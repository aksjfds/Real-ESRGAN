"""GPU task handlers used by stage-isolated worker processes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, cast

import numpy as np
import torch

from .frame_transport import (
    PendingD2H,
    PinnedD2HStager,
    PinnedH2DStager,
    begin_cuda_frames_to_slots,
    copy_host_frames_to_slots,
)
from .npp_resize import NppLanczosResizer
from .sr_runtime import infer_cuda_batch, infer_cuda_tensor
from .task_protocol import (
    BVSResult,
    BVSTask,
    FrameInput,
    FrameStorage,
    RIFEResult,
    RIFETask,
    ResultPayload,
    SRResult,
    SRTask,
    TaskKind,
    WorkerTask,
)
from .worker_protocols import BasicVSRExecutor, RIFEExecutor


@dataclass
class WorkerContext:
    device: torch.device
    dtype: np.dtype
    input_view: np.ndarray
    frame_output_view: np.ndarray
    bvs: BasicVSRExecutor | None = None
    rife: RIFEExecutor | None = None
    sr_model: torch.nn.Module | None = None
    sr_output_view: np.ndarray | None = None
    h2d_stager: PinnedH2DStager | None = None
    d2h_stager: PinnedD2HStager | None = None
    compute_done: Callable[[], None] | None = None
    npp_resizer: NppLanczosResizer | None = None
    npp_checked: bool = False
    npp_active_logged: bool = False
    sr_micro_batch_enabled: bool = True


def _notify_compute_done(context: WorkerContext) -> None:
    callback = context.compute_done
    if callback is None:
        return
    context.compute_done = None
    callback()


def resolve_frame(context: WorkerContext, frame: FrameInput) -> np.ndarray:
    if frame.storage is FrameStorage.INPUT:
        return context.input_view[frame.slot]
    if frame.storage is FrameStorage.OUTPUT:
        return context.frame_output_view[frame.slot]
    raise RuntimeError(f"Unsupported frame storage: {frame.storage!r}")


def _require_bvs(context: WorkerContext) -> BasicVSRExecutor:
    if context.bvs is None:
        raise RuntimeError("BVS task submitted to a worker without BasicVSR++")
    return context.bvs


def _require_h2d_stager(context: WorkerContext) -> PinnedH2DStager:
    if context.h2d_stager is None:
        context.h2d_stager = PinnedH2DStager(context.device, slots=2)
    return context.h2d_stager


def _require_d2h_stager(context: WorkerContext) -> PinnedD2HStager:
    if context.d2h_stager is None:
        context.d2h_stager = PinnedD2HStager(context.device)
    return context.d2h_stager


def _pack_bvs_device_groups(
    task: BVSTask,
    device_groups: list[torch.Tensor],
) -> tuple[torch.Tensor | None, tuple[int, ...], tuple[int, ...]]:
    if len(device_groups) != len(task.groups):
        raise RuntimeError("BVS device output group count mismatch")

    output_cursor = 0
    emitted_counts: list[int] = []
    emitted_batches: list[torch.Tensor] = []
    all_slots: list[int] = []

    for group, enhanced in zip(task.groups, device_groups):
        emitted = enhanced[group.emit_start : group.emit_end]
        emitted_count = int(emitted.shape[0])
        emitted_counts.append(emitted_count)

        end = output_cursor + emitted_count
        slots = task.output_slots[output_cursor:end]
        if len(slots) != emitted_count:
            raise RuntimeError("BVS task output-slot accounting mismatch")

        if emitted_count:
            emitted_batches.append(emitted)
            all_slots.extend(int(slot) for slot in slots)
        output_cursor = end

    if output_cursor != len(task.output_slots):
        raise RuntimeError("BVS task did not fill all reserved output slots")

    packed = None
    if emitted_batches:
        packed = (
            emitted_batches[0].contiguous()
            if len(emitted_batches) == 1
            else torch.cat(emitted_batches, dim=0).contiguous()
        )
    return packed, tuple(emitted_counts), tuple(all_slots)


def _copy_bvs_cpu_groups(
    context: WorkerContext,
    task: BVSTask,
    enhanced_groups: list[list[np.ndarray]],
) -> tuple[int, ...]:
    output_cursor = 0
    emitted_counts: list[int] = []
    for group, enhanced in zip(task.groups, enhanced_groups):
        emitted = enhanced[group.emit_start : group.emit_end]
        emitted_count = len(emitted)
        emitted_counts.append(emitted_count)

        end = output_cursor + emitted_count
        slots = task.output_slots[output_cursor:end]
        if len(slots) != emitted_count:
            raise RuntimeError("BVS task output-slot accounting mismatch")

        for slot, frame in zip(slots, emitted):
            np.copyto(context.frame_output_view[slot], frame, casting="no")
        output_cursor = end

    if output_cursor != len(task.output_slots):
        raise RuntimeError("BVS task did not fill all reserved output slots")
    return tuple(emitted_counts)


def run_bvs(context: WorkerContext, task: BVSTask) -> BVSResult:
    bvs = _require_bvs(context)
    clips: list[list[np.ndarray]] = []
    cursor = 0
    for group in task.groups:
        clips.append(
            [context.input_view[cursor + index] for index in range(group.count)]
        )
        cursor += group.count

    before_tiles = int(bvs.tiles)
    device_groups = bvs.enhance_clips_device(clips)
    if device_groups is not None:
        packed, emitted_counts, slots = _pack_bvs_device_groups(task, device_groups)
        pending = None
        if packed is not None:
            pending = begin_cuda_frames_to_slots(
                packed,
                context.frame_output_view,
                slots,
                _require_d2h_stager(context),
            )
        _notify_compute_done(context)
        if pending is not None:
            pending.wait()
    else:
        if len(clips) > 1:
            enhanced_groups = bvs.enhance_clips(clips)
        else:
            enhanced_groups = [bvs.enhance_clip(clips[0])]
        _notify_compute_done(context)
        emitted_counts = _copy_bvs_cpu_groups(context, task, enhanced_groups)

    return BVSResult(
        emitted_counts=emitted_counts,
        tile_size=int(bvs.tile_size),
        tiles=int(bvs.tiles) - before_tiles,
        clips=len(clips),
    )


def run_rife(context: WorkerContext, task: RIFETask) -> RIFEResult:
    if context.rife is None:
        raise RuntimeError("RIFE task submitted to a worker without a RIFE model")

    frame0 = resolve_frame(context, task.frame0)
    frame1 = resolve_frame(context, task.frame1)
    batch = context.rife.interpolate_device(frame0, frame1, task.timesteps)

    expected = len(task.output_slots)
    if batch is None:
        _notify_compute_done(context)
        if expected != 0:
            raise RuntimeError(
                "RIFE returned no CUDA batch for reserved output slots"
            )
        return RIFEResult(count=0)

    count = int(batch.shape[0])
    if count != expected:
        raise RuntimeError(
            f"RIFE returned {count} frames for {expected} reserved output slots"
        )

    if frame0.dtype == np.uint8:
        pending = begin_cuda_frames_to_slots(
            batch,
            context.frame_output_view,
            task.output_slots,
            _require_d2h_stager(context),
        )
        _notify_compute_done(context)
        if pending is not None:
            pending.wait()
    else:
        pending = _require_d2h_stager(context).begin_copy(batch)
        _notify_compute_done(context)
        frames_cpu = pending.wait().astype(frame0.dtype, copy=False)
        copy_host_frames_to_slots(
            frames_cpu,
            context.frame_output_view,
            task.output_slots,
        )

    return RIFEResult(count=count)


def _get_npp_resizer(context: WorkerContext) -> NppLanczosResizer | None:
    if context.npp_checked:
        return context.npp_resizer
    context.npp_checked = True
    try:
        context.npp_resizer = NppLanczosResizer(context.device)
        print(f"[npp] {context.device} Lanczos runtime ready", flush=True)
    except Exception as error:
        context.npp_resizer = None
        print(
            f"[npp] {context.device} Lanczos unavailable; "
            f"using CPU Lanczos4 fallback: {error!r}",
            flush=True,
        )
    return context.npp_resizer


def _cpu_resize_sr_batch(
    frames: np.ndarray,
    output_view: np.ndarray,
    output_slots: tuple[int, ...],
) -> None:
    import cv2

    target_h = int(output_view.shape[1])
    target_w = int(output_view.shape[2])
    for frame, slot in zip(frames, output_slots):
        if frame.shape[0] == target_h and frame.shape[1] == target_w:
            resized = frame
        else:
            resized = cv2.resize(
                frame,
                (target_w, target_h),
                interpolation=cv2.INTER_LANCZOS4,
            )
        np.copyto(output_view[int(slot)], resized, casting="no")


def _begin_sr_cuda_frames_to_slots(
    frames: torch.Tensor,
    output_view: np.ndarray,
    slots: tuple[int, ...],
    stager: PinnedD2HStager,
) -> PendingD2H | None:
    """Use master uint8 transport; extend the same direct path to uint16 SR."""
    if frames.dtype == torch.uint8:
        return begin_cuda_frames_to_slots(frames, output_view, slots, stager)
    if frames.dtype != torch.uint16:
        raise TypeError(f"SR transport supports uint8/uint16, got {frames.dtype}")
    if frames.device.type != "cuda" or frames.ndim != 4 or not frames.is_contiguous():
        raise RuntimeError("uint16 SR transport requires contiguous CUDA [N,H,W,C]")
    if np.dtype(output_view.dtype) != np.dtype(np.uint16):
        raise TypeError(
            f"uint16 SR output requires uint16 shared pool, got {output_view.dtype}"
        )

    slot_ids = tuple(int(slot) for slot in slots)
    count = int(frames.shape[0])
    if count != len(slot_ids):
        raise RuntimeError(
            f"SR transport frame/slot mismatch: frames={count}, slots={len(slot_ids)}"
        )
    expected_shape = tuple(int(value) for value in output_view.shape[1:])
    actual_shape = tuple(int(value) for value in frames.shape[1:])
    if actual_shape != expected_shape:
        raise RuntimeError(
            f"SR transport frame shape mismatch: {actual_shape} != {expected_shape}"
        )
    if len(set(slot_ids)) != len(slot_ids):
        raise RuntimeError("SR transport received duplicate output slots")
    for slot in slot_ids:
        if slot < 0 or slot >= int(output_view.shape[0]):
            raise RuntimeError(
                f"SR transport output slot out of range: {slot}/{output_view.shape[0]}"
            )
    if count == 0:
        return None

    first = slot_ids[0]
    contiguous = slot_ids == tuple(range(first, first + count))
    if contiguous:
        return stager.begin_copy(
            frames,
            target=output_view[first : first + count],
        )
    return stager.begin_copy(
        frames,
        finalize=lambda array: copy_host_frames_to_slots(
            array,
            output_view,
            slot_ids,
        ),
    )


def _store_sr_cuda_batch(
    context: WorkerContext,
    result_cuda: torch.Tensor,
    output_slots: tuple[int, ...],
    *,
    publish_boundary: bool,
) -> None:
    if context.sr_output_view is None:
        raise RuntimeError("SR worker output shared memory is unavailable")

    target_h = int(context.sr_output_view.shape[1])
    target_w = int(context.sr_output_view.shape[2])
    source_h = int(result_cuda.shape[1])
    source_w = int(result_cuda.shape[2])
    resized_cuda = result_cuda

    if source_h != target_h or source_w != target_w:
        resizer = _get_npp_resizer(context)
        if resizer is not None:
            try:
                resized_cuda = resizer.resize_batch(
                    result_cuda,
                    target_h,
                    target_w,
                )
                if not context.npp_active_logged:
                    context.npp_active_logged = True
                    print(
                        f"[npp] {context.device} Lanczos active: "
                        f"{source_w}x{source_h} -> {target_w}x{target_h} | "
                        f"dtype={result_cuda.dtype}",
                        flush=True,
                    )
            except Exception as error:
                try:
                    torch.cuda.current_stream(context.device).synchronize()
                except Exception:
                    pass
                context.npp_resizer = None
                context.npp_checked = True
                print(
                    f"[npp] {context.device} Lanczos failed; disabling NPP and "
                    f"using CPU Lanczos4 fallback: {error!r}",
                    flush=True,
                )

        if context.npp_resizer is None:
            pending = _require_d2h_stager(context).begin_copy(result_cuda)
            if publish_boundary:
                _notify_compute_done(context)
            frames_cpu = pending.wait()
            _cpu_resize_sr_batch(
                frames_cpu,
                context.sr_output_view,
                output_slots,
            )
            return

    pending = _begin_sr_cuda_frames_to_slots(
        resized_cuda,
        context.sr_output_view,
        output_slots,
        _require_d2h_stager(context),
    )
    if publish_boundary:
        _notify_compute_done(context)
    if pending is not None:
        pending.wait()


def _run_sr_sequential_fallback(
    context: WorkerContext,
    frames: tuple[np.ndarray, ...],
    output_slots: tuple[int, ...],
) -> None:
    if context.sr_model is None:
        raise RuntimeError("SR worker model is unavailable")
    for frame, slot in zip(frames, output_slots):
        result_cuda = infer_cuda_tensor(
            context.sr_model,
            frame,
            context.device,
            h2d_stager=_require_h2d_stager(context),
        )
        _store_sr_cuda_batch(
            context,
            result_cuda.unsqueeze(0),
            (slot,),
            publish_boundary=False,
        )
        del result_cuda
    _notify_compute_done(context)


def run_sr(context: WorkerContext, task: SRTask) -> SRResult:
    if context.sr_model is None:
        raise RuntimeError("SR task submitted to a worker without an SR model")
    if context.sr_output_view is None:
        raise RuntimeError("SR worker output shared memory is unavailable")

    if not task.frame_ids or len(task.frame_ids) != len(task.frames):
        raise RuntimeError("SR task frame ids/inputs are empty or misaligned")
    if len(task.frames) != len(task.output_slots):
        raise RuntimeError("SR task inputs/output slots are misaligned")

    slot_count = int(context.sr_output_view.shape[0])
    output_slots = tuple(int(value) for value in task.output_slots)
    if len(set(output_slots)) != len(output_slots):
        raise RuntimeError("SR task contains duplicate output slots")
    for slot in output_slots:
        if slot < 0 or slot >= slot_count:
            raise RuntimeError(f"SR output slot out of range: {slot}/{slot_count}")

    frames = tuple(resolve_frame(context, value) for value in task.frames)
    if np.dtype(context.dtype) not in {np.dtype(np.uint8), np.dtype(np.uint16)}:
        raise TypeError(f"SR worker supports uint8/uint16 frames, got {context.dtype}")

    if len(frames) > 1 and context.sr_micro_batch_enabled:
        try:
            result_cuda = infer_cuda_batch(
                context.sr_model,
                frames,
                context.device,
                h2d_stager=_require_h2d_stager(context),
            )
        except torch.cuda.OutOfMemoryError:
            context.sr_micro_batch_enabled = False
            torch.cuda.empty_cache()
            print(
                f"[sr] {context.device} micro-batch={len(frames)} OOM; "
                "locking this worker to batch=1 fallback",
                flush=True,
            )
            _run_sr_sequential_fallback(context, frames, output_slots)
        else:
            _store_sr_cuda_batch(
                context,
                result_cuda,
                output_slots,
                publish_boundary=True,
            )
            del result_cuda
    elif len(frames) > 1:
        _run_sr_sequential_fallback(context, frames, output_slots)
    else:
        result_cuda = infer_cuda_batch(
            context.sr_model,
            frames,
            context.device,
            h2d_stager=_require_h2d_stager(context),
        )
        _store_sr_cuda_batch(
            context,
            result_cuda,
            output_slots,
            publish_boundary=True,
        )
        del result_cuda

    return SRResult(
        frame_ids=tuple(int(value) for value in task.frame_ids),
        output_slots=output_slots,
    )


Handler = Callable[[WorkerContext, WorkerTask], ResultPayload]


def build_temporal_handlers() -> dict[TaskKind, Handler]:
    return {
        TaskKind.BVS: cast(Handler, run_bvs),
        TaskKind.RIFE: cast(Handler, run_rife),
    }


def build_sr_handlers() -> dict[TaskKind, Handler]:
    return {TaskKind.SR: cast(Handler, run_sr)}
