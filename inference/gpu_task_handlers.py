"""GPU task handlers used by stage-isolated worker processes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, cast

import numpy as np
import torch

from . import runtime_api as base
from .frame_transport import (
    PinnedD2HStager,
    PinnedH2DStager,
    begin_cuda_frames_to_slots,
    copy_host_frames_to_slots,
)
from .npp_resize import NppLanczosResizer
from .sr_runtime import infer_cuda_batch, infer_cuda_tensor, probe_cuda_uint16
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
    sr_uint16_cuda_enabled: bool = True


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

    pending = begin_cuda_frames_to_slots(
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


def _run_sr_cpu_output_fallback(
    context: WorkerContext,
    frames: tuple[np.ndarray, ...],
    output_slots: tuple[int, ...],
) -> None:
    """Stable path that never requires eager CUDA uint16 tensors."""
    if context.sr_model is None or context.sr_output_view is None:
        raise RuntimeError("SR worker fallback is missing model/output storage")

    import cv2

    target_h = int(context.sr_output_view.shape[1])
    target_w = int(context.sr_output_view.shape[2])
    for frame, slot in zip(frames, output_slots):
        result = base.infer_frame(context.sr_model, frame, context.device)
        if result.shape[0] != target_h or result.shape[1] != target_w:
            result = cv2.resize(
                result,
                (target_w, target_h),
                interpolation=cv2.INTER_LANCZOS4,
            )
        np.copyto(context.sr_output_view[int(slot)], result, casting="no")
        del result
    _notify_compute_done(context)


def probe_sr_worker(context: WorkerContext) -> None:
    """Exercise the exact batch=1 SR/resize/transport path before WorkerReady."""
    if context.sr_model is None or context.sr_output_view is None:
        raise RuntimeError("SR startup probe requires model and shared output storage")
    if int(context.input_view.shape[0]) < 1 or int(context.sr_output_view.shape[0]) < 1:
        raise RuntimeError("SR startup probe requires at least one input/output slot")

    dtype = np.dtype(context.dtype)
    if dtype not in {np.dtype(np.uint8), np.dtype(np.uint16)}:
        raise TypeError(f"SR worker supports uint8/uint16 frames, got {dtype}")

    sample = context.input_view[0]
    sample.fill(0)
    path = "cuda"
    uint16_detail = "not applicable"

    if dtype == np.dtype(np.uint16):
        context.sr_uint16_cuda_enabled, uint16_detail = probe_cuda_uint16(
            context.device
        )
        if not context.sr_uint16_cuda_enabled:
            path = "cpu-output-fallback"
            print(
                f"[sr] {context.device} CUDA uint16 fast path unavailable; "
                f"using stable 10-bit fallback | {uint16_detail}",
                flush=True,
            )

    try:
        if dtype == np.dtype(np.uint16) and not context.sr_uint16_cuda_enabled:
            _run_sr_cpu_output_fallback(context, (sample,), (0,))
        else:
            result_cuda = infer_cuda_batch(
                context.sr_model,
                (sample,),
                context.device,
                h2d_stager=_require_h2d_stager(context),
            )
            _store_sr_cuda_batch(
                context,
                result_cuda,
                (0,),
                publish_boundary=False,
            )
            del result_cuda
        torch.cuda.current_stream(context.device).synchronize()
    except torch.cuda.OutOfMemoryError as error:
        torch.cuda.empty_cache()
        if dtype == np.dtype(np.uint16) and context.sr_uint16_cuda_enabled:
            context.sr_uint16_cuda_enabled = False
            path = "cpu-output-fallback"
            print(
                f"[sr] {context.device} 10-bit CUDA fast path OOM at batch=1; "
                "retrying startup probe with stable CPU-output fallback",
                flush=True,
            )
            try:
                _run_sr_cpu_output_fallback(context, (sample,), (0,))
                torch.cuda.current_stream(context.device).synchronize()
            except torch.cuda.OutOfMemoryError as fallback_error:
                torch.cuda.empty_cache()
                raise RuntimeError(
                    "APISR full-frame batch=1 SR probe ran out of VRAM even on "
                    f"the stable 10-bit fallback at {sample.shape[1]}x{sample.shape[0]} "
                    f"on {context.device}."
                ) from fallback_error
        else:
            raise RuntimeError(
                "APISR full-frame batch=1 SR probe ran out of VRAM at "
                f"{sample.shape[1]}x{sample.shape[0]} on {context.device}."
            ) from error
    finally:
        torch.cuda.empty_cache()

    resize_needed = (
        int(context.sr_output_view.shape[1]) != int(sample.shape[0]) * 4
        or int(context.sr_output_view.shape[2]) != int(sample.shape[1]) * 4
    )
    if resize_needed:
        resize_path = "npp" if context.npp_resizer is not None else "cpu-lanczos"
    else:
        resize_path = "native-4x"
    print(
        f"[sr] {context.device} startup probe passed | input="
        f"{sample.shape[1]}x{sample.shape[0]} | dtype={dtype} | "
        f"path={path} | resize={resize_path} | batch=1",
        flush=True,
    )


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
    dtype = np.dtype(context.dtype)
    if dtype not in {np.dtype(np.uint8), np.dtype(np.uint16)}:
        raise TypeError(f"SR worker supports uint8/uint16 frames, got {context.dtype}")

    if dtype == np.dtype(np.uint16) and not context.sr_uint16_cuda_enabled:
        _run_sr_cpu_output_fallback(context, frames, output_slots)
    elif len(frames) > 1 and context.sr_micro_batch_enabled:
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
        try:
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
        except torch.cuda.OutOfMemoryError:
            if dtype != np.dtype(np.uint16):
                raise
            context.sr_uint16_cuda_enabled = False
            torch.cuda.empty_cache()
            print(
                f"[sr] {context.device} 10-bit CUDA batch=1 OOM after startup; "
                "switching this worker to stable CPU-output fallback",
                flush=True,
            )
            _run_sr_cpu_output_fallback(context, frames, output_slots)

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
