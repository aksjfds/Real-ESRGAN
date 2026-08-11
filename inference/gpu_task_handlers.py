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
    copy_cuda_frame_to_array,
    copy_cuda_frames_to_slots,
)
from .sr_runtime import infer_cuda_u8_tensor
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
        context.h2d_stager = PinnedH2DStager(context.device)
    return context.h2d_stager


def _require_d2h_stager(context: WorkerContext) -> PinnedD2HStager:
    if context.d2h_stager is None:
        context.d2h_stager = PinnedD2HStager(context.device)
    return context.d2h_stager


def _pack_bvs_device_groups(
    task: BVSTask,
    device_groups: list[torch.Tensor],
) -> tuple[torch.Tensor | None, tuple[int, ...], tuple[int, ...]]:
    """Pack all emitted BVS frames before the compute boundary for one D2H."""
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
            np.copyto(
                context.frame_output_view[slot],
                frame,
                casting="no",
            )
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
            [
                context.input_view[cursor + index]
                for index in range(group.count)
            ]
        )
        cursor += group.count

    before_tiles = int(bvs.tiles)
    device_groups = bvs.enhance_clips_device(clips)
    if device_groups is not None:
        packed, emitted_counts, slots = _pack_bvs_device_groups(
            task,
            device_groups,
        )
        # Packing is a CUDA op and is complete/enqueued before this boundary.
        # Only pinned D2H + CPU shared-memory scatter remains afterward.
        _notify_compute_done(context)
        if packed is not None:
            copy_cuda_frames_to_slots(
                packed,
                context.frame_output_view,
                slots,
                stager=_require_d2h_stager(context),
            )
    else:
        if len(clips) > 1:
            enhanced_groups = bvs.enhance_clips(clips)
        else:
            enhanced_groups = [bvs.enhance_clip(clips[0])]
        _notify_compute_done(context)
        emitted_counts = _copy_bvs_cpu_groups(
            context,
            task,
            enhanced_groups,
        )

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
    count = context.rife.interpolate_into(
        frame0,
        frame1,
        task.timesteps,
        context.frame_output_view,
        task.output_slots,
        compute_done=lambda: _notify_compute_done(context),
    )
    _notify_compute_done(context)
    if count != len(task.output_slots):
        raise RuntimeError(
            "RIFE returned a different frame count than reserved output slots"
        )
    return RIFEResult(count=count)


def run_sr(context: WorkerContext, task: SRTask) -> SRResult:
    if context.sr_model is None:
        raise RuntimeError("SR task submitted to a worker without Real-ESRGAN")
    if context.sr_output_view is None:
        raise RuntimeError("SR worker output shared memory is unavailable")

    output_slot = int(task.output_slot)
    if output_slot < 0 or output_slot >= int(context.sr_output_view.shape[0]):
        raise RuntimeError(
            f"SR output slot out of range: {output_slot}/"
            f"{context.sr_output_view.shape[0]}"
        )
    target = context.sr_output_view[output_slot]
    frame = resolve_frame(context, task.frame)

    if context.dtype == np.dtype(np.uint8):
        result_cuda = infer_cuda_u8_tensor(
            context.sr_model,
            frame,
            context.device,
            h2d_stager=_require_h2d_stager(context),
        )
        _notify_compute_done(context)
        copy_cuda_frame_to_array(
            result_cuda,
            target,
            _require_d2h_stager(context),
        )
        del result_cuda
    else:
        result = base.infer_frame(
            context.sr_model,
            frame,
            context.device,
        )
        _notify_compute_done(context)
        np.copyto(
            target,
            result,
            casting="no",
        )
        del result

    return SRResult(
        frame_id=task.frame_id,
        output_slot=output_slot,
    )


Handler = Callable[[WorkerContext, WorkerTask], ResultPayload]


def build_temporal_handlers() -> dict[TaskKind, Handler]:
    return {
        TaskKind.BVS: cast(Handler, run_bvs),
        TaskKind.RIFE: cast(Handler, run_rife),
    }


def build_sr_handlers() -> dict[TaskKind, Handler]:
    return {
        TaskKind.SR: cast(Handler, run_sr),
    }
