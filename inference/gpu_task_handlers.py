"""GPU task handlers used by stage-isolated worker processes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, cast

import numpy as np
import torch

from . import runtime_api as base
from .frame_transport import copy_cuda_frames_to_slots
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
    sr_output_tensor: torch.Tensor | None = None
    sr_output_slot: object | None = None


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


def _copy_bvs_device_groups(
    context: WorkerContext,
    task: BVSTask,
    device_groups: list[torch.Tensor],
) -> tuple[int, ...]:
    if len(device_groups) != len(task.groups):
        raise RuntimeError("BVS device output group count mismatch")

    output_cursor = 0
    emitted_counts: list[int] = []
    for group, enhanced in zip(task.groups, device_groups):
        emitted = enhanced[group.emit_start : group.emit_end]
        emitted_count = int(emitted.shape[0])
        emitted_counts.append(emitted_count)

        end = output_cursor + emitted_count
        slots = task.output_slots[output_cursor:end]
        if len(slots) != emitted_count:
            raise RuntimeError("BVS task output-slot accounting mismatch")

        copy_cuda_frames_to_slots(
            emitted,
            context.frame_output_view,
            slots,
        )
        output_cursor = end

    if output_cursor != len(task.output_slots):
        raise RuntimeError("BVS task did not fill all reserved output slots")
    return tuple(emitted_counts)


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
        emitted_counts = _copy_bvs_device_groups(
            context,
            task,
            device_groups,
        )
    else:
        if len(clips) > 1:
            enhanced_groups = bvs.enhance_clips(clips)
        else:
            enhanced_groups = [bvs.enhance_clip(clips[0])]
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
    )
    if count != len(task.output_slots):
        raise RuntimeError(
            "RIFE returned a different frame count than reserved output slots"
        )
    return RIFEResult(count=count)


def run_sr(context: WorkerContext, task: SRTask) -> SRResult:
    if context.sr_model is None:
        raise RuntimeError("SR task submitted to a worker without Real-ESRGAN")
    if context.sr_output_slot is None:
        raise RuntimeError("SR worker output semaphore is unavailable")
    if context.sr_output_view is None or context.sr_output_tensor is None:
        raise RuntimeError("SR worker output shared memory is unavailable")

    frame = resolve_frame(context, task.frame)

    if context.dtype == np.dtype(np.uint8):
        result_cuda = infer_cuda_u8_tensor(
            context.sr_model,
            frame,
            context.device,
        )
        context.sr_output_slot.acquire()
        context.sr_output_tensor.copy_(
            result_cuda,
            non_blocking=False,
        )
        del result_cuda
    else:
        result = base.infer_frame(
            context.sr_model,
            frame,
            context.device,
        )
        context.sr_output_slot.acquire()
        np.copyto(
            context.sr_output_view,
            result,
            casting="no",
        )
        del result

    return SRResult(frame_id=task.frame_id)


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
