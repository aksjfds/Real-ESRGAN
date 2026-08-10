"""GPU task handlers used by the process-isolated worker core."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch

from . import runtime as base
from .task_protocol import (
    BVSResult,
    BVSTask,
    FrameInput,
    FrameStorage,
    RIFEResult,
    RIFETask,
    SRResult,
    SRTask,
    TaskKind,
)


@dataclass
class WorkerContext:
    device: torch.device
    dtype: np.dtype
    bvs: object
    rife: object | None
    sr_model: torch.nn.Module
    input_view: np.ndarray
    frame_output_view: np.ndarray
    sr_output_view: np.ndarray
    sr_output_tensor: torch.Tensor
    sr_output_slot: object


def resolve_frame(context: WorkerContext, frame: FrameInput) -> np.ndarray:
    if frame.storage is FrameStorage.INPUT:
        return context.input_view[frame.slot]
    if frame.storage is FrameStorage.OUTPUT:
        return context.frame_output_view[frame.slot]
    raise RuntimeError(f"Unsupported frame storage: {frame.storage!r}")


def run_bvs(context: WorkerContext, task: BVSTask) -> BVSResult:
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

    before_tiles = int(getattr(context.bvs, "tiles", 0))
    if len(clips) > 1 and hasattr(context.bvs, "enhance_clips"):
        enhanced_groups = context.bvs.enhance_clips(clips)
    else:
        enhanced_groups = [
            context.bvs.enhance_clip(frames)
            for frames in clips
        ]

    output_cursor = 0
    emitted_counts: list[int] = []
    for group, enhanced in zip(task.groups, enhanced_groups):
        emitted = enhanced[group.emit_start : group.emit_end]
        emitted_count = len(emitted)
        emitted_counts.append(emitted_count)

        end = output_cursor + emitted_count
        slots = task.output_slots[output_cursor:end]
        if len(slots) != emitted_count:
            raise RuntimeError(
                "BVS task output-slot accounting mismatch"
            )

        for slot, frame in zip(slots, emitted):
            np.copyto(
                context.frame_output_view[slot],
                frame,
                casting="no",
            )
        output_cursor = end

    if output_cursor != len(task.output_slots):
        raise RuntimeError(
            "BVS task did not fill all reserved output slots"
        )

    return BVSResult(
        emitted_counts=tuple(emitted_counts),
        tile_size=int(getattr(context.bvs, "tile_size", 0)),
        tiles=int(getattr(context.bvs, "tiles", 0)) - before_tiles,
        clips=len(clips),
    )


def run_rife(context: WorkerContext, task: RIFETask) -> RIFEResult:
    if context.rife is None:
        raise RuntimeError(
            "RIFE task submitted to a worker without a RIFE model"
        )

    frame0 = resolve_frame(context, task.frame0)
    frame1 = resolve_frame(context, task.frame1)
    generated = context.rife.interpolate_many(
        frame0,
        frame1,
        task.timesteps,
    )

    if len(generated) != len(task.output_slots):
        raise RuntimeError(
            "RIFE returned a different frame count than reserved output slots"
        )

    for slot, frame in zip(task.output_slots, generated):
        np.copyto(
            context.frame_output_view[slot],
            frame,
            casting="no",
        )

    return RIFEResult(count=len(generated))


def run_sr(context: WorkerContext, task: SRTask) -> SRResult:
    frame = resolve_frame(context, task.frame)

    if context.dtype == np.dtype(np.uint8):
        from . import v51_runtime as v51

        result_cuda = v51._infer_cuda_u8_tensor(
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


Handler = Callable[[WorkerContext, object], object]


def build_handlers() -> dict[TaskKind, Handler]:
    """Return the task registry consumed by the generic GPU worker loop."""

    return {
        TaskKind.BVS: run_bvs,
        TaskKind.RIFE: run_rife,
        TaskKind.SR: run_sr,
    }
