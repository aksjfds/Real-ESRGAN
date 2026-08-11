"""CUDA-to-shared-memory frame transport helpers.

This module owns the GPU-to-host transport boundary used by GPU task handlers.
Model runtimes return CUDA tensors; task handlers decide which frames are emitted;
this helper performs one blocking D2H transfer per emitted batch and copies the
result into pre-reserved shared-memory slots.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch


def copy_cuda_frames_to_slots(
    frames: torch.Tensor,
    output_view: np.ndarray,
    slots: Sequence[int],
) -> None:
    """Copy one CUDA frame batch into shared-memory slots with one D2H sync."""
    if frames.device.type != "cuda":
        raise RuntimeError(
            f"Expected CUDA BVS output, got device={frames.device}"
        )
    if frames.dtype != torch.uint8:
        raise TypeError(
            f"Expected uint8 BVS output, got dtype={frames.dtype}"
        )
    if frames.ndim != 4:
        raise ValueError(
            f"Expected BVS output shape [T,H,W,C], got {tuple(frames.shape)}"
        )

    count = int(frames.shape[0])
    slot_ids = tuple(int(slot) for slot in slots)
    if count != len(slot_ids):
        raise RuntimeError(
            f"BVS transport frame/slot mismatch: frames={count}, slots={len(slot_ids)}"
        )
    if len(set(slot_ids)) != len(slot_ids):
        raise RuntimeError("BVS transport received duplicate output slots")

    expected_shape = tuple(int(value) for value in output_view.shape[1:])
    frame_shape = tuple(int(value) for value in frames.shape[1:])
    if frame_shape != expected_shape:
        raise RuntimeError(
            "BVS transport frame shape mismatch: "
            f"frames={frame_shape}, slots={expected_shape}"
        )
    if output_view.dtype != np.dtype(np.uint8):
        raise TypeError(
            f"Expected uint8 shared output pool, got dtype={output_view.dtype}"
        )

    slot_count = int(output_view.shape[0])
    for slot in slot_ids:
        if slot < 0 or slot >= slot_count:
            raise RuntimeError(
                f"BVS transport output slot out of range: {slot}/{slot_count}"
            )

    if count == 0:
        return

    # GPU -> CPU in the reverse direction is only safe to consume after the
    # transfer completes. A single blocking batch copy avoids the previous
    # per-frame synchronization while keeping correctness explicit.
    frames_cpu = frames.contiguous().cpu().numpy()
    if frames_cpu.dtype != output_view.dtype:
        raise TypeError(
            "BVS transport dtype changed during D2H: "
            f"{frames_cpu.dtype} != {output_view.dtype}"
        )

    for slot, frame in zip(slot_ids, frames_cpu):
        np.copyto(output_view[slot], frame, casting="no")
