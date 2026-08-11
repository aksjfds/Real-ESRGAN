"""Explicit Real-ESRGAN CUDA helpers used by unified GPU workers."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from .frame_transport import PinnedH2DStager


def infer_cuda_u8_batch(
    model: torch.nn.Module,
    frames: Sequence[np.ndarray],
    device: torch.device,
    *,
    h2d_stager: PinnedH2DStager | None = None,
) -> torch.Tensor:
    """Run a small uint8 SR batch and keep packed HWC results on CUDA."""
    values = tuple(frames)
    if not values:
        raise ValueError("SR batch cannot be empty")

    shape = values[0].shape
    for frame in values:
        if frame.dtype != np.uint8:
            raise TypeError(f"SR uint8 batch received {frame.dtype}")
        if frame.shape != shape:
            raise ValueError("SR micro-batch frames must have identical shapes")

    device_frames: list[torch.Tensor] = []
    for frame in values:
        host = torch.from_numpy(frame)
        if h2d_stager is None:
            tensor = host.to(device, non_blocking=True)
        else:
            tensor = h2d_stager.copy(host)
        device_frames.append(tensor.permute(2, 0, 1))

    tensor = torch.stack(device_frames, dim=0)
    tensor = tensor.half()
    tensor.div_(255.0)
    tensor = tensor.contiguous(memory_format=torch.channels_last)
    with torch.inference_mode():
        output = model(tensor)
        output.clamp_(0, 1)
        output = output.mul_(255.0).round_().byte()
    return output.permute(0, 2, 3, 1).contiguous()


def infer_cuda_u8_tensor(
    model: torch.nn.Module,
    frame: np.ndarray,
    device: torch.device,
    *,
    h2d_stager: PinnedH2DStager | None = None,
) -> torch.Tensor:
    """Compatibility single-frame wrapper around the SR micro-batch path."""
    return infer_cuda_u8_batch(
        model,
        (frame,),
        device,
        h2d_stager=h2d_stager,
    )[0]
