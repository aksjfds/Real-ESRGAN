"""Explicit Real-ESRGAN CUDA helpers used by unified GPU workers."""

from __future__ import annotations

import numpy as np
import torch

from .frame_transport import PinnedH2DStager


def infer_cuda_u8_tensor(
    model: torch.nn.Module,
    frame: np.ndarray,
    device: torch.device,
    *,
    h2d_stager: PinnedH2DStager | None = None,
) -> torch.Tensor:
    """Run the established uint8 SR path and keep the result on CUDA."""
    host = torch.from_numpy(frame)
    if h2d_stager is None:
        tensor = host.to(device, non_blocking=True)
    else:
        tensor = h2d_stager.copy(host)

    tensor = tensor.permute(2, 0, 1).unsqueeze(0)
    tensor = tensor.half()
    tensor.div_(255.0)
    tensor = tensor.contiguous(memory_format=torch.channels_last)
    with torch.inference_mode():
        output = model(tensor)
        output.clamp_(0, 1)
        output = output.mul_(255.0).round_().byte()
    return output[0].permute(1, 2, 0).contiguous()
