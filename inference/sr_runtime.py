"""Explicit CUDA SR helpers used by unified GPU workers."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from .frame_transport import PinnedH2DStager


def _model_input_dtype(model: torch.nn.Module) -> torch.dtype:
    configured = getattr(model, "sr_input_dtype", None)
    if isinstance(configured, torch.dtype):
        return configured
    try:
        return next(model.parameters()).dtype
    except StopIteration:
        return torch.float32


def _uint16_torch_dtype() -> torch.dtype:
    dtype = getattr(torch, "uint16", None)
    if not isinstance(dtype, torch.dtype):
        raise RuntimeError(
            "This PyTorch build does not expose torch.uint16; "
            "use the stable 10-bit CPU-output fallback."
        )
    return dtype


def _frame_output_spec(dtype: np.dtype) -> tuple[float, torch.dtype]:
    dtype = np.dtype(dtype)
    if dtype == np.dtype(np.uint8):
        return 255.0, torch.uint8
    if dtype == np.dtype(np.uint16):
        return 65535.0, _uint16_torch_dtype()
    raise TypeError(f"SR CUDA path supports uint8/uint16, got {dtype}")


def probe_cuda_uint16(device: torch.device) -> tuple[bool, str]:
    """Probe the exact eager-mode uint16 primitives needed by the fast SR path."""
    if device.type != "cuda":
        return False, "device is not CUDA"
    try:
        uint16 = _uint16_torch_dtype()
        source = np.array([0, 1, 1023, 32768, 65535], dtype=np.uint16)
        host = torch.from_numpy(source)
        pinned = torch.empty(
            host.shape,
            dtype=uint16,
            device="cpu",
            pin_memory=True,
        )
        pinned.copy_(host)
        stream = torch.cuda.current_stream(device)
        cuda_value = pinned.to(device, non_blocking=True)
        restored = (
            cuda_value.to(dtype=torch.float32)
            .div_(65535.0)
            .mul_(65535.0)
            .round_()
            .to(dtype=uint16)
        )
        host_out = torch.empty(
            restored.shape,
            dtype=uint16,
            device="cpu",
            pin_memory=True,
        )
        host_out.copy_(restored, non_blocking=True)
        stream.synchronize()
        if not np.array_equal(host_out.numpy(), source):
            return False, "uint16 CUDA round-trip changed sample values"
        return True, "uint16 CUDA H2D/cast/D2H probe passed"
    except Exception as error:
        try:
            torch.cuda.current_stream(device).synchronize()
        except Exception:
            pass
        return False, f"{type(error).__name__}: {error}"


def infer_cuda_batch(
    model: torch.nn.Module,
    frames: Sequence[np.ndarray],
    device: torch.device,
    *,
    h2d_stager: PinnedH2DStager | None = None,
) -> torch.Tensor:
    """Run a small uint8/uint16 SR batch and keep packed HWC results on CUDA."""
    values = tuple(frames)
    if not values:
        raise ValueError("SR batch cannot be empty")

    shape = values[0].shape
    dtype = np.dtype(values[0].dtype)
    max_value, output_dtype = _frame_output_spec(dtype)
    for frame in values:
        if np.dtype(frame.dtype) != dtype:
            raise TypeError("SR micro-batch frames must have identical dtypes")
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
    tensor = tensor.to(dtype=_model_input_dtype(model))
    tensor.div_(max_value)
    if bool(getattr(model, "sr_channels_last", True)):
        tensor = tensor.contiguous(memory_format=torch.channels_last)
    else:
        tensor = tensor.contiguous()

    with torch.inference_mode():
        output = model(tensor)
        output.clamp_(0, 1)
        output = output.mul_(max_value).round_().to(dtype=output_dtype)
    return output.permute(0, 2, 3, 1).contiguous()


def infer_cuda_tensor(
    model: torch.nn.Module,
    frame: np.ndarray,
    device: torch.device,
    *,
    h2d_stager: PinnedH2DStager | None = None,
) -> torch.Tensor:
    """Single-frame wrapper around the generic SR micro-batch path."""
    return infer_cuda_batch(
        model,
        (frame,),
        device,
        h2d_stager=h2d_stager,
    )[0]


def infer_cuda_u8_batch(
    model: torch.nn.Module,
    frames: Sequence[np.ndarray],
    device: torch.device,
    *,
    h2d_stager: PinnedH2DStager | None = None,
) -> torch.Tensor:
    """Compatibility wrapper that requires uint8 input."""
    values = tuple(frames)
    if any(np.dtype(frame.dtype) != np.dtype(np.uint8) for frame in values):
        raise TypeError("infer_cuda_u8_batch requires uint8 frames")
    return infer_cuda_batch(model, values, device, h2d_stager=h2d_stager)


def infer_cuda_u8_tensor(
    model: torch.nn.Module,
    frame: np.ndarray,
    device: torch.device,
    *,
    h2d_stager: PinnedH2DStager | None = None,
) -> torch.Tensor:
    """Compatibility single-frame uint8 wrapper."""
    if np.dtype(frame.dtype) != np.dtype(np.uint8):
        raise TypeError("infer_cuda_u8_tensor requires a uint8 frame")
    return infer_cuda_tensor(model, frame, device, h2d_stager=h2d_stager)
