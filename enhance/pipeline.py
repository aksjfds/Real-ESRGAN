"""Resident official Real-ESRGAN frame enhancement pipeline."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch


@dataclass(frozen=True)
class PipelineConfig:
    native_scale: int = 4
    fp16: bool = True
    channels_last: bool = True


class StageTimings:
    def __init__(self) -> None:
        self.values: dict[str, float] = {"realesrgan": 0.0}


class FrameEnhancementPipeline:
    def __init__(self, model: torch.nn.Module, device: torch.device, config: PipelineConfig):
        self.model = model
        self.device = device
        self.config = config
        self.timings = StageTimings()

    def enhance_batch(self, frames: Sequence[np.ndarray]) -> list[np.ndarray]:
        """Run the official model and return native-scale FP32 RGB frames."""
        if not frames:
            return []
        if any(frame.ndim != 3 or frame.shape[2] != 3 for frame in frames):
            raise TypeError("Inputs must be HWC RGB")
        if any(frame.dtype not in (np.uint8, np.float32) for frame in frames):
            raise TypeError("Inputs must be uint8 RGB or float32 RGB in [0,1]")
        if len({frame.dtype for frame in frames}) != 1:
            raise TypeError("A worker batch cannot mix uint8 and float32 frames")

        rgb = np.stack(frames)
        tensor = torch.from_numpy(rgb).permute(0, 3, 1, 2).to(self.device, non_blocking=True)
        if rgb.dtype == np.uint8:
            tensor = tensor.float().div_(255.0)
        else:
            if not np.isfinite(rgb).all():
                raise ValueError("Float input contains NaN or Inf")
            tensor = tensor.float().clamp_(0.0, 1.0)

        inference_input = tensor
        if self.config.fp16 and self.device.type == "cuda":
            inference_input = inference_input.half()
        if self.config.channels_last and self.device.type == "cuda":
            inference_input = inference_input.contiguous(memory_format=torch.channels_last)

        started = time.monotonic()
        with torch.inference_mode():
            native = self.model(inference_input).float()
        self.timings.values["realesrgan"] += time.monotonic() - started

        expected = (
            tensor.shape[-2] * self.config.native_scale,
            tensor.shape[-1] * self.config.native_scale,
        )
        if tuple(native.shape[-2:]) != expected:
            raise RuntimeError(
                f"Model output dimensions {tuple(native.shape[-2:])} do not match {expected}"
            )
        final = native.clamp(0, 1).permute(0, 2, 3, 1).contiguous().cpu().numpy()
        return list(final.astype(np.float32, copy=False))

    def timing_snapshot(self) -> dict[str, float]:
        return dict(self.timings.values)


class RawVideoWriter:
    """Interface marker; the persistent FFmpeg implementation remains in realesrgan.py."""
