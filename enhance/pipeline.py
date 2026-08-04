"""Resident Real-ESRGAN frame enhancement pipeline.

Internal contract:
- decoded input: uint8 HWC RGB
- model input: NCHW RGB in [0, 1]
- model convolutions may use FP16
- ensemble accumulation, branch recombination and back projection use FP32
- worker output: model-native-scale float32 HWC RGB in [0, 1]

Final Lanczos resizing, dehalo and range limiting are intentionally performed in
one shared parent-side path for both full-frame and tiled inference.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch.nn import functional as F

from .ops import BackProjectionRefiner, EnsembleEngine, lanczos_resize
from .srvgg_enhanced import EnhancedSRVGGNetCompact, adaptive_residual_strength


@dataclass(frozen=True)
class PipelineConfig:
    tta_mode: str = "none"
    tta_batch_size: int = 1
    shift_ensemble: str = "none"
    residual_mode: str = "official"
    residual_strength: float = 1.0
    residual_flat_strength: float = 0.9
    residual_edge_strength: float = 1.0
    residual_edge_low: float = 0.05
    residual_edge_high: float = 0.20
    base_correction: float = 0.0
    back_projection_iterations: int = 0
    back_projection_strength: float = 0.2
    back_projection_kernel: str = "lanczos"
    back_projection_clamp: float = 0.05
    native_scale: int = 4
    pre_pad: int = 0
    fp16: bool = True
    channels_last: bool = True


class StageTimings:
    def __init__(self) -> None:
        self.values: dict[str, float] = {
            name: 0.0
            for name in (
                "tta",
                "shift_ensemble",
                "realesrgan",
                "residual_control",
                "base_correction",
                "back_projection",
            )
        }

    def add(self, name: str, started: float) -> None:
        self.values[name] += time.monotonic() - started


class SourceAnalyzer:
    """Interface marker; the parent-side implementation lives in analysis.py."""


class DescaleBackend:
    """Descale is never emulated with resize; unavailable means a hard bypass/error."""

    @staticmethod
    def available() -> bool:
        try:
            import vapoursynth as vs  # type: ignore

            return hasattr(vs.core, "descale") and (
                hasattr(vs.core, "ffms2") or hasattr(vs.core, "lsmas")
            )
        except ImportError:
            return False


class RealESRGANBackend:
    def __init__(self, model: torch.nn.Module, config: PipelineConfig, timings: StageTimings):
        self.model = model
        self.config = config
        self.timings = timings

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if isinstance(self.model, EnhancedSRVGGNetCompact):
            started = time.monotonic()
            components = self.model.forward_components(x)
            self.timings.add("realesrgan", started)

            started = time.monotonic()
            if self.config.residual_mode == "official":
                strength: float | torch.Tensor = 1.0
            elif self.config.residual_mode == "global":
                strength = self.config.residual_strength
            elif self.config.residual_mode == "adaptive":
                strength = adaptive_residual_strength(
                    x.float(),
                    components.residual.shape[-2:],
                    self.config.residual_flat_strength,
                    self.config.residual_edge_strength,
                    self.config.residual_edge_low,
                    self.config.residual_edge_high,
                )
            else:
                raise ValueError(f"Unknown residual mode: {self.config.residual_mode}")

            output = components.nearest_base.float() + components.residual.float() * strength
            self.timings.add("residual_control", started)

            if self.config.base_correction:
                started = time.monotonic()
                lanczos_base = lanczos_resize(
                    x.float(),
                    (x.shape[-2] * self.config.native_scale, x.shape[-1] * self.config.native_scale),
                )
                output = output + self.config.base_correction * (
                    lanczos_base - components.nearest_base.float()
                )
                self.timings.add("base_correction", started)
            return output

        if self.config.residual_mode != "official" or self.config.base_correction:
            raise ValueError("Residual/base controls are only compatible with SRVGG models")
        started = time.monotonic()
        output = self.model(x)
        self.timings.add("realesrgan", started)
        return output


class AnimePostProcessor:
    """Anime4K shaders are applied by the persistent FFmpeg writer, never here."""


class FrameEnhancementPipeline:
    def __init__(self, model: torch.nn.Module, device: torch.device, config: PipelineConfig):
        self.model = model
        self.device = device
        self.config = config
        self.timings = StageTimings()
        self.realesrgan = RealESRGANBackend(model, config, self.timings)
        self.ensemble = EnsembleEngine(config.tta_mode, config.tta_batch_size, config.shift_ensemble)
        self.back_projection = BackProjectionRefiner(
            config.back_projection_iterations,
            config.back_projection_strength,
            config.back_projection_kernel,
            config.back_projection_clamp,
        )

    def _model_call(self, x: torch.Tensor) -> torch.Tensor:
        return self.realesrgan(x)

    def enhance_batch(self, frames: Sequence[np.ndarray]) -> list[np.ndarray]:
        """Return model-native-scale outputs.

        ``pre_pad`` follows the official Real-ESRGAN convention: reflect-pad only
        the right and bottom edges, run the model, then crop the native-scale
        result before any arbitrary-scale resampling.  This avoids the sampling
        phase shift that occurs when a padded image is resized first and cropped
        afterwards at a non-integer final scale.
        """
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
        original_height, original_width = tensor.shape[-2:]

        if self.config.pre_pad:
            pad_mode = (
                "reflect"
                if original_height > self.config.pre_pad and original_width > self.config.pre_pad
                else "replicate"
            )
            tensor = F.pad(tensor, (0, self.config.pre_pad, 0, self.config.pre_pad), mode=pad_mode)

        model_input = tensor
        inference_input = tensor
        if self.config.fp16 and self.device.type == "cuda":
            inference_input = tensor.half()
        if self.config.channels_last and self.device.type == "cuda":
            inference_input = inference_input.contiguous(memory_format=torch.channels_last)

        started = time.monotonic()
        with torch.inference_mode():
            native = self.ensemble(inference_input, self._model_call, self.config.native_scale).float()
        ensemble_elapsed = time.monotonic() - started
        if self.config.tta_mode == "x8":
            self.timings.values["tta"] += ensemble_elapsed
        if self.config.shift_ensemble != "none":
            self.timings.values["shift_ensemble"] += ensemble_elapsed

        started = time.monotonic()
        refined = self.back_projection(native, model_input.float())
        self.timings.add("back_projection", started)

        if self.config.pre_pad:
            native_height = original_height * self.config.native_scale
            native_width = original_width * self.config.native_scale
            refined = refined[:, :, :native_height, :native_width]

        final = refined.clamp(0, 1).permute(0, 2, 3, 1).contiguous().cpu().numpy().astype(np.float32)
        return list(final)

    def timing_snapshot(self) -> dict[str, float]:
        return dict(self.timings.values)


class RawVideoWriter:
    """Interface marker; the persistent FFmpeg implementation remains in realesrgan.py."""
