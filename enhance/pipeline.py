"""Resident, modular frame enhancement pipeline.

Internal contract: NCHW RGB tensors in [0, 1]. Model inference may use FP16;
all accumulation, fusion, refinement and final resizing use FP32.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from .ops import (
    BackProjectionRefiner,
    EnsembleEngine,
    adaptive_dehalo,
    lanczos_resize,
    soft_range_compress,
)
from .srvgg_enhanced import EnhancedSRVGGNetCompact, adaptive_residual_strength


@dataclass(frozen=True)
class PipelineConfig:
    preprocess_backend: str = "none"
    preprocess_strength: float = 1.0
    preprocess_model_path: str = ""
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
    cugan_ensemble: bool = False
    cugan_model_path: str = ""
    cugan_alpha: float = 1.0
    cugan_global_weight: float = 0.25
    cugan_mask_mode: str = "adaptive"
    back_projection_iterations: int = 0
    back_projection_strength: float = 0.2
    back_projection_kernel: str = "lanczos"
    back_projection_clamp: float = 0.05
    native_scale: int = 4
    final_scale: float = 2.0
    tile_size: int = 0
    tile_pad: int = 10
    pre_pad: int = 0
    fp16: bool = True
    channels_last: bool = True
    dehalo_strength: float = 0.0
    dehalo_radius: int = 2
    range_limit: float = 0.0
    range_radius: int = 2
    overshoot: float = 1.0
    undershoot: float = 1.0


class StageTimings:
    def __init__(self) -> None:
        self.values: dict[str, float] = {
            name: 0.0
            for name in (
                "preprocess",
                "tta",
                "shift_ensemble",
                "realesrgan",
                "residual_control",
                "base_correction",
                "realcugan",
                "ensemble_fusion",
                "back_projection",
                "lanczos",
                "dehalo",
                "range_limit",
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


class Preprocessor:
    VERIFIED_BACKENDS = {"none"}

    def __init__(self, backend: str, strength: float, model_path: str):
        self.backend = backend
        self.strength = strength
        self.model_path = model_path
        if backend not in self.VERIFIED_BACKENDS:
            raise RuntimeError(
                f"Preprocessor {backend!r} is unavailable in this build. Its official repository/API and "
                "checkpoint have not been installed and verified; no substitute will be used."
            )

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x


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
            # Convolutions may run in FP16, but all branch recombination and
            # adaptive strength arithmetic is intentionally FP32.
            output = components.nearest_base.float() + components.residual.float() * strength
            self.timings.add("residual_control", started)
            lanczos_base = None
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


class RealCUGANBackend:
    def __init__(self, enabled: bool, model_path: str):
        self.enabled = enabled
        self.model_path = model_path
        if enabled:
            raise RuntimeError(
                "Real-CUGAN ensemble is unavailable: no verified resident 4x PyTorch backend/checkpoint "
                "was provided. Real-ESRGAN will not be silently replaced."
            )


class AnimePostProcessor:
    """Anime4K shaders are applied by the persistent FFmpeg writer, never here."""


class FrameEnhancementPipeline:
    def __init__(self, model: torch.nn.Module, device: torch.device, config: PipelineConfig):
        self.model = model
        self.device = device
        self.config = config
        self.timings = StageTimings()
        self.preprocessor = Preprocessor(
            config.preprocess_backend, config.preprocess_strength, config.preprocess_model_path
        )
        self.realesrgan = RealESRGANBackend(model, config, self.timings)
        self.realcugan = RealCUGANBackend(config.cugan_ensemble, config.cugan_model_path)
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
        if not frames:
            return []
        if any(frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3 for frame in frames):
            raise TypeError("Decoded inputs must be uint8 HWC RGB")
        rgb = np.stack(frames)
        tensor = torch.from_numpy(rgb).permute(0, 3, 1, 2).to(self.device, non_blocking=True)
        tensor = tensor.float().div_(255.0)
        started = time.monotonic()
        processed = self.preprocessor(tensor)
        self.timings.add("preprocess", started)
        model_input = processed
        inference_input = processed
        if self.config.fp16 and self.device.type == "cuda":
            inference_input = processed.half()
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

        target_size = (
            round(model_input.shape[-2] * self.config.final_scale),
            round(model_input.shape[-1] * self.config.final_scale),
        )
        started = time.monotonic()
        final = refined if self.config.final_scale == self.config.native_scale else lanczos_resize(refined, target_size)
        self.timings.add("lanczos", started)

        started = time.monotonic()
        final = adaptive_dehalo(final, self.config.dehalo_strength, self.config.dehalo_radius)
        self.timings.add("dehalo", started)
        started = time.monotonic()
        if self.config.range_limit > 0:
            reference = lanczos_resize(model_input.float(), target_size)
            final = soft_range_compress(
                final,
                reference,
                self.config.range_limit,
                self.config.range_radius,
                self.config.overshoot,
                self.config.undershoot,
            )
        self.timings.add("range_limit", started)
        final = final.clamp(0, 1).permute(0, 2, 3, 1).contiguous().cpu().numpy().astype(np.float32)
        return list(final)

    def timing_snapshot(self) -> dict[str, float]:
        return dict(self.timings.values)


class RawVideoWriter:
    """Interface marker; the persistent FFmpeg implementation remains in realesrgan.py."""
