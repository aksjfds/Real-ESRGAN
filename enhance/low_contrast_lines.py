"""Selective low-contrast line enhancement for float RGB video frames.

The enhancer works only on luma. A weak-edge mask selects coherent low-gradient
structures, while a dilated hard mask protects strong edges and their immediate
neighborhood. Pixels outside the final mask are copied without modification.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


_LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
_SCHARR_NORMALIZER = 32.0


@dataclass(frozen=True)
class LowContrastLineConfig:
    enabled: bool = True
    strength: float = 1.0
    gradient_min: float = 0.004
    gradient_max: float = 0.025
    protect_gradient: float = 0.040
    coherence_min: float = 0.45
    guided_radius: int = 4
    guided_eps: float = 4.0e-4
    temporal: float = 0.65
    max_delta: float = 0.025

    def validate(self) -> None:
        if self.strength < 0:
            raise ValueError("line strength must be non-negative")
        if not 0 <= self.gradient_min < self.gradient_max < self.protect_gradient:
            raise ValueError(
                "line gradients must satisfy 0 <= min < max < protect"
            )
        if not 0 <= self.coherence_min <= 1:
            raise ValueError("line coherence must be in [0,1]")
        if self.guided_radius < 1:
            raise ValueError("line guided radius must be at least 1")
        if self.guided_eps <= 0:
            raise ValueError("line guided epsilon must be positive")
        if not 0 <= self.temporal < 1:
            raise ValueError("line temporal smoothing must be in [0,1)")
        if self.max_delta <= 0:
            raise ValueError("line max delta must be positive")


def _smoothstep(edge0: float, edge1: float, value: np.ndarray) -> np.ndarray:
    if edge1 <= edge0:
        raise ValueError("smoothstep requires edge1 > edge0")
    x = np.clip((value - edge0) / (edge1 - edge0), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _guided_base(luma: np.ndarray, radius: int, eps: float) -> np.ndarray:
    """Self-guided grayscale filter using the original local-linear equations."""
    kernel = (2 * radius + 1, 2 * radius + 1)
    mean = cv2.boxFilter(
        luma, -1, kernel, normalize=True, borderType=cv2.BORDER_REFLECT101
    )
    corr = cv2.boxFilter(
        luma * luma,
        -1,
        kernel,
        normalize=True,
        borderType=cv2.BORDER_REFLECT101,
    )
    variance = np.maximum(corr - mean * mean, 0.0)
    a = variance / (variance + eps)
    b = mean * (1.0 - a)
    mean_a = cv2.boxFilter(
        a, -1, kernel, normalize=True, borderType=cv2.BORDER_REFLECT101
    )
    mean_b = cv2.boxFilter(
        b, -1, kernel, normalize=True, borderType=cv2.BORDER_REFLECT101
    )
    return mean_a * luma + mean_b


def _gradient_and_coherence(luma: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    smoothed = cv2.GaussianBlur(
        luma, (0, 0), sigmaX=0.65, sigmaY=0.65, borderType=cv2.BORDER_REFLECT101
    )
    gx = cv2.Scharr(smoothed, cv2.CV_32F, 1, 0) / _SCHARR_NORMALIZER
    gy = cv2.Scharr(smoothed, cv2.CV_32F, 0, 1) / _SCHARR_NORMALIZER
    gradient = cv2.magnitude(gx, gy)

    jxx = cv2.GaussianBlur(gx * gx, (0, 0), 1.0)
    jyy = cv2.GaussianBlur(gy * gy, (0, 0), 1.0)
    jxy = cv2.GaussianBlur(gx * gy, (0, 0), 1.0)
    anisotropy = np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy * jxy)
    coherence = anisotropy / (jxx + jyy + 1.0e-8)
    return gradient, np.clip(coherence, 0.0, 1.0)


class LowContrastLineEnhancer:
    """Stateful frame enhancer with conservative temporal mask smoothing."""

    def __init__(self, config: LowContrastLineConfig):
        config.validate()
        self.config = config
        self.previous_mask: Optional[np.ndarray] = None
        self.previous_preview: Optional[np.ndarray] = None
        self.frames = 0
        self.changed_fraction_sum = 0.0
        self.max_observed_delta = 0.0

    def reset(self) -> None:
        self.previous_mask = None
        self.previous_preview = None

    def _scene_changed(self, luma: np.ndarray) -> bool:
        preview = cv2.resize(luma, (64, 36), interpolation=cv2.INTER_AREA)
        changed = False
        if self.previous_preview is not None:
            changed = float(np.mean(np.abs(preview - self.previous_preview))) > 0.12
        self.previous_preview = preview
        return changed

    def _mask(self, luma: np.ndarray) -> np.ndarray:
        gradient, coherence = _gradient_and_coherence(luma)
        cfg = self.config

        enter = _smoothstep(cfg.gradient_min, cfg.gradient_max, gradient)
        leave = 1.0 - _smoothstep(
            cfg.gradient_max, cfg.protect_gradient, gradient
        )
        coherent = _smoothstep(
            cfg.coherence_min,
            min(1.0, cfg.coherence_min + 0.20),
            coherence,
        )
        weak = enter * leave * coherent

        kernel = np.ones((3, 3), dtype=np.uint8)
        support = cv2.dilate(weak, kernel, iterations=1)
        local_max = cv2.dilate(luma, np.ones((5, 5), dtype=np.uint8))
        local_min = cv2.erode(luma, np.ones((5, 5), dtype=np.uint8))
        local_range = local_max - local_min
        strong = np.logical_or(
            gradient >= cfg.protect_gradient,
            local_range >= 2.0 * cfg.protect_gradient,
        ).astype(np.uint8)
        strong = cv2.dilate(strong, kernel, iterations=1).astype(np.float32)
        current = support * (1.0 - strong)

        if self._scene_changed(luma):
            self.previous_mask = None
        if self.previous_mask is not None and self.previous_mask.shape == current.shape:
            retained = np.minimum(self.previous_mask, support)
            current = (1.0 - cfg.temporal) * current + cfg.temporal * retained
            current *= 1.0 - strong
        self.previous_mask = current.copy()
        return np.clip(current, 0.0, 1.0)

    def enhance(self, frame: np.ndarray) -> np.ndarray:
        if not self.config.enabled:
            return frame
        if frame.dtype != np.float32 or frame.ndim != 3 or frame.shape[2] != 3:
            raise TypeError("LowContrastLineEnhancer expects float32 HWC RGB")
        if not np.isfinite(frame).all():
            raise ValueError("Non-finite frame passed to low-contrast line enhancer")

        luma = np.tensordot(frame, _LUMA, axes=([2], [0])).astype(np.float32)
        mask = self._mask(luma)
        base = _guided_base(
            luma, self.config.guided_radius, self.config.guided_eps
        )
        detail = luma - base
        delta = self.config.strength * mask * detail
        delta = np.clip(delta, -self.config.max_delta, self.config.max_delta)

        min_rgb = frame.min(axis=2)
        max_rgb = frame.max(axis=2)
        delta = np.minimum(delta, 1.0 - max_rgb)
        delta = np.maximum(delta, -min_rgb)

        changed = np.abs(delta) > 1.0e-7
        self.frames += 1
        self.changed_fraction_sum += float(np.mean(changed))
        if changed.any():
            self.max_observed_delta = max(
                self.max_observed_delta, float(np.max(np.abs(delta[changed])))
            )
        if not changed.any():
            return frame

        output = frame.copy()
        output += delta[..., None]
        return np.ascontiguousarray(output, dtype=np.float32)

    def summary(self) -> str:
        average = self.changed_fraction_sum / max(1, self.frames)
        return (
            f"frames={self.frames}, mean_changed_pixels={100.0 * average:.3f}%, "
            f"max_luma_delta={self.max_observed_delta:.6f}"
        )


__all__ = ["LowContrastLineConfig", "LowContrastLineEnhancer"]
