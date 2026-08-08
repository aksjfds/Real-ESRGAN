"""Conservative source-constrained dark-line restoration for anime SR.

This module never sharpens the whole image. It estimates thin dark structures
from the source frame, aligns that contrast map to the super-resolved frame,
and only restores contrast that AnimeVideo-v3 appears to have attenuated.
When confidence is low the baseline SR pixel is returned unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import cv2
import numpy as np


_EPS = 1e-6
_KR = 0.2126
_KG = 0.7152
_KB = 0.0722


@dataclass(frozen=True)
class LineRestoreConfig:
    enabled: bool = True
    strength: float = 1.0
    max_contrast_recovery: float = 0.18
    max_darkening: float = 0.10
    min_reference_contrast: float = 0.025
    edge_threshold: float = 0.010
    orientation_floor: float = 0.55
    source_close_radius: int = 2
    guide_radius: int = 4
    guide_eps: float = 1e-4


def _luma(rgb: np.ndarray) -> np.ndarray:
    return (
        rgb[..., 0] * _KR
        + rgb[..., 1] * _KG
        + rgb[..., 2] * _KB
    ).astype(np.float32, copy=False)


def _smoothstep(value: np.ndarray, low: float, high: float) -> np.ndarray:
    if high <= low:
        return (value >= high).astype(np.float32)
    x = np.clip((value - low) / (high - low), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _closing_base(y: np.ndarray, radius: int) -> np.ndarray:
    radius = max(1, int(radius))
    size = radius * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    return cv2.morphologyEx(y, cv2.MORPH_CLOSE, kernel)


def _dark_contrast(y: np.ndarray, radius: int) -> Tuple[np.ndarray, np.ndarray]:
    base = _closing_base(y, radius)
    contrast = np.maximum((base - y) / np.maximum(base, _EPS), 0.0)
    return contrast.astype(np.float32), base.astype(np.float32)


def _gradient(y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    gx = cv2.Sobel(y, cv2.CV_32F, 1, 0, ksize=3, scale=0.125)
    gy = cv2.Sobel(y, cv2.CV_32F, 0, 1, ksize=3, scale=0.125)
    mag = np.sqrt(gx * gx + gy * gy)
    return gx, gy, mag


def _box_mean(array: np.ndarray, radius: int) -> np.ndarray:
    size = max(1, int(radius)) * 2 + 1
    return cv2.boxFilter(
        array,
        cv2.CV_32F,
        (size, size),
        normalize=True,
        borderType=cv2.BORDER_REFLECT101,
    )


def guided_filter(
    guide: np.ndarray,
    signal: np.ndarray,
    radius: int,
    eps: float,
) -> np.ndarray:
    """Single-channel guided filter with ``guide`` as structural guidance."""
    guide = np.ascontiguousarray(guide, dtype=np.float32)
    signal = np.ascontiguousarray(signal, dtype=np.float32)
    mean_i = _box_mean(guide, radius)
    mean_p = _box_mean(signal, radius)
    corr_i = _box_mean(guide * guide, radius)
    corr_ip = _box_mean(guide * signal, radius)
    var_i = np.maximum(corr_i - mean_i * mean_i, 0.0)
    cov_ip = corr_ip - mean_i * mean_p
    a = cov_ip / (var_i + max(float(eps), 1e-8))
    b = mean_p - a * mean_i
    return _box_mean(a, radius) * guide + _box_mean(b, radius)


def _resize_float(array: np.ndarray, width: int, height: int) -> np.ndarray:
    if array.shape[1] == width and array.shape[0] == height:
        return np.ascontiguousarray(array, dtype=np.float32)
    return cv2.resize(array, (width, height), interpolation=cv2.INTER_LINEAR).astype(
        np.float32, copy=False
    )


def restore_dark_lines(
    source_rgb: np.ndarray,
    sr_rgb: np.ndarray,
    config: LineRestoreConfig,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Restore only source-supported dark-line contrast in an SR frame.

    The operation is intentionally one-sided: pixels can only be darkened, never
    brightened, and only where source/SR edge direction agrees. RGB channels are
    scaled together so hue is not rotated by the correction.
    """
    baseline = np.ascontiguousarray(sr_rgb, dtype=np.float32)
    if not config.enabled:
        return baseline, {"modified_fraction": 0.0, "mean_darkening": 0.0, "max_darkening": 0.0}
    if baseline.ndim != 3 or baseline.shape[2] != 3:
        raise ValueError(f"Expected SR RGB HWC frame, got {baseline.shape}")
    source = np.ascontiguousarray(source_rgb, dtype=np.float32)
    if source.ndim != 3 or source.shape[2] != 3:
        raise ValueError(f"Expected source RGB HWC frame, got {source.shape}")
    if not np.isfinite(source).all() or not np.isfinite(baseline).all():
        raise ValueError("Non-finite values are not accepted by line restoration")

    # Measurements operate in display-referred [0, 1] RGB, while the returned
    # image keeps the unclipped SR baseline outside corrected pixels.
    source_measure = np.clip(source, 0.0, 1.0)
    sr_measure = np.clip(baseline, 0.0, 1.0)
    source_y = _luma(source_measure)
    sr_y = _luma(sr_measure)
    target_h, target_w = sr_y.shape

    source_contrast, _ = _dark_contrast(source_y, config.source_close_radius)
    scale = max(target_w / max(source_y.shape[1], 1), target_h / max(source_y.shape[0], 1))
    target_radius = max(2, int(round(config.source_close_radius * scale)))
    sr_contrast, sr_base = _dark_contrast(sr_y, target_radius)

    # Jointly upsample the source dark-line map and let the SR luminance place
    # sub-pixel boundaries. This avoids blindly injecting source high-frequency.
    reference = _resize_float(source_contrast, target_w, target_h)
    reference = guided_filter(
        sr_y,
        reference,
        radius=max(1, int(round(config.guide_radius * max(scale / 2.0, 1.0)))),
        eps=config.guide_eps,
    )
    reference = np.clip(reference, 0.0, 1.0)

    src_gx, src_gy, src_mag = _gradient(source_y)
    sr_gx, sr_gy, sr_mag = _gradient(sr_y)
    src_gx = _resize_float(src_gx, target_w, target_h)
    src_gy = _resize_float(src_gy, target_w, target_h)
    src_mag = _resize_float(src_mag, target_w, target_h)

    denom = np.maximum(src_mag * sr_mag, _EPS)
    orientation = np.abs((src_gx * sr_gx + src_gy * sr_gy) / denom)
    orientation = np.clip(orientation, 0.0, 1.0)

    ref_conf = _smoothstep(
        reference,
        config.min_reference_contrast,
        config.min_reference_contrast * 4.0,
    )
    src_edge_conf = _smoothstep(
        src_mag,
        config.edge_threshold,
        config.edge_threshold * 3.0,
    )
    sr_edge_conf = _smoothstep(
        sr_mag,
        config.edge_threshold,
        config.edge_threshold * 3.0,
    )
    orient_conf = _smoothstep(orientation, config.orientation_floor, 1.0)
    confidence = ref_conf * src_edge_conf * sr_edge_conf * orient_conf

    deficit = np.maximum(reference - sr_contrast, 0.0)
    recovery = np.minimum(deficit, config.max_contrast_recovery)
    recovery *= np.clip(config.strength, 0.0, 2.0) * confidence

    darkening = np.minimum(sr_base * recovery, config.max_darkening).astype(np.float32)
    y_out = np.maximum(sr_y - darkening, 0.0)
    ratio = np.ones_like(sr_y, dtype=np.float32)
    valid = sr_y > 1e-5
    ratio[valid] = np.clip(y_out[valid] / sr_y[valid], 0.0, 1.0)

    restored = baseline * ratio[..., None]
    restored = np.ascontiguousarray(restored, dtype=np.float32)

    changed = darkening > 1e-5
    stats = {
        "modified_fraction": float(np.mean(changed)),
        "mean_darkening": float(np.mean(darkening[changed])) if np.any(changed) else 0.0,
        "max_darkening": float(np.max(darkening)) if darkening.size else 0.0,
        "mean_confidence": float(np.mean(confidence[changed])) if np.any(changed) else 0.0,
    }
    return restored, stats
