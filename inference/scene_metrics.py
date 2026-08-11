"""CPU-only scene signatures shared by clip and interpolation planning."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class SceneSignature:
    luma: np.ndarray
    histogram: np.ndarray


def _frame_to_float_rgb(frame: np.ndarray) -> np.ndarray:
    if frame.dtype == np.uint8:
        return np.ascontiguousarray(frame.astype(np.float32) / 255.0)
    if frame.dtype.kind == "u" and frame.dtype.itemsize == 2:
        return np.ascontiguousarray(frame.astype(np.float32) / 65535.0)
    if frame.dtype == np.float32:
        if not np.isfinite(frame).all():
            raise ValueError("Scene metric input contains NaN or Inf")
        return np.ascontiguousarray(np.clip(frame, 0.0, 1.0), dtype=np.float32)
    raise TypeError(f"Unsupported scene metric dtype: {frame.dtype}")


def scene_signature(frame: np.ndarray) -> SceneSignature:
    """Compute the exact 64x64 representation used by the existing metric."""
    rgb = _frame_to_float_rgb(frame)
    small = cv2.resize(rgb, (64, 64), interpolation=cv2.INTER_AREA)
    luma = (
        0.2126 * small[..., 0]
        + 0.7152 * small[..., 1]
        + 0.0722 * small[..., 2]
    )
    histogram, _ = np.histogram(
        luma,
        bins=32,
        range=(0.0, 1.0),
        density=False,
    )
    histogram = histogram.astype(np.float64) / max(histogram.sum(), 1)
    return SceneSignature(
        luma=np.ascontiguousarray(luma),
        histogram=histogram,
    )


def scene_difference_from_signatures(
    previous: SceneSignature,
    current: SceneSignature,
) -> float:
    mad = float(np.mean(np.abs(previous.luma - current.luma)))
    hist_distance = 0.5 * float(
        np.abs(previous.histogram - current.histogram).sum()
    )
    return 0.7 * mad + 0.3 * hist_distance


def scene_difference(previous: np.ndarray, current: np.ndarray) -> float:
    return scene_difference_from_signatures(
        scene_signature(previous),
        scene_signature(current),
    )
