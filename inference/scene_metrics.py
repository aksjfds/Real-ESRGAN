"""CPU-only scene-difference metric shared by clip and interpolation planning."""

from __future__ import annotations

import cv2
import numpy as np


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


def scene_difference(previous: np.ndarray, current: np.ndarray) -> float:
    prev = _frame_to_float_rgb(previous)
    curr = _frame_to_float_rgb(current)
    prev_small = cv2.resize(prev, (64, 64), interpolation=cv2.INTER_AREA)
    curr_small = cv2.resize(curr, (64, 64), interpolation=cv2.INTER_AREA)
    prev_luma = (
        0.2126 * prev_small[..., 0]
        + 0.7152 * prev_small[..., 1]
        + 0.0722 * prev_small[..., 2]
    )
    curr_luma = (
        0.2126 * curr_small[..., 0]
        + 0.7152 * curr_small[..., 1]
        + 0.0722 * curr_small[..., 2]
    )
    mad = float(np.mean(np.abs(prev_luma - curr_luma)))
    hist_prev, _ = np.histogram(prev_luma, bins=32, range=(0.0, 1.0), density=False)
    hist_curr, _ = np.histogram(curr_luma, bins=32, range=(0.0, 1.0), density=False)
    hist_prev = hist_prev.astype(np.float64) / max(hist_prev.sum(), 1)
    hist_curr = hist_curr.astype(np.float64) / max(hist_curr.sum(), 1)
    hist_distance = 0.5 * float(np.abs(hist_prev - hist_curr).sum())
    return 0.7 * mad + 0.3 * hist_distance
