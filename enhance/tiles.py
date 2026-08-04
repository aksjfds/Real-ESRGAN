"""Official-style context tiles with single-write native-scale stitching.

The tile core stride equals ``tile_size``.  Context is clipped at the image
boundary, matching Real-ESRGAN's official ``tile_process`` behavior; no overlap
averaging or synthetic reflect context is introduced at the outer image edge.
``pre_pad`` is a separate global right/bottom pad, also matching the official
implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import cv2
import numpy as np


@dataclass(frozen=True)
class TileRegion:
    index: int
    x0: int
    y0: int
    x1: int
    y1: int
    crop_x0: int
    crop_y0: int


def axis_starts(length: int, tile_size: int) -> list[int]:
    if tile_size <= 0 or length <= tile_size:
        return [0]
    return list(range(0, length, tile_size))


class TileProcessor:
    def __init__(self, tile_size: int, tile_pad: int, pre_pad: int, scale: float, verify: bool = False):
        self.tile_size = tile_size
        self.tile_pad = tile_pad
        self.pre_pad = pre_pad
        self.scale = scale
        self.verify = verify

    def _global_pad(self, frame: np.ndarray) -> np.ndarray:
        if self.pre_pad <= 0:
            return frame
        border = cv2.BORDER_REFLECT_101 if min(frame.shape[:2]) > self.pre_pad else cv2.BORDER_REPLICATE
        return cv2.copyMakeBorder(frame, 0, self.pre_pad, 0, self.pre_pad, border)

    def split(self, frame: np.ndarray) -> tuple[list[np.ndarray], list[TileRegion]]:
        padded = self._global_pad(frame)
        height, width = padded.shape[:2]
        if self.tile_size == 0:
            return [np.ascontiguousarray(padded)], [TileRegion(0, 0, 0, width, height, 0, 0)]

        patches: list[np.ndarray] = []
        regions: list[TileRegion] = []
        index = 0
        for y0 in axis_starts(height, self.tile_size):
            for x0 in axis_starts(width, self.tile_size):
                y1 = min(y0 + self.tile_size, height)
                x1 = min(x0 + self.tile_size, width)
                cy0 = max(y0 - self.tile_pad, 0)
                cx0 = max(x0 - self.tile_pad, 0)
                cy1 = min(y1 + self.tile_pad, height)
                cx1 = min(x1 + self.tile_pad, width)
                patch = np.ascontiguousarray(padded[cy0:cy1, cx0:cx1])
                patches.append(patch)
                regions.append(
                    TileRegion(
                        index=index,
                        x0=x0,
                        y0=y0,
                        x1=x1,
                        y1=y1,
                        crop_x0=round((x0 - cx0) * self.scale),
                        crop_y0=round((y0 - cy0) * self.scale),
                    )
                )
                index += 1
        return patches, regions

    def stitch(
        self,
        outputs: Mapping[int, np.ndarray],
        regions: Sequence[TileRegion],
        input_width: int,
        input_height: int,
    ) -> np.ndarray:
        padded_width = input_width + self.pre_pad
        padded_height = input_height + self.pre_pad
        output_width = round(padded_width * self.scale)
        output_height = round(padded_height * self.scale)
        result = np.empty((output_height, output_width, 3), dtype=np.float32)
        coverage = np.zeros((output_height, output_width), dtype=np.uint8) if self.verify else None

        for region in regions:
            ox0, ox1 = round(region.x0 * self.scale), round(region.x1 * self.scale)
            oy0, oy1 = round(region.y0 * self.scale), round(region.y1 * self.scale)
            width, height = ox1 - ox0, oy1 - oy0
            tile = outputs[region.index]
            cropped = tile[
                region.crop_y0 : region.crop_y0 + height,
                region.crop_x0 : region.crop_x0 + width,
            ]
            if cropped.shape[:2] != (height, width):
                raise RuntimeError(
                    f"Tile {region.index} crop mismatch: {cropped.shape[:2]} != {(height, width)}"
                )
            result[oy0:oy1, ox0:ox1] = cropped
            if coverage is not None:
                coverage[oy0:oy1, ox0:ox1] += 1

        if coverage is not None and not np.all(coverage == 1):
            unique, counts = np.unique(coverage, return_counts=True)
            raise RuntimeError(f"Tile coverage must be exactly one: {dict(zip(unique.tolist(), counts.tolist()))}")

        # Official pre_pad is appended at right/bottom, so remove it in native
        # coordinates by retaining the top-left original image extent.
        result = result[: round(input_height * self.scale), : round(input_width * self.scale)]
        expected = (round(input_height * self.scale), round(input_width * self.scale), 3)
        if result.shape != expected:
            raise RuntimeError(f"Global pre-pad crop mismatch: {result.shape} != {expected}")
        return np.ascontiguousarray(result, dtype=np.float32)


def full_frame_lanczos(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize one stitched FP32 RGB frame exactly once with Lanczos4."""
    if frame.dtype != np.float32 or frame.ndim != 3 or frame.shape[2] != 3:
        raise TypeError("full_frame_lanczos expects float32 HWC RGB")
    if frame.shape[:2] == (height, width):
        return np.ascontiguousarray(frame, dtype=np.float32)
    resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LANCZOS4)
    return np.ascontiguousarray(resized, dtype=np.float32)


def full_frame_dehalo(frame: np.ndarray, strength: float, radius: int) -> np.ndarray:
    if strength <= 0:
        return np.ascontiguousarray(frame, dtype=np.float32)
    kernel = radius * 2 + 1
    smooth = cv2.blur(frame, (kernel, kernel), borderType=cv2.BORDER_REFLECT_101)
    luma = 0.2126 * frame[..., 0] + 0.7152 * frame[..., 1] + 0.0722 * frame[..., 2]
    luma_smooth = cv2.blur(luma, (3, 3), borderType=cv2.BORDER_REFLECT_101)
    edge = np.abs(luma - luma_smooth)
    halo = np.abs(frame - smooth).mean(axis=2)
    mask = np.clip((halo - 0.01) / 0.04, 0, 1) * np.clip((edge - 0.01) / 0.08, 0, 1)
    corrected = frame - (frame - smooth) * mask[..., None] * min(strength, 1.0)
    return np.ascontiguousarray(corrected, dtype=np.float32)


def full_frame_range_limit(
    frame: np.ndarray,
    reference_lr: np.ndarray,
    strength: float,
    radius: int,
    overshoot: float,
    undershoot: float,
) -> np.ndarray:
    if strength <= 0:
        return np.ascontiguousarray(frame, dtype=np.float32)
    reference = full_frame_lanczos(reference_lr, frame.shape[1], frame.shape[0])
    kernel = np.ones((radius * 2 + 1, radius * 2 + 1), np.uint8)
    local_min = cv2.erode(reference, kernel, borderType=cv2.BORDER_REFLECT_101)
    local_max = cv2.dilate(reference, kernel, borderType=cv2.BORDER_REFLECT_101)
    mean = cv2.blur(reference, kernel.shape, borderType=cv2.BORDER_REFLECT_101)
    variance = cv2.blur(reference * reference, kernel.shape, borderType=cv2.BORDER_REFLECT_101) - mean * mean
    std = np.sqrt(np.maximum(variance, 0))
    lower = local_min - undershoot * std
    upper = local_max + overshoot * std
    softness = np.maximum(std * 0.5 + 1.0 / 255.0, 1.0 / 65535.0)
    below = lower - softness * np.tanh((lower - frame) / softness)
    above = upper + softness * np.tanh((frame - upper) / softness)
    compressed = np.where(frame < lower, below, np.where(frame > upper, above, frame))
    return np.ascontiguousarray(frame + min(strength, 1.0) * (compressed - frame), dtype=np.float32)


def finalize_frame(
    native_frame: np.ndarray,
    reference_lr: np.ndarray,
    output_width: int,
    output_height: int,
    dehalo_strength: float,
    dehalo_radius: int,
    range_strength: float,
    range_radius: int,
    overshoot: float,
    undershoot: float,
) -> np.ndarray:
    """Shared parent-side finalization for full-frame and tiled inference."""
    output = full_frame_lanczos(native_frame, output_width, output_height)
    output = full_frame_dehalo(output, dehalo_strength, dehalo_radius)
    output = full_frame_range_limit(
        output,
        reference_lr,
        range_strength,
        range_radius,
        overshoot,
        undershoot,
    )
    return np.ascontiguousarray(np.clip(output, 0.0, 1.0), dtype=np.float32)


def verify_dimensions(width: int, height: int, scale: float) -> tuple[int, int]:
    return round(width * scale), round(height * scale)
