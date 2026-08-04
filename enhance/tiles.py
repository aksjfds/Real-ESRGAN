"""Context-padded tiles with single-write stitching (no overlap averaging)."""

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
    context: int


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

    @property
    def context(self) -> int:
        return self.tile_pad

    def _global_pad(self, frame: np.ndarray) -> np.ndarray:
        if self.pre_pad <= 0:
            return frame
        border = cv2.BORDER_REFLECT_101 if min(frame.shape[:2]) > 1 else cv2.BORDER_REPLICATE
        return cv2.copyMakeBorder(
            frame,
            self.pre_pad,
            self.pre_pad,
            self.pre_pad,
            self.pre_pad,
            border,
        )

    def split(self, frame: np.ndarray) -> tuple[list[np.ndarray], list[TileRegion]]:
        frame = self._global_pad(frame)
        height, width = frame.shape[:2]
        if self.tile_size == 0:
            return [np.ascontiguousarray(frame)], [TileRegion(0, 0, 0, width, height, 0)]
        patches: list[np.ndarray] = []
        regions: list[TileRegion] = []
        index = 0
        context = self.context
        for y0 in axis_starts(height, self.tile_size):
            for x0 in axis_starts(width, self.tile_size):
                y1 = min(y0 + self.tile_size, height)
                x1 = min(x0 + self.tile_size, width)
                cy0, cx0 = max(0, y0 - context), max(0, x0 - context)
                cy1, cx1 = min(height, y1 + context), min(width, x1 + context)
                patch = frame[cy0:cy1, cx0:cx1]
                top, left = context - (y0 - cy0), context - (x0 - cx0)
                bottom, right = context - (cy1 - y1), context - (cx1 - x1)
                if top or bottom or left or right:
                    border = cv2.BORDER_REFLECT_101 if min(patch.shape[:2]) > 1 else cv2.BORDER_REPLICATE
                    patch = cv2.copyMakeBorder(patch, top, bottom, left, right, border)
                expected = (y1 - y0 + 2 * context, x1 - x0 + 2 * context)
                if patch.shape[:2] != expected:
                    raise RuntimeError(f"Tile context shape mismatch: {patch.shape[:2]} != {expected}")
                patches.append(np.ascontiguousarray(patch))
                regions.append(TileRegion(index, x0, y0, x1, y1, context))
                index += 1
        return patches, regions

    def stitch(
        self,
        outputs: Mapping[int, np.ndarray],
        regions: Sequence[TileRegion],
        input_width: int,
        input_height: int,
    ) -> np.ndarray:
        padded_width = input_width + 2 * self.pre_pad
        padded_height = input_height + 2 * self.pre_pad
        output_width = round(padded_width * self.scale)
        output_height = round(padded_height * self.scale)
        result = np.empty((output_height, output_width, 3), dtype=np.float32)
        coverage = np.zeros((output_height, output_width), dtype=np.uint8) if self.verify else None
        for region in regions:
            ox0, ox1 = round(region.x0 * self.scale), round(region.x1 * self.scale)
            oy0, oy1 = round(region.y0 * self.scale), round(region.y1 * self.scale)
            local_x0 = round(region.context * self.scale)
            local_y0 = round(region.context * self.scale)
            width, height = ox1 - ox0, oy1 - oy0
            tile = outputs[region.index]
            cropped = tile[local_y0 : local_y0 + height, local_x0 : local_x0 + width]
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
        if self.pre_pad:
            x0 = round(self.pre_pad * self.scale)
            y0 = round(self.pre_pad * self.scale)
            x1 = x0 + round(input_width * self.scale)
            y1 = y0 + round(input_height * self.scale)
            result = result[y0:y1, x0:x1]
        expected = (round(input_height * self.scale), round(input_width * self.scale), 3)
        if result.shape != expected:
            raise RuntimeError(f"Global pre-pad crop mismatch: {result.shape} != {expected}")
        return result


def full_frame_lanczos(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize one stitched FP32 RGB frame exactly once with Lanczos4."""
    if frame.dtype != np.float32 or frame.ndim != 3 or frame.shape[2] != 3:
        raise TypeError("full_frame_lanczos expects float32 HWC RGB")
    if frame.shape[:2] == (height, width):
        return frame
    resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LANCZOS4)
    return np.ascontiguousarray(resized, dtype=np.float32)


def full_frame_dehalo(frame: np.ndarray, strength: float, radius: int) -> np.ndarray:
    if strength <= 0:
        return frame
    kernel = radius * 2 + 1
    smooth = cv2.blur(frame, (kernel, kernel), borderType=cv2.BORDER_REFLECT_101)
    luma = 0.2126 * frame[..., 0] + 0.7152 * frame[..., 1] + 0.0722 * frame[..., 2]
    luma_smooth = cv2.blur(luma, (3, 3), borderType=cv2.BORDER_REFLECT_101)
    edge = np.abs(luma - luma_smooth)
    halo = np.abs(frame - smooth).mean(axis=2)
    mask = np.clip((halo - 0.01) / 0.04, 0, 1) * np.clip((edge - 0.01) / 0.08, 0, 1)
    return np.ascontiguousarray(frame - (frame - smooth) * mask[..., None] * min(strength, 1.0), dtype=np.float32)


def full_frame_range_limit(
    frame: np.ndarray,
    reference_lr: np.ndarray,
    strength: float,
    radius: int,
    overshoot: float,
    undershoot: float,
) -> np.ndarray:
    if strength <= 0:
        return frame
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


def verify_dimensions(width: int, height: int, scale: float) -> tuple[int, int]:
    return round(width * scale), round(height * scale)
