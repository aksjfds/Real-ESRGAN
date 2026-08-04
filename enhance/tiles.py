"""Official-style context tiles with single-write native-scale stitching."""

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
    def __init__(self, tile_size: int, tile_pad: int, scale: float, verify: bool = False):
        self.tile_size = tile_size
        self.tile_pad = tile_pad
        self.scale = scale
        self.verify = verify

    def split(self, frame: np.ndarray) -> tuple[list[np.ndarray], list[TileRegion]]:
        height, width = frame.shape[:2]
        if self.tile_size == 0:
            return [np.ascontiguousarray(frame)], [TileRegion(0, 0, 0, width, height, 0, 0)]

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
                patch = np.ascontiguousarray(frame[cy0:cy1, cx0:cx1])
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
        output_width = round(input_width * self.scale)
        output_height = round(input_height * self.scale)
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

        expected = (round(input_height * self.scale), round(input_width * self.scale), 3)
        if result.shape != expected:
            raise RuntimeError(f"Stitched frame shape mismatch: {result.shape} != {expected}")
        return np.ascontiguousarray(result, dtype=np.float32)


def full_frame_lanczos(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize one stitched FP32 RGB frame exactly once with Lanczos4."""
    if frame.dtype != np.float32 or frame.ndim != 3 or frame.shape[2] != 3:
        raise TypeError("full_frame_lanczos expects float32 HWC RGB")
    if frame.shape[:2] == (height, width):
        return np.ascontiguousarray(frame, dtype=np.float32)
    resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LANCZOS4)
    return np.ascontiguousarray(resized, dtype=np.float32)


def verify_dimensions(width: int, height: int, scale: float) -> tuple[int, int]:
    return round(width * scale), round(height * scale)
