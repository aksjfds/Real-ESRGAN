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
        return self.tile_pad + self.pre_pad

    def split(self, frame: np.ndarray) -> tuple[list[np.ndarray], list[TileRegion]]:
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
        output_width = round(input_width * self.scale)
        output_height = round(input_height * self.scale)
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
        return result


def verify_dimensions(width: int, height: int, scale: float) -> tuple[int, int]:
    return round(width * scale), round(height * scale)
