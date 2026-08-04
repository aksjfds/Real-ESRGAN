"""Quality-oriented runtime wrapper for the BasicVSR++ video preprocessor.

The underlying network, checkpoint loader, and tensor normalization remain in
``enhance/basicvsrpp.py``. This wrapper only replaces the spatial tile scheduler,
tile overlap composition, temporal clip composition, and scene-cut score.
"""

from __future__ import annotations

import importlib.util
import math
import sys
import time
from collections import deque
from pathlib import Path
from typing import Deque, Optional, Sequence

import cv2
import numpy as np
import torch


# Load the original implementation under a private module name. ``enhance``
# aliases ``enhance.basicvsrpp`` to this wrapper in its package initializer, so
# importing the original by its public name would recurse.
_LEGACY_PATH = Path(__file__).resolve().with_name("basicvsrpp.py")
_LEGACY_NAME = "enhance._basicvsrpp_legacy"
_spec = importlib.util.spec_from_file_location(_LEGACY_NAME, _LEGACY_PATH)
if _spec is None or _spec.loader is None:  # pragma: no cover - import failure
    raise ImportError(f"Unable to load the BasicVSR++ implementation at {_LEGACY_PATH}")
_legacy = importlib.util.module_from_spec(_spec)
sys.modules[_LEGACY_NAME] = _legacy
_spec.loader.exec_module(_legacy)

# Preserve every public symbol from the original module unless it is replaced
# below. Existing imports and checkpoint-related utilities therefore keep the
# same API.
for _name, _value in vars(_legacy).items():
    if not _name.startswith("_"):
        globals()[_name] = _value


def _axis_starts(length: int, tile_size: int, overlap: int) -> list[int]:
    """Return overlapping starts without a very narrow trailing core tile."""
    if tile_size <= 0 or length <= tile_size:
        return [0]
    overlap = max(1, min(overlap, tile_size - 1))
    stride = tile_size - overlap
    starts = list(range(0, length, stride))
    minimum_tail = max(64, tile_size // 2)
    tail = length - starts[-1]
    if tail < minimum_tail and len(starts) > 1:
        starts[-1] = max(starts[-2] + 1, length - minimum_tail)
    return sorted(set(starts))


def _axis_weight(
    start: int,
    end: int,
    index: int,
    starts: Sequence[int],
    ends: Sequence[int],
) -> np.ndarray:
    """Create complementary cosine ramps for actual neighboring overlaps."""
    size = end - start
    weight = np.ones(size, dtype=np.float32)
    if index > 0:
        left_overlap = max(0, ends[index - 1] - start)
        left_overlap = min(left_overlap, size)
        if left_overlap:
            phase = (np.arange(left_overlap, dtype=np.float32) + 0.5) / left_overlap
            weight[:left_overlap] *= np.sin(phase * (math.pi / 2.0)) ** 2
    if index + 1 < len(starts):
        right_overlap = max(0, end - starts[index + 1])
        right_overlap = min(right_overlap, size)
        if right_overlap:
            phase = (np.arange(right_overlap, dtype=np.float32) + 0.5) / right_overlap
            weight[-right_overlap:] *= np.cos(phase * (math.pi / 2.0)) ** 2
    return weight


class BasicVSRPPPreprocessor(_legacy.BasicVSRPPPreprocessor):
    """BasicVSR++ preprocessor with overlap-add spatial tile composition."""

    def _enhance_tensor(self, clip: torch.Tensor) -> torch.Tensor:
        _n, _t, _c, height, width = clip.shape
        if self.config.tile_size == 0:
            return self._run_model(clip).cpu()

        tile_size = self.config.tile_size
        context = self.config.tile_pad
        blend_overlap = min(max(context * 2, 32), max(1, tile_size // 4))
        x_starts = _axis_starts(width, tile_size, blend_overlap)
        y_starts = _axis_starts(height, tile_size, blend_overlap)
        x_ends = [min(start + tile_size, width) for start in x_starts]
        y_ends = [min(start + tile_size, height) for start in y_starts]
        x_weights = [
            _axis_weight(start, end, index, x_starts, x_ends)
            for index, (start, end) in enumerate(zip(x_starts, x_ends))
        ]
        y_weights = [
            _axis_weight(start, end, index, y_starts, y_ends)
            for index, (start, end) in enumerate(zip(y_starts, y_ends))
        ]

        result = torch.zeros_like(clip, dtype=torch.float32, device="cpu")
        weight_sum = torch.zeros((height, width), dtype=torch.float32, device="cpu")

        for y_index, (y0, y1) in enumerate(zip(y_starts, y_ends)):
            context_y0 = max(0, y0 - context)
            context_y1 = min(height, y1 + context)
            for x_index, (x0, x1) in enumerate(zip(x_starts, x_ends)):
                context_x0 = max(0, x0 - context)
                context_x1 = min(width, x1 + context)
                patch = clip[..., context_y0:context_y1, context_x0:context_x1]
                enhanced = self._run_model(patch).cpu()
                crop_y0 = y0 - context_y0
                crop_x0 = x0 - context_x0
                core_height = y1 - y0
                core_width = x1 - x0
                core = enhanced[
                    ...,
                    crop_y0 : crop_y0 + core_height,
                    crop_x0 : crop_x0 + core_width,
                ]
                spatial_weight = np.multiply.outer(y_weights[y_index], x_weights[x_index])
                weight = torch.from_numpy(spatial_weight).view(1, 1, 1, core_height, core_width)
                result[..., y0:y1, x0:x1] += core.float() * weight
                weight_sum[y0:y1, x0:x1] += weight[0, 0, 0]
                self.tiles += 1

        if not torch.all(weight_sum > 0):
            raise RuntimeError("BasicVSR++ tile blending left uncovered output pixels")
        return result / weight_sum.view(1, 1, 1, height, width)


def scene_difference(previous: np.ndarray, current: np.ndarray) -> float:
    """Scene-cut score combining luminance, color, histogram, and edges."""
    prev = _legacy.frame_to_float_rgb(previous)
    curr = _legacy.frame_to_float_rgb(current)
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
    luma_mad = float(np.mean(np.abs(prev_luma - curr_luma)))
    color_mad = float(np.mean(np.abs(prev_small - curr_small)))

    hist_prev, _ = np.histogram(prev_luma, bins=32, range=(0.0, 1.0), density=False)
    hist_curr, _ = np.histogram(curr_luma, bins=32, range=(0.0, 1.0), density=False)
    hist_prev = hist_prev.astype(np.float64) / max(hist_prev.sum(), 1)
    hist_curr = hist_curr.astype(np.float64) / max(hist_curr.sum(), 1)
    hist_distance = 0.5 * float(np.abs(hist_prev - hist_curr).sum())

    def edge_map(luma: np.ndarray) -> np.ndarray:
        grad_x = cv2.Sobel(luma, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(luma, cv2.CV_32F, 0, 1, ksize=3)
        return np.clip(cv2.magnitude(grad_x, grad_y) / 4.0, 0.0, 1.0)

    edge_mad = float(np.mean(np.abs(edge_map(prev_luma) - edge_map(curr_luma))))
    return 0.35 * luma_mad + 0.20 * color_mad + 0.25 * hist_distance + 0.20 * edge_mad


def _temporal_weights(length: int) -> np.ndarray:
    if length <= 1:
        return np.ones(length, dtype=np.float32)
    # The offset Hann window has no zero endpoints, so segment boundaries and
    # short tail clips remain numerically well-defined after normalization.
    return np.hanning(length + 2)[1:-1].astype(np.float32)


class BasicVSRPPStreamReader:
    """Streaming overlapping clips with temporal overlap-add composition."""

    def __init__(self, reader: object, preprocessor: BasicVSRPPPreprocessor):
        self.reader = reader
        self.preprocessor = preprocessor
        self.clip_length = preprocessor.config.clip_length
        self.overlap = preprocessor.config.clip_overlap
        self.hop = self.clip_length - 2 * self.overlap
        if self.hop < 1:
            raise ValueError("BasicVSR++ clip overlap leaves no temporal hop")

        self.buffer_frames: list[np.ndarray] = []
        self.buffer_indices: list[int] = []
        self.accumulated: dict[int, np.ndarray] = {}
        self.accumulated_weight: dict[int, float] = {}
        self.output: Deque[np.ndarray] = deque()
        self.next_source_index = 0
        self.next_emit_index = 0
        self.last_source_frame: Optional[np.ndarray] = None
        self.eof = False
        self.finished = False
        self.decode_elapsed = 0.0
        self.scene_cuts = 0

    def _read_source(self) -> Optional[np.ndarray]:
        started = time.monotonic()
        frame = self.reader.read()
        self.decode_elapsed += time.monotonic() - started
        return frame

    def _process_window(self, frames: Sequence[np.ndarray], indices: Sequence[int]) -> None:
        enhanced = self.preprocessor.enhance_clip(frames)
        weights = _temporal_weights(len(enhanced))
        for index, frame, weight in zip(indices, enhanced, weights):
            weighted = np.ascontiguousarray(frame, dtype=np.float32) * float(weight)
            if index in self.accumulated:
                self.accumulated[index] += weighted
                self.accumulated_weight[index] += float(weight)
            else:
                self.accumulated[index] = weighted
                self.accumulated_weight[index] = float(weight)

    def _emit_before(self, cutoff: int) -> None:
        while self.next_emit_index < cutoff:
            index = self.next_emit_index
            if index not in self.accumulated:
                raise RuntimeError(f"Missing BasicVSR++ prediction for frame {index}")
            weight = self.accumulated_weight.pop(index)
            if weight <= 0:
                raise RuntimeError(f"Invalid BasicVSR++ temporal weight for frame {index}")
            frame = self.accumulated.pop(index) / weight
            self.output.append(np.ascontiguousarray(np.clip(frame, 0.0, 1.0), dtype=np.float32))
            self.next_emit_index += 1

    def _process_full_windows(self) -> None:
        while len(self.buffer_frames) >= self.clip_length:
            frames = self.buffer_frames[: self.clip_length]
            indices = self.buffer_indices[: self.clip_length]
            self._process_window(frames, indices)
            next_start = indices[0] + self.hop
            del self.buffer_frames[: self.hop]
            del self.buffer_indices[: self.hop]
            self._emit_before(next_start)

    def _finish_segment(self) -> None:
        if self.buffer_frames:
            # A partial tail is only reprocessed when at least one of its frames
            # has not yet appeared in a full clip. This avoids unnecessary
            # duplicate predictions for a fully covered tail.
            if any(index not in self.accumulated for index in self.buffer_indices):
                self._process_window(self.buffer_frames, self.buffer_indices)
            segment_end = self.buffer_indices[-1] + 1
            self._emit_before(segment_end)
        self.buffer_frames.clear()
        self.buffer_indices.clear()
        self.accumulated.clear()
        self.accumulated_weight.clear()
        self.last_source_frame = None

    def read(self) -> Optional[np.ndarray]:
        while not self.output:
            if self.eof:
                if not self.finished:
                    self._finish_segment()
                    self.finished = True
                if not self.output:
                    return None
                break

            frame = self._read_source()
            if frame is None:
                self.eof = True
                continue

            threshold = self.preprocessor.config.scene_threshold
            if (
                self.last_source_frame is not None
                and threshold > 0
                and scene_difference(self.last_source_frame, frame) >= threshold
            ):
                self._finish_segment()
                self.scene_cuts += 1

            index = self.next_source_index
            self.next_source_index += 1
            self.buffer_frames.append(frame)
            self.buffer_indices.append(index)
            self.last_source_frame = frame
            self._process_full_windows()

        return self.output.popleft()

    def close(self) -> None:
        self.reader.close()
        self.preprocessor.close()


__all__ = sorted(
    {
        name
        for name in globals()
        if not name.startswith("_") and name not in {"cv2", "np", "torch", "math", "sys", "time"}
    }
)
