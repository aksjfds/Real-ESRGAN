"""Quality-oriented runtime wrapper for the BasicVSR++ video preprocessor.

The underlying network, checkpoint loader, and tensor normalization remain in
``enhance/basicvsrpp.py``. This wrapper replaces only spatial tile scheduling,
overlap composition, temporal clip composition, scene-cut scoring, and device
scheduling. Model topology, checkpoint parameters, clip geometry, and blend
weights are unchanged.
"""

from __future__ import annotations

import importlib.util
import math
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Deque, Optional, Sequence

import cv2
import numpy as np
import torch


_LEGACY_PATH = Path(__file__).resolve().with_name("basicvsrpp.py")
_LEGACY_NAME = "enhance._basicvsrpp_legacy"
_spec = importlib.util.spec_from_file_location(_LEGACY_NAME, _LEGACY_PATH)
if _spec is None or _spec.loader is None:  # pragma: no cover
    raise ImportError(f"Unable to load the BasicVSR++ implementation at {_LEGACY_PATH}")
_legacy = importlib.util.module_from_spec(_spec)
sys.modules[_LEGACY_NAME] = _legacy
_spec.loader.exec_module(_legacy)

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
    """Multi-GPU BasicVSR++ with GPU-resident overlap-add tile composition."""

    def __init__(
        self,
        config: object,
        checkpoint_dir: Optional[Path] = None,
        model: Optional[torch.nn.Module] = None,
        gpu_ids: Optional[Sequence[int]] = None,
    ):
        super().__init__(config, checkpoint_dir=checkpoint_dir, model=model)

        if model is not None or self.device.type != "cuda":
            selected = [self.device.index] if self.device.type == "cuda" else []
        elif gpu_ids is None:
            selected = list(range(torch.cuda.device_count()))
        else:
            selected = [int(gpu_id) for gpu_id in gpu_ids]
        selected = list(dict.fromkeys(gpu_id for gpu_id in selected if gpu_id is not None))
        primary = int(self.device.index) if self.device.index is not None else 0
        if primary in selected:
            selected.remove(primary)
        selected.insert(0, primary)
        for gpu_id in selected:
            if gpu_id < 0 or gpu_id >= torch.cuda.device_count():
                raise ValueError(
                    f"BasicVSR++ requested cuda:{gpu_id}, but {torch.cuda.device_count()} GPU(s) are visible"
                )

        self._extra_runners: list[_legacy.BasicVSRPPPreprocessor] = []
        for gpu_id in selected[1:]:
            replica_config = replace(config, gpu_id=gpu_id)
            replica = _legacy.BasicVSRPPPreprocessor(
                replica_config,
                checkpoint_dir=checkpoint_dir,
            )
            self._extra_runners.append(replica)
        self._runners: list[_legacy.BasicVSRPPPreprocessor] = [self, *self._extra_runners]
        self._executor = (
            ThreadPoolExecutor(max_workers=len(self._runners), thread_name_prefix="basicvsrpp-gpu")
            if len(self._runners) > 1
            else None
        )
        self._device_weight_cache: dict[tuple[object, ...], torch.Tensor] = {}
        self._closed = False

        self.model_gpu_seconds = 0.0
        self.gpu_blend_seconds = 0.0
        self.d2h_seconds = 0.0
        self.cpu_reduce_seconds = 0.0
        self.parallel_wall_seconds = 0.0
        print(
            "[basicvsrpp] spatial tile devices="
            + ",".join(f"cuda:{runner.device.index}" for runner in self._runners),
            flush=True,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=False)
        for runner in self._extra_runners:
            runner.close()
        print(
            "[basicvsrpp timing] "
            f"parallel_wall={self.parallel_wall_seconds:.1f}s, "
            f"model_gpu_sum={self.model_gpu_seconds:.1f}s, "
            f"gpu_blend_sum={self.gpu_blend_seconds:.1f}s, "
            f"d2h={self.d2h_seconds:.1f}s, "
            f"cpu_reduce={self.cpu_reduce_seconds:.1f}s",
            flush=True,
        )
        super().close()

    def _device_weight(
        self,
        runner: _legacy.BasicVSRPPPreprocessor,
        key: tuple[object, ...],
        spatial_weight: np.ndarray,
    ) -> torch.Tensor:
        cache_key = (runner.device.index, *key)
        weight = self._device_weight_cache.get(cache_key)
        if weight is None:
            weight = torch.from_numpy(spatial_weight).to(
                runner.device,
                dtype=torch.float32,
                non_blocking=True,
            ).view(1, 1, 1, spatial_weight.shape[0], spatial_weight.shape[1])
            self._device_weight_cache[cache_key] = weight
        return weight

    def _run_lane(
        self,
        runner: _legacy.BasicVSRPPPreprocessor,
        clip: torch.Tensor,
        jobs: Sequence[tuple[object, ...]],
    ) -> tuple[torch.Tensor, torch.Tensor, float, float, float]:
        _n, _t, _c, height, width = clip.shape
        device = runner.device
        result = torch.zeros(clip.shape, dtype=torch.float32, device=device)
        weight_sum = torch.zeros((height, width), dtype=torch.float32, device=device)
        model_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        lane_start = torch.cuda.Event(enable_timing=True)
        lane_end = torch.cuda.Event(enable_timing=True)

        with torch.cuda.device(device):
            lane_start.record()
            for job in jobs:
                (
                    tile_index,
                    y_index,
                    x_index,
                    y0,
                    y1,
                    x0,
                    x1,
                    context_y0,
                    context_y1,
                    context_x0,
                    context_x1,
                    spatial_weight,
                ) = job
                patch = clip[..., context_y0:context_y1, context_x0:context_x1]
                model_start = torch.cuda.Event(enable_timing=True)
                model_end = torch.cuda.Event(enable_timing=True)
                model_start.record()
                enhanced = runner._run_model(patch)
                model_end.record()
                model_events.append((model_start, model_end))

                crop_y0 = y0 - context_y0
                crop_x0 = x0 - context_x0
                core_height = y1 - y0
                core_width = x1 - x0
                core = enhanced[
                    ...,
                    crop_y0 : crop_y0 + core_height,
                    crop_x0 : crop_x0 + core_width,
                ]
                weight = self._device_weight(
                    runner,
                    (height, width, tile_index, y_index, x_index),
                    spatial_weight,
                )
                result[..., y0:y1, x0:x1].add_(core.float() * weight)
                weight_sum[y0:y1, x0:x1].add_(weight[0, 0, 0])
            lane_end.record()
            torch.cuda.synchronize(device)

        lane_gpu_seconds = lane_start.elapsed_time(lane_end) / 1000.0
        model_seconds = sum(start.elapsed_time(end) for start, end in model_events) / 1000.0
        blend_seconds = max(0.0, lane_gpu_seconds - model_seconds)

        transfer_started = time.monotonic()
        result_cpu = result.cpu()
        weight_cpu = weight_sum.cpu()
        d2h_seconds = time.monotonic() - transfer_started
        return result_cpu, weight_cpu, model_seconds, blend_seconds, d2h_seconds

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

        jobs: list[tuple[object, ...]] = []
        tile_index = 0
        for y_index, (y0, y1) in enumerate(zip(y_starts, y_ends)):
            context_y0 = max(0, y0 - context)
            context_y1 = min(height, y1 + context)
            for x_index, (x0, x1) in enumerate(zip(x_starts, x_ends)):
                context_x0 = max(0, x0 - context)
                context_x1 = min(width, x1 + context)
                spatial_weight = np.multiply.outer(y_weights[y_index], x_weights[x_index])
                jobs.append(
                    (
                        tile_index,
                        y_index,
                        x_index,
                        y0,
                        y1,
                        x0,
                        x1,
                        context_y0,
                        context_y1,
                        context_x0,
                        context_x1,
                        spatial_weight,
                    )
                )
                tile_index += 1
        self.tiles += len(jobs)

        lanes: list[list[tuple[object, ...]]] = [[] for _ in self._runners]
        lane_loads = [0 for _ in self._runners]
        for job in sorted(
            jobs,
            key=lambda item: (item[8] - item[7]) * (item[10] - item[9]),
            reverse=True,
        ):
            lane_index = min(range(len(lanes)), key=lane_loads.__getitem__)
            lanes[lane_index].append(job)
            lane_loads[lane_index] += (job[8] - job[7]) * (job[10] - job[9])

        wall_started = time.monotonic()
        if self._executor is None:
            lane_results = [self._run_lane(self._runners[0], clip, lanes[0])]
        else:
            futures = [
                self._executor.submit(self._run_lane, runner, clip, lane_jobs)
                for runner, lane_jobs in zip(self._runners, lanes)
                if lane_jobs
            ]
            lane_results = [future.result() for future in futures]
        self.parallel_wall_seconds += time.monotonic() - wall_started

        reduce_started = time.monotonic()
        result = torch.zeros(clip.shape, dtype=torch.float32, device="cpu")
        weight_sum = torch.zeros((height, width), dtype=torch.float32, device="cpu")
        for partial, partial_weight, model_seconds, blend_seconds, d2h_seconds in lane_results:
            result.add_(partial)
            weight_sum.add_(partial_weight)
            self.model_gpu_seconds += model_seconds
            self.gpu_blend_seconds += blend_seconds
            self.d2h_seconds += d2h_seconds
        if not torch.all(weight_sum > 0):
            raise RuntimeError("BasicVSR++ tile blending left uncovered output pixels")
        result.div_(weight_sum.view(1, 1, 1, height, width))
        self.cpu_reduce_seconds += time.monotonic() - reduce_started
        return result


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
        if not name.startswith("_")
        and name
        not in {
            "cv2",
            "np",
            "torch",
            "math",
            "sys",
            "time",
            "ThreadPoolExecutor",
        }
    }
)
