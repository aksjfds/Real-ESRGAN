"""Balanced multi-GPU scheduler for BasicVSR++ + full-frame Real-ESRGAN.

Profiles A and single-GPU B/C delegate to the existing pipeline unchanged.
For multi-GPU B/C, BasicVSR++ clips are processed in parallel across all
requested GPUs in small ordered batches. Real-ESRGAN then uses all requested
GPUs full-frame. This avoids a fixed 1-GPU producer / N-1-GPU consumer split
when BasicVSR++ is the throughput bottleneck.
"""

from __future__ import annotations

import builtins
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Deque, Optional, Sequence

import numpy as np
import torch

from . import pipeline as base_pipeline


_BALANCED_GPU_IDS: tuple[int, ...] = ()


class _AggregatePreprocessorStats:
    def __init__(self, preprocessors: Sequence[object]) -> None:
        self._preprocessors = preprocessors

    @property
    def elapsed(self) -> float:
        return sum(float(getattr(item, "elapsed", 0.0)) for item in self._preprocessors)

    @property
    def clips(self) -> int:
        return sum(int(getattr(item, "clips", 0)) for item in self._preprocessors)

    @property
    def tiles(self) -> int:
        return sum(int(getattr(item, "tiles", 0)) for item in self._preprocessors)

    @property
    def tile_size(self) -> int:
        values = [int(getattr(item, "tile_size", 0)) for item in self._preprocessors]
        return min((value for value in values if value > 0), default=0)


class BalancedBasicVSRPPStreamReader:
    """Scene-aware BasicVSR++ reader that parallelizes independent clips.

    Adjacent clips keep exactly the same temporal overlap policy as the original
    stream reader. The only change is that up to one clip per requested GPU is
    enhanced concurrently, then emitted in deterministic source order.
    """

    def __init__(self, reader, preprocessor) -> None:
        from . import basicvsrpp as bvsr

        self.reader = reader
        self.clip_length = int(preprocessor.config.clip_length)
        self.overlap = int(preprocessor.config.clip_overlap)
        self.buffer: list[np.ndarray] = []
        self.output: Deque[np.ndarray] = deque()
        self.pending: Optional[np.ndarray] = None
        self.eof = False
        self.segment_end = False
        self.first_chunk = True
        self.decode_elapsed = 0.0
        self.scene_cuts = 0
        self.closed = False

        self.preprocessors = [preprocessor]
        primary_gpu = int(preprocessor.config.gpu_id)
        checkpoint_dir = Path(bvsr.__file__).resolve().parent / "weights"
        if preprocessor.config.model_path:
            checkpoint_path = Path(preprocessor.config.model_path).expanduser().resolve()
        else:
            checkpoint_path = checkpoint_dir / bvsr.BASICVSRPP_TRACK1_URL.rsplit("/", 1)[-1]

        for gpu_id in _BALANCED_GPU_IDS:
            if gpu_id == primary_gpu:
                continue
            cfg = replace(
                preprocessor.config,
                gpu_id=int(gpu_id),
                model_path=str(checkpoint_path),
            )
            try:
                replica = bvsr.BasicVSRPPPreprocessor(cfg, checkpoint_dir=checkpoint_dir)
            except Exception as error:
                builtins.print(
                    f"[basicvsrpp] cuda:{gpu_id} replica unavailable; "
                    f"continuing with {len(self.preprocessors)} restoration GPU(s): {error}",
                    flush=True,
                )
                continue
            self.preprocessors.append(replica)

        self.preprocessor = _AggregatePreprocessorStats(self.preprocessors)
        self.executor = ThreadPoolExecutor(
            max_workers=len(self.preprocessors),
            thread_name_prefix="basicvsrpp-gpu",
        )
        builtins.print(
            "[basicvsrpp] balanced clip workers: "
            + ", ".join(f"cuda:{item.config.gpu_id}" for item in self.preprocessors),
            flush=True,
        )

    def _read_source(self):
        import time

        started = time.monotonic()
        frame = self.reader.read()
        self.decode_elapsed += time.monotonic() - started
        return frame

    def _next_task(self):
        from .basicvsrpp import scene_difference

        while True:
            if self.segment_end or self.eof:
                if self.buffer:
                    frames = list(self.buffer)
                    emit_start = 0 if self.first_chunk else self.overlap
                    emit_end = len(frames)
                    self.buffer = []
                    self.first_chunk = True
                    self.segment_end = False
                    if self.pending is not None:
                        self.buffer.append(self.pending)
                        self.pending = None
                    return frames, emit_start, emit_end
                self.segment_end = False
                if self.pending is not None:
                    self.buffer.append(self.pending)
                    self.pending = None
                    continue
                if self.eof:
                    return None

            frame = self._read_source()
            if frame is None:
                self.eof = True
                continue

            if self.buffer:
                threshold = float(self.preprocessors[0].config.scene_threshold)
                if threshold > 0 and scene_difference(self.buffer[-1], frame) >= threshold:
                    self.pending = frame
                    self.segment_end = True
                    self.scene_cuts += 1
                    continue

            self.buffer.append(frame)
            if len(self.buffer) == self.clip_length:
                frames = list(self.buffer)
                if self.first_chunk:
                    emit_start = 0
                    emit_end = self.clip_length - self.overlap
                    self.first_chunk = False
                else:
                    emit_start = self.overlap
                    emit_end = self.clip_length - self.overlap
                retain = 2 * self.overlap
                self.buffer = self.buffer[-retain:] if retain else []
                return frames, emit_start, emit_end

    @staticmethod
    def _enhance(preprocessor, frames):
        torch.cuda.set_device(int(preprocessor.config.gpu_id))
        return preprocessor.enhance_clip(frames)

    def _fill_output(self) -> None:
        tasks = []
        for _ in self.preprocessors:
            task = self._next_task()
            if task is None:
                break
            tasks.append(task)

        if not tasks:
            return

        futures = [
            self.executor.submit(self._enhance, self.preprocessors[index], task[0])
            for index, task in enumerate(tasks)
        ]
        for task, future in zip(tasks, futures):
            enhanced = future.result()
            _frames, emit_start, emit_end = task
            self.output.extend(enhanced[emit_start:emit_end])

    def read(self) -> Optional[np.ndarray]:
        while not self.output:
            self._fill_output()
            if not self.output:
                return None
        return self.output.popleft()

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.executor.shutdown(wait=True, cancel_futures=True)
        try:
            self.reader.close()
        finally:
            for item in self.preprocessors:
                try:
                    torch.cuda.set_device(int(item.config.gpu_id))
                    item.close()
                except Exception:
                    pass


class SynchronousFrameReader:
    """Compatibility wrapper that intentionally disables background prefetch.

    Balanced BasicVSR++ already occupies all restoration GPUs while it prepares
    a clip batch. Background prefetch would make BasicVSR++ and full-frame SR
    fight for the same devices, so batches are produced on demand instead.
    """

    def __init__(self, reader, depth: int) -> None:
        del depth
        self.reader = reader

    def read(self) -> Optional[np.ndarray]:
        return self.reader.read()

    def close(self) -> None:
        self.reader.close()


def _balanced_profile_settings(args, gpu_ids):
    del args
    profile_name = str(getattr(_balanced_profile_settings, "_profile_name", "A"))
    profile = base_pipeline.SOURCE_PROFILES[profile_name]
    if profile_name == "A":
        return profile_name, profile, gpu_ids, None
    if any(gpu is None for gpu in gpu_ids):
        raise RuntimeError("BasicVSR++ profiles B/C require CUDA")
    return profile_name, profile, gpu_ids, int(gpu_ids[0])


def process_video(args) -> None:
    """Use the original v5 pipeline unless multi-GPU B/C needs balancing."""

    global _BALANCED_GPU_IDS

    profile_name = str(getattr(args, "source_profile", "A")).upper()
    if profile_name == "A":
        base_pipeline.process_video(args)
        return

    gpu_ids = base_pipeline.base.parse_gpu_ids(args.gpu_ids)
    if len(gpu_ids) < 2:
        base_pipeline.process_video(args)
        return
    if any(gpu is None for gpu in gpu_ids):
        raise RuntimeError("BasicVSR++ profiles B/C require CUDA")

    from . import basicvsrpp as bvsr

    _BALANCED_GPU_IDS = tuple(int(gpu) for gpu in gpu_ids)
    _balanced_profile_settings._profile_name = profile_name

    old_profile_settings = base_pipeline._profile_settings
    old_async_reader = base_pipeline.AsyncFrameReader
    old_stream_reader = bvsr.BasicVSRPPStreamReader
    had_print = "print" in base_pipeline.__dict__
    old_print = base_pipeline.__dict__.get("print")

    def pipeline_print(*values, **kwargs):
        if values and isinstance(values[0], str):
            text = values[0]
            if text.startswith("Denoise :") and "dedicated cuda:" in text:
                prefix = text.split("| dedicated cuda:", 1)[0].rstrip()
                values = (
                    prefix
                    + " | balanced clip-parallel GPUs="
                    + ",".join(f"cuda:{gpu}" for gpu in _BALANCED_GPU_IDS),
                    *values[1:],
                )
            elif text == "Pipeline: BasicVSR++ producer overlaps full-frame SR through bounded host prefetch":
                values = (
                    "Pipeline: balanced phases | BasicVSR++ clip-parallel on all GPUs | "
                    "Real-ESRGAN full-frame on all GPUs",
                    *values[1:],
                )
        builtins.print(*values, **kwargs)

    try:
        base_pipeline._profile_settings = _balanced_profile_settings
        base_pipeline.AsyncFrameReader = SynchronousFrameReader
        bvsr.BasicVSRPPStreamReader = BalancedBasicVSRPPStreamReader
        base_pipeline.print = pipeline_print
        base_pipeline.process_video(args)
    finally:
        base_pipeline._profile_settings = old_profile_settings
        base_pipeline.AsyncFrameReader = old_async_reader
        bvsr.BasicVSRPPStreamReader = old_stream_reader
        if had_print:
            base_pipeline.print = old_print
        else:
            base_pipeline.__dict__.pop("print", None)
        _BALANCED_GPU_IDS = ()
