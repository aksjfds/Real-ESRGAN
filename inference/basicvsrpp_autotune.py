"""Runtime autotuning for BasicVSR++ without changing restoration semantics."""

from __future__ import annotations

import json
import math
import time
from collections import deque
from dataclasses import replace
from pathlib import Path
from typing import Deque, Optional, Sequence

import numpy as np
import torch

from . import basicvsrpp as bvsr


_CACHE_VERSION = 1
_MIB = 1024 * 1024
_GIB = 1024 * _MIB
_INSTALLED = False
_SOURCE_WIDTH = 0
_SOURCE_HEIGHT = 0
_ORIGINAL_PREPROCESSOR = bvsr.BasicVSRPPPreprocessor
_ORIGINAL_STREAM_READER = bvsr.BasicVSRPPStreamReader


def _cache_path() -> Path:
    return Path.home() / ".cache" / "realesrgan" / "basicvsrpp-autotune-v1.json"


def _load_cache() -> dict[str, dict[str, object]]:
    path = _cache_path()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {}
    return data if isinstance(data, dict) else {}


def _save_cache(data: dict[str, dict[str, object]]) -> None:
    path = _cache_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        temporary.replace(path)
    except OSError:
        pass


def _tile_candidates(base: int, width: int, height: int) -> list[int]:
    limit = 1536
    if width > 0 and height > 0:
        limit = min(limit, max(width, height))
    values = [base, 640, 768, 896, 1024, 1152, 1280, 1536]
    return sorted({value for value in values if value >= base and (value <= limit or value == base)})


def _clip_candidates(base: int) -> list[int]:
    return [value for value in range(base, min(15, base + 6) + 1, 2)]


def _batch_candidates(total_bytes: int) -> list[int]:
    values = [1, 2]
    if total_bytes >= 24 * _GIB:
        values.append(3)
    return values


def _cache_key(preprocessor: "AutoTunedBasicVSRPPPreprocessor", free_bytes: int, total_bytes: int) -> str:
    device_name = torch.cuda.get_device_name(preprocessor.device)
    free_bucket = free_bytes // (512 * _MIB)
    return "|".join(
        (
            str(_CACHE_VERSION),
            device_name,
            str(total_bytes // _MIB),
            str(free_bucket),
            str(torch.__version__),
            str(torch.version.cuda),
            f"{_SOURCE_WIDTH}x{_SOURCE_HEIGHT}",
            str(preprocessor.config.clip_length),
            str(preprocessor.config.clip_overlap),
            str(preprocessor.config.tile_pad),
            "fp16" if preprocessor.config.fp16 else "fp32",
        )
    )


def _estimated_tiles(tile_size: int) -> int:
    if _SOURCE_WIDTH <= 0 or _SOURCE_HEIGHT <= 0:
        return 1
    return max(1, math.ceil(_SOURCE_WIDTH / tile_size) * math.ceil(_SOURCE_HEIGHT / tile_size))


def _probe_shape(tile_size: int, pad: int) -> tuple[int, int]:
    height = tile_size if _SOURCE_HEIGHT <= 0 else min(tile_size, _SOURCE_HEIGHT)
    width = tile_size if _SOURCE_WIDTH <= 0 else min(tile_size, _SOURCE_WIDTH)
    return max(256, height + 2 * pad), max(256, width + 2 * pad)


class AutoTunedBasicVSRPPPreprocessor(_ORIGINAL_PREPROCESSOR):
    """BasicVSR++ preprocessor that benchmarks safe tile/clip/batch choices once."""

    def __init__(self, config, checkpoint_dir: Path):
        self.clip_batch = 1
        self.autotune_score = 0.0
        self.autotune_peak_mib = 0.0
        super().__init__(config, checkpoint_dir)
        self._autotune()

    def _apply_choice(self, tile_size: int, clip_length: int, clip_batch: int, *, cached: bool) -> None:
        self.config = replace(self.config, tile_size=int(tile_size), clip_length=int(clip_length))
        self.tile_size = int(tile_size)
        self.clip_batch = max(1, int(clip_batch))
        origin = "cache" if cached else "benchmark"
        print(
            f"[basicvsrpp-autotune] cuda:{self.config.gpu_id} {origin} selected "
            f"tile={self.tile_size} clip={self.config.clip_length} batch={self.clip_batch}",
            flush=True,
        )

    def _probe(self, tile_size: int, clip_length: int, clip_batch: int) -> Optional[dict[str, float]]:
        pad = int(self.config.tile_pad)
        probe_h, probe_w = _probe_shape(tile_size, pad)
        emitted = max(1, clip_length - 2 * int(self.config.clip_overlap))
        tiles = _estimated_tiles(tile_size)
        device = self.device

        torch.cuda.set_device(device)
        torch.cuda.empty_cache()
        free_before, total = torch.cuda.mem_get_info(device)
        baseline_reserved = torch.cuda.memory_reserved(device)
        headroom = max(2 * _GIB, int(total * 0.20))
        if free_before <= headroom:
            return None

        seed = torch.zeros((1, 1, 3, probe_h, probe_w), dtype=torch.float32)
        clip = seed.expand(clip_batch, clip_length, -1, -1, -1)
        output = None
        try:
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            output = self._run_model(clip).cpu()
            torch.cuda.synchronize(device)
            elapsed = max(time.perf_counter() - started, 1e-6)
            peak_reserved = torch.cuda.max_memory_reserved(device)
            peak_allocated = torch.cuda.max_memory_allocated(device)
            extra_reserved = max(0, peak_reserved - baseline_reserved)
            safe = free_before - extra_reserved >= headroom
            score = (clip_batch * emitted) / (elapsed * tiles)
            return {
                "score": float(score) if safe else 0.0,
                "elapsed": float(elapsed),
                "peak_reserved": float(peak_reserved),
                "peak_allocated": float(peak_allocated),
                "free_before": float(free_before),
                "total": float(total),
                "safe": 1.0 if safe else 0.0,
            }
        except (torch.cuda.OutOfMemoryError, RuntimeError) as error:
            message = str(error).lower()
            if not isinstance(error, torch.cuda.OutOfMemoryError) and "out of memory" not in message:
                raise
            return None
        finally:
            del output
            del clip
            del seed
            torch.cuda.empty_cache()

    def _autotune(self) -> None:
        device = self.device
        torch.cuda.set_device(device)
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        cache = _load_cache()
        key = _cache_key(self, free_bytes, total_bytes)
        cached = cache.get(key)
        if isinstance(cached, dict):
            try:
                tile = int(cached["tile_size"])
                clip = int(cached["clip_length"])
                batch = int(cached["clip_batch"])
                if tile >= self.config.tile_size and clip >= self.config.clip_length and batch >= 1:
                    self._apply_choice(tile, clip, batch, cached=True)
                    return
            except (KeyError, TypeError, ValueError):
                pass

        base_tile = int(self.config.tile_size)
        base_clip = int(self.config.clip_length)
        best_tile = base_tile
        best_clip = base_clip
        best_batch = 1
        best_score = -1.0
        best_peak = 0.0

        print(
            f"[basicvsrpp-autotune] cuda:{self.config.gpu_id} searching "
            f"tile/clip/batch | free={free_bytes / _GIB:.2f} GiB total={total_bytes / _GIB:.2f} GiB",
            flush=True,
        )

        warm = self._probe(base_tile, base_clip, 1)
        if warm is None:
            print("[basicvsrpp-autotune] baseline probe unavailable; keeping profile defaults", flush=True)
            self._apply_choice(base_tile, base_clip, 1, cached=False)
            return

        no_gain = 0
        for tile in _tile_candidates(base_tile, _SOURCE_WIDTH, _SOURCE_HEIGHT):
            result = self._probe(tile, base_clip, 1)
            if result is None or result["safe"] < 0.5:
                if tile > best_tile:
                    break
                continue
            score = result["score"]
            print(
                f"[basicvsrpp-autotune] probe tile={tile} clip={base_clip} batch=1 "
                f"score={score:.4f} peak={result['peak_reserved'] / _GIB:.2f} GiB",
                flush=True,
            )
            if score > best_score * 1.02:
                best_score = score
                best_tile = tile
                best_peak = result["peak_reserved"]
                no_gain = 0
            elif tile > best_tile:
                no_gain += 1
                if no_gain >= 2:
                    break

        for clip_length in _clip_candidates(base_clip):
            result = self._probe(best_tile, clip_length, 1)
            if result is None or result["safe"] < 0.5:
                break
            score = result["score"]
            print(
                f"[basicvsrpp-autotune] probe tile={best_tile} clip={clip_length} batch=1 "
                f"score={score:.4f} peak={result['peak_reserved'] / _GIB:.2f} GiB",
                flush=True,
            )
            if score > best_score * 1.02:
                best_score = score
                best_clip = clip_length
                best_peak = result["peak_reserved"]

        local_tiles = sorted({base_tile, best_tile, max(base_tile, best_tile - 256), min(1536, best_tile + 256)})
        for tile in local_tiles:
            if tile < base_tile:
                continue
            if _SOURCE_WIDTH > 0 and _SOURCE_HEIGHT > 0 and tile > max(_SOURCE_WIDTH, _SOURCE_HEIGHT):
                continue
            result = self._probe(tile, best_clip, 1)
            if result is None or result["safe"] < 0.5:
                continue
            score = result["score"]
            if score > best_score * 1.02:
                best_score = score
                best_tile = tile
                best_peak = result["peak_reserved"]

        for batch in _batch_candidates(total_bytes)[1:]:
            result = self._probe(best_tile, best_clip, batch)
            if result is None or result["safe"] < 0.5:
                break
            score = result["score"]
            print(
                f"[basicvsrpp-autotune] probe tile={best_tile} clip={best_clip} batch={batch} "
                f"score={score:.4f} peak={result['peak_reserved'] / _GIB:.2f} GiB",
                flush=True,
            )
            if score > best_score * 1.03:
                best_score = score
                best_batch = batch
                best_peak = result["peak_reserved"]
            else:
                break

        self.autotune_score = max(0.0, best_score)
        self.autotune_peak_mib = best_peak / _MIB
        self._apply_choice(best_tile, best_clip, best_batch, cached=False)
        cache[key] = {
            "tile_size": best_tile,
            "clip_length": best_clip,
            "clip_batch": best_batch,
            "score": self.autotune_score,
            "peak_mib": self.autotune_peak_mib,
        }
        _save_cache(cache)

    def enhance_clips(self, clips: Sequence[Sequence[np.ndarray]]) -> list[list[np.ndarray]]:
        """Enhance independent clips in the model's N dimension when shapes match."""
        if not clips:
            return []
        if len(clips) == 1:
            return [super().enhance_clip(clips[0])]
        lengths = {len(frames) for frames in clips}
        if len(lengths) != 1 or next(iter(lengths)) < 2:
            return [super().enhance_clip(frames) for frames in clips]

        first = clips[0][0]
        first_shape = first.shape
        first_dtype = np.dtype(first.dtype)
        for frames in clips:
            if any(frame.shape != first_shape or np.dtype(frame.dtype) != first_dtype for frame in frames):
                return [super().enhance_clip(item) for item in clips]

        originals = np.stack(
            [[bvsr.frame_to_float_rgb(frame) for frame in frames] for frames in clips], axis=0
        )
        tensor = torch.from_numpy(originals).permute(0, 1, 4, 2, 3)
        started = time.monotonic()
        try:
            enhanced = self._enhance_tensor(tensor).permute(0, 1, 3, 4, 2).numpy()
        except RuntimeError as error:
            if "out of gpu memory" not in str(error).lower() and "out of memory" not in str(error).lower():
                raise
            torch.cuda.empty_cache()
            midpoint = max(1, len(clips) // 2)
            return self.enhance_clips(clips[:midpoint]) + self.enhance_clips(clips[midpoint:])
        self.elapsed += time.monotonic() - started
        self.clips += len(clips)
        mixed = originals + self.config.strength * (enhanced - originals)
        return [
            [bvsr.float_rgb_to_dtype(frame, first_dtype) for frame in clip]
            for clip in mixed
        ]


class AutoTunedBasicVSRPPStreamReader:
    """Scene-aware stream that uses the selected per-GPU clip batch."""

    def __init__(self, reader, preprocessor: AutoTunedBasicVSRPPPreprocessor):
        self.reader = reader
        self.preprocessor = preprocessor
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

    def _read_source(self):
        started = time.monotonic()
        frame = self.reader.read()
        self.decode_elapsed += time.monotonic() - started
        return frame

    def _next_task(self):
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
                threshold = float(self.preprocessor.config.scene_threshold)
                if threshold > 0 and bvsr.scene_difference(self.buffer[-1], frame) >= threshold:
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

    def _fill_output(self) -> None:
        tasks = []
        for _ in range(max(1, int(getattr(self.preprocessor, "clip_batch", 1)))):
            task = self._next_task()
            if task is None:
                break
            tasks.append(task)
        if not tasks:
            return
        enhanced_groups = self.preprocessor.enhance_clips([task[0] for task in tasks])
        for task, enhanced in zip(tasks, enhanced_groups):
            _frames, emit_start, emit_end = task
            self.output.extend(enhanced[emit_start:emit_end])
        torch.cuda.set_device(int(self.preprocessor.config.gpu_id))
        torch.cuda.empty_cache()

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
        try:
            self.reader.close()
        finally:
            self.preprocessor.close()


def _balanced_fill_output(self) -> None:
    """Replacement for balanced_pipeline._fill_output with per-GPU clip batches."""
    groups = []
    exhausted = False
    for preprocessor in self.preprocessors:
        tasks = []
        for _ in range(max(1, int(getattr(preprocessor, "clip_batch", 1)))):
            task = self._next_task()
            if task is None:
                exhausted = True
                break
            tasks.append(task)
        if tasks:
            groups.append((preprocessor, tasks))
        if exhausted:
            break

    if not groups:
        return

    def run_group(preprocessor, tasks):
        torch.cuda.set_device(int(preprocessor.config.gpu_id))
        if hasattr(preprocessor, "enhance_clips"):
            return preprocessor.enhance_clips([task[0] for task in tasks])
        return [preprocessor.enhance_clip(task[0]) for task in tasks]

    futures = [
        self.executor.submit(run_group, preprocessor, tasks)
        for preprocessor, tasks in groups
    ]
    for (_preprocessor, tasks), future in zip(groups, futures):
        enhanced_groups = future.result()
        for task, enhanced in zip(tasks, enhanced_groups):
            _frames, emit_start, emit_end = task
            self.output.extend(enhanced[emit_start:emit_end])

    for preprocessor, _tasks in groups:
        try:
            torch.cuda.set_device(int(preprocessor.config.gpu_id))
            torch.cuda.empty_cache()
        except Exception:
            pass


def install_autotune(source_width: int = 0, source_height: int = 0) -> None:
    """Install the autotuned preprocessor/stream into both base and balanced pipelines."""
    global _INSTALLED, _SOURCE_WIDTH, _SOURCE_HEIGHT
    _SOURCE_WIDTH = max(0, int(source_width))
    _SOURCE_HEIGHT = max(0, int(source_height))
    if _INSTALLED:
        return
    _INSTALLED = True

    bvsr.BasicVSRPPPreprocessor = AutoTunedBasicVSRPPPreprocessor
    bvsr.BasicVSRPPStreamReader = AutoTunedBasicVSRPPStreamReader

    try:
        from . import balanced_pipeline
        balanced_pipeline.BalancedBasicVSRPPStreamReader._fill_output = _balanced_fill_output
    except Exception:
        pass
