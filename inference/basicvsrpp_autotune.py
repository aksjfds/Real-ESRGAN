"""Fast quality-constrained autotuning for BasicVSR++.

The selected source profile defines restoration quality. The tuner never changes
strength, temporal overlap, tile padding, scene-cut behavior, or clip length.
It only searches execution parameters that should preserve those semantics:
spatial tile size (never below the profile baseline) and, on large-memory GPUs,
independent clip batch size.
"""

from __future__ import annotations

import json
import math
import time
from collections import deque
from pathlib import Path
from typing import Deque, Optional, Sequence

import numpy as np
import torch

from . import basicvsrpp as bvsr


_CACHE_VERSION = 3
_MIB = 1024 * 1024
_GIB = 1024 * _MIB
_INSTALLED = False
_SOURCE_WIDTH = 0
_SOURCE_HEIGHT = 0
_ORIGINAL_PREPROCESSOR = bvsr.BasicVSRPPPreprocessor
_ORIGINAL_STREAM_READER = bvsr.BasicVSRPPStreamReader


def _cache_path() -> Path:
    return Path.home() / ".cache" / "realesrgan" / "basicvsrpp-autotune-v3.json"


def _load_cache() -> dict[str, dict[str, object]]:
    try:
        data = json.loads(_cache_path().read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {}
    return data if isinstance(data, dict) else {}


def _save_cache(data: dict[str, dict[str, object]]) -> None:
    path = _cache_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        temporary.replace(path)
    except OSError:
        pass


def _round4(value: float | int) -> int:
    return max(256, int(math.ceil(float(value) / 4.0) * 4))


def _source_shape(tile_size: int) -> tuple[int, int]:
    if _SOURCE_WIDTH > 0 and _SOURCE_HEIGHT > 0:
        return _SOURCE_HEIGHT, _SOURCE_WIDTH
    return tile_size, tile_size


def _tile_count(tile_size: int) -> int:
    height, width = _source_shape(tile_size)
    return max(1, math.ceil(width / tile_size) * math.ceil(height / tile_size))


def _fast_tile_candidates(base: int, width: int, height: int) -> list[int]:
    """Keep only a few source-specific topology representatives.

    Autotuning startup time matters. Rather than benchmarking every size, build
    several topology breakpoints and retain at most four: the baseline plus the
    three candidates that reduce the number of source-frame tiles the most.
    """
    if width <= 0 or height <= 0:
        return [base, 768, 1024][:3]

    long_edge = max(width, height)
    short_edge = min(width, height)
    limit = min(1536, long_edge)
    raw = {
        base,
        _round4(short_edge / 2),
        _round4(long_edge / 4),
        _round4(long_edge / 3),
        _round4(long_edge / 2),
        _round4(short_edge),
        limit,
    }
    raw = {value for value in raw if base <= value <= limit}

    representatives: dict[int, int] = {}
    for value in sorted(raw):
        count = _tile_count(value)
        representatives.setdefault(count, value)

    candidates = {base}
    ranked = sorted(
        ((count, value) for count, value in representatives.items() if value != base),
        key=lambda item: (item[0], item[1]),
    )
    candidates.update(value for _count, value in ranked[:3])
    return sorted(candidates)


def _cache_key(preprocessor: "AutoTunedBasicVSRPPPreprocessor", total_bytes: int) -> str:
    return "|".join(
        (
            str(_CACHE_VERSION),
            torch.cuda.get_device_name(preprocessor.device),
            str(total_bytes // _MIB),
            str(torch.__version__),
            str(torch.version.cuda),
            f"{_SOURCE_WIDTH}x{_SOURCE_HEIGHT}",
            str(preprocessor.config.clip_length),
            str(preprocessor.config.clip_overlap),
            str(preprocessor.config.tile_size),
            str(preprocessor.config.tile_pad),
            str(preprocessor.config.scene_threshold),
            f"{float(preprocessor.config.strength):.6f}",
            "fp16" if preprocessor.config.fp16 else "fp32",
        )
    )


class AutoTunedBasicVSRPPPreprocessor(_ORIGINAL_PREPROCESSOR):
    """Choose the fastest quality-preserving BasicVSR++ execution parameters."""

    def __init__(self, config, checkpoint_dir: Path):
        self.clip_batch = 1
        self.autotune_score = 0.0
        self.autotune_peak_mib = 0.0
        self._profile_tile_floor = int(config.tile_size)
        self._profile_clip = int(config.clip_length)
        self._profile_strength = float(config.strength)
        self._profile_overlap = int(config.clip_overlap)
        self._profile_tile_pad = int(config.tile_pad)
        self._profile_scene_threshold = float(config.scene_threshold)
        super().__init__(config, checkpoint_dir)
        self._autotune()

    def _headroom(self, total_bytes: int) -> int:
        return max(2 * _GIB, int(total_bytes * 0.20))

    def _quality_guard(self, tile_size: int, clip_batch: int) -> None:
        if tile_size < self._profile_tile_floor:
            raise RuntimeError("autotuner must not reduce the BasicVSR++ tile baseline")
        if clip_batch < 1:
            raise RuntimeError("autotuner clip batch must be positive")
        if int(self.config.clip_length) != self._profile_clip:
            raise RuntimeError("autotuner changed BasicVSR++ clip length")
        if float(self.config.strength) != self._profile_strength:
            raise RuntimeError("autotuner changed BasicVSR++ strength")
        if int(self.config.clip_overlap) != self._profile_overlap:
            raise RuntimeError("autotuner changed BasicVSR++ clip overlap")
        if int(self.config.tile_pad) != self._profile_tile_pad:
            raise RuntimeError("autotuner changed BasicVSR++ tile padding")
        if float(self.config.scene_threshold) != self._profile_scene_threshold:
            raise RuntimeError("autotuner changed BasicVSR++ scene-cut threshold")

    def _apply_choice(
        self,
        tile_size: int,
        clip_batch: int,
        *,
        cached: bool,
        score: float = 0.0,
        peak_mib: float = 0.0,
    ) -> None:
        self._quality_guard(tile_size, clip_batch)
        self.tile_size = int(tile_size)
        self.clip_batch = int(clip_batch)
        self.autotune_score = float(score)
        self.autotune_peak_mib = float(peak_mib)
        tag = "cache" if cached else "selected"
        print(
            f"[autotuner] {tag} | cuda:{self.config.gpu_id} | "
            f"tile={self.tile_size} clip={self._profile_clip} batch={self.clip_batch}"
            + (f" | {self.autotune_score:.4f} effective-fps" if self.autotune_score > 0 else ""),
            flush=True,
        )

    def _warmup(self) -> None:
        """One small untimed pass to avoid lazy CUDA initialization bias."""
        device = self.device
        torch.cuda.set_device(device)
        probe = torch.zeros((1, min(3, self._profile_clip), 3, 320, 320), dtype=torch.float32)
        output = None
        try:
            output = self._run_model(probe).cpu()
            torch.cuda.synchronize(device)
        finally:
            del output
            del probe

    def _probe_full_frame(
        self,
        tile_size: int,
        clip_length: int,
        clip_batch: int,
    ) -> Optional[dict[str, float]]:
        self._quality_guard(tile_size, clip_batch)
        device = self.device
        torch.cuda.set_device(device)
        free_before, total = torch.cuda.mem_get_info(device)
        headroom = self._headroom(total)
        if free_before <= headroom:
            return None

        height, width = _source_shape(tile_size)
        clip = torch.zeros((clip_batch, clip_length, 3, height, width), dtype=torch.float32)
        output = None
        old_tiles = self.tiles
        try:
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            output = self._enhance_with_tile(clip, tile_size)
            torch.cuda.synchronize(device)
            elapsed = max(time.perf_counter() - started, 1e-6)
            peak_reserved = torch.cuda.max_memory_reserved(device)
            peak_allocated = torch.cuda.max_memory_allocated(device)
            free_after, _ = torch.cuda.mem_get_info(device)
            safe = free_after >= headroom
            emitted = max(1, clip_length - 2 * self._profile_overlap)
            score = (clip_batch * emitted) / elapsed
            return {
                "score": float(score) if safe else 0.0,
                "elapsed": float(elapsed),
                "peak_reserved": float(peak_reserved),
                "peak_allocated": float(peak_allocated),
                "safe": 1.0 if safe else 0.0,
            }
        except (torch.cuda.OutOfMemoryError, RuntimeError) as error:
            message = str(error).lower()
            if not isinstance(error, torch.cuda.OutOfMemoryError) and "out of memory" not in message:
                raise
            torch.cuda.empty_cache()
            return None
        finally:
            self.tiles = old_tiles
            del output
            del clip

    def _cached_choice_is_safe(self, cached: dict[str, object], total_bytes: int) -> bool:
        try:
            peak_bytes = int(float(cached.get("peak_mib", 0.0)) * _MIB)
        except (TypeError, ValueError):
            return False
        free_now, _ = torch.cuda.mem_get_info(self.device)
        reserved_now = torch.cuda.memory_reserved(self.device)
        extra_needed = max(0, peak_bytes - reserved_now)
        return free_now - extra_needed >= self._headroom(total_bytes)

    def _autotune(self) -> None:
        device = self.device
        torch.cuda.set_device(device)
        _free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        cache = _load_cache()
        key = _cache_key(self, total_bytes)
        cached = cache.get(key)

        if isinstance(cached, dict) and self._cached_choice_is_safe(cached, total_bytes):
            try:
                self._apply_choice(
                    int(cached["tile_size"]),
                    int(cached["clip_batch"]),
                    cached=True,
                    score=float(cached.get("effective_fps", 0.0)),
                    peak_mib=float(cached.get("peak_mib", 0.0)),
                )
                return
            except (KeyError, TypeError, ValueError, RuntimeError):
                pass

        candidates = _fast_tile_candidates(
            self._profile_tile_floor,
            _SOURCE_WIDTH,
            _SOURCE_HEIGHT,
        )
        print(
            f"[autotuner] search | cuda:{self.config.gpu_id} | "
            f"{_SOURCE_WIDTH or '?'}x{_SOURCE_HEIGHT or '?'} | "
            f"clip={self._profile_clip} | tiles={','.join(map(str, candidates))}",
            flush=True,
        )

        self._warmup()

        quick_clip = min(self._profile_clip, 5)
        quick_results: list[tuple[float, int]] = []
        quick_text: list[str] = []
        for tile in candidates:
            result = self._probe_full_frame(tile, quick_clip, 1)
            if result is None or result["safe"] < 0.5:
                quick_text.append(f"{tile}=OOM")
                continue
            rank_score = 1.0 / max(result["elapsed"], 1e-6)
            quick_results.append((rank_score, tile))
            quick_text.append(f"{tile}={rank_score:.3f}")
        print("[autotuner] quick | " + "  ".join(quick_text), flush=True)

        if not quick_results:
            self._apply_choice(self._profile_tile_floor, 1, cached=False)
            return

        quick_results.sort(reverse=True)
        verify_tiles = [tile for _score, tile in quick_results[:2]]

        quick_map = {tile: score for score, tile in quick_results}
        baseline_score = quick_map.get(self._profile_tile_floor, 0.0)
        second_score = quick_results[min(1, len(quick_results) - 1)][0]
        if (
            self._profile_tile_floor not in verify_tiles
            and baseline_score >= second_score * 0.90
        ):
            verify_tiles.append(self._profile_tile_floor)

        best_tile = self._profile_tile_floor
        best_score = -1.0
        best_peak = 0.0
        verify_text: list[str] = []
        for tile in verify_tiles:
            result = self._probe_full_frame(tile, self._profile_clip, 1)
            if result is None or result["safe"] < 0.5:
                verify_text.append(f"{tile}=OOM")
                continue
            score = result["score"]
            verify_text.append(f"{tile}={score:.4f}")
            if score > best_score:
                best_score = score
                best_tile = tile
                best_peak = result["peak_reserved"]
        print("[autotuner] verify | " + "  ".join(verify_text), flush=True)

        if best_score < 0:
            self._apply_choice(self._profile_tile_floor, 1, cached=False)
            return

        best_batch = 1

        if total_bytes >= 24 * _GIB:
            batch_result = self._probe_full_frame(best_tile, self._profile_clip, 2)
            if batch_result is not None and batch_result["safe"] >= 0.5:
                batch_score = batch_result["score"]
                print(f"[autotuner] batch | 1={best_score:.4f}  2={batch_score:.4f}", flush=True)
                if batch_score > best_score * 1.02:
                    best_score = batch_score
                    best_batch = 2
                    best_peak = batch_result["peak_reserved"]

                    if total_bytes >= 48 * _GIB:
                        batch3 = self._probe_full_frame(best_tile, self._profile_clip, 3)
                        if batch3 is not None and batch3["safe"] >= 0.5:
                            batch3_score = batch3["score"]
                            print(
                                f"[autotuner] batch3 | 2={best_score:.4f}  3={batch3_score:.4f}",
                                flush=True,
                            )
                            if batch3_score > best_score * 1.02:
                                best_score = batch3_score
                                best_batch = 3
                                best_peak = batch3["peak_reserved"]

        peak_mib = best_peak / _MIB
        self._apply_choice(
            best_tile,
            best_batch,
            cached=False,
            score=best_score,
            peak_mib=peak_mib,
        )
        cache[key] = {
            "tile_size": best_tile,
            "clip_batch": best_batch,
            "effective_fps": best_score,
            "peak_mib": peak_mib,
        }
        _save_cache(cache)

    def enhance_clips(self, clips: Sequence[Sequence[np.ndarray]]) -> list[list[np.ndarray]]:
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
            [[bvsr.frame_to_float_rgb(frame) for frame in frames] for frames in clips],
            axis=0,
        )
        tensor = torch.from_numpy(originals).permute(0, 1, 4, 2, 3)
        started = time.monotonic()
        try:
            enhanced = self._enhance_tensor(tensor).permute(0, 1, 3, 4, 2).numpy()
        except RuntimeError as error:
            message = str(error).lower()
            if "out of gpu memory" not in message and "out of memory" not in message:
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

    def release_unused_cache_if_tight(self) -> None:
        device = self.device
        torch.cuda.set_device(device)
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        if free_bytes < self._headroom(total_bytes):
            torch.cuda.empty_cache()


class AutoTunedBasicVSRPPStreamReader:
    """Scene-aware stream that uses the selected independent-clip batch."""

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
        self.preprocessor.release_unused_cache_if_tight()

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
            if hasattr(preprocessor, "release_unused_cache_if_tight"):
                preprocessor.release_unused_cache_if_tight()
        except Exception:
            pass


def install_autotune(source_width: int = 0, source_height: int = 0) -> None:
    """Install BasicVSR++ autotuning without changing Real-ESRGAN execution."""
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
