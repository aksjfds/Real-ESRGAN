"""v5.1 quality-preserving pipeline optimizations.

This module only changes execution and transfer paths. It does not alter the
selected BasicVSR++ profile, Real-ESRGAN model/full-frame inference, Lanczos4,
or encoding parameters.
"""

from __future__ import annotations

import multiprocessing as mp
from multiprocessing import shared_memory
import queue
import threading
import time
import traceback
from collections import deque
from dataclasses import asdict
from pathlib import Path
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.nn import functional as F

from . import pipeline as pipeline
from . import runtime as base


_MIB = 1024 * 1024
_TIMEOUT = 300.0
_PIPELINE_INSTALLED = False
_BVSR_INSTALLED = False


def _format_eta(seconds: float) -> str:
    if not np.isfinite(seconds) or seconds < 0:
        return "--:--"
    value = int(round(seconds))
    hours, value = divmod(value, 3600)
    minutes, secs = divmod(value, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


class StableOutputPump(pipeline.OutputPump):
    """Output pump with one stable throughput/ETA instead of two rate estimates."""

    def __init__(self, workers, writer, width: int, height: int, progress, started: float):
        self.first_output_at: Optional[float] = None
        self.rate_history: Deque[Tuple[int, float]] = deque()
        # Do not show tqdm's short-window rate/remaining estimate. The postfix
        # below is based on completed output frames and a much longer window.
        progress.bar_format = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}, {postfix}]"
        super().__init__(workers, writer, width, height, progress, started)

    def _stable_rate(self, now: float) -> float:
        self.rate_history.append((self.processed, now))
        while len(self.rate_history) > 2 and now - self.rate_history[0][1] > 120.0:
            self.rate_history.popleft()
        if len(self.rate_history) >= 2:
            old_count, old_time = self.rate_history[0]
            span = now - old_time
            if span >= 20.0 and self.processed > old_count:
                return (self.processed - old_count) / span
        if self.first_output_at is None:
            return 0.0
        return max(0.0, (self.processed - 1) / max(now - self.first_output_at, 1e-6))

    def _run(self) -> None:
        try:
            while True:
                item = self.queue.get()
                if item is None:
                    self.queue.task_done()
                    break
                _frame_id, worker_id = item
                try:
                    t = time.monotonic()
                    frame = pipeline._resize(self.workers.output(worker_id), self.width, self.height)
                    self.resize_seconds += time.monotonic() - t
                    t = time.monotonic()
                    self.writer.write(frame)
                    self.write_seconds += time.monotonic() - t
                    self.processed += 1
                    now = time.monotonic()
                    if self.first_output_at is None:
                        self.first_output_at = now
                    self.progress.update(1)
                    rate = self._stable_rate(now)
                    total = self.progress.total or 0
                    remaining = max(0, int(total) - self.processed)
                    eta = remaining / rate if rate > 1e-9 else float("inf")
                    self.progress.set_postfix_str(
                        f"{rate:.3f} frame/s | ETA {_format_eta(eta)}",
                        refresh=False,
                    )
                finally:
                    self.workers.release(worker_id)
                    self.queue.task_done()
        except Exception as error:
            self.error = error
            self.traceback_text = traceback.format_exc()


def _infer_cuda_u8_tensor(model: torch.nn.Module, frame: np.ndarray, device: torch.device) -> torch.Tensor:
    """Match runtime.infer_frame's uint8 math but keep the result on CUDA."""
    tensor = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).to(device, non_blocking=True)
    tensor = tensor.half()
    tensor.div_(255.0)
    tensor = tensor.contiguous(memory_format=torch.channels_last)
    with torch.inference_mode():
        output = model(tensor)
        output.clamp_(0, 1)
        output = output.mul_(255.0).round_().byte()
    return output[0].permute(1, 2, 0).contiguous()


def _worker_v51(
    worker_id: int,
    gpu_id: Optional[int],
    input_queue: mp.Queue,
    result_queue: mp.Queue,
    output_slot: mp.Semaphore,
    config_dict: Dict[str, object],
    input_name: str,
    output_name: str,
    input_shape: Tuple[int, int, int],
    output_shape: Tuple[int, int, int],
    dtype_str: str,
) -> None:
    """SR worker that writes CUDA uint8 output directly into shared memory."""
    input_shm = output_shm = None
    try:
        config = base.WorkerConfig(**config_dict)  # type: ignore[arg-type]
        if gpu_id is None:
            device = torch.device("cpu")
        else:
            torch.cuda.set_device(gpu_id)
            device = torch.device(f"cuda:{gpu_id}")
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cuda.matmul.allow_tf32 = True
        model, _ = base.load_worker_model(config, device)
        dtype = np.dtype(dtype_str)
        input_shm = shared_memory.SharedMemory(name=input_name)
        output_shm = shared_memory.SharedMemory(name=output_name)
        input_view = np.ndarray(input_shape, dtype=dtype, buffer=input_shm.buf)
        output_view = np.ndarray(output_shape, dtype=dtype, buffer=output_shm.buf)
        output_tensor = torch.from_numpy(output_view)
        result_queue.put(("ready", worker_id, str(device)))

        while True:
            frame_id = input_queue.get()
            if frame_id is None:
                break
            started = time.monotonic()
            if device.type == "cuda" and dtype == np.dtype(np.uint8):
                result_cuda = _infer_cuda_u8_tensor(model, input_view, device)
                compute_seconds = time.monotonic() - started
                # Preserve the one-frame look-ahead of v5.0: inference may finish
                # while the previous shared output is still being consumed.
                output_slot.acquire()
                # Blocking D2H is deliberate. The shared-memory frame must be
                # complete before the result metadata becomes visible.
                copy_started = time.monotonic()
                output_tensor.copy_(result_cuda, non_blocking=False)
                infer_seconds = compute_seconds + (time.monotonic() - copy_started)
                del result_cuda
            else:
                # Keep the established CPU/10-bit path bit-for-bit at the Python
                # level; only the common 8-bit CUDA path removes the extra host copy.
                result = base.infer_frame(model, input_view, device)
                infer_seconds = time.monotonic() - started
                output_slot.acquire()
                np.copyto(output_view, result, casting="no")
                del result
            result_queue.put(("result", worker_id, int(frame_id), infer_seconds))
    except Exception as error:
        result_queue.put(("error", worker_id, repr(error), traceback.format_exc()))
    finally:
        if input_shm is not None:
            input_shm.close()
        if output_shm is not None:
            output_shm.close()


def install_pipeline_optimizations() -> None:
    """Install transfer/progress optimizations without changing SR semantics."""
    global _PIPELINE_INSTALLED
    if _PIPELINE_INSTALLED:
        return
    _PIPELINE_INSTALLED = True
    pipeline._worker = _worker_v51
    pipeline.OutputPump = StableOutputPump

    # Balanced BasicVSR++ clips run in parallel across GPUs. Reporting the sum
    # of per-GPU elapsed times makes the stage look roughly Nx slower than its
    # wall time; the longest worker is the useful approximation for this log.
    try:
        from . import balanced_pipeline

        def parallel_elapsed(stats) -> float:
            return max(
                (float(getattr(item, "elapsed", 0.0)) for item in stats._preprocessors),
                default=0.0,
            )

        balanced_pipeline._AggregatePreprocessorStats.elapsed = property(parallel_elapsed)
    except Exception:
        pass


def _run_model_device(preprocessor, clip_gpu: torch.Tensor) -> torch.Tensor:
    """Run BasicVSR++ from a CUDA float32 clip and return CUDA float32 output."""
    from . import basicvsrpp as bvsr

    padded, original_h, original_w = bvsr._pad_to_model_size(clip_gpu)
    model_input = padded.to(dtype=preprocessor.dtype)
    try:
        with torch.inference_mode():
            output = preprocessor.model(model_input)
    except RuntimeError as error:
        message = str(error).lower()
        half_failure = preprocessor.dtype == torch.float16 and any(
            token in message
            for token in (
                "not implemented for 'half'",
                "not implemented for half",
                "expected scalar type float",
                "deform_conv2d",
            )
        )
        if not half_failure:
            raise
        print("[basicvsrpp] FP16 operator path failed; switching to FP32", flush=True)
        preprocessor.model.float()
        preprocessor.dtype = torch.float32
        torch.cuda.empty_cache()
        model_input = padded.float()
        with torch.inference_mode():
            output = preprocessor.model(model_input)
    return output[..., :original_h, :original_w].float()


def _enhance_u8_with_tile(preprocessor, clip_cpu: torch.Tensor, tile_size: int) -> np.ndarray:
    """Keep BVS assembly/blend/quantization on CUDA; transfer only final uint8."""
    device = preprocessor.device
    torch.cuda.set_device(device)
    original = clip_cpu.to(device=device, dtype=torch.float32, non_blocking=True)
    _n, _t, _c, height, width = original.shape
    pad = int(preprocessor.config.tile_pad)
    flat = original.reshape(-1, 3, height, width)
    mode = "reflect" if min(height, width) > pad and min(height, width) > 1 else "replicate"
    padded_flat = F.pad(flat, (pad, pad, pad, pad), mode=mode) if pad else flat
    padded = padded_flat.view(
        original.shape[0], original.shape[1], 3, height + 2 * pad, width + 2 * pad
    )
    restored = torch.empty_like(original, dtype=torch.float32, device=device)
    tile_count = 0
    for y0 in range(0, height, tile_size):
        y1 = min(y0 + tile_size, height)
        for x0 in range(0, width, tile_size):
            x1 = min(x0 + tile_size, width)
            patch = padded[..., y0 : y1 + 2 * pad, x0 : x1 + 2 * pad]
            enhanced = _run_model_device(preprocessor, patch)
            restored[..., y0:y1, x0:x1] = enhanced[
                ..., pad : pad + (y1 - y0), pad : pad + (x1 - x0)
            ]
            tile_count += 1
    preprocessor.tiles += tile_count

    strength = float(preprocessor.config.strength)
    if strength >= 1.0:
        mixed = restored
    else:
        mixed = original + strength * (restored - original)
    quantized = mixed.clamp_(0.0, 1.0).mul_(255.0).round_().to(torch.uint8)
    return quantized.permute(0, 1, 3, 4, 2).contiguous().cpu().numpy()


def _enhance_u8(preprocessor, clip_cpu: torch.Tensor) -> np.ndarray:
    requested = int(preprocessor.tile_size)
    candidates = [requested]
    for fallback in (384, 320, 256):
        if fallback < requested and fallback not in candidates:
            candidates.append(fallback)
    last_error: Optional[BaseException] = None
    for tile_size in candidates:
        try:
            output = _enhance_u8_with_tile(preprocessor, clip_cpu, tile_size)
            if tile_size != preprocessor.tile_size:
                preprocessor.tile_size = tile_size
                print(f"[basicvsrpp] VRAM fallback locked to tile={tile_size}", flush=True)
            return output
        except torch.cuda.OutOfMemoryError as error:
            last_error = error
            torch.cuda.empty_cache()
            print(f"[basicvsrpp] tile={tile_size} OOM; retrying smaller tile", flush=True)
    raise RuntimeError("BasicVSR++ ran out of GPU memory even at tile=256") from last_error


def _enhance_clip_v51(self, frames: Sequence[np.ndarray]) -> list[np.ndarray]:
    if not frames:
        return []
    first_shape = frames[0].shape
    first_dtype = np.dtype(frames[0].dtype)
    if any(frame.shape != first_shape or np.dtype(frame.dtype) != first_dtype for frame in frames):
        raise ValueError("All BasicVSR++ clip frames must have identical shape/dtype")
    if len(frames) == 1:
        return [np.ascontiguousarray(frame) for frame in frames]
    if first_dtype != np.dtype(np.uint8):
        return self._v51_original_enhance_clip(frames)

    from . import basicvsrpp as bvsr

    originals = np.stack([bvsr.frame_to_float_rgb(frame) for frame in frames])
    clip = torch.from_numpy(originals).permute(0, 3, 1, 2).unsqueeze(0)
    started = time.monotonic()
    enhanced_u8 = _enhance_u8(self, clip)[0]
    self.elapsed += time.monotonic() - started
    self.clips += 1
    return [np.ascontiguousarray(frame) for frame in enhanced_u8]


def _enhance_clips_v51(self, clips: Sequence[Sequence[np.ndarray]]) -> list[list[np.ndarray]]:
    if not clips:
        return []
    if len(clips) == 1:
        return [_enhance_clip_v51(self, clips[0])]

    lengths = {len(frames) for frames in clips}
    if len(lengths) != 1 or next(iter(lengths)) < 2:
        return [_enhance_clip_v51(self, frames) for frames in clips]
    first = clips[0][0]
    first_shape = first.shape
    first_dtype = np.dtype(first.dtype)
    for frames in clips:
        if any(frame.shape != first_shape or np.dtype(frame.dtype) != first_dtype for frame in frames):
            return [_enhance_clip_v51(self, item) for item in clips]
    if first_dtype != np.dtype(np.uint8):
        return self._v51_original_enhance_clips(clips)

    from . import basicvsrpp as bvsr

    originals = np.stack(
        [[bvsr.frame_to_float_rgb(frame) for frame in frames] for frames in clips], axis=0
    )
    tensor = torch.from_numpy(originals).permute(0, 1, 4, 2, 3)
    started = time.monotonic()
    enhanced_u8 = _enhance_u8(self, tensor)
    self.elapsed += time.monotonic() - started
    self.clips += len(clips)
    return [
        [np.ascontiguousarray(frame) for frame in group]
        for group in enhanced_u8
    ]


def _probe_full_frame_v51(self, tile_size: int, clip_length: int, clip_batch: int):
    """Benchmark the optimized uint8 execution path used by real 8-bit video."""
    self._quality_guard(tile_size, clip_batch)
    device = self.device
    torch.cuda.set_device(device)
    free_before, total = torch.cuda.mem_get_info(device)
    headroom = self._headroom(total)
    if free_before <= headroom:
        return None

    from . import basicvsrpp_autotune as tune

    height, width = tune._source_shape(tile_size)
    # Keep the probe source float32 to avoid spending benchmark time on random
    # generation/conversion; real execution also reaches _enhance_u8 as float32.
    clip = torch.zeros((clip_batch, clip_length, 3, height, width), dtype=torch.float32)
    old_tiles = self.tiles
    try:
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        output = _enhance_u8_with_tile(self, clip, tile_size)
        torch.cuda.synchronize(device)
        elapsed = max(time.perf_counter() - started, 1e-6)
        peak_reserved = torch.cuda.max_memory_reserved(device)
        peak_allocated = torch.cuda.max_memory_allocated(device)
        free_after, _ = torch.cuda.mem_get_info(device)
        safe = free_after >= headroom
        emitted = max(1, clip_length - 2 * self._profile_overlap)
        score = (clip_batch * emitted) / elapsed
        del output
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
        del clip


def install_basicvsrpp_optimizations(bit_depth: Optional[int] = None) -> None:
    """Install optimized 8-bit BVS post-processing after autotune is installed."""
    global _BVSR_INSTALLED
    if _BVSR_INSTALLED:
        return
    _BVSR_INSTALLED = True

    from . import basicvsrpp as bvsr
    from . import basicvsrpp_autotune as tune

    cls = tune.AutoTunedBasicVSRPPPreprocessor
    if not hasattr(cls, "_v51_original_enhance_clip"):
        cls._v51_original_enhance_clip = cls.enhance_clip
    if not hasattr(cls, "_v51_original_enhance_clips"):
        cls._v51_original_enhance_clips = cls.enhance_clips

    cls.enhance_clip = _enhance_clip_v51
    cls.enhance_clips = _enhance_clips_v51
    if bit_depth == 8:
        cls._probe_full_frame = _probe_full_frame_v51
    bvsr.BasicVSRPPPreprocessor = cls

    # The optimized data path changes the performance characteristics of tile
    # sizes, so do not reuse v3 measurements made with float32 D2H per tile.
    tune._CACHE_VERSION = 4
    tune._cache_path = lambda: Path.home() / ".cache" / "realesrgan" / "basicvsrpp-autotune-v4.json"
