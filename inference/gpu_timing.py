"""CUDA-event timing and per-GPU busy/wait diagnostics for the v5.2 scheduler.

This module changes instrumentation only. Model math, scheduling decisions,
quality parameters, resize, and encoding are untouched.
"""
from __future__ import annotations

import builtins
import multiprocessing as mp
from multiprocessing import shared_memory
import re
import threading
import time
import traceback
from collections import defaultdict
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch.nn import functional as F

from . import pipeline
from . import runtime as base
from . import v51_runtime

_INSTALLED = False
_LOCK = threading.Lock()
_STAGE_GPU: dict[str, dict[int, float]] = {
    "bvs": defaultdict(float),
    "sr": defaultdict(float),
}
_TASK_START: Optional[float] = None
_TASK_END: Optional[float] = None
_SR_SAMPLE_COUNT = 0


def _reset_metrics() -> None:
    global _TASK_START, _TASK_END, _SR_SAMPLE_COUNT
    with _LOCK:
        _STAGE_GPU["bvs"].clear()
        _STAGE_GPU["sr"].clear()
        _TASK_START = None
        _TASK_END = None
        _SR_SAMPLE_COUNT = 0


def _record(stage: str, gpu_id: Optional[int], gpu_seconds: float, wall_start: float, wall_end: float) -> None:
    global _TASK_START, _TASK_END
    if gpu_id is None:
        return
    with _LOCK:
        _STAGE_GPU[stage][int(gpu_id)] += max(0.0, float(gpu_seconds))
        _TASK_START = wall_start if _TASK_START is None else min(_TASK_START, wall_start)
        _TASK_END = wall_end if _TASK_END is None else max(_TASK_END, wall_end)


def _snapshot() -> dict:
    with _LOCK:
        bvs = dict(_STAGE_GPU["bvs"])
        sr = dict(_STAGE_GPU["sr"])
        start = _TASK_START
        end = _TASK_END
        sr_samples = _SR_SAMPLE_COUNT
    gpus = sorted(set(bvs) | set(sr))
    window = max(0.0, (end - start)) if start is not None and end is not None else 0.0
    per_gpu = {}
    for gpu in gpus:
        bvs_s = float(bvs.get(gpu, 0.0))
        sr_s = float(sr.get(gpu, 0.0))
        busy = bvs_s + sr_s
        wait = max(0.0, window - busy)
        ratio = (100.0 * busy / window) if window > 1e-9 else 0.0
        per_gpu[gpu] = {
            "bvs": bvs_s,
            "sr": sr_s,
            "busy": busy,
            "wait": wait,
            "ratio": ratio,
        }
    return {"window": window, "per_gpu": per_gpu, "sr_samples": sr_samples}


def _worker_timed(
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
    """v5.1 SR worker with CUDA Event timing instead of host launch timing."""
    input_shm = output_shm = None
    try:
        config = base.WorkerConfig(**config_dict)  # type: ignore[arg-type]
        if gpu_id is None:
            device = torch.device("cpu")
            start_event = end_event = None
        else:
            torch.cuda.set_device(gpu_id)
            device = torch.device(f"cuda:{gpu_id}")
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cuda.matmul.allow_tf32 = True
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

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
            wall_started = time.monotonic()

            if device.type == "cuda" and dtype == np.dtype(np.uint8):
                assert start_event is not None and end_event is not None
                start_event.record()
                result_cuda = v51_runtime._infer_cuda_u8_tensor(model, input_view, device)
                end_event.record()

                # Keep v5.1's one-frame look-ahead and blocking D2H semantics.
                # The event pair ends before output-slot waiting, so queue
                # backpressure is not misreported as GPU compute time.
                output_slot.acquire()
                output_tensor.copy_(result_cuda, non_blocking=False)
                end_event.synchronize()
                gpu_seconds = start_event.elapsed_time(end_event) / 1000.0
                del result_cuda
            else:
                # Preserve established CPU/10-bit math. On CUDA, events still
                # measure device work rather than asynchronous host launch time.
                if device.type == "cuda":
                    assert start_event is not None and end_event is not None
                    start_event.record()
                else:
                    cpu_started = time.monotonic()
                result = base.infer_frame(model, input_view, device)
                if device.type == "cuda":
                    end_event.record()
                    end_event.synchronize()
                    gpu_seconds = start_event.elapsed_time(end_event) / 1000.0
                else:
                    gpu_seconds = time.monotonic() - cpu_started
                output_slot.acquire()
                np.copyto(output_view, result, casting="no")
                del result

            wall_finished = time.monotonic()
            result_queue.put(
                (
                    "result",
                    worker_id,
                    int(frame_id),
                    float(gpu_seconds),
                    float(wall_started),
                    float(wall_finished),
                    gpu_id,
                )
            )
    except Exception as error:
        result_queue.put(("error", worker_id, repr(error), traceback.format_exc()))
    finally:
        if input_shm is not None:
            input_shm.close()
        if output_shm is not None:
            output_shm.close()


def _timed_enhance_u8_with_tile(preprocessor, clip_cpu: torch.Tensor, tile_size: int) -> np.ndarray:
    """v5.1 uint8 BVS path with device-side CUDA Event timing."""
    device = preprocessor.device
    torch.cuda.set_device(device)
    wall_started = time.monotonic()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()

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
            enhanced = v51_runtime._run_model_device(preprocessor, patch)
            restored[..., y0:y1, x0:x1] = enhanced[
                ..., pad : pad + (y1 - y0), pad : pad + (x1 - x0)
            ]
            tile_count += 1
    preprocessor.tiles += tile_count

    strength = float(preprocessor.config.strength)
    mixed = restored if strength >= 1.0 else original + strength * (restored - original)
    quantized = mixed.clamp_(0.0, 1.0).mul_(255.0).round_().to(torch.uint8)
    end_event.record()

    # The existing blocking .cpu() comes after the end event and therefore
    # guarantees that elapsed_time can be read without adding a new sync point.
    result = quantized.permute(0, 1, 3, 4, 2).contiguous().cpu().numpy()
    gpu_seconds = start_event.elapsed_time(end_event) / 1000.0
    _record("bvs", int(preprocessor.config.gpu_id), gpu_seconds, wall_started, time.monotonic())
    return result


def _timed_original_enhance_with_tile(self, clip: torch.Tensor, tile_size: int) -> torch.Tensor:
    """10-bit/original BVS tiled path with per-tile CUDA Event timing."""
    _n, _t, _c, height, width = clip.shape
    pad = self.config.tile_pad
    flat = clip.reshape(-1, 3, height, width)
    mode = "reflect" if min(height, width) > pad and min(height, width) > 1 else "replicate"
    padded_flat = F.pad(flat, (pad, pad, pad, pad), mode=mode) if pad else flat
    padded = padded_flat.view(clip.shape[0], clip.shape[1], 3, height + 2 * pad, width + 2 * pad)
    result = torch.empty_like(clip, dtype=torch.float32, device="cpu")
    tile_count = 0
    gpu_seconds = 0.0
    wall_started = time.monotonic()
    torch.cuda.set_device(int(self.config.gpu_id))
    for y0 in range(0, height, tile_size):
        y1 = min(y0 + tile_size, height)
        for x0 in range(0, width, tile_size):
            x1 = min(x0 + tile_size, width)
            patch = padded[..., y0 : y1 + 2 * pad, x0 : x1 + 2 * pad]
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            enhanced = self._run_model(patch)
            end_event.record()
            enhanced_cpu = enhanced.cpu()
            end_event.synchronize()
            gpu_seconds += start_event.elapsed_time(end_event) / 1000.0
            result[..., y0:y1, x0:x1] = enhanced_cpu[
                ..., pad : pad + (y1 - y0), pad : pad + (x1 - x0)
            ]
            tile_count += 1
    self.tiles += tile_count
    _record("bvs", int(self.config.gpu_id), gpu_seconds, wall_started, time.monotonic())
    return result


def _install_result_collector() -> None:
    if getattr(pipeline._result, "_gpu_timing_wrapped", False):
        return
    original_result = pipeline._result

    def timed_result(msg: tuple):
        global _SR_SAMPLE_COUNT
        result = original_result(msg)
        if msg and msg[0] == "result" and len(msg) >= 7:
            _record("sr", msg[6], float(msg[3]), float(msg[4]), float(msg[5]))
            if msg[6] is not None:
                with _LOCK:
                    _SR_SAMPLE_COUNT += 1
        return result

    timed_result._gpu_timing_wrapped = True  # type: ignore[attr-defined]
    pipeline._result = timed_result


def _install_balanced_reset() -> None:
    from . import balanced_pipeline

    cls = balanced_pipeline.BalancedBasicVSRPPStreamReader
    if getattr(cls.__init__, "_gpu_timing_wrapped", False):
        return
    original_init = cls.__init__

    def timed_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        # Replica construction may run/load autotune probes. Clear those so
        # reported busy/wait covers only actual video scheduling.
        _reset_metrics()

    timed_init._gpu_timing_wrapped = True  # type: ignore[attr-defined]
    cls.__init__ = timed_init


def _install_log_formatter() -> None:
    from . import v52_scheduler

    if getattr(v52_scheduler, "_gpu_timing_print_installed", False):
        return

    def timing_print(*values, **kwargs):
        if values and isinstance(values[0], str):
            text = values[0]
            if text.startswith("GPU     :"):
                text = text.replace(
                    " | FP16 + channels_last | FP16 + channels_last",
                    " | FP16 + channels_last",
                )
                values = (text, *values[1:])
            elif text.startswith("Timing  :"):
                snap = _snapshot()
                per_gpu = snap["per_gpu"]
                if per_gpu:
                    busy_values = [item["busy"] for item in per_gpu.values()]
                    wait_values = [item["wait"] for item in per_gpu.values()]
                    bvs_max = max((item["bvs"] for item in per_gpu.values()), default=0.0)
                    sr_total = sum(item["sr"] for item in per_gpu.values())
                    clip_match = re.search(r"/([0-9]+) clips", text)
                    clips = int(clip_match.group(1)) if clip_match else 0
                    sr_avg = sr_total / max(int(snap["sr_samples"]), 1)
                    replacement = (
                        f"gpu_busy_avg={sum(busy_values)/len(busy_values):.1f}s | "
                        f"gpu_wait_avg={sum(wait_values)/len(wait_values):.1f}s | "
                    )
                    text = re.sub(r"scheduler_idle=[^|]+\|\s*", replacement, text)
                    text = re.sub(
                        r"basicvsr=[^|]+",
                        f"bvs_gpu_max={bvs_max:.1f}s/{clips} clips ",
                        text,
                    )
                    text = re.sub(
                        r"sr_gpu_avg=[0-9.]+s/frame",
                        f"sr_gpu_avg={sr_avg:.3f}s/frame",
                        text,
                    )
                    builtins.print(text, *values[1:], **kwargs)
                    details = []
                    for gpu, item in per_gpu.items():
                        details.append(
                            f"cuda:{gpu} busy={item['busy']:.1f}s "
                            f"(BVS={item['bvs']:.1f}s SR={item['sr']:.1f}s) "
                            f"wait={item['wait']:.1f}s busy_ratio={item['ratio']:.1f}%"
                        )
                    builtins.print(
                        f"GPU time: window={snap['window']:.1f}s | " + " | ".join(details),
                        flush=kwargs.get("flush", True),
                    )
                    return
        builtins.print(*values, **kwargs)

    v52_scheduler.print = timing_print
    v52_scheduler._gpu_timing_print_installed = True


def install_gpu_timing(enable_bvs: bool = False) -> None:
    """Install CUDA-event metrics without changing model/scheduler semantics."""
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True
    _reset_metrics()

    # install_pipeline_optimizations() runs before this and installs v5.1's
    # transfer worker. Replace only its timing method, keeping the data path.
    pipeline._worker = _worker_timed
    _install_result_collector()
    _install_log_formatter()

    if enable_bvs:
        from . import basicvsrpp as bvsr

        # The optimized 8-bit path calls this v51 module-level function.
        v51_runtime._enhance_u8_with_tile = _timed_enhance_u8_with_tile

        # The 10-bit/original path uses the class method below.
        bvsr.BasicVSRPPPreprocessor._enhance_with_tile = _timed_original_enhance_with_tile
        _install_balanced_reset()
