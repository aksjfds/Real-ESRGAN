"""v5.1 quality-preserving SR pipeline optimizations.

This module only changes Real-ESRGAN transfer/progress execution and exposes a
CUDA BasicVSR++ model helper used by later fixed-parameter execution patches.
"""
from __future__ import annotations

import multiprocessing as mp
from multiprocessing import shared_memory
import time
import traceback
from collections import deque
from dataclasses import asdict
from typing import Deque, Dict, Optional, Tuple

import numpy as np
import torch

from . import pipeline
from . import runtime as base


_PIPELINE_INSTALLED = False


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
                output_slot.acquire()
                copy_started = time.monotonic()
                output_tensor.copy_(result_cuda, non_blocking=False)
                infer_seconds = compute_seconds + (time.monotonic() - copy_started)
                del result_cuda
            else:
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
