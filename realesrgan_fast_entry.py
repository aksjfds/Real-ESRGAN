#!/usr/bin/env python3
"""AutoDL RTX 4090 single-GPU entry point for the fast Real-ESRGAN runner.

This entry keeps the shared-memory transport and automatic tile/batch probing,
but replaces the multi-GPU scheduler with a dedicated single-worker runtime:

* exactly one CUDA device, one worker process and one model copy;
* all Real-ESRGAN regions are processed by that worker without GPU lane splits;
* BasicVSR++ is restricted to the same single CUDA device;
* multi-GPU and CPU values for ``--gpu-ids`` are rejected explicitly;
* large output buffers use POSIX shared memory when safe and file-backed mmap
  otherwise.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import queue
import shutil
import sys
import tempfile
import time
from pathlib import Path
from types import ModuleType
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch


_REPOSITORY_ROOT = Path(__file__).resolve().parent


def _load_local_realesrgan() -> ModuleType:
    """Load the adjacent CLI/video runner without relying on import search order."""
    module_path = _REPOSITORY_ROOT / "realesrgan.py"
    if not module_path.is_file():
        raise FileNotFoundError(f"Local Real-ESRGAN runner not found: {module_path}")

    spec = importlib.util.spec_from_file_location("realesrgan", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to create an import spec for {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["realesrgan"] = module
    spec.loader.exec_module(module)

    required = (
        "build_parser",
        "process_video",
        "PersistentWorkers",
        "TileProcessor",
        "WorkerConfig",
        "BasicVSRPPPreprocessor",
    )
    missing = [name for name in required if not hasattr(module, name)]
    if missing:
        raise ImportError(
            f"Local runner {module_path} is missing required symbols: {missing}"
        )
    return module


base = _load_local_realesrgan()

import realesrgan_fast as fast  # noqa: E402  (requires local-module bootstrap)

fast.base = base


# Preserve the existing AutoDL/Kaggle-safe shared-buffer fallback.
_original_shared_array_init = fast.SharedArray.__init__


def _guarded_shared_array_init(
    self: fast.SharedArray,
    shape: tuple[int, ...],
    dtype: np.dtype,
    prefix: str,
) -> None:
    resolved_shape = tuple(int(value) for value in shape)
    resolved_dtype = np.dtype(dtype)
    required = int(np.prod(resolved_shape, dtype=np.int64)) * resolved_dtype.itemsize
    shm_path = Path("/dev/shm")
    use_posix_shm = False
    if shm_path.is_dir():
        try:
            free = shutil.disk_usage(shm_path).free
            use_posix_shm = required <= int(free * 0.70)
        except OSError:
            use_posix_shm = False

    if use_posix_shm:
        _original_shared_array_init(self, resolved_shape, resolved_dtype, prefix)
        return

    self.shape = resolved_shape
    self.dtype = resolved_dtype
    self.size = required
    self.shm = None
    root = Path("/root/autodl-tmp")
    if not root.is_dir():
        root = _REPOSITORY_ROOT
    if not root.is_dir():
        root = Path(tempfile.gettempdir())
    fd, path = tempfile.mkstemp(prefix=f"{prefix}-", suffix=".mmap", dir=root)
    os.close(fd)
    with open(path, "wb") as handle:
        handle.truncate(required)
    self.path = path
    self.array = np.memmap(path, mode="r+", dtype=resolved_dtype, shape=resolved_shape)
    self.spec = fast.SharedArraySpec("mmap", path, resolved_shape, resolved_dtype.str)


fast.SharedArray.__init__ = _guarded_shared_array_init


def _parse_single_gpu(value: str) -> list[Optional[int]]:
    """Resolve exactly one CUDA device for the AutoDL RTX 4090 runtime."""
    normalized = str(value).strip().lower()
    if normalized == "cpu":
        raise ValueError("AutoDL v4.1 requires one CUDA GPU; CPU inference is disabled")
    if not normalized or "," in normalized:
        raise ValueError("AutoDL v4.1 accepts exactly one --gpu-ids value, for example: 0")
    try:
        gpu_id = int(normalized)
    except ValueError as error:
        raise ValueError("--gpu-ids must be one non-negative CUDA device number") from error
    if gpu_id < 0:
        raise ValueError("--gpu-ids must be non-negative")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; AutoDL v4.1 requires an RTX 4090")
    visible = torch.cuda.device_count()
    if gpu_id >= visible:
        raise ValueError(
            f"Requested CUDA device {gpu_id}, but only {visible} device(s) are visible"
        )
    return [gpu_id]


base.parse_gpu_ids = _parse_single_gpu


# Prevent the quality wrapper from enumerating every visible GPU. This keeps
# BasicVSR++ and Real-ESRGAN on the same logical cuda:0 selected by AutoDL.
_OriginalBasicVSRPPPreprocessor = base.BasicVSRPPPreprocessor


class SingleGPUBasicVSRPPPreprocessor(_OriginalBasicVSRPPPreprocessor):
    def __init__(self, config: object, *args: object, **kwargs: object) -> None:
        gpu_id = int(getattr(config, "gpu_id"))
        kwargs["gpu_ids"] = (gpu_id,)
        super().__init__(config, *args, **kwargs)


base.BasicVSRPPPreprocessor = SingleGPUBasicVSRPPPreprocessor


class SingleGPUSharedMemoryWorker(fast.SharedMemoryWorkers):
    """One-process shared-memory worker for a single RTX 4090."""

    def __init__(
        self,
        gpu_ids: Sequence[Optional[int]],
        config: base.WorkerConfig,
    ) -> None:
        if len(gpu_ids) != 1 or gpu_ids[0] is None:
            raise RuntimeError(
                "AutoDL v4.1 requires exactly one CUDA GPU; use --gpu-ids 0"
            )

        self.context = fast.mp.get_context("spawn")
        self.output_queue = self.context.Queue()
        self.input_queue = self.context.Queue(maxsize=1)
        self.gpu_id = int(gpu_ids[0])
        self.config = config
        self.stage_timings: Dict[str, float] = {}
        self.input_buffer: Optional[fast.SharedArray] = None
        self.output_buffer: Optional[fast.SharedArray] = None
        self.frame_shape: Optional[tuple[int, int, int]] = None
        self.selected_tile: Optional[int] = None
        self.batch_size = max(1, int(config.batch_size))
        self._closed = False
        self._request_id = 0

        self.process = self.context.Process(
            target=fast.fast_worker_main,
            args=(
                0,
                self.gpu_id,
                self.input_queue,
                self.output_queue,
                fast.asdict(config),
            ),
            daemon=True,
        )
        self.process.start()
        try:
            self._wait_ready()
        except Exception:
            self.close()
            raise
        fast._ACTIVE_WORKERS = self

    def _next_request(self) -> int:
        self._request_id += 1
        return self._request_id

    def _get_message(self, timeout: Optional[float] = None) -> tuple:
        try:
            message = self.output_queue.get(timeout=timeout)
        except queue.Empty as error:
            raise TimeoutError("Timed out waiting for the RTX 4090 worker") from error
        if message[0] == "error":
            raise RuntimeError(
                f"RTX 4090 worker failed: {message[2]}\n{message[3]}"
            )
        return message

    def _wait_ready(self) -> None:
        message = self._get_message(timeout=300.0)
        if message[0] != "ready" or message[1] != 0:
            raise RuntimeError(f"Unexpected worker startup message: {message[0]}")
        print(
            f"[gpu] single shared-memory model resident on {message[2]}",
            flush=True,
        )

    def _wait_result(self, expected: str, identity: int) -> tuple:
        message = self._get_message()
        if message[0] != expected or message[1] != 0 or message[2] != identity:
            raise RuntimeError(
                f"Unexpected single-GPU worker message: {message[0]}"
            )
        return message

    def _allocate(self, frame: np.ndarray) -> None:
        shape = tuple(int(value) for value in frame.shape)
        if self.frame_shape == shape:
            return
        if self.input_buffer is not None:
            self.input_buffer.close()
        if self.output_buffer is not None:
            self.output_buffer.close()

        output_dtype = np.float16 if self.config.fp16 else np.float32
        self.input_buffer = fast.SharedArray(shape, np.float32, "realesrgan-input")
        self.output_buffer = fast.SharedArray(
            (
                shape[0] * self.config.native_scale,
                shape[1] * self.config.native_scale,
                3,
            ),
            output_dtype,
            "realesrgan-output",
        )
        self.frame_shape = shape

        self.input_queue.put(
            ("attach", self.input_buffer.spec, self.output_buffer.spec)
        )
        message = self._get_message()
        if message[0] != "attached" or message[1] != 0:
            raise RuntimeError(f"Unexpected shared-buffer message: {message[0]}")

        print(
            f"[shared-memory] input={self.input_buffer.spec.backend}:"
            f"{self.input_buffer.size / 2**20:.1f}MiB, "
            f"output={self.output_buffer.spec.backend}:"
            f"{self.output_buffer.size / 2**20:.1f}MiB, "
            f"dtype={output_dtype}",
            flush=True,
        )

    def _auto_tune(self, tile_pad: int) -> None:
        if self.frame_shape is None:
            raise RuntimeError("Frame buffers must be allocated before auto tuning")
        height, width = self.frame_shape[:2]
        selected = max(fast._MIN_TILE_SIZE, min(fast._MAX_TILE_SIZE, 256))

        if fast._AUTO_TILE:
            for candidate in fast._tile_candidates(width, height, fast._MAX_TILE_SIZE):
                request = self._next_request()
                self.input_queue.put(("probe_tile", request, candidate, tile_pad))
                message = self._wait_result("probe_tile_result", request)
                if bool(message[3]):
                    selected = candidate
                    print(
                        f"[auto-tile] selected={selected}, "
                        f"gpu0={message[4]:.3f}s/free={message[6] / 2**30:.2f}GiB",
                        flush=True,
                    )
                    break
                print(
                    f"[auto-tile] rejected={candidate} due to RTX 4090 OOM",
                    flush=True,
                )
        else:
            selected = max(fast._MIN_TILE_SIZE, int(fast._REQUESTED_TILE_SIZE))

        self.selected_tile = selected

        if fast._AUTO_BATCH:
            best = 1
            for candidate in fast._batch_candidates(fast._MAX_BATCH_SIZE):
                request = self._next_request()
                self.input_queue.put(
                    ("probe_batch", request, selected, tile_pad, candidate)
                )
                message = self._wait_result("probe_batch_result", request)
                if not bool(message[3]):
                    break
                best = candidate
            self.batch_size = best
        else:
            self.batch_size = max(1, int(fast._REQUESTED_BATCH_SIZE))

        print(
            f"[auto-batch] tile={self.selected_tile}, batch={self.batch_size}, gpu=0",
            flush=True,
        )

    def infer_tiles(
        self,
        frame_id: int,
        regions: Sequence[fast.FastTileRegion],
    ) -> Dict[int, np.ndarray]:
        # The complete region list stays on one lane; no GPU partitioning or
        # cross-device result merge remains in the AutoDL runtime.
        self.input_queue.put(
            ("tiles", frame_id, list(regions), self.batch_size)
        )
        message = self._wait_result("tiles_result", frame_id)
        for name, value in message[3].items():
            self.stage_timings[name] = self.stage_timings.get(name, 0.0) + float(value)
        return {}

    def infer_frames(
        self,
        batch_id: int,
        indexed_frames: Sequence[Tuple[int, np.ndarray]],
    ) -> Dict[int, np.ndarray]:
        del batch_id, indexed_frames
        raise RuntimeError(
            "The AutoDL fast runner uses single-GPU auto-tuned tiles; "
            "full-frame queue transport is disabled"
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.input_queue.put_nowait(None)
        except queue.Full:
            pass
        self.process.join(timeout=10)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=5)
        self.input_queue.close()
        self.output_queue.close()
        if self.input_buffer is not None:
            self.input_buffer.close()
            self.input_buffer = None
        if self.output_buffer is not None:
            self.output_buffer.close()
            self.output_buffer = None
        if fast._ACTIVE_WORKERS is self:
            fast._ACTIVE_WORKERS = None


fast.SharedMemoryWorkers = SingleGPUSharedMemoryWorker


_original_build_parser = fast.build_parser


def _single_gpu_build_parser() -> argparse.ArgumentParser:
    parser = _original_build_parser()
    parser.description = (
        "AutoDL RTX 4090 single-GPU shared-memory Real-ESRGAN video enhancement."
    )
    for action in parser._actions:
        if action.dest == "gpu_ids":
            action.default = "0"
            action.help = "Single CUDA device only; AutoDL RTX 4090 uses 0"
        elif action.dest == "max_tile_size":
            action.default = 1536
        elif action.dest == "max_batch_size":
            action.default = 32
    return parser


fast.build_parser = _single_gpu_build_parser


if __name__ == "__main__":
    fast.mp.freeze_support()
    fast.main()
