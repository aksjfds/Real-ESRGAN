#!/usr/bin/env python3
"""AutoDL RTX 4090 single-GPU entry point for the fast Real-ESRGAN runner.

This entry keeps the shared-memory transport and automatic tile/batch probing,
but replaces the multi-GPU scheduler with a dedicated single-worker runtime:

* exactly one CUDA device, one worker process and one model copy;
* all Real-ESRGAN regions are processed by that worker without GPU lane splits;
* BasicVSR++ is restricted to the same single CUDA device;
* H2D, model execution and D2H use a two-slot CUDA-stream pipeline;
* automatic tuning selects the fastest safe tile/batch combination rather than
  merely the largest tile that fits;
* multi-GPU and CPU values for ``--gpu-ids`` are rejected explicitly;
* large output buffers use POSIX shared memory when safe and file-backed mmap
  otherwise.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import os
import queue
import shutil
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from types import ModuleType
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch


_REPOSITORY_ROOT = Path(__file__).resolve().parent
_MIN_QUALITY_TILE = 512
_PIPELINE_DEPTH = 2


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
        raise ValueError("AutoDL v4.2 requires one CUDA GPU; CPU inference is disabled")
    if not normalized or "," in normalized:
        raise ValueError("AutoDL v4.2 accepts exactly one --gpu-ids value, for example: 0")
    try:
        gpu_id = int(normalized)
    except ValueError as error:
        raise ValueError("--gpu-ids must be one non-negative CUDA device number") from error
    if gpu_id < 0:
        raise ValueError("--gpu-ids must be non-negative")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; AutoDL v4.2 requires an RTX 4090")
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


_original_worker_process_regions = fast._worker_process_regions


def _pipelined_worker_process_regions(
    model: torch.nn.Module,
    input_array: np.ndarray,
    output_array: np.ndarray,
    regions: Sequence[fast.FastTileRegion],
    batch_size: int,
    native_scale: int,
    device: torch.device,
    fp16: bool,
    channels_last: bool,
) -> dict[str, float]:
    """Overlap CPU packing, H2D, model compute and D2H with two CUDA slots.

    The old path synchronized the entire CUDA device after every model batch and
    then copied each core back synchronously. This implementation keeps model
    calls ordered on one compute stream while the next pinned input is copied on
    an H2D stream and the previous result is copied to pinned CPU buffers on a
    D2H stream. Pixel operations and crop coordinates are unchanged.
    """
    grouped: dict[tuple[int, int, int], list[fast.FastTileRegion]] = defaultdict(list)
    for region in regions:
        grouped[region.patch_shape].append(region)

    chunks: list[list[fast.FastTileRegion]] = []
    for shape in sorted(grouped):
        group = grouped[shape]
        for offset in range(0, len(group), max(1, batch_size)):
            chunks.append(group[offset : offset + max(1, batch_size)])

    timings = {
        "realesrgan_cpu_pack": 0.0,
        "realesrgan_h2d": 0.0,
        "realesrgan_model_gpu": 0.0,
        "realesrgan_d2h": 0.0,
        "realesrgan_pipeline_wait": 0.0,
        "realesrgan_shared_write": 0.0,
    }
    if not chunks:
        return timings

    h2d_stream = torch.cuda.Stream(device=device)
    compute_stream = torch.cuda.Stream(device=device)
    d2h_stream = torch.cuda.Stream(device=device)
    current_stream = torch.cuda.current_stream(device)
    h2d_stream.wait_stream(current_stream)
    compute_stream.wait_stream(current_stream)
    d2h_stream.wait_stream(current_stream)
    output_dtype = torch.float16 if fp16 else torch.float32
    model_dtype = torch.float16 if fp16 else torch.float32
    pending: list[dict[str, object]] = []

    def flush_oldest() -> None:
        item = pending.pop(0)
        wait_started = time.monotonic()
        done_event = item["done_event"]
        assert isinstance(done_event, torch.cuda.Event)
        done_event.synchronize()
        timings["realesrgan_pipeline_wait"] += time.monotonic() - wait_started

        for prefix, start_name, end_name in (
            ("realesrgan_h2d", "h2d_start", "h2d_end"),
            ("realesrgan_model_gpu", "model_start", "model_end"),
            ("realesrgan_d2h", "d2h_start", "d2h_end"),
        ):
            start_event = item[start_name]
            end_event = item[end_name]
            assert isinstance(start_event, torch.cuda.Event)
            assert isinstance(end_event, torch.cuda.Event)
            timings[prefix] += start_event.elapsed_time(end_event) / 1000.0

        write_started = time.monotonic()
        cpu_cores = item["cpu_cores"]
        chunk = item["chunk"]
        assert isinstance(cpu_cores, list)
        assert isinstance(chunk, list)
        for region, cpu_core in zip(chunk, cpu_cores):
            assert isinstance(region, fast.FastTileRegion)
            assert isinstance(cpu_core, torch.Tensor)
            oy0, oy1 = region.y0 * native_scale, region.y1 * native_scale
            ox0, ox1 = region.x0 * native_scale, region.x1 * native_scale
            np.copyto(
                output_array[oy0:oy1, ox0:ox1],
                cpu_core.numpy(),
                casting="unsafe",
            )
        timings["realesrgan_shared_write"] += time.monotonic() - write_started

    try:
        for chunk in chunks:
            pack_started = time.monotonic()
            patch_h, patch_w, _ = chunk[0].patch_shape
            pinned_input = torch.empty(
                (len(chunk), 3, patch_h, patch_w),
                dtype=torch.float32,
                device="cpu",
                pin_memory=True,
            )
            for index, region in enumerate(chunk):
                view = input_array[
                    region.context_y0 : region.context_y1,
                    region.context_x0 : region.context_x1,
                ]
                source = torch.from_numpy(view).permute(2, 0, 1)
                pinned_input[index].copy_(source)
            timings["realesrgan_cpu_pack"] += time.monotonic() - pack_started

            h2d_start = torch.cuda.Event(enable_timing=True)
            h2d_end = torch.cuda.Event(enable_timing=True)
            h2d_ready = torch.cuda.Event()
            with torch.cuda.stream(h2d_stream):
                h2d_start.record(h2d_stream)
                gpu_input = pinned_input.to(
                    device=device,
                    dtype=model_dtype,
                    non_blocking=True,
                )
                if channels_last:
                    gpu_input = gpu_input.contiguous(memory_format=torch.channels_last)
                h2d_end.record(h2d_stream)
                h2d_ready.record(h2d_stream)

            model_start = torch.cuda.Event(enable_timing=True)
            model_end = torch.cuda.Event(enable_timing=True)
            model_ready = torch.cuda.Event()
            with torch.cuda.stream(compute_stream):
                compute_stream.wait_event(h2d_ready)
                model_start.record(compute_stream)
                with torch.inference_mode():
                    native = model(gpu_input)
                    native.clamp_(0.0, 1.0)
                model_end.record(compute_stream)
                model_ready.record(compute_stream)
            gpu_input.record_stream(compute_stream)

            cpu_cores: list[torch.Tensor] = []
            d2h_start = torch.cuda.Event(enable_timing=True)
            d2h_end = torch.cuda.Event(enable_timing=True)
            done_event = torch.cuda.Event()
            with torch.cuda.stream(d2h_stream):
                d2h_stream.wait_event(model_ready)
                d2h_start.record(d2h_stream)
                for batch_index, region in enumerate(chunk):
                    crop_x0 = (region.x0 - region.context_x0) * native_scale
                    crop_y0 = (region.y0 - region.context_y0) * native_scale
                    core_w = (region.x1 - region.x0) * native_scale
                    core_h = (region.y1 - region.y0) * native_scale
                    core = native[
                        batch_index,
                        :,
                        crop_y0 : crop_y0 + core_h,
                        crop_x0 : crop_x0 + core_w,
                    ].permute(1, 2, 0)
                    cpu_core = torch.empty(
                        (core_h, core_w, 3),
                        dtype=output_dtype,
                        device="cpu",
                        pin_memory=True,
                    )
                    cpu_core.copy_(core, non_blocking=True)
                    cpu_cores.append(cpu_core)
                d2h_end.record(d2h_stream)
                done_event.record(d2h_stream)
            native.record_stream(d2h_stream)

            pending.append(
                {
                    "chunk": list(chunk),
                    "pinned_input": pinned_input,
                    "gpu_input": gpu_input,
                    "native": native,
                    "cpu_cores": cpu_cores,
                    "h2d_start": h2d_start,
                    "h2d_end": h2d_end,
                    "model_start": model_start,
                    "model_end": model_end,
                    "d2h_start": d2h_start,
                    "d2h_end": d2h_end,
                    "done_event": done_event,
                }
            )
            if len(pending) >= _PIPELINE_DEPTH:
                flush_oldest()

        while pending:
            flush_oldest()
        return timings
    except torch.cuda.OutOfMemoryError:
        try:
            torch.cuda.synchronize(device)
        except RuntimeError:
            pass
        pending.clear()
        torch.cuda.empty_cache()
        print(
            "[pipeline] two-slot overlap OOM; falling back to synchronous single-slot inference",
            flush=True,
        )
        return _original_worker_process_regions(
            model,
            input_array,
            output_array,
            regions,
            batch_size,
            native_scale,
            device,
            fp16,
            channels_last,
        )


fast._worker_process_regions = _pipelined_worker_process_regions


class SingleGPUSharedMemoryWorker(fast.SharedMemoryWorkers):
    """One-process shared-memory worker for a single RTX 4090."""

    def __init__(
        self,
        gpu_ids: Sequence[Optional[int]],
        config: base.WorkerConfig,
    ) -> None:
        if len(gpu_ids) != 1 or gpu_ids[0] is None:
            raise RuntimeError(
                "AutoDL v4.2 requires exactly one CUDA GPU; use --gpu-ids 0"
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

    @staticmethod
    def _shape_groups(
        width: int,
        height: int,
        tile_size: int,
        tile_pad: int,
    ) -> dict[tuple[int, int, int], int]:
        counts: dict[tuple[int, int, int], int] = defaultdict(int)
        for region in fast._regions(width, height, tile_size, tile_pad):
            counts[region.patch_shape] += 1
        return counts

    def _probe_batch(
        self,
        tile_size: int,
        tile_pad: int,
        batch: int,
    ) -> tuple[bool, float]:
        request = self._next_request()
        self.input_queue.put(
            ("probe_batch", request, tile_size, tile_pad, batch)
        )
        message = self._wait_result("probe_batch_result", request)
        return bool(message[3]), float(message[4])

    def _auto_tune(self, tile_pad: int) -> None:
        if self.frame_shape is None:
            raise RuntimeError("Frame buffers must be allocated before auto tuning")
        height, width = self.frame_shape[:2]

        if fast._AUTO_TILE:
            tile_candidates = [
                value
                for value in fast._tile_candidates(width, height, fast._MAX_TILE_SIZE)
                if value >= _MIN_QUALITY_TILE
            ]
            if not tile_candidates:
                tile_candidates = [_MIN_QUALITY_TILE]
        else:
            tile_candidates = [
                max(_MIN_QUALITY_TILE, int(fast._REQUESTED_TILE_SIZE))
            ]

        best_choice: Optional[tuple[float, int, int, float]] = None
        for tile_size in tile_candidates:
            groups = self._shape_groups(width, height, tile_size, tile_pad)
            region_count = sum(groups.values())
            largest_group = max(groups.values(), default=1)
            if fast._AUTO_BATCH:
                batch_candidates = sorted(
                    {
                        value
                        for value in (
                            *fast._batch_candidates(fast._MAX_BATCH_SIZE),
                            largest_group,
                        )
                        if 1 <= value <= largest_group
                    }
                )
            else:
                batch_candidates = [
                    min(largest_group, max(1, int(fast._REQUESTED_BATCH_SIZE)))
                ]

            max_area = max((shape[0] * shape[1] for shape in groups), default=1)
            successful: dict[int, float] = {}
            for batch in batch_candidates:
                ok, seconds = self._probe_batch(tile_size, tile_pad, batch)
                if not ok:
                    break
                successful[batch] = seconds

            if not successful:
                print(
                    f"[auto-tune] tile={tile_size} rejected due to RTX 4090 OOM",
                    flush=True,
                )
                continue

            for batch, probe_seconds in successful.items():
                estimated = 0.0
                for shape, count in groups.items():
                    area_ratio = (shape[0] * shape[1]) / max_area
                    estimated += math.ceil(count / batch) * probe_seconds * area_ratio
                throughput = (
                    batch * max_area / max(probe_seconds, 1e-9) / 1_000_000.0
                )
                print(
                    f"[auto-tune] tile={tile_size}, batch={batch}, "
                    f"regions={region_count}, estimate={estimated:.3f}s/frame, "
                    f"probe={throughput:.1f} MPix/s",
                    flush=True,
                )
                choice = (estimated, tile_size, batch, throughput)
                if best_choice is None or choice < best_choice:
                    best_choice = choice

        if best_choice is None:
            raise RuntimeError(
                "No quality-preserving tile/batch combination fit on the RTX 4090"
            )

        estimated, self.selected_tile, self.batch_size, throughput = best_choice
        print(
            f"[auto-tune] selected tile={self.selected_tile}, "
            f"batch={self.batch_size}, estimated={estimated:.3f}s/frame, "
            f"probe={throughput:.1f} MPix/s, gpu=0",
            flush=True,
        )

    def infer_tiles(
        self,
        frame_id: int,
        regions: Sequence[fast.FastTileRegion],
    ) -> Dict[int, np.ndarray]:
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
        "AutoDL RTX 4090 single-GPU pipelined Real-ESRGAN video enhancement."
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
