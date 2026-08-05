#!/usr/bin/env python3
"""Shared-memory, auto-tuned Real-ESRGAN runner.

This module reuses the project's validated video, color, BasicVSR++, model-loading,
stitching, Lanczos, encoding, and CLI implementations. It replaces only the
Real-ESRGAN tile transport/scheduling path:

- persistent shared input/output arrays instead of pickled NumPy tile payloads;
- crop tile context inside each GPU worker;
- write FP16 model cores to shared output (FP32 when the model is FP32);
- group equal-shape tiles before batching;
- probe the largest safe tile and batch on the actual resident-GPU state.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import queue
import tempfile
import time
import traceback
from collections import defaultdict
from dataclasses import asdict, dataclass
from multiprocessing import shared_memory
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

import realesrgan as base


_AUTO_TILE = True
_AUTO_BATCH = True
_MAX_TILE_SIZE = 1024
_MAX_BATCH_SIZE = 16
_REQUESTED_TILE_SIZE = 256
_REQUESTED_BATCH_SIZE = 4
_MIN_TILE_SIZE = 256
_ACTIVE_WORKERS: Optional["SharedMemoryWorkers"] = None


@dataclass(frozen=True)
class FastTileRegion:
    index: int
    x0: int
    y0: int
    x1: int
    y1: int
    context_x0: int
    context_y0: int
    context_x1: int
    context_y1: int

    @property
    def patch_shape(self) -> tuple[int, int, int]:
        return (
            self.context_y1 - self.context_y0,
            self.context_x1 - self.context_x0,
            3,
        )

    @property
    def area(self) -> int:
        return (self.context_y1 - self.context_y0) * (
            self.context_x1 - self.context_x0
        )


@dataclass(frozen=True)
class SharedArraySpec:
    backend: str
    location: str
    shape: tuple[int, ...]
    dtype: str


class SharedArray:
    """Persistent cross-process ndarray with POSIX-shm and mmap fallback."""

    def __init__(self, shape: Sequence[int], dtype: np.dtype, prefix: str):
        self.shape = tuple(int(value) for value in shape)
        self.dtype = np.dtype(dtype)
        self.size = int(np.prod(self.shape, dtype=np.int64)) * self.dtype.itemsize
        self.shm: Optional[shared_memory.SharedMemory] = None
        self.path: Optional[str] = None
        self.array: np.ndarray

        try:
            self.shm = shared_memory.SharedMemory(create=True, size=self.size)
            self.array = np.ndarray(self.shape, dtype=self.dtype, buffer=self.shm.buf)
            self.spec = SharedArraySpec(
                "shm", self.shm.name, self.shape, self.dtype.str
            )
        except (OSError, BufferError):
            if self.shm is not None:
                try:
                    self.shm.close()
                    self.shm.unlink()
                except FileNotFoundError:
                    pass
                self.shm = None
            root = Path("/kaggle/working")
            if not root.is_dir():
                root = Path(tempfile.gettempdir())
            fd, path = tempfile.mkstemp(prefix=f"{prefix}-", suffix=".mmap", dir=root)
            os.close(fd)
            with open(path, "wb") as handle:
                handle.truncate(self.size)
            self.path = path
            self.array = np.memmap(path, mode="r+", dtype=self.dtype, shape=self.shape)
            self.spec = SharedArraySpec("mmap", path, self.shape, self.dtype.str)

    def close(self) -> None:
        if isinstance(self.array, np.memmap):
            self.array.flush()
        del self.array
        if self.shm is not None:
            try:
                self.shm.close()
            finally:
                try:
                    self.shm.unlink()
                except FileNotFoundError:
                    pass
            self.shm = None
        if self.path is not None:
            try:
                os.unlink(self.path)
            except FileNotFoundError:
                pass
            self.path = None


class AttachedArray:
    def __init__(self, spec: SharedArraySpec):
        self.spec = spec
        self.shm: Optional[shared_memory.SharedMemory] = None
        dtype = np.dtype(spec.dtype)
        if spec.backend == "shm":
            self.shm = shared_memory.SharedMemory(name=spec.location, create=False)
            self.array = np.ndarray(spec.shape, dtype=dtype, buffer=self.shm.buf)
        elif spec.backend == "mmap":
            self.array = np.memmap(
                spec.location, mode="r+", dtype=dtype, shape=spec.shape
            )
        else:
            raise ValueError(f"Unknown shared array backend: {spec.backend}")

    def close(self) -> None:
        if isinstance(self.array, np.memmap):
            self.array.flush()
        del self.array
        if self.shm is not None:
            self.shm.close()
            self.shm = None


def _tile_candidates(width: int, height: int, maximum: int) -> list[int]:
    """Largest candidates that do not create a very narrow trailing core."""
    upper = min(maximum, max(width, height))
    standard = [1536, 1280, 1024, 896, 768, 640, 576, 512, 448, 384, 320, 256]

    def quality_safe(length: int, tile: int) -> bool:
        if length <= tile:
            return True
        tail = length % tile
        return tail == 0 or tail >= max(128, tile // 3)

    available = [
        value for value in standard if _MIN_TILE_SIZE <= value <= upper
    ]
    safe = [
        value
        for value in available
        if quality_safe(width, value) and quality_safe(height, value)
    ]
    unsafe_fallback = [value for value in available if value not in safe]
    candidates = safe + unsafe_fallback
    if not candidates:
        candidates = [min(max(width, height), max(_MIN_TILE_SIZE, upper))]
    return list(dict.fromkeys(candidates))


def _batch_candidates(maximum: int) -> list[int]:
    standard = [1, 2, 4, 6, 8, 12, 16, 24, 32]
    return [value for value in standard if value <= maximum]


def _regions(
    width: int,
    height: int,
    tile_size: int,
    tile_pad: int,
) -> list[FastTileRegion]:
    result: list[FastTileRegion] = []
    index = 0
    for y0 in base.axis_starts(height, tile_size):
        for x0 in base.axis_starts(width, tile_size):
            y1 = min(y0 + tile_size, height)
            x1 = min(x0 + tile_size, width)
            result.append(
                FastTileRegion(
                    index=index,
                    x0=x0,
                    y0=y0,
                    x1=x1,
                    y1=y1,
                    context_x0=max(0, x0 - tile_pad),
                    context_y0=max(0, y0 - tile_pad),
                    context_x1=min(width, x1 + tile_pad),
                    context_y1=min(height, y1 + tile_pad),
                )
            )
            index += 1
    return result


def _attach_arrays(
    input_spec: SharedArraySpec,
    output_spec: SharedArraySpec,
) -> tuple[AttachedArray, AttachedArray]:
    return AttachedArray(input_spec), AttachedArray(output_spec)


def _prepare_tensor(
    batch: np.ndarray,
    device: torch.device,
    fp16: bool,
    channels_last: bool,
) -> torch.Tensor:
    tensor = torch.from_numpy(batch).permute(0, 3, 1, 2).to(
        device, dtype=torch.float32, non_blocking=True
    )
    tensor.clamp_(0.0, 1.0)
    if fp16 and device.type == "cuda":
        tensor = tensor.half()
    if channels_last and device.type == "cuda":
        tensor = tensor.contiguous(memory_format=torch.channels_last)
    return tensor


def _probe_shape(
    model: torch.nn.Module,
    input_array: np.ndarray,
    device: torch.device,
    fp16: bool,
    channels_last: bool,
    patch_h: int,
    patch_w: int,
    batch_size: int,
) -> tuple[bool, float]:
    h, w = input_array.shape[:2]
    y0 = max(0, (h - patch_h) // 2)
    x0 = max(0, (w - patch_w) // 2)
    patch = np.ascontiguousarray(input_array[y0 : y0 + patch_h, x0 : x0 + patch_w])
    batch = np.repeat(patch[None, ...], batch_size, axis=0)
    try:
        tensor = _prepare_tensor(batch, device, fp16, channels_last)
        started = torch.cuda.Event(enable_timing=True)
        ended = torch.cuda.Event(enable_timing=True)
        with torch.inference_mode():
            started.record()
            output = model(tensor)
            ended.record()
            torch.cuda.synchronize(device)
        seconds = started.elapsed_time(ended) / 1000.0
        del output, tensor, batch
        torch.cuda.empty_cache()
        return True, seconds
    except torch.cuda.OutOfMemoryError:
        del batch
        torch.cuda.empty_cache()
        return False, 0.0


def _worker_process_regions(
    model: torch.nn.Module,
    input_array: np.ndarray,
    output_array: np.ndarray,
    regions: Sequence[FastTileRegion],
    batch_size: int,
    native_scale: int,
    device: torch.device,
    fp16: bool,
    channels_last: bool,
) -> dict[str, float]:
    grouped: dict[tuple[int, int, int], list[FastTileRegion]] = defaultdict(list)
    for region in regions:
        grouped[region.patch_shape].append(region)

    timings = {
        "realesrgan_h2d": 0.0,
        "realesrgan_model_gpu": 0.0,
        "realesrgan_d2h": 0.0,
        "realesrgan_shared_write": 0.0,
    }

    for shape in sorted(grouped):
        group = grouped[shape]
        for offset in range(0, len(group), batch_size):
            chunk = group[offset : offset + batch_size]
            patches = np.stack(
                [
                    input_array[
                        region.context_y0 : region.context_y1,
                        region.context_x0 : region.context_x1,
                    ]
                    for region in chunk
                ]
            )

            h2d_started = time.monotonic()
            tensor = _prepare_tensor(patches, device, fp16, channels_last)
            timings["realesrgan_h2d"] += time.monotonic() - h2d_started

            model_start = torch.cuda.Event(enable_timing=True)
            model_end = torch.cuda.Event(enable_timing=True)
            with torch.inference_mode():
                model_start.record()
                native = model(tensor)
                native.clamp_(0.0, 1.0)
                model_end.record()
                torch.cuda.synchronize(device)
            timings["realesrgan_model_gpu"] += (
                model_start.elapsed_time(model_end) / 1000.0
            )

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
                ].permute(1, 2, 0).contiguous()

                d2h_started = time.monotonic()
                if fp16:
                    cpu_core = core.to("cpu", dtype=torch.float16).numpy()
                else:
                    cpu_core = core.float().cpu().numpy()
                timings["realesrgan_d2h"] += time.monotonic() - d2h_started

                write_started = time.monotonic()
                oy0, oy1 = region.y0 * native_scale, region.y1 * native_scale
                ox0, ox1 = region.x0 * native_scale, region.x1 * native_scale
                output_array[oy0:oy1, ox0:ox1] = cpu_core
                timings["realesrgan_shared_write"] += time.monotonic() - write_started

            del native, tensor, patches
    return timings


def fast_worker_main(
    worker_id: int,
    gpu_id: Optional[int],
    input_queue: mp.Queue,
    output_queue: mp.Queue,
    config_dict: Dict[str, object],
) -> None:
    input_attached: Optional[AttachedArray] = None
    output_attached: Optional[AttachedArray] = None
    try:
        config = base.WorkerConfig(**config_dict)  # type: ignore[arg-type]
        if gpu_id is None:
            raise RuntimeError("The shared-memory fast path requires CUDA workers")
        torch.cuda.set_device(gpu_id)
        device = torch.device(f"cuda:{gpu_id}")
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        model, native_scale = base.load_worker_model(config, device)
        output_queue.put(("ready", worker_id, str(device)))

        while True:
            job = input_queue.get()
            if job is None:
                break
            command = job[0]

            if command == "attach":
                if input_attached is not None:
                    input_attached.close()
                if output_attached is not None:
                    output_attached.close()
                input_attached, output_attached = _attach_arrays(job[1], job[2])
                output_queue.put(("attached", worker_id))
                continue

            if input_attached is None or output_attached is None:
                raise RuntimeError("Shared arrays were not attached before inference")

            if command == "probe_tile":
                request_id, tile_size, tile_pad = job[1], int(job[2]), int(job[3])
                patch_h = min(input_attached.array.shape[0], tile_size + 2 * tile_pad)
                patch_w = min(input_attached.array.shape[1], tile_size + 2 * tile_pad)
                free_before, total = torch.cuda.mem_get_info(device)
                ok, seconds = _probe_shape(
                    model,
                    input_attached.array,
                    device,
                    config.fp16,
                    config.channels_last,
                    patch_h,
                    patch_w,
                    1,
                )
                free_after, _ = torch.cuda.mem_get_info(device)
                output_queue.put(
                    (
                        "probe_tile_result",
                        worker_id,
                        request_id,
                        ok,
                        seconds,
                        free_before,
                        free_after,
                        total,
                    )
                )
                continue

            if command == "probe_batch":
                request_id = job[1]
                tile_size, tile_pad, batch_size = map(int, job[2:5])
                patch_h = min(input_attached.array.shape[0], tile_size + 2 * tile_pad)
                patch_w = min(input_attached.array.shape[1], tile_size + 2 * tile_pad)
                ok, seconds = _probe_shape(
                    model,
                    input_attached.array,
                    device,
                    config.fp16,
                    config.channels_last,
                    patch_h,
                    patch_w,
                    batch_size,
                )
                output_queue.put(
                    (
                        "probe_batch_result",
                        worker_id,
                        request_id,
                        ok,
                        seconds,
                        batch_size,
                    )
                )
                continue

            if command == "tiles":
                frame_id, regions, batch_size = job[1], job[2], int(job[3])
                timings = _worker_process_regions(
                    model,
                    input_attached.array,
                    output_attached.array,
                    regions,
                    batch_size,
                    native_scale,
                    device,
                    config.fp16,
                    config.channels_last,
                )
                output_queue.put(("tiles_result", worker_id, frame_id, timings))
                continue

            raise RuntimeError(f"Unknown fast worker command: {command}")
    except Exception as error:
        output_queue.put(("error", worker_id, repr(error), traceback.format_exc()))
    finally:
        if input_attached is not None:
            input_attached.close()
        if output_attached is not None:
            output_attached.close()


class SharedMemoryWorkers:
    def __init__(self, gpu_ids: Sequence[Optional[int]], config: base.WorkerConfig):
        global _ACTIVE_WORKERS
        if any(gpu_id is None for gpu_id in gpu_ids):
            raise RuntimeError("Shared-memory workers require CUDA GPU IDs")
        self.context = mp.get_context("spawn")
        self.output_queue = self.context.Queue()
        self.input_queues = [self.context.Queue(maxsize=1) for _ in gpu_ids]
        self.processes = []
        self.gpu_ids = [int(gpu_id) for gpu_id in gpu_ids if gpu_id is not None]
        self.config = config
        self.stage_timings: Dict[str, float] = {}
        self.input_buffer: Optional[SharedArray] = None
        self.output_buffer: Optional[SharedArray] = None
        self.frame_shape: Optional[tuple[int, int, int]] = None
        self.selected_tile: Optional[int] = None
        self.worker_batches: list[int] = [max(1, config.batch_size)] * len(self.gpu_ids)
        self._closed = False
        self._request_id = 0

        for worker_id, gpu_id in enumerate(self.gpu_ids):
            process = self.context.Process(
                target=fast_worker_main,
                args=(
                    worker_id,
                    gpu_id,
                    self.input_queues[worker_id],
                    self.output_queue,
                    asdict(config),
                ),
                daemon=True,
            )
            process.start()
            self.processes.append(process)
        try:
            self._wait_ready()
        except Exception:
            self.close()
            raise
        _ACTIVE_WORKERS = self

    def _next_request(self) -> int:
        self._request_id += 1
        return self._request_id

    def _wait_ready(self) -> None:
        ready = 0
        deadline = time.monotonic() + 300
        while ready < len(self.processes):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Timed out while loading Real-ESRGAN workers")
            try:
                message = self.output_queue.get(timeout=remaining)
            except queue.Empty as error:
                raise TimeoutError("Timed out while loading Real-ESRGAN workers") from error
            if message[0] == "error":
                raise RuntimeError(
                    f"Worker {message[1]} failed: {message[2]}\n{message[3]}"
                )
            if message[0] == "ready":
                print(
                    f"[gpu] shared-memory worker={message[1]} model resident on {message[2]}",
                    flush=True,
                )
                ready += 1

    def _wait_messages(self, expected: str, identity: int) -> list[tuple]:
        results: list[tuple] = []
        while len(results) < len(self.processes):
            message = self.output_queue.get()
            if message[0] == "error":
                raise RuntimeError(
                    f"Worker {message[1]} failed: {message[2]}\n{message[3]}"
                )
            if message[0] != expected or message[2] != identity:
                raise RuntimeError(f"Unexpected worker message: {message[0]}")
            results.append(message)
        return results

    def _allocate(self, frame: np.ndarray) -> None:
        shape = tuple(frame.shape)
        if self.frame_shape == shape:
            return
        if self.input_buffer is not None:
            self.input_buffer.close()
        if self.output_buffer is not None:
            self.output_buffer.close()

        output_dtype = np.float16 if self.config.fp16 else np.float32
        self.input_buffer = SharedArray(shape, np.float32, "realesrgan-input")
        self.output_buffer = SharedArray(
            (
                shape[0] * self.config.native_scale,
                shape[1] * self.config.native_scale,
                3,
            ),
            output_dtype,
            "realesrgan-output",
        )
        self.frame_shape = shape

        for input_queue in self.input_queues:
            input_queue.put(
                ("attach", self.input_buffer.spec, self.output_buffer.spec)
            )
        attached = 0
        while attached < len(self.processes):
            message = self.output_queue.get()
            if message[0] == "error":
                raise RuntimeError(
                    f"Worker {message[1]} failed: {message[2]}\n{message[3]}"
                )
            if message[0] != "attached":
                raise RuntimeError(f"Unexpected worker message: {message[0]}")
            attached += 1

        print(
            f"[shared-memory] input={self.input_buffer.spec.backend}:"
            f"{self.input_buffer.size / 2**20:.1f}MiB, "
            f"output={self.output_buffer.spec.backend}:"
            f"{self.output_buffer.size / 2**20:.1f}MiB, "
            f"dtype={output_dtype}",
            flush=True,
        )

    def prepare_frame(self, frame: np.ndarray, tile_pad: int) -> None:
        self._allocate(frame)
        assert self.input_buffer is not None
        if frame.dtype == np.uint8:
            np.multiply(frame, 1.0 / 255.0, out=self.input_buffer.array, casting="unsafe")
        elif frame.dtype == np.float32:
            np.clip(frame, 0.0, 1.0, out=self.input_buffer.array)
        else:
            raise TypeError(f"Unsupported shared input dtype: {frame.dtype}")

        if self.selected_tile is None:
            self._auto_tune(tile_pad)

    def _auto_tune(self, tile_pad: int) -> None:
        assert self.frame_shape is not None
        height, width = self.frame_shape[:2]
        fallback_tile = max(_MIN_TILE_SIZE, min(_MAX_TILE_SIZE, 256))
        selected = fallback_tile

        if _AUTO_TILE:
            for candidate in _tile_candidates(width, height, _MAX_TILE_SIZE):
                request = self._next_request()
                for input_queue in self.input_queues:
                    input_queue.put(("probe_tile", request, candidate, tile_pad))
                messages = self._wait_messages("probe_tile_result", request)
                if all(bool(message[3]) for message in messages):
                    selected = candidate
                    print(
                        f"[auto-tile] selected={selected}, "
                        + ", ".join(
                            f"gpu{message[1]}={message[4]:.3f}s/"
                            f"free={message[6] / 2**30:.2f}GiB"
                            for message in messages
                        ),
                        flush=True,
                    )
                    break
                print(
                    f"[auto-tile] rejected={candidate} due to OOM on at least one GPU",
                    flush=True,
                )
        else:
            selected = max(_MIN_TILE_SIZE, int(_REQUESTED_TILE_SIZE))

        self.selected_tile = selected

        if _AUTO_BATCH:
            batches: list[int] = []
            candidates = _batch_candidates(_MAX_BATCH_SIZE)
            for worker_id, input_queue in enumerate(self.input_queues):
                best = 1
                for candidate in candidates:
                    request = self._next_request()
                    input_queue.put(
                        (
                            "probe_batch",
                            request,
                            selected,
                            tile_pad,
                            candidate,
                        )
                    )
                    while True:
                        message = self.output_queue.get()
                        if message[0] == "error":
                            raise RuntimeError(
                                f"Worker {message[1]} failed: {message[2]}\n{message[3]}"
                            )
                        if (
                            message[0] == "probe_batch_result"
                            and message[1] == worker_id
                            and message[2] == request
                        ):
                            break
                        raise RuntimeError(
                            f"Unexpected worker message during batch probe: {message[0]}"
                        )
                    if not bool(message[3]):
                        break
                    best = candidate
                batches.append(best)
            self.worker_batches = batches
        else:
            self.worker_batches = [max(1, _REQUESTED_BATCH_SIZE)] * len(self.processes)

        print(
            f"[auto-batch] tile={self.selected_tile}, per_gpu={self.worker_batches}",
            flush=True,
        )

    def selected_regions(self, tile_pad: int) -> list[FastTileRegion]:
        if self.selected_tile is None or self.frame_shape is None:
            raise RuntimeError("Auto tuning has not completed")
        height, width = self.frame_shape[:2]
        return _regions(width, height, self.selected_tile, tile_pad)

    def infer_tiles(
        self,
        frame_id: int,
        regions: Sequence[FastTileRegion],
    ) -> Dict[int, np.ndarray]:
        lanes: list[list[FastTileRegion]] = [[] for _ in self.processes]
        loads = [0] * len(self.processes)
        for region in sorted(regions, key=lambda item: item.area, reverse=True):
            lane = min(range(len(lanes)), key=loads.__getitem__)
            lanes[lane].append(region)
            loads[lane] += region.area

        for worker_id, input_queue in enumerate(self.input_queues):
            input_queue.put(
                (
                    "tiles",
                    frame_id,
                    lanes[worker_id],
                    self.worker_batches[worker_id],
                )
            )
        messages = self._wait_messages("tiles_result", frame_id)
        for message in messages:
            for name, value in message[3].items():
                self.stage_timings[name] = self.stage_timings.get(name, 0.0) + float(value)
        return {}

    def infer_frames(
        self,
        batch_id: int,
        indexed_frames: Sequence[Tuple[int, np.ndarray]],
    ) -> Dict[int, np.ndarray]:
        raise RuntimeError(
            "The fast runner always uses auto-tuned tiles; full-frame queue transport is disabled"
        )

    def output_float32(self) -> np.ndarray:
        if self.output_buffer is None:
            raise RuntimeError("Shared output is not allocated")
        return np.ascontiguousarray(
            self.output_buffer.array.astype(np.float32, copy=True)
        )

    def close(self) -> None:
        global _ACTIVE_WORKERS
        if self._closed:
            return
        self._closed = True
        for input_queue in self.input_queues:
            try:
                input_queue.put_nowait(None)
            except queue.Full:
                pass
        for process in self.processes:
            process.join(timeout=10)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        for input_queue in self.input_queues:
            input_queue.close()
        self.output_queue.close()
        if self.input_buffer is not None:
            self.input_buffer.close()
            self.input_buffer = None
        if self.output_buffer is not None:
            self.output_buffer.close()
            self.output_buffer = None
        if _ACTIVE_WORKERS is self:
            _ACTIVE_WORKERS = None


class AutoTileProcessor:
    """Adapter matching the original TileProcessor API without tile payloads."""

    def __init__(
        self,
        tile_size: int,
        tile_pad: int,
        scale: float,
        verify: bool = False,
    ):
        self.requested_tile_size = tile_size
        self.tile_pad = tile_pad
        self.scale = int(scale)
        self.verify = verify
        self._regions: list[FastTileRegion] = []

    def split(
        self,
        frame: np.ndarray,
    ) -> tuple[list[FastTileRegion], list[FastTileRegion]]:
        if _ACTIVE_WORKERS is None:
            raise RuntimeError("Shared-memory workers are unavailable")
        _ACTIVE_WORKERS.prepare_frame(frame, self.tile_pad)
        self._regions = _ACTIVE_WORKERS.selected_regions(self.tile_pad)
        return self._regions, self._regions

    def stitch(
        self,
        outputs: Mapping[int, np.ndarray],
        regions: Sequence[FastTileRegion],
        input_width: int,
        input_height: int,
    ) -> np.ndarray:
        del outputs, regions
        if _ACTIVE_WORKERS is None:
            raise RuntimeError("Shared-memory workers are unavailable")
        expected = (
            input_height * self.scale,
            input_width * self.scale,
            3,
        )
        result = _ACTIVE_WORKERS.output_float32()
        if result.shape != expected:
            raise RuntimeError(f"Shared output shape mismatch: {result.shape} != {expected}")
        return result


def build_parser() -> argparse.ArgumentParser:
    parser = base.build_parser()
    parser.description = (
        "Shared-memory, auto-tuned multi-GPU Real-ESRGAN video enhancement."
    )
    parser.add_argument(
        "--auto-tile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Probe the largest tile that succeeds on every selected GPU",
    )
    parser.add_argument(
        "--max-tile-size",
        type=int,
        default=1024,
        help="Upper bound for automatic Real-ESRGAN tile probing",
    )
    parser.add_argument(
        "--auto-batch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Probe the largest safe batch independently on each GPU",
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=16,
        help="Upper bound for automatic per-GPU batch probing",
    )
    return parser


def main() -> None:
    global _AUTO_TILE, _AUTO_BATCH, _MAX_TILE_SIZE, _MAX_BATCH_SIZE
    global _REQUESTED_TILE_SIZE, _REQUESTED_BATCH_SIZE
    parser = build_parser()
    args = parser.parse_args()
    base.apply_legacy_args(args, os.sys.argv[1:])
    base.validate_args(args)
    if args.max_tile_size < _MIN_TILE_SIZE or args.max_tile_size % 4:
        raise ValueError("--max-tile-size must be at least 256 and divisible by 4")
    if args.max_batch_size < 1:
        raise ValueError("--max-batch-size must be positive")

    _AUTO_TILE = bool(args.auto_tile)
    _AUTO_BATCH = bool(args.auto_batch)
    _MAX_TILE_SIZE = int(args.max_tile_size)
    _MAX_BATCH_SIZE = int(args.max_batch_size)
    _REQUESTED_TILE_SIZE = int(args.tile_size or _MIN_TILE_SIZE)
    _REQUESTED_BATCH_SIZE = int(args.batch_size)

    # Keep the validated base process loop and replace only the worker/tiler
    # implementations it resolves through module globals.
    base.PersistentWorkers = SharedMemoryWorkers
    base.TileProcessor = AutoTileProcessor

    # Force the tile branch; the adapter selects the actual size after the
    # BasicVSR++ and Real-ESRGAN models are resident on the GPUs.
    if args.tile_size == 0:
        args.tile_size = _MIN_TILE_SIZE
    base.process_video(args)


if __name__ == "__main__":
    mp.freeze_support()
    main()
