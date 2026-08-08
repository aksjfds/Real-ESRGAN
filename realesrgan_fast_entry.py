#!/usr/bin/env python3
"""Kaggle dual-GPU optimized entry point for the shared-memory Real-ESRGAN runner.

The base runner remains multi-GPU: one persistent process and one model copy per
selected CUDA device.  This entry adds the performance mechanisms proven in the
AutoDL runtime, adapted to multiple GPUs instead of replacing the scheduler with
a single-GPU worker:

* guarded POSIX shared-memory allocation with /kaggle/working mmap fallback;
* a two-slot H2D / model / D2H CUDA-stream pipeline inside every GPU worker;
* speed-based tile and per-GPU batch auto tuning;
* throughput-aware tile scheduling for heterogeneous GPUs;
* quality-safe automatic tiles, preferring sizes >= 512;
* 1536 max-tile and 32 max-batch defaults.
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
    )
    missing = [name for name in required if not hasattr(module, name)]
    if missing:
        raise ImportError(
            f"Local runner {module_path} is missing required symbols: {missing}"
        )
    return module


base = _load_local_realesrgan()

import realesrgan_fast as fast  # noqa: E402

fast.base = base


# Avoid asking multiprocessing.shared_memory to allocate arrays that clearly do
# not fit in Docker's /dev/shm quota.  The fallback is persistent file-backed
# mmap under Kaggle working storage, preserving zero-pickle tile transport.
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
    root = Path("/kaggle/working")
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


# Upgrade every GPU worker independently.  Under multiprocessing spawn this
# entry module is re-imported in child processes, so the patched worker function
# is present before fast_worker_main begins processing jobs.
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
    """Overlap CPU packing, H2D, model compute and D2H with two CUDA slots."""
    grouped: dict[tuple[int, int, int], list[fast.FastTileRegion]] = defaultdict(list)
    for region in regions:
        grouped[region.patch_shape].append(region)

    chunks: list[list[fast.FastTileRegion]] = []
    for shape in sorted(grouped):
        group = grouped[shape]
        step = max(1, int(batch_size))
        for offset in range(0, len(group), step):
            chunks.append(group[offset : offset + step])

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
            f"[pipeline] {device} two-slot overlap OOM; falling back to synchronous inference",
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


class SpeedTunedMultiGPUWorkers(fast.SharedMemoryWorkers):
    """Multi-GPU workers tuned for minimum combined frame completion time."""

    def __init__(
        self,
        gpu_ids: Sequence[Optional[int]],
        config: base.WorkerConfig,
    ) -> None:
        if len(gpu_ids) < 1 or any(item is None for item in gpu_ids):
            raise RuntimeError("Kaggle fast runtime requires one or more CUDA GPU IDs")
        super().__init__(gpu_ids, config)
        self.worker_throughputs: list[float] = [1.0] * len(self.processes)
        self.worker_full_work_estimates: list[float] = [1.0] * len(self.processes)

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
        worker_id: int,
        tile_size: int,
        tile_pad: int,
        batch: int,
    ) -> tuple[bool, float]:
        request = self._next_request()
        self.input_queues[worker_id].put(
            ("probe_batch", request, tile_size, tile_pad, batch)
        )
        while True:
            try:
                message = self.output_queue.get(timeout=300.0)
            except queue.Empty as error:
                raise TimeoutError(
                    f"Timed out probing tile={tile_size}, batch={batch} on gpu{worker_id}"
                ) from error
            if message[0] == "error":
                raise RuntimeError(
                    f"Worker {message[1]} failed: {message[2]}\n{message[3]}"
                )
            if (
                message[0] == "probe_batch_result"
                and message[1] == worker_id
                and message[2] == request
            ):
                return bool(message[3]), float(message[4])
            raise RuntimeError(
                f"Unexpected worker message during speed probe: {message[0]}"
            )

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
                tile_candidates = [
                    max(fast._MIN_TILE_SIZE, min(fast._MAX_TILE_SIZE, max(width, height)))
                ]
        else:
            tile_candidates = [
                max(fast._MIN_TILE_SIZE, int(fast._REQUESTED_TILE_SIZE))
            ]

        best_global: Optional[
            tuple[float, int, list[int], list[float], list[float]]
        ] = None

        for tile_size in tile_candidates:
            groups = self._shape_groups(width, height, tile_size, tile_pad)
            region_count = sum(groups.values())
            largest_group = max(groups.values(), default=1)
            max_area = max((shape[0] * shape[1] for shape in groups), default=1)
            total_area = float(
                sum(shape[0] * shape[1] * count for shape, count in groups.items())
            )

            worker_batches: list[int] = []
            worker_estimates: list[float] = []
            worker_throughputs: list[float] = []
            tile_valid = True

            for worker_id in range(len(self.processes)):
                if fast._AUTO_BATCH:
                    batch_candidates = sorted(
                        {
                            value
                            for value in (
                                *fast._batch_candidates(fast._MAX_BATCH_SIZE),
                                largest_group,
                            )
                            if 1 <= value <= min(largest_group, fast._MAX_BATCH_SIZE)
                        }
                    )
                else:
                    batch_candidates = [
                        min(largest_group, max(1, int(fast._REQUESTED_BATCH_SIZE)))
                    ]

                best_worker: Optional[tuple[float, int, float]] = None
                for batch in batch_candidates:
                    ok, probe_seconds = self._probe_batch(
                        worker_id, tile_size, tile_pad, batch
                    )
                    if not ok:
                        break
                    estimated_full_work = 0.0
                    for shape, count in groups.items():
                        area_ratio = (shape[0] * shape[1]) / max_area
                        estimated_full_work += (
                            math.ceil(count / batch) * probe_seconds * area_ratio
                        )
                    throughput = total_area / max(estimated_full_work, 1e-9)
                    print(
                        f"[auto-tune] tile={tile_size}, gpu={worker_id}, batch={batch}, "
                        f"regions={region_count}, full_work={estimated_full_work:.3f}s, "
                        f"throughput={throughput / 1_000_000.0:.1f} MPix/s",
                        flush=True,
                    )
                    choice = (estimated_full_work, batch, throughput)
                    if best_worker is None or choice[0] < best_worker[0]:
                        best_worker = choice

                if best_worker is None:
                    tile_valid = False
                    print(
                        f"[auto-tune] tile={tile_size} rejected: gpu{worker_id} OOM",
                        flush=True,
                    )
                    break

                estimate, batch, throughput = best_worker
                worker_estimates.append(estimate)
                worker_batches.append(batch)
                worker_throughputs.append(throughput)

            if not tile_valid:
                continue

            # If worker i takes Ti seconds to process all work alone, its rate is
            # W/Ti.  With balanced scheduling the ideal combined makespan is
            # 1 / sum(1/Ti).  This directly generalizes the AutoDL single-GPU
            # estimate to N independent GPU workers.
            combined_estimate = 1.0 / sum(
                1.0 / max(value, 1e-9) for value in worker_estimates
            )
            print(
                f"[auto-tune] tile={tile_size}, combined={combined_estimate:.3f}s/frame, "
                f"batches={worker_batches}",
                flush=True,
            )
            choice = (
                combined_estimate,
                tile_size,
                worker_batches,
                worker_estimates,
                worker_throughputs,
            )
            if best_global is None or choice[0] < best_global[0]:
                best_global = choice

        if best_global is None:
            raise RuntimeError(
                "No quality-preserving tile/batch combination fit on every selected GPU"
            )

        (
            combined_estimate,
            self.selected_tile,
            self.worker_batches,
            self.worker_full_work_estimates,
            self.worker_throughputs,
        ) = best_global
        print(
            f"[auto-tune] selected tile={self.selected_tile}, "
            f"per_gpu_batch={self.worker_batches}, "
            f"estimated={combined_estimate:.3f}s/frame, "
            f"per_gpu_throughput={[round(v / 1_000_000.0, 1) for v in self.worker_throughputs]} MPix/s",
            flush=True,
        )

    def infer_tiles(
        self,
        frame_id: int,
        regions: Sequence[fast.FastTileRegion],
    ) -> Dict[int, np.ndarray]:
        lanes: list[list[fast.FastTileRegion]] = [[] for _ in self.processes]
        loads = [0.0] * len(self.processes)
        throughputs = [max(value, 1e-9) for value in self.worker_throughputs]

        # Longest-processing-time greedy scheduling, but normalized by measured
        # per-GPU throughput instead of assuming identical GPUs.
        for region in sorted(regions, key=lambda item: item.area, reverse=True):
            lane = min(
                range(len(lanes)),
                key=lambda index: (loads[index] + region.area) / throughputs[index],
            )
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


fast.SharedMemoryWorkers = SpeedTunedMultiGPUWorkers


_original_build_parser = fast.build_parser


def build_parser() -> argparse.ArgumentParser:
    parser = _original_build_parser()
    parser.description = (
        "Kaggle multi-GPU pipelined Real-ESRGAN video enhancement."
    )
    for action in parser._actions:
        if action.dest == "max_tile_size":
            action.default = 1536
            action.help = "Upper bound for speed-based automatic tile tuning"
        elif action.dest == "max_batch_size":
            action.default = 32
            action.help = "Upper bound for speed-based per-GPU batch tuning"
    return parser


fast.build_parser = build_parser


if __name__ == "__main__":
    fast.mp.freeze_support()
    fast.main()
