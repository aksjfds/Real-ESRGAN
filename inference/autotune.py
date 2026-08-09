"""Automatic quality-safe tile/batch tuning for the master inference pipeline."""

from __future__ import annotations

import math
import multiprocessing as mp
import queue
import time
import traceback
from dataclasses import dataclass
from typing import Dict, Optional

import torch


_MIN_QUALITY_TILE = 512
_TILE_CANDIDATES = (1536, 1280, 1024, 896, 768, 640, 576, 512)
_BATCH_CANDIDATES = (1, 2, 4, 6, 8, 12, 16, 24, 32)


@dataclass(frozen=True)
class TuneResult:
    tile_size: int
    batch_size: int
    estimated_seconds: float
    throughput_mpix: float
    tested: int
    rejected_oom: int
    probe_gpu: int
    search_seconds: float


def _axis_starts(length: int, tile_size: int, overlap: int) -> list[int]:
    if length <= tile_size:
        return [0]
    return list(range(0, length, tile_size - overlap))


def _tile_count(width: int, height: int, tile_size: int, overlap: int) -> int:
    return len(_axis_starts(width, tile_size, overlap)) * len(
        _axis_starts(height, tile_size, overlap)
    )


def _probe_forward(
    model: torch.nn.Module,
    device: torch.device,
    fp16: bool,
    channels_last: bool,
    patch_h: int,
    patch_w: int,
    batch_size: int,
) -> tuple[bool, float]:
    """Measure the native model forward only.

    Final target scaling is intentionally excluded because the runtime performs
    one full-frame Lanczos resize after reconstruction, independent of the
    tile/batch candidate being tested.
    """
    dtype = torch.float16 if fp16 else torch.float32
    tensor = None
    output = None
    try:
        tensor = torch.zeros(
            (batch_size, 3, patch_h, patch_w),
            dtype=dtype,
            device=device,
        )
        if channels_last:
            tensor = tensor.contiguous(memory_format=torch.channels_last)

        started = torch.cuda.Event(enable_timing=True)
        ended = torch.cuda.Event(enable_timing=True)
        with torch.inference_mode():
            started.record()
            output = model(tensor)
            ended.record()
            torch.cuda.synchronize(device)
        seconds = started.elapsed_time(ended) / 1000.0
        return True, max(seconds, 1e-9)
    except torch.cuda.OutOfMemoryError:
        return False, 0.0
    finally:
        del output, tensor
        torch.cuda.empty_cache()


def _probe_worker(
    result_queue: mp.Queue,
    gpu_id: int,
    config_dict: Dict[str, object],
    width: int,
    height: int,
    gpu_count: int,
    overlap: int,
    max_tile_size: int,
    max_batch_size: int,
    requested_tile: int,
    requested_batch: int,
) -> None:
    try:
        from inference import runtime as base

        torch.cuda.set_device(gpu_id)
        device = torch.device(f"cuda:{gpu_id}")
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True

        config = base.WorkerConfig(**config_dict)  # type: ignore[arg-type]
        model, _native_scale = base.load_worker_model(config, device)

        tile_candidates = [
            tile
            for tile in _TILE_CANDIDATES
            if tile <= max_tile_size and tile > overlap * 2
        ]
        if _MIN_QUALITY_TILE <= requested_tile <= max_tile_size:
            tile_candidates.append(requested_tile)
        tile_candidates = list(dict.fromkeys(tile_candidates))
        if not tile_candidates:
            raise RuntimeError("No quality-preserving tile candidate is available")

        best: Optional[tuple[float, int, int, float]] = None
        tested = 0
        rejected_oom = 0

        for tile_size in tile_candidates:
            count = _tile_count(width, height, tile_size, overlap)
            worker_counts = [
                len(range(worker_id, count, max(1, gpu_count)))
                for worker_id in range(max(1, gpu_count))
            ]
            max_worker_tiles = max(worker_counts, default=1)

            batch_candidates = [
                value
                for value in _BATCH_CANDIDATES
                if value <= max_batch_size and value <= max_worker_tiles
            ]
            if 1 <= requested_batch <= min(max_batch_size, max_worker_tiles):
                batch_candidates.append(requested_batch)
            if max_worker_tiles <= max_batch_size:
                batch_candidates.append(max_worker_tiles)
            batch_candidates = sorted(set(batch_candidates))

            for batch_size in batch_candidates:
                ok, seconds = _probe_forward(
                    model,
                    device,
                    bool(config.fp16),
                    bool(config.channels_last),
                    tile_size,
                    tile_size,
                    batch_size,
                )
                if not ok:
                    rejected_oom += 1
                    break

                tested += 1
                batches_per_worker = max(
                    math.ceil(worker_tiles / batch_size)
                    for worker_tiles in worker_counts
                )
                estimated = batches_per_worker * seconds
                throughput = (
                    batch_size * tile_size * tile_size / seconds / 1_000_000.0
                )
                choice = (estimated, tile_size, batch_size, throughput)
                if best is None or choice[0] < best[0]:
                    best = choice

        if best is None:
            raise RuntimeError(
                "No quality-preserving tile/batch combination fit on the probe GPU"
            )

        estimated, tile_size, batch_size, throughput = best
        result_queue.put(
            (
                "ok",
                {
                    "tile_size": tile_size,
                    "batch_size": batch_size,
                    "estimated_seconds": estimated,
                    "throughput_mpix": throughput,
                    "tested": tested,
                    "rejected_oom": rejected_oom,
                    "probe_gpu": gpu_id,
                },
            )
        )
    except Exception as error:
        result_queue.put(("error", repr(error), traceback.format_exc()))


def select_parameters(
    *,
    config_dict: Dict[str, object],
    gpu_id: int,
    width: int,
    height: int,
    gpu_count: int,
    overlap: int,
    max_tile_size: int,
    max_batch_size: int,
    requested_tile: int,
    requested_batch: int,
) -> TuneResult:
    """Return the fastest measured quality-safe tiled configuration."""
    if max_tile_size < _MIN_QUALITY_TILE:
        raise ValueError(f"--max-tile-size must be at least {_MIN_QUALITY_TILE}")
    if max_batch_size < 1:
        raise ValueError("--max-batch-size must be at least 1")

    context = mp.get_context("spawn")
    result_queue = context.Queue(maxsize=1)
    started = time.monotonic()
    process = context.Process(
        target=_probe_worker,
        args=(
            result_queue,
            int(gpu_id),
            config_dict,
            int(width),
            int(height),
            int(gpu_count),
            int(overlap),
            int(max_tile_size),
            int(max_batch_size),
            int(requested_tile),
            int(requested_batch),
        ),
    )
    process.start()
    try:
        try:
            message = result_queue.get(timeout=300.0)
        except queue.Empty as error:
            raise TimeoutError("Automatic tile/batch tuning timed out") from error
    finally:
        process.join(timeout=5)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
        result_queue.close()

    if message[0] == "error":
        raise RuntimeError(f"Automatic tuning failed: {message[1]}\n{message[2]}")

    values = message[1]
    return TuneResult(
        tile_size=int(values["tile_size"]),
        batch_size=int(values["batch_size"]),
        estimated_seconds=float(values["estimated_seconds"]),
        throughput_mpix=float(values["throughput_mpix"]),
        tested=int(values["tested"]),
        rejected_oom=int(values["rejected_oom"]),
        probe_gpu=int(values["probe_gpu"]),
        search_seconds=time.monotonic() - started,
    )
