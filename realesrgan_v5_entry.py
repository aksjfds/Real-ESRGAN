#!/usr/bin/env python3
"""Quality-first Kaggle entry for AnimeVideo-v3.

v5.0 keeps the v4.2 dual-GPU scheduler but changes the quality-critical path:

* decode FFmpeg frames as RGB48LE and convert once to float32 [0, 1];
* force FP32 AnimeVideo-v3 inference and disable TF32/reduced FP16 reductions;
* retain unclamped native x4 model cores until full-frame reconstruction;
* keep the existing Lanczos4 arbitrary-scale baseline for compatibility;
* restore only source-supported dark-line contrast after x4 -> final-scale resize;
* keep line restoration fail-open: uncertain/error pixels return the SR baseline.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import time
import traceback
from collections import defaultdict
from typing import Dict, Optional, Sequence

import numpy as np
import torch

import realesrgan_fast_entry as v42
from enhance.line_restore import LineRestoreConfig, restore_dark_lines


base = v42.base
fast = v42.fast

_CURRENT_SOURCE_FRAME: Optional[np.ndarray] = None
_LINE_CONFIG = LineRestoreConfig()
_LINE_FRAME_INDEX = 0


def _disable_reduced_precision() -> None:
    """Prefer IEEE FP32 over TF32/reduced-precision fast paths."""
    try:
        torch.backends.cudnn.allow_tf32 = False
    except (AttributeError, RuntimeError):
        pass
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
    except (AttributeError, RuntimeError):
        pass
    try:
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    except (AttributeError, RuntimeError):
        pass
    try:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    except (AttributeError, RuntimeError):
        pass


class HighPrecisionRawVideoReader:
    """Decode to RGB48LE so 10-bit sources are not truncated to RGB24 first."""

    def __init__(
        self,
        input_path,
        ffmpeg_bin: str,
        width: int,
        height: int,
        fps_rate: str,
        start: float,
        duration: float,
    ) -> None:
        self.frame_bytes = width * height * 3 * 2
        vf = (
            f"scale={width}:{height}:"
            "flags=lanczos+accurate_rnd+full_chroma_int,"
            f"fps={fps_rate}"
        )
        command = [ffmpeg_bin, "-hide_banner", "-loglevel", "error"]
        if start > 0:
            command += ["-ss", f"{start:.6f}"]
        command += ["-i", str(input_path)]
        command += [
            "-t", f"{duration:.6f}",
            "-vf", vf,
            "-an",
            "-f", "rawvideo",
            "-pix_fmt", "rgb48le",
            "pipe:1",
        ]
        self.width = width
        self.height = height
        self.process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def read(self) -> Optional[np.ndarray]:
        assert self.process.stdout is not None
        data = self.process.stdout.read(self.frame_bytes)
        if not data:
            return None
        if len(data) != self.frame_bytes:
            raise RuntimeError(
                f"ffmpeg returned a partial RGB48 frame "
                f"({len(data)}/{self.frame_bytes} bytes)."
            )
        frame16 = np.frombuffer(data, dtype="<u2").reshape(
            self.height, self.width, 3
        )
        frame = frame16.astype(np.float32)
        frame *= 1.0 / 65535.0
        return np.ascontiguousarray(frame)

    def close(self) -> None:
        if self.process.stdout is not None:
            self.process.stdout.close()
        stderr = b""
        if self.process.stderr is not None:
            stderr = self.process.stderr.read()
            self.process.stderr.close()
        return_code = self.process.wait()
        if return_code != 0:
            raise RuntimeError(
                f"ffmpeg RGB48 decode failed (exit {return_code}):\n"
                f"{stderr.decode(errors='replace')}"
            )


def _quality_sync_worker_process_regions(
    model: torch.nn.Module,
    input_array: np.ndarray,
    output_array: np.ndarray,
    regions: Sequence[fast.FastTileRegion],
    native_scale: int,
    device: torch.device,
    channels_last: bool,
) -> Dict[str, float]:
    """One-tile-at-a-time FP32 fallback without early clamp."""
    timings = {
        "realesrgan_cpu_pack": 0.0,
        "realesrgan_h2d": 0.0,
        "realesrgan_model_gpu": 0.0,
        "realesrgan_d2h": 0.0,
        "realesrgan_pipeline_wait": 0.0,
        "realesrgan_shared_write": 0.0,
    }
    for region in regions:
        pack_started = time.monotonic()
        patch = np.ascontiguousarray(
            input_array[
                region.context_y0 : region.context_y1,
                region.context_x0 : region.context_x1,
            ]
        )
        pinned = torch.empty(
            (1, 3, patch.shape[0], patch.shape[1]),
            dtype=torch.float32,
            device="cpu",
            pin_memory=True,
        )
        pinned[0].copy_(torch.from_numpy(patch).permute(2, 0, 1))
        timings["realesrgan_cpu_pack"] += time.monotonic() - pack_started

        h2d_started = time.monotonic()
        tensor = pinned.to(device=device, dtype=torch.float32, non_blocking=True)
        if channels_last:
            tensor = tensor.contiguous(memory_format=torch.channels_last)
        torch.cuda.synchronize(device)
        timings["realesrgan_h2d"] += time.monotonic() - h2d_started

        model_start = torch.cuda.Event(enable_timing=True)
        model_end = torch.cuda.Event(enable_timing=True)
        with torch.inference_mode():
            model_start.record()
            native = model(tensor)
            model_end.record()
            torch.cuda.synchronize(device)
        timings["realesrgan_model_gpu"] += model_start.elapsed_time(model_end) / 1000.0

        crop_x0 = (region.x0 - region.context_x0) * native_scale
        crop_y0 = (region.y0 - region.context_y0) * native_scale
        core_w = (region.x1 - region.x0) * native_scale
        core_h = (region.y1 - region.y0) * native_scale
        core = native[
            0,
            :,
            crop_y0 : crop_y0 + core_h,
            crop_x0 : crop_x0 + core_w,
        ].permute(1, 2, 0).contiguous()

        d2h_started = time.monotonic()
        cpu_core = core.float().cpu().numpy()
        timings["realesrgan_d2h"] += time.monotonic() - d2h_started

        write_started = time.monotonic()
        oy0, oy1 = region.y0 * native_scale, region.y1 * native_scale
        ox0, ox1 = region.x0 * native_scale, region.x1 * native_scale
        np.copyto(output_array[oy0:oy1, ox0:ox1], cpu_core, casting="unsafe")
        timings["realesrgan_shared_write"] += time.monotonic() - write_started
        del native, tensor, pinned, patch

    return timings


def _quality_worker_process_regions(
    model: torch.nn.Module,
    input_array: np.ndarray,
    output_array: np.ndarray,
    regions: Sequence[fast.FastTileRegion],
    batch_size: int,
    native_scale: int,
    device: torch.device,
    fp16: bool,
    channels_last: bool,
) -> Dict[str, float]:
    """Two-slot FP32 pipeline that preserves raw model values until stitching."""
    del fp16
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
                pinned_input[index].copy_(torch.from_numpy(view).permute(2, 0, 1))
            timings["realesrgan_cpu_pack"] += time.monotonic() - pack_started

            h2d_start = torch.cuda.Event(enable_timing=True)
            h2d_end = torch.cuda.Event(enable_timing=True)
            h2d_ready = torch.cuda.Event()
            with torch.cuda.stream(h2d_stream):
                h2d_start.record(h2d_stream)
                gpu_input = pinned_input.to(
                    device=device,
                    dtype=torch.float32,
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
                        dtype=torch.float32,
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
            if len(pending) >= 2:
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
            f"[quality-pipeline] {device} overlap OOM; retrying sequential FP32 tiles",
            flush=True,
        )
        return _quality_sync_worker_process_regions(
            model,
            input_array,
            output_array,
            regions,
            native_scale,
            device,
            channels_last,
        )


def quality_worker_main(
    worker_id: int,
    gpu_id: Optional[int],
    input_queue,
    output_queue,
    config_dict: Dict[str, object],
) -> None:
    """Resident GPU worker with full-precision math policy."""
    input_attached = None
    output_attached = None
    try:
        config = base.WorkerConfig(**config_dict)
        if gpu_id is None:
            raise RuntimeError("v5.0 quality runtime requires CUDA workers")
        torch.cuda.set_device(gpu_id)
        device = torch.device(f"cuda:{gpu_id}")
        _disable_reduced_precision()
        torch.backends.cudnn.benchmark = True
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
                input_attached, output_attached = fast._attach_arrays(job[1], job[2])
                output_queue.put(("attached", worker_id))
                continue

            if input_attached is None or output_attached is None:
                raise RuntimeError("Shared arrays were not attached before inference")

            if command == "probe_tile":
                request_id, tile_size, tile_pad = job[1], int(job[2]), int(job[3])
                patch_h = min(input_attached.array.shape[0], tile_size + 2 * tile_pad)
                patch_w = min(input_attached.array.shape[1], tile_size + 2 * tile_pad)
                free_before, total = torch.cuda.mem_get_info(device)
                ok, seconds = fast._probe_shape(
                    model,
                    input_attached.array,
                    device,
                    False,
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
                ok, seconds = fast._probe_shape(
                    model,
                    input_attached.array,
                    device,
                    False,
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
                timings = fast._worker_process_regions(
                    model,
                    input_attached.array,
                    output_attached.array,
                    regions,
                    batch_size,
                    native_scale,
                    device,
                    False,
                    config.channels_last,
                )
                output_queue.put(("tiles_result", worker_id, frame_id, timings))
                continue

            raise RuntimeError(f"Unknown v5 worker command: {command}")
    except Exception as error:
        output_queue.put(("error", worker_id, repr(error), traceback.format_exc()))
    finally:
        if input_attached is not None:
            input_attached.close()
        if output_attached is not None:
            output_attached.close()


class SourceAwareAutoTileProcessor(fast.AutoTileProcessor):
    """Remember the matching source frame for final line-contrast constraints."""

    def split(self, frame: np.ndarray):
        global _CURRENT_SOURCE_FRAME
        _CURRENT_SOURCE_FRAME = np.ascontiguousarray(frame.astype(np.float32, copy=False))
        return super().split(frame)


def quality_finalize_output_frame(
    native_output: np.ndarray,
    output_width: int,
    output_height: int,
    timings: Dict[str, float],
) -> np.ndarray:
    """Resize native x4 SR, then restore only high-confidence missing dark lines."""
    global _LINE_FRAME_INDEX
    stage_started = time.monotonic()
    baseline = base.full_frame_lanczos(native_output, output_width, output_height)
    timings["lanczos"] = timings.get("lanczos", 0.0) + (time.monotonic() - stage_started)
    baseline = np.ascontiguousarray(baseline, dtype=np.float32)

    source = _CURRENT_SOURCE_FRAME
    if source is None or not _LINE_CONFIG.enabled:
        return baseline

    stage_started = time.monotonic()
    try:
        restored, stats = restore_dark_lines(source, baseline, _LINE_CONFIG)
    except Exception as error:
        print(
            f"[line-restore-warning] fail-open to AnimeVideo-v3 baseline: {error}",
            flush=True,
        )
        return baseline
    timings["line_restore"] = timings.get("line_restore", 0.0) + (
        time.monotonic() - stage_started
    )

    if _LINE_FRAME_INDEX == 0 or (_LINE_FRAME_INDEX + 1) % 60 == 0:
        print(
            "[line-restore] "
            f"frame={_LINE_FRAME_INDEX}, "
            f"modified={stats['modified_fraction'] * 100.0:.2f}%, "
            f"mean_darkening={stats['mean_darkening']:.5f}, "
            f"max_darkening={stats['max_darkening']:.5f}, "
            f"confidence={stats['mean_confidence']:.3f}",
            flush=True,
        )
    _LINE_FRAME_INDEX += 1
    return restored


_original_build_parser = fast.build_parser


def build_parser() -> argparse.ArgumentParser:
    parser = _original_build_parser()
    parser.description = (
        "Kaggle dual-GPU AnimeVideo-v3 quality-first runtime with "
        "source-constrained dark-line restoration."
    )
    for action in parser._actions:
        if action.dest == "fp16":
            action.default = False
            action.help = "v5.0 forces FP32 to avoid precision loss"
    parser.add_argument(
        "--line-restore",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Restore only source-supported dark-line contrast after final scaling",
    )
    parser.add_argument("--line-strength", type=float, default=1.0)
    parser.add_argument("--line-max-recovery", type=float, default=0.18)
    parser.add_argument("--line-max-darkening", type=float, default=0.10)
    parser.add_argument("--line-min-contrast", type=float, default=0.025)
    parser.add_argument("--line-edge-threshold", type=float, default=0.010)
    parser.add_argument("--line-orientation-floor", type=float, default=0.55)
    return parser


fast.build_parser = build_parser
fast._worker_process_regions = _quality_worker_process_regions
fast.fast_worker_main = quality_worker_main
fast.AutoTileProcessor = SourceAwareAutoTileProcessor
base.RawVideoReader = HighPrecisionRawVideoReader
base.finalize_output_frame = quality_finalize_output_frame


def main() -> None:
    global _LINE_CONFIG
    _disable_reduced_precision()

    parser = fast.build_parser()
    args = parser.parse_args()
    base.apply_legacy_args(args, os.sys.argv[1:])
    base.validate_args(args)

    if args.max_tile_size < fast._MIN_TILE_SIZE or args.max_tile_size % 4:
        raise ValueError(
            f"--max-tile-size must be at least {fast._MIN_TILE_SIZE} and divisible by 4"
        )
    if args.max_batch_size < 1:
        raise ValueError("--max-batch-size must be positive")
    if not 0.0 <= args.line_strength <= 2.0:
        raise ValueError("--line-strength must be between 0 and 2")
    if not 0.0 <= args.line_max_recovery <= 1.0:
        raise ValueError("--line-max-recovery must be between 0 and 1")
    if not 0.0 <= args.line_max_darkening <= 1.0:
        raise ValueError("--line-max-darkening must be between 0 and 1")
    if not 0.0 <= args.line_min_contrast <= 1.0:
        raise ValueError("--line-min-contrast must be between 0 and 1")
    if args.line_edge_threshold < 0.0:
        raise ValueError("--line-edge-threshold must be non-negative")
    if not 0.0 <= args.line_orientation_floor <= 1.0:
        raise ValueError("--line-orientation-floor must be between 0 and 1")

    if args.fp16:
        print(
            "[quality] --fp16 requested but v5.0 forces FP32; "
            "use the v4.2 entry for speed-first FP16 inference.",
            flush=True,
        )
    args.fp16 = False

    fast._AUTO_TILE = bool(args.auto_tile)
    fast._AUTO_BATCH = bool(args.auto_batch)
    fast._MAX_TILE_SIZE = int(args.max_tile_size)
    fast._MAX_BATCH_SIZE = int(args.max_batch_size)
    fast._REQUESTED_TILE_SIZE = int(args.tile_size or fast._MIN_TILE_SIZE)
    fast._REQUESTED_BATCH_SIZE = int(args.batch_size)

    _LINE_CONFIG = LineRestoreConfig(
        enabled=bool(args.line_restore),
        strength=float(args.line_strength),
        max_contrast_recovery=float(args.line_max_recovery),
        max_darkening=float(args.line_max_darkening),
        min_reference_contrast=float(args.line_min_contrast),
        edge_threshold=float(args.line_edge_threshold),
        orientation_floor=float(args.line_orientation_floor),
    )

    base.PersistentWorkers = fast.SharedMemoryWorkers
    base.TileProcessor = fast.AutoTileProcessor
    if args.tile_size == 0:
        args.tile_size = fast._MIN_TILE_SIZE

    print(
        "[quality] decode=rgb48le, model=fp32, tf32=off, "
        "native_model_output=unclamped, line_restore="
        f"{_LINE_CONFIG.enabled}",
        flush=True,
    )
    base.process_video(args)


if __name__ == "__main__":
    fast.mp.freeze_support()
    main()
