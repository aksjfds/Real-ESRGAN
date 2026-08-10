"""Low-coupling video orchestration for BVS -> RIFE -> Real-ESRGAN."""

from __future__ import annotations

from fractions import Fraction
import sys
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

from . import pipeline as base_pipeline
from . import runtime as base
from .clip_source import ClipSource
from .gpu_workers import UnifiedGPUWorkers
from .scheduler_loop import run_scheduler
from .scheduler_state import SchedulerState
from .timeline import TimelinePlanner, ceil_fraction


def process_video(args) -> None:
    requested_gpu_ids = base_pipeline.base.parse_gpu_ids(args.gpu_ids)
    if len(requested_gpu_ids) < 2:
        raise RuntimeError(
            "Unified GPU scheduling requires at least two CUDA GPUs"
        )
    if any(gpu is None for gpu in requested_gpu_ids):
        raise RuntimeError(
            "Unified GPU scheduling requires CUDA GPUs"
        )
    if (
        base_pipeline.base._require_encoder is None
        or base_pipeline.base._writer_type is None
    ):
        raise RuntimeError(
            "Encoding backend is not configured. Run through root inference.py."
        )

    base = base_pipeline.base
    base.require_binary(args.ffmpeg_bin)
    base.require_binary(args.ffprobe_bin)
    base._require_encoder(args.ffmpeg_bin, args.video_codec)

    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input video not found: {input_path}")
    if input_path == output_path:
        raise ValueError("Input and output paths must be different.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_video = output_path.with_name(
        output_path.stem + ".video_only.tmp.mp4"
    )

    info = base.probe_video(input_path, args.ffprobe_bin)
    source_rate = Fraction(info.fps_num, info.fps_den)
    output_rate = Fraction(
        str(float(args.rife_fps))
    ).limit_denominator(100000)
    if output_rate < source_rate:
        raise ValueError(
            "--rife-fps must be >= source FPS; "
            f"got {float(output_rate):g} < {float(source_rate):g}"
        )

    rife_enabled = output_rate > source_rate
    source_fps = float(source_rate)
    output_fps = float(output_rate)
    source_rate_text = (
        f"{source_rate.numerator}/{source_rate.denominator}"
    )
    output_rate_text = (
        f"{output_rate.numerator}/{output_rate.denominator}"
    )

    in_w, in_h = info.width, info.height
    out_w = int(round(in_w * args.scale))
    out_h = int(round(in_h * args.scale))
    if out_w % 2 or out_h % 2:
        raise ValueError(
            f"4:2:0 output needs even dimensions, got {out_w}x{out_h}."
        )

    start, duration, expected_source = base.resolve_range(
        info,
        args.start_time,
        args.test_seconds,
        source_rate,
    )
    expected_output = max(
        1,
        int(round(duration * output_fps)),
    )
    end = start + duration

    gpu_ids = [int(gpu) for gpu in requested_gpu_ids]
    strength = float(args.bvs_strength)
    clip_length = int(args.bvs_clip_length)
    tile_size = int(args.bvs_tile_size)
    batch_size = int(args.bvs_batch_size)
    overlap = 2
    scene_threshold = 0.30

    native_scale = base._model_native_scale(args.model)
    config = base.WorkerConfig(
        args.model,
        base.resolve_model_paths(args),
    )
    dtype = (
        np.dtype("<u2")
        if info.bit_depth == 10
        else np.dtype(np.uint8)
    )
    input_shape = (in_h, in_w, 3)
    native_shape = (
        in_h * native_scale,
        in_w * native_scale,
        3,
    )

    from .checkpoint_parts import resolve_checkpoint

    checkpoint_path = resolve_checkpoint(
        Path(__file__).resolve().parent / "weights"
    )
    rife_weights = ""
    if rife_enabled:
        from .rife425 import resolve_rife425_weights

        rife_weights = str(resolve_rife425_weights())

    planner = TimelinePlanner(
        source_rate=source_rate,
        output_rate=output_rate,
        expected_output=expected_output,
        duplicate_threshold=0.002,
        scene_threshold=scene_threshold,
    )

    max_targets_per_interval = max(
        1,
        ceil_fraction(output_rate / source_rate) + 1,
    )
    input_slots = max(2, clip_length * batch_size)
    frame_output_slots = (
        2 * input_slots
        + max(4, max_targets_per_interval)
    )
    bvs_headroom = (
        input_slots
        + (max_targets_per_interval if rife_enabled else 0)
    )

    mode = (
        "test"
        if args.test_seconds > 0
        else "full/selected range"
    )
    if float(args.scale) == float(native_scale):
        scale_text = f"native={native_scale}x"
    else:
        scale_text = (
            f"native={native_scale}x -> final={args.scale:g}x | "
            "resample=full-frame Lanczos4"
        )

    from .scheduler_reporting import print_run_header

    print_run_header(
        input_path=input_path,
        in_w=in_w,
        in_h=in_h,
        info=info,
        out_w=out_w,
        out_h=out_h,
        output_fps=output_fps,
        video_codec=args.video_codec,
        output_pix_fmt=base._output_pixel_format(
            args.video_codec,
            info.bit_depth,
        ),
        mode=mode,
        start=base.format_seconds(start),
        end=base.format_seconds(end),
        duration=duration,
        expected_source=expected_source,
        expected_output=expected_output,
        model=args.model,
        scale_text=scale_text,
        strength=strength,
        clip_length=clip_length,
        tile_size=tile_size,
        batch_size=batch_size,
        rife_enabled=rife_enabled,
        source_fps=source_fps,
    )

    writer = None
    workers = None
    pump = None
    progress = None
    raw_reader = None
    clip_source = None

    started = time.monotonic()
    worker_model_time = 0.0
    flush_time = 0.0
    audio_time = 0.0
    scheduler_idle = 0.0
    state = None
    clean = False

    try:
        writer = base._writer_type(
            temp_video,
            args.ffmpeg_bin,
            out_w,
            out_h,
            output_rate_text,
            output_rate_text,
            args.video_codec,
            args.crf,
            args.preset,
            args.cq,
            args.nvenc_preset,
            args.encode_gpu,
        )
        raw_reader = base.RawVideoReader(
            input_path,
            args.ffmpeg_bin,
            in_w,
            in_h,
            source_rate_text,
            start,
            duration,
            info.bit_depth,
        )
        clip_source = ClipSource(
            raw_reader,
            clip_length=clip_length,
            overlap=overlap,
            scene_threshold=scene_threshold,
        )
        raw_reader = None

        bvs_config = {
            "gpu_id": 0,
            "strength": strength,
            "clip_length": clip_length,
            "clip_overlap": overlap,
            "tile_size": tile_size,
            "tile_pad": 32,
            "fp16": True,
            "scene_threshold": scene_threshold,
            "model_path": str(checkpoint_path),
        }

        model_started = time.monotonic()
        workers = UnifiedGPUWorkers(
            gpu_ids=gpu_ids,
            config=config,
            bvs_config=bvs_config,
            rife_weights=rife_weights,
            input_shape=input_shape,
            sr_output_shape=native_shape,
            dtype=dtype,
            input_slots=input_slots,
            frame_output_slots=frame_output_slots,
        )
        worker_model_time = time.monotonic() - model_started

        print(
            "Pipeline: unified GPU workers ready | "
            f"shared_memory={workers.memory_mib:.0f} MiB | "
            f"frame_slots={frame_output_slots}/GPU",
            flush=True,
        )
        print(
            "Pipeline: locality-aware scheduling prefers the GPU that "
            "already owns each FrameHandle",
            flush=True,
        )
        print(flush=True)

        progress = tqdm(
            total=expected_output,
            desc="Real-ESRGAN",
            unit="frame",
            dynamic_ncols=True,
            mininterval=1.0,
            file=sys.stdout,
        )
        pump = base_pipeline.OutputPump(
            workers,
            writer,
            out_w,
            out_h,
            progress,
            started,
        )

        state = SchedulerState(
            workers=workers,
            planner=planner,
            clip_source=clip_source,
            gpu_ids=gpu_ids,
            batch_size=batch_size,
            bvs_headroom=bvs_headroom,
            expected_output=expected_output,
        )
        scheduler_idle = run_scheduler(
            state,
            pump,
            expected_output,
            frame_output_slots,
        )
        pump.finish()

        flush_started = time.monotonic()
        writer.close()
        flush_time = time.monotonic() - flush_started
        writer = None
        clean = True
    finally:
        if pump is not None:
            try:
                pump.stop()
            except Exception:
                pass
        if progress is not None:
            progress.close()
        if clip_source is not None:
            try:
                clip_source.close()
            except Exception:
                pass
        if raw_reader is not None:
            try:
                raw_reader.close()
            except Exception:
                pass
        if writer is not None:
            try:
                writer.close()
            except Exception:
                pass
        if workers is not None:
            workers.close()

    processed = pump.processed if pump is not None else 0
    if not clean or processed == 0 or state is None:
        raise RuntimeError("No complete video was encoded.")

    actual_duration = processed / output_fps
    audio_started = time.monotonic()
    base.mux_audio(
        temp_video,
        input_path,
        output_path,
        args.ffmpeg_bin,
        start,
        actual_duration,
        info.has_audio,
        args.audio_codec,
        args.audio_bitrate,
    )
    audio_time = time.monotonic() - audio_started
    if temp_video.exists():
        temp_video.unlink()

    elapsed = time.monotonic() - started
    size_mib = output_path.stat().st_size / 2**20
    bitrate = (
        output_path.stat().st_size
        * 8
        / max(actual_duration, 1e-6)
        / 1_000_000
    )
    fps = processed / max(elapsed, 1e-6)
    selected_tile = min(
        state.selected_tiles,
        default=tile_size,
    )
    decode_elapsed = float(clip_source.decode_elapsed)
    scene_cuts = int(clip_source.scene_cuts)

    from .scheduler_reporting import print_completed

    print_completed(
        base=base,
        processed=processed,
        start=start,
        actual_duration=actual_duration,
        fps=fps,
        elapsed=elapsed,
        worker_model_time=worker_model_time,
        decode_elapsed=decode_elapsed,
        state=state,
        scheduler_idle=scheduler_idle,
        pump=pump,
        flush_time=flush_time,
        audio_time=audio_time,
        selected_tile=selected_tile,
        clip_length=clip_length,
        batch_size=batch_size,
        strength=strength,
        scene_cuts=scene_cuts,
        size_mib=size_mib,
        bitrate=bitrate,
        output_path=output_path,
    )
