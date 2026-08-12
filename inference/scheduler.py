"""Low-coupling video orchestration for BVS -> RIFE -> Real-ESRGAN."""

from __future__ import annotations

from fractions import Fraction
import json
from pathlib import Path
import re
import subprocess
import time

import numpy as np

from audio.runtime import mux_audio as mux_output_audio
from . import runtime_api as base
from .clip_source import ClipSource
from .gpu_workers import UnifiedGPUWorkers
from .output_runtime import OutputPump, create_progress
from .scheduler_loop import run_scheduler
from .scheduler_state import SchedulerState
from .timeline import TimelinePlanner, ceil_fraction


def _expected_output_bit_depth(args, source_bit_depth: int) -> int:
    if args.video_codec == "av1_nvenc":
        return max(int(source_bit_depth), int(args.av1_bit_depth))
    return int(source_bit_depth)


def _expected_codec_name(video_codec: str) -> str:
    if video_codec in {"av1_nvenc", "libsvtav1", "libaom-av1"}:
        return "av1"
    if video_codec in {"hevc_nvenc", "libx265"}:
        return "hevc"
    if video_codec in {"h264_nvenc", "libx264"}:
        return "h264"
    raise ValueError(f"Unsupported output codec for verification: {video_codec}")


def _stream_bit_depth(stream: dict) -> tuple[int, str]:
    pix_fmt = str(stream.get("pix_fmt") or "unknown").lower()
    raw_value = str(stream.get("bits_per_raw_sample") or "").strip()
    raw_bits = int(raw_value) if raw_value.isdigit() and int(raw_value) > 0 else 0
    if "p010" in pix_fmt:
        bits = 10
    else:
        match = re.search(r"(?:p|gray|rgb|bgr)(9|10|12|14|16)(?:le|be)$", pix_fmt)
        bits = int(match.group(1)) if match else raw_bits or 8
    if bits <= 8:
        return 8, pix_fmt
    if bits == 10:
        return 10, pix_fmt
    raise RuntimeError(
        f"Final output pixel format {pix_fmt!r} reports {bits}-bit samples; "
        "only 8-bit and 10-bit are supported."
    )


def _verify_output_file(
    output_path: Path,
    ffprobe_bin: str,
    *,
    video_codec: str,
    expected_width: int,
    expected_height: int,
    expected_fps: float,
    expected_bit_depth: int,
    expected_audio_streams: int,
    expect_enhanced_audio: bool,
) -> str:
    command = [
        ffprobe_bin,
        "-v", "error",
        "-print_format", "json",
        "-show_streams",
        str(output_path),
    ]
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"final ffprobe verification failed:\n{detail}")

    data = json.loads(result.stdout)
    streams = data.get("streams", [])
    videos = [stream for stream in streams if stream.get("codec_type") == "video"]
    audios = [stream for stream in streams if stream.get("codec_type") == "audio"]
    if len(videos) != 1:
        raise RuntimeError(f"Final output must contain exactly one video stream; got {len(videos)}")

    video = videos[0]
    actual_codec = str(video.get("codec_name") or "unknown")
    expected_codec = _expected_codec_name(video_codec)
    if actual_codec != expected_codec:
        raise RuntimeError(
            f"Final output codec mismatch: expected {expected_codec}, got {actual_codec}"
        )

    width = int(video.get("width") or 0)
    height = int(video.get("height") or 0)
    if (width, height) != (expected_width, expected_height):
        raise RuntimeError(
            "Final output dimensions mismatch: "
            f"expected {expected_width}x{expected_height}, got {width}x{height}"
        )

    rate_text = str(video.get("avg_frame_rate") or video.get("r_frame_rate") or "0/0")
    try:
        actual_fps = float(Fraction(rate_text))
    except (ValueError, ZeroDivisionError) as error:
        raise RuntimeError(f"Final output has invalid frame rate: {rate_text!r}") from error
    if abs(actual_fps - expected_fps) > max(1e-3, expected_fps * 1e-6):
        raise RuntimeError(
            f"Final output FPS mismatch: expected {expected_fps:.6f}, got {actual_fps:.6f}"
        )

    actual_bit_depth, pix_fmt = _stream_bit_depth(video)
    if actual_bit_depth != expected_bit_depth:
        raise RuntimeError(
            "Final output bit depth mismatch: "
            f"expected {expected_bit_depth}-bit, got {actual_bit_depth}-bit ({pix_fmt})"
        )

    profile = str(video.get("profile") or "unknown")
    if expected_codec == "av1" and profile.lower() != "main":
        raise RuntimeError(f"Final AV1 profile mismatch: expected Main, got {profile}")
    if expected_codec == "hevc":
        expected_profile = "main 10" if expected_bit_depth == 10 else "main"
        if profile.lower() != expected_profile:
            raise RuntimeError(
                f"Final HEVC profile mismatch: expected {expected_profile.title()}, got {profile}"
            )

    if len(audios) != expected_audio_streams:
        raise RuntimeError(
            "Final output audio stream count mismatch: "
            f"expected {expected_audio_streams}, got {len(audios)}"
        )
    if expect_enhanced_audio:
        defaults = [int(stream.get("disposition", {}).get("default", 0)) for stream in audios]
        if defaults != [1, 0]:
            raise RuntimeError(
                "Final enhanced/original audio default dispositions are incorrect: "
                f"default={defaults}"
            )

    return (
        f"{actual_codec} | profile={profile} | {actual_bit_depth}-bit ({pix_fmt}) | "
        f"{width}x{height} | {actual_fps:.3f} fps | audio={len(audios)}"
    )


def process_video(args) -> None:
    requested_gpu_ids = base.parse_gpu_ids(args.gpu_ids)
    if len(requested_gpu_ids) < 1:
        raise RuntimeError(
            "Unified GPU scheduling requires at least one CUDA GPU"
        )
    if any(gpu is None for gpu in requested_gpu_ids):
        raise RuntimeError("Unified GPU scheduling requires CUDA GPUs")

    require_encoder, writer_type = base.get_encoding_backend()
    base.require_binary(args.ffmpeg_bin)
    base.require_binary(args.ffprobe_bin)
    require_encoder(args.ffmpeg_bin, args.video_codec)

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
    probe_runtime = getattr(writer_type, "probe_runtime", None)
    if callable(probe_runtime):
        probe_runtime(args, info.bit_depth)

    source_rate = Fraction(info.fps_num, info.fps_den)
    requested_rife_fps = float(args.rife_fps)
    if requested_rife_fps == 0.0:
        output_rate = source_rate
    else:
        output_rate = Fraction(str(requested_rife_fps)).limit_denominator(100000)
        if output_rate < source_rate:
            raise ValueError(
                "--rife-fps must be 0 or >= source FPS; "
                f"got {float(output_rate):g} < {float(source_rate):g}"
            )

    rife_enabled = requested_rife_fps > 0.0 and output_rate > source_rate
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

    native_scale = base.model_native_scale(args.model)
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
    final_output_shape = (out_h, out_w, 3)

    from .checkpoint_parts import resolve_checkpoint

    checkpoint_path = resolve_checkpoint(
        Path(__file__).resolve().parent / "weights"
    )
    rife_weights = ""
    if rife_enabled:
        from .rife425_api import resolve_rife425_weights

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
            "resample=NPP Lanczos CUDA (CPU Lanczos4 fallback)"
        )

    output_bit_depth = _expected_output_bit_depth(args, info.bit_depth)
    output_pix_fmt = base.output_pixel_format(
        args.video_codec,
        output_bit_depth,
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
        output_bit_depth=output_bit_depth,
        output_pix_fmt=output_pix_fmt,
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
        gpu_timing=bool(args.gpu_timing),
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
    scheduler_wait = 0.0
    state = None
    clean = False

    try:
        writer = writer_type(
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
            sr_output_shape=final_output_shape,
            dtype=dtype,
            input_slots=input_slots,
            frame_output_slots=frame_output_slots,
            enable_gpu_timing=bool(args.gpu_timing),
        )
        worker_model_time = time.monotonic() - model_started

        print(
            "Pipeline: unified GPU workers ready | "
            f"shared_memory={workers.memory_mib:.0f} MiB | "
            f"frame_slots={frame_output_slots}/GPU",
            flush=True,
        )
        print(
            "Pipeline: event-driven control IPC | locality-aware "
            "FrameHandle scheduling | SR micro-batch<=2",
            flush=True,
        )
        print(flush=True)

        progress = create_progress(expected_output)
        pump = OutputPump(
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
        scheduler_wait = run_scheduler(
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
    mux_output_audio(
        temp_video,
        input_path,
        output_path,
        args.ffmpeg_bin,
        start,
        actual_duration,
        info.has_audio,
        args.audio_codec,
        args.audio_bitrate,
        enhance=bool(args.audio_enhance),
    )
    audio_time = time.monotonic() - audio_started

    expected_audio_streams = (
        0
        if not info.has_audio
        else (2 if bool(args.audio_enhance) else 1)
    )
    verification_text = _verify_output_file(
        output_path,
        args.ffprobe_bin,
        video_codec=args.video_codec,
        expected_width=out_w,
        expected_height=out_h,
        expected_fps=output_fps,
        expected_bit_depth=output_bit_depth,
        expected_audio_streams=expected_audio_streams,
        expect_enhanced_audio=bool(args.audio_enhance and info.has_audio),
    )
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
        scheduler_wait=scheduler_wait,
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
        verification_text=verification_text,
        output_path=output_path,
    )
