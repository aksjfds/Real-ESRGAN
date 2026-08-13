#!/usr/bin/env python3
"""Unified APISR video inference entry point."""

from __future__ import annotations

import argparse
import multiprocessing as mp

from inference.cuda_memory_policy import configure_cuda_allocator_env

# Configure before importing torch-bearing runtime modules so spawned CUDA workers
# inherit the allocator policy before their first CUDA allocation. Explicit user
# PYTORCH_ALLOC_CONF/PYTORCH_CUDA_ALLOC_CONF values always win.
configure_cuda_allocator_env()

from encode import runtime as encode_runtime
from inference import runtime as inference_runtime
from inference import runtime_api as pipeline_runtime
from inference import scheduler as inference_pipeline
from inference.run_lock import exclusive_output_run


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="BVS + RIFE + APISR GRL GPU video inference.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", required=True, help="Input video path")
    parser.add_argument("--output", required=True, help="Output MP4 path")
    parser.add_argument(
        "--model",
        choices=tuple(pipeline_runtime.MODEL_URLS),
        default=pipeline_runtime.DEFAULT_MODEL_NAME,
    )
    parser.add_argument(
        "--model-path",
        default="",
        help="Optional local .pth override",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=2.0,
        help="Final output scale",
    )
    parser.add_argument(
        "--deband-strength",
        type=float,
        default=0.0,
        help=(
            "FFmpeg pre-model deband threshold; 0 disables, otherwise "
            "must be within 0.00003..0.5"
        ),
    )
    parser.add_argument(
        "--gpu-ids",
        default="0",
        help="Comma-separated CUDA GPU IDs; defaults to one GPU",
    )
    parser.add_argument(
        "--audio-codec",
        choices=("aac", "copy"),
        default="aac",
    )
    parser.add_argument("--audio-bitrate", default="192k")
    parser.add_argument(
        "--audio-enhance",
        action="store_true",
        help="Enable FFmpeg dialogue-focused audio enhancement",
    )
    parser.add_argument(
        "--start-time",
        type=float,
        default=0.0,
        help="Source start in seconds",
    )
    parser.add_argument(
        "--test-seconds",
        type=float,
        default=0.0,
        help="0 processes to end; use 10 for a test",
    )
    parser.add_argument("--ffmpeg-bin", default="ffmpeg")
    parser.add_argument("--ffprobe-bin", default="ffprobe")

    parser.add_argument(
        "--bvs-tile-size",
        type=int,
        default=640,
        help="BasicVSR++ spatial tile size",
    )
    parser.add_argument(
        "--bvs-clip-length",
        type=int,
        default=13,
        help="BasicVSR++ temporal clip length",
    )
    parser.add_argument(
        "--bvs-batch-size",
        type=int,
        default=1,
        help="Independent BasicVSR++ clips per GPU task",
    )
    parser.add_argument(
        "--bvs-strength",
        type=float,
        default=1.0,
        help="BasicVSR++ residual blend strength in (0,1]",
    )
    parser.add_argument(
        "--rife-fps",
        type=float,
        default=60.0,
        help=(
            "Practical-RIFE 4.25 target/output FPS; 0 disables RIFE and "
            "keeps source FPS, otherwise must be >= source FPS"
        ),
    )
    parser.add_argument(
        "--gpu-timing",
        action="store_true",
        help=(
            "Enable native per-task CUDA Event timing. "
            "Adds synchronization overhead only when enabled."
        ),
    )
    encode_runtime.extend_parser(parser)
    return parser


def _validate_args(args) -> None:
    if not 0.0 < float(args.bvs_strength) <= 1.0:
        raise ValueError("--bvs-strength must be in (0, 1]")
    if int(args.bvs_clip_length) < 2:
        raise ValueError("--bvs-clip-length must be at least 2")
    if int(args.bvs_tile_size) < 256 or int(args.bvs_tile_size) % 4:
        raise ValueError(
            "--bvs-tile-size must be >=256 and divisible by 4"
        )
    if int(args.bvs_batch_size) < 1:
        raise ValueError("--bvs-batch-size must be at least 1")
    if float(args.rife_fps) < 0:
        raise ValueError("--rife-fps must be 0 or a positive FPS")
    deband_strength = float(args.deband_strength)
    if deband_strength < 0.0 or deband_strength > 0.5:
        raise ValueError("--deband-strength must be 0 or within 0.00003..0.5")
    if 0.0 < deband_strength < 0.00003:
        raise ValueError("--deband-strength must be 0 or within 0.00003..0.5")
    if bool(args.audio_enhance) and args.audio_codec != "aac":
        raise ValueError("--audio-enhance requires --audio-codec aac")


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    _validate_args(args)
    if args.audio_enhance:
        from audio.runtime import validate_runtime

        validate_runtime(args.ffmpeg_bin)
    pipeline_runtime.configure_deband(args.deband_strength)
    with exclusive_output_run(args.output):
        encode_runtime.prepare_runtime(inference_runtime, args)
        inference_pipeline.process_video(args)


if __name__ == "__main__":
    mp.freeze_support()
    main()
