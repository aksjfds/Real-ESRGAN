#!/usr/bin/env python3
"""Unified Real-ESRGAN video inference entry point."""

from __future__ import annotations

import argparse
import multiprocessing as mp

from encode import runtime as encode_runtime
from inference import runtime as inference_runtime
from inference import v52_scheduler as inference_pipeline
from inference.run_lock import exclusive_output_run


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="BVS + RIFE + Real-ESRGAN multi-GPU video inference.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", required=True, help="Input video path")
    parser.add_argument("--output", required=True, help="Output MP4 path")
    parser.add_argument(
        "--model",
        choices=tuple(inference_runtime.MODEL_URLS),
        default="realesr-animevideov3",
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
        "--gpu-ids",
        default="0,1",
        help="Comma-separated CUDA GPU IDs",
    )
    parser.add_argument(
        "--audio-codec",
        choices=("aac", "copy"),
        default="aac",
    )
    parser.add_argument("--audio-bitrate", default="192k")
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


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    _validate_args(args)
    with exclusive_output_run(args.output):
        encode_runtime.prepare_runtime(inference_runtime, args)
        inference_pipeline.process_video(args)


if __name__ == "__main__":
    mp.freeze_support()
    main()
