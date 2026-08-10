#!/usr/bin/env python3
"""Unified Real-ESRGAN video inference entry point."""
from __future__ import annotations

import multiprocessing as mp

from encode import runtime as encode_runtime
from inference import runtime as inference_runtime
from inference import v52_scheduler as inference_pipeline
from inference.progress_log import install_persistent_progress
from inference.v51_runtime import install_pipeline_optimizations


def _validate_basicvsrpp_args(args) -> None:
    if not 0.0 < float(args.bvs_strength) <= 1.0:
        raise ValueError("--bvs-strength must be in (0, 1]")
    if int(args.bvs_clip_length) < 2:
        raise ValueError("--bvs-clip-length must be at least 2")
    if int(args.bvs_tile_size) < 256 or int(args.bvs_tile_size) % 4:
        raise ValueError("--bvs-tile-size must be >=256 and divisible by 4")
    if int(args.bvs_batch_size) < 1:
        raise ValueError("--bvs-batch-size must be at least 1")


def main() -> None:
    parser = inference_runtime.build_parser()
    parser.add_argument(
        "--bvs-tile-size",
        type=int,
        default=640,
        help="BasicVSR++ spatial tile size. Default: 640.",
    )
    parser.add_argument(
        "--bvs-clip-length",
        type=int,
        default=13,
        help="BasicVSR++ temporal clip length. Default: 13.",
    )
    parser.add_argument(
        "--bvs-batch-size",
        type=int,
        default=1,
        help="Independent BasicVSR++ clips per GPU task. Default: 1.",
    )
    parser.add_argument(
        "--bvs-strength",
        type=float,
        default=1.0,
        help="BasicVSR++ residual blend strength in (0,1]. Default: 1.0.",
    )
    parser.add_argument(
        "--gpu-timing",
        action="store_true",
        help="Enable CUDA-event GPU busy/wait diagnostics (adds profiling overhead).",
    )
    encode_runtime.extend_parser(parser)
    args = parser.parse_args()
    _validate_basicvsrpp_args(args)
    encode_runtime.prepare_runtime(inference_runtime, args)

    install_pipeline_optimizations()
    install_persistent_progress()

    # BasicVSR++ is always enabled with explicit fixed parameters. The checkpoint
    # is assembled from bundled repository parts; parameters are used directly and
    # no runtime parameter search is installed.
    from inference import basicvsrpp
    from inference.checkpoint_parts import resolve_checkpoint
    from inference.v54_runtime import install_basicvsrpp_execution_optimizations

    basicvsrpp.download_checkpoint = resolve_checkpoint
    install_basicvsrpp_execution_optimizations()

    if args.gpu_timing:
        from inference.gpu_timing import install_gpu_timing

        install_gpu_timing(enable_bvs=True)

    inference_pipeline.process_video(args)


if __name__ == "__main__":
    mp.freeze_support()
    main()
