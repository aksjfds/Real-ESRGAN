#!/usr/bin/env python3
"""Unified Real-ESRGAN video inference entry point."""

from __future__ import annotations

import multiprocessing as mp
from pathlib import Path

from encode import runtime as encode_runtime
from inference import runtime as inference_runtime
from inference import v52_scheduler as inference_pipeline
from inference.gpu_timing import install_gpu_timing
from inference.source_profiles import PROFILE_CHOICES, SOURCE_PROFILES
from inference.v51_runtime import install_basicvsrpp_optimizations, install_pipeline_optimizations


def main() -> None:
    parser = inference_runtime.build_parser()
    parser.add_argument(
        "--source-profile",
        choices=PROFILE_CHOICES,
        default="A",
        help=(
            "A=BasicVSR++ off; B=25%%; C=50%%; D=75%%; "
            "E=100%% full-strength NTIRE compressed-video restoration"
        ),
    )
    encode_runtime.extend_parser(parser)
    args = parser.parse_args()
    encode_runtime.prepare_runtime(inference_runtime, args)
    install_pipeline_optimizations()

    # Keep the CLI as the single public entry while allowing the pipeline to
    # share the extended A-E profile table without duplicating configuration.
    inference_pipeline.base_pipeline.SOURCE_PROFILES = SOURCE_PROFILES

    source_bit_depth = None
    if args.source_profile != "A":
        # B-E always use the checkpoint parts bundled in inference/weights.
        from inference import basicvsrpp
        from inference.basicvsrpp_autotune import install_autotune
        from inference.checkpoint_parts import resolve_checkpoint

        basicvsrpp.SOURCE_PROFILES = SOURCE_PROFILES
        basicvsrpp.download_checkpoint = resolve_checkpoint
        try:
            source = inference_runtime.probe_video(
                Path(args.input).expanduser().resolve(),
                args.ffprobe_bin,
            )
            install_autotune(source.width, source.height)
            source_bit_depth = source.bit_depth
        except Exception as error:
            print(
                f"[autotuner] source probe unavailable ({error}); using hardware-only search",
                flush=True,
            )
            install_autotune()
        install_basicvsrpp_optimizations(source_bit_depth)

    # CUDA Events replace host launch timing for SR/BVS and report per-GPU
    # timed-device busy/wait without changing the scheduler or model math.
    install_gpu_timing(enable_bvs=args.source_profile != "A")
    inference_pipeline.process_video(args)


if __name__ == "__main__":
    mp.freeze_support()
    main()
