#!/usr/bin/env python3
"""Unified Real-ESRGAN video inference entry point."""

from __future__ import annotations

import multiprocessing as mp
from pathlib import Path

from encode import runtime as encode_runtime
from inference import balanced_pipeline as inference_pipeline
from inference import runtime as inference_runtime


def main() -> None:
    parser = inference_runtime.build_parser()
    parser.add_argument(
        "--source-profile",
        choices=("A", "B", "C"),
        default="A",
        help=(
            "A=BasicVSR++ off; B=light NTIRE compressed-video restoration; "
            "C=stronger restoration for visibly compressed/noisy sources"
        ),
    )
    encode_runtime.extend_parser(parser)
    args = parser.parse_args()
    encode_runtime.prepare_runtime(inference_runtime, args)

    if args.source_profile != "A":
        # B/C always use the checkpoint parts bundled in inference/weights.
        from inference import basicvsrpp
        from inference.basicvsrpp_autotune import install_autotune
        from inference.checkpoint_parts import resolve_checkpoint

        basicvsrpp.download_checkpoint = resolve_checkpoint
        try:
            source = inference_runtime.probe_video(
                Path(args.input).expanduser().resolve(),
                args.ffprobe_bin,
            )
            install_autotune(source.width, source.height)
        except Exception as error:
            print(
                f"[basicvsrpp-autotune] source probe unavailable ({error}); using hardware-only search",
                flush=True,
            )
            install_autotune()

    inference_pipeline.process_video(args)


if __name__ == "__main__":
    mp.freeze_support()
    main()
