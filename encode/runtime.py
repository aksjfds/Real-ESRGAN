"""Encoding backend selection and CLI integration."""

from __future__ import annotations

import argparse
from types import ModuleType

from . import av1, hevc


ALL_CODECS = tuple(sorted(hevc.CODECS | av1.CODECS))


def extend_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--video-codec",
        choices=ALL_CODECS,
        default="hevc_nvenc",
        help=(
            "Video encoder: CPU HEVC=libx265, GPU HEVC=hevc_nvenc, "
            "CPU AV1=libsvtav1/libaom-av1, GPU AV1=av1_nvenc"
        ),
    )
    parser.add_argument("--crf", type=int, default=18, help="Software encoder quality; lower is higher quality")
    parser.add_argument(
        "--preset",
        choices=("ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower"),
        default="medium",
        help="x264/x265 software encoder preset",
    )
    parser.add_argument("--cq", type=int, default=18, help="NVENC quality target; lower is higher quality")
    parser.add_argument("--nvenc-preset", choices=tuple(f"p{i}" for i in range(1, 8)), default="p7")
    parser.add_argument("--encode-gpu", type=int, default=0)
    parser.add_argument(
        "--svtav1-preset",
        type=int,
        default=6,
        help="SVT-AV1 speed/efficiency preset, 0-13; higher is faster",
    )
    parser.add_argument(
        "--aom-cpu-used",
        type=int,
        default=6,
        help="libaom-av1 quality/speed setting, 0-8; higher is faster",
    )
    return parser


def prepare_runtime(base: ModuleType, args: argparse.Namespace) -> None:
    """Validate and install the selected encoding backend into inference runtime."""
    base.validate_args(args)
    base.require_binary(args.ffmpeg_bin)

    if args.video_codec in av1.CODECS:
        args.video_codec = av1.resolve_requested_encoder(args.ffmpeg_bin, args.video_codec)
        av1.validate_args(args)
        av1.configure(args)
        av1.require_encoder(args.ffmpeg_bin, args.video_codec)
        av1.probe_encoder_runtime(args)
        base.set_encoding_backend(av1.require_encoder, av1.RawVideoWriter)
        return

    hevc.validate_args(args)
    hevc.require_encoder(args.ffmpeg_bin, args.video_codec)
    base.set_encoding_backend(hevc.require_encoder, hevc.RawVideoWriter)
