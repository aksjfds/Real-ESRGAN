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
        default="av1_nvenc",
        help=(
            "Video encoder; Notebook defaults to AV1 NVENC. "
            "HEVC/H.264/software backends remain available through the CLI."
        ),
    )
    parser.add_argument("--crf", type=int, default=18, help="Software encoder quality; lower is higher quality")
    parser.add_argument(
        "--preset",
        choices=("ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower"),
        default="medium",
        help="x264/x265 software encoder preset",
    )
    parser.add_argument("--cq", type=int, default=18, help="NVENC CQ target or CONSTQP QP; lower is higher quality")
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

    # AV1 NVENC high-quality controls.
    parser.add_argument(
        "--av1-profile",
        choices=("main",),
        default="main",
        help="Compatibility-only option; NVENC AV1 output is fixed to Main and final output is verified with ffprobe",
    )
    parser.add_argument(
        "--av1-bit-depth",
        type=int,
        choices=(8, 10),
        default=8,
        help=(
            "Minimum AV1 NVENC output bit depth: 8 keeps 8-bit sources at 8-bit; "
            "10 promotes 8-bit sources to 10-bit. 10-bit sources are never down-converted."
        ),
    )
    parser.add_argument("--av1-tune", choices=("hq",), default="hq")
    parser.add_argument(
        "--av1-rc",
        choices=("vbr", "cbr", "constqp"),
        default="vbr",
        help="AV1 NVENC rate control: VBR uses --cq, CBR requires --av1-bitrate, CONSTQP uses --cq as QP",
    )
    parser.add_argument("--av1-bitrate", default="0")
    parser.add_argument(
        "--av1-multipass",
        choices=("disabled", "qres", "fullres"),
        default="fullres",
    )
    parser.add_argument("--av1-rc-lookahead", type=int, default=28)
    parser.add_argument("--av1-spatial-aq", type=int, choices=(0, 1), default=1)
    parser.add_argument("--av1-temporal-aq", type=int, choices=(0, 1), default=0)
    parser.add_argument("--av1-aq-strength", type=int, default=8)
    parser.add_argument(
        "--av1-b-ref-mode",
        choices=("disabled", "each", "middle"),
        default="middle",
    )
    parser.add_argument("--av1-b-frames", type=int, default=3)
    parser.add_argument("--av1-gop-size", type=int, default=240)
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
        base.set_encoding_backend(av1.require_encoder, av1.RawVideoWriter)
        return

    hevc.validate_args(args)
    hevc.require_encoder(args.ffmpeg_bin, args.video_codec)
    base.set_encoding_backend(hevc.require_encoder, hevc.RawVideoWriter)
