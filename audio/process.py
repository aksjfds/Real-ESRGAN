#!/usr/bin/env python3
"""Standalone media bypass/audio enhancement entry point."""

from __future__ import annotations

import argparse
from pathlib import Path

from .runtime import process_media


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Copy video unchanged and optionally enhance its audio.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--audio-enhance", action="store_true")
    parser.add_argument("--audio-codec", choices=("aac", "copy"), default="aac")
    parser.add_argument("--audio-bitrate", default="256k")
    parser.add_argument("--start-time", type=float, default=0.0)
    parser.add_argument("--test-seconds", type=float, default=0.0)
    parser.add_argument("--ffmpeg-bin", default="ffmpeg")
    parser.add_argument("--ffprobe-bin", default="ffprobe")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    process_media(
        Path(args.input),
        Path(args.output),
        args.ffmpeg_bin,
        args.ffprobe_bin,
        float(args.start_time),
        float(args.test_seconds),
        args.audio_codec,
        args.audio_bitrate,
        enhance=bool(args.audio_enhance),
    )


if __name__ == "__main__":
    main()
