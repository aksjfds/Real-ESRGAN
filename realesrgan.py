#!/usr/bin/env python3
"""AutoDL v4.3 single-GPU Real-ESRGAN entry point.

This is the only top-level executable.  The v4.2 core and AutoDL runtime are
kept as internal modules under ``enhance`` so notebook users invoke only:

    python realesrgan.py ...
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Optional, Sequence

import torch

from enhance import autodl_runtime as runtime


base = runtime.base
fast = runtime.fast
runtime._REPOSITORY_ROOT = Path(__file__).resolve().parent

_SOURCE_PROFILES = {
    "A": {
        "description": "clean/high-quality source; Real-ESRGAN only",
        "basicvsrpp": False,
        "strength": 0.0,
        "clip_length": 7,
    },
    "B": {
        "description": "nearly clean source with mild compression or temporal instability",
        "basicvsrpp": True,
        "strength": 0.25,
        "clip_length": 7,
    },
    "C": {
        "description": "visibly compressed/noisy source",
        "basicvsrpp": True,
        "strength": 0.75,
        "clip_length": 9,
    },
}

_SVTAV1_PRESET = 6


def _apply_source_profile(args: argparse.Namespace) -> None:
    key = str(getattr(args, "source_profile", "A")).upper()
    if key not in _SOURCE_PROFILES:
        raise ValueError(f"Unknown --source-profile {key!r}; choose A, B or C")
    profile = _SOURCE_PROFILES[key]
    args.source_profile = key
    args.basicvsrpp = bool(profile["basicvsrpp"])
    args.basicvsrpp_track = 1
    args.basicvsrpp_model_path = ""
    args.basicvsrpp_gpu = 0
    args.basicvsrpp_fp16 = True
    args.basicvsrpp_clip_length = int(profile["clip_length"])
    args.basicvsrpp_clip_overlap = 2
    args.basicvsrpp_tile_size = 512
    args.basicvsrpp_tile_pad = 32
    args.basicvsrpp_strength = float(profile["strength"])
    args.basicvsrpp_scene_threshold = 0.30


def _parse_single_gpu(value: str) -> list[Optional[int]]:
    normalized = str(value).strip().lower()
    if normalized == "cpu":
        raise ValueError("AutoDL v4.3 requires one CUDA GPU; CPU inference is disabled")
    if not normalized or "," in normalized:
        raise ValueError("AutoDL v4.3 accepts exactly one --gpu-ids value, for example: 0")
    try:
        gpu_id = int(normalized)
    except ValueError as error:
        raise ValueError("--gpu-ids must be one non-negative CUDA device number") from error
    if gpu_id < 0:
        raise ValueError("--gpu-ids must be non-negative")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; AutoDL v4.3 requires an RTX 4090")
    visible = torch.cuda.device_count()
    if gpu_id >= visible:
        raise ValueError(
            f"Requested CUDA device {gpu_id}, but only {visible} device(s) are visible"
        )
    return [gpu_id]


base.parse_gpu_ids = _parse_single_gpu


_original_require_encoder = base.require_encoder


def _require_encoder(ffmpeg_bin: str, encoder: str) -> None:
    try:
        _original_require_encoder(ffmpeg_bin, encoder)
    except RuntimeError as error:
        if encoder == "libsvtav1":
            raise RuntimeError(
                "FFmpeg does not provide libsvtav1. Install/use an FFmpeg build "
                "configured with --enable-libsvtav1."
            ) from error
        raise


base.require_encoder = _require_encoder

_original_resolve_output_pix_fmt = base.resolve_output_pix_fmt


def _resolve_output_pix_fmt(ffmpeg_bin: str, codec: str, requested: str) -> str:
    if codec != "libsvtav1":
        return _original_resolve_output_pix_fmt(ffmpeg_bin, codec, requested)

    selected = "yuv420p10le" if requested == "auto" else requested
    supported = base.encoder_pixel_formats(ffmpeg_bin, codec)
    if selected in supported:
        return selected
    if requested != "auto":
        raise RuntimeError(
            f"Encoder {codec} does not advertise explicitly requested pixel "
            f"format {selected}; supported={sorted(supported)}"
        )
    if "yuv420p" in supported:
        print(
            "[warning] libsvtav1 does not advertise yuv420p10le; "
            "auto falling back to yuv420p",
            flush=True,
        )
        return "yuv420p"
    raise RuntimeError(
        f"Encoder {codec} has no supported 4:2:0 output format; "
        f"supported={sorted(supported)}"
    )


base.resolve_output_pix_fmt = _resolve_output_pix_fmt

_OriginalRawVideoWriter = base.RawVideoWriter


class RawVideoWriter(_OriginalRawVideoWriter):
    """Use the v4.2 writer unchanged except for the libsvtav1 command."""

    def __init__(
        self,
        path: Path,
        ffmpeg_bin: str,
        width: int,
        height: int,
        input_fps_rate: str,
        output_fps_rate: str,
        codec: str,
        crf: int,
        preset: str,
        cq: int,
        nvenc_preset: str,
        encode_gpu: int,
        output_pix_fmt: str,
        video_filters: Sequence[str],
        color_spec: object,
    ) -> None:
        if codec != "libsvtav1":
            super().__init__(
                path,
                ffmpeg_bin,
                width,
                height,
                input_fps_rate,
                output_fps_rate,
                codec,
                crf,
                preset,
                cq,
                nvenc_preset,
                encode_gpu,
                output_pix_fmt,
                video_filters,
                color_spec,
            )
            return

        command = [
            ffmpeg_bin,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb48le",
            "-s:v",
            f"{width}x{height}",
            "-r",
            input_fps_rate,
            "-i",
            "pipe:0",
            "-an",
        ]
        filters = list(video_filters)
        if output_fps_rate != input_fps_rate:
            filters.append(f"fps={output_fps_rate}")
        if filters:
            command += ["-vf", ",".join(filters)]
        command += [
            "-c:v",
            "libsvtav1",
            "-preset",
            str(_SVTAV1_PRESET),
            "-crf",
            str(crf),
            "-pix_fmt",
            output_pix_fmt,
            *base.color_output_args(color_spec),
            "-tag:v",
            "av01",
            str(path),
        ]
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )


base.RawVideoWriter = RawVideoWriter

_original_validate_args = base.validate_args


def _validate_args(args: argparse.Namespace) -> None:
    _apply_source_profile(args)
    original_crf = int(args.crf)
    if args.video_codec == "libsvtav1" and original_crf > 51:
        # Run all common v4.2 checks without letting its x26x-specific CRF cap
        # reject the wider SVT-AV1 range.
        args.crf = 51
    try:
        _original_validate_args(args)
    finally:
        args.crf = original_crf

    if args.video_codec == "libsvtav1":
        if not 0 <= original_crf <= 63:
            raise ValueError("--crf must be between 0 and 63 for libsvtav1")
        if not 0 <= args.svtav1_preset <= 13:
            raise ValueError("--svtav1-preset must be between 0 and 13")


base.validate_args = _validate_args

_original_process_video = base.process_video


def _process_video(args: argparse.Namespace) -> None:
    global _SVTAV1_PRESET
    _apply_source_profile(args)
    _SVTAV1_PRESET = int(args.svtav1_preset)
    profile = _SOURCE_PROFILES[args.source_profile]
    print(
        f"[source-profile] {args.source_profile}: {profile['description']}; "
        f"basicvsrpp={args.basicvsrpp}, strength={args.basicvsrpp_strength:g}, "
        f"clip={args.basicvsrpp_clip_length}",
        flush=True,
    )
    _original_process_video(args)


base.process_video = _process_video

_original_build_parser = fast.build_parser


def build_parser() -> argparse.ArgumentParser:
    parser = _original_build_parser()
    for action in parser._actions:
        if action.dest == "video_codec":
            choices = tuple(action.choices or ())
            if "libsvtav1" not in choices:
                action.choices = choices + ("libsvtav1",)
    parser.add_argument(
        "--source-profile",
        choices=("A", "B", "C"),
        default="A",
        help=(
            "A=clean source/no BasicVSR++; B=mild compression/BasicVSR++ 0.25; "
            "C=visible compression/BasicVSR++ 0.75"
        ),
    )
    parser.add_argument(
        "--svtav1-preset",
        type=int,
        default=6,
        help="SVT-AV1 speed/efficiency preset, 0-13; higher is faster",
    )
    return parser


fast.build_parser = build_parser


def main() -> None:
    fast.main()


if __name__ == "__main__":
    fast.mp.freeze_support()
    main()
