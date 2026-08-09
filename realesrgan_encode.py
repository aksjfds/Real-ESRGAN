#!/usr/bin/env python3
"""Encoding-capability wrapper for the master/2.7 Real-ESRGAN runtime.

The underlying inference implementation remains ``realesrgan.py``.  This entry
only extends FFmpeg output support so the same inference path can encode:

* CPU HEVC: libx265
* GPU HEVC: hevc_nvenc
* CPU AV1:  libsvtav1
* GPU AV1:  av1_nvenc

H.264 options from the 2.7 runtime remain available for compatibility.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Sequence

import realesrgan as base


_SVTAV1_PRESET = 6


def _require_encoder(ffmpeg_bin: str, encoder: str) -> None:
    """Keep the base encoder check but provide codec-specific diagnostics."""
    try:
        _ORIGINAL_REQUIRE_ENCODER(ffmpeg_bin, encoder)
    except RuntimeError as error:
        if encoder == "libsvtav1":
            raise RuntimeError(
                "FFmpeg does not provide libsvtav1. Use an FFmpeg build configured "
                "with --enable-libsvtav1, or select hevc_nvenc/libx265."
            ) from error
        if encoder == "av1_nvenc":
            raise RuntimeError(
                "FFmpeg does not provide av1_nvenc. Use an FFmpeg build with NVENC AV1 "
                "support and an AV1-capable NVIDIA GPU, or select libsvtav1/libx265."
            ) from error
        raise


def _codec_args(
    codec: str,
    crf: int,
    preset: str,
    cq: int,
    nvenc_preset: str,
    encode_gpu: int,
    svtav1_preset: int,
) -> list[str]:
    if codec in {"libx264", "libx265"}:
        return ["-preset", preset, "-crf", str(crf)]

    if codec == "libsvtav1":
        return ["-preset", str(svtav1_preset), "-crf", str(crf)]

    if codec not in {"h264_nvenc", "hevc_nvenc", "av1_nvenc"}:
        raise ValueError(f"Unsupported video encoder: {codec}")

    # Use the same quality-first NVENC policy as master/2.7 HEVC.  FFmpeg's
    # av1_nvenc exposes the same VBR CQ, multipass, lookahead and AQ controls.
    args = [
        "-gpu", str(encode_gpu),
        "-preset", nvenc_preset,
        "-tune", "hq",
        "-rc", "vbr",
        "-cq", str(cq),
        "-b:v", "0",
        "-multipass", "fullres",
        "-spatial-aq", "1",
        "-temporal-aq", "1",
        "-rc-lookahead", "32",
    ]
    if codec in {"h264_nvenc", "hevc_nvenc"}:
        args += ["-bf", "3"]
    return args


def _codec_tag(codec: str) -> list[str]:
    if codec in {"libx265", "hevc_nvenc"}:
        return ["-tag:v", "hvc1"]
    if codec in {"libsvtav1", "av1_nvenc"}:
        return ["-tag:v", "av01"]
    return []


class RawVideoWriter:
    """2.7 RGB24 writer extended with separate software/NVENC AV1 paths."""

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
    ) -> None:
        command = [
            ffmpeg_bin,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s:v",
            f"{width}x{height}",
            "-r",
            input_fps_rate,
            "-i",
            "pipe:0",
            "-an",
        ]
        if output_fps_rate != input_fps_rate:
            command += ["-vf", f"fps={output_fps_rate}"]

        command += ["-c:v", codec]
        command += _codec_args(
            codec,
            crf,
            preset,
            cq,
            nvenc_preset,
            encode_gpu,
            _SVTAV1_PRESET,
        )
        # Keep master/2.7's 8-bit 4:2:0 output semantics.  The encoding feature
        # extension must not silently change the existing inference pixel path.
        command += ["-pix_fmt", "yuv420p"]
        command += _codec_tag(codec)
        command.append(str(path))
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def write(self, frame) -> None:
        assert self.process.stdin is not None
        try:
            self.process.stdin.write(memoryview(base.np.ascontiguousarray(frame)).cast("B"))
        except BrokenPipeError as error:
            detail = (
                self.process.stderr.read().decode(errors="replace")
                if self.process.stderr
                else ""
            )
            raise RuntimeError(f"ffmpeg encoder closed its input early:\n{detail}") from error

    def close(self) -> None:
        if self.process.stdin is not None:
            self.process.stdin.close()
        stderr = self.process.stderr.read() if self.process.stderr is not None else b""
        if self.process.stderr is not None:
            self.process.stderr.close()
        return_code = self.process.wait()
        if return_code != 0:
            raise RuntimeError(
                f"ffmpeg encode failed (exit {return_code}):\n"
                f"{stderr.decode(errors='replace')}"
            )


def _probe_encoder_runtime(args: argparse.Namespace) -> None:
    """Verify the selected encoder can actually initialize on this machine/GPU."""
    command = [
        args.ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        "color=black:size=128x128:rate=1,format=rgb24",
        "-frames:v",
        "1",
        "-c:v",
        args.video_codec,
    ]
    command += _codec_args(
        args.video_codec,
        args.crf,
        args.preset,
        args.cq,
        args.nvenc_preset,
        args.encode_gpu,
        args.svtav1_preset,
    )
    command += ["-pix_fmt", "yuv420p", "-f", "null", "-"]

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if result.returncode == 0:
        print(f"[encoder] runtime OK: {args.video_codec}", flush=True)
        return

    detail = result.stdout.strip()
    if args.video_codec == "av1_nvenc":
        raise RuntimeError(
            "av1_nvenc is present in FFmpeg but could not initialize. NVIDIA AV1 "
            "hardware encoding requires an AV1-capable GPU (Ada generation or newer "
            "in NVIDIA's current Video Codec SDK documentation).\n"
            f"FFmpeg output:\n{detail}"
        )
    raise RuntimeError(
        f"{args.video_codec} runtime probe failed (exit {result.returncode}):\n{detail}"
    )


_ORIGINAL_REQUIRE_ENCODER = base.require_encoder
_ORIGINAL_VALIDATE_ARGS = base.validate_args


def validate_args(args: argparse.Namespace) -> None:
    """Reuse 2.7 validation while widening AV1 quality ranges where appropriate."""
    original_crf = int(args.crf)
    original_cq = int(args.cq)

    if args.video_codec == "libsvtav1" and original_crf > 51:
        args.crf = 51
    if args.video_codec == "av1_nvenc" and original_cq > 51:
        args.cq = 51
    try:
        _ORIGINAL_VALIDATE_ARGS(args)
    finally:
        args.crf = original_crf
        args.cq = original_cq

    if args.video_codec == "libsvtav1":
        if not 0 <= original_crf <= 63:
            raise ValueError("--crf must be between 0 and 63 for libsvtav1")
        if not 0 <= args.svtav1_preset <= 13:
            raise ValueError("--svtav1-preset must be between 0 and 13")

    if args.video_codec == "av1_nvenc" and not 0 <= original_cq <= 63:
        raise ValueError("--cq must be between 0 and 63 for av1_nvenc")


def build_parser() -> argparse.ArgumentParser:
    parser = base.build_parser()
    for action in parser._actions:
        if action.dest == "video_codec":
            action.choices = (
                "libx264",
                "libx265",
                "h264_nvenc",
                "hevc_nvenc",
                "libsvtav1",
                "av1_nvenc",
            )
            action.help = (
                "Video encoder: CPU HEVC=libx265, GPU HEVC=hevc_nvenc, "
                "CPU AV1=libsvtav1, GPU AV1=av1_nvenc"
            )
            break
    parser.add_argument(
        "--svtav1-preset",
        type=int,
        default=6,
        help="SVT-AV1 speed/efficiency preset, 0-13; higher is faster",
    )
    return parser


# Patch only the encoding-facing globals used by base.process_video().
base.require_encoder = _require_encoder
base.RawVideoWriter = RawVideoWriter
base.validate_args = validate_args


def main() -> None:
    global _SVTAV1_PRESET
    args = build_parser().parse_args()
    validate_args(args)
    _SVTAV1_PRESET = int(args.svtav1_preset)

    base.require_binary(args.ffmpeg_bin)
    _require_encoder(args.ffmpeg_bin, args.video_codec)
    _probe_encoder_runtime(args)
    base.process_video(args)


if __name__ == "__main__":
    base.mp.freeze_support()
    main()
