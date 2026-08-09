#!/usr/bin/env python3
"""Encoding-capability wrapper for the master/2.7 Real-ESRGAN runtime.

The underlying inference implementation remains ``realesrgan.py``. This entry
only extends FFmpeg output support so the same inference path can encode:

* CPU HEVC: libx265
* GPU HEVC: hevc_nvenc
* CPU AV1:  libsvtav1, with libaom-av1 fallback when SVT-AV1 is unavailable
* GPU AV1:  av1_nvenc

H.264 options from the 2.7 runtime remain available for compatibility.
"""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType


def _load_base_runtime() -> ModuleType:
    """Load the root ``realesrgan.py`` explicitly, not the ``realesrgan`` package.

    This repository contains both ``realesrgan.py`` and a ``realesrgan/`` package.
    A normal ``import realesrgan`` can therefore resolve to the package, which does
    not expose the video-runtime functions used by this wrapper.
    """
    module_name = "_master_realesrgan_runtime"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing

    runtime_path = Path(__file__).resolve().with_name("realesrgan.py")
    spec = importlib.util.spec_from_file_location(module_name, runtime_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load Real-ESRGAN runtime from {runtime_path}")
    module = importlib.util.module_from_spec(spec)
    # Register before execution so multiprocessing/dataclasses can resolve the
    # module by name when spawned workers import/pickle runtime symbols.
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


base = _load_base_runtime()

_SVTAV1_PRESET = 6
_AOM_CPU_USED = 6


_ORIGINAL_REQUIRE_ENCODER = base.require_encoder
_ORIGINAL_VALIDATE_ARGS = base.validate_args


def _encoder_available(ffmpeg_bin: str, encoder: str) -> bool:
    try:
        _ORIGINAL_REQUIRE_ENCODER(ffmpeg_bin, encoder)
        return True
    except RuntimeError:
        return False


def _resolve_requested_encoder(ffmpeg_bin: str, requested: str) -> str:
    """Resolve environment-sensitive encoder aliases before video inference."""
    if requested == "libsvtav1" and not _encoder_available(ffmpeg_bin, requested):
        if _encoder_available(ffmpeg_bin, "libaom-av1"):
            print(
                "[encoder-warning] FFmpeg has no libsvtav1; falling back to "
                "CPU AV1 encoder libaom-av1.",
                flush=True,
            )
            return "libaom-av1"
        raise RuntimeError(
            "FFmpeg provides neither libsvtav1 nor libaom-av1, so CPU AV1 encoding "
            "is unavailable in this environment. libsvtav1 requires an FFmpeg build "
            "configured with --enable-libsvtav1."
        )
    return requested


def _require_encoder(ffmpeg_bin: str, encoder: str) -> None:
    """Keep the base encoder check but provide codec-specific diagnostics."""
    try:
        _ORIGINAL_REQUIRE_ENCODER(ffmpeg_bin, encoder)
    except RuntimeError as error:
        if encoder == "libsvtav1":
            raise RuntimeError(
                "FFmpeg does not provide libsvtav1. Use an FFmpeg build configured "
                "with --enable-libsvtav1, or use libaom-av1/libx265."
            ) from error
        if encoder == "libaom-av1":
            raise RuntimeError(
                "FFmpeg does not provide libaom-av1. CPU AV1 encoding is unavailable "
                "unless this FFmpeg build contains libsvtav1 or libaom-av1."
            ) from error
        if encoder == "av1_nvenc":
            raise RuntimeError(
                "FFmpeg does not provide av1_nvenc. Use an FFmpeg build with NVENC AV1 "
                "support and an AV1-capable NVIDIA GPU, or select CPU AV1/HEVC."
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
    aom_cpu_used: int,
) -> list[str]:
    if codec in {"libx264", "libx265"}:
        return ["-preset", preset, "-crf", str(crf)]

    if codec == "libsvtav1":
        return ["-preset", str(svtav1_preset), "-crf", str(crf)]

    if codec == "libaom-av1":
        # Constant-quality AV1. -b:v 0 selects unconstrained constant quality;
        # cpu-used controls the documented quality/speed tradeoff (0..8).
        return [
            "-crf", str(crf),
            "-b:v", "0",
            "-cpu-used", str(aom_cpu_used),
            "-row-mt", "1",
        ]

    if codec not in {"h264_nvenc", "hevc_nvenc", "av1_nvenc"}:
        raise ValueError(f"Unsupported video encoder: {codec}")

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
    if codec in {"libsvtav1", "libaom-av1", "av1_nvenc"}:
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
            _AOM_CPU_USED,
        )
        # Preserve master/2.7 output semantics: 8-bit YUV 4:2:0.
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
    """Verify the selected encoder can actually initialize before model inference."""
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
        args.aom_cpu_used,
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
            "av1_nvenc is present in FFmpeg but could not initialize. The selected "
            "NVIDIA GPU/driver may not support AV1 NVENC.\n"
            f"FFmpeg output:\n{detail}"
        )
    raise RuntimeError(
        f"{args.video_codec} runtime probe failed (exit {result.returncode}):\n{detail}"
    )


def validate_args(args: argparse.Namespace) -> None:
    """Reuse 2.7 validation while widening AV1 quality ranges."""
    original_crf = int(args.crf)
    original_cq = int(args.cq)

    if args.video_codec in {"libsvtav1", "libaom-av1"} and original_crf > 51:
        args.crf = 51
    if args.video_codec == "av1_nvenc" and original_cq > 51:
        args.cq = 51
    try:
        _ORIGINAL_VALIDATE_ARGS(args)
    finally:
        args.crf = original_crf
        args.cq = original_cq

    if args.video_codec in {"libsvtav1", "libaom-av1"}:
        if not 0 <= original_crf <= 63:
            raise ValueError("--crf must be between 0 and 63 for CPU AV1")
    if not 0 <= args.svtav1_preset <= 13:
        raise ValueError("--svtav1-preset must be between 0 and 13")
    if not 0 <= args.aom_cpu_used <= 8:
        raise ValueError("--aom-cpu-used must be between 0 and 8")
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
                "libaom-av1",
                "av1_nvenc",
            )
            action.help = (
                "Video encoder: CPU HEVC=libx265, GPU HEVC=hevc_nvenc, "
                "CPU AV1=libsvtav1/libaom-av1, GPU AV1=av1_nvenc. "
                "Requesting libsvtav1 automatically falls back to libaom-av1 if needed."
            )
            break
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


# Patch only encoding-facing globals used by base.process_video().
base.require_encoder = _require_encoder
base.RawVideoWriter = RawVideoWriter
base.validate_args = validate_args


def main() -> None:
    global _SVTAV1_PRESET, _AOM_CPU_USED
    args = build_parser().parse_args()
    validate_args(args)

    base.require_binary(args.ffmpeg_bin)
    args.video_codec = _resolve_requested_encoder(args.ffmpeg_bin, args.video_codec)
    # Validate again after environment-driven fallback so codec-specific limits
    # are applied to the actual encoder that will run.
    validate_args(args)

    _SVTAV1_PRESET = int(args.svtav1_preset)
    _AOM_CPU_USED = int(args.aom_cpu_used)

    _require_encoder(args.ffmpeg_bin, args.video_codec)
    _probe_encoder_runtime(args)
    base.process_video(args)


if __name__ == "__main__":
    base.mp.freeze_support()
    main()
