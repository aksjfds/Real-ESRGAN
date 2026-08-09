"""AV1 FFmpeg encoders."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np


CODECS = {"libsvtav1", "libaom-av1", "av1_nvenc"}
_SVTAV1_PRESET = 6
_AOM_CPU_USED = 6


def require_encoder(ffmpeg_bin: str, encoder: str) -> None:
    result = subprocess.run(
        [ffmpeg_bin, "-hide_banner", "-encoders"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"ffmpeg encoder probe failed (exit {result.returncode}):\n{detail}")
    if encoder not in (result.stdout + result.stderr):
        raise RuntimeError(f"ffmpeg does not provide the requested video encoder: {encoder}")


def encoder_available(ffmpeg_bin: str, encoder: str) -> bool:
    try:
        require_encoder(ffmpeg_bin, encoder)
        return True
    except RuntimeError:
        return False


def resolve_requested_encoder(ffmpeg_bin: str, requested: str) -> str:
    if requested == "libsvtav1" and not encoder_available(ffmpeg_bin, requested):
        if encoder_available(ffmpeg_bin, "libaom-av1"):
            print(
                "[encoder-warning] FFmpeg has no libsvtav1; falling back to "
                "CPU AV1 encoder libaom-av1.",
                flush=True,
            )
            return "libaom-av1"
        raise RuntimeError(
            "FFmpeg provides neither libsvtav1 nor libaom-av1, so CPU AV1 encoding "
            "is unavailable in this environment."
        )
    return requested


def validate_args(args) -> None:
    if args.video_codec not in CODECS:
        raise ValueError(f"Unsupported AV1 encoder: {args.video_codec}")
    if args.video_codec in {"libsvtav1", "libaom-av1"} and not 0 <= args.crf <= 63:
        raise ValueError("--crf must be between 0 and 63 for CPU AV1")
    if args.video_codec == "av1_nvenc" and not 0 <= args.cq <= 63:
        raise ValueError("--cq must be between 0 and 63 for av1_nvenc")
    if not 0 <= args.svtav1_preset <= 13:
        raise ValueError("--svtav1-preset must be between 0 and 13")
    if not 0 <= args.aom_cpu_used <= 8:
        raise ValueError("--aom-cpu-used must be between 0 and 8")
    if args.encode_gpu < 0:
        raise ValueError("--encode-gpu must be non-negative.")


def _codec_args(
    codec: str,
    crf: int,
    cq: int,
    nvenc_preset: str,
    encode_gpu: int,
    svtav1_preset: int,
    aom_cpu_used: int,
) -> list[str]:
    if codec == "libsvtav1":
        return ["-preset", str(svtav1_preset), "-crf", str(crf)]

    if codec == "libaom-av1":
        return [
            "-crf", str(crf),
            "-b:v", "0",
            "-cpu-used", str(aom_cpu_used),
            "-row-mt", "1",
        ]

    if codec == "av1_nvenc":
        return [
            "-gpu", str(encode_gpu),
            "-preset", nvenc_preset,
            "-tune", "hq",
            "-rc", "vbr",
            "-cq", str(cq),
            "-b:v", "0",
            "-multipass", "fullres",
            "-spatial_aq", "1",
            "-temporal_aq", "1",
            "-rc-lookahead", "32",
        ]

    raise ValueError(f"Unsupported AV1 encoder: {codec}")


class RawVideoWriter:
    """RGB24 FFmpeg writer for CPU/NVENC AV1."""

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
        del preset
        if codec not in CODECS:
            raise RuntimeError(f"Non-AV1 codec reached AV1 writer: {codec}")

        command = [
            ffmpeg_bin,
            "-y",
            "-hide_banner",
            "-loglevel", "error",
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "-s:v", f"{width}x{height}",
            "-r", input_fps_rate,
            "-i", "pipe:0",
            "-an",
        ]
        if output_fps_rate != input_fps_rate:
            command += ["-vf", f"fps={output_fps_rate}"]

        command += ["-c:v", codec]
        command += _codec_args(
            codec,
            crf,
            cq,
            nvenc_preset,
            encode_gpu,
            _SVTAV1_PRESET,
            _AOM_CPU_USED,
        )
        command += ["-pix_fmt", "yuv420p", "-tag:v", "av01", str(path)]
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    def write(self, frame: np.ndarray) -> None:
        assert self.process.stdin is not None
        try:
            self.process.stdin.write(memoryview(np.ascontiguousarray(frame)).cast("B"))
        except BrokenPipeError as error:
            detail = self.process.stderr.read().decode(errors="replace") if self.process.stderr else ""
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


def probe_encoder_runtime(args) -> None:
    command = [
        args.ffmpeg_bin,
        "-hide_banner",
        "-loglevel", "error",
        "-f", "lavfi",
        "-i", "color=black:size=128x128:rate=1,format=rgb24",
        "-frames:v", "1",
        "-c:v", args.video_codec,
        *_codec_args(
            args.video_codec,
            args.crf,
            args.cq,
            args.nvenc_preset,
            args.encode_gpu,
            args.svtav1_preset,
            args.aom_cpu_used,
        ),
        "-pix_fmt", "yuv420p",
        "-f", "null", "-",
    ]
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


def configure(args) -> None:
    global _SVTAV1_PRESET, _AOM_CPU_USED
    _SVTAV1_PRESET = int(args.svtav1_preset)
    _AOM_CPU_USED = int(args.aom_cpu_used)
