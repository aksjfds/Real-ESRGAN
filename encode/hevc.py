"""H.264/HEVC FFmpeg encoders.

This module contains all software x264/x265 and NVIDIA NVENC H.264/HEVC
encoding details.  The parameter set intentionally matches the previously
working master/2.7 writer.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np


CODECS = {"libx264", "libx265", "h264_nvenc", "hevc_nvenc"}


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


def validate_args(args) -> None:
    if args.video_codec not in CODECS:
        raise ValueError(f"Unsupported H.264/HEVC encoder: {args.video_codec}")
    if not 0 <= args.crf <= 51:
        raise ValueError("--crf must be between 0 and 51.")
    if not 0 <= args.cq <= 51:
        raise ValueError("--cq must be between 0 and 51.")
    if args.encode_gpu < 0:
        raise ValueError("--encode-gpu must be non-negative.")


class RawVideoWriter:
    """RGB24 FFmpeg writer for software H.264/HEVC and NVENC H.264/HEVC."""

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
        if codec not in CODECS:
            raise RuntimeError(f"Non-H.264/HEVC codec reached HEVC writer: {codec}")

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
        if codec in {"libx264", "libx265"}:
            command += ["-preset", preset, "-crf", str(crf)]
        else:
            command += [
                "-gpu",
                str(encode_gpu),
                "-preset",
                nvenc_preset,
                "-tune",
                "hq",
                "-rc",
                "vbr",
                "-cq",
                str(cq),
                "-b:v",
                "0",
                "-multipass",
                "fullres",
                "-spatial_aq",
                "1",
                "-temporal_aq",
                "1",
                "-rc-lookahead",
                "32",
                "-bf",
                "3",
            ]

        command += ["-pix_fmt", "yuv420p"]
        if codec in {"libx265", "hevc_nvenc"}:
            command += ["-tag:v", "hvc1"]
        command.append(str(path))
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
