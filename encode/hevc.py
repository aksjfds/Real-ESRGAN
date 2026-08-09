"""H.264/HEVC FFmpeg encoders.

This module contains all software x264/x265 and NVIDIA NVENC H.264/HEVC
encoding details. The parameter set intentionally matches the previously
working master/2.7 writer while preserving source bit depth automatically.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional

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


def _frame_pixel_formats(frame: np.ndarray, codec: str) -> tuple[str, str]:
    if frame.dtype.kind == "u" and frame.dtype.itemsize == 1:
        return "rgb24", "yuv420p"
    if frame.dtype.kind == "u" and frame.dtype.itemsize == 2:
        # NVENC consumes 10-bit 4:2:0 through P010; software encoders use the
        # planar 10-bit format. Both produce a 10-bit encoded video stream.
        output_pix_fmt = "p010le" if codec.endswith("_nvenc") else "yuv420p10le"
        return "rgb48le", output_pix_fmt
    raise RuntimeError(f"Unsupported inference frame dtype for encoding: {frame.dtype}")


class RawVideoWriter:
    """FFmpeg writer that selects 8-bit or 10-bit output from the first frame."""

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

        self.path = path
        self.ffmpeg_bin = ffmpeg_bin
        self.width = width
        self.height = height
        self.input_fps_rate = input_fps_rate
        self.output_fps_rate = output_fps_rate
        self.codec = codec
        self.crf = crf
        self.preset = preset
        self.cq = cq
        self.nvenc_preset = nvenc_preset
        self.encode_gpu = encode_gpu
        self.process: Optional[subprocess.Popen] = None

    def _start(self, frame: np.ndarray) -> None:
        if self.process is not None:
            return

        raw_pix_fmt, output_pix_fmt = _frame_pixel_formats(frame, self.codec)
        command = [
            self.ffmpeg_bin,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            raw_pix_fmt,
            "-s:v",
            f"{self.width}x{self.height}",
            "-r",
            self.input_fps_rate,
            "-i",
            "pipe:0",
            "-an",
        ]
        if self.output_fps_rate != self.input_fps_rate:
            command += ["-vf", f"fps={self.output_fps_rate}"]

        command += ["-c:v", self.codec]
        if self.codec in {"libx264", "libx265"}:
            command += ["-preset", self.preset, "-crf", str(self.crf)]
        else:
            command += [
                "-gpu",
                str(self.encode_gpu),
                "-preset",
                self.nvenc_preset,
                "-tune",
                "hq",
                "-rc",
                "vbr",
                "-cq",
                str(self.cq),
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

        command += ["-pix_fmt", output_pix_fmt]
        if self.codec in {"libx265", "hevc_nvenc"}:
            command += ["-tag:v", "hvc1"]
        command.append(str(self.path))

        print(
            f"[encoder] pixel format: {raw_pix_fmt} -> {output_pix_fmt} ({self.codec})",
            flush=True,
        )
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    def write(self, frame: np.ndarray) -> None:
        frame = np.ascontiguousarray(frame)
        self._start(frame)
        assert self.process is not None and self.process.stdin is not None
        try:
            self.process.stdin.write(memoryview(frame).cast("B"))
        except BrokenPipeError as error:
            detail = self.process.stderr.read().decode(errors="replace") if self.process.stderr else ""
            raise RuntimeError(f"ffmpeg encoder closed its input early:\n{detail}") from error

    def close(self) -> None:
        if self.process is None:
            return
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
