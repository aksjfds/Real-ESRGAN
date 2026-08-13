"""Narrow public compatibility boundary for the legacy inference runtime.

Current v6.x orchestration imports this module instead of reaching into private
attributes of inference.runtime directly. Legacy implementations remain intact.
The APISR branch installs its SR-only backend here so spawned GPU workers see
exactly the same model registry and loader as the parent process.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Callable, Type

import numpy as np

from . import runtime as _legacy
from .apisr_backend import install_apisr_backend


install_apisr_backend(_legacy)

MODEL_URLS = _legacy.MODEL_URLS
VideoInfo = _legacy.VideoInfo
WorkerConfig = _legacy.WorkerConfig

_DEBAND_STRENGTH = 0.0
_DEBAND_RANGE = 16


def configure_deband(strength: float) -> None:
    global _DEBAND_STRENGTH
    _DEBAND_STRENGTH = float(strength)


class RawVideoReader(_legacy.RawVideoReader):
    """Legacy reader with optional FFmpeg deband before RGB extraction."""

    def __init__(
        self,
        input_path: Path,
        ffmpeg_bin: str,
        width: int,
        height: int,
        fps_rate: str,
        start: float,
        duration: float,
        bit_depth: int,
    ) -> None:
        strength = float(_DEBAND_STRENGTH)
        if strength <= 0.0:
            super().__init__(
                input_path,
                ffmpeg_bin,
                width,
                height,
                fps_rate,
                start,
                duration,
                bit_depth,
            )
            return

        self.width = width
        self.height = height
        self.pixel_format = "rgb48le" if bit_depth == 10 else "rgb24"
        self.dtype = np.dtype("<u2") if bit_depth == 10 else np.dtype(np.uint8)
        self.frame_bytes = width * height * 3 * self.dtype.itemsize

        threshold = f"{strength:.8g}"
        filter_chain = (
            f"deband=1thr={threshold}:2thr={threshold}:3thr={threshold}:"
            f"range={_DEBAND_RANGE}:blur=1,"
            f"fps={fps_rate}"
        )

        command = [ffmpeg_bin, "-hide_banner", "-loglevel", "error"]
        if start > 0:
            command += ["-ss", f"{start:.6f}"]
        command += ["-i", str(input_path)]
        command += [
            "-t",
            f"{duration:.6f}",
            "-vf",
            filter_chain,
            "-an",
            "-f",
            "rawvideo",
            "-pix_fmt",
            self.pixel_format,
            "pipe:1",
        ]
        print(
            "[deband] enabled before BasicVSR++: "
            f"strength={threshold}, range={_DEBAND_RANGE}, blur=1",
            flush=True,
        )
        self.process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )


format_seconds = _legacy.format_seconds
require_binary = _legacy.require_binary
probe_video = _legacy.probe_video
resolve_range = _legacy.resolve_range
parse_gpu_ids = _legacy.parse_gpu_ids
resolve_model_paths = _legacy.resolve_model_paths
load_worker_model = _legacy.load_worker_model
infer_frame = _legacy.infer_frame
mux_audio = _legacy.mux_audio


def model_native_scale(name: str) -> int:
    return _legacy._model_native_scale(name)


def output_pixel_format(codec: str, bit_depth: int) -> str:
    return _legacy._output_pixel_format(codec, bit_depth)


def get_encoding_backend() -> tuple[Callable[[str, str], None], Type]:
    require_encoder = _legacy._require_encoder
    writer_type = _legacy._writer_type
    if require_encoder is None or writer_type is None:
        raise RuntimeError(
            "Encoding backend is not configured. Run through root inference.py."
        )
    return require_encoder, writer_type
