"""FFmpeg/ffprobe media I/O without audio processing."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    fps_num: int
    fps_den: int
    duration: float
    frames: Optional[int]
    has_audio: bool
    pix_fmt: str
    bit_depth: int

    @property
    def fps(self) -> float:
        return self.fps_num / self.fps_den


def format_seconds(value: float) -> str:
    hours = int(value // 3600)
    minutes = int((value % 3600) // 60)
    seconds = value % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"


def run_checked(command: Sequence[str], label: str) -> subprocess.CompletedProcess:
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"{label} failed (exit {result.returncode}):\n{detail}")
    return result


def require_binary(name: str) -> None:
    if shutil.which(name) is None:
        raise FileNotFoundError(f"Required executable '{name}' was not found.")


def parse_rate(value: str) -> Fraction:
    if not value or value in {"0/0", "N/A"}:
        raise ValueError(f"Invalid video frame rate: {value!r}")
    rate = Fraction(value)
    if rate <= 0:
        raise ValueError(f"Invalid video frame rate: {value!r}")
    return rate


def _stream_bit_depth(stream: dict) -> Tuple[int, str]:
    pix_fmt = str(stream.get("pix_fmt") or "unknown").lower()
    raw_value = str(stream.get("bits_per_raw_sample") or "").strip()
    raw_bits = int(raw_value) if raw_value.isdigit() and int(raw_value) > 0 else 0
    if "p010" in pix_fmt:
        bits = 10
    else:
        match = re.search(r"(?:p|gray|rgb|bgr)(9|10|12|14|16)(?:le|be)$", pix_fmt)
        bits = int(match.group(1)) if match else raw_bits or 8
    if bits <= 8:
        return 8, pix_fmt
    if bits == 10:
        return 10, pix_fmt
    raise RuntimeError(f"Source pixel format {pix_fmt!r} reports {bits}-bit samples; only 8-bit and 10-bit are supported.")


def probe_video(path: Path, ffprobe_bin: str) -> VideoInfo:
    data = json.loads(run_checked([
        ffprobe_bin, "-v", "error", "-print_format", "json",
        "-show_streams", "-show_format", str(path),
    ], "ffprobe").stdout)
    video_streams = [item for item in data.get("streams", []) if item.get("codec_type") == "video"]
    if not video_streams:
        raise ValueError(f"No video stream found in {path}")
    stream = video_streams[0]
    rate = parse_rate(stream.get("avg_frame_rate") or stream.get("r_frame_rate"))
    duration_value = stream.get("duration") or data.get("format", {}).get("duration")
    if duration_value in {None, "N/A"}:
        raise ValueError("The input has no usable duration metadata.")
    frame_value = stream.get("nb_frames")
    frames = int(frame_value) if frame_value not in {None, "N/A"} else None
    bit_depth, pix_fmt = _stream_bit_depth(stream)
    return VideoInfo(
        width=int(stream["width"]), height=int(stream["height"]),
        fps_num=rate.numerator, fps_den=rate.denominator,
        duration=float(duration_value), frames=frames,
        has_audio=any(item.get("codec_type") == "audio" for item in data.get("streams", [])),
        pix_fmt=pix_fmt, bit_depth=bit_depth,
    )


def resolve_range(info: VideoInfo, start: float, test_seconds: float, output_rate: Fraction) -> Tuple[float, float, int]:
    if start < 0 or start >= info.duration:
        raise ValueError(f"--start-time must be in [0, {info.duration:.3f}).")
    available = info.duration - start
    duration = min(test_seconds, available) if test_seconds > 0 else available
    if duration <= 0:
        raise ValueError("Selected video range is empty.")
    return start, duration, max(1, int(round(duration * float(output_rate))))


class RawVideoReader:
    def __init__(self, input_path: Path, ffmpeg_bin: str, width: int, height: int, fps_rate: str, start: float, duration: float, bit_depth: int) -> None:
        self.width = width
        self.height = height
        self.pixel_format = "rgb48le" if bit_depth == 10 else "rgb24"
        self.dtype = np.dtype("<u2") if bit_depth == 10 else np.dtype(np.uint8)
        self.frame_bytes = width * height * 3 * self.dtype.itemsize
        command = [ffmpeg_bin, "-hide_banner", "-loglevel", "error"]
        if start > 0:
            command += ["-ss", f"{start:.6f}"]
        command += ["-i", str(input_path), "-t", f"{duration:.6f}", "-vf", f"fps={fps_rate}", "-an", "-f", "rawvideo", "-pix_fmt", self.pixel_format, "pipe:1"]
        self.process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def read(self) -> Optional[np.ndarray]:
        assert self.process.stdout is not None
        data = self.process.stdout.read(self.frame_bytes)
        if not data:
            return None
        if len(data) != self.frame_bytes:
            raise RuntimeError(f"ffmpeg returned a partial raw frame ({len(data)}/{self.frame_bytes} bytes).")
        return np.frombuffer(data, dtype=self.dtype).reshape(self.height, self.width, 3)

    def close(self) -> None:
        if self.process.stdout is not None:
            self.process.stdout.close()
        stderr = b""
        if self.process.stderr is not None:
            stderr = self.process.stderr.read()
            self.process.stderr.close()
        return_code = self.process.wait()
        if return_code != 0:
            raise RuntimeError(f"ffmpeg decode failed (exit {return_code}):\n{stderr.decode(errors='replace')}")


def mux_original_audio(silent_video: Path, input_path: Path, output_path: Path, ffmpeg_bin: str, start: float, duration: float, has_audio: bool) -> None:
    """Mux the first source audio stream using stream copy only."""
    if not has_audio:
        silent_video.replace(output_path)
        return
    run_checked([
        ffmpeg_bin, "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(silent_video),
        "-ss", f"{start:.6f}", "-t", f"{duration:.6f}", "-i", str(input_path),
        "-map", "0:v:0", "-map", "1:a:0", "-map_metadata", "1",
        "-c", "copy", "-shortest", "-avoid_negative_ts", "make_zero",
        "-movflags", "+faststart", str(output_path),
    ], "original audio stream-copy mux")


def parse_gpu_ids(value: str) -> List[Optional[int]]:
    if value.strip().lower() == "cpu":
        return [None]
    try:
        ids = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise ValueError("--gpu-ids must be 'cpu' or comma-separated IDs such as 0,1.") from error
    if not ids or len(ids) != len(set(ids)) or min(ids) < 0:
        raise ValueError("--gpu-ids must contain unique, non-negative GPU numbers.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Use --gpu-ids cpu for compatibility.")
    count = torch.cuda.device_count()
    if max(ids) >= count:
        raise ValueError(f"Requested GPU {max(ids)}, but only {count} CUDA device(s) are visible.")
    return ids


def output_pixel_format(codec: str, bit_depth: int) -> str:
    if bit_depth == 8:
        return "yuv420p"
    return "p010le" if codec.endswith("_nvenc") else "yuv420p10le"
