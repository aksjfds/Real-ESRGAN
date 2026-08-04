"""Uniform source sampling and fixed-algorithm degradation metrics."""

from __future__ import annotations

import shutil
import subprocess
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class FrameMetrics:
    timestamp: float
    jpeg_block: float
    ringing: float
    high_frequency_noise: float
    laplacian_sharpness: float
    directional_blur: float
    local_contrast: float
    flat_ratio: float
    line_ratio: float


class SourceAnalyzer:
    def __init__(self, ffmpeg_bin: str):
        self.ffmpeg_bin = ffmpeg_bin

    def _decode_frame(self, path: Path, timestamp: float, width: int, height: int) -> np.ndarray:
        command = [
            self.ffmpeg_bin,
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{timestamp:.6f}",
            "-i",
            str(path),
            "-frames:v",
            "1",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ]
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode:
            raise RuntimeError(result.stderr.decode(errors="replace"))
        expected = width * height * 3
        if len(result.stdout) != expected:
            raise RuntimeError(f"Analysis frame has {len(result.stdout)} bytes, expected {expected}")
        return np.frombuffer(result.stdout, np.uint8).reshape(height, width, 3)

    @staticmethod
    def metrics(frame: np.ndarray, timestamp: float) -> FrameMetrics:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        lap = cv2.Laplacian(gray, cv2.CV_32F)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        gradient = np.hypot(gx, gy)
        blur = cv2.GaussianBlur(gray, (0, 0), 1.0)
        noise = gray - blur
        # Compare the pixels immediately across each internal 8×8 boundary.
        # ``7::8`` contains one extra element when the dimension is exactly a
        # multiple of eight, so bound it to the number of right/bottom samples.
        right = gray[:, 8::8]
        left = gray[:, 7::8][:, : right.shape[1]]
        bottom = gray[8::8]
        top = gray[7::8][: bottom.shape[0]]
        vertical_boundaries = float(np.abs(right - left).mean()) if right.size else 0.0
        horizontal_boundaries = float(np.abs(bottom - top).mean()) if bottom.size else 0.0
        ordinary_x = np.abs(np.diff(gray, axis=1)).mean() + 1e-6
        ordinary_y = np.abs(np.diff(gray, axis=0)).mean() + 1e-6
        jpeg_block = float((vertical_boundaries / ordinary_x + horizontal_boundaries / ordinary_y) / 2)
        edge_mask = gradient > np.percentile(gradient, 85)
        ringing = float(np.abs(noise)[edge_mask].mean()) if edge_mask.any() else 0.0
        mean = cv2.blur(gray, (9, 9))
        local_std = np.sqrt(np.maximum(cv2.blur(gray * gray, (9, 9)) - mean * mean, 0))
        return FrameMetrics(
            timestamp=timestamp,
            jpeg_block=jpeg_block,
            ringing=ringing,
            high_frequency_noise=float(np.median(np.abs(noise)) * 1.4826),
            laplacian_sharpness=float(lap.var()),
            directional_blur=float(abs(np.abs(gx).mean() - np.abs(gy).mean())),
            local_contrast=float(local_std.mean()),
            flat_ratio=float((local_std < 0.015).mean()),
            line_ratio=float((gradient > 0.12).mean()),
        )

    def analyze(
        self,
        path: Path,
        width: int,
        height: int,
        start: float,
        duration: float,
        samples: int,
    ) -> list[FrameMetrics]:
        timestamps = np.linspace(start, max(start, start + duration - 0.001), samples)
        return [self.metrics(self._decode_frame(path, float(ts), width, height), float(ts)) for ts in timestamps]

    @staticmethod
    def recommend(rows: list[FrameMetrics]) -> tuple[str, str]:
        avg = {key: float(np.mean([getattr(row, key) for row in rows])) for key in asdict(rows[0]) if key != "timestamp"}
        if avg["jpeg_block"] > 1.35 and avg["ringing"] > 0.02:
            return "jpeg-artifacts", "JPEG block and edge-ringing scores are both elevated"
        if avg["directional_blur"] > 0.035 and avg["laplacian_sharpness"] < 0.01:
            return "directional-motion-blur", "strong directional blur with low sharpness"
        if avg["laplacian_sharpness"] < 0.004 and avg["directional_blur"] < 0.015:
            return "defocus-blur", "low non-directional sharpness"
        if avg["high_frequency_noise"] > 0.025:
            return "high-frequency-noise", "mixed high-frequency noise"
        return "ordinary-anime-compression", "ordinary compressed anime profile"


def getnative_available() -> bool:
    return shutil.which("getnative") is not None


@dataclass(frozen=True)
class NativeCandidate:
    frame: int
    kernel: str
    height: int
    error: float | None
    confidence: float | None
    raw_summary: str


def run_getnative(
    input_path: Path,
    frames: list[int],
    kernels: list[str],
    min_height: int,
    max_height: int,
) -> list[NativeCandidate]:
    executable = shutil.which("getnative")
    if executable is None:
        raise RuntimeError(
            "native analysis requires the real getnative CLI plus VapourSynth descale and a source plugin"
        )
    candidates: list[NativeCandidate] = []
    for frame in frames:
        for kernel in kernels:
            command = [
                executable,
                "--frame",
                str(frame),
                "--kernel",
                kernel,
                "--min-height",
                str(min_height),
                "--max-height",
                str(max_height),
                "--no-save",
                str(input_path),
            ]
            result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if result.returncode:
                raise RuntimeError(
                    f"getnative failed for frame={frame}, kernel={kernel}:\n"
                    f"{result.stderr.strip() or result.stdout.strip()}"
                )
            text = result.stdout + "\n" + result.stderr
            match = re.search(r"Native resolution\(s\) \(best guess\):\s*(\d+)p", text)
            if match is None:
                raise RuntimeError(f"getnative returned no parseable best guess:\n{text[-2000:]}")
            candidates.append(
                NativeCandidate(frame, kernel, int(match.group(1)), None, None, text[-1000:].strip())
            )
    return candidates
