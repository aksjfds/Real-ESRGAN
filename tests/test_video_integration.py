"""Opt-in CUDA/video integration checks.

Set REALESRGAN_INTEGRATION=1, REALESRGAN_TEST_VIDEO and
REALESRGAN_ANIMEVIDEO_CHECKPOINT to run these expensive ten-second tests.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch


ENABLED = os.environ.get("REALESRGAN_INTEGRATION") == "1"
VIDEO = os.environ.get("REALESRGAN_TEST_VIDEO", "")
CHECKPOINT = os.environ.get("REALESRGAN_ANIMEVIDEO_CHECKPOINT", "")

pytestmark = pytest.mark.skipif(
    not ENABLED or not VIDEO or not CHECKPOINT or torch.cuda.device_count() < 2,
    reason="requires explicit opt-in, a test video/checkpoint, and two CUDA GPUs",
)


def probe_duration(path: Path, selector: str) -> float | None:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            selector,
            "-show_entries",
            "stream=duration",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    streams = json.loads(result.stdout).get("streams", [])
    if not streams or streams[0].get("duration") in (None, "N/A"):
        return None
    return float(streams[0]["duration"])


@pytest.mark.parametrize("preset", ["baseline", "safe"])
def test_ten_second_presets_dual_gpu_and_av_sync(tmp_path: Path, preset: str):
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("ffmpeg/ffprobe are required")
    output = tmp_path / f"{preset}.mp4"
    command = [
        sys.executable,
        str(Path(__file__).parents[1] / "realesrgan.py"),
        "--input",
        VIDEO,
        "--output",
        str(output),
        "--model",
        "realesr-animevideov3",
        "--model-path",
        CHECKPOINT,
        "--quality-preset",
        preset,
        "--scale",
        "2",
        "--gpu-ids",
        "0,1",
        "--tile-size",
        "256",
        "--batch-size",
        "4",
        "--start-time",
        "0",
        "--test-seconds",
        "10",
        "--video-codec",
        "hevc_nvenc",
        "--output-pix-fmt",
        "auto",
    ]
    result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    assert result.returncode == 0, result.stdout[-8000:]
    assert result.stdout.count("model resident on cuda:") == 2
    assert "cuda:0" in result.stdout and "cuda:1" in result.stdout
    assert output.is_file() and output.stat().st_size > 0
    video_duration = probe_duration(output, "v:0")
    audio_duration = probe_duration(output, "a:0")
    assert video_duration is not None and abs(video_duration - 10.0) < 0.15
    if audio_duration is not None:
        assert abs(audio_duration - video_duration) < 0.15
