"""Low-coupling FFmpeg audio enhancement and mux boundary for v8."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
from typing import Sequence


def _run_checked(command: Sequence[str], label: str) -> subprocess.CompletedProcess:
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"{label} failed (exit {result.returncode}):\n{detail}")
    return result


def _voice_filter() -> str:
    """Conservative dialogue-focused chain for already-clean anime audio."""
    return ",".join(
        (
            "highpass=f=55",
            "equalizer=f=500:t=q:w=1:g=-0.8",
            "equalizer=f=3000:t=q:w=1:g=0.8",
            "highshelf=f=11000:t=q:w=0.7:g=0.8",
            (
                "acompressor=threshold=0.125:ratio=1.8:attack=20:"
                "release=120:makeup=1:knee=2.828:link=average:detection=rms"
            ),
            "deesser=i=0.12:m=0.35:f=0.55",
            "loudnorm=I=-16:LRA=7:TP=-1.5",
            "aresample=48000",
        )
    )


def mux_audio(
    silent_video: Path,
    input_path: Path,
    output_path: Path,
    ffmpeg_bin: str,
    start: float,
    duration: float,
    has_audio: bool,
    audio_codec: str,
    audio_bitrate: str,
    *,
    enhance: bool = False,
) -> None:
    """Mux source audio into processed video; preserve v7 behavior when disabled."""
    if not has_audio:
        silent_video.replace(output_path)
        return

    if enhance and audio_codec != "aac":
        raise ValueError(
            "Audio enhancement requires AUDIO_CODEC='aac' because filtered audio "
            "must be re-encoded."
        )

    base = [
        ffmpeg_bin,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(silent_video),
    ]

    if enhance:
        filtergraph = (
            f"[1:a:0]atrim=start=0:duration={duration:.6f},"
            f"asetpts=PTS-STARTPTS,{_voice_filter()}[a]"
        )
        command = base + [
            "-ss",
            f"{start:.6f}",
            "-t",
            f"{duration:.6f}",
            "-i",
            str(input_path),
            "-filter_complex",
            filtergraph,
            "-map",
            "0:v:0",
            "-map",
            "[a]",
            "-map_metadata",
            "1",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            audio_bitrate,
            "-ar",
            "48000",
            "-shortest",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
        _run_checked(command, "audio enhancement mux")
        return

    if audio_codec == "aac":
        command = base + [
            "-ss",
            f"{start:.6f}",
            "-t",
            f"{duration:.6f}",
            "-i",
            str(input_path),
            "-filter_complex",
            f"[1:a:0]atrim=start=0:duration={duration:.6f},asetpts=PTS-STARTPTS[a]",
            "-map",
            "0:v:0",
            "-map",
            "[a]",
            "-map_metadata",
            "1",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            audio_bitrate,
            "-shortest",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
    else:
        command = base + [
            "-ss",
            f"{start:.6f}",
            "-t",
            f"{duration:.6f}",
            "-i",
            str(input_path),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-map_metadata",
            "1",
            "-c",
            "copy",
            "-shortest",
            "-avoid_negative_ts",
            "make_zero",
            "-movflags",
            "+faststart",
            str(output_path),
        ]

    try:
        _run_checked(command, "audio mux")
    except RuntimeError:
        if audio_codec != "copy":
            raise
        print("[warning] audio stream copy failed; retrying with AAC", flush=True)
        mux_audio(
            silent_video,
            input_path,
            output_path,
            ffmpeg_bin,
            start,
            duration,
            has_audio,
            "aac",
            audio_bitrate,
            enhance=False,
        )


def _probe_media(input_path: Path, ffprobe_bin: str) -> tuple[float, bool]:
    command = [
        ffprobe_bin,
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_streams",
        "-show_format",
        str(input_path),
    ]
    data = json.loads(_run_checked(command, "ffprobe").stdout)
    duration_value = data.get("format", {}).get("duration")
    if duration_value in {None, "N/A"}:
        raise ValueError("The input has no usable duration metadata.")
    has_audio = any(
        stream.get("codec_type") == "audio" for stream in data.get("streams", [])
    )
    return float(duration_value), has_audio


def process_media(
    input_path: Path,
    output_path: Path,
    ffmpeg_bin: str,
    ffprobe_bin: str,
    start: float,
    test_seconds: float,
    audio_codec: str,
    audio_bitrate: str,
    *,
    enhance: bool,
) -> None:
    """Bypass video enhancement and optionally enhance audio with video stream copy."""
    input_path = input_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input video not found: {input_path}")
    if input_path == output_path:
        raise ValueError("Input and output paths must be different.")
    if enhance and audio_codec != "aac":
        raise ValueError(
            "Audio enhancement requires AUDIO_CODEC='aac' because filtered audio "
            "must be re-encoded."
        )

    total_duration, has_audio = _probe_media(input_path, ffprobe_bin)
    if start < 0 or start >= total_duration:
        raise ValueError(f"START_TIME must be in [0, {total_duration:.3f}).")
    available = total_duration - start
    duration = min(test_seconds, available) if test_seconds > 0 else available
    if duration <= 0:
        raise ValueError("Selected media range is empty.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg_bin,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{start:.6f}",
        "-t",
        f"{duration:.6f}",
        "-i",
        str(input_path),
        "-map",
        "0:v:0",
    ]

    if not has_audio:
        command += ["-c:v", "copy"]
    elif enhance:
        command += [
            "-map",
            "0:a:0",
            "-c:v",
            "copy",
            "-af",
            _voice_filter(),
            "-c:a",
            "aac",
            "-b:a",
            audio_bitrate,
            "-ar",
            "48000",
        ]
    elif audio_codec == "aac":
        command += [
            "-map",
            "0:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            audio_bitrate,
        ]
    else:
        command += ["-map", "0:a:0", "-c", "copy"]

    command += [
        "-map_metadata",
        "0",
        "-avoid_negative_ts",
        "make_zero",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    _run_checked(command, "media bypass/audio processing")
