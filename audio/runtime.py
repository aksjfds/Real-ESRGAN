"""Low-coupling v8.1 AI dialogue enhancement and mux boundary."""

from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time
from typing import Iterator, Sequence

from .backend import separate_dialogue, validate_backend
from .dsp import decode_stereo, encode_wav, enhance_mix

_SAMPLE_RATE = 48000
_LOUDNESS_I = -16.0
_LOUDNESS_LRA = 7.0
_LOUDNESS_TP = -1.5


def _run_checked(
    command: Sequence[str],
    label: str,
    *,
    capture: bool = True,
) -> subprocess.CompletedProcess:
    result = subprocess.run(
        list(command),
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
        text=True,
    )
    if result.returncode != 0:
        detail = ""
        if capture:
            detail = (result.stderr or result.stdout or "").strip()
        suffix = f"\n{detail}" if detail else ""
        raise RuntimeError(f"{label} failed (exit {result.returncode}){suffix}")
    return result


def validate_runtime(ffmpeg_bin: str) -> None:
    """Fail before video inference if the isolated v8.1 backend is unavailable."""
    if shutil.which(ffmpeg_bin) is None:
        raise FileNotFoundError(f"Required executable '{ffmpeg_bin}' was not found.")
    validate_backend()


def _extract_source_audio(
    input_path: Path,
    target: Path,
    ffmpeg_bin: str,
    start: float,
    duration: float,
) -> None:
    _run_checked(
        [
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
            "0:a:0",
            "-vn",
            "-ac",
            "2",
            "-ar",
            str(_SAMPLE_RATE),
            "-c:a",
            "pcm_f32le",
            str(target),
        ],
        "source audio extraction",
    )


def _parse_loudnorm(stderr: str) -> dict[str, float]:
    matches = re.findall(r"\{(?:.|\n)*?\}", stderr)
    for block in reversed(matches):
        try:
            data = json.loads(block)
        except json.JSONDecodeError:
            continue
        required = ("input_i", "input_tp", "input_lra", "input_thresh", "target_offset")
        if not all(key in data for key in required):
            continue
        try:
            return {key: float(data[key]) for key in required}
        except (TypeError, ValueError):
            continue
    raise RuntimeError("Unable to parse EBU R128 loudnorm measurement")


def _measure_loudness(processed_wav: Path, ffmpeg_bin: str) -> dict[str, float]:
    filter_text = (
        f"loudnorm=I={_LOUDNESS_I:g}:LRA={_LOUDNESS_LRA:g}:"
        f"TP={_LOUDNESS_TP:g}:print_format=json"
    )
    result = subprocess.run(
        [
            ffmpeg_bin,
            "-hide_banner",
            "-loglevel",
            "info",
            "-i",
            str(processed_wav),
            "-af",
            filter_text,
            "-f",
            "null",
            "-",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"loudness measurement failed (exit {result.returncode}):\n{detail}")
    return _parse_loudnorm(result.stderr)


def _linear_loudnorm_filter(measured: dict[str, float]) -> str:
    return (
        f"loudnorm=I={_LOUDNESS_I:g}:LRA={_LOUDNESS_LRA:g}:TP={_LOUDNESS_TP:g}:"
        f"measured_I={measured['input_i']:.6f}:"
        f"measured_LRA={measured['input_lra']:.6f}:"
        f"measured_TP={measured['input_tp']:.6f}:"
        f"measured_thresh={measured['input_thresh']:.6f}:"
        f"offset={measured['target_offset']:.6f}:linear=true:print_format=summary"
    )


@contextmanager
def _enhanced_audio(
    input_path: Path,
    ffmpeg_bin: str,
    start: float,
    duration: float,
    work_parent: Path,
) -> Iterator[tuple[Path, dict[str, float]]]:
    """Produce a bounded-lifetime 48 kHz enhanced WAV plus loudness measurement."""
    with tempfile.TemporaryDirectory(prefix="realesrgan-audio-v81-", dir=work_parent) as temp_dir:
        root = Path(temp_dir)
        source_wav = root / "source.wav"
        stem_dir = root / "stems"
        processed_wav = root / "processed.wav"

        stage = time.monotonic()
        _extract_source_audio(input_path, source_wav, ffmpeg_bin, start, duration)
        extract_s = time.monotonic() - stage

        stage = time.monotonic()
        dialogue_wav = separate_dialogue(source_wav, stem_dir)
        separation_s = time.monotonic() - stage

        stage = time.monotonic()
        original = decode_stereo(source_wav, ffmpeg_bin)
        dialogue = decode_stereo(dialogue_wav, ffmpeg_bin)
        enhanced = enhance_mix(original, dialogue)
        encode_wav(enhanced, processed_wav, ffmpeg_bin)
        dsp_s = time.monotonic() - stage

        stage = time.monotonic()
        measured = _measure_loudness(processed_wav, ffmpeg_bin)
        measure_s = time.monotonic() - stage

        print(
            "[audio-v8.1] Mel-Band RoFormer + adaptive DSP ready | "
            f"extract={extract_s:.1f}s | separation={separation_s:.1f}s | "
            f"dsp={dsp_s:.1f}s | loudness_scan={measure_s:.1f}s",
            flush=True,
        )
        yield processed_wav, measured


def _mux_enhanced(
    video_source: Path,
    processed_wav: Path,
    output_path: Path,
    ffmpeg_bin: str,
    audio_bitrate: str,
    measured: dict[str, float],
    *,
    metadata_source: Path | None = None,
    start: float | None = None,
    duration: float | None = None,
) -> None:
    command = [ffmpeg_bin, "-y", "-hide_banner", "-loglevel", "error"]
    if start is not None:
        command += ["-ss", f"{start:.6f}"]
    if duration is not None:
        command += ["-t", f"{duration:.6f}"]
    command += ["-i", str(video_source), "-i", str(processed_wav)]
    metadata_index = 0
    if metadata_source is not None and metadata_source != video_source:
        command += ["-i", str(metadata_source)]
        metadata_index = 2
    command += [
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-map_metadata",
        str(metadata_index),
        "-c:v",
        "copy",
        "-af",
        _linear_loudnorm_filter(measured),
        "-c:a",
        "aac",
        "-b:a",
        audio_bitrate,
        "-ar",
        str(_SAMPLE_RATE),
        "-shortest",
        "-avoid_negative_ts",
        "make_zero",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    _run_checked(command, "v8.1 enhanced audio mux")


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
    """Mux source or v8.1 enhanced audio into an already processed video."""
    if not has_audio:
        silent_video.replace(output_path)
        return

    if enhance:
        if audio_codec != "aac":
            raise ValueError("v8.1 audio enhancement requires AUDIO_CODEC='aac'")
        validate_runtime(ffmpeg_bin)
        with _enhanced_audio(input_path, ffmpeg_bin, start, duration, output_path.parent) as (
            processed_wav,
            measured,
        ):
            _mux_enhanced(
                silent_video,
                processed_wav,
                output_path,
                ffmpeg_bin,
                audio_bitrate,
                measured,
                metadata_source=input_path,
            )
        return

    base = [
        ffmpeg_bin,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(silent_video),
    ]
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
    data = json.loads(
        _run_checked(
            [
                ffprobe_bin,
                "-v",
                "error",
                "-print_format",
                "json",
                "-show_streams",
                "-show_format",
                str(input_path),
            ],
            "ffprobe",
        ).stdout
    )
    duration_value = data.get("format", {}).get("duration")
    if duration_value in {None, "N/A"}:
        raise ValueError("The input has no usable duration metadata.")
    has_audio = any(stream.get("codec_type") == "audio" for stream in data.get("streams", []))
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
    """Copy video unchanged and optionally run the complete v8.1 audio pipeline."""
    input_path = input_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input video not found: {input_path}")
    if input_path == output_path:
        raise ValueError("Input and output paths must be different.")

    total_duration, has_audio = _probe_media(input_path, ffprobe_bin)
    if start < 0 or start >= total_duration:
        raise ValueError(f"START_TIME must be in [0, {total_duration:.3f}).")
    available = total_duration - start
    duration = min(test_seconds, available) if test_seconds > 0 else available
    if duration <= 0:
        raise ValueError("Selected media range is empty.")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not has_audio:
        _run_checked(
            [
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
                "-c:v",
                "copy",
                "-map_metadata",
                "0",
                "-avoid_negative_ts",
                "make_zero",
                "-movflags",
                "+faststart",
                str(output_path),
            ],
            "video bypass",
        )
        return

    if enhance:
        if audio_codec != "aac":
            raise ValueError("v8.1 audio enhancement requires AUDIO_CODEC='aac'")
        validate_runtime(ffmpeg_bin)
        with _enhanced_audio(input_path, ffmpeg_bin, start, duration, output_path.parent) as (
            processed_wav,
            measured,
        ):
            _mux_enhanced(
                input_path,
                processed_wav,
                output_path,
                ffmpeg_bin,
                audio_bitrate,
                measured,
                metadata_source=input_path,
                start=start,
                duration=duration,
            )
        return

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
        "-map",
        "0:a:0",
        "-map_metadata",
        "0",
        "-c:v",
        "copy",
    ]
    if audio_codec == "aac":
        command += ["-c:a", "aac", "-b:a", audio_bitrate]
    else:
        command += ["-c:a", "copy"]
    command += [
        "-avoid_negative_ts",
        "make_zero",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    _run_checked(command, "media bypass")
