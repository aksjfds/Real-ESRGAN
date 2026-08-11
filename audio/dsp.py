"""Adaptive dialogue DSP and voice-aware background ducking for v8.1."""

from __future__ import annotations

import math
from pathlib import Path
import subprocess
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

_SAMPLE_RATE = 48000
_N_FFT = 2048
_HOP = 512


def _run(command: Sequence[str], label: str, *, input_bytes: bytes | None = None) -> subprocess.CompletedProcess:
    result = subprocess.run(
        list(command),
        input=input_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        detail = result.stderr.decode(errors="replace").strip()
        raise RuntimeError(f"{label} failed (exit {result.returncode}):\n{detail}")
    return result


def decode_stereo(path: Path, ffmpeg_bin: str) -> np.ndarray:
    result = _run(
        [
            ffmpeg_bin,
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(path),
            "-vn",
            "-ac",
            "2",
            "-ar",
            str(_SAMPLE_RATE),
            "-f",
            "f32le",
            "-acodec",
            "pcm_f32le",
            "pipe:1",
        ],
        "audio decode",
    )
    samples = np.frombuffer(result.stdout, dtype="<f4")
    if samples.size == 0 or samples.size % 2:
        raise RuntimeError(f"Invalid decoded audio length for {path}")
    return samples.reshape(-1, 2).T.copy()


def encode_wav(audio: np.ndarray, path: Path, ffmpeg_bin: str) -> None:
    if audio.ndim != 2 or audio.shape[0] != 2:
        raise ValueError(f"Expected stereo [2,N], got {audio.shape}")
    path.parent.mkdir(parents=True, exist_ok=True)
    interleaved = np.ascontiguousarray(audio.T.astype("<f4", copy=False))
    _run(
        [
            ffmpeg_bin,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "f32le",
            "-acodec",
            "pcm_f32le",
            "-ar",
            str(_SAMPLE_RATE),
            "-ac",
            "2",
            "-i",
            "pipe:0",
            "-c:a",
            "pcm_f32le",
            str(path),
        ],
        "audio encode",
        input_bytes=interleaved.tobytes(),
    )


def _smooth_curve(values: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return values
    kernel = radius * 2 + 1
    padded = F.pad(values[None, None], (radius, radius), mode="replicate")
    return F.avg_pool1d(padded, kernel_size=kernel, stride=1)[0, 0]


def _compress_dialogue(wave: torch.Tensor, sample_rate: int) -> torch.Tensor:
    """Stereo-linked gentle compressor that avoids flattening quiet breath detail."""
    mono = wave.abs().mean(dim=0)
    frame = max(1, int(sample_rate * 0.010))
    count = int(math.ceil(mono.numel() / frame))
    padded = F.pad(mono, (0, count * frame - mono.numel()))
    rms = padded.view(count, frame).square().mean(dim=1).sqrt().clamp_min(1e-7)
    db = 20.0 * torch.log10(rms)
    threshold = -18.0
    ratio = 1.6
    over = (db - threshold).clamp_min(0.0)
    gain_db = -over * (1.0 - 1.0 / ratio)
    gain_db = torch.where(db < -32.0, gain_db * 0.35, gain_db)
    gain_db = _smooth_curve(gain_db, radius=6)
    gain = torch.pow(10.0, gain_db / 20.0).repeat_interleave(frame)[: wave.shape[1]]
    return wave * gain.unsqueeze(0)


def _spectral_process(dialogue: torch.Tensor, background: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    device = dialogue.device
    window = torch.hann_window(_N_FFT, device=device)
    d_spec = torch.stft(dialogue, n_fft=_N_FFT, hop_length=_HOP, window=window, return_complex=True)
    b_spec = torch.stft(background, n_fft=_N_FFT, hop_length=_HOP, window=window, return_complex=True)
    freq = torch.linspace(0.0, _SAMPLE_RATE / 2.0, d_spec.shape[-2], device=device)
    d_mag = d_spec.abs().mean(dim=0).clamp_min(1e-7)
    b_mag = b_spec.abs().mean(dim=0).clamp_min(1e-7)

    def band(lo: float, hi: float) -> torch.Tensor:
        return (freq >= lo) & (freq < hi)

    speech_band = band(1200.0, 5000.0)
    sibilant_band = band(5500.0, 9500.0)
    air_band = band(10000.0, 16000.0)

    speech_energy = d_mag[speech_band].square().mean(dim=0).sqrt()
    sib_energy = d_mag[sibilant_band].square().mean(dim=0).sqrt()
    air_energy = d_mag[air_band].square().mean(dim=0).sqrt()
    bg_speech = b_mag[speech_band].square().mean(dim=0).sqrt()

    speech_db = 20.0 * torch.log10(speech_energy.clamp_min(1e-7))
    activity = torch.sigmoid((speech_db + 38.0) / 4.0)

    brightness = (sib_energy + air_energy) / speech_energy.clamp_min(1e-7)
    presence_db = 1.35 * activity * torch.sigmoid((0.75 - brightness) * 4.0)

    sib_ratio = sib_energy / speech_energy.clamp_min(1e-7)
    deess_db = -4.0 * activity * torch.sigmoid((sib_ratio - 0.72) * 8.0)

    air_ratio = air_energy / speech_energy.clamp_min(1e-7)
    air_db = (
        0.8
        * activity
        * torch.sigmoid((0.38 - air_ratio) * 7.0)
        * torch.sigmoid((0.80 - sib_ratio) * 8.0)
    )

    presence_db = _smooth_curve(presence_db, 3)
    deess_db = _smooth_curve(deess_db, 2)
    air_db = _smooth_curve(air_db, 3)

    gain_db = torch.zeros_like(d_mag)
    gain_db[band(1800.0, 4800.0)] += presence_db
    gain_db[sibilant_band] += deess_db
    gain_db[air_band] += air_db

    low = freq < 70.0
    low_gain = ((freq / 70.0).clamp(0.0, 1.0) ** 2).clamp_min(0.02)
    spectral_gain = torch.pow(10.0, gain_db / 20.0)
    spectral_gain[low] *= low_gain[low, None]
    d_spec = d_spec * spectral_gain.unsqueeze(0)

    dominance = speech_energy / (speech_energy + bg_speech + 1e-7)
    duck_db = _smooth_curve(-2.0 * activity * dominance, 4)
    bg_gain = torch.ones_like(b_mag)
    bg_gain[band(1500.0, 5000.0)] = torch.pow(10.0, duck_db / 20.0)
    b_spec = b_spec * bg_gain.unsqueeze(0)

    length = dialogue.shape[1]
    dialogue_out = torch.istft(d_spec, n_fft=_N_FFT, hop_length=_HOP, window=window, length=length)
    background_out = torch.istft(b_spec, n_fft=_N_FFT, hop_length=_HOP, window=window, length=length)
    return dialogue_out, background_out


def _enhance_chunk(original: np.ndarray, dialogue: np.ndarray, device: torch.device) -> np.ndarray:
    background = original - dialogue
    d = torch.from_numpy(dialogue).to(device=device, dtype=torch.float32)
    b = torch.from_numpy(background).to(device=device, dtype=torch.float32)
    with torch.inference_mode():
        d, b = _spectral_process(d, b)
        d = _compress_dialogue(d, _SAMPLE_RATE)
        d = d * (10.0 ** (0.4 / 20.0))
        mixed = d + b
    return mixed.cpu().numpy().astype(np.float32, copy=False)


def enhance_mix(original: np.ndarray, dialogue: np.ndarray) -> np.ndarray:
    """Bounded-memory adaptive processing with overlap-add for full episodes."""
    if original.shape[0] != 2 or dialogue.shape[0] != 2:
        raise ValueError("v8.1 audio DSP requires stereo input")
    length = min(original.shape[1], dialogue.shape[1])
    if length <= 0:
        raise RuntimeError("Empty audio passed to v8.1 DSP")
    original = original[:, :length]
    dialogue = dialogue[:, :length]
    if not np.isfinite(original).all() or not np.isfinite(dialogue).all():
        raise RuntimeError("Non-finite audio samples detected before v8.1 DSP")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    chunk = 20 * _SAMPLE_RATE
    overlap = 1 * _SAMPLE_RATE
    step = chunk - overlap
    output = np.zeros((2, length), dtype=np.float32)
    weights = np.zeros(length, dtype=np.float32)

    for start in range(0, length, step):
        end = min(length, start + chunk)
        processed = _enhance_chunk(original[:, start:end], dialogue[:, start:end], device)
        n = processed.shape[1]
        weight = np.ones(n, dtype=np.float32)
        fade = min(overlap, n // 2)
        if fade > 0 and start > 0:
            weight[:fade] = np.linspace(0.0, 1.0, fade, endpoint=False).astype(np.float32)
        if fade > 0 and end < length:
            weight[-fade:] = np.linspace(1.0, 0.0, fade, endpoint=False).astype(np.float32)
        output[:, start:end] += processed * weight[None, :]
        weights[start:end] += weight
        if end >= length:
            break

    output /= np.maximum(weights, 1e-6)[None, :]
    peak = float(np.max(np.abs(output))) if output.size else 0.0
    if peak > 0.94:
        output *= np.float32(0.94 / peak)
    if not np.isfinite(output).all():
        raise RuntimeError("v8.1 DSP produced non-finite samples")
    return output
