"""Isolated Mel-Band RoFormer backend management for v8.1 audio."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Sequence

AUDIO_SEPARATOR_VERSION = "0.44.5"
ROFORMER_MODEL = "vocals_mel_band_roformer.ckpt"


def _run(command: Sequence[str], label: str, *, capture: bool = False) -> subprocess.CompletedProcess:
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


def cache_root() -> Path:
    override = os.environ.get("REALESRGAN_AUDIO_CACHE", "").strip()
    if override:
        root = Path(override).expanduser()
    elif Path("/kaggle/working").is_dir():
        root = Path("/kaggle/working/realesrgan-audio-cache")
    else:
        root = Path.home() / ".cache" / "realesrgan-audio"
    return root.resolve()


def venv_python() -> Path:
    root = cache_root()
    return root / "venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def model_dir() -> Path:
    return cache_root() / "models"


def worker_path() -> Path:
    return Path(__file__).resolve().with_name("separator_worker.py")


def _installed_version(python_bin: Path) -> str:
    if not python_bin.is_file():
        return ""
    result = subprocess.run(
        [
            str(python_bin),
            "-c",
            "import importlib.metadata as m; print(m.version('audio-separator'))",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def prepare_backend() -> None:
    """Create isolated dependencies and download/load the pinned RoFormer model."""
    root = cache_root()
    root.mkdir(parents=True, exist_ok=True)
    python_bin = venv_python()
    if not python_bin.is_file():
        print(f"[audio-v8.1] creating isolated environment: {python_bin.parent.parent}", flush=True)
        _run(
            [sys.executable, "-m", "venv", "--system-site-packages", str(python_bin.parent.parent)],
            "audio venv creation",
        )

    if _installed_version(python_bin) != AUDIO_SEPARATOR_VERSION:
        print(f"[audio-v8.1] installing audio-separator=={AUDIO_SEPARATOR_VERSION}", flush=True)
        _run(
            [
                str(python_bin),
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                f"audio-separator=={AUDIO_SEPARATOR_VERSION}",
            ],
            "audio-separator installation",
        )

    model_dir().mkdir(parents=True, exist_ok=True)
    print(f"[audio-v8.1] preparing Mel-Band RoFormer: {ROFORMER_MODEL}", flush=True)
    _run(
        [
            str(python_bin),
            str(worker_path()),
            "--prepare",
            "--model-dir",
            str(model_dir()),
        ],
        "RoFormer model preparation",
    )


def validate_backend() -> None:
    python_bin = venv_python()
    version = _installed_version(python_bin)
    if version != AUDIO_SEPARATOR_VERSION:
        raise RuntimeError(
            "v8.1 audio backend is not prepared. Run `python -m audio.prepare` "
            f"before inference (expected audio-separator {AUDIO_SEPARATOR_VERSION}, got {version or 'missing'})."
        )
    if not model_dir().is_dir():
        raise RuntimeError("v8.1 RoFormer model cache is missing; run `python -m audio.prepare`.")


def separate_dialogue(input_wav: Path, output_dir: Path) -> Path:
    """Run the isolated separator and return the generated dialogue stem path."""
    validate_backend()
    output_dir.mkdir(parents=True, exist_ok=True)
    result = _run(
        [
            str(venv_python()),
            str(worker_path()),
            "--input",
            str(input_wav),
            "--output-dir",
            str(output_dir),
            "--model-dir",
            str(model_dir()),
        ],
        "Mel-Band RoFormer separation",
        capture=True,
    )
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("RoFormer separation returned no output metadata")
    try:
        payload = json.loads(lines[-1])
    except json.JSONDecodeError as error:
        raise RuntimeError(f"Invalid RoFormer worker output: {lines[-1]}") from error
    path = Path(payload["dialogue"]).resolve()
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"RoFormer dialogue stem was not created: {path}")
    return path
