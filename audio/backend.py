"""Isolated Mel-Band RoFormer backend management for v8.1 audio."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Sequence

AUDIO_SEPARATOR_VERSION = "0.44.5"
AUDIO_SEPARATOR_SPEC = f"audio-separator[gpu]=={AUDIO_SEPARATOR_VERSION}"
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


def venv_root() -> Path:
    return cache_root() / "venv"


def venv_python() -> Path:
    root = venv_root()
    return root / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


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


def _onnxruntime_importable(python_bin: Path) -> bool:
    if not python_bin.is_file():
        return False
    result = subprocess.run(
        [str(python_bin), "-c", "import onnxruntime"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def _host_pip_prefix() -> list[str]:
    """Resolve a host pip that can manage the pip-less isolated venv via --python."""
    result = subprocess.run(
        [sys.executable, "-m", "pip", "--version"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if result.returncode == 0:
        return [sys.executable, "-m", "pip"]

    for name in ("pip", "pip3"):
        executable = shutil.which(name)
        if executable:
            return [executable]
    raise RuntimeError(
        "A host pip installation is required to prepare the isolated v8.1 audio environment."
    )


def _ensure_venv() -> Path:
    """Create or recover the isolated environment without invoking ensurepip."""
    root = venv_root()
    python_bin = venv_python()
    config = root / "pyvenv.cfg"

    if python_bin.is_file() and config.is_file():
        return python_bin

    if root.exists():
        print(f"[audio-v8.1] removing incomplete environment: {root}", flush=True)
        shutil.rmtree(root)

    print(f"[audio-v8.1] creating isolated environment: {root}", flush=True)
    _run(
        [
            sys.executable,
            "-m",
            "venv",
            "--without-pip",
            "--system-site-packages",
            str(root),
        ],
        "audio venv creation",
    )
    if not python_bin.is_file() or not config.is_file():
        raise RuntimeError(f"audio venv creation did not produce a usable environment: {root}")
    return python_bin


def prepare_backend() -> None:
    """Create isolated dependencies and download/load the pinned RoFormer model."""
    root = cache_root()
    root.mkdir(parents=True, exist_ok=True)
    python_bin = _ensure_venv()

    version = _installed_version(python_bin)
    runtime_ready = _onnxruntime_importable(python_bin)
    if version != AUDIO_SEPARATOR_VERSION or not runtime_ready:
        reason = (
            f"version={version or 'missing'}"
            if version != AUDIO_SEPARATOR_VERSION
            else "onnxruntime=missing"
        )
        print(
            f"[audio-v8.1] installing {AUDIO_SEPARATOR_SPEC} ({reason})",
            flush=True,
        )
        _run(
            [
                *_host_pip_prefix(),
                "--python",
                str(venv_root()),
                "install",
                "--upgrade",
                "--disable-pip-version-check",
                "--no-input",
                AUDIO_SEPARATOR_SPEC,
            ],
            "audio-separator GPU installation",
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
    validate_backend()


def validate_backend() -> None:
    python_bin = venv_python()
    version = _installed_version(python_bin)
    if version != AUDIO_SEPARATOR_VERSION:
        raise RuntimeError(
            "v8.1 audio backend is not prepared. Run `python -m audio.prepare` "
            f"before inference (expected audio-separator {AUDIO_SEPARATOR_VERSION}, got {version or 'missing'})."
        )
    if not _onnxruntime_importable(python_bin):
        raise RuntimeError(
            "v8.1 audio backend is missing ONNX Runtime. Run `python -m audio.prepare` "
            "to install the audio-separator GPU extra."
        )
    model_path = model_dir() / ROFORMER_MODEL
    if not model_path.is_file() or model_path.stat().st_size == 0:
        raise RuntimeError(
            f"v8.1 RoFormer model cache is missing {ROFORMER_MODEL}; run `python -m audio.prepare`."
        )


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
