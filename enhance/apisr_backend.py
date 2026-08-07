"""APISR GRL integration for the existing AutoDL video pipeline.

The video decoder, shared-memory transport, auto tile/batch selection, stitching,
color handling and encoder stay unchanged. This module only swaps the SR model
registry/checkpoint loader to the official APISR 4x GRL GAN model.
"""

from __future__ import annotations

import importlib
import os
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from types import ModuleType
from typing import Any, Dict

import torch


APISR_MODEL_NAME = "APISR_GRL"
APISR_NATIVE_SCALE = 4
APISR_SOURCE_COMMIT = "c0c0407ba68c0bc5026e43da05f0e7c1cf7b9b95"
APISR_SOURCE_ARCHIVE_URL = (
    f"https://github.com/Kiteretsu77/APISR/archive/{APISR_SOURCE_COMMIT}.zip"
)
APISR_WEIGHT_URL = (
    "https://github.com/Kiteretsu77/APISR/releases/download/"
    "v0.1.0/4x_APISR_GRL_GAN_generator.pth"
)

_INSTALLED = False


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _configured_source_dir() -> Path | None:
    value = os.environ.get("APISR_SOURCE_DIR", "").strip()
    if not value:
        return None
    path = Path(value).expanduser().resolve()
    if not (path / "architecture" / "grl.py").is_file():
        raise FileNotFoundError(
            "APISR_SOURCE_DIR must point at an APISR checkout containing "
            f"architecture/grl.py; received: {path}"
        )
    return path


def _cached_source_dir() -> Path:
    return (
        _repository_root()
        / "weights"
        / "_apisr_source"
        / APISR_SOURCE_COMMIT
    )


def _ensure_apisr_source() -> Path:
    configured = _configured_source_dir()
    if configured is not None:
        return configured

    target = _cached_source_dir()
    if (target / "architecture" / "grl.py").is_file():
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(
        tempfile.mkdtemp(prefix="apisr-source-", dir=str(target.parent))
    )
    archive = temporary_root / "apisr.zip"
    try:
        print(
            f"[APISR] downloading source commit {APISR_SOURCE_COMMIT}",
            flush=True,
        )
        urllib.request.urlretrieve(APISR_SOURCE_ARCHIVE_URL, archive)
        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(temporary_root)

        extracted = temporary_root / f"APISR-{APISR_SOURCE_COMMIT}"
        if not (extracted / "architecture" / "grl.py").is_file():
            candidates = [
                path
                for path in temporary_root.iterdir()
                if path.is_dir() and (path / "architecture" / "grl.py").is_file()
            ]
            if len(candidates) != 1:
                raise RuntimeError(
                    "Downloaded APISR archive did not contain architecture/grl.py"
                )
            extracted = candidates[0]

        if target.exists():
            shutil.rmtree(target)
        extracted.replace(target)
    except Exception:
        if target.exists() and not (target / "architecture" / "grl.py").is_file():
            shutil.rmtree(target, ignore_errors=True)
        raise
    finally:
        shutil.rmtree(temporary_root, ignore_errors=True)

    print(f"[APISR] source cached at {target}", flush=True)
    return target


def _load_grl_class() -> type[torch.nn.Module]:
    source_root = _ensure_apisr_source()
    source_text = str(source_root)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)

    try:
        module = importlib.import_module("architecture.grl")
    except ModuleNotFoundError as error:
        if error.name in {"fairscale", "omegaconf", "timm"}:
            raise RuntimeError(
                "APISR GRL dependency is missing. Install this branch's "
                "requirements.txt before inference."
            ) from error
        raise

    grl = getattr(module, "GRL", None)
    if grl is None:
        raise ImportError("APISR architecture.grl does not export GRL")
    return grl


def build_apisr_grl() -> torch.nn.Module:
    """Construct the exact 4x paper GRL topology used by APISR's loader."""
    grl = _load_grl_class()
    return grl(
        upscale=APISR_NATIVE_SCALE,
        img_size=64,
        window_size=8,
        depths=[4, 4, 4, 4],
        embed_dim=64,
        num_heads_window=[2, 2, 2, 2],
        num_heads_stripe=[2, 2, 2, 2],
        mlp_ratio=2,
        qkv_proj_type="linear",
        anchor_proj_type="avgpool",
        anchor_window_down_factor=2,
        out_proj_type="linear",
        conv_type="1conv",
        upsampler="nearest+conv",
    )


def _normalize_state(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    prefix = "_orig_mod."
    if state and all(key.startswith(prefix) for key in state):
        return {key[len(prefix) :]: value for key, value in state.items()}
    return state


def install_apisr_backend(base: ModuleType) -> None:
    """Patch the loaded core module in place before fast workers are spawned."""
    global _INSTALLED
    if _INSTALLED:
        return

    base.MODEL_URLS[APISR_MODEL_NAME] = (APISR_WEIGHT_URL,)

    original_build_model = base.build_model
    original_checkpoint_state = base.checkpoint_state
    original_build_parser = base.build_parser
    original_validate_args = base.validate_args
    original_process_video = base.process_video

    def build_model(name: str) -> tuple[torch.nn.Module, int]:
        if name == APISR_MODEL_NAME:
            return build_apisr_grl(), APISR_NATIVE_SCALE
        return original_build_model(name)

    def checkpoint_state(path: str) -> Dict[str, torch.Tensor]:
        checkpoint: Dict[str, Any] = base.torch_load_cpu(path)
        state = checkpoint.get("model_state_dict")
        if isinstance(state, dict):
            return _normalize_state(state)
        return original_checkpoint_state(path)

    def build_parser():
        parser = original_build_parser()
        parser.description = (
            "AutoDL video enhancement using the official APISR 4x GRL GAN model."
        )
        for action in parser._actions:
            if action.dest == "model":
                action.default = APISR_MODEL_NAME
                action.help = (
                    "SR model. This APISR branch defaults to APISR_GRL "
                    "(official 4x paper GAN weight)."
                )
        return parser

    def validate_args(args) -> None:
        original_validate_args(args)
        if args.model == APISR_MODEL_NAME and float(args.denoise_strength) != 1.0:
            raise ValueError(
                "--denoise-strength is a Real-ESRGAN DNI option and is not "
                "supported by APISR_GRL; use 1.0."
            )

    def process_video(args) -> None:
        if args.model == APISR_MODEL_NAME:
            if bool(args.fp16):
                print(
                    "[APISR] GRL FP16 is disabled because the official APISR "
                    "implementation documents FP16 issues; using FP32.",
                    flush=True,
                )
                args.fp16 = False
            if bool(args.channels_last):
                print(
                    "[APISR] disabling channels-last for GRL compatibility.",
                    flush=True,
                )
                args.channels_last = False
            # GRL is substantially more memory hungry than the original
            # compact Real-ESRGAN model. Let the AutoDL tuner fall back to
            # 256px tiles and use a single pipeline slot so OOM recovery has a
            # genuinely conservative configuration available.
            runtime = sys.modules.get("enhance.autodl_runtime")
            if runtime is not None:
                if hasattr(runtime, "_MIN_QUALITY_TILE"):
                    runtime._MIN_QUALITY_TILE = 256
                if hasattr(runtime, "_PIPELINE_DEPTH"):
                    runtime._PIPELINE_DEPTH = 1
            _ensure_apisr_source()
        original_process_video(args)

    base.build_model = build_model
    base.checkpoint_state = checkpoint_state
    base.build_parser = build_parser
    base.validate_args = validate_args
    base.process_video = process_video
    base.APISR_MODEL_NAME = APISR_MODEL_NAME
    base.APISR_SOURCE_COMMIT = APISR_SOURCE_COMMIT
    base.APISR_WEIGHT_URL = APISR_WEIGHT_URL
    _INSTALLED = True


__all__ = [
    "APISR_MODEL_NAME",
    "APISR_NATIVE_SCALE",
    "APISR_SOURCE_COMMIT",
    "APISR_WEIGHT_URL",
    "build_apisr_grl",
    "install_apisr_backend",
]
