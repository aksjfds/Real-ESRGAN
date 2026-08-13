"""APISR GRL model adapter for the master video inference runtime.

Only the SR model registry/loading path is changed. Scheduling, shared-memory
transport, BasicVSR++, RIFE, resize, encoding, and audio remain owned by the
master runtime.
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

import numpy as np
import torch
import torch.nn.functional as F


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


def _cache_root() -> Path:
    return Path(__file__).resolve().parent / "weights" / "_apisr_source"


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


def _ensure_apisr_source() -> Path:
    configured = _configured_source_dir()
    if configured is not None:
        return configured

    target = _cache_root() / APISR_SOURCE_COMMIT
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
                "APISR GRL dependency is missing. Install requirements.txt "
                "before inference."
            ) from error
        raise

    grl = getattr(module, "GRL", None)
    if grl is None:
        raise ImportError("APISR architecture.grl does not export GRL")
    return grl


def build_apisr_grl() -> torch.nn.Module:
    """Construct the official APISR 4x GRL paper-model topology."""
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


class APISRGRL(torch.nn.Module):
    """Dimension-preserving wrapper around the official 4x GRL network."""

    sr_input_dtype = torch.float32
    sr_channels_last = False

    def __init__(self, network: torch.nn.Module) -> None:
        super().__init__()
        self.network = network

    @staticmethod
    def _pad(frame: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        height = int(frame.shape[-2])
        width = int(frame.shape[-1])
        pad_h = (-height) % 4
        pad_w = (-width) % 4
        if not pad_h and not pad_w:
            return frame, height, width
        mode = "reflect" if height > 1 and width > 1 else "replicate"
        return F.pad(frame, (0, pad_w, 0, pad_h), mode=mode), height, width

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        outputs: list[torch.Tensor] = []
        for index in range(int(value.shape[0])):
            frame, height, width = self._pad(value[index : index + 1])
            output = self.network(frame)
            outputs.append(
                output[
                    ...,
                    : height * APISR_NATIVE_SCALE,
                    : width * APISR_NATIVE_SCALE,
                ]
            )
        return torch.cat(outputs, dim=0)


def install_apisr_backend(base: ModuleType) -> None:
    """Replace only the master runtime's SR model backend with APISR GRL."""
    global _INSTALLED
    if _INSTALLED:
        return

    base.MODEL_URLS.clear()
    base.MODEL_URLS[APISR_MODEL_NAME] = (APISR_WEIGHT_URL,)
    original_resolve_model_paths = base.resolve_model_paths

    def resolve_model_paths(args):
        if args.model == APISR_MODEL_NAME:
            _ensure_apisr_source()
        return original_resolve_model_paths(args)

    def build_model(name: str) -> tuple[torch.nn.Module, int]:
        if name != APISR_MODEL_NAME:
            raise ValueError(f"Unsupported APISR branch model: {name}")
        return build_apisr_grl(), APISR_NATIVE_SCALE

    def checkpoint_state(path: str) -> Dict[str, torch.Tensor]:
        checkpoint: Dict[str, Any] = base.torch_load_cpu(path)
        state = checkpoint.get("model_state_dict")
        if not isinstance(state, dict):
            raise KeyError(f"No model_state_dict found in APISR checkpoint {path}")
        return _normalize_state(state)

    def model_native_scale(name: str) -> int:
        if name != APISR_MODEL_NAME:
            raise ValueError(f"Unsupported APISR branch model: {name}")
        return APISR_NATIVE_SCALE

    def load_worker_model(config, device: torch.device):
        network, native_scale = build_model(config.model_name)
        network.load_state_dict(checkpoint_state(config.model_paths[0]), strict=True)
        network.eval().requires_grad_(False)
        network.to(device=device, dtype=torch.float32)
        model = APISRGRL(network)
        model.eval().requires_grad_(False)
        model.to(device=device, dtype=torch.float32)
        print(
            f"[APISR] {device} GRL ready | FP32 | native={native_scale}x",
            flush=True,
        )
        return model, native_scale

    def infer_frame(model: torch.nn.Module, frame: np.ndarray, device: torch.device) -> np.ndarray:
        is_10bit = frame.dtype.kind == "u" and frame.dtype.itemsize == 2
        max_value = 65535.0 if is_10bit else 255.0
        tensor = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0)
        tensor = tensor.to(device=device, dtype=torch.float32, non_blocking=True)
        tensor.div_(max_value)
        with torch.inference_mode():
            output = model(tensor)
            output.clamp_(0, 1)
            output = output.mul_(max_value).round_()
        array = output[0].permute(1, 2, 0).contiguous().cpu().numpy()
        return array.astype(np.uint16 if is_10bit else np.uint8)

    base.resolve_model_paths = resolve_model_paths
    base.build_model = build_model
    base.checkpoint_state = checkpoint_state
    base._model_native_scale = model_native_scale
    base.load_worker_model = load_worker_model
    base.infer_frame = infer_frame
    base.APISR_MODEL_NAME = APISR_MODEL_NAME
    base.APISR_SOURCE_COMMIT = APISR_SOURCE_COMMIT
    base.APISR_WEIGHT_URL = APISR_WEIGHT_URL
    _INSTALLED = True


__all__ = [
    "APISR_MODEL_NAME",
    "APISR_NATIVE_SCALE",
    "APISR_SOURCE_COMMIT",
    "APISR_WEIGHT_URL",
    "APISRGRL",
    "build_apisr_grl",
    "install_apisr_backend",
]
