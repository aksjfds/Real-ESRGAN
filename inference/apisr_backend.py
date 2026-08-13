"""APISR GRL backend for the master video inference runtime.

This module owns only APISR-specific model/source/weight behavior. Scheduling,
shared-memory transport, BasicVSR++, RIFE, resize, encoding, and audio remain
owned by the master runtime.
"""

from __future__ import annotations

from functools import lru_cache
import hashlib
import importlib
import os
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch


APISR_MODEL_NAME = "APISR_GRL"
DEFAULT_MODEL_NAME = APISR_MODEL_NAME
APISR_NATIVE_SCALE = 4
APISR_SOURCE_COMMIT = "c0c0407ba68c0bc5026e43da05f0e7c1cf7b9b95"
APISR_GRL_BLOB_SHA = "41bdfa5949dca89bab1e8c163afed256168b7b05"
APISR_SOURCE_ARCHIVE_URL = (
    f"https://github.com/Kiteretsu77/APISR/archive/{APISR_SOURCE_COMMIT}.zip"
)
APISR_WEIGHT_URL = (
    "https://github.com/Kiteretsu77/APISR/releases/download/"
    "v0.1.0/4x_APISR_GRL_GAN_generator.pth"
)
APISR_WEIGHT_SIZE = 6_479_400
MODEL_URLS = {APISR_MODEL_NAME: (APISR_WEIGHT_URL,)}


def _cache_root() -> Path:
    configured = os.environ.get("APISR_CACHE_DIR", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return Path.home() / ".cache" / "realesrgan" / "apisr"


def _git_blob_sha(path: Path) -> str:
    data = path.read_bytes()
    header = f"blob {len(data)}\0".encode("ascii")
    return hashlib.sha1(header + data).hexdigest()


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _validate_source_dir(path: Path, *, require_pinned: bool) -> Path:
    path = path.expanduser().resolve()
    grl_path = path / "architecture" / "grl.py"
    common_init = path / "architecture" / "grl_common" / "__init__.py"
    if not grl_path.is_file() or not common_init.is_file():
        raise FileNotFoundError(
            "APISR source must contain architecture/grl.py and "
            f"architecture/grl_common; received: {path}"
        )
    if require_pinned:
        actual = _git_blob_sha(grl_path)
        if actual != APISR_GRL_BLOB_SHA:
            raise RuntimeError(
                "Cached APISR GRL source failed pinned-commit integrity check: "
                f"expected git blob {APISR_GRL_BLOB_SHA}, got {actual}"
            )
    return path


def _configured_source_dir() -> Path | None:
    value = os.environ.get("APISR_SOURCE_DIR", "").strip()
    if not value:
        return None
    return _validate_source_dir(Path(value), require_pinned=False)


def _ensure_apisr_source() -> Path:
    configured = _configured_source_dir()
    if configured is not None:
        return configured

    target = _cache_root() / "source" / APISR_SOURCE_COMMIT
    if target.exists():
        try:
            return _validate_source_dir(target, require_pinned=True)
        except (FileNotFoundError, RuntimeError):
            shutil.rmtree(target, ignore_errors=True)

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
        if not extracted.is_dir():
            candidates = [
                path
                for path in temporary_root.iterdir()
                if path.is_dir() and (path / "architecture" / "grl.py").is_file()
            ]
            if len(candidates) != 1:
                raise RuntimeError(
                    "Downloaded APISR archive did not contain one valid source tree"
                )
            extracted = candidates[0]

        _validate_source_dir(extracted, require_pinned=True)
        extracted.replace(target)
    finally:
        shutil.rmtree(temporary_root, ignore_errors=True)

    validated = _validate_source_dir(target, require_pinned=True)
    print(f"[APISR] source cached at {validated}", flush=True)
    return validated


def _ensure_official_weight() -> Path:
    target = _cache_root() / "weights" / APISR_WEIGHT_URL.rsplit("/", 1)[-1]
    if target.is_file() and target.stat().st_size == APISR_WEIGHT_SIZE:
        return target
    if target.exists():
        target.unlink()

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".part")
    try:
        print(f"[APISR] downloading {APISR_WEIGHT_URL}", flush=True)
        urllib.request.urlretrieve(APISR_WEIGHT_URL, temporary)
        actual_size = temporary.stat().st_size
        if actual_size != APISR_WEIGHT_SIZE:
            raise RuntimeError(
                "Downloaded APISR weight size mismatch: "
                f"expected {APISR_WEIGHT_SIZE}, got {actual_size}"
            )
        temporary.replace(target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return target


def resolve_model_paths(args) -> tuple[str, ...]:
    if args.model != APISR_MODEL_NAME:
        raise ValueError(f"Unsupported APISR branch model: {args.model}")

    # Fail before worker startup if the pinned source cannot be prepared.
    _ensure_apisr_source()

    if args.model_path:
        primary = Path(args.model_path).expanduser().resolve()
        if not primary.is_file():
            raise FileNotFoundError(f"Model weight not found: {primary}")
        return (str(primary),)

    return (str(_ensure_official_weight()),)


def _loaded_module_path(module) -> Path | None:
    value = getattr(module, "__file__", None)
    if not value:
        return None
    return Path(value).resolve()


@lru_cache(maxsize=1)
def _load_grl_class() -> type[torch.nn.Module]:
    source_root = _ensure_apisr_source()
    source_text = str(source_root)

    existing_arch = sys.modules.get("architecture")
    if existing_arch is not None:
        locations = [
            Path(value).resolve()
            for value in getattr(existing_arch, "__path__", ())
        ]
        if not any(_is_within(location, source_root) for location in locations):
            raise RuntimeError(
                "A conflicting top-level 'architecture' package is already loaded; "
                "cannot safely import the pinned APISR GRL source."
            )

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

    module_path = _loaded_module_path(module)
    if module_path is None or not _is_within(module_path, source_root):
        raise RuntimeError(
            "Imported architecture.grl does not come from the prepared APISR "
            f"source tree: {module_path!s}"
        )

    grl = getattr(module, "GRL", None)
    if grl is None:
        raise ImportError("APISR architecture.grl does not export GRL")
    return grl


def _cached_grl_class() -> type[torch.nn.Module]:
    grl = _load_grl_class()

    class CachedGRL(grl):  # type: ignore[misc, valid-type]
        """Official GRL with one-entry dynamic-resolution table/mask cache."""

        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self._apisr_cache_key: tuple[int, int, str] | None = None
            self._apisr_cache_value: dict[str, torch.Tensor] | None = None

        def get_table_index_mask(self, device=None, input_resolution=None):
            if input_resolution == self.input_resolution or input_resolution is None:
                return super().get_table_index_mask(device, input_resolution)

            key = (
                int(input_resolution[0]),
                int(input_resolution[1]),
                str(device),
            )
            if self._apisr_cache_key != key or self._apisr_cache_value is None:
                self._apisr_cache_value = super().get_table_index_mask(
                    device,
                    input_resolution,
                )
                self._apisr_cache_key = key
            return self._apisr_cache_value

    CachedGRL.__name__ = "CachedAPISRGRL"
    CachedGRL.__qualname__ = "CachedAPISRGRL"
    return CachedGRL


def build_apisr_grl() -> torch.nn.Module:
    """Construct the official APISR 4x GRL paper-model topology."""
    grl = _cached_grl_class()
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


def _torch_load_cpu(path: str) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _checkpoint_state(path: str) -> Dict[str, torch.Tensor]:
    checkpoint = _torch_load_cpu(path)
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, dict):
        raise KeyError(f"No model_state_dict found in APISR checkpoint {path}")

    prefix = "_orig_mod."
    if state and all(key.startswith(prefix) for key in state):
        return {key[len(prefix) :]: value for key, value in state.items()}
    return state


class APISRGRL(torch.nn.Module):
    """Runtime metadata wrapper around the official GRL network."""

    sr_input_dtype = torch.float32
    sr_channels_last = False

    def __init__(self, network: torch.nn.Module) -> None:
        super().__init__()
        self.network = network

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        # GRL itself owns padding/cropping and supports BCHW batches. Keeping the
        # batch intact preserves the master's SR micro-batch fast path.
        return self.network(value)


def model_native_scale(name: str) -> int:
    if name != APISR_MODEL_NAME:
        raise ValueError(f"Unsupported APISR branch model: {name}")
    return APISR_NATIVE_SCALE


def load_worker_model(config, device: torch.device):
    if config.model_name != APISR_MODEL_NAME:
        raise ValueError(f"Unsupported APISR branch model: {config.model_name}")

    network = build_apisr_grl()
    network.load_state_dict(_checkpoint_state(config.model_paths[0]), strict=True)
    model = APISRGRL(network)
    model.eval().requires_grad_(False)
    model.to(device=device, dtype=torch.float32)
    print(
        f"[APISR] {device} GRL ready | FP32 | native={APISR_NATIVE_SCALE}x",
        flush=True,
    )
    return model, APISR_NATIVE_SCALE


def infer_frame(
    model: torch.nn.Module,
    frame: np.ndarray,
    device: torch.device,
) -> np.ndarray:
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


__all__ = [
    "APISR_MODEL_NAME",
    "APISR_NATIVE_SCALE",
    "APISR_SOURCE_COMMIT",
    "APISR_WEIGHT_URL",
    "APISRGRL",
    "DEFAULT_MODEL_NAME",
    "MODEL_URLS",
    "build_apisr_grl",
    "infer_frame",
    "load_worker_model",
    "model_native_scale",
    "resolve_model_paths",
]
