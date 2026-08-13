"""APISR GRL backend for the master video inference runtime.

This module owns only APISR-specific model/source/weight behavior. Scheduling,
shared-memory transport, BasicVSR++, RIFE, resize, encoding, and audio remain
owned by the master runtime.
"""

from __future__ import annotations

from contextlib import contextmanager
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
from typing import Any, Dict, Iterator

import numpy as np
import torch


APISR_MODEL_NAME = "APISR_GRL"
DEFAULT_MODEL_NAME = APISR_MODEL_NAME
APISR_NATIVE_SCALE = 4
APISR_SOURCE_COMMIT = "fabe8332413bc7f4024e6db39141c68692e88ea5"
APISR_SOURCE_ARCHIVE_URL = (
    f"https://github.com/Kiteretsu77/APISR/archive/{APISR_SOURCE_COMMIT}.zip"
)
APISR_WEIGHT_URL = (
    "https://github.com/Kiteretsu77/APISR/releases/download/"
    "v0.1.0/4x_APISR_GRL_GAN_generator.pth"
)
APISR_WEIGHT_SIZE = 6_479_400
APISR_WEIGHT_SHA256 = "56fff250139563dea59c4ca81af19cc098d94dc3abaad23640f14cec488e5da1"
MODEL_URLS = {APISR_MODEL_NAME: (APISR_WEIGHT_URL,)}

# Git blob IDs from the APISR v0.1.0 source commit. Validate the complete GRL
# runtime source boundary, not only architecture/grl.py.
APISR_SOURCE_BLOBS = {
    "architecture/grl.py": "635842d101eb6ab562257c6181a6e5c012f611e3",
    "architecture/grl_common/__init__.py": "711842e9c7673427001700c318bf446227f5e834",
    "architecture/grl_common/common_edsr.py": "8d0da6e0ad593d97bb1bccb5f7a75a622670322e",
    "architecture/grl_common/mixed_attn_block.py": "d845b7fa67302ec4f566c94aa0a4a0b9b0185f45",
    "architecture/grl_common/mixed_attn_block_efficient.py": "3cb78c23b79281423d6065f72f307b560543bd1c",
    "architecture/grl_common/ops.py": "37406bd8795b61781eaca1d4a854547eff1725a0",
    "architecture/grl_common/resblock.py": "af1999c8d07a99d6aae1fc33bb4fb98670acbf4f",
    "architecture/grl_common/swin_v1_block.py": "26ed1e291de57f29cbeea54af3e8af9b119b7476",
    "architecture/grl_common/swin_v2_block.py": "e62f13704ee2fe5e1674cf6316df8137597688c3",
    "architecture/grl_common/upsample.py": "86155d0efeec42abd1ba12ef263c50357709b625",
}


def _cache_root() -> Path:
    configured = os.environ.get("APISR_CACHE_DIR", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return Path.home() / ".cache" / "realesrgan" / "apisr"


def _git_blob_sha(path: Path) -> str:
    data = path.read_bytes()
    header = f"blob {len(data)}\0".encode("ascii")
    return hashlib.sha1(header + data).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


@contextmanager
def _cache_lock(name: str) -> Iterator[None]:
    """Serialize first-use cache writes; POSIX workers may start concurrently."""
    lock_root = _cache_root() / "locks"
    lock_root.mkdir(parents=True, exist_ok=True)
    handle = (lock_root / f"{name}.lock").open("a+b")
    locked = False
    try:
        if os.name == "posix":
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            locked = True
        yield
    finally:
        if locked:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def _validate_source_dir(path: Path, *, require_pinned: bool) -> Path:
    path = path.expanduser().resolve()
    grl_path = path / "architecture" / "grl.py"
    common_root = path / "architecture" / "grl_common"
    if not grl_path.is_file() or not common_root.is_dir():
        raise FileNotFoundError(
            "APISR source must contain architecture/grl.py and "
            f"architecture/grl_common; received: {path}"
        )
    if require_pinned:
        for relative, expected in APISR_SOURCE_BLOBS.items():
            candidate = path / relative
            if not candidate.is_file():
                raise RuntimeError(f"Pinned APISR source file is missing: {relative}")
            actual = _git_blob_sha(candidate)
            if actual != expected:
                raise RuntimeError(
                    "Cached APISR source failed pinned-release integrity check: "
                    f"{relative}: expected git blob {expected}, got {actual}"
                )
    return path


def _configured_source_dir() -> Path | None:
    value = os.environ.get("APISR_SOURCE_DIR", "").strip()
    if not value:
        return None
    # Explicit local override intentionally permits modified APISR source.
    return _validate_source_dir(Path(value), require_pinned=False)


def _ensure_apisr_source() -> Path:
    configured = _configured_source_dir()
    if configured is not None:
        return configured

    target = _cache_root() / "source" / APISR_SOURCE_COMMIT
    with _cache_lock("source"):
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
                f"[APISR] downloading v0.1.0 source commit {APISR_SOURCE_COMMIT}",
                flush=True,
            )
            urllib.request.urlretrieve(APISR_SOURCE_ARCHIVE_URL, archive)
            with zipfile.ZipFile(archive) as bundle:
                bundle.extractall(temporary_root)

            extracted = temporary_root / f"APISR-{APISR_SOURCE_COMMIT}"
            if not extracted.is_dir():
                candidates = [
                    item
                    for item in temporary_root.iterdir()
                    if item.is_dir() and (item / "architecture" / "grl.py").is_file()
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


def _validate_official_weight(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size != APISR_WEIGHT_SIZE:
        return False
    return _sha256(path) == APISR_WEIGHT_SHA256


def _ensure_official_weight() -> Path:
    target = _cache_root() / "weights" / APISR_WEIGHT_URL.rsplit("/", 1)[-1]
    with _cache_lock("weight"):
        if _validate_official_weight(target):
            return target
        if target.exists():
            target.unlink()

        target.parent.mkdir(parents=True, exist_ok=True)
        handle = tempfile.NamedTemporaryFile(
            prefix=target.name + ".",
            suffix=".part",
            dir=target.parent,
            delete=False,
        )
        temporary = Path(handle.name)
        handle.close()
        try:
            print(f"[APISR] downloading {APISR_WEIGHT_URL}", flush=True)
            urllib.request.urlretrieve(APISR_WEIGHT_URL, temporary)
            if temporary.stat().st_size != APISR_WEIGHT_SIZE:
                raise RuntimeError(
                    "Downloaded APISR weight size mismatch: "
                    f"expected {APISR_WEIGHT_SIZE}, got {temporary.stat().st_size}"
                )
            digest = _sha256(temporary)
            if digest != APISR_WEIGHT_SHA256:
                raise RuntimeError(
                    "Downloaded APISR weight SHA256 mismatch: "
                    f"expected {APISR_WEIGHT_SHA256}, got {digest}"
                )
            temporary.replace(target)
        finally:
            if temporary.exists():
                temporary.unlink()
        return target


def resolve_model_paths(args) -> tuple[str, ...]:
    if args.model != APISR_MODEL_NAME:
        raise ValueError(f"Unsupported APISR branch model: {args.model}")

    # Fail before worker startup if the source boundary cannot be prepared.
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
        locations = [Path(value).resolve() for value in getattr(existing_arch, "__path__", ())]
        if not any(_is_within(location, source_root) for location in locations):
            raise RuntimeError(
                "A conflicting top-level 'architecture' package is already loaded; "
                "cannot safely import the pinned APISR GRL source."
            )

    inserted = source_text not in sys.path
    if inserted:
        sys.path.insert(0, source_text)
    try:
        try:
            module = importlib.import_module("architecture.grl")
        except ModuleNotFoundError as error:
            if error.name in {"fairscale", "omegaconf", "timm"}:
                raise RuntimeError(
                    "APISR GRL dependency is missing. Install requirements.txt "
                    "before inference."
                ) from error
            raise
    finally:
        if inserted:
            try:
                sys.path.remove(source_text)
            except ValueError:
                pass

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
    """Construct the APISR v0.1.0 official 4x GRL GAN topology."""
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
        # GRL owns padding/cropping and supports BCHW batches. Keeping the batch
        # intact preserves the master's SR micro-batch fast path.
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


def warmup_worker_model(
    model: torch.nn.Module,
    device: torch.device,
    input_shape: tuple[int, int, int],
) -> None:
    """Fail fast on full-frame batch=1 VRAM and prebuild the GRL resolution cache."""
    height, width, channels = (int(value) for value in input_shape)
    if channels != 3 or height <= 0 or width <= 0:
        raise ValueError(f"Invalid APISR input shape: {input_shape}")

    sample = None
    output = None
    try:
        sample = torch.zeros(
            (1, channels, height, width),
            dtype=torch.float32,
            device=device,
        )
        with torch.inference_mode():
            output = model(sample)
        expected = (1, 3, height * APISR_NATIVE_SCALE, width * APISR_NATIVE_SCALE)
        if tuple(int(value) for value in output.shape) != expected:
            raise RuntimeError(
                f"APISR warmup output shape mismatch: expected {expected}, "
                f"got {tuple(output.shape)}"
            )
        torch.cuda.current_stream(device).synchronize()
    except torch.cuda.OutOfMemoryError as error:
        torch.cuda.empty_cache()
        raise RuntimeError(
            "APISR GRL full-frame batch=1 warmup ran out of VRAM at "
            f"{width}x{height} on {device}; inference cannot safely start."
        ) from error
    finally:
        del output
        del sample

    # Release transient warmup allocations back to the other persistent GPU
    # process; the GRL resolution cache remains referenced by the model.
    torch.cuda.empty_cache()
    print(
        f"[APISR] {device} full-frame warmup passed | input={width}x{height} | batch=1",
        flush=True,
    )


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
    "APISR_WEIGHT_SHA256",
    "APISR_WEIGHT_URL",
    "APISRGRL",
    "DEFAULT_MODEL_NAME",
    "MODEL_URLS",
    "build_apisr_grl",
    "infer_frame",
    "load_worker_model",
    "model_native_scale",
    "resolve_model_paths",
    "warmup_worker_model",
]
