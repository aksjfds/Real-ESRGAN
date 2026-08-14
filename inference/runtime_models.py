"""Model loading and per-frame inference primitives."""

from __future__ import annotations

import sys
import types
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch

try:  # pragma: no cover - depends on installed torchvision
    import torchvision.transforms.functional_tensor  # noqa: F401
except (ImportError, ModuleNotFoundError):  # pragma: no cover
    import torchvision.transforms.functional as _tv_functional

    _functional_tensor = types.ModuleType("torchvision.transforms.functional_tensor")
    _functional_tensor.rgb_to_grayscale = _tv_functional.rgb_to_grayscale
    sys.modules["torchvision.transforms.functional_tensor"] = _functional_tensor

from basicsr.archs.rrdbnet_arch import RRDBNet
from .models.srvgg_arch import SRVGGNetCompact


MODEL_URLS = {
    "RealESRGAN_x4plus": (
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
    ),
    "RealESRNet_x4plus": (
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth",
    ),
    "RealESRGAN_x4plus_anime_6B": (
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
    ),
    "RealESRGAN_x2plus": (
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
    ),
    "realesr-animevideov3": (
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth",
    ),
    "realesr-general-x4v3": (
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
    ),
}


@dataclass(frozen=True)
class WorkerConfig:
    model_name: str
    model_paths: Tuple[str, ...]


def download_file(url: str, target: Path) -> Path:
    if target.is_file() and target.stat().st_size > 0:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".part")
    print(f"[model] downloading {url}", flush=True)
    try:
        urllib.request.urlretrieve(url, temporary)
        temporary.replace(target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return target


def resolve_model_paths(args) -> Tuple[str, ...]:
    if args.model_path:
        primary = Path(args.model_path).expanduser().resolve()
        if not primary.is_file():
            raise FileNotFoundError(f"Model weight not found: {primary}")
        return (str(primary),)
    urls = MODEL_URLS[args.model]
    weight_dir = Path(__file__).resolve().parent / "weights"
    return tuple(
        str(download_file(url, weight_dir / url.rsplit("/", 1)[-1]))
        for url in urls
    )


def build_model(name: str) -> tuple[torch.nn.Module, int]:
    if name == "RealESRGAN_x4plus":
        return RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4), 4
    if name == "RealESRNet_x4plus":
        return RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4), 4
    if name == "RealESRGAN_x4plus_anime_6B":
        return RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4), 4
    if name == "RealESRGAN_x2plus":
        return RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2), 2
    if name == "realesr-animevideov3":
        return SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type="prelu"), 4
    if name == "realesr-general-x4v3":
        return SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type="prelu"), 4
    raise ValueError(f"Unsupported model: {name}")


def model_native_scale(name: str) -> int:
    return 2 if name == "RealESRGAN_x2plus" else 4


def torch_load_cpu(path: str) -> Dict[str, object]:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def checkpoint_state(path: str) -> Dict[str, torch.Tensor]:
    checkpoint = torch_load_cpu(path)
    if "params_ema" in checkpoint:
        return checkpoint["params_ema"]  # type: ignore[return-value]
    if "params" in checkpoint:
        return checkpoint["params"]  # type: ignore[return-value]
    raise KeyError(f"No params or params_ema found in {path}")


def load_worker_model(config: WorkerConfig, device: torch.device) -> tuple[torch.nn.Module, int]:
    model, native_scale = build_model(config.model_name)
    model.load_state_dict(checkpoint_state(config.model_paths[0]), strict=True)
    model.eval().requires_grad_(False)
    if device.type == "cuda":
        model.half()
        model.to(device=device, memory_format=torch.channels_last)
    else:
        model.to(device)
    return model, native_scale


def infer_frame(model: torch.nn.Module, frame: np.ndarray, device: torch.device) -> np.ndarray:
    is_10bit = frame.dtype.kind == "u" and frame.dtype.itemsize == 2
    if is_10bit:
        tensor = torch.from_numpy(frame.astype(np.float32, copy=False)).permute(2, 0, 1).unsqueeze(0).to(device, non_blocking=True)
        tensor.div_(65535.0)
        if device.type == "cuda":
            tensor = tensor.half()
    else:
        tensor = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).to(device, non_blocking=True)
        tensor = tensor.half() if device.type == "cuda" else tensor.float()
        tensor.div_(255.0)
    if device.type == "cuda":
        tensor = tensor.contiguous(memory_format=torch.channels_last)
    with torch.inference_mode():
        output = model(tensor)
        output.clamp_(0, 1)
        output = output.float().mul_(65535.0).round_() if is_10bit else output.mul_(255.0).round_().byte()
    array = output[0].permute(1, 2, 0).contiguous().cpu().numpy()
    return array.astype(np.uint16) if is_10bit else array
