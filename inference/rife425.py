"""Practical-RIFE 4.25 inference-only adapter for arbitrary-timestep interpolation.

Architecture adapted from hzwer/Practical-RIFE (MIT). The official 4.25 archive
is downloaded on first use and only flownet.pkl is consumed.
"""
from __future__ import annotations

import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

RIFE425_FILE_ID = "1ZKjcbmt1hypiFprJPIKW0Tt0lr_2i7bg"
RIFE425_CACHE = Path.home() / ".cache" / "realesrgan" / "rife-v4.25"
_WARP_GRID: dict[tuple[str, str, int, int, int], torch.Tensor] = {}


def _warp(x: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    n, _c, h, w = flow.shape
    key = (str(flow.device), str(flow.dtype), n, h, w)
    grid = _WARP_GRID.get(key)
    if grid is None:
        horizontal = torch.linspace(-1.0, 1.0, w, device=flow.device, dtype=flow.dtype)
        vertical = torch.linspace(-1.0, 1.0, h, device=flow.device, dtype=flow.dtype)
        horizontal = horizontal.view(1, 1, 1, w).expand(n, -1, h, -1)
        vertical = vertical.view(1, 1, h, 1).expand(n, -1, -1, w)
        grid = torch.cat((horizontal, vertical), 1)
        _WARP_GRID[key] = grid
    normalized = torch.cat(
        (
            flow[:, 0:1] / max((x.shape[3] - 1.0) / 2.0, 1.0),
            flow[:, 1:2] / max((x.shape[2] - 1.0) / 2.0, 1.0),
        ),
        1,
    )
    sample_grid = (grid + normalized).permute(0, 2, 3, 1)
    return F.grid_sample(
        x, sample_grid, mode="bilinear", padding_mode="border", align_corners=True
    )


def _conv(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, 2, 1), nn.LeakyReLU(0.2, True))


class _Head(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.cnn0 = nn.Conv2d(3, 16, 3, 2, 1)
        self.cnn1 = nn.Conv2d(16, 16, 3, 1, 1)
        self.cnn2 = nn.Conv2d(16, 16, 3, 1, 1)
        self.cnn3 = nn.ConvTranspose2d(16, 4, 4, 2, 1)
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.cnn0(x))
        x = self.relu(self.cnn1(x))
        x = self.relu(self.cnn2(x))
        return self.cnn3(x)


class _ResConv(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, 1, 1)
        self.beta = nn.Parameter(torch.ones((1, channels, 1, 1)))
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.conv(x) * self.beta + x)


class _IFBlock(nn.Module):
    def __init__(self, in_planes: int, channels: int) -> None:
        super().__init__()
        self.conv0 = nn.Sequential(_conv(in_planes, channels // 2), _conv(channels // 2, channels))
        self.convblock = nn.Sequential(*[_ResConv(channels) for _ in range(8)])
        self.lastconv = nn.Sequential(
            nn.ConvTranspose2d(channels, 4 * 13, 4, 2, 1),
            nn.PixelShuffle(2),
        )

    def forward(
        self, x: torch.Tensor, flow: torch.Tensor | None, scale: float
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = F.interpolate(x, scale_factor=1.0 / scale, mode="bilinear", align_corners=False)
        if flow is not None:
            flow = F.interpolate(
                flow, scale_factor=1.0 / scale, mode="bilinear", align_corners=False
            ) * (1.0 / scale)
            x = torch.cat((x, flow), 1)
        feat = self.convblock(self.conv0(x))
        tmp = self.lastconv(feat)
        tmp = F.interpolate(tmp, scale_factor=scale, mode="bilinear", align_corners=False)
        return tmp[:, :4] * scale, tmp[:, 4:5], tmp[:, 5:]


class _IFNet425(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block0 = _IFBlock(15, 192)
        self.block1 = _IFBlock(28, 128)
        self.block2 = _IFBlock(28, 96)
        self.block3 = _IFBlock(28, 64)
        self.block4 = _IFBlock(28, 32)
        self.encode = _Head()

    def forward(self, x: torch.Tensor, timestep: float) -> torch.Tensor:
        channel = x.shape[1] // 2
        img0, img1 = x[:, :channel], x[:, channel:]
        timestep_map = (x[:, :1] * 0 + 1) * float(timestep)
        f0, f1 = self.encode(img0[:, :3]), self.encode(img1[:, :3])
        warped0, warped1 = img0, img1
        flow = mask = feat = None
        blocks = (self.block0, self.block1, self.block2, self.block3, self.block4)
        for block, scale in zip(blocks, (16.0, 8.0, 4.0, 2.0, 1.0)):
            if flow is None:
                flow, mask, feat = block(
                    torch.cat((img0[:, :3], img1[:, :3], f0, f1, timestep_map), 1),
                    None,
                    scale,
                )
            else:
                wf0 = _warp(f0, flow[:, :2])
                wf1 = _warp(f1, flow[:, 2:4])
                delta, mask, feat = block(
                    torch.cat((warped0[:, :3], warped1[:, :3], wf0, wf1, timestep_map, mask, feat), 1),
                    flow,
                    scale,
                )
                flow = flow + delta
            warped0 = _warp(img0, flow[:, :2])
            warped1 = _warp(img1, flow[:, 2:4])
        blend = torch.sigmoid(mask)
        return warped0 * blend + warped1 * (1.0 - blend)


def resolve_rife425_weights() -> Path:
    target = RIFE425_CACHE / "flownet.pkl"
    if target.is_file() and target.stat().st_size > 10_000_000:
        return target
    RIFE425_CACHE.mkdir(parents=True, exist_ok=True)
    archive = RIFE425_CACHE / "RIFEv4.25_0919.zip"
    if not archive.is_file() or archive.stat().st_size < 10_000_000:
        try:
            import gdown
        except ImportError as error:
            raise RuntimeError("RIFE 4.25 requires gdown; install repository requirements.txt") from error
        print("[rife] downloading official Practical-RIFE 4.25 model archive", flush=True)
        downloaded = gdown.download(id=RIFE425_FILE_ID, output=str(archive), quiet=False)
        if not downloaded or not archive.is_file():
            raise RuntimeError("Failed to download official Practical-RIFE 4.25 archive")
    with tempfile.TemporaryDirectory(prefix="rife425-") as temp:
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(temp)
        matches = list(Path(temp).rglob("flownet.pkl"))
        if not matches:
            raise RuntimeError("RIFE 4.25 archive does not contain flownet.pkl")
        shutil.copy2(matches[0], target)
    return target


def _load_state(path: Path) -> tuple[dict[str, torch.Tensor], int]:
    """Load only inference IFNet weights and drop training-only teacher tensors."""
    try:
        obj = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        obj = torch.load(path, map_location="cpu")
    if not isinstance(obj, dict):
        raise TypeError("Unexpected RIFE checkpoint object")

    state: dict[str, torch.Tensor] = {}
    ignored_teacher = 0
    for raw_key, value in obj.items():
        if not isinstance(value, torch.Tensor):
            continue
        key = str(raw_key).removeprefix("module.")
        if key.startswith("teacher."):
            ignored_teacher += 1
            continue
        state[key] = value

    if not state:
        raise RuntimeError("RIFE 4.25 checkpoint contains no inference weights")
    return state, ignored_teacher


class RIFE425Interpolator:
    def __init__(self, gpu_id: int, weights: Path) -> None:
        self.gpu_id = int(gpu_id)
        self.device = torch.device(f"cuda:{self.gpu_id}")
        torch.cuda.set_device(self.gpu_id)
        self.model = _IFNet425().eval().requires_grad_(False)
        state, ignored_teacher = _load_state(weights)
        try:
            self.model.load_state_dict(state, strict=True)
        except RuntimeError as error:
            raise RuntimeError(
                "RIFE 4.25 inference checkpoint mismatch after filtering training-only teacher.* weights"
            ) from error
        self.dtype = torch.float16
        self.model.to(self.device, dtype=self.dtype)
        self.elapsed = 0.0
        self.frames = 0
        suffix = f" | ignored_teacher={ignored_teacher}" if ignored_teacher else ""
        print(
            f"[rife] Practical-RIFE 4.25 loaded on cuda:{self.gpu_id} (FP16){suffix}",
            flush=True,
        )

    @staticmethod
    def _to_float(frame: np.ndarray) -> tuple[np.ndarray, float]:
        if frame.dtype == np.uint8:
            return frame.astype(np.float32) / 255.0, 255.0
        if frame.dtype.kind == "u" and frame.dtype.itemsize == 2:
            return frame.astype(np.float32) / 65535.0, 65535.0
        raise TypeError(f"RIFE supports uint8/uint16 frames, got {frame.dtype}")

    def interpolate_many(
        self, frame0: np.ndarray, frame1: np.ndarray, timesteps: Sequence[float]
    ) -> list[np.ndarray]:
        if not timesteps:
            return []
        if frame0.shape != frame1.shape or frame0.dtype != frame1.dtype:
            raise ValueError("RIFE frame pairs must have identical shape/dtype")
        import time
        started = time.monotonic()
        torch.cuda.set_device(self.gpu_id)
        a, scale_value = self._to_float(frame0)
        b, _ = self._to_float(frame1)
        h, w = frame0.shape[:2]
        ph = ((h - 1) // 128 + 1) * 128
        pw = ((w - 1) // 128 + 1) * 128
        ta = torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0).to(
            self.device, dtype=self.dtype, non_blocking=True
        )
        tb = torch.from_numpy(b).permute(2, 0, 1).unsqueeze(0).to(
            self.device, dtype=self.dtype, non_blocking=True
        )
        if ph != h or pw != w:
            ta = F.pad(ta, (0, pw - w, 0, ph - h))
            tb = F.pad(tb, (0, pw - w, 0, ph - h))
        outputs: list[np.ndarray] = []
        with torch.inference_mode():
            pair = torch.cat((ta, tb), 1)
            for timestep in timesteps:
                out = self.model(pair, float(timestep))[0, :, :h, :w]
                out = out.clamp_(0.0, 1.0).mul_(scale_value).round()
                dtype = torch.uint8 if frame0.dtype == np.uint8 else torch.int32
                arr = out.to(dtype).permute(1, 2, 0).contiguous().cpu().numpy()
                if frame0.dtype != np.uint8:
                    arr = arr.astype(frame0.dtype, copy=False)
                outputs.append(np.ascontiguousarray(arr))
        self.elapsed += time.monotonic() - started
        self.frames += len(outputs)
        return outputs

    def close(self) -> None:
        if hasattr(self, "model"):
            del self.model
        torch.cuda.empty_cache()
