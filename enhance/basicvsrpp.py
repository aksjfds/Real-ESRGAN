# SPDX-License-Identifier: Apache-2.0
"""BasicVSR++ compressed-video enhancement preprocessor.

This is a dependency-light inference port of OpenMMLab MMagic v1.2.0's
``BasicVSRPlusPlusNet``.  The network topology and parameter names are kept
checkpoint-compatible with the official NTIRE 2021 compressed-video models,
while the DCNv2 operator is provided by ``torchvision.ops.deform_conv2d`` so a
separately compiled MMCV extension is not required.

Official source/config references:
- MMagic v1.2.0 BasicVSR++ implementation
- basicvsr-pp_c128n25_600k_ntire-decompress-track1.py

Internal contract:
- input/output frames: RGB float32 in [0, 1]
- model input/output: N,T,C,H,W
- NTIRE decompression checkpoints use same-resolution output
"""

from __future__ import annotations

import math
import time
import urllib.request
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Optional, Protocol, Sequence

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import deform_conv2d


BASICVSRPP_TRACK_URLS: dict[int, str] = {
    1: (
        "https://download.openmmlab.com/mmediting/restorers/basicvsr_plusplus/"
        "basicvsr_plusplus_c128n25_ntire_decompress_track1_20210223-7b2eba02.pth"
    ),
    2: (
        "https://download.openmmlab.com/mmediting/restorers/basicvsr_plusplus/"
        "basicvsr_plusplus_c128n25_ntire_decompress_track2_20210314-eeae05e6.pth"
    ),
    3: (
        "https://download.openmmlab.com/mmediting/restorers/basicvsr_plusplus/"
        "basicvsr_plusplus_c128n25_ntire_decompress_track3_20210304-6daf4a40.pth"
    ),
}


class FrameReader(Protocol):
    def read(self) -> Optional[np.ndarray]: ...

    def close(self) -> None: ...


def _pair(value: int | tuple[int, int]) -> tuple[int, int]:
    return value if isinstance(value, tuple) else (value, value)


def flow_warp(
    x: torch.Tensor,
    flow: torch.Tensor,
    interpolation: str = "bilinear",
    padding_mode: str = "zeros",
    align_corners: bool = True,
) -> torch.Tensor:
    """Warp ``x`` using pixel-space flow in ``(x, y)`` order."""
    if x.shape[-2:] != flow.shape[1:3]:
        raise ValueError(f"flow size {flow.shape[1:3]} does not match feature size {x.shape[-2:]}")
    n, _c, h, w = x.shape
    grid_y, grid_x = torch.meshgrid(
        torch.arange(h, device=x.device, dtype=x.dtype),
        torch.arange(w, device=x.device, dtype=x.dtype),
        indexing="ij",
    )
    grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0).expand(n, -1, -1, -1)
    grid = grid + flow.to(dtype=x.dtype)
    if w > 1:
        grid_x_norm = 2.0 * grid[..., 0] / (w - 1) - 1.0
    else:
        grid_x_norm = torch.zeros_like(grid[..., 0])
    if h > 1:
        grid_y_norm = 2.0 * grid[..., 1] / (h - 1) - 1.0
    else:
        grid_y_norm = torch.zeros_like(grid[..., 1])
    normalized = torch.stack((grid_x_norm, grid_y_norm), dim=-1)
    return F.grid_sample(
        x,
        normalized,
        mode=interpolation,
        padding_mode=padding_mode,
        align_corners=align_corners,
    )


class ResidualBlockNoBN(nn.Module):
    def __init__(self, mid_channels: int = 64, residual_scale: float = 1.0):
        super().__init__()
        self.residual_scale = residual_scale
        self.conv1 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.conv2(self.relu(self.conv1(x)))
        return x + residual * self.residual_scale


def make_layer(block: type[nn.Module], num_blocks: int, **kwargs: object) -> nn.Sequential:
    return nn.Sequential(*(block(**kwargs) for _ in range(num_blocks)))


class PixelShufflePack(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        scale_factor: int,
        upsample_kernel: int,
    ):
        super().__init__()
        self.scale_factor = scale_factor
        self.upsample_conv = nn.Conv2d(
            in_channels,
            out_channels * scale_factor * scale_factor,
            upsample_kernel,
            padding=(upsample_kernel - 1) // 2,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.pixel_shuffle(self.upsample_conv(x), self.scale_factor)


class ResidualBlocksWithInputConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = 64, num_blocks: int = 30):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            make_layer(ResidualBlockNoBN, num_blocks, mid_channels=out_channels),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.main(feat)


class ConvModuleLite(nn.Module):
    """State-dict-compatible subset of MMCV ConvModule used by SPyNet."""

    def __init__(self, in_channels: int, out_channels: int, activate: bool):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 7, 1, 3, bias=True)
        self.activate = nn.ReLU(inplace=True) if activate else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activate(self.conv(x))


class SPyNetBasicModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.basic_module = nn.Sequential(
            ConvModuleLite(8, 32, True),
            ConvModuleLite(32, 64, True),
            ConvModuleLite(64, 32, True),
            ConvModuleLite(32, 16, True),
            ConvModuleLite(16, 2, False),
        )

    def forward(self, tensor_input: torch.Tensor) -> torch.Tensor:
        return self.basic_module(tensor_input)


class SPyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.basic_module = nn.ModuleList([SPyNetBasicModule() for _ in range(6)])
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def compute_flow(self, ref: torch.Tensor, supp: torch.Tensor) -> torch.Tensor:
        n, _c, h, w = ref.shape
        ref_pyramid = [(ref - self.mean) / self.std]
        supp_pyramid = [(supp - self.mean) / self.std]
        for _ in range(5):
            ref_pyramid.append(F.avg_pool2d(ref_pyramid[-1], 2, 2, count_include_pad=False))
            supp_pyramid.append(F.avg_pool2d(supp_pyramid[-1], 2, 2, count_include_pad=False))
        ref_pyramid.reverse()
        supp_pyramid.reverse()
        flow = ref.new_zeros(n, 2, h // 32, w // 32)
        for level, (ref_level, supp_level) in enumerate(zip(ref_pyramid, supp_pyramid)):
            if level == 0:
                flow_up = flow
            else:
                flow_up = F.interpolate(flow, scale_factor=2, mode="bilinear", align_corners=True) * 2.0
            warped = flow_warp(
                supp_level,
                flow_up.permute(0, 2, 3, 1),
                padding_mode="border",
            )
            flow = flow_up + self.basic_module[level](torch.cat((ref_level, warped, flow_up), dim=1))
        return flow

    def forward(self, ref: torch.Tensor, supp: torch.Tensor) -> torch.Tensor:
        h, w = ref.shape[-2:]
        h_up = h if h % 32 == 0 else 32 * (h // 32 + 1)
        w_up = w if w % 32 == 0 else 32 * (w // 32 + 1)
        ref_up = F.interpolate(ref, size=(h_up, w_up), mode="bilinear", align_corners=False)
        supp_up = F.interpolate(supp, size=(h_up, w_up), mode="bilinear", align_corners=False)
        flow = F.interpolate(
            self.compute_flow(ref_up, supp_up),
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        )
        flow[:, 0] *= float(w) / float(w_up)
        flow[:, 1] *= float(h) / float(h_up)
        return flow


class SecondOrderDeformableAlignment(nn.Module):
    """MMagic-compatible second-order DCNv2 alignment using torchvision."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        deform_groups: int = 1,
        bias: bool = True,
        max_residue_magnitude: int = 10,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.deform_groups = deform_groups
        self.max_residue_magnitude = max_residue_magnitude
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels // groups, *self.kernel_size)
        )
        self.bias = nn.Parameter(torch.empty(out_channels)) if bias else None
        self.conv_offset = nn.Sequential(
            nn.Conv2d(3 * out_channels + 4, out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(out_channels, 27 * deform_groups, 3, 1, 1),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_channels * self.kernel_size[0] * self.kernel_size[1] // self.groups
            bound = 1.0 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        nn.init.zeros_(self.conv_offset[-1].weight)
        nn.init.zeros_(self.conv_offset[-1].bias)

    def forward(
        self,
        x: torch.Tensor,
        extra_feat: torch.Tensor,
        flow_1: torch.Tensor,
        flow_2: torch.Tensor,
    ) -> torch.Tensor:
        extra_feat = torch.cat((extra_feat, flow_1, flow_2), dim=1)
        out = self.conv_offset(extra_feat)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = self.max_residue_magnitude * torch.tanh(torch.cat((o1, o2), dim=1))
        offset_1, offset_2 = torch.chunk(offset, 2, dim=1)
        offset_1 = offset_1 + flow_1.flip(1).repeat(1, offset_1.size(1) // 2, 1, 1)
        offset_2 = offset_2 + flow_2.flip(1).repeat(1, offset_2.size(1) // 2, 1, 1)
        offset = torch.cat((offset_1, offset_2), dim=1)
        mask = torch.sigmoid(mask)
        return deform_conv2d(
            x,
            offset,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            mask=mask,
        )


class BasicVSRPlusPlusNet(nn.Module):
    """Official BasicVSR++ topology with same-resolution NTIRE output."""

    def __init__(
        self,
        mid_channels: int = 128,
        num_blocks: int = 25,
        max_residue_magnitude: int = 10,
        is_low_res_input: bool = False,
        cpu_cache_length: int = 100,
    ):
        super().__init__()
        self.mid_channels = mid_channels
        self.is_low_res_input = is_low_res_input
        self.cpu_cache_length = cpu_cache_length
        self.spynet = SPyNet()
        if is_low_res_input:
            self.feat_extract = ResidualBlocksWithInputConv(3, mid_channels, 5)
        else:
            self.feat_extract = nn.Sequential(
                nn.Conv2d(3, mid_channels, 3, 2, 1),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                nn.Conv2d(mid_channels, mid_channels, 3, 2, 1),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                ResidualBlocksWithInputConv(mid_channels, mid_channels, 5),
            )
        self.deform_align = nn.ModuleDict()
        self.backbone = nn.ModuleDict()
        modules = ("backward_1", "forward_1", "backward_2", "forward_2")
        for index, module_name in enumerate(modules):
            self.deform_align[module_name] = SecondOrderDeformableAlignment(
                2 * mid_channels,
                mid_channels,
                3,
                padding=1,
                deform_groups=16,
                max_residue_magnitude=max_residue_magnitude,
            )
            self.backbone[module_name] = ResidualBlocksWithInputConv(
                (2 + index) * mid_channels,
                mid_channels,
                num_blocks,
            )
        self.reconstruction = ResidualBlocksWithInputConv(5 * mid_channels, mid_channels, 5)
        self.upsample1 = PixelShufflePack(mid_channels, mid_channels, 2, upsample_kernel=3)
        self.upsample2 = PixelShufflePack(mid_channels, 64, 2, upsample_kernel=3)
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)
        self.img_upsample = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.is_mirror_extended = False
        self.cpu_cache = False

    def check_if_mirror_extended(self, lqs: torch.Tensor) -> None:
        self.is_mirror_extended = False
        if lqs.size(1) % 2 == 0:
            first, second = torch.chunk(lqs, 2, dim=1)
            if torch.norm(first - second.flip(1)) == 0:
                self.is_mirror_extended = True

    def compute_flow(self, lqs: torch.Tensor) -> tuple[Optional[torch.Tensor], torch.Tensor]:
        n, t, c, h, w = lqs.shape
        lqs_1 = lqs[:, :-1].reshape(-1, c, h, w)
        lqs_2 = lqs[:, 1:].reshape(-1, c, h, w)
        flows_backward = self.spynet(lqs_1, lqs_2).view(n, t - 1, 2, h, w)
        flows_forward = None
        if not self.is_mirror_extended:
            flows_forward = self.spynet(lqs_2, lqs_1).view(n, t - 1, 2, h, w)
        return flows_forward, flows_backward

    def propagate(
        self,
        feats: dict[str, list[torch.Tensor]],
        flows: torch.Tensor,
        module_name: str,
    ) -> dict[str, list[torch.Tensor]]:
        n, t, _c, h, w = flows.shape
        frame_idx = list(range(t + 1))
        flow_idx = list(range(-1, t))
        mapping_idx = list(range(len(feats["spatial"])))
        mapping_idx += mapping_idx[::-1]
        if "backward" in module_name:
            frame_idx = frame_idx[::-1]
            flow_idx = frame_idx
        feat_prop = flows.new_zeros(n, self.mid_channels, h, w)
        for i, idx in enumerate(frame_idx):
            feat_current = feats["spatial"][mapping_idx[idx]]
            if i > 0:
                flow_n1 = flows[:, flow_idx[i]]
                cond_n1 = flow_warp(feat_prop, flow_n1.permute(0, 2, 3, 1))
                feat_n2 = torch.zeros_like(feat_prop)
                flow_n2 = torch.zeros_like(flow_n1)
                cond_n2 = torch.zeros_like(cond_n1)
                if i > 1:
                    feat_n2 = feats[module_name][-2]
                    flow_n2 = flows[:, flow_idx[i - 1]]
                    flow_n2 = flow_n1 + flow_warp(flow_n2, flow_n1.permute(0, 2, 3, 1))
                    cond_n2 = flow_warp(feat_n2, flow_n2.permute(0, 2, 3, 1))
                cond = torch.cat((cond_n1, feat_current, cond_n2), dim=1)
                feat_prop = torch.cat((feat_prop, feat_n2), dim=1)
                feat_prop = self.deform_align[module_name](feat_prop, cond, flow_n1, flow_n2)
            feat = [feat_current]
            feat.extend(feats[key][idx] for key in feats if key not in ("spatial", module_name))
            feat.append(feat_prop)
            feat_prop = feat_prop + self.backbone[module_name](torch.cat(feat, dim=1))
            feats[module_name].append(feat_prop)
        if "backward" in module_name:
            feats[module_name].reverse()
        return feats

    def upsample(self, lqs: torch.Tensor, feats: dict[str, list[torch.Tensor]]) -> torch.Tensor:
        outputs: list[torch.Tensor] = []
        num_outputs = len(feats["spatial"])
        mapping_idx = list(range(num_outputs))
        mapping_idx += mapping_idx[::-1]
        for i in range(lqs.size(1)):
            hr = [feats[key].pop(0) for key in feats if key != "spatial"]
            hr.insert(0, feats["spatial"][mapping_idx[i]])
            hr_tensor = self.reconstruction(torch.cat(hr, dim=1))
            hr_tensor = self.lrelu(self.upsample1(hr_tensor))
            hr_tensor = self.lrelu(self.upsample2(hr_tensor))
            hr_tensor = self.lrelu(self.conv_hr(hr_tensor))
            hr_tensor = self.conv_last(hr_tensor)
            if self.is_low_res_input:
                hr_tensor = hr_tensor + self.img_upsample(lqs[:, i])
            else:
                hr_tensor = hr_tensor + lqs[:, i]
            outputs.append(hr_tensor)
        return torch.stack(outputs, dim=1)

    def forward(self, lqs: torch.Tensor) -> torch.Tensor:
        if lqs.ndim != 5 or lqs.shape[2] != 3:
            raise ValueError("BasicVSR++ expects N,T,3,H,W input")
        n, t, c, h, w = lqs.shape
        if t < 2:
            return lqs
        if h % 4 or w % 4:
            raise ValueError(f"BasicVSR++ same-resolution input must be divisible by 4, got {w}x{h}")
        self.cpu_cache = False  # short streaming clips are deliberately kept on one GPU
        if self.is_low_res_input:
            lqs_downsample = lqs.clone()
        else:
            lqs_downsample = F.interpolate(
                lqs.reshape(-1, c, h, w),
                scale_factor=0.25,
                mode="bicubic",
                align_corners=False,
            ).view(n, t, c, h // 4, w // 4)
        if lqs_downsample.shape[-2] < 64 or lqs_downsample.shape[-1] < 64:
            raise ValueError(
                "BasicVSR++ decompression tiles must be at least 256x256 before the internal 1/4 downsample"
            )
        self.check_if_mirror_extended(lqs)
        feats: dict[str, list[torch.Tensor]] = {}
        feats_tensor = self.feat_extract(lqs.reshape(-1, c, h, w))
        feat_h, feat_w = feats_tensor.shape[-2:]
        feats_tensor = feats_tensor.view(n, t, -1, feat_h, feat_w)
        feats["spatial"] = [feats_tensor[:, i] for i in range(t)]
        flows_forward, flows_backward = self.compute_flow(lqs_downsample)
        for iteration in (1, 2):
            for direction in ("backward", "forward"):
                module_name = f"{direction}_{iteration}"
                feats[module_name] = []
                if direction == "backward":
                    flows = flows_backward
                elif flows_forward is not None:
                    flows = flows_forward
                else:
                    flows = flows_backward.flip(1)
                feats = self.propagate(feats, flows, module_name)
        return self.upsample(lqs, feats)


@dataclass(frozen=True)
class BasicVSRPPConfig:
    track: int = 1
    model_path: str = ""
    gpu_id: int = 0
    fp16: bool = True
    clip_length: int = 7
    clip_overlap: int = 2
    tile_size: int = 512
    tile_pad: int = 32
    strength: float = 1.0
    scene_threshold: float = 0.30


def download_basicvsrpp_checkpoint(track: int, target_dir: Path) -> Path:
    if track not in BASICVSRPP_TRACK_URLS:
        raise ValueError(f"Unsupported BasicVSR++ track: {track}")
    url = BASICVSRPP_TRACK_URLS[track]
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / url.rsplit("/", 1)[-1]
    if target.is_file() and target.stat().st_size > 0:
        print(f"[basicvsrpp] using cached checkpoint: {target}", flush=True)
        return target
    temporary = target.with_suffix(target.suffix + ".part")
    print(f"[basicvsrpp] downloading {url}", flush=True)
    try:
        urllib.request.urlretrieve(url, temporary)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def _torch_load_cpu(path: Path) -> dict[str, object]:
    try:
        loaded = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        loaded = torch.load(path, map_location="cpu")
    if not isinstance(loaded, dict):
        raise TypeError(f"Unsupported BasicVSR++ checkpoint object: {type(loaded)}")
    return loaded


def _extract_state_dict(checkpoint: dict[str, object]) -> dict[str, torch.Tensor]:
    candidate: object = checkpoint
    for key in ("state_dict", "params_ema", "params"):
        if key in checkpoint:
            candidate = checkpoint[key]
            break
    if not isinstance(candidate, dict):
        raise KeyError("BasicVSR++ checkpoint has no usable state_dict/params")
    return {str(key): value for key, value in candidate.items() if isinstance(value, torch.Tensor)}


def normalize_basicvsrpp_state(
    raw_state: dict[str, torch.Tensor],
    model: nn.Module,
) -> dict[str, torch.Tensor]:
    """Map MMagic/MMEditing wrapper prefixes to generator parameter names."""
    model_keys = set(model.state_dict())
    normalized: dict[str, torch.Tensor] = {}
    generator_like = False
    for original_key, value in raw_state.items():
        key = original_key
        while key.startswith("module."):
            key = key[len("module.") :]
        for prefix in ("model.generator.", "generator."):
            if key.startswith(prefix):
                key = key[len(prefix) :]
                generator_like = True
                break
        if key in model_keys:
            normalized[key] = value
        elif original_key.startswith(("generator.", "module.generator.", "model.generator.")):
            raise KeyError(f"Unexpected BasicVSR++ generator key after prefix removal: {original_key}")
    missing = model_keys - set(normalized)
    allowed_missing = {"spynet.mean", "spynet.std"}
    hard_missing = sorted(missing - allowed_missing)
    if hard_missing:
        mode = "generator-prefixed" if generator_like else "unprefixed"
        raise KeyError(f"BasicVSR++ checkpoint ({mode}) is missing {len(hard_missing)} model keys: {hard_missing[:8]}")
    return normalized


def load_basicvsrpp_checkpoint(model: nn.Module, path: Path) -> None:
    raw = _extract_state_dict(_torch_load_cpu(path))
    state = normalize_basicvsrpp_state(raw, model)
    result = model.load_state_dict(state, strict=False)
    allowed_missing = {"spynet.mean", "spynet.std"}
    hard_missing = sorted(set(result.missing_keys) - allowed_missing)
    if hard_missing or result.unexpected_keys:
        raise RuntimeError(
            f"BasicVSR++ checkpoint mismatch: missing={hard_missing}, unexpected={result.unexpected_keys}"
        )
    print(
        f"[basicvsrpp] checkpoint loaded: keys={len(state)}, path={path}",
        flush=True,
    )


def frame_to_float_rgb(frame: np.ndarray) -> np.ndarray:
    if frame.dtype == np.uint8:
        return np.ascontiguousarray(frame.astype(np.float32) / 255.0)
    if frame.dtype == np.float32:
        if not np.isfinite(frame).all():
            raise ValueError("BasicVSR++ input contains NaN or Inf")
        return np.ascontiguousarray(np.clip(frame, 0.0, 1.0), dtype=np.float32)
    raise TypeError(f"Unsupported frame dtype for BasicVSR++: {frame.dtype}")


def _pad_to_model_size(x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
    """Pad N,T,C,H,W on right/bottom to >=256 and divisible by four."""
    h, w = x.shape[-2:]
    target_h = max(256, ((h + 3) // 4) * 4)
    target_w = max(256, ((w + 3) // 4) * 4)
    pad_h, pad_w = target_h - h, target_w - w
    if pad_h == 0 and pad_w == 0:
        return x, h, w
    flat = x.reshape(-1, x.shape[2], h, w)
    mode = "reflect" if h > pad_h and w > pad_w and h > 1 and w > 1 else "replicate"
    flat = F.pad(flat, (0, pad_w, 0, pad_h), mode=mode)
    return flat.view(x.shape[0], x.shape[1], x.shape[2], target_h, target_w), h, w


class BasicVSRPPPreprocessor:
    def __init__(
        self,
        config: BasicVSRPPConfig,
        checkpoint_dir: Optional[Path] = None,
        model: Optional[nn.Module] = None,
    ):
        self.config = config
        if config.track not in BASICVSRPP_TRACK_URLS:
            raise ValueError("BasicVSR++ track must be 1, 2 or 3")
        if not 0.0 <= config.strength <= 1.0:
            raise ValueError("BasicVSR++ strength must be in [0,1]")
        if config.tile_size != 0 and (config.tile_size < 256 or config.tile_size % 4):
            raise ValueError("BasicVSR++ tile size must be 0 or >=256 and divisible by 4")
        if config.tile_pad < 0 or config.tile_pad % 4:
            raise ValueError("BasicVSR++ tile pad must be non-negative and divisible by 4")
        if config.clip_length < 2:
            raise ValueError("BasicVSR++ clip length must be at least 2")
        if not 0 <= config.clip_overlap < config.clip_length / 2:
            raise ValueError("BasicVSR++ clip overlap must satisfy 0 <= overlap < clip_length/2")

        if model is None:
            if not torch.cuda.is_available():
                raise RuntimeError("BasicVSR++ compressed-video preprocessing requires CUDA")
            if config.gpu_id < 0 or config.gpu_id >= torch.cuda.device_count():
                raise ValueError(
                    f"BasicVSR++ requested cuda:{config.gpu_id}, but {torch.cuda.device_count()} GPU(s) are visible"
                )
            self.device = torch.device(f"cuda:{config.gpu_id}")
            checkpoint_path = (
                Path(config.model_path).expanduser().resolve()
                if config.model_path
                else download_basicvsrpp_checkpoint(
                    config.track,
                    checkpoint_dir or Path(__file__).resolve().parents[1] / "weights",
                )
            )
            if not checkpoint_path.is_file():
                raise FileNotFoundError(f"BasicVSR++ checkpoint not found: {checkpoint_path}")
            built_model = BasicVSRPlusPlusNet(
                mid_channels=128,
                num_blocks=25,
                is_low_res_input=False,
                cpu_cache_length=100,
            )
            load_basicvsrpp_checkpoint(built_model, checkpoint_path)
            self.model = built_model.eval().requires_grad_(False).to(self.device)
            if config.fp16:
                self.model.half()
        else:
            self.model = model.eval().requires_grad_(False)
            parameter = next(self.model.parameters(), None)
            self.device = parameter.device if parameter is not None else torch.device("cpu")
        self.dtype = torch.float16 if config.fp16 and self.device.type == "cuda" else torch.float32
        self.elapsed = 0.0
        self.clips = 0
        self.tiles = 0

    def close(self) -> None:
        if self.device.type == "cuda":
            del self.model
            torch.cuda.empty_cache()

    def _run_model(self, clip: torch.Tensor) -> torch.Tensor:
        padded, original_h, original_w = _pad_to_model_size(clip)
        padded = padded.to(self.device, dtype=self.dtype, non_blocking=True)
        try:
            with torch.inference_mode():
                output = self.model(padded)
        except torch.cuda.OutOfMemoryError as error:
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            raise RuntimeError(
                "BasicVSR++ ran out of GPU memory. Reduce --basicvsrpp-tile-size or "
                "--basicvsrpp-clip-length."
            ) from error
        except RuntimeError as error:
            message = str(error).lower()
            half_failure = self.dtype == torch.float16 and any(
                token in message
                for token in (
                    "not implemented for 'half'",
                    "not implemented for half",
                    "expected scalar type float",
                    "deform_conv2d",
                )
            )
            if not half_failure:
                raise
            print(
                "[basicvsrpp] FP16 operator path failed; retrying this run in FP32",
                flush=True,
            )
            self.model.float()
            self.dtype = torch.float32
            padded = padded.float()
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            with torch.inference_mode():
                output = self.model(padded)
        return output[..., :original_h, :original_w].float()

    def _enhance_tensor(self, clip: torch.Tensor) -> torch.Tensor:
        _n, _t, _c, height, width = clip.shape
        if self.config.tile_size == 0:
            return self._run_model(clip).cpu()
        tile_size = self.config.tile_size
        pad = self.config.tile_pad
        flat = clip.reshape(-1, 3, height, width)
        mode = "reflect" if min(height, width) > pad and min(height, width) > 1 else "replicate"
        padded_flat = F.pad(flat, (pad, pad, pad, pad), mode=mode) if pad else flat
        padded = padded_flat.view(clip.shape[0], clip.shape[1], 3, height + 2 * pad, width + 2 * pad)
        result = torch.empty_like(clip, dtype=torch.float32, device="cpu")
        for y0 in range(0, height, tile_size):
            y1 = min(y0 + tile_size, height)
            for x0 in range(0, width, tile_size):
                x1 = min(x0 + tile_size, width)
                patch = padded[..., y0 : y1 + 2 * pad, x0 : x1 + 2 * pad]
                enhanced = self._run_model(patch).cpu()
                result[..., y0:y1, x0:x1] = enhanced[
                    ..., pad : pad + (y1 - y0), pad : pad + (x1 - x0)
                ]
                self.tiles += 1
        return result

    def enhance_clip(self, frames: Sequence[np.ndarray]) -> list[np.ndarray]:
        if not frames:
            return []
        first_shape = frames[0].shape
        if any(frame.shape != first_shape for frame in frames):
            raise ValueError("All BasicVSR++ clip frames must have identical dimensions")
        originals = np.stack([frame_to_float_rgb(frame) for frame in frames])
        if len(frames) == 1 or self.config.strength == 0:
            return [np.ascontiguousarray(frame, dtype=np.float32) for frame in originals]
        clip = torch.from_numpy(originals).permute(0, 3, 1, 2).unsqueeze(0)
        started = time.monotonic()
        enhanced = self._enhance_tensor(clip).squeeze(0).permute(0, 2, 3, 1).numpy()
        self.elapsed += time.monotonic() - started
        self.clips += 1
        mixed = originals + self.config.strength * (enhanced - originals)
        return [np.ascontiguousarray(np.clip(frame, 0.0, 1.0), dtype=np.float32) for frame in mixed]


def scene_difference(previous: np.ndarray, current: np.ndarray) -> float:
    """Low-cost scene-cut score in [0,1]-like units."""
    prev = frame_to_float_rgb(previous)
    curr = frame_to_float_rgb(current)
    prev_small = cv2.resize(prev, (64, 64), interpolation=cv2.INTER_AREA)
    curr_small = cv2.resize(curr, (64, 64), interpolation=cv2.INTER_AREA)
    prev_luma = 0.2126 * prev_small[..., 0] + 0.7152 * prev_small[..., 1] + 0.0722 * prev_small[..., 2]
    curr_luma = 0.2126 * curr_small[..., 0] + 0.7152 * curr_small[..., 1] + 0.0722 * curr_small[..., 2]
    mad = float(np.mean(np.abs(prev_luma - curr_luma)))
    hist_prev, _ = np.histogram(prev_luma, bins=32, range=(0.0, 1.0), density=False)
    hist_curr, _ = np.histogram(curr_luma, bins=32, range=(0.0, 1.0), density=False)
    hist_prev = hist_prev.astype(np.float64) / max(hist_prev.sum(), 1)
    hist_curr = hist_curr.astype(np.float64) / max(hist_curr.sum(), 1)
    hist_distance = 0.5 * float(np.abs(hist_prev - hist_curr).sum())
    return 0.7 * mad + 0.3 * hist_distance


class BasicVSRPPStreamReader:
    """Convert a frame reader into an overlapping BasicVSR++ clip stream."""

    def __init__(self, reader: FrameReader, preprocessor: BasicVSRPPPreprocessor):
        self.reader = reader
        self.preprocessor = preprocessor
        self.clip_length = preprocessor.config.clip_length
        self.overlap = preprocessor.config.clip_overlap
        self.buffer: list[np.ndarray] = []
        self.output: Deque[np.ndarray] = deque()
        self.pending: Optional[np.ndarray] = None
        self.eof = False
        self.segment_end = False
        self.first_chunk = True
        self.decode_elapsed = 0.0
        self.scene_cuts = 0

    def _read_source(self) -> Optional[np.ndarray]:
        started = time.monotonic()
        frame = self.reader.read()
        self.decode_elapsed += time.monotonic() - started
        return frame

    def _regular_chunk(self) -> None:
        enhanced = self.preprocessor.enhance_clip(self.buffer)
        if self.first_chunk:
            emit_end = self.clip_length - self.overlap
            self.output.extend(enhanced[:emit_end])
            self.first_chunk = False
        else:
            self.output.extend(enhanced[self.overlap : self.clip_length - self.overlap])
        retain = 2 * self.overlap
        self.buffer = self.buffer[-retain:] if retain else []

    def _finish_segment(self) -> None:
        if self.buffer:
            enhanced = self.preprocessor.enhance_clip(self.buffer)
            if self.first_chunk:
                self.output.extend(enhanced)
            else:
                self.output.extend(enhanced[self.overlap :])
        self.buffer = []
        self.first_chunk = True
        self.segment_end = False
        if self.pending is not None:
            self.buffer.append(self.pending)
            self.pending = None

    def read(self) -> Optional[np.ndarray]:
        while not self.output:
            if self.segment_end or self.eof:
                self._finish_segment()
                if self.output:
                    break
                if self.eof and self.pending is None and not self.buffer:
                    return None
                continue
            frame = self._read_source()
            if frame is None:
                self.eof = True
                continue
            if self.buffer and self.preprocessor.config.scene_threshold > 0:
                if scene_difference(self.buffer[-1], frame) >= self.preprocessor.config.scene_threshold:
                    self.pending = frame
                    self.segment_end = True
                    self.scene_cuts += 1
                    continue
            self.buffer.append(frame)
            if len(self.buffer) == self.clip_length:
                self._regular_chunk()
        return self.output.popleft()

    def close(self) -> None:
        self.reader.close()
        self.preprocessor.close()
