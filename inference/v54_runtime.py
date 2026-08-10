"""Fixed-parameter BasicVSR++ execution optimizations.

These patches preserve the BasicVSR++ model, weights, clip policy, tiling,
scene-cut behavior, and restoration arithmetic. They only reduce execution and
transfer overhead: cache invariant warp grids, avoid temporary zero tensors that
are immediately overwritten, keep 8-bit clips compact across H2D, and use a
conservative channels-last layout for the main Conv2d-heavy restoration blocks.
"""
from __future__ import annotations

import time
from typing import Dict, Sequence, Tuple

import numpy as np
import torch
from torch.nn import functional as F

from . import basicvsrpp as bvsr


_INSTALLED = False
_FLOW_GRID_CACHE: Dict[Tuple[torch.device, torch.dtype, int, int], torch.Tensor] = {}


def _flow_warp_cached(
    x: torch.Tensor,
    flow: torch.Tensor,
    interpolation: str = "bilinear",
    padding_mode: str = "zeros",
    align_corners: bool = True,
) -> torch.Tensor:
    """Match BasicVSR++ flow_warp while reusing its invariant pixel grid."""
    if x.shape[-2:] != flow.shape[1:3]:
        raise ValueError(f"flow size {flow.shape[1:3]} does not match feature size {x.shape[-2:]}")
    n, _c, h, w = x.shape
    key = (x.device, x.dtype, int(h), int(w))
    base_grid = _FLOW_GRID_CACHE.get(key)
    if base_grid is None:
        grid_y, grid_x = torch.meshgrid(
            torch.arange(h, device=x.device, dtype=x.dtype),
            torch.arange(w, device=x.device, dtype=x.dtype),
            indexing="ij",
        )
        base_grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0)
        _FLOW_GRID_CACHE[key] = base_grid

    grid = base_grid.expand(n, -1, -1, -1) + flow.to(dtype=x.dtype)
    gx = 2.0 * grid[..., 0] / (w - 1) - 1.0 if w > 1 else torch.zeros_like(grid[..., 0])
    gy = 2.0 * grid[..., 1] / (h - 1) - 1.0 if h > 1 else torch.zeros_like(grid[..., 1])
    normalized = torch.stack((gx, gy), dim=-1)
    return F.grid_sample(
        x,
        normalized,
        mode=interpolation,
        padding_mode=padding_mode,
        align_corners=align_corners,
    )


def _propagate_v54(
    self,
    feats: dict[str, list[torch.Tensor]],
    flows: torch.Tensor,
    module_name: str,
) -> dict[str, list[torch.Tensor]]:
    """Match BasicVSR++ propagation without zero tensors that get overwritten."""
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
            cond_n1 = bvsr.flow_warp(feat_prop, flow_n1.permute(0, 2, 3, 1))
            if i > 1:
                feat_n2 = feats[module_name][-2]
                flow_n2 = flows[:, flow_idx[i - 1]]
                flow_n2 = flow_n1 + bvsr.flow_warp(
                    flow_n2, flow_n1.permute(0, 2, 3, 1)
                )
                cond_n2 = bvsr.flow_warp(feat_n2, flow_n2.permute(0, 2, 3, 1))
            else:
                feat_n2 = torch.zeros_like(feat_prop)
                flow_n2 = torch.zeros_like(flow_n1)
                cond_n2 = torch.zeros_like(cond_n1)
            cond = torch.cat((cond_n1, feat_current, cond_n2), dim=1)
            feat_prop = self.deform_align[module_name](
                torch.cat((feat_prop, feat_n2), dim=1), cond, flow_n1, flow_n2
            )
        feat = [feat_current]
        feat.extend(feats[key][idx] for key in feats if key not in ("spatial", module_name))
        feat.append(feat_prop)
        feat_prop = feat_prop + self.backbone[module_name](torch.cat(feat, dim=1))
        feats[module_name].append(feat_prop)
    if "backward" in module_name:
        feats[module_name].reverse()
    return feats


def _enhance_u8_with_tile(preprocessor, clip_cpu: torch.Tensor, tile_size: int) -> np.ndarray:
    """Transfer uint8 clips compactly, then normalize and restore on CUDA."""
    from . import v51_runtime as v51

    device = preprocessor.device
    torch.cuda.set_device(device)

    timing = None
    start_event = end_event = None
    wall_started = 0.0
    try:
        from . import gpu_timing

        if getattr(gpu_timing, "_INSTALLED", False):
            timing = gpu_timing
            wall_started = time.monotonic()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
    except Exception:
        timing = None
        start_event = end_event = None

    if clip_cpu.dtype == torch.uint8:
        original_u8 = clip_cpu.to(device=device, non_blocking=True)
        original = original_u8.to(dtype=torch.float32)
        original.div_(255.0)
        del original_u8
    else:
        original = clip_cpu.to(device=device, dtype=torch.float32, non_blocking=True)

    _n, _t, _c, height, width = original.shape
    pad = int(preprocessor.config.tile_pad)
    flat = original.reshape(-1, 3, height, width)
    mode = "reflect" if min(height, width) > pad and min(height, width) > 1 else "replicate"
    padded_flat = F.pad(flat, (pad, pad, pad, pad), mode=mode) if pad else flat
    padded = padded_flat.view(
        original.shape[0], original.shape[1], 3, height + 2 * pad, width + 2 * pad
    )
    restored = torch.empty_like(original, dtype=torch.float32, device=device)
    tile_count = 0
    for y0 in range(0, height, tile_size):
        y1 = min(y0 + tile_size, height)
        for x0 in range(0, width, tile_size):
            x1 = min(x0 + tile_size, width)
            patch = padded[..., y0 : y1 + 2 * pad, x0 : x1 + 2 * pad]
            enhanced = v51._run_model_device(preprocessor, patch)
            restored[..., y0:y1, x0:x1] = enhanced[
                ..., pad : pad + (y1 - y0), pad : pad + (x1 - x0)
            ]
            tile_count += 1
    preprocessor.tiles += tile_count

    strength = float(preprocessor.config.strength)
    mixed = restored if strength >= 1.0 else original + strength * (restored - original)
    quantized = mixed.clamp_(0.0, 1.0).mul_(255.0).round_().to(torch.uint8)

    if start_event is not None and end_event is not None:
        end_event.record()
    result = quantized.permute(0, 1, 3, 4, 2).contiguous().cpu().numpy()
    if timing is not None and start_event is not None and end_event is not None:
        gpu_seconds = start_event.elapsed_time(end_event) / 1000.0
        timing._record(
            "bvs",
            int(preprocessor.config.gpu_id),
            gpu_seconds,
            wall_started,
            time.monotonic(),
        )
    return result


def _enhance_u8(preprocessor, clip_cpu: torch.Tensor) -> np.ndarray:
    requested = int(preprocessor.tile_size)
    candidates = [requested]
    for fallback in (384, 320, 256):
        if fallback < requested and fallback not in candidates:
            candidates.append(fallback)
    last_error = None
    for tile_size in candidates:
        try:
            output = _enhance_u8_with_tile(preprocessor, clip_cpu, tile_size)
            if tile_size != preprocessor.tile_size:
                preprocessor.tile_size = tile_size
                print(f"[basicvsrpp] VRAM fallback locked to tile={tile_size}", flush=True)
            return output
        except torch.cuda.OutOfMemoryError as error:
            last_error = error
            torch.cuda.empty_cache()
            print(f"[basicvsrpp] tile={tile_size} OOM; retrying smaller tile", flush=True)
    raise RuntimeError("BasicVSR++ ran out of GPU memory even at tile=256") from last_error


def _enhance_clip(self, frames: Sequence[np.ndarray]) -> list[np.ndarray]:
    if not frames:
        return []
    first_shape = frames[0].shape
    first_dtype = np.dtype(frames[0].dtype)
    if any(frame.shape != first_shape or np.dtype(frame.dtype) != first_dtype for frame in frames):
        raise ValueError("All BasicVSR++ clip frames must have identical shape/dtype")
    if len(frames) == 1:
        return [np.ascontiguousarray(frame) for frame in frames]
    if first_dtype != np.dtype(np.uint8):
        return self._fixed_original_enhance_clip(frames)

    packed = np.stack(frames, axis=0)
    clip = torch.from_numpy(packed).permute(0, 3, 1, 2).unsqueeze(0)
    started = time.monotonic()
    enhanced_u8 = _enhance_u8(self, clip)[0]
    self.elapsed += time.monotonic() - started
    self.clips += 1
    return [np.ascontiguousarray(frame) for frame in enhanced_u8]


def _enhance_clips(self, clips: Sequence[Sequence[np.ndarray]]) -> list[list[np.ndarray]]:
    if not clips:
        return []
    if len(clips) == 1:
        return [_enhance_clip(self, clips[0])]

    lengths = {len(frames) for frames in clips}
    if len(lengths) != 1 or next(iter(lengths)) < 2:
        return [_enhance_clip(self, frames) for frames in clips]
    first = clips[0][0]
    first_shape = first.shape
    first_dtype = np.dtype(first.dtype)
    for frames in clips:
        if any(frame.shape != first_shape or np.dtype(frame.dtype) != first_dtype for frame in frames):
            return [_enhance_clip(self, item) for item in clips]
    if first_dtype != np.dtype(np.uint8):
        return [self._fixed_original_enhance_clip(frames) for frames in clips]

    packed = np.stack([np.stack(frames, axis=0) for frames in clips], axis=0)
    tensor = torch.from_numpy(packed).permute(0, 1, 4, 2, 3)
    started = time.monotonic()
    enhanced_u8 = _enhance_u8(self, tensor)
    self.elapsed += time.monotonic() - started
    self.clips += len(clips)
    return [
        [np.ascontiguousarray(frame) for frame in group]
        for group in enhanced_u8
    ]


def _enable_selective_channels_last(preprocessor) -> None:
    """Convert only the main Conv2d-heavy BVS blocks to channels-last weights.

    This deliberately leaves SPyNet, deformable-alignment offset convolutions,
    PixelShuffle/output convolutions, and the custom deform_conv2d weight alone.
    No input/output wrapper is installed, so scheduler and CUDA thread behavior
    are unchanged from the fixed-parameter v5.7 path.
    """
    if preprocessor.dtype != torch.float16:
        return
    major, _minor = torch.cuda.get_device_capability(preprocessor.device)
    if major < 7:
        return

    converter = getattr(torch.nn.utils, "convert_conv2d_weight_memory_format", None)
    if converter is None:
        print(
            "[basicvsrpp] selective channels_last unavailable in this PyTorch build; using NCHW",
            flush=True,
        )
        return

    model = preprocessor.model
    targets = (
        model.feat_extract,
        model.backbone,
        model.reconstruction,
    )
    converted = 0
    for module in targets:
        converted += sum(1 for item in module.modules() if isinstance(item, torch.nn.Conv2d))
        converter(module, torch.channels_last)

    print(
        f"[basicvsrpp] selective channels_last enabled: {converted} Conv2d weights | "
        "SPyNet/deform-align/upsample kept unchanged",
        flush=True,
    )


def install_basicvsrpp_execution_optimizations() -> None:
    """Install fixed-parameter BasicVSR++ execution optimizations."""
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True

    bvsr.flow_warp = _flow_warp_cached
    bvsr.BasicVSRPlusPlusNet.propagate = _propagate_v54

    cls = bvsr.BasicVSRPPPreprocessor
    if not hasattr(cls, "_fixed_original_enhance_clip"):
        cls._fixed_original_enhance_clip = cls.enhance_clip
    cls.enhance_clip = _enhance_clip
    cls.enhance_clips = _enhance_clips

    if not hasattr(cls, "_fixed_original_init"):
        cls._fixed_original_init = cls.__init__

        def init_with_selective_channels_last(self, *args, **kwargs):
            self._fixed_original_init(*args, **kwargs)
            _enable_selective_channels_last(self)

        cls.__init__ = init_with_selective_channels_last
