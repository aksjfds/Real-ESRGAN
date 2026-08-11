"""Explicit BasicVSR++ execution optimizations for the unified GPU worker.

This module replaces the former v54 runtime monkey-patches with concrete model
and preprocessor classes. Model weights and restoration arithmetic are unchanged.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.nn import functional as F

from . import basicvsrpp as base


_FLOW_GRID_CACHE: dict[
    tuple[torch.device, torch.dtype, int, int],
    torch.Tensor,
] = {}


def _flow_warp_cached(
    x: torch.Tensor,
    flow: torch.Tensor,
    interpolation: str = "bilinear",
    padding_mode: str = "zeros",
    align_corners: bool = True,
) -> torch.Tensor:
    if x.shape[-2:] != flow.shape[1:3]:
        raise ValueError(
            f"flow size {flow.shape[1:3]} does not match feature size "
            f"{x.shape[-2:]}"
        )
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
    gx = (
        2.0 * grid[..., 0] / (w - 1) - 1.0
        if w > 1
        else torch.zeros_like(grid[..., 0])
    )
    gy = (
        2.0 * grid[..., 1] / (h - 1) - 1.0
        if h > 1
        else torch.zeros_like(grid[..., 1])
    )
    normalized = torch.stack((gx, gy), dim=-1)
    return F.grid_sample(
        x,
        normalized,
        mode=interpolation,
        padding_mode=padding_mode,
        align_corners=align_corners,
    )


class OptimizedSPyNet(base.SPyNet):
    def compute_flow(
        self,
        ref: torch.Tensor,
        supp: torch.Tensor,
    ) -> torch.Tensor:
        n, _c, h, w = ref.shape
        ref_pyramid = [(ref - self.mean) / self.std]
        supp_pyramid = [(supp - self.mean) / self.std]
        for _ in range(5):
            ref_pyramid.append(
                F.avg_pool2d(
                    ref_pyramid[-1],
                    2,
                    2,
                    count_include_pad=False,
                )
            )
            supp_pyramid.append(
                F.avg_pool2d(
                    supp_pyramid[-1],
                    2,
                    2,
                    count_include_pad=False,
                )
            )
        ref_pyramid.reverse()
        supp_pyramid.reverse()
        flow = ref.new_zeros(n, 2, h // 32, w // 32)
        for level, (ref_level, supp_level) in enumerate(
            zip(ref_pyramid, supp_pyramid)
        ):
            flow_up = (
                flow
                if level == 0
                else F.interpolate(
                    flow,
                    scale_factor=2,
                    mode="bilinear",
                    align_corners=True,
                )
                * 2.0
            )
            warped = _flow_warp_cached(
                supp_level,
                flow_up.permute(0, 2, 3, 1),
                padding_mode="border",
            )
            flow = flow_up + self.basic_module[level](
                torch.cat((ref_level, warped, flow_up), dim=1)
            )
        return flow


class OptimizedBasicVSRPlusPlusNet(base.BasicVSRPlusPlusNet):
    def __init__(self, mid_channels: int = 64, num_blocks: int = 7):
        super().__init__(mid_channels=mid_channels, num_blocks=num_blocks)
        self.spynet = OptimizedSPyNet()

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
                cond_n1 = _flow_warp_cached(
                    feat_prop,
                    flow_n1.permute(0, 2, 3, 1),
                )
                if i > 1:
                    feat_n2 = feats[module_name][-2]
                    flow_n2 = flows[:, flow_idx[i - 1]]
                    flow_n2 = flow_n1 + _flow_warp_cached(
                        flow_n2,
                        flow_n1.permute(0, 2, 3, 1),
                    )
                    cond_n2 = _flow_warp_cached(
                        feat_n2,
                        flow_n2.permute(0, 2, 3, 1),
                    )
                else:
                    feat_n2 = torch.zeros_like(feat_prop)
                    flow_n2 = torch.zeros_like(flow_n1)
                    cond_n2 = torch.zeros_like(cond_n1)
                cond = torch.cat(
                    (cond_n1, feat_current, cond_n2),
                    dim=1,
                )
                feat_prop = self.deform_align[module_name](
                    torch.cat((feat_prop, feat_n2), dim=1),
                    cond,
                    flow_n1,
                    flow_n2,
                )

            feat = [feat_current]
            feat.extend(
                feats[key][idx]
                for key in feats
                if key not in ("spatial", module_name)
            )
            feat.append(feat_prop)
            feat_prop = feat_prop + self.backbone[module_name](
                torch.cat(feat, dim=1)
            )
            feats[module_name].append(feat_prop)

        if "backward" in module_name:
            feats[module_name].reverse()
        return feats


class OptimizedBasicVSRPPPreprocessor(base.BasicVSRPPPreprocessor):
    """Explicit v5.8/v5.4 execution path without runtime class patching."""

    def __init__(
        self,
        config: base.BasicVSRPPConfig,
        checkpoint_dir: Path,
    ) -> None:
        self.config = config
        if not 0.0 < config.strength <= 1.0:
            raise ValueError("BasicVSR++ strength must be in (0,1]")
        if config.tile_size < 256 or config.tile_size % 4:
            raise ValueError(
                "BasicVSR++ tile size must be >=256 and divisible by 4"
            )
        if config.tile_pad < 0 or config.tile_pad % 4:
            raise ValueError(
                "BasicVSR++ tile pad must be non-negative and divisible by 4"
            )
        if config.clip_length < 2:
            raise ValueError("BasicVSR++ clip length must be at least 2")
        if not 0 <= config.clip_overlap < config.clip_length / 2:
            raise ValueError(
                "BasicVSR++ clip overlap must satisfy "
                "0 <= overlap < clip_length/2"
            )
        if not torch.cuda.is_available():
            raise RuntimeError("BasicVSR++ preprocessing requires CUDA")
        if config.gpu_id < 0 or config.gpu_id >= torch.cuda.device_count():
            raise ValueError(
                f"BasicVSR++ requested cuda:{config.gpu_id}, but "
                f"{torch.cuda.device_count()} GPU(s) are visible"
            )

        self.device = torch.device(f"cuda:{config.gpu_id}")
        checkpoint = (
            Path(config.model_path).expanduser().resolve()
            if config.model_path
            else base.download_checkpoint(checkpoint_dir)
        )
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"BasicVSR++ checkpoint not found: {checkpoint}"
            )

        with torch.cuda.device(self.device):
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cuda.matmul.allow_tf32 = True
            model = OptimizedBasicVSRPlusPlusNet(
                mid_channels=128,
                num_blocks=25,
            )
            base.load_checkpoint(model, checkpoint)
            self.model = model.eval().requires_grad_(False).to(self.device)
            if config.fp16:
                self.model.half()

            self.dtype = (
                torch.float16 if config.fp16 else torch.float32
            )
            self._enable_selective_channels_last()

        self.elapsed = 0.0
        self.clips = 0
        self.tiles = 0
        self.tile_size = config.tile_size

    def _enable_selective_channels_last(self) -> None:
        if self.dtype != torch.float16:
            return
        major, _minor = torch.cuda.get_device_capability(self.device)
        if major < 7:
            return

        converter = getattr(
            torch.nn.utils,
            "convert_conv2d_weight_memory_format",
            None,
        )
        if converter is None:
            print(
                "[basicvsrpp] selective channels_last unavailable; "
                "using NCHW",
                flush=True,
            )
            return

        targets = (
            self.model.feat_extract,
            self.model.backbone,
            self.model.reconstruction,
        )
        converted = 0
        for module in targets:
            converted += sum(
                1
                for item in module.modules()
                if isinstance(item, torch.nn.Conv2d)
            )
            converter(module, torch.channels_last)
        print(
            "[basicvsrpp] selective channels_last enabled: "
            f"{converted} Conv2d weights | "
            "SPyNet/deform-align/upsample kept unchanged",
            flush=True,
        )

    def close(self) -> None:
        if hasattr(self, "model"):
            del self.model
        with torch.cuda.device(self.device):
            torch.cuda.empty_cache()

    def _run_model_device(
        self,
        clip_gpu: torch.Tensor,
    ) -> torch.Tensor:
        padded, original_h, original_w = base._pad_to_model_size(clip_gpu)
        model_input = padded.to(dtype=self.dtype)
        try:
            with torch.inference_mode():
                output = self.model(model_input)
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
                "[basicvsrpp] FP16 operator path failed; switching to FP32",
                flush=True,
            )
            self.model.float()
            self.dtype = torch.float32
            torch.cuda.empty_cache()
            model_input = padded.float()
            with torch.inference_mode():
                output = self.model(model_input)
        return output[..., :original_h, :original_w].float()

    def _enhance_u8_with_tile(
        self,
        clip_cpu: torch.Tensor,
        tile_size: int,
    ) -> np.ndarray:
        original_u8 = clip_cpu.to(
            device=self.device,
            non_blocking=True,
        )
        original = original_u8.to(dtype=torch.float32)
        original.div_(255.0)
        del original_u8

        _n, _t, _c, height, width = original.shape
        pad = int(self.config.tile_pad)
        flat = original.reshape(-1, 3, height, width)
        mode = (
            "reflect"
            if min(height, width) > pad and min(height, width) > 1
            else "replicate"
        )
        padded_flat = (
            F.pad(flat, (pad, pad, pad, pad), mode=mode)
            if pad
            else flat
        )
        padded = padded_flat.view(
            original.shape[0],
            original.shape[1],
            3,
            height + 2 * pad,
            width + 2 * pad,
        )
        restored = torch.empty_like(
            original,
            dtype=torch.float32,
            device=self.device,
        )
        tile_count = 0
        for y0 in range(0, height, tile_size):
            y1 = min(y0 + tile_size, height)
            for x0 in range(0, width, tile_size):
                x1 = min(x0 + tile_size, width)
                patch = padded[
                    ...,
                    y0 : y1 + 2 * pad,
                    x0 : x1 + 2 * pad,
                ]
                enhanced = self._run_model_device(patch)
                restored[..., y0:y1, x0:x1] = enhanced[
                    ...,
                    pad : pad + (y1 - y0),
                    pad : pad + (x1 - x0),
                ]
                tile_count += 1
        self.tiles += tile_count

        strength = float(self.config.strength)
        mixed = (
            restored
            if strength >= 1.0
            else original + strength * (restored - original)
        )
        quantized = (
            mixed.clamp_(0.0, 1.0)
            .mul_(255.0)
            .round_()
            .to(torch.uint8)
        )
        return (
            quantized.permute(0, 1, 3, 4, 2)
            .contiguous()
            .cpu()
            .numpy()
        )

    def _enhance_u8(self, clip_cpu: torch.Tensor) -> np.ndarray:
        requested = int(self.tile_size)
        candidates = [requested]
        for fallback in (384, 320, 256):
            if fallback < requested and fallback not in candidates:
                candidates.append(fallback)

        last_error = None
        for tile_size in candidates:
            try:
                output = self._enhance_u8_with_tile(
                    clip_cpu,
                    tile_size,
                )
                if tile_size != self.tile_size:
                    self.tile_size = tile_size
                    print(
                        "[basicvsrpp] VRAM fallback locked to "
                        f"tile={tile_size}",
                        flush=True,
                    )
                return output
            except torch.cuda.OutOfMemoryError as error:
                last_error = error
                torch.cuda.empty_cache()
                print(
                    f"[basicvsrpp] tile={tile_size} OOM; "
                    "retrying smaller tile",
                    flush=True,
                )
        raise RuntimeError(
            "BasicVSR++ ran out of GPU memory even at tile=256"
        ) from last_error

    def enhance_clip(
        self,
        frames: Sequence[np.ndarray],
    ) -> list[np.ndarray]:
        if not frames:
            return []
        first_shape = frames[0].shape
        first_dtype = np.dtype(frames[0].dtype)
        if any(
            frame.shape != first_shape
            or np.dtype(frame.dtype) != first_dtype
            for frame in frames
        ):
            raise ValueError(
                "All BasicVSR++ clip frames must have identical shape/dtype"
            )
        if len(frames) == 1:
            return [np.ascontiguousarray(frame) for frame in frames]
        if first_dtype != np.dtype(np.uint8):
            return super().enhance_clip(frames)

        packed = np.stack(frames, axis=0)
        clip = (
            torch.from_numpy(packed)
            .permute(0, 3, 1, 2)
            .unsqueeze(0)
        )
        started = time.monotonic()
        enhanced_u8 = self._enhance_u8(clip)[0]
        self.elapsed += time.monotonic() - started
        self.clips += 1
        return [
            np.ascontiguousarray(frame)
            for frame in enhanced_u8
        ]

    def enhance_clips(
        self,
        clips: Sequence[Sequence[np.ndarray]],
    ) -> list[list[np.ndarray]]:
        if not clips:
            return []
        if len(clips) == 1:
            return [self.enhance_clip(clips[0])]

        lengths = {len(frames) for frames in clips}
        if len(lengths) != 1 or next(iter(lengths)) < 2:
            return [self.enhance_clip(frames) for frames in clips]

        first = clips[0][0]
        first_shape = first.shape
        first_dtype = np.dtype(first.dtype)
        for frames in clips:
            if any(
                frame.shape != first_shape
                or np.dtype(frame.dtype) != first_dtype
                for frame in frames
            ):
                return [self.enhance_clip(item) for item in clips]
        if first_dtype != np.dtype(np.uint8):
            return [super().enhance_clip(frames) for frames in clips]

        packed = np.stack(
            [np.stack(frames, axis=0) for frames in clips],
            axis=0,
        )
        tensor = torch.from_numpy(packed).permute(0, 1, 4, 2, 3)
        started = time.monotonic()
        enhanced_u8 = self._enhance_u8(tensor)
        self.elapsed += time.monotonic() - started
        self.clips += len(clips)
        return [
            [np.ascontiguousarray(frame) for frame in group]
            for group in enhanced_u8
        ]
