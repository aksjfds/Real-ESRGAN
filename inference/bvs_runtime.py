"""Current explicit BasicVSR++ runtime with direct shared-slot output support."""

from __future__ import annotations

import time
from typing import Sequence

import numpy as np
import torch
from torch.nn import functional as F

from . import basicvsrpp_api as api
from .frame_transport import PinnedH2DStager
from .optimized_basicvsrpp import OptimizedBasicVSRPPPreprocessor


class BasicVSRRuntime(OptimizedBasicVSRPPPreprocessor):
    """Preserve v5.8 math while keeping uint8 BVS results on CUDA."""

    def _stage_h2d(self, clip_cpu: torch.Tensor) -> torch.Tensor:
        stager = getattr(self, "_pinned_h2d_stager", None)
        if stager is None:
            stager = PinnedH2DStager(self.device)
            self._pinned_h2d_stager = stager
        return stager.copy(clip_cpu)

    def _run_model_device(self, clip_gpu: torch.Tensor) -> torch.Tensor:
        padded, original_h, original_w = api.pad_to_model_size(clip_gpu)
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

    def _enhance_u8_with_tile_device(
        self,
        clip_cpu: torch.Tensor,
        tile_size: int,
    ) -> torch.Tensor:
        original_u8 = self._stage_h2d(clip_cpu)
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
        return (
            mixed.clamp_(0.0, 1.0)
            .mul_(255.0)
            .round_()
            .to(torch.uint8)
            .permute(0, 1, 3, 4, 2)
            .contiguous()
        )

    def _enhance_u8_device(self, clip_cpu: torch.Tensor) -> torch.Tensor:
        requested = int(self.tile_size)
        candidates = [requested]
        for fallback in (384, 320, 256):
            if fallback < requested and fallback not in candidates:
                candidates.append(fallback)

        last_error: BaseException | None = None
        for tile_size in candidates:
            try:
                output = self._enhance_u8_with_tile_device(
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
                    f"[basicvsrpp] tile={tile_size} OOM; retrying smaller tile",
                    flush=True,
                )

        raise RuntimeError(
            "BasicVSR++ ran out of GPU memory even at tile=256"
        ) from last_error

    @staticmethod
    def _common_clip_format(
        clips: Sequence[Sequence[np.ndarray]],
    ) -> tuple[tuple[int, ...], np.dtype] | None:
        if not clips or not clips[0]:
            return None
        shape = clips[0][0].shape
        dtype = np.dtype(clips[0][0].dtype)
        for frames in clips:
            if not frames:
                return None
            if any(
                frame.shape != shape or np.dtype(frame.dtype) != dtype
                for frame in frames
            ):
                return None
        return shape, dtype

    def enhance_clips_device(
        self,
        clips: Sequence[Sequence[np.ndarray]],
    ) -> list[torch.Tensor] | None:
        """Return uint8 restored clips on CUDA; None requests CPU fallback."""
        common = self._common_clip_format(clips)
        if common is None:
            return None
        _shape, dtype = common
        if dtype != np.dtype(np.uint8):
            return None

        lengths = {len(frames) for frames in clips}
        if len(lengths) != 1 or next(iter(lengths)) < 2:
            return None

        packed = np.stack(
            [np.stack(frames, axis=0) for frames in clips],
            axis=0,
        )
        tensor = torch.from_numpy(packed).permute(0, 1, 4, 2, 3)
        started = time.monotonic()
        enhanced = self._enhance_u8_device(tensor)
        self.elapsed += time.monotonic() - started
        self.clips += len(clips)
        return [enhanced[index] for index in range(enhanced.shape[0])]

    def enhance_clip(self, frames: Sequence[np.ndarray]) -> list[np.ndarray]:
        device_groups = self.enhance_clips_device((frames,))
        if device_groups is None:
            return api.BasicVSRPPPreprocessor.enhance_clip(self, frames)
        array = device_groups[0].cpu().numpy()
        return [np.ascontiguousarray(frame) for frame in array]

    def enhance_clips(
        self,
        clips: Sequence[Sequence[np.ndarray]],
    ) -> list[list[np.ndarray]]:
        device_groups = self.enhance_clips_device(clips)
        if device_groups is None:
            return [
                api.BasicVSRPPPreprocessor.enhance_clip(self, frames)
                for frames in clips
            ]
        return [
            [np.ascontiguousarray(frame) for frame in group.cpu().numpy()]
            for group in device_groups
        ]
