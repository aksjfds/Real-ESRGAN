"""Explicit Practical-RIFE 4.25 runtime optimized for shared-frame slots."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.nn import functional as F

from .rife425_api import IFNet425, load_rife425_state


class OptimizedRIFE425Interpolator:
    def __init__(self, gpu_id: int, weights: Path) -> None:
        self.gpu_id = int(gpu_id)
        self.device = torch.device(f"cuda:{self.gpu_id}")
        self.model = IFNet425().eval().requires_grad_(False)
        state, ignored_teacher, ignored_caltime = load_rife425_state(weights)
        try:
            self.model.load_state_dict(state, strict=True)
        except RuntimeError as error:
            raise RuntimeError(
                "RIFE 4.25 inference checkpoint mismatch after filtering known "
                "training-only teacher.* / caltime.* weights"
            ) from error

        self.dtype = torch.float16
        with torch.cuda.device(self.device):
            self.model.to(self.device, dtype=self.dtype)

        self.elapsed = 0.0
        self.frames = 0
        ignored = []
        if ignored_teacher:
            ignored.append(f"teacher={ignored_teacher}")
        if ignored_caltime:
            ignored.append(f"caltime={ignored_caltime}")
        suffix = (
            f" | ignored_training={'/'.join(ignored)}"
            if ignored
            else ""
        )
        print(
            f"[rife] Practical-RIFE 4.25 loaded on cuda:{self.gpu_id} "
            f"(FP16){suffix}",
            flush=True,
        )

    def _compact_to_cuda(self, frame: np.ndarray) -> tuple[torch.Tensor, float]:
        if frame.dtype == np.uint8:
            scale = 255.0
            tensor = (
                torch.from_numpy(frame)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .to(self.device, non_blocking=True)
            )
            tensor = tensor.to(dtype=self.dtype)
            tensor.div_(scale)
            return tensor, scale

        if frame.dtype.kind == "u" and frame.dtype.itemsize == 2:
            scale = 65535.0
            array = frame.astype(np.float32) / scale
            tensor = (
                torch.from_numpy(array)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .to(self.device, dtype=self.dtype, non_blocking=True)
            )
            return tensor, scale

        raise TypeError(
            f"RIFE supports uint8/uint16 frames, got {frame.dtype}"
        )

    def interpolate_into(
        self,
        frame0: np.ndarray,
        frame1: np.ndarray,
        timesteps: Sequence[float],
        output_view: np.ndarray,
        output_slots: Sequence[int],
    ) -> int:
        if not timesteps:
            return 0
        if len(timesteps) != len(output_slots):
            raise ValueError("RIFE timesteps/output slots must have equal length")
        if frame0.shape != frame1.shape or frame0.dtype != frame1.dtype:
            raise ValueError("RIFE frame pairs must have identical shape/dtype")

        started = time.monotonic()
        ta, scale_value = self._compact_to_cuda(frame0)
        tb, _ = self._compact_to_cuda(frame1)
        h, w = frame0.shape[:2]
        ph = ((h - 1) // 128 + 1) * 128
        pw = ((w - 1) // 128 + 1) * 128
        if ph != h or pw != w:
            ta = F.pad(ta, (0, pw - w, 0, ph - h))
            tb = F.pad(tb, (0, pw - w, 0, ph - h))

        with torch.inference_mode():
            pair = torch.cat((ta, tb), 1)
            for timestep, slot in zip(timesteps, output_slots):
                out = self.model(pair, float(timestep))[0, :, :h, :w]
                out = out.clamp_(0.0, 1.0).mul_(scale_value).round()

                if frame0.dtype == np.uint8:
                    out_u8 = (
                        out.to(torch.uint8)
                        .permute(1, 2, 0)
                        .contiguous()
                    )
                    target = torch.from_numpy(output_view[int(slot)])
                    target.copy_(out_u8, non_blocking=False)
                else:
                    array = (
                        out.to(torch.int32)
                        .permute(1, 2, 0)
                        .contiguous()
                        .cpu()
                        .numpy()
                        .astype(frame0.dtype, copy=False)
                    )
                    np.copyto(
                        output_view[int(slot)],
                        array,
                        casting="no",
                    )

        self.elapsed += time.monotonic() - started
        self.frames += len(timesteps)
        return len(timesteps)

    def close(self) -> None:
        if hasattr(self, "model"):
            del self.model
        with torch.cuda.device(self.device):
            torch.cuda.empty_cache()
