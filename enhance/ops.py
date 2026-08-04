"""Deterministic FP32 tensor operations used by enhancement stages."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import torch
from torch.nn import functional as F


TTA_SPECS: tuple[tuple[bool, bool, bool], ...] = (
    (False, False, False),
    (False, True, False),
    (False, False, True),
    (False, True, True),
    (True, False, False),
    (True, True, False),
    (True, False, True),
    (True, True, True),
)


def tta_transform(x: torch.Tensor, spec: tuple[bool, bool, bool]) -> torch.Tensor:
    transpose, horizontal, vertical = spec
    if transpose:
        x = x.transpose(-2, -1)
    if horizontal:
        x = torch.flip(x, (-1,))
    if vertical:
        x = torch.flip(x, (-2,))
    return x


def tta_inverse(x: torch.Tensor, spec: tuple[bool, bool, bool]) -> torch.Tensor:
    transpose, horizontal, vertical = spec
    if vertical:
        x = torch.flip(x, (-2,))
    if horizontal:
        x = torch.flip(x, (-1,))
    if transpose:
        x = x.transpose(-2, -1)
    return x


class EnsembleEngine:
    def __init__(self, tta_mode: str = "none", tta_batch_size: int = 1, shift_mode: str = "none"):
        self.tta_mode = tta_mode
        self.tta_batch_size = tta_batch_size
        self.shift_mode = shift_mode

    @property
    def tta_count(self) -> int:
        return 8 if self.tta_mode == "x8" else 1

    @property
    def phases(self) -> tuple[tuple[int, int], ...]:
        if self.shift_mode == "x4":
            return ((0, 0), (1, 0), (0, 1), (1, 1))
        if self.shift_mode == "x2":
            return ((0, 0), (1, 1))
        return ((0, 0),)

    @property
    def call_count(self) -> int:
        return self.tta_count * len(self.phases)

    def _tta(self, x: torch.Tensor, model_fn: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
        specs = TTA_SPECS if self.tta_mode == "x8" else TTA_SPECS[:1]
        total: torch.Tensor | None = None
        # Non-square transpose variants cannot share a batch. Sequential mode
        # is the stable default; equally shaped consecutive variants are batched.
        offset = 0
        while offset < len(specs):
            first_shape = tta_transform(x, specs[offset]).shape
            group = [specs[offset]]
            for spec in specs[offset + 1 : offset + self.tta_batch_size]:
                if tta_transform(x, spec).shape != first_shape:
                    break
                group.append(spec)
            transformed = torch.cat([tta_transform(x, spec) for spec in group], dim=0)
            outputs = model_fn(transformed)
            chunks = outputs.chunk(len(group), dim=0)
            for spec, chunk in zip(group, chunks):
                restored = tta_inverse(chunk, spec).float()
                total = restored if total is None else total + restored
            offset += len(group)
        assert total is not None
        return total / len(specs)

    def __call__(
        self,
        x: torch.Tensor,
        model_fn: Callable[[torch.Tensor], torch.Tensor],
        native_scale: int,
    ) -> torch.Tensor:
        if len(self.phases) == 1:
            return self._tta(x, model_fn)
        height, width = x.shape[-2:]
        accumulated: torch.Tensor | None = None
        identity: torch.Tensor | None = None
        max_dx = max(dx for dx, _dy in self.phases)
        max_dy = max(dy for _dx, dy in self.phases)
        valid_h = height * native_scale - max_dy * native_scale
        valid_w = width * native_scale - max_dx * native_scale
        for dx, dy in self.phases:
            padded = F.pad(x, (1, 1, 1, 1), mode="reflect")
            shifted = padded[:, :, 1 - dy : 1 - dy + height, 1 - dx : 1 - dx + width]
            sr = self._tta(shifted, model_fn)
            aligned = torch.roll(sr, shifts=(-dy * native_scale, -dx * native_scale), dims=(-2, -1))
            if dx == 0 and dy == 0:
                identity = aligned.float()
            if accumulated is None:
                accumulated = torch.zeros_like(aligned[:, :, :valid_h, :valid_w], dtype=torch.float32)
            accumulated[:, :, :valid_h, :valid_w] += aligned[:, :, :valid_h, :valid_w].float()
        assert accumulated is not None and identity is not None
        # Every ensemble phase contributes to one identical common region.
        # Borders that cannot be aligned for all phases come from identity,
        # avoiding the previous position-dependent phase count.
        result = identity.clone()
        result[:, :, :valid_h, :valid_w] = accumulated / float(len(self.phases))
        return result


def _lanczos_weights(
    input_size: int,
    output_size: int,
    device: torch.device,
    a: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    scale = output_size / input_size
    support = a / min(scale, 1.0)
    taps = max(2 * a, int(math.ceil(support)) * 2)
    positions = (torch.arange(output_size, device=device, dtype=torch.float32) + 0.5) / scale - 0.5
    left = torch.floor(positions - support + 1).to(torch.long)
    offsets = torch.arange(taps, device=device)
    indices = left[:, None] + offsets[None, :]
    distance = positions[:, None] - indices.float()
    filter_distance = distance * min(scale, 1.0)
    weights = torch.sinc(filter_distance) * torch.sinc(filter_distance / a)
    weights = torch.where(filter_distance.abs() < a, weights, torch.zeros_like(weights))
    weights /= weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
    return indices.clamp(0, input_size - 1), weights


def lanczos_resize(
    x: torch.Tensor,
    size: tuple[int, int],
    a: int = 4,
    chunk_size: int = 128,
) -> torch.Tensor:
    """Separable FP32 Lanczos resize for NCHW tensors."""
    if tuple(x.shape[-2:]) == tuple(size):
        return x.float()
    x = x.float()
    out_h, out_w = size
    ix, wx = _lanczos_weights(x.shape[-1], out_w, x.device, a)
    width_parts = []
    for start in range(0, out_w, chunk_size):
        end = min(start + chunk_size, out_w)
        gathered_x = x[:, :, :, ix[start:end]]
        width_parts.append(
            (gathered_x * wx[start:end].view(1, 1, 1, end - start, -1)).sum(-1)
        )
    x = torch.cat(width_parts, dim=-1)
    iy, wy = _lanczos_weights(x.shape[-2], out_h, x.device, a)
    height_parts = []
    for start in range(0, out_h, chunk_size):
        end = min(start + chunk_size, out_h)
        gathered_y = x[:, :, iy[start:end], :]
        height_parts.append(
            (gathered_y * wy[start:end].view(1, 1, end - start, -1, 1)).sum(-2)
        )
    return torch.cat(height_parts, dim=-2)


def resize_kernel(x: torch.Tensor, size: tuple[int, int], kernel: str) -> torch.Tensor:
    if kernel == "lanczos":
        return lanczos_resize(x, size)
    if kernel == "area":
        return F.interpolate(x.float(), size=size, mode="area")
    if kernel == "bicubic":
        return F.interpolate(x.float(), size=size, mode="bicubic", align_corners=False, antialias=True)
    raise ValueError(f"Unsupported resize kernel: {kernel}")


class BackProjectionRefiner:
    def __init__(self, iterations: int, strength: float, kernel: str, error_clamp: float):
        self.iterations = iterations
        self.strength = strength
        self.kernel = kernel
        self.error_clamp = error_clamp
        self.errors: list[float] = []

    def __call__(self, sr: torch.Tensor, target_lr: torch.Tensor) -> torch.Tensor:
        if self.iterations <= 0:
            self.errors = []
            return sr.float()
        current = sr.float()
        reconstructed = resize_kernel(current, target_lr.shape[-2:], self.kernel)
        initial_error = float(
            (target_lr.float() - reconstructed).square().mean().sqrt().item()
        )
        self.errors = [initial_error]
        for _ in range(self.iterations):
            reconstructed = resize_kernel(current, target_lr.shape[-2:], self.kernel)
            error = target_lr.float() - reconstructed
            limited = self.error_clamp * torch.tanh(error / max(self.error_clamp, 1e-6))
            correction = resize_kernel(limited, current.shape[-2:], self.kernel)
            candidate = (current + self.strength * correction).clamp(-0.05, 1.05)
            candidate_error = resize_kernel(candidate, target_lr.shape[-2:], self.kernel)
            candidate_metric = float((target_lr.float() - candidate_error).square().mean().sqrt().item())
            if candidate_metric > self.errors[-1] + 1e-8:
                break
            current = candidate
            self.errors.append(candidate_metric)
        return current.clamp(0, 1)


def soft_range_compress(
    output: torch.Tensor,
    reference: torch.Tensor,
    strength: float,
    radius: int,
    overshoot: float,
    undershoot: float,
) -> torch.Tensor:
    if strength <= 0:
        return output
    kernel = radius * 2 + 1
    local_max = F.max_pool2d(reference, kernel, stride=1, padding=radius)
    local_min = -F.max_pool2d(-reference, kernel, stride=1, padding=radius)
    mean = F.avg_pool2d(reference, kernel, stride=1, padding=radius)
    variance = F.avg_pool2d(reference.square(), kernel, stride=1, padding=radius) - mean.square()
    std = variance.clamp_min(0).sqrt()
    lower = local_min - undershoot * std
    upper = local_max + overshoot * std
    softness = (std * 0.5 + 1.0 / 255.0).clamp_min(1.0 / 65535.0)
    below = lower - softness * torch.tanh((lower - output) / softness)
    above = upper + softness * torch.tanh((output - upper) / softness)
    compressed = torch.where(output < lower, below, torch.where(output > upper, above, output))
    return torch.lerp(output, compressed, min(strength, 1.0))


def adaptive_dehalo(output: torch.Tensor, strength: float, radius: int) -> torch.Tensor:
    if strength <= 0:
        return output
    kernel = radius * 2 + 1
    smooth = F.avg_pool2d(output, kernel, stride=1, padding=radius)
    luma = 0.2126 * output[:, 0:1] + 0.7152 * output[:, 1:2] + 0.0722 * output[:, 2:3]
    edge = (luma - F.avg_pool2d(luma, 3, stride=1, padding=1)).abs()
    halo = (output - smooth).abs().mean(1, keepdim=True)
    mask = ((halo - 0.01) / 0.04).clamp(0, 1) * ((edge - 0.01) / 0.08).clamp(0, 1)
    corrected = output - (output - smooth) * mask * min(strength, 1.0)
    return corrected


def mask_is_valid(mask: torch.Tensor) -> bool:
    return bool(torch.isfinite(mask).all() and mask.min() >= 0 and mask.max() <= 1)
