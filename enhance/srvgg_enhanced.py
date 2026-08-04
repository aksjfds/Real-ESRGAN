"""Strict-checkpoint-compatible SRVGG decomposition and residual controls."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from realesrgan.archs.srvgg_arch import SRVGGNetCompact as OfficialSRVGGNetCompact


@dataclass(frozen=True)
class SRVGGComponents:
    residual: torch.Tensor
    nearest_base: torch.Tensor
    official_output: torch.Tensor


class EnhancedSRVGGNetCompact(OfficialSRVGGNetCompact):
    """Official topology with no additional parameters or state-dict keys."""

    def forward_components(self, x: torch.Tensor) -> SRVGGComponents:
        residual = x
        for layer in self.body:
            residual = layer(residual)
        residual = self.upsampler(residual)
        nearest_base = F.interpolate(x, scale_factor=self.upscale, mode="nearest")
        return SRVGGComponents(
            residual=residual,
            nearest_base=nearest_base,
            official_output=residual + nearest_base,
        )

    def forward(
        self,
        x: torch.Tensor,
        enhancement_disabled: bool = False,
    ) -> torch.Tensor:
        # enhancement_disabled is intentionally accepted for compatibility
        # tests. Ordinary forward remains exactly the official computation.
        del enhancement_disabled
        return self.forward_components(x).official_output

    def enhanced_forward(
        self,
        x: torch.Tensor,
        *,
        residual_mode: str = "official",
        residual_strength: float = 1.0,
        residual_flat_strength: float = 0.9,
        residual_edge_strength: float = 1.0,
        residual_edge_low: float = 0.05,
        residual_edge_high: float = 0.20,
        base_correction: float = 0.0,
        lanczos_base: torch.Tensor | None = None,
    ) -> torch.Tensor:
        components = self.forward_components(x)
        if residual_mode == "official":
            strength: float | torch.Tensor = 1.0
        elif residual_mode == "global":
            strength = residual_strength
        elif residual_mode == "adaptive":
            strength = adaptive_residual_strength(
                x.float(),
                components.residual.shape[-2:],
                residual_flat_strength,
                residual_edge_strength,
                residual_edge_low,
                residual_edge_high,
            )
        else:
            raise ValueError(f"Unknown residual mode: {residual_mode}")
        output = components.nearest_base.float() + components.residual.float() * strength
        if base_correction:
            if lanczos_base is None:
                raise ValueError("lanczos_base is required when base_correction is non-zero")
            output = output.float() + base_correction * (
                lanczos_base.float() - components.nearest_base.float()
            )
        return output


def adaptive_residual_strength(
    x: torch.Tensor,
    output_size: tuple[int, int],
    flat_strength: float,
    edge_strength: float,
    edge_low: float,
    edge_high: float,
) -> torch.Tensor:
    """Build a smooth fixed-algorithm edge/variance strength map in FP32."""
    if x.dtype != torch.float32:
        x = x.float()
    luma = 0.2126 * x[:, 0:1] + 0.7152 * x[:, 1:2] + 0.0722 * x[:, 2:3]
    sobel_x = x.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3) / 8
    sobel_y = sobel_x.transpose(-1, -2)
    gx = F.conv2d(luma, sobel_x, padding=1)
    gy = F.conv2d(luma, sobel_y, padding=1)
    gradient = torch.sqrt(gx.square() + gy.square() + 1e-12)
    mean = F.avg_pool2d(luma, 5, stride=1, padding=2)
    variance = F.avg_pool2d(luma.square(), 5, stride=1, padding=2) - mean.square()
    score = gradient + variance.clamp_min(0).sqrt() * 0.5
    denom = max(edge_high - edge_low, 1e-6)
    normalized = ((score - edge_low) / denom).clamp(0, 1)
    normalized = F.avg_pool2d(normalized, 5, stride=1, padding=2)
    normalized = F.interpolate(normalized, size=output_size, mode="bilinear", align_corners=False)
    return flat_strength + normalized * (edge_strength - flat_strength)


def assert_no_extra_state(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    """Strict load helper with an explicit success contract."""
    result = model.load_state_dict(state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(
            f"Strict checkpoint load failed: missing={result.missing_keys}, "
            f"unexpected={result.unexpected_keys}"
        )
