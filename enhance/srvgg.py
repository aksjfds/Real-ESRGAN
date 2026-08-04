"""Strict-checkpoint-compatible official SRVGG model helper."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from realesrgan.archs.srvgg_arch import SRVGGNetCompact as OfficialSRVGGNetCompact


class SRVGGNetCompact(OfficialSRVGGNetCompact):
    """Official topology with the existing FP32 residual/base recombination."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        for layer in self.body:
            residual = layer(residual)
        residual = self.upsampler(residual)
        nearest_base = F.interpolate(x, scale_factor=self.upscale, mode="nearest")
        return residual.float() + nearest_base.float()


def assert_no_extra_state(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    result = model.load_state_dict(state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(
            f"Strict checkpoint load failed: missing={result.missing_keys}, "
            f"unexpected={result.unexpected_keys}"
        )


__all__ = ["SRVGGNetCompact", "assert_no_extra_state"]
