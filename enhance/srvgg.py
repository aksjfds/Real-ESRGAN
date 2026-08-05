"""Strict-checkpoint-compatible SRVGG model used by Real-ESRGAN.

The project intentionally does not require the installable ``realesrgan``
Python package.  This module vendors the small official SRVGGNetCompact
architecture so Kaggle only needs the dependencies listed in requirements.txt.
The module/parameter names match the official implementation exactly, which
preserves strict checkpoint compatibility.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class SRVGGNetCompact(nn.Module):
    """Official compact VGG-style super-resolution topology.

    The layer layout and state-dict keys match
    ``realesrgan.archs.srvgg_arch.SRVGGNetCompact``.  The residual/base
    recombination remains FP32, as required by the existing video pipeline.
    """

    def __init__(
        self,
        num_in_ch: int = 3,
        num_out_ch: int = 3,
        num_feat: int = 64,
        num_conv: int = 16,
        upscale: int = 4,
        act_type: str = "prelu",
    ) -> None:
        super().__init__()
        self.num_in_ch = num_in_ch
        self.num_out_ch = num_out_ch
        self.num_feat = num_feat
        self.num_conv = num_conv
        self.upscale = upscale
        self.act_type = act_type

        def activation() -> nn.Module:
            if act_type == "relu":
                return nn.ReLU(inplace=True)
            if act_type == "prelu":
                return nn.PReLU(num_parameters=num_feat)
            if act_type == "leakyrelu":
                return nn.LeakyReLU(negative_slope=0.1, inplace=True)
            raise ValueError(
                "act_type must be one of: relu, prelu, leakyrelu; "
                f"received {act_type!r}"
            )

        self.body = nn.ModuleList()
        self.body.append(nn.Conv2d(num_in_ch, num_feat, 3, 1, 1))
        self.body.append(activation())
        for _ in range(num_conv):
            self.body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            self.body.append(activation())
        self.body.append(
            nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1)
        )
        self.upsampler = nn.PixelShuffle(upscale)

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
