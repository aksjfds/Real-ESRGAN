"""Minimal stubs for testing the custom enhancement modules outside the upstream repo."""

from __future__ import annotations

import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch import nn
from torch.nn import functional as F


class StubSRVGGNetCompact(nn.Module):
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
        del num_feat, num_conv, act_type
        self.upscale = upscale
        # The custom class only requires iterable body and an upsampler.
        self.body = nn.ModuleList([nn.Identity()])
        self.upsampler = nn.Identity()
        self.num_in_ch = num_in_ch
        self.num_out_ch = num_out_ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, scale_factor=self.upscale, mode="nearest")


package = types.ModuleType("realesrgan")
package.__path__ = []  # type: ignore[attr-defined]
archs = types.ModuleType("realesrgan.archs")
archs.__path__ = []  # type: ignore[attr-defined]
srvgg = types.ModuleType("realesrgan.archs.srvgg_arch")
srvgg.SRVGGNetCompact = StubSRVGGNetCompact
sys.modules.setdefault("realesrgan", package)
sys.modules.setdefault("realesrgan.archs", archs)
sys.modules.setdefault("realesrgan.archs.srvgg_arch", srvgg)

# BasicSR stub so the top-level runner can be imported for non-model unit tests.
basicsr = types.ModuleType("basicsr")
basicsr.__path__ = []  # type: ignore[attr-defined]
basicsr_archs = types.ModuleType("basicsr.archs")
basicsr_archs.__path__ = []  # type: ignore[attr-defined]
rrdb = types.ModuleType("basicsr.archs.rrdbnet_arch")
rrdb.RRDBNet = StubSRVGGNetCompact
sys.modules.setdefault("basicsr", basicsr)
sys.modules.setdefault("basicsr.archs", basicsr_archs)
sys.modules.setdefault("basicsr.archs.rrdbnet_arch", rrdb)
