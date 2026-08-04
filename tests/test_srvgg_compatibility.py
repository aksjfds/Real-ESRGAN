import os

import pytest
import torch

from enhance.srvgg_enhanced import EnhancedSRVGGNetCompact
from realesrgan.archs.srvgg_arch import SRVGGNetCompact as OfficialSRVGGNetCompact


def make(cls):
    return cls(3, 3, 64, 16, 4, "prelu")


def test_strict_state_and_disabled_output_match():
    official = make(OfficialSRVGGNetCompact).eval()
    enhanced = make(EnhancedSRVGGNetCompact).eval()
    state = official.state_dict()
    official.load_state_dict(state, strict=True)
    enhanced.load_state_dict(state, strict=True)
    x = torch.rand(1, 3, 8, 9)
    torch.testing.assert_close(
        official(x), enhanced(x, enhancement_disabled=True), rtol=1e-5, atol=1e-6
    )


def test_real_checkpoint_strict_when_provided():
    path = os.environ.get("REALESRGAN_ANIMEVIDEO_CHECKPOINT")
    if not path:
        pytest.skip("set REALESRGAN_ANIMEVIDEO_CHECKPOINT for downloaded-checkpoint test")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state = checkpoint.get("params_ema", checkpoint.get("params"))
    make(OfficialSRVGGNetCompact).load_state_dict(state, strict=True)
    make(EnhancedSRVGGNetCompact).load_state_dict(state, strict=True)
