import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from enhance.pipeline import FrameEnhancementPipeline, PipelineConfig
from enhance.tiles import full_frame_lanczos


class Nearest4x(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, scale_factor=4, mode="nearest")


def run(pre_pad: int) -> np.ndarray:
    frame = np.random.default_rng(4).integers(0, 256, (19, 23, 3), dtype=np.uint8)
    pipeline = FrameEnhancementPipeline(
        Nearest4x(),
        torch.device("cpu"),
        PipelineConfig(native_scale=4, pre_pad=pre_pad, fp16=False, channels_last=False),
    )
    native = pipeline.enhance_batch([frame])[0]
    assert native.shape == (19 * 4, 23 * 4, 3)
    return full_frame_lanczos(native, round(23 * 2.5), round(19 * 2.5))


def test_prepad_is_cropped_before_noninteger_lanczos() -> None:
    no_pad = run(0)
    with_pad = run(3)
    np.testing.assert_allclose(with_pad, no_pad, rtol=0, atol=0)
