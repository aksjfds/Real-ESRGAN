import numpy as np
import torch

from enhance.pipeline import FrameEnhancementPipeline, PipelineConfig
from enhance.srvgg_enhanced import EnhancedSRVGGNetCompact


def test_writer_boundary_receives_float32_rgb():
    model = EnhancedSRVGGNetCompact(3, 3, 4, 1, 4, "prelu").eval()
    config = PipelineConfig(fp16=False, final_scale=2, native_scale=4)
    pipeline = FrameEnhancementPipeline(model, torch.device("cpu"), config)
    output = pipeline.enhance_batch([np.zeros((6, 7, 3), dtype=np.uint8)])[0]
    assert output.dtype == np.float32
    assert output.shape == (12, 14, 3)
    assert np.isfinite(output).all()
    assert 0 <= output.min() <= output.max() <= 1
