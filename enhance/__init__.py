"""Composable enhancement stages for the Real-ESRGAN video runner."""

import sys

from . import basicvsrpp_stage2 as _basicvsrpp_stage2

# Keep the public BasicVSR++ import path stable while routing runtime
# composition through the quality-preserving stage-two wrapper. The wrapper
# keeps the original network/checkpoint and stage-one spatial/temporal fusion,
# then auto-selects only geometry that is at least as strong as the prior
# 7-frame/512-tile quality baseline.
sys.modules[f"{__name__}.basicvsrpp"] = _basicvsrpp_stage2

from .basicvsrpp_stage2 import (
    BASICVSRPP_TRACK_URLS,
    BasicVSRPPConfig,
    BasicVSRPPPreprocessor,
    BasicVSRPPStreamReader,
    BasicVSRPlusPlusNet,
)
from .pipeline import FrameEnhancementPipeline, PipelineConfig
from .srvgg import SRVGGNetCompact, assert_no_extra_state

__all__ = [
    "BASICVSRPP_TRACK_URLS",
    "BasicVSRPPConfig",
    "BasicVSRPPPreprocessor",
    "BasicVSRPPStreamReader",
    "BasicVSRPlusPlusNet",
    "FrameEnhancementPipeline",
    "PipelineConfig",
    "SRVGGNetCompact",
    "assert_no_extra_state",
]
