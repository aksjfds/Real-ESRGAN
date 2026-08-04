"""Composable enhancement stages for the Real-ESRGAN video runner."""

import sys

from . import basicvsrpp_quality as _basicvsrpp_quality

# Keep the public BasicVSR++ import path stable while routing runtime
# composition through the quality wrapper. The wrapper keeps the original
# network/checkpoint implementation and only changes frame composition.
sys.modules[f"{__name__}.basicvsrpp"] = _basicvsrpp_quality

from .basicvsrpp_quality import (
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
