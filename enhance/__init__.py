"""Composable enhancement stages for the Real-ESRGAN video runner."""

import sys

from . import basicvsrpp_stage2_guard as _basicvsrpp_stage2_guard

# Keep the public BasicVSR++ import path stable while routing runtime
# composition through the quality-preserving stage-two wrapper and its
# full-runtime-memory safety guard. The original network/checkpoint and the
# stage-one spatial/temporal fusion remain unchanged.
sys.modules[f"{__name__}.basicvsrpp"] = _basicvsrpp_stage2_guard

from .basicvsrpp_stage2_guard import (
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
