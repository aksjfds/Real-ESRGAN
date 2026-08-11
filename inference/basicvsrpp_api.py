"""Public compatibility surface for BasicVSR++ model/runtime primitives."""

from __future__ import annotations

from . import basicvsrpp as _legacy

BasicVSRPPConfig = _legacy.BasicVSRPPConfig
BasicVSRPPPreprocessor = _legacy.BasicVSRPPPreprocessor
BasicVSRPlusPlusNet = _legacy.BasicVSRPlusPlusNet
SPyNet = _legacy.SPyNet
load_checkpoint = _legacy.load_checkpoint
download_checkpoint = _legacy.download_checkpoint
pad_to_model_size = _legacy._pad_to_model_size
