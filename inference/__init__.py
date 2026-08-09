"""Inference package for the retained Real-ESRGAN video runtime."""

from __future__ import annotations

import sys

from . import models as _models

# The retained 2.7 runtime still imports the historical package path
# ``realesrgan.archs``. Keep that import working internally while the repository
# itself uses the clearer ``inference`` naming.
_this_module = sys.modules[__name__]
sys.modules.setdefault("realesrgan", _this_module)
sys.modules.setdefault("realesrgan.archs", _models)
sys.modules.setdefault(
    "realesrgan.archs.srvgg_arch",
    sys.modules[f"{__name__}.models.srvgg_arch"],
)
setattr(_this_module, "archs", _models)

from .models import SRVGGNetCompact

__all__ = ["SRVGGNetCompact"]
