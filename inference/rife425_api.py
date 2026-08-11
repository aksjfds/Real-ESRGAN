"""Public compatibility surface for Practical-RIFE 4.25 primitives."""

from __future__ import annotations

from .rife425 import (
    _IFNet425 as IFNet425,
    _load_state as load_rife425_state,
    resolve_rife425_weights,
)

__all__ = (
    "IFNet425",
    "load_rife425_state",
    "resolve_rife425_weights",
)
