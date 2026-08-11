"""Structural interfaces for GPU task runtimes."""

from __future__ import annotations

from typing import Callable, Protocol, Sequence

import numpy as np
import torch


class BasicVSRExecutor(Protocol):
    tiles: int
    clips: int
    tile_size: int

    def enhance_clip(self, frames: Sequence[np.ndarray]) -> list[np.ndarray]: ...

    def enhance_clips(
        self,
        clips: Sequence[Sequence[np.ndarray]],
    ) -> list[list[np.ndarray]]: ...

    def enhance_clips_device(
        self,
        clips: Sequence[Sequence[np.ndarray]],
    ) -> list[torch.Tensor] | None: ...

    def close(self) -> None: ...


class RIFEExecutor(Protocol):
    def interpolate_into(
        self,
        frame0: np.ndarray,
        frame1: np.ndarray,
        timesteps: Sequence[float],
        output_view: np.ndarray,
        output_slots: Sequence[int],
        *,
        compute_done: Callable[[], None] | None = None,
    ) -> int: ...

    def close(self) -> None: ...
