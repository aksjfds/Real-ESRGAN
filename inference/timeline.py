"""Pure output-timeline planning for source-FPS to target-FPS interpolation."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from .scene_metrics import (
    SceneSignature,
    scene_difference_from_signatures,
    scene_signature,
)


@dataclass(frozen=True)
class IntervalPlan:
    direct_targets: tuple[int, ...]
    rife_targets: tuple[int, ...]
    timesteps: tuple[float, ...]


def ceil_fraction(value: Fraction) -> int:
    return -(-value.numerator // value.denominator)


def targets_for_source(
    source_id: int,
    source_rate: Fraction,
    output_rate: Fraction,
    expected_output: int,
) -> tuple[list[int], list[float]]:
    ratio = output_rate / source_rate
    start = max(0, ceil_fraction(Fraction(source_id) * ratio))
    end = min(
        expected_output,
        ceil_fraction(Fraction(source_id + 1) * ratio),
    )

    targets: list[int] = []
    timesteps: list[float] = []
    for target in range(start, end):
        alpha = Fraction(target, 1) / ratio - source_id
        targets.append(target)
        timesteps.append(
            float(max(Fraction(0), min(Fraction(1), alpha)))
        )
    return targets, timesteps


class TimelinePlanner:
    """Convert restored source intervals into direct frames or RIFE jobs."""

    def __init__(
        self,
        source_rate: Fraction,
        output_rate: Fraction,
        expected_output: int,
        duplicate_threshold: float = 0.002,
        scene_threshold: float = 0.30,
    ) -> None:
        self.source_rate = source_rate
        self.output_rate = output_rate
        self.expected_output = int(expected_output)
        self.duplicate_threshold = float(duplicate_threshold)
        self.scene_threshold = float(scene_threshold)
        self.rife_enabled = output_rate > source_rate

        self._cached_signature_source: int | None = None
        self._cached_signature: SceneSignature | None = None

    def _interval_difference(
        self,
        source_id: int,
        current: np.ndarray,
        nxt: np.ndarray,
    ) -> float:
        if (
            self._cached_signature_source == source_id
            and self._cached_signature is not None
        ):
            current_signature = self._cached_signature
        else:
            current_signature = scene_signature(current)

        next_signature = scene_signature(nxt)
        self._cached_signature_source = source_id + 1
        self._cached_signature = next_signature
        return scene_difference_from_signatures(
            current_signature,
            next_signature,
        )

    def plan_interval(
        self,
        source_id: int,
        current: np.ndarray,
        nxt: np.ndarray,
    ) -> IntervalPlan:
        targets, timesteps = targets_for_source(
            source_id,
            self.source_rate,
            self.output_rate,
            self.expected_output,
        )

        direct_targets: list[int] = []
        rife_targets: list[int] = []
        rife_times: list[float] = []
        for target, alpha in zip(targets, timesteps):
            if alpha <= 1e-8:
                direct_targets.append(target)
            else:
                rife_targets.append(target)
                rife_times.append(alpha)

        if not rife_targets:
            return IntervalPlan(tuple(direct_targets), (), ())

        difference = self._interval_difference(source_id, current, nxt)
        if (
            not self.rife_enabled
            or difference <= self.duplicate_threshold
            or difference >= self.scene_threshold
        ):
            direct_targets.extend(rife_targets)
            return IntervalPlan(tuple(direct_targets), (), ())

        return IntervalPlan(
            tuple(direct_targets),
            tuple(rife_targets),
            tuple(rife_times),
        )

    def final_targets(self, source_id: int) -> tuple[int, ...]:
        targets, _ = targets_for_source(
            source_id,
            self.source_rate,
            self.output_rate,
            self.expected_output,
        )
        return tuple(targets)
