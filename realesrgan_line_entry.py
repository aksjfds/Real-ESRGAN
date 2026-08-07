#!/usr/bin/env python3
"""AutoDL v5.0: v4.2 runtime plus selective low-contrast line enhancement."""

from __future__ import annotations

import argparse

import realesrgan_fast_entry as v42
from enhance.low_contrast_lines import (
    LowContrastLineConfig,
    LowContrastLineEnhancer,
)

base = v42.base
fast = v42.fast
_original_parser = fast.build_parser
_original_validate = base.validate_args
_original_process_video = base.process_video
_OriginalRawVideoWriter = base.RawVideoWriter
_ACTIVE_CONFIG = LowContrastLineConfig(enabled=False)


def _parser() -> argparse.ArgumentParser:
    parser = _original_parser()
    parser.description = (
        "AutoDL RTX 4090 v4.2 pipeline with selective low-contrast line enhancement."
    )
    parser.add_argument(
        "--low-contrast-lines",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enhance coherent weak lines while hard-protecting strong edges",
    )
    parser.add_argument("--line-strength", type=float, default=1.0)
    parser.add_argument("--line-gradient-min", type=float, default=0.004)
    parser.add_argument("--line-gradient-max", type=float, default=0.025)
    parser.add_argument("--line-protect-gradient", type=float, default=0.040)
    parser.add_argument("--line-coherence", type=float, default=0.45)
    parser.add_argument("--line-guided-radius", type=int, default=4)
    parser.add_argument("--line-guided-eps", type=float, default=4.0e-4)
    parser.add_argument("--line-temporal", type=float, default=0.65)
    parser.add_argument("--line-max-delta", type=float, default=0.025)
    return parser


def _config(args: argparse.Namespace) -> LowContrastLineConfig:
    return LowContrastLineConfig(
        enabled=bool(args.low_contrast_lines),
        strength=float(args.line_strength),
        gradient_min=float(args.line_gradient_min),
        gradient_max=float(args.line_gradient_max),
        protect_gradient=float(args.line_protect_gradient),
        coherence_min=float(args.line_coherence),
        guided_radius=int(args.line_guided_radius),
        guided_eps=float(args.line_guided_eps),
        temporal=float(args.line_temporal),
        max_delta=float(args.line_max_delta),
    )


def _validate(args: argparse.Namespace) -> None:
    _original_validate(args)
    _config(args).validate()


class _LineAwareWriter(_OriginalRawVideoWriter):
    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self._line_enhancer = LowContrastLineEnhancer(_ACTIVE_CONFIG)
        self._line_reported = False

    def write(self, frame):
        return super().write(self._line_enhancer.enhance(frame))

    def close(self) -> None:
        super().close()
        if not self._line_reported and _ACTIVE_CONFIG.enabled:
            print(
                f"[low-contrast-lines] {self._line_enhancer.summary()}",
                flush=True,
            )
            self._line_reported = True


def _process_video(args: argparse.Namespace) -> None:
    global _ACTIVE_CONFIG
    _ACTIVE_CONFIG = _config(args)
    if _ACTIVE_CONFIG.enabled:
        print(
            "[low-contrast-lines] enabled, "
            f"strength={_ACTIVE_CONFIG.strength:g}, "
            f"gradient={_ACTIVE_CONFIG.gradient_min:g}-"
            f"{_ACTIVE_CONFIG.gradient_max:g}, "
            f"protect>={_ACTIVE_CONFIG.protect_gradient:g}, "
            f"coherence>={_ACTIVE_CONFIG.coherence_min:g}, "
            f"radius={_ACTIVE_CONFIG.guided_radius}, "
            f"temporal={_ACTIVE_CONFIG.temporal:g}, "
            f"max_delta={_ACTIVE_CONFIG.max_delta:g}",
            flush=True,
        )
    else:
        print("[low-contrast-lines] disabled", flush=True)
    _original_process_video(args)


fast.build_parser = _parser
base.validate_args = _validate
base.RawVideoWriter = _LineAwareWriter
base.process_video = _process_video


if __name__ == "__main__":
    fast.mp.freeze_support()
    fast.main()
