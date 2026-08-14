"""Compatibility facade for core model and media runtime primitives."""

from __future__ import annotations

import argparse
from typing import Callable, Optional, Protocol, Type

import numpy as np

from .runtime_media import (
    RawVideoReader,
    VideoInfo,
    format_seconds,
    mux_original_audio,
    output_pixel_format as _output_pixel_format,
    parse_gpu_ids,
    parse_rate,
    probe_video,
    require_binary,
    resolve_range,
    run_checked,
)
from .runtime_models import (
    MODEL_URLS,
    WorkerConfig,
    build_model,
    checkpoint_state,
    download_file,
    infer_frame,
    load_worker_model,
    model_native_scale as _model_native_scale,
    resolve_model_paths,
    torch_load_cpu,
)


class VideoWriter(Protocol):
    def write(self, frame: np.ndarray) -> None: ...
    def close(self) -> None: ...


_RequireEncoder = Callable[[str, str], None]
_WriterType = Type[VideoWriter]
_require_encoder: Optional[_RequireEncoder] = None
_writer_type: Optional[_WriterType] = None


def set_encoding_backend(require_encoder_fn: _RequireEncoder, writer_type: _WriterType) -> None:
    global _require_encoder, _writer_type
    _require_encoder = require_encoder_fn
    _writer_type = writer_type


def validate_args(args: argparse.Namespace) -> None:
    if args.scale <= 0:
        raise ValueError("--scale must be positive.")
