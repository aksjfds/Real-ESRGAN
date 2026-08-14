"""Narrow public compatibility boundary for the legacy inference runtime.

Current v6.x orchestration imports this module instead of reaching into private
attributes of inference.runtime directly. Legacy implementations remain intact.
"""

from __future__ import annotations

from typing import Callable, Type

from . import runtime as _legacy


MODEL_URLS = _legacy.MODEL_URLS
VideoInfo = _legacy.VideoInfo
WorkerConfig = _legacy.WorkerConfig
RawVideoReader = _legacy.RawVideoReader

format_seconds = _legacy.format_seconds
require_binary = _legacy.require_binary
probe_video = _legacy.probe_video
resolve_range = _legacy.resolve_range
parse_gpu_ids = _legacy.parse_gpu_ids
resolve_model_paths = _legacy.resolve_model_paths
load_worker_model = _legacy.load_worker_model
infer_frame = _legacy.infer_frame
mux_original_audio = _legacy.mux_original_audio


def model_native_scale(name: str) -> int:
    return _legacy._model_native_scale(name)


def output_pixel_format(codec: str, bit_depth: int) -> str:
    return _legacy._output_pixel_format(codec, bit_depth)


def get_encoding_backend() -> tuple[Callable[[str, str], None], Type]:
    require_encoder = _legacy._require_encoder
    writer_type = _legacy._writer_type
    if require_encoder is None or writer_type is None:
        raise RuntimeError(
            "Encoding backend is not configured. Run through root inference.py."
        )
    return require_encoder, writer_type
