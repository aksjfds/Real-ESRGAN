from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def load_runner():
    path = Path(__file__).resolve().parents[1] / "realesrgan.py"
    spec = importlib.util.spec_from_file_location("realesrgan_video_runner", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_color_metadata_defaults_and_filter_chain() -> None:
    runner = load_runner()
    info = runner.VideoInfo(
        width=1920,
        height=1080,
        fps_num=24000,
        fps_den=1001,
        duration=10.0,
        frames=240,
        has_audio=True,
        color_range=None,
        color_space=None,
        color_primaries=None,
        color_transfer=None,
        chroma_location=None,
    )
    spec = runner.resolve_color_spec(info, "preserve", "reject")
    assert spec.range == "tv"
    assert spec.space == "bt709"
    assert spec.primaries == "bt709"
    assert spec.transfer == "bt709"
    chain = runner.color_filter_chain(spec, "yuv420p10le", [])
    text = ",".join(chain)
    assert "in_range=pc" in text
    assert "out_color_matrix=bt709" in text
    assert "format=yuv420p10le" in text


def test_hdr_rejected_by_default() -> None:
    runner = load_runner()
    info = runner.VideoInfo(
        width=3840,
        height=2160,
        fps_num=24,
        fps_den=1,
        duration=1.0,
        frames=24,
        has_audio=False,
        color_range="tv",
        color_space="bt2020nc",
        color_primaries="bt2020",
        color_transfer="smpte2084",
        chroma_location="left",
    )
    try:
        runner.resolve_color_spec(info, "preserve", "reject")
    except ValueError as error:
        assert "HDR" in str(error)
    else:
        raise AssertionError("HDR input should be rejected by default")
