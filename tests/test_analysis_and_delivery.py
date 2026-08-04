from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load_file_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_analysis_common_video_size():
    analysis = load_file_module("analysis_standalone", ROOT / "enhance" / "analysis.py")
    frame = np.random.default_rng(7).integers(0, 256, (1080, 1920, 3), dtype=np.uint8)
    metrics = analysis.SourceAnalyzer.metrics(frame, 0.0)
    for value in vars(metrics).values():
        assert np.isfinite(value)


def test_safe_preset_removed_and_basicvsrpp_exposed():
    script = (ROOT / "realesrgan.py").read_text()
    assert "--quality-preset" not in script
    assert '"safe"' not in script
    for option in (
        "--basicvsrpp",
        "--basicvsrpp-track",
        "--basicvsrpp-clip-length",
        "--basicvsrpp-tile-size",
    ):
        assert option in script


def test_notebook_has_one_basicvsrpp_run():
    notebook = json.loads((ROOT / "realesrgan.ipynb").read_text())
    text = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    assert "QUALITY_PRESET" not in text
    assert "RUN_BOTH_10S_TESTS" not in text
    assert "BASICVSRPP = True" in text
    assert "--basicvsrpp-track" in text
