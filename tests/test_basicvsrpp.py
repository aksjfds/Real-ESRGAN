from __future__ import annotations

import importlib.util
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "enhance" / "basicvsrpp.py"
SPEC = importlib.util.spec_from_file_location("basicvsrpp_standalone", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
basicvsrpp = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = basicvsrpp
SPEC.loader.exec_module(basicvsrpp)


class ListReader:
    def __init__(self, frames: list[np.ndarray]):
        self.frames = frames
        self.index = 0
        self.closed = False

    def read(self):
        if self.index >= len(self.frames):
            return None
        frame = self.frames[self.index]
        self.index += 1
        return frame

    def close(self):
        self.closed = True


class IdentityPreprocessor:
    def __init__(self, config):
        self.config = config
        self.elapsed = 0.0
        self.clips = 0
        self.tiles = 0
        self.closed = False

    def enhance_clip(self, frames):
        self.clips += 1
        return [basicvsrpp.frame_to_float_rgb(frame) for frame in frames]

    def close(self):
        self.closed = True


def numbered_frames(count: int) -> list[np.ndarray]:
    return [np.full((8, 8, 3), index, dtype=np.uint8) for index in range(count)]


def test_stream_reader_emits_every_frame_once():
    config = basicvsrpp.BasicVSRPPConfig(
        clip_length=7,
        clip_overlap=2,
        scene_threshold=0,
    )
    source = ListReader(numbered_frames(20))
    preprocessor = IdentityPreprocessor(config)
    reader = basicvsrpp.BasicVSRPPStreamReader(source, preprocessor)
    values = []
    while True:
        frame = reader.read()
        if frame is None:
            break
        values.append(int(round(float(frame[0, 0, 0] * 255))))
    assert values == list(range(20))
    reader.close()
    assert source.closed and preprocessor.closed


def test_stream_reader_resets_at_scene_cut():
    config = basicvsrpp.BasicVSRPPConfig(
        clip_length=5,
        clip_overlap=2,
        scene_threshold=0.10,
    )
    dark = [np.zeros((16, 16, 3), np.uint8) for _ in range(4)]
    bright = [np.full((16, 16, 3), 255, np.uint8) for _ in range(4)]
    reader = basicvsrpp.BasicVSRPPStreamReader(
        ListReader(dark + bright), IdentityPreprocessor(config)
    )
    outputs = []
    while True:
        frame = reader.read()
        if frame is None:
            break
        outputs.append(frame)
    assert len(outputs) == 8
    assert reader.scene_cuts == 1
    assert max(float(frame.mean()) for frame in outputs[:4]) == 0.0
    assert min(float(frame.mean()) for frame in outputs[4:]) == 1.0


def test_checkpoint_prefix_normalization():
    model = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1))
    expected = model.state_dict()
    raw = {f"generator.{key}": value.clone() for key, value in expected.items()}
    normalized = basicvsrpp.normalize_basicvsrpp_state(raw, model)
    assert set(normalized) == set(expected)


def test_dcnv2_operator_shape():
    layer = basicvsrpp.SecondOrderDeformableAlignment(
        in_channels=32,
        out_channels=16,
        kernel_size=3,
        padding=1,
        deform_groups=4,
    )
    x = torch.rand(1, 32, 8, 8)
    extra = torch.rand(1, 3 * 16, 8, 8)
    flow1 = torch.zeros(1, 2, 8, 8)
    flow2 = torch.zeros(1, 2, 8, 8)
    result = layer(x, extra, flow1, flow2)
    assert result.shape == (1, 16, 8, 8)
    assert torch.isfinite(result).all()


def test_invalid_clip_overlap_is_rejected():
    base = basicvsrpp.BasicVSRPPConfig()
    # Validation occurs before checkpoint/device setup.
    with pytest.raises(ValueError, match="clip overlap"):
        basicvsrpp.BasicVSRPPPreprocessor(
            replace(base, clip_length=4, clip_overlap=2),
            model=nn.Identity(),
        )


def test_tiny_basicvsrpp_forward_same_resolution():
    model = basicvsrpp.BasicVSRPlusPlusNet(
        mid_channels=16,
        num_blocks=1,
        is_low_res_input=False,
    ).eval()
    clip = torch.rand(1, 2, 3, 256, 256)
    with torch.inference_mode():
        output = model(clip)
    assert output.shape == clip.shape
    assert torch.isfinite(output).all()
