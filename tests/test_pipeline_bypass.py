import pytest
import torch

from enhance.pipeline import Preprocessor, RealCUGANBackend


def test_disabled_optional_models_load_no_weights():
    pre = Preprocessor("none", 1.0, "/does/not/exist")
    x = torch.rand(1, 3, 4, 4)
    assert pre(x) is x
    RealCUGANBackend(False, "/does/not/exist")


def test_unverified_optional_models_fail_explicitly():
    with pytest.raises(RuntimeError, match="unavailable"):
        Preprocessor("waifu2x", 1.0, "")
    with pytest.raises(RuntimeError, match="unavailable"):
        RealCUGANBackend(True, "")
