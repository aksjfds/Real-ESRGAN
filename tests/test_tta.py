import torch

from enhance.ops import TTA_SPECS, EnsembleEngine, tta_inverse, tta_transform


def test_all_tta_transforms_are_exactly_invertible():
    x = torch.arange(2 * 3 * 5 * 7).reshape(2, 3, 5, 7)
    for spec in TTA_SPECS:
        assert torch.equal(tta_inverse(tta_transform(x, spec), spec), x)


def test_x8_identity_model_is_identity():
    x = torch.rand(1, 3, 7, 9)
    result = EnsembleEngine("x8", 2, "none")(x, lambda value: value, 1)
    torch.testing.assert_close(result, x)
