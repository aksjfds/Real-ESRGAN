import torch
from torch.nn import functional as F

from enhance.ops import EnsembleEngine, TTA_SPECS, tta_inverse, tta_transform


def test_tta_round_trip() -> None:
    x = torch.arange(2 * 3 * 5 * 7).reshape(2, 3, 5, 7)
    for spec in TTA_SPECS:
        assert torch.equal(tta_inverse(tta_transform(x, spec), spec), x)


def test_shift_common_region_and_identity_border() -> None:
    x = torch.rand(1, 3, 9, 11)
    engine = EnsembleEngine(tta_mode="none", shift_mode="x4")

    def model_fn(value: torch.Tensor) -> torch.Tensor:
        return F.interpolate(value, scale_factor=4, mode="nearest")

    result = engine(x, model_fn, native_scale=4)
    identity = model_fn(x).float()
    assert result.shape == identity.shape
    # Right/bottom borders outside the common phase region come from identity.
    assert torch.equal(result[:, :, -4:, :], identity[:, :, -4:, :])
    assert torch.equal(result[:, :, :, -4:], identity[:, :, :, -4:])
