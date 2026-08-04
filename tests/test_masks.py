import torch

from enhance.ops import mask_is_valid, soft_range_compress
from enhance.srvgg_enhanced import adaptive_residual_strength


def test_adaptive_mask_is_finite_bounded_and_smooth():
    x = torch.rand(1, 3, 16, 16)
    mask = adaptive_residual_strength(x, (64, 64), 0.0, 1.0, 0.02, 0.2)
    assert mask_is_valid(mask)
    assert (mask[:, :, :, 1:] - mask[:, :, :, :-1]).abs().max() < 0.5


def test_range_limit_has_no_nan():
    reference = torch.rand(1, 3, 16, 16)
    output = reference * 1.3 - 0.1
    result = soft_range_compress(output, reference, 0.1, 2, 1.0, 1.0)
    assert torch.isfinite(result).all()
