import torch
from torch.nn import functional as F

from enhance.ops import EnsembleEngine


def test_shift_modes_preserve_shape_and_coordinates():
    x = torch.rand(1, 3, 8, 10)
    model = lambda value: F.interpolate(value, scale_factor=4, mode="nearest")
    expected = model(x)
    for mode in ("x2", "x4"):
        result = EnsembleEngine("none", 1, mode)(x, model, 4)
        assert result.shape == expected.shape
        torch.testing.assert_close(result[:, :, 4:-4, 4:-4], expected[:, :, 4:-4, 4:-4])
