import torch

from enhance.ops import BackProjectionRefiner, resize_kernel


def test_back_projection_does_not_increase_reconstruction_error():
    lr = torch.rand(1, 3, 8, 9)
    sr = resize_kernel(lr, (32, 36), "lanczos") * 0.9
    before = (resize_kernel(sr, lr.shape[-2:], "lanczos") - lr).square().mean()
    refiner = BackProjectionRefiner(3, 0.2, "lanczos", 0.05)
    refined = refiner(sr, lr)
    after = (resize_kernel(refined, lr.shape[-2:], "lanczos") - lr).square().mean()
    assert after <= before + 1e-8
    assert all(next_error <= error + 1e-8 for error, next_error in zip(refiner.errors, refiner.errors[1:]))
