from enhance.tiles import verify_dimensions


def test_fractional_scale_dimensions_use_global_rounding():
    assert verify_dimensions(17, 13, 1.5) == (26, 20)
    assert verify_dimensions(1920, 1080, 2.0) == (3840, 2160)
