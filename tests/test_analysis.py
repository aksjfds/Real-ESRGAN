import numpy as np

from enhance.analysis import SourceAnalyzer


def test_eight_pixel_boundary_slices_support_exact_multiples():
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    metrics = SourceAnalyzer.metrics(frame, 0.0)
    assert np.isfinite(metrics.jpeg_block)


def test_eight_pixel_boundary_slices_support_non_multiples():
    frame = np.zeros((17, 19, 3), dtype=np.uint8)
    metrics = SourceAnalyzer.metrics(frame, 0.0)
    assert np.isfinite(metrics.jpeg_block)
