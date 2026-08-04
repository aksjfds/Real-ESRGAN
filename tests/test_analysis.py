import numpy as np

from enhance.analysis import SourceAnalyzer


def test_common_video_dimensions_do_not_broadcast_fail() -> None:
    for width, height in ((1920, 1080), (1280, 720), (16, 16), (15, 15)):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        metrics = SourceAnalyzer.metrics(frame, 0.0)
        assert np.isfinite(metrics.jpeg_block)
        assert np.isfinite(metrics.ringing)
