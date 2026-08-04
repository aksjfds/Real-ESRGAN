import numpy as np

from enhance.tiles import TileProcessor


def test_tiles_write_every_output_pixel_once():
    frame = np.arange(13 * 17 * 3, dtype=np.uint8).reshape(13, 17, 3)
    processor = TileProcessor(8, 2, 1, 1.5, verify=True)
    patches, regions = processor.split(frame)
    outputs = {}
    for patch, region in zip(patches, regions):
        outputs[region.index] = np.zeros(
            (round(patch.shape[0] * 1.5), round(patch.shape[1] * 1.5), 3), dtype=np.float32
        )
    output = processor.stitch(outputs, regions, frame.shape[1], frame.shape[0])
    assert output.shape == (round(13 * 1.5), round(17 * 1.5), 3)
    assert np.isfinite(output).all()
