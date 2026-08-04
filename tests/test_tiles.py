import numpy as np

from enhance.tiles import TileProcessor, full_frame_lanczos


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


def test_pre_pad_is_global_and_tile_pad_is_per_tile_context():
    frame = np.zeros((9, 11, 3), dtype=np.uint8)
    processor = TileProcessor(8, tile_pad=2, pre_pad=3, scale=4, verify=True)
    patches, regions = processor.split(frame)
    assert regions[0].context == 2
    # Global dimensions are 15×17; the first 8×8 core receives only tile_pad.
    assert patches[0].shape[:2] == (12, 12)
    outputs = {
        region.index: np.zeros((patch.shape[0] * 4, patch.shape[1] * 4, 3), np.float32)
        for patch, region in zip(patches, regions)
    }
    stitched = processor.stitch(outputs, regions, 11, 9)
    assert stitched.shape == (36, 44, 3)


def test_native_stitch_then_one_full_frame_lanczos():
    native = np.random.default_rng(0).random((52, 68, 3), dtype=np.float32)
    final = full_frame_lanczos(native, 34, 26)
    assert final.dtype == np.float32
    assert final.shape == (26, 34, 3)
