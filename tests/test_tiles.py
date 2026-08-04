import numpy as np
import cv2

from enhance.tiles import TileProcessor, finalize_frame


def dummy_native(tile: np.ndarray, scale: int = 4) -> np.ndarray:
    return cv2.resize(tile.astype(np.float32) / 255.0, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)


def test_tile_native_stitch_matches_full_nearest_and_coverage() -> None:
    frame = np.random.default_rng(9).integers(0, 256, (37, 53, 3), dtype=np.uint8)
    processor = TileProcessor(tile_size=16, tile_pad=3, pre_pad=5, scale=4, verify=True)
    patches, regions = processor.split(frame)
    outputs = {region.index: dummy_native(patch) for patch, region in zip(patches, regions)}
    stitched = processor.stitch(outputs, regions, frame.shape[1], frame.shape[0])
    expected = dummy_native(frame)
    np.testing.assert_allclose(stitched, expected, rtol=0, atol=0)


def test_shared_finalizer_is_deterministic() -> None:
    frame = np.random.default_rng(1).integers(0, 256, (24, 32, 3), dtype=np.uint8)
    native = dummy_native(frame)
    a = finalize_frame(native, frame.astype(np.float32) / 255.0, 64, 48, 0.1, 2, 0.1, 2, 1.0, 1.0)
    b = finalize_frame(native, frame.astype(np.float32) / 255.0, 64, 48, 0.1, 2, 0.1, 2, 1.0, 1.0)
    np.testing.assert_allclose(a, b, rtol=0, atol=0)
    assert a.dtype == np.float32
    assert a.shape == (48, 64, 3)


def test_noninteger_scale_after_native_stitch_matches_full_frame() -> None:
    frame = np.random.default_rng(11).integers(0, 256, (31, 47, 3), dtype=np.uint8)
    processor = TileProcessor(tile_size=13, tile_pad=4, pre_pad=3, scale=4, verify=True)
    patches, regions = processor.split(frame)
    outputs = {region.index: dummy_native(patch) for patch, region in zip(patches, regions)}
    stitched_native = processor.stitch(outputs, regions, frame.shape[1], frame.shape[0])
    full_native = dummy_native(frame)
    target_w, target_h = round(frame.shape[1] * 2.5), round(frame.shape[0] * 2.5)
    tiled = finalize_frame(stitched_native, frame.astype(np.float32) / 255.0, target_w, target_h, 0, 2, 0, 2, 1, 1)
    full = finalize_frame(full_native, frame.astype(np.float32) / 255.0, target_w, target_h, 0, 2, 0, 2, 1, 1)
    np.testing.assert_allclose(tiled, full, rtol=0, atol=0)
