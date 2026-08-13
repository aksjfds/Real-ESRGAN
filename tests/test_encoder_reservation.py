from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from inference.encoder_reservation import reserve_nvenc_for_workers


class FakeWriter:
    instances: list["FakeWriter"] = []
    events: list[str] = []

    def __init__(self, path, ffmpeg_bin, width, height, input_fps, output_fps, codec, crf, preset, cq, nvenc_preset, encode_gpu):
        self.path = Path(path)
        self.width = int(width)
        self.height = int(height)
        self.codec = str(codec)
        self.encode_gpu = int(encode_gpu)
        self.frames: list[np.ndarray] = []
        self.closed = False
        FakeWriter.instances.append(self)

    def write(self, frame: np.ndarray) -> None:
        role = "reservation" if "nvenc-reservation" in self.path.name else "real"
        FakeWriter.events.append(f"{role}-write")
        self.frames.append(np.array(frame, copy=True))

    def close(self) -> None:
        role = "reservation" if "nvenc-reservation" in self.path.name else "real"
        FakeWriter.events.append(f"{role}-close")
        self.closed = True


class EncoderReservationTests(unittest.TestCase):
    def setUp(self) -> None:
        FakeWriter.instances.clear()
        FakeWriter.events.clear()

    def _real_writer(self, path: Path) -> FakeWriter:
        return FakeWriter(path, "ffmpeg", 8, 6, "25", "25", "hevc_nvenc", 23, "medium", 18, "p7", 0)

    def test_nvenc_reservation_lives_until_first_real_frame(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            output = Path(raw) / "video.mp4"
            real = self._real_writer(output)
            wrapped = reserve_nvenc_for_workers(
                real,
                writer_type=FakeWriter,
                temp_video=output,
                ffmpeg_bin="ffmpeg",
                width=8,
                height=6,
                fps_rate="25",
                codec="hevc_nvenc",
                crf=23,
                preset="medium",
                cq=18,
                nvenc_preset="p7",
                encode_gpu=0,
                worker_gpu_ids=[0, 1],
                dtype=np.dtype(np.uint8),
            )
            self.assertEqual(len(FakeWriter.instances), 2)
            reservation = FakeWriter.instances[1]
            self.assertFalse(reservation.closed)
            self.assertEqual(len(reservation.frames), 2)
            self.assertEqual(reservation.frames[0].shape, (6, 8, 3))
            self.assertEqual(reservation.frames[0].dtype, np.uint8)
            self.assertEqual(FakeWriter.events, ["reservation-write", "reservation-write"])

            wrapped.write(np.ones((6, 8, 3), dtype=np.uint8))
            self.assertEqual(
                FakeWriter.events,
                ["reservation-write", "reservation-write", "reservation-close", "real-write"],
            )
            self.assertTrue(reservation.closed)
            wrapped.close()
            self.assertEqual(FakeWriter.events[-1], "real-close")

    def test_non_nvenc_or_separate_encode_gpu_is_untouched(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            output = Path(raw) / "video.mp4"
            real = self._real_writer(output)
            same = reserve_nvenc_for_workers(
                real, writer_type=FakeWriter, temp_video=output, ffmpeg_bin="ffmpeg",
                width=8, height=6, fps_rate="25", codec="libx265", crf=23,
                preset="medium", cq=18, nvenc_preset="p7", encode_gpu=0,
                worker_gpu_ids=[0, 1], dtype=np.dtype(np.uint8),
            )
            self.assertIs(same, real)
            same = reserve_nvenc_for_workers(
                real, writer_type=FakeWriter, temp_video=output, ffmpeg_bin="ffmpeg",
                width=8, height=6, fps_rate="25", codec="hevc_nvenc", crf=23,
                preset="medium", cq=18, nvenc_preset="p7", encode_gpu=2,
                worker_gpu_ids=[0, 1], dtype=np.dtype(np.uint8),
            )
            self.assertIs(same, real)
            self.assertEqual(len(FakeWriter.instances), 1)


if __name__ == "__main__":
    unittest.main()
