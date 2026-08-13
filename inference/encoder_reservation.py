"""Reserve an NVENC session while GPU inference workers establish memory policy."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


class ReservedNVENCWriter:
    """Swap a primed temporary NVENC session for the real writer on first frame."""

    def __init__(
        self,
        writer: Any,
        reservation_writer: Any,
        reservation_path: Path,
        codec: str,
        encode_gpu: int,
    ) -> None:
        self._writer = writer
        self._reservation_writer = reservation_writer
        self._reservation_path = Path(reservation_path)
        self._codec = str(codec)
        self._encode_gpu = int(encode_gpu)
        self._reservation_released = False

    def _release_reservation(self) -> None:
        if self._reservation_released:
            return
        self._reservation_released = True
        reservation = self._reservation_writer
        self._reservation_writer = None
        try:
            if reservation is not None:
                reservation.close()
        finally:
            self._reservation_path.unlink(missing_ok=True)
        print(
            f"[encode] released primed {self._codec} reservation on "
            f"cuda:{self._encode_gpu}; starting real output session",
            flush=True,
        )

    def write(self, frame: np.ndarray) -> None:
        # Keep the real NVENC footprint unavailable to inference until the first
        # actual output frame is ready, then hand that footprint directly to the
        # real writer with no dummy frame entering the final video.
        self._release_reservation()
        self._writer.write(frame)

    def close(self) -> None:
        reservation_error: BaseException | None = None
        try:
            self._release_reservation()
        except BaseException as error:
            reservation_error = error
        try:
            self._writer.close()
        except BaseException:
            if reservation_error is not None:
                raise reservation_error
            raise
        if reservation_error is not None:
            raise reservation_error


def reserve_nvenc_for_workers(
    writer: Any,
    *,
    writer_type: type,
    temp_video: Path,
    ffmpeg_bin: str,
    width: int,
    height: int,
    fps_rate: str,
    codec: str,
    crf: int,
    preset: str,
    cq: int,
    nvenc_preset: str,
    encode_gpu: int,
    worker_gpu_ids: tuple[int, ...] | list[int],
    dtype: np.dtype,
) -> Any:
    """Prime and hold the exact NVENC session when it shares a worker GPU."""
    encode_gpu = int(encode_gpu)
    worker_ids = {int(value) for value in worker_gpu_ids}
    if not str(codec).endswith('_nvenc') or encode_gpu not in worker_ids:
        return writer

    reservation_path = Path(temp_video).with_name(
        f".{Path(temp_video).name}.nvenc-reservation.mp4"
    )
    reservation_path.unlink(missing_ok=True)
    reservation = writer_type(
        reservation_path,
        ffmpeg_bin,
        int(width),
        int(height),
        str(fps_rate),
        str(fps_rate),
        codec,
        int(crf),
        preset,
        int(cq),
        nvenc_preset,
        encode_gpu,
    )

    # Two full-size frames make FFmpeg consume more than a pipe buffer and force
    # the exact output encoder session to be opened before worker model startup.
    # The session then remains idle but alive, so its buffers stay resident.
    sample = np.zeros((int(height), int(width), 3), dtype=np.dtype(dtype))
    try:
        reservation.write(sample)
        reservation.write(sample)
    except BaseException:
        try:
            reservation.close()
        except Exception:
            pass
        reservation_path.unlink(missing_ok=True)
        raise
    finally:
        del sample

    print(
        f"[encode] primed {codec} reservation on cuda:{encode_gpu} | "
        f"{width}x{height} | held until first real output frame",
        flush=True,
    )
    return ReservedNVENCWriter(
        writer,
        reservation,
        reservation_path,
        codec,
        encode_gpu,
    )
