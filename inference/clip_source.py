"""CPU-only scene-aware BasicVSR++ clip assembly."""

from __future__ import annotations

import time

import numpy as np

from .scene_metrics import scene_difference


class ClipSource:
    """Assemble overlapping temporal clips without owning any CUDA state."""

    def __init__(
        self,
        reader,
        clip_length: int,
        overlap: int,
        scene_threshold: float,
    ) -> None:
        self.reader = reader
        self.clip_length = int(clip_length)
        self.overlap = int(overlap)
        self.scene_threshold = float(scene_threshold)

        self.buffer: list[np.ndarray] = []
        self.pending: np.ndarray | None = None
        self.eof = False
        self.segment_end = False
        self.first_chunk = True
        self.decode_elapsed = 0.0
        self.scene_cuts = 0
        self.closed = False

    def _read_source(self) -> np.ndarray | None:
        started = time.monotonic()
        frame = self.reader.read()
        self.decode_elapsed += time.monotonic() - started
        return frame

    def next_task(self) -> tuple[list[np.ndarray], int, int] | None:
        while True:
            if self.segment_end or self.eof:
                if self.buffer:
                    frames = list(self.buffer)
                    emit_start = 0 if self.first_chunk else self.overlap
                    emit_end = len(frames)
                    self.buffer = []
                    self.first_chunk = True
                    self.segment_end = False
                    if self.pending is not None:
                        self.buffer.append(self.pending)
                        self.pending = None
                    return frames, emit_start, emit_end

                self.segment_end = False
                if self.pending is not None:
                    self.buffer.append(self.pending)
                    self.pending = None
                    continue
                if self.eof:
                    return None

            frame = self._read_source()
            if frame is None:
                self.eof = True
                continue

            if (
                self.buffer
                and self.scene_threshold > 0
                and scene_difference(self.buffer[-1], frame) >= self.scene_threshold
            ):
                self.pending = frame
                self.segment_end = True
                self.scene_cuts += 1
                continue

            self.buffer.append(frame)
            if len(self.buffer) != self.clip_length:
                continue

            frames = list(self.buffer)
            if self.first_chunk:
                emit_start = 0
                emit_end = self.clip_length - self.overlap
                self.first_chunk = False
            else:
                emit_start = self.overlap
                emit_end = self.clip_length - self.overlap

            retain = 2 * self.overlap
            self.buffer = self.buffer[-retain:] if retain else []
            return frames, emit_start, emit_end

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.reader.close()
