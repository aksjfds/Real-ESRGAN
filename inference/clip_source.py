"""CPU-only scene-aware BasicVSR++ clip assembly."""

from __future__ import annotations

import time

import numpy as np

from .scene_metrics import (
    SceneSignature,
    scene_difference_from_signatures,
    scene_signature,
)


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

        self._last_signature: SceneSignature | None = None
        self._pending_signature: SceneSignature | None = None

    def _read_source(self) -> np.ndarray | None:
        started = time.monotonic()
        frame = self.reader.read()
        self.decode_elapsed += time.monotonic() - started
        return frame

    def _adopt_pending(self) -> None:
        if self.pending is None:
            self._last_signature = None
            return
        self.buffer.append(self.pending)
        self.pending = None
        self._last_signature = self._pending_signature
        self._pending_signature = None

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
                    self._adopt_pending()
                    return frames, emit_start, emit_end

                self.segment_end = False
                if self.pending is not None:
                    self._adopt_pending()
                    continue
                if self.eof:
                    return None

            frame = self._read_source()
            if frame is None:
                self.eof = True
                continue

            current_signature = (
                scene_signature(frame)
                if self.scene_threshold > 0
                else None
            )
            if self.buffer and self.scene_threshold > 0:
                previous_signature = self._last_signature
                if previous_signature is None:
                    previous_signature = scene_signature(self.buffer[-1])
                assert current_signature is not None
                difference = scene_difference_from_signatures(
                    previous_signature,
                    current_signature,
                )
                if difference >= self.scene_threshold:
                    self.pending = frame
                    self._pending_signature = current_signature
                    self.segment_end = True
                    self.scene_cuts += 1
                    continue

            self.buffer.append(frame)
            self._last_signature = current_signature
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
            if not self.buffer:
                self._last_signature = None
            return frames, emit_start, emit_end

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.reader.close()
