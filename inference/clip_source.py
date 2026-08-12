"""CPU-only scene-aware BasicVSR++ clip assembly with one-item prefetch."""

from __future__ import annotations

import queue
import threading
import time

import numpy as np

from .scene_metrics import (
    SceneSignature,
    scene_difference_from_signatures,
    scene_signature,
)


_ClipTask = tuple[list[np.ndarray], int, int] | None
_PREFETCH_DEPTH = 1


class ClipSource:
    """Assemble overlapping temporal clips and prefetch one task ahead."""

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
        self._prefetch_queue: queue.Queue[tuple[str, object]] = queue.Queue(
            maxsize=_PREFETCH_DEPTH
        )
        self._prefetch_stop = threading.Event()
        self._prefetch_thread = threading.Thread(
            target=self._prefetch_loop,
            name="bvs-clip-prefetch",
            daemon=True,
        )
        self._prefetch_thread.start()

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

    def _next_task_sync(self) -> _ClipTask:
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

    def _publish_prefetch(self, kind: str, payload: object) -> bool:
        while not self._prefetch_stop.is_set():
            try:
                self._prefetch_queue.put((kind, payload), timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def _prefetch_loop(self) -> None:
        try:
            while not self._prefetch_stop.is_set():
                task = self._next_task_sync()
                if not self._publish_prefetch("task", task):
                    return
                if task is None:
                    return
        except BaseException as error:
            if not self._prefetch_stop.is_set():
                self._publish_prefetch("error", error)

    def next_task(self) -> _ClipTask:
        if self.closed:
            raise RuntimeError("ClipSource is closed.")
        kind, payload = self._prefetch_queue.get()
        if kind == "error":
            if isinstance(payload, BaseException):
                raise RuntimeError("BVS clip prefetch failed.") from payload
            raise RuntimeError(f"BVS clip prefetch failed: {payload!r}")
        if kind != "task":
            raise RuntimeError(f"Unknown BVS prefetch message: {kind!r}")
        return payload  # type: ignore[return-value]

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self._prefetch_stop.set()

        while True:
            try:
                self._prefetch_queue.get_nowait()
            except queue.Empty:
                break

        try:
            self.reader.close()
        finally:
            self._prefetch_thread.join(timeout=5.0)
            while True:
                try:
                    self._prefetch_queue.get_nowait()
                except queue.Empty:
                    break
