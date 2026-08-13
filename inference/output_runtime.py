"""Explicit output resize, progress, and fail-fast encode-pump runtime."""

from __future__ import annotations

from collections import deque
import os
import queue
import sys
import threading
import time
from typing import Deque

import cv2
import numpy as np
from tqdm import tqdm as _tqdm

_HEARTBEAT_INTERVAL = 60.0
_RATE_WINDOW = 120.0
_QUEUE_PUT_TIMEOUT = 0.25


class _InteractiveTqdm(_tqdm):
    def __init__(self, *args, **kwargs):
        kwargs["file"] = sys.stderr
        kwargs["disable"] = False
        kwargs["miniters"] = 1
        kwargs["mininterval"] = min(float(kwargs.get("mininterval", 0.5)), 0.5)
        kwargs["maxinterval"] = min(float(kwargs.get("maxinterval", 2.0)), 2.0)
        super().__init__(*args, **kwargs)


class _SilentTqdm(_tqdm):
    def __init__(self, *args, **kwargs):
        kwargs["disable"] = True
        super().__init__(*args, **kwargs)


def create_progress(total: int):
    interactive = (
        os.environ.get("KAGGLE_KERNEL_RUN_TYPE", "").strip().lower()
        == "interactive"
    )
    cls = _InteractiveTqdm if interactive else _SilentTqdm
    return cls(
        total=int(total),
        desc="Real-ESRGAN",
        unit="frame",
        dynamic_ncols=True,
        mininterval=1.0,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}{postfix}]",
    )


def _format_duration(seconds: float) -> str:
    if seconds < 0 or seconds == float("inf"):
        return "--:--"
    value = max(0, int(round(seconds)))
    hours, value = divmod(value, 3600)
    minutes, secs = divmod(value, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def resize_frame(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    if frame.shape[1] == width and frame.shape[0] == height:
        return np.ascontiguousarray(frame)
    return np.ascontiguousarray(
        cv2.resize(
            frame,
            (width, height),
            interpolation=cv2.INTER_LANCZOS4,
        )
    )


class OutputPump:
    """Encode from explicit per-GPU SR output slots.

    Each GPU has two shared SR buffers. The writer may retain one slot while the
    SR worker fills the other, so resize/encoder backpressure no longer blocks
    every next SR inference and no extra native-resolution CPU staging copy is
    required.
    """

    def __init__(
        self,
        workers,
        writer,
        width: int,
        height: int,
        progress,
        started: float,
        heartbeat_interval: float = _HEARTBEAT_INTERVAL,
    ) -> None:
        self.workers = workers
        self.writer = writer
        self.width = int(width)
        self.height = int(height)
        self.progress = progress
        self.started = float(started)
        self.queue: queue.Queue[tuple[int, int, int] | None] = queue.Queue(
            maxsize=max(2, workers.count * workers.sr_output_buffers)
        )
        self.processed = 0
        self.resize_seconds = 0.0
        self.write_seconds = 0.0
        self.error: BaseException | None = None
        self.traceback_text = ""
        self.first_output_at: float | None = None
        self.rate_history: Deque[tuple[int, float]] = deque()

        self._heartbeat_interval = max(1.0, float(heartbeat_interval))
        self._heartbeat_stop = threading.Event()
        self._heartbeat_closed = False
        self._batch_mode = bool(getattr(progress, "disable", False))
        self._process_lock = threading.Lock()

        self.thread = threading.Thread(
            target=self._run,
            name="realesrgan-output",
            daemon=True,
        )
        self.thread.start()
        self._heartbeat_thread = None
        if self._batch_mode:
            self._heartbeat_thread = threading.Thread(
                target=self._heartbeat_run,
                name="realesrgan-progress-log",
                daemon=True,
            )
            self._heartbeat_thread.start()

    def _stable_rate(self, now: float) -> float:
        self.rate_history.append((self.processed, now))
        while (
            len(self.rate_history) > 2
            and now - self.rate_history[0][1] > _RATE_WINDOW
        ):
            self.rate_history.popleft()
        if len(self.rate_history) >= 2:
            old_count, old_time = self.rate_history[0]
            span = now - old_time
            if span >= 20.0 and self.processed > old_count:
                return (self.processed - old_count) / span
        if self.first_output_at is None:
            return 0.0
        return max(
            0.0,
            (self.processed - 1) / max(now - self.first_output_at, 1e-6),
        )

    def _process_item(self, item: tuple[int, int, int]) -> None:
        _frame_id, worker_id, output_slot = item
        try:
            mark = time.monotonic()
            frame = resize_frame(
                self.workers.output(worker_id, output_slot),
                self.width,
                self.height,
            )
            self.resize_seconds += time.monotonic() - mark

            mark = time.monotonic()
            self.writer.write(frame)
            self.write_seconds += time.monotonic() - mark

            self.processed += 1
            now = time.monotonic()
            if self.first_output_at is None:
                self.first_output_at = now
            self.progress.update(1)
            rate = self._stable_rate(now)
            total = int(self.progress.total or 0)
            remaining = max(0, total - self.processed)
            eta = remaining / rate if rate > 1e-9 else float("inf")
            self.progress.set_postfix_str(
                f"{rate:.3f} frame/s | ETA {_format_duration(eta)}",
                refresh=False,
            )
        finally:
            self.workers.release(worker_id, output_slot)

    def encoder_handoff_worker_id(self) -> int | None:
        if not bool(getattr(self.writer, "handoff_pending", False)):
            return None
        gpu_id = getattr(self.writer, "handoff_gpu_id", None)
        if gpu_id is None:
            return None
        try:
            return self.workers.gpu_ids.index(int(gpu_id))
        except ValueError:
            return None

    def handoff_first_output(
        self,
        frame_id: int,
        worker_id: int,
        output_slot: int,
    ) -> None:
        """Atomically trim the shared GPU then start the real encoder on frame 0."""
        target = self.encoder_handoff_worker_id()
        if target is None:
            raise RuntimeError("NVENC handoff requested without a shared encode GPU")
        self.check()
        self.workers.trim_cuda_cache(target)
        print(
            f"[encode] cuda:{self.workers.gpu_ids[target]} workers idle/cache-trimmed; "
            "performing NVENC handoff",
            flush=True,
        )
        with self._process_lock:
            self._process_item((int(frame_id), int(worker_id), int(output_slot)))
        self.check()

    def _run(self) -> None:
        import traceback

        try:
            while True:
                item = self.queue.get()
                if item is None:
                    self.queue.task_done()
                    break
                try:
                    with self._process_lock:
                        self._process_item(item)
                finally:
                    self.queue.task_done()
        except Exception as error:
            self.error = error
            self.traceback_text = traceback.format_exc()

    def _heartbeat_emit(self, final: bool = False) -> None:
        now = time.monotonic()
        total = int(self.progress.total or 0)
        rate = self._stable_rate(now)
        percent = 100.0 * self.processed / total if total > 0 else 0.0
        remaining = max(0, total - self.processed)
        eta = remaining / rate if rate > 1e-9 else float("inf")
        speed = (
            f"{rate:.3f} frame/s"
            if rate > 1e-9
            else "waiting for output"
        )
        suffix = (
            " | done"
            if final and total > 0 and self.processed >= total
            else ""
        )
        print(
            f"[progress] {self.processed}/{total or '?'} | "
            f"{percent:.1f}% | {speed} | "
            f"elapsed {_format_duration(now - self.started)} | "
            f"ETA {_format_duration(eta)}{suffix}",
            flush=True,
        )

    def _heartbeat_run(self) -> None:
        self._heartbeat_emit()
        while not self._heartbeat_stop.wait(self._heartbeat_interval):
            self._heartbeat_emit()

    def _stop_heartbeat(self, final: bool) -> None:
        if self._heartbeat_closed:
            return
        self._heartbeat_closed = True
        self._heartbeat_stop.set()
        if (
            self._heartbeat_thread is not None
            and self._heartbeat_thread.is_alive()
        ):
            self._heartbeat_thread.join(timeout=2.0)
        if self._batch_mode:
            self._heartbeat_emit(final=final)

    def check(self) -> None:
        if self.error is not None:
            raise RuntimeError(
                f"Output pipeline failed: {self.error!r}\n{self.traceback_text}"
            ) from self.error

    def _enqueue(self, item: tuple[int, int, int] | None) -> None:
        """Enqueue without allowing a dead output thread to strand the scheduler."""
        while True:
            self.check()
            if not self.thread.is_alive():
                raise RuntimeError("Output pipeline thread exited unexpectedly")
            try:
                self.queue.put(item, timeout=_QUEUE_PUT_TIMEOUT)
                return
            except queue.Full:
                continue

    def put(self, frame_id: int, worker_id: int, output_slot: int) -> None:
        self._enqueue((int(frame_id), int(worker_id), int(output_slot)))
        self.check()

    def finish(self) -> None:
        self.check()
        self._enqueue(None)
        self.thread.join()
        self.check()
        self._stop_heartbeat(final=True)

    def stop(self) -> None:
        self._stop_heartbeat(final=False)
        if not self.thread.is_alive():
            return
        try:
            self._enqueue(None)
        except Exception:
            return
        self.thread.join(timeout=10)
