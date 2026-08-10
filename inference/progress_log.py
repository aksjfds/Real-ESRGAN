"""Durable newline progress logging for saved cloud notebook logs."""
from __future__ import annotations

from collections import deque
import os
import sys
import threading
import time
from typing import Deque, Tuple

from . import pipeline

_INSTALLED = False
_LOG_INTERVAL = 10.0
_RATE_WINDOW = 120.0


def _format_duration(seconds: float) -> str:
    if seconds < 0 or seconds == float("inf"):
        return "--:--"
    value = max(0, int(round(seconds)))
    hours, value = divmod(value, 3600)
    minutes, secs = divmod(value, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _durable_write(line: str) -> None:
    """Write a real newline record, bypassing TextIO/tqdm carriage-return buffering."""
    payload = ("\n" + line + "\n").encode("utf-8", errors="replace")
    try:
        fd = sys.stdout.fileno()
        os.write(fd, payload)
        return
    except Exception:
        pass
    print(line, flush=True)


def _is_kaggle_interactive() -> bool:
    """Return True only for Kaggle's foreground interactive notebook session."""
    return os.environ.get("KAGGLE_KERNEL_RUN_TYPE", "").strip().lower() == "interactive"


def install_persistent_progress(interval: float = _LOG_INTERVAL) -> None:
    """Install one heartbeat logger around whichever OutputPump is active."""
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True

    # The original v5.3 PersistentTqdm logger prints real newline records from
    # set_postfix(), which collides with the interactive carriage-return bar.
    # Disable that path everywhere; batch logging is handled by the heartbeat.
    persistent_tqdm = getattr(pipeline, "PersistentTqdm", None)
    if persistent_tqdm is not None and hasattr(persistent_tqdm, "_persistent_log"):
        persistent_tqdm._persistent_log = lambda *args, **kwargs: None

    # Kaggle exposes the run mode to child processes too. In the foreground
    # Interactive session, leave stdout/tqdm completely alone so the live bar
    # can repaint normally. Save & Run uses Batch and continues below to install
    # the durable newline heartbeat for the persisted online logs.
    if _is_kaggle_interactive():
        return

    base_cls = pipeline.OutputPump
    if getattr(base_cls, "_persistent_heartbeat_wrapped", False):
        return

    class PersistentProgressOutputPump(base_cls):
        _persistent_heartbeat_wrapped = True

        def __init__(self, *args, **kwargs):
            self._heartbeat_interval = max(1.0, float(interval))
            self._heartbeat_stop = threading.Event()
            self._heartbeat_closed = False
            self._heartbeat_samples: Deque[Tuple[int, float]] = deque()
            super().__init__(*args, **kwargs)
            self._heartbeat_thread = threading.Thread(
                target=self._heartbeat_run,
                name="realesrgan-progress-log",
                daemon=True,
            )
            self._heartbeat_thread.start()

        def _heartbeat_rate(self, processed: int, now: float) -> float:
            self._heartbeat_samples.append((processed, now))
            while (
                len(self._heartbeat_samples) > 2
                and now - self._heartbeat_samples[0][1] > _RATE_WINDOW
            ):
                self._heartbeat_samples.popleft()
            if len(self._heartbeat_samples) >= 2:
                old_count, old_time = self._heartbeat_samples[0]
                span = now - old_time
                delta = processed - old_count
                if span > 1e-6 and delta > 0:
                    return delta / span
            elapsed = max(now - float(self.started), 1e-6)
            return processed / elapsed if processed > 0 else 0.0

        def _heartbeat_emit(self, final: bool = False) -> None:
            now = time.monotonic()
            processed = max(0, int(getattr(self, "processed", 0)))
            total = max(0, int(getattr(self.progress, "total", 0) or 0))
            elapsed = max(0.0, now - float(self.started))
            rate = self._heartbeat_rate(processed, now)
            percent = 100.0 * processed / total if total > 0 else 0.0
            remaining = max(0, total - processed)
            eta = remaining / rate if rate > 1e-9 else float("inf")
            if rate > 1e-9:
                speed = f"{rate:.3f} frame/s"
                eta_text = _format_duration(eta)
            else:
                speed = "waiting for output"
                eta_text = "--:--"
            suffix = " | done" if final and total > 0 and processed >= total else ""
            total_text = str(total) if total > 0 else "?"
            _durable_write(
                f"[progress] {processed}/{total_text} | {percent:.1f}% | "
                f"{speed} | elapsed {_format_duration(elapsed)} | ETA {eta_text}{suffix}"
            )

        def _heartbeat_run(self) -> None:
            # Emit immediately so saved logs prove the heartbeat is installed,
            # then keep emitting even while BVS/SR is producing no output frames.
            self._heartbeat_emit()
            while not self._heartbeat_stop.wait(self._heartbeat_interval):
                self._heartbeat_emit()

        def _stop_heartbeat(self, final: bool = False) -> None:
            if self._heartbeat_closed:
                return
            self._heartbeat_closed = True
            self._heartbeat_stop.set()
            thread = getattr(self, "_heartbeat_thread", None)
            if thread is not None and thread.is_alive():
                thread.join(timeout=2.0)
            self._heartbeat_emit(final=final)

        def finish(self) -> None:
            try:
                super().finish()
            finally:
                self._stop_heartbeat(final=True)

        def stop(self) -> None:
            self._stop_heartbeat(final=False)
            super().stop()

    pipeline.OutputPump = PersistentProgressOutputPump
