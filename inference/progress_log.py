"""Durable progress logging for saved cloud notebook logs."""
from __future__ import annotations

from collections import deque
import os
import sys
import threading
import time
from typing import Deque, Optional, Tuple

from tqdm import tqdm as _tqdm

from . import pipeline

_INSTALLED = False
_LOG_INTERVAL = 60.0
_RATE_WINDOW = 120.0
_EMIT_LOCK = threading.Lock()
_LAST_EMIT_SIGNATURE: Optional[Tuple[int, int]] = None
_LAST_EMIT_AT = 0.0


class _InteractiveTqdm(_tqdm):
    """Console tqdm tuned for a foreground Kaggle notebook child process."""

    def __init__(self, *args, **kwargs):
        # tqdm defaults to stderr. Force that default here because the project
        # historically passed sys.stdout explicitly, whose carriage-return
        # redraws are not reliably surfaced by Kaggle's notebook subprocess UI.
        kwargs["file"] = sys.stderr
        kwargs["disable"] = False
        # The workload is bursty (BVS clips then SR frames). Dynamic miniters can
        # otherwise become too large after a fast burst and make the bar appear
        # frozen during the following slower interval.
        kwargs["miniters"] = 1
        kwargs["mininterval"] = min(float(kwargs.get("mininterval", 0.5)), 0.5)
        kwargs["maxinterval"] = min(float(kwargs.get("maxinterval", 2.0)), 2.0)
        super().__init__(*args, **kwargs)


class _SilentTqdm(_tqdm):
    """Keep tqdm's counters/API but never render a carriage-return bar."""

    def __init__(self, *args, **kwargs):
        kwargs["disable"] = True
        super().__init__(*args, **kwargs)


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
    """Write exactly one newline-terminated record to the persisted log stream."""
    payload = (line + "\n").encode("utf-8", errors="replace")
    try:
        fd = sys.stdout.fileno()
        os.write(fd, payload)
        return
    except Exception:
        pass
    print(line, flush=True)


def _kaggle_run_type() -> str:
    return os.environ.get("KAGGLE_KERNEL_RUN_TYPE", "").strip().lower()


def _is_kaggle_interactive() -> bool:
    """Return True only for Kaggle's foreground interactive notebook session."""
    return _kaggle_run_type() == "interactive"


def _install_clean_postfix_format() -> None:
    """Remove the duplicate comma before tqdm's string postfix."""
    base_cls = pipeline.OutputPump
    if getattr(base_cls, "_clean_postfix_format_installed", False):
        return

    original_init = base_cls.__init__

    def clean_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        # tqdm's {postfix} is automatically prefixed with ', ' when it is a
        # normal string. Do not add another comma in bar_format.
        self.progress.bar_format = (
            "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}{postfix}]"
        )

    base_cls.__init__ = clean_init
    base_cls._clean_postfix_format_installed = True


def _install_interactive_tqdm() -> None:
    """Route every interactive progress constructor through stderr."""
    # Base/A/single-GPU paths construct pipeline.PersistentTqdm directly.
    if hasattr(pipeline, "PersistentTqdm"):
        pipeline.PersistentTqdm = _InteractiveTqdm

    # v5.2 multi-GPU B-E scheduler imported tqdm into its own module namespace.
    try:
        from . import v52_scheduler

        v52_scheduler.tqdm = _InteractiveTqdm
    except Exception:
        pass


def _disable_batch_tqdm() -> None:
    """Keep batch logs clean by suppressing all carriage-return tqdm rendering."""
    # Base/A/single-GPU paths construct pipeline.PersistentTqdm directly.
    if hasattr(pipeline, "PersistentTqdm"):
        pipeline.PersistentTqdm = _SilentTqdm

    # v5.2 multi-GPU B-E scheduler imported tqdm into its own module namespace.
    try:
        from . import v52_scheduler

        v52_scheduler.tqdm = _SilentTqdm
    except Exception:
        pass


def install_persistent_progress(interval: float = _LOG_INTERVAL) -> None:
    """Keep interactive tqdm live; install one clean heartbeat for batch logs."""
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True

    # The original v5.3 PersistentTqdm logger writes newline records from
    # set_postfix(), which conflicts with interactive carriage-return redraws.
    persistent_tqdm = getattr(pipeline, "PersistentTqdm", None)
    if persistent_tqdm is not None and hasattr(persistent_tqdm, "_persistent_log"):
        persistent_tqdm._persistent_log = lambda *args, **kwargs: None

    # Fix the interactive bar independently of the saved-log heartbeat.
    _install_clean_postfix_format()

    # Foreground notebook: use a normal console tqdm on stderr. The inference
    # process is a subprocess, so a notebook widget tqdm is not appropriate here.
    if _is_kaggle_interactive():
        _install_interactive_tqdm()
        return

    # Save & Run / Batch: carriage-return bars are useless in persisted logs and
    # are the source of giant concatenated Real-ESRGAN lines. Keep tqdm's counter
    # object but render nothing; the heartbeat below is the sole progress logger.
    _disable_batch_tqdm()

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
            self._heartbeat_last_zero_at = float("-inf")
            super().__init__(*args, **kwargs)
            self._heartbeat_thread = threading.Thread(
                target=self._heartbeat_run,
                name="realesrgan-progress-log",
                daemon=True,
            )
            self._heartbeat_thread.start()

        def _heartbeat_rate(self, processed: int, now: float) -> float:
            if processed <= 0:
                return 0.0

            # Mirror StableOutputPump: prefer a long rolling window, then fall
            # back to average throughput since the first completed output frame.
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
                if span >= 20.0 and delta > 0:
                    return delta / span

            first_output_at = getattr(self, "first_output_at", None)
            if first_output_at is not None:
                return max(
                    0.0,
                    (processed - 1) / max(now - float(first_output_at), 1e-6),
                )

            first_count, first_time = self._heartbeat_samples[0]
            span = now - first_time
            delta = processed - first_count
            return delta / span if span > 1e-6 and delta > 0 else 0.0

        def _heartbeat_emit(self, final: bool = False) -> None:
            global _LAST_EMIT_AT, _LAST_EMIT_SIGNATURE

            now = time.monotonic()
            processed = max(0, int(getattr(self, "processed", 0)))
            total = max(0, int(getattr(self.progress, "total", 0) or 0))

            # Batch progress is intentionally sparse: one record per configured
            # heartbeat interval, including while waiting for the first output.
            if processed == 0 and not final:
                if now - self._heartbeat_last_zero_at < self._heartbeat_interval:
                    return
                self._heartbeat_last_zero_at = now

            # If more than one OutputPump exists briefly, suppress near-identical
            # heartbeat records instead of printing duplicate rows a few ms apart.
            signature = (processed, total)
            with _EMIT_LOCK:
                duplicate_window = min(2.0, max(0.5, self._heartbeat_interval / 4.0))
                if (
                    not final
                    and signature == _LAST_EMIT_SIGNATURE
                    and now - _LAST_EMIT_AT < duplicate_window
                ):
                    return
                _LAST_EMIT_SIGNATURE = signature
                _LAST_EMIT_AT = now

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
