"""Single-run guard for one output path.

The lock is intentionally process-level rather than scheduler-level. It prevents
accidental duplicate notebook launches from running two full GPU pipelines against
the same output path, while automatically releasing the lock if the owner exits.
"""

from __future__ import annotations

from contextlib import contextmanager
import errno
import hashlib
import os
from pathlib import Path
import tempfile
from typing import Iterator, TextIO


@contextmanager
def exclusive_output_run(output_path: str | Path) -> Iterator[None]:
    """Allow only one inference process for a resolved output path on POSIX."""
    if os.name != "posix":
        yield
        return

    import fcntl

    resolved = str(Path(output_path).expanduser().resolve())
    digest = hashlib.sha256(resolved.encode("utf-8")).hexdigest()[:24]
    lock_dir = Path(tempfile.gettempdir()) / "realesrgan-run-locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / f"{digest}.lock"

    handle: TextIO = lock_path.open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(
                handle.fileno(),
                fcntl.LOCK_EX | fcntl.LOCK_NB,
            )
        except OSError as error:
            if error.errno not in (errno.EACCES, errno.EAGAIN):
                raise
            handle.seek(0)
            owner = handle.read().strip() or "unknown"
            raise RuntimeError(
                "Another Real-ESRGAN inference process is already writing "
                f"{resolved} (lock owner pid={owner})."
            ) from error

        handle.seek(0)
        handle.truncate()
        handle.write(str(os.getpid()))
        handle.flush()
        yield
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()
