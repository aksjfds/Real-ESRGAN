"""Reassemble repository-split BasicVSR++ checkpoint parts at runtime."""

from __future__ import annotations

import atexit
import hashlib
import os
import tempfile
import threading
from pathlib import Path

_MODEL_NAME = "basicvsr_plusplus_c128n25_ntire_decompress_track1_20210223-7b2eba02.pth"
_EXPECTED_SHA256_PREFIX = "7b2eba02"
_LOCK = threading.Lock()
_MERGED_PATH: Path | None = None


def _cleanup() -> None:
    global _MERGED_PATH
    path = _MERGED_PATH
    _MERGED_PATH = None
    if path is not None:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass


atexit.register(_cleanup)


def _parts(checkpoint_dir: Path) -> list[Path]:
    parts = sorted(checkpoint_dir.glob(f"{_MODEL_NAME}.part[0-9][0-9]"))
    if not parts:
        raise FileNotFoundError(
            "Bundled BasicVSR++ checkpoint parts are missing from "
            f"{checkpoint_dir}. Expected {_MODEL_NAME}.part01, part02, ..."
        )
    expected = [checkpoint_dir / f"{_MODEL_NAME}.part{index:02d}" for index in range(1, len(parts) + 1)]
    if parts != expected:
        raise RuntimeError(
            "BasicVSR++ checkpoint parts are incomplete or non-contiguous: "
            + ", ".join(path.name for path in parts)
        )
    for part in parts:
        if part.stat().st_size <= 0:
            raise RuntimeError(f"BasicVSR++ checkpoint part is empty: {part}")
    return parts


def resolve_checkpoint(checkpoint_dir: Path) -> Path:
    """Merge ordered repository parts into one temporary verified checkpoint."""

    global _MERGED_PATH
    checkpoint_dir = Path(checkpoint_dir).resolve()
    with _LOCK:
        if _MERGED_PATH is not None and _MERGED_PATH.is_file():
            return _MERGED_PATH

        parts = _parts(checkpoint_dir)
        fd, temporary_name = tempfile.mkstemp(
            prefix=f"realesrgan-basicvsrpp-{os.getpid()}-",
            suffix=".pth",
        )
        os.close(fd)
        temporary = Path(temporary_name)
        digest = hashlib.sha256()
        try:
            with temporary.open("wb") as output:
                for part in parts:
                    with part.open("rb") as source:
                        while True:
                            block = source.read(8 * 1024 * 1024)
                            if not block:
                                break
                            output.write(block)
                            digest.update(block)
            sha256 = digest.hexdigest()
            if not sha256.startswith(_EXPECTED_SHA256_PREFIX):
                raise RuntimeError(
                    "BasicVSR++ checkpoint checksum mismatch after joining parts: "
                    f"sha256={sha256}, expected prefix={_EXPECTED_SHA256_PREFIX}"
                )
            _MERGED_PATH = temporary
            print(
                f"[basicvsrpp] joined {len(parts)} bundled checkpoint parts | sha256={sha256[:12]}...",
                flush=True,
            )
            return temporary
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
