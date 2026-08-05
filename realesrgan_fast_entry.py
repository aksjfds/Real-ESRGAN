#!/usr/bin/env python3
"""Kaggle-safe entry point for the shared-memory Real-ESRGAN runner.

Large native-scale output arrays may exceed Docker's POSIX shared-memory quota.
Use POSIX shared memory when there is ample capacity and transparently fall back
to a persistent file-backed mmap otherwise. Both paths avoid Queue pickling of
frame/tile arrays.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import numpy as np

import realesrgan_fast as fast


_original_shared_array_init = fast.SharedArray.__init__


def _guarded_shared_array_init(
    self: fast.SharedArray,
    shape: tuple[int, ...],
    dtype: np.dtype,
    prefix: str,
) -> None:
    resolved_shape = tuple(int(value) for value in shape)
    resolved_dtype = np.dtype(dtype)
    required = int(np.prod(resolved_shape, dtype=np.int64)) * resolved_dtype.itemsize
    shm_path = Path("/dev/shm")
    use_posix_shm = False
    if shm_path.is_dir():
        try:
            free = shutil.disk_usage(shm_path).free
            use_posix_shm = required <= int(free * 0.70)
        except OSError:
            use_posix_shm = False

    if use_posix_shm:
        _original_shared_array_init(self, resolved_shape, resolved_dtype, prefix)
        return

    self.shape = resolved_shape
    self.dtype = resolved_dtype
    self.size = required
    self.shm = None
    root = Path("/kaggle/working")
    if not root.is_dir():
        root = Path(tempfile.gettempdir())
    fd, path = tempfile.mkstemp(prefix=f"{prefix}-", suffix=".mmap", dir=root)
    os.close(fd)
    with open(path, "wb") as handle:
        handle.truncate(required)
    self.path = path
    self.array = np.memmap(path, mode="r+", dtype=resolved_dtype, shape=resolved_shape)
    self.spec = fast.SharedArraySpec("mmap", path, resolved_shape, resolved_dtype.str)


fast.SharedArray.__init__ = _guarded_shared_array_init


if __name__ == "__main__":
    fast.mp.freeze_support()
    fast.main()
