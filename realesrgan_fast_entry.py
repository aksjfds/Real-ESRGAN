#!/usr/bin/env python3
"""Kaggle-safe entry point for the shared-memory Real-ESRGAN runner.

Large native-scale output arrays may exceed Docker's POSIX shared-memory quota.
Use POSIX shared memory when there is ample capacity and transparently fall back
to a persistent file-backed mmap otherwise. Both paths avoid Queue pickling of
frame/tile arrays.

The repository contains a top-level ``realesrgan.py`` runner while the installed
official project exposes the ``realesrgan`` package used by ``enhance.srvgg``.
Keep those two modules under different names: import the official package after
temporarily removing the repository directory from ``sys.path``, then load the
local runner as ``_realesrgan_local_runner`` and inject it only into the fast
runner's ``base`` reference.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import shutil
import sys
import tempfile
from pathlib import Path
from types import ModuleType

import numpy as np


_REPOSITORY_ROOT = Path(__file__).resolve().parent
_LOCAL_RUNNER_NAME = "_realesrgan_local_runner"


def _resolved_path(entry: str) -> Path:
    """Resolve one sys.path entry, including the empty current-directory entry."""
    return Path(entry or os.getcwd()).resolve()


def _import_official_realesrgan() -> ModuleType:
    """Import the installed ``realesrgan`` package without local-file shadowing."""
    existing = sys.modules.get("realesrgan")
    if existing is not None and hasattr(existing, "__path__"):
        return existing
    if existing is not None:
        del sys.modules["realesrgan"]

    original_path = list(sys.path)
    try:
        sys.path[:] = [
            entry
            for entry in original_path
            if _resolved_path(entry) != _REPOSITORY_ROOT
        ]
        importlib.invalidate_caches()
        package = importlib.import_module("realesrgan")
    finally:
        sys.path[:] = original_path

    if not hasattr(package, "__path__"):
        raise ImportError(
            "The installed 'realesrgan' import is not a package. "
            "Reinstall the repository requirements before running the notebook."
        )
    try:
        importlib.import_module("realesrgan.archs.srvgg_arch")
    except ModuleNotFoundError as error:
        raise ImportError(
            "The installed Real-ESRGAN package does not provide "
            "realesrgan.archs.srvgg_arch. Reinstall requirements."
        ) from error
    return package


def _load_local_runner() -> ModuleType:
    """Load the adjacent runner under a private name, never as ``realesrgan``."""
    module_path = _REPOSITORY_ROOT / "realesrgan.py"
    if not module_path.is_file():
        raise FileNotFoundError(f"Local Real-ESRGAN runner not found: {module_path}")

    spec = importlib.util.spec_from_file_location(_LOCAL_RUNNER_NAME, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to create an import spec for {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[_LOCAL_RUNNER_NAME] = module
    spec.loader.exec_module(module)

    required = ("build_parser", "process_video", "PersistentWorkers", "TileProcessor")
    missing = [name for name in required if not hasattr(module, name)]
    if missing:
        raise ImportError(
            f"Local runner {module_path} is missing required symbols: {missing}"
        )
    return module


_OFFICIAL_PACKAGE = _import_official_realesrgan()
_LOCAL_BASE = _load_local_runner()

import realesrgan_fast as fast  # noqa: E402

# ``realesrgan_fast`` imports the official package at module import time because
# that package must retain its public name for enhance.srvgg. Only its runtime
# dependency is replaced with the adjacent CLI/video runner.
fast.base = _LOCAL_BASE


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
