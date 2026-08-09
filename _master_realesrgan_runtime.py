"""Importable alias for the master/2.7 top-level video runtime.

The repository contains both ``realesrgan.py`` and a ``realesrgan/`` package.
AV1 encoding support needs to reuse the top-level runtime while remaining
importable by multiprocessing's spawn method. Executing the original runtime in
this real module namespace keeps worker functions picklable/importable without
duplicating the core implementation.
"""

from __future__ import annotations

from pathlib import Path

_RUNTIME_PATH = Path(__file__).resolve().with_name("realesrgan.py")
exec(compile(_RUNTIME_PATH.read_bytes(), str(_RUNTIME_PATH), "exec"), globals(), globals())
