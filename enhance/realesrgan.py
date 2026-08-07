"""Internal loader for the v4.2 core used by the v4.3 top-level entry."""

import sys
from pathlib import Path

_core_path = Path(__file__).resolve().with_name("realesrgan_core.py")
_public_path = Path(__file__).resolve().parents[1] / "realesrgan.py"
if not _core_path.is_file():
    raise FileNotFoundError(f"Internal Real-ESRGAN core not found: {_core_path}")

# Preserve the v4.2 core's repository-root-relative paths for weights and assets.
globals()["__file__"] = str(_public_path)
exec(compile(_core_path.read_bytes(), str(_core_path), "exec"), globals(), globals())

# The APISR branch reuses the validated video/runtime stack and swaps only the
# super-resolution model/checkpoint integration before fast workers are built.
from enhance.apisr_backend import install_apisr_backend

install_apisr_backend(sys.modules[__name__])
