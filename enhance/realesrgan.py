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


# AutoDL's single-GPU runtime historically injects ``gpu_ids=(gpu_id,)`` into
# BasicVSRPPPreprocessor.  The current dependency-light BasicVSR++ port accepts
# only ``config``, ``checkpoint_dir`` and ``model`` and already selects its CUDA
# device from ``config.gpu_id``.  Profiles B/C are the only paths that construct
# BasicVSR++, so profile A hid this signature mismatch.  Keep the AutoDL runtime
# contract intact here and consume/validate the legacy keyword before delegating
# to the actual preprocessor.
_OriginalBasicVSRPPPreprocessor = BasicVSRPPPreprocessor


class APISRBasicVSRPPPreprocessor(_OriginalBasicVSRPPPreprocessor):
    def __init__(self, config: object, *args: object, **kwargs: object) -> None:
        gpu_ids = kwargs.pop("gpu_ids", None)
        if gpu_ids is not None:
            resolved = tuple(int(value) for value in gpu_ids)
            if len(resolved) != 1:
                raise ValueError(
                    "APISR AutoDL BasicVSR++ requires exactly one CUDA device"
                )
            configured_gpu = int(getattr(config, "gpu_id"))
            if resolved[0] != configured_gpu:
                raise ValueError(
                    "BasicVSR++ GPU mismatch: "
                    f"runtime requested cuda:{resolved[0]}, "
                    f"config requested cuda:{configured_gpu}"
                )
        super().__init__(config, *args, **kwargs)


BasicVSRPPPreprocessor = APISRBasicVSRPPPreprocessor
