"""Internal loader for the v4.2 core used by the v4.3 top-level entry."""

import sys
from dataclasses import replace
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
# BasicVSRPPPreprocessor. The dependency-light BasicVSR++ port already selects
# its CUDA device from ``config.gpu_id``, so consume/validate that legacy
# keyword here. For APISR + BasicVSR++ we also use a conservative execution
# configuration: torchvision deform_conv2d has had Float/Half mixed-precision
# compatibility failures in some builds, while profile C's 512px/9-frame path
# has a much larger activation footprint. Force the temporal preprocessor to
# FP32 and 256px tiles; APISR GRL itself is already intentionally FP32.
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

        if bool(getattr(config, "fp16", False)) or int(getattr(config, "tile_size", 256)) > 256:
            config = replace(config, fp16=False, tile_size=256)
            print(
                "[basicvsrpp] APISR compatibility mode: fp16=False, tile_size=256",
                flush=True,
            )

        super().__init__(config, *args, **kwargs)


BasicVSRPPPreprocessor = APISRBasicVSRPPPreprocessor
