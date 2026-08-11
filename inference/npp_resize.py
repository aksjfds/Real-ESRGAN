"""NPP Lanczos resize for contiguous uint8 HWC CUDA tensors.

The wrapper binds NVIDIA NPP's application-managed stream-context resize API to
the PyTorch current CUDA stream.  It is intentionally narrow: the active video
pipeline only needs packed 8-bit C3 images, so unsupported environments can
fall back without leaking NPP details into the scheduler or model runtime.
"""

from __future__ import annotations

import ctypes
import ctypes.util
from pathlib import Path
import site
import sys

import torch


_NPP_SUCCESS = 0
_NPPI_INTER_LANCZOS = 16


class NppiSize(ctypes.Structure):
    _fields_ = [("width", ctypes.c_int), ("height", ctypes.c_int)]


class NppiRect(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_int),
        ("y", ctypes.c_int),
        ("width", ctypes.c_int),
        ("height", ctypes.c_int),
    ]


class NppStreamContext(ctypes.Structure):
    _fields_ = [
        ("hStream", ctypes.c_void_p),
        ("nCudaDeviceId", ctypes.c_int),
        ("nMultiProcessorCount", ctypes.c_int),
        ("nMaxThreadsPerMultiProcessor", ctypes.c_int),
        ("nMaxThreadsPerBlock", ctypes.c_int),
        ("nSharedMemPerBlock", ctypes.c_size_t),
        ("nCudaDevAttrComputeCapabilityMajor", ctypes.c_int),
        ("nCudaDevAttrComputeCapabilityMinor", ctypes.c_int),
        ("nStreamFlags", ctypes.c_uint),
        ("nReserved0", ctypes.c_int),
    ]


def _library_candidates(stem: str) -> list[str]:
    candidates: list[str] = []
    found = ctypes.util.find_library(stem)
    if found:
        candidates.append(found)

    cuda_major = ""
    if torch.version.cuda:
        cuda_major = str(torch.version.cuda).split(".", 1)[0]
    if cuda_major:
        candidates.append(f"lib{stem}.so.{cuda_major}")
    candidates.append(f"lib{stem}.so")

    roots = [
        Path("/usr/local/cuda/lib64"),
        Path("/usr/local/cuda/targets/x86_64-linux/lib"),
        Path("/usr/lib/x86_64-linux-gnu"),
    ]
    try:
        roots.extend(Path(value) / "nvidia" / "npp" / "lib" for value in site.getsitepackages())
    except Exception:
        pass
    for value in sys.path:
        if value:
            roots.append(Path(value) / "nvidia" / "npp" / "lib")

    for root in roots:
        try:
            matches = sorted(root.glob(f"lib{stem}.so*"), reverse=True)
        except OSError:
            continue
        candidates.extend(str(path) for path in matches)

    unique: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        unique.append(candidate)
    return unique


def _load_library(stem: str) -> ctypes.CDLL:
    errors: list[str] = []
    for candidate in _library_candidates(stem):
        try:
            return ctypes.CDLL(candidate)
        except OSError as error:
            errors.append(f"{candidate}: {error}")
    detail = "; ".join(errors[-3:]) if errors else "library not found"
    raise RuntimeError(f"NPP {stem} unavailable ({detail})")


def _check_npp(status: int, operation: str) -> None:
    if int(status) != _NPP_SUCCESS:
        raise RuntimeError(f"{operation} failed with NppStatus={int(status)}")


class NppLanczosResizer:
    """Resize uint8 HWC CUDA batches with NPP Lanczos on the current stream."""

    def __init__(self, device: torch.device) -> None:
        if device.type != "cuda":
            raise ValueError("NppLanczosResizer requires a CUDA device")
        self.device = torch.device(device)
        self.nppc = _load_library("nppc")
        self.nppig = _load_library("nppig")
        self._bound_stream: int | None = None

        self.nppc.nppGetStream.restype = ctypes.c_void_p
        self.nppc.nppSetStream.argtypes = [ctypes.c_void_p]
        self.nppc.nppSetStream.restype = ctypes.c_int
        self.nppc.nppGetStreamContext.argtypes = [
            ctypes.POINTER(NppStreamContext)
        ]
        self.nppc.nppGetStreamContext.restype = ctypes.c_int

        resize = self.nppig.nppiResize_8u_C3R_Ctx
        resize.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            NppiSize,
            NppiRect,
            ctypes.c_void_p,
            ctypes.c_int,
            NppiSize,
            NppiRect,
            ctypes.c_int,
            NppStreamContext,
        ]
        resize.restype = ctypes.c_int
        self._resize = resize

    def _stream_context(self) -> NppStreamContext:
        stream = torch.cuda.current_stream(self.device)
        stream_ptr = int(stream.cuda_stream)
        current = self.nppc.nppGetStream()
        current_ptr = 0 if current is None else int(current)
        if current_ptr != stream_ptr or self._bound_stream != stream_ptr:
            _check_npp(
                self.nppc.nppSetStream(ctypes.c_void_p(stream_ptr)),
                "nppSetStream",
            )
            self._bound_stream = stream_ptr

        context = NppStreamContext()
        _check_npp(
            self.nppc.nppGetStreamContext(ctypes.byref(context)),
            "nppGetStreamContext",
        )
        return context

    def resize_batch(
        self,
        frames: torch.Tensor,
        out_height: int,
        out_width: int,
    ) -> torch.Tensor:
        if frames.device.type != "cuda":
            raise RuntimeError("NPP resize source must be CUDA")
        if frames.dtype != torch.uint8:
            raise TypeError(f"NPP resize expects uint8, got {frames.dtype}")
        if frames.ndim != 4 or int(frames.shape[-1]) != 3:
            raise ValueError(
                f"NPP resize expects [N,H,W,3], got {tuple(frames.shape)}"
            )
        if not frames.is_contiguous():
            raise RuntimeError("NPP resize source must be contiguous HWC")

        count, in_height, in_width, _channels = (
            int(value) for value in frames.shape
        )
        out_height = int(out_height)
        out_width = int(out_width)
        if out_height <= 0 or out_width <= 0:
            raise ValueError("NPP resize target dimensions must be positive")
        if in_height == out_height and in_width == out_width:
            return frames

        output = torch.empty(
            (count, out_height, out_width, 3),
            dtype=torch.uint8,
            device=frames.device,
        )
        context = self._stream_context()
        src_size = NppiSize(in_width, in_height)
        src_roi = NppiRect(0, 0, in_width, in_height)
        dst_size = NppiSize(out_width, out_height)
        dst_roi = NppiRect(0, 0, out_width, out_height)
        src_step = in_width * 3
        dst_step = out_width * 3

        for index in range(count):
            source = frames[index]
            target = output[index]
            status = self._resize(
                ctypes.c_void_p(int(source.data_ptr())),
                src_step,
                src_size,
                src_roi,
                ctypes.c_void_p(int(target.data_ptr())),
                dst_step,
                dst_size,
                dst_roi,
                _NPPI_INTER_LANCZOS,
                context,
            )
            _check_npp(status, "nppiResize_8u_C3R_Ctx")

        return output
