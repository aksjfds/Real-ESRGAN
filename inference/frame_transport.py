"""Pinned/direct host-device transport helpers for the GPU video pipeline.

Long-lived shared-memory mappings may be registered in-place with CUDA so
supported packed integer paths can transfer directly between shared slots and
CUDA without a second CPU staging memcpy. Registration is opportunistic:
unsupported platforms/dtypes fall back to the reusable pinned staging path.
D2H is enqueued before the worker publishes its compute boundary, allowing the
copy engine to start as soon as the producer stream finishes while preserving
heavy-compute exclusivity.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import torch


def _torch_dtype_for_numpy(dtype: np.dtype) -> torch.dtype:
    value = np.dtype(dtype)
    if value == np.dtype(np.uint8):
        return torch.uint8
    if value == np.dtype(np.uint16):
        candidate = getattr(torch, "uint16", None)
        if isinstance(candidate, torch.dtype):
            return candidate
        raise TypeError("This PyTorch build does not expose torch.uint16")
    raise TypeError(f"Unsupported packed transport dtype: {value}")


class CudaHostRegistration:
    """Own one cudaHostRegister() registration for a long-lived NumPy mapping."""

    def __init__(self, array: np.ndarray) -> None:
        if not array.flags.c_contiguous:
            raise ValueError("cudaHostRegister requires a contiguous NumPy mapping")
        if int(array.nbytes) <= 0:
            raise ValueError("cudaHostRegister requires a non-empty mapping")

        self.array = array
        self.tensor = torch.from_numpy(array)
        self.ptr = int(self.tensor.data_ptr())
        self.nbytes = int(array.nbytes)
        self.owned = False
        self.closed = False

        if self.tensor.is_pinned():
            return

        cudart = torch.cuda.cudart()
        torch.cuda.check_error(cudart.cudaHostRegister(self.ptr, self.nbytes, 0))
        self.owned = True

        if not self.tensor.is_pinned():
            try:
                torch.cuda.check_error(cudart.cudaHostUnregister(self.ptr))
            finally:
                self.owned = False
            raise RuntimeError(
                "cudaHostRegister succeeded but PyTorch does not report the mapping as pinned"
            )

    @property
    def registered(self) -> bool:
        return bool(self.tensor.is_pinned())

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        if not self.owned:
            return
        torch.cuda.check_error(torch.cuda.cudart().cudaHostUnregister(self.ptr))
        self.owned = False


class _H2DSlot:
    def __init__(self, event: torch.cuda.Event) -> None:
        self.buffer: torch.Tensor | None = None
        self.shape: tuple[int, ...] | None = None
        self.dtype: torch.dtype | None = None
        self.event = event
        self.in_flight = False


class PinnedH2DStager:
    """Asynchronous H2D with direct registered-memory fast path and fallback."""

    def __init__(self, device: torch.device, slots: int = 1) -> None:
        if device.type != "cuda":
            raise ValueError("PinnedH2DStager requires a CUDA device")
        if int(slots) < 1:
            raise ValueError("PinnedH2DStager slots must be positive")
        self.device = device
        self.stream = torch.cuda.Stream(device=device)
        self._slots = [_H2DSlot(torch.cuda.Event()) for _ in range(int(slots))]
        self._cursor = 0

    @staticmethod
    def _ensure(source: torch.Tensor, slot: _H2DSlot) -> torch.Tensor:
        shape = tuple(int(value) for value in source.shape)
        if (
            slot.buffer is None
            or slot.shape != shape
            or slot.dtype != source.dtype
        ):
            slot.buffer = torch.empty(
                shape,
                dtype=source.dtype,
                device="cpu",
                pin_memory=True,
            )
            slot.shape = shape
            slot.dtype = source.dtype
        return slot.buffer

    def copy(self, source: torch.Tensor) -> torch.Tensor:
        """Enqueue H2D, avoiding the staging memcpy for registered/pinned input."""
        if source.device.type != "cpu":
            raise RuntimeError(f"Expected CPU source, got device={source.device}")

        if source.is_pinned():
            with torch.cuda.stream(self.stream):
                device_tensor = source.to(self.device, non_blocking=True)
            consumer = torch.cuda.current_stream(self.device)
            consumer.wait_stream(self.stream)
            device_tensor.record_stream(consumer)
            return device_tensor

        slot = self._slots[self._cursor]
        self._cursor = (self._cursor + 1) % len(self._slots)

        if slot.in_flight:
            slot.event.synchronize()
            slot.in_flight = False

        host = self._ensure(source, slot)
        host.copy_(source, non_blocking=False)

        with torch.cuda.stream(self.stream):
            device_tensor = host.to(self.device, non_blocking=True)
            slot.event.record(self.stream)

        consumer = torch.cuda.current_stream(self.device)
        consumer.wait_event(slot.event)
        device_tensor.record_stream(consumer)
        slot.in_flight = True
        return device_tensor


class PendingD2H:
    """One enqueued D2H transfer whose completion is synchronized exactly once."""

    def __init__(
        self,
        owner: "PinnedD2HStager",
        host_tensor: torch.Tensor,
        host_array: np.ndarray,
        finalize: Callable[[np.ndarray], None] | None,
        direct: bool,
    ) -> None:
        self._owner = owner
        self._host_tensor = host_tensor
        self._host_array = host_array
        self._finalize = finalize
        self.direct = bool(direct)
        self._done = False

    def wait(self) -> np.ndarray:
        if self._done:
            return self._host_array
        self._owner._finish(self)
        self._done = True
        if self._finalize is not None:
            self._finalize(self._host_array)
        return self._host_array


class PinnedD2HStager:
    """Reusable D2H copy stream with registered-shared-memory direct path."""

    def __init__(self, device: torch.device) -> None:
        if device.type != "cuda":
            raise ValueError("PinnedD2HStager requires a CUDA device")
        self.device = device
        self.stream = torch.cuda.Stream(device=device)
        self.done_event = torch.cuda.Event()
        self.buffer: torch.Tensor | None = None
        self._shape: tuple[int, ...] | None = None
        self._dtype: torch.dtype | None = None
        self._pending: PendingD2H | None = None

    def _ensure(self, source: torch.Tensor) -> torch.Tensor:
        shape = tuple(int(value) for value in source.shape)
        if (
            self.buffer is None
            or self._shape != shape
            or self._dtype != source.dtype
        ):
            self.buffer = torch.empty(
                shape,
                dtype=source.dtype,
                device="cpu",
                pin_memory=True,
            )
            self._shape = shape
            self._dtype = source.dtype
        return self.buffer

    @staticmethod
    def _direct_target(
        source: torch.Tensor,
        target: np.ndarray | None,
    ) -> torch.Tensor | None:
        if target is None or not target.flags.c_contiguous:
            return None
        try:
            target_tensor = torch.from_numpy(target)
        except (TypeError, RuntimeError):
            return None
        if tuple(target_tensor.shape) != tuple(source.shape):
            return None
        if target_tensor.dtype != source.dtype:
            return None
        if not target_tensor.is_pinned():
            return None
        return target_tensor

    def begin_copy(
        self,
        source: torch.Tensor,
        *,
        target: np.ndarray | None = None,
        finalize: Callable[[np.ndarray], None] | None = None,
    ) -> PendingD2H:
        """Enqueue one D2H without blocking the host on its completion."""
        if source.device.type != "cuda":
            raise RuntimeError(f"Expected CUDA source, got device={source.device}")
        if not source.is_contiguous():
            raise RuntimeError(
                "D2H source must be contiguous before the compute boundary"
            )
        if self._pending is not None:
            raise RuntimeError("D2H stager already has an in-flight transfer")

        direct_tensor = self._direct_target(source, target)
        direct = direct_tensor is not None
        if direct:
            if finalize is not None:
                raise RuntimeError("Direct D2H target cannot also use a finalizer")
            host_tensor = direct_tensor
            host_array = target
        else:
            host_tensor = self._ensure(source)
            host_array = host_tensor.numpy()
            if target is not None:
                if tuple(target.shape) != tuple(source.shape):
                    raise RuntimeError(
                        f"D2H target shape mismatch: {target.shape} != {tuple(source.shape)}"
                    )
                expected_dtype = _torch_dtype_for_numpy(target.dtype)
                if expected_dtype != source.dtype:
                    raise TypeError(
                        f"D2H target dtype mismatch: {target.dtype} != {source.dtype}"
                    )
                if finalize is not None:
                    raise RuntimeError(
                        "D2H target and explicit finalizer are mutually exclusive"
                    )

                def copy_target(array: np.ndarray) -> None:
                    np.copyto(target, array, casting="no")

                finalize = copy_target

        producer = torch.cuda.current_stream(self.device)
        self.stream.wait_stream(producer)
        with torch.cuda.stream(self.stream):
            host_tensor.copy_(source, non_blocking=True)
            self.done_event.record(self.stream)

        pending = PendingD2H(
            self,
            host_tensor,
            host_array,
            finalize,
            direct,
        )
        self._pending = pending
        return pending

    def _finish(self, pending: PendingD2H) -> None:
        if self._pending is not pending:
            raise RuntimeError("Unexpected D2H completion token")
        try:
            self.done_event.synchronize()
        finally:
            self._pending = None

    def copy(self, source: torch.Tensor) -> np.ndarray:
        """Compatibility helper: enqueue then wait for pinned staging output."""
        return self.begin_copy(source).wait()


def copy_host_frames_to_slots(
    frames: np.ndarray,
    output_view: np.ndarray,
    slots: Sequence[int],
) -> None:
    """Copy a host frame batch to shared slots, using one copy when contiguous."""
    count = int(frames.shape[0])
    slot_ids = tuple(int(slot) for slot in slots)
    if count != len(slot_ids):
        raise RuntimeError(
            f"Host transport frame/slot mismatch: frames={count}, slots={len(slot_ids)}"
        )
    if count == 0:
        return

    first = slot_ids[0]
    contiguous = slot_ids == tuple(range(first, first + count))
    if contiguous:
        np.copyto(output_view[first : first + count], frames, casting="no")
        return

    for slot, frame in zip(slot_ids, frames):
        np.copyto(output_view[slot], frame, casting="no")


def _validate_cuda_frame_batch(
    frames: torch.Tensor,
    output_view: np.ndarray,
    slots: Sequence[int],
) -> tuple[int, tuple[int, ...]]:
    if frames.device.type != "cuda":
        raise RuntimeError(f"Expected CUDA frame batch, got device={frames.device}")
    expected_dtype = _torch_dtype_for_numpy(output_view.dtype)
    if frames.dtype != expected_dtype:
        raise TypeError(
            f"CUDA/shared output dtype mismatch: {frames.dtype} != {output_view.dtype}"
        )
    if frames.ndim != 4:
        raise ValueError(
            f"Expected CUDA frame batch [T,H,W,C], got {tuple(frames.shape)}"
        )
    if not frames.is_contiguous():
        raise RuntimeError("CUDA frame batch must be contiguous HWC")

    count = int(frames.shape[0])
    slot_ids = tuple(int(slot) for slot in slots)
    if count != len(slot_ids):
        raise RuntimeError(
            f"Transport frame/slot mismatch: frames={count}, slots={len(slot_ids)}"
        )
    if len(set(slot_ids)) != len(slot_ids):
        raise RuntimeError("Transport received duplicate output slots")

    expected_shape = tuple(int(value) for value in output_view.shape[1:])
    frame_shape = tuple(int(value) for value in frames.shape[1:])
    if frame_shape != expected_shape:
        raise RuntimeError(
            f"Transport frame shape mismatch: frames={frame_shape}, slots={expected_shape}"
        )

    slot_count = int(output_view.shape[0])
    for slot in slot_ids:
        if slot < 0 or slot >= slot_count:
            raise RuntimeError(
                f"Transport output slot out of range: {slot}/{slot_count}"
            )
    return count, slot_ids


def begin_cuda_frames_to_slots(
    frames: torch.Tensor,
    output_view: np.ndarray,
    slots: Sequence[int],
    stager: PinnedD2HStager,
) -> PendingD2H | None:
    """Enqueue a batch D2H, direct to registered shared slots when contiguous."""
    count, slot_ids = _validate_cuda_frame_batch(frames, output_view, slots)
    if count == 0:
        return None

    first = slot_ids[0]
    contiguous = slot_ids == tuple(range(first, first + count))
    if contiguous:
        return stager.begin_copy(
            frames,
            target=output_view[first : first + count],
        )

    return stager.begin_copy(
        frames,
        finalize=lambda array: copy_host_frames_to_slots(
            array,
            output_view,
            slot_ids,
        ),
    )


def copy_cuda_frames_to_slots(
    frames: torch.Tensor,
    output_view: np.ndarray,
    slots: Sequence[int],
    stager: PinnedD2HStager | None = None,
) -> None:
    """Compatibility helper for one synchronized frame-batch D2H."""
    transfer = stager or PinnedD2HStager(frames.device)
    pending = begin_cuda_frames_to_slots(frames, output_view, slots, transfer)
    if pending is not None:
        pending.wait()


def begin_cuda_frame_to_array(
    frame: torch.Tensor,
    output_view: np.ndarray,
    stager: PinnedD2HStager,
) -> PendingD2H:
    """Enqueue one HWC frame D2H directly to registered output when possible."""
    if frame.device.type != "cuda":
        raise RuntimeError(f"Expected CUDA SR output, got device={frame.device}")
    expected_shape = tuple(int(value) for value in output_view.shape)
    actual_shape = tuple(int(value) for value in frame.shape)
    if actual_shape != expected_shape:
        raise RuntimeError(
            f"SR transport shape mismatch: {actual_shape} != {expected_shape}"
        )
    expected_dtype = _torch_dtype_for_numpy(output_view.dtype)
    if expected_dtype != frame.dtype:
        raise TypeError(
            f"SR transport dtype mismatch: {output_view.dtype} != {frame.dtype}"
        )
    return stager.begin_copy(frame, target=output_view)


def copy_cuda_frame_to_array(
    frame: torch.Tensor,
    output_view: np.ndarray,
    stager: PinnedD2HStager,
) -> None:
    """Compatibility helper for one synchronized frame-batch D2H."""
    begin_cuda_frame_to_array(frame, output_view, stager).wait()
