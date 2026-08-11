"""Pinned host/device transfer helpers for the GPU video pipeline.

CUDA model outputs are drained through reusable page-locked host buffers on
non-default copy streams. H2D input staging likewise reuses pinned buffers so
large shared-memory frames do not pay pageable-memory DMA staging on every task.
The scheduler separates CUDA compute completion from transport completion, which
lets the opposite stage overlap copy/CPU drain without running two heavy model
compute phases at the same time.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch


class _H2DSlot:
    def __init__(self, event: torch.cuda.Event) -> None:
        self.buffer: torch.Tensor | None = None
        self.shape: tuple[int, ...] | None = None
        self.dtype: torch.dtype | None = None
        self.event = event
        self.in_flight = False


class PinnedH2DStager:
    """Reusable pinned CPU input staging and asynchronous H2D copy."""

    def __init__(self, device: torch.device, slots: int = 1) -> None:
        if device.type != "cuda":
            raise ValueError("PinnedH2DStager requires a CUDA device")
        if int(slots) < 1:
            raise ValueError("PinnedH2DStager slots must be positive")
        self.device = device
        self.stream = torch.cuda.Stream(device=device)
        self._slots = [
            _H2DSlot(torch.cuda.Event())
            for _ in range(int(slots))
        ]
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
        """Stage a CPU tensor and enqueue H2D on the reusable copy stream."""
        if source.device.type != "cpu":
            raise RuntimeError(
                f"Expected CPU source, got device={source.device}"
            )

        # If a caller already owns page-locked memory, avoid the otherwise
        # unavoidable CPU staging memcpy. The caller remains responsible for
        # keeping that host tensor alive until its consumer stream has waited.
        if source.is_pinned():
            with torch.cuda.stream(self.stream):
                device_tensor = source.to(self.device, non_blocking=True)
            consumer = torch.cuda.current_stream(self.device)
            consumer.wait_stream(self.stream)
            device_tensor.record_stream(consumer)
            return device_tensor

        slot = self._slots[self._cursor]
        self._cursor = (self._cursor + 1) % len(self._slots)

        # A pinned host buffer must not be modified while an async DMA still
        # reads from it. Reuse the slot-local CUDA Event instead of allocating
        # a fresh Event for every frame/task.
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


class PinnedD2HStager:
    """Reusable pinned CPU staging for one CUDA worker process."""

    def __init__(self, device: torch.device) -> None:
        if device.type != "cuda":
            raise ValueError("PinnedD2HStager requires a CUDA device")
        self.device = device
        self.stream = torch.cuda.Stream(device=device)
        self.done_event = torch.cuda.Event()
        self.buffer: torch.Tensor | None = None
        self._shape: tuple[int, ...] | None = None
        self._dtype: torch.dtype | None = None

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

    def copy(self, source: torch.Tensor) -> np.ndarray:
        """Drain one contiguous CUDA tensor to pinned host memory."""
        if source.device.type != "cuda":
            raise RuntimeError(
                f"Expected CUDA source, got device={source.device}"
            )
        if not source.is_contiguous():
            raise RuntimeError(
                "D2H source must be contiguous before the compute boundary"
            )
        host = self._ensure(source)

        producer = torch.cuda.current_stream(self.device)
        self.stream.wait_stream(producer)
        with torch.cuda.stream(self.stream):
            host.copy_(source, non_blocking=True)
            self.done_event.record(self.stream)
        self.done_event.synchronize()
        return host.numpy()


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
        np.copyto(
            output_view[first : first + count],
            frames,
            casting="no",
        )
        return

    for slot, frame in zip(slot_ids, frames):
        np.copyto(output_view[slot], frame, casting="no")


def copy_cuda_frames_to_slots(
    frames: torch.Tensor,
    output_view: np.ndarray,
    slots: Sequence[int],
    stager: PinnedD2HStager | None = None,
) -> None:
    """Copy one CUDA frame batch into shared-memory slots with one D2H sync."""
    if frames.device.type != "cuda":
        raise RuntimeError(
            f"Expected CUDA BVS output, got device={frames.device}"
        )
    if frames.dtype != torch.uint8:
        raise TypeError(
            f"Expected uint8 BVS output, got dtype={frames.dtype}"
        )
    if frames.ndim != 4:
        raise ValueError(
            f"Expected BVS output shape [T,H,W,C], got {tuple(frames.shape)}"
        )

    count = int(frames.shape[0])
    slot_ids = tuple(int(slot) for slot in slots)
    if count != len(slot_ids):
        raise RuntimeError(
            f"BVS transport frame/slot mismatch: frames={count}, slots={len(slot_ids)}"
        )
    if len(set(slot_ids)) != len(slot_ids):
        raise RuntimeError("BVS transport received duplicate output slots")

    expected_shape = tuple(int(value) for value in output_view.shape[1:])
    frame_shape = tuple(int(value) for value in frames.shape[1:])
    if frame_shape != expected_shape:
        raise RuntimeError(
            "BVS transport frame shape mismatch: "
            f"frames={frame_shape}, slots={expected_shape}"
        )
    if output_view.dtype != np.dtype(np.uint8):
        raise TypeError(
            f"Expected uint8 shared output pool, got dtype={output_view.dtype}"
        )

    slot_count = int(output_view.shape[0])
    for slot in slot_ids:
        if slot < 0 or slot >= slot_count:
            raise RuntimeError(
                f"BVS transport output slot out of range: {slot}/{slot_count}"
            )

    if count == 0:
        return

    transfer = stager or PinnedD2HStager(frames.device)
    frames_cpu = transfer.copy(frames)
    if frames_cpu.dtype != output_view.dtype:
        raise TypeError(
            "BVS transport dtype changed during D2H: "
            f"{frames_cpu.dtype} != {output_view.dtype}"
        )

    copy_host_frames_to_slots(frames_cpu, output_view, slot_ids)


def copy_cuda_frame_to_array(
    frame: torch.Tensor,
    output_view: np.ndarray,
    stager: PinnedD2HStager,
) -> None:
    """Drain one CUDA HWC frame into an existing shared-memory ndarray."""
    if frame.device.type != "cuda":
        raise RuntimeError(
            f"Expected CUDA SR output, got device={frame.device}"
        )
    expected_shape = tuple(int(value) for value in output_view.shape)
    actual_shape = tuple(int(value) for value in frame.shape)
    if actual_shape != expected_shape:
        raise RuntimeError(
            f"SR transport shape mismatch: {actual_shape} != {expected_shape}"
        )

    frame_cpu = stager.copy(frame)
    if frame_cpu.dtype != output_view.dtype:
        raise TypeError(
            f"SR transport dtype mismatch: {frame_cpu.dtype} != {output_view.dtype}"
        )
    np.copyto(output_view, frame_cpu, casting="no")
