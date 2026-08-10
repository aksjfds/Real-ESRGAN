"""Reference-counted shared-memory slots for intermediate video frames."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .task_protocol import FrameHandle


class FrameSlotPool:
    def __init__(
        self,
        gpu_ids: Sequence[int],
        slots_per_worker: int,
        views: Sequence[np.ndarray],
    ) -> None:
        self.gpu_ids = [int(value) for value in gpu_ids]
        self.slots_per_worker = int(slots_per_worker)
        self.views = list(views)

        self._free = [
            set(range(self.slots_per_worker))
            for _ in self.gpu_ids
        ]
        self._refs = [
            [0] * self.slots_per_worker
            for _ in self.gpu_ids
        ]
        self._generations = [
            [0] * self.slots_per_worker
            for _ in self.gpu_ids
        ]

    def available(self, worker_id: int) -> int:
        return len(self._free[worker_id])

    def can_reserve(self, worker_id: int, count: int) -> bool:
        return self.available(worker_id) >= int(count)

    def reserve(
        self,
        worker_id: int,
        count: int,
    ) -> tuple[FrameHandle, ...]:
        count = int(count)
        if count < 0:
            raise ValueError("Frame-slot reservation count cannot be negative")
        if count == 0:
            return ()
        if not self.can_reserve(worker_id, count):
            raise BufferError(
                f"cuda:{self.gpu_ids[worker_id]} has only "
                f"{self.available(worker_id)} free frame slots; "
                f"{count} required"
            )

        handles: list[FrameHandle] = []
        for slot in sorted(self._free[worker_id])[:count]:
            self._free[worker_id].remove(slot)
            self._generations[worker_id][slot] += 1
            self._refs[worker_id][slot] = 1
            handles.append(
                FrameHandle(
                    worker_id=worker_id,
                    slot=slot,
                    generation=self._generations[worker_id][slot],
                )
            )
        return tuple(handles)

    def _validate(self, handle: FrameHandle) -> None:
        worker_id = int(handle.worker_id)
        slot = int(handle.slot)

        if worker_id < 0 or worker_id >= len(self.gpu_ids):
            raise RuntimeError(f"Invalid frame-handle worker: {worker_id}")
        if slot < 0 or slot >= self.slots_per_worker:
            raise RuntimeError(f"Invalid frame-handle slot: {slot}")
        if (
            self._generations[worker_id][slot] != handle.generation
            or self._refs[worker_id][slot] <= 0
        ):
            raise RuntimeError(
                "Stale or released frame handle: "
                f"worker={worker_id}, slot={slot}, "
                f"generation={handle.generation}"
            )

    def retain(self, handle: FrameHandle, count: int = 1) -> None:
        count = int(count)
        if count < 0:
            raise ValueError("Retain count cannot be negative")
        if count == 0:
            return

        self._validate(handle)
        self._refs[handle.worker_id][handle.slot] += count

    def release(self, handle: FrameHandle, count: int = 1) -> None:
        count = int(count)
        if count < 0:
            raise ValueError("Release count cannot be negative")
        if count == 0:
            return

        self._validate(handle)
        worker_id = handle.worker_id
        slot = handle.slot
        refs = self._refs[worker_id][slot] - count
        if refs < 0:
            raise RuntimeError(
                f"Frame-handle refcount underflow: {handle}"
            )

        self._refs[worker_id][slot] = refs
        if refs == 0:
            self._free[worker_id].add(slot)

    def release_many(self, handles: Sequence[FrameHandle]) -> None:
        for handle in handles:
            self.release(handle)

    def view(self, handle: FrameHandle) -> np.ndarray:
        self._validate(handle)
        return self.views[handle.worker_id][handle.slot]

    def live_count(self) -> int:
        return sum(
            1
            for worker_refs in self._refs
            for refs in worker_refs
            if refs > 0
        )
