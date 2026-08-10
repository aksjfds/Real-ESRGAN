"""Typed GPU task submission with FrameHandle locality and refcounting."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .frame_pool import FrameSlotPool
from .gpu_transport import GPUWorkerTransport
from .task_protocol import (
    BVSGroup,
    BVSTask,
    FrameHandle,
    FrameInput,
    FrameStorage,
    RIFETask,
    SRTask,
)


class UnifiedGPUWorkers(GPUWorkerTransport):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.frames = FrameSlotPool(
            self.gpu_ids,
            self.frame_output_slots,
            self.frame_output_views,
        )

    def available_frame_slots(self, worker_id: int) -> int:
        return self.frames.available(worker_id)

    def can_reserve_frames(self, worker_id: int, count: int) -> bool:
        return self.frames.can_reserve(worker_id, count)

    def retain(self, handle: FrameHandle, count: int = 1) -> None:
        self.frames.retain(handle, count)

    def release_handle(
        self,
        handle: FrameHandle,
        count: int = 1,
    ) -> None:
        self.frames.release(handle, count)

    def release_handles(
        self,
        handles: Sequence[FrameHandle],
    ) -> None:
        self.frames.release_many(handles)

    def frame_view(self, handle: FrameHandle) -> np.ndarray:
        return self.frames.view(handle)

    def live_frame_handles(self) -> int:
        return self.frames.live_count()

    def _copy_raw_input(
        self,
        worker_id: int,
        slot: int,
        frame: np.ndarray,
    ) -> None:
        if (
            frame.shape != self.input_shape
            or np.dtype(frame.dtype) != self.dtype
        ):
            raise ValueError(
                f"Unexpected worker input {frame.shape}/{frame.dtype}; "
                f"expected {self.input_shape}/{self.dtype}"
            )
        np.copyto(
            self.input_views[worker_id][slot],
            frame,
            casting="no",
        )

    def _prepare_handle_input(
        self,
        worker_id: int,
        handle: FrameHandle,
        input_slot: int,
    ) -> tuple[FrameInput, tuple[FrameHandle, ...]]:
        if handle.worker_id == worker_id:
            self.frames.view(handle)
            return (
                FrameInput(FrameStorage.OUTPUT, handle.slot),
                (handle,),
            )

        np.copyto(
            self.input_views[worker_id][input_slot],
            self.frames.view(handle),
            casting="no",
        )
        self.frames.release(handle)
        return FrameInput(FrameStorage.INPUT, input_slot), ()

    def submit_bvs(
        self,
        worker_id: int,
        task_id: int,
        groups: Sequence[tuple[Sequence[np.ndarray], int, int]],
    ) -> tuple[FrameHandle, ...]:
        descriptors: list[BVSGroup] = []
        cursor = 0
        emitted_count = 0

        for frames, emit_start, emit_end in groups:
            count = len(frames)
            if cursor + count > self.input_slots:
                raise RuntimeError(
                    "BVS task exceeds unified GPU input buffer"
                )
            for frame in frames:
                self._copy_raw_input(worker_id, cursor, frame)
                cursor += 1

            descriptors.append(
                BVSGroup(
                    count=count,
                    emit_start=int(emit_start),
                    emit_end=int(emit_end),
                )
            )
            emitted_count += max(
                0,
                int(emit_end) - int(emit_start),
            )

        outputs = self.frames.reserve(worker_id, emitted_count)
        task = BVSTask(
            task_id=int(task_id),
            groups=tuple(descriptors),
            output_slots=tuple(handle.slot for handle in outputs),
        )
        try:
            self.task_queues[worker_id].put(task)
        except Exception:
            self.frames.release_many(outputs)
            raise
        return outputs

    def submit_rife(
        self,
        worker_id: int,
        task_id: int,
        frame0: FrameHandle,
        frame1: FrameHandle,
        timesteps: Sequence[float],
    ) -> tuple[
        tuple[FrameHandle, ...],
        tuple[FrameHandle, ...],
    ]:
        values = tuple(float(value) for value in timesteps)
        outputs = self.frames.reserve(worker_id, len(values))
        deferred: list[FrameHandle] = []

        try:
            input0, deferred0 = self._prepare_handle_input(
                worker_id,
                frame0,
                0,
            )
            input1, deferred1 = self._prepare_handle_input(
                worker_id,
                frame1,
                1,
            )
            deferred.extend(deferred0)
            deferred.extend(deferred1)
            self.task_queues[worker_id].put(
                RIFETask(
                    task_id=int(task_id),
                    frame0=input0,
                    frame1=input1,
                    timesteps=values,
                    output_slots=tuple(
                        handle.slot
                        for handle in outputs
                    ),
                )
            )
        except Exception:
            self.frames.release_many(outputs)
            self.frames.release_many(deferred)
            raise

        return outputs, tuple(deferred)

    def submit_sr(
        self,
        worker_id: int,
        task_id: int,
        frame_id: int,
        frame: FrameHandle,
    ) -> tuple[FrameHandle, ...]:
        self.claim_sr_output(worker_id)
        deferred: tuple[FrameHandle, ...] = ()

        try:
            frame_input, deferred = self._prepare_handle_input(
                worker_id,
                frame,
                0,
            )
            self.task_queues[worker_id].put(
                SRTask(
                    task_id=int(task_id),
                    frame_id=int(frame_id),
                    frame=frame_input,
                )
            )
        except Exception:
            with self._sr_lock:
                self._sr_available[worker_id] = True
            self.frames.release_many(deferred)
            raise

        return deferred
