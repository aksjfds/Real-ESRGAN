"""Shared-memory and event-driven lifecycle for stage-isolated GPU workers."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict
from multiprocessing import connection, shared_memory
from pathlib import Path
import multiprocessing as mp
import queue
import shutil
import threading
import time
from typing import Sequence

import numpy as np

from . import runtime as base
from .gpu_worker_process import (
    gpu_sr_worker_main,
    gpu_temporal_worker_main,
)
from .task_protocol import TaskError, WorkerReady, WorkerRole

_TASK_TIMEOUT = 300.0


class GPUWorkerTransport:
    """Two permanent CUDA processes per GPU: temporal and SR.

    The scheduler still permits only one heavy task per logical GPU at a time.
    Splitting the processes isolates model/allocator/cuDNN state without changing
    task ordering or allowing uncontrolled same-GPU concurrency.
    """

    def __init__(
        self,
        gpu_ids: Sequence[int],
        config: base.WorkerConfig,
        bvs_config: dict[str, object],
        rife_weights: str,
        input_shape: tuple[int, int, int],
        sr_output_shape: tuple[int, int, int],
        dtype: np.dtype,
        input_slots: int,
        frame_output_slots: int,
        enable_gpu_timing: bool = False,
    ) -> None:
        self.context = mp.get_context("spawn")
        self.gpu_ids = [int(value) for value in gpu_ids]
        self.count = len(self.gpu_ids)
        self.dtype = np.dtype(dtype)
        self.input_shape = tuple(int(x) for x in input_shape)
        self.sr_output_shape = tuple(int(x) for x in sr_output_shape)
        self.input_slots = int(input_slots)
        self.frame_output_slots = int(frame_output_slots)
        self.enable_gpu_timing = bool(enable_gpu_timing)
        self.closed = False

        if self.count < 1:
            raise ValueError(
                "GPUWorkerTransport requires at least one CUDA GPU"
            )

        self.temporal_task_queues = [
            self.context.SimpleQueue()
            for _ in self.gpu_ids
        ]
        self.sr_task_queues = [
            self.context.SimpleQueue()
            for _ in self.gpu_ids
        ]

        self.result_receivers = []
        self.result_senders = []
        self.receiver_meta: list[tuple[int, WorkerRole]] = []
        for worker_id in range(self.count):
            for role in (WorkerRole.TEMPORAL, WorkerRole.SR):
                receiver, sender = self.context.Pipe(duplex=False)
                self.result_receivers.append(receiver)
                self.result_senders.append(sender)
                self.receiver_meta.append((worker_id, role))

        self._closed_result_receivers: set[int] = set()
        self._wakeup_receiver, self._wakeup_sender = self.context.Pipe(
            duplex=False
        )
        self._pending_messages = deque()

        self.sr_output_slots = [
            self.context.Semaphore(1)
            for _ in self.gpu_ids
        ]
        self._sr_available = [True] * self.count
        self._sr_lock = threading.Lock()

        self.input_shms: list[shared_memory.SharedMemory] = []
        self.frame_output_shms: list[shared_memory.SharedMemory] = []
        self.sr_output_shms: list[shared_memory.SharedMemory] = []
        self.input_views: list[np.ndarray] = []
        self.frame_output_views: list[np.ndarray] = []
        self.sr_output_views: list[np.ndarray] = []

        self.temporal_processes = []
        self.sr_processes = []
        self.processes = []
        self.process_meta: list[tuple[int, WorkerRole]] = []

        frame_bytes = (
            int(np.prod(self.input_shape, dtype=np.int64))
            * self.dtype.itemsize
        )
        sr_bytes = (
            int(np.prod(self.sr_output_shape, dtype=np.int64))
            * self.dtype.itemsize
        )
        required_bytes = self.count * (
            frame_bytes * self.input_slots
            + frame_bytes * self.frame_output_slots
            + sr_bytes
        )
        self._check_shared_memory_capacity(required_bytes)

        try:
            self._create_shared_memory(frame_bytes, sr_bytes)
            self._start_processes(config, bvs_config, rife_weights)
            for sender in self.result_senders:
                sender.close()
            self.result_senders = []
            self._wait_ready()
        except Exception:
            self.close()
            raise

    def _create_shared_memory(
        self,
        frame_bytes: int,
        sr_bytes: int,
    ) -> None:
        for _ in self.gpu_ids:
            input_shm = shared_memory.SharedMemory(
                create=True,
                size=frame_bytes * self.input_slots,
            )
            frame_output_shm = shared_memory.SharedMemory(
                create=True,
                size=frame_bytes * self.frame_output_slots,
            )
            sr_output_shm = shared_memory.SharedMemory(
                create=True,
                size=sr_bytes,
            )
            self.input_shms.append(input_shm)
            self.frame_output_shms.append(frame_output_shm)
            self.sr_output_shms.append(sr_output_shm)
            self.input_views.append(
                np.ndarray(
                    (self.input_slots, *self.input_shape),
                    dtype=self.dtype,
                    buffer=input_shm.buf,
                )
            )
            self.frame_output_views.append(
                np.ndarray(
                    (self.frame_output_slots, *self.input_shape),
                    dtype=self.dtype,
                    buffer=frame_output_shm.buf,
                )
            )
            self.sr_output_views.append(
                np.ndarray(
                    self.sr_output_shape,
                    dtype=self.dtype,
                    buffer=sr_output_shm.buf,
                )
            )

    def _sender_for(
        self,
        worker_id: int,
        role: WorkerRole,
    ):
        for sender, meta in zip(self.result_senders, self.receiver_meta):
            if meta == (worker_id, role):
                return sender
        raise RuntimeError(
            f"Missing result sender for cuda:{worker_id}/{role.value}"
        )

    def _start_processes(
        self,
        config: base.WorkerConfig,
        bvs_config: dict[str, object],
        rife_weights: str,
    ) -> None:
        config_dict = asdict(config)
        for worker_id, gpu_id in enumerate(self.gpu_ids):
            temporal = self.context.Process(
                target=gpu_temporal_worker_main,
                args=(
                    worker_id,
                    gpu_id,
                    self.temporal_task_queues[worker_id],
                    self._sender_for(worker_id, WorkerRole.TEMPORAL),
                    dict(bvs_config),
                    str(rife_weights or ""),
                    self.input_shms[worker_id].name,
                    self.frame_output_shms[worker_id].name,
                    self.input_slots,
                    self.frame_output_slots,
                    self.input_shape,
                    self.dtype.str,
                    self.enable_gpu_timing,
                ),
                daemon=True,
            )
            temporal.start()
            self.temporal_processes.append(temporal)
            self.processes.append(temporal)
            self.process_meta.append((worker_id, WorkerRole.TEMPORAL))

            sr = self.context.Process(
                target=gpu_sr_worker_main,
                args=(
                    worker_id,
                    gpu_id,
                    self.sr_task_queues[worker_id],
                    self._sender_for(worker_id, WorkerRole.SR),
                    self.sr_output_slots[worker_id],
                    config_dict,
                    self.input_shms[worker_id].name,
                    self.frame_output_shms[worker_id].name,
                    self.sr_output_shms[worker_id].name,
                    self.input_slots,
                    self.frame_output_slots,
                    self.input_shape,
                    self.sr_output_shape,
                    self.dtype.str,
                    self.enable_gpu_timing,
                ),
                daemon=True,
            )
            sr.start()
            self.sr_processes.append(sr)
            self.processes.append(sr)
            self.process_meta.append((worker_id, WorkerRole.SR))

    @staticmethod
    def _check_shared_memory_capacity(required_bytes: int) -> None:
        shm_path = Path("/dev/shm")
        if not shm_path.exists():
            return
        try:
            free = shutil.disk_usage(shm_path).free
        except OSError:
            return
        reserve = max(64 * 2**20, int(required_bytes * 0.05))
        if required_bytes + reserve > free:
            raise RuntimeError(
                "Insufficient /dev/shm capacity for GPU workers: "
                f"need about {(required_bytes + reserve) / 2**20:.0f} MiB, "
                f"available {free / 2**20:.0f} MiB"
            )

    @property
    def memory_mib(self) -> float:
        return (
            sum(
                shm.size
                for shm in (
                    self.input_shms
                    + self.frame_output_shms
                    + self.sr_output_shms
                )
            )
            / 2**20
        )

    def _connection_set(self):
        receivers = [
            receiver
            for receiver in self.result_receivers
            if id(receiver) not in self._closed_result_receivers
        ]
        return (
            receivers
            + [self._wakeup_receiver]
            + [process.sentinel for process in self.processes]
        )

    def _drain_result_receiver(self, receiver) -> bool:
        changed = False
        while True:
            try:
                if not receiver.poll():
                    break
                self._pending_messages.append(receiver.recv())
                changed = True
            except (EOFError, OSError):
                self._closed_result_receivers.add(id(receiver))
                changed = True
                break
        return changed

    def _drain_ready(self, ready) -> bool:
        changed = False
        receiver_ids = {
            id(receiver): receiver
            for receiver in self.result_receivers
        }
        sentinel_meta = {
            process.sentinel: meta
            for process, meta in zip(self.processes, self.process_meta)
        }

        for item in ready:
            receiver = receiver_ids.get(id(item))
            if receiver is not None:
                changed |= self._drain_result_receiver(receiver)
                continue

            if item is self._wakeup_receiver:
                while self._wakeup_receiver.poll():
                    try:
                        self._wakeup_receiver.recv_bytes()
                    except (EOFError, OSError):
                        break
                    changed = True
                continue

            meta = sentinel_meta.get(item)
            if meta is not None:
                worker_id, role = meta
                process = self._process_for(worker_id, role)
                if not process.is_alive():
                    changed = True

        return changed

    def wait_for_event(self, timeout: float | None) -> bool:
        if self._pending_messages:
            return True
        ready = connection.wait(self._connection_set(), timeout=timeout)
        if not ready:
            return False
        return self._drain_ready(ready)

    def _process_for(
        self,
        worker_id: int,
        role: WorkerRole,
    ):
        processes = (
            self.temporal_processes
            if role is WorkerRole.TEMPORAL
            else self.sr_processes
        )
        return processes[worker_id]

    def _startup_exit_error(
        self,
        ready_workers: set[tuple[int, WorkerRole]],
    ) -> RuntimeError | None:
        for process, meta in zip(self.processes, self.process_meta):
            if meta in ready_workers:
                continue
            if not process.is_alive():
                worker_id, role = meta
                return RuntimeError(
                    f"cuda:{self.gpu_ids[worker_id]} {role.value} worker "
                    f"exited before READY (exitcode={process.exitcode})"
                )
        return None

    def _wait_ready(self) -> None:
        expected = {
            (worker_id, role)
            for worker_id in range(self.count)
            for role in (WorkerRole.TEMPORAL, WorkerRole.SR)
        }
        ready_workers: set[tuple[int, WorkerRole]] = set()
        deadline = time.monotonic() + _TASK_TIMEOUT

        while ready_workers != expected:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                missing = sorted(
                    f"{worker}:{role.value}"
                    for worker, role in expected - ready_workers
                )
                raise TimeoutError(
                    "Timed out loading GPU workers; missing=" + ",".join(missing)
                )

            self.wait_for_event(remaining)
            while self._pending_messages:
                message = self._pending_messages.popleft()
                if isinstance(message, TaskError):
                    role = (
                        message.role.value
                        if message.role is not None
                        else "unknown"
                    )
                    raise RuntimeError(
                        f"GPU worker {message.worker_id}/{role} failed "
                        f"during startup: {message.error}\n"
                        f"{message.traceback_text}"
                    )
                if not isinstance(message, WorkerReady):
                    raise RuntimeError(
                        "Unexpected GPU startup message: "
                        f"{type(message).__name__}"
                    )
                ready_workers.add(
                    (message.worker_id, message.role)
                )

            startup_error = self._startup_exit_error(ready_workers)
            if startup_error is not None:
                raise startup_error

    def can_submit_sr(self, worker_id: int) -> bool:
        with self._sr_lock:
            return self._sr_available[worker_id]

    def claim_sr_output(self, worker_id: int) -> None:
        with self._sr_lock:
            if not self._sr_available[worker_id]:
                raise RuntimeError(
                    f"cuda:{self.gpu_ids[worker_id]} SR output slot is busy"
                )
            self._sr_available[worker_id] = False

    def result(self, block: bool = False):
        if not self._pending_messages:
            if block:
                deadline = time.monotonic() + _TASK_TIMEOUT
                while not self._pending_messages:
                    left = deadline - time.monotonic()
                    if left <= 0:
                        raise TimeoutError(
                            "Timed out waiting for GPU worker"
                        )
                    self.wait_for_event(left)
            else:
                self.wait_for_event(0.0)
        if not self._pending_messages:
            raise queue.Empty
        return self._pending_messages.popleft()

    def output(self, worker_id: int) -> np.ndarray:
        return self.sr_output_views[worker_id]

    def release(self, worker_id: int) -> None:
        self.sr_output_slots[worker_id].release()
        with self._sr_lock:
            self._sr_available[worker_id] = True
        try:
            self._wakeup_sender.send_bytes(b"1")
        except (BrokenPipeError, OSError):
            pass

    def is_alive(self, worker_id: int) -> bool:
        return (
            self.temporal_processes[worker_id].is_alive()
            and self.sr_processes[worker_id].is_alive()
        )

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True

        for slot in self.sr_output_slots:
            try:
                slot.release()
            except Exception:
                pass

        for task_queue in self.temporal_task_queues + self.sr_task_queues:
            try:
                task_queue.put(None)
            except Exception:
                pass

        for process in self.processes:
            process.join(timeout=10)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)

        for task_queue in self.temporal_task_queues + self.sr_task_queues:
            try:
                task_queue.close()
            except Exception:
                pass

        for receiver in self.result_receivers:
            try:
                receiver.close()
            except Exception:
                pass
        for sender in self.result_senders:
            try:
                sender.close()
            except Exception:
                pass
        for conn in (self._wakeup_receiver, self._wakeup_sender):
            try:
                conn.close()
            except Exception:
                pass

        for shm in (
            self.input_shms
            + self.frame_output_shms
            + self.sr_output_shms
        ):
            try:
                shm.close()
                shm.unlink()
            except FileNotFoundError:
                pass
            except Exception:
                pass
