"""Shared-memory and event-driven process lifecycle for per-GPU workers."""

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
from .gpu_worker_process import gpu_worker_main
from .task_protocol import TaskError, WorkerReady

_TASK_TIMEOUT = 300.0


class GPUWorkerTransport:
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

        self.task_queues = [
            self.context.SimpleQueue()
            for _ in self.gpu_ids
        ]
        self.result_receivers = []
        self.result_senders = []
        for _ in self.gpu_ids:
            receiver, sender = self.context.Pipe(duplex=False)
            self.result_receivers.append(receiver)
            self.result_senders.append(sender)

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
        self.processes = []

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

            for worker_id, gpu_id in enumerate(self.gpu_ids):
                process = self.context.Process(
                    target=gpu_worker_main,
                    args=(
                        worker_id,
                        gpu_id,
                        self.task_queues[worker_id],
                        self.result_senders[worker_id],
                        self.sr_output_slots[worker_id],
                        asdict(config),
                        dict(bvs_config),
                        str(rife_weights or ""),
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
                process.start()
                self.processes.append(process)

            for sender in self.result_senders:
                sender.close()
            self.result_senders = []
            self._wait_ready()
        except Exception:
            self.close()
            raise

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
                "Insufficient /dev/shm capacity for unified GPU workers: "
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
        return (
            list(self.result_receivers)
            + [self._wakeup_receiver]
            + [process.sentinel for process in self.processes]
        )

    def _drain_ready(self, ready) -> bool:
        changed = False
        receiver_ids = {
            id(receiver): receiver
            for receiver in self.result_receivers
        }
        sentinel_to_worker = {
            process.sentinel: worker_id
            for worker_id, process in enumerate(self.processes)
        }
        for item in ready:
            receiver = receiver_ids.get(id(item))
            if receiver is not None:
                while receiver.poll():
                    self._pending_messages.append(receiver.recv())
                    changed = True
                continue
            if item is self._wakeup_receiver:
                while self._wakeup_receiver.poll():
                    try:
                        self._wakeup_receiver.recv_bytes()
                    except EOFError:
                        break
                    changed = True
                continue
            worker_id = sentinel_to_worker.get(item)
            if worker_id is not None:
                if not self.processes[worker_id].is_alive():
                    changed = True
        return changed

    def wait_for_event(self, timeout: float | None) -> bool:
        if self._pending_messages:
            return True
        ready = connection.wait(self._connection_set(), timeout=timeout)
        if not ready:
            return False
        return self._drain_ready(ready)

    def _wait_ready(self) -> None:
        ready_workers: set[int] = set()
        deadline = time.monotonic() + _TASK_TIMEOUT
        while len(ready_workers) < self.count:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Timed out loading unified GPU workers")
            self.wait_for_event(remaining)
            while self._pending_messages:
                message = self._pending_messages.popleft()
                if isinstance(message, TaskError):
                    raise RuntimeError(
                        f"GPU worker {message.worker_id} failed during startup: "
                        f"{message.error}\n{message.traceback_text}"
                    )
                if not isinstance(message, WorkerReady):
                    raise RuntimeError(
                        "Unexpected unified GPU startup message: "
                        f"{type(message).__name__}"
                    )
                ready_workers.add(message.worker_id)

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
                            "Timed out waiting for unified GPU worker"
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
        return self.processes[worker_id].is_alive()

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True

        for slot in self.sr_output_slots:
            try:
                slot.release()
            except Exception:
                pass
        for task_queue in self.task_queues:
            try:
                task_queue.put(None)
            except Exception:
                pass
        for process in self.processes:
            process.join(timeout=10)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)

        for task_queue in self.task_queues:
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
