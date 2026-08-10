"""Shared-memory and process lifecycle for permanent per-GPU workers."""

from __future__ import annotations

import multiprocessing as mp
from multiprocessing import shared_memory
import queue
import threading
import time
from dataclasses import asdict
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
    ) -> None:
        self.context = mp.get_context("spawn")
        self.gpu_ids = [int(value) for value in gpu_ids]
        self.count = len(self.gpu_ids)
        self.dtype = np.dtype(dtype)
        self.input_shape = tuple(int(x) for x in input_shape)
        self.sr_output_shape = tuple(
            int(x) for x in sr_output_shape
        )
        self.input_slots = int(input_slots)
        self.frame_output_slots = int(frame_output_slots)
        self.closed = False

        if self.count < 1:
            raise ValueError(
                "GPUWorkerTransport requires at least one CUDA GPU"
            )

        self.result_queue = self.context.Queue()
        self.task_queues = [
            self.context.Queue(maxsize=1)
            for _ in self.gpu_ids
        ]
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
                        (
                            self.frame_output_slots,
                            *self.input_shape,
                        ),
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
                        self.result_queue,
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
                    ),
                    daemon=True,
                )
                process.start()
                self.processes.append(process)

            self._wait_ready()
        except Exception:
            self.close()
            raise

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

    def _wait_ready(self) -> None:
        ready: set[int] = set()
        deadline = time.monotonic() + _TASK_TIMEOUT
        while len(ready) < self.count:
            left = deadline - time.monotonic()
            if left <= 0:
                raise TimeoutError(
                    "Timed out loading unified GPU workers"
                )
            try:
                message = self.result_queue.get(timeout=left)
            except queue.Empty as error:
                raise TimeoutError(
                    "Timed out loading unified GPU workers"
                ) from error

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
            ready.add(message.worker_id)

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
        if block:
            try:
                return self.result_queue.get(timeout=_TASK_TIMEOUT)
            except queue.Empty as error:
                raise TimeoutError(
                    "Timed out waiting for unified GPU worker"
                ) from error
        return self.result_queue.get_nowait()

    def output(self, worker_id: int) -> np.ndarray:
        return self.sr_output_views[worker_id]

    def release(self, worker_id: int) -> None:
        self.sr_output_slots[worker_id].release()
        with self._sr_lock:
            self._sr_available[worker_id] = True

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
                task_queue.put_nowait(None)
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
        try:
            self.result_queue.close()
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
