"""Fail-fast event transport layered on the stable per-GPU process transport."""

from __future__ import annotations

import time

from .gpu_transport import GPUWorkerTransport, _TASK_TIMEOUT
from .task_protocol import TaskError, WorkerReady


class StableGPUWorkerTransport(GPUWorkerTransport):
    """Treat result-pipe EOF as worker lifecycle state instead of recv failure."""

    def __init__(self, *args, **kwargs) -> None:
        self._closed_result_receivers: set[int] = set()
        super().__init__(*args, **kwargs)

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

    def _drain_result_receiver(self, worker_id: int, receiver) -> bool:
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
        receiver_to_worker = {
            id(receiver): worker_id
            for worker_id, receiver in enumerate(self.result_receivers)
        }
        sentinel_to_worker = {
            process.sentinel: worker_id
            for worker_id, process in enumerate(self.processes)
        }

        for item in ready:
            worker_id = receiver_to_worker.get(id(item))
            if worker_id is not None:
                changed |= self._drain_result_receiver(worker_id, item)
                continue

            if item is self._wakeup_receiver:
                while self._wakeup_receiver.poll():
                    try:
                        self._wakeup_receiver.recv_bytes()
                    except (EOFError, OSError):
                        break
                    changed = True
                continue

            worker_id = sentinel_to_worker.get(item)
            if worker_id is not None and not self.processes[worker_id].is_alive():
                changed = True

        return changed

    def _startup_exit_error(self, ready_workers: set[int]) -> RuntimeError | None:
        for worker_id, process in enumerate(self.processes):
            receiver = self.result_receivers[worker_id]
            pipe_closed = id(receiver) in self._closed_result_receivers
            if worker_id in ready_workers:
                continue
            if pipe_closed or not process.is_alive():
                return RuntimeError(
                    f"GPU worker {worker_id} exited before reporting READY "
                    f"(exitcode={process.exitcode})"
                )
        return None

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

            startup_error = self._startup_exit_error(ready_workers)
            if startup_error is not None:
                raise startup_error
