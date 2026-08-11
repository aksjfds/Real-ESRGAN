"""Stage-isolated GPU worker entrypoints with permanent CUDA affinity."""

from __future__ import annotations

from multiprocessing import shared_memory
from pathlib import Path
import os
import time
import traceback

import numpy as np
import torch

from . import runtime_api as base
from .bvs_runtime import BasicVSRRuntime
from .gpu_task_handlers import (
    WorkerContext,
    build_sr_handlers,
    build_temporal_handlers,
)
from .optimized_rife425 import OptimizedRIFE425Interpolator
from .task_protocol import (
    TaskComputeDone,
    TaskError,
    TaskResult,
    TaskStarted,
    WorkerReady,
    WorkerRole,
)


def _configure_worker_cpu_threads() -> None:
    """Keep CUDA worker processes from oversubscribing notebook CPU cores."""
    cpu_count = max(1, int(os.cpu_count() or 1))
    torch.set_num_threads(1 if cpu_count > 1 else cpu_count)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def _run_task_loop(
    *,
    worker_id: int,
    gpu_id: int,
    role: WorkerRole,
    task_queue,
    result_conn,
    context: WorkerContext,
    handlers,
    enable_gpu_timing: bool,
) -> None:
    def send(message) -> None:
        result_conn.send(message)

    send(WorkerReady(worker_id, gpu_id, role))

    while True:
        task = task_queue.get()
        if task is None:
            break

        handler = handlers.get(task.kind)
        if handler is None:
            raise RuntimeError(
                f"{role.value} worker has no handler for {task.kind!r}"
            )

        send(TaskStarted(worker_id, task.task_id, task.kind))
        started = time.monotonic()
        start_event = None
        if enable_gpu_timing:
            start_event = torch.cuda.Event(enable_timing=True)
            start_event.record()

        compute_notified = False
        gpu_seconds: float | None = None

        def notify_compute_done() -> None:
            nonlocal compute_notified, gpu_seconds
            if compute_notified:
                return

            boundary = torch.cuda.Event(enable_timing=enable_gpu_timing)
            boundary.record()
            boundary.synchronize()
            if start_event is not None:
                gpu_seconds = start_event.elapsed_time(boundary) / 1000.0

            compute_notified = True
            send(TaskComputeDone(worker_id, task.task_id, task.kind))

        context.compute_done = notify_compute_done
        payload = handler(context, task)
        if not compute_notified:
            notify_compute_done()
        context.compute_done = None

        send(
            TaskResult(
                worker_id=worker_id,
                task_id=task.task_id,
                kind=task.kind,
                seconds=time.monotonic() - started,
                payload=payload,
                gpu_seconds=gpu_seconds,
            )
        )


def gpu_temporal_worker_main(
    worker_id: int,
    gpu_id: int,
    task_queue,
    result_conn,
    bvs_config_dict: dict[str, object],
    rife_weights: str,
    input_name: str,
    frame_output_name: str,
    input_slots: int,
    frame_output_slots: int,
    input_shape: tuple[int, int, int],
    dtype_str: str,
    enable_gpu_timing: bool,
) -> None:
    input_shm = None
    frame_output_shm = None
    bvs = None
    rife = None

    try:
        _configure_worker_cpu_threads()
        device = torch.device(f"cuda:{gpu_id}")
        dtype = np.dtype(dtype_str)

        with torch.cuda.device(device):
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cuda.matmul.allow_tf32 = True

            from .basicvsrpp_api import BasicVSRPPConfig

            local_bvs_config = dict(bvs_config_dict)
            local_bvs_config["gpu_id"] = int(gpu_id)
            bvs = BasicVSRRuntime(
                BasicVSRPPConfig(**local_bvs_config),
                checkpoint_dir=Path(__file__).resolve().parent / "weights",
            )

            if rife_weights:
                rife = OptimizedRIFE425Interpolator(
                    gpu_id,
                    Path(rife_weights),
                )

            input_shm = shared_memory.SharedMemory(name=input_name)
            frame_output_shm = shared_memory.SharedMemory(
                name=frame_output_name
            )

            input_view = np.ndarray(
                (input_slots, *input_shape),
                dtype=dtype,
                buffer=input_shm.buf,
            )
            frame_output_view = np.ndarray(
                (frame_output_slots, *input_shape),
                dtype=dtype,
                buffer=frame_output_shm.buf,
            )
            context = WorkerContext(
                device=device,
                dtype=dtype,
                input_view=input_view,
                frame_output_view=frame_output_view,
                bvs=bvs,
                rife=rife,
            )
            _run_task_loop(
                worker_id=worker_id,
                gpu_id=gpu_id,
                role=WorkerRole.TEMPORAL,
                task_queue=task_queue,
                result_conn=result_conn,
                context=context,
                handlers=build_temporal_handlers(),
                enable_gpu_timing=enable_gpu_timing,
            )
    except Exception as error:
        try:
            result_conn.send(
                TaskError(
                    worker_id,
                    repr(error),
                    traceback.format_exc(),
                    WorkerRole.TEMPORAL,
                )
            )
        except Exception:
            pass
    finally:
        for model in (rife, bvs):
            try:
                if model is not None:
                    model.close()
            except Exception:
                pass
        for shm in (input_shm, frame_output_shm):
            if shm is not None:
                try:
                    shm.close()
                except Exception:
                    pass
        try:
            result_conn.close()
        except Exception:
            pass


def gpu_sr_worker_main(
    worker_id: int,
    gpu_id: int,
    task_queue,
    result_conn,
    config_dict: dict[str, object],
    input_name: str,
    frame_output_name: str,
    sr_output_name: str,
    input_slots: int,
    frame_output_slots: int,
    sr_output_buffers: int,
    input_shape: tuple[int, int, int],
    sr_output_shape: tuple[int, int, int],
    dtype_str: str,
    enable_gpu_timing: bool,
) -> None:
    input_shm = None
    frame_output_shm = None
    sr_output_shm = None
    sr_model = None

    try:
        _configure_worker_cpu_threads()
        device = torch.device(f"cuda:{gpu_id}")
        dtype = np.dtype(dtype_str)

        with torch.cuda.device(device):
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cuda.matmul.allow_tf32 = True

            config = base.WorkerConfig(**config_dict)
            sr_model, _native_scale = base.load_worker_model(config, device)

            input_shm = shared_memory.SharedMemory(name=input_name)
            frame_output_shm = shared_memory.SharedMemory(
                name=frame_output_name
            )
            sr_output_shm = shared_memory.SharedMemory(name=sr_output_name)

            input_view = np.ndarray(
                (input_slots, *input_shape),
                dtype=dtype,
                buffer=input_shm.buf,
            )
            frame_output_view = np.ndarray(
                (frame_output_slots, *input_shape),
                dtype=dtype,
                buffer=frame_output_shm.buf,
            )
            sr_output_view = np.ndarray(
                (sr_output_buffers, *sr_output_shape),
                dtype=dtype,
                buffer=sr_output_shm.buf,
            )
            context = WorkerContext(
                device=device,
                dtype=dtype,
                input_view=input_view,
                frame_output_view=frame_output_view,
                sr_model=sr_model,
                sr_output_view=sr_output_view,
            )
            _run_task_loop(
                worker_id=worker_id,
                gpu_id=gpu_id,
                role=WorkerRole.SR,
                task_queue=task_queue,
                result_conn=result_conn,
                context=context,
                handlers=build_sr_handlers(),
                enable_gpu_timing=enable_gpu_timing,
            )
    except Exception as error:
        try:
            result_conn.send(
                TaskError(
                    worker_id,
                    repr(error),
                    traceback.format_exc(),
                    WorkerRole.SR,
                )
            )
        except Exception:
            pass
    finally:
        try:
            if sr_model is not None:
                del sr_model
        except Exception:
            pass
        for shm in (input_shm, frame_output_shm, sr_output_shm):
            if shm is not None:
                try:
                    shm.close()
                except Exception:
                    pass
        try:
            result_conn.close()
        except Exception:
            pass


gpu_worker_main = gpu_temporal_worker_main
