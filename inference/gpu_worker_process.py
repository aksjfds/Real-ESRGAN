"""Child-process entrypoint for a permanently GPU-affine worker."""

from __future__ import annotations

from multiprocessing import shared_memory
from pathlib import Path
import time
import traceback

import numpy as np
import torch

from . import runtime as base
from .gpu_task_handlers import WorkerContext, build_handlers
from .task_protocol import TaskError, TaskResult, TaskStarted, WorkerReady


def gpu_worker_main(
    worker_id: int,
    gpu_id: int,
    task_queue,
    result_queue,
    sr_output_slot,
    config_dict: dict[str, object],
    bvs_config_dict: dict[str, object],
    rife_weights: str,
    input_name: str,
    frame_output_name: str,
    sr_output_name: str,
    input_slots: int,
    frame_output_slots: int,
    input_shape: tuple[int, int, int],
    sr_output_shape: tuple[int, int, int],
    dtype_str: str,
) -> None:
    input_shm = None
    frame_output_shm = None
    sr_output_shm = None
    bvs = None
    rife = None
    sr_model = None

    try:
        device = torch.device(f"cuda:{gpu_id}")
        dtype = np.dtype(dtype_str)

        with torch.cuda.device(device):
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cuda.matmul.allow_tf32 = True

            from . import v54_runtime
            from .basicvsrpp import (
                BasicVSRPPConfig,
                BasicVSRPPPreprocessor,
            )

            v54_runtime.install_basicvsrpp_execution_optimizations()
            local_bvs_config = dict(bvs_config_dict)
            local_bvs_config["gpu_id"] = int(gpu_id)
            bvs = BasicVSRPPPreprocessor(
                BasicVSRPPConfig(**local_bvs_config),
                checkpoint_dir=Path(__file__).resolve().parent / "weights",
            )

            if rife_weights:
                from .rife425 import RIFE425Interpolator

                rife = RIFE425Interpolator(gpu_id, Path(rife_weights))

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
                sr_output_shape,
                dtype=dtype,
                buffer=sr_output_shm.buf,
            )
            context = WorkerContext(
                device=device,
                dtype=dtype,
                bvs=bvs,
                rife=rife,
                sr_model=sr_model,
                input_view=input_view,
                frame_output_view=frame_output_view,
                sr_output_view=sr_output_view,
                sr_output_tensor=torch.from_numpy(sr_output_view),
                sr_output_slot=sr_output_slot,
            )
            handlers = build_handlers()
            result_queue.put(WorkerReady(worker_id, gpu_id))

            while True:
                task = task_queue.get()
                if task is None:
                    break

                handler = handlers.get(task.kind)
                if handler is None:
                    raise RuntimeError(
                        f"No GPU handler registered for {task.kind!r}"
                    )

                result_queue.put(
                    TaskStarted(worker_id, task.task_id, task.kind)
                )
                started = time.monotonic()
                payload = handler(context, task)
                result_queue.put(
                    TaskResult(
                        worker_id,
                        task.task_id,
                        task.kind,
                        time.monotonic() - started,
                        payload,
                    )
                )
    except Exception as error:
        try:
            result_queue.put(
                TaskError(
                    worker_id,
                    repr(error),
                    traceback.format_exc(),
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
