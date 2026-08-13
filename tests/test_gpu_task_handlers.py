from __future__ import annotations

import sys
import unittest
from unittest import mock

import numpy as np
import torch

from inference import gpu_task_handlers as handlers
from inference.task_protocol import FrameInput, FrameStorage, SRTask


class GPUTaskHandlerTests(unittest.TestCase):
    @staticmethod
    def _context() -> handlers.WorkerContext:
        return handlers.WorkerContext(
            device=torch.device("cpu"),
            dtype=np.dtype(np.uint8),
            input_view=np.zeros((2, 2, 2, 3), dtype=np.uint8),
            frame_output_view=np.zeros((2, 2, 2, 3), dtype=np.uint8),
            sr_model=torch.nn.Identity(),
            sr_output_view=np.zeros((2, 8, 8, 3), dtype=np.uint8),
            h2d_stager=object(),  # patched inference never dereferences it
        )

    def test_microbatch_retry_runs_after_oom_handler_exits(self) -> None:
        context = self._context()
        task = SRTask(
            task_id=1,
            frame_ids=(10, 11),
            frames=(
                FrameInput(FrameStorage.INPUT, 0),
                FrameInput(FrameStorage.INPUT, 1),
            ),
            output_slots=(0, 1),
        )
        events: list[str] = []

        def fail_batch(*args, **kwargs):
            events.append("batch2")
            raise torch.cuda.OutOfMemoryError("synthetic batch2 OOM")

        def cleanup() -> None:
            self.assertIsNone(sys.exc_info()[0])
            events.append("cleanup")

        def fallback(*args, **kwargs) -> None:
            self.assertIsNone(sys.exc_info()[0])
            events.append("batch1")

        with (
            mock.patch.object(handlers, "infer_cuda_batch", side_effect=fail_batch),
            mock.patch.object(handlers, "_release_cuda_after_oom", side_effect=cleanup),
            mock.patch.object(handlers, "_run_sr_sequential_fallback", side_effect=fallback),
        ):
            result = handlers.run_sr(context, task)

        self.assertEqual(events, ["batch2", "cleanup", "batch1"])
        self.assertFalse(context.sr_micro_batch_enabled)
        self.assertEqual(result.frame_ids, (10, 11))

    def test_probe_honors_model_microbatch_safety(self) -> None:
        class ProbeModel(torch.nn.Identity):
            sr_micro_batch_safe = False

        context = self._context()
        context.sr_model = ProbeModel()
        fake_result = torch.zeros((1, 8, 8, 3), dtype=torch.uint8)
        fake_stream = mock.Mock()

        with (
            mock.patch.object(handlers, "infer_cuda_batch", return_value=fake_result),
            mock.patch.object(handlers, "_store_sr_cuda_batch"),
            mock.patch.object(torch.cuda, "current_stream", return_value=fake_stream),
            mock.patch.object(torch.cuda, "empty_cache"),
        ):
            handlers.probe_sr_worker(context)

        self.assertFalse(context.sr_micro_batch_enabled)
        fake_stream.synchronize.assert_called_once()


if __name__ == "__main__":
    unittest.main()
