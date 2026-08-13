from __future__ import annotations

import unittest

import numpy as np

from inference.output_runtime import OutputPump


class _Progress:
    disable = False
    total = 1

    def update(self, _count):
        pass

    def set_postfix_str(self, *_args, **_kwargs):
        pass


class _Writer:
    def __init__(self, events):
        self.events = events
        self.handoff_pending = True
        self.handoff_gpu_id = 0

    def write(self, _frame):
        self.events.append("write")
        self.handoff_pending = False

    def close(self):
        pass


class _Workers:
    gpu_ids = [0]
    count = 1
    sr_output_buffers = 2

    def __init__(self, events):
        self.events = events
        self.frame = np.zeros((4, 4, 3), dtype=np.uint8)

    def trim_cuda_cache(self, worker_id):
        self.events.append(f"trim:{worker_id}")
        return ()

    def output(self, _worker_id, _slot):
        return self.frame

    def release(self, worker_id, slot):
        self.events.append(f"release:{worker_id}:{slot}")


class EncoderHandoffTests(unittest.TestCase):
    def test_trim_precedes_real_writer_and_slot_release(self):
        events = []
        workers = _Workers(events)
        writer = _Writer(events)
        pump = OutputPump(workers, writer, 4, 4, _Progress(), 0.0)
        try:
            self.assertEqual(pump.encoder_handoff_worker_id(), 0)
            pump.handoff_first_output(0, 0, 1)
            self.assertEqual(events, ["trim:0", "write", "release:0:1"])
            self.assertIsNone(pump.encoder_handoff_worker_id())
        finally:
            pump.stop()


if __name__ == "__main__":
    unittest.main()
