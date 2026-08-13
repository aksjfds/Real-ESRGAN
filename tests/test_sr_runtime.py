from __future__ import annotations

import unittest

import numpy as np
import torch

from inference import frame_transport, sr_runtime


class SRRuntimeTests(unittest.TestCase):
    def test_uint8_output_spec(self) -> None:
        scale, dtype = sr_runtime._frame_output_spec(np.dtype(np.uint8))
        self.assertEqual(scale, 255.0)
        self.assertEqual(dtype, torch.uint8)

    def test_uint16_spec_matches_runtime_capability(self) -> None:
        candidate = getattr(torch, "uint16", None)
        if isinstance(candidate, torch.dtype):
            scale, dtype = sr_runtime._frame_output_spec(np.dtype(np.uint16))
            self.assertEqual(scale, 65535.0)
            self.assertEqual(dtype, candidate)
            self.assertEqual(
                frame_transport._torch_dtype_for_numpy(np.dtype(np.uint16)),
                candidate,
            )
        else:
            with self.assertRaises(RuntimeError):
                sr_runtime._frame_output_spec(np.dtype(np.uint16))
            with self.assertRaises(TypeError):
                frame_transport._torch_dtype_for_numpy(np.dtype(np.uint16))

    def test_uint16_cuda_probe_is_false_on_cpu(self) -> None:
        supported, detail = sr_runtime.probe_cuda_uint16(torch.device("cpu"))
        self.assertFalse(supported)
        self.assertIn("not CUDA", detail)


if __name__ == "__main__":
    unittest.main()
