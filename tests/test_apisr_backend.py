from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest
from unittest import mock

import torch


MODULE_PATH = Path(__file__).resolve().parents[1] / "inference" / "apisr_backend.py"
spec = importlib.util.spec_from_file_location("apisr_backend_test_target", MODULE_PATH)
assert spec is not None and spec.loader is not None
backend = importlib.util.module_from_spec(spec)
spec.loader.exec_module(backend)


class CountingNetwork(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return value


class FakeGRL(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.input_resolution = (64, 64)
        self.calls = 0

    def get_table_index_mask(self, device=None, input_resolution=None):
        self.calls += 1
        return {"call": torch.tensor(self.calls)}


class APISRBackendTests(unittest.TestCase):
    def test_wrapper_preserves_batch(self) -> None:
        network = CountingNetwork()
        model = backend.APISRGRL(network)
        value = torch.zeros((2, 3, 8, 8))
        output = model(value)
        self.assertEqual(network.calls, 1)
        self.assertEqual(tuple(output.shape), tuple(value.shape))

    def test_dynamic_resolution_cache_is_one_entry(self) -> None:
        with mock.patch.object(backend, "_load_grl_class", return_value=FakeGRL):
            cached_type = backend._cached_grl_class()
        model = cached_type()
        first = model.get_table_index_mask("cpu", (128, 128))
        second = model.get_table_index_mask("cpu", (128, 128))
        self.assertIs(first, second)
        self.assertEqual(model.calls, 1)
        third = model.get_table_index_mask("cpu", (256, 128))
        self.assertEqual(model.calls, 2)
        self.assertIsNot(first, third)

    def test_release_integrity_constants(self) -> None:
        self.assertEqual(
            backend.APISR_SOURCE_COMMIT,
            "fabe8332413bc7f4024e6db39141c68692e88ea5",
        )
        self.assertEqual(len(backend.APISR_WEIGHT_SHA256), 64)
        self.assertIn("architecture/grl.py", backend.APISR_SOURCE_BLOBS)
        self.assertGreaterEqual(len(backend.APISR_SOURCE_BLOBS), 10)


if __name__ == "__main__":
    unittest.main()
