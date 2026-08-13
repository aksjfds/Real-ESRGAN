from __future__ import annotations

import unittest
from unittest import mock

import torch

from inference import cuda_memory_policy as policy


class CudaMemoryPolicyTests(unittest.TestCase):
    def test_user_allocator_policy_is_preserved(self) -> None:
        env = {"PYTORCH_ALLOC_CONF": "backend:cudaMallocAsync"}
        value, changed = policy.configure_cuda_allocator_env(env)
        self.assertEqual(value, "backend:cudaMallocAsync")
        self.assertFalse(changed)

    def test_default_policy_uses_expandable_segments(self) -> None:
        env: dict[str, str] = {}
        value, changed = policy.configure_cuda_allocator_env(env)
        self.assertTrue(changed)
        self.assertIn("expandable_segments:True", value)
        self.assertIn("garbage_collection_threshold:0.80", value)

    def test_trim_runs_only_under_low_free_high_cache_pressure(self) -> None:
        device = torch.device("cuda:0")
        gib = 1024 ** 3
        with (
            mock.patch.object(torch.cuda, "mem_get_info", side_effect=[(600 * 1024**2, 15 * gib), (4 * gib, 15 * gib)]),
            mock.patch.object(torch.cuda, "memory_allocated", return_value=7 * gib),
            mock.patch.object(torch.cuda, "memory_reserved", return_value=11 * gib),
            mock.patch.object(torch.cuda, "empty_cache") as empty_cache,
        ):
            result = policy.trim_cuda_cache_under_pressure(device, enabled=True)
        self.assertIsNotNone(result)
        empty_cache.assert_called_once_with()
        assert result is not None
        self.assertGreater(result.free_after, result.free_before)

    def test_trim_skips_when_free_memory_is_healthy(self) -> None:
        device = torch.device("cuda:0")
        gib = 1024 ** 3
        with (
            mock.patch.object(torch.cuda, "mem_get_info", return_value=(4 * gib, 15 * gib)),
            mock.patch.object(torch.cuda, "memory_allocated", return_value=7 * gib),
            mock.patch.object(torch.cuda, "memory_reserved", return_value=11 * gib),
            mock.patch.object(torch.cuda, "empty_cache") as empty_cache,
        ):
            result = policy.trim_cuda_cache_under_pressure(device, enabled=True)
        self.assertIsNone(result)
        empty_cache.assert_not_called()


if __name__ == "__main__":
    unittest.main()
