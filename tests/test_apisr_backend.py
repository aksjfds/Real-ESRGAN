from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
import zipfile

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
    def tearDown(self) -> None:
        backend._load_grl_class.cache_clear()
        backend._cached_grl_class.cache_clear()

    def test_wrapper_preserves_batch(self) -> None:
        network = CountingNetwork()
        model = backend.APISRGRL(network)
        value = torch.zeros((2, 3, 8, 8))
        output = model(value)
        self.assertEqual(network.calls, 1)
        self.assertEqual(tuple(output.shape), tuple(value.shape))

    def test_dynamic_resolution_cache_is_one_entry(self) -> None:
        backend._cached_grl_class.cache_clear()
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
        self.assertFalse(backend.SR_MATMUL_TF32)

    def test_pinned_source_validation_rejects_changed_file(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            for relative in backend.APISR_SOURCE_BLOBS:
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"placeholder")

            def fake_blob_sha(path: Path) -> str:
                relative = path.relative_to(root).as_posix()
                if relative == "architecture/grl.py":
                    return "0" * 40
                return backend.APISR_SOURCE_BLOBS[relative]

            with mock.patch.object(backend, "_git_blob_sha", side_effect=fake_blob_sha):
                with self.assertRaises(RuntimeError):
                    backend._validate_source_dir(root, require_pinned=True)

    def test_grl_import_restores_entire_sys_path(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            grl_path = root / "architecture" / "grl.py"
            grl_path.parent.mkdir(parents=True, exist_ok=True)
            grl_path.write_text("# fake", encoding="utf-8")
            original = list(sys.path)
            previous_arch = sys.modules.pop("architecture", None)

            def fake_import(name: str):
                self.assertEqual(name, "architecture.grl")
                sys.path.append("/tmp/apisr-upstream-side-effect")
                return SimpleNamespace(__file__=str(grl_path), GRL=FakeGRL)

            backend._load_grl_class.cache_clear()
            try:
                with mock.patch.object(backend, "_ensure_apisr_source", return_value=root):
                    with mock.patch.object(
                        backend.importlib,
                        "import_module",
                        side_effect=fake_import,
                    ):
                        self.assertIs(backend._load_grl_class(), FakeGRL)
                self.assertEqual(sys.path, original)
            finally:
                sys.path[:] = original
                if previous_arch is not None:
                    sys.modules["architecture"] = previous_arch
                else:
                    sys.modules.pop("architecture", None)

    def test_archive_extraction_ignores_unrelated_files(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            archive = root / "source.zip"
            destination = root / "out"
            prefix = f"APISR-{backend.APISR_SOURCE_COMMIT}/"
            with zipfile.ZipFile(archive, "w") as bundle:
                for relative in backend.APISR_SOURCE_BLOBS:
                    bundle.writestr(prefix + relative, b"runtime")
                bundle.writestr(prefix + "train_code/unused.py", b"unused")
            backend._extract_pinned_source(archive, destination)
            for relative in backend.APISR_SOURCE_BLOBS:
                self.assertTrue((destination / relative).is_file())
            self.assertFalse((destination / "train_code" / "unused.py").exists())


if __name__ == "__main__":
    unittest.main()
