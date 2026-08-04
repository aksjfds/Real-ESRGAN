from argparse import Namespace
import importlib.util
from pathlib import Path
import sys


spec = importlib.util.spec_from_file_location("realesrgan_cli", Path(__file__).parents[1] / "realesrgan.py")
assert spec is not None and spec.loader is not None
cli = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = cli
spec.loader.exec_module(cli)


def test_anime4k_probe_uses_complete_hardware_filter_graph(monkeypatch, tmp_path):
    shader = tmp_path / "line.hook"
    shader.write_text("//!HOOK MAIN\n")
    monkeypatch.setattr(cli, "require_libplacebo", lambda _ffmpeg: None)
    filters = cli.anime4k_filters(
        Namespace(
            anime4k=True,
            anime4k_strength=1.0,
            anime4k_shaders=shader.name,
            anime4k_shader_dir=str(tmp_path),
            ffmpeg_bin="ffmpeg",
        )
    )
    assert filters[0:2] == ["format=yuv444p16le", "hwupload"]
    assert "custom_shader_path" in filters[2]
    assert filters[-2:] == ["hwdownload", "format=yuv444p16le"]

    captured = []
    monkeypatch.setattr(cli, "run_checked", lambda command, _label: captured.append(command))
    cli.probe_encoder_runtime("ffmpeg", "libx265", "yuv420p10le", 64, 64, 0, filters)
    command = captured[0]
    assert command[command.index("-init_hw_device") + 1] == "vulkan=anime4k"
    graph = command[command.index("-vf") + 1]
    assert graph == ",".join(filters)
