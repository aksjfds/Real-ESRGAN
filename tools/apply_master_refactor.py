from __future__ import annotations

import json
from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if old not in text:
        raise RuntimeError(f"{label} not found")
    return text.replace(old, new, 1)


# ---- inference/runtime.py -------------------------------------------------
runtime_path = Path("inference/runtime.py")
text = runtime_path.read_text()
text = text.replace("import threading\n", "")
text = text.replace("from datetime import datetime\n", "")
model_import = "from .models.srvgg_arch import SRVGGNetCompact\n"
if "from . import autotune\n" not in text:
    text = replace_once(text, model_import, model_import + "from . import autotune\n", "autotune import point")

start = text.index("\ndef now_text() -> str:\n")
end = text.index("\ndef format_seconds", start)
text = text[:start] + text[end:]

text = replace_once(
    text,
    '    if target.is_file() and target.stat().st_size > 0:\n'
    '        print(f"[model] using cached weight: {target}", flush=True)\n'
    '        return target\n',
    '    if target.is_file() and target.stat().st_size > 0:\n'
    '        return target\n',
    "cached model log",
)

start = text.index("\nclass PeriodicRefresh:\n")
end = text.index("\ndef mux_audio(", start)
text = text[:start] + text[end:]

text = text.replace(
    '    if args.progress_interval <= 0:\n'
    '        raise ValueError("--progress-interval must be positive.")\n',
    '',
)
text = text.replace(
    '    parser.add_argument("--progress-interval", type=float, default=60.0, help="Forced progress refresh interval")\n',
    '',
)

old_tile_parser = '''    parser.add_argument(
        "--tile-size",
        type=int,
        default=256,
        help="0 uses fastest full-frame inference; use tiles only as an OOM fallback",
    )
    parser.add_argument("--overlap", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4, help="Tiles per inference batch on each GPU")
'''
new_tile_parser = '''    parser.add_argument(
        "--tile-size",
        type=int,
        default=256,
        help="Manual/fallback tile size; 0 uses full-frame inference",
    )
    parser.add_argument(
        "--auto-tile",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Benchmark full-frame and quality-preserving tile sizes, then use the fastest safe choice",
    )
    parser.add_argument(
        "--max-tile-size",
        type=int,
        default=1536,
        help="Largest tile considered by automatic tuning; auto-tile never tests below 512",
    )
    parser.add_argument("--overlap", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4, help="Manual/fallback tiles per inference batch on each GPU")
    parser.add_argument(
        "--auto-batch",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Benchmark safe batch sizes for tiled candidates",
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=32,
        help="Largest batch considered by automatic tuning",
    )
'''
text = replace_once(text, old_tile_parser, new_tile_parser, "tile parser block")

validation_marker = '    if args.batch_size < 1:\n        raise ValueError("--batch-size must be at least 1.")\n'
validation_extra = validation_marker + (
    '    if args.auto_tile and args.max_tile_size < 512:\n'
    '        raise ValueError("--max-tile-size must be at least 512 when auto-tile is enabled.")\n'
    '    if args.auto_batch and args.max_batch_size < 1:\n'
    '        raise ValueError("--max-batch-size must be at least 1 when auto-batch is enabled.")\n'
)
text = replace_once(text, validation_marker, validation_extra, "validation block")

start = text.index("\ndef log_devices(")
end = text.index("\ndef process_video(", start)
text = text[:start] + text[end:]

start = text.index("    model_paths = resolve_model_paths(args)\n", text.index("def process_video"))
end_marker = "    log_devices(gpu_ids, effective_fp16)\n"
end = text.index(end_marker, start) + len(end_marker)
startup = '''    model_paths = resolve_model_paths(args)
    requested_config = WorkerConfig(
        model_name=args.model,
        model_paths=model_paths,
        denoise_strength=args.denoise_strength,
        scale=args.scale,
        tile_size=args.tile_size,
        batch_size=args.batch_size,
        fp16=effective_fp16,
        channels_last=effective_channels_last,
    )

    tune_result = None
    cuda_gpu_ids = [int(gpu_id) for gpu_id in gpu_ids if gpu_id is not None]
    if (args.auto_tile or args.auto_batch) and cuda_gpu_ids:
        tune_gpu = min(
            cuda_gpu_ids,
            key=lambda gpu_id: torch.cuda.get_device_properties(gpu_id).total_memory,
        )
        tune_result = autotune.select_parameters(
            config_dict=asdict(requested_config),
            gpu_id=tune_gpu,
            width=input_width,
            height=input_height,
            gpu_count=len(cuda_gpu_ids),
            overlap=args.overlap,
            auto_tile=args.auto_tile,
            max_tile_size=args.max_tile_size,
            auto_batch=args.auto_batch,
            max_batch_size=args.max_batch_size,
            requested_tile=args.tile_size,
            requested_batch=args.batch_size,
        )
        args.tile_size = tune_result.tile_size
        args.batch_size = tune_result.batch_size

    config = WorkerConfig(
        model_name=args.model,
        model_paths=model_paths,
        denoise_strength=args.denoise_strength,
        scale=args.scale,
        tile_size=args.tile_size,
        batch_size=args.batch_size,
        fp16=effective_fp16,
        channels_last=effective_channels_last,
    )

    source_bit_depth = int(globals().get("_source_bit_depth", 8))
    source_pix_fmt = str(globals().get("_source_pix_fmt", "unknown"))
    if source_bit_depth == 10:
        output_pix_fmt = "p010le" if args.video_codec.endswith("_nvenc") else "yuv420p10le"
    else:
        output_pix_fmt = "yuv420p"

    if gpu_ids == [None]:
        device_text = "CPU | FP32"
    else:
        device_parts = []
        for gpu_id in cuda_gpu_ids:
            props = torch.cuda.get_device_properties(gpu_id)
            device_parts.append(
                f"cuda:{gpu_id} {props.name} {props.total_memory / 2**30:.1f}GiB"
            )
        precision = "FP16" if effective_fp16 else "FP32"
        device_text = "; ".join(device_parts) + f" | {precision}"

    mode = "test" if args.test_seconds > 0 else "full/selected range"
    print("=== Real-ESRGAN ===", flush=True)
    print(
        f"Input   : {input_path.name} | {info.width}x{info.height} | "
        f"{info.fps:.3f} fps | {source_bit_depth}-bit ({source_pix_fmt})",
        flush=True,
    )
    print(
        f"Output  : {output_width}x{output_height} | {output_fps:.3f} fps | "
        f"{source_bit_depth}-bit ({output_pix_fmt}) | {args.video_codec}",
        flush=True,
    )
    print(
        f"Range   : {mode} | {format_seconds(start)} -> {format_seconds(end)} | "
        f"{duration:.3f}s | {expected_frames} inference frames / {expected_output_frames} output frames",
        flush=True,
    )
    print(
        f"Model   : {args.model} | scale={args.scale:g}x | channels_last={effective_channels_last}",
        flush=True,
    )
    print(f"GPU     : {device_text}", flush=True)
    if tune_result is not None:
        print(
            f"Tuning  : tile={tune_result.tile_label} | batch={tune_result.batch_size} | "
            f"estimate={tune_result.estimated_seconds:.3f}s/frame | "
            f"tested={tune_result.tested}, OOM={tune_result.rejected_oom} | "
            f"search={tune_result.search_seconds:.1f}s",
            flush=True,
        )
    elif args.tile_size == 0:
        print(f"Tuning  : manual | full-frame | parallel_frames={len(gpu_ids)}", flush=True)
    else:
        print(
            f"Tuning  : manual | tile={args.tile_size} | overlap={args.overlap} | batch={args.batch_size}",
            flush=True,
        )
    print(flush=True)
'''
text = text[:start] + startup + text[end:]

text = replace_once(
    text,
    '            mininterval=1.0,\n        )\n',
    '            mininterval=1.0,\n            file=sys.stdout,\n        )\n',
    "tqdm stdout",
)
progress_start = text.index(
    "        try:\n            with PeriodicRefresh(progress, args.progress_interval):\n",
    text.index("progress = tqdm("),
)
body_start = progress_start + len(
    "        try:\n            with PeriodicRefresh(progress, args.progress_interval):\n"
)
progress_end = text.index("        finally:\n            progress.close()\n", body_start)
body = text[body_start:progress_end]
dedented = "".join(
    line[4:] if line.startswith("    ") else line
    for line in body.splitlines(keepends=True)
)
text = text[:progress_start] + "        try:\n" + dedented + text[progress_end:]

summary_start = text.index(
    '    print(\n        f"[range] actual_start=',
    text.index("elapsed = time.monotonic() - started"),
)
summary_end = text.index("\n\ndef build_parser", summary_start)
summary = '''    size_mib = output_path.stat().st_size / (1024**2)
    bitrate_mbps = output_path.stat().st_size * 8 / max(actual_duration, 1e-6) / 1_000_000
    average_fps = processed / max(elapsed, 1e-6)
    encode_time = timings["write"] + timings["encode_flush"]
    print("\n=== Completed ===", flush=True)
    print(
        f"Frames  : {processed} | {format_seconds(start)} -> "
        f"{format_seconds(start + actual_duration)} | duration={actual_duration:.3f}s",
        flush=True,
    )
    print(f"Speed   : {average_fps:.3f} frame/s | processing={elapsed:.1f}s", flush=True)
    print(
        f"Timing  : model={timings['model_startup']:.1f}s | decode={timings['decode']:.1f}s | "
        f"inference={timings['inference']:.1f}s | blend={timings['blend']:.1f}s | "
        f"encode={encode_time:.1f}s | audio={timings['audio_mux']:.1f}s",
        flush=True,
    )
    print(f"File    : {size_mib:.2f} MiB | {bitrate_mbps:.2f} Mb/s", flush=True)
    print(f"Output  : {output_path}", flush=True)
'''
text = text[:summary_start] + summary + text[summary_end:]
runtime_path.write_text(text)


# ---- inference/bitdepth.py ------------------------------------------------
bitdepth_path = Path("inference/bitdepth.py")
bitdepth = bitdepth_path.read_text()
old = '''    _SOURCE_BIT_DEPTH, pix_fmt = _stream_bit_depth(streams[0])
    decode_format = "rgb48le" if _SOURCE_BIT_DEPTH == 10 else "rgb24"
    output_format = "10-bit" if _SOURCE_BIT_DEPTH == 10 else "8-bit"
    print(
        f"[bit-depth] source_pix_fmt={pix_fmt}, source={_SOURCE_BIT_DEPTH}-bit, "
        f"inference_rgb={decode_format}, output={output_format}",
        flush=True,
    )
    return info
'''
new = '''    _SOURCE_BIT_DEPTH, pix_fmt = _stream_bit_depth(streams[0])
    base._source_bit_depth = _SOURCE_BIT_DEPTH
    base._source_pix_fmt = pix_fmt
    return info
'''
bitdepth = replace_once(bitdepth, old, new, "bit-depth output block")
bitdepth = replace_once(
    bitdepth,
    '    runtime._bitdepth_preservation_installed = True\n',
    '    runtime._source_bit_depth = _SOURCE_BIT_DEPTH\n'
    '    runtime._source_pix_fmt = "unknown"\n'
    '    runtime._bitdepth_preservation_installed = True\n',
    "bit-depth install tail",
)
bitdepth_path.write_text(bitdepth)


# ---- encoder logging ------------------------------------------------------
hevc_path = Path("encode/hevc.py")
hevc = hevc_path.read_text()
hevc = hevc.replace(
    '''        print(
            f"[encoder] pixel format: {raw_pix_fmt} -> {output_pix_fmt} ({self.codec})",
            flush=True,
        )
''',
    '',
)
hevc_path.write_text(hevc)

av1_path = Path("encode/av1.py")
av1 = av1_path.read_text()
av1 = av1.replace(
    '''            print(
                f"[encoder] libaom parallelism: row-mt=1, tiles={tile_name}, "
                f"async_queue={_AOM_QUEUE_DEPTH}",
                flush=True,
            )
''',
    '',
)
av1 = av1.replace(
    '''        print(
            f"[encoder] pixel format: {raw_pix_fmt} -> {output_pix_fmt} ({self.codec})",
            flush=True,
        )
''',
    '',
)
av1 = av1.replace(
    '        print(f"[encoder] runtime OK: {args.video_codec}", flush=True)\n        return\n',
    '        return\n',
)
av1_path.write_text(av1)


# ---- notebook -------------------------------------------------------------
notebook_path = Path("realesrgan.ipynb")
notebook = json.loads(notebook_path.read_text())
notebook["cells"][0]["source"] = '''!rm -rf /kaggle/working/Real-ESRGAN
!git clone --depth 1 --branch master https://github.com/aksjfds/Real-ESRGAN.git /kaggle/working/Real-ESRGAN
!pip install -q -r /kaggle/working/Real-ESRGAN/requirements.txt
!cd /kaggle/working/Real-ESRGAN && python -m py_compile inference.py inference/*.py encode/*.py inference/models/*.py
!cd /kaggle/working/Real-ESRGAN && python inference.py --help >/dev/null
!ffmpeg -hide_banner -encoders 2>/dev/null | grep -E "(libx265|hevc_nvenc|libsvtav1|libaom-av1|av1_nvenc)" || true'''.splitlines(keepends=True)
notebook["cells"][1]["source"] = '''INPUT_VIDEO = "/kaggle/input/datasets/rustacean1/hanime/bz.mp4"
OUTPUT_VIDEO = "/kaggle/working/realesrgan.mp4"

MODEL = "realesr-animevideov3"
MODEL_PATH = ""
SCALE = 2
FPS = "24000/1001"

START_TIME = 0
TEST_SECONDS = 10

DENOISE_STRENGTH = 1.0
FP16 = True
CHANNELS_LAST = True
INPUT_WIDTH = 0
INPUT_HEIGHT = 0

# 自动寻找预计整帧最快的 full-frame / tile + batch 组合。
# 自动 tile 不会测试小于 512 的 tile，避免为了速度牺牲画质。
AUTO_TILE = True
MAX_TILE_SIZE = 1536
AUTO_BATCH = True
MAX_BATCH_SIZE = 32
TILE_SIZE = 512       # 关闭 AUTO_TILE 时使用，也作为手动回退值
OVERLAP = 32
BATCH_SIZE = 4        # 关闭 AUTO_BATCH 时使用
GPU_IDS = "0,1"

# 编码器：
#   CPU HEVC = "libx265"
#   GPU HEVC = "hevc_nvenc"
#   CPU AV1  = "libsvtav1"  # 缺失时自动尝试 libaom-av1
#   CPU AV1  = "libaom-av1"
#   GPU AV1  = "av1_nvenc"
VIDEO_CODEC = "hevc_nvenc"
CRF = 18
PRESET = "medium"
SVTAV1_PRESET = 6
AOM_CPU_USED = 6
CQ = 18
NVENC_PRESET = "p7"
ENCODE_GPU = 0

AUDIO_CODEC = "copy"
AUDIO_BITRATE = "192k"
'''.splitlines(keepends=True)
notebook["cells"][2]["source"] = '''import subprocess
import sys


def boolean_flag(enabled, yes, no):
    return yes if enabled else no


command = [
    sys.executable, "/kaggle/working/Real-ESRGAN/inference.py",
    "--input", INPUT_VIDEO,
    "--output", OUTPUT_VIDEO,
    "--model", MODEL,
    "--model-path", MODEL_PATH,
    "--denoise-strength", str(DENOISE_STRENGTH),
    "--scale", str(SCALE),
    "--fps", str(FPS),
    boolean_flag(FP16, "--fp16", "--no-fp16"),
    boolean_flag(CHANNELS_LAST, "--channels-last", "--no-channels-last"),
    "--input-width", str(INPUT_WIDTH),
    "--input-height", str(INPUT_HEIGHT),
    boolean_flag(AUTO_TILE, "--auto-tile", "--no-auto-tile"),
    "--max-tile-size", str(MAX_TILE_SIZE),
    "--tile-size", str(TILE_SIZE),
    "--overlap", str(OVERLAP),
    boolean_flag(AUTO_BATCH, "--auto-batch", "--no-auto-batch"),
    "--max-batch-size", str(MAX_BATCH_SIZE),
    "--batch-size", str(BATCH_SIZE),
    "--gpu-ids", GPU_IDS,
    "--video-codec", VIDEO_CODEC,
    "--crf", str(CRF),
    "--preset", PRESET,
    "--svtav1-preset", str(SVTAV1_PRESET),
    "--aom-cpu-used", str(AOM_CPU_USED),
    "--cq", str(CQ),
    "--nvenc-preset", NVENC_PRESET,
    "--encode-gpu", str(ENCODE_GPU),
    "--audio-codec", AUDIO_CODEC,
    "--audio-bitrate", AUDIO_BITRATE,
    "--start-time", str(START_TIME),
    "--test-seconds", str(TEST_SECONDS),
    "--ffmpeg-bin", "ffmpeg",
    "--ffprobe-bin", "ffprobe",
]

_result = subprocess.run(command, check=True)
'''.splitlines(keepends=True)
for cell in notebook["cells"]:
    cell["outputs"] = []
    cell["execution_count"] = None
notebook_path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n")


# ---- README ---------------------------------------------------------------
readme_path = Path("README.md")
readme = readme_path.read_text()
readme = readme.replace(
    '- `inference/runtime.py`：视频解码、Real-ESRGAN 模型、多 GPU worker、tile、融合、进度与音频封装。\n',
    '- `inference/runtime.py`：视频解码、Real-ESRGAN 模型、多 GPU worker、tile、融合、进度与音频封装。\n'
    '- `inference/autotune.py`：在实际 GPU 上自动测试 full-frame / tile / batch，并选择预计整帧最快的安全组合。\n',
)
if "## 自动调参" not in readme:
    readme = readme.replace(
        "## 使用\n",
        "## 自动调参\n\n"
        "Notebook 默认启用 `AUTO_TILE` 和 `AUTO_BATCH`。程序会在实际 GPU 上测试 full-frame 与不小于 512 的 tile，"
        "并在 tiled 候选中测试 batch，过滤 OOM 后按当前多 GPU 调度方式估算整帧耗时，选择最快组合。"
        "关闭自动选项后仍可使用 `TILE_SIZE` / `BATCH_SIZE` 手动指定。\n\n"
        "## 使用\n",
    )
readme_path.write_text(readme)


# ---- validation -----------------------------------------------------------
runtime_final = runtime_path.read_text()
notebook_final = notebook_path.read_text()
assert "PROGRESS_INTERVAL" not in notebook_final
assert "--progress-interval" not in runtime_final
assert "PeriodicRefresh" not in runtime_final
assert "progress_interval" not in runtime_final
assert "--auto-tile" in runtime_final
assert "--auto-batch" in runtime_final
assert "file=sys.stdout" in runtime_final
assert "CompletedProcess" not in notebook_final
assert "[command]" not in notebook_final
assert len(notebook["cells"]) == 3
