#!/usr/bin/env python3
"""Modular, float-first multi-GPU Real-ESRGAN video enhancement for Kaggle.

The parent process owns video decoding, direct tile stitching, progress reporting and
encoding.  Exactly one persistent worker (and therefore one model copy) is
created for every selected GPU.  Workers process fixed-size tiles in batches.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import queue
import shutil
import subprocess
import sys
import threading
import time
import traceback
import types
import urllib.request
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime
from fractions import Fraction
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm

# BasicSR 1.4.2 imports a module removed by newer torchvision releases.  Kaggle
# images often contain such a newer torchvision, so provide the one symbol that
# BasicSR needs before importing it.
try:  # pragma: no cover - depends on the installed torchvision version
    import torchvision.transforms.functional_tensor  # noqa: F401
except (ImportError, ModuleNotFoundError):  # pragma: no cover
    import torchvision.transforms.functional as _tv_functional

    _functional_tensor = types.ModuleType("torchvision.transforms.functional_tensor")
    _functional_tensor.rgb_to_grayscale = _tv_functional.rgb_to_grayscale
    sys.modules["torchvision.transforms.functional_tensor"] = _functional_tensor

from basicsr.archs.rrdbnet_arch import RRDBNet
from enhance.analysis import SourceAnalyzer, run_getnative
from enhance.pipeline import DescaleBackend, FrameEnhancementPipeline, PipelineConfig
from enhance.srvgg_enhanced import EnhancedSRVGGNetCompact, assert_no_extra_state
from enhance.tiles import (
    TileProcessor,
    axis_starts,
    full_frame_dehalo,
    full_frame_lanczos,
    full_frame_range_limit,
)


MODEL_URLS = {
    "RealESRGAN_x4plus": (
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
    ),
    "RealESRNet_x4plus": (
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth",
    ),
    "RealESRGAN_x4plus_anime_6B": (
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
    ),
    "RealESRGAN_x2plus": (
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
    ),
    "realesr-animevideov3": (
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth",
    ),
    "realesr-general-x4v3": (
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth",
    ),
}


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    fps_num: int
    fps_den: int
    duration: float
    frames: Optional[int]
    has_audio: bool
    color_range: Optional[str]
    color_space: Optional[str]
    color_primaries: Optional[str]
    color_transfer: Optional[str]
    chroma_location: Optional[str]

    @property
    def fps(self) -> float:
        return self.fps_num / self.fps_den


@dataclass(frozen=True)
class ColorSpec:
    range: str
    space: str
    primaries: str
    transfer: str
    chroma_location: Optional[str]
    inferred: bool = False

    @property
    def scale_matrix(self) -> str:
        if self.space in {"smpte170m", "bt470bg"}:
            return "bt601"
        if self.space in {"bt2020nc", "bt2020c"}:
            return "bt2020"
        if self.space == "smpte240m":
            return "smpte240m"
        return "bt709"

    @property
    def is_hdr(self) -> bool:
        return self.transfer in {"smpte2084", "arib-std-b67"} or self.primaries == "bt2020"


def _known_color(value: object) -> Optional[str]:
    text = str(value or "").strip().lower()
    return None if text in {"", "unknown", "unspecified", "reserved", "n/a", "none"} else text


def resolve_color_spec(info: VideoInfo, policy: str, hdr_policy: str) -> ColorSpec:
    if policy == "bt709":
        spec = ColorSpec("tv", "bt709", "bt709", "bt709", "left", inferred=True)
    else:
        space = _known_color(info.color_space)
        primaries = _known_color(info.color_primaries)
        transfer = _known_color(info.color_transfer)
        range_value = _known_color(info.color_range)
        inferred = not all((space, primaries, transfer, range_value))
        if space is None or space == "gbr":
            space = "bt709" if info.height >= 720 else "smpte170m"
        if primaries is None:
            primaries = "bt2020" if space.startswith("bt2020") else ("bt709" if info.height >= 720 else "smpte170m")
        if transfer is None:
            transfer = "bt709" if info.height >= 720 else "smpte170m"
        if range_value in {"pc", "jpeg", "full"}:
            range_value = "pc"
        else:
            range_value = "tv"
        chroma = _known_color(info.chroma_location)
        spec = ColorSpec(range_value, space, primaries, transfer, chroma, inferred=inferred)
    if spec.is_hdr and hdr_policy == "reject":
        raise ValueError(
            "HDR/BT.2020 input was detected. This Real-ESRGAN pipeline processes nonlinear RGB code values "
            "and is not HDR-aware. Use --hdr-policy passthrough only if that limitation is intentional."
        )
    if spec.is_hdr:
        print("[warning] HDR metadata is being passed through; the model itself is not HDR-linear", flush=True)
    return spec


def color_filter_chain(spec: ColorSpec, output_pix_fmt: str, anime_filters: Sequence[str]) -> list[str]:
    filters = [
        f"setparams=range=pc:color_primaries={spec.primaries}:color_trc={spec.transfer}:colorspace=gbr",
        f"scale=in_range=pc:out_range={spec.range}:out_color_matrix={spec.scale_matrix}",
    ]
    if anime_filters:
        filters.append("format=yuv444p16le")
        filters.extend(anime_filters)
        filters.append("format=yuv444p16le")
    filters.extend(
        [
            f"format={output_pix_fmt}",
            f"setparams=range={spec.range}:color_primaries={spec.primaries}:"
            f"color_trc={spec.transfer}:colorspace={spec.space}",
        ]
    )
    return filters


def color_output_args(spec: ColorSpec) -> list[str]:
    args = [
        "-color_range", spec.range,
        "-colorspace", spec.space,
        "-color_primaries", spec.primaries,
        "-color_trc", spec.transfer,
    ]
    if spec.chroma_location:
        args += ["-chroma_sample_location", spec.chroma_location]
    return args


@dataclass(frozen=True)
class WorkerConfig:
    model_name: str
    model_paths: Tuple[str, ...]
    denoise_strength: float
    tta_mode: str
    tta_batch_size: int
    shift_ensemble: str
    residual_mode: str
    residual_strength: float
    residual_flat_strength: float
    residual_edge_strength: float
    residual_edge_low: float
    residual_edge_high: float
    base_correction: float
    back_projection_iterations: int
    back_projection_strength: float
    back_projection_kernel: str
    back_projection_clamp: float
    native_scale: int
    pre_pad: int
    batch_size: int
    fp16: bool
    channels_last: bool


def now_text() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def format_seconds(value: float) -> str:
    hours = int(value // 3600)
    minutes = int((value % 3600) // 60)
    seconds = value % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"


def run_checked(command: Sequence[str], label: str) -> subprocess.CompletedProcess:
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"{label} failed (exit {result.returncode}):\n{detail}")
    return result


def require_binary(name: str) -> None:
    if shutil.which(name) is None:
        raise FileNotFoundError(
            f"Required executable '{name}' was not found. Kaggle normally includes ffmpeg; "
            "otherwise install it before running inference."
        )


def require_encoder(ffmpeg_bin: str, encoder: str) -> None:
    result = run_checked([ffmpeg_bin, "-hide_banner", "-encoders"], "ffmpeg encoder probe")
    if encoder not in (result.stdout + result.stderr):
        raise RuntimeError(f"ffmpeg does not provide the requested video encoder: {encoder}")


def encoder_pixel_formats(ffmpeg_bin: str, encoder: str) -> set[str]:
    result = run_checked([ffmpeg_bin, "-hide_banner", "-h", f"encoder={encoder}"], "encoder format probe")
    text = result.stdout + result.stderr
    for line in text.splitlines():
        if "Supported pixel formats:" in line:
            return set(line.split("Supported pixel formats:", 1)[1].strip().split())
    return set()


def resolve_output_pix_fmt(ffmpeg_bin: str, codec: str, requested: str) -> str:
    defaults = {
        "hevc_nvenc": "p010le",
        "libx265": "yuv420p10le",
        "h264_nvenc": "yuv420p",
        "libx264": "yuv420p",
    }
    selected = defaults[codec] if requested == "auto" else requested
    supported = encoder_pixel_formats(ffmpeg_bin, codec)
    if selected in supported:
        return selected
    if requested != "auto":
        raise RuntimeError(
            f"Encoder {codec} does not advertise explicitly requested pixel format {selected}; "
            f"supported={sorted(supported)}"
        )
    fallback = "yuv420p"
    if fallback not in supported:
        raise RuntimeError(f"Encoder {codec} has no supported automatic 4:2:0 output format")
    print(
        f"[warning] {codec} does not advertise {selected}; auto falling back to {fallback}",
        flush=True,
    )
    return fallback


def probe_encoder_runtime(
    ffmpeg_bin: str,
    codec: str,
    pixel_format: str,
    width: int,
    height: int,
    encode_gpu: int,
    video_filters: Sequence[str],
    color_spec: ColorSpec,
    anime4k_enabled: bool,
) -> None:
    """Open the same RGB48/filter/encoder graph used by the real writer."""
    command = [ffmpeg_bin, "-hide_banner", "-loglevel", "error"]
    if anime4k_enabled:
        command += ["-init_hw_device", "vulkan=anime4k", "-filter_hw_device", "anime4k"]
    command += [
        "-f", "lavfi",
        "-i", f"color=black:size={width}x{height}:rate=1,format=rgb48le",
        "-frames:v", "1",
    ]
    if video_filters:
        command += ["-vf", ",".join(video_filters)]
    command += ["-c:v", codec]
    if codec in {"h264_nvenc", "hevc_nvenc"}:
        command += ["-gpu", str(encode_gpu)]
    command += ["-pix_fmt", pixel_format, *color_output_args(color_spec), "-f", "null", "-"]
    run_checked(command, f"{codec} RGB48 runtime probe at {width}x{height}/{pixel_format}")
    print(f"[encoder] runtime OK: {codec}, RGB48 -> {width}x{height}/{pixel_format}", flush=True)
    if anime4k_enabled:
        print("[anime4k] complete single-frame filter graph runtime OK", flush=True)


def require_libplacebo(ffmpeg_bin: str) -> None:
    result = run_checked([ffmpeg_bin, "-hide_banner", "-filters"], "FFmpeg filter probe")
    if "libplacebo" not in (result.stdout + result.stderr):
        raise RuntimeError("Anime4K requested, but this FFmpeg build has no libplacebo filter")


def parse_rate(value: str) -> Fraction:
    if not value or value in {"0/0", "N/A"}:
        raise ValueError(f"Invalid video frame rate: {value!r}")
    rate = Fraction(value)
    if rate <= 0:
        raise ValueError(f"Invalid video frame rate: {value!r}")
    return rate


def parse_output_rate(value: str, source_rate: Fraction) -> Fraction:
    """Resolve an output FPS value while preserving rational rates exactly."""
    normalized = str(value).strip().lower()
    if normalized in {"", "0", "auto", "source", "original"}:
        return source_rate
    try:
        return parse_rate(normalized)
    except (ValueError, ZeroDivisionError) as error:
        raise ValueError(
            "--fps must be source/auto/0, a positive number such as 23 or 60, "
            "or a rational rate such as 24000/1001."
        ) from error


def probe_video(path: Path, ffprobe_bin: str) -> VideoInfo:
    command = [
        ffprobe_bin,
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_streams",
        "-show_format",
        str(path),
    ]
    data = json.loads(run_checked(command, "ffprobe").stdout)
    video_streams = [item for item in data.get("streams", []) if item.get("codec_type") == "video"]
    if not video_streams:
        raise ValueError(f"No video stream found in {path}")
    stream = video_streams[0]
    rate = parse_rate(stream.get("avg_frame_rate") or stream.get("r_frame_rate"))
    duration_value = stream.get("duration") or data.get("format", {}).get("duration")
    if duration_value in {None, "N/A"}:
        raise ValueError("The input has no usable duration metadata.")
    frame_value = stream.get("nb_frames")
    frames = int(frame_value) if frame_value not in {None, "N/A"} else None
    return VideoInfo(
        width=int(stream["width"]),
        height=int(stream["height"]),
        fps_num=rate.numerator,
        fps_den=rate.denominator,
        duration=float(duration_value),
        frames=frames,
        has_audio=any(item.get("codec_type") == "audio" for item in data.get("streams", [])),
        color_range=_known_color(stream.get("color_range")),
        color_space=_known_color(stream.get("color_space")),
        color_primaries=_known_color(stream.get("color_primaries")),
        color_transfer=_known_color(stream.get("color_transfer")),
        chroma_location=_known_color(stream.get("chroma_location")),
    )


def choose_input_size(info: VideoInfo, width: int, height: int) -> Tuple[int, int]:
    if width == 0 and height == 0:
        return info.width, info.height
    if width == 0:
        width = round(info.width * height / info.height)
    elif height == 0:
        height = round(info.height * width / info.width)
    if width < 2 or height < 2:
        raise ValueError("Input width and height must be zero or at least 2 pixels.")
    return width, height


def resolve_range(
    info: VideoInfo,
    start: float,
    test_seconds: float,
    output_rate: Fraction,
) -> Tuple[float, float, int]:
    if start < 0 or start >= info.duration:
        raise ValueError(f"--start-time must be in [0, {info.duration:.3f}).")
    available = info.duration - start
    duration = min(test_seconds, available) if test_seconds > 0 else available
    if duration <= 0:
        raise ValueError("Selected video range is empty.")
    expected = max(1, int(round(duration * float(output_rate))))
    return start, duration, expected


def download_file(url: str, target: Path) -> Path:
    if target.is_file() and target.stat().st_size > 0:
        print(f"[model] using cached weight: {target}", flush=True)
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".part")
    print(f"[model] downloading {url}", flush=True)
    try:
        urllib.request.urlretrieve(url, temporary)
        temporary.replace(target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return target


def resolve_model_paths(args: argparse.Namespace) -> Tuple[str, ...]:
    if args.model_path:
        primary = Path(args.model_path).expanduser().resolve()
        if not primary.is_file():
            raise FileNotFoundError(f"Model weight not found: {primary}")
        if args.model == "realesr-general-x4v3" and args.denoise_strength != 1.0:
            raise ValueError(
                "A custom realesr-general-x4v3 weight can only use --denoise-strength 1. "
                "Use the standard downloadable pair for DNI."
            )
        return (str(primary),)

    urls = MODEL_URLS[args.model]
    if args.model != "realesr-general-x4v3" or args.denoise_strength == 1.0:
        urls = urls[:1]
    weight_dir = Path(__file__).resolve().parent / "weights"
    return tuple(str(download_file(url, weight_dir / url.rsplit("/", 1)[-1])) for url in urls)


def build_model(name: str) -> Tuple[torch.nn.Module, int]:
    if name == "RealESRGAN_x4plus":
        return RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4), 4
    if name == "RealESRNet_x4plus":
        return RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4), 4
    if name == "RealESRGAN_x4plus_anime_6B":
        return RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4), 4
    if name == "RealESRGAN_x2plus":
        return RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2), 2
    if name == "realesr-animevideov3":
        return EnhancedSRVGGNetCompact(
            num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type="prelu"
        ), 4
    if name == "realesr-general-x4v3":
        return EnhancedSRVGGNetCompact(
            num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type="prelu"
        ), 4
    raise ValueError(f"Unsupported model: {name}")


def torch_load_cpu(path: str) -> Dict[str, object]:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch before weights_only was added
        return torch.load(path, map_location="cpu")


def checkpoint_state(path: str) -> Dict[str, torch.Tensor]:
    checkpoint = torch_load_cpu(path)
    if "params_ema" in checkpoint:
        return checkpoint["params_ema"]  # type: ignore[return-value]
    if "params" in checkpoint:
        return checkpoint["params"]  # type: ignore[return-value]
    raise KeyError(f"No params or params_ema found in {path}")


def load_worker_model(config: WorkerConfig, device: torch.device) -> Tuple[torch.nn.Module, int]:
    model, native_scale = build_model(config.model_name)
    state = checkpoint_state(config.model_paths[0])
    if len(config.model_paths) == 2:
        weak_state = checkpoint_state(config.model_paths[1])
        strength = config.denoise_strength
        state = {key: strength * value + (1.0 - strength) * weak_state[key] for key, value in state.items()}
    assert_no_extra_state(model, state)
    print(
        f"[checkpoint] strict=True OK, model={config.model_name}, "
        f"keys={len(state)}, path={config.model_paths[0]}",
        flush=True,
    )
    model.eval().requires_grad_(False)
    if config.fp16 and device.type == "cuda":
        model.half()
    if config.channels_last and device.type == "cuda":
        model.to(device=device, memory_format=torch.channels_last)
    else:
        model.to(device)
    if native_scale != config.native_scale:
        raise RuntimeError(
            f"Configured native scale {config.native_scale} does not match model scale {native_scale}"
        )
    return model, native_scale


def worker_main(
    worker_id: int,
    gpu_id: Optional[int],
    input_queue: mp.Queue,
    output_queue: mp.Queue,
    config_dict: Dict[str, object],
) -> None:
    try:
        config = WorkerConfig(**config_dict)  # type: ignore[arg-type]
        if gpu_id is None:
            device = torch.device("cpu")
        else:
            torch.cuda.set_device(gpu_id)
            device = torch.device(f"cuda:{gpu_id}")
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cuda.matmul.allow_tf32 = True
        model, native_scale = load_worker_model(config, device)
        pipeline = FrameEnhancementPipeline(
            model,
            device,
            PipelineConfig(
                tta_mode=config.tta_mode,
                tta_batch_size=config.tta_batch_size,
                shift_ensemble=config.shift_ensemble,
                residual_mode=config.residual_mode,
                residual_strength=config.residual_strength,
                residual_flat_strength=config.residual_flat_strength,
                residual_edge_strength=config.residual_edge_strength,
                residual_edge_low=config.residual_edge_low,
                residual_edge_high=config.residual_edge_high,
                base_correction=config.base_correction,
                back_projection_iterations=config.back_projection_iterations,
                back_projection_strength=config.back_projection_strength,
                back_projection_kernel=config.back_projection_kernel,
                back_projection_clamp=config.back_projection_clamp,
                native_scale=native_scale,
                pre_pad=config.pre_pad,
                fp16=config.fp16,
                channels_last=config.channels_last,
            ),
        )
        output_queue.put(("ready", worker_id, str(device)))
        while True:
            job = input_queue.get()
            if job is None:
                break
            job_type, job_id, indexed_patches = job
            results = []
            timing_before = pipeline.timing_snapshot()
            # Full-frame jobs contain at most one frame per GPU.  Tile jobs use
            # the configured batch size to improve Tensor Core occupancy.
            batch_size = config.batch_size if job_type == "tiles" else 1
            offset = 0
            while offset < len(indexed_patches):
                first_shape = indexed_patches[offset][1].shape
                chunk = []
                while (
                    offset < len(indexed_patches)
                    and len(chunk) < batch_size
                    and indexed_patches[offset][1].shape == first_shape
                ):
                    chunk.append(indexed_patches[offset])
                    offset += 1
                indices = [item[0] for item in chunk]
                patches = [item[1] for item in chunk]
                outputs = pipeline.enhance_batch(patches)
                results.extend(zip(indices, outputs))
            timing_after = pipeline.timing_snapshot()
            timing_delta = {
                name: timing_after.get(name, 0.0) - timing_before.get(name, 0.0)
                for name in timing_after
            }
            output_queue.put((f"{job_type}_result", worker_id, job_id, results, timing_delta))
    except Exception as error:  # send failures to the parent instead of hanging it
        output_queue.put(("error", worker_id, repr(error), traceback.format_exc()))


class PersistentWorkers:
    def __init__(self, gpu_ids: Sequence[Optional[int]], config: WorkerConfig):
        self.context = mp.get_context("spawn")
        self.output_queue = self.context.Queue()
        self.input_queues = [self.context.Queue(maxsize=1) for _ in gpu_ids]
        self.processes = []
        self.stage_timings: Dict[str, float] = {}
        for worker_id, gpu_id in enumerate(gpu_ids):
            process = self.context.Process(
                target=worker_main,
                args=(worker_id, gpu_id, self.input_queues[worker_id], self.output_queue, asdict(config)),
                daemon=True,
            )
            process.start()
            self.processes.append(process)
        try:
            self._wait_until_ready(len(gpu_ids))
        except Exception:
            self.close()
            raise

    def _wait_until_ready(self, count: int) -> None:
        ready = 0
        deadline = time.monotonic() + 300
        while ready < count:
            timeout = max(0.1, deadline - time.monotonic())
            if timeout <= 0:
                raise TimeoutError("Timed out while loading models on GPU workers.")
            try:
                message = self.output_queue.get(timeout=timeout)
            except queue.Empty as error:
                raise TimeoutError("Timed out while loading models on GPU workers.") from error
            if message[0] == "error":
                raise RuntimeError(f"Worker {message[1]} failed during startup: {message[2]}\n{message[3]}")
            if message[0] == "ready":
                print(f"[gpu] worker={message[1]} model resident on {message[2]}", flush=True)
                ready += 1

    def _infer_distributed(
        self,
        job_type: str,
        job_id: int,
        indexed_images: Sequence[Tuple[int, np.ndarray]],
    ) -> Dict[int, np.ndarray]:
        worker_count = len(self.processes)
        for worker_id, input_queue in enumerate(self.input_queues):
            input_queue.put((job_type, job_id, indexed_images[worker_id::worker_count]))
        merged: Dict[int, np.ndarray] = {}
        received = 0
        while received < worker_count:
            message = self.output_queue.get()
            if message[0] == "error":
                if job_type == "frames":
                    hint = (
                        "\nFull-frame OOM fallback: use --tile-size 576 --tile-pad 16 --batch-size 2; "
                        "then try tile-size 256 with batch 16, 8, or 4. Keep FP16 enabled."
                    )
                else:
                    hint = (
                        "\nTile OOM fallback: lower --batch-size first, then lower --tile-size. "
                        "Keep FP16 enabled unless diagnosing a numerical issue."
                    )
                raise RuntimeError(f"Worker {message[1]} failed: {message[2]}\n{message[3]}{hint}")
            if message[0] != f"{job_type}_result" or message[2] != job_id:
                raise RuntimeError(f"Unexpected worker message: {message[0]}")
            merged.update(message[3])
            if len(message) > 4:
                for name, value in message[4].items():
                    self.stage_timings[name] = self.stage_timings.get(name, 0.0) + float(value)
            received += 1
        if len(merged) != len(indexed_images):
            raise RuntimeError(f"Expected {len(indexed_images)} outputs, received {len(merged)}.")
        return merged

    def infer_tiles(self, frame_id: int, patches: Sequence[np.ndarray]) -> Dict[int, np.ndarray]:
        return self._infer_distributed("tiles", frame_id, list(enumerate(patches)))

    def infer_frames(
        self, batch_id: int, indexed_frames: Sequence[Tuple[int, np.ndarray]]
    ) -> Dict[int, np.ndarray]:
        if len(indexed_frames) > len(self.processes):
            raise ValueError("Full-frame batches may contain at most one frame per GPU worker.")
        return self._infer_distributed("frames", batch_id, indexed_frames)

    def close(self) -> None:
        for input_queue in self.input_queues:
            try:
                input_queue.put_nowait(None)
            except queue.Full:
                pass
        for process in self.processes:
            process.join(timeout=10)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        for item in self.input_queues:
            item.close()
        self.output_queue.close()


class RawVideoReader:
    def __init__(
        self,
        input_path: Path,
        ffmpeg_bin: str,
        width: int,
        height: int,
        fps_rate: str,
        start: float,
        duration: float,
    ) -> None:
        self.frame_bytes = width * height * 3
        vf = f"scale={width}:{height}:flags=lanczos,fps={fps_rate}"
        command = [ffmpeg_bin, "-hide_banner", "-loglevel", "error"]
        if start > 0:
            # Input-side seeking is still accurate while transcoding (ffmpeg's
            # accurate_seek is enabled by default) and avoids decoding minutes
            # of video before an arbitrary 10-second test range.
            command += ["-ss", f"{start:.6f}"]
        command += ["-i", str(input_path)]
        command += [
            "-t",
            f"{duration:.6f}",
            "-vf",
            vf,
            "-an",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ]
        self.width = width
        self.height = height
        self.process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def read(self) -> Optional[np.ndarray]:
        assert self.process.stdout is not None
        data = self.process.stdout.read(self.frame_bytes)
        if not data:
            return None
        if len(data) != self.frame_bytes:
            raise RuntimeError(f"ffmpeg returned a partial raw frame ({len(data)}/{self.frame_bytes} bytes).")
        return np.frombuffer(data, dtype=np.uint8).reshape(self.height, self.width, 3)

    def close(self) -> None:
        if self.process.stdout is not None:
            self.process.stdout.close()
        stderr = b""
        if self.process.stderr is not None:
            stderr = self.process.stderr.read()
            self.process.stderr.close()
        return_code = self.process.wait()
        if return_code != 0:
            raise RuntimeError(f"ffmpeg decode failed (exit {return_code}):\n{stderr.decode(errors='replace')}")


class DescaleRawVideoReader(RawVideoReader):
    """Persistent vspipe → FFmpeg RGB reader using the real descale plugin."""

    def __init__(
        self,
        input_path: Path,
        ffmpeg_bin: str,
        width: int,
        height: int,
        fps_rate: str,
        start: float,
        duration: float,
        source_fps: Fraction,
        kernel: str,
    ) -> None:
        vspipe = shutil.which("vspipe")
        if vspipe is None:
            raise RuntimeError("--descale requires vspipe in PATH")
        try:
            import vapoursynth as vs  # type: ignore

            if not hasattr(vs.core, "descale") or not hasattr(vs.core, "ffms2"):
                raise RuntimeError("VapourSynth descale and ffms2 plugins must both be loaded")
        except ImportError as error:
            raise RuntimeError("--descale requires a compatible VapourSynth Python installation") from error
        start_frame = round(start * float(source_fps))
        frame_count = max(1, round(duration * float(source_fps)))
        function = {"bilinear": "Debilinear", "bicubic": "Debicubic", "lanczos": "Delanczos"}[kernel]
        handle = tempfile.NamedTemporaryFile("w", suffix=".vpy", prefix="realesrgan-descale-", delete=False)
        self.script_path = Path(handle.name)
        script = (
            "import vapoursynth as vs\n"
            "core = vs.core\n"
            f"src = core.ffms2.Source(source={str(input_path)!r})\n"
            f"src = src[{start_frame}:{start_frame + frame_count}]\n"
            f"out = core.descale.{function}(src, width={width}, height={height})\n"
            "out.set_output()\n"
        )
        handle.write(script)
        handle.close()
        self.vspipe_process = subprocess.Popen(
            [vspipe, "--container", "y4m", str(self.script_path), "-"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert self.vspipe_process.stdout is not None
        command = [
            ffmpeg_bin,
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            "pipe:0",
            "-vf",
            f"fps={fps_rate}",
            "-an",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ]
        self.frame_bytes = width * height * 3
        self.width = width
        self.height = height
        self.process = subprocess.Popen(
            command,
            stdin=self.vspipe_process.stdout,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.vspipe_process.stdout.close()

    def close(self) -> None:
        ffmpeg_error: Exception | None = None
        try:
            super().close()
        except Exception as error:
            ffmpeg_error = error
        vspipe_stderr = self.vspipe_process.stderr.read() if self.vspipe_process.stderr else b""
        if self.vspipe_process.stderr:
            self.vspipe_process.stderr.close()
        vspipe_code = self.vspipe_process.wait()
        self.script_path.unlink(missing_ok=True)
        if ffmpeg_error is not None:
            raise ffmpeg_error
        if vspipe_code:
            raise RuntimeError(
                f"vspipe descale failed (exit {vspipe_code}):\n{vspipe_stderr.decode(errors='replace')}"
            )


class RawVideoWriter:
    def __init__(
        self,
        path: Path,
        ffmpeg_bin: str,
        width: int,
        height: int,
        input_fps_rate: str,
        output_fps_rate: str,
        codec: str,
        crf: int,
        preset: str,
        cq: int,
        nvenc_preset: str,
        encode_gpu: int,
        output_pix_fmt: str,
        video_filters: Sequence[str],
        color_spec: ColorSpec,
        anime4k_enabled: bool,
    ) -> None:
        command = [
            ffmpeg_bin,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
        ]
        if anime4k_enabled:
            command += ["-init_hw_device", "vulkan=anime4k", "-filter_hw_device", "anime4k"]
        command += [
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb48le",
            "-s:v",
            f"{width}x{height}",
            "-r",
            input_fps_rate,
            "-i",
            "pipe:0",
            "-an",
        ]
        filters = list(video_filters)
        if output_fps_rate != input_fps_rate:
            # Raising FPS duplicates already-enhanced frames here instead of
            # running identical source frames through the model repeatedly.
            filters.append(f"fps={output_fps_rate}")
        if filters:
            command += ["-vf", ",".join(filters)]
        command += ["-c:v", codec]
        if codec in {"libx264", "libx265"}:
            command += ["-preset", preset, "-crf", str(crf)]
        else:
            # NVENC runs on T4's dedicated encoder block.  CQ is the quality
            # target; it is intentionally separate from software-codec CRF.
            command += [
                "-gpu",
                str(encode_gpu),
                "-preset",
                nvenc_preset,
                "-tune",
                "hq",
                "-rc",
                "vbr",
                "-cq",
                str(cq),
                "-b:v",
                "0",
                "-multipass",
                "fullres",
                "-spatial_aq",
                "1",
                "-temporal_aq",
                "1",
                "-rc-lookahead",
                "32",
                "-bf",
                "3",
            ]
        command += ["-pix_fmt", output_pix_fmt, *color_output_args(color_spec)]
        if codec in {"libx265", "hevc_nvenc"}:
            command += ["-tag:v", "hvc1"]
        command.append(str(path))
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    def write(self, frame: np.ndarray) -> None:
        assert self.process.stdin is not None
        if frame.dtype != np.float32 or frame.ndim != 3 or frame.shape[2] != 3:
            raise TypeError("RawVideoWriter expects float32 HWC RGB")
        if not np.isfinite(frame).all():
            raise ValueError("Non-finite value found before rgb48le encoding")
        frame16 = np.rint(np.clip(frame, 0.0, 1.0) * 65535.0).astype("<u2")
        try:
            self.process.stdin.write(memoryview(np.ascontiguousarray(frame16)).cast("B"))
        except BrokenPipeError as error:
            detail = self.process.stderr.read().decode(errors="replace") if self.process.stderr else ""
            raise RuntimeError(f"ffmpeg encoder closed its input early:\n{detail}") from error

    def close(self) -> None:
        if self.process.stdin is not None:
            self.process.stdin.close()
        stderr = self.process.stderr.read() if self.process.stderr is not None else b""
        if self.process.stderr is not None:
            self.process.stderr.close()
        return_code = self.process.wait()
        if return_code != 0:
            raise RuntimeError(f"ffmpeg encode failed (exit {return_code}):\n{stderr.decode(errors='replace')}")


class PeriodicRefresh:
    def __init__(self, progress: tqdm, interval: float):
        self.progress = progress
        self.interval = interval
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, name="progress-refresh", daemon=True)

    def _run(self) -> None:
        while not self.stop_event.wait(self.interval):
            self.progress.refresh()

    def __enter__(self) -> "PeriodicRefresh":
        self.thread.start()
        return self

    def __exit__(self, *_args: object) -> None:
        self.stop_event.set()
        self.thread.join(timeout=2)
        self.progress.refresh()


def mux_audio(
    silent_video: Path,
    input_path: Path,
    output_path: Path,
    ffmpeg_bin: str,
    start: float,
    duration: float,
    has_audio: bool,
    audio_codec: str,
    audio_bitrate: str,
) -> None:
    if not has_audio:
        silent_video.replace(output_path)
        print("[audio] input has no audio stream; wrote video-only output", flush=True)
        return
    base = [ffmpeg_bin, "-y", "-hide_banner", "-loglevel", "error", "-i", str(silent_video)]
    if audio_codec == "aac":
        command = base + [
            "-ss",
            f"{start:.6f}",
            "-t",
            f"{duration:.6f}",
            "-i",
            str(input_path),
            "-filter_complex",
            f"[1:a:0]atrim=start=0:duration={duration:.6f},asetpts=PTS-STARTPTS[a]",
            "-map",
            "0:v:0",
            "-map",
            "[a]",
            "-map_metadata",
            "1",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            audio_bitrate,
            "-shortest",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
    else:
        command = base + [
            "-ss",
            f"{start:.6f}",
            "-t",
            f"{duration:.6f}",
            "-i",
            str(input_path),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-map_metadata",
            "1",
            "-c",
            "copy",
            "-shortest",
            "-avoid_negative_ts",
            "make_zero",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
    try:
        run_checked(command, "audio mux")
    except RuntimeError:
        if audio_codec != "copy":
            raise
        print("[audio] stream copy failed; retrying with AAC for MP4 compatibility", flush=True)
        mux_audio(
            silent_video,
            input_path,
            output_path,
            ffmpeg_bin,
            start,
            duration,
            has_audio,
            "aac",
            audio_bitrate,
        )


def parse_gpu_ids(value: str) -> List[Optional[int]]:
    if value.strip().lower() == "cpu":
        return [None]
    try:
        ids = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise ValueError("--gpu-ids must be 'cpu' or a comma-separated list such as 0,1.") from error
    if not ids or len(ids) != len(set(ids)) or min(ids) < 0:
        raise ValueError("--gpu-ids must contain unique, non-negative GPU numbers.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Use --gpu-ids cpu only for a slow compatibility run.")
    count = torch.cuda.device_count()
    if max(ids) >= count:
        raise ValueError(f"Requested GPU {max(ids)}, but only {count} CUDA device(s) are visible.")
    return ids


def apply_quality_preset(args: argparse.Namespace, argv: Sequence[str]) -> None:
    """Apply preset defaults without overriding explicitly supplied CLI options."""
    explicit = {item.split("=", 1)[0] for item in argv if item.startswith("--")}
    presets: Dict[str, Dict[str, object]] = {
        "baseline": {
            "tta": "none",
            "shift_ensemble": "none",
            "residual_mode": "official",
            "base_correction": 0.0,
            "back_projection_iterations": 0,
            "dehalo_strength": 0.0,
            "range_limit": 0.0,
        },
        "safe": {
            "tta": "x8",
            "shift_ensemble": "none",
            "residual_mode": "official",
            "base_correction": 0.0,
            "back_projection_iterations": 1,
            "dehalo_strength": 0.0,
            "range_limit": 0.1,
        },
    }
    option_names = {
        "tta": "--tta",
        "shift_ensemble": "--shift-ensemble",
        "residual_mode": "--residual-mode",
        "base_correction": "--base-correction",
        "back_projection_iterations": "--back-projection-iterations",
        "dehalo_strength": "--dehalo-strength",
        "range_limit": "--range-limit",
    }
    for name, value in presets[args.quality_preset].items():
        positive = option_names[name]
        negative = "--no-" + positive[2:] if isinstance(value, bool) else ""
        if positive not in explicit and negative not in explicit:
            setattr(args, name, value)
    if args.overlap is not None:
        if "--tile-pad" in explicit:
            raise ValueError("--overlap and --tile-pad cannot be supplied together")
        args.tile_pad = args.overlap
        print("[deprecated] --overlap now maps to --tile-pad; no feather blending is performed", flush=True)


def anime4k_filters(args: argparse.Namespace) -> list[str]:
    if not args.anime4k:
        return []
    require_libplacebo(args.ffmpeg_bin)
    if args.anime4k_strength != 1.0:
        raise ValueError(
            "Official Anime4K .hook shaders do not expose one universal strength parameter; "
            "--anime4k-strength must remain 1.0"
        )
    if not args.anime4k_shaders:
        raise ValueError(
            "Anime4K requires explicit --anime4k-shaders filenames; preset names are not emulated"
        )
    root = Path(args.anime4k_shader_dir).expanduser().resolve()
    filters = ["hwupload"]
    for name in (item.strip() for item in args.anime4k_shaders.split(",") if item.strip()):
        shader = (root / name).resolve()
        if root not in shader.parents or not shader.is_file():
            raise FileNotFoundError(f"Anime4K shader not found below shader directory: {shader}")
        shader_path = str(shader).replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")
        filters.append(f"libplacebo=custom_shader_path='{shader_path}'")
    filters.append("hwdownload")
    return filters


def validate_args(args: argparse.Namespace) -> None:
    if not 0 < args.scale <= 4:
        raise ValueError("--scale/final scale must satisfy 0 < scale <= 4.")
    if args.tile_size != 0 and (args.tile_size < 64 or args.tile_size % 4):
        raise ValueError("--tile-size must be 0 (full frame), or at least 64 and divisible by 4.")
    if args.tile_pad < 0 or args.pre_pad < 0:
        raise ValueError("--tile-pad and --pre-pad must be non-negative")
    if args.tile_size and args.tile_pad >= args.tile_size // 2:
        raise ValueError("--tile-pad must be less than half of --tile-size")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")
    if not 0 <= args.crf <= 51:
        raise ValueError("--crf must be between 0 and 51.")
    if not 0 <= args.cq <= 51:
        raise ValueError("--cq must be between 0 and 51.")
    if args.encode_gpu < 0:
        raise ValueError("--encode-gpu must be non-negative.")
    if not 0 <= args.denoise_strength <= 1:
        raise ValueError("--denoise-strength must be between 0 and 1.")
    if args.progress_interval <= 0:
        raise ValueError("--progress-interval must be positive.")
    if args.native_samples < 1:
        raise ValueError("--native-samples must be at least 1")
    if not 0 < args.native_confidence <= 1:
        raise ValueError("--native-confidence must be in (0, 1]")
    if args.native_min_height <= 0 or args.native_max_height < args.native_min_height:
        raise ValueError("Invalid native height search range")
    native_kernels = [item.strip() for item in args.native_kernels.split(",") if item.strip()]
    if not native_kernels or any(item not in {"bilinear", "bicubic", "lanczos"} for item in native_kernels):
        raise ValueError("--native-kernels must be a non-empty comma-separated subset of bilinear,bicubic,lanczos")
    if args.tta_batch_size not in {1, 2, 4, 8}:
        raise ValueError("--tta-batch-size must be 1, 2, 4, or 8")
    if not 0 <= args.back_projection_iterations <= 3:
        raise ValueError("--back-projection-iterations must be between 0 and 3")
    if args.back_projection_strength < 0 or args.back_projection_clamp <= 0:
        raise ValueError("Back-projection strength must be non-negative and clamp positive")
    if args.descale and args.native_height == 0 and args.native_analysis == "off":
        raise ValueError("--descale requires --native-height or native analysis")
    if args.descale and (args.input_width or args.input_height):
        raise ValueError(
            "--descale cannot be combined with --input-width/--input-height; "
            "use --native-height to choose the actual descale target"
        )
    if args.descale and args.native_height > 0 and args.native_kernel == "auto" and args.native_analysis == "off":
        raise ValueError("Explicit --descale requires a concrete --native-kernel when analysis is off")
    if not 0 <= args.base_correction <= 0.5:
        raise ValueError("--base-correction must be between 0 and 0.5")
    if args.residual_edge_high <= args.residual_edge_low:
        raise ValueError("--residual-edge-high must be greater than --residual-edge-low")
    if min(args.residual_strength, args.residual_flat_strength, args.residual_edge_strength) < 0:
        raise ValueError("Residual strengths must be non-negative")
    if args.dehalo_radius < 1 or args.range_radius < 1:
        raise ValueError("Dehalo/range radii must be positive")
    if args.dehalo_strength < 0 or args.range_limit < 0:
        raise ValueError("Dehalo/range strengths must be non-negative")
    if args.overshoot < 0 or args.undershoot < 0:
        raise ValueError("--overshoot and --undershoot must be non-negative")


def log_devices(gpu_ids: Sequence[Optional[int]], fp16: bool) -> None:
    for gpu_id in gpu_ids:
        if gpu_id is None:
            print(f"[device] CPU, fp16=False", flush=True)
        else:
            props = torch.cuda.get_device_properties(gpu_id)
            memory_gib = props.total_memory / (1024**3)
            print(
                f"[device] cuda:{gpu_id} {props.name}, memory={memory_gib:.1f} GiB, fp16={fp16}",
                flush=True,
            )


def finalize_output_frame(
    native_output: np.ndarray,
    reference_frame: np.ndarray,
    output_width: int,
    output_height: int,
    args: argparse.Namespace,
    timings: Dict[str, float],
) -> np.ndarray:
    """Apply one identical parent-side finalization path to every frame."""
    stage_started = time.monotonic()
    output = full_frame_lanczos(native_output, output_width, output_height)
    timings["lanczos"] += time.monotonic() - stage_started

    stage_started = time.monotonic()
    output = full_frame_dehalo(output, args.dehalo_strength, args.dehalo_radius)
    timings["dehalo"] += time.monotonic() - stage_started

    stage_started = time.monotonic()
    output = full_frame_range_limit(
        output,
        reference_frame.astype(np.float32) / 255.0,
        args.range_limit,
        args.range_radius,
        args.overshoot,
        args.undershoot,
    )
    timings["range_limit"] += time.monotonic() - stage_started
    return np.ascontiguousarray(np.clip(output, 0.0, 1.0), dtype=np.float32)


def process_video(args: argparse.Namespace) -> None:
    require_binary(args.ffmpeg_bin)
    require_binary(args.ffprobe_bin)
    require_encoder(args.ffmpeg_bin, args.video_codec)
    output_pix_fmt = resolve_output_pix_fmt(args.ffmpeg_bin, args.video_codec, args.output_pix_fmt)
    anime_filters = anime4k_filters(args)
    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input video not found: {input_path}")
    if input_path == output_path:
        raise ValueError("Input and output paths must be different.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_video = output_path.with_name(output_path.stem + ".video_only.tmp.mp4")

    info = probe_video(input_path, args.ffprobe_bin)
    color_spec = resolve_color_spec(info, args.color_policy, args.hdr_policy)
    writer_filters = color_filter_chain(color_spec, output_pix_fmt, anime_filters)
    source_rate = Fraction(info.fps_num, info.fps_den)
    output_rate = parse_output_rate(args.fps, source_rate)
    inference_rate = min(source_rate, output_rate)
    output_fps = float(output_rate)
    output_fps_rate = f"{output_rate.numerator}/{output_rate.denominator}"
    inference_fps = float(inference_rate)
    inference_fps_rate = f"{inference_rate.numerator}/{inference_rate.denominator}"
    input_width, input_height = choose_input_size(info, args.input_width, args.input_height)
    start, duration, expected_frames = resolve_range(info, args.start_time, args.test_seconds, inference_rate)
    expected_output_frames = max(1, int(round(duration * output_fps)))
    end = start + duration
    gpu_ids = parse_gpu_ids(args.gpu_ids)
    effective_fp16 = args.fp16 and gpu_ids != [None]
    effective_channels_last = args.channels_last and gpu_ids != [None]
    native_model_scale = 2 if args.model == "RealESRGAN_x2plus" else 4
    if args.scale > native_model_scale:
        raise ValueError(
            f"Final scale {args.scale:g} exceeds model-native scale {native_model_scale}; "
            "post-model enlargement is not allowed"
        )

    analysis_rows = []
    native_analysis_elapsed = 0.0
    if args.native_analysis != "off":
        analysis_started = time.monotonic()
        analysis_rows = SourceAnalyzer(args.ffmpeg_bin).analyze(
            input_path, info.width, info.height, start, duration, args.native_samples
        )
        for row in analysis_rows:
            print(f"[analysis] {json.dumps(asdict(row), sort_keys=True)}", flush=True)
        recommendation, reason = SourceAnalyzer.recommend(analysis_rows)
        print(f"[analysis] degradation_profile={recommendation}, reason={reason}", flush=True)
        native_analysis_elapsed = time.monotonic() - analysis_started
        print(f"[timing] native/degradation_analysis={native_analysis_elapsed:.2f}s", flush=True)

    selected_native_height = args.native_height
    selected_native_kernel = args.native_kernel
    if selected_native_height > 0 and selected_native_kernel != "auto":
        print(
            f"[native] explicit candidate height={selected_native_height}, kernel={selected_native_kernel}; "
            "automatic getnative skipped",
            flush=True,
        )
    elif args.native_analysis != "off":
        last_source_frame = max(
            0,
            (info.frames if info.frames is not None else int(info.duration * info.fps)) - 1,
        )
        sample_frames = [
            min(last_source_frame, round((start + fraction * duration) * info.fps))
            for fraction in np.linspace(0.0, 0.999, args.native_samples)
        ]
        try:
            candidates = run_getnative(
                input_path,
                sample_frames,
                [item.strip() for item in args.native_kernels.split(",") if item.strip()],
                args.native_min_height,
                args.native_max_height,
            )
            for candidate in candidates:
                print(f"[native] {json.dumps(asdict(candidate), sort_keys=True)}", flush=True)
            groups: Dict[Tuple[int, str], int] = {}
            for candidate in candidates:
                key = (candidate.height, candidate.kernel)
                groups[key] = groups.get(key, 0) + 1
            best, count = max(groups.items(), key=lambda item: item[1])
            # Every source frame is tested with every kernel.  Confidence is
            # therefore agreement across sampled frames for one (height,
            # kernel), not its share of all frame×kernel trials.
            confidence = count / max(args.native_samples, 1)
            if confidence >= args.native_confidence and best[0] < info.height * 0.95:
                selected_native_height, selected_native_kernel = best
                if args.native_analysis == "auto" and not args.descale:
                    if DescaleBackend.available() and shutil.which("vspipe") is not None:
                        args.descale = True
                        auto_status = "auto-enabled"
                    else:
                        auto_status = "unavailable; keeping source size"
                else:
                    auto_status = "enabled" if args.descale else "report-only"
                print(
                    f"[native] consensus height={best[0]}, kernel={best[1]}, confidence={confidence:.3f}; "
                    f"descale={auto_status}",
                    flush=True,
                )
            else:
                print(
                    f"[native] no safe consensus: best={best}, confidence={confidence:.3f}; keeping source size",
                    flush=True,
                )
        except Exception as error:
            if args.native_analysis == "report":
                raise
            print(f"[native] auto disabled: {error}", flush=True)

    if args.descale:
        if not DescaleBackend.available() or shutil.which("vspipe") is None:
            raise RuntimeError(
                "--descale requires a working vspipe plus VapourSynth descale and ffms2 plugins. "
                "No ordinary resize fallback is permitted."
            )
        if selected_native_height <= 0 or selected_native_kernel == "auto":
            raise RuntimeError(
                "Native analysis did not produce a safe height/kernel consensus; "
                "provide both --native-height and --native-kernel or disable --descale"
            )
        input_height = selected_native_height
        input_width = max(2, int(round(input_height * info.width / info.height)))
        input_width -= input_width % 2

    output_width = int(round(input_width * args.scale))
    output_height = int(round(input_height * args.scale))
    if output_width % 2 or output_height % 2:
        raise ValueError(
            f"4:2:0 encoding needs even output dimensions, got {output_width}x{output_height}. "
            "Adjust the selected dimensions or scale."
        )
    probe_encoder_runtime(
        args.ffmpeg_bin,
        args.video_codec,
        output_pix_fmt,
        output_width,
        output_height,
        args.encode_gpu,
        writer_filters,
        color_spec,
        args.anime4k,
    )

    # Download/resolve the main checkpoint only after all inexpensive source,
    # capability and encoder checks have succeeded.
    model_paths = resolve_model_paths(args)

    config = WorkerConfig(
        model_name=args.model,
        model_paths=model_paths,
        denoise_strength=args.denoise_strength,
        tta_mode=args.tta,
        tta_batch_size=args.tta_batch_size,
        shift_ensemble=args.shift_ensemble,
        residual_mode=args.residual_mode,
        residual_strength=args.residual_strength,
        residual_flat_strength=args.residual_flat_strength,
        residual_edge_strength=args.residual_edge_strength,
        residual_edge_low=args.residual_edge_low,
        residual_edge_high=args.residual_edge_high,
        base_correction=args.base_correction,
        back_projection_iterations=args.back_projection_iterations,
        back_projection_strength=args.back_projection_strength,
        back_projection_kernel=args.back_projection_kernel,
        back_projection_clamp=args.back_projection_clamp,
        native_scale=native_model_scale,
        # Every worker returns the complete model-native-scale result.  The
        # parent applies one shared final Lanczos/dehalo/range path for both
        # full-frame and tiled inference.
        pre_pad=0 if args.tile_size else args.pre_pad,
        batch_size=args.batch_size,
        fp16=effective_fp16,
        channels_last=effective_channels_last,
    )
    tile_processor = TileProcessor(
        args.tile_size,
        args.tile_pad,
        args.pre_pad,
        native_model_scale,
        args.tile_verify_coverage,
    )

    mode = "timed test" if args.test_seconds > 0 else "selected/full range"
    print(f"[run] wall_start={now_text()}", flush=True)
    print(f"[input] {input_path}", flush=True)
    print(
        f"[input] source={info.width}x{info.height}, inference={input_width}x{input_height}, "
        f"output={output_width}x{output_height}, source_fps={info.fps:.6f}, "
        f"inference_fps={inference_fps:.6f} ({inference_fps_rate}), "
        f"output_fps={output_fps:.6f} ({output_fps_rate}), audio={info.has_audio}",
        flush=True,
    )
    print(
        f"[range] mode={mode}, start={format_seconds(start)}, end={format_seconds(end)}, "
        f"duration={duration:.3f}s, expected_inference_frames={expected_frames}, "
        f"expected_output_frames={expected_output_frames}",
        flush=True,
    )
    tta_count = 8 if args.tta == "x8" else 1
    shift_count = {"none": 1, "x2": 2, "x4": 4}[args.shift_ensemble]
    print(
        f"[pipeline] model={args.model}, weight={model_paths[0]}, strict_load=True, "
        f"native_scale={native_model_scale}, final_scale={args.scale:g}, "
        f"descale={'on' if args.descale else 'off'}, native_kernel={selected_native_kernel}, "
        f"tta={tta_count}, shift={shift_count}, model_calls={tta_count * shift_count}, "
        f"residual={args.residual_mode}, base_correction={args.base_correction:g}, "
        f"back_projection={args.back_projection_iterations}, "
        f"internal=fp32/fp16-model, raw=rgb48le, encode_pix_fmt={output_pix_fmt}, "
        f"anime4k_shaders={args.anime4k_shaders or 'none'}",
        flush=True,
    )
    print(
        f"[color] policy={args.color_policy}, inferred={color_spec.inferred}, "
        f"range={color_spec.range}, space={color_spec.space}, primaries={color_spec.primaries}, "
        f"transfer={color_spec.transfer}, chroma={color_spec.chroma_location or 'unspecified'}",
        flush=True,
    )
    if tta_count * shift_count > 8:
        print("[warning] ensemble requires many model calls and has high runtime/VRAM pressure", flush=True)
    if args.tile_size == 0:
        print(
            f"[inference] full-frame mode, parallel_frames={len(gpu_ids)}, "
            f"channels_last={effective_channels_last}",
            flush=True,
        )
    else:
        stride = args.tile_size
        tile_count = len(axis_starts(input_width + args.pre_pad, args.tile_size)) * len(
            axis_starts(input_height + args.pre_pad, args.tile_size)
        )
        print(
            f"[tiles] size={args.tile_size}, tile_pad={args.tile_pad}, pre_pad={args.pre_pad}, "
            f"stride={stride}, global_prepad=True, native_stitch_scale={native_model_scale}, "
            f"full_frame_lanczos={args.scale != native_model_scale}, direct_stitch=True, "
            f"tiles_per_frame={tile_count}, batch_per_gpu={args.batch_size}, "
            f"channels_last={effective_channels_last}",
            flush=True,
        )
    log_devices(gpu_ids, effective_fp16)

    reader: Optional[RawVideoReader] = None
    writer: Optional[RawVideoWriter] = None
    workers: Optional[PersistentWorkers] = None
    processed = 0
    started = time.monotonic()
    timings = {
        "model_startup": 0.0,
        "native_analysis": native_analysis_elapsed,
        "descale": 0.0,
        "decode": 0.0,
        "inference": 0.0,
        "tta": 0.0,
        "shift_ensemble": 0.0,
        "realesrgan": 0.0,
        "residual_control": 0.0,
        "base_correction": 0.0,
        "back_projection": 0.0,
        "lanczos": 0.0,
        "tile_crop_stitch": 0.0,
        "dehalo": 0.0,
        "range_limit": 0.0,
        "anime4k": 0.0,
        "float_to_rgb48": 0.0,
        "encode_flush": 0.0,
        "audio_mux": 0.0,
    }
    clean_video_ready = False
    try:
        stage_started = time.monotonic()
        workers = PersistentWorkers(gpu_ids, config)
        timings["model_startup"] += time.monotonic() - stage_started
        if args.descale:
            reader = DescaleRawVideoReader(
                input_path,
                args.ffmpeg_bin,
                input_width,
                input_height,
                inference_fps_rate,
                start,
                duration,
                source_rate,
                selected_native_kernel,
            )
        else:
            reader = RawVideoReader(
                input_path,
                args.ffmpeg_bin,
                input_width,
                input_height,
                inference_fps_rate,
                start,
                duration,
            )
        writer = RawVideoWriter(
            temporary_video,
            args.ffmpeg_bin,
            output_width,
            output_height,
            inference_fps_rate,
            output_fps_rate,
            args.video_codec,
            args.crf,
            args.preset,
            args.cq,
            args.nvenc_preset,
            args.encode_gpu,
            output_pix_fmt,
            writer_filters,
            color_spec,
            args.anime4k,
        )
        progress = tqdm(
            total=expected_frames,
            desc="Real-ESRGAN",
            unit="frame",
            dynamic_ncols=True,
            mininterval=1.0,
        )
        try:
            with PeriodicRefresh(progress, args.progress_interval):
                if args.tile_size == 0:
                    batch_id = 0
                    while True:
                        indexed_frames = []
                        stage_started = time.monotonic()
                        for _ in gpu_ids:
                            frame = reader.read()
                            if frame is None:
                                break
                            indexed_frames.append((processed + len(indexed_frames), frame))
                        timings["descale" if args.descale else "decode"] += time.monotonic() - stage_started
                        if not indexed_frames:
                            break
                        stage_started = time.monotonic()
                        frame_outputs = workers.infer_frames(batch_id, indexed_frames)
                        timings["inference"] += time.monotonic() - stage_started
                        for frame_id, reference_frame in indexed_frames:
                            output = finalize_output_frame(
                                frame_outputs[frame_id],
                                reference_frame,
                                output_width,
                                output_height,
                                args,
                                timings,
                            )
                            stage_started = time.monotonic()
                            writer.write(output)
                            timings["float_to_rgb48"] += time.monotonic() - stage_started
                        processed += len(indexed_frames)
                        batch_id += 1
                        progress.update(len(indexed_frames))
                        elapsed = max(time.monotonic() - started, 1e-6)
                        progress.set_postfix(fps=f"{processed / elapsed:.3f}", refresh=False)
                else:
                    while True:
                        stage_started = time.monotonic()
                        frame = reader.read()
                        timings["descale" if args.descale else "decode"] += time.monotonic() - stage_started
                        if frame is None:
                            break
                        patches, tile_infos = tile_processor.split(frame)
                        stage_started = time.monotonic()
                        tile_outputs = workers.infer_tiles(processed, patches)
                        timings["inference"] += time.monotonic() - stage_started
                        stage_started = time.monotonic()
                        output = tile_processor.stitch(
                            tile_outputs, tile_infos, input_width, input_height
                        )
                        timings["tile_crop_stitch"] = timings.get("tile_crop_stitch", 0.0) + (
                            time.monotonic() - stage_started
                        )
                        output = finalize_output_frame(
                            output,
                            frame,
                            output_width,
                            output_height,
                            args,
                            timings,
                        )
                        stage_started = time.monotonic()
                        writer.write(output)
                        timings["float_to_rgb48"] += time.monotonic() - stage_started
                        processed += 1
                        progress.update(1)
                        elapsed = max(time.monotonic() - started, 1e-6)
                        progress.set_postfix(fps=f"{processed / elapsed:.3f}", refresh=False)
        finally:
            progress.close()
        reader.close()
        reader = None
        stage_started = time.monotonic()
        writer.close()
        timings["encode_flush"] += time.monotonic() - stage_started
        writer = None
        clean_video_ready = True
    finally:
        if reader is not None:
            try:
                reader.close()
            except Exception:
                pass
        if writer is not None:
            try:
                writer.close()
            except Exception:
                pass
        if workers is not None:
            for name, value in workers.stage_timings.items():
                timings[name] = timings.get(name, 0.0) + value
            workers.close()

    if not clean_video_ready or processed == 0:
        raise RuntimeError("No complete video was encoded.")
    actual_duration = processed / inference_fps
    output_frames = int(round(actual_duration * output_fps))
    stage_started = time.monotonic()
    mux_audio(
        temporary_video,
        input_path,
        output_path,
        args.ffmpeg_bin,
        start,
        actual_duration,
        info.has_audio,
        args.audio_codec,
        args.audio_bitrate,
    )
    timings["audio_mux"] += time.monotonic() - stage_started
    if temporary_video.exists():
        temporary_video.unlink()
    elapsed = time.monotonic() - started
    print(
        f"[range] actual_start={format_seconds(start)}, actual_end={format_seconds(start + actual_duration)}, "
        f"processed_inference_frames={processed}, output_frames={output_frames}, "
        f"output_duration={actual_duration:.3f}s",
        flush=True,
    )
    print(
        f"[run] wall_end={now_text()}, elapsed={elapsed:.1f}s, average={processed / max(elapsed, 1e-6):.3f} frame/s",
        flush=True,
    )
    print(
        "[timing] " + ", ".join(f"{name}={value:.1f}s" for name, value in timings.items()),
        flush=True,
    )
    size_mib = output_path.stat().st_size / (1024**2)
    bitrate_mbps = output_path.stat().st_size * 8 / max(actual_duration, 1e-6) / 1_000_000
    print(f"[size] {size_mib:.2f} MiB, average_bitrate={bitrate_mbps:.2f} Mb/s", flush=True)
    print(f"[output] {output_path}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Persistent multi-GPU, float-first Real-ESRGAN video enhancement.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", required=True, help="Input video path")
    parser.add_argument("--output", required=True, help="Output MP4 path")
    parser.add_argument("--model", choices=tuple(MODEL_URLS), default="realesr-animevideov3")
    parser.add_argument("--model-path", default="", help="Optional local .pth override")
    parser.add_argument("--denoise-strength", type=float, default=1.0, help="DNI strength for general-x4v3")
    parser.add_argument("--scale", type=float, default=2.0, help="Final output scale")
    parser.add_argument(
        "--quality-preset",
        choices=("baseline", "safe"),
        default="safe",
    )
    parser.add_argument(
        "--fps",
        default="source",
        help="Output FPS: source/auto/0, a number such as 23 or 60, or 24000/1001",
    )
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--input-width", type=int, default=0, help="0 keeps source/aspect-derived width")
    parser.add_argument("--input-height", type=int, default=0, help="0 keeps source/aspect-derived height")
    parser.add_argument(
        "--tile-size",
        type=int,
        default=256,
        help="0 uses fastest full-frame inference; use tiles only as an OOM fallback",
    )
    parser.add_argument("--tile-pad", type=int, default=10, help="Model context around each direct-write tile")
    parser.add_argument("--pre-pad", type=int, default=0)
    parser.add_argument("--tile-verify-coverage", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--overlap", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--batch-size", type=int, default=4, help="Tiles per inference batch on each GPU")
    parser.add_argument("--gpu-ids", default="0,1", help="Comma-separated IDs, or cpu")
    parser.add_argument("--channels-last", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--native-analysis", choices=("off", "report", "auto"), default="off")
    parser.add_argument("--native-samples", type=int, default=5)
    parser.add_argument("--native-min-height", type=int, default=500)
    parser.add_argument("--native-max-height", type=int, default=1080)
    parser.add_argument("--native-kernels", default="bilinear,bicubic,lanczos")
    parser.add_argument("--native-confidence", type=float, default=0.85)
    parser.add_argument("--native-height", type=int, default=0)
    parser.add_argument(
        "--native-kernel", choices=("auto", "bilinear", "bicubic", "lanczos"), default="auto"
    )
    parser.add_argument("--descale", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--tta", choices=("none", "x8"), default="x8")
    parser.add_argument("--tta-batch-size", type=int, choices=(1, 2, 4, 8), default=1)
    parser.add_argument("--shift-ensemble", choices=("none", "x2", "x4"), default="none")
    parser.add_argument("--residual-mode", choices=("official", "global", "adaptive"), default="official")
    parser.add_argument("--residual-strength", type=float, default=1.0)
    parser.add_argument("--residual-flat-strength", type=float, default=0.9)
    parser.add_argument("--residual-edge-strength", type=float, default=1.0)
    parser.add_argument("--residual-edge-low", type=float, default=0.05)
    parser.add_argument("--residual-edge-high", type=float, default=0.20)
    parser.add_argument("--base-correction", type=float, default=0.0)

    parser.add_argument("--back-projection-iterations", type=int, default=1)
    parser.add_argument("--back-projection-strength", type=float, default=0.2)
    parser.add_argument(
        "--back-projection-kernel", choices=("area", "bicubic", "lanczos"), default="lanczos"
    )
    parser.add_argument("--back-projection-clamp", type=float, default=0.05)
    parser.add_argument("--dehalo-strength", type=float, default=0.0)
    parser.add_argument("--dehalo-radius", type=int, default=2)
    parser.add_argument("--range-limit", type=float, default=0.1)
    parser.add_argument("--range-radius", type=int, default=2)
    parser.add_argument("--overshoot", type=float, default=1.0)
    parser.add_argument("--undershoot", type=float, default=1.0)

    parser.add_argument("--anime4k", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--anime4k-shader-dir", default="")
    parser.add_argument("--anime4k-strength", type=float, default=1.0)
    parser.add_argument("--anime4k-shaders", default="")
    parser.add_argument(
        "--video-codec",
        choices=("libx264", "libx265", "h264_nvenc", "hevc_nvenc"),
        default="hevc_nvenc",
    )
    parser.add_argument("--crf", type=int, default=18, help="Lower is higher video quality/larger file")
    parser.add_argument(
        "--preset",
        choices=("ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower"),
        default="medium",
    )
    parser.add_argument("--cq", type=int, default=18, help="NVENC quality target; lower is higher quality")
    parser.add_argument("--nvenc-preset", choices=tuple(f"p{i}" for i in range(1, 8)), default="p7")
    parser.add_argument("--encode-gpu", type=int, default=0)
    parser.add_argument(
        "--output-pix-fmt",
        choices=("auto", "yuv420p", "yuv420p10le", "p010le"),
        default="auto",
    )
    parser.add_argument(
        "--color-policy",
        choices=("preserve", "bt709"),
        default="preserve",
        help="Preserve/infer source SDR metadata, or force BT.709 limited-range output",
    )
    parser.add_argument(
        "--hdr-policy",
        choices=("reject", "passthrough"),
        default="reject",
        help="Reject HDR by default because the SR model is not HDR-linear",
    )
    parser.add_argument("--audio-codec", choices=("aac", "copy"), default="aac")
    parser.add_argument("--audio-bitrate", default="192k")
    parser.add_argument("--start-time", type=float, default=0.0, help="Arbitrary source start in seconds")
    parser.add_argument("--test-seconds", type=float, default=0.0, help="0 processes to end; use 10 for a test")
    parser.add_argument("--progress-interval", type=float, default=60.0, help="Forced progress refresh interval")
    parser.add_argument("--ffmpeg-bin", default="ffmpeg")
    parser.add_argument("--ffprobe-bin", default="ffprobe")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    apply_quality_preset(args, sys.argv[1:])
    validate_args(args)
    process_video(args)


if __name__ == "__main__":
    mp.freeze_support()
    main()
