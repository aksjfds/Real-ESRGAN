#!/usr/bin/env python3
"""Multi-GPU tiled Real-ESRGAN video inference for Kaggle.

The parent process owns video decoding, overlap blending, progress reporting and
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
from dataclasses import asdict, dataclass
from datetime import datetime
from fractions import Fraction
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
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
from realesrgan.archs.srvgg_arch import SRVGGNetCompact


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

    @property
    def fps(self) -> float:
        return self.fps_num / self.fps_den

@dataclass(frozen=True)
class TileInfo:
    index: int
    x0: int
    y0: int
    x1: int
    y1: int


@dataclass(frozen=True)
class WorkerConfig:
    model_name: str
    model_paths: Tuple[str, ...]
    denoise_strength: float
    scale: float
    tile_size: int
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
        return SRVGGNetCompact(
            num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type="prelu"
        ), 4
    if name == "realesr-general-x4v3":
        return SRVGGNetCompact(
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
    model.load_state_dict(state, strict=True)
    model.eval().requires_grad_(False)
    if config.fp16 and device.type == "cuda":
        model.half()
    if config.channels_last and device.type == "cuda":
        model.to(device=device, memory_format=torch.channels_last)
    else:
        model.to(device)
    return model, native_scale


def infer_image_batch(
    model: torch.nn.Module,
    patches: Sequence[np.ndarray],
    device: torch.device,
    fp16: bool,
    native_scale: int,
    output_scale: float,
    channels_last: bool,
) -> List[np.ndarray]:
    # The raw pipeline is RGB end-to-end, matching the model's training order.
    input_height, input_width = patches[0].shape[:2]
    rgb = np.stack(patches)
    tensor = torch.from_numpy(rgb).permute(0, 3, 1, 2).to(device, non_blocking=True)
    tensor = tensor.half() if fp16 and device.type == "cuda" else tensor.float()
    if channels_last and device.type == "cuda":
        tensor = tensor.contiguous(memory_format=torch.channels_last)
    tensor.div_(255.0)
    with torch.inference_mode():
        output = model(tensor)
        if output_scale != native_scale:
            output_height = max(1, int(round(input_height * output_scale)))
            output_width = max(1, int(round(input_width * output_scale)))
            output = F.interpolate(
                output, size=(output_height, output_width), mode="bicubic", align_corners=False
            )
        output = output.clamp_(0, 1).mul_(255).round_().byte()
    array = output.permute(0, 2, 3, 1).contiguous().cpu().numpy()
    return list(array)


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
        output_queue.put(("ready", worker_id, str(device)))
        while True:
            job = input_queue.get()
            if job is None:
                break
            job_type, job_id, indexed_patches = job
            results = []
            # Full-frame jobs contain at most one frame per GPU.  Tile jobs use
            # the configured batch size to improve Tensor Core occupancy.
            batch_size = config.batch_size if job_type == "tiles" else 1
            for offset in range(0, len(indexed_patches), batch_size):
                chunk = indexed_patches[offset : offset + batch_size]
                indices = [item[0] for item in chunk]
                patches = [item[1] for item in chunk]
                outputs = infer_image_batch(
                    model,
                    patches,
                    device,
                    config.fp16,
                    native_scale,
                    config.scale,
                    config.channels_last,
                )
                results.extend(zip(indices, outputs))
            output_queue.put((f"{job_type}_result", worker_id, job_id, results))
    except Exception as error:  # send failures to the parent instead of hanging it
        output_queue.put(("error", worker_id, repr(error), traceback.format_exc()))


class PersistentWorkers:
    def __init__(self, gpu_ids: Sequence[Optional[int]], config: WorkerConfig):
        self.context = mp.get_context("spawn")
        self.output_queue = self.context.Queue()
        self.input_queues = [self.context.Queue(maxsize=1) for _ in gpu_ids]
        self.processes = []
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
                        "\nFull-frame OOM fallback: use --tile-size 576 --overlap 32 --batch-size 2; "
                        "then try 256/32 with batch 16, 8, or 4. Keep FP16 enabled."
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
    ) -> None:
        command = [
            ffmpeg_bin,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s:v",
            f"{width}x{height}",
            "-r",
            input_fps_rate,
            "-i",
            "pipe:0",
            "-an",
        ]
        if output_fps_rate != input_fps_rate:
            # Raising FPS duplicates already-enhanced frames here instead of
            # running identical source frames through the model repeatedly.
            command += ["-vf", f"fps={output_fps_rate}"]
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
        command += ["-pix_fmt", "yuv420p"]
        if codec in {"libx265", "hevc_nvenc"}:
            command += ["-tag:v", "hvc1"]
        command.append(str(path))
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    def write(self, frame: np.ndarray) -> None:
        assert self.process.stdin is not None
        try:
            self.process.stdin.write(memoryview(np.ascontiguousarray(frame)).cast("B"))
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


def axis_starts(length: int, tile_size: int, overlap: int) -> List[int]:
    if length <= tile_size:
        return [0]
    stride = tile_size - overlap
    # Let the final tile be smaller and reflect-pad it to tile_size.  Moving a
    # full-size final tile back to the frame edge can create an almost complete
    # duplicate tile when a dimension is only slightly larger than tile_size.
    return list(range(0, length, stride))


def split_tiles(frame: np.ndarray, tile_size: int, overlap: int) -> Tuple[List[np.ndarray], List[TileInfo]]:
    height, width = frame.shape[:2]
    patches: List[np.ndarray] = []
    infos: List[TileInfo] = []
    index = 0
    for y0 in axis_starts(height, tile_size, overlap):
        for x0 in axis_starts(width, tile_size, overlap):
            y1 = min(y0 + tile_size, height)
            x1 = min(x0 + tile_size, width)
            patch = frame[y0:y1, x0:x1]
            pad_bottom = tile_size - patch.shape[0]
            pad_right = tile_size - patch.shape[1]
            if pad_bottom or pad_right:
                border = cv2.BORDER_REFLECT_101 if min(patch.shape[:2]) > 1 else cv2.BORDER_REPLICATE
                patch = cv2.copyMakeBorder(patch, 0, pad_bottom, 0, pad_right, border)
            patches.append(np.ascontiguousarray(patch))
            infos.append(TileInfo(index, x0, y0, x1, y1))
            index += 1
    return patches, infos


def feather_axis(length: int, fade: int, fade_start: bool, fade_end: bool) -> np.ndarray:
    weights = np.ones(length, dtype=np.float32)
    fade = min(fade, length // 2)
    if fade > 0:
        ramp = np.linspace(0.0, 1.0, fade, endpoint=False, dtype=np.float32)
        if fade_start:
            weights[:fade] = ramp
        if fade_end:
            weights[-fade:] = ramp[::-1]
    return weights


def blend_tiles(
    outputs: Dict[int, np.ndarray],
    infos: Sequence[TileInfo],
    input_width: int,
    input_height: int,
    scale: float,
    overlap: int,
) -> np.ndarray:
    output_width = int(round(input_width * scale))
    output_height = int(round(input_height * scale))
    accumulator = np.zeros((output_height, output_width, 3), dtype=np.float32)
    weight_sum = np.zeros((output_height, output_width, 1), dtype=np.float32)
    fade = max(1, int(round(overlap * scale))) if overlap else 0
    for info in infos:
        ox0 = int(round(info.x0 * scale))
        oy0 = int(round(info.y0 * scale))
        ox1 = int(round(info.x1 * scale))
        oy1 = int(round(info.y1 * scale))
        height = oy1 - oy0
        width = ox1 - ox0
        tile = outputs[info.index]
        if tile.shape[0] < height or tile.shape[1] < width:
            tile = cv2.resize(tile, (width, height), interpolation=cv2.INTER_CUBIC)
        else:
            tile = tile[:height, :width]
        wx = feather_axis(width, fade, info.x0 > 0, info.x1 < input_width)
        wy = feather_axis(height, fade, info.y0 > 0, info.y1 < input_height)
        weight = (wy[:, None] * wx[None, :])[:, :, None]
        accumulator[oy0:oy1, ox0:ox1] += tile.astype(np.float32) * weight
        weight_sum[oy0:oy1, ox0:ox1] += weight
    if np.any(weight_sum <= 0):
        raise RuntimeError("Tile fusion produced uncovered output pixels; check tile/overlap settings.")
    # Keep the large full-frame operations in-place to avoid several extra
    # 95 MiB temporaries at 4K (and roughly 380 MiB each at 8K).
    np.divide(accumulator, weight_sum, out=accumulator)
    np.rint(accumulator, out=accumulator)
    np.clip(accumulator, 0, 255, out=accumulator)
    return accumulator.astype(np.uint8)


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


def validate_args(args: argparse.Namespace) -> None:
    if args.scale <= 0:
        raise ValueError("--scale must be positive.")
    if args.tile_size != 0 and (args.tile_size < 64 or args.tile_size % 4):
        raise ValueError("--tile-size must be 0 (full frame), or at least 64 and divisible by 4.")
    if args.tile_size == 0:
        if args.overlap != 0:
            raise ValueError("--overlap must be 0 when --tile-size is 0 (full-frame mode).")
    elif args.overlap < 0 or args.overlap >= args.tile_size // 2:
        raise ValueError("--overlap must be non-negative and less than half of --tile-size.")
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


def process_video(args: argparse.Namespace) -> None:
    require_binary(args.ffmpeg_bin)
    require_binary(args.ffprobe_bin)
    require_encoder(args.ffmpeg_bin, args.video_codec)
    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input video not found: {input_path}")
    if input_path == output_path:
        raise ValueError("Input and output paths must be different.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_video = output_path.with_name(output_path.stem + ".video_only.tmp.mp4")

    info = probe_video(input_path, args.ffprobe_bin)
    source_rate = Fraction(info.fps_num, info.fps_den)
    output_rate = parse_output_rate(args.fps, source_rate)
    inference_rate = min(source_rate, output_rate)
    output_fps = float(output_rate)
    output_fps_rate = f"{output_rate.numerator}/{output_rate.denominator}"
    inference_fps = float(inference_rate)
    inference_fps_rate = f"{inference_rate.numerator}/{inference_rate.denominator}"
    input_width, input_height = choose_input_size(info, args.input_width, args.input_height)
    output_width = int(round(input_width * args.scale))
    output_height = int(round(input_height * args.scale))
    if output_width % 2 or output_height % 2:
        raise ValueError(
            f"yuv420p needs even output dimensions, got {output_width}x{output_height}. "
            "Adjust --input-width/--input-height or use an integer scale producing even dimensions."
        )
    start, duration, expected_frames = resolve_range(info, args.start_time, args.test_seconds, inference_rate)
    expected_output_frames = max(1, int(round(duration * output_fps)))
    end = start + duration
    gpu_ids = parse_gpu_ids(args.gpu_ids)
    effective_fp16 = args.fp16 and gpu_ids != [None]
    effective_channels_last = args.channels_last and gpu_ids != [None]
    model_paths = resolve_model_paths(args)
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
    if args.tile_size == 0:
        print(
            f"[inference] full-frame mode, parallel_frames={len(gpu_ids)}, "
            f"channels_last={effective_channels_last}",
            flush=True,
        )
    else:
        stride = args.tile_size - args.overlap
        tile_count = len(axis_starts(input_width, args.tile_size, args.overlap)) * len(
            axis_starts(input_height, args.tile_size, args.overlap)
        )
        print(
            f"[tiles] size={args.tile_size}, overlap={args.overlap}, stride={stride}, "
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
        "decode": 0.0,
        "inference": 0.0,
        "blend": 0.0,
        "write": 0.0,
        "encode_flush": 0.0,
        "audio_mux": 0.0,
    }
    clean_video_ready = False
    try:
        stage_started = time.monotonic()
        workers = PersistentWorkers(gpu_ids, config)
        timings["model_startup"] += time.monotonic() - stage_started
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
                        timings["decode"] += time.monotonic() - stage_started
                        if not indexed_frames:
                            break
                        stage_started = time.monotonic()
                        frame_outputs = workers.infer_frames(batch_id, indexed_frames)
                        timings["inference"] += time.monotonic() - stage_started
                        stage_started = time.monotonic()
                        for frame_id, _frame in indexed_frames:
                            writer.write(frame_outputs[frame_id])
                        timings["write"] += time.monotonic() - stage_started
                        processed += len(indexed_frames)
                        batch_id += 1
                        progress.update(len(indexed_frames))
                        elapsed = max(time.monotonic() - started, 1e-6)
                        progress.set_postfix(fps=f"{processed / elapsed:.3f}", refresh=False)
                else:
                    while True:
                        stage_started = time.monotonic()
                        frame = reader.read()
                        timings["decode"] += time.monotonic() - stage_started
                        if frame is None:
                            break
                        patches, tile_infos = split_tiles(frame, args.tile_size, args.overlap)
                        stage_started = time.monotonic()
                        tile_outputs = workers.infer_tiles(processed, patches)
                        timings["inference"] += time.monotonic() - stage_started
                        stage_started = time.monotonic()
                        output = blend_tiles(
                            tile_outputs,
                            tile_infos,
                            input_width,
                            input_height,
                            args.scale,
                            args.overlap,
                        )
                        timings["blend"] += time.monotonic() - stage_started
                        stage_started = time.monotonic()
                        writer.write(output)
                        timings["write"] += time.monotonic() - stage_started
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
        description="Persistent multi-GPU, overlap-blended Real-ESRGAN video inference.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", required=True, help="Input video path")
    parser.add_argument("--output", required=True, help="Output MP4 path")
    parser.add_argument("--model", choices=tuple(MODEL_URLS), default="realesr-animevideov3")
    parser.add_argument("--model-path", default="", help="Optional local .pth override")
    parser.add_argument("--denoise-strength", type=float, default=1.0, help="DNI strength for general-x4v3")
    parser.add_argument("--scale", type=float, default=2.0, help="Final output scale")
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
    parser.add_argument("--overlap", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4, help="Tiles per inference batch on each GPU")
    parser.add_argument("--gpu-ids", default="0,1", help="Comma-separated IDs, or cpu")
    parser.add_argument("--channels-last", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--video-codec",
        choices=("libx264", "libx265", "h264_nvenc", "hevc_nvenc"),
        default="libx264",
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
    validate_args(args)
    process_video(args)


if __name__ == "__main__":
    mp.freeze_support()
    main()
