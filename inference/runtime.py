#!/usr/bin/env python3
"""Multi-GPU Real-ESRGAN video inference runtime.

The model always runs at its native scale. When the requested output scale is
smaller (for example AnimeVideo-v3 native x4 -> final x2), the runtime first
reconstructs the complete native-resolution frame and then performs one
full-frame Lanczos4 resize. No per-patch bicubic resize is used.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import queue
import re
import shutil
import subprocess
import sys
import time
import traceback
import types
import urllib.request
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from typing import Callable, Dict, List, Optional, Protocol, Sequence, Tuple, Type

import cv2
import numpy as np
import torch
from tqdm import tqdm

try:  # pragma: no cover - depends on installed torchvision
    import torchvision.transforms.functional_tensor  # noqa: F401
except (ImportError, ModuleNotFoundError):  # pragma: no cover
    import torchvision.transforms.functional as _tv_functional

    _functional_tensor = types.ModuleType("torchvision.transforms.functional_tensor")
    _functional_tensor.rgb_to_grayscale = _tv_functional.rgb_to_grayscale
    sys.modules["torchvision.transforms.functional_tensor"] = _functional_tensor

from basicsr.archs.rrdbnet_arch import RRDBNet
from .models.srvgg_arch import SRVGGNetCompact
from . import autotune


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


class VideoWriter(Protocol):
    def write(self, frame: np.ndarray) -> None: ...
    def close(self) -> None: ...


_RequireEncoder = Callable[[str, str], None]
_WriterType = Type[VideoWriter]
_require_encoder: Optional[_RequireEncoder] = None
_writer_type: Optional[_WriterType] = None


def set_encoding_backend(require_encoder_fn: _RequireEncoder, writer_type: _WriterType) -> None:
    global _require_encoder, _writer_type
    _require_encoder = require_encoder_fn
    _writer_type = writer_type


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    fps_num: int
    fps_den: int
    duration: float
    frames: Optional[int]
    has_audio: bool
    pix_fmt: str
    bit_depth: int

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
        raise FileNotFoundError(f"Required executable '{name}' was not found.")


def parse_rate(value: str) -> Fraction:
    if not value or value in {"0/0", "N/A"}:
        raise ValueError(f"Invalid video frame rate: {value!r}")
    rate = Fraction(value)
    if rate <= 0:
        raise ValueError(f"Invalid video frame rate: {value!r}")
    return rate


def parse_output_rate(value: str, source_rate: Fraction) -> Fraction:
    normalized = str(value).strip().lower()
    if normalized in {"", "0", "auto", "source", "original"}:
        return source_rate
    try:
        return parse_rate(normalized)
    except (ValueError, ZeroDivisionError) as error:
        raise ValueError(
            "--fps must be source/auto/0, a positive number, or a rational rate such as 24000/1001."
        ) from error


def _stream_bit_depth(stream: dict) -> Tuple[int, str]:
    pix_fmt = str(stream.get("pix_fmt") or "unknown").lower()
    raw_value = str(stream.get("bits_per_raw_sample") or "").strip()
    raw_bits = int(raw_value) if raw_value.isdigit() and int(raw_value) > 0 else 0
    if "p010" in pix_fmt:
        bits = 10
    else:
        match = re.search(r"(?:p|gray|rgb|bgr)(9|10|12|14|16)(?:le|be)$", pix_fmt)
        bits = int(match.group(1)) if match else raw_bits or 8
    if bits <= 8:
        return 8, pix_fmt
    if bits == 10:
        return 10, pix_fmt
    raise RuntimeError(
        f"Source pixel format {pix_fmt!r} reports {bits}-bit samples; only 8-bit and 10-bit are supported."
    )


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
    bit_depth, pix_fmt = _stream_bit_depth(stream)
    return VideoInfo(
        width=int(stream["width"]),
        height=int(stream["height"]),
        fps_num=rate.numerator,
        fps_den=rate.denominator,
        duration=float(duration_value),
        frames=frames,
        has_audio=any(item.get("codec_type") == "audio" for item in data.get("streams", [])),
        pix_fmt=pix_fmt,
        bit_depth=bit_depth,
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
            raise ValueError("A custom realesr-general-x4v3 weight can only use --denoise-strength 1.")
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


def _model_native_scale(name: str) -> int:
    return 2 if name == "RealESRGAN_x2plus" else 4


def torch_load_cpu(path: str) -> Dict[str, object]:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
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


def _is_uint16(frame: np.ndarray) -> bool:
    return frame.dtype.kind == "u" and frame.dtype.itemsize == 2


def _full_frame_lanczos(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize a completed native-scale frame once with Lanczos4."""
    if frame.shape[1] == width and frame.shape[0] == height:
        return frame
    if frame.dtype.kind != "u" or frame.dtype.itemsize not in {1, 2}:
        raise RuntimeError(f"Unsupported inference frame dtype for resize: {frame.dtype}")
    max_value = float(np.iinfo(frame.dtype).max)
    resized = cv2.resize(
        frame.astype(np.float32, copy=False),
        (width, height),
        interpolation=cv2.INTER_LANCZOS4,
    )
    np.rint(resized, out=resized)
    np.clip(resized, 0, max_value, out=resized)
    return resized.astype(frame.dtype)


def infer_image_batch(
    model: torch.nn.Module,
    patches: Sequence[np.ndarray],
    device: torch.device,
    fp16: bool,
    native_scale: int,
    output_scale: float,
    channels_last: bool,
) -> List[np.ndarray]:
    """Run the model at native scale; never resize individual patches."""
    del native_scale, output_scale
    is_10bit = _is_uint16(patches[0])

    if is_10bit:
        rgb = np.stack(patches).astype(np.float32, copy=False)
        tensor = torch.from_numpy(rgb).permute(0, 3, 1, 2).to(device, non_blocking=True)
        tensor.div_(65535.0)
        if fp16 and device.type == "cuda":
            tensor = tensor.half()
    else:
        rgb = np.stack(patches)
        tensor = torch.from_numpy(rgb).permute(0, 3, 1, 2).to(device, non_blocking=True)
        tensor = tensor.half() if fp16 and device.type == "cuda" else tensor.float()
        tensor.div_(255.0)

    if channels_last and device.type == "cuda":
        tensor = tensor.contiguous(memory_format=torch.channels_last)

    with torch.inference_mode():
        output = model(tensor)
        output.clamp_(0, 1)
        if is_10bit:
            output = output.float().mul_(65535.0).round_()
        else:
            output = output.mul_(255.0).round_().byte()

    array = output.permute(0, 2, 3, 1).contiguous().cpu().numpy()
    if is_10bit:
        array = array.astype(np.uint16)
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
            batch_size = config.batch_size if job_type == "tiles" else 1
            for offset in range(0, len(indexed_patches), batch_size):
                chunk = indexed_patches[offset : offset + batch_size]
                if not chunk:
                    continue
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
    except Exception as error:
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
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Timed out while loading models on workers.")
            try:
                message = self.output_queue.get(timeout=remaining)
            except queue.Empty as error:
                raise TimeoutError("Timed out while loading models on workers.") from error
            if message[0] == "error":
                raise RuntimeError(f"Worker {message[1]} failed during startup: {message[2]}\n{message[3]}")
            if message[0] == "ready":
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
                raise RuntimeError(f"Worker {message[1]} failed: {message[2]}\n{message[3]}")
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
        self,
        batch_id: int,
        indexed_frames: Sequence[Tuple[int, np.ndarray]],
    ) -> Dict[int, np.ndarray]:
        if len(indexed_frames) > len(self.processes):
            raise ValueError("Full-frame batches may contain at most one frame per worker.")
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
        bit_depth: int,
    ) -> None:
        self.width = width
        self.height = height
        self.pixel_format = "rgb48le" if bit_depth == 10 else "rgb24"
        self.dtype = np.dtype("<u2") if bit_depth == 10 else np.dtype(np.uint8)
        self.frame_bytes = width * height * 3 * self.dtype.itemsize
        vf = f"scale={width}:{height}:flags=lanczos,fps={fps_rate}"
        command = [ffmpeg_bin, "-hide_banner", "-loglevel", "error"]
        if start > 0:
            command += ["-ss", f"{start:.6f}"]
        command += ["-i", str(input_path)]
        command += [
            "-t", f"{duration:.6f}",
            "-vf", vf,
            "-an",
            "-f", "rawvideo",
            "-pix_fmt", self.pixel_format,
            "pipe:1",
        ]
        self.process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def read(self) -> Optional[np.ndarray]:
        assert self.process.stdout is not None
        data = self.process.stdout.read(self.frame_bytes)
        if not data:
            return None
        if len(data) != self.frame_bytes:
            raise RuntimeError(f"ffmpeg returned a partial raw frame ({len(data)}/{self.frame_bytes} bytes).")
        return np.frombuffer(data, dtype=self.dtype).reshape(self.height, self.width, 3)

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


def axis_starts(length: int, tile_size: int, overlap: int) -> List[int]:
    if length <= tile_size:
        return [0]
    return list(range(0, length, tile_size - overlap))


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
    """Blend model-native tile outputs before any final resize."""
    if not outputs:
        raise RuntimeError("Tile fusion received no model outputs.")
    sample = next(iter(outputs.values()))
    output_dtype = sample.dtype
    max_value = float(np.iinfo(output_dtype).max)
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
            tile = cv2.resize(tile, (width, height), interpolation=cv2.INTER_LANCZOS4)
        else:
            tile = tile[:height, :width]
        wx = feather_axis(width, fade, info.x0 > 0, info.x1 < input_width)
        wy = feather_axis(height, fade, info.y0 > 0, info.y1 < input_height)
        weight = (wy[:, None] * wx[None, :])[:, :, None]
        accumulator[oy0:oy1, ox0:ox1] += tile.astype(np.float32) * weight
        weight_sum[oy0:oy1, ox0:ox1] += weight

    if np.any(weight_sum <= 0):
        raise RuntimeError("Tile fusion produced uncovered output pixels.")
    np.divide(accumulator, weight_sum, out=accumulator)
    np.rint(accumulator, out=accumulator)
    np.clip(accumulator, 0, max_value, out=accumulator)
    return accumulator.astype(output_dtype)


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
        return
    base = [ffmpeg_bin, "-y", "-hide_banner", "-loglevel", "error", "-i", str(silent_video)]
    if audio_codec == "aac":
        command = base + [
            "-ss", f"{start:.6f}",
            "-t", f"{duration:.6f}",
            "-i", str(input_path),
            "-filter_complex", f"[1:a:0]atrim=start=0:duration={duration:.6f},asetpts=PTS-STARTPTS[a]",
            "-map", "0:v:0",
            "-map", "[a]",
            "-map_metadata", "1",
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", audio_bitrate,
            "-shortest",
            "-movflags", "+faststart",
            str(output_path),
        ]
    else:
        command = base + [
            "-ss", f"{start:.6f}",
            "-t", f"{duration:.6f}",
            "-i", str(input_path),
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-map_metadata", "1",
            "-c", "copy",
            "-shortest",
            "-avoid_negative_ts", "make_zero",
            "-movflags", "+faststart",
            str(output_path),
        ]
    try:
        run_checked(command, "audio mux")
    except RuntimeError:
        if audio_codec != "copy":
            raise
        print("[warning] audio stream copy failed; retrying with AAC", flush=True)
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
        raise ValueError("--gpu-ids must be 'cpu' or comma-separated IDs such as 0,1.") from error
    if not ids or len(ids) != len(set(ids)) or min(ids) < 0:
        raise ValueError("--gpu-ids must contain unique, non-negative GPU numbers.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Use --gpu-ids cpu for compatibility.")
    count = torch.cuda.device_count()
    if max(ids) >= count:
        raise ValueError(f"Requested GPU {max(ids)}, but only {count} CUDA device(s) are visible.")
    return ids


def validate_args(args: argparse.Namespace) -> None:
    if args.scale <= 0:
        raise ValueError("--scale must be positive.")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")
    if args.overlap < 0:
        raise ValueError("--overlap must be non-negative.")
    if not 0 <= args.denoise_strength <= 1:
        raise ValueError("--denoise-strength must be between 0 and 1.")
    if not args.full_frame:
        if args.tile_size < 64 or args.tile_size % 4:
            raise ValueError("--tile-size must be at least 64 and divisible by 4.")
        if args.overlap >= args.tile_size // 2:
            raise ValueError("--overlap must be less than half of --tile-size.")
        if args.max_tile_size < 512:
            raise ValueError("--max-tile-size must be at least 512.")
        if args.max_batch_size < 1:
            raise ValueError("--max-batch-size must be at least 1.")


def _output_pixel_format(codec: str, bit_depth: int) -> str:
    if bit_depth == 8:
        return "yuv420p"
    return "p010le" if codec.endswith("_nvenc") else "yuv420p10le"


def _device_text(gpu_ids: Sequence[Optional[int]], fp16: bool) -> str:
    if gpu_ids == [None]:
        return "CPU | FP32"
    parts = []
    for gpu_id in gpu_ids:
        assert gpu_id is not None
        props = torch.cuda.get_device_properties(gpu_id)
        parts.append(f"cuda:{gpu_id} {props.name} {props.total_memory / 2**30:.1f}GiB")
    return "; ".join(parts) + (" | FP16" if fp16 else " | FP32")


def process_video(args: argparse.Namespace) -> None:
    if _require_encoder is None or _writer_type is None:
        raise RuntimeError("Encoding backend is not configured. Run through root inference.py.")

    require_binary(args.ffmpeg_bin)
    require_binary(args.ffprobe_bin)
    _require_encoder(args.ffmpeg_bin, args.video_codec)

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
        raise ValueError(f"4:2:0 output needs even dimensions, got {output_width}x{output_height}.")

    start, duration, expected_frames = resolve_range(
        info, args.start_time, args.test_seconds, inference_rate
    )
    expected_output_frames = max(1, int(round(duration * output_fps)))
    end = start + duration
    gpu_ids = parse_gpu_ids(args.gpu_ids)
    effective_fp16 = args.fp16 and gpu_ids != [None]
    effective_channels_last = args.channels_last and gpu_ids != [None]
    model_paths = resolve_model_paths(args)
    native_scale = _model_native_scale(args.model)

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
    if args.full_frame:
        args.tile_size = 0
        args.batch_size = 1
    elif cuda_gpu_ids:
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
            max_tile_size=args.max_tile_size,
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

    mode = "test" if args.test_seconds > 0 else "full/selected range"
    print("=== Real-ESRGAN ===", flush=True)
    print(
        f"Input   : {input_path.name} | {info.width}x{info.height} | "
        f"{info.fps:.3f} fps | {info.bit_depth}-bit ({info.pix_fmt})",
        flush=True,
    )
    print(
        f"Output  : {output_width}x{output_height} | {output_fps:.3f} fps | "
        f"{info.bit_depth}-bit ({_output_pixel_format(args.video_codec, info.bit_depth)}) | {args.video_codec}",
        flush=True,
    )
    print(
        f"Range   : {mode} | {format_seconds(start)} -> {format_seconds(end)} | "
        f"{duration:.3f}s | {expected_frames} inference / {expected_output_frames} output frames",
        flush=True,
    )
    if float(args.scale) == float(native_scale):
        resample_text = f"native={native_scale}x"
    else:
        resample_text = (
            f"native={native_scale}x -> final={args.scale:g}x | resample=full-frame Lanczos4"
        )
    print(
        f"Model   : {args.model} | {resample_text} | channels_last={effective_channels_last}",
        flush=True,
    )
    print(f"GPU     : {_device_text(gpu_ids, effective_fp16)}", flush=True)
    if args.full_frame:
        print(
            f"Mode    : FULL_FRAME=True | full-frame | parallel_frames={len(gpu_ids)}",
            flush=True,
        )
    elif tune_result is not None:
        print(
            f"Mode    : FULL_FRAME=False | auto tile={tune_result.tile_size} | "
            f"batch={tune_result.batch_size} | estimate={tune_result.estimated_seconds:.3f}s/frame | "
            f"tested={tune_result.tested}, OOM={tune_result.rejected_oom} | "
            f"search={tune_result.search_seconds:.1f}s",
            flush=True,
        )
    else:
        print(
            f"Mode    : FULL_FRAME=False | CPU fallback tile={args.tile_size} | batch={args.batch_size}",
            flush=True,
        )
    print(flush=True)

    reader: Optional[RawVideoReader] = None
    writer: Optional[VideoWriter] = None
    workers: Optional[PersistentWorkers] = None
    processed = 0
    started = time.monotonic()
    timings = {
        "model_startup": 0.0,
        "decode": 0.0,
        "inference": 0.0,
        "blend": 0.0,
        "resize": 0.0,
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
            info.bit_depth,
        )
        writer = _writer_type(
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
            file=sys.stdout,
        )
        try:
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
                    resized_outputs: Dict[int, np.ndarray] = {}
                    for frame_id, _frame in indexed_frames:
                        resized_outputs[frame_id] = _full_frame_lanczos(
                            frame_outputs[frame_id], output_width, output_height
                        )
                    timings["resize"] += time.monotonic() - stage_started

                    stage_started = time.monotonic()
                    for frame_id, _frame in indexed_frames:
                        writer.write(resized_outputs[frame_id])
                    timings["write"] += time.monotonic() - stage_started

                    processed += len(indexed_frames)
                    batch_id += 1
                    progress.update(len(indexed_frames))
                    elapsed_now = max(time.monotonic() - started, 1e-6)
                    progress.set_postfix(fps=f"{processed / elapsed_now:.3f}", refresh=False)
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
                    native_output = blend_tiles(
                        tile_outputs,
                        tile_infos,
                        input_width,
                        input_height,
                        native_scale,
                        args.overlap,
                    )
                    timings["blend"] += time.monotonic() - stage_started

                    stage_started = time.monotonic()
                    output = _full_frame_lanczos(native_output, output_width, output_height)
                    timings["resize"] += time.monotonic() - stage_started

                    stage_started = time.monotonic()
                    writer.write(output)
                    timings["write"] += time.monotonic() - stage_started

                    processed += 1
                    progress.update(1)
                    elapsed_now = max(time.monotonic() - started, 1e-6)
                    progress.set_postfix(fps=f"{processed / elapsed_now:.3f}", refresh=False)
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
    size_mib = output_path.stat().st_size / (1024**2)
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
        f"resize={timings['resize']:.1f}s | encode={encode_time:.1f}s | "
        f"audio={timings['audio_mux']:.1f}s",
        flush=True,
    )
    print(f"File    : {size_mib:.2f} MiB | {bitrate_mbps:.2f} Mb/s", flush=True)
    print(f"Output  : {output_path}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Persistent multi-GPU Real-ESRGAN video inference.",
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
        help="Output FPS: source/auto/0, number, or rational rate such as 24000/1001",
    )
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--input-width", type=int, default=0, help="0 keeps source/aspect-derived width")
    parser.add_argument("--input-height", type=int, default=0, help="0 keeps source/aspect-derived height")
    parser.add_argument(
        "--full-frame",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="True uses quality-first full-frame inference; False enables automatic tile+batch tuning",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=512,
        help="Fallback/additional tile candidate used when --no-full-frame",
    )
    parser.add_argument("--max-tile-size", type=int, default=1536)
    parser.add_argument("--overlap", type=int, default=32)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Fallback/additional batch candidate used when --no-full-frame",
    )
    parser.add_argument("--max-batch-size", type=int, default=32)
    parser.add_argument("--gpu-ids", default="0,1", help="Comma-separated IDs, or cpu")
    parser.add_argument("--channels-last", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--audio-codec", choices=("aac", "copy"), default="aac")
    parser.add_argument("--audio-bitrate", default="192k")
    parser.add_argument("--start-time", type=float, default=0.0, help="Source start in seconds")
    parser.add_argument("--test-seconds", type=float, default=0.0, help="0 processes to end; use 10 for a test")
    parser.add_argument("--ffmpeg-bin", default="ffmpeg")
    parser.add_argument("--ffprobe-bin", default="ffprobe")
    return parser
