"""Automatic 8-bit/10-bit precision handling for video inference.

The retained inference runtime is intentionally left unchanged for its original
8-bit path. This module installs only the pieces needed to preserve 10-bit
sources through FFmpeg decode, Real-ESRGAN inference and tile fusion.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import re
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from . import runtime as base


_ORIGINAL_PROBE_VIDEO = base.probe_video
_SOURCE_BIT_DEPTH = 8


def _stream_bit_depth(stream: dict) -> Tuple[int, str]:
    pix_fmt = str(stream.get("pix_fmt") or "unknown").lower()
    raw_value = str(stream.get("bits_per_raw_sample") or "").strip()
    raw_bits = int(raw_value) if raw_value.isdigit() and int(raw_value) > 0 else 0

    # FFmpeg formats such as yuv420p10le/gbrp10le carry the bit depth after
    # the component layout. p010le is the common semi-planar 10-bit format.
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
        f"Source pixel format {pix_fmt!r} reports {bits}-bit samples. "
        "This pipeline currently preserves 8-bit and 10-bit sources only."
    )


def probe_video(path: Path, ffprobe_bin: str) -> base.VideoInfo:
    global _SOURCE_BIT_DEPTH

    info = _ORIGINAL_PROBE_VIDEO(path, ffprobe_bin)
    command = [
        ffprobe_bin,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=pix_fmt,bits_per_raw_sample",
        "-of",
        "json",
        str(path),
    ]
    data = json.loads(base.run_checked(command, "ffprobe bit-depth").stdout)
    streams = data.get("streams", [])
    if not streams:
        raise RuntimeError("ffprobe did not return the primary video stream for bit-depth detection.")

    _SOURCE_BIT_DEPTH, pix_fmt = _stream_bit_depth(streams[0])
    decode_format = "rgb48le" if _SOURCE_BIT_DEPTH == 10 else "rgb24"
    output_format = "10-bit" if _SOURCE_BIT_DEPTH == 10 else "8-bit"
    print(
        f"[bit-depth] source_pix_fmt={pix_fmt}, source={_SOURCE_BIT_DEPTH}-bit, "
        f"inference_rgb={decode_format}, output={output_format}",
        flush=True,
    )
    return info


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
        self.width = width
        self.height = height
        self.pixel_format = "rgb48le" if _SOURCE_BIT_DEPTH == 10 else "rgb24"
        self.dtype = np.dtype("<u2") if _SOURCE_BIT_DEPTH == 10 else np.dtype(np.uint8)
        self.frame_bytes = width * height * 3 * self.dtype.itemsize

        vf = f"scale={width}:{height}:flags=lanczos,fps={fps_rate}"
        command = [ffmpeg_bin, "-hide_banner", "-loglevel", "error"]
        if start > 0:
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
            self.pixel_format,
            "pipe:1",
        ]
        self.process = base.subprocess.Popen(command, stdout=base.subprocess.PIPE, stderr=base.subprocess.PIPE)

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


def _is_10bit_frame(frame: np.ndarray) -> bool:
    return frame.dtype.kind == "u" and frame.dtype.itemsize == 2


def infer_image_batch(
    model: torch.nn.Module,
    patches: Sequence[np.ndarray],
    device: torch.device,
    fp16: bool,
    native_scale: int,
    output_scale: float,
    channels_last: bool,
) -> List[np.ndarray]:
    # Keep the original 8-bit path byte-for-byte so existing inputs retain the
    # same performance and numerical behavior.
    if not _is_10bit_frame(patches[0]):
        return base.infer_image_batch(
            model,
            patches,
            device,
            fp16,
            native_scale,
            output_scale,
            channels_last,
        )

    input_height, input_width = patches[0].shape[:2]
    rgb = np.stack(patches).astype(np.float32, copy=False)
    tensor = torch.from_numpy(rgb).permute(0, 3, 1, 2).to(device, non_blocking=True)
    tensor.div_(65535.0)
    if fp16 and device.type == "cuda":
        tensor = tensor.half()
    if channels_last and device.type == "cuda":
        tensor = tensor.contiguous(memory_format=torch.channels_last)

    with torch.inference_mode():
        output = model(tensor)
        if output_scale != native_scale:
            output_height = max(1, int(round(input_height * output_scale)))
            output_width = max(1, int(round(input_width * output_scale)))
            output = F.interpolate(
                output,
                size=(output_height, output_width),
                mode="bicubic",
                align_corners=False,
            )
        # Convert to FP32 before multiplying by 65535 so FP16 never overflows
        # near full white. NumPy performs the final uint16 packing reliably on
        # PyTorch versions where eager uint16 support is still limited.
        output = output.clamp_(0, 1).float().mul_(65535.0).round_()

    array = output.permute(0, 2, 3, 1).contiguous().cpu().numpy().astype(np.uint16)
    return list(array)


def worker_main(
    worker_id: int,
    gpu_id: Optional[int],
    input_queue: mp.Queue,
    output_queue: mp.Queue,
    config_dict: Dict[str, object],
) -> None:
    try:
        config = base.WorkerConfig(**config_dict)  # type: ignore[arg-type]
        if gpu_id is None:
            device = torch.device("cpu")
        else:
            torch.cuda.set_device(gpu_id)
            device = torch.device(f"cuda:{gpu_id}")
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cuda.matmul.allow_tf32 = True

        model, native_scale = base.load_worker_model(config, device)
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


def blend_tiles(
    outputs: Dict[int, np.ndarray],
    infos: Sequence[base.TileInfo],
    input_width: int,
    input_height: int,
    scale: float,
    overlap: int,
) -> np.ndarray:
    if not outputs:
        raise RuntimeError("Tile fusion received no model outputs.")

    sample = next(iter(outputs.values()))
    output_dtype = sample.dtype
    if output_dtype.kind != "u" or output_dtype.itemsize not in {1, 2}:
        raise RuntimeError(f"Unsupported tile output dtype: {output_dtype}")
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
            tile = cv2.resize(tile, (width, height), interpolation=cv2.INTER_CUBIC)
        else:
            tile = tile[:height, :width]
        wx = base.feather_axis(width, fade, info.x0 > 0, info.x1 < input_width)
        wy = base.feather_axis(height, fade, info.y0 > 0, info.y1 < input_height)
        weight = (wy[:, None] * wx[None, :])[:, :, None]
        accumulator[oy0:oy1, ox0:ox1] += tile.astype(np.float32) * weight
        weight_sum[oy0:oy1, ox0:ox1] += weight

    if np.any(weight_sum <= 0):
        raise RuntimeError("Tile fusion produced uncovered output pixels; check tile/overlap settings.")
    np.divide(accumulator, weight_sum, out=accumulator)
    np.rint(accumulator, out=accumulator)
    np.clip(accumulator, 0, max_value, out=accumulator)
    return accumulator.astype(output_dtype)


def install(runtime=base) -> None:
    """Install automatic bit-depth preservation into the retained runtime."""
    if getattr(runtime, "_bitdepth_preservation_installed", False):
        return
    runtime.probe_video = probe_video
    runtime.RawVideoReader = RawVideoReader
    runtime.worker_main = worker_main
    runtime.blend_tiles = blend_tiles
    runtime._bitdepth_preservation_installed = True
