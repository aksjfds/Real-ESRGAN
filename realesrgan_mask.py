#!/usr/bin/env python3
"""Run the masked Real-ESRGAN/ArtCNN video upscaling pipeline."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import shutil
import subprocess
import threading
import time
import urllib.request
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from fractions import Fraction
from pathlib import Path
from typing import Iterable, Optional, Sequence


REALESRGAN_ARCHIVE_URL = (
    "https://github.com/AmusementClub/vs-mlrt/releases/download/"
    "model-20211209/RealESRGANv3_v1.7z"
)
REALESRGAN_ARCHIVE_SHA256 = (
    "ee25320125aac83b5662c770a76d961126fc5e2dc759c461ae67afa33b138ba9"
)
ARTCNN_ARCHIVE_URL = (
    "https://github.com/AmusementClub/vs-mlrt/releases/download/"
    "external-models/artcnn_v8.7z"
)
ARTCNN_ARCHIVE_SHA256 = (
    "011623661f7273fe77c71bf419868da9700633eeea7bddcdac61a680951c6ab0"
)
VSMLRT_SCRIPT_URL = (
    "https://raw.githubusercontent.com/AmusementClub/vs-mlrt/"
    "1a14847b0652271d266efbfb691de15fb04bf988/scripts/vsmlrt.py"
)
VSMLRT_SCRIPT_SHA256 = "440703b8dc6ce265b3edcc4f1ac67cfc61d7ca128619e1c56eae3fd80af44dc1"


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    fps: Fraction
    duration: float
    frames: Optional[int]
    has_audio: bool
    matrix: str
    transfer: str
    primaries: str
    color_range: str
    vfr_suspected: bool


def now_text() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def format_seconds(value: float) -> str:
    hours = int(value // 3600)
    minutes = int((value % 3600) // 60)
    seconds = value % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"


def run_checked(command: Sequence[str], label: str) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"{label} failed (exit {result.returncode}):\n{detail}")
    return result


def require_binary(name: str) -> str:
    resolved = shutil.which(name)
    if not resolved:
        raise FileNotFoundError(f"Required executable not found: {name}")
    return resolved


def parse_rate(value: str) -> Fraction:
    rate = Fraction(value)
    if rate <= 0:
        raise ValueError(f"Invalid frame rate: {value!r}")
    return rate


def parse_output_rate(value: str, source: Fraction) -> Fraction:
    normalized = value.strip().lower()
    if normalized in {"", "0", "auto", "source", "original"}:
        return source
    try:
        return parse_rate(normalized)
    except (ValueError, ZeroDivisionError) as error:
        raise ValueError(
            "--fps must be source/auto/0, a positive number, or a rational such as 24000/1001"
        ) from error


def normalize_matrix(value: str, width: int, height: int) -> str:
    mapping = {
        "bt709": "709",
        "smpte170m": "170m",
        "bt470bg": "470bg",
        "bt2020nc": "2020ncl",
    }
    if value in mapping:
        return mapping[value]
    return "709" if width >= 1280 or height > 576 else "170m"


def normalize_transfer(value: str, matrix: str) -> str:
    mapping = {
        "bt709": "709",
        "smpte170m": "170m",
        "gamma28": "470bg",
        "bt2020-10": "2020_10",
    }
    return mapping.get(value, "709" if matrix == "709" else matrix)


def normalize_primaries(value: str, matrix: str) -> str:
    mapping = {
        "bt709": "709",
        "smpte170m": "170m",
        "bt470bg": "470bg",
        "bt2020": "2020",
    }
    return mapping.get(value, "709" if matrix == "709" else matrix)


def probe_video(path: Path, ffprobe: str) -> VideoInfo:
    command = [
        ffprobe,
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_streams",
        "-show_format",
        str(path),
    ]
    data = json.loads(run_checked(command, "ffprobe").stdout)
    videos = [stream for stream in data.get("streams", []) if stream.get("codec_type") == "video"]
    if not videos:
        raise ValueError(f"No video stream found in {path}")
    stream = videos[0]
    rate_text = stream.get("avg_frame_rate") or stream.get("r_frame_rate")
    if not rate_text or rate_text in {"0/0", "N/A"}:
        raise ValueError("Input video has no usable frame rate")
    fps = parse_rate(rate_text)
    nominal_text = stream.get("r_frame_rate")
    nominal_fps = (
        parse_rate(nominal_text)
        if nominal_text and nominal_text not in {"0/0", "N/A"}
        else fps
    )
    vfr_suspected = abs(float(nominal_fps - fps)) / float(fps) > 0.001
    duration_value = stream.get("duration") or data.get("format", {}).get("duration")
    if duration_value in {None, "N/A"}:
        raise ValueError("Input video has no usable duration")
    frame_value = stream.get("nb_frames")
    parsed_frames = (
        int(frame_value)
        if frame_value not in {None, "N/A"} and int(frame_value) > 0
        else None
    )
    width, height = int(stream["width"]), int(stream["height"])
    matrix = normalize_matrix(stream.get("color_space", ""), width, height)
    return VideoInfo(
        width=width,
        height=height,
        fps=fps,
        duration=float(duration_value),
        frames=parsed_frames,
        has_audio=any(item.get("codec_type") == "audio" for item in data.get("streams", [])),
        matrix=matrix,
        transfer=normalize_transfer(stream.get("color_transfer", ""), matrix),
        primaries=normalize_primaries(stream.get("color_primaries", ""), matrix),
        color_range="full" if stream.get("color_range") == "pc" else "limited",
        vfr_suspected=vfr_suspected,
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, target: Path, sha256: str | None = None) -> Path:
    if target.is_file() and target.stat().st_size > 0:
        if sha256 is None or file_sha256(target) == sha256:
            print(f"[download] cached: {target}", flush=True)
            return target
        print(f"[download] replacing cache with unexpected checksum: {target}", flush=True)
        target.unlink()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".part")
    print(f"[download] {url}", flush=True)
    started = time.monotonic()
    last_report = started

    def reporthook(blocks: int, block_size: int, total_size: int) -> None:
        nonlocal last_report
        current = time.monotonic()
        if current - last_report < 60:
            return
        received = blocks * block_size
        percent = 100 * received / total_size if total_size > 0 else 0.0
        print(
            f"[download-progress] time={now_text()}, file={target.name}, "
            f"{percent:.1f}%, elapsed={format_seconds(current - started)}",
            flush=True,
        )
        last_report = current

    try:
        urllib.request.urlretrieve(url, temporary, reporthook=reporthook)
        if sha256 is not None:
            actual = file_sha256(temporary)
            if actual != sha256:
                raise RuntimeError(
                    f"Checksum mismatch for {url}: expected {sha256}, got {actual}"
                )
        temporary.replace(target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return target


def extract_archive(archive: Path, destination: Path) -> None:
    marker = destination / f".{archive.name}.extracted"
    if marker.exists():
        return
    try:
        import py7zr
    except ImportError as error:
        raise RuntimeError("py7zr is required to extract model archives") from error
    destination.mkdir(parents=True, exist_ok=True)
    print(f"[model] extracting {archive.name}", flush=True)
    with py7zr.SevenZipFile(archive, mode="r") as package:
        package.extractall(destination)
    marker.touch()


def find_model(root: Path, filename: str) -> Path:
    matches = list(root.rglob(filename))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected exactly one {filename} below {root}, found {len(matches)}")
    return matches[0].resolve()


def prepare_assets(work_dir: Path) -> dict[str, Path]:
    download_dir = work_dir / "downloads"
    model_dir = work_dir / "models"
    real_archive = download(
        REALESRGAN_ARCHIVE_URL,
        download_dir / "RealESRGANv3_v1.7z",
        REALESRGAN_ARCHIVE_SHA256,
    )
    art_archive = download(
        ARTCNN_ARCHIVE_URL,
        download_dir / "artcnn_v8.7z",
        ARTCNN_ARCHIVE_SHA256,
    )
    extract_archive(real_archive, model_dir)
    extract_archive(art_archive, model_dir)
    vsmlrt_script = download(
        VSMLRT_SCRIPT_URL,
        work_dir / "vsmlrt.py",
        VSMLRT_SCRIPT_SHA256,
    )
    return {
        "vsmlrt": vsmlrt_script.resolve(),
        "realesrgan": find_model(model_dir, "realesr-animevideov3.onnx"),
        "artcnn_luma": find_model(model_dir, "ArtCNN_R16F96.onnx"),
        "artcnn_chroma": find_model(model_dir, "ArtCNN_R16F96_Chroma.onnx"),
    }


def ffmpeg_video_args(args: argparse.Namespace) -> list[str]:
    if args.video_codec in {"hevc_nvenc", "h264_nvenc"}:
        result = [
            "-c:v",
            args.video_codec,
            "-gpu",
            str(args.encode_gpu),
            "-preset",
            args.nvenc_preset,
            "-tune",
            "hq",
            "-rc",
            "vbr",
            "-cq",
            str(args.cq),
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
    else:
        result = [
            "-c:v",
            args.video_codec,
            "-crf",
            str(args.crf),
            "-preset",
            args.preset,
        ]
        if args.video_codec in {"libx264", "libx265"}:
            result.extend(["-tune", "animation"])
    return result


def color_args(info: VideoInfo) -> list[str]:
    ffmpeg_matrix = {
        "709": "bt709",
        "170m": "smpte170m",
        "470bg": "bt470bg",
        "2020ncl": "bt2020nc",
    }.get(info.matrix, "bt709")
    ffmpeg_transfer = {
        "709": "bt709",
        "170m": "smpte170m",
        "470bg": "gamma28",
        "2020_10": "bt2020-10",
    }.get(info.transfer, "bt709")
    ffmpeg_primaries = {
        "709": "bt709",
        "170m": "smpte170m",
        "470bg": "bt470bg",
        "2020": "bt2020",
    }.get(info.primaries, "bt709")
    return [
        "-colorspace",
        ffmpeg_matrix,
        "-color_trc",
        ffmpeg_transfer,
        "-color_primaries",
        ffmpeg_primaries,
        "-color_range",
        "tv",
    ]


def output_dimensions(args: argparse.Namespace, info: VideoInfo) -> tuple[int, int]:
    width, height = args.input_width, args.input_height
    if width == 0 and height == 0:
        width, height = info.width, info.height
    elif width == 0:
        width = round(info.width * height / info.height)
    elif height == 0:
        height = round(info.height * width / info.width)
    width = max(2, width + width % 2)
    height = max(2, height + height % 2)
    return int(round(width * args.scale)), int(round(height * args.scale))


def nvenc_smoke_test(args: argparse.Namespace, width: int, height: int) -> tuple[bool, str]:
    original = args.video_codec
    args.video_codec = "hevc_nvenc"
    try:
        command = [
            args.ffmpeg_bin,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"color=c=black:size={width}x{height}:rate=1",
            "-frames:v",
            "1",
            *ffmpeg_video_args(args),
            "-pix_fmt",
            "yuv420p",
            "-f",
            "null",
            "-",
        ]
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return result.returncode == 0, (result.stderr.strip() or result.stdout.strip())
    finally:
        args.video_codec = original


def resolve_video_codec(args: argparse.Namespace, info: VideoInfo) -> None:
    if args.video_codec not in {"auto", "hevc_nvenc"}:
        return
    width, height = output_dimensions(args, info)
    ok, detail = nvenc_smoke_test(args, width, height)
    if ok:
        args.video_codec = "hevc_nvenc"
        print(f"[encoder] hevc_nvenc runtime OK at {width}x{height}", flush=True)
        return
    if args.video_codec == "hevc_nvenc":
        raise RuntimeError(
            f"hevc_nvenc failed its {width}x{height} runtime test:\n{detail}"
        )
    args.video_codec = "libx265"
    tail = "\n".join(detail.splitlines()[-4:])
    print(
        f"[encoder] hevc_nvenc runtime test failed at {width}x{height}; "
        f"falling back to 8-bit libx265:\n{tail}",
        flush=True,
    )


def tail_text(lines: Iterable[str], count: int = 30) -> str:
    return "\n".join(list(lines)[-count:])


class PipeState:
    def __init__(self, expected_frames: int):
        self.expected_frames = expected_frames
        self.frames = 0
        self.stop = threading.Event()
        self.vspipe_log: deque[str] = deque(maxlen=100)
        self.ffmpeg_log: deque[str] = deque(maxlen=100)
        self.lock = threading.Lock()


def drain_stream(lines: Iterable[str], log: deque[str], state: PipeState, parse_frames: bool = False) -> None:
    pattern = re.compile(r"(?:Frame|frame)\s*[:=]\s*(\d+)")
    for raw in lines:
        line = raw.rstrip()
        if line:
            log.append(line)
        if parse_frames:
            match = pattern.search(line)
            if match:
                with state.lock:
                    state.frames = max(state.frames, int(match.group(1)))


def progress_reporter(
    state: PipeState,
    interval: float,
    start: float,
    label: str,
    monitor_gpu: bool,
) -> None:
    suffix = f":{label}" if label else ""
    while not state.stop.wait(interval):
        with state.lock:
            frames = state.frames
        elapsed = max(0.001, time.monotonic() - start)
        speed = frames / elapsed
        percent = 100 * frames / state.expected_frames if state.expected_frames else 0.0
        eta = (
            (state.expected_frames - frames) / speed
            if speed > 0 and state.expected_frames > frames
            else 0.0
        )
        print(
            f"[progress{suffix}] time={now_text()}, frames={frames}/{state.expected_frames}, "
            f"{percent:.2f}%, speed={speed:.3f} fps, elapsed={format_seconds(elapsed)}, "
            f"eta={format_seconds(eta)}",
            flush=True,
        )
        if not monitor_gpu:
            continue
        nvidia_smi = shutil.which("nvidia-smi")
        if nvidia_smi:
            try:
                result = subprocess.run(
                    [
                        nvidia_smi,
                        "--query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw",
                        "--format=csv,noheader,nounits",
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    text=True,
                    timeout=10,
                )
            except (OSError, subprocess.TimeoutExpired):
                continue
            if result.returncode == 0:
                rows = " | ".join(
                    row.strip() for row in result.stdout.splitlines() if row.strip()
                )
                print(
                    "[gpu-util] index, util%, memory MiB used/total, power W: " + rows,
                    flush=True,
                )


def vspipe_arguments(
    args: argparse.Namespace,
    info: VideoInfo,
    assets: dict[str, Path],
    backend: str,
    start_frame: int,
    frame_count: int,
    output_rate: Fraction,
    gpu_ids: Optional[str] = None,
) -> list[str]:
    script = Path(args.vpy_script).resolve()
    values = {
        "INPUT": Path(args.input).resolve(),
        "VSMLRT_PATH": assets["vsmlrt"],
        "REALESRGAN_MODEL": assets["realesrgan"],
        "ARTCNN_LUMA_MODEL": assets["artcnn_luma"],
        "ARTCNN_CHROMA_MODEL": assets["artcnn_chroma"],
        "ENGINE_DIR": Path(args.engine_dir).resolve(),
        "BACKEND": backend,
        "GPU_IDS": gpu_ids if gpu_ids is not None else args.gpu_ids,
        "START_FRAME": start_frame,
        "FRAME_COUNT": frame_count,
        "OUTPUT_SCALE": args.scale,
        "OUTPUT_FPS_NUM": output_rate.numerator,
        "OUTPUT_FPS_DEN": output_rate.denominator,
        "INPUT_WIDTH": args.input_width,
        "INPUT_HEIGHT": args.input_height,
        "TILE_SIZE": args.tile_size,
        "OVERLAP": args.overlap,
        "FP16": int(args.fp16),
        "NUM_STREAMS": args.num_streams,
        "BATCH_SIZE": args.batch_size,
        "PREFER_NHWC": int(args.prefer_nhwc),
        "WORKSPACE_GIB": args.workspace_gib,
        "USE_CUDA_GRAPH": int(args.cuda_graph),
        "CORE_THREADS": args.core_threads,
        "CACHE_MIB": args.cache_mib,
        "MATRIX": info.matrix,
        "TRANSFER": info.transfer,
        "PRIMARIES": info.primaries,
        "INPUT_RANGE": info.color_range,
        "OUTPUT_RANGE": "limited",
        "MASK_LOW": args.mask_low,
        "MASK_HIGH": args.mask_high,
        "MASK_DARK_BOOST": args.mask_dark_boost,
        "MASK_STRENGTH": args.mask_strength,
        "MASK_INFLATE": args.mask_inflate,
        "MASK_FEATHER": args.mask_feather,
        "MASK_PREFILTER": args.mask_prefilter,
        "MASK_SUPPORT_LOW": args.mask_support_low,
        "MASK_SUPPORT_HIGH": args.mask_support_high,
        "MASK_SUPPORT_RADIUS": args.mask_support_radius,
    }
    command = [args.vspipe_bin, "--progress", "--container", "y4m"]
    for name, value in values.items():
        command.extend(["--arg", f"{name}={value}"])
    command.extend([str(script), "-"])
    return command


def probe_backend(command: list[str], progress_interval: float) -> tuple[bool, str]:
    info_command = command.copy()
    if "--progress" in info_command:
        info_command.remove("--progress")
    if "--container" in info_command:
        container_index = info_command.index("--container")
        del info_command[container_index : container_index + 2]
    info_command.insert(1, "--info")
    if info_command[-1] == "-":
        info_command.pop()
    process = subprocess.Popen(
        info_command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    started = time.monotonic()
    while True:
        try:
            stdout, stderr = process.communicate(timeout=progress_interval)
            break
        except subprocess.TimeoutExpired:
            print(
                f"[backend-progress] time={now_text()}, "
                f"elapsed={format_seconds(time.monotonic() - started)}, still preparing",
                flush=True,
            )
    return process.returncode == 0, (stderr.strip() or stdout.strip())


def choose_backend(
    args: argparse.Namespace,
    command_factory,
) -> tuple[str, list[str]]:
    candidates = ["trt", "ort_cuda"] if args.backend == "auto" else [args.backend]
    failures: list[str] = []
    for candidate in candidates:
        command = command_factory(candidate)
        print(f"[backend] probing {candidate}", flush=True)
        ok, detail = probe_backend(command, args.progress_interval)
        if ok:
            print(f"[backend] selected {candidate}", flush=True)
            return candidate, command
        failures.append(f"{candidate}:\n{detail}")
        detail_tail = "\n".join(detail.splitlines()[-12:])
        print(
            f"[backend] {candidate} unavailable"
            + (f":\n{detail_tail}" if detail_tail else ""),
            flush=True,
        )
    raise RuntimeError("No usable VapourSynth ML backend:\n" + "\n\n".join(failures))


def encode_video(
    args: argparse.Namespace,
    info: VideoInfo,
    vspipe_command: list[str],
    temporary_video: Path,
    expected_frames: int,
    *,
    label: str = "",
    monitor_gpu: bool = True,
) -> None:
    ffmpeg_command = [
        args.ffmpeg_bin,
        "-y",
        "-hide_banner",
        "-loglevel",
        "warning",
        "-nostats",
        "-stats_period",
        "1",
        "-progress",
        "pipe:2",
        "-i",
        "pipe:0",
        "-an",
        *ffmpeg_video_args(args),
        "-pix_fmt",
        "yuv420p",
        *color_args(info),
        str(temporary_video),
    ]
    start = time.monotonic()
    state = PipeState(expected_frames)
    vspipe = subprocess.Popen(
        vspipe_command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=False,
    )
    assert vspipe.stdout is not None and vspipe.stderr is not None
    ffmpeg = subprocess.Popen(
        ffmpeg_command,
        stdin=vspipe.stdout,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=False,
    )
    vspipe.stdout.close()
    assert ffmpeg.stderr is not None

    def decode_stream(binary_stream):
        pending = b""
        while True:
            chunk = os.read(binary_stream.fileno(), 4096)
            if not chunk:
                break
            records = re.split(rb"[\r\n]+", pending + chunk)
            pending = records.pop()
            for record in records:
                if record:
                    yield record.decode("utf-8", errors="replace")
        if pending:
            yield pending.decode("utf-8", errors="replace")

    vs_text = iter(decode_stream(vspipe.stderr))
    ff_text = iter(decode_stream(ffmpeg.stderr))
    vs_thread = threading.Thread(
        target=drain_stream,
        args=(vs_text, state.vspipe_log, state, True),
        daemon=True,
    )
    ff_thread = threading.Thread(
        target=drain_stream,
        args=(ff_text, state.ffmpeg_log, state, True),
        daemon=True,
    )
    progress_thread = threading.Thread(
        target=progress_reporter,
        args=(state, args.progress_interval, start, label, monitor_gpu),
        daemon=True,
    )
    vs_thread.start()
    ff_thread.start()
    progress_thread.start()
    ffmpeg_code = ffmpeg.wait()
    vspipe_code = vspipe.wait()
    state.stop.set()
    vs_thread.join(timeout=5)
    ff_thread.join(timeout=5)
    progress_thread.join(timeout=5)
    if vspipe_code or ffmpeg_code:
        prefix = f"{label}: " if label else ""
        raise RuntimeError(
            f"{prefix}Pipeline failed: vspipe={vspipe_code}, ffmpeg={ffmpeg_code}\n"
            f"vspipe tail:\n{tail_text(state.vspipe_log)}\n"
            f"ffmpeg tail:\n{tail_text(state.ffmpeg_log)}"
        )
    elapsed = time.monotonic() - start
    suffix = f":{label}" if label else ""
    print(
        f"[inference-end{suffix}] wall_end={now_text()}, elapsed={format_seconds(elapsed)}, "
        f"frames={expected_frames}, average={expected_frames / max(elapsed, 0.001):.3f} fps",
        flush=True,
    )


def split_frame_chunks(
    start_frame: int,
    frame_count: int,
    gpu_ids: Sequence[int],
) -> list[tuple[int, int, int, int]]:
    worker_count = min(len(gpu_ids), frame_count)
    base, remainder = divmod(frame_count, worker_count)
    chunks: list[tuple[int, int, int, int]] = []
    cursor = start_frame
    for index, device_id in enumerate(gpu_ids[:worker_count]):
        count = base + (1 if index < remainder else 0)
        chunks.append((index, device_id, cursor, count))
        cursor += count
    return chunks


def concat_video_chunks(
    args: argparse.Namespace,
    chunks: Sequence[Path],
    output: Path,
) -> None:
    concat_file = output.with_name(output.name + ".concat.txt")
    lines = []
    for chunk in chunks:
        escaped = str(chunk.resolve()).replace("'", "'\\''")
        lines.append(f"file '{escaped}'\n")
    concat_file.write_text("".join(lines), encoding="utf-8")
    try:
        run_checked(
            [
                args.ffmpeg_bin,
                "-y",
                "-hide_banner",
                "-loglevel",
                "warning",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(concat_file),
                "-map",
                "0:v:0",
                "-c",
                "copy",
                "-avoid_negative_ts",
                "make_zero",
                str(output),
            ],
            "parallel video concat",
        )
    finally:
        concat_file.unlink(missing_ok=True)


def encode_video_parallel(
    args: argparse.Namespace,
    info: VideoInfo,
    assets: dict[str, Path],
    backend: str,
    start_frame: int,
    frame_count: int,
    output_rate: Fraction,
    temporary_video: Path,
    gpu_ids: Sequence[int],
) -> None:
    chunks = split_frame_chunks(start_frame, frame_count, gpu_ids)
    chunk_paths = [
        temporary_video.with_name(f"{temporary_video.stem}.part{index:02d}.mkv")
        for index, _, _, _ in chunks
    ]
    for path in chunk_paths:
        path.unlink(missing_ok=True)

    print(
        f"[parallel] mode=independent-contiguous-chunks, workers={len(chunks)}, "
        "models=resident-per-gpu",
        flush=True,
    )
    for index, device_id, chunk_start, chunk_count in chunks:
        print(
            f"[parallel] part={index}, gpu={device_id}, "
            f"source_frames={chunk_start}:{chunk_start + chunk_count}, count={chunk_count}",
            flush=True,
        )

    started = time.monotonic()

    def run_chunk(
        index: int,
        device_id: int,
        chunk_start: int,
        chunk_count: int,
    ) -> None:
        worker_args = copy.copy(args)
        if worker_args.video_codec in {"hevc_nvenc", "h264_nvenc"}:
            worker_args.encode_gpu = device_id
        command = vspipe_arguments(
            worker_args,
            info,
            assets,
            backend,
            chunk_start,
            chunk_count,
            output_rate,
            gpu_ids=str(device_id),
        )
        encode_video(
            worker_args,
            info,
            command,
            chunk_paths[index],
            chunk_count,
            label=f"gpu{device_id}/part{index}",
            monitor_gpu=index == 0,
        )

    try:
        with ThreadPoolExecutor(max_workers=len(chunks)) as executor:
            futures = [
                executor.submit(run_chunk, index, device_id, chunk_start, chunk_count)
                for index, device_id, chunk_start, chunk_count in chunks
            ]
            for future in futures:
                future.result()
        concat_video_chunks(args, chunk_paths, temporary_video)
    finally:
        for path in chunk_paths:
            path.unlink(missing_ok=True)

    elapsed = time.monotonic() - started
    print(
        f"[parallel-end] wall_end={now_text()}, elapsed={format_seconds(elapsed)}, "
        f"frames={frame_count}, aggregate={frame_count / max(elapsed, 0.001):.3f} fps",
        flush=True,
    )


def mux_audio(
    args: argparse.Namespace,
    info: VideoInfo,
    temporary_video: Path,
    output: Path,
    start: float,
    duration: float,
) -> None:
    if not info.has_audio:
        temporary_video.replace(output)
        print("[audio] input has no audio stream", flush=True)
        return
    command = [args.ffmpeg_bin, "-y", "-hide_banner", "-loglevel", "warning"]
    if start > 0:
        command.extend(["-ss", f"{start:.9f}"])
    if args.test_seconds > 0:
        command.extend(["-t", f"{duration:.9f}"])
    command.extend(
        [
            "-i",
            str(Path(args.input).resolve()),
            "-i",
            str(temporary_video),
            "-map",
            "1:v:0",
            "-map",
            "0:a:0?",
            "-c:v",
            "copy",
        ]
    )
    if args.audio_codec == "copy":
        command.extend(["-c:a", "copy"])
    else:
        command.extend(
            [
                "-c:a",
                "aac",
                "-b:a",
                args.audio_bitrate,
                "-af",
                "aresample=async=1:first_pts=0",
            ]
        )
    command.extend(["-shortest", "-movflags", "+faststart", str(output)])
    run_checked(command, "audio mux")
    temporary_video.unlink(missing_ok=True)


def validate_args(args: argparse.Namespace) -> None:
    input_path = Path(args.input).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input video not found: {input_path}")
    if not Path(args.vpy_script).expanduser().is_file():
        raise FileNotFoundError(f"VapourSynth script not found: {args.vpy_script}")
    output_path = Path(args.output).expanduser().resolve()
    if output_path == input_path:
        raise ValueError("--output must not overwrite --input")
    if output_path.suffix.lower() not in {".mp4", ".mkv"}:
        raise ValueError("--output must use .mp4 or .mkv")
    if args.scale not in {2.0, 3.0, 4.0}:
        raise ValueError("--scale must be 2, 3, or 4 for this hybrid pipeline")
    if args.start_time < 0 or args.test_seconds < 0:
        raise ValueError("Start and test duration must be non-negative")
    if args.tile_size < 0 or args.overlap < 0:
        raise ValueError("Tile size and overlap must be non-negative")
    if args.input_width < 0 or args.input_height < 0:
        raise ValueError("Input width and height must be non-negative")
    if args.tile_size and args.overlap * 2 >= args.tile_size:
        raise ValueError("--overlap must be smaller than half of --tile-size")
    if args.num_streams <= 0 or args.batch_size <= 0 or args.workspace_gib <= 0:
        raise ValueError("Streams, batch size, and TensorRT workspace must be positive")
    if args.core_threads <= 0 or args.cache_mib <= 0:
        raise ValueError("Core threads and cache size must be positive")
    if not 0 <= args.mask_low < args.mask_high <= 1:
        raise ValueError("Mask thresholds must satisfy 0 <= low < high <= 1")
    if args.mask_dark_boost < 0 or args.mask_strength < 0:
        raise ValueError("Mask dark boost and strength must be non-negative")
    if args.mask_inflate < 0 or args.mask_feather < 0 or args.mask_prefilter < 0:
        raise ValueError("Mask morphology settings must be non-negative")
    if not 0 <= args.mask_support_low < args.mask_support_high <= 1:
        raise ValueError("Mask support thresholds must satisfy 0 <= low < high <= 1")
    if args.mask_support_radius < 0:
        raise ValueError("--mask-support-radius must be non-negative")
    if not 0 <= args.cq <= 51 or not 0 <= args.crf <= 51:
        raise ValueError("CQ and CRF must be between 0 and 51")
    if args.encode_gpu < 0:
        raise ValueError("--encode-gpu must be non-negative")
    if args.progress_interval <= 0:
        raise ValueError("--progress-interval must be positive")
    gpu_ids = [item.strip() for item in args.gpu_ids.split(",") if item.strip()]
    if not gpu_ids or any(not item.isdigit() for item in gpu_ids):
        raise ValueError("--gpu-ids must be a comma-separated list such as 0,1")
    if len(gpu_ids) != len(set(gpu_ids)):
        raise ValueError("--gpu-ids must not contain duplicates")


def process(args: argparse.Namespace) -> None:
    validate_args(args)
    args.ffmpeg_bin = require_binary(args.ffmpeg_bin)
    args.ffprobe_bin = require_binary(args.ffprobe_bin)
    args.vspipe_bin = require_binary(args.vspipe_bin)
    input_path = Path(args.input).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    work_dir = Path(args.work_dir).expanduser().resolve()
    Path(args.engine_dir).expanduser().resolve().mkdir(parents=True, exist_ok=True)

    info = probe_video(input_path, args.ffprobe_bin)
    resolve_video_codec(args, info)
    target_width, target_height = output_dimensions(args, info)
    if args.start_time >= info.duration:
        raise ValueError(f"--start-time must be below {info.duration:.3f}s")
    duration = info.duration - args.start_time
    if args.test_seconds > 0:
        duration = min(duration, args.test_seconds)
    start_frame = int(round(args.start_time * float(info.fps)))
    if info.frames is not None and start_frame >= info.frames:
        raise ValueError("--start-time rounds beyond the last decodable video frame")
    actual_start = start_frame / float(info.fps)
    source_frames = max(1, int(round(duration * float(info.fps))))
    if info.frames is not None and info.frames > 0:
        source_frames = min(source_frames, info.frames - start_frame)
    if source_frames <= 0:
        raise ValueError("Selected range contains no decodable video frames")
    selected_duration = source_frames / float(info.fps)
    output_rate = parse_output_rate(args.fps, info.fps)
    expected_frames = max(1, int(round(selected_duration * float(output_rate))))
    output_duration = expected_frames / float(output_rate)
    assets = prepare_assets(work_dir)

    print(f"[run] wall_start={now_text()}", flush=True)
    print(f"[input] {input_path}", flush=True)
    print(
        f"[input] source={info.width}x{info.height}, output={target_width}x{target_height}, "
        f"source_fps={float(info.fps):.6f} "
        f"({info.fps}), output_fps={float(output_rate):.6f} ({output_rate}), "
        f"audio={info.has_audio}, matrix={info.matrix}, range={info.color_range}",
        flush=True,
    )
    if target_width > 3840 or target_height > 2160:
        print(
            "[warning] output exceeds 3840x2160; verify --scale and "
            "--input-width/--input-height if the target is 4K UHD",
            flush=True,
        )
    if info.vfr_suspected:
        print(
            "[warning] input may be variable-frame-rate; this Y4M pipeline normalizes it "
            "to ffprobe avg_frame_rate. Check sync on the 10-second test before a full run.",
            flush=True,
        )
    print(
        f"[range] requested_start={format_seconds(args.start_time)}, "
        f"frame_aligned_start={format_seconds(actual_start)}, "
        f"requested_duration={format_seconds(duration)}, "
        f"selected_duration={format_seconds(selected_duration)}, "
        f"source_frames={source_frames}, expected_output_frames={expected_frames}, "
        f"output_duration={format_seconds(output_duration)}",
        flush=True,
    )
    print(
        f"[pipeline] line=realesr-animevideov3, flat=ArtCNN_R16F96, "
        f"chroma=ArtCNN_R16F96_Chroma, scale={args.scale:g}, gpu_ids={args.gpu_ids}",
        flush=True,
    )

    gpu_ids = [int(item.strip()) for item in args.gpu_ids.split(",") if item.strip()]

    def command_for(backend: str) -> list[str]:
        return vspipe_arguments(
            args,
            info,
            assets,
            backend,
            start_frame,
            source_frames,
            output_rate,
            gpu_ids=str(gpu_ids[0]),
        )

    backend, _ = choose_backend(args, command_for)
    if backend == "ort_cuda" and args.cuda_graph:
        print(
            "[backend] CUDA Graph requested but disabled for ORT CUDA because the "
            "multi-model T4 graph is not capture-safe",
            flush=True,
        )
    print(
        f"[inference-start] wall_start={now_text()}, backend={backend}, "
        f"tile={args.tile_size}, overlap={args.overlap}, batch={args.batch_size}, "
        f"streams={args.num_streams}, fp16={args.fp16}, "
        f"half_io={args.fp16}, prefer_nhwc={args.prefer_nhwc}",
        flush=True,
    )
    temporary_video = output.with_name(output.stem + ".video-only.tmp.mkv")
    if temporary_video.exists():
        temporary_video.unlink()
    if len(gpu_ids) > 1 and output_rate == info.fps:
        encode_video_parallel(
            args,
            info,
            assets,
            backend,
            start_frame,
            source_frames,
            output_rate,
            temporary_video,
            gpu_ids,
        )
    else:
        if len(gpu_ids) > 1:
            print(
                "[scheduler] exact multi-process chunking currently requires --fps source; "
                f"using cuda:{gpu_ids[0]} for non-source FPS",
                flush=True,
            )
        vspipe_command = vspipe_arguments(
            args,
            info,
            assets,
            backend,
            start_frame,
            source_frames,
            output_rate,
            gpu_ids=str(gpu_ids[0]),
        )
        encode_video(args, info, vspipe_command, temporary_video, expected_frames)
    mux_audio(args, info, temporary_video, output, actual_start, output_duration)
    final_info = probe_video(output, args.ffprobe_bin)
    delta = abs(final_info.duration - output_duration)
    size_mib = output.stat().st_size / 1024**2
    print(
        f"[output] {output}\n"
        f"[verify] duration={final_info.duration:.3f}s, target={output_duration:.3f}s, "
        f"delta={delta:.3f}s, fps={final_info.fps}, audio={final_info.has_audio}\n"
        f"[size] {size_mib:.2f} MiB\n"
        f"[run-end] wall_end={now_text()}",
        flush=True,
    )
    frame_tolerance = 1.5 / float(output_rate)
    if delta > max(0.15, frame_tolerance):
        print(
            f"[warning] output duration differs by {delta:.3f}s; inspect audio sync before a full run",
            flush=True,
        )


def build_parser() -> argparse.ArgumentParser:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Multi-GPU VapourSynth Real-ESRGAN/ArtCNN anime upscaler",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--vpy-script", default=str(here / "realesrgan_mask.vpy"))
    parser.add_argument("--work-dir", default="./realesrgan-mask-assets")
    parser.add_argument("--engine-dir", default="./realesrgan-mask-engines")
    parser.add_argument(
        "--scale",
        type=float,
        default=2.0,
        help="Output width/height multiplier; 720p to 2160p uses 3",
    )
    parser.add_argument("--fps", default="source")
    parser.add_argument("--start-time", type=float, default=0.0)
    parser.add_argument("--test-seconds", type=float, default=10.0)
    parser.add_argument("--input-width", type=int, default=0)
    parser.add_argument("--input-height", type=int, default=0)

    parser.add_argument("--backend", choices=("auto", "trt", "ort_cuda"), default="auto")
    parser.add_argument(
        "--gpu-ids",
        default="0,1",
        help="Comma-separated CUDA devices; each runs an independent contiguous frame chunk",
    )
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tile-size", type=int, default=384)
    parser.add_argument("--overlap", type=int, default=24)
    parser.add_argument("--num-streams", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--prefer-nhwc",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--workspace-gib", type=float, default=4.0)
    parser.add_argument("--cuda-graph", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--core-threads", type=int, default=8)
    parser.add_argument("--cache-mib", type=int, default=2048)

    parser.add_argument("--mask-low", type=float, default=0.040)
    parser.add_argument("--mask-high", type=float, default=0.140)
    parser.add_argument("--mask-dark-boost", type=float, default=0.5)
    parser.add_argument("--mask-strength", type=float, default=1.0)
    parser.add_argument("--mask-inflate", type=int, default=1)
    parser.add_argument("--mask-feather", type=int, default=1)
    parser.add_argument("--mask-prefilter", type=int, default=2)
    parser.add_argument("--mask-support-low", type=float, default=0.012)
    parser.add_argument("--mask-support-high", type=float, default=0.035)
    parser.add_argument("--mask-support-radius", type=int, default=2)

    parser.add_argument(
        "--video-codec",
        choices=("auto", "hevc_nvenc", "h264_nvenc", "libx264", "libx265"),
        default="auto",
    )
    parser.add_argument("--cq", type=int, default=14)
    parser.add_argument("--nvenc-preset", choices=tuple(f"p{i}" for i in range(1, 8)), default="p7")
    parser.add_argument("--encode-gpu", type=int, default=0)
    parser.add_argument("--crf", type=int, default=14)
    parser.add_argument("--preset", default="slow")
    parser.add_argument("--audio-codec", choices=("copy", "aac"), default="aac")
    parser.add_argument("--audio-bitrate", default="192k")
    parser.add_argument("--progress-interval", type=float, default=60.0)
    parser.add_argument("--vspipe-bin", default="vspipe")
    parser.add_argument("--ffmpeg-bin", default="ffmpeg")
    parser.add_argument("--ffprobe-bin", default="ffprobe")
    return parser


def main() -> None:
    process(build_parser().parse_args())


if __name__ == "__main__":
    main()
