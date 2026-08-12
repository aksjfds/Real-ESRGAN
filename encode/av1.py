"""AV1 FFmpeg encoders with automatic 8-bit/10-bit output."""

from __future__ import annotations

import queue
import subprocess
import threading
from pathlib import Path
from typing import Optional

import numpy as np


CODECS = {"libsvtav1", "libaom-av1", "av1_nvenc"}
_SVTAV1_PRESET = 6
_AOM_CPU_USED = 6
_AOM_QUEUE_DEPTH = 2

# v8.5 AV1 NVENC configuration. configure() overrides these from CLI args.
_AV1_PROFILE = "main"
_AV1_TUNE = "hq"
_AV1_RC = "vbr"
_AV1_BITRATE = "0"
_AV1_MULTIPASS = "fullres"
_AV1_RC_LOOKAHEAD = 28
_AV1_SPATIAL_AQ = 1
_AV1_TEMPORAL_AQ = 1
_AV1_AQ_STRENGTH = 8
_AV1_B_REF_MODE = "middle"
_AV1_B_FRAMES = 3
_AV1_GOP_SIZE = 240


def require_encoder(ffmpeg_bin: str, encoder: str) -> None:
    result = subprocess.run(
        [ffmpeg_bin, "-hide_banner", "-encoders"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"ffmpeg encoder probe failed (exit {result.returncode}):\n{detail}")
    if encoder not in (result.stdout + result.stderr):
        raise RuntimeError(f"ffmpeg does not provide the requested video encoder: {encoder}")


def encoder_available(ffmpeg_bin: str, encoder: str) -> bool:
    try:
        require_encoder(ffmpeg_bin, encoder)
        return True
    except RuntimeError:
        return False


def resolve_requested_encoder(ffmpeg_bin: str, requested: str) -> str:
    if requested == "libsvtav1" and not encoder_available(ffmpeg_bin, requested):
        if encoder_available(ffmpeg_bin, "libaom-av1"):
            print(
                "[warning] FFmpeg has no libsvtav1; falling back to CPU AV1 encoder libaom-av1.",
                flush=True,
            )
            return "libaom-av1"
        raise RuntimeError(
            "FFmpeg provides neither libsvtav1 nor libaom-av1, so CPU AV1 encoding is unavailable."
        )
    return requested


def validate_args(args) -> None:
    if args.video_codec not in CODECS:
        raise ValueError(f"Unsupported AV1 encoder: {args.video_codec}")
    if args.video_codec in {"libsvtav1", "libaom-av1"} and not 0 <= args.crf <= 63:
        raise ValueError("--crf must be between 0 and 63 for CPU AV1")
    if args.video_codec == "av1_nvenc":
        if not 0 <= args.cq <= 63:
            raise ValueError("--cq must be between 0 and 63 for av1_nvenc")
        if args.av1_profile != "main":
            raise ValueError("AV1 NVENC uses AV1 Main profile.")
        if not 0 <= args.av1_b_frames <= 31:
            raise ValueError("--av1-b-frames must be between 0 and 31")
        max_lookahead = max(0, 31 - int(args.av1_b_frames))
        if not 0 <= args.av1_rc_lookahead <= max_lookahead:
            raise ValueError(
                f"--av1-rc-lookahead must be between 0 and {max_lookahead} "
                f"when --av1-b-frames={args.av1_b_frames}"
            )
        if not 1 <= args.av1_aq_strength <= 15:
            raise ValueError("--av1-aq-strength must be between 1 and 15")
        if args.av1_gop_size < 1:
            raise ValueError("--av1-gop-size must be at least 1")
        if args.av1_b_ref_mode != "disabled" and args.av1_b_frames == 0:
            raise ValueError("--av1-b-ref-mode requires --av1-b-frames > 0")
    if not 0 <= args.svtav1_preset <= 13:
        raise ValueError("--svtav1-preset must be between 0 and 13")
    if not 0 <= args.aom_cpu_used <= 8:
        raise ValueError("--aom-cpu-used must be between 0 and 8")
    if args.encode_gpu < 0:
        raise ValueError("--encode-gpu must be non-negative.")


def _codec_args(
    codec: str,
    crf: int,
    cq: int,
    nvenc_preset: str,
    encode_gpu: int,
    svtav1_preset: int,
    aom_cpu_used: int,
) -> list[str]:
    if codec == "libsvtav1":
        return ["-preset", str(svtav1_preset), "-crf", str(crf)]
    if codec == "libaom-av1":
        return [
            "-crf", str(crf),
            "-b:v", "0",
            "-cpu-used", str(aom_cpu_used),
            "-row-mt", "1",
        ]
    if codec == "av1_nvenc":
        return [
            "-gpu", str(encode_gpu),
            "-preset", nvenc_preset,
            "-tune", _AV1_TUNE,
            "-rc", _AV1_RC,
            "-cq", str(cq),
            "-b:v", _AV1_BITRATE,
            "-multipass", _AV1_MULTIPASS,
            "-rc-lookahead", str(_AV1_RC_LOOKAHEAD),
            "-spatial-aq", str(_AV1_SPATIAL_AQ),
            "-temporal-aq", str(_AV1_TEMPORAL_AQ),
            "-aq-strength", str(_AV1_AQ_STRENGTH),
            "-b_ref_mode", _AV1_B_REF_MODE,
            "-bf", str(_AV1_B_FRAMES),
            "-g", str(_AV1_GOP_SIZE),
        ]
    raise ValueError(f"Unsupported AV1 encoder: {codec}")


def _libaom_tile_args(width: int, height: int) -> tuple[list[str], str]:
    if width >= 3840 or height >= 2160:
        return ["-tiles", "2x2"], "2x2"
    if width >= 1920 or height >= 1080:
        return ["-tiles", "2x1"], "2x1"
    return [], "1x1"


def _frame_pixel_formats(frame: np.ndarray, codec: str) -> tuple[str, str]:
    if frame.dtype.kind == "u" and frame.dtype.itemsize == 1:
        return "rgb24", "yuv420p"
    if frame.dtype.kind == "u" and frame.dtype.itemsize == 2:
        output_pix_fmt = "p010le" if codec == "av1_nvenc" else "yuv420p10le"
        return "rgb48le", output_pix_fmt
    raise RuntimeError(f"Unsupported inference frame dtype for encoding: {frame.dtype}")


class RawVideoWriter:
    """AV1 writer with automatic bit depth and asynchronous libaom stdin."""

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
        del preset
        if codec not in CODECS:
            raise RuntimeError(f"Non-AV1 codec reached AV1 writer: {codec}")

        self.path = path
        self.ffmpeg_bin = ffmpeg_bin
        self.width = width
        self.height = height
        self.input_fps_rate = input_fps_rate
        self.output_fps_rate = output_fps_rate
        self.codec = codec
        self.crf = crf
        self.cq = cq
        self.nvenc_preset = nvenc_preset
        self.encode_gpu = encode_gpu
        self.process: Optional[subprocess.Popen] = None

        self._queue: Optional[queue.Queue[Optional[np.ndarray]]] = None
        self._thread: Optional[threading.Thread] = None
        self._worker_error: Optional[BaseException] = None
        self._closed = False

        if codec == "libaom-av1":
            self._queue = queue.Queue(maxsize=_AOM_QUEUE_DEPTH)
            self._thread = threading.Thread(
                target=self._write_loop,
                name="libaom-av1-writer",
                daemon=True,
            )
            self._thread.start()

    def _start(self, frame: np.ndarray) -> None:
        if self.process is not None:
            return

        raw_pix_fmt, output_pix_fmt = _frame_pixel_formats(frame, self.codec)
        command = [
            self.ffmpeg_bin,
            "-y",
            "-hide_banner",
            "-loglevel", "error",
            "-f", "rawvideo",
            "-pix_fmt", raw_pix_fmt,
            "-s:v", f"{self.width}x{self.height}",
            "-r", self.input_fps_rate,
            "-i", "pipe:0",
            "-an",
        ]
        if self.output_fps_rate != self.input_fps_rate:
            command += ["-vf", f"fps={self.output_fps_rate}"]

        command += ["-c:v", self.codec]
        command += _codec_args(
            self.codec,
            self.crf,
            self.cq,
            self.nvenc_preset,
            self.encode_gpu,
            _SVTAV1_PRESET,
            _AOM_CPU_USED,
        )
        if self.codec == "av1_nvenc" and output_pix_fmt == "p010le":
            command += ["-highbitdepth", "1"]
        if self.codec == "libaom-av1":
            tile_args, _ = _libaom_tile_args(self.width, self.height)
            command += tile_args

        command += ["-pix_fmt", output_pix_fmt, "-tag:v", "av01", str(self.path)]
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    def _write_frame_sync(self, frame: np.ndarray) -> None:
        assert self.process is not None and self.process.stdin is not None
        self.process.stdin.write(memoryview(frame).cast("B"))

    def _write_loop(self) -> None:
        assert self._queue is not None
        while True:
            frame = self._queue.get()
            try:
                if frame is None:
                    return
                self._write_frame_sync(frame)
            except BaseException as error:
                self._worker_error = error
                return
            finally:
                self._queue.task_done()

    def _raise_worker_error(self) -> None:
        if self._worker_error is not None:
            raise RuntimeError("background AV1 encoder writer failed") from self._worker_error

    def write(self, frame: np.ndarray) -> None:
        if self._closed:
            raise RuntimeError("cannot write to a closed AV1 encoder")
        frame = np.ascontiguousarray(frame)
        self._start(frame)
        if self._queue is None:
            try:
                self._write_frame_sync(frame)
            except BrokenPipeError as error:
                assert self.process is not None
                detail = self.process.stderr.read().decode(errors="replace") if self.process.stderr else ""
                raise RuntimeError(f"ffmpeg encoder closed its input early:\n{detail}") from error
            return

        while True:
            self._raise_worker_error()
            try:
                self._queue.put(frame, timeout=0.1)
                return
            except queue.Full:
                continue

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True

        if self._queue is not None and self._thread is not None:
            while self._thread.is_alive():
                try:
                    self._queue.put(None, timeout=0.1)
                    break
                except queue.Full:
                    if self._worker_error is not None:
                        break
            self._thread.join()

        if self.process is None:
            return
        if self.process.stdin is not None:
            try:
                self.process.stdin.close()
            except BrokenPipeError:
                pass
        stderr = self.process.stderr.read() if self.process.stderr is not None else b""
        if self.process.stderr is not None:
            self.process.stderr.close()
        return_code = self.process.wait()

        if self._worker_error is not None:
            raise RuntimeError(
                "background AV1 encoder writer failed:\n"
                f"{stderr.decode(errors='replace')}"
            ) from self._worker_error
        if return_code != 0:
            raise RuntimeError(
                f"ffmpeg encode failed (exit {return_code}):\n"
                f"{stderr.decode(errors='replace')}"
            )


def probe_encoder_runtime(args) -> None:
    probe_size = "640x360" if args.video_codec == "av1_nvenc" else "128x128"
    command = [
        args.ffmpeg_bin,
        "-hide_banner",
        "-loglevel", "error",
        "-f", "lavfi",
        "-i", f"color=black:size={probe_size}:rate=1,format=rgb24",
        "-frames:v", "1",
        "-c:v", args.video_codec,
        *_codec_args(
            args.video_codec,
            args.crf,
            args.cq,
            args.nvenc_preset,
            args.encode_gpu,
            args.svtav1_preset,
            args.aom_cpu_used,
        ),
        "-pix_fmt", "yuv420p",
        "-f", "null", "-",
    ]
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if result.returncode == 0:
        return

    detail = result.stdout.strip()
    if args.video_codec == "av1_nvenc":
        raise RuntimeError(
            "av1_nvenc is present in FFmpeg but the v8.5 runtime probe failed. "
            "Check the FFmpeg options below and the selected NVIDIA GPU/driver.\n"
            f"FFmpeg output:\n{detail}"
        )
    raise RuntimeError(
        f"{args.video_codec} runtime probe failed (exit {result.returncode}):\n{detail}"
    )


def configure(args) -> None:
    global _SVTAV1_PRESET, _AOM_CPU_USED
    global _AV1_PROFILE, _AV1_TUNE, _AV1_RC, _AV1_BITRATE
    global _AV1_MULTIPASS, _AV1_RC_LOOKAHEAD, _AV1_SPATIAL_AQ, _AV1_TEMPORAL_AQ
    global _AV1_AQ_STRENGTH, _AV1_B_REF_MODE, _AV1_B_FRAMES, _AV1_GOP_SIZE

    _SVTAV1_PRESET = int(args.svtav1_preset)
    _AOM_CPU_USED = int(args.aom_cpu_used)

    _AV1_PROFILE = str(args.av1_profile)
    _AV1_TUNE = str(args.av1_tune)
    _AV1_RC = str(args.av1_rc)
    _AV1_BITRATE = str(args.av1_bitrate)
    _AV1_MULTIPASS = str(args.av1_multipass)
    _AV1_RC_LOOKAHEAD = int(args.av1_rc_lookahead)
    _AV1_SPATIAL_AQ = int(args.av1_spatial_aq)
    _AV1_TEMPORAL_AQ = int(args.av1_temporal_aq)
    _AV1_AQ_STRENGTH = int(args.av1_aq_strength)
    _AV1_B_REF_MODE = str(args.av1_b_ref_mode)
    _AV1_B_FRAMES = int(args.av1_b_frames)
    _AV1_GOP_SIZE = int(args.av1_gop_size)
