"""Dynamic multi-GPU scheduler for BasicVSR++ + full-frame Real-ESRGAN.

The scheduler removes the global BVS->SR phase barrier. Each GPU runs exactly
one heavy task at a time (BasicVSR++ clip or Real-ESRGAN frame), while different
GPUs may be in different stages. Quality-related model parameters are unchanged.
"""
from __future__ import annotations

import heapq
import queue
import sys
import time
from fractions import Fraction
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import torch
from tqdm import tqdm

from . import pipeline as base_pipeline


class _RestoredTask:
    __slots__ = ("frames", "emit_start", "emit_end", "output_start")

    def __init__(self, frames, emit_start: int, emit_end: int, output_start: int) -> None:
        self.frames = frames
        self.emit_start = int(emit_start)
        self.emit_end = int(emit_end)
        self.output_start = int(output_start)

    @property
    def emit_count(self) -> int:
        return max(0, self.emit_end - self.emit_start)


def _run_bvs_group(preprocessor, tasks: Sequence[_RestoredTask]):
    torch.cuda.set_device(int(preprocessor.config.gpu_id))
    clips = [task.frames for task in tasks]
    if hasattr(preprocessor, "enhance_clips") and len(clips) > 1:
        return preprocessor.enhance_clips(clips)
    return [preprocessor.enhance_clip(frames) for frames in clips]


def process_video(args) -> None:
    """Run v5.2 dynamic scheduling for multi-GPU B-E; otherwise use base pipeline."""
    profile_name = str(getattr(args, "source_profile", "A")).upper()
    requested_gpu_ids = base_pipeline.base.parse_gpu_ids(args.gpu_ids)
    if profile_name == "A" or len(requested_gpu_ids) < 2:
        base_pipeline.process_video(args)
        return
    if any(gpu is None for gpu in requested_gpu_ids):
        raise RuntimeError("BasicVSR++ profiles B-E require CUDA")

    if base_pipeline.base._require_encoder is None or base_pipeline.base._writer_type is None:
        raise RuntimeError("Encoding backend is not configured. Run through root inference.py.")
    base = base_pipeline.base
    base.require_binary(args.ffmpeg_bin)
    base.require_binary(args.ffprobe_bin)
    base._require_encoder(args.ffmpeg_bin, args.video_codec)

    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input video not found: {input_path}")
    if input_path == output_path:
        raise ValueError("Input and output paths must be different.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_video = output_path.with_name(output_path.stem + ".video_only.tmp.mp4")

    info = base.probe_video(input_path, args.ffprobe_bin)
    source_rate = Fraction(info.fps_num, info.fps_den)
    output_rate = base.parse_output_rate(args.fps, source_rate)
    inference_rate = min(source_rate, output_rate)
    output_fps = float(output_rate)
    output_rate_text = f"{output_rate.numerator}/{output_rate.denominator}"
    inference_fps = float(inference_rate)
    inference_rate_text = f"{inference_rate.numerator}/{inference_rate.denominator}"
    in_w, in_h = info.width, info.height
    out_w, out_h = int(round(in_w * args.scale)), int(round(in_h * args.scale))
    if out_w % 2 or out_h % 2:
        raise ValueError(f"4:2:0 output needs even dimensions, got {out_w}x{out_h}.")
    start, duration, expected = base.resolve_range(info, args.start_time, args.test_seconds, inference_rate)
    expected_output = max(1, int(round(duration * output_fps)))
    end = start + duration
    gpu_ids = [int(gpu) for gpu in requested_gpu_ids]
    profile = base_pipeline.SOURCE_PROFILES[profile_name]
    native_scale = base._model_native_scale(args.model)
    config = base.WorkerConfig(args.model, base.resolve_model_paths(args))
    dtype = np.dtype("<u2") if info.bit_depth == 10 else np.dtype(np.uint8)
    input_shape = (in_h, in_w, 3)
    native_shape = (in_h * native_scale, in_w * native_scale, 3)

    mode = "test" if args.test_seconds > 0 else "full/selected range"
    print("=== Real-ESRGAN ===", flush=True)
    print(
        f"Input   : {input_path.name} | {in_w}x{in_h} | {info.fps:.3f} fps | "
        f"{info.bit_depth}-bit ({info.pix_fmt})",
        flush=True,
    )
    print(
        f"Output  : {out_w}x{out_h} | {output_fps:.3f} fps | {info.bit_depth}-bit "
        f"({base._output_pixel_format(args.video_codec, info.bit_depth)}) | {args.video_codec}",
        flush=True,
    )
    print(
        f"Range   : {mode} | {base.format_seconds(start)} -> {base.format_seconds(end)} | "
        f"{duration:.3f}s | {expected} inference / {expected_output} output frames",
        flush=True,
    )
    scale_text = (
        f"native={native_scale}x"
        if float(args.scale) == float(native_scale)
        else f"native={native_scale}x -> final={args.scale:g}x | resample=full-frame Lanczos4"
    )
    print(f"Model   : {args.model} | {scale_text}", flush=True)
    print(
        f"Denoise : {profile_name} | BasicVSR++ NTIRE Track 1 | strength={profile['strength']:.2f} | "
        f"clip={profile['clip_length']} | tile=512(auto fallback) | dynamic GPUs="
        + ",".join(f"cuda:{gpu}" for gpu in gpu_ids),
        flush=True,
    )
    print(f"GPU     : SR={base._device_text(gpu_ids)} | FP16 + channels_last", flush=True)
    print(
        f"Mode    : dynamic full-frame | parallel_gpus={len(gpu_ids)} | IPC=shared-memory",
        flush=True,
    )
    print(flush=True)

    writer = workers = pump = progress = raw_reader = task_source = None
    preprocessors = []
    started = time.monotonic()
    sr_model_time = bvs_model_time = flush_time = audio_time = 0.0
    gpu_work = scheduler_idle = 0.0
    bvs_elapsed = decode_elapsed = 0.0
    bvs_clips = bvs_tiles = scene_cuts = 0
    selected_tile = 0
    clean = False

    pending: Dict[int, int] = {}
    next_output = 0
    restored_heap: list[tuple[int, np.ndarray]] = []
    sr_active: set[int] = set()
    bvs_active: dict[int, tuple[object, list[_RestoredTask]]] = {}
    bvs_eof = False
    next_restored_id = 0
    bvs_jobs = sr_jobs = 0

    try:
        writer = base._writer_type(
            temp_video,
            args.ffmpeg_bin,
            out_w,
            out_h,
            inference_rate_text,
            output_rate_text,
            args.video_codec,
            args.crf,
            args.preset,
            args.cq,
            args.nvenc_preset,
            args.encode_gpu,
        )

        t = time.monotonic()
        workers = base_pipeline.SharedWorkers(gpu_ids, config, input_shape, native_shape, dtype)
        sr_model_time = time.monotonic() - t

        raw_reader = base.RawVideoReader(
            input_path,
            args.ffmpeg_bin,
            in_w,
            in_h,
            inference_rate_text,
            start,
            duration,
            info.bit_depth,
        )

        from . import balanced_pipeline
        from .basicvsrpp import BasicVSRPPConfig, BasicVSRPPPreprocessor

        balanced_pipeline._BALANCED_GPU_IDS = tuple(gpu_ids)
        t = time.monotonic()
        primary = BasicVSRPPPreprocessor(
            BasicVSRPPConfig(
                gpu_id=gpu_ids[0],
                strength=float(profile["strength"]),
                clip_length=int(profile["clip_length"]),
                clip_overlap=2,
                tile_size=512,
                tile_pad=32,
                fp16=True,
                scene_threshold=0.30,
            ),
            checkpoint_dir=Path(__file__).resolve().parent / "weights",
        )
        task_source = balanced_pipeline.BalancedBasicVSRPPStreamReader(raw_reader, primary)
        preprocessors = list(task_source.preprocessors)
        raw_reader = None
        bvs_model_time = time.monotonic() - t

        if len(preprocessors) != len(gpu_ids):
            raise RuntimeError(
                f"Dynamic scheduler needs one BasicVSR++ instance per SR GPU; "
                f"got {len(preprocessors)} for {len(gpu_ids)} GPUs"
            )

        print(
            f"Pipeline: shared_memory={workers.memory_mib:.0f} MiB | "
            "dynamic per-GPU BVS/SR task scheduling",
            flush=True,
        )
        print(
            "Pipeline: no global BVS->SR barrier | at least one BVS producer is kept active "
            "while restoration work remains",
            flush=True,
        )
        print(flush=True)

        progress = tqdm(
            total=expected,
            desc="Real-ESRGAN",
            unit="frame",
            dynamic_ncols=True,
            mininterval=1.0,
            file=sys.stdout,
        )
        pump = base_pipeline.OutputPump(workers, writer, out_w, out_h, progress, started)

        overlap = int(primary.config.clip_overlap)
        clip_length = int(primary.config.clip_length)
        sr_trigger = max(1, clip_length - 2 * overlap)

        def schedule_bvs(worker_id: int) -> bool:
            nonlocal bvs_eof, next_restored_id, bvs_jobs
            if bvs_eof:
                return False
            preprocessor = preprocessors[worker_id]
            batch = max(1, int(getattr(preprocessor, "clip_batch", 1)))
            tasks: list[_RestoredTask] = []
            for _ in range(batch):
                raw_task = task_source._next_task()
                if raw_task is None:
                    bvs_eof = True
                    break
                frames, emit_start, emit_end = raw_task
                task = _RestoredTask(frames, emit_start, emit_end, next_restored_id)
                next_restored_id += task.emit_count
                tasks.append(task)
            if not tasks:
                return False
            future = task_source.executor.submit(_run_bvs_group, preprocessor, tasks)
            bvs_active[worker_id] = (future, tasks)
            bvs_jobs += len(tasks)
            return True

        def schedule_sr(worker_id: int) -> bool:
            nonlocal sr_jobs
            if not restored_heap:
                return False
            frame_id, frame = heapq.heappop(restored_heap)
            workers.submit(worker_id, frame_id, frame)
            sr_active.add(worker_id)
            sr_jobs += 1
            return True

        def handle_bvs_done(worker_id: int) -> None:
            future, tasks = bvs_active.pop(worker_id)
            enhanced_groups = future.result()
            if len(enhanced_groups) != len(tasks):
                raise RuntimeError("BasicVSR++ dynamic worker returned an unexpected clip count")
            for task, enhanced in zip(tasks, enhanced_groups):
                emitted = enhanced[task.emit_start : task.emit_end]
                if len(emitted) != task.emit_count:
                    raise RuntimeError("BasicVSR++ dynamic worker returned an unexpected frame count")
                for offset, frame in enumerate(emitted):
                    heapq.heappush(restored_heap, (task.output_start + offset, frame))
            preprocessor = preprocessors[worker_id]
            try:
                if hasattr(preprocessor, "release_unused_cache_if_tight"):
                    preprocessor.release_unused_cache_if_tight()
            except Exception:
                pass

        while True:
            pump.check()
            made_progress = False

            while True:
                try:
                    worker_id, frame_id, seconds = base_pipeline._result(workers.result(False))
                except queue.Empty:
                    break
                gpu_work += seconds
                sr_active.discard(worker_id)
                pending[frame_id] = worker_id
                made_progress = True

            for worker_id in list(bvs_active):
                future, _tasks = bvs_active[worker_id]
                if future.done():
                    handle_bvs_done(worker_id)
                    made_progress = True

            while next_output in pending:
                worker_id = pending.pop(next_output)
                pump.put(next_output, worker_id)
                next_output += 1
                made_progress = True

            for worker_id in range(len(gpu_ids)):
                if worker_id in sr_active or worker_id in bvs_active:
                    continue

                other_bvs_running = any(other != worker_id for other in bvs_active)
                if restored_heap and (
                    bvs_eof or (other_bvs_running and len(restored_heap) >= sr_trigger)
                ):
                    if schedule_sr(worker_id):
                        made_progress = True
                        continue

                if not bvs_eof and schedule_bvs(worker_id):
                    made_progress = True
                    continue

                if restored_heap and schedule_sr(worker_id):
                    made_progress = True

            if bvs_eof and not bvs_active and not restored_heap and not sr_active:
                break

            if not made_progress:
                idle_started = time.monotonic()
                time.sleep(0.01)
                scheduler_idle += time.monotonic() - idle_started

        if pending:
            raise RuntimeError(f"Pipeline ended with {len(pending)} out-of-order SR frame(s).")
        if next_output != next_restored_id:
            raise RuntimeError(
                f"Pipeline output mismatch: emitted={next_output}, restored={next_restored_id}"
            )

        pump.finish()
        decode_elapsed = float(task_source.decode_elapsed)
        scene_cuts = int(task_source.scene_cuts)
        bvs_elapsed = max(
            (float(getattr(item, "elapsed", 0.0)) for item in preprocessors),
            default=0.0,
        )
        bvs_clips = sum(int(getattr(item, "clips", 0)) for item in preprocessors)
        bvs_tiles = sum(int(getattr(item, "tiles", 0)) for item in preprocessors)
        selected_tile = min(
            (
                int(getattr(item, "tile_size", 0))
                for item in preprocessors
                if int(getattr(item, "tile_size", 0)) > 0
            ),
            default=0,
        )
        task_source.close()
        task_source = None
        t = time.monotonic()
        writer.close()
        flush_time = time.monotonic() - t
        writer = None
        clean = True
    finally:
        try:
            from . import balanced_pipeline

            balanced_pipeline._BALANCED_GPU_IDS = ()
        except Exception:
            pass
        if pump is not None:
            try:
                pump.stop()
            except Exception:
                pass
        if progress is not None:
            progress.close()
        if task_source is not None:
            try:
                task_source.close()
            except Exception:
                pass
        if raw_reader is not None:
            try:
                raw_reader.close()
            except Exception:
                pass
        if writer is not None:
            try:
                writer.close()
            except Exception:
                pass
        if workers is not None:
            workers.close()

    processed = pump.processed if pump is not None else 0
    if not clean or processed == 0:
        raise RuntimeError("No complete video was encoded.")

    actual_duration = processed / inference_fps
    t = time.monotonic()
    base.mux_audio(
        temp_video,
        input_path,
        output_path,
        args.ffmpeg_bin,
        start,
        actual_duration,
        info.has_audio,
        args.audio_codec,
        args.audio_bitrate,
    )
    audio_time = time.monotonic() - t
    if temp_video.exists():
        temp_video.unlink()

    elapsed = time.monotonic() - started
    size_mib = output_path.stat().st_size / 2**20
    bitrate = output_path.stat().st_size * 8 / max(actual_duration, 1e-6) / 1_000_000
    fps = processed / max(elapsed, 1e-6)
    gpu_avg = gpu_work / max(processed, 1)
    print("\n=== Completed ===", flush=True)
    print(
        f"Frames  : {processed} | {base.format_seconds(start)} -> "
        f"{base.format_seconds(start + actual_duration)} | duration={actual_duration:.3f}s",
        flush=True,
    )
    print(f"Speed   : {fps:.3f} frame/s | processing={elapsed:.1f}s", flush=True)
    print(
        f"Timing  : sr_model={sr_model_time:.1f}s | bvs_model={bvs_model_time:.1f}s | "
        f"decode={decode_elapsed:.1f}s | basicvsr={bvs_elapsed:.1f}s/{bvs_clips} clips | "
        f"scheduler_idle={scheduler_idle:.1f}s | sr_gpu_avg={gpu_avg:.3f}s/frame | "
        f"resize={pump.resize_seconds:.1f}s | write={pump.write_seconds:.1f}s | "
        f"flush={flush_time:.1f}s | audio={audio_time:.1f}s",
        flush=True,
    )
    print(
        f"BasicVSR: tile={selected_tile} | tiles={bvs_tiles} | scene_cuts={scene_cuts}",
        flush=True,
    )
    print(
        f"Scheduler: bvs_jobs={bvs_jobs} | sr_jobs={sr_jobs} | restored_frames={next_restored_id}",
        flush=True,
    )
    print(f"File    : {size_mib:.2f} MiB | {bitrate:.2f} Mb/s", flush=True)
    print(f"Output  : {output_path}", flush=True)
