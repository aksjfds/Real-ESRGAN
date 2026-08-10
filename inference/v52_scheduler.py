"""Process-isolated multi-GPU scheduler: BasicVSR++ -> RIFE -> Real-ESRGAN.

v6.1 keeps one spawned process per CUDA device for the entire run. BVS, RIFE
and SR are separate task types; the main process owns only decoding, temporal
task assembly, scheduling, ordering, progress and encoding.
"""
from __future__ import annotations
from collections import deque
from dataclasses import asdict, dataclass
import heapq
import multiprocessing as mp
from multiprocessing import shared_memory
import queue
import sys
import time
import traceback
from fractions import Fraction
from pathlib import Path
from typing import Deque, Dict, Sequence
import numpy as np
import torch
from tqdm import tqdm
from . import pipeline as base_pipeline
from . import runtime as base
_TASK_TIMEOUT = 300.0

def _gpu_worker(worker_id: int, gpu_id: int, task_queue, result_queue, sr_output_slot, config_dict: dict[str, object], bvs_config_dict: dict[str, object], rife_weights: str, input_name: str, frame_output_name: str, sr_output_name: str, input_slots: int, frame_output_slots: int, input_shape: tuple[int, int, int], sr_output_shape: tuple[int, int, int], dtype_str: str) -> None:
    input_shm = frame_output_shm = sr_output_shm = None
    bvs = rife = sr_model = None
    try:
        device = torch.device(f'cuda:{gpu_id}')
        dtype = np.dtype(dtype_str)
        with torch.cuda.device(device):
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cuda.matmul.allow_tf32 = True
            from . import v54_runtime
            from .basicvsrpp import BasicVSRPPConfig, BasicVSRPPPreprocessor
            v54_runtime.install_basicvsrpp_execution_optimizations()
            local_bvs_config = dict(bvs_config_dict)
            local_bvs_config['gpu_id'] = int(gpu_id)
            bvs_config = BasicVSRPPConfig(**local_bvs_config)
            bvs = BasicVSRPPPreprocessor(bvs_config, checkpoint_dir=Path(__file__).resolve().parent / 'weights')
            if rife_weights:
                from .rife425 import RIFE425Interpolator
                rife = RIFE425Interpolator(gpu_id, Path(rife_weights))
            config = base.WorkerConfig(**config_dict)
            sr_model, _native_scale = base.load_worker_model(config, device)
            input_shm = shared_memory.SharedMemory(name=input_name)
            frame_output_shm = shared_memory.SharedMemory(name=frame_output_name)
            sr_output_shm = shared_memory.SharedMemory(name=sr_output_name)
            input_view = np.ndarray((input_slots, *input_shape), dtype=dtype, buffer=input_shm.buf)
            frame_output_view = np.ndarray((frame_output_slots, *input_shape), dtype=dtype, buffer=frame_output_shm.buf)
            sr_output_view = np.ndarray(sr_output_shape, dtype=dtype, buffer=sr_output_shm.buf)
            sr_output_tensor = torch.from_numpy(sr_output_view)
            result_queue.put(('ready', worker_id, gpu_id))
            while True:
                task = task_queue.get()
                if task is None:
                    break
                kind = str(task[0])
                task_id = int(task[1])
                result_queue.put(('started', worker_id, task_id, kind))
                started = time.monotonic()
                if kind == 'bvs':
                    descriptors = task[2]
                    clips: list[list[np.ndarray]] = []
                    cursor = 0
                    for count, _emit_start, _emit_end in descriptors:
                        count = int(count)
                        clips.append([input_view[cursor + index] for index in range(count)])
                        cursor += count
                    before_tiles = int(getattr(bvs, 'tiles', 0))
                    if len(clips) > 1 and hasattr(bvs, 'enhance_clips'):
                        enhanced_groups = bvs.enhance_clips(clips)
                    else:
                        enhanced_groups = [bvs.enhance_clip(frames) for frames in clips]
                    output_cursor = 0
                    emitted_counts: list[int] = []
                    for descriptor, enhanced in zip(descriptors, enhanced_groups):
                        _count, emit_start, emit_end = descriptor
                        emitted = enhanced[int(emit_start):int(emit_end)]
                        emitted_count = len(emitted)
                        if output_cursor + emitted_count > frame_output_slots:
                            raise RuntimeError('Unified GPU worker frame-output buffer is too small for BVS result')
                        for frame in emitted:
                            np.copyto(frame_output_view[output_cursor], frame, casting='no')
                            output_cursor += 1
                        emitted_counts.append(emitted_count)
                    elapsed = time.monotonic() - started
                    result_queue.put(('result', worker_id, task_id, kind, elapsed, {'count': output_cursor, 'emitted_counts': tuple(emitted_counts), 'tile_size': int(getattr(bvs, 'tile_size', 0)), 'tiles': int(getattr(bvs, 'tiles', 0)) - before_tiles, 'clips': len(clips)}))
                    continue
                if kind == 'rife':
                    if rife is None:
                        raise RuntimeError('RIFE task submitted to a worker without RIFE')
                    timesteps = tuple((float(value) for value in task[2]))
                    if len(timesteps) > frame_output_slots:
                        raise RuntimeError('Unified GPU worker frame-output buffer is too small for RIFE result')
                    generated = rife.interpolate_many(input_view[0], input_view[1], timesteps)
                    for index, frame in enumerate(generated):
                        np.copyto(frame_output_view[index], frame, casting='no')
                    elapsed = time.monotonic() - started
                    result_queue.put(('result', worker_id, task_id, kind, elapsed, {'count': len(generated)}))
                    continue
                if kind == 'sr':
                    frame_id = int(task[2])
                    from . import v51_runtime as v51
                    if dtype == np.dtype(np.uint8):
                        result_cuda = v51._infer_cuda_u8_tensor(sr_model, input_view[0], device)
                        sr_output_slot.acquire()
                        sr_output_tensor.copy_(result_cuda, non_blocking=False)
                        del result_cuda
                    else:
                        result = base.infer_frame(sr_model, input_view[0], device)
                        sr_output_slot.acquire()
                        np.copyto(sr_output_view, result, casting='no')
                        del result
                    elapsed = time.monotonic() - started
                    result_queue.put(('result', worker_id, task_id, kind, elapsed, {'frame_id': frame_id}))
                    continue
                raise RuntimeError(f'Unknown unified GPU task kind: {kind!r}')
    except Exception as error:
        try:
            result_queue.put(('error', worker_id, repr(error), traceback.format_exc()))
        except Exception:
            pass
    finally:
        try:
            if rife is not None:
                rife.close()
        except Exception:
            pass
        try:
            if bvs is not None:
                bvs.close()
        except Exception:
            pass
        try:
            if sr_model is not None:
                del sr_model
        except Exception:
            pass
        for shm in (input_shm, frame_output_shm, sr_output_shm):
            if shm is not None:
                try:
                    shm.close()
                except Exception:
                    pass

class UnifiedGPUWorkers:
    """One spawned process and one task queue per CUDA device."""
    def __init__(self, gpu_ids: Sequence[int], config: base.WorkerConfig, bvs_config: dict[str, object], rife_weights: str, input_shape: tuple[int, int, int], sr_output_shape: tuple[int, int, int], dtype: np.dtype, input_slots: int, frame_output_slots: int) -> None:
        self.context = mp.get_context('spawn')
        self.gpu_ids = [int(value) for value in gpu_ids]
        self.count = len(self.gpu_ids)
        self.dtype = np.dtype(dtype)
        self.input_shape = tuple((int(x) for x in input_shape))
        self.sr_output_shape = tuple((int(x) for x in sr_output_shape))
        self.input_slots = int(input_slots)
        self.frame_output_slots = int(frame_output_slots)
        self.closed = False
        if self.count < 1:
            raise ValueError('UnifiedGPUWorkers requires at least one CUDA GPU')
        if self.input_slots < 2:
            raise ValueError('UnifiedGPUWorkers input_slots must be at least 2')
        if self.frame_output_slots < 1:
            raise ValueError('UnifiedGPUWorkers frame_output_slots must be positive')
        self.result_queue = self.context.Queue()
        self.task_queues = [self.context.Queue(maxsize=1) for _ in self.gpu_ids]
        self.sr_output_slots = [self.context.Semaphore(1) for _ in self.gpu_ids]
        self.input_shms: list[shared_memory.SharedMemory] = []
        self.frame_output_shms: list[shared_memory.SharedMemory] = []
        self.sr_output_shms: list[shared_memory.SharedMemory] = []
        self.input_views: list[np.ndarray] = []
        self.frame_output_views: list[np.ndarray] = []
        self.sr_output_views: list[np.ndarray] = []
        self.processes = []
        in_frame_bytes = int(np.prod(self.input_shape, dtype=np.int64)) * self.dtype.itemsize
        sr_bytes = int(np.prod(self.sr_output_shape, dtype=np.int64)) * self.dtype.itemsize
        try:
            for _ in self.gpu_ids:
                input_shm = shared_memory.SharedMemory(create=True, size=in_frame_bytes * self.input_slots)
                frame_output_shm = shared_memory.SharedMemory(create=True, size=in_frame_bytes * self.frame_output_slots)
                sr_output_shm = shared_memory.SharedMemory(create=True, size=sr_bytes)
                self.input_shms.append(input_shm)
                self.frame_output_shms.append(frame_output_shm)
                self.sr_output_shms.append(sr_output_shm)
                self.input_views.append(np.ndarray((self.input_slots, *self.input_shape), dtype=self.dtype, buffer=input_shm.buf))
                self.frame_output_views.append(np.ndarray((self.frame_output_slots, *self.input_shape), dtype=self.dtype, buffer=frame_output_shm.buf))
                self.sr_output_views.append(np.ndarray(self.sr_output_shape, dtype=self.dtype, buffer=sr_output_shm.buf))
            for worker_id, gpu_id in enumerate(self.gpu_ids):
                process = self.context.Process(target=_gpu_worker, args=(worker_id, gpu_id, self.task_queues[worker_id], self.result_queue, self.sr_output_slots[worker_id], asdict(config), dict(bvs_config), str(rife_weights or ''), self.input_shms[worker_id].name, self.frame_output_shms[worker_id].name, self.sr_output_shms[worker_id].name, self.input_slots, self.frame_output_slots, self.input_shape, self.sr_output_shape, self.dtype.str), daemon=True)
                process.start()
                self.processes.append(process)
            self._wait_ready()
        except Exception:
            self.close()
            raise
    @property
    def memory_mib(self) -> float:
        return sum((shm.size for shm in self.input_shms + self.frame_output_shms + self.sr_output_shms)) / 2 ** 20
    def _wait_ready(self) -> None:
        ready: set[int] = set()
        deadline = time.monotonic() + _TASK_TIMEOUT
        while len(ready) < self.count:
            left = deadline - time.monotonic()
            if left <= 0:
                raise TimeoutError('Timed out loading unified GPU workers')
            try:
                message = self.result_queue.get(timeout=left)
            except queue.Empty as error:
                raise TimeoutError('Timed out loading unified GPU workers') from error
            if message[0] == 'error':
                raise RuntimeError(f'GPU worker {message[1]} failed during startup: {message[2]}\n{message[3]}')
            if message[0] != 'ready':
                raise RuntimeError(f'Unexpected unified GPU startup message: {message[0]!r}')
            ready.add(int(message[1]))
    def _copy_input(self, worker_id: int, slot: int, frame: np.ndarray) -> None:
        if frame.shape != self.input_shape or np.dtype(frame.dtype) != self.dtype:
            raise ValueError(f'Unexpected worker input {frame.shape}/{frame.dtype}; expected {self.input_shape}/{self.dtype}')
        np.copyto(self.input_views[worker_id][slot], frame, casting='no')
    def submit_bvs(self, worker_id: int, task_id: int, groups: Sequence[tuple[Sequence[np.ndarray], int, int]]) -> None:
        descriptors: list[tuple[int, int, int]] = []
        cursor = 0
        for frames, emit_start, emit_end in groups:
            count = len(frames)
            if cursor + count > self.input_slots:
                raise RuntimeError('BVS task exceeds unified GPU input buffer')
            for frame in frames:
                self._copy_input(worker_id, cursor, frame)
                cursor += 1
            descriptors.append((count, int(emit_start), int(emit_end)))
        self.task_queues[worker_id].put(('bvs', int(task_id), tuple(descriptors)))
    def submit_rife(self, worker_id: int, task_id: int, frame0: np.ndarray, frame1: np.ndarray, timesteps: Sequence[float]) -> None:
        values = tuple((float(value) for value in timesteps))
        if len(values) > self.frame_output_slots:
            raise RuntimeError('RIFE task exceeds unified GPU frame-output buffer')
        self._copy_input(worker_id, 0, frame0)
        self._copy_input(worker_id, 1, frame1)
        self.task_queues[worker_id].put(('rife', int(task_id), values))
    def submit_sr(self, worker_id: int, task_id: int, frame_id: int, frame: np.ndarray) -> None:
        self._copy_input(worker_id, 0, frame)
        self.task_queues[worker_id].put(('sr', int(task_id), int(frame_id)))
    def result(self, block: bool=False):
        if block:
            try:
                return self.result_queue.get(timeout=_TASK_TIMEOUT)
            except queue.Empty as error:
                raise TimeoutError('Timed out waiting for unified GPU worker') from error
        return self.result_queue.get_nowait()
    def take_frames(self, worker_id: int, count: int) -> list[np.ndarray]:
        count = int(count)
        if count < 0 or count > self.frame_output_slots:
            raise ValueError(f'Invalid unified GPU frame-output count: {count}')
        return [np.array(self.frame_output_views[worker_id][index], copy=True, order='C') for index in range(count)]
    def output(self, worker_id: int) -> np.ndarray:
        return self.sr_output_views[worker_id]
    def release(self, worker_id: int) -> None:
        self.sr_output_slots[worker_id].release()
    def is_alive(self, worker_id: int) -> bool:
        return self.processes[worker_id].is_alive()
    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        for slot in self.sr_output_slots:
            try:
                slot.release()
            except Exception:
                pass
        for task_queue in self.task_queues:
            try:
                task_queue.put_nowait(None)
            except Exception:
                pass
        for process in self.processes:
            process.join(timeout=10)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        for task_queue in self.task_queues:
            try:
                task_queue.close()
            except Exception:
                pass
        try:
            self.result_queue.close()
        except Exception:
            pass
        for shm in self.input_shms + self.frame_output_shms + self.sr_output_shms:
            try:
                shm.close()
                shm.unlink()
            except FileNotFoundError:
                pass
            except Exception:
                pass

@dataclass
class _RestoredTask:
    frames: list[np.ndarray]
    emit_start: int
    emit_end: int
    output_start: int
    @property
    def emit_count(self) -> int:
        return max(0, int(self.emit_end) - int(self.emit_start))

@dataclass
class _RifeJob:
    frame0: np.ndarray
    frame1: np.ndarray
    targets: tuple[int, ...]
    timesteps: tuple[float, ...]

@dataclass
class _ActiveTask:
    task_id: int
    kind: str
    submitted_at: float
    started_at: float | None
    meta: object

class _ClipSource:
    """CPU-only scene-aware clip assembler."""
    def __init__(self, reader, clip_length: int, overlap: int, scene_threshold: float) -> None:
        self.reader = reader
        self.clip_length = int(clip_length)
        self.overlap = int(overlap)
        self.scene_threshold = float(scene_threshold)
        self.buffer: list[np.ndarray] = []
        self.pending: np.ndarray | None = None
        self.eof = False
        self.segment_end = False
        self.first_chunk = True
        self.decode_elapsed = 0.0
        self.scene_cuts = 0
        self.closed = False
    def _read_source(self):
        started = time.monotonic()
        frame = self.reader.read()
        self.decode_elapsed += time.monotonic() - started
        return frame
    def next_task(self):
        from .basicvsrpp import scene_difference
        while True:
            if self.segment_end or self.eof:
                if self.buffer:
                    frames = list(self.buffer)
                    emit_start = 0 if self.first_chunk else self.overlap
                    emit_end = len(frames)
                    self.buffer = []
                    self.first_chunk = True
                    self.segment_end = False
                    if self.pending is not None:
                        self.buffer.append(self.pending)
                        self.pending = None
                    return (frames, emit_start, emit_end)
                self.segment_end = False
                if self.pending is not None:
                    self.buffer.append(self.pending)
                    self.pending = None
                    continue
                if self.eof:
                    return None
            frame = self._read_source()
            if frame is None:
                self.eof = True
                continue
            if self.buffer and self.scene_threshold > 0:
                if scene_difference(self.buffer[-1], frame) >= self.scene_threshold:
                    self.pending = frame
                    self.segment_end = True
                    self.scene_cuts += 1
                    continue
            self.buffer.append(frame)
            if len(self.buffer) == self.clip_length:
                frames = list(self.buffer)
                if self.first_chunk:
                    emit_start = 0
                    emit_end = self.clip_length - self.overlap
                    self.first_chunk = False
                else:
                    emit_start = self.overlap
                    emit_end = self.clip_length - self.overlap
                retain = 2 * self.overlap
                self.buffer = self.buffer[-retain:] if retain else []
                return (frames, emit_start, emit_end)
    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.reader.close()

def _ceil_fraction(value: Fraction) -> int:
    return -(-value.numerator // value.denominator)

def _rife_targets_for_source(source_id: int, source_rate: Fraction, output_rate: Fraction, expected_output: int) -> tuple[list[int], list[float]]:
    ratio = output_rate / source_rate
    start = max(0, _ceil_fraction(Fraction(source_id) * ratio))
    end = min(expected_output, _ceil_fraction(Fraction(source_id + 1) * ratio))
    targets: list[int] = []
    timesteps: list[float] = []
    for target in range(start, end):
        alpha = Fraction(target, 1) / ratio - source_id
        targets.append(target)
        timesteps.append(float(max(Fraction(0), min(Fraction(1), alpha))))
    return (targets, timesteps)

def process_video(args) -> None:
    requested_gpu_ids = base_pipeline.base.parse_gpu_ids(args.gpu_ids)
    if len(requested_gpu_ids) < 2:
        raise RuntimeError('v6.1 unified GPU scheduling requires at least two CUDA GPUs')
    if any((gpu is None for gpu in requested_gpu_ids)):
        raise RuntimeError('v6.1 unified GPU scheduling requires CUDA GPUs')
    if base_pipeline.base._require_encoder is None or base_pipeline.base._writer_type is None:
        raise RuntimeError('Encoding backend is not configured. Run through root inference.py.')
    base = base_pipeline.base
    base.require_binary(args.ffmpeg_bin)
    base.require_binary(args.ffprobe_bin)
    base._require_encoder(args.ffmpeg_bin, args.video_codec)
    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f'Input video not found: {input_path}')
    if input_path == output_path:
        raise ValueError('Input and output paths must be different.')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_video = output_path.with_name(output_path.stem + '.video_only.tmp.mp4')
    info = base.probe_video(input_path, args.ffprobe_bin)
    source_rate = Fraction(info.fps_num, info.fps_den)
    output_rate = Fraction(str(float(args.rife_fps))).limit_denominator(100000)
    if output_rate < source_rate:
        raise ValueError(f'--rife-fps must be >= source FPS; got {float(output_rate):g} < {float(source_rate):g}')
    rife_enabled = output_rate > source_rate
    source_fps = float(source_rate)
    output_fps = float(output_rate)
    source_rate_text = f'{source_rate.numerator}/{source_rate.denominator}'
    output_rate_text = f'{output_rate.numerator}/{output_rate.denominator}'
    in_w, in_h = (info.width, info.height)
    out_w, out_h = (int(round(in_w * args.scale)), int(round(in_h * args.scale)))
    if out_w % 2 or out_h % 2:
        raise ValueError(f'4:2:0 output needs even dimensions, got {out_w}x{out_h}.')
    start, duration, expected_source = base.resolve_range(info, args.start_time, args.test_seconds, source_rate)
    expected_output = max(1, int(round(duration * output_fps)))
    end = start + duration
    gpu_ids = [int(gpu) for gpu in requested_gpu_ids]
    strength = float(args.bvs_strength)
    clip_length = int(args.bvs_clip_length)
    tile_size = int(args.bvs_tile_size)
    batch_size = int(args.bvs_batch_size)
    overlap = 2
    scene_threshold = 0.3
    native_scale = base._model_native_scale(args.model)
    config = base.WorkerConfig(args.model, base.resolve_model_paths(args))
    dtype = np.dtype('<u2') if info.bit_depth == 10 else np.dtype(np.uint8)
    input_shape = (in_h, in_w, 3)
    native_shape = (in_h * native_scale, in_w * native_scale, 3)
    from .checkpoint_parts import resolve_checkpoint
    checkpoint_path = resolve_checkpoint(Path(__file__).resolve().parent / 'weights')
    rife_weights = ''
    if rife_enabled:
        from .rife425 import resolve_rife425_weights
        rife_weights = str(resolve_rife425_weights())
    max_targets_per_interval = max(1, _ceil_fraction(output_rate / source_rate) + 1)
    input_slots = max(2, clip_length * batch_size)
    frame_output_slots = max(clip_length * batch_size, max_targets_per_interval)
    mode = 'test' if args.test_seconds > 0 else 'full/selected range'
    print('=== Real-ESRGAN ===', flush=True)
    print(f'Input   : {input_path.name} | {in_w}x{in_h} | {info.fps:.3f} fps | {info.bit_depth}-bit ({info.pix_fmt})', flush=True)
    print(f'Output  : {out_w}x{out_h} | {output_fps:.3f} fps | {info.bit_depth}-bit ({base._output_pixel_format(args.video_codec, info.bit_depth)}) | {args.video_codec}', flush=True)
    print(f'Range   : {mode} | {base.format_seconds(start)} -> {base.format_seconds(end)} | {duration:.3f}s | {expected_source} source / {expected_output} output frames', flush=True)
    scale_text = f'native={native_scale}x' if float(args.scale) == float(native_scale) else f'native={native_scale}x -> final={args.scale:g}x | resample=full-frame Lanczos4'
    print(f'Model   : {args.model} | {scale_text}', flush=True)
    print(f'Denoise : BasicVSR++ NTIRE Track 1 | strength={strength:.2f} | clip={clip_length} | tile={tile_size}(OOM fallback) | batch={batch_size}', flush=True)
    if rife_enabled:
        print(f'Interp  : Practical-RIFE 4.25 | {source_fps:.3f} -> {output_fps:.3f} fps | arbitrary timestep | scene-cut/duplicate guard | FP16', flush=True)
    else:
        print('Interp  : target FPS equals source FPS; RIFE inference bypassed', flush=True)
    print('GPU     : one spawned process per device | each process owns BVS + RIFE + SR', flush=True)
    print('Mode    : independent BVS / RIFE / SR tasks | no shared CUDA ThreadPoolExecutor', flush=True)
    print(flush=True)
    writer = workers = pump = progress = raw_reader = clip_source = None
    started = time.monotonic()
    worker_model_time = flush_time = audio_time = 0.0
    scheduler_idle = 0.0
    bvs_seconds = rife_seconds = sr_seconds = 0.0
    bvs_clips = bvs_tiles = rife_frames = 0
    selected_tiles: list[int] = []
    bvs_jobs = rife_jobs_count = sr_jobs = 0
    clean = False
    pending: Dict[int, int] = {}
    next_output = 0
    next_source_id = 0
    next_interval = 0
    next_task_id = 0
    bvs_eof = False
    final_interval_flushed = False
    generated_targets: set[int] = set()
    restored_sources: dict[int, np.ndarray] = {}
    restored_heap: list[tuple[int, np.ndarray]] = []
    rife_queue: Deque[_RifeJob] = deque()
    active: list[_ActiveTask | None] = [None for _ in gpu_ids]
    last_status = time.monotonic()
    last_status_output = 0
    timeout_by_kind = {'bvs': 300.0, 'rife': 180.0, 'sr': 120.0}
    def next_id() -> int:
        nonlocal next_task_id
        value = next_task_id
        next_task_id += 1
        return value
    def add_target(frame_id: int, frame: np.ndarray) -> None:
        frame_id = int(frame_id)
        if frame_id in generated_targets:
            raise RuntimeError(f'Timeline produced duplicate target frame {frame_id}')
        generated_targets.add(frame_id)
        heapq.heappush(restored_heap, (frame_id, frame))
    def promote_intervals(final: bool=False) -> bool:
        nonlocal next_interval, final_interval_flushed
        made = False
        from .basicvsrpp import scene_difference
        while next_interval in restored_sources and next_interval + 1 in restored_sources:
            current = restored_sources.pop(next_interval)
            nxt = restored_sources[next_interval + 1]
            targets, timesteps = _rife_targets_for_source(next_interval, source_rate, output_rate, expected_output)
            model_targets: list[int] = []
            model_times: list[float] = []
            for target, alpha in zip(targets, timesteps):
                if alpha <= 1e-08:
                    add_target(target, current)
                else:
                    model_targets.append(target)
                    model_times.append(alpha)
            if model_targets:
                difference = scene_difference(current, nxt)
                if not rife_enabled or difference <= 0.002 or difference >= 0.3:
                    for target in model_targets:
                        add_target(target, current)
                else:
                    rife_queue.append(_RifeJob(current, nxt, tuple(model_targets), tuple(model_times)))
            next_interval += 1
            made = True
        if final and (not final_interval_flushed) and (next_source_id > 0) and (next_interval == next_source_id - 1) and (next_interval in restored_sources):
            current = restored_sources.pop(next_interval)
            targets, _timesteps = _rife_targets_for_source(next_interval, source_rate, output_rate, expected_output)
            for target in targets:
                add_target(target, current)
            next_interval += 1
            final_interval_flushed = True
            made = True
        return made
    try:
        writer = base._writer_type(temp_video, args.ffmpeg_bin, out_w, out_h, output_rate_text, output_rate_text, args.video_codec, args.crf, args.preset, args.cq, args.nvenc_preset, args.encode_gpu)
        raw_reader = base.RawVideoReader(input_path, args.ffmpeg_bin, in_w, in_h, source_rate_text, start, duration, info.bit_depth)
        clip_source = _ClipSource(raw_reader, clip_length=clip_length, overlap=overlap, scene_threshold=scene_threshold)
        raw_reader = None
        bvs_config = {'gpu_id': 0, 'strength': strength, 'clip_length': clip_length, 'clip_overlap': overlap, 'tile_size': tile_size, 'tile_pad': 32, 'fp16': True, 'scene_threshold': scene_threshold, 'model_path': str(checkpoint_path)}
        t = time.monotonic()
        workers = UnifiedGPUWorkers(gpu_ids=gpu_ids, config=config, bvs_config=bvs_config, rife_weights=rife_weights, input_shape=input_shape, sr_output_shape=native_shape, dtype=dtype, input_slots=input_slots, frame_output_slots=frame_output_slots)
        worker_model_time = time.monotonic() - t
        print(f'Pipeline: unified GPU workers ready | shared_memory={workers.memory_mib:.0f} MiB', flush=True)
        print('Pipeline: each CUDA process has one task queue and permanent device affinity', flush=True)
        print(flush=True)
        progress = tqdm(total=expected_output, desc='Real-ESRGAN', unit='frame', dynamic_ncols=True, mininterval=1.0, file=sys.stdout)
        pump = base_pipeline.OutputPump(workers, writer, out_w, out_h, progress, started)
        def schedule_bvs(worker_id: int) -> bool:
            nonlocal bvs_eof, next_source_id, bvs_jobs
            if bvs_eof:
                return False
            groups: list[tuple[Sequence[np.ndarray], int, int]] = []
            assignments: list[tuple[int, int]] = []
            for _ in range(batch_size):
                raw_task = clip_source.next_task()
                if raw_task is None:
                    bvs_eof = True
                    break
                frames, emit_start, emit_end = raw_task
                emit_count = max(0, int(emit_end) - int(emit_start))
                groups.append((frames, emit_start, emit_end))
                assignments.append((next_source_id, emit_count))
                next_source_id += emit_count
            if not groups:
                return False
            task_id = next_id()
            workers.submit_bvs(worker_id, task_id, groups)
            active[worker_id] = _ActiveTask(task_id, 'bvs', time.monotonic(), None, tuple(assignments))
            bvs_jobs += len(groups)
            return True
        def schedule_rife(worker_id: int) -> bool:
            nonlocal rife_jobs_count
            if not rife_queue:
                return False
            job = rife_queue.popleft()
            task_id = next_id()
            workers.submit_rife(worker_id, task_id, job.frame0, job.frame1, job.timesteps)
            active[worker_id] = _ActiveTask(task_id, 'rife', time.monotonic(), None, job.targets)
            rife_jobs_count += 1
            return True
        def schedule_sr(worker_id: int) -> bool:
            nonlocal sr_jobs
            if not restored_heap:
                return False
            frame_id, frame = heapq.heappop(restored_heap)
            task_id = next_id()
            workers.submit_sr(worker_id, task_id, frame_id, frame)
            active[worker_id] = _ActiveTask(task_id, 'sr', time.monotonic(), None, frame_id)
            sr_jobs += 1
            return True
        while True:
            pump.check()
            made_progress = False
            while True:
                try:
                    message = workers.result(False)
                except queue.Empty:
                    break
                tag = message[0]
                if tag == 'error':
                    raise RuntimeError(f'GPU worker {message[1]} failed: {message[2]}\n{message[3]}')
                if tag == 'started':
                    worker_id = int(message[1])
                    task_id = int(message[2])
                    kind = str(message[3])
                    state = active[worker_id]
                    if state is None or state.task_id != task_id or state.kind != kind:
                        raise RuntimeError(f'Unexpected worker start: gpu={worker_id} task={task_id}/{kind}')
                    state.started_at = time.monotonic()
                    made_progress = True
                    continue
                if tag != 'result':
                    raise RuntimeError(f'Unexpected unified GPU message: {tag!r}')
                worker_id = int(message[1])
                task_id = int(message[2])
                kind = str(message[3])
                seconds = float(message[4])
                payload = message[5]
                state = active[worker_id]
                if state is None or state.task_id != task_id or state.kind != kind:
                    raise RuntimeError(f'Unexpected worker result: gpu={worker_id} task={task_id}/{kind}')
                if kind == 'bvs':
                    count = int(payload['count'])
                    frames = workers.take_frames(worker_id, count)
                    assignments = state.meta
                    emitted_counts = tuple((int(x) for x in payload['emitted_counts']))
                    if len(assignments) != len(emitted_counts):
                        raise RuntimeError('BVS worker returned mismatched assignment count')
                    cursor = 0
                    for (output_start, expected_count), emitted_count in zip(assignments, emitted_counts):
                        if emitted_count != expected_count:
                            raise RuntimeError(f'BVS worker emitted {emitted_count} frames, expected {expected_count}')
                        for offset in range(emitted_count):
                            restored_sources[int(output_start) + offset] = frames[cursor]
                            cursor += 1
                    if cursor != count:
                        raise RuntimeError('BVS worker frame-output accounting mismatch')
                    bvs_seconds += seconds
                    bvs_clips += int(payload.get('clips', len(assignments)))
                    bvs_tiles += int(payload.get('tiles', 0))
                    tile_value = int(payload.get('tile_size', 0))
                    if tile_value > 0:
                        selected_tiles.append(tile_value)
                elif kind == 'rife':
                    targets = tuple((int(value) for value in state.meta))
                    count = int(payload['count'])
                    frames = workers.take_frames(worker_id, count)
                    if count != len(targets):
                        raise RuntimeError(f'RIFE worker returned {count} frames for {len(targets)} targets')
                    for target, frame in zip(targets, frames):
                        add_target(target, frame)
                    rife_seconds += seconds
                    rife_frames += count
                elif kind == 'sr':
                    frame_id = int(payload['frame_id'])
                    if frame_id != int(state.meta):
                        raise RuntimeError('SR worker returned an unexpected frame id')
                    pending[frame_id] = worker_id
                    sr_seconds += seconds
                active[worker_id] = None
                made_progress = True
            if promote_intervals(final=bvs_eof and (not any((state is not None and state.kind == 'bvs' for state in active)))):
                made_progress = True
            while next_output in pending:
                worker_id = pending.pop(next_output)
                pump.put(next_output, worker_id)
                next_output += 1
                made_progress = True
            for worker_id, state in enumerate(active):
                if state is not None:
                    continue
                bvs_running = any((item is not None and item.kind == 'bvs' for item in active))
                if not bvs_eof and (not bvs_running):
                    if schedule_bvs(worker_id):
                        made_progress = True
                        continue
                if rife_queue and schedule_rife(worker_id):
                    made_progress = True
                    continue
                if restored_heap and schedule_sr(worker_id):
                    made_progress = True
                    continue
                if not bvs_eof and schedule_bvs(worker_id):
                    made_progress = True
                    continue
            now = time.monotonic()
            for worker_id, state in enumerate(active):
                if not workers.is_alive(worker_id):
                    raise RuntimeError(f'cuda:{gpu_ids[worker_id]} worker process exited unexpectedly')
                if state is None:
                    continue
                reference = state.started_at or state.submitted_at
                limit = timeout_by_kind[state.kind]
                if now - reference > limit:
                    phase = 'running' if state.started_at is not None else 'queued'
                    raise TimeoutError(f'cuda:{gpu_ids[worker_id]} {state.kind.upper()} task {state.task_id} stalled while {phase} for {now - reference:.1f}s')
            if now - last_status >= 30.0:
                status = []
                for worker_id, state in enumerate(active):
                    if state is None:
                        text = 'IDLE'
                    else:
                        reference = state.started_at or state.submitted_at
                        phase = '' if state.started_at is not None else '/queued'
                        text = f'{state.kind.upper()}#{state.task_id}{phase} {now - reference:.1f}s'
                    status.append(f'cuda:{gpu_ids[worker_id]}={text}')
                print(f'[gpu-status] output={next_output}/{expected_output} | ' + ' | '.join(status), flush=True)
                last_status = now
                last_status_output = next_output
            all_idle = all((state is None for state in active))
            no_generation_work = bvs_eof and final_interval_flushed and (not restored_sources) and (not rife_queue) and (not restored_heap) and all_idle
            if no_generation_work:
                break
            if not made_progress:
                idle_started = time.monotonic()
                time.sleep(0.01)
                scheduler_idle += time.monotonic() - idle_started
        if pending:
            raise RuntimeError(f'Pipeline ended with {len(pending)} out-of-order SR frame(s); next expected frame is {next_output}')
        if len(generated_targets) != expected_output:
            missing = sorted(set(range(expected_output)) - generated_targets)[:8]
            raise RuntimeError(f'Timeline generated {len(generated_targets)}/{expected_output} targets; missing={missing}')
        if next_output != expected_output:
            raise RuntimeError(f'Pipeline output mismatch: emitted={next_output}, expected={expected_output}')
        pump.finish()
        t = time.monotonic()
        writer.close()
        flush_time = time.monotonic() - t
        writer = None
        clean = True
    finally:
        if pump is not None:
            try:
                pump.stop()
            except Exception:
                pass
        if progress is not None:
            progress.close()
        if clip_source is not None:
            try:
                clip_source.close()
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
        raise RuntimeError('No complete video was encoded.')
    actual_duration = processed / output_fps
    t = time.monotonic()
    base.mux_audio(temp_video, input_path, output_path, args.ffmpeg_bin, start, actual_duration, info.has_audio, args.audio_codec, args.audio_bitrate)
    audio_time = time.monotonic() - t
    if temp_video.exists():
        temp_video.unlink()
    elapsed = time.monotonic() - started
    size_mib = output_path.stat().st_size / 2 ** 20
    bitrate = output_path.stat().st_size * 8 / max(actual_duration, 1e-06) / 1000000
    fps = processed / max(elapsed, 1e-06)
    selected_tile = min(selected_tiles, default=tile_size)
    decode_elapsed = float(clip_source.decode_elapsed) if clip_source is not None else 0.0
    scene_cuts = int(clip_source.scene_cuts) if clip_source is not None else 0
    print('\n=== Completed ===', flush=True)
    print(f'Frames  : {processed} | {base.format_seconds(start)} -> {base.format_seconds(start + actual_duration)} | duration={actual_duration:.3f}s', flush=True)
    print(f'Speed   : {fps:.3f} frame/s | processing={elapsed:.1f}s', flush=True)
    print(f'Timing  : gpu_models={worker_model_time:.1f}s | decode={decode_elapsed:.1f}s | basicvsr={bvs_seconds:.1f}s/{bvs_clips} clips | rife={rife_seconds:.1f}s/{rife_frames} generated | sr={sr_seconds:.1f}s/{sr_jobs} frames | scheduler_idle={scheduler_idle:.1f}s | resize={pump.resize_seconds:.1f}s | write={pump.write_seconds:.1f}s | flush={flush_time:.1f}s | audio={audio_time:.1f}s', flush=True)
    print(f'BasicVSR: tile={selected_tile} | clip={clip_length} | batch={batch_size} | strength={strength:.2f} | tiles={bvs_tiles} | scene_cuts={scene_cuts}', flush=True)
    print(f'Scheduler: bvs_jobs={bvs_jobs} | rife_jobs={rife_jobs_count} | sr_jobs={sr_jobs} | source_frames={next_source_id} | output_frames={processed}', flush=True)
    print(f'File    : {size_mib:.2f} MiB | {bitrate:.2f} Mb/s', flush=True)
    print(f'Output  : {output_path}', flush=True)
