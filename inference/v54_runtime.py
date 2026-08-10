"""v5.4-v5.5 BasicVSR++ execution-only optimizations.

These patches preserve the BasicVSR++ model, weights, clip policy, tiling,
scene-cut behavior, and arithmetic. v5.4 reuses invariant warp grids and avoids
temporary zero tensors that were immediately overwritten. v5.5 keeps 8-bit
clips compact across H2D and prepares future clip tasks on a bounded CPU thread.
"""
from __future__ import annotations

import queue
import threading
import time
import traceback
from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np
import torch
from torch.nn import functional as F

from . import basicvsrpp as bvsr


_INSTALLED = False
_FLOW_GRID_CACHE: Dict[Tuple[torch.device, torch.dtype, int, int], torch.Tensor] = {}


def _flow_warp_cached(
    x: torch.Tensor,
    flow: torch.Tensor,
    interpolation: str = "bilinear",
    padding_mode: str = "zeros",
    align_corners: bool = True,
) -> torch.Tensor:
    """Match BasicVSR++ flow_warp while reusing its invariant pixel grid."""
    if x.shape[-2:] != flow.shape[1:3]:
        raise ValueError(f"flow size {flow.shape[1:3]} does not match feature size {x.shape[-2:]}")
    n, _c, h, w = x.shape
    key = (x.device, x.dtype, int(h), int(w))
    base_grid = _FLOW_GRID_CACHE.get(key)
    if base_grid is None:
        grid_y, grid_x = torch.meshgrid(
            torch.arange(h, device=x.device, dtype=x.dtype),
            torch.arange(w, device=x.device, dtype=x.dtype),
            indexing="ij",
        )
        base_grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0)
        _FLOW_GRID_CACHE[key] = base_grid

    grid = base_grid.expand(n, -1, -1, -1) + flow.to(dtype=x.dtype)
    gx = 2.0 * grid[..., 0] / (w - 1) - 1.0 if w > 1 else torch.zeros_like(grid[..., 0])
    gy = 2.0 * grid[..., 1] / (h - 1) - 1.0 if h > 1 else torch.zeros_like(grid[..., 1])
    normalized = torch.stack((gx, gy), dim=-1)
    return F.grid_sample(
        x, normalized, mode=interpolation, padding_mode=padding_mode, align_corners=align_corners
    )


def _propagate_v54(
    self,
    feats: dict[str, list[torch.Tensor]],
    flows: torch.Tensor,
    module_name: str,
) -> dict[str, list[torch.Tensor]]:
    """Match BasicVSR++ propagation without zero tensors that get overwritten."""
    n, t, _c, h, w = flows.shape
    frame_idx = list(range(t + 1))
    flow_idx = list(range(-1, t))
    mapping_idx = list(range(len(feats["spatial"])))
    mapping_idx += mapping_idx[::-1]
    if "backward" in module_name:
        frame_idx = frame_idx[::-1]
        flow_idx = frame_idx
    feat_prop = flows.new_zeros(n, self.mid_channels, h, w)
    for i, idx in enumerate(frame_idx):
        feat_current = feats["spatial"][mapping_idx[idx]]
        if i > 0:
            flow_n1 = flows[:, flow_idx[i]]
            cond_n1 = bvsr.flow_warp(feat_prop, flow_n1.permute(0, 2, 3, 1))
            if i > 1:
                feat_n2 = feats[module_name][-2]
                flow_n2 = flows[:, flow_idx[i - 1]]
                flow_n2 = flow_n1 + bvsr.flow_warp(
                    flow_n2, flow_n1.permute(0, 2, 3, 1)
                )
                cond_n2 = bvsr.flow_warp(feat_n2, flow_n2.permute(0, 2, 3, 1))
            else:
                feat_n2 = torch.zeros_like(feat_prop)
                flow_n2 = torch.zeros_like(flow_n1)
                cond_n2 = torch.zeros_like(cond_n1)
            cond = torch.cat((cond_n1, feat_current, cond_n2), dim=1)
            feat_prop = self.deform_align[module_name](
                torch.cat((feat_prop, feat_n2), dim=1), cond, flow_n1, flow_n2
            )
        feat = [feat_current]
        feat.extend(feats[key][idx] for key in feats if key not in ("spatial", module_name))
        feat.append(feat_prop)
        feat_prop = feat_prop + self.backbone[module_name](torch.cat(feat, dim=1))
        feats[module_name].append(feat_prop)
    if "backward" in module_name:
        feats[module_name].reverse()
    return feats


def _enhance_u8_with_tile_v55(preprocessor, clip_cpu: torch.Tensor, tile_size: int) -> np.ndarray:
    """Transfer uint8 clips compactly, then normalize and restore entirely on CUDA."""
    from . import v51_runtime as v51

    device = preprocessor.device
    torch.cuda.set_device(device)

    timing = None
    start_event = end_event = None
    wall_started = 0.0
    try:
        from . import gpu_timing

        if getattr(gpu_timing, "_INSTALLED", False):
            timing = gpu_timing
            wall_started = time.monotonic()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
    except Exception:
        timing = None
        start_event = end_event = None

    if clip_cpu.dtype == torch.uint8:
        # Keep PCIe payload at one byte/channel. The float conversion and /255
        # happen after the compact uint8 tensor reaches the target GPU.
        original_u8 = clip_cpu.to(device=device, non_blocking=True)
        original = original_u8.to(dtype=torch.float32)
        original.div_(255.0)
        del original_u8
    else:
        # Compatibility path for probes/callers that still provide float clips.
        original = clip_cpu.to(device=device, dtype=torch.float32, non_blocking=True)

    _n, _t, _c, height, width = original.shape
    pad = int(preprocessor.config.tile_pad)
    flat = original.reshape(-1, 3, height, width)
    mode = "reflect" if min(height, width) > pad and min(height, width) > 1 else "replicate"
    padded_flat = F.pad(flat, (pad, pad, pad, pad), mode=mode) if pad else flat
    padded = padded_flat.view(
        original.shape[0], original.shape[1], 3, height + 2 * pad, width + 2 * pad
    )
    restored = torch.empty_like(original, dtype=torch.float32, device=device)
    tile_count = 0
    for y0 in range(0, height, tile_size):
        y1 = min(y0 + tile_size, height)
        for x0 in range(0, width, tile_size):
            x1 = min(x0 + tile_size, width)
            patch = padded[..., y0 : y1 + 2 * pad, x0 : x1 + 2 * pad]
            enhanced = v51._run_model_device(preprocessor, patch)
            restored[..., y0:y1, x0:x1] = enhanced[
                ..., pad : pad + (y1 - y0), pad : pad + (x1 - x0)
            ]
            tile_count += 1
    preprocessor.tiles += tile_count

    strength = float(preprocessor.config.strength)
    mixed = restored if strength >= 1.0 else original + strength * (restored - original)
    quantized = mixed.clamp_(0.0, 1.0).mul_(255.0).round_().to(torch.uint8)

    if start_event is not None and end_event is not None:
        end_event.record()
    result = quantized.permute(0, 1, 3, 4, 2).contiguous().cpu().numpy()
    if timing is not None and start_event is not None and end_event is not None:
        gpu_seconds = start_event.elapsed_time(end_event) / 1000.0
        timing._record(
            "bvs",
            int(preprocessor.config.gpu_id),
            gpu_seconds,
            wall_started,
            time.monotonic(),
        )
    return result


def _enhance_u8_v55(preprocessor, clip_cpu: torch.Tensor) -> np.ndarray:
    requested = int(preprocessor.tile_size)
    candidates = [requested]
    for fallback in (384, 320, 256):
        if fallback < requested and fallback not in candidates:
            candidates.append(fallback)
    last_error = None
    for tile_size in candidates:
        try:
            output = _enhance_u8_with_tile_v55(preprocessor, clip_cpu, tile_size)
            if tile_size != preprocessor.tile_size:
                preprocessor.tile_size = tile_size
                print(f"[basicvsrpp] VRAM fallback locked to tile={tile_size}", flush=True)
            return output
        except torch.cuda.OutOfMemoryError as error:
            last_error = error
            torch.cuda.empty_cache()
            print(f"[basicvsrpp] tile={tile_size} OOM; retrying smaller tile", flush=True)
    raise RuntimeError("BasicVSR++ ran out of GPU memory even at tile=256") from last_error


def _enhance_clip_v55(self, frames: Sequence[np.ndarray]) -> list[np.ndarray]:
    if not frames:
        return []
    first_shape = frames[0].shape
    first_dtype = np.dtype(frames[0].dtype)
    if any(frame.shape != first_shape or np.dtype(frame.dtype) != first_dtype for frame in frames):
        raise ValueError("All BasicVSR++ clip frames must have identical shape/dtype")
    if len(frames) == 1:
        return [np.ascontiguousarray(frame) for frame in frames]
    if first_dtype != np.dtype(np.uint8):
        return self._v51_original_enhance_clip(frames)

    packed = np.stack(frames, axis=0)
    clip = torch.from_numpy(packed).permute(0, 3, 1, 2).unsqueeze(0)
    started = time.monotonic()
    enhanced_u8 = _enhance_u8_v55(self, clip)[0]
    self.elapsed += time.monotonic() - started
    self.clips += 1
    return [np.ascontiguousarray(frame) for frame in enhanced_u8]


def _enhance_clips_v55(self, clips: Sequence[Sequence[np.ndarray]]) -> list[list[np.ndarray]]:
    if not clips:
        return []
    if len(clips) == 1:
        return [_enhance_clip_v55(self, clips[0])]

    lengths = {len(frames) for frames in clips}
    if len(lengths) != 1 or next(iter(lengths)) < 2:
        return [_enhance_clip_v55(self, frames) for frames in clips]
    first = clips[0][0]
    first_shape = first.shape
    first_dtype = np.dtype(first.dtype)
    for frames in clips:
        if any(frame.shape != first_shape or np.dtype(frame.dtype) != first_dtype for frame in frames):
            return [_enhance_clip_v55(self, item) for item in clips]
    if first_dtype != np.dtype(np.uint8):
        return self._v51_original_enhance_clips(clips)

    packed = np.stack([np.stack(frames, axis=0) for frames in clips], axis=0)
    tensor = torch.from_numpy(packed).permute(0, 1, 4, 2, 3)
    started = time.monotonic()
    enhanced_u8 = _enhance_u8_v55(self, tensor)
    self.elapsed += time.monotonic() - started
    self.clips += len(clips)
    return [
        [np.ascontiguousarray(frame) for frame in group]
        for group in enhanced_u8
    ]


def _probe_full_frame_v55(self, tile_size: int, clip_length: int, clip_batch: int):
    """Autotune the same compact uint8 H2D path used by real 8-bit clips."""
    self._quality_guard(tile_size, clip_batch)
    device = self.device
    torch.cuda.set_device(device)
    free_before, total = torch.cuda.mem_get_info(device)
    headroom = self._headroom(total)
    if free_before <= headroom:
        return None

    from . import basicvsrpp_autotune as tune

    height, width = tune._source_shape(tile_size)
    clip = torch.zeros((clip_batch, clip_length, 3, height, width), dtype=torch.uint8)
    old_tiles = self.tiles
    try:
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        output = _enhance_u8_with_tile_v55(self, clip, tile_size)
        torch.cuda.synchronize(device)
        elapsed = max(time.perf_counter() - started, 1e-6)
        peak_reserved = torch.cuda.max_memory_reserved(device)
        peak_allocated = torch.cuda.max_memory_allocated(device)
        free_after, _ = torch.cuda.mem_get_info(device)
        safe = free_after >= headroom
        emitted = max(1, clip_length - 2 * self._profile_overlap)
        score = (clip_batch * emitted) / elapsed
        del output
        return {
            "score": float(score) if safe else 0.0,
            "elapsed": float(elapsed),
            "peak_reserved": float(peak_reserved),
            "peak_allocated": float(peak_allocated),
            "safe": 1.0 if safe else 0.0,
        }
    except (torch.cuda.OutOfMemoryError, RuntimeError) as error:
        message = str(error).lower()
        if not isinstance(error, torch.cuda.OutOfMemoryError) and "out of memory" not in message:
            raise
        torch.cuda.empty_cache()
        return None
    finally:
        self.tiles = old_tiles
        del clip


class _ProducerFailure:
    def __init__(self, error: BaseException, traceback_text: str) -> None:
        self.error = error
        self.traceback_text = traceback_text


def _install_clip_producer() -> None:
    """Prefetch exact existing v5.2 clip tasks without changing task semantics."""
    from . import balanced_pipeline

    cls = balanced_pipeline.BalancedBasicVSRPPStreamReader
    if getattr(cls, "_v55_clip_producer_installed", False):
        return

    original_init = cls.__init__
    original_next_task = cls._next_task
    original_close = cls.close

    def producer_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        depth = max(
            2,
            sum(max(1, int(getattr(item, "clip_batch", 1))) for item in self.preprocessors),
        )
        self._v55_task_queue = queue.Queue(maxsize=depth)
        self._v55_task_stop = threading.Event()

        def put_item(item) -> bool:
            while not self._v55_task_stop.is_set():
                try:
                    self._v55_task_queue.put(item, timeout=0.25)
                    return True
                except queue.Full:
                    continue
            return False

        def producer_run() -> None:
            try:
                while not self._v55_task_stop.is_set():
                    item = original_next_task(self)
                    if not put_item(item):
                        return
                    if item is None:
                        return
            except Exception as error:
                put_item(_ProducerFailure(error, traceback.format_exc()))

        self._v55_task_thread = threading.Thread(
            target=producer_run,
            name="basicvsrpp-clip-producer",
            daemon=True,
        )
        self._v55_task_thread.start()

    def prefetched_next_task(self):
        item = self._v55_task_queue.get()
        if isinstance(item, _ProducerFailure):
            raise RuntimeError(
                f"BasicVSR++ clip producer failed: {item.error!r}\n{item.traceback_text}"
            ) from item.error
        return item

    def producer_close(self) -> None:
        stop = getattr(self, "_v55_task_stop", None)
        if stop is not None:
            stop.set()
        try:
            original_close(self)
        finally:
            thread = getattr(self, "_v55_task_thread", None)
            if thread is not None and thread.is_alive():
                thread.join(timeout=5.0)

    cls.__init__ = producer_init
    cls._next_task = prefetched_next_task
    cls.close = producer_close
    cls._v55_clip_producer_installed = True


def _install_v55_late_patches() -> None:
    """Apply v5.5 after v5.1/autotune installers have selected their classes."""
    from . import basicvsrpp_autotune as tune
    from . import v51_runtime as v51

    cls = tune.AutoTunedBasicVSRPPPreprocessor
    cls.enhance_clip = _enhance_clip_v55
    cls.enhance_clips = _enhance_clips_v55

    # v5.1 only replaces the autotune probe for 8-bit sources. Detect that
    # exact case so 10-bit sources keep their established float/uint16 probe.
    if cls._probe_full_frame is v51._probe_full_frame_v51:
        cls._probe_full_frame = _probe_full_frame_v55
        tune._CACHE_VERSION = 5
        tune._cache_path = lambda: Path.home() / ".cache" / "realesrgan" / "basicvsrpp-autotune-v5.json"

    bvsr.BasicVSRPPPreprocessor = cls
    _install_clip_producer()


def _install_late_hook() -> None:
    from . import v52_scheduler

    if getattr(v52_scheduler, "_v55_late_hook_installed", False):
        return
    original_process_video = v52_scheduler.process_video

    def process_video_v55(args) -> None:
        _install_v55_late_patches()
        original_process_video(args)

    v52_scheduler.process_video = process_video_v55
    v52_scheduler._v55_late_hook_installed = True


def install_basicvsrpp_execution_optimizations() -> None:
    """Install v5.4/v5.5 execution patches without changing restoration semantics."""
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True
    bvsr.flow_warp = _flow_warp_cached
    bvsr.BasicVSRPlusPlusNet.propagate = _propagate_v54
    _install_late_hook()
