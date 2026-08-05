"""Quality-preserving second-stage runtime tuning for BasicVSR++.

This layer builds on :mod:`enhance.basicvsrpp_quality` and changes only runtime
geometry selection:

* prefer a longer temporal window (configured by the caller, normally 9/2);
* benchmark the largest quality-safe spatial tiles on the actual resident-GPU
  state and choose the lowest estimated-work candidate that fits every GPU;
* never select a tile smaller than the configured baseline;
* if the longer clip cannot fit even at the baseline tile, fall back to the
  previous 7-frame clip while retaining the same baseline tile quality;
* enable cuDNN convolution autotuning for the small set of fixed probe shapes.

The network topology, checkpoint, FP16/FP32 policy, tile padding, cosine spatial
blend, temporal overlap-add, scene detection, and output mixing are unchanged.
"""

from __future__ import annotations

import time
from concurrent.futures import Future
from dataclasses import replace
from typing import Optional, Sequence

import numpy as np
import torch

from . import basicvsrpp_quality as _quality


for _name, _value in vars(_quality).items():
    if not _name.startswith("_"):
        globals()[_name] = _value


_MAX_AUTO_TILE = 1024
_KNOWN_TILE_SIZES = (1024, 896, 768, 640, 576, 512, 448, 384, 320, 256)
_PREVIOUS_CLIP_LENGTH = 7


def _is_cuda_oom(error: BaseException) -> bool:
    text = str(error).lower()
    return isinstance(error, torch.cuda.OutOfMemoryError) or any(
        token in text
        for token in (
            "cuda out of memory",
            "out of memory",
            "ran out of gpu memory",
            "cublas_status_alloc_failed",
        )
    )


def _tile_plan(
    width: int,
    height: int,
    tile_size: int,
    tile_pad: int,
) -> tuple[int, int, int]:
    """Return tile count, total context pixels, and largest context patch."""
    overlap = min(max(tile_pad * 2, 32), max(1, tile_size // 4))
    x_starts = _quality._axis_starts(width, tile_size, overlap)
    y_starts = _quality._axis_starts(height, tile_size, overlap)
    tile_count = 0
    total_context_pixels = 0
    largest_context_patch = 0
    for y0 in y_starts:
        y1 = min(y0 + tile_size, height)
        context_y0 = max(0, y0 - tile_pad)
        context_y1 = min(height, y1 + tile_pad)
        for x0 in x_starts:
            x1 = min(x0 + tile_size, width)
            context_x0 = max(0, x0 - tile_pad)
            context_x1 = min(width, x1 + tile_pad)
            area = (context_y1 - context_y0) * (context_x1 - context_x0)
            tile_count += 1
            total_context_pixels += area
            largest_context_patch = max(largest_context_patch, area)
    return tile_count, total_context_pixels, largest_context_patch


def _candidate_tiles(
    width: int,
    height: int,
    baseline: int,
    tile_pad: int,
) -> list[int]:
    if baseline <= 0:
        return [0]

    upper = max(baseline, min(_MAX_AUTO_TILE, max(width, height)))
    values = {
        value
        for value in (*_KNOWN_TILE_SIZES, baseline)
        if baseline <= value <= upper and value % 4 == 0
    }
    full_frame_candidate = ((max(width, height) + 3) // 4) * 4
    if baseline <= full_frame_candidate <= _MAX_AUTO_TILE:
        values.add(full_frame_candidate)

    def rank(tile_size: int) -> tuple[int, int, int, int]:
        count, total, largest = _tile_plan(width, height, tile_size, tile_pad)
        return total, count, largest, -tile_size

    return sorted(values, key=rank)


def _representative_patch(
    clip: torch.Tensor,
    tile_size: int,
    tile_pad: int,
) -> torch.Tensor:
    height, width = clip.shape[-2:]
    patch_height = min(height, tile_size + 2 * tile_pad)
    patch_width = min(width, tile_size + 2 * tile_pad)
    y0 = max(0, (height - patch_height) // 2)
    x0 = max(0, (width - patch_width) // 2)
    return clip[..., y0 : y0 + patch_height, x0 : x0 + patch_width]


class BasicVSRPPPreprocessor(_quality.BasicVSRPPPreprocessor):
    """Auto-tuned BasicVSR++ without reducing spatial or temporal context."""

    def __init__(
        self,
        config: object,
        checkpoint_dir: Optional[object] = None,
        model: Optional[torch.nn.Module] = None,
        gpu_ids: Optional[Sequence[int]] = None,
    ):
        super().__init__(
            config,
            checkpoint_dir=checkpoint_dir,
            model=model,
            gpu_ids=gpu_ids,
        )
        self.requested_clip_length = int(self.config.clip_length)
        self.baseline_tile_size = int(self.config.tile_size)
        self.autotuned = self.baseline_tile_size == 0
        self.autotune_seconds = 0.0
        self.autotune_attempts = 0
        self.selected_estimated_ratio = 1.0

        if any(runner.device.type == "cuda" for runner in self._runners):
            torch.backends.cudnn.benchmark = True

    def _set_runtime_geometry(self, clip_length: int, tile_size: int) -> None:
        for runner in self._runners:
            runner.config = replace(
                runner.config,
                clip_length=int(clip_length),
                tile_size=int(tile_size),
            )
        self.config = replace(
            self.config,
            clip_length=int(clip_length),
            tile_size=int(tile_size),
        )
        self._device_weight_cache.clear()

    def _probe_runner(
        self,
        runner: object,
        patch: torch.Tensor,
    ) -> tuple[bool, float, int, str]:
        device = runner.device
        if device.type != "cuda":
            return True, 0.0, 0, "cpu"
        started = time.monotonic()
        try:
            with torch.cuda.device(device):
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
                output = runner._run_model(patch)
                torch.cuda.synchronize(device)
                peak = int(torch.cuda.max_memory_allocated(device))
                del output
                torch.cuda.empty_cache()
            return True, time.monotonic() - started, peak, ""
        except BaseException as error:
            if device.type == "cuda":
                with torch.cuda.device(device):
                    torch.cuda.empty_cache()
            if _is_cuda_oom(error):
                return False, time.monotonic() - started, 0, str(error)
            raise

    def _probe_candidate(
        self,
        clip: torch.Tensor,
        tile_size: int,
    ) -> list[tuple[bool, float, int, str]]:
        patch = _representative_patch(clip, tile_size, self.config.tile_pad)
        if self._executor is None:
            return [self._probe_runner(self._runners[0], patch)]
        futures: list[Future] = [
            self._executor.submit(self._probe_runner, runner, patch)
            for runner in self._runners
        ]
        return [future.result() for future in futures]

    def calibrate(self, frames: Sequence[np.ndarray]) -> None:
        """Select clip/tile geometry once from real frames and resident GPUs."""
        if self.autotuned or self.baseline_tile_size == 0:
            return
        if len(frames) < 2:
            return

        started = time.monotonic()
        originals = np.stack(
            [_quality.frame_to_float_rgb(frame) for frame in frames]
        )
        full_clip = torch.from_numpy(originals).permute(0, 3, 1, 2).unsqueeze(0)
        height, width = originals.shape[1:3]
        candidates = _candidate_tiles(
            width,
            height,
            self.baseline_tile_size,
            self.config.tile_pad,
        )

        clip_options = [min(self.requested_clip_length, len(frames))]
        if (
            self.requested_clip_length > _PREVIOUS_CLIP_LENGTH
            and len(frames) >= _PREVIOUS_CLIP_LENGTH
            and _PREVIOUS_CLIP_LENGTH not in clip_options
        ):
            clip_options.append(_PREVIOUS_CLIP_LENGTH)

        selected: Optional[tuple[int, int]] = None
        for clip_length in clip_options:
            clip = full_clip[:, :clip_length]
            for tile_size in candidates:
                self.autotune_attempts += 1
                results = self._probe_candidate(clip, tile_size)
                count, total_pixels, _largest = _tile_plan(
                    width,
                    height,
                    tile_size,
                    self.config.tile_pad,
                )
                ratio = total_pixels / max(width * height, 1)
                status = ", ".join(
                    (
                        f"gpu{runner.device.index}:"
                        f"{'ok' if result[0] else 'oom'}/"
                        f"{result[1]:.2f}s/"
                        f"peak={result[2] / 2**30:.2f}GiB"
                    )
                    for runner, result in zip(self._runners, results)
                )
                print(
                    f"[basicvsrpp auto] clip={clip_length}, tile={tile_size}, "
                    f"tiles={count}, context_ratio={ratio:.3f}, {status}",
                    flush=True,
                )
                if all(result[0] for result in results):
                    selected = clip_length, tile_size
                    self.selected_estimated_ratio = ratio
                    break
            if selected is not None:
                break

        if selected is None:
            raise RuntimeError(
                "BasicVSR++ auto tuning could not fit the previous quality "
                f"baseline (clip={_PREVIOUS_CLIP_LENGTH}, "
                f"tile={self.baseline_tile_size})."
            )

        self._set_runtime_geometry(*selected)
        self.autotuned = True
        self.autotune_seconds += time.monotonic() - started
        fallback = selected[0] < self.requested_clip_length
        print(
            f"[basicvsrpp auto] selected clip={selected[0]}, "
            f"overlap={self.config.clip_overlap}, tile={selected[1]}, "
            f"baseline_tile={self.baseline_tile_size}, "
            f"fallback_to_previous_clip={fallback}, "
            f"autotune={self.autotune_seconds:.1f}s",
            flush=True,
        )

    def enhance_clip(self, frames: Sequence[np.ndarray]) -> list[np.ndarray]:
        if not self.autotuned and len(frames) >= self.requested_clip_length:
            self.calibrate(frames[: self.requested_clip_length])
        return super().enhance_clip(frames)

    def close(self) -> None:
        if not self._closed:
            print(
                f"[basicvsrpp auto] attempts={self.autotune_attempts}, "
                f"selected_clip={self.config.clip_length}, "
                f"selected_tile={self.config.tile_size}, "
                f"context_ratio={self.selected_estimated_ratio:.3f}, "
                f"autotune={self.autotune_seconds:.1f}s",
                flush=True,
            )
        super().close()


class BasicVSRPPStreamReader(_quality.BasicVSRPPStreamReader):
    """Adapt the stream geometry after the first real-frame memory probe."""

    def __init__(self, reader: object, preprocessor: BasicVSRPPPreprocessor):
        super().__init__(reader, preprocessor)
        self._stage2_geometry_ready = preprocessor.autotuned

    def _apply_selected_geometry(self) -> None:
        self.clip_length = int(self.preprocessor.config.clip_length)
        self.overlap = int(self.preprocessor.config.clip_overlap)
        self.hop = self.clip_length - 2 * self.overlap
        if self.hop < 1:
            raise ValueError("BasicVSR++ selected clip overlap leaves no hop")
        self._stage2_geometry_ready = True

    def _process_full_windows(self) -> None:
        if (
            not self._stage2_geometry_ready
            and len(self.buffer_frames) >= self.clip_length
        ):
            self.preprocessor.calibrate(
                self.buffer_frames[: self.clip_length]
            )
            self._apply_selected_geometry()

        while len(self.buffer_frames) >= self.clip_length:
            frames = self.buffer_frames[: self.clip_length]
            indices = self.buffer_indices[: self.clip_length]
            self._process_window(frames, indices)
            next_start = indices[0] + self.hop
            del self.buffer_frames[: self.hop]
            del self.buffer_indices[: self.hop]
            self._emit_before(next_start)


__all__ = sorted(
    {
        name
        for name in globals()
        if not name.startswith("_")
        and name
        not in {
            "np",
            "torch",
            "time",
            "replace",
            "Future",
            "Optional",
            "Sequence",
        }
    }
)
