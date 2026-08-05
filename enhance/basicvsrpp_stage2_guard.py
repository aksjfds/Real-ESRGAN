"""Safety guard for BasicVSR++ stage-two auto tuning.

The stage-two tuner probes model patches. This layer additionally reserves the
same full-clip FP32 accumulation buffers used by the real multi-GPU tile lane,
so a candidate is accepted only when the complete runtime allocation fits.

For a scene segment that ends before the requested 9-frame window, repeated
copies of the last frame are used only for the memory probe. Real inference
still receives the original frames unchanged.
"""

from __future__ import annotations

import time
from concurrent.futures import Future
from typing import Sequence

import torch

from . import basicvsrpp_stage2 as _stage2


for _name, _value in vars(_stage2).items():
    if not _name.startswith("_"):
        globals()[_name] = _value


class BasicVSRPPPreprocessor(_stage2.BasicVSRPPPreprocessor):
    """Stage-two tuner with full runtime-memory reservation."""

    def _probe_runner(
        self,
        runner: object,
        patch: torch.Tensor,
        full_shape: Sequence[int],
    ) -> tuple[bool, float, int, str]:
        device = runner.device
        if device.type != "cuda":
            return True, 0.0, 0, "cpu"
        started = time.monotonic()
        reserve_result = None
        reserve_weight = None
        output = None
        try:
            with torch.cuda.device(device):
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
                reserve_result = torch.empty(
                    tuple(int(value) for value in full_shape),
                    dtype=torch.float32,
                    device=device,
                )
                reserve_weight = torch.empty(
                    (int(full_shape[-2]), int(full_shape[-1])),
                    dtype=torch.float32,
                    device=device,
                )
                output = runner._run_model(patch)
                torch.cuda.synchronize(device)
                peak = int(torch.cuda.max_memory_allocated(device))
                del output, reserve_result, reserve_weight
                output = reserve_result = reserve_weight = None
                torch.cuda.empty_cache()
            return True, time.monotonic() - started, peak, ""
        except BaseException as error:
            del output, reserve_result, reserve_weight
            if device.type == "cuda":
                with torch.cuda.device(device):
                    torch.cuda.empty_cache()
            if _stage2._is_cuda_oom(error):
                return False, time.monotonic() - started, 0, str(error)
            raise

    def _probe_candidate(
        self,
        clip: torch.Tensor,
        tile_size: int,
    ) -> list[tuple[bool, float, int, str]]:
        patch = _stage2._representative_patch(
            clip,
            tile_size,
            self.config.tile_pad,
        )
        full_shape = tuple(int(value) for value in clip.shape)
        if self._executor is None:
            return [
                self._probe_runner(
                    self._runners[0],
                    patch,
                    full_shape,
                )
            ]
        futures: list[Future] = [
            self._executor.submit(
                self._probe_runner,
                runner,
                patch,
                full_shape,
            )
            for runner in self._runners
        ]
        return [future.result() for future in futures]


class BasicVSRPPStreamReader(_stage2.BasicVSRPPStreamReader):
    """Calibrate safely even when the first scene is shorter than 9 frames."""

    def _finish_segment(self) -> None:
        if (
            not self._stage2_geometry_ready
            and len(self.buffer_frames) >= _stage2._PREVIOUS_CLIP_LENGTH
        ):
            probe_frames = list(self.buffer_frames)
            requested = int(self.preprocessor.requested_clip_length)
            while len(probe_frames) < requested:
                probe_frames.append(probe_frames[-1])
            self.preprocessor.calibrate(probe_frames[:requested])
            self._apply_selected_geometry()

        if self._stage2_geometry_ready:
            self._process_full_windows()
        super()._finish_segment()


__all__ = sorted(
    {
        name
        for name in globals()
        if not name.startswith("_")
        and name not in {"torch", "time", "Future", "Sequence"}
    }
)
