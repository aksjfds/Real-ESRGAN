#!/usr/bin/env python3
"""Unified Real-ESRGAN video inference entry point."""

from __future__ import annotations

import multiprocessing as mp
from pathlib import Path

from encode import runtime as encode_runtime
from inference import runtime as inference_runtime
from inference import v52_scheduler as inference_pipeline
from inference.progress_log import install_persistent_progress
from inference.source_profiles import PROFILE_CHOICES, SOURCE_PROFILES
from inference.v51_runtime import install_basicvsrpp_optimizations, install_pipeline_optimizations


def _install_v55_repeat_run_fix() -> None:
    """Make v5.5 clip prefetch deterministic and fail-fast across repeated runs."""
    import queue
    import threading
    import traceback

    from inference import balanced_pipeline
    from inference import v54_runtime

    cls = balanced_pipeline.BalancedBasicVSRPPStreamReader
    if getattr(cls, "_v55_repeat_run_fix_installed", False):
        return

    # Capture the original synchronous reader methods before v5.5 installs its
    # Queue.get()-based producer wrapper. We keep the exact same task generator,
    # scene-cut logic, overlap policy, and frame ordering.
    base_init = cls.__init__
    base_next_task = cls._next_task
    base_close = cls.close

    # v5.1/autotune is already installed when this helper is called, so it is now
    # safe to apply v5.5's late uint8-H2D patches. This also installs the old
    # producer wrapper, which is immediately replaced below with the deterministic
    # implementation built from the captured synchronous methods.
    v54_runtime._install_v55_late_patches()

    class ProducerFailure:
        def __init__(self, error: BaseException, traceback_text: str) -> None:
            self.error = error
            self.traceback_text = traceback_text

    def safe_init(self, *args, **kwargs):
        base_init(self, *args, **kwargs)
        depth = max(
            2,
            sum(max(1, int(getattr(item, "clip_batch", 1))) for item in self.preprocessors),
        )
        self._v55_task_queue = queue.Queue(maxsize=depth)
        self._v55_task_stop = threading.Event()
        self._v55_task_thread = None

        # Prime one complete scheduling window synchronously. On a cached second
        # run, the scheduler can otherwise start before the background producer
        # has prepared GPU1's first clip and block inside Queue.get().
        reached_eof = False
        try:
            for _ in range(depth):
                item = base_next_task(self)
                self._v55_task_queue.put_nowait(item)
                if item is None:
                    reached_eof = True
                    break
        except Exception as error:
            self._v55_task_queue.put_nowait(ProducerFailure(error, traceback.format_exc()))
            reached_eof = True

        if reached_eof:
            return

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
                    item = base_next_task(self)
                    if not put_item(item):
                        return
                    if item is None:
                        return
            except Exception as error:
                put_item(ProducerFailure(error, traceback.format_exc()))

        self._v55_task_thread = threading.Thread(
            target=producer_run,
            name="basicvsrpp-clip-producer",
            daemon=True,
        )
        self._v55_task_thread.start()

    def safe_next_task(self):
        # Do not allow the dynamic scheduler to hang forever on an empty queue.
        # Normally the queue is prefilled and continuously replenished; timeout
        # polling is only a liveness guard for an unexpectedly stopped producer.
        while True:
            try:
                item = self._v55_task_queue.get(timeout=0.5)
                break
            except queue.Empty:
                thread = getattr(self, "_v55_task_thread", None)
                if thread is None or not thread.is_alive():
                    raise RuntimeError(
                        "BasicVSR++ clip producer stopped before emitting EOF; "
                        "refusing to leave the scheduler blocked at 0%."
                    )

        if isinstance(item, ProducerFailure):
            raise RuntimeError(
                f"BasicVSR++ clip producer failed: {item.error!r}\n{item.traceback_text}"
            ) from item.error
        return item

    def safe_close(self) -> None:
        stop = getattr(self, "_v55_task_stop", None)
        if stop is not None:
            stop.set()
        thread = getattr(self, "_v55_task_thread", None)
        if thread is not None and thread.is_alive():
            thread.join(timeout=5.0)
        try:
            base_close(self)
        finally:
            if thread is not None and thread.is_alive():
                thread.join(timeout=1.0)

    cls.__init__ = safe_init
    cls._next_task = safe_next_task
    cls.close = safe_close
    cls._v55_repeat_run_fix_installed = True


def main() -> None:
    parser = inference_runtime.build_parser()
    parser.add_argument(
        "--source-profile",
        choices=PROFILE_CHOICES,
        default="A",
        help=(
            "A=BasicVSR++ off; B=25%%; C=50%%; D=75%%; "
            "E=100%% full-strength NTIRE compressed-video restoration"
        ),
    )
    parser.add_argument(
        "--gpu-timing",
        action="store_true",
        help="Enable CUDA-event GPU busy/wait diagnostics (adds profiling overhead).",
    )
    encode_runtime.extend_parser(parser)
    args = parser.parse_args()
    encode_runtime.prepare_runtime(inference_runtime, args)
    install_pipeline_optimizations()
    # Durable saved-log heartbeat is independent of tqdm and wraps the active
    # OutputPump after v5.1 transfer/progress optimizations are installed.
    install_persistent_progress()

    # Keep the CLI as the single public entry while allowing the pipeline to
    # share the extended A-E profile table without duplicating configuration.
    inference_pipeline.base_pipeline.SOURCE_PROFILES = SOURCE_PROFILES

    source_bit_depth = None
    if args.source_profile != "A":
        # B-E always use the checkpoint parts bundled in inference/weights.
        from inference import basicvsrpp
        from inference.basicvsrpp_autotune import install_autotune
        from inference.checkpoint_parts import resolve_checkpoint
        from inference.v54_runtime import install_basicvsrpp_execution_optimizations

        # v5.4 only changes execution: cache invariant warp grids and avoid
        # temporary CUDA zero tensors that are immediately overwritten.
        install_basicvsrpp_execution_optimizations()
        basicvsrpp.SOURCE_PROFILES = SOURCE_PROFILES
        basicvsrpp.download_checkpoint = resolve_checkpoint
        try:
            source = inference_runtime.probe_video(
                Path(args.input).expanduser().resolve(),
                args.ffprobe_bin,
            )
            install_autotune(source.width, source.height)
            source_bit_depth = source.bit_depth
        except Exception as error:
            print(
                f"[autotuner] source probe unavailable ({error}); using hardware-only search",
                flush=True,
            )
            install_autotune()
        install_basicvsrpp_optimizations(source_bit_depth)
        _install_v55_repeat_run_fix()

    # CUDA-event timing is diagnostic instrumentation. Keep it off in normal
    # inference so Event synchronization cannot serialize the production path.
    if args.gpu_timing:
        from inference.gpu_timing import install_gpu_timing

        install_gpu_timing(enable_bvs=args.source_profile != "A")
    inference_pipeline.process_video(args)


if __name__ == "__main__":
    mp.freeze_support()
    main()
