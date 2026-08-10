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


def _disable_v55_clip_producer() -> None:
    """Keep v5.5 uint8-H2D optimizations but disable its queue-based clip producer.

    The dynamic v5.2 scheduler already requests clips synchronously for each free
    restoration GPU. Wrapping that request in a blocking Queue.get() can stall the
    scheduler between GPU0 and GPU1 dispatch, leaving GPU1 idle while progress
    remains at 0%. Source decode is tiny compared with BasicVSR++ compute in the
    measured workload, so restoring the original synchronous task generator trades
    negligible overlap for deterministic multi-GPU scheduling.
    """
    from inference import v54_runtime

    if getattr(v54_runtime, "_v56_clip_producer_disabled", False):
        return

    # v5.5's late patch also installs the compact uint8 H2D execution path. Keep
    # that optimization, but permanently turn its optional queue producer into a
    # no-op before any late patch can wrap BalancedBasicVSRPPStreamReader.
    v54_runtime._install_clip_producer = lambda: None
    v54_runtime._v56_clip_producer_disabled = True

    # Apply the remaining v5.5 late patches now. The v5.6 cache policy installed
    # below will override the temporary v5 cache selection before model creation.
    v54_runtime._install_v55_late_patches()


def _install_v56_selective_channels_last() -> None:
    """Use channels-last only inside BasicVSR++ convolution-heavy 4D islands."""
    import torch
    from torch import nn

    from inference import basicvsrpp_autotune as tune
    from inference import v54_runtime

    cls = tune.AutoTunedBasicVSRPPPreprocessor
    if getattr(cls, "_v56_selective_channels_last_installed", False):
        return

    class ChannelsLastIsland(nn.Module):
        """Run one convolution island as NHWC, then restore NCHW for other ops."""

        def __init__(self, module: nn.Module):
            super().__init__()
            self.module = module
            # Checkpoint loading has already finished by the time this wrapper is
            # created, so changing weight memory format cannot alter state-dict keys.
            self.module.to(memory_format=torch.channels_last)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = x.contiguous(memory_format=torch.channels_last)
            output = self.module(x)
            # grid_sample and torchvision deform_conv2d stay on the established
            # contiguous NCHW path; only the dense convolution islands use NHWC.
            return output.contiguous(memory_format=torch.contiguous_format)

    def wrap(module: nn.Module) -> nn.Module:
        if isinstance(module, ChannelsLastIsland):
            return module
        return ChannelsLastIsland(module)

    def apply_model(model: nn.Module) -> None:
        if getattr(model, "_v56_selective_channels_last_applied", False):
            return

        # These are the long Conv2d-heavy regions. Keeping conversion boundaries
        # around whole islands amortizes NCHW<->NHWC copies over many convolutions.
        model.feat_extract = wrap(model.feat_extract)
        model.reconstruction = wrap(model.reconstruction)

        for index in range(len(model.spynet.basic_module)):
            model.spynet.basic_module[index] = wrap(model.spynet.basic_module[index])

        for name in list(model.backbone.keys()):
            model.backbone[name] = wrap(model.backbone[name])

        # Only offset prediction convolutions use channels-last here. The actual
        # deform_conv2d weight/input layout remains untouched.
        for name in list(model.deform_align.keys()):
            alignment = model.deform_align[name]
            alignment.conv_offset = wrap(alignment.conv_offset)

        model._v56_selective_channels_last_applied = True
        print(
            "[basicvsrpp] selective channels_last enabled for convolution islands",
            flush=True,
        )

    original_autotune = cls._autotune

    def autotune_v56(self) -> None:
        # AutoTunedBasicVSRPPPreprocessor calls _autotune only after the checkpoint
        # is loaded, the model is on CUDA, and FP16 selection is complete.
        apply_model(self.model)
        original_autotune(self)

    cls._autotune = autotune_v56
    cls._v56_selective_channels_last_installed = True

    def select_v6_cache() -> None:
        # The execution layout changed, so do not reuse v5 timing decisions. The
        # first v5.6 run re-benchmarks tile choice; following runs use this cache.
        tune._CACHE_VERSION = 6
        tune._cache_path = lambda: Path.home() / ".cache" / "realesrgan" / "basicvsrpp-autotune-v6.json"

    select_v6_cache()

    # v5.5's process_video late hook reapplies its v5 cache settings immediately
    # before model construction. Wrap that hook so v6 remains the final cache
    # policy whenever it runs, including repeated notebook executions.
    original_late_patches = v54_runtime._install_v55_late_patches

    def late_patches_v56() -> None:
        original_late_patches()
        select_v6_cache()

    v54_runtime._install_v55_late_patches = late_patches_v56


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
        _disable_v55_clip_producer()
        _install_v56_selective_channels_last()

    # CUDA-event timing is diagnostic instrumentation. Keep it off in normal
    # inference so Event synchronization cannot serialize the production path.
    if args.gpu_timing:
        from inference.gpu_timing import install_gpu_timing

        install_gpu_timing(enable_bvs=args.source_profile != "A")
    inference_pipeline.process_video(args)


if __name__ == "__main__":
    mp.freeze_support()
    main()
