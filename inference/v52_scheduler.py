"""Compatibility entrypoint for the v6.2 modular scheduler."""

from . import scheduler, scheduler_reporting


def process_video(args) -> None:
    minimum_output_bit_depth = (
        int(args.av1_bit_depth)
        if args.video_codec == "av1_nvenc"
        else 0
    )
    scheduler_reporting.configure_output_bit_depth(
        minimum_output_bit_depth
    )
    scheduler.process_video(args)


__all__ = ["process_video"]
