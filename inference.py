#!/usr/bin/env python3
"""Unified Real-ESRGAN video inference entry point."""

from __future__ import annotations

import multiprocessing as mp

from encode import runtime as encode_runtime
from inference import bitdepth as bitdepth_runtime
from inference import runtime as inference_runtime


def main() -> None:
    bitdepth_runtime.install(inference_runtime)
    parser = inference_runtime.build_parser()
    encode_runtime.extend_parser(parser)
    args = parser.parse_args()
    encode_runtime.prepare_runtime(inference_runtime, args)
    inference_runtime.process_video(args)


if __name__ == "__main__":
    mp.freeze_support()
    main()
