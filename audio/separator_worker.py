#!/usr/bin/env python3
"""Worker executed inside the isolated audio-separator environment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

MODEL = "vocals_mel_band_roformer.ckpt"


def _separator(model_dir: Path, output_dir: Path | None = None):
    from audio_separator.separator import Separator

    kwargs = {
        "model_file_dir": str(model_dir),
        "output_format": "WAV",
        "normalization_threshold": 1.0,
        "amplification_threshold": 0.0,
        "sample_rate": 48000,
        "use_soundfile": True,
        "use_autocast": True,
        "output_single_stem": "Vocals",
        "mdxc_params": {
            "segment_size": 256,
            "override_model_segment_size": False,
            "batch_size": 1,
            "overlap": 8,
            "pitch_shift": 0,
        },
    }
    if output_dir is not None:
        kwargs["output_dir"] = str(output_dir)
    separator = Separator(**kwargs)
    separator.load_model(model_filename=MODEL)
    return separator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    args.model_dir = args.model_dir.resolve()
    args.model_dir.mkdir(parents=True, exist_ok=True)
    if args.prepare:
        _separator(args.model_dir)
        return
    if args.input is None or args.output_dir is None:
        parser.error("--input and --output-dir are required unless --prepare is used")

    args.input = args.input.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    separator = _separator(args.model_dir, args.output_dir)
    output_files = separator.separate(
        str(args.input),
        {"Vocals": "dialogue_v81"},
    )
    candidates = []
    for value in output_files:
        path = Path(value)
        if not path.is_absolute():
            path = args.output_dir / path.name
        candidates.append(path.resolve())
    dialogue = next(
        (path for path in candidates if "dialogue_v81" in path.name.lower()),
        None,
    )
    if dialogue is None and len(candidates) == 1:
        dialogue = candidates[0]
    if dialogue is None or not dialogue.is_file():
        raise RuntimeError(f"Unable to identify dialogue stem from: {output_files}")
    print(json.dumps({"dialogue": str(dialogue)}), flush=True)


if __name__ == "__main__":
    main()
