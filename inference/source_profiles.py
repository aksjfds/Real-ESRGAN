"""Source-quality profiles shared by the CLI and BasicVSR++ pipeline."""

from __future__ import annotations

SOURCE_PROFILES = {
    "A": {
        "strength": 0.0,
        "clip_length": 0,
        "description": "off / clean source",
    },
    "B": {
        "strength": 0.25,
        "clip_length": 7,
        "description": "light restoration / mild compression-noise",
    },
    "C": {
        "strength": 0.50,
        "clip_length": 9,
        "description": "medium restoration / visible compression-noise",
    },
    "D": {
        "strength": 0.75,
        "clip_length": 11,
        "description": "strong restoration / heavy compression-noise",
    },
    "E": {
        "strength": 1.00,
        "clip_length": 13,
        "description": "full BasicVSR++ restoration strength",
    },
}

PROFILE_CHOICES = tuple(SOURCE_PROFILES)
