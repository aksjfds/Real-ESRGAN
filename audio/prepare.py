#!/usr/bin/env python3
"""Prepare the isolated v8.1 audio backend before expensive video inference."""

from __future__ import annotations

from .backend import prepare_backend


def main() -> None:
    prepare_backend()
    print("[audio-v8.1] backend ready", flush=True)


if __name__ == "__main__":
    main()
