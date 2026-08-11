"""Compatibility alias for the fail-fast stage-isolated GPU transport."""

from __future__ import annotations

from .gpu_transport import GPUWorkerTransport, _TASK_TIMEOUT


class StableGPUWorkerTransport(GPUWorkerTransport):
    """Backward-compatible name; fail-fast handling now lives in the base transport."""

    pass


__all__ = ["StableGPUWorkerTransport", "_TASK_TIMEOUT"]
