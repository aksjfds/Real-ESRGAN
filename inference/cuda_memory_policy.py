"""CUDA allocator policy for memory-constrained multi-process inference."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import MutableMapping

DEFAULT_ALLOC_CONF = "expandable_segments:True,garbage_collection_threshold:0.80"
_MIN_RECLAIMABLE_BYTES = 512 * 1024 * 1024
_MIN_FREE_BYTES = 1280 * 1024 * 1024
_MIN_FREE_FRACTION = 0.10


def configure_cuda_allocator_env(
    environ: MutableMapping[str, str] | None = None,
) -> tuple[str, bool]:
    """Set a fragmentation-resistant default without overriding user policy."""
    env = os.environ if environ is None else environ
    for name in ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF"):
        configured = str(env.get(name, "")).strip()
        if configured:
            return configured, False
    env["PYTORCH_ALLOC_CONF"] = DEFAULT_ALLOC_CONF
    return DEFAULT_ALLOC_CONF, True


@dataclass(frozen=True)
class CacheTrimResult:
    free_before: int
    free_after: int
    allocated: int
    reserved: int
    reclaimable: int


def trim_cuda_cache_under_pressure(device, *, enabled: bool) -> CacheTrimResult | None:
    """Release unused cache only when global free VRAM is genuinely tight."""
    if not enabled or getattr(device, "type", None) != "cuda":
        return None

    import torch

    free_before, total = torch.cuda.mem_get_info(device)
    allocated = int(torch.cuda.memory_allocated(device))
    reserved = int(torch.cuda.memory_reserved(device))
    reclaimable = max(0, reserved - allocated)
    free_floor = max(
        _MIN_FREE_BYTES,
        int(int(total) * _MIN_FREE_FRACTION),
    )
    if int(free_before) >= free_floor or reclaimable < _MIN_RECLAIMABLE_BYTES:
        return None

    torch.cuda.empty_cache()
    free_after, _ = torch.cuda.mem_get_info(device)
    return CacheTrimResult(
        free_before=int(free_before),
        free_after=int(free_after),
        allocated=allocated,
        reserved=reserved,
        reclaimable=reclaimable,
    )
