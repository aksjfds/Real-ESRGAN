"""v5.4 BasicVSR++ execution-only optimizations.

These patches preserve the BasicVSR++ model, weights, clip policy, tiling,
scene-cut behavior, and arithmetic. They only reuse invariant warp grids and
avoid temporary zero tensors that were immediately overwritten.
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch.nn import functional as F

from . import basicvsrpp as bvsr


_INSTALLED = False
_FLOW_GRID_CACHE: Dict[Tuple[torch.device, torch.dtype, int, int], torch.Tensor] = {}


def _flow_warp_cached(
    x: torch.Tensor,
    flow: torch.Tensor,
    interpolation: str = "bilinear",
    padding_mode: str = "zeros",
    align_corners: bool = True,
) -> torch.Tensor:
    """Match BasicVSR++ flow_warp while reusing its invariant pixel grid."""
    if x.shape[-2:] != flow.shape[1:3]:
        raise ValueError(f"flow size {flow.shape[1:3]} does not match feature size {x.shape[-2:]}")
    n, _c, h, w = x.shape
    key = (x.device, x.dtype, int(h), int(w))
    base_grid = _FLOW_GRID_CACHE.get(key)
    if base_grid is None:
        grid_y, grid_x = torch.meshgrid(
            torch.arange(h, device=x.device, dtype=x.dtype),
            torch.arange(w, device=x.device, dtype=x.dtype),
            indexing="ij",
        )
        base_grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0)
        _FLOW_GRID_CACHE[key] = base_grid

    grid = base_grid.expand(n, -1, -1, -1) + flow.to(dtype=x.dtype)
    gx = 2.0 * grid[..., 0] / (w - 1) - 1.0 if w > 1 else torch.zeros_like(grid[..., 0])
    gy = 2.0 * grid[..., 1] / (h - 1) - 1.0 if h > 1 else torch.zeros_like(grid[..., 1])
    normalized = torch.stack((gx, gy), dim=-1)
    return F.grid_sample(
        x, normalized, mode=interpolation, padding_mode=padding_mode, align_corners=align_corners
    )


def _propagate_v54(
    self,
    feats: dict[str, list[torch.Tensor]],
    flows: torch.Tensor,
    module_name: str,
) -> dict[str, list[torch.Tensor]]:
    """Match BasicVSR++ propagation without zero tensors that get overwritten."""
    n, t, _c, h, w = flows.shape
    frame_idx = list(range(t + 1))
    flow_idx = list(range(-1, t))
    mapping_idx = list(range(len(feats["spatial"])))
    mapping_idx += mapping_idx[::-1]
    if "backward" in module_name:
        frame_idx = frame_idx[::-1]
        flow_idx = frame_idx
    feat_prop = flows.new_zeros(n, self.mid_channels, h, w)
    for i, idx in enumerate(frame_idx):
        feat_current = feats["spatial"][mapping_idx[idx]]
        if i > 0:
            flow_n1 = flows[:, flow_idx[i]]
            cond_n1 = bvsr.flow_warp(feat_prop, flow_n1.permute(0, 2, 3, 1))
            if i > 1:
                feat_n2 = feats[module_name][-2]
                flow_n2 = flows[:, flow_idx[i - 1]]
                flow_n2 = flow_n1 + bvsr.flow_warp(
                    flow_n2, flow_n1.permute(0, 2, 3, 1)
                )
                cond_n2 = bvsr.flow_warp(feat_n2, flow_n2.permute(0, 2, 3, 1))
            else:
                feat_n2 = torch.zeros_like(feat_prop)
                flow_n2 = torch.zeros_like(flow_n1)
                cond_n2 = torch.zeros_like(cond_n1)
            cond = torch.cat((cond_n1, feat_current, cond_n2), dim=1)
            feat_prop = self.deform_align[module_name](
                torch.cat((feat_prop, feat_n2), dim=1), cond, flow_n1, flow_n2
            )
        feat = [feat_current]
        feat.extend(feats[key][idx] for key in feats if key not in ("spatial", module_name))
        feat.append(feat_prop)
        feat_prop = feat_prop + self.backbone[module_name](torch.cat(feat, dim=1))
        feats[module_name].append(feat_prop)
    if "backward" in module_name:
        feats[module_name].reverse()
    return feats


def install_basicvsrpp_execution_optimizations() -> None:
    """Install v5.4 execution patches without changing restoration semantics."""
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True
    bvsr.flow_warp = _flow_warp_cached
    bvsr.BasicVSRPlusPlusNet.propagate = _propagate_v54
