# src/custom_sam_peft/models/semantic.py
"""Head-free semantic marginalization over the SAM3 grounding output dict (§6).

Pure functions; the model itself is unchanged. Consumes only keys the existing
forward already produces and validates: pred_logits, pred_masks,
presence_logit_dec (+ semantic_seg for the opt-in source path).
"""

from __future__ import annotations

import torch
from torch import Tensor

_EPS = 1e-6


def marginalize_group(
    outputs: dict[str, Tensor],
    b: int,
    k: int,
    *,
    query_reduce: str,
    source: str,
) -> Tensor:
    """(N=b*k columns) -> (b, k, H, W) per-concept foreground LOGITS for this group.

    column n corresponds to image n//k and concept n%k (image-major / class-minor).
    """
    if source == "semantic_seg":
        # surfaced single-channel foreground map (§6.4); (N,1,H,W) -> (b,k,H,W).
        seg = outputs["semantic_seg"]  # (N, 1, H, W)
        _n, _, h, w = seg.shape
        return seg.reshape(b, k, h, w)

    pred_logits = outputs["pred_logits"]  # (N, Q, 1)
    pred_masks = outputs["pred_masks"]  # (N, Q, H, W)
    presence = outputs["presence_logit_dec"]  # (N, 1)
    _n, _q, h, w = pred_masks.shape

    obj_q = torch.sigmoid(pred_logits[..., 0])  # (N, Q)
    mask_q = torch.sigmoid(pred_masks)  # (N, Q, H, W)
    pres = torch.sigmoid(presence[:, 0])  # (N,)

    weighted = obj_q[:, :, None, None] * mask_q  # (N, Q, H, W)
    if query_reduce == "max":
        fg = weighted.amax(dim=1)  # (N, H, W) in [0,1]
    elif query_reduce == "sum":
        fg = weighted.sum(dim=1)  # (N, H, W) in [0, +)
    else:
        raise ValueError(f"unknown query_reduce: {query_reduce!r}")
    fg = pres[:, None, None] * fg  # gate by presence
    fg = fg.clamp(_EPS, 1.0 - _EPS)
    fg_logits = torch.log(fg) - torch.log1p(-fg)  # logit(fg)
    return fg_logits.reshape(b, k, h, w)


def build_semantic_logits(
    group_logit_slices: list[Tensor],
    *,
    background_logit: float,
) -> Tensor:
    """Concat per-group (b, k_g, H, W) slices along concept axis, prepend bg -> (B, K+1, H, W)."""
    concept = torch.cat(group_logit_slices, dim=1)  # (B, K, H, W)
    b, _, h, w = concept.shape
    bg = torch.full(
        (b, 1, h, w), float(background_logit), device=concept.device, dtype=concept.dtype
    )
    return torch.cat([bg, concept], dim=1)  # (B, K+1, H, W), channel 0 = background


def semantic_argmax(sem_logits: Tensor) -> Tensor:
    """(B, K+1, H, W) -> (B, H, W) int64 in {0..K}; channel 0 == background."""
    return sem_logits.argmax(dim=1)
