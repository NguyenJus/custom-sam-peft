"""SemanticLoss: multi-class CE + region loss over (B, K+1, H, W) logits (spec §7.4).

Marginalization (Phase C) yields a (B, K+1, H, W) per-concept foreground logit
volume (channel 0 = background). This module turns that volume + dense GT labels
into {"ce", "region", "total"}. The per-class region math reuses terms/mask.py's
_dice / _tversky_index / _kervadec_boundary applied per channel over (B, H, W).

ignore_index pixels are void: excluded from CE (via F.cross_entropy ignore_index)
and zeroed in both prediction and one-hot target before every region reduction.
GT is nearest-downsampled to the logit resolution (bilinear would invent
fractional class ids). The background channel (0) is a real argmax class and is
INCLUDED in both CE and region.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from custom_sam_peft.data.base import SemanticTarget
from custom_sam_peft.models.losses.semantic_presets import ResolvedSemanticLoss
from custom_sam_peft.models.losses.terms.mask import (
    _dice,
    _kervadec_boundary,
    _tversky_index,
)

# A family fn takes (logits, labels, valid, probs, ignore_index, num_channels) and
# returns (ce_term, region_term). Either term may be a graph-connected zero.
_FamilyFn = Callable[[Tensor, Tensor, Tensor, Tensor, int, int], tuple[Tensor, Tensor]]
_FamilyBuilder = Callable[[ResolvedSemanticLoss], _FamilyFn]
_PerClassFn = Callable[[Tensor, Tensor], Tensor]


def _zero_like(logits: Tensor) -> Tensor:
    """A finite, graph-connected scalar zero (keeps autograd happy on empty terms)."""
    return logits.sum() * 0.0


def _stack_downsample_labels(
    targets: list[SemanticTarget], size: tuple[int, int], device: torch.device
) -> tuple[Tensor, int]:
    """Stack each target's labels, nearest-downsample to `size` -> (B, H, W) int64."""
    ignore_index = targets[0].ignore_index
    maps: list[Tensor] = []
    for tgt in targets:
        lbl = tgt.labels
        if tuple(lbl.shape[-2:]) != size:
            lbl = F.interpolate(lbl[None, None].float(), size=size, mode="nearest")[0, 0]
        maps.append(lbl.to(torch.int64))
    return torch.stack(maps, dim=0).to(device), ignore_index


def _multiclass_ce(logits: Tensor, labels: Tensor, ignore_index: int) -> Tensor:
    valid = labels != ignore_index
    if not bool(valid.any()):
        return _zero_like(logits)
    return F.cross_entropy(logits, labels, ignore_index=ignore_index)


def _multiclass_focal_ce(
    logits: Tensor, labels: Tensor, ignore_index: int, gamma: float, alpha: float
) -> Tensor:
    valid = labels != ignore_index
    if not bool(valid.any()):
        return _zero_like(logits)
    logp = F.log_softmax(logits, dim=1)  # (B, C, H, W)
    safe = torch.where(valid, labels, torch.zeros_like(labels))  # gather needs in-range idx
    logp_t = logp.gather(1, safe[:, None]).squeeze(1)  # (B, H, W)
    p_t = logp_t.exp()
    focal = -alpha * (1.0 - p_t).pow(gamma) * logp_t
    return focal[valid].mean()


def _region_per_class(
    probs: Tensor, labels: Tensor, valid: Tensor, num_channels: int, per_class: _PerClassFn
) -> Tensor:
    valid_f = valid.float()
    terms: list[Tensor] = []
    for c in range(num_channels):
        pred_c = probs[:, c] * valid_f  # zero at void
        tgt_c = ((labels == c) & valid).float()  # zero at void
        terms.append(per_class(pred_c, tgt_c))
    return torch.stack(terms).mean()


def _dice_region(probs: Tensor, labels: Tensor, valid: Tensor, num_channels: int) -> Tensor:
    return _region_per_class(probs, labels, valid, num_channels, lambda p, t: _dice(p, t))


def _focal_tversky_region(
    probs: Tensor, labels: Tensor, valid: Tensor, num_channels: int, alpha: float, gamma: float
) -> Tensor:
    def per_class(p: Tensor, t: Tensor) -> Tensor:
        ti = _tversky_index(p, t, alpha)
        return (1.0 - ti).pow(gamma).mean()

    return _region_per_class(probs, labels, valid, num_channels, per_class)


def _boundary_region(
    probs: Tensor, labels: Tensor, valid: Tensor, num_channels: int, boundary_weight: float
) -> Tensor:
    def per_class(p: Tensor, t: Tensor) -> Tensor:
        dice = _dice(p, t)
        if boundary_weight <= 0.0:
            return dice
        return boundary_weight * _kervadec_boundary(p, t) + (1.0 - boundary_weight) * dice

    return _region_per_class(probs, labels, valid, num_channels, per_class)


# --- family builders: each returns a _FamilyFn closure capturing resolved knobs ---


def _build_ce_dice(r: ResolvedSemanticLoss) -> _FamilyFn:
    def fn(
        logits: Tensor, labels: Tensor, valid: Tensor, probs: Tensor, ii: int, c: int
    ) -> tuple[Tensor, Tensor]:
        return _multiclass_ce(logits, labels, ii), _dice_region(probs, labels, valid, c)

    return fn


def _build_focal_dice(r: ResolvedSemanticLoss) -> _FamilyFn:
    def fn(
        logits: Tensor, labels: Tensor, valid: Tensor, probs: Tensor, ii: int, c: int
    ) -> tuple[Tensor, Tensor]:
        ce = _multiclass_focal_ce(logits, labels, ii, r.focal_gamma, r.focal_alpha)
        return ce, _dice_region(probs, labels, valid, c)

    return fn


def _build_focal_tversky(r: ResolvedSemanticLoss) -> _FamilyFn:
    def fn(
        logits: Tensor, labels: Tensor, valid: Tensor, probs: Tensor, ii: int, c: int
    ) -> tuple[Tensor, Tensor]:
        ce = _multiclass_focal_ce(logits, labels, ii, r.focal_gamma, r.focal_alpha)
        region = _focal_tversky_region(probs, labels, valid, c, r.tversky_alpha, r.tversky_gamma)
        return ce, region

    return fn


def _build_boundary(r: ResolvedSemanticLoss) -> _FamilyFn:
    def fn(
        logits: Tensor, labels: Tensor, valid: Tensor, probs: Tensor, ii: int, c: int
    ) -> tuple[Tensor, Tensor]:
        region = _boundary_region(probs, labels, valid, c, r.boundary_weight)
        return _multiclass_ce(logits, labels, ii), region

    return fn


def _build_ce_only(r: ResolvedSemanticLoss) -> _FamilyFn:
    def fn(
        logits: Tensor, labels: Tensor, valid: Tensor, probs: Tensor, ii: int, c: int
    ) -> tuple[Tensor, Tensor]:
        return _multiclass_ce(logits, labels, ii), _zero_like(logits)

    return fn


def _build_dice_only(r: ResolvedSemanticLoss) -> _FamilyFn:
    def fn(
        logits: Tensor, labels: Tensor, valid: Tensor, probs: Tensor, ii: int, c: int
    ) -> tuple[Tensor, Tensor]:
        return _zero_like(logits), _dice_region(probs, labels, valid, c)

    return fn


SEM_FAMILY_BUILDERS: dict[str, _FamilyBuilder] = {
    "ce_dice": _build_ce_dice,
    "focal_dice": _build_focal_dice,
    "focal_tversky": _build_focal_tversky,
    "boundary": _build_boundary,
    "ce": _build_ce_only,
    "dice": _build_dice_only,
}


class SemanticLoss(nn.Module):
    """Multi-class semantic loss over a (B, K+1, H, W) foreground-logit volume."""

    def __init__(self, resolved: ResolvedSemanticLoss, family_fn: _FamilyFn) -> None:
        super().__init__()
        self.w_ce = float(resolved.w_ce)
        self.w_region = float(resolved.w_region)
        self.sem_family = resolved.sem_family
        self._family_fn = family_fn

    def forward(self, sem_logits: Tensor, targets: list[SemanticTarget]) -> dict[str, Tensor]:
        if sem_logits.ndim != 4:
            raise ValueError(
                f"sem_logits must be (B, K+1, H, W); got shape {tuple(sem_logits.shape)}"
            )
        if len(targets) != sem_logits.shape[0]:
            raise ValueError(f"targets length {len(targets)} != batch size {sem_logits.shape[0]}")
        size = (int(sem_logits.shape[-2]), int(sem_logits.shape[-1]))
        labels, ignore_index = _stack_downsample_labels(targets, size, sem_logits.device)
        valid = labels != ignore_index
        probs = F.softmax(sem_logits, dim=1)
        num_channels = int(sem_logits.shape[1])
        ce, region = self._family_fn(sem_logits, labels, valid, probs, ignore_index, num_channels)
        total = self.w_ce * ce + self.w_region * region
        return {"ce": ce, "region": region, "total": total}


def build_semantic_loss(resolved: ResolvedSemanticLoss) -> SemanticLoss:
    """Instantiate the SemanticLoss for the resolved sem_family + weights (§7.4)."""
    family_fn = SEM_FAMILY_BUILDERS[resolved.sem_family](resolved)
    return SemanticLoss(resolved, family_fn)
