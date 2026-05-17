"""SAM 3.1 training losses (per-class, open-vocab head)."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn.functional import binary_cross_entropy_with_logits, interpolate

from esam3.config.schema import LossConfig  # noqa: F401
from esam3.data.base import Instance  # noqa: F401
from esam3.models.matching import (  # noqa: F401
    CanonicalOutputs,
    HungarianMatcher,
    meta_to_canonical,
)


def _dice_loss(pred_logits: Tensor, target: Tensor) -> Tensor:
    p = pred_logits.sigmoid().flatten(1)
    t = target.flatten(1).float()
    num = 2 * (p * t).sum(-1) + 1.0
    den = p.sum(-1) + t.sum(-1) + 1.0
    return (1.0 - num / den).mean()


def mask_loss(pred: Tensor, target: Tensor) -> Tensor:
    """0.5 · Dice + 0.5 · BCE on matched mask pairs.

    `pred` and `target` are (N, H_p, W_p) and (N, H_t, W_t). If the spatial
    shapes differ, `pred` is bilinear-upsampled to the target resolution.
    """
    if pred.shape[-2:] != target.shape[-2:]:
        pred = interpolate(
            pred[:, None], size=target.shape[-2:], mode="bilinear", align_corners=False
        )[:, 0]
    bce = binary_cross_entropy_with_logits(pred, target.float())
    dice = _dice_loss(pred, target)
    return 0.5 * dice + 0.5 * bce


def box_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError("filled in by spec: spec/model-loading")


def objectness_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError("filled in by spec: spec/model-loading")


def total_loss(outputs: dict[str, torch.Tensor], targets: dict[str, torch.Tensor]) -> torch.Tensor:
    raise NotImplementedError("filled in by spec: spec/model-loading")
