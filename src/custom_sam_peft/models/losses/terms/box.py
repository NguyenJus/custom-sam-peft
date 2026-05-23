"""Box-axis loss term classes (spec §8.2)."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn.functional import smooth_l1_loss


def _cxcywh_to_xyxy(box: Tensor) -> Tensor:
    cx, cy, w, h = box.unbind(-1)
    return torch.stack([cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h], dim=-1)


def _giou_pairwise(b1_xyxy: Tensor, b2_xyxy: Tensor) -> Tensor:
    area1 = (b1_xyxy[:, 2] - b1_xyxy[:, 0]) * (b1_xyxy[:, 3] - b1_xyxy[:, 1])
    area2 = (b2_xyxy[:, 2] - b2_xyxy[:, 0]) * (b2_xyxy[:, 3] - b2_xyxy[:, 1])
    lt = torch.max(b1_xyxy[:, :2], b2_xyxy[:, :2])
    rb = torch.min(b1_xyxy[:, 2:], b2_xyxy[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, 0] * wh[:, 1]
    union = area1 + area2 - inter
    iou = inter / union.clamp(min=1e-7)
    lt_c = torch.min(b1_xyxy[:, :2], b2_xyxy[:, :2])
    rb_c = torch.max(b1_xyxy[:, 2:], b2_xyxy[:, 2:])
    wh_c = (rb_c - lt_c).clamp(min=0)
    area_c = wh_c[:, 0] * wh_c[:, 1]
    return iou - (area_c - union) / area_c.clamp(min=1e-7)


class _BoxTermBase(nn.Module):
    def __init__(self, **_unused: float) -> None:
        super().__init__()


class L1GIoULoss(_BoxTermBase):
    """Today's box_loss: smoothL1(p, t) + (1 - GIoU(p, t)).mean()."""

    def forward(self, pred_cxcywh: Tensor, target_cxcywh: Tensor) -> Tensor:
        if pred_cxcywh.numel() == 0:
            return pred_cxcywh.new_zeros(())
        l1 = smooth_l1_loss(pred_cxcywh, target_cxcywh, reduction="mean")
        giou = _giou_pairwise(_cxcywh_to_xyxy(pred_cxcywh), _cxcywh_to_xyxy(target_cxcywh))
        return l1 + (1.0 - giou).mean()


class GIoUOnlyLoss(_BoxTermBase):
    """GIoU-only box loss without the L1 regularisation term."""

    def forward(self, pred_cxcywh: Tensor, target_cxcywh: Tensor) -> Tensor:
        if pred_cxcywh.numel() == 0:
            return pred_cxcywh.new_zeros(())
        giou = _giou_pairwise(_cxcywh_to_xyxy(pred_cxcywh), _cxcywh_to_xyxy(target_cxcywh))
        return (1.0 - giou).mean()


class CIoULoss(_BoxTermBase):
    """Zheng et al. 2020 — IoU - ρ²(p,t)/c² - α·v with aspect-ratio penalty."""

    def forward(self, pred_cxcywh: Tensor, target_cxcywh: Tensor) -> Tensor:
        if pred_cxcywh.numel() == 0:
            return pred_cxcywh.new_zeros(())
        p_xyxy = _cxcywh_to_xyxy(pred_cxcywh)
        t_xyxy = _cxcywh_to_xyxy(target_cxcywh)
        # IoU
        area1 = (p_xyxy[:, 2] - p_xyxy[:, 0]) * (p_xyxy[:, 3] - p_xyxy[:, 1])
        area2 = (t_xyxy[:, 2] - t_xyxy[:, 0]) * (t_xyxy[:, 3] - t_xyxy[:, 1])
        lt = torch.max(p_xyxy[:, :2], t_xyxy[:, :2])
        rb = torch.min(p_xyxy[:, 2:], t_xyxy[:, 2:])
        wh = (rb - lt).clamp(min=0)
        inter = wh[:, 0] * wh[:, 1]
        union = area1 + area2 - inter
        iou = inter / union.clamp(min=1e-7)
        # Enclosing-box diagonal²
        lt_c = torch.min(p_xyxy[:, :2], t_xyxy[:, :2])
        rb_c = torch.max(p_xyxy[:, 2:], t_xyxy[:, 2:])
        c2 = (rb_c - lt_c).pow(2).sum(dim=-1).clamp(min=1e-7)
        # Center distance²
        rho2 = (pred_cxcywh[:, :2] - target_cxcywh[:, :2]).pow(2).sum(dim=-1)
        # Aspect-ratio penalty v and α
        w1, h1 = pred_cxcywh[:, 2].clamp(min=1e-7), pred_cxcywh[:, 3].clamp(min=1e-7)
        w2, h2 = target_cxcywh[:, 2].clamp(min=1e-7), target_cxcywh[:, 3].clamp(min=1e-7)
        v = (4.0 / (math.pi**2)) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)).pow(2)
        alpha = v / (1.0 - iou + v).clamp(min=1e-7)
        ciou = iou - rho2 / c2 - alpha * v
        return (1.0 - ciou).mean()
