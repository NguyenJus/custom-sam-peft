"""SAM 3.1 training losses (per-class, open-vocab head)."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn.functional import binary_cross_entropy_with_logits, interpolate

from custom_sam_peft.config.schema import LossConfig
from custom_sam_peft.data.base import Instance
from custom_sam_peft.models.matching import (
    CanonicalOutputs,
    HungarianMatcher,
    meta_to_canonical,
)

# Demoted from LossConfig to module-level constants per audit Section E (#93).
# These are not user-tunable: the values were the same across every config
# example. If you need to tune focal weights, re-promote with a YAML schema
# change and a tracked-feature issue.
_FOCAL_GAMMA = 2.0
_FOCAL_ALPHA = 0.25


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


def _box_cxcywh_to_xyxy(box: Tensor) -> Tensor:
    cx, cy, w, h = box.unbind(-1)
    return torch.stack([cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h], dim=-1)


def _giou_pairwise(b1: Tensor, b2: Tensor) -> Tensor:
    """Element-wise GIoU between two (N, 4) tensors in xyxy."""
    area1 = (b1[:, 2] - b1[:, 0]) * (b1[:, 3] - b1[:, 1])
    area2 = (b2[:, 2] - b2[:, 0]) * (b2[:, 3] - b2[:, 1])
    lt = torch.max(b1[:, :2], b2[:, :2])
    rb = torch.min(b1[:, 2:], b2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, 0] * wh[:, 1]
    union = area1 + area2 - inter
    iou = inter / union.clamp(min=1e-7)
    lt_c = torch.min(b1[:, :2], b2[:, :2])
    rb_c = torch.max(b1[:, 2:], b2[:, 2:])
    wh_c = (rb_c - lt_c).clamp(min=0)
    area_c = wh_c[:, 0] * wh_c[:, 1]
    return iou - (area_c - union) / area_c.clamp(min=1e-7)


def box_loss(pred: Tensor, target: Tensor) -> Tensor:
    """smoothL1 + (1 - GIoU) on matched box pairs. Boxes are normalized cxcywh."""
    smooth_l1 = torch.nn.functional.smooth_l1_loss(pred, target, reduction="mean")
    giou = _giou_pairwise(_box_cxcywh_to_xyxy(pred), _box_cxcywh_to_xyxy(target))
    return smooth_l1 + (1.0 - giou).mean()


def _focal_bce(logits: Tensor, targets: Tensor, gamma: float = 2.0, alpha: float = 0.25) -> Tensor:
    """Sigmoid focal BCE, mean-reduced. logits and targets broadcastable to the same shape."""
    p = logits.sigmoid()
    ce = binary_cross_entropy_with_logits(logits, targets.float(), reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    return (alpha_t * (1 - p_t).pow(gamma) * ce).mean()


def objectness_loss(
    obj_logits: Tensor,
    matched_mask: Tensor,
    gamma: float = 2.0,
    alpha: float = 0.25,
) -> Tensor:
    """Per-query binary focal BCE.

    obj_logits:    (B, Q) — Meta's `pred_logits` squeezed.
    matched_mask:  (B, Q) bool — True for queries assigned to some target by the matcher.
    """
    return _focal_bce(obj_logits, matched_mask.float(), gamma=gamma, alpha=alpha)


def presence_loss(
    img_presence: Tensor,
    image_has_target: Tensor,
) -> Tensor:
    """Image-level binary BCE on the global presence logit.

    img_presence:     (B,) — Meta's `presence_logit_dec` squeezed.
    image_has_target: (B,) bool — True if the image contains any instance of the
                      current prompt class.
    """
    return binary_cross_entropy_with_logits(img_presence, image_has_target.float())


def _gather_matched_boxes_masks(
    canonical: CanonicalOutputs,
    targets: list[list[Instance]],
    indices: list[tuple[Tensor, Tensor]],
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Concatenate matched (pred_box, tgt_box, pred_mask, tgt_mask) across the batch."""
    pred_boxes, tgt_boxes, pred_masks, tgt_masks = [], [], [], []
    for i, (pred_idx, tgt_idx) in enumerate(indices):
        if pred_idx.numel() == 0:
            continue
        pred_boxes.append(canonical.pred_boxes[i, pred_idx])
        tgt_boxes.append(
            torch.stack([targets[i][j].box for j in tgt_idx.tolist()]).to(
                canonical.pred_boxes.device
            )
        )
        pred_masks.append(canonical.pred_masks[i, pred_idx])
        tgt_masks.append(
            torch.stack([targets[i][j].mask for j in tgt_idx.tolist()]).to(
                canonical.pred_masks.device
            )
        )
    if not pred_boxes:
        empty_b = canonical.pred_boxes.new_zeros((0, 4))
        empty_m = canonical.pred_masks.new_zeros((0, 1, 1))
        return empty_b, empty_b, empty_m, empty_m
    return (
        torch.cat(pred_boxes),
        torch.cat(tgt_boxes),
        torch.cat(pred_masks),
        torch.cat(tgt_masks),
    )


def _matched_query_mask(
    canonical: CanonicalOutputs,
    indices: list[tuple[Tensor, Tensor]],
) -> Tensor:
    """Bool (B, Q): True where a query is matched to some target."""
    b, q = canonical.obj_logits.shape
    mask = torch.zeros((b, q), dtype=torch.bool, device=canonical.obj_logits.device)
    for i, (pred_idx, _) in enumerate(indices):
        if pred_idx.numel() > 0:
            mask[i, pred_idx] = True
    return mask


def _image_has_target(targets: list[list[Instance]], device: torch.device) -> Tensor:
    """Bool (B,): True if image has any target instance of the current prompt class."""
    return torch.tensor([len(t) > 0 for t in targets], dtype=torch.bool, device=device)


def total_loss(
    outputs: dict[str, Tensor],
    targets: list[list[Instance]],
    cfg: LossConfig,
) -> dict[str, Tensor]:
    """Run matching, compute per-component losses, return dict with 'total' summed.

    `outputs` is Meta's raw per-class forward dict. `targets[i]` is the list of
    GT instances of the prompt's class for image i (may be empty).
    """
    canonical = meta_to_canonical(outputs)
    matcher = HungarianMatcher(
        lambda_l1=0.0,
        lambda_giou=0.0,
        lambda_mask=cfg.matcher_weights.lambda_mask,
    )
    indices = matcher(canonical, targets)

    pred_boxes_m, tgt_boxes_m, pred_masks_m, tgt_masks_m = _gather_matched_boxes_masks(
        canonical, targets, indices
    )
    matched_mask = _matched_query_mask(canonical, indices)
    has_target = _image_has_target(targets, canonical.img_presence.device)

    zero = canonical.obj_logits.new_zeros(())
    losses: dict[str, Tensor] = {
        "mask": mask_loss(pred_masks_m, tgt_masks_m) if pred_masks_m.numel() > 0 else zero,
        "box": box_loss(pred_boxes_m, tgt_boxes_m) if pred_boxes_m.numel() > 0 else zero,
        "obj": objectness_loss(
            canonical.obj_logits,
            matched_mask,
            gamma=_FOCAL_GAMMA,
            alpha=_FOCAL_ALPHA,
        ),
        "presence": presence_loss(canonical.img_presence, has_target),
    }
    losses["total"] = (
        cfg.w_mask * losses["mask"]
        + cfg.w_box * losses["box"]
        + cfg.w_obj * losses["obj"]
        + cfg.w_presence * losses["presence"]
    )
    return losses
