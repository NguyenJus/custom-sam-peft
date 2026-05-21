"""Adapter + Hungarian matcher for SAM 3.1 training.

`meta_to_canonical` is the SINGLE point in the codebase that knows Meta's
native output dict key names. If Meta renames a field, only this function
breaks. Filled in by Task 5 once the actual key names are inspected against
a real `Sam3Wrapper` forward pass.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class CanonicalOutputs:
    """Adapter output of `meta_to_canonical`. Per-class (one prompt per call).

    Shapes (B = batch size, Q = number of decoder queries):
      obj_logits:   (B, Q)         per-query binary score (text-image similarity).
                                   Positive = "this query detects an instance of the
                                   current prompt class." From Meta's `pred_logits`
                                   after squeezing the trailing size-1 dim.
      pred_boxes:   (B, Q, 4)      normalized cx,cy,w,h in [0, 1].
      pred_masks:   (B, Q, H, W)   instance mask logits; H=W=288 at 1008-px input.
      img_presence: (B,)           image-level binary score "does this image contain
                                   any instance of the current prompt class." From
                                   Meta's `presence_logit_dec` after squeezing.
    """

    obj_logits: Tensor
    pred_boxes: Tensor
    pred_masks: Tensor
    img_presence: Tensor


def meta_to_canonical(outputs: dict[str, Tensor]) -> CanonicalOutputs:
    """Convert Meta SAM 3.1's native output dict to CanonicalOutputs.

    SINGLE point of contact for Meta key names. Update only this function if
    Meta renames a field.

    Meta keys (from `sam3.model.sam3_image.Sam3Image.forward_grounding`):
      "pred_logits":        (B, Q, 1)  per-query text-image similarity logit.
      "pred_boxes":         (B, Q, 4)  normalized cx,cy,w,h.
      "pred_masks":         (B, Q, H, W)  instance mask logits (288x288 at 1008px).
      "presence_logit_dec": (B, 1)     single global presence logit per image.

    The trailing size-1 dims of pred_logits and presence_logit_dec are squeezed.
    """
    pred_logits: Tensor = outputs["pred_logits"]
    presence: Tensor = outputs["presence_logit_dec"]
    return CanonicalOutputs(
        obj_logits=pred_logits.squeeze(-1),
        pred_boxes=outputs["pred_boxes"],
        pred_masks=outputs["pred_masks"],
        img_presence=presence.squeeze(-1),
    )


from scipy.optimize import linear_sum_assignment  # noqa: E402
from torch.nn.functional import interpolate  # noqa: E402

from custom_sam_peft.data.base import Instance  # noqa: E402


def _box_cxcywh_to_xyxy(box: Tensor) -> Tensor:
    cx, cy, w, h = box.unbind(-1)
    x1, y1 = cx - 0.5 * w, cy - 0.5 * h
    x2, y2 = cx + 0.5 * w, cy + 0.5 * h
    return torch.stack([x1, y1, x2, y2], dim=-1)


def _giou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """Generalized IoU between every pair in boxes1 (N,4) and boxes2 (M,4), xyxy."""
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2[None, :] - inter
    iou = inter / union.clamp(min=1e-7)
    lt_c = torch.min(boxes1[:, None, :2], boxes2[None, :, :2])
    rb_c = torch.max(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh_c = (rb_c - lt_c).clamp(min=0)
    area_c = wh_c[:, :, 0] * wh_c[:, :, 1]
    return iou - (area_c - union) / area_c.clamp(min=1e-7)


def _dice_cost(pred_masks: Tensor, tgt_masks: Tensor) -> Tensor:
    """Dice cost between every pred (Q, H, W) and target (N, H, W) mask. Returns (Q, N)."""
    p = pred_masks.sigmoid().flatten(1)  # (Q, H*W)
    t = tgt_masks.flatten(1).float()  # (N, H*W)
    num = 2 * p @ t.t()
    den = p.sum(-1)[:, None] + t.sum(-1)[None, :]
    return 1.0 - (num + 1.0) / (den + 1.0)


class HungarianMatcher:
    """DETR-style bipartite matcher for per-class SAM 3.1 outputs.

    No class-cost term: prompts encode class identity, so the only meaningful
    pairwise affinities are geometric (L1, GIoU on cxcywh boxes) and mask (Dice).
    Non-differentiable; called under `@torch.no_grad()`.
    """

    def __init__(
        self,
        lambda_l1: float,
        lambda_giou: float,
        lambda_mask: float,
    ) -> None:
        self.lambda_l1 = lambda_l1
        self.lambda_giou = lambda_giou
        self.lambda_mask = lambda_mask

    @torch.no_grad()
    def __call__(
        self,
        outputs: CanonicalOutputs,
        targets: list[list[Instance]],
    ) -> list[tuple[Tensor, Tensor]]:
        b = outputs.obj_logits.shape[0]
        mask_h, mask_w = outputs.pred_masks.shape[-2:]
        results: list[tuple[Tensor, Tensor]] = []
        for i in range(b):
            tgts = targets[i]
            if len(tgts) == 0:
                results.append(
                    (
                        torch.empty(0, dtype=torch.long),
                        torch.empty(0, dtype=torch.long),
                    )
                )
                continue
            # Matching is @torch.no_grad and the per-image volume is small
            # (queries x targets is at most a few hundred), so we upcast model
            # outputs to fp32 here. torch.cdist has no bf16/fp16 kernel on
            # CPU or CUDA (NotImplementedError "cdist_cuda" / "cdist" for
            # BFloat16), and downstream cost terms expect a consistent dtype.
            # Targets are already fp32 from the dataset.
            pred_boxes_i = outputs.pred_boxes[i].float()
            pred_masks_i = outputs.pred_masks[i].float()

            tgt_boxes = torch.stack([t.box for t in tgts]).to(outputs.pred_boxes.device)
            cost_l1 = torch.cdist(pred_boxes_i, tgt_boxes, p=1)
            cost_giou = -_giou(
                _box_cxcywh_to_xyxy(pred_boxes_i),
                _box_cxcywh_to_xyxy(tgt_boxes),
            )

            tgt_masks = torch.stack([t.mask for t in tgts]).to(outputs.pred_masks.device)
            tgt_masks_low = interpolate(
                tgt_masks[None].float(),
                size=(mask_h, mask_w),
                mode="bilinear",
                align_corners=False,
            )[0]
            cost_mask = _dice_cost(pred_masks_i, tgt_masks_low)

            cost = (
                self.lambda_l1 * cost_l1
                + self.lambda_giou * cost_giou
                + self.lambda_mask * cost_mask
            )
            row_ind, col_ind = linear_sum_assignment(cost.cpu().numpy())
            results.append(
                (
                    torch.as_tensor(row_ind, dtype=torch.long),
                    torch.as_tensor(col_ind, dtype=torch.long),
                )
            )
        return results
