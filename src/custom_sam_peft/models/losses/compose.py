"""Loss-bundle composer (spec §8). Builds a LossBundle from a ResolvedLosses."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from custom_sam_peft.config._internal import MatcherWeights
from custom_sam_peft.data.base import Instance
from custom_sam_peft.models.losses.presets import ResolvedLosses
from custom_sam_peft.models.losses.terms import box as box_terms
from custom_sam_peft.models.losses.terms import mask as mask_terms
from custom_sam_peft.models.losses.terms import obj as obj_terms
from custom_sam_peft.models.losses.terms import presence as presence_terms
from custom_sam_peft.models.matching import (
    CanonicalOutputs,
    HungarianMatcher,
    meta_to_canonical,
)

# ---------------------------------------------------------------------------
# Term registries — keyed by the family literal strings from schema.py.
# Missing keys raise KeyError, which is unreachable because pydantic validates
# the literal at config-load time.
# ---------------------------------------------------------------------------

_MASK_TERMS: dict[str, type] = {
    "bce": mask_terms.BCELoss,
    "dice": mask_terms.DiceLoss,
    "dice_bce": mask_terms.DiceBCELoss,
    "focal_bce": mask_terms.FocalBCELoss,
    "focal_dice": mask_terms.FocalDiceLoss,
    "focal_tversky": mask_terms.FocalTverskyLoss,
    "boundary": mask_terms.BoundaryLoss,
}

_BOX_TERMS: dict[str, type] = {
    "l1_giou": box_terms.L1GIoULoss,
    "giou_only": box_terms.GIoUOnlyLoss,
    "ciou": box_terms.CIoULoss,
}

_OBJ_TERMS: dict[str, type] = {
    "bce": obj_terms.BCELoss,
    "focal_bce": obj_terms.FocalBCELoss,
}

_PRESENCE_TERMS: dict[str, type] = {
    "bce": presence_terms.BCELoss,
    "focal_bce": presence_terms.FocalBCELoss,
}


# ---------------------------------------------------------------------------
# Helpers — moved verbatim from the pre-#112 monolith losses.py
# ---------------------------------------------------------------------------


def _gather_matched_boxes_masks(
    canonical: CanonicalOutputs,
    targets: list[list[Instance]],
    indices: list[tuple[Tensor, Tensor]],
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
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
    b, q = canonical.obj_logits.shape
    mask = torch.zeros((b, q), dtype=torch.bool, device=canonical.obj_logits.device)
    for i, (pred_idx, _) in enumerate(indices):
        if pred_idx.numel() > 0:
            mask[i, pred_idx] = True
    return mask


def _image_has_target(targets: list[list[Instance]], device: torch.device) -> Tensor:
    return torch.tensor([len(t) > 0 for t in targets], dtype=torch.bool, device=device)


# ---------------------------------------------------------------------------
# LossBundle + builder
# ---------------------------------------------------------------------------


class LossBundle:
    """Pre-instantiated four-term loss bundle. Built once per trainer init."""

    def __init__(
        self,
        mask_term: torch.nn.Module,
        box_term: torch.nn.Module,
        obj_term: torch.nn.Module,
        presence_term: torch.nn.Module,
        *,
        weights: tuple[float, float, float, float],
        matcher_weights: MatcherWeights,
    ) -> None:
        self.mask_term = mask_term
        self.box_term = box_term
        self.obj_term = obj_term
        self.presence_term = presence_term
        self.w_mask, self.w_box, self.w_obj, self.w_presence = weights
        self.matcher = HungarianMatcher(
            lambda_l1=matcher_weights.lambda_l1,
            lambda_giou=matcher_weights.lambda_giou,
            lambda_mask=matcher_weights.lambda_mask,
        )

    def forward(
        self,
        outputs: dict[str, Tensor],
        targets: list[list[Instance]],
    ) -> dict[str, Tensor]:
        """Run the matcher and four-term forward.

        Returns a dict of {mask, box, obj, presence, total} losses.
        """
        canonical = meta_to_canonical(outputs)
        indices = self.matcher(canonical, targets)
        pred_boxes_m, tgt_boxes_m, pred_masks_m, tgt_masks_m = _gather_matched_boxes_masks(
            canonical, targets, indices
        )
        matched_mask = _matched_query_mask(canonical, indices)
        has_target = _image_has_target(targets, canonical.img_presence.device)
        zero = canonical.obj_logits.new_zeros(())
        losses: dict[str, Tensor] = {
            "mask": (
                self.mask_term(pred_masks_m, tgt_masks_m) if pred_masks_m.numel() > 0 else zero
            ),
            "box": (self.box_term(pred_boxes_m, tgt_boxes_m) if pred_boxes_m.numel() > 0 else zero),
            "obj": self.obj_term(canonical.obj_logits, matched_mask),
            "presence": self.presence_term(canonical.img_presence, has_target),
        }
        losses["total"] = (
            self.w_mask * losses["mask"]
            + self.w_box * losses["box"]
            + self.w_obj * losses["obj"]
            + self.w_presence * losses["presence"]
        )
        return losses


def build_loss_bundle(resolved: ResolvedLosses) -> LossBundle:
    """Instantiate the four chosen term classes from the resolved knob set."""
    hp: dict[str, Any] = dict(
        focal_gamma=resolved.focal_gamma,
        focal_alpha=resolved.focal_alpha,
        tversky_alpha=resolved.tversky_alpha,
        tversky_gamma=resolved.tversky_gamma,
        boundary_weight=resolved.boundary_weight,
    )
    mask_term = _MASK_TERMS[resolved.mask_family](**hp)
    box_term = _BOX_TERMS[resolved.box_family](**hp)
    obj_term = _OBJ_TERMS[resolved.obj_family](**hp)
    presence_term = _PRESENCE_TERMS[resolved.presence_family](**hp)
    weights = (resolved.w_mask, resolved.w_box, resolved.w_obj, resolved.w_presence)
    return LossBundle(
        mask_term,
        box_term,
        obj_term,
        presence_term,
        weights=weights,
        matcher_weights=resolved.matcher_weights,
    )
