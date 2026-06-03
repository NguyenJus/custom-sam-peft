"""Pure-function postprocess: model output dict → COCO results entries.

No model, no dataset, no I/O. All tensor → COCO conversion lives here so it
can be unit-tested against the tiny_sam3_stub without touching pycocotools'
scoring machinery.
"""

from __future__ import annotations

import numpy as np
import pycocotools.mask as mask_utils
import torch
import torch.nn.functional as F
from torch import Tensor

from custom_sam_peft.eval import _profile  # TEMP #250 Phase 1 — removed in Phase 2


def _denorm_cxcywh_to_xywh(boxes_norm: Tensor, original_hw: tuple[int, int]) -> Tensor:
    """boxes_norm: (N, 4) normalized cxcywh in [0, 1]. Returns (N, 4) absolute xywh, clamped."""
    h, w = original_hw
    cx = boxes_norm[:, 0] * w
    cy = boxes_norm[:, 1] * h
    bw = boxes_norm[:, 2] * w
    bh = boxes_norm[:, 3] * h
    x = (cx - bw / 2).clamp(min=0.0, max=float(w))
    y = (cy - bh / 2).clamp(min=0.0, max=float(h))
    x2 = (cx + bw / 2).clamp(min=0.0, max=float(w))
    y2 = (cy + bh / 2).clamp(min=0.0, max=float(h))
    bw = (x2 - x).clamp(min=0.0)
    bh = (y2 - y).clamp(min=0.0)
    return torch.stack([x, y, bw, bh], dim=-1)


def _upsample_mask_logits(masks_logits: Tensor, original_hw: tuple[int, int]) -> Tensor:
    """masks_logits: (N, H_m, W_m) float. Returns (N, H, W) bilinear-upsampled."""
    return F.interpolate(
        masks_logits.unsqueeze(1),
        size=original_hw,
        mode="bilinear",
        align_corners=False,
    ).squeeze(1)


def queries_to_coco_results(
    outputs: dict[str, Tensor],
    image_id: int,
    category_id: int,
    original_hw: tuple[int, int],
    mask_threshold: float = 0.0,
    *,
    max_dets: int | None = None,
) -> list[dict[str, object]]:
    """Convert one per-class forward output into a list of COCO results entries.

    Required keys in ``outputs``: ``pred_logits`` (1, N, 1), ``pred_boxes``
    (1, N, 4) normalized cxcywh, ``pred_masks`` (1, N, H_m, W_m) logits,
    ``presence_logit_dec`` (1, 1).

    Score = ``sigmoid(pred_logits) * sigmoid(presence_logit_dec)``.
    Masks are bilinear-upsampled to ``original_hw`` and binarized at
    ``mask_threshold`` on the logits.

    When ``max_dets`` is given, keep only queries whose score is >= the
    ``max_dets``-th-highest score (a threshold, NOT exactly ``max_dets`` —
    boundary ties are kept as a superset). This is mAP-EXACT: pycocotools'
    COCOeval already truncates to ``max(params.maxDets)`` (=100) detections by
    score per (image, category), so dropping the strictly-lower-scored remainder
    cannot change the metric. Citation: pycocotools maxDets=[1,10,100] semantics.
    ``max_dets=None`` (default) returns ALL queries unchanged (predict/visualize
    need every query).
    """
    pred_logits = outputs["pred_logits"]
    pred_boxes = outputs["pred_boxes"]
    pred_masks = outputs["pred_masks"]
    presence = outputs["presence_logit_dec"]

    if pred_logits.shape[0] != 1:
        raise ValueError(f"postprocess expects batch=1; got {pred_logits.shape[0]}")
    if len(original_hw) != 2:
        raise ValueError(f"original_hw must be (H, W); got {original_hw!r}")

    n = pred_logits.shape[1]
    if n == 0:
        return []

    # --- scores ---
    p_obj = torch.sigmoid(pred_logits.float()).squeeze(-1).squeeze(0)  # (N,)
    p_presence = torch.sigmoid(presence.float()).reshape(())  # scalar
    scores = p_obj * p_presence  # (N,)
    if not torch.isfinite(scores).all():
        raise RuntimeError(
            "non-finite scores in postprocess; check model outputs "
            "(pred_logits or presence_logit_dec contains NaN/Inf)"
        )

    # --- top-N filter (mAP-exact; spec §3.3) ---
    # Rank by the SAME score COCOeval uses; keep all queries with score >= the
    # max_dets-th-highest score (threshold, not exactly max_dets) so the survivor
    # set is a SUPERSET of whatever 100 the scorer would keep under its own
    # tie-break. Done BEFORE upsample/transfer/RLE so those costs only touch
    # survivors.
    keep_idx: Tensor | None = None
    if max_dets is not None and n > max_dets:
        kth = torch.topk(scores, max_dets).values.min()  # the max_dets-th-highest score
        keep_idx = (scores >= kth).nonzero(as_tuple=False).squeeze(-1)  # (M,), M >= max_dets
        scores = scores[keep_idx]

    m = scores.shape[0]  # survivor count (== n when no filter / n <= max_dets)

    # --- boxes ---
    boxes_norm = pred_boxes.float().squeeze(0)  # (N, 4)
    if not torch.isfinite(boxes_norm).all():
        raise RuntimeError(
            "non-finite box coordinates in postprocess; check model outputs "
            "(pred_boxes contains NaN/Inf)"
        )
    if keep_idx is not None:
        boxes_norm = boxes_norm[keep_idx]
    boxes_xywh = _denorm_cxcywh_to_xywh(boxes_norm, original_hw)  # (M, 4)

    # --- masks ---
    _profile.note(N=int(n), mask_logit_hw=tuple(pred_masks.shape[-2:]))  # TEMP #250
    masks_logits = pred_masks.float().squeeze(0)  # (N, H_m, W_m)
    if not torch.isfinite(masks_logits).all():
        raise RuntimeError(
            "non-finite mask logits in postprocess; check model outputs "
            "(pred_masks contains NaN/Inf)"
        )
    if keep_idx is not None:
        masks_logits = masks_logits[keep_idx]
    with _profile.bucket("mask_upsample"):  # TEMP #250
        masks_up = _upsample_mask_logits(masks_logits, original_hw)  # (M, H, W)
    with _profile.bucket("transfer_binarize"):  # TEMP #250
        masks_bin = (masks_up > mask_threshold).cpu().numpy()  # (M, H, W) bool

    entries: list[dict[str, object]] = []
    with _profile.bucket("transfer_binarize"):  # TEMP #250 (box/score device->host)
        boxes_list = boxes_xywh.cpu().tolist()
        scores_list = scores.cpu().tolist()
    with _profile.bucket("rle_encode"):  # TEMP #250
        # Batched RLE: encode all survivor masks in ONE pycocotools call.
        # masks_bin is (M, H, W) bool; encode wants Fortran (H, W, M) uint8.
        if m:
            masks_fortran = np.asfortranarray(
                np.ascontiguousarray(masks_bin).transpose(1, 2, 0).astype(np.uint8)
            )
            rles = mask_utils.encode(masks_fortran)  # list[M] of RLE dicts
        else:
            rles = []
        for i in range(m):
            rle = rles[i]
            counts = rle["counts"]
            rle["counts"] = counts.decode("ascii") if isinstance(counts, bytes) else counts
            entries.append(
                {
                    "image_id": int(image_id),
                    "category_id": int(category_id),
                    "bbox": [float(v) for v in boxes_list[i]],
                    "score": float(scores_list[i]),
                    "segmentation": rle,
                }
            )
    return entries
