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


def _logits_to_rle(mask_bin: np.ndarray[tuple[int, int], np.dtype[np.uint8]]) -> dict[str, object]:
    """mask_bin: (H, W) bool/uint8. Returns pycocotools RLE dict with ascii counts."""
    rle: dict[str, object] = mask_utils.encode(np.asfortranarray(mask_bin.astype(np.uint8)))
    counts = rle["counts"]
    rle["counts"] = counts.decode("ascii") if isinstance(counts, bytes) else counts
    return rle


def queries_to_coco_results(
    outputs: dict[str, Tensor],
    image_id: int,
    category_id: int,
    original_hw: tuple[int, int],
    mask_threshold: float = 0.0,
) -> list[dict[str, object]]:
    """Convert one per-class forward output into a list of COCO results entries.

    Required keys in ``outputs``: ``pred_logits`` (1, N, 1), ``pred_boxes``
    (1, N, 4) normalized cxcywh, ``pred_masks`` (1, N, H_m, W_m) logits,
    ``presence_logit_dec`` (1, 1).

    Score = ``sigmoid(pred_logits) * sigmoid(presence_logit_dec)``.
    Masks are bilinear-upsampled to ``original_hw`` and binarized at
    ``mask_threshold`` on the logits.

    All queries are returned; no filtering or NMS applied.
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

    # --- boxes ---
    boxes_norm = pred_boxes.float().squeeze(0)  # (N, 4)
    if not torch.isfinite(boxes_norm).all():
        raise RuntimeError(
            "non-finite box coordinates in postprocess; check model outputs "
            "(pred_boxes contains NaN/Inf)"
        )
    boxes_xywh = _denorm_cxcywh_to_xywh(boxes_norm, original_hw)  # (N, 4)

    # --- masks ---
    masks_logits = pred_masks.float().squeeze(0)  # (N, H_m, W_m)
    if not torch.isfinite(masks_logits).all():
        raise RuntimeError(
            "non-finite mask logits in postprocess; check model outputs "
            "(pred_masks contains NaN/Inf)"
        )
    masks_up = _upsample_mask_logits(masks_logits, original_hw)  # (N, H, W)
    masks_bin = (masks_up > mask_threshold).cpu().numpy()  # (N, H, W) bool

    entries: list[dict[str, object]] = []
    boxes_list = boxes_xywh.cpu().tolist()
    scores_list = scores.cpu().tolist()
    for i in range(n):
        entries.append(
            {
                "image_id": int(image_id),
                "category_id": int(category_id),
                "bbox": [float(v) for v in boxes_list[i]],
                "score": float(scores_list[i]),
                "segmentation": _logits_to_rle(masks_bin[i]),
            }
        )
    return entries
