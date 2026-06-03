"""Fast GPU dense-IoU AP proxy for lite (in-training) validation (#269).

Lite-mode validation pays the full pycocotools COCO-mAP cost in-loop, whose
dominant term is the serial single-threaded per-query RLE encode. The single
``mAP`` scalar lite eval produces drives only the in-loop control consumers
(best-checkpoint selection and early-stop; #264 decoupled the LR schedule from
the metric), each an ordering / threshold comparison against the metric's own
history. A monotone-faithful
GPU dense-IoU AP proxy is therefore sufficient in-loop; exact COCO mAP is kept
for the standalone / final report.

This module holds the pure functions of that proxy: a flattened area-sum IoU
matmul and the AP-from-IoU aggregation. No model / dataset / IO imports.

Five ordering-critical COCOeval rules replicated here (spike §2/§3.3):

  (a) iouThr sweep over the requested thresholds, with per-category AP being the
      mean over thresholds and overall ``mAP`` the mean over GT-bearing
      categories;
  (b) per-category pooling across the lite image subset into ONE PR curve (NOT a
      per-image-then-mean): a global score-descending sort, cumulative TP/FP,
      and an AP via precision-envelope all-point integration;
  (c) per IoU threshold INDEPENDENTLY, per-image score-descending greedy
      matching — each det grabs its highest-IoU unmatched GT with ``iou >= t``
      (one GT per pred); matched = TP, else FP; a match at one threshold is not
      reused at another;
  (d) no-GT-category masking before the mean: a category with zero GT across the
      subset is EXCLUDED from the mean (a category with GT but no TP counts
      AP 0 and is included);
  (e) re-truncation to the top ``max_dets`` by score, per (image, category),
      BEFORE matching (postprocess keeps a superset).

Simplifications taken (absolute-value-only; do not reorder checkpoints — spike
§2): all-point / trapezoidal PR-area instead of the 101-point interpolation;
single area range ("all"); a single ``maxDets`` cap (no 1/10 slices); no
``iscrowd`` modified-IoU branch (repo GT is always ``iscrowd=0``).

The lite ``mAP`` produced here is a PROXY, not exact COCO mAP. It is gated by the
§8.2 pre-enablement validation gate (Spearman rho >= 0.95 over real checkpoints
plus a ``min_delta`` scale check), filed as a follow-up and NOT run yet.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np
import torch
from torch import Tensor

# Float arrays of unconstrained shape/precision (the IoU matrix is 2-D, scores
# 1-D, and entries arrive as float32 from CUDA or float64 from numpy) — mirrors
# the repo's typed-ndarray idiom (e.g. eval/metrics.py) for the mypy gate.
FloatArray = np.ndarray[Any, np.dtype[np.floating[Any]]]


class ProxyEntry(NamedTuple):
    """One ``(image, category)`` group's dense-IoU evidence for the proxy.

    Fields:
        image_id: source image id (matching runs per image).
        category_id: category id (AP pools per category across images).
        iou: ``(m, M)`` float IoU matrix, m preds x M GT for this group.
        scores: ``(m,)`` detection scores aligned with the rows of ``iou``.
    """

    image_id: int
    category_id: int
    iou: FloatArray
    scores: FloatArray


def dense_iou_matrix(pred_masks_bin: Tensor, gt_masks_bin: Tensor) -> Tensor:
    """Dense pred x GT IoU via a flattened area-sum matmul (fp32 accumulation).

    Args:
        pred_masks_bin: ``(m, H, W)`` bool tensor of binarized predictions,
            on-device.
        gt_masks_bin: ``(M, H, W)`` bool tensor of binarized GT, same device.

    Returns:
        ``(m, M)`` float32 IoU tensor on the same device. The naive
        ``(m, M, H, W)`` intersection is never materialized; intersection is a
        matmul and areas are row-sums, all accumulated in fp32.
    """
    device = pred_masks_bin.device
    m = pred_masks_bin.shape[0]
    big_m = gt_masks_bin.shape[0]
    if m == 0 or big_m == 0:
        return torch.zeros((m, big_m), dtype=torch.float32, device=device)

    pred_f = pred_masks_bin.flatten(1).float()  # (m, H*W)
    gt_f = gt_masks_bin.flatten(1).float()  # (M, H*W)
    inter = pred_f @ gt_f.T  # (m, M)
    pred_a = pred_f.sum(1)  # (m,)
    gt_a = gt_f.sum(1)  # (M,)
    union = pred_a[:, None] + gt_a[None, :] - inter
    return inter / union.clamp(min=1)


def _ap_all_point(recall: FloatArray, precision: FloatArray) -> float:
    """All-point AP: precision-envelope (monotone) integrated over recall.

    Prepends a recall-0 anchor, takes the running max of precision from the high
    end (the envelope), then sums ``(recall[i] - recall[i-1]) * env[i]``.
    """
    rec = np.concatenate(([0.0], recall))
    prec = np.concatenate(([0.0], precision))
    env = np.maximum.accumulate(prec[::-1])[::-1]
    return float(np.sum(np.diff(rec) * env[1:]))


def _category_ap_at_threshold(cat_entries: list[ProxyEntry], thr: float, npig: int) -> float:
    """Pooled-PR-curve AP for one category at one IoU threshold (rules b, c, e).

    Per-image greedy matching, score-pooled cumulative TP/FP across images.
    """
    pooled_scores: list[float] = []
    pooled_tp: list[bool] = []
    for entry in cat_entries:
        iou = entry.iou  # (m, M)
        big_m = iou.shape[1]
        gt_matched = np.zeros(big_m, dtype=bool)
        # score-descending, stable (mergesort) within this image.
        order = np.argsort(-entry.scores, kind="mergesort")
        for det in order:
            row = iou[det]
            best_iou = -1.0
            best_gt = -1
            for gt_idx in range(big_m):
                if gt_matched[gt_idx]:
                    continue
                if row[gt_idx] >= thr and row[gt_idx] > best_iou:
                    best_iou = float(row[gt_idx])
                    best_gt = gt_idx
            is_tp = best_gt >= 0
            if is_tp:
                gt_matched[best_gt] = True
            pooled_scores.append(float(entry.scores[det]))
            pooled_tp.append(is_tp)

    if not pooled_scores:
        return 0.0

    scores_arr = np.asarray(pooled_scores, dtype=np.float64)
    tp_arr = np.asarray(pooled_tp, dtype=np.float64)
    # global score-descending, stable (mergesort).
    order = np.argsort(-scores_arr, kind="mergesort")
    tp_sorted = tp_arr[order]
    fp_sorted = 1.0 - tp_sorted
    cum_tp = np.cumsum(tp_sorted)
    cum_fp = np.cumsum(fp_sorted)
    recall = cum_tp / npig
    precision = cum_tp / np.maximum(cum_tp + cum_fp, 1.0)
    return _ap_all_point(recall, precision)


def proxy_map_from_iou(
    entries: list[ProxyEntry], iou_thresholds: list[float], max_dets: int
) -> dict[str, float]:
    """Aggregate per-(image, category) IoU evidence into a proxy ``mAP`` dict.

    Replicates the five ordering-critical COCOeval rules (see module docstring).
    Returns ``{"mAP": ...}`` plus ``mAP_50`` / ``mAP_75`` iff 0.5 / 0.75 are in
    ``iou_thresholds``. Empty ``entries`` returns zeros.
    """
    keys = ["mAP"]
    if 0.5 in iou_thresholds:
        keys.append("mAP_50")
    if 0.75 in iou_thresholds:
        keys.append("mAP_75")

    if not entries:
        return dict.fromkeys(keys, 0.0)

    # rule (e): re-truncate to top-max_dets by score per (image, category).
    by_cat: dict[int, list[ProxyEntry]] = {}
    for entry in entries:
        if entry.scores.shape[0] > max_dets:
            keep = np.argsort(-entry.scores, kind="mergesort")[:max_dets]
            entry = ProxyEntry(
                entry.image_id, entry.category_id, entry.iou[keep], entry.scores[keep]
            )
        by_cat.setdefault(entry.category_id, []).append(entry)

    # per category: mean-over-thresholds AP, and per-threshold AP for mAP_50/75.
    cat_mean_ap: list[float] = []
    cat_ap_at: dict[float, list[float]] = {t: [] for t in iou_thresholds}
    for cat_entries in by_cat.values():
        # rule (d): npig per category = total GT across that category's entries.
        npig = int(sum(e.iou.shape[1] for e in cat_entries))
        if npig == 0:
            continue  # no-GT category masked before the mean
        per_thr = [_category_ap_at_threshold(cat_entries, t, npig) for t in iou_thresholds]
        cat_mean_ap.append(float(np.mean(per_thr)))
        for t, ap in zip(iou_thresholds, per_thr, strict=True):
            cat_ap_at[t].append(ap)

    out: dict[str, float] = {}
    out["mAP"] = float(np.mean(cat_mean_ap)) if cat_mean_ap else 0.0
    if "mAP_50" in keys:
        out["mAP_50"] = float(np.mean(cat_ap_at[0.5])) if cat_ap_at[0.5] else 0.0
    if "mAP_75" in keys:
        out["mAP_75"] = float(np.mean(cat_ap_at[0.75])) if cat_ap_at[0.75] else 0.0
    return out
