"""Pure-function postprocess: model output dict → COCO results entries.

No model, no dataset, no I/O. All tensor → COCO conversion lives here so it
can be unit-tested against the tiny_sam3_stub without touching pycocotools'
scoring machinery.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pycocotools.mask as mask_utils
import torch
import torch.nn.functional as F
from torch import Tensor

from custom_sam_peft import profiling


# Module-level pinned-buffer pool. Eval is single-threaded (pycocotools holds the
# GIL; CPU-parallelism is a documented dead end, see #253), so no lock is needed.
class _PinnedHostBuffer:
    """Reusable, grow-only pinned host buffer for D2H mask transfer."""

    def __init__(self) -> None:
        self._buf: Tensor | None = None
        self._device: torch.device | None = None

    def view_for(self, numel: int, device: torch.device) -> Tensor:
        """Return a contiguous 1-D pinned bool slice of length ``numel``,
        pinned for ``device``. Grows (and re-pins) only when required."""
        if self._buf is None or self._buf.numel() < numel or self._device != device:
            self._buf = torch.empty(numel, dtype=torch.bool, pin_memory=True)
            self._device = device
        return self._buf[:numel]


_PINNED_HOST = _PinnedHostBuffer()


_BoolArray = np.ndarray[Any, np.dtype[np.bool_]]


def _binarize_to_host(masks_up: Tensor, mask_threshold: float) -> _BoolArray:
    """Threshold ``masks_up`` to bool and copy to host.

    Bit-identical to ``(masks_up > mask_threshold).cpu().numpy()``. On CUDA, uses a
    reused pinned host buffer with a ``non_blocking=True`` copy + explicit
    synchronize; on CPU, falls back to the plain ``.numpy()`` path (no pinned
    machinery), keeping stub/CPU eval and tests working unchanged.

    WARNING: the CUDA path returns a numpy VIEW into the reused pinned buffer. The
    caller MUST consume it before the next ``_binarize_to_host`` call. See spec
    invariant 2 (the immediately-following RLE block copies it via
    ``np.ascontiguousarray(...).astype(np.uint8)``).
    """
    gpu_bool = masks_up > mask_threshold  # bool, contiguous, on input's device

    if gpu_bool.device.type != "cuda":
        # CPU fallback: bit-identical to the old .cpu().numpy() path.
        return gpu_bool.numpy()  # ndarray[bool_] at runtime

    numel = gpu_bool.numel()
    flat = _PINNED_HOST.view_for(numel, gpu_bool.device)
    view = flat.view(gpu_bool.shape)  # (M, H, W) bool, pinned host
    view.copy_(gpu_bool, non_blocking=True)
    torch.cuda.synchronize()  # required before the host reads the numpy
    return view.numpy()  # zero-copy view of the pinned buffer


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


def score_and_topk_filter(
    outputs: dict[str, Tensor], max_dets: int | None
) -> tuple[Tensor, Tensor | None]:
    """Compute per-query scores and the top-``max_dets`` survivor filter.

    Score = ``sigmoid(pred_logits) * sigmoid(presence_logit_dec)`` (fp32, on the
    same device as the inputs). Raises ``RuntimeError`` if any score is non-finite.

    When ``max_dets`` is given and there are more than ``max_dets`` queries, keep
    only queries whose score is >= the ``max_dets``-th-highest score (a threshold,
    NOT exactly ``max_dets`` — boundary ties are kept as a SUPERSET). Returns
    ``(scores, keep_idx)`` where ``scores`` is the post-filter ``(M,)`` score
    vector and ``keep_idx`` is the ``(M,)`` survivor index tensor, or ``None``
    when no filter was applied (``max_dets is None`` or ``n <= max_dets``). When
    ``keep_idx is None`` the returned ``scores`` covers ALL ``n`` queries.

    Shared by both the exact RLE path (``queries_to_coco_results``) and the lite
    dense-IoU proxy so the two can never drift on score formula or tie-break.
    """
    pred_logits = outputs["pred_logits"]
    presence = outputs["presence_logit_dec"]

    p_obj = torch.sigmoid(pred_logits.float()).squeeze(-1).squeeze(0)  # (N,)
    p_presence = torch.sigmoid(presence.float()).reshape(())  # scalar
    scores = p_obj * p_presence  # (N,)
    if not torch.isfinite(scores).all():
        raise RuntimeError(
            "non-finite scores in postprocess; check model outputs "
            "(pred_logits or presence_logit_dec contains NaN/Inf)"
        )

    n = scores.shape[0]
    keep_idx: Tensor | None = None
    if max_dets is not None and n > max_dets:
        kth = torch.topk(scores, max_dets).values.min()  # the max_dets-th-highest score
        keep_idx = (scores >= kth).nonzero(as_tuple=False).squeeze(-1)  # (M,), M >= max_dets
        scores = scores[keep_idx]
    return scores, keep_idx


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

    if pred_logits.shape[0] != 1:
        raise ValueError(f"postprocess expects batch=1; got {pred_logits.shape[0]}")
    if len(original_hw) != 2:
        raise ValueError(f"original_hw must be (H, W); got {original_hw!r}")

    n = pred_logits.shape[1]
    if n == 0:
        return []

    # --- scores + top-N filter (mAP-exact; spec §3.3) ---
    # Shared with the lite dense-IoU proxy via score_and_topk_filter so the score
    # formula and tie-break can never drift. The filter ranks by the SAME score
    # COCOeval uses and keeps all queries with score >= the max_dets-th-highest
    # score (threshold, not exactly max_dets) — a SUPERSET of whatever 100 the
    # scorer would keep under its own tie-break. Done BEFORE upsample/transfer/RLE
    # so those costs only touch survivors. The non-finite RuntimeError raises here.
    scores, keep_idx = score_and_topk_filter(outputs, max_dets)

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
    # Guard the note at the call site: its arguments (int(n), tuple(...)) are
    # evaluated before note() can short-circuit, and this is the hot per-image
    # postprocess loop (#250 bottleneck) — so honor the no-op guarantee here.
    if profiling.is_enabled():
        profiling.note(N=int(n), mask_logit_hw=tuple(pred_masks.shape[-2:]))
    masks_logits = pred_masks.float().squeeze(0)  # (N, H_m, W_m)
    if not torch.isfinite(masks_logits).all():
        raise RuntimeError(
            "non-finite mask logits in postprocess; check model outputs "
            "(pred_masks contains NaN/Inf)"
        )
    if keep_idx is not None:
        masks_logits = masks_logits[keep_idx]
    with profiling.bucket("eval.mask_upsample"):
        masks_up = _upsample_mask_logits(masks_logits, original_hw)  # (M, H, W)
    with profiling.bucket("eval.transfer_binarize"):
        masks_bin = _binarize_to_host(masks_up, mask_threshold)  # (M, H, W) bool

    entries: list[dict[str, object]] = []
    with profiling.bucket("eval.box_transfer"):
        boxes_list = boxes_xywh.cpu().tolist()
        scores_list = scores.cpu().tolist()
    # Batched RLE: encode all survivor masks in ONE pycocotools call.
    # masks_bin is (M, H, W) bool; encode wants Fortran (H, W, M) uint8.
    with profiling.bucket("eval.rle_encode"):
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
