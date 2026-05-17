"""Mask-panel rendering for image-logging in the training loop.

`render_mask_panel(image, gt_masks, pred_mask, class_name)` returns a single
(H, 3*W, 3) uint8 strip: the un-modified image, a GT overlay, and a pred
overlay. The function is pure (no torch, no I/O), so the trainer is free to
call it under `torch.no_grad()` or from a worker.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

_GT_COLOR: NDArray[Any] = np.array([0, 255, 0], dtype=np.float32)
_PRED_COLOR: NDArray[Any] = np.array([255, 0, 0], dtype=np.float32)
_OVERLAY_ALPHA = 0.5


def _overlay(
    image: NDArray[np.uint8], mask: NDArray[Any], color: NDArray[Any]
) -> NDArray[np.uint8]:
    """Alpha-blend `color` onto `image` where `mask > 0`."""
    out = image.astype(np.float32).copy()
    mask_f = mask.astype(np.float32)[..., None]
    out = out * (1.0 - _OVERLAY_ALPHA * mask_f) + color * (_OVERLAY_ALPHA * mask_f)
    return np.clip(out, 0.0, 255.0).astype(np.uint8)  # type: ignore[no-any-return]


def render_mask_panel(
    image: NDArray[np.uint8],
    gt_masks: list[NDArray[Any]],
    pred_mask: NDArray[Any],
    class_name: str,
) -> NDArray[np.uint8]:
    """Compose image | GT-overlay | pred-overlay horizontally.

    Empty `gt_masks` → the GT panel is just `image` un-overlaid.
    """
    del class_name
    gt_union = (
        np.any(np.stack(gt_masks, axis=0), axis=0).astype(np.float32)
        if gt_masks
        else np.zeros(image.shape[:2], dtype=np.float32)
    )
    pred_bin = (pred_mask >= 0.5).astype(np.float32)
    gt_panel = _overlay(image, gt_union, _GT_COLOR)
    pred_panel = _overlay(image, pred_bin, _PRED_COLOR)
    return np.concatenate([image, gt_panel, pred_panel], axis=1).astype(np.uint8)  # type: ignore[no-any-return]
