"""Pixel-grid composition tests for render_mask_panel."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from esam3.train.visualize import render_mask_panel


def _checker(h: int, w: int) -> NDArray[Any]:
    """A simple non-uniform image so the renderer doesn't optimize away a no-op."""
    grid = np.zeros((h, w, 3), dtype=np.uint8)
    grid[::2, ::2, :] = 255
    return grid


def test_panel_shape_and_dtype() -> None:
    img = _checker(16, 16)
    gt = [np.zeros((16, 16), dtype=bool)]
    pred = np.zeros((16, 16), dtype=np.float32)
    panel = render_mask_panel(img, gt, pred, class_name="cat")
    assert panel.shape == (16, 48, 3)
    assert panel.dtype == np.uint8
    assert not np.isnan(panel).any()


def test_panel_handles_empty_gt() -> None:
    img = _checker(16, 16)
    panel = render_mask_panel(img, [], np.zeros((16, 16), dtype=np.float32), class_name="cat")
    assert panel.shape == (16, 48, 3)


def test_panel_overlay_visible() -> None:
    img = np.full((16, 16, 3), 128, dtype=np.uint8)
    gt = [np.ones((16, 16), dtype=bool)]
    pred = np.ones((16, 16), dtype=np.float32)
    panel = render_mask_panel(img, gt, pred, class_name="cat")
    raw = panel[:, :16, :]
    gt_overlay = panel[:, 16:32, :]
    pred_overlay = panel[:, 32:, :]
    assert not np.array_equal(raw, gt_overlay)
    assert not np.array_equal(raw, pred_overlay)
