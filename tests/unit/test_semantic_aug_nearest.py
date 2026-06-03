"""Label-map augmentation must use nearest interp + ignore-fill padding (§5.5/§4.5)."""

from __future__ import annotations

import json

import numpy as np
import torch
from PIL import Image

import custom_sam_peft.data.mask_png  # noqa: F401 — triggers @register side-effect
from custom_sam_peft._registry import lookup


def _cfg(img_dir, lbl_dir, cm):
    # Match the real builder cfg-dict shape used by test_data_mask_png.py.
    return {
        "train": {"images": str(img_dir), "annotations": str(lbl_dir)},
        "val": None,
        "semantic": {"class_map": str(cm), "ignore_index": 255, "label_suffix": ".png"},
        "channels": 3,
        "text_prompt": {"mode": "all"},
    }


def test_label_map_resize_preserves_integer_class_ids(tmp_path):
    img_dir = tmp_path / "img"
    lbl_dir = tmp_path / "lbl"
    img_dir.mkdir()
    lbl_dir.mkdir()
    Image.fromarray(np.zeros((37, 37, 3), dtype=np.uint8)).save(img_dir / "a.png")
    lbl = np.zeros((37, 37), dtype=np.uint8)
    lbl[:18] = 1
    lbl[18:] = 2
    Image.fromarray(lbl, mode="L").save(lbl_dir / "a.png")
    cm = tmp_path / "cm.json"
    cm.write_text(json.dumps({"0": "background", "1": "road", "2": "building"}))
    ds = lookup("dataset", "mask_png")(
        _cfg(img_dir, lbl_dir, cm), model_name="sam3.1", pipeline="train"
    )
    labels = ds[0].semantic.labels
    present = set(labels.unique().tolist())
    assert present <= {0, 1, 2, 255}, f"fractional/blended ids leaked: {present}"
    assert labels.dtype == torch.int64


def test_padding_fills_label_with_ignore_index_not_background(tmp_path):
    img_dir = tmp_path / "img"
    lbl_dir = tmp_path / "lbl"
    img_dir.mkdir()
    lbl_dir.mkdir()
    # Wide, short image -> LongestMaxSize scales width to image_size (~1008), height < image_size
    # -> vertical padding fills the bottom region.
    # 64x16 -> longest edge (64) scales to 1008 -> height becomes 252 -> pad to 1008 vertically.
    Image.fromarray(np.zeros((16, 64, 3), dtype=np.uint8)).save(img_dir / "a.png")
    lbl = np.ones((16, 64), dtype=np.uint8)  # entirely class 1 (road); NO background, NO void
    Image.fromarray(lbl, mode="L").save(lbl_dir / "a.png")
    cm = tmp_path / "cm.json"
    cm.write_text(json.dumps({"0": "background", "1": "road", "2": "building"}))
    ds = lookup("dataset", "mask_png")(
        _cfg(img_dir, lbl_dir, cm), model_name="sam3.1", pipeline="eval"
    )
    labels = ds[0].semantic.labels
    present = set(labels.unique().tolist())
    # Real content is all class 1 (road). Padding must be ignore_index (255), NOT background (0).
    assert 1 in present, "the real road pixels should survive"
    assert 0 not in present, "padding leaked as background (0) instead of ignore_index"
    assert 255 in present, "padded region should be ignore_index (255)"
