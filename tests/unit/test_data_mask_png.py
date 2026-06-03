# tests/unit/test_data_mask_png.py
"""MaskPngDataset against a synthetic temp tree (SS5.3)."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from PIL import Image

# Direct import triggers @register("dataset", "mask_png") side-effect.
import custom_sam_peft.data.mask_png  # noqa: F401
from custom_sam_peft._registry import lookup
from custom_sam_peft.data.base import Example, SemanticTarget, TextPrompts


def _make_tree(tmp_path):
    img_dir = tmp_path / "img"
    lbl_dir = tmp_path / "lbl"
    img_dir.mkdir()
    lbl_dir.mkdir()
    for stem in ("a", "b"):
        Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8)).save(img_dir / f"{stem}.png")
        lbl = np.zeros((16, 16), dtype=np.uint8)
        lbl[:8, :8] = 1  # road
        lbl[8:, 8:] = 2  # building
        lbl[0, 0] = 255  # void
        Image.fromarray(lbl, mode="L").save(lbl_dir / f"{stem}.png")
    cm = tmp_path / "cm.json"
    cm.write_text(json.dumps({"0": "background", "1": "road", "2": "building"}))
    return img_dir, lbl_dir, cm


def _cfg(img_dir, lbl_dir, cm):
    # Matches real DataConfig.model_dump() shape: top-level "train"/"val" splits,
    # "semantic" sub-dict, "channels", "text_prompt".
    return {
        "train": {"images": str(img_dir), "annotations": str(lbl_dir)},
        "val": None,
        "semantic": {"class_map": str(cm), "ignore_index": 255, "label_suffix": ".png"},
        "channels": 3,
        "text_prompt": {"mode": "all"},
    }


def _build(tmp_path):
    img_dir, lbl_dir, cm = _make_tree(tmp_path)
    builder = lookup("dataset", "mask_png")
    return builder(_cfg(img_dir, lbl_dir, cm), model_name="sam3.1", pipeline="eval")


def test_class_names_ascending_drop_background(tmp_path):
    ds = _build(tmp_path)
    assert ds.class_names == ["road", "building"]


def test_getitem_returns_semantic_example(tmp_path):
    ds = _build(tmp_path)
    ex = ds[0]
    assert isinstance(ex, Example)
    assert ex.instances == []
    assert isinstance(ex.semantic, SemanticTarget)
    assert ex.semantic.labels.dtype == torch.int64
    assert ex.semantic.ignore_index == 255
    vals = set(ex.semantic.labels.unique().tolist())
    assert vals <= {0, 1, 2, 255}
    assert 255 in vals  # void pixel survives (nearest interp; not normalized)


def test_prompts_are_full_vocabulary_mode_all(tmp_path):
    ds = _build(tmp_path)
    ex = ds[0]
    assert isinstance(ex.prompts, TextPrompts)
    assert ex.prompts.classes == ["road", "building"]


def test_missing_label_pair_raises(tmp_path):
    img_dir, lbl_dir, cm = _make_tree(tmp_path)
    (lbl_dir / "a.png").unlink()
    builder = lookup("dataset", "mask_png")
    with pytest.raises(FileNotFoundError, match="a"):
        builder(_cfg(img_dir, lbl_dir, cm), model_name="sam3.1", pipeline="eval")
