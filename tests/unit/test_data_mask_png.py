# tests/unit/test_data_mask_png.py
"""MaskPngDataset against a synthetic temp tree (SS5.3)."""

from __future__ import annotations

import json
import sys

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


# ---------------------------------------------------------------------------
# Production-registration regression test (Issue 1)
# ---------------------------------------------------------------------------


def test_production_registration_via_bootstrap(tmp_path):
    """lookup("dataset","mask_png") resolves via bootstrap WITHOUT a direct mask_png import.

    Evicts mask_png, its parent data package, and _bootstrap from sys.modules, then
    re-imports _bootstrap to fire its module-level side-effects — same path as the
    production trainer.  Asserts lookup resolves to the builder and that it returns
    a working dataset.
    """
    from custom_sam_peft._registry import _REGISTRY, reset_registry

    # Must evict the data package too so that 'from custom_sam_peft.data import mask_png'
    # in _bootstrap actually re-executes rather than returning the cached submodule.
    mods_to_evict = (
        "custom_sam_peft.data",
        "custom_sam_peft.data.mask_png",
        "custom_sam_peft.data.coco",
        "custom_sam_peft.data.hf",
        "custom_sam_peft._bootstrap",
    )

    saved_mods = {m: sys.modules[m] for m in mods_to_evict if m in sys.modules}
    saved_registry: dict[str, dict] = {k: dict(v) for k, v in _REGISTRY.items()}

    # Capture package-level submodule attributes so sibling tests can still resolve them.
    _pkg_attr_snapshot: list[tuple[object, str, object]] = []
    for dotted in mods_to_evict:
        parts = dotted.rsplit(".", 1)
        if len(parts) == 2:
            parent_name, attr = parts
            parent = sys.modules.get(parent_name)
            if parent is not None and hasattr(parent, attr):
                _pkg_attr_snapshot.append((parent, attr, getattr(parent, attr)))

    try:
        reset_registry()
        for m in mods_to_evict:
            sys.modules.pop(m, None)

        # Import _bootstrap (not mask_png directly) to simulate production wiring.
        import custom_sam_peft._bootstrap  # noqa: F401

        builder = lookup("dataset", "mask_png")
        assert callable(builder), "bootstrap must register mask_png dataset builder"

        # Smoke-check: builder actually constructs a working dataset.
        img_dir, lbl_dir, cm = _make_tree(tmp_path)
        ds = builder(_cfg(img_dir, lbl_dir, cm), model_name="sam3.1", pipeline="eval")
        assert len(ds) == 2
    finally:
        for m in mods_to_evict:
            sys.modules.pop(m, None)
        sys.modules.update(saved_mods)
        _REGISTRY.clear()
        _REGISTRY.update(saved_registry)
        for parent, attr, orig_val in _pkg_attr_snapshot:
            setattr(parent, attr, orig_val)


# ---------------------------------------------------------------------------
# Auto-split (_resolved_image_ids) filtering test (Issue 2)
# ---------------------------------------------------------------------------


def test_resolved_image_ids_filters_to_subset(tmp_path):
    """_resolved_image_ids restricts the dataset to the named image stems."""
    img_dir, lbl_dir, cm = _make_tree(tmp_path)
    builder = lookup("dataset", "mask_png")
    cfg = _cfg(img_dir, lbl_dir, cm)
    # Inject auto-split IDs for the eval pipeline — only stem "a".
    cfg["_resolved_image_ids"] = {"eval": ["a"]}
    ds = builder(cfg, model_name="sam3.1", pipeline="eval")
    assert len(ds) == 1
    ex = ds[0]
    assert ex.image_id == "a"
