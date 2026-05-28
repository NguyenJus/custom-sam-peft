"""Tests for data/hf.py — helpers + dataset + builder."""

from __future__ import annotations

import logging
import re
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any
from unittest.mock import MagicMock
from unittest.mock import patch as mock_patch

import datasets as hf_datasets
import pytest
import torch
from PIL import Image

from custom_sam_peft.config.schema import HFFieldMap, NormalizeConfig, TextPromptConfig
from custom_sam_peft.data.hf import (
    HFFieldError,
    _normalize_bbox,
    _resolve_class_names,
    _resolve_field,
    _validate_required_fields,
)


@contextmanager
def _patch_imagenet_ctx() -> Iterator[None]:
    mock_aip = MagicMock()
    mock_aip.from_pretrained.side_effect = OSError("no cache")
    with mock_patch("transformers.AutoImageProcessor", mock_aip):
        yield


def test_resolve_field_top_level() -> None:
    row = {"image": "x"}
    assert _resolve_field(row, "image") == "x"


def test_resolve_field_dotted_path() -> None:
    row = {"objects": {"bbox": [[0, 0, 1, 1]]}}
    assert _resolve_field(row, "objects.bbox") == [[0, 0, 1, 1]]


def test_resolve_field_missing_raises_keyerror() -> None:
    with pytest.raises(KeyError, match=r"objects\.bbox"):
        _resolve_field({"objects": {}}, "objects.bbox")


def test_normalize_bbox_xywh_to_xyxy() -> None:
    assert _normalize_bbox([10.0, 20.0, 5.0, 7.0], "xywh") == (10.0, 20.0, 15.0, 27.0)


def test_normalize_bbox_xyxy_passthrough() -> None:
    assert _normalize_bbox([1.0, 2.0, 3.0, 4.0], "xyxy") == (1.0, 2.0, 3.0, 4.0)


def _build_hf_dataset(
    n: int = 2,
    *,
    include_segmentation: bool = False,
    use_class_label: bool = True,
) -> hf_datasets.Dataset:
    images = [Image.new("RGB", (8, 8)) for _ in range(n)]
    bboxes = [[[0.0, 0.0, 4.0, 4.0]] for _ in range(n)]
    categories = [[0] for _ in range(n)]
    cols: dict[str, Any] = {
        "image": images,
        "objects": [{"bbox": bboxes[i], "category": categories[i]} for i in range(n)],
        "categories": [["thing"]] * n,
    }
    if include_segmentation:
        for o in cols["objects"]:
            o["segmentation"] = [[[0, 0, 4, 0, 4, 4, 0, 4]]]
    features = None
    if use_class_label:
        features = hf_datasets.Features(
            {
                "image": hf_datasets.Image(),
                "objects": hf_datasets.Sequence(
                    {
                        "bbox": hf_datasets.Sequence(hf_datasets.Value("float32"), length=4),
                        "category": hf_datasets.ClassLabel(names=["thing"]),
                    }
                ),
                "categories": hf_datasets.Sequence(hf_datasets.Value("string")),
            }
        )
    return hf_datasets.Dataset.from_dict(cols, features=features)


def test_validate_required_fields_passes_on_default_schema() -> None:
    ds = _build_hf_dataset(use_class_label=False)
    _validate_required_fields(ds, HFFieldMap(segmentation=None))


def test_validate_required_fields_raises_on_missing_bbox() -> None:
    ds = hf_datasets.Dataset.from_dict(
        {"image": [Image.new("RGB", (8, 8))], "objects": [{"category": [0]}]}
    )
    with pytest.raises(HFFieldError) as exc:
        _validate_required_fields(ds, HFFieldMap(segmentation=None))
    msg = str(exc.value)
    assert "objects.bbox" in msg
    assert "data.hf.field_map.bbox" in msg


def test_resolve_class_names_from_classlabel_in_objects() -> None:
    ds = _build_hf_dataset(use_class_label=True)
    names = _resolve_class_names(ds, HFFieldMap())
    assert names == ["thing"]


# ---------------------------------------------------------------------------
# Task 16: HFDataset + build_hf
# ---------------------------------------------------------------------------

from custom_sam_peft._registry import lookup
from custom_sam_peft.data.base import BoxPrompts, TextPrompts
from custom_sam_peft.data.hf import HFDataset


def _build_eval(image_size: int = 8) -> Any:
    from custom_sam_peft.data.transforms import build_eval_transforms

    return build_eval_transforms(
        image_size, model_name="facebook/sam3.1", normalize=NormalizeConfig()
    )


def _patch_load_dataset(monkeypatch: pytest.MonkeyPatch, ds: hf_datasets.Dataset) -> None:
    def fake(name: str, split: str, **kwargs: object) -> hf_datasets.Dataset:
        return ds

    monkeypatch.setattr("custom_sam_peft.data.hf.hf_load_dataset", fake)


def test_required_fields_validation_default_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bad = hf_datasets.Dataset.from_dict(
        {"image": [Image.new("RGB", (8, 8))], "objects": [{"category": [0]}]}
    )
    _patch_load_dataset(monkeypatch, bad)
    with _patch_imagenet_ctx(), pytest.raises(HFFieldError) as exc:
        HFDataset(
            name="x",
            split="train",
            prompt_mode="bbox",
            transforms=_build_eval(),
            text_prompt=TextPromptConfig(),
            field_map=HFFieldMap(segmentation=None),
        )
    msg = str(exc.value)
    assert "objects.bbox" in msg
    assert "data.hf.field_map.bbox" in msg


def test_field_map_override_picks_alternate_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    features = hf_datasets.Features(
        {
            "image": hf_datasets.Image(),
            "annotations": hf_datasets.Sequence(
                {
                    "bbox": hf_datasets.Sequence(hf_datasets.Value("float32"), length=4),
                    "label": hf_datasets.ClassLabel(names=["thing"]),
                }
            ),
        }
    )
    ds = hf_datasets.Dataset.from_dict(
        {
            "image": [Image.new("RGB", (8, 8))],
            "annotations": [{"bbox": [[0.0, 0.0, 4.0, 4.0]], "label": [0]}],
        },
        features=features,
    )
    _patch_load_dataset(monkeypatch, ds)
    with _patch_imagenet_ctx():
        hfds = HFDataset(
            name="x",
            split="train",
            prompt_mode="bbox",
            transforms=_build_eval(),
            text_prompt=TextPromptConfig(),
            field_map=HFFieldMap(
                bbox="annotations.bbox",
                category="annotations.label",
                segmentation=None,
            ),
        )
    assert len(hfds) == 1
    assert hfds.class_names == ["thing"]


def test_class_names_from_categories_feature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    features = hf_datasets.Features(
        {
            "image": hf_datasets.Image(),
            "objects": hf_datasets.Sequence(
                {
                    "bbox": hf_datasets.Sequence(hf_datasets.Value("float32"), length=4),
                    "category": hf_datasets.ClassLabel(names=["a", "b"]),
                }
            ),
        }
    )
    ds = hf_datasets.Dataset.from_dict(
        {
            "image": [Image.new("RGB", (8, 8))],
            "objects": [{"bbox": [[0.0, 0.0, 4.0, 4.0]], "category": [0]}],
        },
        features=features,
    )
    _patch_load_dataset(monkeypatch, ds)
    with _patch_imagenet_ctx():
        hfds = HFDataset(
            name="x",
            split="train",
            prompt_mode="text",
            transforms=_build_eval(),
            text_prompt=TextPromptConfig(),
            field_map=HFFieldMap(segmentation=None),
        )
    assert hfds.class_names == ["a", "b"]


def test_getitem_text_mode_present(monkeypatch: pytest.MonkeyPatch) -> None:
    features = hf_datasets.Features(
        {
            "image": hf_datasets.Image(),
            "objects": hf_datasets.Sequence(
                {
                    "bbox": hf_datasets.Sequence(hf_datasets.Value("float32"), length=4),
                    "category": hf_datasets.ClassLabel(names=["a", "b"]),
                }
            ),
        }
    )
    ds = hf_datasets.Dataset.from_dict(
        {
            "image": [Image.new("RGB", (8, 8))],
            "objects": [{"bbox": [[0.0, 0.0, 4.0, 4.0]], "category": [1]}],
        },
        features=features,
    )
    _patch_load_dataset(monkeypatch, ds)
    with _patch_imagenet_ctx():
        hfds = HFDataset(
            name="x",
            split="train",
            prompt_mode="text",
            transforms=_build_eval(),
            text_prompt=TextPromptConfig(mode="present"),
            field_map=HFFieldMap(segmentation=None),
        )
    ex = hfds[0]
    assert isinstance(ex.prompts, TextPrompts)
    assert ex.prompts.classes == ["b"]


def test_getitem_bbox_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    ds = _build_hf_dataset(use_class_label=True)
    _patch_load_dataset(monkeypatch, ds)
    with _patch_imagenet_ctx():
        hfds = HFDataset(
            name="x",
            split="train",
            prompt_mode="bbox",
            transforms=_build_eval(),
            text_prompt=TextPromptConfig(),
            field_map=HFFieldMap(segmentation=None),
        )
    ex = hfds[0]
    assert isinstance(ex.prompts, BoxPrompts)
    assert ex.prompts.boxes.dtype == torch.float32
    assert ex.prompts.class_ids.dtype == torch.int64


def test_bbox_format_xywh_conversion(monkeypatch: pytest.MonkeyPatch) -> None:
    features = hf_datasets.Features(
        {
            "image": hf_datasets.Image(),
            "objects": hf_datasets.Sequence(
                {
                    "bbox": hf_datasets.Sequence(hf_datasets.Value("float32"), length=4),
                    "category": hf_datasets.ClassLabel(names=["thing"]),
                }
            ),
        }
    )
    ds = hf_datasets.Dataset.from_dict(
        {
            "image": [Image.new("RGB", (8, 8))],
            "objects": [{"bbox": [[1.0, 2.0, 3.0, 4.0]], "category": [0]}],
        },
        features=features,
    )
    _patch_load_dataset(monkeypatch, ds)
    with _patch_imagenet_ctx():
        hfds = HFDataset(
            name="x",
            split="train",
            prompt_mode="bbox",
            transforms=_build_eval(),
            text_prompt=TextPromptConfig(),
            field_map=HFFieldMap(segmentation=None, bbox_format="xywh"),
        )
    ex = hfds[0]
    box = ex.prompts.boxes[0]
    assert abs(float(box[0]) - 1.0) < 0.5
    assert abs(float(box[2]) - 4.0) < 0.5


def test_masks_from_boxes_when_segmentation_absent(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    ds = _build_hf_dataset(use_class_label=True, include_segmentation=False)
    _patch_load_dataset(monkeypatch, ds)
    caplog.set_level(logging.WARNING, logger="custom_sam_peft.data.hf")
    with _patch_imagenet_ctx():
        hfds = HFDataset(
            name="x",
            split="train",
            prompt_mode="bbox",
            transforms=_build_eval(),
            text_prompt=TextPromptConfig(),
            field_map=HFFieldMap(segmentation=None),
        )
    ex = hfds[0]
    assert ex.instances[0].mask.dtype == torch.bool
    assert int(ex.instances[0].mask.sum()) > 0
    assert any(re.search(r"masks-from-boxes", rec.message) for rec in caplog.records)


def test_register_hf_lookup(monkeypatch: pytest.MonkeyPatch) -> None:
    ds = _build_hf_dataset(use_class_label=True)
    _patch_load_dataset(monkeypatch, ds)
    builder = lookup("dataset", "hf")
    cfg: dict[str, Any] = {
        "format": "hf",
        "train": {"annotations": "unused", "images": "unused"},
        "val": {"annotations": "unused", "images": "unused"},
        "hf": {
            "name": "x",
            "split_train": "train",
            "split_val": "val",
            "field_map": {
                "image": "image",
                "bbox": "objects.bbox",
                "category": "objects.category",
                "segmentation": None,
                "categories_feature": "categories",
                "bbox_format": "xyxy",
            },
        },
        "prompt_mode": "bbox",
        "augmentations": {"preset": "none"},
        "text_prompt": {"mode": "present"},
        "normalize": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
    }
    with _patch_imagenet_ctx():
        hfds = builder(cfg, model_name="facebook/sam3.1", pipeline="eval")
    assert len(hfds) > 0


def _build_many_cat_hf(n_cats: int) -> hf_datasets.Dataset:
    features = hf_datasets.Features(
        {
            "image": hf_datasets.Image(),
            "objects": hf_datasets.Sequence(
                {
                    "bbox": hf_datasets.Sequence(hf_datasets.Value("float32"), length=4),
                    "category": hf_datasets.ClassLabel(names=[f"c{i}" for i in range(n_cats)]),
                }
            ),
        }
    )
    return hf_datasets.Dataset.from_dict(
        {
            "image": [Image.new("RGB", (16, 16))],
            "objects": [
                {
                    "bbox": [[0.0, 0.0, 4.0, 4.0]] * n_cats,
                    "category": list(range(n_cats)),
                }
            ],
        },
        features=features,
    )


def test_multiplex_truncation_text_hf(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    ds = _build_many_cat_hf(20)
    _patch_load_dataset(monkeypatch, ds)
    caplog.set_level(logging.WARNING, logger="custom_sam_peft.data.hf")
    with _patch_imagenet_ctx():
        hfds = HFDataset(
            name="x",
            split="train",
            prompt_mode="text",
            transforms=_build_eval(16),
            text_prompt=TextPromptConfig(mode="all"),
            field_map=HFFieldMap(segmentation=None),
        )
    ex = hfds[0]
    assert isinstance(ex.prompts, TextPrompts)
    assert len(ex.prompts.classes) == 16
    assert any(re.search(r"truncating to 16", rec.message) for rec in caplog.records)


def test_multiplex_truncation_box_hf(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    ds = _build_many_cat_hf(20)
    _patch_load_dataset(monkeypatch, ds)
    caplog.set_level(logging.WARNING, logger="custom_sam_peft.data.hf")
    with _patch_imagenet_ctx():
        hfds = HFDataset(
            name="x",
            split="train",
            prompt_mode="bbox",
            transforms=_build_eval(16),
            text_prompt=TextPromptConfig(),
            field_map=HFFieldMap(segmentation=None),
        )
    ex = hfds[0]
    assert isinstance(ex.prompts, BoxPrompts)
    assert ex.prompts.boxes.shape == (16, 4)


def test_build_hf_train_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    ds = _build_hf_dataset(use_class_label=True)
    _patch_load_dataset(monkeypatch, ds)
    builder = lookup("dataset", "hf")
    cfg: dict[str, Any] = {
        "format": "hf",
        "train": {"annotations": "unused", "images": "unused"},
        "val": {"annotations": "unused", "images": "unused"},
        "hf": {
            "name": "x",
            "split_train": "train",
            "split_val": "val",
            "field_map": {
                "image": "image",
                "bbox": "objects.bbox",
                "category": "objects.category",
                "segmentation": None,
                "categories_feature": "categories",
                "bbox_format": "xyxy",
            },
        },
        "prompt_mode": "bbox",
        "augmentations": {"preset": "none"},
        "text_prompt": {"mode": "present"},
        "normalize": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
    }
    with _patch_imagenet_ctx():
        hfds = builder(cfg, model_name="facebook/sam3.1", pipeline="train")
    assert len(hfds) > 0


# ---------------------------------------------------------------------------
# spec/data-no-val-auto-split (#71): row_indices subset parameter
# ---------------------------------------------------------------------------


def _make_min_ds_with_class_label(n: int) -> hf_datasets.Dataset:
    """Adapter shim: wraps the existing `_build_hf_dataset(n=...)` helper."""
    return _build_hf_dataset(n=n, use_class_label=True)


def test_hfdataset_row_indices_filters_to_subset(monkeypatch: pytest.MonkeyPatch) -> None:
    """Spec §6.2: row_indices restricts the dataset to the requested rows."""
    ds_underlying = _make_min_ds_with_class_label(n=5)
    _patch_load_dataset(monkeypatch, ds_underlying)
    with _patch_imagenet_ctx():
        ds = HFDataset(
            name="x",
            split="train",
            prompt_mode="text",
            transforms=_build_eval(),
            text_prompt=TextPromptConfig(),
            field_map=HFFieldMap(),
            row_indices=[0, 2],
        )
    assert len(ds) == 2


def test_hfdataset_row_indices_out_of_range_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Spec §6.2: out-of-range row_indices raise ValueError."""
    ds_underlying = _make_min_ds_with_class_label(n=3)
    _patch_load_dataset(monkeypatch, ds_underlying)
    with _patch_imagenet_ctx(), pytest.raises(ValueError, match="out of range"):
        HFDataset(
            name="x",
            split="train",
            prompt_mode="text",
            transforms=_build_eval(),
            text_prompt=TextPromptConfig(),
            field_map=HFFieldMap(),
            row_indices=[-1],
        )
    with _patch_imagenet_ctx(), pytest.raises(ValueError, match="out of range"):
        HFDataset(
            name="x",
            split="train",
            prompt_mode="text",
            transforms=_build_eval(),
            text_prompt=TextPromptConfig(),
            field_map=HFFieldMap(),
            row_indices=[100],
        )


def test_hfdataset_image_id_uses_underlying_row_index(monkeypatch: pytest.MonkeyPatch) -> None:
    """Spec §6.2 last paragraph: image_id in the returned Example uses the
    underlying dataset row index (not the post-subset position)."""
    ds_underlying = _make_min_ds_with_class_label(n=5)
    _patch_load_dataset(monkeypatch, ds_underlying)
    with _patch_imagenet_ctx():
        ds = HFDataset(
            name="x",
            split="train",
            prompt_mode="text",
            transforms=_build_eval(),
            text_prompt=TextPromptConfig(),
            field_map=HFFieldMap(),
            row_indices=[2, 4],
        )
    ex0 = ds[0]
    # Subset position 0 → underlying row 2 → image_id == "2".
    assert ex0.image_id == "2"
    ex1 = ds[1]
    assert ex1.image_id == "4"


# ---------------------------------------------------------------------------
# Task 7+8: channel-aware _decode_image in HFDataset
# ---------------------------------------------------------------------------


def test_hf_decode_image_array_branch_channel_aware():
    """HFDataset._decode_image coerces an array row to self._channels."""
    import numpy as np

    from custom_sam_peft.data import hf as hf_mod

    obj = hf_mod.HFDataset.__new__(hf_mod.HFDataset)
    obj._channels = 1
    obj._field_map = type("FM", (), {"image": "image"})()

    raw = {"image": np.zeros((4, 6), np.uint8)}
    out = hf_mod.HFDataset._decode_image(obj, raw)
    assert out.shape == (4, 6, 1)
