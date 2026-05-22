"""HFDataset.image_class_labels lazy cache tests."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest


def _make_hf_dataset(n: int = 4, n_classes: int = 3):
    """Build a minimal HFDataset backed by an in-memory datasets.Dataset."""
    from custom_sam_peft.config.schema import HFFieldMap, TextPromptConfig
    from custom_sam_peft.data.hf import HFDataset

    # categories as a Sequence feature (name→value pairs)
    fake_ds = MagicMock()
    fake_ds.__len__ = lambda self: n
    fake_ds.__getitem__ = lambda self, i: {
        "image": None,
        "objects": {"bbox": [[0, 0, 1, 1]], "category": [i % n_classes], "segmentation": [None]},
    }
    fake_ds.features = {"objects": MagicMock()}

    ds = HFDataset.__new__(HFDataset)
    ds._name = "fake"
    ds._split = "train"
    ds._prompt_mode = "bbox"
    ds._transforms = MagicMock()
    ds._text_prompt_cfg = TextPromptConfig()
    ds._field_map = HFFieldMap()
    ds._seed = 0
    ds._multiplex_cap = 16
    ds._warned_truncation = False
    ds._warned_masks_from_boxes = False
    ds._ds = fake_ds
    ds._class_names = [f"cls{i}" for i in range(n_classes)]
    ds._image_class_labels = None  # cache sentinel
    return ds


def test_image_class_labels_not_computed_before_access() -> None:
    ds = _make_hf_dataset()
    assert ds._image_class_labels is None


def test_image_class_labels_computed_on_first_access(
    caplog: pytest.LogCaptureFixture,
) -> None:
    ds = _make_hf_dataset(n=4, n_classes=3)

    def _resolve_field_stub(row, path):
        parts = path.split(".")
        v = row
        for p in parts:
            v = v[p]
        return v

    with (
        patch("custom_sam_peft.data.hf._resolve_field", side_effect=_resolve_field_stub),
        caplog.at_level(logging.INFO, logger="custom_sam_peft.data.hf"),
    ):
        labels = ds.image_class_labels

    assert labels is not None
    assert len(labels) == 4
    assert all(isinstance(s, frozenset) for s in labels)
    assert any("scanning 4 rows" in r.message for r in caplog.records)


def test_image_class_labels_cached_no_second_scan(
    caplog: pytest.LogCaptureFixture,
) -> None:
    ds = _make_hf_dataset(n=4, n_classes=3)

    def _resolve_field_stub(row, path):
        parts = path.split(".")
        v = row
        for p in parts:
            v = v[p]
        return v

    with (
        patch("custom_sam_peft.data.hf._resolve_field", side_effect=_resolve_field_stub),
        caplog.at_level(logging.INFO, logger="custom_sam_peft.data.hf"),
    ):
        _ = ds.image_class_labels
        caplog.clear()
        _ = ds.image_class_labels

    # Second access must NOT produce a second scan log
    scan_msgs = [r for r in caplog.records if "scanning" in r.message]
    assert scan_msgs == []


def test_image_class_labels_not_accessed_for_random_strategy() -> None:
    """_build_dataset with random strategy never reads image_class_labels."""
    from custom_sam_peft.data.subset import SubsetDataset
    from custom_sam_peft.train.runner import _build_dataset

    inner = MagicMock()
    inner.__len__ = MagicMock(return_value=10)
    inner.class_names = ["x"]
    # image_class_labels should NOT be accessed; track via spec
    accessed = []

    type(inner).image_class_labels = property(lambda self: accessed.append(True) or [frozenset()])

    cfg = MagicMock()
    cfg.data.format = "hf"
    cfg.data.model_dump.return_value = {}
    cfg.model.name = "n/a"
    cfg.data.limit.train = 3
    cfg.data.limit.val = None
    cfg.data.limit.seed = 0
    cfg.data.limit.strategy = "random"

    with patch("custom_sam_peft.train.runner.lookup", return_value=lambda *a, **kw: inner):
        ds = _build_dataset(cfg, "train")

    assert isinstance(ds, SubsetDataset)
    assert accessed == []  # image_class_labels was never accessed
