"""SemanticHFDataset against a tiny in-memory HF dataset (SS5.4)."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

datasets = pytest.importorskip("datasets")

from custom_sam_peft.data.base import Example, SemanticTarget
from custom_sam_peft.data.semantic_hf import SemanticHFDataset


def _tiny_ds():
    imgs = [Image.fromarray(np.zeros((16, 16, 3), np.uint8)) for _ in range(2)]
    lbls = []
    for _ in range(2):
        a = np.zeros((16, 16), np.uint8)
        a[:8] = 1
        a[8:] = 2
        lbls.append(Image.fromarray(a, mode="L"))
    return datasets.Dataset.from_dict({"image": imgs, "annotation": lbls})


def test_semantic_hf_requires_label_map_field():
    ds = _tiny_ds()
    with pytest.raises(ValueError, match="label_map"):
        SemanticHFDataset(
            ds,
            image_field="image",
            label_map_field=None,  # missing -> error
            class_names=["road", "building"],
            ignore_index=255,
            transforms=None,
            channels=3,
        )


def test_semantic_hf_getitem():
    ds = _tiny_ds()
    sds = SemanticHFDataset(
        ds,
        image_field="image",
        label_map_field="annotation",
        class_names=["road", "building"],
        ignore_index=255,
        transforms=None,
        channels=3,
    )
    ex = sds[0]
    assert isinstance(ex, Example)
    assert isinstance(ex.semantic, SemanticTarget)
    assert ex.semantic.labels.dtype == torch.int64
    assert sds.class_names == ["road", "building"]
