"""COCODataset.image_class_labels + limit wrapping via _build_dataset."""

from __future__ import annotations

from pathlib import Path

import pytest

from custom_sam_peft.config.schema import (
    NormalizeConfig,
    TextPromptConfig,
)
from custom_sam_peft.data.coco import COCODataset
from custom_sam_peft.data.transforms import build_eval_transforms


@pytest.fixture
def coco_ds(tiny_coco_dir: Path) -> COCODataset:
    transforms = build_eval_transforms(
        32, model_name="facebook/sam3.1", normalize=NormalizeConfig()
    )
    return COCODataset(
        annotations=str(tiny_coco_dir / "annotations.json"),
        images=str(tiny_coco_dir / "images"),
        prompt_mode="bbox",
        transforms=transforms,
        text_prompt=TextPromptConfig(),
    )


def test_image_class_labels_populated_at_init(coco_ds: COCODataset) -> None:
    labels = coco_ds.image_class_labels
    assert isinstance(labels, list)
    assert len(labels) == len(coco_ds)


def test_image_class_labels_are_frozensets(coco_ds: COCODataset) -> None:
    for entry in coco_ds.image_class_labels:
        assert isinstance(entry, frozenset)
        # All class ids must be valid dense ids (0..C-1)
        for c in entry:
            assert 0 <= c < len(coco_ds.class_names)


def test_image_class_labels_length_matches_image_ids(coco_ds: COCODataset) -> None:
    assert len(coco_ds.image_class_labels) == len(coco_ds)


def test_int_limit_via_build_dataset(tiny_coco_dir: Path) -> None:
    """_build_dataset with an int limit returns a SubsetDataset of the right size."""
    from unittest.mock import MagicMock

    from custom_sam_peft.data.subset import SubsetDataset
    from custom_sam_peft.train.runner import _build_dataset

    cfg = MagicMock()
    cfg.data.format = "coco"
    cfg.data.model_dump.return_value = {
        "format": "coco",
        "train": {
            "annotations": str(tiny_coco_dir / "annotations.json"),
            "images": str(tiny_coco_dir / "images"),
        },
        "val": {
            "annotations": str(tiny_coco_dir / "annotations.json"),
            "images": str(tiny_coco_dir / "images"),
        },
        "prompt_mode": "bbox",
        "image_size": 32,
        "augmentations": {"hflip": False, "color_jitter": 0.0},
        "text_prompt": {"mode": "present", "negatives_per_image": 0, "k": 16},
        "normalize": {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]},
        "limit": {"train": 1, "val": None, "seed": 42, "strategy": "random"},
    }
    cfg.model.name = "facebook/sam3.1"
    cfg.data.limit.train = 1
    cfg.data.limit.val = None
    cfg.data.limit.seed = 42
    cfg.data.limit.strategy = "random"

    ds = _build_dataset(cfg, "train")
    assert isinstance(ds, SubsetDataset)
    assert len(ds) == 1


def test_fraction_limit_rounds_correctly(tiny_coco_dir: Path) -> None:
    """float limit rounds to max(1, round(fraction * n_total))."""
    from unittest.mock import MagicMock

    from custom_sam_peft.data.subset import SubsetDataset
    from custom_sam_peft.train.runner import _build_dataset

    cfg = MagicMock()
    cfg.data.format = "coco"
    cfg.data.model_dump.return_value = {
        "format": "coco",
        "train": {
            "annotations": str(tiny_coco_dir / "annotations.json"),
            "images": str(tiny_coco_dir / "images"),
        },
        "val": {
            "annotations": str(tiny_coco_dir / "annotations.json"),
            "images": str(tiny_coco_dir / "images"),
        },
        "prompt_mode": "bbox",
        "image_size": 32,
        "augmentations": {"hflip": False, "color_jitter": 0.0},
        "text_prompt": {"mode": "present", "negatives_per_image": 0, "k": 16},
        "normalize": {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]},
        "limit": {"train": 0.5, "val": None, "seed": 0, "strategy": "first_n"},
    }
    cfg.model.name = "facebook/sam3.1"
    cfg.data.limit.train = 0.5
    cfg.data.limit.val = None
    cfg.data.limit.seed = 0
    cfg.data.limit.strategy = "first_n"

    # tiny_coco has 2 images; 0.5 * 2 = 1
    ds = _build_dataset(cfg, "train")
    assert isinstance(ds, SubsetDataset)
    assert len(ds) == 1


def test_stratified_limit_preserves_all_classes(tiny_coco_dir: Path) -> None:
    transforms = build_eval_transforms(
        32, model_name="facebook/sam3.1", normalize=NormalizeConfig()
    )
    ds = COCODataset(
        annotations=str(tiny_coco_dir / "annotations.json"),
        images=str(tiny_coco_dir / "images"),
        prompt_mode="bbox",
        transforms=transforms,
        text_prompt=TextPromptConfig(),
    )
    from custom_sam_peft.data.subset import resolve_subset_indices

    labels = ds.image_class_labels
    # Use all images (tiny_coco has 2) — just verify the call works
    idx = resolve_subset_indices(
        len(ds), len(ds), seed=0, strategy="stratified", image_class_labels=labels
    )
    assert len(idx) == len(ds)
