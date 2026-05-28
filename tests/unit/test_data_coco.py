"""Tests for data/coco.py — helpers + dataset + builder."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pytest
from pycocotools.coco import COCO

from custom_sam_peft.config.schema import TextPromptConfig
from custom_sam_peft.data.coco import (
    _build_category_remap,
    _build_text_prompts,
    _decode_segmentation,
    _drop_crowd_only_images,
    _load_coco_index,
)


def test_load_coco_index(tiny_coco_dir: Path) -> None:
    coco = _load_coco_index(tiny_coco_dir / "annotations.json")
    assert isinstance(coco, COCO)
    assert sorted(coco.getImgIds()) == [1, 2]


def test_build_category_remap(tiny_coco_dir: Path) -> None:
    coco = _load_coco_index(tiny_coco_dir / "annotations.json")
    sparse, mapping, names = _build_category_remap(coco)
    assert sparse == [1, 2]
    assert mapping == {1: 0, 2: 1}
    assert names == ["thing_a", "thing_b"]


def test_build_category_remap_handles_sparse_ids(tmp_path: Path) -> None:
    p = tmp_path / "ann.json"
    p.write_text(
        json.dumps(
            {
                "images": [{"id": 1, "file_name": "x.png", "width": 8, "height": 8}],
                "categories": [
                    {"id": 7, "name": "ginger"},
                    {"id": 3, "name": "apple"},
                ],
                "annotations": [],
            }
        )
    )
    coco = _load_coco_index(p)
    sparse, mapping, names = _build_category_remap(coco)
    assert sparse == [3, 7]
    assert mapping == {3: 0, 7: 1}
    assert names == ["apple", "ginger"]


def test_drop_crowd_only_images(tmp_path: Path) -> None:
    p = tmp_path / "ann.json"
    p.write_text(
        json.dumps(
            {
                "images": [
                    {"id": 1, "file_name": "a.png", "width": 8, "height": 8},
                    {"id": 2, "file_name": "b.png", "width": 8, "height": 8},
                ],
                "categories": [{"id": 1, "name": "x"}],
                "annotations": [
                    {
                        "id": 1,
                        "image_id": 1,
                        "category_id": 1,
                        "bbox": [0, 0, 4, 4],
                        "area": 16,
                        "iscrowd": 0,
                    },
                    {
                        "id": 2,
                        "image_id": 2,
                        "category_id": 1,
                        "bbox": [0, 0, 4, 4],
                        "area": 16,
                        "iscrowd": 1,
                    },
                ],
            }
        )
    )
    coco = _load_coco_index(p)
    kept, ann_index, dropped = _drop_crowd_only_images(coco)
    assert kept == [1]
    assert 2 not in ann_index
    assert dropped == 1


def test_decode_segmentation_polygon(tiny_coco_dir: Path) -> None:
    coco = _load_coco_index(tiny_coco_dir / "annotations.json")
    ann = coco.loadAnns([1])[0]
    mask = _decode_segmentation(ann, 32, 32)
    assert mask.shape == (32, 32)
    assert mask.dtype == np.bool_
    assert mask.sum() > 0


def test_decode_segmentation_rle() -> None:
    """A synthetic RLE: a 4x4 mask with all ones."""
    from pycocotools import mask as mu

    rle = mu.encode(np.asfortranarray(np.ones((4, 4), dtype=np.uint8)))
    ann = {"segmentation": rle}
    out = _decode_segmentation(ann, 4, 4)
    assert out.dtype == np.bool_
    assert out.all()


def test_build_text_prompts_present() -> None:
    out = _build_text_prompts(
        present_dense_ids=[1, 0],
        class_names=["zero", "one", "two"],
        cfg=TextPromptConfig(mode="present"),
        rng=random.Random(0),
        image_id=42,
    )
    assert out == ["zero", "one"]


def test_build_text_prompts_all() -> None:
    out = _build_text_prompts(
        present_dense_ids=[1],
        class_names=["a", "b", "c"],
        cfg=TextPromptConfig(mode="all"),
        rng=random.Random(0),
        image_id=7,
    )
    assert out == ["a", "b", "c"]


def test_build_text_prompts_present_plus_negatives() -> None:
    out = _build_text_prompts(
        present_dense_ids=[0],
        class_names=["a", "b", "c", "d", "e"],
        cfg=TextPromptConfig(mode="present_plus_negatives", negatives_per_image=2),
        rng=random.Random(123),
        image_id=1,
    )
    assert out[0] == "a"
    assert len(out) == 3
    assert len(set(out)) == 3


def test_build_text_prompts_sampled_fixed_k_truncates_positives() -> None:
    out = _build_text_prompts(
        present_dense_ids=list(range(10)),
        class_names=[f"c{i}" for i in range(20)],
        cfg=TextPromptConfig(mode="sampled_fixed_k", k=3),
        rng=random.Random(0),
        image_id=1,
    )
    assert len(out) == 3
    assert out == ["c0", "c1", "c2"]


# ---------------------------------------------------------------------------
# Task 13: COCODataset integration tests
# ---------------------------------------------------------------------------

import logging
import re
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any
from unittest.mock import MagicMock
from unittest.mock import patch as mock_patch

import torch

from custom_sam_peft.config.schema import NormalizeConfig
from custom_sam_peft.data.base import TextPrompts
from custom_sam_peft.data.coco import COCODataset


@contextmanager
def _patch_imagenet_ctx() -> Iterator[None]:
    mock_aip = MagicMock()
    mock_aip.from_pretrained.side_effect = OSError("no cache")
    with mock_patch("transformers.AutoImageProcessor", mock_aip):
        yield


def _build_eval(image_size: int = 32) -> Any:
    from custom_sam_peft.data.transforms import build_eval_transforms

    return build_eval_transforms(
        image_size, model_name="facebook/sam3.1", normalize=NormalizeConfig()
    )


def test_class_names_dense_and_ordered(tiny_coco_dir: Path) -> None:
    with _patch_imagenet_ctx():
        ds = COCODataset(
            annotations=str(tiny_coco_dir / "annotations.json"),
            images=str(tiny_coco_dir / "images"),
            transforms=_build_eval(),
            text_prompt=TextPromptConfig(),
        )
    assert ds.class_names == ["thing_a", "thing_b"]
    assert ds.coco_category_ids == [1, 2]


def test_len_drops_empty_after_iscrowd(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    p = tmp_path / "ann.json"
    p.write_text(
        json.dumps(
            {
                "images": [
                    {"id": 1, "file_name": "a.png", "width": 8, "height": 8},
                    {"id": 2, "file_name": "b.png", "width": 8, "height": 8},
                ],
                "categories": [{"id": 1, "name": "x"}],
                "annotations": [
                    {
                        "id": 1,
                        "image_id": 1,
                        "category_id": 1,
                        "bbox": [0, 0, 4, 4],
                        "area": 16,
                        "iscrowd": 0,
                        "segmentation": [[0, 0, 4, 0, 4, 4, 0, 4]],
                    },
                    {
                        "id": 2,
                        "image_id": 2,
                        "category_id": 1,
                        "bbox": [0, 0, 4, 4],
                        "area": 16,
                        "iscrowd": 1,
                        "segmentation": [[0, 0, 4, 0, 4, 4, 0, 4]],
                    },
                ],
            }
        )
    )
    images_dir = tmp_path / "imgs"
    images_dir.mkdir()
    from PIL import Image

    Image.new("RGB", (8, 8)).save(images_dir / "a.png")
    Image.new("RGB", (8, 8)).save(images_dir / "b.png")
    caplog.set_level(logging.INFO, logger="custom_sam_peft.data.coco")
    with _patch_imagenet_ctx():
        ds = COCODataset(
            annotations=str(p),
            images=str(images_dir),
            transforms=_build_eval(8),
            text_prompt=TextPromptConfig(),
        )
    assert len(ds) == 1
    assert any(re.search(r"dropped.*1.*iscrowd", rec.message) for rec in caplog.records)


def test_getitem_text_mode_present(tiny_coco_dir: Path) -> None:
    with _patch_imagenet_ctx():
        ds = COCODataset(
            annotations=str(tiny_coco_dir / "annotations.json"),
            images=str(tiny_coco_dir / "images"),
            transforms=_build_eval(),
            text_prompt=TextPromptConfig(mode="present"),
        )
    ex = ds[0]
    assert isinstance(ex.prompts, TextPrompts)
    assert ex.prompts.classes == ["thing_a", "thing_b"]


def test_getitem_text_mode_all(tiny_coco_dir: Path) -> None:
    with _patch_imagenet_ctx():
        ds = COCODataset(
            annotations=str(tiny_coco_dir / "annotations.json"),
            images=str(tiny_coco_dir / "images"),
            transforms=_build_eval(),
            text_prompt=TextPromptConfig(mode="all"),
        )
    ex = ds[1]
    assert isinstance(ex.prompts, TextPrompts)
    assert ex.prompts.classes == ["thing_a", "thing_b"]


def test_getitem_text_mode_present_plus_negatives(tmp_path: Path) -> None:
    p = tmp_path / "ann.json"
    p.write_text(
        json.dumps(
            {
                "images": [{"id": 1, "file_name": "a.png", "width": 8, "height": 8}],
                "categories": [
                    {"id": 1, "name": "a"},
                    {"id": 2, "name": "b"},
                    {"id": 3, "name": "c"},
                    {"id": 4, "name": "d"},
                ],
                "annotations": [
                    {
                        "id": 1,
                        "image_id": 1,
                        "category_id": 1,
                        "bbox": [0, 0, 4, 4],
                        "area": 16,
                        "iscrowd": 0,
                        "segmentation": [[0, 0, 4, 0, 4, 4, 0, 4]],
                    }
                ],
            }
        )
    )
    images_dir = tmp_path / "imgs"
    images_dir.mkdir()
    from PIL import Image

    Image.new("RGB", (8, 8)).save(images_dir / "a.png")
    with _patch_imagenet_ctx():
        ds = COCODataset(
            annotations=str(p),
            images=str(images_dir),
            transforms=_build_eval(8),
            text_prompt=TextPromptConfig(mode="present_plus_negatives", negatives_per_image=2),
            seed=42,
        )
    ex = ds[0]
    assert isinstance(ex.prompts, TextPrompts)
    assert ex.prompts.classes[0] == "a"
    assert len(ex.prompts.classes) == 3
    assert len(set(ex.prompts.classes)) == 3


def test_getitem_text_mode_sampled_fixed_k(tmp_path: Path) -> None:
    p = tmp_path / "ann.json"
    p.write_text(
        json.dumps(
            {
                "images": [{"id": 1, "file_name": "a.png", "width": 8, "height": 8}],
                "categories": [{"id": i, "name": f"c{i}"} for i in range(1, 6)],
                "annotations": [
                    {
                        "id": 1,
                        "image_id": 1,
                        "category_id": 1,
                        "bbox": [0, 0, 4, 4],
                        "area": 16,
                        "iscrowd": 0,
                        "segmentation": [[0, 0, 4, 0, 4, 4, 0, 4]],
                    }
                ],
            }
        )
    )
    images_dir = tmp_path / "imgs"
    images_dir.mkdir()
    from PIL import Image

    Image.new("RGB", (8, 8)).save(images_dir / "a.png")
    with _patch_imagenet_ctx():
        ds = COCODataset(
            annotations=str(p),
            images=str(images_dir),
            transforms=_build_eval(8),
            text_prompt=TextPromptConfig(mode="sampled_fixed_k", k=3),
            seed=7,
        )
    ex = ds[0]
    assert isinstance(ex.prompts, TextPrompts)
    assert len(ex.prompts.classes) == 3
    assert ex.prompts.classes[0] == "c1"


def _synth_many_cats(tmp_path: Path, n_cats: int) -> tuple[Path, Path]:
    """Build a 1-image COCO with n_cats categories, one annotation each."""
    from PIL import Image

    p = tmp_path / "ann.json"
    images_dir = tmp_path / "imgs"
    images_dir.mkdir()
    Image.new("RGB", (32, 32)).save(images_dir / "a.png")
    p.write_text(
        json.dumps(
            {
                "images": [{"id": 1, "file_name": "a.png", "width": 32, "height": 32}],
                "categories": [{"id": i + 1, "name": f"c{i}"} for i in range(n_cats)],
                "annotations": [
                    {
                        "id": i + 1,
                        "image_id": 1,
                        "category_id": i + 1,
                        "bbox": [0, 0, 4, 4],
                        "area": 16,
                        "iscrowd": 0,
                        "segmentation": [[0, 0, 4, 0, 4, 4, 0, 4]],
                    }
                    for i in range(n_cats)
                ],
            }
        )
    )
    return p, images_dir


def test_multiplex_truncation_text(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    ann, imgs = _synth_many_cats(tmp_path, 20)
    caplog.set_level(logging.WARNING, logger="custom_sam_peft.data.coco")
    with _patch_imagenet_ctx():
        ds = COCODataset(
            annotations=str(ann),
            images=str(imgs),
            transforms=_build_eval(32),
            text_prompt=TextPromptConfig(mode="all"),
        )
    ex = ds[0]
    assert isinstance(ex.prompts, TextPrompts)
    assert len(ex.prompts.classes) == 16
    assert any(re.search(r"truncating to 16", rec.message) for rec in caplog.records)


def test_polygon_segmentation_decoded(tiny_coco_dir: Path) -> None:
    with _patch_imagenet_ctx():
        ds = COCODataset(
            annotations=str(tiny_coco_dir / "annotations.json"),
            images=str(tiny_coco_dir / "images"),
            transforms=_build_eval(32),
            text_prompt=TextPromptConfig(),
        )
    ex = ds[0]
    assert ex.instances[0].mask.shape == (32, 32)
    assert ex.instances[0].mask.dtype == torch.bool
    assert int(ex.instances[0].mask.sum()) > 0


def test_rle_segmentation_decoded(tmp_path: Path) -> None:
    from PIL import Image
    from pycocotools import mask as mu

    rle = mu.encode(np.asfortranarray(np.ones((8, 8), dtype=np.uint8)))
    rle["counts"] = (
        rle["counts"].decode("ascii") if isinstance(rle["counts"], bytes) else rle["counts"]
    )
    p = tmp_path / "ann.json"
    p.write_text(
        json.dumps(
            {
                "images": [{"id": 1, "file_name": "a.png", "width": 8, "height": 8}],
                "categories": [{"id": 1, "name": "x"}],
                "annotations": [
                    {
                        "id": 1,
                        "image_id": 1,
                        "category_id": 1,
                        "bbox": [0, 0, 8, 8],
                        "area": 64,
                        "iscrowd": 0,
                        "segmentation": rle,
                    }
                ],
            }
        )
    )
    images_dir = tmp_path / "imgs"
    images_dir.mkdir()
    Image.new("RGB", (8, 8)).save(images_dir / "a.png")
    with _patch_imagenet_ctx():
        ds = COCODataset(
            annotations=str(p),
            images=str(images_dir),
            transforms=_build_eval(8),
            text_prompt=TextPromptConfig(),
        )
    ex = ds[0]
    assert int(ex.instances[0].mask.sum()) > 0


def test_iscrowd_skipped(tmp_path: Path) -> None:
    from PIL import Image

    p = tmp_path / "ann.json"
    p.write_text(
        json.dumps(
            {
                "images": [{"id": 1, "file_name": "a.png", "width": 8, "height": 8}],
                "categories": [{"id": 1, "name": "x"}],
                "annotations": [
                    {
                        "id": 1,
                        "image_id": 1,
                        "category_id": 1,
                        "bbox": [0, 0, 4, 4],
                        "area": 16,
                        "iscrowd": 0,
                        "segmentation": [[0, 0, 4, 0, 4, 4, 0, 4]],
                    },
                    {
                        "id": 2,
                        "image_id": 1,
                        "category_id": 1,
                        "bbox": [4, 4, 4, 4],
                        "area": 16,
                        "iscrowd": 1,
                        "segmentation": [[4, 4, 8, 4, 8, 8, 4, 8]],
                    },
                ],
            }
        )
    )
    images_dir = tmp_path / "imgs"
    images_dir.mkdir()
    Image.new("RGB", (8, 8)).save(images_dir / "a.png")
    with _patch_imagenet_ctx():
        ds = COCODataset(
            annotations=str(p),
            images=str(images_dir),
            transforms=_build_eval(8),
            text_prompt=TextPromptConfig(),
        )
    ex = ds[0]
    assert len(ex.instances) == 1


def test_dropped_empty_image_logged_once(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    from PIL import Image

    p = tmp_path / "ann.json"
    p.write_text(
        json.dumps(
            {
                "images": [
                    {"id": 1, "file_name": "a.png", "width": 8, "height": 8},
                    {"id": 2, "file_name": "b.png", "width": 8, "height": 8},
                ],
                "categories": [{"id": 1, "name": "x"}],
                "annotations": [
                    {
                        "id": 1,
                        "image_id": 1,
                        "category_id": 1,
                        "bbox": [0, 0, 4, 4],
                        "area": 16,
                        "iscrowd": 0,
                        "segmentation": [[0, 0, 4, 0, 4, 4, 0, 4]],
                    },
                    {
                        "id": 2,
                        "image_id": 2,
                        "category_id": 1,
                        "bbox": [0, 0, 4, 4],
                        "area": 16,
                        "iscrowd": 1,
                        "segmentation": [[0, 0, 4, 0, 4, 4, 0, 4]],
                    },
                ],
            }
        )
    )
    images_dir = tmp_path / "imgs"
    images_dir.mkdir()
    Image.new("RGB", (8, 8)).save(images_dir / "a.png")
    Image.new("RGB", (8, 8)).save(images_dir / "b.png")
    caplog.set_level(logging.INFO, logger="custom_sam_peft.data.coco")
    with _patch_imagenet_ctx():
        COCODataset(
            annotations=str(p),
            images=str(images_dir),
            transforms=_build_eval(8),
            text_prompt=TextPromptConfig(),
        )
    drop_lines = [r for r in caplog.records if re.search(r"dropped.*iscrowd", r.message)]
    assert len(drop_lines) == 1


def test_image_resize_geometry(tiny_coco_dir: Path) -> None:
    with _patch_imagenet_ctx():
        ds = COCODataset(
            annotations=str(tiny_coco_dir / "annotations.json"),
            images=str(tiny_coco_dir / "images"),
            transforms=_build_eval(64),
            text_prompt=TextPromptConfig(),
        )
    ex = ds[0]
    assert ex.image.shape == (3, 64, 64)
    assert ex.instances[0].mask.shape == (64, 64)
    coords = ex.instances[0].box
    assert (coords >= 0).all() and (coords <= 64).all()


def test_sparse_to_dense_remap(tmp_path: Path) -> None:
    from PIL import Image

    p = tmp_path / "ann.json"
    p.write_text(
        json.dumps(
            {
                "images": [{"id": 1, "file_name": "a.png", "width": 8, "height": 8}],
                "categories": [
                    {"id": 7, "name": "g"},
                    {"id": 3, "name": "a"},
                ],
                "annotations": [
                    {
                        "id": 1,
                        "image_id": 1,
                        "category_id": 3,
                        "bbox": [0, 0, 4, 4],
                        "area": 16,
                        "iscrowd": 0,
                        "segmentation": [[0, 0, 4, 0, 4, 4, 0, 4]],
                    },
                    {
                        "id": 2,
                        "image_id": 1,
                        "category_id": 7,
                        "bbox": [4, 0, 4, 4],
                        "area": 16,
                        "iscrowd": 0,
                        "segmentation": [[4, 0, 8, 0, 8, 4, 4, 4]],
                    },
                ],
            }
        )
    )
    images_dir = tmp_path / "imgs"
    images_dir.mkdir()
    Image.new("RGB", (8, 8)).save(images_dir / "a.png")
    with _patch_imagenet_ctx():
        ds = COCODataset(
            annotations=str(p),
            images=str(images_dir),
            transforms=_build_eval(8),
            text_prompt=TextPromptConfig(),
        )
    assert len(ds.class_names) == 2
    assert ds.coco_category_ids == [3, 7]
    assert {int(inst.class_id) for inst in ds[0].instances} == {0, 1}


# ---------------------------------------------------------------------------
# Task 14: build_coco builder
# ---------------------------------------------------------------------------


def test_register_coco_lookup(tiny_coco_dir: Path) -> None:
    from custom_sam_peft._registry import lookup

    builder = lookup("dataset", "coco")
    cfg: dict[str, Any] = {
        "format": "coco",
        "train": {
            "annotations": str(tiny_coco_dir / "annotations.json"),
            "images": str(tiny_coco_dir / "images"),
        },
        "val": {
            "annotations": str(tiny_coco_dir / "annotations.json"),
            "images": str(tiny_coco_dir / "images"),
        },
        "augmentations": {"preset": "natural", "intensity": "medium"},
        "text_prompt": {"mode": "present"},
        "normalize": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
    }
    with _patch_imagenet_ctx():
        ds = builder(cfg, model_name="facebook/sam3.1", pipeline="eval")
    assert len(ds) == 2
    assert ds.class_names == ["thing_a", "thing_b"]


def test_build_coco_train_pipeline_uses_train_transforms(tiny_coco_dir: Path) -> None:
    from custom_sam_peft._registry import lookup

    builder = lookup("dataset", "coco")
    cfg: dict[str, Any] = {
        "format": "coco",
        "train": {
            "annotations": str(tiny_coco_dir / "annotations.json"),
            "images": str(tiny_coco_dir / "images"),
        },
        "val": {
            "annotations": str(tiny_coco_dir / "annotations.json"),
            "images": str(tiny_coco_dir / "images"),
        },
        "augmentations": {"preset": "none"},
        "text_prompt": {"mode": "present"},
        "normalize": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
    }
    with _patch_imagenet_ctx():
        ds = builder(cfg, model_name="facebook/sam3.1", pipeline="train")
    assert len(ds) == 2


def test_deterministic_text_sampling_under_fixed_seed(tmp_path: Path) -> None:
    ann, imgs = _synth_many_cats(tmp_path, 5)

    def build() -> COCODataset:
        with _patch_imagenet_ctx():
            return COCODataset(
                annotations=str(ann),
                images=str(imgs),
                transforms=_build_eval(32),
                text_prompt=TextPromptConfig(mode="sampled_fixed_k", k=4),
                seed=42,
            )

    a = build()[0]
    b = build()[0]
    assert isinstance(a.prompts, TextPrompts)
    assert isinstance(b.prompts, TextPrompts)
    assert a.prompts.classes == b.prompts.classes


# ---------------------------------------------------------------------------
# spec/data-no-val-auto-split (#71): image_ids subset parameter
# ---------------------------------------------------------------------------


def test_cocodataset_image_ids_filters_to_subset(tiny_coco_dir: Path) -> None:
    """Spec §6.1: image_ids restricts the dataset to the requested subset."""
    with _patch_imagenet_ctx():
        full = COCODataset(
            annotations=str(tiny_coco_dir / "annotations.json"),
            images=str(tiny_coco_dir / "images"),
            transforms=_build_eval(),
            text_prompt=TextPromptConfig(),
        )
    all_ids = list(full._image_ids)
    assert len(all_ids) >= 2
    subset = all_ids[:1]
    with _patch_imagenet_ctx():
        ds = COCODataset(
            annotations=str(tiny_coco_dir / "annotations.json"),
            images=str(tiny_coco_dir / "images"),
            transforms=_build_eval(),
            text_prompt=TextPromptConfig(),
            image_ids=subset,
        )
    assert len(ds) == 1
    ex = ds[0]
    assert int(ex.image_id) == subset[0]


def test_cocodataset_image_ids_missing_raises_value_error(tiny_coco_dir: Path) -> None:
    """Spec §6.1: requesting an image_id not present (or crowd-only) raises ValueError."""
    with _patch_imagenet_ctx(), pytest.raises(ValueError, match="not present"):
        COCODataset(
            annotations=str(tiny_coco_dir / "annotations.json"),
            images=str(tiny_coco_dir / "images"),
            transforms=_build_eval(),
            text_prompt=TextPromptConfig(),
            image_ids=[999999],
        )


def test_cocodataset_image_ids_none_preserves_existing_behavior(tiny_coco_dir: Path) -> None:
    """When image_ids is None, the dataset behaves exactly as before."""
    with _patch_imagenet_ctx():
        ds_a = COCODataset(
            annotations=str(tiny_coco_dir / "annotations.json"),
            images=str(tiny_coco_dir / "images"),
            transforms=_build_eval(),
            text_prompt=TextPromptConfig(),
        )
        ds_b = COCODataset(
            annotations=str(tiny_coco_dir / "annotations.json"),
            images=str(tiny_coco_dir / "images"),
            transforms=_build_eval(),
            text_prompt=TextPromptConfig(),
            image_ids=None,
        )
    assert len(ds_a) == len(ds_b)


def test_cocodataset_image_ids_sorted_order_preserved(tiny_coco_dir: Path) -> None:
    """Internal _image_ids list must be in ascending order regardless of caller-supplied order."""
    with _patch_imagenet_ctx():
        full = COCODataset(
            annotations=str(tiny_coco_dir / "annotations.json"),
            images=str(tiny_coco_dir / "images"),
            transforms=_build_eval(),
            text_prompt=TextPromptConfig(),
        )
    ids_sorted_desc = sorted(full._image_ids, reverse=True)
    with _patch_imagenet_ctx():
        ds = COCODataset(
            annotations=str(tiny_coco_dir / "annotations.json"),
            images=str(tiny_coco_dir / "images"),
            transforms=_build_eval(),
            text_prompt=TextPromptConfig(),
            image_ids=ids_sorted_desc,
        )
    assert ds._image_ids == sorted(ds._image_ids)


def test_image_level_leak_invariant_on_tiny_coco(tiny_coco_dir: Path) -> None:
    """Spec §9.4.5: stratified_split on tiny_coco items yields disjoint train/val ids."""
    from custom_sam_peft.config.schema import DataConfig, DataSplit
    from custom_sam_peft.data.splitter import stratified_split
    from custom_sam_peft.data.val_source import _enumerate_coco_items

    data_cfg = DataConfig(
        format="coco",
        train=DataSplit(
            annotations=str(tiny_coco_dir / "annotations.json"),
            images=str(tiny_coco_dir / "images"),
        ),
    )
    items = _enumerate_coco_items(data_cfg)
    if len(items) < 2:
        pytest.skip("tiny_coco has < 2 keep-after-crowd-filter images; cannot test split")
    res = stratified_split(items, fraction=0.5, seed=0)
    assert set(res.train_ids).isdisjoint(set(res.val_ids))


# ---------------------------------------------------------------------------
# Task 7+8: channel-aware _decode_image in COCODataset
# ---------------------------------------------------------------------------


def test_coco_decode_image_uses_channels(tmp_path, monkeypatch):
    """COCODataset._decode_image routes through read_image with self._channels."""
    import numpy as np

    from custom_sam_peft.data import coco as coco_mod

    captured = {}

    def fake_read_image(path, channels):
        captured["channels"] = channels
        return np.zeros((4, 5, channels), np.uint8)

    monkeypatch.setattr(coco_mod, "read_image", fake_read_image, raising=False)
    obj = coco_mod.COCODataset.__new__(coco_mod.COCODataset)
    obj._image_root = tmp_path
    obj._channels = 5
    raw = (1, {"file_name": "a.png"}, [])
    out = coco_mod.COCODataset._decode_image(obj, raw)
    assert out.shape == (4, 5, 5)
    assert captured["channels"] == 5
