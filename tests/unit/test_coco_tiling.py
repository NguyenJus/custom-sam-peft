"""CPU-only: construct a COCODataset over a synthetic oversized raster + COCO
annotations; assert tile-expanded __len__, per-window clipping, empty-tile negatives."""

import json

import numpy as np
import pytest
from PIL import Image

from custom_sam_peft.config.schema import NormalizeConfig, TextPromptConfig
from custom_sam_peft.data.coco import COCODataset
from custom_sam_peft.data.transforms import build_eval_transforms


@pytest.fixture
def _eval_transforms():
    return build_eval_transforms(1008, model_name="<test>", normalize=NormalizeConfig())


@pytest.fixture
def oversized_coco(tmp_path):
    img = (np.random.rand(1500, 1500, 3) * 255).astype(np.uint8)
    imgs_dir = tmp_path / "imgs"
    imgs_dir.mkdir()
    Image.fromarray(img).save(imgs_dir / "big.png")
    coco = {
        "images": [{"id": 1, "file_name": "big.png", "width": 1500, "height": 1500}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [10, 10, 40, 40],
                "area": 1600,
                "iscrowd": 0,
                "segmentation": [[10, 10, 50, 10, 50, 50, 10, 50]],
            },
        ],
        "categories": [{"id": 1, "name": "thing"}],
    }
    ann = tmp_path / "ann.json"
    ann.write_text(json.dumps(coco))
    return str(ann), str(imgs_dir)


def test_C6_oversized_raster_expands_into_tiles(oversized_coco, _eval_transforms):
    ann, imgs = oversized_coco
    ds = COCODataset(
        annotations=ann,
        images=imgs,
        transforms=_eval_transforms,
        text_prompt=TextPromptConfig(),
        channels=3,
    )
    # 1500x1500 @ tile 1008 overlap 0.25 -> 2x2 = 4 windows -> len == 4 (one image)
    assert len(ds) == 4
    # tile containing the top-left object yields >=1 instance; an empty tile is valid.
    n_with, n_empty = 0, 0
    for k in range(len(ds)):
        ex = ds[k]
        if len(ex.instances):
            n_with += 1
        else:
            n_empty += 1
    assert n_with >= 1 and n_empty >= 1  # empty tiles are valid negatives


def test_C6_image_class_labels_aligns_with_expanded_len(oversized_coco, _eval_transforms):
    ann, imgs = oversized_coco
    ds = COCODataset(
        annotations=ann,
        images=imgs,
        transforms=_eval_transforms,
        text_prompt=TextPromptConfig(),
        channels=3,
    )
    # The stratified-subset label cache must be per-sample, not per-image, or the
    # data.limit consumer (resolve_subset_indices) indexes out of range.
    assert len(ds.image_class_labels) == len(ds)
