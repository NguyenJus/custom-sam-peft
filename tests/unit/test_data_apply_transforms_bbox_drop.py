"""Regression tests for bbox-drop invariant in _apply_transforms (COCO and HF paths).

When Albumentations drops an out-of-frame bbox it also removes the parallel
class_labels entry, but the masks list is processed separately and keeps its
original length.  Before the fix this caused:

    zip() argument 2 is longer than argument 1

in _pack_example (both COCO and HF) because strict=True zip over
(out_bboxes, out_masks, out_classes) would find len(out_masks) > len(out_bboxes).

The fix adds `instance_idx` as a second label field so that after the call the
surviving masks can be re-selected by original index, restoring the invariant.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any
from unittest.mock import MagicMock
from unittest.mock import patch as mock_patch

import albumentations as A
import datasets as hf_datasets
import numpy as np
import pytest
from albumentations.pytorch import ToTensorV2
from PIL import Image

from custom_sam_peft.config.schema import HFFieldMap, TextPromptConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@contextmanager
def _patch_imagenet_ctx() -> Iterator[None]:
    """Patch AutoImageProcessor so resolve_normalization falls back to ImageNet defaults."""
    mock_aip = MagicMock()
    mock_aip.from_pretrained.side_effect = OSError("no cache")
    with mock_patch("transformers.AutoImageProcessor", mock_aip):
        yield


def _make_bbox_dropping_compose(image_size: int = 32) -> A.Compose:
    """Build a deterministic A.Compose that ALWAYS drops the first bbox.

    We use Crop to cut off the left half of the image — a bbox in the left half
    (x2 <= image_size/2) will be fully cropped out and Albumentations will drop
    it from bboxes/class_labels, but the corresponding mask entry remains at
    its original index in the masks list (which keeps its full length).

    This compose matches the FIXED production BboxParams (both ``"class_labels"``
    and ``"instance_idx"`` as label fields) so that the fix is exercised end-to-end:

    * ``_apply_transforms`` passes ``instance_idx=list(range(len(masks)))``
    * Albumentations filters ``instance_idx`` in lockstep with ``class_labels``
      when a bbox is dropped (both are label fields)
    * ``_apply_transforms`` re-selects ``out["masks"]`` by the surviving indices
    * all three returned lists have equal length

    Before the fix (when ``instance_idx`` was absent from label_fields and from
    the transform call), ``_apply_transforms`` returned ``len(out_masks)=2`` vs
    ``len(out_bboxes)=1``, and the strict-zip in ``_pack_example`` raised
    ``ValueError: zip() argument 2 is longer than argument 1``.
    """
    half = image_size // 2
    return A.Compose(
        [
            # Crop to right half — any bbox in left half is dropped.
            A.Crop(x_min=half, y_min=0, x_max=image_size, y_max=image_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["class_labels", "instance_idx"],
            min_visibility=0.0,
            min_area=0.0,
        ),
    )


# ---------------------------------------------------------------------------
# COCO path
# ---------------------------------------------------------------------------


def _make_coco_dataset_with_transform(
    tmp_path: Any,
    transforms: A.Compose,
) -> Any:
    """Build a minimal COCODataset backed by a one-image COCO fixture."""
    import json

    from custom_sam_peft.data.coco import COCODataset

    # Write a minimal valid COCO annotations file.
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    ann_path = tmp_path / "annotations.json"

    # 32x32 image, two annotations:
    #   ann 0: bbox fully in the LEFT half  → will be cropped out
    #   ann 1: bbox fully in the RIGHT half → survives
    ann_data = {
        "images": [{"id": 1, "file_name": "img.png", "width": 32, "height": 32}],
        "categories": [{"id": 1, "name": "cat", "supercategory": "animal"}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [2.0, 2.0, 10.0, 10.0],  # left half, x=[2,12]
                "area": 100.0,
                "iscrowd": 0,
                "segmentation": [[2, 2, 12, 2, 12, 12, 2, 12]],
            },
            {
                "id": 2,
                "image_id": 1,
                "category_id": 1,
                "bbox": [18.0, 2.0, 10.0, 10.0],  # right half, x=[18,28]
                "area": 100.0,
                "iscrowd": 0,
                "segmentation": [[18, 2, 28, 2, 28, 12, 18, 12]],
            },
        ],
    }
    ann_path.write_text(json.dumps(ann_data))

    # Write a tiny RGB image.
    img_arr = np.zeros((32, 32, 3), dtype=np.uint8)
    img_arr[:, :, 0] = 128  # non-trivial so normalize doesn't produce NaN
    img = Image.fromarray(img_arr)
    img.save(img_dir / "img.png")

    with _patch_imagenet_ctx():
        ds = COCODataset(
            annotations=str(ann_path),
            images=str(img_dir),
            prompt_mode="bbox",
            transforms=transforms,
            text_prompt=TextPromptConfig(),
        )
    return ds


def test_coco_apply_transforms_bbox_drop_keeps_aligned_lengths(
    tmp_path: Any,
) -> None:
    """After a bbox is dropped by augmentation, bboxes/masks/classes stay equal-length.

    Before the fix _apply_transforms returns len(out_masks)=2, len(out_bboxes)=1
    because Albumentations filters bboxes/class_labels but leaves the masks list
    at its original length.  _pack_example then raises a strict-zip error.
    """
    compose = _make_bbox_dropping_compose(image_size=32)
    ds = _make_coco_dataset_with_transform(tmp_path, compose)

    # Must not raise; must return equal-length outputs.
    _image_tensor, out_bboxes, out_masks, out_classes = ds._apply_transforms(
        np_img=np.zeros((32, 32, 3), dtype=np.uint8),
        bboxes_xyxy=[[2.0, 2.0, 12.0, 12.0], [18.0, 2.0, 28.0, 12.0]],
        masks=[
            np.ones((32, 32), dtype=np.uint8),
            np.ones((32, 32), dtype=np.uint8),
        ],
        class_labels=[0, 0],
    )

    assert len(out_bboxes) == len(out_masks) == len(out_classes), (
        f"After bbox drop, lengths must match: "
        f"bboxes={len(out_bboxes)}, masks={len(out_masks)}, classes={len(out_classes)}"
    )
    # Specifically: 1 bbox survived (the right-half one), so exactly 1 mask/class.
    assert len(out_bboxes) == 1


def test_coco_getitem_bbox_drop_does_not_raise(tmp_path: Any) -> None:
    """Full __getitem__ path does not raise when augmentation drops a bbox."""
    compose = _make_bbox_dropping_compose(image_size=32)
    ds = _make_coco_dataset_with_transform(tmp_path, compose)
    # This calls _apply_transforms → _pack_example (strict zip) internally.
    ex = ds[0]
    assert len(ex.instances) == 1


def test_coco_apply_transforms_empty_annotations_still_works(
    tmp_path: Any,
) -> None:
    """Empty annotation lists (no boxes) work correctly — list(range(0)) == []."""
    compose = _make_bbox_dropping_compose(image_size=32)
    ds = _make_coco_dataset_with_transform(tmp_path, compose)

    _image_tensor, out_bboxes, out_masks, out_classes = ds._apply_transforms(
        np_img=np.zeros((32, 32, 3), dtype=np.uint8),
        bboxes_xyxy=[],
        masks=[],
        class_labels=[],
    )
    assert out_bboxes == []
    assert out_masks == []
    assert out_classes == []


# ---------------------------------------------------------------------------
# HF path
# ---------------------------------------------------------------------------


def _build_hf_dropping_dataset(image_size: int = 32) -> hf_datasets.Dataset:
    """Build a minimal HF Dataset with 2 objects; left-half bbox will be dropped.

    Bboxes are in xyxy format (the HFFieldMap default: bbox_format="xyxy"):
      left-half:  [2, 2, 12, 12]  → x=[2,12]  → cropped out after Crop(x_min=16)
      right-half: [18, 2, 28, 12] → x=[18,28] → survives
    """
    images = [Image.new("RGB", (image_size, image_size))]
    bboxes = [[[2.0, 2.0, 12.0, 12.0], [18.0, 2.0, 28.0, 12.0]]]
    categories = [[0, 0]]
    cols: dict[str, Any] = {
        "image": images,
        "objects": [{"bbox": bboxes[0], "category": categories[0]}],
        "categories": [["thing"]],
    }
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


def _patch_hf_load_dataset(
    monkeypatch: pytest.MonkeyPatch,
    ds: hf_datasets.Dataset,
) -> None:
    def fake(name: str, split: str, **kwargs: object) -> hf_datasets.Dataset:
        return ds

    monkeypatch.setattr("custom_sam_peft.data.hf.hf_load_dataset", fake)


def test_hf_apply_transforms_bbox_drop_keeps_aligned_lengths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """HF _apply_transforms returns equal-length bboxes/masks/classes after a drop."""
    from custom_sam_peft.data.hf import HFDataset

    ds = _build_hf_dropping_dataset(image_size=32)
    _patch_hf_load_dataset(monkeypatch, ds)

    compose = _make_bbox_dropping_compose(image_size=32)
    with _patch_imagenet_ctx():
        hfds = HFDataset(
            name="x",
            split="train",
            prompt_mode="bbox",
            transforms=compose,
            text_prompt=TextPromptConfig(),
            field_map=HFFieldMap(segmentation=None),
        )

    # Call _apply_transforms directly with 2 masks but only 1 bbox surviving.
    _image_tensor, out_bboxes, out_masks, out_classes = hfds._apply_transforms(
        np_img=np.zeros((32, 32, 3), dtype=np.uint8),
        bboxes_xyxy=[(2.0, 2.0, 12.0, 12.0), (18.0, 2.0, 28.0, 12.0)],
        masks=[
            np.ones((32, 32), dtype=np.uint8),
            np.ones((32, 32), dtype=np.uint8),
        ],
        classes=[0, 0],
    )

    assert len(out_bboxes) == len(out_masks) == len(out_classes), (
        f"After bbox drop, lengths must match: "
        f"bboxes={len(out_bboxes)}, masks={len(out_masks)}, classes={len(out_classes)}"
    )
    assert len(out_bboxes) == 1


def test_hf_getitem_bbox_drop_does_not_raise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Full HFDataset __getitem__ does not raise when augmentation drops a bbox."""
    from custom_sam_peft.data.hf import HFDataset

    ds = _build_hf_dropping_dataset(image_size=32)
    _patch_hf_load_dataset(monkeypatch, ds)

    compose = _make_bbox_dropping_compose(image_size=32)
    with _patch_imagenet_ctx():
        hfds = HFDataset(
            name="x",
            split="train",
            prompt_mode="bbox",
            transforms=compose,
            text_prompt=TextPromptConfig(),
            field_map=HFFieldMap(segmentation=None),
        )

    ex = hfds[0]
    assert len(ex.instances) == 1


def test_hf_apply_transforms_empty_annotations_still_works(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """HF path empty annotation list works correctly."""
    from custom_sam_peft.data.hf import HFDataset

    ds = _build_hf_dropping_dataset(image_size=32)
    _patch_hf_load_dataset(monkeypatch, ds)

    compose = _make_bbox_dropping_compose(image_size=32)
    with _patch_imagenet_ctx():
        hfds = HFDataset(
            name="x",
            split="train",
            prompt_mode="bbox",
            transforms=compose,
            text_prompt=TextPromptConfig(),
            field_map=HFFieldMap(segmentation=None),
        )

    _image_tensor, out_bboxes, out_masks, out_classes = hfds._apply_transforms(
        np_img=np.zeros((32, 32, 3), dtype=np.uint8),
        bboxes_xyxy=[],
        masks=[],
        classes=[],
    )
    assert out_bboxes == []
    assert out_masks == []
    assert out_classes == []
