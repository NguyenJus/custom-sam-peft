"""Smoke-import the new data-loading deps so missing-dep failures surface here."""

from __future__ import annotations


def test_albumentations_imports() -> None:
    import albumentations as A

    assert hasattr(A, "Compose")
    assert hasattr(A, "LongestMaxSize")
    assert hasattr(A, "PadIfNeeded")
    assert hasattr(A, "Normalize")
    assert hasattr(A, "HorizontalFlip")
    assert hasattr(A, "ColorJitter")
    assert hasattr(A, "BboxParams")


def test_albumentations_to_tensor_v2_imports() -> None:
    from albumentations.pytorch import ToTensorV2

    assert ToTensorV2 is not None


def test_cv2_imports_headless() -> None:
    import cv2

    assert hasattr(cv2, "INTER_LINEAR")
    assert hasattr(cv2, "BORDER_CONSTANT")


def test_pycocotools_mask_imports() -> None:
    from pycocotools import mask as coco_mask

    assert hasattr(coco_mask, "frPyObjects")
    assert hasattr(coco_mask, "decode")


def test_datasets_imports() -> None:
    import datasets

    assert hasattr(datasets, "Dataset")
    assert hasattr(datasets, "load_dataset")


def test_pillow_imports() -> None:
    from PIL import Image

    assert hasattr(Image, "open")
