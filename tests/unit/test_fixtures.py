"""Verify the fixtures are well-formed."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from tests.fixtures.tiny_sam3_stub import TinySam3Stub


def test_tiny_coco_annotations_load(tiny_coco_dir: Path) -> None:
    data = json.loads((tiny_coco_dir / "annotations.json").read_text())
    assert len(data["images"]) == 2
    assert len(data["categories"]) == 2
    assert len(data["annotations"]) == 3


def test_tiny_coco_images_exist(tiny_coco_dir: Path) -> None:
    for name in ("img_000001.png", "img_000002.png"):
        assert (tiny_coco_dir / "images" / name).is_file()


def test_stub_model_forward_returns_expected_keys(stub_model: TinySam3Stub) -> None:
    image = torch.zeros((2, 3, 32, 32))
    out = stub_model(image, prompts=None)
    assert set(out.keys()) == {"masks", "boxes", "objectness", "class_logits"}
    assert out["masks"].shape == (2, 1, 32, 32)
    assert out["boxes"].shape == (2, 1, 4)
