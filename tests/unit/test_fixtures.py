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
    assert set(out.keys()) == {
        "pred_logits",
        "pred_boxes",
        "pred_masks",
        "presence_logit_dec",
    }
    assert out["pred_logits"].shape == (2, stub_model.num_queries, 1)
    assert out["pred_boxes"].shape == (2, stub_model.num_queries, 4)
    assert out["pred_masks"].shape == (
        2,
        stub_model.num_queries,
        stub_model.mask_size,
        stub_model.mask_size,
    )
    assert out["presence_logit_dec"].shape == (2, 1)
