"""Tests for predict/visualize.py — Phase 4."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pycocotools.mask as mask_utils
from PIL import Image

from custom_sam_peft.predict.visualize import (
    color_for_class,
    write_visualization,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_rle_entry(
    image_id: int = 1,
    category_id: int = 1,
    h: int = 32,
    w: int = 32,
    score: float = 0.9,
    bbox: list[float] | None = None,
) -> dict[str, object]:
    """Return a synthetic entry with an RLE segmentation matching spec §7.1."""
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[4:10, 4:10] = 1  # small filled square
    rle = mask_utils.encode(np.asfortranarray(mask))
    seg: dict[str, object] = {
        "size": list(rle["size"]),
        "counts": rle["counts"].decode("ascii"),
    }
    if bbox is None:
        bbox = [4.0, 4.0, 6.0, 6.0]
    return {
        "image_id": image_id,
        "category_id": category_id,
        "bbox": bbox,
        "score": score,
        "segmentation": seg,
    }


def _make_image(w: int = 32, h: int = 32) -> Image.Image:
    return Image.new("RGB", (w, h), color=(200, 200, 200))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestWriteVisualizationWritesPng:
    """test_visualize_writes_png — file exists and opens via PIL."""

    def test_visualize_writes_png(self, tmp_path: Path) -> None:
        img = _make_image()
        img_path = tmp_path / "test_img.png"
        img.save(img_path)

        entry = _make_rle_entry()
        out = write_visualization(img_path, [entry], tmp_path, prompts=["cat"])

        assert out.exists(), "visualization PNG was not written"
        assert out.suffix == ".png"
        assert out.parent.name == "visualizations"
        # Must open via PIL without error
        reopened = Image.open(out)
        reopened.verify()


class TestColorDeterminism:
    """test_visualize_color_deterministic_per_class — hash is cross-process stable."""

    def test_visualize_color_deterministic_per_class(self) -> None:
        c1 = color_for_class("cat")
        c2 = color_for_class("cat")
        assert c1 == c2, "color_for_class must return the same color on repeated calls"

    def test_visualize_color_differs_per_class(self) -> None:
        # With a palette of ≥ 8 colors and different class names, these should differ.
        # There is a tiny theoretical collision chance; we pick names that differ clearly.
        c_cat = color_for_class("cat")
        c_dog = color_for_class("dog")
        assert c_cat != c_dog, "different class names must produce different palette colors"


class TestScoreLabelDrawn:
    """test_visualize_score_label_drawn — visible non-background pixels exist."""

    def test_visualize_score_label_drawn(self, tmp_path: Path) -> None:
        bg_color = (200, 200, 200)
        img = Image.new("RGB", (128, 128), color=bg_color)
        img_path = tmp_path / "label_test.png"
        img.save(img_path)

        entry = _make_rle_entry(h=128, w=128, score=0.95, bbox=[4.0, 4.0, 60.0, 60.0])
        out = write_visualization(img_path, [entry], tmp_path, prompts=["cat"])

        rendered = np.array(Image.open(out).convert("RGB"))
        # At least some pixels should differ from the solid background
        bg = np.array(bg_color, dtype=np.uint8)
        diff = np.any(rendered != bg, axis=-1)
        assert diff.sum() > 0, "rendered image should have pixels different from background"


class TestEmptyEntries:
    """test_visualize_handles_empty_entries — zero entries writes a copy, no crash."""

    def test_visualize_handles_empty_entries(self, tmp_path: Path) -> None:
        img = _make_image()
        img_path = tmp_path / "empty_entries.png"
        img.save(img_path)

        out = write_visualization(img_path, [], tmp_path, prompts=["cat"])

        assert out.exists(), "PNG should be written even with zero entries"
        reopened = Image.open(out)
        assert reopened.size == img.size


class TestNoSegmentation:
    """test_visualize_skips_when_no_segmentation — bbox-only path, no crash."""

    def test_visualize_skips_when_no_segmentation(self, tmp_path: Path) -> None:
        img = _make_image()
        img_path = tmp_path / "no_seg.png"
        img.save(img_path)

        # Entry without 'segmentation' key — as produced by --save-masks=none
        entry: dict[str, object] = {
            "image_id": 1,
            "category_id": 1,
            "bbox": [4.0, 4.0, 10.0, 10.0],
            "score": 0.8,
        }

        # Must not crash
        out = write_visualization(img_path, [entry], tmp_path, prompts=["cat"])
        assert out.exists()
