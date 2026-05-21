"""One-shot generator for the tiny_coco test fixture.

Run from the repo root once: `uv run python tests/fixtures/make_tiny_coco.py`.
The generated PNGs + JSON are committed; this script exists so the fixture
is reproducible.
"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
OUT = HERE / "tiny_coco"
IMG_DIR = OUT / "images"


def main() -> None:
    IMG_DIR.mkdir(parents=True, exist_ok=True)

    # Two 32x32 RGB images, each a solid color.
    Image.new("RGB", (32, 32), color=(200, 50, 50)).save(IMG_DIR / "img_000001.png")
    Image.new("RGB", (32, 32), color=(50, 200, 50)).save(IMG_DIR / "img_000002.png")

    annotations = {
        "info": {"description": "tiny_coco — custom_sam_peft test fixture", "version": "1.0"},
        "licenses": [],
        "images": [
            {"id": 1, "file_name": "img_000001.png", "width": 32, "height": 32},
            {"id": 2, "file_name": "img_000002.png", "width": 32, "height": 32},
        ],
        "categories": [
            {"id": 1, "name": "thing_a", "supercategory": "thing"},
            {"id": 2, "name": "thing_b", "supercategory": "thing"},
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [4, 4, 12, 12],
                "area": 144,
                "iscrowd": 0,
                "segmentation": [[4, 4, 16, 4, 16, 16, 4, 16]],
            },
            {
                "id": 2,
                "image_id": 1,
                "category_id": 2,
                "bbox": [18, 18, 10, 10],
                "area": 100,
                "iscrowd": 0,
                "segmentation": [[18, 18, 28, 18, 28, 28, 18, 28]],
            },
            {
                "id": 3,
                "image_id": 2,
                "category_id": 1,
                "bbox": [8, 8, 16, 16],
                "area": 256,
                "iscrowd": 0,
                "segmentation": [[8, 8, 24, 8, 24, 24, 8, 24]],
            },
        ],
    }
    (OUT / "annotations.json").write_text(json.dumps(annotations, indent=2))


if __name__ == "__main__":
    main()
