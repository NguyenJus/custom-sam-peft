"""Tests for predict/writers.py — output writers for csp predict.

All tests are CPU-only; no model loading needed.
"""

from __future__ import annotations

import json
import unittest.mock as mock
from pathlib import Path

import numpy as np
import pycocotools.mask as mask_utils
from PIL import Image

from custom_sam_peft.predict.writers import (
    encode_rle_dict,
    select_top_k_per_image_class,
    write_image_id_map,
    write_predictions,
    write_run_json,
)

# ---------------------------------------------------------------------------
# Helpers to build synthetic entries
# ---------------------------------------------------------------------------


def _make_rle(h: int = 16, w: int = 16) -> dict:
    """Create a synthetic pycocotools RLE dict (ASCII counts) for a small mask."""
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[2:6, 3:9] = 1
    return encode_rle_dict(mask)


def _make_entry(
    image_id: int = 1,
    category_id: int = 1,
    score: float = 0.9,
    h: int = 16,
    w: int = 16,
) -> dict:
    return {
        "image_id": image_id,
        "category_id": category_id,
        "bbox": [10.0, 20.0, 30.0, 40.0],
        "score": score,
        "segmentation": _make_rle(h, w),
    }


# ---------------------------------------------------------------------------
# 1. test_write_predictions_json_schema
# ---------------------------------------------------------------------------


def test_write_predictions_json_schema(tmp_path: Path) -> None:
    entries = [_make_entry()]
    write_predictions(
        entries,
        tmp_path,
        save_masks="rle",
        originals={1: (16, 16)},
    )
    pred_file = tmp_path / "predictions.json"
    assert pred_file.exists()
    data = json.loads(pred_file.read_text())
    assert isinstance(data, list)
    assert len(data) == 1
    rec = data[0]
    for field in ("image_id", "category_id", "bbox", "score", "segmentation"):
        assert field in rec, f"missing field: {field}"


# ---------------------------------------------------------------------------
# 2. test_predictions_bbox_is_xywh_in_original_coords
# ---------------------------------------------------------------------------


def test_predictions_bbox_is_xywh_in_original_coords(tmp_path: Path) -> None:
    """Writer must NOT mutate bbox; caller is responsible for coordinate space."""
    original_bbox = [10.5, 20.5, 30.0, 40.0]
    entry = {
        "image_id": 1,
        "category_id": 1,
        "bbox": list(original_bbox),  # copy so mutation would show
        "score": 0.8,
        "segmentation": _make_rle(),
    }
    write_predictions([entry], tmp_path, save_masks="rle", originals={1: (16, 16)})
    data = json.loads((tmp_path / "predictions.json").read_text())
    assert data[0]["bbox"] == original_bbox


# ---------------------------------------------------------------------------
# 3. test_predictions_rle_round_trip_via_pycocotools
# ---------------------------------------------------------------------------


def test_predictions_rle_round_trip_via_pycocotools(tmp_path: Path) -> None:
    """segmentation.counts must be ASCII string decodable by pycocotools."""
    h, w = 32, 32
    entries = [_make_entry(h=h, w=w)]
    write_predictions(entries, tmp_path, save_masks="rle", originals={1: (h, w)})
    data = json.loads((tmp_path / "predictions.json").read_text())
    rle = data[0]["segmentation"]
    # counts must be a string (ASCII-decoded), not bytes
    assert isinstance(rle["counts"], str)
    # pycocotools must be able to decode it back to a 2-D mask
    decoded = mask_utils.decode({"size": rle["size"], "counts": rle["counts"].encode("ascii")})
    assert decoded.shape == (h, w)
    assert decoded.dtype == np.uint8


# ---------------------------------------------------------------------------
# 4. test_image_id_map_json_shape
# ---------------------------------------------------------------------------


def test_image_id_map_json_shape(tmp_path: Path) -> None:
    """Keys must be stringified ints; values must be stringified absolute paths."""
    id_to_path: dict[int, Path] = {
        12345: Path("/abs/path/img001.jpg"),
        99999: Path("/abs/path/img002.png"),
    }
    write_image_id_map(id_to_path, tmp_path)
    data = json.loads((tmp_path / "image_id_map.json").read_text())
    assert set(data.keys()) == {"12345", "99999"}
    assert data["12345"] == "/abs/path/img001.jpg"
    assert data["99999"] == "/abs/path/img002.png"


# ---------------------------------------------------------------------------
# 5. test_run_json_has_all_required_keys
# ---------------------------------------------------------------------------


def test_run_json_has_all_required_keys(tmp_path: Path) -> None:
    """All keys from spec §7.3 must be present; base-model-only → null checkpoint/adapter_kind."""
    required_keys = {
        "model",
        "checkpoint",
        "adapter_kind",
        "merge_adapter",
        "prompts",
        "score_threshold",
        "top_k",
        "mask_threshold",
        "device",
        "dtype",
        "image_size",
        "batch_size",
        "seed",
        "version",
        "git_sha",
        "n_images",
        "n_predictions",
        "elapsed_sec",
    }
    run_meta = {
        "model": "facebook/sam3.1",
        "checkpoint": None,
        "adapter_kind": None,
        "merge_adapter": True,
        "prompts": ["cat", "dog"],
        "score_threshold": 0.3,
        "top_k": 100,
        "mask_threshold": 0.0,
        "device": "cpu",
        "dtype": "float32",
        "image_size": 1024,
        "batch_size": 1,
        "seed": 0,
        "n_images": 2,
        "n_predictions": 5,
        "elapsed_sec": 1.23,
    }
    write_run_json(run_meta, tmp_path)
    data = json.loads((tmp_path / "run.json").read_text())
    assert required_keys <= set(data.keys()), f"missing keys: {required_keys - set(data.keys())}"
    # Base-model-only path: checkpoint and adapter_kind must be null (None → JSON null)
    assert data["checkpoint"] is None
    assert data["adapter_kind"] is None


# ---------------------------------------------------------------------------
# 6. test_run_json_git_sha_optional
# ---------------------------------------------------------------------------


def test_run_json_git_sha_optional(tmp_path: Path) -> None:
    """When called from a non-git path, git_sha must be None (not the string 'None')."""
    run_meta = {
        "model": "facebook/sam3.1",
        "checkpoint": None,
        "adapter_kind": None,
        "merge_adapter": True,
        "prompts": ["cat"],
        "score_threshold": 0.3,
        "top_k": 100,
        "mask_threshold": 0.0,
        "device": "cpu",
        "dtype": "float32",
        "image_size": 1024,
        "batch_size": 1,
        "seed": 0,
        "n_images": 1,
        "n_predictions": 0,
        "elapsed_sec": 0.5,
    }
    # Write run.json from a non-git directory by monkeypatching subprocess.
    # Simulate git failure (non-zero returncode)
    failed_result = mock.MagicMock()
    failed_result.returncode = 128
    failed_result.stdout = ""

    with mock.patch("subprocess.run", return_value=failed_result):
        write_run_json(run_meta, tmp_path)

    data = json.loads((tmp_path / "run.json").read_text())
    assert "git_sha" in data
    assert data["git_sha"] is None, f"expected None, got {data['git_sha']!r}"


# ---------------------------------------------------------------------------
# 7. test_top_k_and_score_threshold_per_image_class
# ---------------------------------------------------------------------------


def test_top_k_and_score_threshold_per_image_class() -> None:
    """Filter score >= threshold FIRST, then group by (image_id, category_id), keep top-K."""
    # image_id=1, category_id=1: 4 entries at various scores
    # image_id=1, category_id=2: 3 entries
    # image_id=2, category_id=1: 2 entries
    entries = [
        # image 1, class 1
        {"image_id": 1, "category_id": 1, "score": 0.9, "bbox": [], "segmentation": {}},
        {"image_id": 1, "category_id": 1, "score": 0.8, "bbox": [], "segmentation": {}},
        {"image_id": 1, "category_id": 1, "score": 0.5, "bbox": [], "segmentation": {}},
        # score 0.2 is below threshold=0.3 → filtered out
        {"image_id": 1, "category_id": 1, "score": 0.2, "bbox": [], "segmentation": {}},
        # image 1, class 2
        {"image_id": 1, "category_id": 2, "score": 0.95, "bbox": [], "segmentation": {}},
        {"image_id": 1, "category_id": 2, "score": 0.7, "bbox": [], "segmentation": {}},
        {"image_id": 1, "category_id": 2, "score": 0.6, "bbox": [], "segmentation": {}},
        # image 2, class 1
        {"image_id": 2, "category_id": 1, "score": 0.85, "bbox": [], "segmentation": {}},
        # score 0.25 is below threshold=0.3 → filtered out
        {"image_id": 2, "category_id": 1, "score": 0.25, "bbox": [], "segmentation": {}},
    ]

    result = select_top_k_per_image_class(entries, score_threshold=0.3, top_k=2)

    # Verify threshold filtering first
    scores = [e["score"] for e in result]
    assert all(s >= 0.3 for s in scores), f"found score below threshold: {scores}"

    # Verify per-group grouping
    by_group: dict[tuple, list] = {}
    for e in result:
        key = (e["image_id"], e["category_id"])
        by_group.setdefault(key, []).append(e)

    # image 1, class 1: 3 above threshold (0.9, 0.8, 0.5), top-2 → (0.9, 0.8)
    assert len(by_group[(1, 1)]) == 2
    assert [e["score"] for e in by_group[(1, 1)]] == [0.9, 0.8]

    # image 1, class 2: all 3 above threshold, top-2 → (0.95, 0.7)
    assert len(by_group[(1, 2)]) == 2
    assert [e["score"] for e in by_group[(1, 2)]] == [0.95, 0.7]

    # image 2, class 1: 1 above threshold (0.85), top-2 → (0.85,)
    assert len(by_group[(2, 1)]) == 1
    assert by_group[(2, 1)][0]["score"] == 0.85

    # Total: 2 + 2 + 1 = 5
    assert len(result) == 5


# ---------------------------------------------------------------------------
# 8. test_save_masks_none_omits_segmentation
# ---------------------------------------------------------------------------


def test_save_masks_none_omits_segmentation(tmp_path: Path) -> None:
    entries = [_make_entry(), _make_entry(image_id=2)]
    write_predictions(entries, tmp_path, save_masks="none", originals={1: (16, 16), 2: (16, 16)})
    data = json.loads((tmp_path / "predictions.json").read_text())
    for rec in data:
        assert "segmentation" not in rec, "segmentation must be absent with save_masks='none'"


# ---------------------------------------------------------------------------
# 9. test_save_masks_rle_default_keeps_segmentation
# ---------------------------------------------------------------------------


def test_save_masks_rle_default_keeps_segmentation(tmp_path: Path) -> None:
    entries = [_make_entry()]
    write_predictions(entries, tmp_path, save_masks="rle", originals={1: (16, 16)})
    data = json.loads((tmp_path / "predictions.json").read_text())
    assert "segmentation" in data[0]
    rle = data[0]["segmentation"]
    # Must be decodable
    decoded = mask_utils.decode({"size": rle["size"], "counts": rle["counts"].encode("ascii")})
    assert decoded.shape == (16, 16)


# ---------------------------------------------------------------------------
# 10. test_save_masks_png_writes_files_and_sets_mask_png
# ---------------------------------------------------------------------------


def test_save_masks_png_writes_files_and_sets_mask_png(tmp_path: Path) -> None:
    h, w = 32, 24
    entries = [_make_entry(image_id=1, category_id=1, h=h, w=w)]
    write_predictions(
        entries,
        tmp_path,
        save_masks="png",
        originals={1: (h, w)},
        id_to_stem={1: "myimage"},
    )
    data = json.loads((tmp_path / "predictions.json").read_text())
    rec = data[0]

    # segmentation field must be dropped
    assert "segmentation" not in rec, "segmentation must be absent with save_masks='png'"

    # mask_png must be present
    assert "mask_png" in rec

    # The relative path must point to an existing file under output_dir
    mask_rel = rec["mask_png"]
    mask_file = tmp_path / mask_rel
    assert mask_file.exists(), f"mask file not found: {mask_file}"

    # Must be inside masks/ subdirectory
    assert mask_rel.startswith("masks/"), f"unexpected path: {mask_rel}"

    # Filename pattern: <stem>_<cat_id>_<inst_idx>.png
    # stem=myimage, cat_id=1, inst_idx=0
    assert mask_rel == "masks/myimage_1_0.png"


# ---------------------------------------------------------------------------
# 11. test_png_mask_dims_match_original_hw
# ---------------------------------------------------------------------------


def test_png_mask_dims_match_original_hw(tmp_path: Path) -> None:
    """PNG dimensions must equal original-image H x W from originals dict."""
    h, w = 48, 36
    entries = [_make_entry(image_id=7, category_id=2, h=h, w=w)]
    write_predictions(
        entries,
        tmp_path,
        save_masks="png",
        originals={7: (h, w)},
        id_to_stem={7: "testimg"},
    )
    data = json.loads((tmp_path / "predictions.json").read_text())
    mask_rel = data[0]["mask_png"]
    mask_file = tmp_path / mask_rel
    img = Image.open(mask_file)
    # PIL .size is (width, height)
    assert img.size == (w, h), f"expected ({w}, {h}), got {img.size}"
