"""Output writers for csp predict: predictions, id-map, run metadata, and masks.

Writes predictions.json, image_id_map.json, run.json, and optional mask PNGs.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pycocotools.mask as mask_utils
from PIL import Image

import custom_sam_peft


def encode_rle_dict(mask: np.ndarray[Any, np.dtype[np.uint8]]) -> dict[str, Any]:
    """Encode a (H, W) uint8 mask to a pycocotools RLE dict with ASCII counts."""
    rle: dict[str, Any] = mask_utils.encode(np.asfortranarray(mask))
    counts = rle["counts"]
    rle["counts"] = counts.decode("ascii") if isinstance(counts, bytes) else counts
    return rle


def decode_rle_to_uint8(rle: dict[str, Any]) -> np.ndarray[Any, np.dtype[np.uint8]]:
    """Decode a pycocotools RLE dict (ASCII or bytes counts) to a (H, W) uint8 mask."""
    # pycocotools expects bytes for the counts field
    decode_rle: dict[str, Any] = dict(rle)
    counts = decode_rle["counts"]
    if isinstance(counts, str):
        decode_rle["counts"] = counts.encode("ascii")
    result: np.ndarray[Any, np.dtype[np.uint8]] = mask_utils.decode(decode_rle)
    return result


def select_top_k_per_image_class(
    entries: list[dict[str, Any]],
    *,
    score_threshold: float,
    top_k: int,
) -> list[dict[str, Any]]:
    """Filter by score >= score_threshold, then keep top-K per (image_id, category_id) group."""
    # Step 1: filter by threshold
    filtered = [e for e in entries if e["score"] >= score_threshold]

    # Step 2: group by (image_id, category_id)
    groups: dict[tuple[Any, Any], list[dict[str, Any]]] = {}
    for entry in filtered:
        key = (entry["image_id"], entry["category_id"])
        groups.setdefault(key, []).append(entry)

    # Step 3: sort each group by score descending, take top-K
    result: list[dict[str, Any]] = []
    for group_entries in groups.values():
        group_entries.sort(key=lambda e: e["score"], reverse=True)
        result.extend(group_entries[:top_k])

    return result


def write_predictions(
    entries: list[dict[str, Any]],
    output_dir: Path,
    *,
    save_masks: Literal["rle", "png", "none"],
    originals: dict[int, tuple[int, int]],
    id_to_stem: dict[int, str] | None = None,
) -> None:
    """Write predictions.json to output_dir; optionally emit mask PNGs.

    ``id_to_stem`` is required (not None) when ``save_masks="png"``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if save_masks == "png" and id_to_stem is None:
        raise ValueError("id_to_stem is required when save_masks='png'")

    # Work on copies so we don't mutate the caller's dicts
    out_entries: list[dict[str, Any]] = []

    if save_masks == "png":
        masks_dir = output_dir / "masks"
        masks_dir.mkdir(parents=True, exist_ok=True)

        # Group by (image_id, category_id) to assign per-group instance indices.
        # We iterate in original list order and track index per (image_id, cat_id).
        inst_counter: dict[tuple[int, int], int] = {}

        for entry in entries:
            image_id = int(entry["image_id"])
            cat_id = int(entry["category_id"])
            key = (image_id, cat_id)

            inst_idx = inst_counter.get(key, 0)
            inst_counter[key] = inst_idx + 1

            if id_to_stem is None:  # pragma: no cover — guarded by ValueError above
                raise ValueError("id_to_stem is required when save_masks='png'")
            stem = id_to_stem[image_id]
            fname = f"{stem}_{cat_id}_{inst_idx}.png"
            rel_path = f"masks/{fname}"
            mask_file = masks_dir / fname

            # Decode RLE → uint8 mask
            mask_arr = decode_rle_to_uint8(entry["segmentation"])

            # Resize to original image dimensions if needed
            h, w = originals[image_id]
            if mask_arr.shape != (h, w):
                # Nearest-neighbor resize via PIL for binary masks
                pil_mask = Image.fromarray(mask_arr * 255)
                pil_mask = pil_mask.resize((w, h), Image.Resampling.NEAREST)
                mask_arr = (np.array(pil_mask) > 127).astype(np.uint8)

            Image.fromarray(mask_arr * 255).save(mask_file)

            rec: dict[str, Any] = {k: v for k, v in entry.items() if k != "segmentation"}
            rec["mask_png"] = rel_path
            out_entries.append(rec)

    elif save_masks == "none":
        for entry in entries:
            rec = {k: v for k, v in entry.items() if k != "segmentation"}
            out_entries.append(rec)

    else:  # "rle" — keep segmentation as-is
        out_entries = [dict(entry) for entry in entries]

    (output_dir / "predictions.json").write_text(json.dumps(out_entries))


def write_image_id_map(id_to_path: dict[int, Path], output_dir: Path) -> None:
    """Write image_id_map.json with stringified int keys and absolute path values."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mapping = {str(k): str(v) for k, v in id_to_path.items()}
    (output_dir / "image_id_map.json").write_text(json.dumps(mapping))


def write_run_json(run_meta: dict[str, Any], output_dir: Path) -> None:
    """Write run.json with all fields from spec §7.3, adding version and git_sha."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve package root for git query
    package_root = Path(custom_sam_peft.__file__).parent

    git_result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],  # noqa: S607
        cwd=package_root,
        check=False,
        capture_output=True,
        text=True,
    )
    git_sha: str | None = git_result.stdout.strip() if git_result.returncode == 0 else None

    record = dict(run_meta)
    record["version"] = custom_sam_peft.__version__
    record["git_sha"] = git_sha

    (output_dir / "run.json").write_text(json.dumps(record))
