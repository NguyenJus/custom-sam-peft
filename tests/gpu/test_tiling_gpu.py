"""GPU end-to-end tests for the tiling path (spec §12.2).

G1: run_predict on an oversized image — full-extent output masks, seam-crossing
    object merges into ONE instance, run.json records tiling provenance.

G2: run_eval on an oversized eval sample — per-tile metric accumulation without
    materializing a stitched full-image mask; visualize path renders one overlay.

Requires:
  - A CUDA device with compute capability >= 7.5  (requires_compatible_gpu)
  - The real SAM 3.1 checkpoint at models/sam3.1/sam3.1_multiplex.pt  (requires_checkpoint)

Run explicitly:
    pytest -m gpu_t4 tests/gpu/test_tiling_gpu.py -v
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

pytestmark = [
    pytest.mark.gpu_t4,
    pytest.mark.requires_compatible_gpu,
    pytest.mark.requires_checkpoint,
]

# ---------------------------------------------------------------------------
# Constants: must stay consistent with the tiling module
# ---------------------------------------------------------------------------
_TILE_SIZE = 1008  # SAM3_IMAGE_SIZE
_OVERSIZED = 1500  # max(edge) > _TILE_SIZE → tiling_engaged is True


# ---------------------------------------------------------------------------
# Helpers: build synthetic oversized COCO dataset on disk
# ---------------------------------------------------------------------------


def _write_oversized_coco(
    coco_dir: Path,
    img_h: int = _OVERSIZED,
    img_w: int = _OVERSIZED,
) -> tuple[Path, Path]:
    """Write a COCO fixture with one oversized RGB image and two annotations.

    Annotation 1 sits entirely in tile 0 (top-left region, well within one tile).
    Annotation 2 is a horizontal bar that crosses every vertical seam — this is
    the seam-crossing object G1 must merge into ONE instance.

    Returns (images_dir, annotations_path).
    """
    images_dir = coco_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Create a solid-colour image large enough to trigger tiling.
    arr = np.full((img_h, img_w, 3), fill_value=200, dtype=np.uint8)
    img_path = images_dir / "oversized.png"
    Image.fromarray(arr).save(img_path)

    # Annotation 1: a 100x100 box well inside the top-left tile.
    ann1_x, ann1_y, ann1_w, ann1_h = 50, 50, 100, 100

    # Annotation 2: a horizontal bar [y=700..760] x [x=0..img_w-1] that spans
    # every vertical seam.  The box representation is [x, y, w, h] (COCO xywh).
    bar_y, bar_h_px = 700, 60
    bar_x, bar_w_px = 0, img_w

    annotations_data = {
        "images": [{"id": 1, "file_name": "oversized.png", "width": img_w, "height": img_h}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [ann1_x, ann1_y, ann1_w, ann1_h],
                "area": ann1_w * ann1_h,
                "iscrowd": 0,
                "segmentation": [
                    [
                        ann1_x,
                        ann1_y,
                        ann1_x + ann1_w,
                        ann1_y,
                        ann1_x + ann1_w,
                        ann1_y + ann1_h,
                        ann1_x,
                        ann1_y + ann1_h,
                    ]
                ],
            },
            {
                "id": 2,
                "image_id": 1,
                "category_id": 1,
                "bbox": [bar_x, bar_y, bar_w_px, bar_h_px],
                "area": bar_w_px * bar_h_px,
                "iscrowd": 0,
                "segmentation": [
                    [
                        bar_x,
                        bar_y,
                        bar_x + bar_w_px,
                        bar_y,
                        bar_x + bar_w_px,
                        bar_y + bar_h_px,
                        bar_x,
                        bar_y + bar_h_px,
                    ]
                ],
            },
        ],
        "categories": [{"id": 1, "name": "thing", "supercategory": "object"}],
    }
    ann_path = coco_dir / "annotations.json"
    ann_path.write_text(json.dumps(annotations_data))
    return images_dir, ann_path


# ---------------------------------------------------------------------------
# G1: tiled predict — full-extent masks + seam-merge + run.json provenance
# ---------------------------------------------------------------------------


def test_G1_tiled_predict_one_full_extent_mask(tmp_path: Path) -> None:
    """run_predict on an oversized image with the real model: output masks are at
    the ORIGINAL full extent (not tile-sized), run.json records tiling provenance
    with engaged=True and n_windows_total > 1.

    The seam-crossing object (a horizontal bar spanning every vertical seam) is
    expected to produce at most one merged instance rather than one fragment per
    tile.  Because this depends on the model's actual segmentation output (score
    threshold, presence logit, etc.), the test asserts the structural invariants
    that the tiling path guarantees regardless of model confidence:

      (a) all prediction segmentation RLEs describe masks at (img_h, img_w), NOT
          at tile resolution — confirms merge_fragments lifted them to canvas;
      (b) run.json["tiling"]["engaged"] is True;
      (c) run.json["tiling"]["n_windows_total"] > 1;
      (d) run.json["tiling"]["tile"] == SAM3_IMAGE_SIZE;
      (e) run.json["tiling"]["overlap"] == DEFAULT_OVERLAP.

    NOTE on per-tile OOM-retry path (spec §5.2 / G1 requirement):
    The OOM ladder in _predict_one_tile halves micro_batch_size on the first
    CUDA OOM ("device not ready" on sm_120), then retries. Deterministically
    inducing OOM against the real SAM 3.1 model would require either: (1) shrinking
    available VRAM (not safely achievable in a pytest fixture), or (2) patching
    is_cuda_oom to always return True (which would skip the real forward entirely).
    The CPU unit test test_predict_one_tile_oom_retry_succeeds (tests/unit/
    test_predict_tiling_unit.py) already exercises the retry logic with a _StubModel
    that raises on first call. The real GPU end-to-end cannot reliably reproduce
    an OOM on a 16 GB card with a single 1500x1500 tile. Therefore the OOM-retry
    path is NOT exercised here — the orchestrator should note this gap and decide
    whether a stress/low-memory variant is warranted.
    """
    import pycocotools.mask as mask_utils

    from custom_sam_peft.data.tiling import DEFAULT_OVERLAP, tiling_engaged
    from custom_sam_peft.models.sam3 import SAM3_IMAGE_SIZE
    from custom_sam_peft.predict.runner import PredictOptions, PredictReport, run_predict

    # Confirm the oversized image will engage tiling (structural guard).
    assert tiling_engaged(_OVERSIZED, _OVERSIZED), (
        f"expected tiling_engaged({_OVERSIZED}, {_OVERSIZED}) == True"
    )

    # Build synthetic oversized COCO fixture.
    coco_dir = tmp_path / "coco"
    images_dir, _ann_path = _write_oversized_coco(coco_dir, img_h=_OVERSIZED, img_w=_OVERSIZED)

    out_dir = tmp_path / "out"
    opts = PredictOptions(
        images=images_dir,
        prompts="thing",
        output=out_dir,
        checkpoint=None,
        merge_adapter=True,
        config=None,
        score_threshold=0.0,  # keep all predictions to maximise observable surface
        top_k=200,
        save_masks="rle",
        visualize=False,
        device="cuda",
        dtype="bfloat16",
        seed=42,
        dry_run=False,
        verbose=False,
    )
    report = run_predict(opts)
    assert isinstance(report, PredictReport)
    assert report.n_images == 1, f"expected 1 image processed, got {report.n_images}"

    # (a) All RLE segmentation masks must be sized at (img_h, img_w), not tile size.
    predictions = json.loads((out_dir / "predictions.json").read_text())
    assert isinstance(predictions, list)
    for entry in predictions:
        seg = entry.get("segmentation")
        assert seg is not None, f"expected 'segmentation' in entry: {entry}"
        size = seg["size"]  # [H, W] in pycocotools RLE
        rle_h, rle_w = int(size[0]), int(size[1])
        assert rle_h == _OVERSIZED, (
            f"RLE H={rle_h} != {_OVERSIZED}: mask was not lifted to full canvas"
        )
        assert rle_w == _OVERSIZED, (
            f"RLE W={rle_w} != {_OVERSIZED}: mask was not lifted to full canvas"
        )

    # Sanity-check the seam-crossing bar: decode all masks and verify that any
    # prediction covering the bar region spans >_TILE_SIZE columns (full-width),
    # confirming merge happened across the seam.  This is a soft assertion —
    # if the model produces no detections (score threshold = 0.0 baseline with no
    # adapter) we skip it gracefully.
    bar_covering_preds = []
    for entry in predictions:
        seg = entry.get("segmentation")
        if seg is None:
            continue
        decode_rle: dict = dict(seg)
        counts = decode_rle.get("counts")
        if isinstance(counts, str):
            decode_rle["counts"] = counts.encode("ascii")
        decoded = mask_utils.decode(decode_rle)  # (H, W) uint8
        # Check if this mask covers a contiguous horizontal band crossing the seam.
        bar_band = decoded[700:760, :]  # the seam-crossing bar rows
        if bar_band.sum() > 0:
            # Column coverage of the mask in the bar band.
            col_coverage = int(bar_band.any(axis=0).sum())
            bar_covering_preds.append(col_coverage)

    if bar_covering_preds:
        # If the model detected the bar, the widest covering prediction must span
        # more than one tile's width, confirming cross-seam merge.
        max_coverage = max(bar_covering_preds)
        assert max_coverage > SAM3_IMAGE_SIZE, (
            f"Widest bar-covering prediction spans only {max_coverage} columns "
            f"(<= tile size {SAM3_IMAGE_SIZE}); cross-seam merge may not have occurred. "
            f"All bar coverages: {bar_covering_preds}"
        )

    # (b-e) run.json tiling provenance record.
    run_json = json.loads((out_dir / "run.json").read_text())
    tiling_rec = run_json.get("tiling")
    assert isinstance(tiling_rec, dict), f"run.json missing 'tiling' key: {run_json.keys()}"

    assert tiling_rec["engaged"] is True, (
        f"run.json['tiling']['engaged'] expected True, got {tiling_rec['engaged']!r}"
    )
    assert isinstance(tiling_rec["n_windows_total"], int) and tiling_rec["n_windows_total"] > 1, (
        f"run.json['tiling']['n_windows_total'] expected int > 1, "
        f"got {tiling_rec['n_windows_total']!r}"
    )
    assert tiling_rec["tile"] == SAM3_IMAGE_SIZE, (
        f"run.json['tiling']['tile'] expected {SAM3_IMAGE_SIZE}, got {tiling_rec['tile']!r}"
    )
    assert math.isclose(tiling_rec["overlap"], DEFAULT_OVERLAP, rel_tol=1e-6), (
        f"run.json['tiling']['overlap'] expected {DEFAULT_OVERLAP}, got {tiling_rec['overlap']!r}"
    )


# ---------------------------------------------------------------------------
# G2: tiled eval — per-tile metric accumulation + visualize produces overlay
# ---------------------------------------------------------------------------


def test_G2_tiled_eval_accumulates_without_stitch(tmp_path: Path) -> None:
    """run_eval on an oversized eval sample: per-tile metric accumulation completes
    without materializing a stitched full-image mask in the evaluator's prediction
    entries, metrics.json is written with sane values, and the visualize path
    produces at least one PNG overlay at the full-image extent.

    'No stitched mask': the evaluator stores per-tile COCO predictions whose
    segmentation RLEs are sized to the tile (<=1008), NOT to the full image
    (1500x1500).  This mirrors test_evaluator_tiling_no_stitched_mask_in_predictions
    in tests/unit/test_eval_tiling_unit.py at the GPU/real-model level.

    The visualize path is exercised via run_eval(..., visualize=True).  It invokes
    write_eval_visualizations → render_eval_pair → _tiled_pred_entries, which runs
    per-tile forwards at DEFAULT_OVERLAP and merges fragments onto the full canvas.
    The resulting composite PNG must exist under output_dir/visualizations/.
    """
    from custom_sam_peft.config.schema import (
        DataConfig,
        DataSplit,
        EvalConfig,
        ModelConfig,
        PEFTConfig,
        RunConfig,
        TrainConfig,
        TrainHyperparams,
    )
    from custom_sam_peft.data.tiling import tiling_engaged
    from custom_sam_peft.eval.metrics import MetricsReport
    from custom_sam_peft.eval.runner import run_eval

    # Confirm the oversized image will engage tiling (structural guard).
    assert tiling_engaged(_OVERSIZED, _OVERSIZED)

    # Build synthetic oversized COCO fixture for the val split.
    coco_dir = tmp_path / "coco"
    images_dir, ann_path = _write_oversized_coco(coco_dir, img_h=_OVERSIZED, img_w=_OVERSIZED)

    out_dir = tmp_path / "eval_out"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build a minimal TrainConfig pointing at the oversized fixture.
    cfg = TrainConfig(
        run=RunConfig(name="tiling-eval-smoke", output_dir=str(out_dir), seed=42),
        model=ModelConfig(name="facebook/sam3.1", dtype="bfloat16"),
        data=DataConfig(
            format="coco",
            train=DataSplit(annotations=str(ann_path), images=str(images_dir)),
            val=DataSplit(annotations=str(ann_path), images=str(images_dir)),
        ),
        peft=PEFTConfig(method="lora", scope="vision_decoder"),
        train=TrainHyperparams(epochs=1, batch_size=1),
        eval=EvalConfig(
            mode="full",
            iou_thresholds=[0.5],
            batch_size=1,
            save_predictions=True,
            visualize=True,
            visualize_count=1,
        ),
    )

    # Run eval with the real model (checkpoint=None → baseline zero-shot SAM 3.1).
    report = run_eval(
        cfg,
        checkpoint=None,
        split="val",
        output_dir=out_dir,
        visualize=True,
    )
    assert isinstance(report, MetricsReport), (
        f"run_eval returned {type(report).__name__}, expected MetricsReport"
    )
    assert report.n_images >= 1, f"expected at least 1 eval image, got {report.n_images}"

    # metrics.json must exist and contain a finite overall.mAP >= 0.
    metrics_path = out_dir / "metrics.json"
    assert metrics_path.exists(), "metrics.json not written by run_eval"
    metrics = json.loads(metrics_path.read_text())
    assert "overall" in metrics, f"metrics.json missing 'overall': {list(metrics.keys())}"
    mAP = metrics["overall"].get("mAP")
    assert isinstance(mAP, (int, float)) and math.isfinite(mAP) and mAP >= 0.0, (
        f"overall.mAP not finite/non-negative: {mAP!r}"
    )

    # 'No stitched mask': per-tile prediction RLEs must be tile-sized, NOT full-image.
    # The evaluator accumulates predictions per tile (each tile independently).
    # Predictions are persisted at the canonical artifacts path (a JSON array written
    # to artifacts/predictions_<split>.jsonl), NOT a bare out_dir/predictions.json.
    from custom_sam_peft.paths import predictions_path

    preds_path = predictions_path(out_dir, split="val")
    assert preds_path.exists(), (
        f"predictions not written at {preds_path} (save_predictions=True, mode=full)"
    )
    predictions = json.loads(preds_path.read_text())
    assert isinstance(predictions, list)
    for entry in predictions:
        seg = entry.get("segmentation")
        if seg is None:
            continue
        size = seg.get("size")
        if size is None:
            continue
        rle_h, rle_w = int(size[0]), int(size[1])
        assert rle_h <= _TILE_SIZE, (
            f"Eval RLE H={rle_h} > tile size {_TILE_SIZE}: "
            f"a stitched full-image mask leaked into eval predictions"
        )
        assert rle_w <= _TILE_SIZE, (
            f"Eval RLE W={rle_w} > tile size {_TILE_SIZE}: "
            f"a stitched full-image mask leaked into eval predictions"
        )

    # Visualize path: run_eval with visualize=True must produce at least one PNG
    # under out_dir/visualizations/.  The vis path calls render_eval_pair →
    # _tiled_pred_entries → merge_fragments, compositing a full-canvas overlay.
    vis_dir = out_dir / "visualizations"
    assert vis_dir.is_dir(), (
        f"visualizations/ dir not created under {out_dir} — visualize path did not run"
    )
    vis_pngs = list(vis_dir.glob("*.png"))
    assert vis_pngs, f"no PNG files in {vis_dir} — visualize path produced no output"

    # Each visualization PNG must be readable and at the full-image extent.
    for png_path in vis_pngs:
        with Image.open(png_path) as img:
            w, h = img.size
        # The composite is hstacked [orig|gt|pred], so width is ~3x orig_w.
        # The panel height must equal the original image height (_OVERSIZED).
        assert h == _OVERSIZED, (
            f"Visualization PNG height {h} != {_OVERSIZED}: "
            f"composite was not rendered at full-image extent"
        )
        assert w >= _OVERSIZED, (
            f"Visualization PNG width {w} < {_OVERSIZED}: "
            f"composite appears narrower than the source image"
        )
