"""GPU smoke: SemanticEvaluator.evaluate on the real SAM3.1 forward (mask_png).

Gated by ``@pytest.mark.gpu_bf16``, ``@requires_compatible_gpu``, and
``@requires_checkpoint``.  Not in CI by default.  Run with::

    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
      pytest -v --tb=short --no-cov \\
        tests/gpu/test_semantic_eval_gpu.py

Exercises the full semantic eval path on the real model (spec §11): the
multiplex forward → ``marginalize_group`` → ``build_semantic_logits`` →
bilinear upsample → ``semantic_argmax`` → streaming ``(K+1,K+1)`` confusion →
``compute_semantic_metrics``.  A tiny 2-image, 3-class mask_png fixture keeps
it within the real-model freeze threshold (one ``evaluate`` call).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import custom_sam_peft.data.mask_png  # noqa: F401 — triggers @register("dataset","mask_png") side-effect
from custom_sam_peft._registry import lookup
from custom_sam_peft.config.schema import EvalConfig, ModelConfig
from custom_sam_peft.eval.metrics import MetricsReport
from custom_sam_peft.eval.semantic_evaluator import SemanticEvaluator
from custom_sam_peft.models.sam3 import load_sam31

pytestmark = [
    pytest.mark.gpu_bf16,
    pytest.mark.requires_compatible_gpu,
    pytest.mark.requires_checkpoint,
]


def _make_mask_png_tree(tmp_path: Path, k: int) -> tuple[Path, Path, Path]:
    """Build a minimal mask_png dataset tree with K concept classes.

    Mirrors tests/gpu/test_semantic_train_step_gpu.py: B=2 RGB images, paired
    label PNGs with pixel values 0..K (0=background, 1..K=concepts) plus one 255
    void pixel at (0,0), and a class_map.json {"0":"background","1":"c1",...}.
    """
    img_dir = tmp_path / "img"
    lbl_dir = tmp_path / "lbl"
    img_dir.mkdir()
    lbl_dir.mkdir()

    h = w = 64
    for stem in ("a", "b"):
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        Image.fromarray(rgb).save(img_dir / f"{stem}.png")

        lbl = np.zeros((h, w), dtype=np.uint8)
        band = max(1, h // (k + 1))
        for c in range(1, k + 1):
            row_start = (c - 1) * band
            row_end = min(c * band, h)
            lbl[row_start:row_end, :] = c
        lbl[0, 0] = 255  # void pixel
        Image.fromarray(lbl, mode="L").save(lbl_dir / f"{stem}.png")

    class_map: dict[str, str] = {"0": "background"}
    for c in range(1, k + 1):
        class_map[str(c)] = f"c{c}"
    cm_path = tmp_path / "class_map.json"
    cm_path.write_text(json.dumps(class_map))

    return img_dir, lbl_dir, cm_path


def test_semantic_evaluate_real_model(tmp_path: Path) -> None:
    """SemanticEvaluator.evaluate on the real SAM3.1 forward — finite mIoU in [0,1].

    Assertions (spec §11):
    - returns a ``MetricsReport``
    - ``overall`` carries ``mIoU`` and ``pixel_acc``, both finite and in [0,1]
    - ``per_class`` is populated (keyed by class name, each {"IoU": ..})
    - ``return_per_example_iou=True`` yields one IoU per image
    """
    k = 3
    img_dir, lbl_dir, cm_path = _make_mask_png_tree(tmp_path, k)

    cfg_dict = {
        "format": "mask_png",
        "train": {"images": str(img_dir), "annotations": str(lbl_dir)},
        "val": None,
        "semantic": {
            "class_map": str(cm_path),
            "ignore_index": 255,
            "label_suffix": ".png",
        },
        "channels": 3,
        "text_prompt": {"mode": "all"},
    }
    builder = lookup("dataset", "mask_png")
    ds = builder(cfg_dict, model_name="facebook/sam3.1", pipeline="eval")

    model = load_sam31(
        ModelConfig(
            name="facebook/sam3.1",
            local_dir="models/sam3.1",
            checkpoint_file="sam3.1_multiplex.pt",
            dtype="bfloat16",
            device="cuda",
        ),
        channels=3,
        channel_semantics="rgb",
    )
    model.eval()

    evaluator = SemanticEvaluator(EvalConfig())
    report, per_example, gt_counts = evaluator.evaluate(model, ds, return_per_example_iou=True)
    assert gt_counts is None  # SemanticEvaluator has no instance concept

    assert isinstance(report, MetricsReport)
    assert "mIoU" in report.overall and "pixel_acc" in report.overall
    miou = report.overall["mIoU"]
    pixel_acc = report.overall["pixel_acc"]
    assert 0.0 <= miou <= 1.0, f"mIoU out of range: {miou}"
    assert 0.0 <= pixel_acc <= 1.0, f"pixel_acc out of range: {pixel_acc}"
    assert report.per_class, "per_class must be populated"
    assert all("IoU" in row for row in report.per_class.values())
    assert len(per_example) == len(ds)
