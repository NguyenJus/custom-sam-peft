"""#142 8 GB-ceiling proof: QLoRA train smoke against min_gpu_qlora.yaml.

Validates that the minimum-supported-card QLoRA configuration
(configs/examples/min_gpu_qlora.yaml — decoder-only narrow-scope QLoRA,
fp16, batch_size=1) fits within an 8 GB VRAM envelope, faithfully modelling
a CC 7.5 / 8 GB card (e.g. GTX 1080, T4).

The test runs the full run_training() path (50 optimizer steps on the
2-image tiny_coco fixture) and asserts:
  1. Peak VRAM <= QLORA_8GB_CEIL_GB (8.0 GB)
  2. Loss drops >= 25% from first to last logged value (overfit signal)
  3. metrics.json overall.mAP is finite and non-negative

Gated by gpu_t4 / requires_compatible_gpu / requires_checkpoint — skipped
on CPU. Run with:
    pytest -m gpu_t4 tests/gpu/test_qlora_8gb_ceiling.py -o "addopts=" -v
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from custom_sam_peft.config.loader import load_config
from custom_sam_peft.train.runner import run_training
from tests.gpu.conftest import _bnb_available, _RecordingTracker

# cite: measured ~5.0 GB peak (GTX 1080, fp16) in
# docs/research/2026-05-24-issue-137-qlora-8gb-feasibility.md. 8.0 GB = target
# minimum-card envelope with ~3 GB margin over the measured peak. Date 2026-05-31.
# 5070 Ti measured peak: 2.348 GB (fp16, min_gpu_qlora, 2026-05-31) — within the 8.0 envelope.
# tbd: #142 — confirm on a real 8 GB card.
QLORA_8GB_CEIL_GB: float = 8.0

pytestmark = [
    pytest.mark.gpu_t4,
    pytest.mark.requires_compatible_gpu,
    pytest.mark.requires_checkpoint,
]

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "examples" / "min_gpu_qlora.yaml"
LOSS_RATIO_CEIL = 0.75


@pytest.mark.requires_bnb
@pytest.mark.skipif(not _bnb_available(), reason="bitsandbytes not installed")
def test_qlora_8gb_ceiling(
    tmp_path: Path,
    tiny_coco_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """50-step QLoRA overfit on tiny_coco via min_gpu_qlora.yaml; asserts 8 GB VRAM ceiling."""
    cfg = load_config(
        CONFIG_PATH,
        overrides=[
            f"data.train.annotations={tiny_coco_dir / 'annotations.json'}",
            f"data.train.images={tiny_coco_dir / 'images'}",
            f"data.val.annotations={tiny_coco_dir / 'annotations.json'}",
            f"data.val.images={tiny_coco_dir / 'images'}",
            f"run.output_dir={tmp_path}",
        ],
    )
    tracker = _RecordingTracker()
    # Patch the consumer's namespace (custom_sam_peft.train.runner) rather than the producer
    # (custom_sam_peft.tracking). See spec §4.2.
    monkeypatch.setattr("custom_sam_peft.train.runner.build_tracker", lambda *_a, **_kw: tracker)

    torch.cuda.reset_peak_memory_stats()
    run_training(cfg)
    peak_vram_gb = torch.cuda.max_memory_allocated() / 1e9

    losses = [s["loss/total"] for _, s in tracker.scalars if s.get("loss/total", 0) > 0]
    assert losses, "expected at least one logged loss scalar"
    assert all(math.isfinite(v) for _, s in tracker.scalars for v in s.values()), (
        "non-finite scalar logged during training"
    )
    assert losses[-1] <= LOSS_RATIO_CEIL * losses[0], (
        f"loss did not drop enough: start={losses[0]:.4f} end={losses[-1]:.4f}"
    )
    assert peak_vram_gb <= QLORA_8GB_CEIL_GB, (
        f"peak VRAM {peak_vram_gb:.2f} GB exceeded 8 GB ceiling {QLORA_8GB_CEIL_GB} GB"
    )

    # Assert the Evaluator's metrics.json overall.mAP is finite.
    runs = sorted(tmp_path.glob("min_gpu_qlora-*"))
    assert runs, f"no run dir under {tmp_path}"
    metrics = json.loads((runs[-1] / "metrics.json").read_text())
    assert "overall" in metrics, f"metrics.json missing 'overall': {metrics}"
    mAP = metrics["overall"].get("mAP")
    assert isinstance(mAP, (int, float)) and math.isfinite(mAP) and mAP >= 0.0, (
        f"overall.mAP not finite/non-negative: {mAP}"
    )
