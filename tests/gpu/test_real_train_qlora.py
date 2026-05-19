"""50-step QLoRA overfit on tiny_coco via run_training(gpu_smoke_qlora.yaml).

Gated by `@pytest.mark.gpu`, `@requires_compatible_gpu`, `@requires_checkpoint`,
plus a per-test `skipif(not _bnb_available())`. Not in CI by default. Run with:
    pytest -m gpu tests/gpu/test_real_train_qlora.py -v

This test exercises the same `run_training(cfg)` seam that `esam3 train` uses,
proving 4-bit base + bf16 LoRA + 8-bit optimizer trains end-to-end on real
SAM 3.1. Loss-ratio and VRAM ceilings are looser than the LoRA smoke because
4-bit base converges slightly slower and pairs with adamw8bit on the 12 GB
recipe (architecture §6).
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from esam3.config.loader import load_config
from esam3.train.runner import run_training
from tests.gpu.conftest import _bnb_available, _RecordingTracker

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.requires_compatible_gpu,
    pytest.mark.requires_checkpoint,
]

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "examples" / "gpu_smoke_qlora.yaml"
LOSS_RATIO_CEIL = 0.75
VRAM_CEIL_GB = 10.0


@pytest.mark.requires_bnb
@pytest.mark.skipif(not _bnb_available(), reason="bitsandbytes not installed")
def test_qlora_overfits_in_50_steps(
    tmp_path: Path,
    tiny_coco_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    # Patch the consumer's namespace (esam3.train.runner) rather than the producer
    # (esam3.tracking). See spec §4.2.
    monkeypatch.setattr("esam3.train.runner.build_tracker", lambda *_a, **_kw: tracker)

    torch.cuda.reset_peak_memory_stats()
    run_training(cfg)
    peak_vram_gb = torch.cuda.max_memory_allocated() / 1e9

    losses = [s["loss/total"] for _, s in tracker.scalars if s["loss/total"] > 0]
    assert losses, "expected at least one logged loss scalar"
    assert all(math.isfinite(v) for _, s in tracker.scalars for v in s.values()), (
        "non-finite scalar logged during training"
    )
    assert losses[-1] <= LOSS_RATIO_CEIL * losses[0], (
        f"loss did not drop enough: start={losses[0]:.4f} end={losses[-1]:.4f}"
    )
    assert peak_vram_gb <= VRAM_CEIL_GB, (
        f"peak VRAM {peak_vram_gb:.2f}GB exceeded ceiling {VRAM_CEIL_GB}GB"
    )
