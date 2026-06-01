"""All-scope LoRA smoke: verify peft.scope=all fits the gpu_t4 VRAM band.

Issue #83 asks whether all-scope LoRA (LoRA adapters on every nn.Linear,
including the frozen ViT trunk) fits the gpu_t4 capability band
(CC >= 7.5, total VRAM <= 16 GB: Tesla T4 / RTX 5070 Ti).

Measured 2026-05-31 on RTX 5070 Ti (CC 12.0, 16 GiB):
    peak VRAM = 3.926 GB (scope=all, 15.4 M trainable params / 1.80% of 856 M,
                          293 target modules, 2-image tiny_coco, epochs=2)
    → well within the 15.0 GB ceiling; #83 closed DONE.

Gated by `@pytest.mark.gpu_t4`, `@requires_compatible_gpu`, and
`@requires_checkpoint`. Not in CI by default. Run with:
    pytest -m gpu_t4 tests/gpu/test_peft_scope_coverage_gpu.py -v
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from custom_sam_peft.config.loader import load_config
from custom_sam_peft.train.runner import run_training
from tests.gpu.conftest import _RecordingTracker

pytestmark = [
    pytest.mark.gpu_t4,
    pytest.mark.requires_compatible_gpu,
    pytest.mark.requires_checkpoint,
]

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "examples" / "gpu_smoke_lora.yaml"
# 15.0 GB ceiling: all-scope LoRA must fit the gpu_t4 band with safe margin.
# Measured peak: 3.926 GB on RTX 5070 Ti (2026-05-31). See module docstring.
VRAM_CEIL_GB = 15.0


def test_all_scope_lora_fits_gpu_t4_band(
    tmp_path: Path,
    tiny_coco_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """All-scope LoRA smoke: run a 2-epoch overfit and assert VRAM + finite loss."""
    cfg = load_config(
        CONFIG_PATH,
        overrides=[
            f"data.train.annotations={tiny_coco_dir / 'annotations.json'}",
            f"data.train.images={tiny_coco_dir / 'images'}",
            f"data.val.annotations={tiny_coco_dir / 'annotations.json'}",
            f"data.val.images={tiny_coco_dir / 'images'}",
            f"run.output_dir={tmp_path}",
            # Override to all-scope LoRA: adapters on every nn.Linear in the model.
            "peft.scope=all",
            # Short budget: peak VRAM is set by the first forward+backward, so 2
            # epochs over the 2-image tiny_coco set (4 grad steps) suffices.
            "train.epochs=2",
            "train.log_every=1",
            # Disable early-stop: the 2-image overfit keeps eval mAP ~0.0, so the
            # loop would halt before completing the 2-epoch budget. Same pattern as
            # test_real_train_qlora_resume.py.
            "train.early_stop.enabled=false",
        ],
    )

    tracker = _RecordingTracker()
    monkeypatch.setattr("custom_sam_peft.train.runner.build_tracker", lambda *_a, **_kw: tracker)

    torch.cuda.reset_peak_memory_stats()
    run_training(cfg)
    peak_vram_gb = torch.cuda.max_memory_allocated() / 1e9

    # At least one finite logged loss (use .get to avoid KeyError on eval-only dicts).
    losses = [s.get("loss/total", 0) for _, s in tracker.scalars if s.get("loss/total", 0) > 0]
    assert losses, "expected at least one logged loss scalar with loss/total > 0"
    assert all(math.isfinite(v) for _, s in tracker.scalars for v in s.values()), (
        "non-finite scalar logged during all-scope LoRA training"
    )

    assert peak_vram_gb <= VRAM_CEIL_GB, (
        f"all-scope LoRA peak VRAM {peak_vram_gb:.2f} GB exceeded gpu_t4 ceiling "
        f"{VRAM_CEIL_GB} GB — does not belong in the gpu_t4 band"
    )
