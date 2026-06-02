"""50-step LoRA overfit on tiny_coco via run_training(gpu_smoke_lora.yaml).

Gated by `@pytest.mark.gpu_t4`, `@requires_compatible_gpu`, and
`@requires_checkpoint`. Not in CI by default. Run with:
    pytest -m gpu_t4 tests/gpu/test_real_train_overfits.py -v

This test exercises the same `run_training(cfg)` seam that `custom_sam_peft train` uses,
so the YAML at configs/examples/gpu_smoke_lora.yaml is both the user-facing
example and the test's source of truth (modulo the monkeypatched tracker).
"""

from __future__ import annotations

import json
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
# 50-step LoRA overfit ceiling, capability-aware. Training IS a guaranteed T4
# capability, but fp16 (the coerced dtype below CC 8.0) converges less in 50
# steps than faithful bf16, so the ceiling is split by band:
#
# - bf16 band (CC >= 8.0): confirmed on RTX 5070 Ti (sm_120) 2026-05-31 — loss
#   0.5222 -> 0.3081 (ratio 0.590 <= 0.70), peak 4.49 GB, ~55 s. No retune (#195).
# - fp16 band (CC < 8.0, bf16 coerced): one real Colab Tesla T4 (CC 7.5) sample
#   2026-06-01 (#212) — loss 0.5704 -> 0.4256 (ratio 0.746). The 0.80 ceiling
#   still requires a meaningful >=20% drop while clearing observed fp16 noise.
#   tbd: pin with a second confirming T4 sample on current main (#212 follow-up).
LOSS_RATIO_CEIL_BF16 = 0.70
LOSS_RATIO_CEIL_FP16 = 0.80
VRAM_CEIL_GB = 14.0


def test_overfits_in_50_steps(
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
    # Patch the consumer's namespace (custom_sam_peft.train.runner) rather than the
    # producer (custom_sam_peft.tracking) — runner.py does
    # `from custom_sam_peft.tracking import build_tracker` at import time, so the
    # bound name lives in runner.__dict__. See spec §4.2.
    monkeypatch.setattr("custom_sam_peft.train.runner.build_tracker", lambda *_a, **_kw: tracker)

    torch.cuda.reset_peak_memory_stats()
    run_training(cfg)
    peak_vram_gb = torch.cuda.max_memory_allocated() / 1e9

    losses = [s["loss/total"] for _, s in tracker.scalars if s.get("loss/total", 0) > 0]
    assert losses, "expected at least one logged loss scalar"
    assert all(math.isfinite(v) for _, s in tracker.scalars for v in s.values()), (
        "non-finite scalar logged during training"
    )
    # Pick the ceiling for the running card's effective dtype band: fp16 below
    # CC 8.0 (bf16 is coerced there), faithful bf16 at/above it.
    cc = torch.cuda.get_device_capability(0)
    loss_ratio_ceil = LOSS_RATIO_CEIL_FP16 if cc < (8, 0) else LOSS_RATIO_CEIL_BF16
    assert losses[-1] <= loss_ratio_ceil * losses[0], (
        f"loss did not drop enough (cc={cc[0]}.{cc[1]}, ceil={loss_ratio_ceil}): "
        f"start={losses[0]:.4f} end={losses[-1]:.4f}"
    )
    assert peak_vram_gb <= VRAM_CEIL_GB, (
        f"peak VRAM {peak_vram_gb:.2f}GB exceeded ceiling {VRAM_CEIL_GB}GB"
    )

    # T2 (spec §6.1): assert the Evaluator's metrics.json overall.mAP is finite.
    runs = sorted(tmp_path.glob("gpu-smoke-lora-*"))
    assert runs, f"no run dir under {tmp_path}"
    metrics = json.loads((runs[-1] / "metrics.json").read_text())
    assert "overall" in metrics, f"metrics.json missing 'overall': {metrics}"
    mAP = metrics["overall"].get("mAP")
    assert isinstance(mAP, (int, float)) and math.isfinite(mAP) and mAP >= 0.0, (
        f"overall.mAP not finite/non-negative: {mAP}"
    )
