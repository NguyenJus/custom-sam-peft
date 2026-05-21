"""50-step QLoRA overfit on tiny_coco via run_training(gpu_smoke_qlora.yaml).

Gated by `@pytest.mark.gpu`, `@requires_compatible_gpu`, `@requires_checkpoint`,
plus a per-test `skipif(not _bnb_available())`. Not in CI by default. Run with:
    pytest -m gpu tests/gpu/test_real_train_qlora.py -v

This test exercises the same `run_training(cfg)` seam that `custom_sam_peft train` uses,
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

from custom_sam_peft.config.loader import load_config
from custom_sam_peft.train.runner import run_training
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
    # Patch the consumer's namespace (custom_sam_peft.train.runner) rather than the producer
    # (custom_sam_peft.tracking). See spec §4.2.
    monkeypatch.setattr("custom_sam_peft.train.runner.build_tracker", lambda *_a, **_kw: tracker)

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


@pytest.mark.requires_bnb
@pytest.mark.skipif(not _bnb_available(), reason="bitsandbytes not installed")
def test_qlora_smoke_fast(
    tmp_path: Path,
    tiny_coco_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fast iteration variant of the QLoRA smoke (~3 min on T4 vs. ~14 min).

    Same ``run_training(cfg)`` path as ``test_qlora_overfits_in_50_steps`` —
    so any dtype, shape, grad-routing, or quantization bug that breaks the
    full overfit also surfaces here — but with two debug-cycle shortcuts:

      - ``train.epochs=2`` override (≈3 training steps on 2 images) instead
        of the 50-step overfit target.
      - Final ``Evaluator.evaluate`` pass monkeypatched to a no-op.

    Assertions are correspondingly looser: every logged scalar must be
    finite and at least one must be logged.  No loss-ratio assertion (3
    steps gives the optimizer no time to overfit) and no VRAM-ceiling
    (smoke is too short to peak activations or quantization buffers).

    Use this for the dtype/grad debug cycle.  Promote to
    ``test_qlora_overfits_in_50_steps`` for full release-tier validation
    once this is green.
    """
    from custom_sam_peft.eval.metrics import MetricsReport

    cfg = load_config(
        CONFIG_PATH,
        overrides=[
            f"data.train.annotations={tiny_coco_dir / 'annotations.json'}",
            f"data.train.images={tiny_coco_dir / 'images'}",
            f"data.val.annotations={tiny_coco_dir / 'annotations.json'}",
            f"data.val.images={tiny_coco_dir / 'images'}",
            f"run.output_dir={tmp_path}",
            "train.epochs=2",
            # gpu_smoke_qlora.yaml ships log_every=10; with only ~3 training
            # steps in the fast smoke, the modulo never hits a boundary and
            # tracker.scalars stays empty. Force log_every=1 so we get a
            # scalar per step and the finite-value assertion has something
            # to assert on.
            "train.log_every=1",
        ],
    )
    tracker = _RecordingTracker()
    monkeypatch.setattr("custom_sam_peft.train.runner.build_tracker", lambda *_a, **_kw: tracker)

    class _SkipEvaluator:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def evaluate(self, model: object, ds: object) -> MetricsReport:
            return MetricsReport()

    monkeypatch.setattr("custom_sam_peft.train.trainer.Evaluator", _SkipEvaluator)

    run_training(cfg)

    assert tracker.scalars, "expected at least one scalar log"
    for _, scalars in tracker.scalars:
        for k, v in scalars.items():
            assert math.isfinite(v), f"non-finite scalar logged: {k}={v}"
