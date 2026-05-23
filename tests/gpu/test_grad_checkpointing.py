"""GPU verification for gradient checkpointing on T4 (#89).

Release-tier (-m gpu); collect-and-skip on CPU. Run on Colab T4:
    pytest -m gpu tests/gpu/test_grad_checkpointing.py -v

Verifies (spec §4 acceptance): no CheckpointError on LoRA AND QLoRA smokes
(forward+backward complete); first-step loss parity vs checkpointing-off (recompute
is numerically exact); peak VRAM measurably LOWER with checkpointing ON than OFF.
The absolute 14/10 GB ceilings live in the existing smoke tests and are unchanged.

Note: OOM-ladder GPU assertion (verifying _train_step_with_oom_ladder enables
use_act_checkpoint=True on real ViT-Det blocks) is deferred to the manual T4 run.
The CPU-level unit test (Task 4) covers the apply-call logic; a GPU integration
test would require heavy scaffolding (full model init + fake OOM injection) and
offers low additional signal over the CPU test. See spec §4 for rationale.
"""

from __future__ import annotations

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

_CFG_DIR = Path(__file__).resolve().parents[2] / "configs" / "examples"
_LORA = _CFG_DIR / "gpu_smoke_lora.yaml"
_QLORA = _CFG_DIR / "gpu_smoke_qlora.yaml"

# Checkpointing-on peak VRAM must be lower than checkpointing-off by at least
# this margin.  0.3 GB is conservative for a ViT-L backbone on T4.
VRAM_MARGIN_GB = 0.3


def _run(
    cfg_path: Path,
    tmp_path: Path,
    tiny_coco_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    grad_ckpt: bool,
) -> tuple[float, float]:
    """Run training with checkpointing on/off; return (first-step loss, peak VRAM GB).

    Forces ``train.log_every=1`` so the very first step is always logged —
    the gpu_smoke_*.yaml configs ship with ``log_every=10``, which would
    produce an empty tracker.scalars on a short run.
    """
    cfg = load_config(
        cfg_path,
        overrides=[
            f"data.train.annotations={tiny_coco_dir / 'annotations.json'}",
            f"data.train.images={tiny_coco_dir / 'images'}",
            f"data.val.annotations={tiny_coco_dir / 'annotations.json'}",
            f"data.val.images={tiny_coco_dir / 'images'}",
            f"run.output_dir={tmp_path}",
            f"model.gradient_checkpointing={'true' if grad_ckpt else 'false'}",
            "train.log_every=1",
        ],
    )
    tracker = _RecordingTracker()
    # Patch the consumer's namespace (custom_sam_peft.train.runner) rather than
    # the producer (custom_sam_peft.tracking) — runner.py does
    # `from custom_sam_peft.tracking import build_tracker` at import time, so the
    # bound name lives in runner.__dict__. See spec §4.2.
    monkeypatch.setattr(
        "custom_sam_peft.train.runner.build_tracker",
        lambda *_a, **_kw: tracker,
    )
    torch.cuda.reset_peak_memory_stats()
    run_training(cfg)
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    losses = [s["loss/total"] for _, s in tracker.scalars if s["loss/total"] > 0]
    assert losses, "expected at least one logged loss/total scalar"
    return losses[0], peak_gb


def test_lora_no_checkpoint_error_and_vram_lower(
    tmp_path: Path,
    tiny_coco_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """LoRA: checkpointing completes without error, first-step loss matches, VRAM lower."""
    off_loss0, off_peak = _run(_LORA, tmp_path / "off", tiny_coco_dir, monkeypatch, grad_ckpt=False)
    on_loss0, on_peak = _run(_LORA, tmp_path / "on", tiny_coco_dir, monkeypatch, grad_ckpt=True)
    assert abs(on_loss0 - off_loss0) <= 1e-2 * max(1.0, abs(off_loss0)), (
        f"first-step loss parity failed: on={on_loss0} off={off_loss0}"
    )
    assert on_peak <= off_peak - VRAM_MARGIN_GB, (
        f"checkpointing did not lower peak VRAM: "
        f"on={on_peak:.2f}GB off={off_peak:.2f}GB "
        f"(expected ≥{VRAM_MARGIN_GB}GB reduction)"
    )


@pytest.mark.requires_bnb
@pytest.mark.skipif(
    not _bnb_available(),
    reason="bitsandbytes not installed",
)
def test_qlora_no_checkpoint_error_and_vram_lower(
    tmp_path: Path,
    tiny_coco_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """QLoRA: checkpointing completes without error, first-step loss matches, VRAM lower."""
    off_loss0, off_peak = _run(
        _QLORA, tmp_path / "off", tiny_coco_dir, monkeypatch, grad_ckpt=False
    )
    on_loss0, on_peak = _run(_QLORA, tmp_path / "on", tiny_coco_dir, monkeypatch, grad_ckpt=True)
    assert abs(on_loss0 - off_loss0) <= 1e-2 * max(1.0, abs(off_loss0)), (
        f"QLoRA first-step loss parity failed: on={on_loss0} off={off_loss0}"
    )
    assert on_peak <= off_peak - VRAM_MARGIN_GB, (
        f"QLoRA checkpointing did not lower peak VRAM: "
        f"on={on_peak:.2f}GB off={off_peak:.2f}GB "
        f"(expected ≥{VRAM_MARGIN_GB}GB reduction)"
    )
