"""Poly schedule + no-val: LR decays without mAP input (#264).

Replaces the old plateau-fallback test. "plateau" is no longer a valid
lr_schedule; poly is the new default. A run with val_ds=None trains to the
horizon with poly decay — no fallback warning, no crash.
"""

from __future__ import annotations

from pathlib import Path
from typing import get_args

import pytest

from custom_sam_peft.config.schema import LRSchedule
from custom_sam_peft.peft_adapters.lora import apply_lora
from custom_sam_peft.train.trainer import Trainer
from tests.fixtures.tiny_sam3_lora_stub import make_stub_wrapper
from tests.integration.test_trainer_evaluator_seam import _make_cfg, _TinyDataset


def test_plateau_not_a_valid_lr_schedule() -> None:
    """'plateau' must not appear in the LRSchedule literal (#264)."""
    valid = get_args(LRSchedule)
    assert "plateau" not in valid, f"'plateau' found in LRSchedule: {valid}"


def test_poly_is_the_default_lr_schedule() -> None:
    """poly is the default lr_schedule after #264."""
    from custom_sam_peft.config.schema import TrainHyperparams

    th = TrainHyperparams(epochs=1)
    assert th.lr_schedule == "poly"


def test_poly_no_val_runs_to_completion(tmp_path: Path) -> None:
    """poly schedule with val_ds=None runs to completion without warning or crash."""
    ds = _TinyDataset()
    wrapper = make_stub_wrapper(dim=8, working=True)
    cfg = _make_cfg(tmp_path)
    cfg = cfg.model_copy(
        update={
            "train": cfg.train.model_copy(
                update={
                    "lr_schedule": "poly",
                    "warmup_steps": 1,
                    "epochs": 3,
                }
            )
        }
    )
    apply_lora(wrapper, cfg.peft)
    base_lr = cfg.train.learning_rate

    # Capture LR to verify decay fired.
    class _LRCapture:
        def __init__(self) -> None:
            self.lr_history: list[float] = []

        def start_run(self, *a, **kw) -> None:
            pass

        def log_scalars(self, step: int, values: dict) -> None:
            if "lr" in values:
                self.lr_history.append(float(values["lr"]))

        def log_images(self, *a, **kw) -> None:
            pass

        def close(self) -> None:
            pass

    lr_capture = _LRCapture()
    trainer = Trainer(wrapper, ds, None, lr_capture, cfg)
    result = trainer.fit(run_dir=tmp_path / "poly-run")

    assert result.run_dir.is_dir()
    assert lr_capture.lr_history, "No LR scalars logged"
    final_lr = lr_capture.lr_history[-1]
    # With poly decay over 3 epochs (6 steps, warmup=1), LR must decay below base_lr.
    assert final_lr < base_lr, (
        f"poly schedule did not decay: final_lr={final_lr:.6e} >= base_lr={base_lr:.6e}"
    )


def test_invalid_lr_schedule_rejected_by_schema() -> None:
    """Pydantic validation rejects 'plateau' as an lr_schedule value."""
    import pydantic

    from custom_sam_peft.config.schema import TrainHyperparams

    with pytest.raises((pydantic.ValidationError, ValueError)):
        TrainHyperparams(epochs=1, lr_schedule="plateau")  # type: ignore[arg-type]
