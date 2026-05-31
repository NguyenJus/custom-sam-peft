"""plateau + no val falls back to cosine with a warning (spec §6.5, §14.2)."""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

from custom_sam_peft.peft_adapters.lora import apply_lora
from custom_sam_peft.tracking.noop import NoopTracker
from custom_sam_peft.train.trainer import Trainer
from tests.fixtures.tiny_sam3_lora_stub import make_stub_wrapper
from tests.integration.test_trainer_evaluator_seam import _make_cfg, _TinyDataset


def test_plateau_no_val_falls_back_to_cosine(tmp_path: Path, caplog) -> None:
    ds = _TinyDataset()
    wrapper = make_stub_wrapper(dim=8, working=True)
    cfg = _make_cfg(tmp_path)
    # Use warmup_steps=1 and epochs=3 so there are enough steps (6 total with
    # _TinyDataset of len=2, batch=1) for cosine decay to be observable after warmup.
    # Under the FIX-1 bug, step_per_train_step receives mode='plateau' (the requested
    # schedule) instead of mode='cosine' (the effective/fallback schedule), so
    # LambdaLR.step() is never called after warmup — LR stays constant at base_lr.
    cfg = cfg.model_copy(
        update={
            "train": cfg.train.model_copy(
                update={
                    "lr_schedule": "plateau",
                    "warmup_steps": 1,
                    "epochs": 3,
                }
            )
        }
    )
    apply_lora(wrapper, cfg.peft)

    # Capture the optimizer so we can read the final LR.
    base_lr = cfg.train.learning_rate

    # val_ds=None -> no plateau signal.
    trainer = Trainer(wrapper, ds, None, NoopTracker(), cfg)
    with caplog.at_level(logging.WARNING):
        result = trainer.fit(run_dir=tmp_path / "fallback-run")

    # Fell back to a per-step LambdaLR (cosine), not ReduceLROnPlateau.
    assert any("falling back to lr_schedule=cosine" in r.message for r in caplog.records)
    # The run completed normally (no early stop, no crash).
    assert result.run_dir.is_dir()
    # config.yaml still echoes the requested plateau.
    saved = yaml.safe_load((result.run_dir / "config.yaml").read_text())
    assert saved["train"]["lr_schedule"] == "plateau"

    # --- FIX-1 regression guard ---
    # The fallback scheduler is a cosine LambdaLR. With warmup_steps=1 and
    # total_steps=6 (3 epochs * 2 steps), cosine decays to near-zero by the end.
    # Under the bug, step_per_train_step uses mode='plateau' and never calls
    # LambdaLR.step(), so LR stays constant at base_lr (decay never fires).
    # After a correct fix, the final LR must be strictly below base_lr.
    metrics_path = result.run_dir / "metrics.json"
    assert metrics_path.exists()
    # Read the LR from the saved config (it's not in metrics.json, but we can
    # check via the scheduler's last-step behaviour indirectly). Instead, use the
    # recorded tracker scalars which include "lr" in every log_scalars flush.
    # Since log_every=1, we get one lr entry per step. Import the recording tracker.
    from custom_sam_peft.tracking.noop import NoopTracker as _NoopTracker  # noqa: F401

    # Re-run with a recording tracker to capture the final LR scalar.
    # (We can't easily inspect the optimizer after fit() since it's a local variable.)
    # Use a dedicated sub-run with a fresh wrapper.
    wrapper2 = make_stub_wrapper(dim=8, working=True)
    apply_lora(wrapper2, cfg.peft)

    class _LRCapture:
        """Minimal tracker that records lr scalars."""

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
    trainer2 = Trainer(wrapper2, ds, None, lr_capture, cfg)
    trainer2.fit(run_dir=tmp_path / "fallback-run-2")

    assert lr_capture.lr_history, "No LR scalars were logged — check log_every setting."
    final_lr = lr_capture.lr_history[-1]
    assert final_lr < base_lr, (
        f"plateau->cosine fallback: final LR {final_lr:.6e} >= base_lr {base_lr:.6e}. "
        "LambdaLR cosine decay did not fire — step_per_train_step likely used "
        "mode='plateau' (requested) instead of mode='cosine' (effective/fallback)."
    )
