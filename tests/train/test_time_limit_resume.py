"""Resume cleanly continues from a time-limited stop's flushed checkpoint (spec §11.4)."""

from __future__ import annotations

from pathlib import Path

import pytest

from custom_sam_peft.peft_adapters.lora import apply_lora
from custom_sam_peft.tracking.noop import NoopTracker
from custom_sam_peft.train.checkpoint import find_latest_checkpoint
from custom_sam_peft.train.trainer import Trainer
from tests.fixtures.tiny_sam3_lora_stub import make_stub_wrapper
from tests.integration.test_trainer_evaluator_seam import _make_cfg, _TinyDataset


def test_resume_after_time_limited_stop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import custom_sam_peft.train.loop as loop_mod

    ds = _TinyDataset()
    cfg = _make_cfg(tmp_path)
    cfg = cfg.model_copy(
        update={"train": cfg.train.model_copy(update={"time_limit": "2h30m", "epochs": 50})}
    )

    # First run: stop on the budget.
    w1 = make_stub_wrapper(dim=8, working=True)
    apply_lora(w1, cfg.peft)
    monkeypatch.setattr(loop_mod.time, "monotonic", lambda: float("inf"))
    run_dir_1 = Path(cfg.run.output_dir) / f"{cfg.run.name}-1"
    r1 = Trainer(w1, ds, ds, NoopTracker(), cfg).fit(run_dir=run_dir_1)
    assert r1.time_limit_stop is not None

    latest = find_latest_checkpoint(cfg)
    assert latest.name.startswith("step_")

    # Second run: no budget, resume from the flushed checkpoint, run to completion.
    monkeypatch.undo()  # revert the inf patch so the second run uses real time
    cfg2 = cfg.model_copy(
        update={"train": cfg.train.model_copy(update={"time_limit": None, "epochs": 1})}
    )
    w2 = make_stub_wrapper(dim=8, working=True)
    apply_lora(w2, cfg2.peft)
    run_dir_2 = Path(cfg2.run.output_dir) / f"{cfg2.run.name}-2"
    r2 = Trainer(w2, ds, ds, NoopTracker(), cfg2).fit(run_dir=run_dir_2, resume_from=latest)
    assert r2.time_limit_stop is None
    assert (run_dir_2 / "adapter").exists()
    assert (run_dir_2 / "metrics.json").exists()
    assert r2.checkpoint_path == run_dir_2 / "adapter"
