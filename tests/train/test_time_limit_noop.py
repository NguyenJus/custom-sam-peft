"""Unset time_limit runs the full loop + finalize exactly as today (spec §11.5)."""

from __future__ import annotations

from pathlib import Path

from custom_sam_peft.eval._artifacts import EvalArtifacts
from custom_sam_peft.peft_adapters.lora import apply_lora
from custom_sam_peft.tracking.noop import NoopTracker
from custom_sam_peft.train.trainer import Trainer
from tests.fixtures.tiny_sam3_lora_stub import make_stub_wrapper
from tests.integration.test_trainer_evaluator_seam import _make_cfg, _TinyDataset


def test_fit_without_time_limit_finalizes_as_today(tmp_path: Path) -> None:
    ds = _TinyDataset()
    wrapper = make_stub_wrapper(dim=8, working=True)
    cfg = _make_cfg(tmp_path)  # no time_limit set (default None)
    apply_lora(wrapper, cfg.peft)

    trainer = Trainer(wrapper, ds, ds, NoopTracker(), cfg)
    result = trainer.fit(run_dir=tmp_path / "noop-run")

    assert isinstance(result, EvalArtifacts)
    assert result.time_limit_stop is None
    assert (tmp_path / "noop-run" / "adapter").exists()
    assert (tmp_path / "noop-run" / "metrics.json").exists()
    assert result.checkpoint_path == tmp_path / "noop-run" / "adapter"
