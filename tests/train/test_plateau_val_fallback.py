"""plateau + no val falls back to cosine with a warning (spec §6.5, §14.2)."""

from __future__ import annotations

import logging
from pathlib import Path

from custom_sam_peft.peft_adapters.lora import apply_lora
from custom_sam_peft.tracking.noop import NoopTracker
from custom_sam_peft.train.trainer import Trainer
from tests.fixtures.tiny_sam3_lora_stub import make_stub_wrapper
from tests.integration.test_trainer_evaluator_seam import _make_cfg, _TinyDataset


def test_plateau_no_val_falls_back_to_cosine(tmp_path: Path, caplog) -> None:
    ds = _TinyDataset()
    wrapper = make_stub_wrapper(dim=8, working=True)
    cfg = _make_cfg(tmp_path)
    cfg = cfg.model_copy(update={"train": cfg.train.model_copy(update={"lr_schedule": "plateau"})})
    apply_lora(wrapper, cfg.peft)

    # val_ds=None → no plateau signal.
    trainer = Trainer(wrapper, ds, None, NoopTracker(), cfg)
    with caplog.at_level(logging.WARNING):
        result = trainer.fit(run_dir=tmp_path / "fallback-run")

    # Fell back to a per-step LambdaLR (cosine), not ReduceLROnPlateau.
    assert any("falling back to lr_schedule=cosine" in r.message for r in caplog.records)
    # The run completed normally (no early stop, no crash).
    assert result.run_dir.is_dir()
    # config.yaml still echoes the requested plateau.
    import yaml

    saved = yaml.safe_load((result.run_dir / "config.yaml").read_text())
    assert saved["train"]["lr_schedule"] == "plateau"
