"""The ladder ticks only on a successful eval, after _maybe_save_best (spec §4.1, §10)."""

from __future__ import annotations

from pathlib import Path

from custom_sam_peft.peft_adapters.lora import apply_lora
from custom_sam_peft.tracking.noop import NoopTracker
from custom_sam_peft.train.trainer import Trainer
from tests.fixtures.tiny_sam3_lora_stub import make_stub_wrapper
from tests.integration.test_trainer_evaluator_seam import _make_cfg, _TinyDataset


def test_failed_eval_does_not_tick(tmp_path: Path, monkeypatch) -> None:
    """An eval that raises advances NEITHER counter (tick is inside the try, after save_best)."""
    import custom_sam_peft.eval.evaluator as ev

    ds = _TinyDataset()
    wrapper = make_stub_wrapper(dim=8, working=True)
    cfg = _make_cfg(tmp_path)
    cfg = cfg.model_copy(
        update={
            "train": cfg.train.model_copy(
                update={"lr_schedule": "plateau", "eval_every": 1, "epochs": 1}
            )
        }
    )
    apply_lora(wrapper, cfg.peft)

    def boom(self, model, dataset, **k):
        raise RuntimeError("eval OOM at batch_size=1")

    monkeypatch.setattr(ev.Evaluator, "evaluate", boom)
    trainer = Trainer(wrapper, ds, ds, NoopTracker(), cfg)
    trainer.fit(run_dir=tmp_path / "tick-run")
    # The ladder exists and was never advanced (all evals failed).
    assert trainer._ladder.evals_without_improvement == 0  # type: ignore[attr-defined]
