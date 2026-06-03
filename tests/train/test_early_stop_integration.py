"""Early stop funnels into close_out; best-as-final on stop + normal completion (§14.7)."""

from __future__ import annotations

import json
from pathlib import Path

from custom_sam_peft.eval._artifacts import EvalArtifacts
from custom_sam_peft.peft_adapters.lora import apply_lora
from custom_sam_peft.tracking.noop import NoopTracker
from custom_sam_peft.train.trainer import Trainer
from tests.fixtures.tiny_sam3_lora_stub import make_stub_wrapper
from tests.integration.test_trainer_evaluator_seam import _make_cfg, _TinyDataset


def test_normal_completion_closes_out_on_best(tmp_path: Path) -> None:
    ds = _TinyDataset()
    wrapper = make_stub_wrapper(dim=8, working=True)
    cfg = _make_cfg(tmp_path)
    cfg = cfg.model_copy(
        update={
            "train": cfg.train.model_copy(
                update={"lr_schedule": "poly", "eval_every": 1, "epochs": 1}
            )
        }
    )
    apply_lora(wrapper, cfg.peft)
    trainer = Trainer(wrapper, ds, ds, NoopTracker(), cfg)
    result = trainer.fit(run_dir=tmp_path / "normal-run")
    assert isinstance(result, EvalArtifacts)
    assert result.checkpoint_path == tmp_path / "normal-run" / "adapter"
    assert result.final_weights in {"best", "last_step"}
    metrics = json.loads((tmp_path / "normal-run" / "metrics.json").read_text())
    assert "final_weights" in metrics


def test_early_stop_stops_before_epochs_and_closes_out(tmp_path: Path, monkeypatch) -> None:
    """Injected plateau mAPs trigger _EarlyStop; fit returns best-as-final artifacts."""
    import custom_sam_peft.eval.evaluator as ev

    ds = _TinyDataset()
    wrapper = make_stub_wrapper(dim=8, working=True)
    cfg = _make_cfg(tmp_path)
    cfg = cfg.model_copy(
        update={
            "train": cfg.train.model_copy(
                update={
                    "lr_schedule": "poly",
                    "eval_every": 1,
                    "epochs": 50,
                    # warmup_floor_steps=0 → adaptive-baseline-only grace: the
                    # first strictly-positive mAP wakes the run, then the flat
                    # mAP plateau accrues the no-improvement counter (#264).
                    "early_stop": cfg.train.early_stop.model_copy(
                        update={"stop_patience": 2, "warmup_floor_steps": 0}
                    ),
                }
            )
        }
    )
    apply_lora(wrapper, cfg.peft)

    # Force every eval to report a flat mAP so the ladder never improves.
    from custom_sam_peft.eval.metrics import MetricsReport

    flat = MetricsReport(overall={"mAP": 0.1}, per_class={}, n_images=1, n_predictions=0)

    def fake_eval(self, model, dataset, **k):
        if k.get("return_per_example_iou"):
            return flat, [0.1]
        return flat

    monkeypatch.setattr(ev.Evaluator, "evaluate", fake_eval)
    trainer = Trainer(wrapper, ds, ds, NoopTracker(), cfg)
    result = trainer.fit(run_dir=tmp_path / "stop-run")
    assert isinstance(result, EvalArtifacts)
    metrics = json.loads((tmp_path / "stop-run" / "metrics.json").read_text())
    assert "final_weights" in metrics
    # Stopped well before 50 epochs (stop_patience=2 → stops within a few evals).
    assert metrics["global_step"] < 50 * len(ds)
