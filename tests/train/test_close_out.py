"""close_out best-restoration + single eval + write (spec §7.2, §14.6)."""

from __future__ import annotations

import json
from pathlib import Path

from custom_sam_peft.eval._artifacts import EvalArtifacts
from custom_sam_peft.peft_adapters.lora import apply_lora
from custom_sam_peft.train.checkpoint import save_adapter
from custom_sam_peft.train.close_out import close_out
from custom_sam_peft.train.ladder import LadderEvents, LrCut
from tests.fixtures.tiny_sam3_lora_stub import make_stub_wrapper
from tests.integration.test_trainer_evaluator_seam import _TinyDataset, _make_cfg


def test_close_out_restores_best_and_writes_adapter(tmp_path: Path) -> None:
    ds = _TinyDataset()
    wrapper = make_stub_wrapper(dim=8, working=True)
    cfg = _make_cfg(tmp_path)
    apply_lora(wrapper, cfg.peft)
    run_dir = tmp_path / "run"
    (run_dir / "best").mkdir(parents=True)
    # Save a distinguishable best/ adapter + best.json.
    save_adapter(wrapper, run_dir / "best" / "adapter")
    (run_dir / "best" / "best.json").write_text(
        json.dumps({"metric": "mAP", "value": 0.8, "global_step": 7})
    )

    events = LadderEvents(cuts=(LrCut(6, 1e-4, 1e-5, 0.5),), stop_reason="early_stop: 10 ...")
    art = close_out(
        run_dir, wrapper, cfg, evaluator_val_ds=ds, oom_state=None,
        final_step=7, final_epoch=0, ladder_events=events,
    )
    assert isinstance(art, EvalArtifacts)
    assert (run_dir / "adapter").is_dir()  # adapter written
    assert art.checkpoint_path == run_dir / "adapter"
    assert art.final_weights == "best"
    assert art.per_example_iou is not None  # single eval returned per-example IoU
    assert art.ladder_events == events  # also rides the returned EvalArtifacts
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert metrics["final_weights"] == "best"
    assert "ladder_events" in metrics


def test_close_out_falls_back_to_last_step_when_no_best(tmp_path: Path) -> None:
    ds = _TinyDataset()
    wrapper = make_stub_wrapper(dim=8, working=True)
    cfg = _make_cfg(tmp_path)
    apply_lora(wrapper, cfg.peft)
    run_dir = tmp_path / "run-nobest"
    run_dir.mkdir(parents=True)
    art = close_out(
        run_dir, wrapper, cfg, evaluator_val_ds=ds, oom_state=None,
        final_step=3, final_epoch=0, ladder_events=None,
    )
    assert art.final_weights == "last_step"
    assert art.ladder_events is None
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert metrics["final_weights"] == "last_step"


def test_close_out_no_val_returns_none_metrics(tmp_path: Path) -> None:
    wrapper = make_stub_wrapper(dim=8, working=True)
    cfg = _make_cfg(tmp_path)
    apply_lora(wrapper, cfg.peft)
    run_dir = tmp_path / "run-noval"
    run_dir.mkdir(parents=True)
    art = close_out(
        run_dir, wrapper, cfg, evaluator_val_ds=None, oom_state=None,
        final_step=3, final_epoch=0, ladder_events=None,
    )
    assert art.final_metrics is None
    assert art.per_example_iou is None
    assert art.ladder_events is None
    assert (run_dir / "adapter").is_dir()
