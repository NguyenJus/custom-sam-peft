"""close_out best-restoration + single eval + write (spec §7.2, §14.6)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from custom_sam_peft.config._internal import ExportConfig
from custom_sam_peft.eval._artifacts import EvalArtifacts
from custom_sam_peft.peft_adapters.lora import apply_lora
from custom_sam_peft.train.checkpoint import save_adapter
from custom_sam_peft.train.close_out import close_out
from custom_sam_peft.train.ladder import LadderEvents, LrCut
from tests.fixtures.tiny_sam3_lora_stub import make_stub_wrapper
from tests.integration.test_trainer_evaluator_seam import _make_cfg, _TinyDataset


def _cfg_no_viz(tmp_path: Path):  # type: ignore[no-untyped-def]
    """Return a cfg with visualize=False so existing tests stay focused on
    their own behaviour and don't trigger the (potentially slow) viz path."""
    base = _make_cfg(tmp_path)
    return base.model_copy(update={"eval": base.eval.model_copy(update={"visualize": False})})


def test_close_out_restores_best_and_writes_adapter(tmp_path: Path) -> None:
    ds = _TinyDataset()
    wrapper = make_stub_wrapper(dim=8, working=True)
    cfg = _cfg_no_viz(tmp_path)
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
        run_dir,
        wrapper,
        cfg,
        evaluator_val_ds=ds,
        oom_state=None,
        final_step=7,
        final_epoch=0,
        ladder_events=events,
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
    cfg = _cfg_no_viz(tmp_path)
    apply_lora(wrapper, cfg.peft)
    run_dir = tmp_path / "run-nobest"
    run_dir.mkdir(parents=True)
    art = close_out(
        run_dir,
        wrapper,
        cfg,
        evaluator_val_ds=ds,
        oom_state=None,
        final_step=3,
        final_epoch=0,
        ladder_events=None,
    )
    assert art.final_weights == "last_step"
    assert art.ladder_events is None
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert metrics["final_weights"] == "last_step"


def test_close_out_no_val_returns_none_metrics(tmp_path: Path) -> None:
    wrapper = make_stub_wrapper(dim=8, working=True)
    cfg = _cfg_no_viz(tmp_path)
    apply_lora(wrapper, cfg.peft)
    run_dir = tmp_path / "run-noval"
    run_dir.mkdir(parents=True)
    art = close_out(
        run_dir,
        wrapper,
        cfg,
        evaluator_val_ds=None,
        oom_state=None,
        final_step=3,
        final_epoch=0,
        ladder_events=None,
    )
    assert art.final_metrics is None
    assert art.per_example_iou is None
    assert art.ladder_events is None
    assert (run_dir / "adapter").is_dir()


# ---------------------------------------------------------------------------
# Visualization pass tests
# ---------------------------------------------------------------------------


def test_close_out_calls_write_eval_visualizations_when_visualize_true(
    tmp_path: Path,
) -> None:
    """close_out must invoke write_eval_visualizations exactly once on the
    with-val path when cfg.eval.visualize is True, passing count=visualize_count."""
    ds = _TinyDataset()
    wrapper = make_stub_wrapper(dim=8, working=True)
    base_cfg = _make_cfg(tmp_path)
    cfg = base_cfg.model_copy(
        update={"eval": base_cfg.eval.model_copy(update={"visualize": True, "visualize_count": 3})}
    )
    apply_lora(wrapper, cfg.peft)
    run_dir = tmp_path / "run-viz"
    run_dir.mkdir(parents=True)

    recorder = MagicMock(return_value=[])
    with patch("custom_sam_peft.eval.visualize.write_eval_visualizations", recorder):
        art = close_out(
            run_dir,
            wrapper,
            cfg,
            evaluator_val_ds=ds,
            oom_state=None,
            final_step=1,
            final_epoch=0,
            ladder_events=None,
        )

    assert isinstance(art, EvalArtifacts)
    recorder.assert_called_once()
    _, kwargs = recorder.call_args
    assert kwargs["count"] == 3


def test_close_out_skips_viz_when_visualize_false(tmp_path: Path) -> None:
    """write_eval_visualizations must NOT be called when cfg.eval.visualize is False."""
    ds = _TinyDataset()
    wrapper = make_stub_wrapper(dim=8, working=True)
    cfg = _cfg_no_viz(tmp_path)
    apply_lora(wrapper, cfg.peft)
    run_dir = tmp_path / "run-noviz"
    run_dir.mkdir(parents=True)

    recorder = MagicMock(return_value=[])
    with patch("custom_sam_peft.eval.visualize.write_eval_visualizations", recorder):
        close_out(
            run_dir,
            wrapper,
            cfg,
            evaluator_val_ds=ds,
            oom_state=None,
            final_step=1,
            final_epoch=0,
            ladder_events=None,
        )

    recorder.assert_not_called()


def test_close_out_skips_viz_when_no_val(tmp_path: Path) -> None:
    """write_eval_visualizations must NOT be called on the no-val path,
    even when cfg.eval.visualize is True."""
    wrapper = make_stub_wrapper(dim=8, working=True)
    base_cfg = _make_cfg(tmp_path)
    cfg = base_cfg.model_copy(update={"eval": base_cfg.eval.model_copy(update={"visualize": True})})
    apply_lora(wrapper, cfg.peft)
    run_dir = tmp_path / "run-noviz-noval"
    run_dir.mkdir(parents=True)

    recorder = MagicMock(return_value=[])
    with patch("custom_sam_peft.eval.visualize.write_eval_visualizations", recorder):
        close_out(
            run_dir,
            wrapper,
            cfg,
            evaluator_val_ds=None,
            oom_state=None,
            final_step=1,
            final_epoch=0,
            ladder_events=None,
        )

    recorder.assert_not_called()


def test_close_out_merge_failure_soft_fails(tmp_path: Path) -> None:
    """When cfg.export.merge=True and save_merged raises, close_out must:
    - set art.merged_export_error to the error string,
    - still write metrics.json,
    - not re-raise (no exception escapes close_out).
    """
    ds = _TinyDataset()
    wrapper = make_stub_wrapper(dim=8, working=True)
    base_cfg = _cfg_no_viz(tmp_path)
    cfg = base_cfg.model_copy(update={"export": ExportConfig(merge=True)})
    apply_lora(wrapper, cfg.peft)
    run_dir = tmp_path / "run-merge-fail"
    run_dir.mkdir(parents=True)

    def _boom(model: object, path: object) -> None:
        raise RuntimeError("simulated merge failure")

    with patch("custom_sam_peft.train.close_out.save_merged", _boom):
        art = close_out(
            run_dir,
            wrapper,
            cfg,
            evaluator_val_ds=ds,
            oom_state=None,
            final_step=1,
            final_epoch=0,
            ladder_events=None,
        )

    assert isinstance(art, EvalArtifacts)
    assert art.merged_export_error == "simulated merge failure"
    assert not (run_dir / "merged").exists(), "merged/ must not exist after failed merge"
    assert (run_dir / "metrics.json").is_file(), "metrics.json must be written even if merge fails"


def test_close_out_viz_failure_does_not_block_metrics(tmp_path: Path) -> None:
    """A viz exception must not prevent metrics.json from being written."""
    ds = _TinyDataset()
    wrapper = make_stub_wrapper(dim=8, working=True)
    base_cfg = _make_cfg(tmp_path)
    cfg = base_cfg.model_copy(update={"eval": base_cfg.eval.model_copy(update={"visualize": True})})
    apply_lora(wrapper, cfg.peft)
    run_dir = tmp_path / "run-viz-fail"
    run_dir.mkdir(parents=True)

    def _boom(*a, **k):  # type: ignore[no-untyped-def]
        raise RuntimeError("simulated viz failure")

    with patch("custom_sam_peft.eval.visualize.write_eval_visualizations", _boom):
        art = close_out(
            run_dir,
            wrapper,
            cfg,
            evaluator_val_ds=ds,
            oom_state=None,
            final_step=1,
            final_epoch=0,
            ladder_events=None,
        )

    assert isinstance(art, EvalArtifacts)
    assert (run_dir / "metrics.json").is_file(), "metrics.json must be written even if viz fails"
