"""Time-limit stop trigger + checkpoint flush (spec §11.3)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

from custom_sam_peft.eval._artifacts import EvalArtifacts
from custom_sam_peft.peft_adapters.lora import apply_lora
from custom_sam_peft.tracking.noop import NoopTracker
from custom_sam_peft.train.loop import _TimeLimitReached, run_epoch
from custom_sam_peft.train.trainer import Trainer
from tests.fixtures.tiny_sam3_lora_stub import make_stub_wrapper
from tests.integration.test_trainer_evaluator_seam import _make_cfg, _TinyDataset


def _loader(ds: _TinyDataset) -> list[dict[str, object]]:
    from custom_sam_peft.data.collate import collate_batch

    return [collate_batch([ds[i]]) for i in range(len(ds))]


def test_run_epoch_flushes_and_raises_on_past_deadline(tmp_path: Path) -> None:
    """A deadline already in the past flushes step_<N>/ and raises _TimeLimitReached."""
    ds = _TinyDataset()
    wrapper = make_stub_wrapper(dim=8, working=True)
    cfg = _make_cfg(tmp_path)
    apply_lora(wrapper, cfg.peft)
    run_dir = tmp_path / "run"
    (run_dir / "checkpoints").mkdir(parents=True)

    optimizer = torch.optim.AdamW([p for p in wrapper.parameters() if p.requires_grad], lr=1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda s: 1.0)

    with pytest.raises(_TimeLimitReached) as exc:
        run_epoch(
            wrapper,
            _loader(ds),
            optimizer,
            scheduler,
            NoopTracker(),
            cfg,
            run_dir,
            epoch=0,
            global_step=0,
            nan_streak=0,
            class_names=ds.class_names,
            on_checkpoint=lambda *a: None,
            on_eval=lambda *a: None,
            deadline=0.0,  # monotonic 0 is always in the past -> fires after step 1
        )
    assert exc.value.step == 1  # stop fires right after the first micro-step
    ckpt = run_dir / "checkpoints" / f"step_{exc.value.step}"
    assert (ckpt / "adapter").exists()
    assert (ckpt / "training_state.pt").exists()


def test_fit_stops_flushes_and_skips_finalize(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """fit() with a near-immediate budget stops, flushes step_<N>/, skips finalize."""
    ds = _TinyDataset()
    wrapper = make_stub_wrapper(dim=8, working=True)
    cfg = _make_cfg(tmp_path)
    # save_every left default; epochs high so the budget, not the epoch count, ends the run.
    cfg = cfg.model_copy(
        update={"train": cfg.train.model_copy(update={"time_limit": "2h30m", "epochs": 50})}
    )
    apply_lora(wrapper, cfg.peft)

    import custom_sam_peft.train.loop as loop_mod

    monkeypatch.setattr(loop_mod.time, "monotonic", lambda: float("inf"))
    trainer = Trainer(wrapper, ds, ds, NoopTracker(), cfg)
    result = trainer.fit(run_dir=tmp_path / "stop-run")

    assert isinstance(result, EvalArtifacts)
    assert result.time_limit_stop is not None
    stop = result.time_limit_stop
    assert stop.stop_step == 1
    assert stop.total_epochs == 50
    assert stop.duration_label == "2h30m"
    # Flushed step checkpoint exists; run_dir/adapter and metrics.json do NOT.
    ckpt = tmp_path / "stop-run" / "checkpoints" / f"step_{stop.stop_step}"
    assert (ckpt / "adapter").exists()
    assert result.checkpoint_path == ckpt / "adapter"
    assert not (tmp_path / "stop-run" / "adapter").exists()
    assert not (tmp_path / "stop-run" / "metrics.json").exists()
    assert result.final_metrics is None


def test_flush_extra_reads_live_values_not_epoch_start_snapshot(tmp_path: Path) -> None:
    """flush_extra lambda is evaluated at deadline time, not at run_epoch call time.

    Regression for the stale-snapshot bug: the three eager ladder/best_metric_value/
    scheduler_kind params were snapshotted at _train_epoch entry; if on_eval mutated
    self._ladder mid-epoch, the flush would write the stale epoch-start state.

    This test verifies the fix by passing a flush_extra whose returned dict comes from
    a mutable container that run_epoch itself cannot observe. We mutate the container
    between epoch-start and deadline, then assert the flushed training_state reflects
    the post-mutation value.
    """
    ds = _TinyDataset()
    wrapper = make_stub_wrapper(dim=8, working=True)
    cfg = _make_cfg(tmp_path)
    apply_lora(wrapper, cfg.peft)
    run_dir = tmp_path / "run"
    (run_dir / "checkpoints").mkdir(parents=True)

    optimizer = torch.optim.AdamW([p for p in wrapper.parameters() if p.requires_grad], lr=1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda s: 1.0)

    # Mutable container: simulates the live trainer state that on_eval would mutate.
    live_state: dict[str, Any] = {
        "ladder": {"best": float("-inf"), "evals_without_improvement": 0},
        "best_metric_value": None,
        "scheduler_kind": "cosine",
    }

    # Sentinel: mutate live_state BEFORE the epoch actually runs (representing
    # a mid-epoch eval that updated the ladder). The flush must capture this
    # updated value, not the epoch-start snapshot.
    post_eval_ladder = {"best": 0.42, "evals_without_improvement": 0}

    calls: list[dict[str, Any]] = []

    def flush_extra() -> dict[str, Any]:
        # Simulate that the trainer's live state was updated by a mid-epoch eval.
        live_state["ladder"] = post_eval_ladder
        live_state["best_metric_value"] = 0.42
        snapshot = dict(live_state)
        calls.append(snapshot)
        return snapshot

    with pytest.raises(_TimeLimitReached):
        run_epoch(
            wrapper,
            _loader(ds),
            optimizer,
            scheduler,
            NoopTracker(),
            cfg,
            run_dir,
            epoch=0,
            global_step=0,
            nan_streak=0,
            class_names=ds.class_names,
            on_checkpoint=lambda *a: None,
            on_eval=lambda *a: None,
            deadline=0.0,  # always in the past -> fires after step 1
            flush_extra=flush_extra,
        )

    # flush_extra must have been called exactly once (at the deadline branch).
    assert len(calls) == 1, "flush_extra should be called exactly once at the deadline"

    # The flushed training_state.pt must contain the post-eval ladder, not epoch-start.
    ckpt_dir = run_dir / "checkpoints" / "step_1"
    assert (ckpt_dir / "training_state.pt").exists()
    state = torch.load(ckpt_dir / "training_state.pt", weights_only=False)
    assert state["ladder"] == post_eval_ladder, (
        f"Expected flushed ladder to reflect post-eval state {post_eval_ladder!r}; "
        f"got {state['ladder']!r}"
    )
    assert state["best_metric_value"] == pytest.approx(0.42), (
        f"Expected flushed best_metric_value=0.42; got {state['best_metric_value']!r}"
    )
