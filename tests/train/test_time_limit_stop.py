"""Time-limit stop trigger + checkpoint flush (spec §11.3)."""

from __future__ import annotations

from pathlib import Path

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
