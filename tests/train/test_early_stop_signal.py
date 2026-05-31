"""run_epoch raises _EarlyStop after an eval when the predicate fires (spec §4.2)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from custom_sam_peft.data.collate import collate_batch
from custom_sam_peft.peft_adapters.lora import apply_lora
from custom_sam_peft.tracking.noop import NoopTracker
from custom_sam_peft.train.loop import _EarlyStop, run_epoch
from tests.fixtures.tiny_sam3_lora_stub import make_stub_wrapper
from tests.integration.test_trainer_evaluator_seam import _make_cfg, _TinyDataset


def _loader(ds: _TinyDataset) -> list[dict[str, object]]:
    return [collate_batch([ds[i]]) for i in range(len(ds))]


def test_run_epoch_raises_early_stop_after_eval(tmp_path: Path) -> None:
    ds = _TinyDataset()
    wrapper = make_stub_wrapper(dim=8, working=True)
    cfg = _make_cfg(tmp_path)
    cfg = cfg.model_copy(update={"train": cfg.train.model_copy(update={"eval_every": 1})})
    apply_lora(wrapper, cfg.peft)
    run_dir = tmp_path / "run"
    (run_dir / "checkpoints").mkdir(parents=True)

    opt = torch.optim.AdamW([p for p in wrapper.parameters() if p.requires_grad], lr=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda s: 1.0)

    # Predicate returns a stop after the first eval.
    fired = {"n": 0}

    def should_stop_early() -> _EarlyStop | None:
        fired["n"] += 1
        return _EarlyStop(step=fired["n"], epoch=0, reason="test stop")

    with pytest.raises(_EarlyStop) as exc:
        run_epoch(
            wrapper,
            _loader(ds),
            opt,
            sched,
            NoopTracker(),
            cfg,
            run_dir,
            epoch=0,
            global_step=0,
            nan_streak=0,
            class_names=ds.class_names,
            on_checkpoint=lambda *a: None,
            on_eval=lambda *a: None,
            should_stop_early=should_stop_early,
        )
    assert exc.value.reason == "test stop"


def test_run_epoch_no_stop_when_predicate_returns_none(tmp_path: Path) -> None:
    ds = _TinyDataset()
    wrapper = make_stub_wrapper(dim=8, working=True)
    cfg = _make_cfg(tmp_path)
    cfg = cfg.model_copy(update={"train": cfg.train.model_copy(update={"eval_every": 1})})
    apply_lora(wrapper, cfg.peft)
    run_dir = tmp_path / "run2"
    (run_dir / "checkpoints").mkdir(parents=True)
    opt = torch.optim.AdamW([p for p in wrapper.parameters() if p.requires_grad], lr=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda s: 1.0)
    # No predicate (None) → runs to the end of the epoch without raising.
    gs, _ = run_epoch(
        wrapper,
        _loader(ds),
        opt,
        sched,
        NoopTracker(),
        cfg,
        run_dir,
        epoch=0,
        global_step=0,
        nan_streak=0,
        class_names=ds.class_names,
        on_checkpoint=lambda *a: None,
        on_eval=lambda *a: None,
        should_stop_early=None,
    )
    assert gs == len(ds)
