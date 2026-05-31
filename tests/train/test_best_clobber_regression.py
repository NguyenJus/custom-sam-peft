"""A post-resume eval must not clobber a better best/ (spec §8.2, §14.5)."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from custom_sam_peft.peft_adapters.lora import apply_lora
from custom_sam_peft.tracking.noop import NoopTracker
from custom_sam_peft.train.checkpoint import save_full_state
from custom_sam_peft.train.trainer import Trainer
from tests.fixtures.tiny_sam3_lora_stub import make_stub_wrapper
from tests.integration.test_trainer_evaluator_seam import _make_cfg, _TinyDataset


def test_resume_reseeds_best_from_best_json(tmp_path: Path) -> None:
    ds = _TinyDataset()
    wrapper = make_stub_wrapper(dim=8, working=True)
    cfg = _make_cfg(tmp_path)
    apply_lora(wrapper, cfg.peft)

    # Build a run_dir with a best/best.json claiming mAP=0.7 and a step checkpoint.
    run_dir = tmp_path / "clobber-run"
    (run_dir / "best").mkdir(parents=True)
    (run_dir / "best" / "best.json").write_text(
        json.dumps({"metric": "mAP", "value": 0.7, "global_step": 5})
    )
    opt = torch.optim.AdamW([p for p in wrapper.parameters() if p.requires_grad], lr=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max")
    state_dir = run_dir / "checkpoints" / "step_5"
    save_full_state(
        state_dir=state_dir,
        wrapper=wrapper,
        optimizer=opt,
        scheduler=sched,
        global_step=5,
        epoch=0,
        nan_streak=0,
        cfg=cfg,
        ladder={"best": 0.7, "evals_without_improvement": 0},
        best_metric_value=0.7,
        scheduler_kind="plateau",
    )

    tracker = NoopTracker()
    trainer = Trainer(wrapper, ds, ds, tracker, cfg)
    # Resume — fit() re-seeds _best_metric_value from best.json BEFORE any eval.
    trainer.fit(run_dir=run_dir, resume_from=state_dir)
    assert trainer._best_metric_value >= 0.7  # type: ignore[attr-defined]
    # best/best.json still claims 0.7 (a worse post-resume eval did not overwrite it).
    saved = json.loads((run_dir / "best" / "best.json").read_text())
    assert saved["value"] >= 0.7
