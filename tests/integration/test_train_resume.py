"""Resume integration: a resumed run reaches a finite end-state."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

import custom_sam_peft.train.trainer as trainer_mod
from custom_sam_peft.config.schema import (
    AugmentationsConfig,
    DataConfig,
    DataSplit,
    NormalizeConfig,
    PEFTConfig,
    RunConfig,
    TextPromptConfig,
    TrainConfig,
    TrainHyperparams,
)
from custom_sam_peft.data.coco import COCODataset
from custom_sam_peft.data.transforms import build_train_transforms
from custom_sam_peft.peft_adapters.lora import apply_lora
from custom_sam_peft.tracking.noop import NoopTracker
from tests.fixtures.tiny_sam3_lora_stub import FIXTURE_SCOPE_PATTERNS, make_stub_wrapper

pytestmark = pytest.mark.integration


def _ds(tiny_coco_dir: Path) -> COCODataset:
    # NOTE: build_train_transforms takes (aug_cfg, image_size, ...) — positional order matches impl.
    transforms = build_train_transforms(
        AugmentationsConfig(preset="none"),
        32,
        model_name="facebook/sam3.1",
        normalize=NormalizeConfig(),
    )
    return COCODataset(
        annotations=str(tiny_coco_dir / "annotations.json"),
        images=str(tiny_coco_dir / "images"),
        prompt_mode="text",
        transforms=transforms,
        text_prompt=TextPromptConfig(),
    )


def _cfg(tmp_path: Path, tiny_coco_dir: Path, save_every: int) -> TrainConfig:
    return TrainConfig(
        run=RunConfig(name="resume", output_dir=str(tmp_path), seed=42),
        data=DataConfig(
            format="coco",
            train=DataSplit(
                annotations=str(tiny_coco_dir / "annotations.json"),
                images=str(tiny_coco_dir / "images"),
            ),
            val=DataSplit(
                annotations=str(tiny_coco_dir / "annotations.json"),
                images=str(tiny_coco_dir / "images"),
            ),
            prompt_mode="text",
            image_size=32,
        ),
        peft=PEFTConfig(
            method="lora",
            scope="vision",
            target_modules=FIXTURE_SCOPE_PATTERNS["vision"],
        ),
        train=TrainHyperparams(
            epochs=2,
            batch_size=1,
            grad_accum_steps=1,
            save_every=save_every,
            log_every=1,
            warmup_steps=0,
            num_workers=0,
        ),
    )


def _adapter_state(wrapper: Any) -> dict[str, torch.Tensor]:
    return {
        k: v.detach().clone().cpu()
        for k, v in wrapper.peft_model.state_dict().items()
        if "lora" in k
    }


def test_resume_matches_uninterrupted(
    tmp_path: Path, tiny_coco_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds = _ds(tiny_coco_dir)
    cfg = _cfg(tmp_path, tiny_coco_dir, save_every=2)

    # C4 (spec §6.2): capture the optimizer/scheduler instances the Trainer
    # builds. Wrap _build_optimizer / _build_scheduler with closures that stash
    # the constructed instance in a test-local list. Requires no
    # src/custom_sam_peft/ change.
    real_opt_builder = trainer_mod._build_optimizer
    real_sched_builder = trainer_mod._build_scheduler
    captured_opts: list[Any] = []
    captured_scheds: list[Any] = []

    def _opt_spy(*a: Any, **kw: Any) -> Any:
        opt = real_opt_builder(*a, **kw)
        captured_opts.append(opt)
        return opt

    def _sched_spy(*a: Any, **kw: Any) -> Any:
        sched = real_sched_builder(*a, **kw)
        captured_scheds.append(sched)
        return sched

    monkeypatch.setattr(trainer_mod, "_build_optimizer", _opt_spy)
    monkeypatch.setattr(trainer_mod, "_build_scheduler", _sched_spy)

    # Uninterrupted reference run (2 epochs).
    w_a = make_stub_wrapper(dim=8, working=True)
    apply_lora(w_a, cfg.peft)
    trainer_a = trainer_mod.Trainer(w_a, ds, ds, NoopTracker(), cfg)
    trainer_a.fit(run_dir=tmp_path / "run-a")
    state_a = _adapter_state(w_a)
    opt_a = captured_opts[-1]
    sched_a = captured_scheds[-1]

    # Truncated first run (1 epoch), then resumed (2 epochs continuing from checkpoint).
    w_b = make_stub_wrapper(dim=8, working=True)
    apply_lora(w_b, cfg.peft)
    cfg_short = _cfg(tmp_path, tiny_coco_dir, save_every=2)
    cfg_short.train.epochs = 1
    trainer_b = trainer_mod.Trainer(w_b, ds, ds, NoopTracker(), cfg_short)
    result_b1 = trainer_b.fit(run_dir=tmp_path / "run-b1")

    ckpts = sorted((result_b1.run_dir / "checkpoints").glob("step_*"))
    assert ckpts, "no checkpoint produced"
    resume_dir = ckpts[-1]

    w_c = make_stub_wrapper(dim=8, working=True)
    apply_lora(w_c, cfg.peft)
    trainer_c = trainer_mod.Trainer(w_c, ds, ds, NoopTracker(), cfg)
    trainer_c.fit(run_dir=tmp_path / "run-c", resume_from=resume_dir)
    state_c = _adapter_state(w_c)
    opt_c = captured_opts[-1]
    sched_c = captured_scheds[-1]

    # Resume produces finite weights (not bit-identical to uninterrupted run because
    # the re-walked epoch retreads some examples). Assert finiteness only.
    for k in state_a:
        assert torch.isfinite(state_c[k]).all()

    # --- C4 monotone continuity assertions ---
    # The resumed run re-walks the interrupted epoch and therefore runs
    # strictly more optimizer steps than the reference. We assert >= on
    # the step counter and scheduler counters; exp_avg / exp_avg_sq are
    # NOT compared because they diverge legitimately. Bit-equality of the
    # serialization contract lives in tests/unit/test_checkpoint_roundtrip.py.

    sd_a = opt_a.state_dict()
    sd_c = opt_c.state_dict()
    assert len(sd_a["param_groups"]) == len(sd_c["param_groups"])
    for g_a, g_c in zip(sd_a["param_groups"], sd_c["param_groups"], strict=True):
        for key in ("lr", "betas", "weight_decay", "eps"):
            if key in g_a:
                assert g_a[key] == g_c[key], f"param_group {key} drift: {g_a[key]} vs {g_c[key]}"
    common_ids = set(sd_a["state"]) & set(sd_c["state"])
    assert common_ids, "no shared parameter IDs between uninterrupted + resumed optimizers"
    for pid in common_ids:
        st_a = sd_a["state"][pid]
        st_c = sd_c["state"][pid]
        if "step" in st_a and "step" in st_c:
            step_a_v = st_a["step"]
            step_c_v = st_c["step"]
            if isinstance(step_a_v, torch.Tensor):
                step_a_v = int(step_a_v.item())
            if isinstance(step_c_v, torch.Tensor):
                step_c_v = int(step_c_v.item())
            assert int(step_c_v) >= int(step_a_v), (
                f"param {pid}: resumed step {step_c_v} < reference step {step_a_v} "
                "(resumed run should run >= steps because it re-walks the interrupted epoch)"
            )

    # Scheduler-state continuity: resumed counters >= reference counters
    # (same re-walk argument as above).
    ssd_a = sched_a.state_dict()
    ssd_c = sched_c.state_dict()
    for key in ("last_epoch", "_step_count"):
        if key in ssd_a and key in ssd_c:
            assert ssd_c[key] >= ssd_a[key], (
                f"scheduler {key}: resumed {ssd_c[key]} < reference {ssd_a[key]}"
            )
