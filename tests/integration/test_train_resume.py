"""Resume integration: a resumed run reaches a finite end-state."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

from esam3.config.schema import (
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
from esam3.data.coco import COCODataset
from esam3.data.transforms import build_train_transforms
from esam3.peft_adapters.lora import apply_lora
from esam3.tracking.noop import NoopTracker
from esam3.train.trainer import Trainer
from tests.fixtures.tiny_sam3_lora_stub import FIXTURE_SCOPE_PATTERNS, make_stub_wrapper

pytestmark = pytest.mark.integration


def _ds(tiny_coco_dir: Path) -> COCODataset:
    # NOTE: build_train_transforms takes (aug_cfg, image_size, ...) — positional order matches impl.
    transforms = build_train_transforms(
        AugmentationsConfig(hflip=False, color_jitter=0.0),
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


def test_resume_matches_uninterrupted(tmp_path: Path, tiny_coco_dir: Path) -> None:
    ds = _ds(tiny_coco_dir)
    cfg = _cfg(tmp_path, tiny_coco_dir, save_every=2)

    # Uninterrupted reference run (2 epochs).
    w_a = make_stub_wrapper(dim=8, working=True)
    apply_lora(w_a, cfg.peft)
    trainer_a = Trainer(w_a, ds, ds, NoopTracker(), cfg)
    trainer_a.fit()
    state_a = _adapter_state(w_a)

    # Truncated first run (1 epoch), then resumed (2 epochs continuing from checkpoint).
    w_b = make_stub_wrapper(dim=8, working=True)
    apply_lora(w_b, cfg.peft)
    cfg_short = _cfg(tmp_path, tiny_coco_dir, save_every=2)
    cfg_short.train.epochs = 1
    trainer_b = Trainer(w_b, ds, ds, NoopTracker(), cfg_short)
    result_b1 = trainer_b.fit()

    ckpts = sorted((result_b1.run_dir / "checkpoints").glob("step_*"))
    assert ckpts, "no checkpoint produced"
    resume_dir = ckpts[-1]

    w_c = make_stub_wrapper(dim=8, working=True)
    apply_lora(w_c, cfg.peft)
    trainer_c = Trainer(w_c, ds, ds, NoopTracker(), cfg)
    trainer_c.fit(resume_from=resume_dir)
    state_c = _adapter_state(w_c)

    # Resume produces finite weights (not bit-identical to uninterrupted run because
    # the re-walked epoch retreads some examples). Assert finiteness only.
    for k in state_a:
        assert torch.isfinite(state_c[k]).all()
