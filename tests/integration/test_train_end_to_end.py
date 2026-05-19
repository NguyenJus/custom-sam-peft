"""End-to-end integration: Trainer.fit() with tiny_coco + LoRA stub."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from esam3.config.schema import (
    AugmentationsConfig,
    DataConfig,
    DataSplit,
    PEFTConfig,
    RunConfig,
    TextPromptConfig,
    TrackingConfig,
    TrainConfig,
    TrainHyperparams,
)
from esam3.data.coco import COCODataset
from esam3.data.transforms import build_eval_transforms, build_train_transforms
from esam3.peft_adapters.lora import apply_lora
from esam3.tracking import build_tracker
from esam3.train.trainer import Trainer
from tests.fixtures.tiny_sam3_lora_stub import FIXTURE_SCOPE_PATTERNS, make_stub_wrapper

pytestmark = pytest.mark.integration


def _ds(tiny_coco_dir: Path, pipeline: str) -> COCODataset:
    from esam3.config.schema import NormalizeConfig

    if pipeline == "train":
        transforms = build_train_transforms(
            AugmentationsConfig(hflip=False, color_jitter=0.0),
            32,
            model_name="facebook/sam3.1",
            normalize=NormalizeConfig(),
        )
    else:
        transforms = build_eval_transforms(
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


@pytest.mark.parametrize("backend", ["none", "tensorboard"])
def test_fit_end_to_end_on_tiny_coco(backend: str, tmp_path: Path, tiny_coco_dir: Path) -> None:
    if backend == "tensorboard":
        pytest.importorskip("tensorboard")
    ds_train = _ds(tiny_coco_dir, "train")
    ds_val = _ds(tiny_coco_dir, "eval")
    wrapper = make_stub_wrapper(dim=8, working=True)

    cfg = TrainConfig(
        run=RunConfig(name="e2e", output_dir=str(tmp_path), seed=0),
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
            epochs=1,
            batch_size=1,
            grad_accum_steps=1,
            save_every=2,
            log_every=1,
            warmup_steps=0,
            num_workers=0,
        ),
        tracking=TrackingConfig(backend=backend),  # type: ignore[arg-type]
    )
    apply_lora(wrapper, cfg.peft)
    run_dir = tmp_path / f"{cfg.run.name}-test"
    tracker = build_tracker(cfg)
    trainer = Trainer(wrapper, ds_train, ds_val, tracker, cfg)
    result = trainer.fit(run_dir=run_dir)

    assert result.run_dir.exists()
    assert (result.run_dir / "adapter" / "adapter_config.json").exists()
    payload = json.loads((result.run_dir / "metrics.json").read_text())
    assert payload["global_step"] >= 1
    ckpts = list((result.run_dir / "checkpoints").glob("step_*"))
    assert ckpts, "expected at least one step_* checkpoint dir"
    assert (ckpts[0] / "training_state.pt").exists()
    assert (ckpts[0] / "adapter").exists()
    if backend == "tensorboard":
        events = list(result.run_dir.glob("events.out.tfevents.*"))
        assert events, "tensorboard backend should write at least one event file"
