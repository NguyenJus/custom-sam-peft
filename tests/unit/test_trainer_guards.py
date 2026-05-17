"""Trainer.__init__ guards: bbox rejection + qlora optimizer coercion."""

from __future__ import annotations

import pytest

from esam3.config.schema import (
    DataConfig,
    DataSplit,
    PEFTConfig,
    RunConfig,
    TrainConfig,
    TrainHyperparams,
)
from esam3.tracking.noop import NoopTracker
from esam3.train.trainer import Trainer, _resolve_optimizer_name


def _cfg(
    prompt_mode: str = "text", peft_method: str = "lora", optimizer: str = "auto"
) -> TrainConfig:
    return TrainConfig(
        run=RunConfig(name="t", output_dir="./runs", seed=0),
        data=DataConfig(
            format="coco",
            train=DataSplit(annotations="a.json", images="i"),
            val=DataSplit(annotations="a.json", images="i"),
            prompt_mode=prompt_mode,
        ),
        peft=PEFTConfig(method=peft_method, scope="vision"),
        train=TrainHyperparams(epochs=1, optimizer=optimizer),
    )


def test_resolve_optimizer_auto_with_qlora() -> None:
    cfg = _cfg(peft_method="qlora", optimizer="auto")
    assert _resolve_optimizer_name(cfg) == "adamw8bit"


def test_resolve_optimizer_auto_with_lora() -> None:
    cfg = _cfg(peft_method="lora", optimizer="auto")
    assert _resolve_optimizer_name(cfg) == "adamw"


def test_resolve_optimizer_explicit_value_honored() -> None:
    cfg = _cfg(peft_method="qlora", optimizer="adamw")
    assert _resolve_optimizer_name(cfg) == "adamw"


def test_trainer_rejects_bbox_prompt_mode(
    stub_model: object, noop_tracker: NoopTracker, tiny_coco_dataset: object
) -> None:
    cfg = _cfg(prompt_mode="bbox")
    with pytest.raises(ValueError, match="prompt_mode='bbox'"):
        Trainer(stub_model, tiny_coco_dataset, tiny_coco_dataset, noop_tracker, cfg)
