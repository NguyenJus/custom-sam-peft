"""Trainer.__init__ guards: qlora optimizer coercion."""

from __future__ import annotations

from custom_sam_peft.config.schema import (
    DataConfig,
    DataSplit,
    PEFTConfig,
    RunConfig,
    TrainConfig,
    TrainHyperparams,
)
from custom_sam_peft.train.trainer import _resolve_optimizer_name


def _cfg(peft_method: str = "lora", optimizer: str = "auto") -> TrainConfig:
    return TrainConfig(
        run=RunConfig(name="t", output_dir="./runs", seed=0),
        data=DataConfig(
            format="coco",
            train=DataSplit(annotations="a.json", images="i"),
            val=DataSplit(annotations="a.json", images="i"),
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
