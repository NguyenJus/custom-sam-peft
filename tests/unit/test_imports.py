"""Smoke test that every public custom_sam_peft submodule imports without raising."""

from __future__ import annotations

import importlib

MODULES = [
    "custom_sam_peft",
    "custom_sam_peft._registry",
    "custom_sam_peft.config",
    "custom_sam_peft.config.schema",
    "custom_sam_peft.config.loader",
    "custom_sam_peft.data",
    "custom_sam_peft.data.base",
    "custom_sam_peft.data.coco",
    "custom_sam_peft.data.hf",
    "custom_sam_peft.data.transforms",
    "custom_sam_peft.data.collate",
    "custom_sam_peft.models",
    "custom_sam_peft.models.sam3",
    "custom_sam_peft.models.losses",
    "custom_sam_peft.peft_adapters",
    "custom_sam_peft.peft_adapters.lora",
    "custom_sam_peft.peft_adapters.qlora",
    "custom_sam_peft.train",
    "custom_sam_peft.train.trainer",
    "custom_sam_peft.train.loop",
    "custom_sam_peft.train.checkpoint",
    "custom_sam_peft.eval",
    "custom_sam_peft.eval.metrics",
    "custom_sam_peft.eval.evaluator",
    "custom_sam_peft.tracking",
    "custom_sam_peft.tracking.base",
    "custom_sam_peft.tracking.tensorboard",
    "custom_sam_peft.tracking.wandb",
    "custom_sam_peft.tracking.noop",
    "custom_sam_peft.cli",
    "custom_sam_peft.cli.main",
    "custom_sam_peft.cli.train_cmd",
    "custom_sam_peft.cli.eval_cmd",
    "custom_sam_peft.cli.export_cmd",
    "custom_sam_peft.cli.init_cmd",
    "custom_sam_peft.cli.doctor_cmd",
]


def test_all_modules_import() -> None:
    for name in MODULES:
        importlib.import_module(name)
