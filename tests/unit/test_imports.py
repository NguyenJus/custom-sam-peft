"""Smoke test that every public esam3 submodule imports without raising."""

from __future__ import annotations

import importlib

MODULES = [
    "esam3",
    "esam3._registry",
    "esam3.config",
    "esam3.config.schema",
    "esam3.config.loader",
    "esam3.data",
    "esam3.data.base",
    "esam3.data.coco",
    "esam3.data.hf",
    "esam3.data.transforms",
    "esam3.data.collate",
    "esam3.models",
    "esam3.models.sam3",
    "esam3.models.losses",
    "esam3.peft_adapters",
    "esam3.peft_adapters.lora",
    "esam3.peft_adapters.qlora",
    "esam3.train",
    "esam3.train.trainer",
    "esam3.train.loop",
    "esam3.train.checkpoint",
    "esam3.eval",
    "esam3.eval.metrics",
    "esam3.eval.evaluator",
    "esam3.tracking",
    "esam3.tracking.base",
    "esam3.tracking.tensorboard",
    "esam3.tracking.wandb",
    "esam3.tracking.noop",
    "esam3.cli",
    "esam3.cli.main",
    "esam3.cli.train_cmd",
    "esam3.cli.eval_cmd",
    "esam3.cli.export_cmd",
    "esam3.cli.init_cmd",
    "esam3.cli.doctor_cmd",
]


def test_all_modules_import() -> None:
    for name in MODULES:
        importlib.import_module(name)
