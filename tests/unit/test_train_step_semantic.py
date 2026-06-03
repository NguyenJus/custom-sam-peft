"""Semantic train_step branch: assemble-then-loss path (Phase C, Task C3)."""

from __future__ import annotations

from typing import Any

import torch

from custom_sam_peft.config.schema import (
    DataConfig,
    DataSplit,
    MultiplexConfig,
    PEFTConfig,
    RunConfig,
    SemanticDataConfig,
    TrainConfig,
    TrainHyperparams,
)
from custom_sam_peft.data.base import Instance, SemanticTarget, TextPrompts
from custom_sam_peft.models.sam3 import Sam3Wrapper
from custom_sam_peft.train.loop import train_step
from tests.fixtures.tiny_sam3_stub import TinySam3Stub


def _make_wrapper() -> Sam3Wrapper:
    return Sam3Wrapper(TinySam3Stub(num_queries=4, mask_size=8), mask_size=8)


def _make_semantic_cfg(*, classes_per_forward: int = 16, **train_overrides: Any) -> TrainConfig:
    train_kwargs: dict[str, Any] = {"epochs": 1, "grad_accum_steps": 1}
    train_kwargs.update(train_overrides)
    return TrainConfig(
        run=RunConfig(name="t", output_dir="./runs", seed=0),
        task="semantic",
        data=DataConfig(
            format="mask_png",
            train=DataSplit(annotations="train", images="train_imgs"),
            val=DataSplit(annotations="val", images="val_imgs"),
            semantic=SemanticDataConfig(class_map="cm.json"),
        ),
        peft=PEFTConfig(method="lora", scope="vision"),
        train=TrainHyperparams(
            multiplex=MultiplexConfig(classes_per_forward=classes_per_forward),
            **train_kwargs,
        ),
    )


def _instance(class_id: int) -> Instance:
    return Instance(
        mask=torch.zeros(8, 8, dtype=torch.bool),
        class_id=class_id,
        box=torch.tensor([1.0, 1.0, 5.0, 5.0]),
    )


def test_train_step_semantic_assemble_then_loss() -> None:
    cfg = _make_semantic_cfg()
    model = _make_wrapper()
    B, K, H, W = 2, 3, 16, 16
    batch = {
        "images": torch.zeros(B, 3, H, W),
        "image_ids": ["a", "b"],
        "prompts": [TextPrompts(["road", "tree", "car"]) for _ in range(B)],
        "instances": [[], []],
        "semantic": [
            SemanticTarget(torch.randint(0, K + 1, (H, W), dtype=torch.int64), 255)
            for _ in range(B)
        ],
    }
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda s: 1.0)
    r = train_step(
        model,
        batch,
        optimizer,
        scheduler,
        cfg,
        class_names=["road", "tree", "car"],
        global_step=0,
        nan_streak=0,
    )
    assert set(r.losses.keys()) == {"ce", "region", "total"}
    assert not r.skipped
    assert r.n_classes == 3
    assert r.images_processed == B
    assert r.grad_norm is not None


def test_train_step_semantic_empty_classes_skips() -> None:
    cfg = _make_semantic_cfg()
    model = _make_wrapper()
    B, H, W = 2, 16, 16
    batch = {
        "images": torch.zeros(B, 3, H, W),
        "image_ids": ["a", "b"],
        "prompts": [TextPrompts([]) for _ in range(B)],
        "instances": [[], []],
        "semantic": [SemanticTarget(torch.zeros(H, W, dtype=torch.int64), 255) for _ in range(B)],
    }
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda s: 1.0)
    r = train_step(
        model,
        batch,
        optimizer,
        scheduler,
        cfg,
        class_names=[],
        global_step=0,
        nan_streak=2,
    )
    assert set(r.losses.keys()) == {"ce", "region", "total"}
    assert r.skipped
    assert r.nan_streak == 2


def test_instance_path_still_returns_instance_keys() -> None:
    """Guard: the instance branch is unaffected by the semantic early-return."""
    from custom_sam_peft.config.schema import DataConfig as _DC

    cfg = TrainConfig(
        run=RunConfig(name="t", output_dir="./runs", seed=0),
        data=_DC(
            format="coco",
            train=DataSplit(annotations="a.json", images="i"),
            val=DataSplit(annotations="a.json", images="i"),
        ),
        peft=PEFTConfig(method="lora", scope="vision"),
        train=TrainHyperparams(
            multiplex=MultiplexConfig(classes_per_forward=16), epochs=1, grad_accum_steps=1
        ),
    )
    model = _make_wrapper()
    batch = {
        "images": torch.zeros(1, 3, 8, 8),
        "image_ids": ["0"],
        "prompts": [TextPrompts(classes=["A"])],
        "instances": [[_instance(0)]],
    }
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda s: 1.0)
    r = train_step(
        model, batch, optimizer, scheduler, cfg, class_names=["A"], global_step=0, nan_streak=0
    )
    assert set(r.losses.keys()) == {"mask", "box", "obj", "presence", "total"}
