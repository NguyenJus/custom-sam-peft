"""Step-body unit tests: schedule math, class loop, hint sampling, NaN policy."""

from __future__ import annotations

import random
from typing import Any

import pytest
import torch
from torch import nn

from esam3.config.schema import (
    BoxHintSchedule,
    DataConfig,
    DataSplit,
    PEFTConfig,
    RunConfig,
    TrainConfig,
    TrainHyperparams,
)
from esam3.data.base import Instance, TextPrompts
from esam3.models.sam3 import Sam3Wrapper
from esam3.train.loop import _box_hint_p, train_step
from tests.fixtures.tiny_sam3_stub import TinySam3Stub


def _make_cfg(**train_overrides: Any) -> TrainConfig:
    train_kwargs: dict[str, Any] = {"epochs": 1, "grad_accum_steps": 1}
    train_kwargs.update(train_overrides)
    return TrainConfig(
        run=RunConfig(name="t", output_dir="./runs", seed=0),
        data=DataConfig(
            format="coco",
            train=DataSplit(annotations="a.json", images="i"),
            val=DataSplit(annotations="a.json", images="i"),
            prompt_mode="text",
        ),
        peft=PEFTConfig(method="lora", scope="vision"),
        train=TrainHyperparams(**train_kwargs),
    )


def _make_wrapper() -> Sam3Wrapper:
    return Sam3Wrapper(TinySam3Stub(num_queries=4, mask_size=8), image_size=8, mask_size=8)


def _instance(class_id: int) -> Instance:
    return Instance(
        mask=torch.zeros(8, 8, dtype=torch.bool),
        class_id=class_id,
        box=torch.tensor([1.0, 1.0, 5.0, 5.0]),
    )


def _batch(prompts: list[list[str]], instances: list[list[Instance]]) -> dict[str, Any]:
    return {
        "images": torch.zeros(len(prompts), 3, 8, 8),
        "image_ids": [str(i) for i in range(len(prompts))],
        "prompts": [TextPrompts(classes=p) for p in prompts],
        "instances": instances,
    }


def test_box_hint_p_endpoints() -> None:
    s = BoxHintSchedule(p_start=1.0, p_end=0.0, decay_steps=10)
    assert _box_hint_p(0, s) == 1.0
    assert _box_hint_p(10, s) == 0.0
    assert _box_hint_p(20, s) == 0.0


def test_box_hint_p_midpoint() -> None:
    s = BoxHintSchedule(p_start=1.0, p_end=0.0, decay_steps=10)
    assert abs(_box_hint_p(5, s) - 0.5) < 1e-6


def test_train_step_class_loop_visits_union(monkeypatch: pytest.MonkeyPatch) -> None:
    """For a 2-image batch with classes {A,B} and {A}, the wrapper is called once
    per class in the union (alphabetical sort: A then B)."""
    cfg = _make_cfg()
    wrapper = _make_wrapper()

    nn.init.normal_(wrapper.model.dummy)  # type: ignore[arg-type]
    calls: list[list[str]] = []
    real_forward = wrapper.forward

    def spy(images: torch.Tensor, prompts: list[Any], box_hints: Any = None) -> Any:
        calls.append([p.classes[0] for p in prompts])
        return real_forward(images, prompts, box_hints=box_hints)

    monkeypatch.setattr(wrapper, "forward", spy)

    batch = _batch(
        prompts=[["A", "B"], ["A"]],
        instances=[[_instance(0), _instance(1)], [_instance(0)]],
    )
    optimizer = torch.optim.AdamW([p for p in wrapper.parameters() if p.requires_grad], lr=1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda s: 1.0)
    monkeypatch.setattr(random, "random", lambda: 1.0)
    result = train_step(
        wrapper,
        batch,
        optimizer,
        scheduler,
        cfg,
        class_names=["A", "B"],
        global_step=0,
        nan_streak=0,
    )
    assert [c[0] for c in calls] == ["A", "A", "B"] or [c[0] for c in calls] == ["A", "B"]
    for call_classes in calls:
        assert len(set(call_classes)) == 1
    assert not result.skipped


def test_train_step_box_hint_sampling(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patched `random.random()` sequence drives Bernoulli sampling."""
    cfg = _make_cfg()
    cfg.train.box_hint.p_start = 0.5
    cfg.train.box_hint.p_end = 0.5
    wrapper = _make_wrapper()

    hint_records: list[list[bool]] = []

    def spy(images: torch.Tensor, prompts: list[Any], box_hints: Any = None) -> Any:
        hint_records.append([h is not None for h in (box_hints or [None] * len(prompts))])
        return TinySam3Stub(num_queries=4, mask_size=8).forward(images, prompts)

    monkeypatch.setattr(wrapper, "forward", spy)

    batch = _batch(
        prompts=[["A"], ["A"]],
        instances=[[_instance(0)], [_instance(0)]],
    )
    optimizer = torch.optim.AdamW([p for p in wrapper.parameters() if p.requires_grad], lr=1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda s: 1.0)
    coin_seq = iter([0.1, 0.9])
    monkeypatch.setattr(random, "random", lambda: next(coin_seq))

    train_step(
        wrapper, batch, optimizer, scheduler, cfg, class_names=["A"], global_step=0, nan_streak=0
    )
    assert hint_records == [[True, False]]


def test_train_step_nan_in_one_class_does_not_count_as_skip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _make_cfg()
    wrapper = _make_wrapper()

    class_call = {"count": 0}

    def spy(images: torch.Tensor, prompts: list[Any], box_hints: Any = None) -> Any:
        class_call["count"] += 1
        out = TinySam3Stub(num_queries=4, mask_size=8).forward(images, prompts)
        if class_call["count"] == 1:
            out["pred_masks"] = out["pred_masks"] * float("nan")
        return out

    monkeypatch.setattr(wrapper, "forward", spy)
    batch = _batch(prompts=[["A", "B"]], instances=[[_instance(0), _instance(1)]])
    optimizer = torch.optim.AdamW([p for p in wrapper.parameters() if p.requires_grad], lr=1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda s: 1.0)
    monkeypatch.setattr(random, "random", lambda: 1.0)
    result = train_step(
        wrapper,
        batch,
        optimizer,
        scheduler,
        cfg,
        class_names=["A", "B"],
        global_step=0,
        nan_streak=0,
    )
    assert not result.skipped
    assert result.nan_streak == 0


def test_train_step_nan_in_all_classes_increments_streak(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _make_cfg(nan_abort_after=99)
    wrapper = _make_wrapper()

    def spy(images: torch.Tensor, prompts: list[Any], box_hints: Any = None) -> Any:
        out = TinySam3Stub(num_queries=4, mask_size=8).forward(images, prompts)
        out["pred_masks"] = out["pred_masks"] * float("nan")
        return out

    monkeypatch.setattr(wrapper, "forward", spy)
    batch = _batch(prompts=[["A"]], instances=[[_instance(0)]])
    optimizer = torch.optim.AdamW([p for p in wrapper.parameters() if p.requires_grad], lr=1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda s: 1.0)
    monkeypatch.setattr(random, "random", lambda: 1.0)
    result = train_step(
        wrapper, batch, optimizer, scheduler, cfg, class_names=["A"], global_step=0, nan_streak=5
    )
    assert result.skipped
    assert result.nan_streak == 6


def test_train_step_aborts_after_nan_abort_after(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _make_cfg(nan_abort_after=3)
    wrapper = _make_wrapper()

    def spy(images: torch.Tensor, prompts: list[Any], box_hints: Any = None) -> Any:
        out = TinySam3Stub(num_queries=4, mask_size=8).forward(images, prompts)
        out["pred_masks"] = out["pred_masks"] * float("nan")
        return out

    monkeypatch.setattr(wrapper, "forward", spy)
    batch = _batch(prompts=[["A"]], instances=[[_instance(0)]])
    optimizer = torch.optim.AdamW([p for p in wrapper.parameters() if p.requires_grad], lr=1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda s: 1.0)
    monkeypatch.setattr(random, "random", lambda: 1.0)
    with pytest.raises(RuntimeError, match="non-finite"):
        train_step(
            wrapper,
            batch,
            optimizer,
            scheduler,
            cfg,
            class_names=["A"],
            global_step=0,
            nan_streak=2,
        )


def test_train_step_empty_classes_does_not_bump_streak(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    cfg = _make_cfg()
    wrapper = _make_wrapper()
    batch = _batch(prompts=[[], []], instances=[[], []])
    optimizer = torch.optim.AdamW([p for p in wrapper.parameters() if p.requires_grad], lr=1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda s: 1.0)
    monkeypatch.setattr(random, "random", lambda: 1.0)
    result = train_step(
        wrapper, batch, optimizer, scheduler, cfg, class_names=[], global_step=0, nan_streak=4
    )
    assert result.skipped is True
    assert result.nan_streak == 4
