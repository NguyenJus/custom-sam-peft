"""Step-body unit tests: class loop, NaN policy."""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import nn

from custom_sam_peft.config.schema import (
    DataConfig,
    DataSplit,
    MultiplexConfig,
    PEFTConfig,
    RunConfig,
    TrainConfig,
    TrainHyperparams,
)
from custom_sam_peft.data.base import Instance, TextPrompts
from custom_sam_peft.models.sam3 import Sam3Wrapper
from custom_sam_peft.train.loop import train_step
from tests.fixtures.tiny_sam3_stub import TinySam3Stub


def _make_cfg(*, classes_per_forward: int = 16, **train_overrides: Any) -> TrainConfig:
    train_kwargs: dict[str, Any] = {"epochs": 1, "grad_accum_steps": 1}
    train_kwargs.update(train_overrides)
    return TrainConfig(
        run=RunConfig(name="t", output_dir="./runs", seed=0),
        data=DataConfig(
            format="coco",
            train=DataSplit(annotations="a.json", images="i"),
            val=DataSplit(annotations="a.json", images="i"),
        ),
        peft=PEFTConfig(method="lora", scope="vision"),
        train=TrainHyperparams(
            multiplex=MultiplexConfig(classes_per_forward=classes_per_forward),
            **train_kwargs,
        ),
    )


def _make_wrapper() -> Sam3Wrapper:
    return Sam3Wrapper(TinySam3Stub(num_queries=4, mask_size=8), mask_size=8)


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


def test_train_step_class_loop_visits_union(monkeypatch: pytest.MonkeyPatch) -> None:
    """For a 2-image batch with classes {A,B} and {A}, the wrapper is called once
    with the full union sorted alphabetically as a single multiplex group
    (default classes_per_forward=16 => G=1)."""
    cfg = _make_cfg()  # default classes_per_forward=16
    wrapper = _make_wrapper()

    nn.init.normal_(wrapper.model.dummy)  # type: ignore[arg-type]
    calls: list[list[str]] = []
    real_forward = wrapper.forward

    def spy(images: torch.Tensor, prompts: list[Any], support: Any = None) -> Any:
        # Record the sorted class list from the first prompt (all prompts share the same list).
        calls.append(list(prompts[0].classes))
        return real_forward(images, prompts, support=support)

    monkeypatch.setattr(wrapper, "forward", spy)

    batch = _batch(
        prompts=[["A", "B"], ["A"]],
        instances=[[_instance(0), _instance(1)], [_instance(0)]],
    )
    optimizer = torch.optim.AdamW([p for p in wrapper.parameters() if p.requires_grad], lr=1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda s: 1.0)
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
    # With classes_per_forward=16, both A and B fit in one group => one forward call.
    assert len(calls) == 1
    # That single call should contain all classes from the union (sorted).
    assert sorted(calls[0]) == ["A", "B"]
    assert not result.skipped


def test_train_step_nan_in_one_class_does_not_count_as_skip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """At classes_per_forward=1, NaN in the first group (class A) does not skip
    the step because class B's group is finite.  This exercises the per-group NaN
    policy: skip only when EVERY group is non-finite."""
    cfg = _make_cfg(classes_per_forward=1)  # K=1 per group → one group per class
    wrapper = _make_wrapper()

    class_call = {"count": 0}

    def spy(images: torch.Tensor, prompts: list[Any], support: Any = None) -> Any:
        class_call["count"] += 1
        out = TinySam3Stub(num_queries=4, mask_size=8).forward(images, prompts)
        if class_call["count"] == 1:
            out["pred_masks"] = out["pred_masks"] * float("nan")
        return out

    monkeypatch.setattr(wrapper, "forward", spy)
    batch = _batch(prompts=[["A", "B"]], instances=[[_instance(0), _instance(1)]])
    optimizer = torch.optim.AdamW([p for p in wrapper.parameters() if p.requires_grad], lr=1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda s: 1.0)
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

    def spy(images: torch.Tensor, prompts: list[Any], support: Any = None) -> Any:
        out = TinySam3Stub(num_queries=4, mask_size=8).forward(images, prompts)
        out["pred_masks"] = out["pred_masks"] * float("nan")
        return out

    monkeypatch.setattr(wrapper, "forward", spy)
    batch = _batch(prompts=[["A"]], instances=[[_instance(0)]])
    optimizer = torch.optim.AdamW([p for p in wrapper.parameters() if p.requires_grad], lr=1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda s: 1.0)
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

    def spy(images: torch.Tensor, prompts: list[Any], support: Any = None) -> Any:
        out = TinySam3Stub(num_queries=4, mask_size=8).forward(images, prompts)
        out["pred_masks"] = out["pred_masks"] * float("nan")
        return out

    monkeypatch.setattr(wrapper, "forward", spy)
    batch = _batch(prompts=[["A"]], instances=[[_instance(0)]])
    optimizer = torch.optim.AdamW([p for p in wrapper.parameters() if p.requires_grad], lr=1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda s: 1.0)
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
    caplog: pytest.LogCaptureFixture,
) -> None:
    cfg = _make_cfg()
    wrapper = _make_wrapper()
    batch = _batch(prompts=[[], []], instances=[[], []])
    optimizer = torch.optim.AdamW([p for p in wrapper.parameters() if p.requires_grad], lr=1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda s: 1.0)
    result = train_step(
        wrapper, batch, optimizer, scheduler, cfg, class_names=[], global_step=0, nan_streak=4
    )
    assert result.skipped is True
    assert result.nan_streak == 4
