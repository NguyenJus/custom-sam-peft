"""Regression guard: at classes_per_forward=1, train_step produces one model
call per class in sorted order. Locked decision §10 R3.

At K=1, _chunked(classes_in_batch, 1) yields single-class groups in sorted
class order.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

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


def _make_cfg_k1(**train_overrides: Any) -> TrainConfig:
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
            multiplex=MultiplexConfig(classes_per_forward=1),
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


def _make_optimizer(wrapper: Sam3Wrapper) -> torch.optim.Optimizer:
    return torch.optim.AdamW([p for p in wrapper.parameters() if p.requires_grad], lr=1e-4)


def _make_scheduler(opt: torch.optim.Optimizer) -> torch.optim.lr_scheduler.LRScheduler:
    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda s: 1.0)


def test_legacy_k1_each_model_call_has_one_class(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """At classes_per_forward=1, each group has exactly one class → one call per class."""
    K_total = 3
    class_names = ["A", "B", "C"]
    batch = _batch(
        [["A", "B", "C"]],
        [[_instance(0), _instance(1), _instance(2)]],
    )

    wrapper = _make_wrapper()
    call_class_lists: list[list[str]] = []
    real_forward = wrapper.forward

    def spy(images: torch.Tensor, prompts: list[Any], support: Any = None) -> Any:
        call_class_lists.append(list(prompts[0].classes))
        return real_forward(images, prompts, support=support)

    monkeypatch.setattr(wrapper, "forward", spy)

    cfg = _make_cfg_k1()
    opt = _make_optimizer(wrapper)
    sched = _make_scheduler(opt)

    train_step(
        wrapper, batch, opt, sched, cfg, class_names=class_names, global_step=0, nan_streak=0
    )

    # classes_per_forward=1 → G=K_total=3 groups, each with 1 class.
    assert len(call_class_lists) == K_total, (
        f"Expected {K_total} model calls, got {len(call_class_lists)}"
    )
    # Classes should be visited in sorted order.
    for i, expected_class in enumerate(sorted(class_names)):
        assert call_class_lists[i] == [expected_class], (
            f"Group {i}: expected [{expected_class}], got {call_class_lists[i]}"
        )


def test_legacy_k1_n_classes_is_K_total() -> None:
    """At classes_per_forward=1, n_classes == K_total (not G, which also == K_total at K=1)."""
    K_total = 3
    class_names = ["A", "B", "C"]
    batch = _batch([["A", "B", "C"]], [[_instance(0), _instance(1), _instance(2)]])

    wrapper = _make_wrapper()
    opt = _make_optimizer(wrapper)
    sched = _make_scheduler(opt)

    cfg = _make_cfg_k1()
    result = train_step(
        wrapper, batch, opt, sched, cfg, class_names=class_names, global_step=0, nan_streak=0
    )
    assert result.n_classes == K_total


def test_legacy_k1_nan_in_one_group_not_skip(monkeypatch: pytest.MonkeyPatch) -> None:
    """At K=1, NaN in one single-class group → step not skipped (other groups finite)."""
    class_names = ["A", "B"]
    batch = _batch([["A", "B"]], [[_instance(0), _instance(1)]])

    wrapper = _make_wrapper()
    call_count = [0]

    def spy(images: torch.Tensor, prompts: list[Any], support: Any = None) -> Any:
        call_count[0] += 1
        out = TinySam3Stub(num_queries=4, mask_size=8).forward(images, prompts)
        # First call (class A) returns NaN.
        if call_count[0] == 1:
            out["pred_masks"] = out["pred_masks"] * float("nan")
        return out

    monkeypatch.setattr(wrapper, "forward", spy)

    cfg = _make_cfg_k1(nan_abort_after=99)
    opt = _make_optimizer(wrapper)
    sched = _make_scheduler(opt)

    result = train_step(
        wrapper, batch, opt, sched, cfg, class_names=class_names, global_step=0, nan_streak=0
    )
    # B class was finite → not skipped.
    assert not result.skipped
    assert result.nan_streak == 0
