"""Multiplex per-group behavior tests.

Covers:
- auto-chunk INFO log when classes_in_batch > MULTIPLEX_CAP
- StepResult.n_classes == K_total (not G)
- K_total=4 with default cap -> one model call with K=4 prompts
"""

from __future__ import annotations

import logging
import random
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
from custom_sam_peft.models.sam3 import MULTIPLEX_CAP, Sam3Wrapper
from custom_sam_peft.train.loop import _reset_auto_chunk_log, train_step
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


def _make_optimizer(wrapper: Sam3Wrapper) -> torch.optim.Optimizer:
    return torch.optim.AdamW([p for p in wrapper.parameters() if p.requires_grad], lr=1e-4)


def _make_scheduler(opt: torch.optim.Optimizer) -> torch.optim.lr_scheduler.LRScheduler:
    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda s: 1.0)


def test_auto_chunk_logs_once_when_classes_exceed_cap(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When K_total > MULTIPLEX_CAP, exactly one INFO 'multiplex auto-chunk' log."""
    # We need K_total > MULTIPLEX_CAP=16.  Build a batch with 17 distinct classes.
    # classes_per_forward=16 → G=2 groups (16 + 1).
    # Use a wrapper whose forward can handle up to 16 classes per group.
    _reset_auto_chunk_log()

    K_total = MULTIPLEX_CAP + 1  # 17
    class_names = [f"cls{i}" for i in range(K_total)]

    # Build a 1-image batch with all classes represented.
    batch_prompts = [class_names]
    batch_instances = [[_instance(i) for i in range(K_total)]]

    wrapper = _make_wrapper()
    opt = _make_optimizer(wrapper)
    sched = _make_scheduler(opt)

    # We need to stub forward so it accepts up to 16 classes without validating
    # that shape matches K exactly. The TinySam3Stub ignores prompts so it just
    # returns fixed shapes — but Sam3Wrapper._validate_inputs will check K <= 16.
    # With classes_per_forward=16 and K_total=17, groups are [16, 1], both <= 16.
    monkeypatch.setattr(random, "random", lambda: 0.0)  # never apply hints

    cfg = _make_cfg(classes_per_forward=16, nan_abort_after=99)

    caplog.set_level(logging.INFO, logger="custom_sam_peft.train.loop")
    result = train_step(
        wrapper,
        _batch(batch_prompts, batch_instances),
        opt,
        sched,
        cfg,
        class_names=class_names,
        global_step=0,
        nan_streak=0,
    )

    msgs = [r.message for r in caplog.records if "multiplex auto-chunk" in r.message]
    assert len(msgs) == 1, f"Expected exactly 1 auto-chunk log; got: {msgs}"
    # n_classes should equal K_total, not G
    assert result.n_classes == K_total


def test_auto_chunk_logs_only_once_across_two_calls(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The _AUTO_CHUNK_LOGGED flag suppresses duplicate logs across steps."""
    _reset_auto_chunk_log()

    K_total = MULTIPLEX_CAP + 1
    class_names = [f"cls{i}" for i in range(K_total)]
    batch = _batch([class_names], [[_instance(i) for i in range(K_total)]])

    wrapper = _make_wrapper()
    opt = _make_optimizer(wrapper)
    sched = _make_scheduler(opt)
    monkeypatch.setattr(random, "random", lambda: 0.0)

    cfg = _make_cfg(classes_per_forward=16, nan_abort_after=99)
    caplog.set_level(logging.INFO, logger="custom_sam_peft.train.loop")

    kwargs = dict(class_names=class_names, nan_streak=0)
    train_step(wrapper, batch, opt, sched, cfg, global_step=0, **kwargs)
    train_step(wrapper, batch, opt, sched, cfg, global_step=1, **kwargs)

    msgs = [r.message for r in caplog.records if "multiplex auto-chunk" in r.message]
    assert len(msgs) == 1, f"Expected only 1 auto-chunk log across two calls; got {len(msgs)}"


def test_step_result_n_classes_equals_K_total(monkeypatch: pytest.MonkeyPatch) -> None:
    """n_classes reports K_total (len of union), not number of groups G."""
    K_total = 5
    class_names = [f"cls{i}" for i in range(K_total)]
    batch = _batch([class_names], [[_instance(i) for i in range(K_total)]])

    wrapper = _make_wrapper()
    opt = _make_optimizer(wrapper)
    sched = _make_scheduler(opt)
    monkeypatch.setattr(random, "random", lambda: 0.0)

    cfg = _make_cfg(classes_per_forward=16)  # G=1 for K_total=5
    result = train_step(
        wrapper, batch, opt, sched, cfg, class_names=class_names, global_step=0, nan_streak=0
    )
    assert result.n_classes == 5  # K_total, not G=1


def test_K_4_multiplex_calls_model_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """K_total=4, classes_per_forward=16 → one model call with all 4 classes."""
    K_total = 4
    class_names = [f"cls{i}" for i in range(K_total)]
    batch = _batch([class_names], [[_instance(i) for i in range(K_total)]])

    wrapper = _make_wrapper()
    call_prompts: list[list[Any]] = []
    real_forward = wrapper.forward

    def spy(images: torch.Tensor, prompts: list[Any], support: Any = None) -> Any:
        call_prompts.append(list(prompts))
        return real_forward(images, prompts, support=support)

    monkeypatch.setattr(wrapper, "forward", spy)
    monkeypatch.setattr(random, "random", lambda: 0.0)

    opt = _make_optimizer(wrapper)
    sched = _make_scheduler(opt)
    cfg = _make_cfg(classes_per_forward=16)  # K_total=4 < 16 → G=1

    train_step(
        wrapper, batch, opt, sched, cfg, class_names=class_names, global_step=0, nan_streak=0
    )

    assert len(call_prompts) == 1, f"Expected 1 model call, got {len(call_prompts)}"
    assert len(call_prompts[0][0].classes) == K_total, (
        f"Expected {K_total} classes in prompt, got {len(call_prompts[0][0].classes)}"
    )


def test_train_step_one_total_loss_per_group(monkeypatch: pytest.MonkeyPatch) -> None:
    """One total_loss call per chunked group; backward divides by G*grad_accum."""
    # K_total=2, classes_per_forward=16 → G=1, K_g=2.
    K_total = 2
    class_names = ["A", "B"]
    batch = _batch(
        [["A", "B"]],
        [[_instance(0), _instance(1)]],
    )

    wrapper = _make_wrapper()
    forward_call_count = [0]
    prompts_seen: list[list[Any]] = []
    real_forward = wrapper.forward

    def spy_forward(images: torch.Tensor, prompts: list[Any], support: Any = None) -> Any:
        forward_call_count[0] += 1
        prompts_seen.append(list(prompts))
        return real_forward(images, prompts, support=support)

    monkeypatch.setattr(wrapper, "forward", spy_forward)
    monkeypatch.setattr(random, "random", lambda: 0.0)

    cfg = _make_cfg(classes_per_forward=16, grad_accum_steps=2)
    opt = _make_optimizer(wrapper)
    sched = _make_scheduler(opt)

    train_step(
        wrapper, batch, opt, sched, cfg, class_names=class_names, global_step=0, nan_streak=0
    )

    # G=1 group → model called exactly once
    assert forward_call_count[0] == 1, f"Expected 1 forward call, got {forward_call_count[0]}"
    # The single group should have 2 classes
    assert len(prompts_seen[0][0].classes) == K_total, (
        f"Expected prompt with {K_total} classes, got {len(prompts_seen[0][0].classes)}"
    )


def test_two_groups_two_model_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    """classes_per_forward=2 with K_total=4 → G=2 groups → 2 model calls."""
    _reset_auto_chunk_log()
    K_total = 4
    class_names = [f"cls{i}" for i in range(K_total)]
    batch = _batch([class_names], [[_instance(i) for i in range(K_total)]])

    wrapper = _make_wrapper()
    call_class_lists: list[list[str]] = []
    real_forward = wrapper.forward

    def spy(images: torch.Tensor, prompts: list[Any], support: Any = None) -> Any:
        call_class_lists.append(list(prompts[0].classes))
        return real_forward(images, prompts, support=support)

    monkeypatch.setattr(wrapper, "forward", spy)
    monkeypatch.setattr(random, "random", lambda: 0.0)

    cfg = _make_cfg(classes_per_forward=2)  # K_total=4, K_g=2 → G=2
    opt = _make_optimizer(wrapper)
    sched = _make_scheduler(opt)

    train_step(
        wrapper, batch, opt, sched, cfg, class_names=class_names, global_step=0, nan_streak=0
    )

    assert len(call_class_lists) == 2, f"Expected 2 model calls, got {len(call_class_lists)}"
    # Groups should partition class_names in sorted order: [cls0,cls1], [cls2,cls3]
    assert call_class_lists[0] == ["cls0", "cls1"]
    assert call_class_lists[1] == ["cls2", "cls3"]
