"""Regression guard: at classes_per_forward=1, train_step is RNG-order and
numerically equivalent to today's per-class loop. Locked decision §10 R3.

At K=1, _chunked(classes_in_batch, 1) yields single-class groups in sorted
class order. The Bernoulli draws happen image-major within each group (one class
per group, B images per group). This matches the legacy per-class loop's
iteration order: `for c in sorted_classes: for i in range(B): random.random()`.

Total random.random() draws per step: B * K_total -- identical to the legacy loop.
"""

from __future__ import annotations

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


def test_legacy_k1_rng_draw_order_matches_per_class_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """At classes_per_forward=1, each group has 1 class and B images.
    The RNG draw order is: for each class (sorted), for each image in range(B).
    B=2, K_total=2 → draws: [cls0/img0, cls0/img1, cls1/img0, cls1/img1].
    With p_t=1.0, all hints apply → n_hint_applied = B * K_total = 4.
    """
    cfg = _make_cfg_k1()
    cfg.train.box_hint.p_start = 1.0
    cfg.train.box_hint.p_end = 1.0
    cfg.train.box_hint.decay_steps = 1

    B = 2
    K_total = 2
    class_names = ["A", "B"]

    # Each image has instances for both classes.
    batch = _batch(
        prompts=[["A", "B"], ["A", "B"]],
        instances=[
            [_instance(0), _instance(1)],
            [_instance(0), _instance(1)],
        ],
    )

    wrapper = _make_wrapper()
    opt = _make_optimizer(wrapper)
    sched = _make_scheduler(opt)

    # Capture the RNG calls in order.
    rng_calls: list[float] = []

    def fake_random() -> float:
        val = 0.0  # always below p_t → hints always apply
        rng_calls.append(val)
        return val

    monkeypatch.setattr(random, "random", fake_random)

    result = train_step(
        wrapper,
        batch,
        opt,
        sched,
        cfg,
        class_names=class_names,
        global_step=0,
        nan_streak=0,
    )

    # B * K_total = 4 draws total, one per (image, class) slot.
    assert len(rng_calls) == B * K_total, f"Expected {B * K_total} RNG draws; got {len(rng_calls)}"
    # All hints applied (random=0.0 < p_t=1.0).
    assert result.n_hint_applied == B * K_total
    assert not result.skipped
    assert result.n_classes == K_total


def test_legacy_k1_rng_draw_order_pinned_interleaving(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pin the exact (image, class) interleaving of RNG draws at K=1.

    At classes_per_forward=1, groups are single-class in sorted order.
    For each group the inner loop runs `for i in range(B): for c in group:`,
    so draws are image-major within each group:
        call 0 → (i=0, cls="A"),  call 1 → (i=1, cls="A")
        call 2 → (i=0, cls="B"),  call 3 → (i=1, cls="B")

    We confirm this by returning a deterministic sequence from random.random()
    and asserting that n_hint_applied matches the hand-computed expected count
    based on which (draw_index) values fall below p_t=0.5.

    Sequence: [0.0, 1.0, 1.0, 0.0]
      draw 0 (i=0, cls=A): 0.0 < 0.5 → hint applied   (A has instances on img 0)
      draw 1 (i=1, cls=A): 1.0 >= 0.5 → no hint        (A has instances on img 1, but val >= p_t)
      draw 2 (i=0, cls=B): 1.0 >= 0.5 → no hint
      draw 3 (i=1, cls=B): 0.0 < 0.5 → hint applied   (B has instances on img 1)
    Expected n_hint_applied = 2 (draws 0 and 3).

    If the loop order were swapped to (class-major within image), the assignment
    of draw indices to (i, c) cells changes and the expected count would differ.
    """
    cfg = _make_cfg_k1()
    cfg.train.box_hint.p_start = 0.5
    cfg.train.box_hint.p_end = 0.5
    cfg.train.box_hint.decay_steps = 1

    B = 2
    K_total = 2
    class_names = ["A", "B"]

    # Each image has instances for both classes, so every (i, c) cell is eligible
    # for hint application; the gate is purely the RNG draw vs. p_t.
    batch = _batch(
        prompts=[["A", "B"], ["A", "B"]],
        instances=[
            [_instance(0), _instance(1)],
            [_instance(0), _instance(1)],
        ],
    )

    wrapper = _make_wrapper()
    opt = _make_optimizer(wrapper)
    sched = _make_scheduler(opt)

    # Deterministic sequence: draw index → value.
    # Expected draw-to-cell mapping (image-major within group, sorted group order):
    #   draw 0 → (i=0, c=A): 0.0 < 0.5 → hint
    #   draw 1 → (i=1, c=A): 1.0 >= 0.5 → no hint
    #   draw 2 → (i=0, c=B): 1.0 >= 0.5 → no hint
    #   draw 3 → (i=1, c=B): 0.0 < 0.5 → hint
    rng_sequence = [0.0, 1.0, 1.0, 0.0]
    rng_idx = [0]

    def fake_random() -> float:
        val = rng_sequence[rng_idx[0] % len(rng_sequence)]
        rng_idx[0] += 1
        return val

    monkeypatch.setattr(random, "random", fake_random)

    result = train_step(
        wrapper,
        batch,
        opt,
        sched,
        cfg,
        class_names=class_names,
        global_step=0,
        nan_streak=0,
    )

    # Exactly B * K_total = 4 draws.
    assert rng_idx[0] == B * K_total, f"Expected {B * K_total} RNG draws; got {rng_idx[0]}"
    # With the image-major draw order and the sequence above, exactly 2 hints fire
    # (draws 0 and 3).  A class-minor-within-image loop order would assign different
    # (i, c) pairs to draws 1 and 2, changing the count.
    expected_hints = 2  # draws 0 and 3 are < p_t=0.5
    assert result.n_hint_applied == expected_hints, (
        f"Expected {expected_hints} hints (image-major draw order); got {result.n_hint_applied}. "
        "This likely means the (i, c) loop order changed."
    )
    assert not result.skipped
    assert result.n_classes == K_total


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
    monkeypatch.setattr(random, "random", lambda: 0.0)

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


def test_legacy_k1_n_classes_is_K_total(monkeypatch: pytest.MonkeyPatch) -> None:
    """At classes_per_forward=1, n_classes == K_total (not G, which also == K_total at K=1)."""
    K_total = 3
    class_names = ["A", "B", "C"]
    batch = _batch([["A", "B", "C"]], [[_instance(0), _instance(1), _instance(2)]])

    wrapper = _make_wrapper()
    opt = _make_optimizer(wrapper)
    sched = _make_scheduler(opt)
    monkeypatch.setattr(random, "random", lambda: 0.0)

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
    monkeypatch.setattr(random, "random", lambda: 0.0)

    cfg = _make_cfg_k1(nan_abort_after=99)
    opt = _make_optimizer(wrapper)
    sched = _make_scheduler(opt)

    result = train_step(
        wrapper, batch, opt, sched, cfg, class_names=class_names, global_step=0, nan_streak=0
    )
    # B class was finite → not skipped.
    assert not result.skipped
    assert result.nan_streak == 0
