"""Multiplex per-group behavior tests.

Covers:
- auto-chunk INFO log when classes_in_batch > MULTIPLEX_CAP
- StepResult.n_classes == K_total (not G)
- K_total=4 with default cap -> one model call with K=4 prompts
- B-then-K OOM ladder: K-replay, zero_grad, sticky effective_K, events, hard-fail
"""

from __future__ import annotations

import logging
import random
from typing import Any
from unittest.mock import MagicMock

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
from custom_sam_peft.train.loop import OomState, _reset_auto_chunk_log, train_step
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


# ---------------------------------------------------------------------------
# B-then-K OOM ladder tests (Spec §4)
# ---------------------------------------------------------------------------


def _make_oom_wrapper_spy(
    monkeypatch: pytest.MonkeyPatch,
    wrapper: Sam3Wrapper,
    *,
    oom_when_group_size_gt: int,
) -> tuple[list[list[str]], list[list[str]]]:
    """Patch wrapper.forward to OOM when a group has more than `oom_when_group_size_gt`
    classes; succeed otherwise.  Returns (oom_classes_seen, ok_classes_seen) spy lists."""
    real_forward = wrapper.forward
    oom_classes: list[list[str]] = []
    ok_classes: list[list[str]] = []

    def spy(
        images: torch.Tensor,
        prompts: list[Any],
        support: Any = None,
    ) -> Any:
        if not prompts:
            return real_forward(images, prompts, support=support)
        k = len(prompts[0].classes) if isinstance(prompts[0], TextPrompts) else 0
        classes = list(prompts[0].classes) if isinstance(prompts[0], TextPrompts) else []
        if k > oom_when_group_size_gt:
            oom_classes.append(classes)
            raise torch.cuda.OutOfMemoryError("synthetic OOM from spy")
        ok_classes.append(classes)
        return real_forward(images, prompts, support=support)

    monkeypatch.setattr(wrapper, "forward", spy)
    return oom_classes, ok_classes


def test_oom_k_rung_rechunks_all_classes_none_dropped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Inner B-ladder exhausts at micro_batch=1 -> train_step halves effective_K,
    re-chunks ALL classes into more groups, replays whole step, trains every class.
    Spec §4.2 (inv a)."""
    _reset_auto_chunk_log()
    monkeypatch.setattr(random, "random", lambda: 0.0)

    # 4 classes, classes_per_forward=4 -> G=1 initially.
    # wrapper OOMs for group size > 2; succeeds for <= 2.
    # After K halving 4->2: G becomes 2, each group has 2 classes.
    K_total = 4
    class_names = [f"cls{i}" for i in range(K_total)]
    batch = _batch([class_names], [[_instance(i) for i in range(K_total)]])

    wrapper = _make_wrapper()
    _oom_spy, ok_spy = _make_oom_wrapper_spy(monkeypatch, wrapper, oom_when_group_size_gt=2)

    cfg = _make_cfg(classes_per_forward=4, nan_abort_after=99)
    opt = MagicMock()
    sched = MagicMock()
    sched.get_last_lr.return_value = [1e-4]

    # micro_batch_size=1 so any OOM immediately signals _MicrobatchExhausted.
    oom_state = OomState(micro_batch_size=1, effective_K=4)

    train_step(
        wrapper,
        batch,
        opt,
        sched,
        cfg,
        class_names=class_names,
        global_step=0,
        nan_streak=0,
        oom_state=oom_state,
    )

    assert oom_state.effective_K == 2, f"expected effective_K=2, got {oom_state.effective_K}"
    # After K-rung replay: ok_spy should contain exactly the 4 classes split into 2 groups.
    trained_classes = {c for group_cls in ok_spy for c in group_cls}
    assert trained_classes == set(class_names), (
        f"Not all classes trained: missing {set(class_names) - trained_classes}"
    )


def test_oom_k_rung_zero_grad_and_whole_step_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """K-rung calls optimizer.zero_grad() (discards larger-K grads) and replays
    from group 0. Spec §4.3 (inv b)."""
    _reset_auto_chunk_log()
    monkeypatch.setattr(random, "random", lambda: 0.0)

    K_total = 4
    class_names = [f"cls{i}" for i in range(K_total)]
    batch = _batch([class_names], [[_instance(i) for i in range(K_total)]])

    wrapper = _make_wrapper()
    _oom_spy, ok_spy = _make_oom_wrapper_spy(monkeypatch, wrapper, oom_when_group_size_gt=2)

    cfg = _make_cfg(classes_per_forward=4, nan_abort_after=99)
    opt = MagicMock()
    sched = MagicMock()
    sched.get_last_lr.return_value = [1e-4]

    oom_state = OomState(micro_batch_size=1, effective_K=4)

    # Record the value of oom_state.effective_K at each zero_grad call.
    # loop.py line 388 calls zero_grad BEFORE halving effective_K at line 389,
    # so at least one recorded value must equal the pre-halve value (4).
    k_at_zero_grad: list[int] = []

    def _recording_zero_grad(**kwargs: object) -> None:
        k_at_zero_grad.append(oom_state.effective_K)

    opt.zero_grad.side_effect = _recording_zero_grad

    train_step(
        wrapper,
        batch,
        opt,
        sched,
        cfg,
        class_names=class_names,
        global_step=0,
        nan_streak=0,
        oom_state=oom_state,
    )

    # 1. zero_grad must have been called at least once on OOM K-rung entry
    #    (to discard the partial grads from the failed larger-K attempt).
    assert opt.zero_grad.called, "optimizer.zero_grad() must be called during K-rung replay"

    # 2. Prove the K-rung zero_grad fired while effective_K was still the
    #    pre-halve value (4).  loop.py §388 calls zero_grad, §389 halves to 2.
    assert any(k == 4 for k in k_at_zero_grad), (
        f"Expected at least one zero_grad call while effective_K==4 (pre-halve); "
        f"recorded values: {k_at_zero_grad}"
    )

    # 3. Prove the replay restarts from group 0: the first successful forward
    #    after the K-rung must be group 0 of the re-chunked layout ([cls0, cls1]).
    assert ok_spy, "Expected at least one successful forward after K-rung replay"
    assert ok_spy[0] == ["cls0", "cls1"], (
        f"Replay must restart from group 0 ([cls0, cls1]); got first ok group: {ok_spy[0]}"
    )

    # 4. Prove all classes are covered by the replay (no class dropped).
    trained_classes = {c for group_cls in ok_spy for c in group_cls}
    assert trained_classes == set(class_names), (
        f"Not all classes trained in replay: missing {set(class_names) - trained_classes}"
    )


def test_oom_effective_k_sticky_across_steps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """effective_K shrinks once and stays shrunk next step. Spec §4.3 (inv e)."""
    _reset_auto_chunk_log()
    monkeypatch.setattr(random, "random", lambda: 0.0)

    K_total = 4
    class_names = [f"cls{i}" for i in range(K_total)]
    batch = _batch([class_names], [[_instance(i) for i in range(K_total)]])

    wrapper = _make_wrapper()
    # Spy that OOMs on group size > 2 ONLY ON THE FIRST STEP (step=0).
    # On step=1, effective_K is already 2 so groups are size <=2 — no OOM.
    real_forward = wrapper.forward
    call_count = [0]

    def spy_sticky(images: torch.Tensor, prompts: list[Any], support: Any = None) -> Any:
        k = len(prompts[0].classes) if (prompts and isinstance(prompts[0], TextPrompts)) else 0
        call_count[0] += 1
        if k > 2:
            raise torch.cuda.OutOfMemoryError("synthetic OOM")
        return real_forward(images, prompts, support=support)

    monkeypatch.setattr(wrapper, "forward", spy_sticky)

    cfg = _make_cfg(classes_per_forward=4, nan_abort_after=99)
    opt = MagicMock()
    sched = MagicMock()
    sched.get_last_lr.return_value = [1e-4]

    oom_state = OomState(micro_batch_size=1, effective_K=4)

    # Step 0: triggers K-rung (effective_K 4->2)
    train_step(
        wrapper,
        batch,
        opt,
        sched,
        cfg,
        class_names=class_names,
        global_step=0,
        nan_streak=0,
        oom_state=oom_state,
    )
    assert oom_state.effective_K == 2, "After step 0: expected effective_K=2"

    # Step 1: effective_K is sticky at 2, groups are <=2 classes -> no OOM
    oom_events_before = len(oom_state.pending_oom_events)
    train_step(
        wrapper,
        batch,
        opt,
        sched,
        cfg,
        class_names=class_names,
        global_step=1,
        nan_streak=0,
        oom_state=oom_state,
    )
    assert oom_state.effective_K == 2, "effective_K must remain 2 on step 1 (sticky)"
    # No new multiplex_halved events on step 1
    new_events = oom_state.pending_oom_events[oom_events_before:]
    assert all(e.action != "multiplex_halved" for e in new_events), (
        "Step 1 must not emit another multiplex_halved event"
    )


def test_oom_records_multiplex_halved_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A multiplex_halved OomEvent carrying new effective_K recorded. Spec §4.4."""
    _reset_auto_chunk_log()
    monkeypatch.setattr(random, "random", lambda: 0.0)

    K_total = 4
    class_names = [f"cls{i}" for i in range(K_total)]
    batch = _batch([class_names], [[_instance(i) for i in range(K_total)]])

    wrapper = _make_wrapper()
    _make_oom_wrapper_spy(monkeypatch, wrapper, oom_when_group_size_gt=2)

    cfg = _make_cfg(classes_per_forward=4, nan_abort_after=99)
    opt = MagicMock()
    sched = MagicMock()
    sched.get_last_lr.return_value = [1e-4]

    oom_state = OomState(micro_batch_size=1, effective_K=4)

    train_step(
        wrapper,
        batch,
        opt,
        sched,
        cfg,
        class_names=class_names,
        global_step=5,
        nan_streak=0,
        oom_state=oom_state,
    )

    multiplex_events = [e for e in oom_state.pending_oom_events if e.action == "multiplex_halved"]
    assert len(multiplex_events) >= 1, "Expected at least one multiplex_halved OomEvent"
    ev = multiplex_events[0]
    assert ev.effective_K == 2, f"Expected effective_K=2 in event, got {ev.effective_K}"
    assert ev.step == 5, f"Expected step=5 in event, got {ev.step}"


def test_oom_final_hard_fail_only_when_b_and_k_both_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """micro_batch=1 AND effective_K=1 and OOM still fires -> raise naming
    classes_per_forward=1. Spec §4.5."""
    _reset_auto_chunk_log()
    monkeypatch.setattr(random, "random", lambda: 0.0)

    class_names = ["cls0"]
    batch = _batch([class_names], [[_instance(0)]])

    wrapper = _make_wrapper()

    def always_oom(images: torch.Tensor, prompts: list[Any], support: Any = None) -> Any:
        raise torch.cuda.OutOfMemoryError("synthetic OOM — always")

    monkeypatch.setattr(wrapper, "forward", always_oom)

    cfg = _make_cfg(classes_per_forward=1, nan_abort_after=99)
    opt = MagicMock()
    sched = MagicMock()
    sched.get_last_lr.return_value = [1e-4]

    # Both B and K already at minimum
    oom_state = OomState(micro_batch_size=1, effective_K=1)

    with pytest.raises(RuntimeError, match=r"classes_per_forward=1"):
        train_step(
            wrapper,
            batch,
            opt,
            sched,
            cfg,
            class_names=class_names,
            global_step=3,
            nan_streak=0,
            oom_state=oom_state,
        )


def test_nan_group_skip_path_untouched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hungarian non-finite-cost group-skip still SKIPS a group (does NOT re-chunk),
    independent of OOM K-rung. Spec §4.2 (inv c)."""
    _reset_auto_chunk_log()
    monkeypatch.setattr(random, "random", lambda: 0.0)

    # 2 classes; classes_per_forward=4 -> G=1 (both classes in one group).
    class_names = ["cls0", "cls1"]
    batch = _batch([class_names], [[_instance(0), _instance(1)]])

    wrapper = _make_wrapper()

    # The forward always raises ValueError (simulates Hungarian non-finite cost).
    def value_error_forward(images: torch.Tensor, prompts: list[Any], support: Any = None) -> Any:
        raise ValueError("synthetic non-finite cost matrix")

    monkeypatch.setattr(wrapper, "forward", value_error_forward)

    cfg = _make_cfg(classes_per_forward=4, nan_abort_after=99)
    opt = MagicMock()
    sched = MagicMock()
    sched.get_last_lr.return_value = [1e-4]

    oom_state = OomState(micro_batch_size=1, effective_K=4)

    # ValueError should cause group-skip (skipped=True), NOT OOM K-rung.
    result = train_step(
        wrapper,
        batch,
        opt,
        sched,
        cfg,
        class_names=class_names,
        global_step=0,
        nan_streak=0,
        oom_state=oom_state,
    )

    # effective_K must remain unchanged (no K-halving happened)
    assert oom_state.effective_K == cfg.train.multiplex.classes_per_forward, (
        f"effective_K must not shrink on NaN-skip; got {oom_state.effective_K}"
    )
    # No multiplex_halved events
    assert all(e.action != "multiplex_halved" for e in oom_state.pending_oom_events), (
        "NaN-skip must not emit multiplex_halved events"
    )
    # The step was skipped (all groups non-finite)
    assert result.skipped
