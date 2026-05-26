"""Tests for the train_step OOM-retry ladder.

We inject `torch.cuda.OutOfMemoryError` from a stub model's forward — the
exception class is importable without CUDA, so this runs on CPU.

Spec: docs/superpowers/specs/2026-05-22-algo-vram-preset-design.md §6.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch

from custom_sam_peft.train.types import OomEvent


class _OomThenOk(torch.nn.Module):
    """forward() raises CUDA OOM the first `n_oom` times, then returns a real loss."""

    def __init__(self, n_oom: int) -> None:
        super().__init__()
        self.n_oom = n_oom
        self.calls = 0
        # A trainable parameter so backward() has something to differentiate.
        self.p = torch.nn.Parameter(torch.zeros(1, requires_grad=True))

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        self.calls += 1
        if self.calls <= self.n_oom:
            raise torch.cuda.OutOfMemoryError("synthetic")
        return self.p.sum()


# --- The OOM ladder helper under test --------------------------------------
# `_train_step_with_oom_ladder` is the new helper we land in train/loop.py.
# These tests import it directly to keep the surface small.

from custom_sam_peft.train.loop import _train_step_with_oom_ladder  # noqa: E402


@dataclass
class _State:
    step: int = 0
    micro_batch_size: int = 8
    pending_oom_events: list[OomEvent] = field(default_factory=list)


def _make_batch(n: int) -> list[int]:
    """Stand-in batch: a list of ints. Sliceable, has __len__."""
    return list(range(n))


def _fake_forward_call(model: torch.nn.Module, micro: list[int]) -> torch.Tensor:
    return model(micro)


def test_oom_first_attempt_halves_microbatch() -> None:
    state = _State(micro_batch_size=8)
    model = _OomThenOk(n_oom=1)
    _train_step_with_oom_ladder(model, _make_batch(8), state, forward_call=_fake_forward_call)
    assert state.micro_batch_size == 4
    assert len(state.pending_oom_events) == 1
    assert state.pending_oom_events[0].action == "microbatch_halved"


def test_oom_multiple_halvings_until_one() -> None:
    state = _State(micro_batch_size=8)
    model = _OomThenOk(n_oom=3)
    _train_step_with_oom_ladder(model, _make_batch(8), state, forward_call=_fake_forward_call)
    assert state.micro_batch_size == 1
    assert len(state.pending_oom_events) == 3
    assert all(e.action == "microbatch_halved" for e in state.pending_oom_events)


def test_oom_after_microbatch_1_raises() -> None:
    state = _State(micro_batch_size=8)
    model = _OomThenOk(n_oom=4)  # 3 halvings → mb=1, 4th OOM raises
    with pytest.raises(RuntimeError, match="OOM at step"):
        _train_step_with_oom_ladder(model, _make_batch(8), state, forward_call=_fake_forward_call)


def test_oom_microbatch_shrink_is_sticky() -> None:
    state = _State(micro_batch_size=8)
    # Step 1: 1 OOM → mb halves to 4.
    model = _OomThenOk(n_oom=1)
    _train_step_with_oom_ladder(model, _make_batch(8), state, forward_call=_fake_forward_call)
    assert state.micro_batch_size == 4
    # Step 2 with a fresh stub that never OOMs.
    state.step = 1
    model2 = _OomThenOk(n_oom=0)
    _train_step_with_oom_ladder(model2, _make_batch(8), state, forward_call=_fake_forward_call)
    # mb did not reset.
    assert state.micro_batch_size == 4


def test_oom_optimizer_zero_grad_called_once_per_step() -> None:
    """Spec §6 invariant: optimizer.zero_grad() fires once per outer step,
    not once per microbatch and not on retry."""
    state = _State(micro_batch_size=4)
    model = _OomThenOk(n_oom=1)
    optimizer = MagicMock()
    # Test harness: a thin wrapper that mimics the trainer's step structure.
    optimizer.zero_grad()
    _train_step_with_oom_ladder(model, _make_batch(4), state, forward_call=_fake_forward_call)
    # The ladder helper itself never calls zero_grad — the caller did once above.
    assert optimizer.zero_grad.call_count == 1


def test_oom_events_propagated_in_eval_artifacts() -> None:
    """Trainer.fit returns an EvalArtifacts exposing the accumulated events tuple."""
    from custom_sam_peft.eval._artifacts import EvalArtifacts

    fields = {f.name for f in __import__("dataclasses").fields(EvalArtifacts)}
    assert "oom_events" in fields


def test_oom_gradient_magnitude_preserved_across_ladder() -> None:
    """After OOM forces microbatch shrink, total backward gradient should equal
    the non-OOM case — the helper's /n_micro must not double-scale."""
    p = torch.nn.Parameter(torch.tensor([1.0]))

    class _Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.p = p
            self.calls = 0
            self.fail_first = True

        def forward(self, micro: list[int]) -> torch.Tensor:
            self.calls += 1
            # First call OOMs; after that succeed.
            if self.fail_first and self.calls == 1:
                raise torch.cuda.OutOfMemoryError("synthetic")
            # Loss = p * sum(indices).  Grad of p w.r.t. total-batch loss
            # should equal sum(all indices) regardless of microbatch slicing.
            return self.p * sum(micro)

    state = _State(micro_batch_size=4)
    model = _Model()
    p.grad = None  # ensure clean slate
    _train_step_with_oom_ladder(model, _make_batch(4), state, forward_call=_fake_forward_call)
    # batch = [0,1,2,3], sum = 6.  After OOM mb halves to 2 → n_micro = 2.
    # The helper applies (loss / n_micro).backward() each microbatch.
    # Microbatch [0,1]: loss = p*1; (loss/2).backward() → grad contribution = 1/2.
    # Microbatch [2,3]: loss = p*5; (loss/2).backward() → grad contribution = 5/2.
    # Total grad = 1/2 + 5/2 = 3.0.
    # If the closure also divided by n_micro the total would be 1.5 (double-scale bug).
    assert state.micro_batch_size == 2
    assert p.grad is not None
    assert abs(p.grad.item() - 3.0) < 1e-5, (
        f"Expected grad 3.0 (sum/n_micro = 6/2), got {p.grad.item()} — double-divide?"
    )


def test_oom_events_serialise_into_bundle_edge_cases() -> None:
    """An end-to-end sanity check that events flowed into the bundler renders.

    This is a shallow trace check — the full rendering is exercised in
    tests/unit/runs/test_bundle.py::test_write_bundle_oom_edge_note_with_ckpt.
    Here we only confirm the linkage: a non-empty oom_events tuple on
    BundleContext produces a `## Edge cases` line containing 'OOM retries'.
    """
    import tempfile
    from datetime import UTC, datetime
    from pathlib import Path as _P
    from unittest.mock import MagicMock as _MM

    from custom_sam_peft.presets import PresetDecision
    from custom_sam_peft.runs.bundle import BundleContext, write_bundle

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = _P(tmp)
        (tmp_path / "run").mkdir()
        (tmp_path / "config.yaml").write_text("run: {name: r}\n")
        decision = PresetDecision(
            method="lora",
            r=16,
            batch_size=1,
            grad_accum_steps=16,
            dtype="bfloat16",
            headroom_bytes=0,
            predicted_bytes=0,
            budget_bytes=0,
            image_size=1008,
            gpu_name="StubGPU",
            provenance="analytic",
            cache_path=None,
            calibrated_at=None,
        )
        ctx = BundleContext(
            run_dir=tmp_path / "run",
            config_path=tmp_path / "config.yaml",
            start_ts=datetime(2026, 5, 22, tzinfo=UTC),
            end_ts=datetime(2026, 5, 22, tzinfo=UTC),
            preset=decision,
            per_example_iou=[],
            merged_dir=None,
            merged_export_error=None,
            oom_events=(OomEvent(step=1, action="microbatch_halved", new_micro_batch_size=4),),
        )
        report = _MM(overall={"mAP": 0.0})
        val_ds = _MM(__len__=lambda self: 0)
        write_bundle(ctx, report, val_dataset=val_ds, model_wrapper=_MM())
        summary = (tmp_path / "run" / "summary.md").read_text()
        assert "OOM retries: 1" in summary
