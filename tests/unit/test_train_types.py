"""Tests for src/custom_sam_peft/train/types.py — frozen dataclasses."""

from __future__ import annotations

import dataclasses

import pytest

from custom_sam_peft.train.types import OomEvent


def test_oom_event_is_frozen() -> None:
    ev = OomEvent(
        step=42,
        action="microbatch_halved",
        new_micro_batch_size=4,
        new_gradient_checkpointing=False,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        ev.step = 99  # type: ignore[misc]


def test_oom_event_field_order_and_types() -> None:
    fields = {f.name: f.type for f in dataclasses.fields(OomEvent)}
    assert list(fields) == [
        "step",
        "action",
        "new_micro_batch_size",
        "new_gradient_checkpointing",
    ]


def test_oom_event_accepts_grad_ckpt_enabled_action() -> None:
    ev = OomEvent(
        step=0,
        action="grad_ckpt_enabled",
        new_micro_batch_size=1,
        new_gradient_checkpointing=True,
    )
    assert ev.action == "grad_ckpt_enabled"
