"""Tests for src/custom_sam_peft/train/types.py — frozen dataclasses."""

from __future__ import annotations

import dataclasses

import pytest

from custom_sam_peft.train.types import OomEvent


def test_oom_event_is_frozen() -> None:
    ev = OomEvent(step=42, action="microbatch_halved", new_micro_batch_size=4)
    with pytest.raises(dataclasses.FrozenInstanceError):
        ev.step = 99  # type: ignore[misc]


def test_oom_event_field_order_and_types() -> None:
    fields = {f.name: f.type for f in dataclasses.fields(OomEvent)}
    assert list(fields) == ["step", "action", "new_micro_batch_size"]


def test_oom_event_only_microbatch_halved_action() -> None:
    ev = OomEvent(step=0, action="microbatch_halved", new_micro_batch_size=1)
    assert ev.action == "microbatch_halved"
