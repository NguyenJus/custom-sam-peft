"""Smoke: OomEvent is importable from both oom.py and train/types.py, same class."""

from __future__ import annotations


def test_oom_event_same_class_from_both_paths() -> None:
    from custom_sam_peft.oom import OomEvent as OomEventNew
    from custom_sam_peft.train.types import OomEvent as OomEventReexport

    assert OomEventNew is OomEventReexport


def test_oom_event_constructs_from_train_types_path() -> None:
    from custom_sam_peft.train.types import OomEvent

    ev = OomEvent(step=1, action="microbatch_halved", new_micro_batch_size=4)
    assert ev.new_micro_batch_size == 4
    assert ev.effective_K is None
