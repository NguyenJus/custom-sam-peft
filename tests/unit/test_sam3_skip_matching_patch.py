"""CPU unit tests for _patch_forward_grounding_skip_matching_on_none_target.

Covers the find_target=None crash fix described at
src/custom_sam_peft/models/sam3.py::_patch_forward_grounding_skip_matching_on_none_target.

sam3's Sam3Image.forward_grounding fires
``self._compute_matching(out, self.back_convert(find_target))`` whenever
``self.training or num_interactive_steps_val > 0``.  Our _Sam3ImageAdapter
passes ``find_target=None``; ``back_convert(None)`` then dereferences
``None.boxes`` and raises an AttributeError on every training-mode forward
(silent in eval mode because the gate is False).  The patch swaps both
``back_convert`` and ``_compute_matching`` with bound-method wrappers that
short-circuit on ``targets is None`` and delegate otherwise.

These tests use a fake stand-in exposing the two methods with sam3's
semantics, so they run on CPU without the gated checkpoint.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import nn

from custom_sam_peft.models.sam3 import _patch_forward_grounding_skip_matching_on_none_target


class _FakeTarget:
    """Stand-in for sam3's BatchedFindTarget, exposing just the `.boxes` attr."""

    def __init__(self, boxes: torch.Tensor) -> None:
        self.boxes = boxes


class _FakeSam3Image(nn.Module):
    """Stand-in for sam3.Sam3Image with the two methods the patch rebinds."""

    def __init__(self) -> None:
        super().__init__()
        self.compute_matching_call_log: list[tuple[int, Any]] = []

    def back_convert(self, targets: Any) -> dict[str, torch.Tensor]:
        # Mirrors sam3's first-line dereference: crashes on None.
        return {"boxes": targets.boxes.view(-1, 4)}

    def _compute_matching(self, out: dict[str, Any], targets: Any) -> None:
        # Mirrors sam3: writes the indices key (the side-effect we don't use).
        self.compute_matching_call_log.append((id(out), targets))
        out["indices"] = ["sentinel"]


def test_unpatched_back_convert_crashes_on_none() -> None:
    """Pin the bug: without the patch, back_convert(None) raises AttributeError."""
    m = _FakeSam3Image()
    with pytest.raises(AttributeError):
        m.back_convert(None)


def test_patched_back_convert_returns_none_when_input_is_none() -> None:
    m = _FakeSam3Image()
    _patch_forward_grounding_skip_matching_on_none_target(m)
    assert m.back_convert(None) is None


def test_patched_back_convert_delegates_for_non_none_targets() -> None:
    """Non-None find_target paths are untouched (eval-time interactive runs)."""
    m = _FakeSam3Image()
    _patch_forward_grounding_skip_matching_on_none_target(m)
    target = _FakeTarget(boxes=torch.arange(12, dtype=torch.float32).view(3, 4))
    out = m.back_convert(target)
    assert out is not None
    assert torch.equal(out["boxes"], torch.arange(12, dtype=torch.float32).view(3, 4))


def test_patched_compute_matching_noop_when_target_is_none() -> None:
    m = _FakeSam3Image()
    _patch_forward_grounding_skip_matching_on_none_target(m)
    out: dict[str, Any] = {}
    m._compute_matching(out, None)
    assert "indices" not in out
    assert m.compute_matching_call_log == []


def test_patched_compute_matching_delegates_for_non_none_targets() -> None:
    """When targets is non-None, the original side-effect still fires."""
    m = _FakeSam3Image()
    _patch_forward_grounding_skip_matching_on_none_target(m)
    out: dict[str, Any] = {}
    m._compute_matching(out, "any-non-none-sentinel")
    assert out["indices"] == ["sentinel"]
    assert len(m.compute_matching_call_log) == 1


def test_idempotency() -> None:
    """Calling the patcher twice rebinds at most once."""
    m = _FakeSam3Image()
    _patch_forward_grounding_skip_matching_on_none_target(m)
    bc1 = m.back_convert
    cm1 = m._compute_matching
    _patch_forward_grounding_skip_matching_on_none_target(m)
    assert m.back_convert is bc1
    assert m._compute_matching is cm1
    assert getattr(m, "_custom_sam_peft_skip_matching_on_none_target_patched", False) is True


def test_missing_methods_are_silent() -> None:
    """Models without back_convert / _compute_matching are skipped, not errored."""
    m = nn.Linear(4, 4)
    _patch_forward_grounding_skip_matching_on_none_target(m)  # must not raise
    assert not getattr(m, "_custom_sam_peft_skip_matching_on_none_target_patched", False)


def test_partial_methods_are_silent() -> None:
    """Only one of the two methods present is treated as 'not a Sam3Image' — skipped."""

    class _OnlyBackConvert(nn.Module):
        def back_convert(self, targets: Any) -> Any:
            return targets

    m = _OnlyBackConvert()
    _patch_forward_grounding_skip_matching_on_none_target(m)
    assert not getattr(m, "_custom_sam_peft_skip_matching_on_none_target_patched", False)
