"""_eval_forward_with_oom_ladder: sticky B halving on OOM; <= 1 warn per call."""

from __future__ import annotations

import logging

import pytest
import torch

from custom_sam_peft.eval.evaluator import _eval_forward_with_oom_ladder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_oom_error() -> torch.cuda.OutOfMemoryError:
    """Construct a torch.cuda.OutOfMemoryError without actually running out of VRAM."""
    return torch.cuda.OutOfMemoryError("CUDA out of memory (synthetic)")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_oom_halves_batch_size_sticky_and_warns_once(caplog) -> None:
    """Ladder halves batch_size on OOM and emits exactly one warning per call."""
    caplog.set_level(logging.WARNING)

    call_count = 0

    def _flaky_model(images, prompts, box_hints=None):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise _make_oom_error()
        # Second call succeeds — return a minimal valid dict.
        b = images.shape[0]
        return {
            "pred_logits": torch.zeros(b, 1, 1),
            "pred_boxes": torch.zeros(b, 1, 4),
            "pred_masks": torch.zeros(b, 1, 4, 4),
            "presence_logit_dec": torch.zeros(b, 1),
        }

    state: dict = {"batch_size": 4, "warned": False}
    images = torch.zeros(4, 3, 8, 8)
    prompts = [None] * 4

    # First call raises OOM (ladder halves and re-raises for outer loop).
    with pytest.raises(torch.cuda.OutOfMemoryError):
        _eval_forward_with_oom_ladder(_flaky_model, images, prompts, state=state)

    # batch_size should be halved.
    assert state["batch_size"] == 2
    assert state["warned"] is True

    # Exactly one warning about "eval OOM" should have been emitted.
    warns = [r for r in caplog.records if "eval OOM" in r.message]
    assert len(warns) == 1

    # Second call with smaller images succeeds; no additional warning.
    caplog.clear()
    images2 = torch.zeros(2, 3, 8, 8)
    prompts2 = [None] * 2
    result = _eval_forward_with_oom_ladder(_flaky_model, images2, prompts2, state=state)
    assert result is not None

    # No second warning emitted (warned=True suppresses it).
    extra_warns = [r for r in caplog.records if "eval OOM" in r.message]
    assert len(extra_warns) == 0


def test_oom_raises_at_B1_floor() -> None:
    """Persistent OOM at B=1 raises a RuntimeError mentioning OOM."""

    def _always_oom(images, prompts, box_hints=None):
        raise _make_oom_error()

    state: dict = {"batch_size": 1, "warned": False}
    images = torch.zeros(1, 3, 8, 8)
    prompts = [None]

    with pytest.raises(RuntimeError, match="OOM"):
        _eval_forward_with_oom_ladder(_always_oom, images, prompts, state=state)
