"""Unit tests for the shared OomLadder (spec §7.1).

CPU-only. on_oom() is called directly; no synthetic CUDA OOM needed except
for the empty_cache-guard tests which patch torch.cuda.is_available/empty_cache.
"""

from __future__ import annotations

import pytest
import torch

from custom_sam_peft.oom import OomDecision, OomEvent, OomLadder


def test_decision_sequence_b_then_k_then_floor_then_terminal() -> None:
    """From (B0=4, K0=4): RETRY_B until B==1, then RETRY_K until K==1, then one
    FLOOR_RETRY, then TERMINAL forever. Assert exact decision + B/K after each."""
    ladder = OomLadder(micro_batch_size=4, effective_K=4)

    assert ladder.on_oom(step=0) is OomDecision.RETRY_B
    assert (ladder.micro_batch_size, ladder.effective_K) == (2, 4)
    assert ladder.on_oom(step=0) is OomDecision.RETRY_B
    assert (ladder.micro_batch_size, ladder.effective_K) == (1, 4)

    assert ladder.on_oom(step=0) is OomDecision.RETRY_K
    assert (ladder.micro_batch_size, ladder.effective_K) == (1, 2)
    assert ladder.on_oom(step=0) is OomDecision.RETRY_K
    assert (ladder.micro_batch_size, ladder.effective_K) == (1, 1)

    assert ladder.on_oom(step=0) is OomDecision.FLOOR_RETRY
    assert (ladder.micro_batch_size, ladder.effective_K) == (1, 1)

    assert ladder.on_oom(step=0) is OomDecision.TERMINAL
    assert ladder.on_oom(step=0) is OomDecision.TERMINAL


def test_b_and_k_are_sticky_monotone() -> None:
    """B and K only ever decrease across the ladder's lifetime."""
    ladder = OomLadder(micro_batch_size=8, effective_K=2)
    seen_b = [ladder.micro_batch_size]
    seen_k = [ladder.effective_K]
    for _ in range(10):
        ladder.on_oom(step=1)
        seen_b.append(ladder.micro_batch_size)
        seen_k.append(ladder.effective_K)
    assert seen_b == sorted(seen_b, reverse=True)
    assert seen_k == sorted(seen_k, reverse=True)
    assert ladder.micro_batch_size == 1
    assert ladder.effective_K == 1


def test_pending_oom_events_emission() -> None:
    """One OomEvent per halving; none for FLOOR_RETRY/TERMINAL. microbatch_halved
    carries new_micro_batch_size with effective_K is None; multiplex_halved carries
    the new effective_K and the current new_micro_batch_size."""
    ladder = OomLadder(micro_batch_size=2, effective_K=2)

    ladder.on_oom(step=7)  # RETRY_B: B 2->1
    ladder.on_oom(step=7)  # RETRY_K: K 2->1
    ladder.on_oom(step=7)  # FLOOR_RETRY: no event
    ladder.on_oom(step=7)  # TERMINAL: no event

    assert len(ladder.pending_oom_events) == 2  # one per halving only
    b_ev, k_ev = ladder.pending_oom_events
    assert b_ev.action == "microbatch_halved"
    assert b_ev.new_micro_batch_size == 1
    assert b_ev.effective_K is None
    assert b_ev.step == 7
    assert k_ev.action == "multiplex_halved"
    assert k_ev.new_micro_batch_size == 1
    assert k_ev.effective_K == 1
    assert k_ev.step == 7


def test_empty_cache_guarded_called_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    """With is_available()->True, every on_oom invokes empty_cache once."""
    calls: list[int] = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append(1))
    ladder = OomLadder(micro_batch_size=2, effective_K=1)
    ladder.on_oom(step=0)  # RETRY_B
    ladder.on_oom(step=0)  # FLOOR_RETRY (B==1, K==1)
    ladder.on_oom(step=0)  # TERMINAL
    assert len(calls) == 3  # one per call, regardless of decision


def test_empty_cache_not_called_when_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """With is_available()->False, empty_cache is never called."""
    calls: list[int] = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append(1))
    ladder = OomLadder(micro_batch_size=2, effective_K=2)
    ladder.on_oom(step=0)
    ladder.on_oom(step=0)
    assert calls == []


def test_degenerate_start_b1_k1_floor_then_terminal() -> None:
    """(B=1, K=1): first on_oom is FLOOR_RETRY, second is TERMINAL; no events."""
    ladder = OomLadder(micro_batch_size=1, effective_K=1)
    assert ladder.on_oom(step=0) is OomDecision.FLOOR_RETRY
    assert ladder.on_oom(step=0) is OomDecision.TERMINAL
    assert ladder.pending_oom_events == []


def test_degenerate_start_b1_k_gt1_halves_k_immediately() -> None:
    """(B=1, K>1): starts halving K immediately (skips the B-rung)."""
    ladder = OomLadder(micro_batch_size=1, effective_K=4)
    assert ladder.on_oom(step=0) is OomDecision.RETRY_K
    assert ladder.effective_K == 2
    assert ladder.pending_oom_events[-1].action == "multiplex_halved"


def test_oom_event_microbatch_defaults_effective_k_none() -> None:
    ev = OomEvent(step=1, action="microbatch_halved", new_micro_batch_size=4)
    assert ev.effective_K is None


def test_oom_event_multiplex_carries_effective_k() -> None:
    ev = OomEvent(step=5, action="multiplex_halved", new_micro_batch_size=1, effective_K=8)
    assert ev.effective_K == 8
