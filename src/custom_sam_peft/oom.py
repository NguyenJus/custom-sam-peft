"""Shared OOM ladder: sticky B-then-K state + the halving decision.

`OomLadder` owns ladder STATE (micro_batch_size B, effective_K K) and the
B-then-K halving DECISION only. It knows nothing about microbatches, image
chunks, class groups, gradients, or replay — those are caller concepts. Train,
eval, and predict each construct a ladder with their per-path initial (B, K)
and map the returned `OomDecision` to their own control flow (spec §3 mapping
table). `OomEvent` lives here (relocated from train/types.py, which re-exports
it for back-compat).

Spec: docs/superpowers/specs/2026-05-29-unified-oom-ladder-design.md
"""

from __future__ import annotations

import enum
import logging
from dataclasses import dataclass, field
from typing import Any, Literal

import torch

_LOG = logging.getLogger(__name__)

__all__ = ["OomDecision", "OomEvent", "OomLadder"]


@dataclass(frozen=True)
class OomEvent:
    """One OOM-halving transition, recorded for telemetry / bundle rendering.

    `action` records the rung:
      - "microbatch_halved": B was halved (effective_K is None).
      - "multiplex_halved": K was halved; carries the new effective_K.

    Fields capture *post*-halving state so downstream rendering can reconstruct
    the run's safety-net history without re-traversing mutable state.
    """

    step: int
    action: Literal["microbatch_halved", "multiplex_halved"]
    new_micro_batch_size: int
    effective_K: int | None = None  # set only for "multiplex_halved" events


class OomDecision(enum.Enum):
    """What a caller should do after one OOM, per the B-then-K policy."""

    RETRY_B = "retry_b"
    RETRY_K = "retry_k"
    FLOOR_RETRY = "floor_retry"
    TERMINAL = "terminal"


def _halve_microbatch(state: Any, step: int | None = None) -> None:
    """Shared B-rung mechanic: halve micro_batch_size + record the transition.

    FIELD-ONLY. Operates solely on the object's *fields* — `micro_batch_size`,
    `pending_oom_events`, `step` — so it works on BOTH `OomLadder` and train's
    field-only `_State` stub (which has no methods). This is the SINGLE
    implementation of the B-halving: `OomLadder.on_oom()`'s B-branch delegates
    to it, and train's inner helper calls it directly (spec §4 "Shared
    _halve_microbatch routine", §5.1). Callers do the `empty_cache()` and the
    `micro_batch_size > 1` guard before calling this.
    """
    if step is not None:
        state.step = step
    state.micro_batch_size //= 2
    state.pending_oom_events.append(
        OomEvent(
            step=state.step,
            action="microbatch_halved",
            new_micro_batch_size=state.micro_batch_size,
        )
    )
    _LOG.warning(
        "OOM at step %d — halving micro_batch_size to %d",
        state.step,
        state.micro_batch_size,
    )


@dataclass
class OomLadder:
    """Sticky, monotonically-decreasing B-then-K OOM state + decision.

    Constructed per path with the initial (micro_batch_size, effective_K).
    on_oom() applies the B-then-K policy; callers map the returned OomDecision
    to their own control flow (spec §3 mapping table). B and K only ever
    decrease (sticky). FLOOR_RETRY is returned at most once per lifetime.
    """

    micro_batch_size: int  # B — only ever decreases
    effective_K: int  # K — only ever decreases
    pending_oom_events: list[OomEvent] = field(default_factory=list)
    step: int = 0  # last-seen step, for telemetry parity with the old OomState
    _floor_retry_used: bool = field(default=False, repr=False, init=False)

    def on_oom(self, step: int | None = None) -> OomDecision:
        """Apply the B-then-K policy to one OOM event.

        1. Guarded torch.cuda.empty_cache() (the #176 robustness guarantee).
        2. If B > 1: delegate to the shared _halve_microbatch() routine
           (halves B, records a microbatch_halved event, warns), RETRY_B.
        3. elif K > 1: halve K, record a multiplex_halved event, warn, RETRY_K.
        4. elif not used: consume the single FLOOR_RETRY.
        5. else: TERMINAL.

        Records at most one OomEvent per call (the halving it performed); none
        for FLOOR_RETRY / TERMINAL.
        """
        if step is not None:
            self.step = step
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if self.micro_batch_size > 1:
            # Delegate to the SINGLE shared B-rung mechanic (also called directly
            # by train's inner helper — spec §4). No duplicated halving logic.
            _halve_microbatch(self, self.step)
            return OomDecision.RETRY_B

        if self.effective_K > 1:
            self.effective_K //= 2
            self.pending_oom_events.append(
                OomEvent(
                    step=self.step,
                    action="multiplex_halved",
                    new_micro_batch_size=self.micro_batch_size,
                    effective_K=self.effective_K,
                )
            )
            _LOG.warning(
                "OOM at step %d after micro_batch=1 — halving effective_K to %d",
                self.step,
                self.effective_K,
            )
            return OomDecision.RETRY_K

        if not self._floor_retry_used:
            self._floor_retry_used = True
            return OomDecision.FLOOR_RETRY

        return OomDecision.TERMINAL
