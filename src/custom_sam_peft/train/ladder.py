"""Early-stop state with cold-mAP grace (#264).

A single no-improvement counter is fed the same mAP at each successful eval,
using one improvement test (mAP > best + min_delta, strict). The counter only
accrues once grace is lifted, where grace requires BOTH:

  - adaptive baseline (PRIMARY): the first eval producing a strictly-positive
    mAP "wakes" the run (`woken`). The baseline floor is 0.0 — no magic number,
    self-scaling. While mAP is pinned at 0.0 (cold) the counter never climbs.
  - warmup_floor_steps (BACKSTOP): a fixed floor in optimizer steps below which
    the counter may not accrue, regardless of mAP.

A model that never produces a non-zero mAP trains to the horizon. LR is now a
pure function of step (poly/cosine/linear/constant LambdaLR) and is never cut.

Note (spec §6.3 A2): _maybe_save_best (trainer.py) saves on strict `>` (no
min_delta); this counter counts improvement on `> best + min_delta`. A tiny
improvement can save a new best yet still count as non-improvement for patience.
Intentional: always save a strictly-better checkpoint; only reset patience on a
meaningfully-better one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class StopDecision:
    should_stop: bool
    reason: str
    triggering_step: int
    triggering_map: float


@dataclass(frozen=True)
class LadderEvents:
    """Accumulated ladder telemetry, threaded into close_out."""

    stop_reason: str | None = None


@dataclass
class LadderState:
    best: float = float("-inf")  # best mAP seen by the ladder
    evals_without_improvement: int = 0  # no-improvement counter
    woken: bool = False  # latches True on the first strictly-positive mAP

    def observe(
        self,
        mAP: float | None,
        step: int,
        cfg: Any,
    ) -> StopDecision:
        """Tick the no-improvement counter on one successful eval.

        A None mAP is a no-op tick. The counter accrues only when grace is
        lifted (woken AND step >= warmup_floor_steps); early stop fires only when
        enabled AND grace is lifted AND the counter reaches stop_patience.
        """
        if mAP is None:
            return StopDecision(False, "", step, float("nan"))

        # Adaptive baseline: latch awake on the first strictly-positive mAP.
        if mAP > 0.0:
            self.woken = True

        grace_lifted = self.woken and step >= int(cfg.train.early_stop.warmup_floor_steps)

        min_delta = float(cfg.train.early_stop.min_delta)
        improved = mAP > self.best + min_delta
        if improved:
            self.best = mAP
            self.evals_without_improvement = 0
        elif grace_lifted:
            # Only accrue once grace is lifted; while cold the counter holds.
            self.evals_without_improvement += 1

        if (
            cfg.train.early_stop.enabled
            and grace_lifted
            and self.evals_without_improvement >= cfg.train.early_stop.stop_patience
        ):
            reason = (
                f"early_stop: {self.evals_without_improvement} evals without mAP "
                f"improvement (>= {cfg.train.early_stop.stop_patience})"
            )
            return StopDecision(True, reason, step, mAP)
        return StopDecision(False, "", step, mAP)

    def state_dict(self) -> dict[str, Any]:
        return {
            "best": self.best,
            "evals_without_improvement": self.evals_without_improvement,
            "woken": self.woken,
        }

    def load_state_dict(self, d: dict[str, Any]) -> None:
        self.best = float(d["best"])
        self.evals_without_improvement = int(d["evals_without_improvement"])
        self.woken = bool(d.get("woken", False))
