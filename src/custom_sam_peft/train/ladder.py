"""Plateau-response ladder state (spec §6.3, §6.4).

Two counters fed the same mAP at each successful eval, sharing one improvement
test (mAP > best + min_delta, strict). Rung 1 reuses ReduceLROnPlateau's internal
bad-eval counter (stepped here in plateau mode); rung 2 is an independent
early-stop counter that resets ONLY on genuine improvement, never on an LR cut.

Note (spec §6.3 A2): _maybe_save_best (trainer.py) saves on strict `>` (no
min_delta); this ladder counts improvement on `> best + min_delta`. A tiny
improvement can save a new best yet still count as non-improvement for patience.
Intentional: always save a strictly-better checkpoint; only reset patience on a
meaningfully-better one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from custom_sam_peft.config.schema import TrainConfig


@dataclass(frozen=True)
class StopDecision:
    should_stop: bool
    reason: str
    triggering_step: int
    triggering_map: float


@dataclass(frozen=True)
class LrCut:
    step: int
    old_lr: float
    new_lr: float
    triggering_map: float


@dataclass(frozen=True)
class LadderEvents:
    """Accumulated ladder telemetry, threaded into close_out (Phase 2)."""

    cuts: tuple[LrCut, ...] = ()
    stop_reason: str | None = None


@dataclass
class LadderState:
    best: float = float("-inf")  # best mAP seen by the ladder
    evals_without_improvement: int = 0  # rung-2 counter
    # rung-1 counter lives inside the ReduceLROnPlateau (not duplicated here)
    last_cut: LrCut | None = field(default=None, compare=False)

    def observe(
        self,
        mAP: float | None,
        step: int,
        scheduler: Any,
        cfg: Any,
    ) -> StopDecision:
        """Tick both rungs on one successful eval. A None mAP is a no-op tick."""
        self.last_cut = None
        if mAP is None:
            return StopDecision(False, "", step, float("nan"))

        min_delta = float(cfg.train.early_stop.min_delta)
        improved = mAP > self.best + min_delta
        if improved:
            self.best = mAP
            self.evals_without_improvement = 0
        else:
            self.evals_without_improvement += 1

        # Rung 1 (plateau mode only): step ReduceLROnPlateau, detect a cut by
        # comparing pre/post param_groups[0]["lr"].
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            opt = scheduler.optimizer
            old_lr = float(opt.param_groups[0]["lr"])
            scheduler.step(mAP)
            new_lr = float(opt.param_groups[0]["lr"])
            if new_lr < old_lr:
                self.last_cut = LrCut(step, old_lr, new_lr, mAP)

        # Rung 2: stop only when enabled and the counter reaches stop_patience.
        if (
            cfg.train.early_stop.enabled
            and self.evals_without_improvement >= cfg.train.early_stop.stop_patience
        ):
            reason = (
                f"early_stop: {self.evals_without_improvement} evals without mAP "
                f"improvement (>= {cfg.train.early_stop.stop_patience})"
            )
            return StopDecision(True, reason, step, mAP)
        return StopDecision(False, "", step, mAP)

    def state_dict(self) -> dict[str, Any]:
        return {"best": self.best, "evals_without_improvement": self.evals_without_improvement}

    def load_state_dict(self, d: dict[str, Any]) -> None:
        self.best = float(d["best"])
        self.evals_without_improvement = int(d["evals_without_improvement"])
