"""Frozen dataclasses shared across the training subsystem.

`OomEvent` records one rung of the trainer's per-step OOM-retry ladder.
The runner accumulates these into a flat list returned in the run result;
the bundler renders the count + final state into summary.md's `## Edge cases`.

Spec: docs/superpowers/specs/2026-05-22-algo-vram-preset-design.md §6.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class OomEvent:
    """One step where the trainer caught OOM and adapted before retrying.

    `action` records the single adaptive rung:
      - "microbatch_halved": `state.micro_batch_size //= 2`, retry same step.

    The fields capture *post*-adaptation state so downstream rendering
    ("OOM retries: N — final micro_batch=M") can reconstruct the run's
    safety-net history without re-traversing the trainer's mutable state.
    """

    step: int
    action: Literal["microbatch_halved"]
    new_micro_batch_size: int
