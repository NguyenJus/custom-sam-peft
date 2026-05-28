"""Verify every stub raises NotImplementedError with a spec: reference."""

from __future__ import annotations

from custom_sam_peft.config.schema import (
    EvalConfig,
)
from custom_sam_peft.eval.evaluator import Evaluator


def test_eval_stubs() -> None:
    # compute_coco_map is implemented (Task 3); Evaluator.evaluate is implemented (Task 4).
    # Nothing left to stub-check in this module — placeholder to keep test collection happy.
    ev = Evaluator(EvalConfig())
    assert ev is not None
