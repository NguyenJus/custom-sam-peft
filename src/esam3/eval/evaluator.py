"""Evaluator — runs a model over a dataset and returns a MetricsReport."""

from __future__ import annotations

from typing import Any

from esam3.config.schema import EvalConfig
from esam3.data.base import Dataset
from esam3.eval.metrics import MetricsReport


class Evaluator:
    """Compute COCO metrics for a model on a dataset.

    Implementation deferred to spec/eval.
    """

    def __init__(self, cfg: EvalConfig) -> None:
        self.cfg = cfg

    def evaluate(self, model: Any, dataset: Dataset) -> MetricsReport:
        raise NotImplementedError("filled in by spec: spec/eval")
