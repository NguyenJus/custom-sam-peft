"""Evaluation metrics — MetricsReport contract + stub computation."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class MetricsReport:
    """Result of an Evaluator.evaluate() call."""

    overall: dict[str, float] = field(default_factory=dict)
    per_class: dict[str, dict[str, float]] = field(default_factory=dict)
    n_images: int = 0
    n_predictions: int = 0


def compute_coco_map(
    predictions: object,
    ground_truth: object,
    iou_thresholds: list[float],
) -> MetricsReport:
    """Compute COCO-style mAP + per-class AP. Stub — see spec/eval."""
    raise NotImplementedError("filled in by spec: spec/eval")
