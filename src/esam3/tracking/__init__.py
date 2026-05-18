"""Tracking subsystem — Tracker Protocol, build_tracker factory, flatten helper."""

from __future__ import annotations

from typing import TYPE_CHECKING

from esam3.tracking.base import Tracker

if TYPE_CHECKING:
    # Type-only import; avoids a runtime tracking → eval dependency so the
    # tracking subsystem remains independent of subsystems 1-6
    # (see architecture §11).
    from esam3.eval.metrics import MetricsReport

__all__ = ["Tracker", "flatten_metrics_report"]


def flatten_metrics_report(
    report: MetricsReport,
    prefix: str = "eval",
) -> dict[str, float]:
    """Render a MetricsReport as a flat scalar dict for log_scalars.

    Duck-typed at runtime: accepts any object with ``overall: dict[str, float]``
    and ``per_class: dict[str, dict[str, float]]``. Keys are namespaced under
    ``prefix``; '/' in class names is replaced with '_'.
    """
    out: dict[str, float] = {f"{prefix}/{k}": float(v) for k, v in report.overall.items()}
    for cls, metrics in report.per_class.items():
        safe = cls.replace("/", "_")
        for k, v in metrics.items():
            out[f"{prefix}/per_class/{safe}/{k}"] = float(v)
    return out
