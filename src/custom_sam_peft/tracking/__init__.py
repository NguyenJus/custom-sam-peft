"""Tracking subsystem — Tracker Protocol, build_tracker factory, flatten helper."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from custom_sam_peft._registry import lookup
from custom_sam_peft.config.schema import TrainConfig
from custom_sam_peft.tracking.base import Tracker

if TYPE_CHECKING:
    # Type-only import; avoids a runtime tracking → eval dependency so the
    # tracking subsystem remains independent of subsystems 1-6
    # (see architecture §11).
    from custom_sam_peft.eval.metrics import MetricsReport

__all__ = ["Tracker", "build_tracker", "flatten_metrics_report"]


def build_tracker(cfg: TrainConfig) -> Tracker:
    """Resolve cfg.tracking.backend to a concrete Tracker.

    Imports the chosen backend module lazily so missing optional extras only
    surface when that backend is actually requested. The @register decorator
    in each backend module wires the factory into _registry on first import.
    """
    backend = cfg.tracking.backend  # Literal["tensorboard", "wandb", "none"]
    if backend == "tensorboard":
        from custom_sam_peft.tracking import tensorboard as _tb  # noqa: F401
    elif backend == "wandb":
        from custom_sam_peft.tracking import wandb as _wb  # noqa: F401
    elif backend == "none":
        from custom_sam_peft.tracking import noop as _noop  # noqa: F401
    else:  # pragma: no cover — pydantic Literal rejects this at config-load
        raise ValueError(f"unknown tracking.backend: {backend!r}")
    factory = lookup("tracker", backend)
    return cast(Tracker, factory(cfg))


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
