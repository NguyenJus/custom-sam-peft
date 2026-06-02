"""Empirical predict-footprint budget for the 8 GB / CC 7.5 minimum-supported card."""

PREDICT_8GB_BUDGET_GB: float = 7.0


def decide_predict_budget_warning(
    measured_bytes: int, budget_gb: float = PREDICT_8GB_BUDGET_GB
) -> tuple[bool, str]:
    """Pure decision: warn iff the measured predict peak exceeds the small-card budget."""
    measured_gb = measured_bytes / (1024**3)
    if measured_gb > budget_gb:
        return True, (
            f"the trained model's predict footprint is ~{measured_gb:.1f} GB; it may not be "
            f"usable for prediction on 8 GB / CC 7.5 GPUs (budget ~{budget_gb:.1f} GB)."
        )
    return False, ""
