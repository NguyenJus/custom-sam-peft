"""Empirical predict-footprint budget for the 8 GB / CC 7.5 minimum-supported card."""

# cite: a CC 7.5 / 8 GB card has ~8.0 GB nominal; subtract ~1.0 GB driver/CUDA-context
# reservation (consistent with the ~1.0 GiB headroom convention in presets.py::_headroom_bytes)
# to get the usable predict budget. Date: 2026-05-31.
# tbd: #142 — replace the ~1.0 GB reservation with a measured figure from a real 8 GB card.
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
