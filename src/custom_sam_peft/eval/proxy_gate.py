"""Checkpoint-sweep proxy-vs-exact gate for the dense-IoU AP proxy (#277).

Phase B harvest: sweep banked checkpoints through lite eval twice (exact vs
proxy) on the SAME lite val subset, then evaluate a PASS/FAIL gate.

Gate semantics (spike §8.2 steps 3-4):

1. **Spearman rho >= 0.95** (cite: spike §8.2 step 3), restricted to
   checkpoints with NON-ZERO exact mAP.  The dead-zone zeros are tied and make
   rank-correlation degenerate (spike §4.3); restricting to non-zero-exact is
   therefore REQUIRED, not optional.

2. **No min_delta-relevant adjacent inversion.**  Precise definition:
   Sort the non-zero-exact checkpoints by exact_map ascending.  For each
   adjacent pair (a, b) — b immediately after a in that sorted order — if
   ``exact_b - exact_a >= min_delta`` (i.e. the early-stop consumer's test
   ``improved = mAP > best + min_delta`` would credit b as a real improvement
   over a), then the PROXY must preserve that ordering: ``proxy_b > proxy_a``.
   A violation (``proxy_b <= proxy_a``) is a **min_delta-relevant inversion**
   and causes a GATE FAIL.

   The spec's parenthetical phrasing ("an inversion smaller than the control
   threshold") is slightly looser; this implementation uses the exact early-stop
   consumer predicate (``>=``) as the gate trigger, which is the conservative
   reading.

   Note on the current-code consumer: min_delta feeds ONLY the early-stop
   improvement test (``train/ladder.py:71-72``, ``improved = mAP > self.best +
   min_delta``).  #264 removed the ReduceLROnPlateau scheduler; the plateau
   path described in the spike (§1, §7b) no longer exists.  Best-checkpoint
   selection is a STRICT ``>`` with no min_delta
   (``train/trainer.py:397``).  The scale check guards one live consumer
   (early-stop), not two.

3. **min_delta scale check** (spike §8.2 step 4 / §7b).  Compute
   ``frac_exact = min_delta / exact_range`` and
   ``frac_proxy = min_delta / proxy_range`` (where range = max - min over the
   gated set).  Report ``ratio = frac_proxy / frac_exact``.  If
   ``ratio > scale_ratio_threshold`` (default 2.0; ``# tbd:`` harvest-time
   tunable), flag ``material_divergence``.  The harness REPORTS and recommends
   recalibration — it does NOT silently rescale min_delta.

IMPORTANT NOTE — "same subset both times" is AUTOMATIC:
The spec (lines 127, 137-138) says "fix the lite subset … and the seed across
both passes."  In the current codebase ``evaluator.py:940-947`` materialises
the lite subset as ``[dataset[i] for i in range(n)]`` where
``n = min(cfg.lite_max_images, len(dataset))``.  There is NO random selection
and NO seed involved — the first-N determinism means that as long as both
passes use the SAME ``lite_max_images``, split, and dataset the subset is
automatically identical.  No seed plumbing is needed.  The spec's
"fix the seed" instruction is moot; this comment records the finding so a
reviewer is not surprised by the absence of seed code.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# FloatArray mirrors the repo's typed-ndarray idiom (see eval/proxy_map.py).
FloatArray = np.ndarray[Any, np.dtype[np.floating[Any]]]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class SweepRecord:
    """One checkpoint's lite eval results: both exact and proxy passes."""

    checkpoint: Path
    step: int
    exact_map: float
    proxy_map: float
    exact_map_50: float | None = None  # mAP_50 from exact pass, if available
    proxy_map_50: float | None = None  # mAP_50 from proxy pass, if available


@dataclass
class ScaleCheck:
    """min_delta scale-fraction analysis over the gated checkpoint set."""

    min_delta: float
    exact_range: float
    proxy_range: float
    frac_exact: float  # min_delta / exact_range (nan/inf when range == 0)
    frac_proxy: float  # min_delta / proxy_range (nan/inf when range == 0)
    ratio: float  # frac_proxy / frac_exact (nan/inf when degenerate)
    material_divergence: bool  # ratio > scale_ratio_threshold
    scale_ratio_threshold: float


@dataclass
class GateResult:
    """Aggregate outcome of the proxy-vs-exact gate over a checkpoint sweep."""

    # Records used (full sweep, including zero-exact ones)
    all_records: list[SweepRecord] = field(default_factory=list)
    # Subset restricted to non-zero exact mAP (the gate's analysis domain)
    gated_records: list[SweepRecord] = field(default_factory=list)

    # Gate part 1: Spearman rank-correlation (spike §8.2 step 3)
    rho: float = float("nan")
    rho_threshold: float = 0.95  # cite: spike §8.2 step 3
    rho_passed: bool = False

    # Gate part 2: adjacent-inversion check
    n_inversions: int = 0
    inversion_pairs: list[tuple[SweepRecord, SweepRecord]] = field(default_factory=list)
    inversions_passed: bool = False

    # Gate part 3: scale check (spike §8.2 step 4)
    scale: ScaleCheck | None = None

    # Overall
    passed: bool = False
    n_gated: int = 0  # len(gated_records)


# ---------------------------------------------------------------------------
# Pure gate functions
# ---------------------------------------------------------------------------


def compute_spearman(
    x: FloatArray,
    y: FloatArray,
) -> float:
    """Spearman rank-correlation with proper average-rank tie handling.

    Uses scipy.stats.spearmanr (scipy is a declared project dependency;
    check pyproject.toml line 27).  Returns nan when n < 2 or when either
    variable has zero variance.

    Args:
        x: 1-D float array of values for the first variable.
        y: 1-D float array of values for the second variable (same length).

    Returns:
        Spearman rho in [-1, 1], or nan when undefined.

    Raises:
        ValueError: if x and y have different lengths.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError(
            f"compute_spearman: x and y must be 1-D arrays of the same length; "
            f"got shapes {x.shape} and {y.shape}"
        )
    n = len(x)
    if n < 2:
        return float("nan")

    from scipy.stats import spearmanr

    result = spearmanr(x, y)
    rho = float(result.statistic)
    # scipy returns nan when variance is zero; propagate that faithfully.
    return rho if math.isfinite(rho) or math.isnan(rho) else float("nan")


def restrict_to_nonzero_exact(records: list[SweepRecord]) -> list[SweepRecord]:
    """Return records where exact_map > 0.

    The dead-zone zeros are tied and make rank-correlation degenerate
    (spike §4.3).  Gate analysis is ALWAYS restricted to this subset.
    """
    return [r for r in records if r.exact_map > 0.0]


def adjacent_inversion_check(
    records: list[SweepRecord],
    min_delta: float,
) -> list[tuple[SweepRecord, SweepRecord]]:
    """Detect min_delta-relevant adjacent inversions in the gated set.

    Precise gate definition (see module docstring):
    Sort records by exact_map ascending.  For each adjacent pair (a, b):
    if ``exact_b - exact_a >= min_delta`` (the early-stop consumer's
    improvement predicate), then ``proxy_b > proxy_a`` MUST hold.
    If ``proxy_b <= proxy_a``, the pair is a min_delta-relevant inversion.

    Returns a list of (a, b) inversion pairs (empty = no inversions = PASS).

    Note: the spec parenthetical says "an inversion smaller than the control
    threshold"; this implementation uses the exact early-stop predicate
    (``exact_delta >= min_delta``) as the trigger — the conservative reading.
    """
    if len(records) < 2:
        return []

    sorted_recs = sorted(records, key=lambda r: r.exact_map)
    inversions: list[tuple[SweepRecord, SweepRecord]] = []
    for a, b in itertools.pairwise(sorted_recs):
        exact_delta = b.exact_map - a.exact_map
        if exact_delta >= min_delta and b.proxy_map <= a.proxy_map:
            inversions.append((a, b))
    return inversions


def scale_check(
    records: list[SweepRecord],
    min_delta: float,
    scale_ratio_threshold: float,
) -> ScaleCheck:
    """Compute min_delta scale fractions and their ratio.

    Reports (not silently fixes) scale divergence so the reviewer can decide
    whether to recalibrate min_delta before trusting the proxy as a control
    input (spike §8.2 step 4 / §7b).

    Degenerate ranges (max == min) yield frac = +inf (via float division)
    to surface that the signal is indistinguishable; ratio is then also inf
    or nan.  The caller receives these sentinel values — no exception is
    raised, and no hard-coded fallback verdict is applied.

    Args:
        records: the non-zero-exact gated records (or all records if called
            separately; the caller is responsible for filtering).
        min_delta: the early-stop ``min_delta`` from EarlyStopConfig
            (default 0.001 in config/schema.py:632).
        scale_ratio_threshold: flag material_divergence when ratio exceeds
            this value.  Default 2.0.  # tbd: tune at harvest time from the
            measured fractions; not derived analytically.

    Returns:
        ScaleCheck dataclass with all intermediate values exposed for
        reporting in the CLI summary.
    """
    if not records:
        return ScaleCheck(
            min_delta=min_delta,
            exact_range=0.0,
            proxy_range=0.0,
            frac_exact=float("inf"),
            frac_proxy=float("inf"),
            ratio=float("nan"),
            material_divergence=False,
            scale_ratio_threshold=scale_ratio_threshold,
        )

    exact_vals = np.array([r.exact_map for r in records], dtype=np.float64)
    proxy_vals = np.array([r.proxy_map for r in records], dtype=np.float64)

    exact_range = float(exact_vals.max() - exact_vals.min())
    proxy_range = float(proxy_vals.max() - proxy_vals.min())

    # Use float division: non-zero / 0.0 → inf (Python float behaviour).
    # Callers must tolerate inf/nan; never raises ZeroDivisionError.
    frac_exact = min_delta / exact_range if exact_range != 0.0 else float("inf")
    frac_proxy = min_delta / proxy_range if proxy_range != 0.0 else float("inf")

    # ratio = frac_proxy / frac_exact; degenerate when either is inf/nan.
    if math.isfinite(frac_exact) and math.isfinite(frac_proxy):
        ratio = frac_proxy / frac_exact if frac_exact != 0.0 else float("inf")
    elif math.isinf(frac_exact) and math.isinf(frac_proxy):
        ratio = float("nan")  # both degenerate — undefined
    else:
        ratio = float("inf")  # one degenerate — ratio blows up

    material_divergence = math.isfinite(ratio) and ratio > scale_ratio_threshold

    return ScaleCheck(
        min_delta=min_delta,
        exact_range=exact_range,
        proxy_range=proxy_range,
        frac_exact=frac_exact,
        frac_proxy=frac_proxy,
        ratio=ratio,
        material_divergence=material_divergence,
        scale_ratio_threshold=scale_ratio_threshold,
    )


def evaluate_gate(
    records: list[SweepRecord],
    min_delta: float,
    rho_threshold: float,
    scale_ratio_threshold: float,
) -> GateResult:
    """Evaluate the full proxy-vs-exact gate over a completed sweep.

    The gate is run over checkpoints with NON-ZERO exact mAP only.
    If n_gated == 0, the gate fails immediately (no signal to evaluate).

    Args:
        records: all SweepRecords from the sweep (including cold/zero ones).
        min_delta: early-stop min_delta from EarlyStopConfig (default 0.001).
        rho_threshold: Spearman rho threshold.  0.95 (cite: spike §8.2 step 3).
        scale_ratio_threshold: material-divergence flag threshold.  2.0.
            # tbd: harvest-time tunable.

    Returns:
        GateResult with all sub-criteria results and overall passed flag.
    """
    gated = restrict_to_nonzero_exact(records)
    result = GateResult(
        all_records=records,
        gated_records=gated,
        n_gated=len(gated),
        rho_threshold=rho_threshold,
    )

    if len(gated) == 0:
        # Cannot evaluate — degenerate; gate fails.
        result.passed = False
        return result

    # Gate part 1: Spearman rho.
    exact_arr = np.array([r.exact_map for r in gated], dtype=np.float64)
    proxy_arr = np.array([r.proxy_map for r in gated], dtype=np.float64)
    result.rho = compute_spearman(exact_arr, proxy_arr)
    result.rho_passed = math.isfinite(result.rho) and result.rho >= rho_threshold

    # Gate part 2: adjacent-inversion check.
    inv_pairs = adjacent_inversion_check(gated, min_delta=min_delta)
    result.inversion_pairs = inv_pairs
    result.n_inversions = len(inv_pairs)
    result.inversions_passed = len(inv_pairs) == 0

    # Gate part 3: scale check (informational + material_divergence flag).
    result.scale = scale_check(
        gated, min_delta=min_delta, scale_ratio_threshold=scale_ratio_threshold
    )

    # Overall PASS requires rho AND no inversions.
    # Material scale divergence is reported but does not auto-fail the gate —
    # it signals that min_delta recalibration should be considered before
    # trusting the proxy as a control input.
    result.passed = result.rho_passed and result.inversions_passed

    return result
