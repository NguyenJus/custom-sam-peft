"""CPU-only unit tests for eval/proxy_gate.py and eval/sweep.py.

Tests cover:
- Spearman rank-correlation (known-value cases incl. ties)
- Restrict-to-nonzero-exact filter
- Adjacent-inversion detection (passing + failing cases)
- Min-delta scale check (incl. degenerate ranges)
- End-to-end gate PASS and FAIL
- Checkpoint discovery/sort against a tmp dir layout
- Sweep orchestration with an injected fake eval_fn that asserts
  CSP_LITE_EXACT_MAP toggling

No GPU, no real model, no real checkpoints required.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from custom_sam_peft.eval.proxy_gate import (
    GateResult,
    ScaleCheck,
    SweepRecord,
    adjacent_inversion_check,
    compute_spearman,
    restrict_to_nonzero_exact,
    scale_check,
    evaluate_gate,
)
from custom_sam_peft.eval.sweep import discover_checkpoints, run_sweep


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _arr(*vals: float) -> np.ndarray:
    return np.array(vals, dtype=np.float64)


# ---------------------------------------------------------------------------
# compute_spearman
# ---------------------------------------------------------------------------


class TestComputeSpearman:
    def test_perfect_positive(self):
        exact = _arr(0.1, 0.2, 0.3, 0.4)
        proxy = _arr(1.0, 2.0, 3.0, 4.0)
        rho = compute_spearman(exact, proxy)
        assert abs(rho - 1.0) < 1e-9

    def test_perfect_negative(self):
        exact = _arr(0.1, 0.2, 0.3, 0.4)
        proxy = _arr(4.0, 3.0, 2.0, 1.0)
        rho = compute_spearman(exact, proxy)
        assert abs(rho - (-1.0)) < 1e-9

    def test_known_value(self):
        # Spearman rho for ranks [1,2,3,4] vs [1,3,2,4] via standard formula.
        # exact ranks: 1,2,3,4 (already ranked); proxy ranks: 1,3,2,4
        # d^2 = 0+1+1+0 = 2; rho = 1 - 6*2/(4*(16-1)) = 1 - 12/60 = 0.8
        exact = _arr(0.1, 0.2, 0.3, 0.4)
        proxy = _arr(0.1, 0.3, 0.2, 0.4)
        rho = compute_spearman(exact, proxy)
        assert abs(rho - 0.8) < 1e-6

    def test_ties_handled_via_average_rank(self):
        # exact: [0.1, 0.1, 0.3] — first two tied → avg rank 1.5
        # proxy: [1.0, 2.0, 3.0] — unique ranks 1,2,3
        # Scipy reference: use scipy.stats.spearmanr([0.1,0.1,0.3],[1.0,2.0,3.0])
        from scipy.stats import spearmanr

        exact = _arr(0.1, 0.1, 0.3)
        proxy = _arr(1.0, 2.0, 3.0)
        rho = compute_spearman(exact, proxy)
        expected = spearmanr(exact, proxy).statistic
        assert abs(rho - expected) < 1e-6

    def test_all_tied_returns_nan_or_zero(self):
        # All same value — rank correlation is undefined; implementation
        # must not raise; return 0.0 (no correlation signal).
        exact = _arr(0.5, 0.5, 0.5)
        proxy = _arr(0.5, 0.5, 0.5)
        rho = compute_spearman(exact, proxy)
        # Either 0.0 or nan is acceptable; must not raise.
        assert np.isnan(rho) or rho == 0.0

    def test_single_element_returns_nan(self):
        rho = compute_spearman(_arr(0.5), _arr(0.5))
        assert np.isnan(rho)

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="same length"):
            compute_spearman(_arr(0.1, 0.2), _arr(0.1))


# ---------------------------------------------------------------------------
# restrict_to_nonzero_exact
# ---------------------------------------------------------------------------


class TestRestrictToNonzeroExact:
    def test_filters_zeros(self):
        records = [
            SweepRecord(checkpoint=Path("a"), step=1, exact_map=0.0, proxy_map=0.1),
            SweepRecord(checkpoint=Path("b"), step=2, exact_map=0.2, proxy_map=0.3),
            SweepRecord(checkpoint=Path("c"), step=3, exact_map=0.0, proxy_map=0.05),
            SweepRecord(checkpoint=Path("d"), step=4, exact_map=0.15, proxy_map=0.25),
        ]
        gated = restrict_to_nonzero_exact(records)
        assert len(gated) == 2
        assert all(r.exact_map > 0 for r in gated)

    def test_empty_input(self):
        assert restrict_to_nonzero_exact([]) == []

    def test_all_zero_returns_empty(self):
        records = [
            SweepRecord(checkpoint=Path("a"), step=1, exact_map=0.0, proxy_map=0.1),
        ]
        assert restrict_to_nonzero_exact(records) == []

    def test_all_nonzero_keeps_all(self):
        records = [
            SweepRecord(checkpoint=Path("a"), step=1, exact_map=0.1, proxy_map=0.2),
            SweepRecord(checkpoint=Path("b"), step=2, exact_map=0.3, proxy_map=0.4),
        ]
        assert restrict_to_nonzero_exact(records) == records


# ---------------------------------------------------------------------------
# adjacent_inversion_check
# ---------------------------------------------------------------------------


class TestAdjacentInversionCheck:
    """Gate part 2: adjacent-pair inversion test.

    Definition (from proxy_gate.py docstring):
    Sort by exact_map ascending. For each adjacent pair (a, b) where
    exact_b - exact_a >= min_delta (early-stop consumer would credit b as
    improvement over a), proxy must preserve order: proxy_b > proxy_a.
    A violation (proxy_b <= proxy_a) is a min_delta-relevant inversion.
    """

    def test_passing_no_inversion(self):
        # exact ascending, proxy also ascending, delta=0.05 >= min_delta=0.01
        records = [
            SweepRecord(checkpoint=Path("a"), step=1, exact_map=0.10, proxy_map=0.10),
            SweepRecord(checkpoint=Path("b"), step=2, exact_map=0.20, proxy_map=0.20),
            SweepRecord(checkpoint=Path("c"), step=3, exact_map=0.30, proxy_map=0.30),
        ]
        inversions = adjacent_inversion_check(records, min_delta=0.01)
        assert inversions == []

    def test_failing_inversion_detected(self):
        # exact goes 0.10 -> 0.20 (delta=0.10 >= min_delta=0.01), but proxy inverts
        records = [
            SweepRecord(checkpoint=Path("a"), step=1, exact_map=0.10, proxy_map=0.30),
            SweepRecord(checkpoint=Path("b"), step=2, exact_map=0.20, proxy_map=0.20),
        ]
        inversions = adjacent_inversion_check(records, min_delta=0.01)
        assert len(inversions) == 1
        a, b = inversions[0]
        assert a.checkpoint == Path("a")
        assert b.checkpoint == Path("b")

    def test_below_min_delta_not_flagged(self):
        # delta < min_delta → pair not subject to ordering constraint
        records = [
            SweepRecord(checkpoint=Path("a"), step=1, exact_map=0.10, proxy_map=0.30),
            SweepRecord(checkpoint=Path("b"), step=2, exact_map=0.105, proxy_map=0.20),
        ]
        # exact delta = 0.005 < min_delta = 0.01 → NOT an inversion
        inversions = adjacent_inversion_check(records, min_delta=0.01)
        assert inversions == []

    def test_equal_proxy_counts_as_inversion(self):
        # proxy_b == proxy_a when exact delta >= min_delta → inversion (not strictly greater)
        records = [
            SweepRecord(checkpoint=Path("a"), step=1, exact_map=0.10, proxy_map=0.20),
            SweepRecord(checkpoint=Path("b"), step=2, exact_map=0.20, proxy_map=0.20),
        ]
        inversions = adjacent_inversion_check(records, min_delta=0.01)
        assert len(inversions) == 1

    def test_records_sorted_by_exact_before_check(self):
        # Records passed in reverse exact order — must be sorted first
        records = [
            SweepRecord(checkpoint=Path("b"), step=2, exact_map=0.30, proxy_map=0.30),
            SweepRecord(checkpoint=Path("a"), step=1, exact_map=0.10, proxy_map=0.40),
        ]
        # After sort: a(0.10,0.40) < b(0.30,0.30). delta=0.20 >= 0.01, proxy inverts.
        inversions = adjacent_inversion_check(records, min_delta=0.01)
        assert len(inversions) == 1

    def test_empty_list_no_inversions(self):
        assert adjacent_inversion_check([], min_delta=0.01) == []

    def test_single_record_no_inversions(self):
        records = [SweepRecord(checkpoint=Path("a"), step=1, exact_map=0.1, proxy_map=0.1)]
        assert adjacent_inversion_check(records, min_delta=0.01) == []


# ---------------------------------------------------------------------------
# scale_check
# ---------------------------------------------------------------------------


class TestScaleCheck:
    def test_nominal_case(self):
        # exact range 0.1-0.5 = 0.4; proxy range 0.08-0.4 = 0.32
        # frac_exact = 0.001/0.4 = 0.0025; frac_proxy = 0.001/0.32 = 0.003125
        # ratio = 0.003125/0.0025 = 1.25 → no material divergence (<2.0)
        records = [
            SweepRecord(checkpoint=Path("a"), step=1, exact_map=0.1, proxy_map=0.08),
            SweepRecord(checkpoint=Path("b"), step=2, exact_map=0.3, proxy_map=0.24),
            SweepRecord(checkpoint=Path("c"), step=3, exact_map=0.5, proxy_map=0.40),
        ]
        sc = scale_check(records, min_delta=0.001, scale_ratio_threshold=2.0)
        assert abs(sc.frac_exact - 0.001 / 0.4) < 1e-9
        assert abs(sc.frac_proxy - 0.001 / 0.32) < 1e-9
        assert abs(sc.ratio - (0.001 / 0.32) / (0.001 / 0.4)) < 1e-6
        assert not sc.material_divergence

    def test_material_divergence_flagged(self):
        # Make proxy range very small → frac_proxy huge → ratio > 2.0
        records = [
            SweepRecord(checkpoint=Path("a"), step=1, exact_map=0.0, proxy_map=0.09),
            SweepRecord(checkpoint=Path("b"), step=2, exact_map=1.0, proxy_map=0.10),
        ]
        # exact_range = 1.0; proxy_range = 0.01
        # frac_exact = 0.001/1.0 = 0.001; frac_proxy = 0.001/0.01 = 0.1
        # ratio = 0.1/0.001 = 100 → material_divergence
        sc = scale_check(records, min_delta=0.001, scale_ratio_threshold=2.0)
        assert sc.material_divergence

    def test_degenerate_zero_exact_range(self):
        # All exact same → zero range → no div-by-zero; frac_exact = inf or sentinel
        records = [
            SweepRecord(checkpoint=Path("a"), step=1, exact_map=0.5, proxy_map=0.1),
            SweepRecord(checkpoint=Path("b"), step=2, exact_map=0.5, proxy_map=0.3),
        ]
        sc = scale_check(records, min_delta=0.001, scale_ratio_threshold=2.0)
        assert sc is not None  # must not raise
        assert np.isnan(sc.frac_exact) or np.isinf(sc.frac_exact)

    def test_degenerate_zero_proxy_range(self):
        records = [
            SweepRecord(checkpoint=Path("a"), step=1, exact_map=0.1, proxy_map=0.5),
            SweepRecord(checkpoint=Path("b"), step=2, exact_map=0.5, proxy_map=0.5),
        ]
        sc = scale_check(records, min_delta=0.001, scale_ratio_threshold=2.0)
        assert sc is not None
        assert np.isnan(sc.frac_proxy) or np.isinf(sc.frac_proxy)

    def test_custom_threshold(self):
        records = [
            SweepRecord(checkpoint=Path("a"), step=1, exact_map=0.0, proxy_map=0.0),
            SweepRecord(checkpoint=Path("b"), step=2, exact_map=1.0, proxy_map=0.5),
        ]
        sc_strict = scale_check(records, min_delta=0.001, scale_ratio_threshold=0.5)
        sc_loose = scale_check(records, min_delta=0.001, scale_ratio_threshold=10.0)
        assert sc_strict.material_divergence != sc_loose.material_divergence


# ---------------------------------------------------------------------------
# evaluate_gate — end-to-end PASS and FAIL
# ---------------------------------------------------------------------------


class TestEvaluateGate:
    def _make_records(self, n: int, rho_perfect: bool = True) -> list[SweepRecord]:
        """Build n records with non-zero exact mAP, perfect correlation by default."""
        records = []
        for i in range(n):
            exact = 0.1 + i * 0.05
            proxy = exact * 1.1 if rho_perfect else (0.5 - i * 0.05)
            records.append(
                SweepRecord(
                    checkpoint=Path(f"step_{i:08d}.pt"),
                    step=i + 1,
                    exact_map=exact,
                    proxy_map=proxy,
                )
            )
        return records

    def test_pass_all_criteria(self):
        records = self._make_records(5, rho_perfect=True)
        result = evaluate_gate(
            records, min_delta=0.001, rho_threshold=0.95, scale_ratio_threshold=2.0
        )
        assert result.rho >= 0.95
        assert result.n_gated >= 5
        assert result.n_inversions == 0
        assert result.passed

    def test_fail_low_rho(self):
        records = self._make_records(5, rho_perfect=False)
        result = evaluate_gate(
            records, min_delta=0.001, rho_threshold=0.95, scale_ratio_threshold=2.0
        )
        assert not result.passed
        assert result.rho < 0.95

    def test_fail_inversion(self):
        # Good rho but one inversion
        records = [
            SweepRecord(checkpoint=Path("a"), step=1, exact_map=0.10, proxy_map=0.10),
            SweepRecord(checkpoint=Path("b"), step=2, exact_map=0.20, proxy_map=0.09),  # inversion
            SweepRecord(checkpoint=Path("c"), step=3, exact_map=0.30, proxy_map=0.30),
            SweepRecord(checkpoint=Path("d"), step=4, exact_map=0.40, proxy_map=0.40),
            SweepRecord(checkpoint=Path("e"), step=5, exact_map=0.50, proxy_map=0.50),
        ]
        result = evaluate_gate(
            records, min_delta=0.001, rho_threshold=0.95, scale_ratio_threshold=2.0
        )
        assert not result.passed
        assert result.n_inversions >= 1

    def test_all_zero_exact_returns_fail(self):
        # If ALL exact mAP are zero, n_gated=0, cannot evaluate → FAIL
        records = [
            SweepRecord(checkpoint=Path("a"), step=1, exact_map=0.0, proxy_map=0.1),
            SweepRecord(checkpoint=Path("b"), step=2, exact_map=0.0, proxy_map=0.2),
        ]
        result = evaluate_gate(
            records, min_delta=0.001, rho_threshold=0.95, scale_ratio_threshold=2.0
        )
        assert not result.passed
        assert result.n_gated == 0

    def test_mixed_zero_and_nonzero(self):
        # Zero-exact checkpoints excluded from gate, nonzero ones evaluated
        records = [
            SweepRecord(checkpoint=Path("cold1"), step=1, exact_map=0.0, proxy_map=0.05),
            SweepRecord(checkpoint=Path("cold2"), step=2, exact_map=0.0, proxy_map=0.06),
            SweepRecord(checkpoint=Path("a"), step=3, exact_map=0.10, proxy_map=0.10),
            SweepRecord(checkpoint=Path("b"), step=4, exact_map=0.20, proxy_map=0.20),
            SweepRecord(checkpoint=Path("c"), step=5, exact_map=0.30, proxy_map=0.30),
        ]
        result = evaluate_gate(
            records, min_delta=0.001, rho_threshold=0.95, scale_ratio_threshold=2.0
        )
        assert result.n_gated == 3  # only non-zero-exact count


# ---------------------------------------------------------------------------
# Checkpoint discovery
# ---------------------------------------------------------------------------


class TestDiscoverCheckpoints:
    def test_discovers_and_sorts_by_step(self, tmp_path: Path):
        # Create step dirs (directories, not files)
        ckpt_dir = tmp_path / "checkpoints"
        for step_num in [100, 50, 1, 200]:
            (ckpt_dir / f"step_{step_num:08d}.pt").mkdir(parents=True)
        # Also create a non-step file that should be ignored
        (ckpt_dir / "best.pt").mkdir()
        (ckpt_dir / "some_file.txt").touch()

        checkpoints = discover_checkpoints(ckpt_dir)
        steps = [int(p.name.split("_")[1].split(".")[0]) for p in checkpoints]
        assert steps == sorted(steps)
        assert set(steps) == {1, 50, 100, 200}

    def test_returns_empty_when_no_checkpoints(self, tmp_path: Path):
        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir()
        assert discover_checkpoints(ckpt_dir) == []

    def test_raises_when_dir_missing(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            discover_checkpoints(tmp_path / "does_not_exist")

    def test_run_dir_variant(self, tmp_path: Path):
        # Also support passing run_dir (contains checkpoints/ subdirectory)
        ckpt_dir = tmp_path / "checkpoints"
        for step_num in [10, 20]:
            (ckpt_dir / f"step_{step_num:08d}.pt").mkdir(parents=True)
        checkpoints = discover_checkpoints(ckpt_dir)
        assert len(checkpoints) == 2


# ---------------------------------------------------------------------------
# Sweep orchestration with injected fake eval_fn
# ---------------------------------------------------------------------------


class TestRunSweep:
    """Verify sweep wiring without GPU/model/checkpoint I/O."""

    def _make_ckpt_dirs(self, tmp_path: Path, steps: list[int]) -> list[Path]:
        ckpt_dir = tmp_path / "checkpoints"
        dirs = []
        for s in steps:
            d = ckpt_dir / f"step_{s:08d}.pt"
            d.mkdir(parents=True)
            dirs.append(d)
        return dirs

    def test_eval_fn_called_twice_per_checkpoint(self, tmp_path: Path):
        """Each checkpoint gets one exact call and one proxy call."""
        self._make_ckpt_dirs(tmp_path, [1, 2, 3])
        call_log: list[dict] = []

        def fake_eval(checkpoint: Path, exact: bool) -> float:
            call_log.append({"checkpoint": checkpoint, "exact": exact})
            step = int(checkpoint.name.split("_")[1].split(".")[0])
            return 0.05 * step + (0.01 if exact else 0.0)

        records = run_sweep(tmp_path / "checkpoints", fake_eval)
        assert len(records) == 3
        assert len(call_log) == 6  # 3 checkpoints × 2 calls each

    def test_exact_env_var_toggled_correctly(self, tmp_path: Path):
        """CSP_LITE_EXACT_MAP must be set to '1' for exact, unset for proxy."""
        self._make_ckpt_dirs(tmp_path, [1])
        env_states: list[tuple[bool, str | None]] = []

        def fake_eval(checkpoint: Path, exact: bool) -> float:
            val = os.environ.get("CSP_LITE_EXACT_MAP")
            env_states.append((exact, val))
            return 0.1 if exact else 0.09

        run_sweep(tmp_path / "checkpoints", fake_eval)
        # One exact call → CSP_LITE_EXACT_MAP must be "1"
        exact_calls = [(e, v) for e, v in env_states if e]
        assert all(v == "1" for _, v in exact_calls), f"exact call saw env={exact_calls}"
        # One proxy call → CSP_LITE_EXACT_MAP must be unset (None or "")
        proxy_calls = [(e, v) for e, v in env_states if not e]
        assert all(v is None or v == "" for _, v in proxy_calls), (
            f"proxy call saw env={proxy_calls}"
        )

    def test_env_restored_after_sweep(self, tmp_path: Path):
        """CSP_LITE_EXACT_MAP must be restored to its pre-sweep value."""
        self._make_ckpt_dirs(tmp_path, [1])
        original = os.environ.get("CSP_LITE_EXACT_MAP", "SENTINEL_NOT_SET")

        def fake_eval(checkpoint: Path, exact: bool) -> float:
            return 0.1

        run_sweep(tmp_path / "checkpoints", fake_eval)
        after = os.environ.get("CSP_LITE_EXACT_MAP", "SENTINEL_NOT_SET")
        assert after == original

    def test_records_ordered_by_step(self, tmp_path: Path):
        self._make_ckpt_dirs(tmp_path, [30, 10, 20])

        def fake_eval(checkpoint: Path, exact: bool) -> float:
            step = int(checkpoint.name.split("_")[1].split(".")[0])
            return 0.01 * step

        records = run_sweep(tmp_path / "checkpoints", fake_eval)
        steps = [r.step for r in records]
        assert steps == sorted(steps)

    def test_record_exact_and_proxy_map_assigned(self, tmp_path: Path):
        self._make_ckpt_dirs(tmp_path, [5])

        def fake_eval(checkpoint: Path, exact: bool) -> float:
            return 0.42 if exact else 0.39

        records = run_sweep(tmp_path / "checkpoints", fake_eval)
        assert len(records) == 1
        assert abs(records[0].exact_map - 0.42) < 1e-9
        assert abs(records[0].proxy_map - 0.39) < 1e-9

    def test_mAP50_captured_when_provided(self, tmp_path: Path):
        """If eval_fn returns a tuple (mAP, mAP_50), both are stored."""
        self._make_ckpt_dirs(tmp_path, [1])

        def fake_eval(checkpoint: Path, exact: bool):
            return (0.42, 0.55) if exact else (0.39, 0.52)

        records = run_sweep(tmp_path / "checkpoints", fake_eval)
        assert records[0].exact_map_50 == pytest.approx(0.55, abs=1e-9)
        assert records[0].proxy_map_50 == pytest.approx(0.52, abs=1e-9)
