"""Unit tests for custom_sam_peft.profiling_report (issue #256 / #273).

CPU-only — pure snapshot-in / report-out, no GPU, no torch required.

Run with:
    uv run pytest tests/unit/test_profiling_report.py -o "addopts=" -p no:cacheprovider -q
"""

from __future__ import annotations

import pytest

from custom_sam_peft.profiling_report import (
    DominantPath,
    attribute_snapshot,
    compare_snapshots,
    render_report,
)

# ---------------------------------------------------------------------------
# Helpers — synthetic snapshots
# ---------------------------------------------------------------------------


def _eval_snap_rle_dominant() -> dict:
    """eval.total parent present; rle_encode dominates (~62%) and N > 100."""
    return {
        "buckets": {
            "eval.total": 10.0,
            "eval.forward": 2.0,
            "eval.mask_upsample": 0.5,
            "eval.transfer_binarize": 0.3,
            "eval.rle_encode": 6.2,
            "eval.gt_rle_encode": 0.5,
            "eval.coco_aggregate": 0.2,
            "eval.dataset_load": 0.3,
        },
        "meta": {
            "N": 200,
            "eval.forwards": 40,
            "mask_logit_hw": [256, 256],
            "n_images": 40,
        },
    }


def _eval_snap_rle_below_threshold() -> dict:
    """rle_encode share < 5% → GO verdict should not fire."""
    return {
        "buckets": {
            "eval.total": 10.0,
            "eval.forward": 9.0,
            "eval.rle_encode": 0.4,
            "eval.coco_aggregate": 0.2,
            "eval.dataset_load": 0.1,
        },
        "meta": {
            "N": 300,
            "eval.forwards": 50,
            "n_images": 50,
        },
    }


def _eval_snap_n_too_small() -> dict:
    """N ≤ 100 → top-100 filter GO should not fire."""
    return {
        "buckets": {
            "eval.total": 10.0,
            "eval.rle_encode": 7.0,
            "eval.forward": 2.5,
            "eval.coco_aggregate": 0.2,
        },
        "meta": {
            "N": 50,
            "eval.forwards": 20,
            "n_images": 20,
        },
    }


def _eval_snap_no_total_parent() -> dict:
    """No eval.total bucket — denominator must be sum of leaves."""
    return {
        "buckets": {
            "eval.forward": 5.0,
            "eval.rle_encode": 3.0,
            "eval.coco_aggregate": 2.0,
        },
        "meta": {
            "N": 150,
        },
    }


def _semantic_eval_snap() -> dict:
    """semantic_eval.total parent; transfer + confusion are the heavy CPU spans."""
    return {
        "buckets": {
            "semantic_eval.total": 8.0,
            "semantic_eval.forward": 3.0,
            "semantic_eval.upsample": 1.0,
            "semantic_eval.transfer": 2.5,
            "semantic_eval.confusion": 1.3,
        },
        "meta": {
            "N": 16,
            "K": 16,
            "semantic_eval.forwards": 10,
            "sem_forward_dtype": "float32",
            "n_images": 10,
        },
    }


def _train_snap() -> dict:
    """train surface; no *.total parent."""
    return {
        "buckets": {
            "train.forward": 6.0,
            "train.loss": 1.0,
            "train.matcher": 2.0,
            "train.backward": 3.0,
            "train.optim_step": 0.5,
        },
        "meta": {
            "N": 8,
        },
    }


def _snap_missing_meta() -> dict:
    """All meta keys absent — must not raise."""
    return {
        "buckets": {
            "eval.total": 5.0,
            "eval.forward": 4.0,
            "eval.rle_encode": 0.8,
        },
        "meta": {},
    }


# ---------------------------------------------------------------------------
# A. Bucket ranking + denominator rule
# ---------------------------------------------------------------------------


class TestBucketRanking:
    def test_eval_total_is_denominator_not_leaf(self) -> None:
        """eval.total must be used as wall denominator, not added as a leaf."""
        data = attribute_snapshot(_eval_snap_rle_dominant())
        names = [r.name for r in data.rows]
        # eval.total must NOT appear in the ranked leaf rows
        assert "eval.total" not in names

    def test_rle_encode_share_uses_total_as_denom(self) -> None:
        """rle_encode share = 6.2 / 10.0 = 62%."""
        data = attribute_snapshot(_eval_snap_rle_dominant())
        rle_row = next(r for r in data.rows if r.name == "eval.rle_encode")
        assert abs(rle_row.share - 0.62) < 0.01

    def test_residual_surfaced_when_total_present(self) -> None:
        """Residual = total (10.0) - sum(leaves: 2+0.5+0.3+6.2+0.5+0.2+0.3=10.0) = 0.0."""
        snap = _eval_snap_rle_dominant()
        # Use a snap with a gap so residual != 0
        snap["buckets"]["eval.total"] = 12.0
        data = attribute_snapshot(snap)
        # Residual should be positive (12 - 10 = 2)
        assert data.residual_seconds > 0.0

    def test_rows_sorted_by_share_descending(self) -> None:
        """Leaf rows must be sorted by share descending."""
        data = attribute_snapshot(_eval_snap_rle_dominant())
        shares = [r.share for r in data.rows]
        assert shares == sorted(shares, reverse=True)

    def test_no_total_parent_uses_sum_of_leaves(self) -> None:
        """Without *.total, denominator = sum of all leaves."""
        data = attribute_snapshot(_eval_snap_no_total_parent())
        # All shares should sum to 1.0 (no residual)
        total_share = sum(r.share for r in data.rows)
        assert abs(total_share - 1.0) < 1e-6

    def test_no_total_parent_residual_is_zero(self) -> None:
        data = attribute_snapshot(_eval_snap_no_total_parent())
        assert data.residual_seconds == pytest.approx(0.0, abs=1e-9)

    def test_surface_detected_eval(self) -> None:
        data = attribute_snapshot(_eval_snap_rle_dominant())
        assert data.surface == "eval"

    def test_surface_detected_semantic_eval(self) -> None:
        data = attribute_snapshot(_semantic_eval_snap())
        assert data.surface == "semantic_eval"

    def test_surface_detected_train(self) -> None:
        data = attribute_snapshot(_train_snap())
        assert data.surface == "train"

    def test_wall_label_when_total_present(self) -> None:
        data = attribute_snapshot(_eval_snap_rle_dominant())
        assert data.pct_label == "% of wall"
        assert data.total_label == "TOTAL(wall)"

    def test_timed_label_when_no_total(self) -> None:
        data = attribute_snapshot(_eval_snap_no_total_parent())
        assert data.pct_label == "% of timed"
        assert data.total_label == "TOTAL(timed)"


# ---------------------------------------------------------------------------
# B. Dominant path + CPU/GPU/sync classification
# ---------------------------------------------------------------------------


class TestDominantPath:
    def test_dominant_bucket_identified(self) -> None:
        data = attribute_snapshot(_eval_snap_rle_dominant())
        # rle_encode is the largest leaf by share
        assert data.dominant.name == "eval.rle_encode"

    def test_dominant_bucket_class_cpu(self) -> None:
        data = attribute_snapshot(_eval_snap_rle_dominant())
        assert data.dominant.kind == "cpu"

    def test_gpu_bucket_classified_correctly(self) -> None:
        data = attribute_snapshot(_eval_snap_rle_dominant())
        forward_row = next(r for r in data.rows if r.name == "eval.forward")
        assert forward_row.kind == "gpu"

    def test_sync_bucket_classified_correctly(self) -> None:
        data = attribute_snapshot(_eval_snap_rle_dominant())
        tb_row = next(r for r in data.rows if r.name == "eval.transfer_binarize")
        assert tb_row.kind == "sync"

    def test_io_bucket_classified_correctly(self) -> None:
        data = attribute_snapshot(_eval_snap_rle_dominant())
        ds_row = next(r for r in data.rows if r.name == "eval.dataset_load")
        assert ds_row.kind == "io"

    def test_unknown_bucket_is_unclassified(self) -> None:
        snap = _eval_snap_rle_dominant()
        snap["buckets"]["eval.some_new_thing"] = 0.5
        data = attribute_snapshot(snap)
        new_row = next(r for r in data.rows if r.name == "eval.some_new_thing")
        assert new_row.kind == "unclassified"

    def test_split_cpu_gpu_sync_io(self) -> None:
        """CPU/GPU/sync/IO shares must sum ≤ 1.0 and match classified buckets."""
        data = attribute_snapshot(_eval_snap_rle_dominant())
        dp = data.dominant
        assert isinstance(dp, DominantPath)
        # The split totals must not exceed 1.0
        total_split = dp.gpu_share + dp.cpu_share + dp.sync_share + dp.io_share
        assert total_split <= 1.0 + 1e-9

    def test_semantic_eval_gpu_forward_classified(self) -> None:
        data = attribute_snapshot(_semantic_eval_snap())
        fwd_row = next(r for r in data.rows if r.name == "semantic_eval.forward")
        assert fwd_row.kind == "gpu"

    def test_semantic_eval_transfer_is_sync(self) -> None:
        data = attribute_snapshot(_semantic_eval_snap())
        tx_row = next(r for r in data.rows if r.name == "semantic_eval.transfer")
        assert tx_row.kind == "sync"

    def test_train_matcher_is_cpu(self) -> None:
        data = attribute_snapshot(_train_snap())
        m_row = next(r for r in data.rows if r.name == "train.matcher")
        assert m_row.kind == "cpu"

    def test_train_backward_is_gpu(self) -> None:
        """Autograd backward is GPU compute, not CPU — the split must reflect that."""
        data = attribute_snapshot(_train_snap())
        bw = next(r for r in data.rows if r.name == "train.backward")
        assert bw.kind == "gpu"


# ---------------------------------------------------------------------------
# C. Structural facts from meta
# ---------------------------------------------------------------------------


class TestStructuralFacts:
    def test_n_extracted(self) -> None:
        data = attribute_snapshot(_eval_snap_rle_dominant())
        assert data.facts["N"] == 200

    def test_forwards_extracted(self) -> None:
        data = attribute_snapshot(_eval_snap_rle_dominant())
        assert data.facts["forwards"] == 40

    def test_n_images_extracted(self) -> None:
        data = attribute_snapshot(_eval_snap_rle_dominant())
        assert data.facts["n_images"] == 40

    def test_mask_logit_hw_extracted(self) -> None:
        data = attribute_snapshot(_eval_snap_rle_dominant())
        assert data.facts["mask_logit_hw"] == [256, 256]

    def test_forwards_per_image_derived(self) -> None:
        """forwards_per_image = forwards / n_images = 40 / 40 = 1.0."""
        data = attribute_snapshot(_eval_snap_rle_dominant())
        assert data.facts["forwards_per_image"] == pytest.approx(1.0)

    def test_sem_forward_dtype_extracted(self) -> None:
        data = attribute_snapshot(_semantic_eval_snap())
        assert data.facts["sem_forward_dtype"] == "float32"

    def test_semantic_forwards_key(self) -> None:
        data = attribute_snapshot(_semantic_eval_snap())
        assert data.facts["forwards"] == 10

    def test_K_extracted_for_semantic(self) -> None:
        """K (semantic class count) is a §3b-required meta key and must surface."""
        data = attribute_snapshot(_semantic_eval_snap())
        assert data.facts["K"] == 16

    def test_K_unknown_when_missing(self) -> None:
        data = attribute_snapshot(_snap_missing_meta())
        assert data.facts["K"] == "unknown"

    def test_missing_meta_keys_degrade_to_unknown(self) -> None:
        data = attribute_snapshot(_snap_missing_meta())
        assert data.facts["N"] == "unknown"
        assert data.facts["forwards"] == "unknown"
        assert data.facts["n_images"] == "unknown"

    def test_missing_mask_logit_hw_is_unknown(self) -> None:
        data = attribute_snapshot(_snap_missing_meta())
        assert data.facts["mask_logit_hw"] == "unknown"

    def test_forwards_per_image_unknown_when_n_images_missing(self) -> None:
        data = attribute_snapshot(_snap_missing_meta())
        assert data.facts["forwards_per_image"] == "unknown"

    def test_no_crash_on_completely_empty_snap(self) -> None:
        data = attribute_snapshot({"buckets": {}, "meta": {}})
        assert data is not None
        assert data.facts["N"] == "unknown"


# ---------------------------------------------------------------------------
# D. GO / NO-GO heuristics
# ---------------------------------------------------------------------------


class TestGoNoGo:
    def test_rle_dominant_and_n_large_gives_go(self) -> None:
        """rle_encode share ≥ 5% AND N > 100 → top-100 query-filter is GO."""
        data = attribute_snapshot(_eval_snap_rle_dominant())
        rle_verdict = next(
            v
            for v in data.verdicts
            if "rle" in v.rule_id.lower() or "postprocess" in v.rule_id.lower()
        )
        assert rle_verdict.verdict == "GO"

    def test_rle_below_threshold_gives_nogo(self) -> None:
        data = attribute_snapshot(_eval_snap_rle_below_threshold())
        rle_verdicts = [
            v
            for v in data.verdicts
            if "rle" in v.rule_id.lower() or "postprocess" in v.rule_id.lower()
        ]
        assert all(v.verdict == "NO-GO" for v in rle_verdicts)

    def test_n_too_small_gives_nogo(self) -> None:
        data = attribute_snapshot(_eval_snap_n_too_small())
        rle_verdicts = [
            v
            for v in data.verdicts
            if "rle" in v.rule_id.lower() or "postprocess" in v.rule_id.lower()
        ]
        assert all(v.verdict == "NO-GO" for v in rle_verdicts)

    def test_custom_threshold_respected(self) -> None:
        """With a very high threshold (0.9), even rle_dominant snap gives NO-GO."""
        data = attribute_snapshot(_eval_snap_rle_dominant(), rle_threshold=0.9)
        rle_verdicts = [
            v
            for v in data.verdicts
            if "rle" in v.rule_id.lower() or "postprocess" in v.rule_id.lower()
        ]
        assert all(v.verdict == "NO-GO" for v in rle_verdicts)

    def test_verdict_cites_rule(self) -> None:
        """Every verdict must carry a non-empty citation field."""
        data = attribute_snapshot(_eval_snap_rle_dominant())
        for v in data.verdicts:
            assert v.citation, f"verdict {v.rule_id!r} has empty citation"

    def test_no_crash_on_non_eval_surface(self) -> None:
        """GO/NO-GO heuristics must not crash on a train snapshot."""
        data = attribute_snapshot(_train_snap())
        assert isinstance(data.verdicts, list)

    def test_rle_dominant_but_n_unknown_reason_does_not_claim_n_small(self) -> None:
        """RLE dominant + N unknown → NO-GO, but the reason must NOT assert N ≤ 100.

        This verdict lands verbatim in the triage doc; claiming a false fact
        ('N=None ≤ 100') would mislead. _snap_missing_meta has rle share 16%
        (dominant) and no N key.
        """
        data = attribute_snapshot(_snap_missing_meta())
        rle_verdict = next(
            v
            for v in data.verdicts
            if "rle" in v.rule_id.lower() or "postprocess" in v.rule_id.lower()
        )
        assert rle_verdict.verdict == "NO-GO"
        assert "≤ 100" not in rle_verdict.reason and "<= 100" not in rle_verdict.reason
        assert "unknown" in rle_verdict.reason.lower()


# ---------------------------------------------------------------------------
# E. Regression detection
# ---------------------------------------------------------------------------


class TestRegressionDetect:
    def _baseline(self) -> dict:
        return {
            "buckets": {
                "eval.total": 10.0,
                "eval.forward": 4.0,
                "eval.rle_encode": 5.0,
                "eval.coco_aggregate": 0.8,
            },
            "meta": {"N": 100},
        }

    def _current_grown_rle(self) -> dict:
        """rle_encode grew from 5.0 (50%) to 7.0 (70%) = +20pp share."""
        return {
            "buckets": {
                "eval.total": 10.0,
                "eval.forward": 2.0,
                "eval.rle_encode": 7.0,
                "eval.coco_aggregate": 0.8,
            },
            "meta": {"N": 100},
        }

    def _current_unchanged(self) -> dict:
        """Same as baseline — no regressions."""
        return dict(self._baseline())

    def test_grown_bucket_flagged(self) -> None:
        flags = compare_snapshots(self._baseline(), self._current_grown_rle())
        assert any(f.name == "eval.rle_encode" for f in flags)

    def test_grown_bucket_shows_delta(self) -> None:
        flags = compare_snapshots(self._baseline(), self._current_grown_rle())
        rle_flag = next(f for f in flags if f.name == "eval.rle_encode")
        # share grew by 0.2 (50% → 70%)
        assert rle_flag.share_delta > 0.0

    def test_unchanged_snap_no_flags(self) -> None:
        flags = compare_snapshots(self._baseline(), self._current_unchanged())
        assert flags == []

    def test_custom_tolerance_suppresses_small_delta(self) -> None:
        """A +5pp change is suppressed when tolerance=0.10."""
        baseline = self._baseline()
        current = {
            "buckets": {
                "eval.total": 10.0,
                "eval.forward": 3.5,
                "eval.rle_encode": 5.5,  # +0.05 share change
                "eval.coco_aggregate": 0.8,
            },
            "meta": {"N": 100},
        }
        flags = compare_snapshots(baseline, current, share_tolerance=0.10)
        assert not any(f.name == "eval.rle_encode" for f in flags)

    def test_new_bucket_in_current_does_not_crash(self) -> None:
        current = self._current_unchanged()
        current["buckets"]["eval.new_thing"] = 1.0
        current["buckets"]["eval.total"] = 11.0
        flags = compare_snapshots(self._baseline(), current)
        # Should not crash; new bucket may or may not be flagged depending on share
        assert isinstance(flags, list)

    def test_bucket_disappeared_in_current_does_not_crash(self) -> None:
        current = {
            "buckets": {
                "eval.total": 10.0,
                "eval.forward": 4.0,
                "eval.coco_aggregate": 0.8,
                # eval.rle_encode removed
            },
            "meta": {"N": 100},
        }
        flags = compare_snapshots(self._baseline(), current)
        assert isinstance(flags, list)


# ---------------------------------------------------------------------------
# F. Report skeleton rendering
# ---------------------------------------------------------------------------


class TestReportRendering:
    def test_report_is_string(self) -> None:
        data = attribute_snapshot(_eval_snap_rle_dominant())
        md = render_report(data)
        assert isinstance(md, str)

    def test_report_has_bucket_table_section(self) -> None:
        data = attribute_snapshot(_eval_snap_rle_dominant())
        md = render_report(data)
        assert "Bucket Ranking" in md or "bucket" in md.lower()

    def test_report_has_dominant_path_section(self) -> None:
        data = attribute_snapshot(_eval_snap_rle_dominant())
        md = render_report(data)
        assert "Dominant" in md or "dominant" in md

    def test_report_has_structural_facts_section(self) -> None:
        data = attribute_snapshot(_eval_snap_rle_dominant())
        md = render_report(data)
        assert "Structural" in md or "structural" in md.lower()

    def test_report_has_go_nogo_section(self) -> None:
        data = attribute_snapshot(_eval_snap_rle_dominant())
        md = render_report(data)
        assert "GO" in md

    def test_report_contains_bucket_names(self) -> None:
        data = attribute_snapshot(_eval_snap_rle_dominant())
        md = render_report(data)
        assert "eval.rle_encode" in md

    def test_report_contains_surface_name(self) -> None:
        data = attribute_snapshot(_eval_snap_rle_dominant())
        md = render_report(data)
        assert "eval" in md

    def test_report_has_regression_section(self) -> None:
        data = attribute_snapshot(_eval_snap_rle_dominant())
        md = render_report(data)
        assert "Regression" in md or "regression" in md.lower()

    def test_report_with_regression_flags(self) -> None:
        baseline = {
            "buckets": {
                "eval.total": 10.0,
                "eval.rle_encode": 5.0,
                "eval.forward": 4.0,
                "eval.coco_aggregate": 0.8,
            },
            "meta": {"N": 100},
        }
        current_snap = _eval_snap_rle_dominant()
        data = attribute_snapshot(current_snap)
        flags = compare_snapshots(baseline, current_snap)
        md = render_report(data, regression_flags=flags)
        assert isinstance(md, str)
        assert "eval.rle_encode" in md

    def test_report_no_crash_on_empty_snapshot(self) -> None:
        data = attribute_snapshot({"buckets": {}, "meta": {}})
        md = render_report(data)
        assert isinstance(md, str)

    def test_report_n_value_in_facts(self) -> None:
        data = attribute_snapshot(_eval_snap_rle_dominant())
        md = render_report(data)
        assert "200" in md  # N = 200

    def test_report_semantic_forward_dtype(self) -> None:
        data = attribute_snapshot(_semantic_eval_snap())
        md = render_report(data)
        assert "float32" in md
