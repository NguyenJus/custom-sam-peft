"""CPU unit tests for the capability-named GPU tier taxonomy.

Tests _satisfied_tiers() set probe and _has_compatible_gpu() CC 7.5 gate.
The 5070 Ti (CC 12.0, 16 GB) satisfies gpu_t4 AND gpu_bf16; the T4 (CC 7.5,
16 GB) satisfies only gpu_t4; CC 6.1 (below CC 7.5 floor) satisfies neither.
"""

from __future__ import annotations

import pytest

import tests.conftest as cf

_GB = 1024**3


def _stub_cuda(monkeypatch, *, available=True, cap=(12, 0), total_gb=16, can_launch=True):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: available)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: cap)

    class _Props:
        total_memory = int(total_gb * _GB)

    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda *a, **k: _Props())
    monkeypatch.setattr(cf, "_torch_can_launch_kernel", lambda *a, **k: can_launch)


# ---------------------------------------------------------------------------
# _satisfied_tiers() parametric cases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cap,total_gb,expected",
    [
        ((12, 0), 16, {"gpu_t4", "gpu_bf16"}),  # 5070 Ti
        ((7, 5), 16, {"gpu_t4"}),  # T4
        ((6, 1), 8, set()),  # CC 6.1 -> nothing (below CC 7.5 floor)
        ((8, 0), 24, {"gpu_xl"}),  # >16 GB -> xl only
    ],
)
def test_satisfied_tiers(monkeypatch, cap, total_gb, expected):
    _stub_cuda(monkeypatch, cap=cap, total_gb=total_gb)
    assert cf._satisfied_tiers() == expected


def test_satisfied_tiers_empty_without_cuda(monkeypatch):
    _stub_cuda(monkeypatch, available=False)
    assert cf._satisfied_tiers() == set()


# ---------------------------------------------------------------------------
# _has_compatible_gpu() gate must be CC 7.5 (not CC 6.0)
# ---------------------------------------------------------------------------


def test_has_compatible_gpu_gate_is_cc_75(monkeypatch):
    _stub_cuda(monkeypatch, cap=(6, 1))
    assert cf._has_compatible_gpu() is False
    _stub_cuda(monkeypatch, cap=(7, 5))
    assert cf._has_compatible_gpu() is True
    _stub_cuda(monkeypatch, cap=(12, 0))
    assert cf._has_compatible_gpu() is True


# ---------------------------------------------------------------------------
# Skip-predicate cases via _should_skip pure helper
# ---------------------------------------------------------------------------


def _make_skip_predicate_cases():
    """Return (marker_tier, satisfied_set, should_be_skipped, reason_substr)."""
    return [
        # gpu_t4 card: t4 runs, bf16 skips (names CC>=8.0 gate)
        ("gpu_t4", {"gpu_t4"}, False, None),
        ("gpu_bf16", {"gpu_t4"}, True, "8.0"),
        # 5070 Ti: both run
        ("gpu_t4", {"gpu_t4", "gpu_bf16"}, False, None),
        ("gpu_bf16", {"gpu_t4", "gpu_bf16"}, False, None),
        # no GPU: all tier items skip
        ("gpu_t4", set(), True, None),
        ("gpu_bf16", set(), True, None),
        ("gpu_xl", set(), True, None),
        # <=16 GB card: gpu_xl skips with a reason naming >16 GB (no #124)
        ("gpu_xl", {"gpu_t4", "gpu_bf16"}, True, "16"),
    ]


@pytest.mark.parametrize(
    "marker_tier,satisfied,should_skip,reason_substr",
    _make_skip_predicate_cases(),
)
def test_should_skip_helper(marker_tier, satisfied, should_skip, reason_substr):
    """_should_skip returns a skip-reason string when the tier is unmet, else None."""
    result = cf._should_skip(marker_tier, satisfied)
    if should_skip:
        assert result is not None, f"expected skip for {marker_tier!r} on {satisfied!r}"
        if reason_substr is not None:
            assert reason_substr in result, f"expected {reason_substr!r} in {result!r}"
    else:
        assert result is None, (
            f"expected no skip for {marker_tier!r} on {satisfied!r}, got {result!r}"
        )
