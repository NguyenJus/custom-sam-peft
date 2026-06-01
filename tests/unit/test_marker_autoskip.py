"""CPU tests for conftest GPU-compat floor and tier autoskip logic."""

from __future__ import annotations

import importlib

import pytest


def _conftest():
    return importlib.import_module("tests.conftest")


@pytest.mark.parametrize(
    ("cap", "expected"),
    [
        # CC 7.5 is the new floor — everything below is False
        ((6, 0), False),
        ((6, 1), False),
        # At and above the floor
        ((7, 5), True),
        ((8, 0), True),
        # CC 5.0 is below the CC 7.5 floor — fails at the gate
        ((5, 0), False),
    ],
)
def test_has_compatible_gpu_floor_is_cc75(monkeypatch, cap, expected) -> None:
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *_a, **_k: cap)
    monkeypatch.setattr(_conftest(), "_torch_can_launch_kernel", lambda: True)
    assert _conftest()._has_compatible_gpu() is expected


def test_compatible_gpu_false_when_kernel_unsupported(monkeypatch) -> None:
    """CC >= 7.5 but the installed torch build cannot launch a kernel -> not a usable GPU."""
    import torch

    conftest = _conftest()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *_a, **_k: (7, 5))
    monkeypatch.setattr(conftest, "_torch_can_launch_kernel", lambda: False)
    assert conftest._has_compatible_gpu() is False


class _FakeItem:
    def __init__(self, *keywords: str) -> None:
        self.keywords = set(keywords)
        self.markers: list[object] = []

    def add_marker(self, marker: object) -> None:
        self.markers.append(marker)


# ---------------------------------------------------------------------------
# New capability-subset skip predicate tests
# All tier-selection via monkeypatching _satisfied_tiers (NOT _current_tier).
# ---------------------------------------------------------------------------


def test_gpu_t4_runs_on_t4_runner(monkeypatch) -> None:
    """A gpu_t4 test runs when the runner satisfies gpu_t4."""
    import tests.conftest as conftest

    monkeypatch.setattr(conftest, "_has_compatible_gpu", lambda: True)
    monkeypatch.setattr(conftest, "_satisfied_tiers", lambda: {"gpu_t4"})
    item = _FakeItem("gpu_t4", "requires_compatible_gpu")
    conftest.pytest_collection_modifyitems(config=None, items=[item])  # type: ignore[arg-type]
    assert not item.markers, "gpu_t4 test should not be skipped when gpu_t4 is satisfied"


def test_gpu_bf16_skipped_on_t4_runner(monkeypatch) -> None:
    """A gpu_bf16 test is skipped when the runner only satisfies gpu_t4 (no bf16 native)."""
    import tests.conftest as conftest

    monkeypatch.setattr(conftest, "_has_compatible_gpu", lambda: True)
    monkeypatch.setattr(conftest, "_satisfied_tiers", lambda: {"gpu_t4"})
    item = _FakeItem("gpu_bf16", "requires_compatible_gpu")
    conftest.pytest_collection_modifyitems(config=None, items=[item])  # type: ignore[arg-type]
    assert item.markers, "gpu_bf16 test was not skipped on a gpu_t4-only runner"
    reason = getattr(item.markers[0], "kwargs", {}).get("reason", "")
    assert "8.0" in reason, f"skip reason should name CC>=8.0 gate; got: {reason!r}"


def test_both_tiers_run_on_5070ti(monkeypatch) -> None:
    """On the RTX 5070 Ti both gpu_t4 and gpu_bf16 tests run."""
    import tests.conftest as conftest

    monkeypatch.setattr(conftest, "_has_compatible_gpu", lambda: True)
    monkeypatch.setattr(conftest, "_satisfied_tiers", lambda: {"gpu_t4", "gpu_bf16"})
    for tier in ("gpu_t4", "gpu_bf16"):
        item = _FakeItem(tier, "requires_compatible_gpu")
        conftest.pytest_collection_modifyitems(config=None, items=[item])  # type: ignore[arg-type]
        assert not item.markers, f"{tier} test should not be skipped on 5070 Ti runner"


def test_all_tiers_skipped_without_gpu(monkeypatch) -> None:
    """With no compatible GPU all tier-marked tests are skipped."""
    import tests.conftest as conftest

    monkeypatch.setattr(conftest, "_has_compatible_gpu", lambda: False)
    monkeypatch.setattr(conftest, "_satisfied_tiers", lambda: set())
    for tier in ("gpu_t4", "gpu_bf16", "gpu_xl"):
        item = _FakeItem(tier, "requires_compatible_gpu")
        conftest.pytest_collection_modifyitems(config=None, items=[item])  # type: ignore[arg-type]
        assert item.markers, f"{tier} test was not skipped when no GPU available"


def test_gpu_xl_skipped_on_16gb_card(monkeypatch) -> None:
    """gpu_xl is skipped on a <=16 GB card; reason names the >16 GB gate (no #124)."""
    import tests.conftest as conftest

    monkeypatch.setattr(conftest, "_has_compatible_gpu", lambda: True)
    monkeypatch.setattr(conftest, "_satisfied_tiers", lambda: {"gpu_t4", "gpu_bf16"})
    item = _FakeItem("gpu_xl", "requires_compatible_gpu")
    conftest.pytest_collection_modifyitems(config=None, items=[item])  # type: ignore[arg-type]
    assert item.markers, "gpu_xl test not skipped on <=16 GB runner"
    reason = getattr(item.markers[0], "kwargs", {}).get("reason", "")
    assert "16" in reason, f"skip reason should name >16 GB gate; got: {reason!r}"
    assert "#124" not in reason, f"skip reason must not reference retired #124; got: {reason!r}"
