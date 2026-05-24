"""CPU tests for conftest GPU-compat floor and tier autoskip logic."""

from __future__ import annotations

import importlib

import pytest


def _conftest():
    return importlib.import_module("tests.conftest")


@pytest.mark.parametrize(
    ("cap", "expected"),
    [((6, 0), True), ((6, 1), True), ((7, 5), True), ((8, 0), True), ((5, 0), False)],
)
def test_has_compatible_gpu_floor_is_cc60(monkeypatch, cap, expected) -> None:
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *_a, **_k: cap)
    monkeypatch.setattr(_conftest(), "_torch_can_launch_kernel", lambda: True)
    assert _conftest()._has_compatible_gpu() is expected


def test_compatible_gpu_false_when_kernel_unsupported(monkeypatch) -> None:
    """CC >= 6.0 but the installed torch build cannot launch a kernel (e.g. cu130
    on a GTX 1080: sm_61 not in the cubin set) -> not a usable GPU."""
    import torch

    conftest = _conftest()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *_a, **_k: (6, 1))
    monkeypatch.setattr(conftest, "_torch_can_launch_kernel", lambda: False)
    assert conftest._has_compatible_gpu() is False
