"""Tests for the Tracker protocol and the noop implementation."""

from __future__ import annotations

import numpy as np
import pytest

from esam3._registry import RegistryError, list_registered, lookup
from esam3.tracking.base import Tracker
from esam3.tracking.noop import NoopTracker, build_noop  # noqa: F401


@pytest.fixture(autouse=True)
def _ensure_noop_registered() -> None:
    """Re-register the noop factory if a sibling test file cleared the registry."""
    import contextlib
    import importlib

    try:
        lookup("tracker", "none")
    except RegistryError:
        from esam3.tracking import noop as _noop_mod

        with contextlib.suppress(RegistryError):
            importlib.reload(_noop_mod)


def test_noop_tracker_conforms_to_protocol() -> None:
    t: Tracker = NoopTracker()
    t.log_scalars(0, {"loss": 1.0})
    t.log_images(0, {"sample": np.zeros((4, 4, 3), dtype=np.uint8)})
    t.close()


def test_noop_registered_under_tracker_kind() -> None:
    assert "none" in list_registered("tracker")
    factory = lookup("tracker", "none")
    instance = factory({})
    assert type(instance).__name__ == "NoopTracker"
    assert type(instance).__module__ == "esam3.tracking.noop"
