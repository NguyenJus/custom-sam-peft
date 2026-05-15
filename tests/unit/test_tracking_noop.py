"""Tests for the Tracker protocol and the noop implementation."""

from __future__ import annotations

import numpy as np

from esam3._registry import lookup, reset_registry
from esam3.tracking.base import Tracker
from esam3.tracking.noop import NoopTracker, build_noop  # noqa: F401


def test_noop_tracker_conforms_to_protocol() -> None:
    t: Tracker = NoopTracker()
    t.log_scalars(0, {"loss": 1.0})
    t.log_images(0, {"sample": np.zeros((4, 4, 3), dtype=np.uint8)})
    t.close()


def test_noop_registered_under_tracker_kind() -> None:
    reset_registry()
    import importlib

    import esam3.tracking.noop as mod

    importlib.reload(mod)
    factory = lookup("tracker", "none")
    instance = factory({})
    assert isinstance(instance, NoopTracker)
