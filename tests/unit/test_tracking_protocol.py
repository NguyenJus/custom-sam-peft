"""Tracker Protocol conformance — catches signature drift early."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from custom_sam_peft.tracking.base import Tracker
from custom_sam_peft.tracking.noop import NoopTracker


def test_noop_is_a_tracker() -> None:
    assert isinstance(NoopTracker(), Tracker)


def test_missing_start_run_is_not_a_tracker() -> None:
    class Incomplete:
        def log_scalars(self, step: int, values: dict[str, float]) -> None:
            pass

        def log_images(self, step: int, images: dict[str, np.ndarray[Any, Any]]) -> None:
            pass

        def close(self) -> None:
            pass

    assert not isinstance(Incomplete(), Tracker)


def test_missing_close_is_not_a_tracker() -> None:
    class Incomplete:
        def start_run(
            self,
            run_dir: Path,
            config: dict[str, Any],
            resume_from: Path | None = None,
        ) -> None:
            pass

        def log_scalars(self, step: int, values: dict[str, float]) -> None:
            pass

        def log_images(self, step: int, images: dict[str, np.ndarray[Any, Any]]) -> None:
            pass

    assert not isinstance(Incomplete(), Tracker)


def test_noop_wants_images_is_false() -> None:
    assert NoopTracker.wants_images is False


def test_local_is_a_tracker_and_wants_images_false() -> None:
    from unittest.mock import MagicMock

    from custom_sam_peft.tracking.local import LocalTracker

    t = LocalTracker(MagicMock())
    assert isinstance(t, Tracker)
    assert t.wants_images is False


def test_missing_wants_images_is_not_a_tracker() -> None:
    class Incomplete:
        def start_run(
            self,
            run_dir: Path,
            config: dict[str, Any],
            resume_from: Path | None = None,
        ) -> None:
            pass

        def log_scalars(self, step: int, values: dict[str, float]) -> None:
            pass

        def log_images(self, step: int, images: dict[str, np.ndarray[Any, Any]]) -> None:
            pass

        def close(self) -> None:
            pass

    assert not isinstance(Incomplete(), Tracker)
