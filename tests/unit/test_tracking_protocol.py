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
