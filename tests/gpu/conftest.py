"""GPU-tier conftest: shared helpers for real-SAM3.1 smoke tests."""

from __future__ import annotations

from esam3.tracking.noop import NoopTracker


class _RecordingTracker(NoopTracker):
    """NoopTracker subclass that captures every (step, scalars) log call.

    Lifted from the inline definition in the previous version of
    test_real_train_overfits.py. Both GPU smoke tests share this instance shape;
    assertions on tracker.scalars are the test surface.
    """

    def __init__(self) -> None:
        self.scalars: list[tuple[int, dict[str, float]]] = []

    def log_scalars(self, step: int, values: dict[str, float]) -> None:
        self.scalars.append((step, values))

    def log_images(self, step: int, images: dict[str, object]) -> None:
        pass

    def close(self) -> None:
        pass


def _bnb_available() -> bool:
    """Return True iff bitsandbytes is importable. Lifted from
    tests/integration/test_peft_qlora_real.py."""
    try:
        import bitsandbytes  # noqa: F401
    except ImportError:
        return False
    return True
