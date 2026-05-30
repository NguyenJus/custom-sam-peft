"""Schema tests for TrainHyperparams.time_limit (spec §11.2)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from custom_sam_peft.config.schema import TrainHyperparams


def test_time_limit_defaults_none() -> None:
    hp = TrainHyperparams(epochs=1)
    assert hp.time_limit is None


@pytest.mark.parametrize(
    ("value", "expected"),
    [("2h30m", "2h30m"), (3600, 3600), ("3600", "3600"), ("1h5m30s", "1h5m30s")],
)
def test_time_limit_stored_verbatim(value: str | int, expected: str | int) -> None:
    """The field is validated but NOT normalized to seconds; it echoes what was passed."""
    hp = TrainHyperparams(epochs=1, time_limit=value)
    assert hp.time_limit == expected
    assert type(hp.time_limit) is type(expected)


@pytest.mark.parametrize("value", [0, -5, "abc", "10x", "", True, False])
def test_time_limit_rejected(value: str | int) -> None:
    with pytest.raises(ValidationError):
        TrainHyperparams(epochs=1, time_limit=value)


def test_time_limit_error_names_bad_value() -> None:
    with pytest.raises(ValidationError, match="10x"):
        TrainHyperparams(epochs=1, time_limit="10x")
