"""Unit tests for the duration parser/formatter (spec §11.1)."""

from __future__ import annotations

import pytest

from custom_sam_peft.config._duration import format_seconds, parse_duration_to_seconds


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("2h30m", 9000),
        ("90m", 5400),
        ("3600s", 3600),
        (3600, 3600),
        ("3600", 3600),
        ("1h", 3600),
        ("45m", 2700),
        ("30s", 30),
        ("1h5m30s", 3930),
        ("  2h30m  ", 9000),  # surrounding whitespace tolerated
    ],
)
def test_parse_accepts(value: str | int, expected: int) -> None:
    assert parse_duration_to_seconds(value) == expected


@pytest.mark.parametrize(
    "value",
    [0, -1, "", "   ", "abc", "10x", "2h30", "-2h", "-5", "0s", "h", "1m2h", True, False],
)
def test_parse_rejects(value: str | int) -> None:
    with pytest.raises(ValueError):
        parse_duration_to_seconds(value)


def test_parse_error_names_the_bad_value() -> None:
    with pytest.raises(ValueError, match="10x"):
        parse_duration_to_seconds("10x")


def test_parse_error_names_bad_int() -> None:
    with pytest.raises(ValueError, match="-5"):
        parse_duration_to_seconds(-5)


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (9000, "2h30m"),
        (3600, "1h"),
        (5400, "1h30m"),  # canonical: 90m collapses to 1h30m
        (90, "1m30s"),
        (45, "45s"),
        (3930, "1h5m30s"),
    ],
)
def test_format(seconds: int, expected: str) -> None:
    assert format_seconds(seconds) == expected


@pytest.mark.parametrize("n", [1, 30, 45, 90, 3600, 3930, 5400, 9000])
def test_round_trip(n: int) -> None:
    assert parse_duration_to_seconds(format_seconds(n)) == n


@pytest.mark.parametrize("seconds", [0, -1])
def test_format_rejects_nonpositive(seconds: int) -> None:
    with pytest.raises(ValueError):
        format_seconds(seconds)
