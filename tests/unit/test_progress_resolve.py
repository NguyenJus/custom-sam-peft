"""Test A: resolve_mode matrix (spec §9 / §4).

Pure-function test — no session, no side effects, no file I/O.
"""

from __future__ import annotations

import pytest

from custom_sam_peft.cli._progress import ProgressMode, resolve_mode


@pytest.mark.parametrize(
    ("cli_flag", "env", "isatty", "is_jupyter", "expected"),
    [
        # Explicit flag wins over everything.
        ("on", {}, False, True, ProgressMode.ON),
        ("on", {"CSP_NO_PROGRESS": "1"}, False, True, ProgressMode.ON),
        ("off", {}, True, False, ProgressMode.OFF),
        ("plain", {}, True, False, ProgressMode.PLAIN),
        # auto/None with CSP_NO_PROGRESS=1 → OFF.
        ("auto", {"CSP_NO_PROGRESS": "1"}, True, False, ProgressMode.OFF),
        (None, {"CSP_NO_PROGRESS": "1"}, True, False, ProgressMode.OFF),
        # Jupyter auto-fallback → PLAIN.
        ("auto", {}, True, True, ProgressMode.PLAIN),
        (None, {}, True, True, ProgressMode.PLAIN),
        # Non-TTY auto-fallback → PLAIN (Jupyter already handled above).
        ("auto", {}, False, False, ProgressMode.PLAIN),
        (None, {}, False, False, ProgressMode.PLAIN),
        # TTY, no env, no Jupyter → ON.
        ("auto", {}, True, False, ProgressMode.ON),
        (None, {}, True, False, ProgressMode.ON),
    ],
)
def test_resolve_mode_matrix(
    cli_flag: str | None,
    env: dict[str, str],
    isatty: bool,
    is_jupyter: bool,
    expected: ProgressMode,
) -> None:
    """Test A: resolve_mode covers flag > env > auto fallback precedence."""
    result = resolve_mode(cli_flag, env, isatty, is_jupyter)
    assert result == expected, (
        f"resolve_mode({cli_flag!r}, {env}, isatty={isatty}, is_jupyter={is_jupyter}) "
        f"returned {result!r}, expected {expected!r}"
    )
