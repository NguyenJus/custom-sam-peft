"""Pure duration parsing/formatting utilities.

No I/O, no logging. Used by the config schema (validation) and the CLI
(early-validate + exit-message rendering). Spec §4.2.
"""

from __future__ import annotations

import re

# One <number><unit> group per unit, each optional, but enforced order h -> m -> s.
# An all-digits string is handled separately (bare seconds) before this matches.
_DURATION_RE = re.compile(r"^(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?$")
_EXAMPLES = 'use e.g. "2h30m", "90m", "3600s", or bare seconds'


def parse_duration_to_seconds(value: str | int) -> int:
    """Parse a duration to a strictly-positive integer number of seconds.

    Accepts a bare int (3600), a bare all-digits string ("3600"), or an
    h/m/s combo ("1h", "45m", "30s", "2h30m", "1h5m30s"). Surrounding
    whitespace is tolerated. Units are lowercase, each at most once, in
    h -> m -> s order. Raises ValueError on non-positive results, empty /
    whitespace-only strings, or any string outside the grammar (e.g. "abc",
    "10x", "2h30" -- a trailing number with no unit).
    """
    if isinstance(value, bool):  # bool is an int subclass; reject explicitly.
        raise ValueError(f"time_limit: {value!r} is not a valid duration ({_EXAMPLES})")
    if isinstance(value, int):
        seconds = value
    else:
        text = value.strip()
        if not text:
            raise ValueError(f"time_limit: {value!r} is not a valid duration ({_EXAMPLES})")
        if text.isdigit():
            seconds = int(text)
        else:
            m = _DURATION_RE.fullmatch(text)
            if m is None or not any(m.groups()):
                raise ValueError(f"time_limit: {value!r} is not a valid duration ({_EXAMPLES})")
            h, mm, s = (int(g) if g else 0 for g in m.groups())
            seconds = h * 3600 + mm * 60 + s
    if seconds <= 0:
        raise ValueError(
            f"time_limit: {value!r} must be a strictly-positive duration ({_EXAMPLES})"
        )
    return seconds


def format_seconds(seconds: int) -> str:
    """Render a positive second count as a canonical human string.

    Collapses to the largest applicable units, dropping zero components:
    9000 -> "2h30m", 3600 -> "1h", 5400 -> "1h30m", 90 -> "1m30s", 45 -> "45s".
    Raises ValueError if ``seconds`` is zero or negative.
    """
    if seconds <= 0:
        raise ValueError(f"format_seconds: expected a positive second count, got {seconds!r}")
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    parts = []
    if h:
        parts.append(f"{h}h")
    if m:
        parts.append(f"{m}m")
    if s:
        parts.append(f"{s}s")
    return "".join(parts)
