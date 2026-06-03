"""Shared git-sha provenance helper for run/export metadata."""

from __future__ import annotations

import subprocess
from pathlib import Path

import custom_sam_peft


def git_sha() -> str | None:
    """Return the short HEAD git sha for the package repo, or None if unavailable."""
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],  # noqa: S607
        cwd=Path(custom_sam_peft.__file__).parent,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None
