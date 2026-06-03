"""Tests for the shared git-sha provenance helper."""

from __future__ import annotations

from custom_sam_peft._provenance import git_sha


def test_git_sha_returns_str_or_none() -> None:
    """git_sha() returns a non-empty str (inside a repo) or None, never raising."""
    sha = git_sha()
    assert sha is None or (isinstance(sha, str) and len(sha) > 0)
