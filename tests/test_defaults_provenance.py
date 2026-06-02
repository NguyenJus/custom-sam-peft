"""Live provenance-completeness gate (issue #192).

Runs all three assertions over the REAL repo and asserts zero violations. After
the inline-tag strip, bare-cell tagging, and doc migration, the repo passes its
own completeness check. A new undocumented default (or an orphaned doc row, or an
untagged preset cell) makes this test fail in the `test` CI job.
"""

from __future__ import annotations

from pathlib import Path

from custom_sam_peft._provenance_check import run_all_checks

_REPO_ROOT = Path(__file__).resolve().parents[1]


def test_repo_defaults_provenance_is_complete() -> None:
    violations = run_all_checks(_REPO_ROOT)
    assert not violations, "Provenance completeness violations:\n" + "\n".join(
        str(v) for v in violations
    )
