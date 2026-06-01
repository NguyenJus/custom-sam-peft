"""Provenance-completeness checker (issue #192).

Internal, not part of the public API. Pure functions take an explicit
repo-root (or explicit doc-text + file paths) so the unit tests can drive the
checker over synthetic fixture trees instead of the live repo.

See ``docs/defaults-provenance.md`` and the design spec
``docs/superpowers/specs/2026-06-01-ci-no-uncited-default-hook-design.md``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

TABLE_MODULES: frozenset[str] = frozenset(
    {"data/aug_presets.py", "models/losses/presets.py"}
)
PROSE_NARRATIVE_HEADERS: frozenset[str] = frozenset(
    {"Verification Standard", "Reference Training Profile"}
)


@dataclass(frozen=True)
class ProvenanceViolation:
    """One actionable provenance failure: what, where, and how to fix it."""

    location: str
    problem: str
    remediation: str

    def __str__(self) -> str:
        return f"{self.location}: {self.problem} — {self.remediation}"


@dataclass(frozen=True)
class Section:
    """A ``## <header>`` block of the provenance doc."""

    header: str
    body: str


_H2 = re.compile(r"^## (.+)$", re.MULTILINE)


def discover_sections(doc_text: str) -> list[Section]:
    """Split the doc into ``## <header>`` sections (body excludes the header line)."""
    matches = list(_H2.finditer(doc_text))
    sections: list[Section] = []
    for i, m in enumerate(matches):
        header = m.group(1).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(doc_text)
        sections.append(Section(header=header, body=doc_text[start:end]))
    return sections
