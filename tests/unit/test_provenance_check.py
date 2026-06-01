# tests/unit/test_provenance_check.py
from __future__ import annotations

from pathlib import Path

from custom_sam_peft._provenance_check import (
    ProvenanceViolation,
    Section,
    discover_sections,
)


def test_provenance_violation_str_renders_contract() -> None:
    v = ProvenanceViolation(
        location="config/schema.py:Foo.bar",
        problem="new undocumented default",
        remediation="add a row to the ## config/schema.py section",
    )
    assert str(v) == (
        "config/schema.py:Foo.bar: new undocumented default — "
        "add a row to the ## config/schema.py section"
    )


def test_discover_sections_splits_on_h2_headers() -> None:
    doc = "# Title\n\nintro\n\n## A\n\nbody a\n\n## B\n\nbody b\n"
    sections = discover_sections(doc)
    headers = [s.header for s in sections]
    assert headers == ["A", "B"]
    assert sections[0].body.strip() == "body a"
    assert isinstance(sections[1], Section)
