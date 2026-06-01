"""Provenance-completeness checker (issue #192).

Internal, not part of the public API. Pure functions take an explicit
repo-root (or explicit doc-text + file paths) so the unit tests can drive the
checker over synthetic fixture trees instead of the live repo.

See ``docs/defaults-provenance.md`` and the design spec
``docs/superpowers/specs/2026-06-01-ci-no-uncited-default-hook-design.md``.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

TABLE_MODULES: frozenset[str] = frozenset({"data/aug_presets.py", "models/losses/presets.py"})
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


SectionClass = Literal["prose", "table", "yaml", "prose-narrative"]

_H2 = re.compile(r"^## (.+)$", re.MULTILINE)


def _unescape_md(header: str) -> str:
    """Strip markdown backslash-escapes (e.g. ``test\\_x`` -> ``test_x``)."""
    return header.replace("\\_", "_")


def resolve_section_path(header: str, repo_root: Path) -> Path | None:
    """Resolve a ``## <header>`` to a file on disk, or ``None`` for prose-narrative.

    Raises ``FileNotFoundError`` if a file-section header resolves to no file.
    """
    if header in PROSE_NARRATIVE_HEADERS:
        return None
    text = _unescape_md(header).strip()
    if text.startswith("cli/templates/"):
        candidate = repo_root / "src/custom_sam_peft" / text
    elif text.startswith("tests/"):
        candidate = repo_root / text
    else:
        candidate = repo_root / "src/custom_sam_peft" / text
    if not candidate.is_file():
        raise FileNotFoundError(
            f"doc section names a path that does not exist: {header!r} (resolved to {candidate})"
        )
    return candidate


def classify_section(section: Section, repo_root: Path) -> SectionClass:
    """Classify a doc section into one of the four classes."""
    path = resolve_section_path(section.header, repo_root)
    if path is None:
        return "prose-narrative"
    rel = _unescape_md(section.header).strip()
    if rel in TABLE_MODULES:
        return "table"
    if path.suffix == ".yaml":
        return "yaml"
    return "prose"


# A table row: leading "| ", cells separated by " | ".
_ROW = re.compile(r"^\|(?P<cells>.+)\|\s*$")
_LOCATION_CELL = re.compile(r"`(?P<loc>[^`]+)`")
# A Location of the form ``symbol (<literal-note>)`` is an in-function literal.
_LITERAL_SUFFIX = re.compile(r"^(?P<symbol>[^()]+?)\s*\((?P<note>[^)]*)\)\s*$")


@dataclass(frozen=True)
class DocRow:
    """A parsed Location key from one doc table row, relative to its section."""

    symbol: str
    is_in_function_literal: bool


def parse_prose_rows(body: str, section_header: str) -> list[DocRow]:
    """Parse a prose section body into per-row Location symbols.

    The Location prefix mirrors ``section_header`` and is stripped to yield the
    bare symbol. Rows whose Location has a ``symbol (<note>)`` parenthetical are
    flagged as in-function literals (exempt in the doc->code direction).
    """
    prefix = _unescape_md(section_header).strip()
    rows: list[DocRow] = []
    for line in body.splitlines():
        m = _ROW.match(line)
        if m is None:
            continue
        loc_match = _LOCATION_CELL.search(m.group("cells").split("|", 1)[0])
        if loc_match is None:
            continue
        loc = loc_match.group("loc").strip()
        if not loc.startswith(f"{prefix}:"):
            continue  # header/separator/non-location rows
        rest = loc[len(prefix) + 1 :]  # drop "prefix:"
        lit = _LITERAL_SUFFIX.match(rest)
        if lit is not None:
            rows.append(DocRow(symbol=lit.group("symbol").strip(), is_in_function_literal=True))
        else:
            rows.append(DocRow(symbol=rest.strip(), is_in_function_literal=False))
    return rows


def extract_default_surface(file_path: Path) -> set[str]:
    """Return the enforced default-surface symbol keys for a prose file.

    Surface = pydantic ``Field(default=...)``/``Field(default_factory=...)``,
    dataclass field defaults, and module-level constant assignments. In-function
    literals are deliberately excluded.
    """
    tree = ast.parse(file_path.read_text(encoding="utf-8"))
    surface: set[str] = set()

    # Module-level constant assignments.
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    surface.add(target.id)
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.value is not None
        ):
            surface.add(node.target.id)

    # Class-body field defaults (pydantic + dataclass): ``name: T = <default>``.
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        for stmt in node.body:
            if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                if stmt.value is not None:
                    surface.add(f"{node.name}.{stmt.target.id}")
            elif isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    if isinstance(target, ast.Name):
                        surface.add(f"{node.name}.{target.id}")
    return surface


def check_prose_section(section: Section, file_path: Path) -> list[ProvenanceViolation]:
    """Assertion 1: symbol<->row bijection over the file's default surface.

    code->doc: every surface symbol must have a doc row.
    doc->code: every doc row naming a *surface* symbol must still resolve;
               rows flagged as in-function literals are exempt in this direction.
    """
    file_disp = _unescape_md(section.header).strip()
    surface = extract_default_surface(file_path)
    rows = parse_prose_rows(section.body, section.header)
    documented_surface_symbols = {r.symbol for r in rows if not r.is_in_function_literal}

    violations: list[ProvenanceViolation] = []

    # code->doc
    for symbol in sorted(surface - documented_surface_symbols):
        violations.append(
            ProvenanceViolation(
                location=f"{file_disp}:{symbol}",
                problem="new undocumented default",
                remediation=(
                    f"add a row to the `## {file_disp}` section of docs/defaults-provenance.md"
                ),
            )
        )

    # doc->code (skip in-function-literal rows)
    for row in rows:
        if row.is_in_function_literal:
            continue
        if row.symbol not in surface:
            violations.append(
                ProvenanceViolation(
                    location=f"{file_disp}:{row.symbol}",
                    problem="stale/orphaned provenance row",
                    remediation=("remove or update the row in docs/defaults-provenance.md"),
                )
            )
    return violations


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
