# CI No-Uncited-Default Hook + Inline-Tag Strip Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a pytest-driven provenance-completeness check (three assertions over `docs/defaults-provenance.md` and in-scope source files) plus a curated inline-tag strip, bare-off-cell tagging, and a doc migration, leaving the repo green under its own check.

**Architecture:** A new internal module `src/custom_sam_peft/_provenance_check.py` exposes pure functions that take an explicit repo-root (or explicit doc-text + file-set) so unit tests can drive it over synthetic fixture trees. The module auto-derives scope from the doc's `## <path>` section headers, classifies each (`prose` / `table` / `yaml` / `prose-narrative`), and dispatches one of three assertions. Synthetic unit tests give the module its own coverage and land first (green in isolation). The repo-conformance edits (strip, bare-cell tagging, doc migration) land before the *live* test `tests/test_defaults_provenance.py`, which runs the check over the real repo and is the final green gate.

**Tech Stack:** Python 3.12, pydantic, pytest + coverage (`--cov-fail-under=80`), ruff (`check` + `format --check`), mypy, markdownlint-cli2. Parsing uses Python's stdlib `ast` for the default surface (prose sections) and line-level regex for table cells / YAML scalars.

---

## Background facts the implementer must know

These are load-bearing repo facts discovered during planning. Bake them in; do not re-derive.

- **CI shape (no new YAML).** `.github/workflows/ci.yml` `test` job runs, in order: `uv run ruff check`, `uv run ruff format --check`, `uv run mypy src/custom_sam_peft`, `uv run pytest`. A separate `lint-hygiene` job runs `markdownlint-cli2 --config .config/markdownlint-cli2.jsonc "**/*.md" "#node_modules"`. The new check rides `uv run pytest`.
- **Coverage gate.** `pyproject.toml` `[tool.pytest.ini_options] addopts` includes `--cov=custom_sam_peft ... --cov-fail-under=80`. The new `_provenance_check.py` is under `src/custom_sam_peft/`, so it counts toward coverage; its unit tests must exercise it thoroughly or the global gate fails.
- **Test layout.** Unit tests live in `tests/unit/test_*.py` and import internals via `from custom_sam_peft.<...> import <...>`. There is `tests/conftest.py` and `tests/unit/__init__.py`. The live enforcement test goes at the top level: `tests/test_defaults_provenance.py` (sibling of `tests/conftest.py`), with no GPU/integration marker so it runs in the normal CPU `test` job.
- **Doc structure (`docs/defaults-provenance.md`).** `## <path>` headers, current set:
  - prose/constant: `## config/_internal.py`, `## config/schema.py`, `## data/channel_semantics.py`, `## data/transforms.py`, `## presets.py`, `## predict/budget.py`, `## tests/gpu/test\_qlora\_8gb\_ceiling.py` (note: header text has **markdown-escaped underscores** `\_`).
  - table: `## data/aug_presets.py`, `## models/losses/presets.py`.
  - yaml: `## cli/templates/config_full.yaml`.
  - prose-narrative (no check): `## Verification Standard`, `## Reference Training Profile`.
- **Doc Location-key convention is SECTION-RELATIVE.** A row's `Location` cell prefix mirrors its section header text, NOT a uniform repo path. Examples actually in the doc:
  - `## presets.py` section rows are keyed `presets.py:MODEL_PARAMS`, `presets.py:forward_only_factor`, etc. (no `src/...` prefix).
  - `## data/channel_semantics.py` rows are keyed `data/channel_semantics.py:_IMAGENET_MEAN`, etc.
  - `## config/schema.py` rows are keyed `config/schema.py:RunConfig.seed`, etc.
  - `## cli/templates/config_full.yaml` rows are keyed `config_full.yaml:run.seed` (basename-rooted dotted path).
  - The three in-function literal rows: `presets.py:_bytes_per_param_for_method (2.0)`, `presets.py:_bytes_per_param_for_method (0.5)`, `presets.py:_optimizer_bytes (*4 literal)` — recognized by the `(<...>)` parenthetical suffix.
  So when matching rows to a section, **strip the section's own header-derived prefix** from each `Location` to get the bare `symbol`, then compare against the file's surface symbols. Do not assume a fixed prefix across sections.
- **Base-path resolution** (header text → file on disk), in order: `cli/templates/…` → `src/custom_sam_peft/cli/templates/…`; `tests/…` → repo-root `tests/…`; everything else → `src/custom_sam_peft/…`. The header `tests/gpu/test\_qlora\_8gb\_ceiling.py` must have its markdown `\_` unescaped to `_` before resolving. A file-section header resolving to no file on disk is a hard FAIL.
- **Default surface (prose sections)** — exactly: pydantic `Field(default=...)` / `Field(default_factory=...)`, dataclass field defaults (`name: T = <default>`), and module-level constant assignments (`NAME = <value>` / `NAME: T = <value>` at module top level). **Excludes** in-function literals.
- **Table modules' tag syntaxes differ.** `aug_presets.py` uses bare `# (a)`–`# (e)` (lowercase). `losses/presets.py` uses `# cite: (A)`–`# cite: (H)` and combined forms `# cite: (A,C)`. Both also allow `# cite: <non-legend>` (e.g. `# cite: empirical`) and `# tbd: …`. The recognizer must accept all of these.
- **`losses/presets.py` table scope** includes the `PRESET_TABLE` dict literal, the three module-level alias-assignment lines `PRESET_TABLE[("microscopy", …)] = dict(...)  # cite: (G)`, AND the `_LEGACY_DEFAULTS` base dict.

---

## File Structure

| File | Responsibility |
| --- | --- |
| `src/custom_sam_peft/_provenance_check.py` (create) | Pure-function checker: doc parsing, section classification, base-path resolution, the three assertions, the `ProvenanceViolation` failure record. |
| `tests/unit/test_provenance_check.py` (create) | Synthetic-fixture unit tests — drive the checker over temp trees; never touch live repo state. Provides the module's coverage. |
| `tests/test_defaults_provenance.py` (create) | Live enforcement test — runs all three assertions over the real repo; asserts zero violations. The final green gate. |
| `src/custom_sam_peft/config/schema.py` (modify) | Strip redundant inline tags; keep the curated head-turner notes. |
| `src/custom_sam_peft/config/_internal.py` (modify) | Keep matcher-weights note; strip `lambda_mask` tbd tag. |
| `src/custom_sam_peft/data/channel_semantics.py` (modify) | Strip all inline tags to doc-only. |
| `src/custom_sam_peft/data/transforms.py` (modify) | Keep HED + processor-stats divergence notes; strip the rest. |
| `src/custom_sam_peft/presets.py` (modify) | Strip all inline cite tags to doc-only. |
| `src/custom_sam_peft/predict/budget.py` (modify) | Strip the budget-derivation tags to doc-only. |
| `src/custom_sam_peft/data/aug_presets.py` (modify) | Tag bare off-cells (`# (a)` / `# (d)` / new `# (e)`); keep legend-letter system. |
| `src/custom_sam_peft/models/losses/presets.py` (modify) | Tag bare off-cells + `_LEGACY_DEFAULTS` (`# cite: (B)` / `(A,C)` / `(F)`); keep legend system. |
| `docs/defaults-provenance.md` (modify) | Preamble reword, `Tag` column redefinition, add `(e)` aug-legend row. |

---

## Phasing overview

Five phases, each leaving CI green.

- **Phase 1 — Checker core + section classification + Assertion 1 (prose), with synthetic unit tests.** Lands `_provenance_check.py` with doc parsing, classification, base-path resolution, and the prose-section symbol⇄row bijection, all exercised by synthetic fixtures. Green in isolation (no live-repo test yet).
- **Phase 2 — Assertions 2 & 3 (table + yaml) added to the checker, with synthetic unit tests.** Extends the module; still no live test. Green in isolation.
- **Phase 3 — Bare-off-cell tagging in the two table modules + the `(e)` aug-legend doc row.** Makes the table modules Assertion-2-clean. Value-preserving edits only.
- **Phase 4 — Curated inline-tag strip of prose/constant files + doc preamble/Tag-column migration.** Makes the prose files Assertion-1-clean and the doc principle-correct.
- **Phase 5 — Live enforcement test + final green gate.** Adds `tests/test_defaults_provenance.py`, runs the full check over the real repo, and confirms the whole CI matrix is green.

### Phase-boundary interface contracts

- **End of Phase 1 exposes** (consumed by Phases 2 and 5):
  - `discover_sections(doc_text: str) -> list[Section]` where `Section` is a dataclass `Section(header: str, body: str)` (`header` is the raw header text after `## `; `body` is the markdown between this header and the next `## `).
  - `classify_section(section: Section, repo_root: Path) -> SectionClass` returning a `SectionClass` enum/`Literal["prose", "table", "yaml", "prose-narrative"]`. Internally calls `resolve_section_path`.
  - `resolve_section_path(header: str, repo_root: Path) -> Path | None` — applies the `cli/templates/` / `tests/` / else `src/custom_sam_peft/` rules (unescaping markdown `\_`), returns `None` if the header is in the prose-narrative allow-set, raises/records a hard-fail violation if a file-section header resolves to a missing file.
  - `ProvenanceViolation` dataclass: `ProvenanceViolation(location: str, problem: str, remediation: str)` with `__str__` rendering `"{location}: {problem} — {remediation}"`. This is the single failure record type all assertions emit.
  - `check_prose_section(section: Section, file_path: Path) -> list[ProvenanceViolation]` — Assertion 1.
  - `extract_default_surface(file_path: Path) -> set[str]` — surface symbols (used by Assertion 1 and unit-tested directly).
  - The table-module allow-set constant `TABLE_MODULES = {"data/aug_presets.py", "models/losses/presets.py"}` and the prose-narrative allow-set `PROSE_NARRATIVE_HEADERS = {"Verification Standard", "Reference Training Profile"}`.
- **End of Phase 2 exposes** (consumed by Phase 5):
  - `check_table_section(section: Section, file_path: Path) -> list[ProvenanceViolation]` — Assertion 2.
  - `check_yaml_section(section: Section, yaml_path: Path, schema_default_paths: set[str]) -> list[ProvenanceViolation]` — Assertion 3.
  - `run_all_checks(repo_root: Path) -> list[ProvenanceViolation]` — the top-level driver: reads `docs/defaults-provenance.md`, discovers + classifies sections, dispatches each to its assertion, aggregates ALL violations, returns them. This is the one function the live test calls.
- **End of Phase 3 guarantees** (consumed by Phase 5): every preset-table value line in `aug_presets.py` and `losses/presets.py` (incl. `_LEGACY_DEFAULTS` + alias lines) carries a recognized tag whose legend letter is defined in that module's doc legend — i.e. Assertion 2 over the real repo is clean. No cell *value* changed.
- **End of Phase 4 guarantees** (consumed by Phase 5): every prose-section file's default surface ⇄ doc rows is a bijection (no undocumented surface symbol, no orphaned surface-symbol row) — i.e. Assertion 1 over the real repo is clean. The doc preamble + `Tag` column reflect the new principle. The curated keep-list notes are exactly the surviving inline notes.
- **End of Phase 5:** `run_all_checks(repo_root)` over the real repo returns `[]`; `tests/test_defaults_provenance.py` is green; full CI (ruff check, ruff format --check, mypy, pytest with coverage, markdownlint) is green.

---

## Phase 1 — Checker core, classification, Assertion 1 (prose)

**Interface contract this phase EXPOSES:** `discover_sections`, `Section`, `classify_section`, `resolve_section_path`, `ProvenanceViolation`, `extract_default_surface`, `check_prose_section`, `TABLE_MODULES`, `PROSE_NARRATIVE_HEADERS` (signatures above). Phases 2 & 5 build on these without re-reading the implementation.

**CI state at phase end:** GREEN. The module + its unit tests land together; no live-repo test exists yet, so nothing depends on repo conformance. The unit tests cover the new module so the 80% gate holds.

### Task 1.1: Module scaffold + `ProvenanceViolation` + `Section`

**Files:**

- Create: `src/custom_sam_peft/_provenance_check.py`
- Test: `tests/unit/test_provenance_check.py`

- [ ] **Step 1: Write the failing test** (difficulty: easy)

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_provenance_check.py -v`
Expected: FAIL — `ModuleNotFoundError` / `ImportError` (module + symbols not defined).

- [ ] **Step 3: Write minimal implementation**

```python
# src/custom_sam_peft/_provenance_check.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_provenance_check.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/custom_sam_peft/_provenance_check.py tests/unit/test_provenance_check.py
git commit -m "feat(#192): provenance checker scaffold — Section + ProvenanceViolation"
```

### Task 1.2: Base-path resolution + section classification

**Files:**

- Modify: `src/custom_sam_peft/_provenance_check.py`
- Test: `tests/unit/test_provenance_check.py`

- [ ] **Step 1: Write the failing test** (difficulty: medium)

```python
# append to tests/unit/test_provenance_check.py
import pytest

from custom_sam_peft._provenance_check import classify_section, resolve_section_path


def _make_repo(tmp_path: Path) -> Path:
    """Minimal fixture repo tree mirroring the real base-path roots."""
    (tmp_path / "src/custom_sam_peft/config").mkdir(parents=True)
    (tmp_path / "src/custom_sam_peft/config/schema.py").write_text("x = 1\n")
    (tmp_path / "src/custom_sam_peft/data").mkdir(parents=True)
    (tmp_path / "src/custom_sam_peft/data/aug_presets.py").write_text("PRESET_TABLE = {}\n")
    (tmp_path / "src/custom_sam_peft/cli/templates").mkdir(parents=True)
    (tmp_path / "src/custom_sam_peft/cli/templates/config_full.yaml").write_text("a: 1\n")
    (tmp_path / "tests/gpu").mkdir(parents=True)
    (tmp_path / "tests/gpu/test_qlora_8gb_ceiling.py").write_text("Q = 8\n")
    return tmp_path


def test_resolve_src_rooted(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    p = resolve_section_path("config/schema.py", repo)
    assert p == repo / "src/custom_sam_peft/config/schema.py"


def test_resolve_cli_templates_rooted(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    p = resolve_section_path("cli/templates/config_full.yaml", repo)
    assert p == repo / "src/custom_sam_peft/cli/templates/config_full.yaml"


def test_resolve_tests_rooted_unescapes_markdown_underscores(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    # The doc header escapes underscores for markdown: test\_qlora\_8gb...
    p = resolve_section_path(r"tests/gpu/test\_qlora\_8gb\_ceiling.py", repo)
    assert p == repo / "tests/gpu/test_qlora_8gb_ceiling.py"


def test_resolve_prose_narrative_returns_none(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    assert resolve_section_path("Verification Standard", repo) is None
    assert resolve_section_path("Reference Training Profile", repo) is None


def test_classify_section_kinds(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)

    def cls(header: str) -> str:
        return classify_section(Section(header=header, body=""), repo)

    assert cls("config/schema.py") == "prose"
    assert cls("data/aug_presets.py") == "table"
    assert cls("cli/templates/config_full.yaml") == "yaml"
    assert cls("Verification Standard") == "prose-narrative"
    assert cls(r"tests/gpu/test\_qlora\_8gb\_ceiling.py") == "prose"


def test_classify_missing_file_is_hard_fail(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    with pytest.raises(FileNotFoundError):
        classify_section(Section(header="config/does_not_exist.py", body=""), repo)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_provenance_check.py -k "resolve or classify" -v`
Expected: FAIL — `ImportError` for `classify_section` / `resolve_section_path`.

- [ ] **Step 3: Write minimal implementation**

```python
# add to src/custom_sam_peft/_provenance_check.py
from pathlib import Path
from typing import Literal

SectionClass = Literal["prose", "table", "yaml", "prose-narrative"]


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
            f"doc section names a path that does not exist: {header!r} "
            f"(resolved to {candidate})"
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_provenance_check.py -k "resolve or classify" -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/custom_sam_peft/_provenance_check.py tests/unit/test_provenance_check.py
git commit -m "feat(#192): base-path resolution + section classification"
```

### Task 1.3: `extract_default_surface` (ast-based)

**Files:**

- Modify: `src/custom_sam_peft/_provenance_check.py`
- Test: `tests/unit/test_provenance_check.py`

- [ ] **Step 1: Write the failing test** (difficulty: medium)

```python
# append to tests/unit/test_provenance_check.py
from custom_sam_peft._provenance_check import extract_default_surface


def test_surface_collects_pydantic_dataclass_and_module_constants(tmp_path: Path) -> None:
    src = '''
from pydantic import BaseModel, Field
from dataclasses import dataclass

MODULE_CONST = 7
TYPED_CONST: int = 8


class Cfg(BaseModel):
    a: int = Field(default=3)
    b: list[int] = Field(default_factory=list)
    plain: int = 5


@dataclass
class DC:
    x: float = 1.5


def helper() -> int:
    magic = 42  # in-function literal, NOT surface
    return magic
'''
    f = tmp_path / "m.py"
    f.write_text(src)
    surface = extract_default_surface(f)
    assert "MODULE_CONST" in surface
    assert "TYPED_CONST" in surface
    assert "Cfg.a" in surface
    assert "Cfg.b" in surface
    assert "Cfg.plain" in surface
    assert "DC.x" in surface
    # In-function literal is excluded.
    assert not any("magic" in s for s in surface)
    assert "helper" not in surface
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_provenance_check.py -k surface -v`
Expected: FAIL — `ImportError` for `extract_default_surface`.

- [ ] **Step 3: Write minimal implementation**

```python
# add to src/custom_sam_peft/_provenance_check.py
import ast


def _is_field_call(node: ast.expr) -> bool:
    """True if node is a ``Field(...)`` / ``...Field(...)`` call."""
    return (
        isinstance(node, ast.Call)
        and (
            (isinstance(node.func, ast.Name) and node.func.id == "Field")
            or (isinstance(node.func, ast.Attribute) and node.func.attr == "Field")
        )
    )


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
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.value is not None:
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
```

Note: `_is_field_call` is provided for the reviewer's intent — the surface rule treats *any* class-body annotated/plain assignment with a value as a field default (whether or not the RHS is a `Field(...)` call), which matches the spec's "dataclass field defaults" + "pydantic `Field(default=...)`" union. Keep `_is_field_call` only if a later refinement needs to distinguish; otherwise the implementer MAY drop it to satisfy ruff's unused-symbol check. (Decide at implementation: if unused, delete it — do not leave dead code.)

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_provenance_check.py -k surface -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/custom_sam_peft/_provenance_check.py tests/unit/test_provenance_check.py
git commit -m "feat(#192): ast-based default-surface extraction"
```

### Task 1.4: Doc-row parsing for a prose section

**Files:**

- Modify: `src/custom_sam_peft/_provenance_check.py`
- Test: `tests/unit/test_provenance_check.py`

- [ ] **Step 1: Write the failing test** (difficulty: medium)

```python
# append to tests/unit/test_provenance_check.py
from custom_sam_peft._provenance_check import parse_prose_rows


def test_parse_prose_rows_strips_section_prefix_and_flags_literals() -> None:
    # The body's Location keys are section-relative (prefix == header text).
    body = (
        "| Location | Value | Tag | Full reference | Verifying quote | Notes |\n"
        "| --- | --- | --- | --- | --- | --- |\n"
        "| `presets.py:MODEL_PARAMS` | `1` | `# tbd: #191` | — | — | n |\n"
        "| `presets.py:forward_only_factor` | `0.25` | `# cite: x` | — | — | n |\n"
        "| `presets.py:_optimizer_bytes (*4 literal)` | `4x` | `# cite: y` | — | — | n |\n"
    )
    rows = parse_prose_rows(body, section_header="presets.py")
    by_symbol = {r.symbol: r for r in rows}
    assert by_symbol["MODEL_PARAMS"].is_in_function_literal is False
    assert by_symbol["forward_only_factor"].is_in_function_literal is False
    # The parenthetical-suffix row is recognized as an in-function literal.
    lit = next(r for r in rows if r.is_in_function_literal)
    assert lit.symbol == "_optimizer_bytes"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_provenance_check.py -k parse_prose -v`
Expected: FAIL — `ImportError` for `parse_prose_rows`.

- [ ] **Step 3: Write minimal implementation**

```python
# add to src/custom_sam_peft/_provenance_check.py

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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_provenance_check.py -k parse_prose -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/custom_sam_peft/_provenance_check.py tests/unit/test_provenance_check.py
git commit -m "feat(#192): doc-row parsing with in-function-literal detection"
```

### Task 1.5: `check_prose_section` — Assertion 1 bijection

**Files:**

- Modify: `src/custom_sam_peft/_provenance_check.py`
- Test: `tests/unit/test_provenance_check.py`

- [ ] **Step 1: Write the failing test** (difficulty: medium)

```python
# append to tests/unit/test_provenance_check.py
from custom_sam_peft._provenance_check import check_prose_section


def _prose_doc_body(rows: str) -> str:
    return (
        "| Location | Value | Tag | Full reference | Verifying quote | Notes |\n"
        "| --- | --- | --- | --- | --- | --- |\n" + rows
    )


def test_prose_documented_default_passes(tmp_path: Path) -> None:
    f = tmp_path / "schema.py"
    f.write_text("from pydantic import BaseModel, Field\n\n\nclass C(BaseModel):\n    a: int = Field(default=3)\n")
    body = _prose_doc_body("| `schema.py:C.a` | `3` | `# cite: x` | — | — | n |\n")
    section = Section(header="schema.py", body=body)
    assert check_prose_section(section, f) == []


def test_prose_undocumented_default_fails_codedoc(tmp_path: Path) -> None:
    f = tmp_path / "schema.py"
    f.write_text("from pydantic import BaseModel, Field\n\n\nclass C(BaseModel):\n    a: int = Field(default=3)\n")
    body = _prose_doc_body("")  # no rows at all
    section = Section(header="schema.py", body=body)
    violations = check_prose_section(section, f)
    assert len(violations) == 1
    msg = str(violations[0])
    assert "schema.py:C.a" in msg
    assert "undocumented default" in msg
    assert "add a row" in msg


def test_prose_orphaned_row_fails_doccode(tmp_path: Path) -> None:
    f = tmp_path / "schema.py"
    f.write_text("from pydantic import BaseModel, Field\n\n\nclass C(BaseModel):\n    a: int = Field(default=3)\n")
    body = _prose_doc_body(
        "| `schema.py:C.a` | `3` | `# cite: x` | — | — | n |\n"
        "| `schema.py:C.gone` | `9` | `# cite: y` | — | — | n |\n"
    )
    section = Section(header="schema.py", body=body)
    violations = check_prose_section(section, f)
    assert len(violations) == 1
    msg = str(violations[0])
    assert "schema.py:C.gone" in msg
    assert "orphaned" in msg or "stale" in msg
    assert "remove or update" in msg


def test_prose_in_function_literal_row_exempt_from_doccode(tmp_path: Path) -> None:
    # A doc row of `symbol (<literal>)` form whose base symbol is absent => no failure.
    f = tmp_path / "presets.py"
    f.write_text("MODEL_PARAMS = 1\n")
    body = _prose_doc_body(
        "| `presets.py:MODEL_PARAMS` | `1` | `# tbd: x` | — | — | n |\n"
        "| `presets.py:_optimizer_bytes (*4 literal)` | `4x` | `# cite: y` | — | — | n |\n"
    )
    section = Section(header="presets.py", body=body)
    assert check_prose_section(section, f) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_provenance_check.py -k prose -v`
Expected: FAIL — `ImportError` for `check_prose_section`.

- [ ] **Step 3: Write minimal implementation**

```python
# add to src/custom_sam_peft/_provenance_check.py


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
                    f"add a row to the `## {file_disp}` section of "
                    "docs/defaults-provenance.md"
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
                    remediation=(
                        "remove or update the row in docs/defaults-provenance.md"
                    ),
                )
            )
    return violations
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_provenance_check.py -k prose -v`
Expected: PASS.

- [ ] **Step 5: Run full module + lint gates**

Run: `uv run pytest tests/unit/test_provenance_check.py -v && uv run ruff check src/custom_sam_peft/_provenance_check.py tests/unit/test_provenance_check.py && uv run ruff format --check src/custom_sam_peft/_provenance_check.py tests/unit/test_provenance_check.py && uv run mypy src/custom_sam_peft/_provenance_check.py`
Expected: all PASS. (If `_is_field_call` is unused, delete it now so ruff passes.)

- [ ] **Step 6: Commit**

```bash
git add src/custom_sam_peft/_provenance_check.py tests/unit/test_provenance_check.py
git commit -m "feat(#192): Assertion 1 — prose symbol<->row bijection"
```

---

## Phase 2 — Assertions 2 & 3 (table + yaml) + driver

**Interface contract this phase CONSUMES (from Phase 1):** `Section`, `ProvenanceViolation`, `classify_section`, `resolve_section_path`, `discover_sections`, `check_prose_section`, `TABLE_MODULES`.

**Interface contract this phase EXPOSES (to Phase 5):** `check_table_section`, `check_yaml_section`, `run_all_checks` (signatures in the overview). `run_all_checks(repo_root)` is the single entry point the live test calls.

**CI state at phase end:** GREEN. Still no live-repo test; the new assertions are exercised only by synthetic fixtures.

### Task 2.1: Tag recognizer for table cells

**Files:**

- Modify: `src/custom_sam_peft/_provenance_check.py`
- Test: `tests/unit/test_provenance_check.py`

- [ ] **Step 1: Write the failing test** (difficulty: medium)

```python
# append to tests/unit/test_provenance_check.py
from custom_sam_peft._provenance_check import recognize_cell_tag


def test_recognize_cell_tag_forms() -> None:
    # Bare aug-style legend letter.
    assert recognize_cell_tag('"hflip": True,  # (a)') == {"letters": ["a"], "kind": "legend"}
    # cite-style single + combined legend letters.
    assert recognize_cell_tag('"x": 1,  # cite: (A)') == {"letters": ["A"], "kind": "legend"}
    assert recognize_cell_tag('"x": 1,  # cite: (A,C)') == {"letters": ["A", "C"], "kind": "legend"}
    # Non-legend cite.
    assert recognize_cell_tag('"x": 1,  # cite: empirical')["kind"] == "cite"
    # tbd.
    assert recognize_cell_tag('"x": 1,  # tbd: #191')["kind"] == "tbd"
    # No tag => None.
    assert recognize_cell_tag('"x": 1,') is None
    assert recognize_cell_tag('"vflip": False,') is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_provenance_check.py -k recognize_cell_tag -v`
Expected: FAIL — `ImportError`.

- [ ] **Step 3: Write minimal implementation**

```python
# add to src/custom_sam_peft/_provenance_check.py
from typing import TypedDict


class CellTag(TypedDict):
    """A recognized inline tag on a preset-table value line."""

    kind: str  # "legend" | "cite" | "tbd"
    letters: list[str]


# Bare aug form: ``# (a)`` or ``# (a,b)``. cite form: ``# cite: (A)`` / ``(A,C)``.
_LEGEND_BARE = re.compile(r"#\s*\(([A-Za-z](?:\s*,\s*[A-Za-z])*)\)\s*$")
_LEGEND_CITE = re.compile(r"#\s*cite:\s*\(([A-Za-z](?:\s*,\s*[A-Za-z])*)\)")
_CITE_ANY = re.compile(r"#\s*cite:\s*(\S.*)$")
_TBD_ANY = re.compile(r"#\s*tbd:\s*(\S.*)$")


def recognize_cell_tag(line: str) -> CellTag | None:
    """Recognize an inline tag on a preset-table value line, or ``None``."""
    for pat in (_LEGEND_CITE, _LEGEND_BARE):
        m = pat.search(line)
        if m is not None:
            letters = [s.strip() for s in m.group(1).split(",")]
            return CellTag(kind="legend", letters=letters)
    if _CITE_ANY.search(line) is not None:
        return CellTag(kind="cite", letters=[])
    if _TBD_ANY.search(line) is not None:
        return CellTag(kind="tbd", letters=[])
    return None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_provenance_check.py -k recognize_cell_tag -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/custom_sam_peft/_provenance_check.py tests/unit/test_provenance_check.py
git commit -m "feat(#192): table-cell tag recognizer (bare + cite + tbd forms)"
```

### Task 2.2: Preset-table cell extraction + legend parsing

**Files:**

- Modify: `src/custom_sam_peft/_provenance_check.py`
- Test: `tests/unit/test_provenance_check.py`

- [ ] **Step 1: Write the failing test** (difficulty: medium)

```python
# append to tests/unit/test_provenance_check.py
from custom_sam_peft._provenance_check import (
    extract_preset_cell_lines,
    parse_doc_legend_letters,
)


def test_extract_preset_cell_lines_includes_legacy_and_alias(tmp_path: Path) -> None:
    src = (
        'PRESET_TABLE = {\n'
        '    ("a", "b"): {\n'
        '        "k": 1,  # cite: (A)\n'
        '    },\n'
        '}\n'
        'PRESET_TABLE[("c", "d")] = dict(PRESET_TABLE[("a", "b")])  # cite: (G)\n'
        '_LEGACY_DEFAULTS: dict[str, Any] = {\n'
        '    "w": 1.0,  # cite: (B)\n'
        '}\n'
    )
    f = tmp_path / "presets.py"
    f.write_text(src)
    cells = extract_preset_cell_lines(f)
    texts = [c.text for c in cells]
    assert any('"k": 1' in t for t in texts)
    assert any("PRESET_TABLE[(\"c\", \"d\")]" in t for t in texts)  # alias line
    assert any('"w": 1.0' in t for t in texts)  # _LEGACY_DEFAULTS cell
    # Each cell carries its 1-based line number.
    assert all(c.lineno >= 1 for c in cells)


def test_parse_doc_legend_letters() -> None:
    body = (
        "### Legend\n\n"
        "| Letter | Meaning |\n"
        "| --- | --- |\n"
        "| (a) | domain |\n"
        "| (e) | recipe |\n"
    )
    assert parse_doc_legend_letters(body) == {"a", "e"}

    cite_body = (
        "### Citation legend\n\n"
        "| Letter | Source | Establishes |\n"
        "| --- | --- | --- |\n"
        "| A | x | y |\n"
        "| H | x | y |\n"
    )
    assert parse_doc_legend_letters(cite_body) == {"A", "H"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_provenance_check.py -k "extract_preset_cell_lines or parse_doc_legend" -v`
Expected: FAIL — `ImportError`.

- [ ] **Step 3: Write minimal implementation**

```python
# add to src/custom_sam_peft/_provenance_check.py


@dataclass(frozen=True)
class CellLine:
    """A preset-table value line: source text + 1-based line number."""

    text: str
    lineno: int


# A value line inside a dict literal: ``"key": <value>,`` (optionally tagged).
_DICT_VALUE_LINE = re.compile(r'^\s*"[^"]+"\s*:\s*.+,?\s*(#.*)?$')
# A module-level alias assignment line in losses/presets.py.
_ALIAS_LINE = re.compile(r"^\s*PRESET_TABLE\[\(.+\)\]\s*=\s*dict\(.+\)")


def _dict_literal_spans(lines: list[str], dict_names: tuple[str, ...]) -> list[range]:
    """Return line-index ranges (0-based, exclusive end) of named top-level dict literals.

    A dict opens on a line matching ``<name>...= {`` and closes on the first
    line that is exactly ``}`` (top-level, no leading whitespace).
    """
    spans: list[range] = []
    for i, line in enumerate(lines):
        stripped = line.rstrip()
        if any(
            re.match(rf"^{re.escape(name)}\b.*=\s*\{{\s*$", line) for name in dict_names
        ):
            for j in range(i + 1, len(lines)):
                if lines[j].rstrip() == "}":
                    spans.append(range(i + 1, j))
                    break
    return spans


def extract_preset_cell_lines(file_path: Path) -> list[CellLine]:
    """All preset-table value lines for Assertion 2 (PRESET_TABLE + aliases + _LEGACY_DEFAULTS)."""
    lines = file_path.read_text(encoding="utf-8").splitlines()
    cells: list[CellLine] = []
    spans = _dict_literal_spans(lines, ("PRESET_TABLE", "_LEGACY_DEFAULTS"))
    in_span: set[int] = set()
    for span in spans:
        in_span.update(span)
    for idx, line in enumerate(lines):
        if idx in in_span and _DICT_VALUE_LINE.match(line):
            cells.append(CellLine(text=line, lineno=idx + 1))
        elif _ALIAS_LINE.match(line):
            cells.append(CellLine(text=line, lineno=idx + 1))
    return cells


_LEGEND_ROW_BARE = re.compile(r"^\|\s*\(([A-Za-z])\)\s*\|")
_LEGEND_ROW_PLAIN = re.compile(r"^\|\s*([A-Za-z])\s*\|")


def parse_doc_legend_letters(section_body: str) -> set[str]:
    """Collect legend letters defined in a table module's doc legend sub-table.

    Accepts both ``| (a) | … |`` (aug) and ``| A | … |`` (losses) row forms.
    """
    letters: set[str] = set()
    for line in section_body.splitlines():
        m = _LEGEND_ROW_BARE.match(line)
        if m is not None:
            letters.add(m.group(1))
            continue
        m = _LEGEND_ROW_PLAIN.match(line)
        if m is not None and m.group(1) not in {"L"}:  # skip the "Letter" header
            letters.add(m.group(1))
    return letters
```

Note on the `_LEGEND_ROW_PLAIN` header guard: the losses legend header row is `| Letter | Source | Establishes |`; the single-letter capture would grab `L` from `Letter`. The guard `m.group(1) not in {"L"}` drops it. The aug legend header `| Letter | Meaning |` is likewise dropped (its first cell is `Letter`, captured as `L`). Separator rows `| --- | … |` do not match either pattern. If a real legend ever defined letter `L`, revisit — none do today (`aug` uses `a–e`, `losses` uses `A–H`).

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_provenance_check.py -k "extract_preset_cell_lines or parse_doc_legend" -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/custom_sam_peft/_provenance_check.py tests/unit/test_provenance_check.py
git commit -m "feat(#192): preset-cell extraction + doc-legend letter parsing"
```

### Task 2.3: `check_table_section` — Assertion 2

**Files:**

- Modify: `src/custom_sam_peft/_provenance_check.py`
- Test: `tests/unit/test_provenance_check.py`

- [ ] **Step 1: Write the failing test** (difficulty: medium)

```python
# append to tests/unit/test_provenance_check.py
from custom_sam_peft._provenance_check import check_table_section


def _table_module(tmp_path: Path, cells: str) -> Path:
    f = tmp_path / "aug_presets.py"
    f.write_text(f"PRESET_TABLE = {{\n    (\"a\", \"b\"): {{\n{cells}    }},\n}}\n")
    return f


def _legend_body(letters: str) -> str:
    rows = "".join(f"| ({c}) | meaning |\n" for c in letters)
    return "### Legend\n\n| Letter | Meaning |\n| --- | --- |\n" + rows


def test_table_all_cells_tagged_and_resolved_passes(tmp_path: Path) -> None:
    f = _table_module(tmp_path, '        "k": 1,  # (a)\n')
    section = Section(header="data/aug_presets.py", body=_legend_body("a"))
    assert check_table_section(section, f) == []


def test_table_untagged_cell_fails(tmp_path: Path) -> None:
    f = _table_module(tmp_path, '        "k": 1,\n')
    section = Section(header="data/aug_presets.py", body=_legend_body("a"))
    violations = check_table_section(section, f)
    assert len(violations) == 1
    msg = str(violations[0])
    assert "aug_presets.py:" in msg
    assert "untagged" in msg


def test_table_undefined_legend_letter_fails(tmp_path: Path) -> None:
    f = _table_module(tmp_path, '        "k": 1,  # (z)\n')
    section = Section(header="data/aug_presets.py", body=_legend_body("a"))  # no (z)
    violations = check_table_section(section, f)
    assert len(violations) == 1
    msg = str(violations[0])
    assert "z" in msg
    assert "undefined legend letter" in msg
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_provenance_check.py -k check_table -v`
Expected: FAIL — `ImportError`.

- [ ] **Step 3: Write minimal implementation**

```python
# add to src/custom_sam_peft/_provenance_check.py


def check_table_section(section: Section, file_path: Path) -> list[ProvenanceViolation]:
    """Assertion 2: every preset-table cell carries a tag; legend letters resolve."""
    file_disp = _unescape_md(section.header).strip()
    defined_letters = parse_doc_legend_letters(section.body)
    violations: list[ProvenanceViolation] = []
    for cell in extract_preset_cell_lines(file_path):
        tag = recognize_cell_tag(cell.text)
        if tag is None:
            violations.append(
                ProvenanceViolation(
                    location=f"{file_disp}:{cell.lineno}",
                    problem="untagged preset cell",
                    remediation="add a legend letter, `# cite:`, or `# tbd:` tag",
                )
            )
            continue
        if tag["kind"] == "legend":
            for letter in tag["letters"]:
                if letter not in defined_letters:
                    violations.append(
                        ProvenanceViolation(
                            location=f"{file_disp}:{cell.lineno}",
                            problem=f"undefined legend letter `{letter}`",
                            remediation=(
                                f"define it in the legend under `## {file_disp}` "
                                "in docs/defaults-provenance.md"
                            ),
                        )
                    )
    return violations
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_provenance_check.py -k check_table -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/custom_sam_peft/_provenance_check.py tests/unit/test_provenance_check.py
git commit -m "feat(#192): Assertion 2 — table tag-presence + legend resolution"
```

### Task 2.4: `check_yaml_section` — Assertion 3 + `schema_default_paths`

**Files:**

- Modify: `src/custom_sam_peft/_provenance_check.py`
- Test: `tests/unit/test_provenance_check.py`

- [ ] **Step 1: Write the failing test** (difficulty: medium)

```python
# append to tests/unit/test_provenance_check.py
import yaml as _yaml  # PyYAML is a base dependency (pyproject `pyyaml>=6.0`), imported across src/

from custom_sam_peft._provenance_check import (
    check_yaml_section,
    yaml_scalar_dotted_paths,
)


def test_yaml_scalar_dotted_paths_flattens(tmp_path: Path) -> None:
    f = tmp_path / "config_full.yaml"
    f.write_text("run:\n  seed: 42\ndata:\n  text_prompt:\n    mode: present\n")
    paths = yaml_scalar_dotted_paths(f)
    assert "run.seed" in paths
    assert "data.text_prompt.mode" in paths


def _yaml_doc_body(rows: str) -> str:
    return (
        "| Location | Value | Tag | Full reference | Verifying quote | Notes |\n"
        "| --- | --- | --- | --- | --- | --- |\n" + rows
    )


def test_yaml_missing_crosslink_for_schema_echo_fails(tmp_path: Path) -> None:
    f = tmp_path / "config_full.yaml"
    f.write_text("run:\n  seed: 42\n")
    body = _yaml_doc_body("")  # no cross-link row for run.seed
    section = Section(header="cli/templates/config_full.yaml", body=body)
    # run.seed echoes a schema default.
    violations = check_yaml_section(section, f, schema_default_paths={"run.seed"})
    assert len(violations) == 1
    msg = str(violations[0])
    assert "config_full.yaml:run.seed" in msg
    assert "cross-link" in msg


def test_yaml_template_only_key_not_required(tmp_path: Path) -> None:
    f = tmp_path / "config_full.yaml"
    f.write_text("run:\n  output_dir: ./x\n")
    body = _yaml_doc_body("")
    section = Section(header="cli/templates/config_full.yaml", body=body)
    # run.output_dir is NOT in the schema-default set -> no requirement.
    assert check_yaml_section(section, f, schema_default_paths=set()) == []


def test_yaml_present_crosslink_passes(tmp_path: Path) -> None:
    f = tmp_path / "config_full.yaml"
    f.write_text("run:\n  seed: 42\n")
    body = _yaml_doc_body("| `config_full.yaml:run.seed` | `42` | `cross-link` | x | — | n |\n")
    section = Section(header="cli/templates/config_full.yaml", body=body)
    assert check_yaml_section(section, f, schema_default_paths={"run.seed"}) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_provenance_check.py -k "yaml" -v`
Expected: FAIL — `ImportError`.

- [ ] **Step 3: Write minimal implementation**

```python
# add to src/custom_sam_peft/_provenance_check.py
from typing import Any

import yaml


def yaml_scalar_dotted_paths(yaml_path: Path) -> set[str]:
    """Dotted paths of every scalar leaf in a YAML file (lists treated as leaves)."""
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    paths: set[str] = set()

    def walk(node: Any, prefix: str) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                walk(value, f"{prefix}.{key}" if prefix else str(key))
        else:
            if prefix:
                paths.add(prefix)

    walk(data, "")
    return paths


def _yaml_crosslink_keys(section_body: str, file_disp_basename: str) -> set[str]:
    """Dotted paths that already have a cross-link row in the yaml doc section."""
    keys: set[str] = set()
    for line in section_body.splitlines():
        m = _ROW.match(line)
        if m is None:
            continue
        loc_match = _LOCATION_CELL.search(m.group("cells").split("|", 1)[0])
        if loc_match is None:
            continue
        loc = loc_match.group("loc").strip()
        if loc.startswith(f"{file_disp_basename}:"):
            keys.add(loc[len(file_disp_basename) + 1 :])
    return keys


def check_yaml_section(
    section: Section, yaml_path: Path, schema_default_paths: set[str]
) -> list[ProvenanceViolation]:
    """Assertion 3: every template scalar echoing a schema default has a cross-link row."""
    basename = yaml_path.name  # rows are keyed "config_full.yaml:<dotted>"
    documented = _yaml_crosslink_keys(section.body, basename)
    template_paths = yaml_scalar_dotted_paths(yaml_path)
    echoing = template_paths & schema_default_paths
    violations: list[ProvenanceViolation] = []
    for dotted in sorted(echoing - documented):
        violations.append(
            ProvenanceViolation(
                location=f"{basename}:{dotted}",
                problem="template scalar echoes a schema default but has no cross-link row",
                remediation=(
                    "add a `cross-link` row to the "
                    "`## cli/templates/config_full.yaml` section"
                ),
            )
        )
    return violations
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_provenance_check.py -k "yaml" -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/custom_sam_peft/_provenance_check.py tests/unit/test_provenance_check.py
git commit -m "feat(#192): Assertion 3 — yaml cross-link coverage"
```

### Task 2.5: `run_all_checks` driver + schema-default-path derivation

**Files:**

- Modify: `src/custom_sam_peft/_provenance_check.py`
- Test: `tests/unit/test_provenance_check.py`

The driver needs the set of dotted schema-default paths (for Assertion 3). Derive it from `config/schema.py` by mapping the pydantic model graph to its dotted YAML paths. To avoid coupling the checker to the live schema in unit tests, `run_all_checks` computes it via a helper `schema_default_dotted_paths(repo_root)` that imports and walks the schema models; the unit test for the *driver* uses a synthetic mini-repo and passes the doc text directly to the lower-level dispatch, so it does not need the real schema.

- [ ] **Step 1: Write the failing test** (difficulty: hard)

```python
# append to tests/unit/test_provenance_check.py
from custom_sam_peft._provenance_check import run_all_checks


def test_run_all_checks_aggregates_and_dispatches(tmp_path: Path) -> None:
    # Build a tiny repo: one prose file with a missing doc row.
    (tmp_path / "src/custom_sam_peft/config").mkdir(parents=True)
    (tmp_path / "src/custom_sam_peft/config/schema.py").write_text(
        "from pydantic import BaseModel, Field\n\n\nclass C(BaseModel):\n    a: int = Field(default=3)\n"
    )
    (tmp_path / "docs").mkdir(parents=True)
    (tmp_path / "docs/defaults-provenance.md").write_text(
        "# Defaults Provenance\n\n"
        "## config/schema.py\n\n"
        "| Location | Value | Tag | Full reference | Verifying quote | Notes |\n"
        "| --- | --- | --- | --- | --- | --- |\n"
        # no row for C.a -> one code->doc violation expected
        "\n"
        "## Verification Standard\n\nnarrative, no check.\n"
    )
    violations = run_all_checks(tmp_path)
    assert len(violations) == 1
    assert "config/schema.py:C.a" in str(violations[0])
```

For the driver test we avoid yaml/table sections (so no real schema import is needed). `run_all_checks` must skip Assertion 3 gracefully when no yaml section is present, and must derive `schema_default_paths` only when a yaml section exists (lazy).

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_provenance_check.py -k run_all_checks -v`
Expected: FAIL — `ImportError`.

- [ ] **Step 3: Write minimal implementation**

```python
# add to src/custom_sam_peft/_provenance_check.py

DOC_REL_PATH = "docs/defaults-provenance.md"


def schema_default_dotted_paths(repo_root: Path) -> set[str]:
    """Dotted YAML paths for every pydantic field that has a default, from the schema.

    Walks ``custom_sam_peft.config.schema.RunConfig`` (the root config model) and
    yields the dotted path of each field whose model field has a default. Used
    only for Assertion 3; imported lazily so the unit tests that pass explicit
    schema-default sets do not need the real schema importable.
    """
    from pydantic import BaseModel

    from custom_sam_peft.config.schema import RunConfig

    paths: set[str] = set()

    def walk(model: type[BaseModel], prefix: str) -> None:
        for name, field in model.model_fields.items():
            dotted = f"{prefix}.{name}" if prefix else name
            annotation = field.annotation
            nested = annotation
            # Unwrap Optional[...] / X | None to find a nested BaseModel.
            args = getattr(annotation, "__args__", ())
            for arg in args:
                if isinstance(arg, type) and issubclass(arg, BaseModel):
                    nested = arg
                    break
            if isinstance(nested, type) and issubclass(nested, BaseModel):
                walk(nested, dotted)
            else:
                has_default = (
                    field.default is not None or field.default_factory is not None
                ) or not field.is_required()
                if has_default:
                    paths.add(dotted)

    walk(RunConfig, "")
    return paths


def run_all_checks(repo_root: Path) -> list[ProvenanceViolation]:
    """Run all three assertions over the real repo; return ALL violations."""
    doc_text = (repo_root / DOC_REL_PATH).read_text(encoding="utf-8")
    violations: list[ProvenanceViolation] = []
    schema_paths: set[str] | None = None
    for section in discover_sections(doc_text):
        kind = classify_section(section, repo_root)
        if kind == "prose-narrative":
            continue
        path = resolve_section_path(section.header, repo_root)
        assert path is not None  # non-narrative => resolved (else FileNotFoundError)
        if kind == "prose":
            violations.extend(check_prose_section(section, path))
        elif kind == "table":
            violations.extend(check_table_section(section, path))
        elif kind == "yaml":
            if schema_paths is None:
                schema_paths = schema_default_dotted_paths(repo_root)
            violations.extend(check_yaml_section(section, path, schema_paths))
    return violations
```

Note: `classify_section` raises `FileNotFoundError` for a missing file-section header; `run_all_checks` lets it propagate, so a doc section naming a nonexistent path is a hard test failure (the spec's "hard FAIL"). The live test (Phase 5) will surface it as an error.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_provenance_check.py -k run_all_checks -v`
Expected: PASS.

- [ ] **Step 5: Run the full unit suite + lint/type gates**

Run: `uv run pytest tests/unit/test_provenance_check.py -v && uv run ruff check src/custom_sam_peft/_provenance_check.py tests/unit/test_provenance_check.py && uv run ruff format --check src/custom_sam_peft/_provenance_check.py tests/unit/test_provenance_check.py && uv run mypy src/custom_sam_peft/_provenance_check.py`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add src/custom_sam_peft/_provenance_check.py tests/unit/test_provenance_check.py
git commit -m "feat(#192): run_all_checks driver + schema-default-path derivation"
```

---

## Phase 3 — Bare-off-cell tagging (table modules) + `(e)` aug-legend doc row

**Interface contract this phase CONSUMES (from Phase 2):** the Assertion-2 tag grammar (bare `# (x)`, `# cite: (X)`, `# cite: (X,Y)`) and the legend-resolution rule (every used letter defined in the module's doc legend). The edits in this phase produce exactly the tag forms Phase 2's recognizer accepts and the letters Phase 4's/this-phase's doc legend defines.

**Interface contract this phase EXPOSES (to Phase 5):** the two table modules are Assertion-2-clean — every preset cell carries a recognized tag, every used legend letter is defined.

**CI state at phase end:** GREEN. These are value-preserving inline-comment edits + one additive doc row; they touch no behavior. ruff/format/mypy unaffected (comments only); markdownlint must pass on the doc edit. (The live test does not exist yet, so partial conformance is fine — but doing the table modules fully here keeps Phase 5 a pure verification step.)

> **CRITICAL — value-preserving only:** every edit in this phase appends or changes an inline `# comment` on an existing line. **No literal value may change.** After each file's edits, diff with `git diff -w --stat` and eyeball that only comments moved.

### Task 3.1: Tag bare off-cells in `data/aug_presets.py`

**Files:**

- Modify: `src/custom_sam_peft/data/aug_presets.py`

Apply the spec's four-way classification to the untagged cells in `PRESET_TABLE` (lines ~50–171). The untagged bare cells today are the `False` booleans and `0.0` magnitudes with no trailing `# (x)`. Tag each per these rules (spec Assertion-2 note):

- **Off symmetry booleans** (`vflip`/`rotate90` that are `False` for a domain-symmetry reason) → `# (a)` (domain convention covers on AND off states).
- **Domain-not-applicable off magnitudes**: clinical/diagnostic lock-offs already using `# (d)` keep `# (d)`; H&E-not-applicable `stain_jitter: 0.0` zeros (outside the H&E domains) → `# (a)` (domain fact).
- **Intensity-tier-omission off magnitudes** (`blur`/`gauss_noise`/`rotate_arbitrary`/`color_jitter` that are `0.0` simply because the gentler tier omits them) → the new `# (e)`.

The concrete untagged cells to tag (verified against the current file):

| Preset×intensity block | Cell | New tag | Why |
| --- | --- | --- | --- |
| `("natural","safe")` | `"vflip": False` | `# (a)` | natural has no vertical symmetry (domain convention) |
| `("natural","safe")` | `"rotate90": False` | `# (a)` | natural has a canonical "up" (domain convention) |
| `("natural","safe")` | `"rotate_arbitrary": 0.0` | `# (e)` | omitted at safe tier |
| `("natural","safe")` | `"stain_jitter": 0.0` | `# (a)` | natural is not H&E (domain-not-applicable) |
| `("natural","safe")` | `"blur": 0.0` | `# (e)` | omitted at safe tier |
| `("natural","safe")` | `"gauss_noise": 0.0` | `# (e)` | omitted at safe tier |
| `("natural","medium")` | `"vflip": False` | `# (a)` | as above |
| `("natural","medium")` | `"rotate90": False` | `# (a)` | as above |
| `("natural","medium")` | `"rotate_arbitrary": 0.0` | `# (e)` | omitted at medium tier |
| `("natural","medium")` | `"stain_jitter": 0.0` | `# (a)` | not H&E |
| `("natural","medium")` | `"blur": 0.0` | `# (e)` | omitted at medium tier |
| `("natural","medium")` | `"gauss_noise": 0.0` | `# (e)` | omitted at medium tier |
| `("natural","aggressive")` | `"stain_jitter": 0.0` | `# (a)` | not H&E |
| `("medical","safe")` | `"rotate_arbitrary": 0.0` | `# (e)` | omitted at safe tier |
| `("medical","safe")` | `"stain_jitter": 0.0` | `# (a)` | H&E domain but tier omits stain jitter at safe → treat as intensity omission `# (e)` IF the value being off is a tier choice, else `# (a)`. **Resolution: `# (e)`** — `stain_jitter` IS applicable to medical/H&E (it is on at medium/aggressive: 0.03/0.07), so its `0.0` at the safe tier is an intensity-tier omission, not domain-not-applicable. |
| `("medical","safe")` | `"blur": 0.0` | `# (e)` | omitted at safe tier |
| `("medical","safe")` | `"gauss_noise": 0.0` | `# (e)` | omitted at safe tier |
| `("medical","medium")` | `"blur": 0.0` | `# (e)` | omitted at medium tier (gauss_noise/stain on, blur off) |
| `("satellite","safe")` | `"rotate_arbitrary": 0.0` | `# (e)` | omitted at safe tier |
| `("satellite","safe")` | `"color_jitter": 0.0` | `# (e)` | omitted at safe tier (satellite uses color_jitter at medium=0.05/aggressive=0.1) |
| `("satellite","safe")` | `"stain_jitter": 0.0` | `# (a)` | satellite is not H&E (domain-not-applicable) |
| `("satellite","safe")` | `"blur": 0.0` | `# (e)` | omitted at safe tier |
| `("satellite","safe")` | `"gauss_noise": 0.0` | `# (e)` | omitted at safe tier |
| `("satellite","medium")` | `"rotate_arbitrary": 0.0` | `# (e)` | omitted at medium tier |
| `("satellite","medium")` | `"stain_jitter": 0.0` | `# (a)` | not H&E |
| `("satellite","medium")` | `"blur": 0.0` | `# (e)` | omitted at medium tier |
| `("satellite","medium")` | `"gauss_noise": 0.0` | `# (e)` | omitted at medium tier |
| `("microscopy","safe")` | `"rotate_arbitrary": 0.0` | `# (e)` | omitted at safe tier |
| `("microscopy","safe")` | `"stain_jitter": 0.0` | `# (a)` | fluorescence microscopy is not H&E |
| `("microscopy","safe")` | `"blur": 0.0` | `# (e)` | omitted at safe tier |
| `("microscopy","safe")` | `"gauss_noise": 0.0` | `# (e)` | omitted at safe tier |
| `("microscopy","medium")` | `"rotate_arbitrary": 0.0` | `# (e)` | omitted at medium tier |
| `("microscopy","medium")` | `"stain_jitter": 0.0` | `# (a)` | not H&E |
| `("microscopy","medium")` | `"blur": 0.0` | `# (e)` | omitted at medium tier |
| `("microscopy","medium")` | `"gauss_noise": 0.0` | `# (e)` | omitted at medium tier |

> **IMPLEMENTER VERIFICATION — do not trust this table blind.** Before editing, run the discovery command in Step 1 to list the *actual* untagged cell lines in the current file (line numbers shift as the file evolves). Tag every untagged value line, classifying each by the three-way rule (off-symmetry-boolean `# (a)` / not-H&E-`stain_jitter` `# (a)` / intensity-tier-omission `# (e)`). The table above is the design intent; the command output is ground truth. The medical-domain `color_jitter: 0.0` and `hflip/vflip/rotate90: False` cells already carry `# (d)` and stay as-is. The aggressive-tier blocks are already fully tagged.

- [ ] **Step 1: Discover the exact untagged cells**

Run: `uv run python -c "import re; lines=open('src/custom_sam_peft/data/aug_presets.py').read().splitlines();
import sys
start=next(i for i,l in enumerate(lines) if l.startswith('PRESET_TABLE'));
end=next(i for i in range(start+1,len(lines)) if lines[i].rstrip()=='}');
[print(i+1, repr(lines[i])) for i in range(start+1,end) if re.match(r'\s*\"[^\"]+\"\s*:', lines[i]) and '#' not in lines[i]]"`
Expected: prints the line numbers + text of every untagged `"key": value` line in `PRESET_TABLE`. This is the authoritative work-list.

- [ ] **Step 2: Apply the tags**

For each untagged line from Step 1, append the classified tag. Example edits (exact strings will be matched against the live file):

```python
# before:  "vflip": False,
# after:   "vflip": False,  # (a)

# before:  "rotate_arbitrary": 0.0,
# after:   "rotate_arbitrary": 0.0,  # (e)

# before:  "stain_jitter": 0.0,   (in a non-H&E domain block)
# after:   "stain_jitter": 0.0,  # (a)

# before:  "blur": 0.0,
# after:   "blur": 0.0,  # (e)
```

Use two spaces before `#` to match the file's existing style (`"hflip": True,  # (a)`).

- [ ] **Step 3: Verify no value changed + every cell now tagged**

Run: `git diff -w src/custom_sam_peft/data/aug_presets.py` — confirm every hunk is a comment-only addition (with `-w`, whitespace-only; here the change is the trailing comment, so review each hunk shows `,` → `,  # (x)`).
Run the discovery command from Step 1 again.
Expected: it prints **nothing** (no untagged cells remain).

- [ ] **Step 4: Run aug-preset behavior tests + lint**

Run: `uv run pytest tests/unit/test_aug_presets.py -v && uv run ruff check src/custom_sam_peft/data/aug_presets.py && uv run ruff format --check src/custom_sam_peft/data/aug_presets.py`
Expected: PASS (no value changed → behavior identical).

- [ ] **Step 5: Commit**

```bash
git add src/custom_sam_peft/data/aug_presets.py
git commit -m "feat(#192): tag bare off-cells in aug_presets (a/d/e honest justification)"
```

### Task 3.2: Tag bare off-cells + `_LEGACY_DEFAULTS` in `models/losses/presets.py`

**Files:**

- Modify: `src/custom_sam_peft/models/losses/presets.py`

Per spec: every `boundary_weight: 0.0` cell across the presets → `# cite: (B)` (off-by-default / opt-in folds into legend `(B)` "preserved pre-#112"). The `_LEGACY_DEFAULTS` base-dict cells (the `preset == "none"` values) are trust-bearing and must each carry a tag:

| `_LEGACY_DEFAULTS` cell | Tag |
| --- | --- |
| `"mask_family": "dice_bce"` | `# cite: (B)` |
| `"box_family": "l1_giou"` | `# cite: (B)` |
| `"obj_family": "focal_bce"` | `# cite: (B)` |
| `"presence_family": "bce"` | `# cite: (B)` |
| `"w_mask": 1.0` | `# cite: (B)` |
| `"w_box": 0.0` | `# cite: (B)` |
| `"w_obj": 1.0` | `# cite: (B)` |
| `"w_presence": 1.0` | `# cite: (B)` |
| `"focal_gamma": 2.0` | `# cite: (A,C)` |
| `"focal_alpha": 0.25` | `# cite: (A,C)` |
| `"tversky_alpha": 0.5` | `# cite: (F)` |
| `"tversky_gamma": 1.0` | `# cite: (F)` |
| `"boundary_weight": 0.0` | `# cite: (B)` |

> **Do NOT tag any `boundary_weight: 0.0` or `_LEGACY_DEFAULTS` cell `# cite: (H)`.** `(H)` is the boundary-loss-ON citation and applies only to `boundary_weight: 0.2` (already tagged `# cite: (A,H)` at line ~183). Verify it stays untouched.
>
> The `_LEGACY_DEFAULTS` `tversky_alpha: 0.5` / `tversky_gamma: 1.0` lines currently carry trailing prose comments (`# neutral — Dice-equivalent; …`). Replace/augment so the line carries a recognized tag: change `# neutral — …` to `# cite: (F)  (neutral — Dice-equivalent; ignored by dice_bce)` so the recognizer sees `# cite: (F)` AND the human note survives. (The recognizer's `_LEGEND_CITE` matches `# cite: (F)` anywhere in the line.)

- [ ] **Step 1: Discover untagged preset cells**

Run: `uv run python -c "import re; lines=open('src/custom_sam_peft/models/losses/presets.py').read().splitlines();
spans=[]
for name in ('PRESET_TABLE','_LEGACY_DEFAULTS'):
    for i,l in enumerate(lines):
        if re.match(rf'^{name}\b.*=\s*\{{\s*$', l):
            for j in range(i+1,len(lines)):
                if lines[j].rstrip()=='}': spans.append(range(i+1,j)); break
idx=set()
[idx.update(s) for s in spans]
[print(i+1, repr(lines[i])) for i in sorted(idx) if re.match(r'\s*\"[^\"]+\"\s*:', lines[i]) and not re.search(r'#\s*(cite|tbd)', lines[i])]"`
Expected: prints the untagged `boundary_weight: 0.0` lines AND the `_LEGACY_DEFAULTS` cells (most carry no tag today; two carry prose-only comments which the regex `#\s*(cite|tbd)` correctly flags as untagged).

- [ ] **Step 2: Apply the tags** per the table above. For the two prose-comment lines, fold the existing note in after the tag:

```python
# before:  "tversky_alpha": 0.5,  # neutral — Dice-equivalent; ignored by dice_bce
# after:   "tversky_alpha": 0.5,  # cite: (F)  (neutral — Dice-equivalent; ignored by dice_bce)
# before:  "tversky_gamma": 1.0,  # neutral — Tversky-equivalent; ignored by dice_bce
# after:   "tversky_gamma": 1.0,  # cite: (F)  (neutral — Tversky-equivalent; ignored by dice_bce)
```

For each `boundary_weight: 0.0` across PRESET_TABLE → append `  # cite: (B)`.

- [ ] **Step 3: Verify no value changed + every cell tagged**

Run: `git diff -w src/custom_sam_peft/models/losses/presets.py` (confirm comment-only).
Run the Step-1 discovery command again.
Expected: prints **nothing**. Also confirm `boundary_weight: 0.2` still reads `# cite: (A,H)` and no `(H)` was added to any `0.0` cell: `grep -n "boundary_weight" src/custom_sam_peft/models/losses/presets.py`.

- [ ] **Step 4: Run loss-preset behavior tests + lint**

Run: `uv run pytest tests/unit/test_loss_presets.py -v && uv run ruff check src/custom_sam_peft/models/losses/presets.py && uv run ruff format --check src/custom_sam_peft/models/losses/presets.py`
Expected: PASS (`tests/unit/test_loss_presets.py` exists; no value changed → behavior identical).

- [ ] **Step 5: Commit**

```bash
git add src/custom_sam_peft/models/losses/presets.py
git commit -m "feat(#192): tag bare off-cells + _LEGACY_DEFAULTS in losses presets ((B)/(A,C)/(F))"
```

### Task 3.3: Add the `(e)` aug-legend row to the doc

**Files:**

- Modify: `docs/defaults-provenance.md`

- [ ] **Step 1: Add the `(e)` row** to the `### Legend` table under `## data/aug_presets.py`, immediately after the `(d)` row:

```markdown
| (e) | Augmentation omitted at this preset's intensity tier — recipe choice; no citation. |
```

This is purely additive; do not reword `(a)`–`(d)`.

- [ ] **Step 2: Markdown-lint the doc**

Run: `uvx --from nodejs-bin@22.9.0 --with markdownlint-cli2 markdownlint-cli2 --config .config/markdownlint-cli2.jsonc docs/defaults-provenance.md`
(If a system `npx` is available, `npx --yes markdownlint-cli2 --config .config/markdownlint-cli2.jsonc docs/defaults-provenance.md` is equivalent — match the CI invocation.)
Expected: no findings (exit 0). Fix any line-length/table findings before committing.

- [ ] **Step 3: Commit**

```bash
git add docs/defaults-provenance.md
git commit -m "docs(#192): add (e) intensity-tier-omission aug legend row"
```

---

## Phase 4 — Curated inline-tag strip (prose files) + doc preamble/Tag-column migration

**Interface contract this phase CONSUMES (from Phase 1):** Assertion 1's bijection rule. Stripping an inline tag does **not** remove a default *value*, so the surface symbol stays and its doc row stays — the bijection is preserved. The keep-list notes are cosmetic comments the checker ignores. (The checker reads the *default surface* and *doc rows*, never the inline tag text in prose files.)

**Interface contract this phase EXPOSES (to Phase 5):** every prose-section file is Assertion-1-clean (it already is, value-wise; this phase only removes redundant comments and migrates the doc prose). The doc preamble + `Tag` column reflect the new principle. Exactly the curated keep-list notes survive.

**CI state at phase end:** GREEN. Comment-only code edits (ruff/format/mypy unaffected); doc edits pass markdownlint. The live test still does not exist; but after this phase the repo IS conformant, so Phase 5 is pure verification.

> **CRITICAL — the strip removes inline `# cite:` / `# tbd:` comments only; never a value, never a docstring, never a non-tag comment.** A "strip" turns `seed: int = 42  # cite: degenerate-case (...)` into `seed: int = 42`. Multi-line `# cite:` comment blocks (e.g. the LR-schedule block in `TrainHyperparams`) are removed in full. After each file, `git diff` must show only comment deletions (plus the keep-list lines unchanged).

### Keep-list (verbatim — the reviewer's veto checklist)

These inline notes **survive** the strip; everything else `# cite:`/`# tbd:` in the six prose files is removed:

- `config/schema.py: NormalizeConfig.mean` and `NormalizeConfig.std` (the `max_length=16  # cite: …` lines at ~269/274 carry the note today; keep them) — ImageNet regression-bait.
- `data/transforms.py: KNOWN_PROCESSOR_STATS["facebook/sam3.1"]` (line ~61 `# cite: empirically verified …`) — source of the `[0.5,0.5,0.5]` stats; keep.
- `config/_internal.py: MatcherWeights.lambda_l1` and `lambda_giou` (lines 32–33 `# cite: degenerate-case (...)`) — read like disabled box terms; keep.
- `config/schema.py: TrainHyperparams.lr_schedule = "plateau"` (line ~566; today the note is the multi-line `# cite: ReduceLROnPlateau … # tbd: #197 — the cosine->plateau default flip.` block at ~567-568) — keep the cosine→plateau flip note.
- `data/transforms.py: _HED_FROM_RGB_MATRIX` (line ~77 `# cite: Ruifrok & Johnston 2001`) — name reflects the inverse's role; keep.
- `config/schema.py: TextPromptConfig.k = 16` (line 241 `# cite: models/sam3.py:MULTIPLEX_CAP`) AND `MultiplexConfig.classes_per_forward = 16` (line 517, same tag) — coupling invariant; keep both.
- `config/schema.py: EvalConfig.mask_threshold = 0.0` (line 645 `# cite: degenerate-case (logit boundary; …)`) — logit-threshold reads as "off"; keep.
- `cli/templates/config_full.yaml`: `text_prompt.negatives_per_image: 4` and `text_prompt.mode: present_plus_negatives` (lines 31–32) — KEEP a one-line inline note on each (they currently have none; **add** a short divergence note, see Task 4.4).

> **STRIP (resolved decisions):** `config/schema.py: TextPromptConfig.negatives_per_image = 0` (line ~240 `# cite: empirical (...)`) → strip. `config/schema.py: PEFTConfig.scope = "vision_decoder"` (line 496 `# tbd: #191 (...)`) → strip. The `_STATS_DIVERGENCE_ATOL` note (transforms line 68) is NOT on the confident-keep list → strip. Everything else in the six prose files → strip.

### Task 4.1: Strip `config/_internal.py` (keep matcher-weights notes)

**Files:**

- Modify: `src/custom_sam_peft/config/_internal.py`

- [ ] **Step 1: Strip the one non-keep tag.** Lines 32–33 (`lambda_l1`, `lambda_giou`) are keep-list → leave their `# cite: degenerate-case (...)` notes. Line 34 (`lambda_mask`) is not keep-list → strip:

```python
# before:  lambda_mask: float = 5.0  # tbd: #191 (mask-only matcher cost; no verified source)
# after:   lambda_mask: float = 5.0
```

- [ ] **Step 2: Verify** `git diff src/custom_sam_peft/config/_internal.py` shows only the line-34 comment removed; lines 32–33 unchanged.

- [ ] **Step 3: Lint** `uv run ruff check src/custom_sam_peft/config/_internal.py && uv run ruff format --check src/custom_sam_peft/config/_internal.py`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add src/custom_sam_peft/config/_internal.py
git commit -m "refactor(#192): strip redundant lambda_mask tag (keep matcher-weights notes)"
```

### Task 4.2: Strip `data/channel_semantics.py` (full strip)

**Files:**

- Modify: `src/custom_sam_peft/data/channel_semantics.py`

No channel_semantics symbol is on the keep-list → strip all four tags (lines 15, 16, 44, 52):

```python
# 15:  _IMAGENET_MEAN = (0.485, 0.456, 0.406)  # cite: ImageNet-1k stats (torchvision); see provenance doc
#  ->  _IMAGENET_MEAN = (0.485, 0.456, 0.406)
# 16:  _IMAGENET_STD = (0.229, 0.224, 0.225)  # cite: ...
#  ->  _IMAGENET_STD = (0.229, 0.224, 0.225)
# 44:  # cite: degenerate-case (neutral alpha) -- 0.5 mean/std maps alpha in [0,1] to [-1,1]   (this is a standalone comment line — remove the line)
# 52:  normalize_default=((0.449,), (0.226,)),  # cite: torchvision grayscale-ImageNet
#  ->  normalize_default=((0.449,), (0.226,)),
```

> Line 44 is a standalone `# cite:` comment line, not a trailing comment. Removing it must not orphan the value it annotated; confirm the value on the next code line is untouched. If the comment documents a multi-line expression, keep the code, drop only the comment line.

- [ ] **Step 1: Strip all four.** (Read lines 40–55 first to confirm line 44's exact context before deleting.)

- [ ] **Step 2: Verify** `git diff src/custom_sam_peft/data/channel_semantics.py` — comment-only deletions; the `_IMAGENET_*` tuples and `normalize_default` values unchanged.

- [ ] **Step 3: Lint + behavior** `uv run pytest tests/unit/test_channel_semantics.py -v && uv run ruff check src/custom_sam_peft/data/channel_semantics.py && uv run ruff format --check src/custom_sam_peft/data/channel_semantics.py`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add src/custom_sam_peft/data/channel_semantics.py
git commit -m "refactor(#192): strip inline provenance tags in channel_semantics (doc-only)"
```

### Task 4.3: Strip `data/transforms.py`, `presets.py`, `predict/budget.py` (keep HED + processor-stats)

**Files:**

- Modify: `src/custom_sam_peft/data/transforms.py`
- Modify: `src/custom_sam_peft/presets.py`
- Modify: `src/custom_sam_peft/predict/budget.py`

`transforms.py` keep-list: line 61 (`KNOWN_PROCESSOR_STATS … # cite: empirically verified`) and line 77 (`_HED_FROM_RGB_MATRIX … # cite: Ruifrok & Johnston 2001`). Strip the rest:

```python
# 68:  _STATS_DIVERGENCE_ATOL = 1e-3  # cite: empirical (...)   -> strip comment
# 88:  _GAUSS_NOISE_MAX_VAR: float = 0.05  # tbd: #191           -> strip comment
# 89:  _GAUSS_BLUR_MAX_SIGMA: float = 3.0  # tbd: #191           -> strip comment
```

`presets.py`: no symbol is on the keep-list → strip every `# cite:` (lines 41, 46, 48, 50, 52, 54, 62, 80, 168, 170, 177, 194). Several are standalone comment lines above a constant (41/46/48/.../62/168/170) and some trailing (80). For standalone `# cite:` lines, remove the comment line; keep the constant. The multi-line `# cite: measured on NVIDIA … FLASH` block at ~62 spans more than one line — remove the full `# cite:` block (read 58–66 first).

`predict/budget.py`: lines 3 and 6 are standalone `# cite:` / `# tbd:` comment lines above `PREDICT_8GB_BUDGET_GB`. Strip both comment lines; keep the constant.

- [ ] **Step 1: Read each file's tag regions** (`transforms.py` 55–90, `presets.py` 38–200, `budget.py` 1–10) to capture exact comment text, then strip per above. Keep lines 61 and 77 in `transforms.py`.

- [ ] **Step 2: Verify** `git diff -w src/custom_sam_peft/data/transforms.py src/custom_sam_peft/presets.py src/custom_sam_peft/predict/budget.py` — comment-only; all constant values unchanged; transforms lines 61/77 retained.

- [ ] **Step 3: Lint + behavior + type** `uv run pytest tests/unit/test_data_transforms.py -v && uv run ruff check src/custom_sam_peft/data/transforms.py src/custom_sam_peft/presets.py src/custom_sam_peft/predict/budget.py && uv run ruff format --check src/custom_sam_peft/data/transforms.py src/custom_sam_peft/presets.py src/custom_sam_peft/predict/budget.py && uv run mypy src/custom_sam_peft`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add src/custom_sam_peft/data/transforms.py src/custom_sam_peft/presets.py src/custom_sam_peft/predict/budget.py
git commit -m "refactor(#192): strip inline tags in transforms/presets/budget (keep HED + processor-stats)"
```

### Task 4.4: Strip `config/schema.py` (apply keep-list + add yaml divergence notes)

**Files:**

- Modify: `src/custom_sam_peft/config/schema.py`
- Modify: `src/custom_sam_peft/cli/templates/config_full.yaml`

In `schema.py`, **keep** these inline notes (do not strip): `mean`/`std` max_length lines (269, 274), `k=16` (241), `classes_per_forward=16` (517), `mask_threshold=0.0` (645), the `lr_schedule` cosine→plateau block (the multi-line `# cite: … # tbd: #197 …` at ~566-568 — keep the part annotating `lr_schedule`).

**Strip** every other `# cite:`/`# tbd:` in `schema.py`, including: `seed` (110, 361), `dtype` (117), `negatives_per_image` (240), `max_pixel_value` (285), `fraction` (340), `quant_type`/`compute_dtype`/`use_double_quant` (486–488), `r`/`alpha`/`dropout`/`scope` (493–496), `bias` (505), the rung-1 reduce-on-plateau `factor`/`patience`/`min_lr`/`min_delta` comment blocks (524–557 — these annotate `ReduceLROnPlateauConfig`/`EarlyStoppingConfig` knobs, NOT `lr_schedule`; strip), `batch_size`/`grad_accum_steps`/`optimizer`/`learning_rate`/`warmup_steps` (562–569 — but the `lr_schedule` field at 566 keeps its note; the `# cite: ReduceLROnPlateau … #197` block at 567–568 annotates `lr_schedule`, KEEP), `log_every` (578), `max_grad_norm` (580), the host-RAM-floor `# tbd:` (613), `nan_abort_after` (628), the workers cap `# tbd:` (633), `iou_thresholds` COCO cite (640), `mode`/`lite_max_images`/`visualize`/`visualize_count` (643–649).

> This file is the largest strip. **The keep/strip boundary around `lr_schedule` (566) is subtle:** lines 567–568 are a comment block; determine which field they annotate by reading 560–570. The `# cite: ReduceLROnPlateau … # tbd: #197 — the cosine->plateau default flip.` text is the `lr_schedule` flip note → KEEP. The rung-1 `factor`/`min_lr`/`patience`/`min_delta` blocks (520–557) annotate the plateau/early-stop *sub-config knobs*, not `lr_schedule` → STRIP.

- [ ] **Step 1: Read `schema.py` 105–125, 235–300, 480–570, 605–650** to map every tag to its field, then strip the non-keep tags. Preserve all field defaults and docstrings.

- [ ] **Step 2: Add the yaml divergence notes** in `config_full.yaml` (these get *added*, not stripped — they currently have none):

```yaml
  text_prompt:
    mode: present_plus_negatives  # differs from schema default `present` — pairs with negatives_per_image
    negatives_per_image: 4  # differs from schema default `0` — headroom for typical COCO present-class counts
```

- [ ] **Step 3: Verify** `git diff src/custom_sam_peft/config/schema.py` — only non-keep comments removed; all keep-list notes present; no value/docstring changed. `git diff src/custom_sam_peft/cli/templates/config_full.yaml` — only the two divergence comments added.

- [ ] **Step 4: Lint + type + schema tests** `uv run pytest tests/unit/test_config_schema.py tests/unit/test_config_examples.py -v && uv run ruff check src/custom_sam_peft/config/schema.py && uv run ruff format --check src/custom_sam_peft/config/schema.py && uv run mypy src/custom_sam_peft`
Expected: PASS. (config_full.yaml is YAML — not ruff/mypy-scanned, but `test_config_examples.py` loads the templates; ensure the added comments don't break parsing.)

- [ ] **Step 5: Commit**

```bash
git add src/custom_sam_peft/config/schema.py src/custom_sam_peft/cli/templates/config_full.yaml
git commit -m "refactor(#192): strip schema tags per keep-list; add yaml divergence notes"
```

### Task 4.5: Doc migration — preamble reword + `Tag`-column redefinition

**Files:**

- Modify: `docs/defaults-provenance.md`

- [ ] **Step 1: Reword the preamble.** Replace the sentence:

```markdown
default hyperparameter in `custom-sam-peft`. Inline `# cite:` / `# tbd:` tags in
the code are deliberately terse pointers into the rows below.
```

with:

```markdown
default hyperparameter in `custom-sam-peft`. This document is the **home** for
provenance; inline `# cite:` / `# tbd:` tags in the code are **no longer the
primary code↔doc pointer**. A small curated set of head-turner defaults retains
an inline note purely as a reader's "wait, that's intentional" guard, not as the
canonical provenance pointer. A CI completeness check (`tests/test_defaults_provenance.py`)
keeps this registry in sync with the code.
```

- [ ] **Step 2: Redefine the `Tag` column** row-schema bullet. Replace:

```markdown
- **Tag** — the inline tag class applied (mirrors the code), or `index-only` for
  untagged self-evident structural/string defaults.
```

with:

```markdown
- **Tag** — the provenance class of the row — one of `cite`, `tbd`, `index-only`,
  or `cross-link`. This is the row's classification in this registry; it no longer
  mirrors an inline code tag (most defaults now carry no inline tag).
```

Do not rewrite any per-row `Tag` cell values; only the definition changes.

- [ ] **Step 3: Markdown-lint** `uvx --from nodejs-bin@22.9.0 --with markdownlint-cli2 markdownlint-cli2 --config .config/markdownlint-cli2.jsonc docs/defaults-provenance.md` (or the CI `npx` form).
Expected: no findings. Fix line-length wraps before committing.

- [ ] **Step 4: Commit**

```bash
git add docs/defaults-provenance.md
git commit -m "docs(#192): reword preamble + redefine Tag column as provenance class"
```

---

## Phase 5 — Live enforcement test + final green gate

**Interface contract this phase CONSUMES:** `run_all_checks(repo_root) -> list[ProvenanceViolation]` (Phase 2). The repo is now conformant (Phases 3–4): Assertions 1–3 over the real repo return `[]`.

**Interface contract this phase EXPOSES:** none downstream — this is the terminal phase. It opens the PR.

**CI state at phase end:** GREEN, including the new live test. This is the acceptance gate.

### Task 5.1: Add the live enforcement test

**Files:**

- Create: `tests/test_defaults_provenance.py`

- [ ] **Step 1: Write the test** (difficulty: easy)

```python
# tests/test_defaults_provenance.py
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
```

> `parents[1]` from `tests/test_defaults_provenance.py` is the repo root (where `docs/` and `src/` live). Confirm by checking `(_REPO_ROOT / "docs/defaults-provenance.md").is_file()` if the assertion errors with a path problem.

- [ ] **Step 2: Run the live test**

Run: `uv run pytest tests/test_defaults_provenance.py -v`
Expected: PASS. If it FAILS, the message lists every violation (file:symbol / file:line + what's missing + remediation) — fix the offending file/doc (a missed strip artifact, an untagged cell, or a stale doc row) and re-run. Do NOT weaken the check.

- [ ] **Step 3: Commit**

```bash
git add tests/test_defaults_provenance.py
git commit -m "test(#192): live defaults-provenance completeness gate over real repo"
```

### Task 5.2: Full-suite + full-gate verification

**Files:** none (verification only).

- [ ] **Step 1: Run the complete CI gate locally**

Run: `uv run ruff check && uv run ruff format --check && uv run mypy src/custom_sam_peft && uv run pytest`
Expected: all PASS, including `--cov-fail-under=80` (the new module is covered by `tests/unit/test_provenance_check.py`). If coverage on `_provenance_check.py` is below threshold, add a targeted unit test for the uncovered branch (e.g. the `schema_default_dotted_paths` Optional-unwrap path or the alias-line/`_LEGACY_DEFAULTS` extraction) — do not lower the gate.

> **GPU-suite caution (from project memory):** a bare `uv run pytest` runs the whole suite in one process. If running locally on the 16 GB dev box risks the real-model GPU tests, scope to the relevant CPU dirs for the inner loop — `uv run pytest tests/unit/test_provenance_check.py tests/test_defaults_provenance.py -o "addopts="` to bypass the global coverage gate during iteration — but the FINAL gate must be the full `uv run pytest` (CI runs it), so run it once at the end (or rely on CI) rather than skipping it.

- [ ] **Step 2: Markdown-lint all docs**

Run: `npx --yes markdownlint-cli2 --config .config/markdownlint-cli2.jsonc "**/*.md" "#node_modules"` (CI's exact invocation), or the `uvx --from nodejs-bin@22.9.0 --with markdownlint-cli2 …` equivalent if no system node. Include the spec + this plan (CI lints all `**/*.md`).
Expected: no findings.

- [ ] **Step 3: Sanity-check the keep-list veto**

Run: `for f in config/schema.py config/_internal.py data/channel_semantics.py data/transforms.py presets.py predict/budget.py; do echo "== $f =="; grep -nE '#\s*(cite|tbd):' "src/custom_sam_peft/$f"; done`
Expected: the ONLY surviving `# cite:`/`# tbd:` lines are the confident-keep set (schema mean/std/k/classes_per_forward/mask_threshold/lr_schedule-flip; internal lambda_l1/lambda_giou; transforms processor-stats/HED). Everything else stripped. This is the reviewer's veto checklist — confirm it matches before PR.

- [ ] **Step 4: Open the PR** (orchestrator close-out — link spec + plan, `--assignee @me`, `--label`).

---

## Self-Review (planner's own pass against the spec)

**Spec coverage:**

- Checker module `_provenance_check.py`, pure functions, explicit repo-root → Phase 1–2. ✓
- Scope auto-derived from `## <path>` headers; no second path list → `discover_sections` + `classify_section`, Task 1.2/1.5. ✓
- Section classification (4 classes) + base-path resolution + missing-file hard fail → Task 1.2. ✓
- Default surface (pydantic/dataclass/module-const; excludes in-function literals) → Task 1.3. ✓
- Assertion 1 both directions + in-function-literal exemption → Task 1.4/1.5. ✓
- Assertion 2 tag-presence + legend resolution + both tag syntaxes + `_LEGACY_DEFAULTS` + alias lines → Task 2.1/2.2/2.3. ✓
- Assertion 3 cross-link coverage + schema-echo definition → Task 2.4/2.5. ✓
- Failure-output contract (file:symbol / file:line + missing + remediation; all violations) → `ProvenanceViolation` + each `check_*` aggregating, never short-circuiting. ✓
- Bare-off-cell tagging (aug a/d/e; losses B/A,C/F; never (H)) → Phase 3. ✓
- `(e)` aug legend row added; losses legend unchanged; no rows deleted → Task 3.3 + Task 4.5 (preamble/Tag only). ✓
- Curated strip with keep-list + resolved keep/strip decisions → Phase 4 (4.1–4.4). ✓
- Doc preamble reword + Tag-column redefinition → Task 4.5. ✓
- Unit tests over synthetic fixtures (6 cases incl. in-function exclusion) → Tasks 1.1–2.5 collectively (documented-pass, undocumented-fail, orphaned-fail, untagged-cell-fail, undefined-letter-fail, in-function-exclusion). ✓
- Live test green after strip+tagging+doc → Phase 5. ✓
- Lint gates (ruff/format/mypy/markdownlint) → verification steps in every phase + Task 5.2. ✓
- Out-of-scope untouched; no value changes → enforced by "value-preserving only" guards in Phases 3–4. ✓

**Placeholder scan:** no TODO/TBD/"handle edge cases" placeholders; every code step shows code; every strip task names exact lines + before/after.

**Type consistency:** `ProvenanceViolation`, `Section`, `DocRow`, `CellLine`, `CellTag`, `SectionClass` used consistently; `check_prose_section`/`check_table_section`/`check_yaml_section`/`run_all_checks` signatures match the phase-boundary contracts and the live test's single call to `run_all_checks`.

## Coverage gaps / risks for the orchestrator

1. **Line numbers drift.** Every strip/tag task cites current line numbers (verified against the live worktree at planning time), but they shift as edits land within a file. Each task includes a discovery command or "read region first" step — implementers must use those as ground truth, not the cited numbers.
2. **`schema_default_dotted_paths` is the trickiest piece (Task 2.5, hard).** Walking the pydantic model graph to dotted YAML paths can mis-handle `Optional[NestedModel]`, `Literal`/`Union` annotations, or list-typed fields. The unit tests pass explicit schema-default sets to dodge this; only the live Assertion-3 run exercises the real walker. If the live test (Phase 5) reports spurious Assertion-3 violations, the bug is almost certainly here — debug `schema_default_dotted_paths` against `config_full.yaml`'s actual cross-link rows, not the checker's table logic. Cross-check: the doc's yaml section already enumerates the expected cross-link keys, so the walker's output should be a superset that intersects the template's scalars to exactly those keys.
3. **`_dict_literal_spans` brittleness.** Cell extraction keys off `<NAME> ... = {` opening and a bare `}` close at column 0. If a future edit indents the closing brace or splits the dict, extraction silently drops cells. The synthetic tests cover the happy path; the live test catches real-repo regressions. Acceptable for this PR (the two modules' formatting is stable and ruff-enforced).
4. **markdownlint tool discovery.** Project memory notes CI uses `npx markdownlint-cli2` but local runs may have no system node → use the `uvx --from nodejs-bin@… --with markdownlint-cli2` form. The plan offers both; the orchestrator should confirm which works in the worktree before the doc-edit commits.
5. **Coverage gate timing.** Phase 1 and 2 land the module *with* its unit tests, so coverage stays ≥80% at every commit. If an implementer commits `_provenance_check.py` additions ahead of their tests within a task, the intermediate state could dip below 80% — keep each task's test+impl in one commit (the task structure already does this).
6. **`yaml` import — RESOLVED at planning.** PyYAML is already a base runtime dependency (`pyproject.toml` line 16: `pyyaml>=6.0`) and imported across `src/` (config loader, trainer, runner). The checker's `import yaml` is safe; no dependency change needed.
