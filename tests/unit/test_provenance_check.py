# tests/unit/test_provenance_check.py
from __future__ import annotations

from pathlib import Path

import pytest

from custom_sam_peft._provenance_check import (
    ProvenanceViolation,
    Section,
    check_prose_section,
    classify_section,
    discover_sections,
    extract_default_surface,
    parse_prose_rows,
    resolve_section_path,
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


def test_surface_collects_pydantic_dataclass_and_module_constants(tmp_path: Path) -> None:
    src = """
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
"""
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


def _prose_doc_body(rows: str) -> str:
    return (
        "| Location | Value | Tag | Full reference | Verifying quote | Notes |\n"
        "| --- | --- | --- | --- | --- | --- |\n" + rows
    )


_SCHEMA_SRC = (
    "from pydantic import BaseModel, Field\n\n\n"
    "class C(BaseModel):\n"
    "    a: int = Field(default=3)\n"
)


def test_prose_documented_default_passes(tmp_path: Path) -> None:
    f = tmp_path / "schema.py"
    f.write_text(_SCHEMA_SRC)
    body = _prose_doc_body("| `schema.py:C.a` | `3` | `# cite: x` | — | — | n |\n")
    section = Section(header="schema.py", body=body)
    assert check_prose_section(section, f) == []


def test_prose_undocumented_default_fails_codedoc(tmp_path: Path) -> None:
    f = tmp_path / "schema.py"
    f.write_text(_SCHEMA_SRC)
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
    f.write_text(_SCHEMA_SRC)
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
