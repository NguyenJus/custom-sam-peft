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
    module_assign_names,
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


def test_surface_collects_literal_module_consts_and_nested_class_fields(tmp_path: Path) -> None:
    src = """
from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Literal
from pydantic import BaseModel, Field

__all__ = ["Cfg"]                       # dunder -> excluded
pytestmark = []                         # pytest marker -> excluded
Dtype = Literal["a", "b"]               # type alias -> excluded
_LOG = logging.getLogger(__name__)      # Call RHS -> excluded (not literal)
_GB = 1024 ** 3                         # BinOp RHS -> excluded (not literal)
MODEL_PARAMS = 5_000_000_000            # literal -> INCLUDED
_IMAGENET_MEAN = (0.485, 0.456, 0.406)  # Tuple literal -> INCLUDED


@dataclass
class Wandb:                            # imported-style container (here: local)
    project: str = "p"


class Inner(BaseModel):
    k: int = Field(default=16)
    model_config = {"x": 1}            # model_config -> excluded


class Cfg(BaseModel):
    epochs: int                        # required (no default) -> exempt
    plain: int = 5                     # leaf
    inner: Inner = Field(default_factory=Inner)   # local container -> recurse, suppress
    wandb: Wandb = Field(default_factory=Wandb)   # local dataclass container -> recurse
"""
    f = tmp_path / "m.py"
    f.write_text(src)
    surface = extract_default_surface(f)
    # literal module constants
    assert "MODEL_PARAMS" in surface
    assert "_IMAGENET_MEAN" in surface
    # excluded module symbols
    for excluded in ("__all__", "pytestmark", "Dtype", "_LOG", "_GB"):
        assert excluded not in surface, excluded
    # nested recursion + container suppression (keyed by defining class)
    assert "Inner.k" in surface
    assert "Cfg.inner" not in surface  # container suppressed, not a leaf
    assert "Wandb.project" in surface  # dataclass container recursed
    assert "Cfg.wandb" not in surface
    # required field exempt; model_config excluded
    assert "Cfg.epochs" not in surface
    assert "Cfg.plain" in surface
    assert "Inner.model_config" not in surface

    # module_assign_names returns ALL module-level names (any RHS) for the
    # doc->code module-constant exemption.
    names = module_assign_names(f)
    assert {"MODEL_PARAMS", "_LOG", "_GB", "Dtype"} <= names


def test_surface_excludes_override_mirror_real_classes(tmp_path: Path) -> None:
    # The real impl hard-codes OVERRIDE_MIRROR = {"AugmentationOverrides", "LossOverrides"}.
    src = """
from pydantic import BaseModel, Field


class LossOverrides(BaseModel):
    w_box: float | None = Field(default=None)
    mask_family: str | None = None


class TextPromptConfig(BaseModel):
    k: int = Field(default=16)
"""
    f = tmp_path / "schema.py"
    f.write_text(src)
    surface = extract_default_surface(f)
    assert not any(s.startswith("LossOverrides.") for s in surface)
    assert "TextPromptConfig.k" in surface


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
    # AMENDED (#192): DocRow also carries the Value cell and a subscript-key flag.
    assert by_symbol["MODEL_PARAMS"].value == "1"
    assert by_symbol["MODEL_PARAMS"].is_subscript_key is False


def test_parse_prose_rows_flags_subscript_keys_and_required_value() -> None:
    body = (
        "| Location | Value | Tag | Full reference | Verifying quote | Notes |\n"
        "| --- | --- | --- | --- | --- | --- |\n"
        '| `schema.py:CHANNEL_SEMANTICS["rgb"].x` | `1` | `index-only` | — | — | n |\n'
        "| `schema.py:T.epochs` | `required (slot)` | `# cite: x` | — | — | n |\n"
    )
    rows = parse_prose_rows(body, section_header="schema.py")
    sub = next(r for r in rows if r.is_subscript_key)
    assert "[" in sub.symbol
    epochs = next(r for r in rows if r.symbol == "T.epochs")
    assert "required" in epochs.value.lower()
    assert epochs.is_subscript_key is False


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


def test_prose_required_field_row_exempt_from_doccode(tmp_path: Path) -> None:
    # `epochs` is required (no default) -> NOT in the surface; its row's Value cell
    # marks it `required`, so the row is not an orphan.
    f = tmp_path / "schema.py"
    f.write_text("from pydantic import BaseModel\n\n\nclass T(BaseModel):\n    epochs: int\n")
    body = _prose_doc_body(
        "| `schema.py:T.epochs` | `required (template slot)` | `# cite: x` | — | — | n |\n"
    )
    section = Section(header="schema.py", body=body)
    assert check_prose_section(section, f) == []


def test_prose_subscript_keyed_row_exempt_from_doccode(tmp_path: Path) -> None:
    # A subscript/call-path key the AST surface cannot emit -> exempt; the
    # container constant itself gets its own (index-only) row.
    f = tmp_path / "channel_semantics.py"
    f.write_text("CHANNEL_SEMANTICS = {'rgb': 1}\n")
    body = _prose_doc_body(
        "| `channel_semantics.py:CHANNEL_SEMANTICS` | `{...}` | `index-only` | — | — | n |\n"
        '| `channel_semantics.py:CHANNEL_SEMANTICS["rgb"].x` | `1` | `# cite: y` | — | — | n |\n'
    )
    section = Section(header="channel_semantics.py", body=body)
    assert check_prose_section(section, f) == []


def test_prose_module_constant_row_exempt_even_with_call_rhs(tmp_path: Path) -> None:
    # `_HED = np.array(...)` is a Call RHS -> NOT in the literal surface, so it is
    # not demanded code->doc; but a doc row pointing at it must NOT be an orphan
    # (it is a real module-level assigned name).
    f = tmp_path / "transforms.py"
    f.write_text("import numpy as np\n\n_HED = np.array([[1.0]])\n")
    body = _prose_doc_body("| `transforms.py:_HED` | `[[1.0]]` | `# cite: x` | — | — | n |\n")
    section = Section(header="transforms.py", body=body)
    assert check_prose_section(section, f) == []


def test_prose_stale_module_row_is_orphan(tmp_path: Path) -> None:
    # A doc row naming a module symbol that no longer exists IS an orphan.
    f = tmp_path / "presets.py"
    f.write_text("A_FIXED = 0\n")
    body = _prose_doc_body(
        "| `presets.py:A_FIXED` | `0` | `# cite: x` | — | — | n |\n"
        "| `presets.py:BASE_ACTIVATION_AT_1024` | `gone` | `# cite: y` | — | — | n |\n"
    )
    section = Section(header="presets.py", body=body)
    violations = check_prose_section(section, f)
    assert len(violations) == 1
    assert "BASE_ACTIVATION_AT_1024" in str(violations[0])


def test_surface_terminates_on_container_cycle(tmp_path: Path) -> None:
    # A mutual cycle: A.b defaults to B() and B.a defaults to A().
    # Without the cycle guard this would cause infinite recursion / non-termination.
    # The guard is per-descent-path, so both classes still emit their own leaf fields.
    src = """
from pydantic import BaseModel, Field


class A(BaseModel):
    x: int = Field(default=1)
    b: object = Field(default_factory=B)


class B(BaseModel):
    y: int = Field(default=2)
    a: object = Field(default_factory=A)
"""
    f = tmp_path / "cycle.py"
    f.write_text(src)
    # Must return without RecursionError / hanging.
    surface = extract_default_surface(f)
    # Each class's own leaf fields must be present.
    assert "A.x" in surface
    assert "B.y" in surface
    # The container fields themselves are suppressed (they're container references).
    assert "A.b" not in surface
    assert "B.a" not in surface


def test_prose_required_when_value_is_not_exempt(tmp_path: Path) -> None:
    # A Value cell like "field is required when X enabled" contains "required" but
    # does NOT start with it; the tightened startswith check must NOT exempt such a
    # row from the orphan check (the old `in` check would have wrongly exempted it).
    f = tmp_path / "schema.py"
    f.write_text(
        "from pydantic import BaseModel, Field\n\n\n"
        "class C(BaseModel):\n    a: int = Field(default=3)\n"
    )
    # "field is required only when mask_decoder enabled" — contains "required" mid-string.
    gone_row = (
        "| `schema.py:C.gone` | `field is required only when mask_decoder enabled`"
        " | `# cite: y` | — | — | n |\n"
    )
    body = _prose_doc_body("| `schema.py:C.a` | `3` | `# cite: x` | — | — | n |\n" + gone_row)
    section = Section(header="schema.py", body=body)
    violations = check_prose_section(section, f)
    # C.gone is an orphan; "field is required only when ..." must NOT exempt it.
    assert len(violations) == 1
    assert "C.gone" in str(violations[0])


def test_prose_defining_class_rekey(tmp_path: Path) -> None:
    # Outer-rooted doc rows (TrainHyperparams.early_stop.*) are re-keyed to the
    # defining-class form (EarlyStopConfig.*) before matching the surface.
    f = tmp_path / "schema.py"
    f.write_text(
        "from pydantic import BaseModel, Field\n\n\n"
        "class EarlyStopConfig(BaseModel):\n    enabled: bool = True\n\n\n"
        "class TrainHyperparams(BaseModel):\n"
        "    early_stop: EarlyStopConfig = Field(default_factory=EarlyStopConfig)\n"
    )
    body = _prose_doc_body(
        "| `schema.py:TrainHyperparams.early_stop.enabled` | `True` | `index-only` | — | — | n |\n"
    )
    section = Section(header="schema.py", body=body)
    assert check_prose_section(section, f) == []


# ---------------------------------------------------------------------------
# Phase 2 tests
# ---------------------------------------------------------------------------

import yaml as _yaml  # noqa: E402,F401 — PyYAML is a base dependency; mid-file per-task append

from custom_sam_peft._provenance_check import (  # noqa: E402
    check_table_section,
    check_yaml_section,
    extract_preset_cell_lines,
    parse_doc_legend_letters,
    recognize_cell_tag,
    run_all_checks,
    yaml_scalar_dotted_paths,
)


def test_extract_preset_cell_lines_includes_legacy_and_alias(tmp_path: Path) -> None:
    src = (
        "PRESET_TABLE = {\n"
        '    ("a", "b"): {\n'
        '        "k": 1,  # cite: (A)\n'
        "    },\n"
        "}\n"
        'PRESET_TABLE[("c", "d")] = dict(PRESET_TABLE[("a", "b")])  # cite: (G)\n'
        "_LEGACY_DEFAULTS: dict[str, Any] = {\n"
        '    "w": 1.0,  # cite: (B)\n'
        "}\n"
    )
    f = tmp_path / "presets.py"
    f.write_text(src)
    cells = extract_preset_cell_lines(f)
    texts = [c.text for c in cells]
    assert any('"k": 1' in t for t in texts)
    assert any('PRESET_TABLE[("c", "d")]' in t for t in texts)  # alias line
    assert any('"w": 1.0' in t for t in texts)  # _LEGACY_DEFAULTS cell
    # Each cell carries its 1-based line number.
    assert all(c.lineno >= 1 for c in cells)


def test_parse_doc_legend_letters() -> None:
    body = "### Legend\n\n| Letter | Meaning |\n| --- | --- |\n| (a) | domain |\n| (e) | recipe |\n"
    assert parse_doc_legend_letters(body) == {"a", "e"}

    cite_body = (
        "### Citation legend\n\n"
        "| Letter | Source | Establishes |\n"
        "| --- | --- | --- |\n"
        "| A | x | y |\n"
        "| H | x | y |\n"
    )
    assert parse_doc_legend_letters(cite_body) == {"A", "H"}


def test_run_all_checks_aggregates_and_dispatches(tmp_path: Path) -> None:
    # Build a tiny repo: one prose file with a missing doc row.
    (tmp_path / "src/custom_sam_peft/config").mkdir(parents=True)
    (tmp_path / "src/custom_sam_peft/config/schema.py").write_text(
        "from pydantic import BaseModel, Field\n\n\n"
        "class C(BaseModel):\n    a: int = Field(default=3)\n"
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


def _table_module(tmp_path: Path, cells: str) -> Path:
    f = tmp_path / "aug_presets.py"
    f.write_text(f'PRESET_TABLE = {{\n    ("a", "b"): {{\n{cells}    }},\n}}\n')
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


def test_extract_preset_cell_lines_multiple_tuple_key_entries(tmp_path: Path) -> None:
    # Two tuple-key entries: the inner `},` closes must NOT prematurely end the
    # top-level span; all cells from both entries are collected.
    src = (
        "PRESET_TABLE = {\n"
        '    ("a", "b"): {\n'
        '        "k": 1,  # (a)\n'
        "    },\n"
        '    ("c", "d"): {\n'
        '        "m": 2,  # (a)\n'
        "    },\n"
        "}\n"
    )
    f = tmp_path / "aug_presets.py"
    f.write_text(src)
    texts = [c.text for c in extract_preset_cell_lines(f)]
    assert any('"k": 1' in t for t in texts)
    assert any('"m": 2' in t for t in texts)
    assert not any('("a", "b")' in t for t in texts)  # tuple-key lines are not cells
