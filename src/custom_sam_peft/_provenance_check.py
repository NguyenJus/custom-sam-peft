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
from typing import Any, Literal, TypedDict

import yaml

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
    """A parsed doc table row, relative to its section (amended #192).

    ``value`` is the raw **Value** cell text (used for the required-field
    exemption).  ``is_subscript_key`` flags a bare symbol that contains ``[``
    (a subscript/call-path key the AST surface cannot emit).
    ``is_in_function_literal`` flags a ``symbol (<note>)`` parenthetical with
    NO ``[`` before the paren — indicating a literal embedded inside a function
    body rather than a module- or class-level default.
    """

    symbol: str
    is_in_function_literal: bool
    value: str
    is_subscript_key: bool


def _cell_inner(cell: str) -> str:
    """Strip the backtick wrapper from a markdown cell, if present."""
    m = re.search(r"`([^`]+)`", cell)
    return m.group(1).strip() if m else cell.strip()


def parse_prose_rows(body: str, section_header: str) -> list[DocRow]:
    """Parse a prose section body into per-row DocRows (amended #192).

    The Location prefix mirrors ``section_header`` and is stripped to yield the
    bare symbol. Rows whose Location has a ``symbol (<note>)`` parenthetical (with
    NO ``[`` before the paren) are in-function literals. Rows whose bare symbol
    contains ``[`` or ``(`` are subscript/call-path keys. The **Value** cell
    (2nd column) is captured for the required-field exemption.
    """
    prefix = _unescape_md(section_header).strip()
    rows: list[DocRow] = []
    for line in body.splitlines():
        m = _ROW.match(line)
        if m is None:
            continue
        cells = m.group("cells").split("|")
        loc_match = _LOCATION_CELL.search(cells[0])
        if loc_match is None:
            continue
        loc = loc_match.group("loc").strip()
        if not loc.startswith(f"{prefix}:"):
            continue  # header/separator/non-location rows
        rest = loc[len(prefix) + 1 :]  # drop "prefix:"
        value = _cell_inner(cells[1]) if len(cells) > 1 else ""
        # subscript/call-path key: a bracket or a paren that is part of a path
        # (e.g. CHANNEL_SEMANTICS["rgb"].x). Detect a '[' anywhere first.
        if "[" in rest:
            rows.append(
                DocRow(
                    symbol=rest.strip(),
                    is_in_function_literal=False,
                    value=value,
                    is_subscript_key=True,
                )
            )
            continue
        lit = _LITERAL_SUFFIX.match(rest)
        if lit is not None:
            # ``symbol (<note>)`` with no bracket -> in-function literal.
            rows.append(
                DocRow(
                    symbol=lit.group("symbol").strip(),
                    is_in_function_literal=True,
                    value=value,
                    is_subscript_key=False,
                )
            )
        else:
            rows.append(
                DocRow(
                    symbol=rest.strip(),
                    is_in_function_literal=False,
                    value=value,
                    is_subscript_key=False,
                )
            )
    return rows


# Override-mirror config classes: all fields default to None ("inherit from the
# preset table"); their value provenance lives in the TABLE modules, not prose.
OVERRIDE_MIRROR_CLASSES: frozenset[str] = frozenset({"AugmentationOverrides", "LossOverrides"})
_ALIAS_BASES: frozenset[str] = frozenset({"Literal", "Union", "Optional"})
_LITERAL_NODES = (ast.Constant, ast.Tuple, ast.List, ast.Dict, ast.Set)


def _is_type_alias_rhs(value: ast.expr | None) -> bool:
    """RHS is ``Literal[...]`` / ``Union[...]`` / ``Optional[...]`` (a type alias)."""
    if not isinstance(value, ast.Subscript):
        return False
    base = value.value
    return (isinstance(base, ast.Name) and base.id in _ALIAS_BASES) or (
        isinstance(base, ast.Attribute) and base.attr in _ALIAS_BASES
    )


def _is_literal_value(value: ast.expr | None) -> bool:
    """RHS is a literal value node (Constant/Tuple/List/Dict/Set or -Constant)."""
    if isinstance(value, _LITERAL_NODES):
        return True
    return isinstance(value, ast.UnaryOp) and isinstance(value.operand, ast.Constant)


def _is_field_call(value: ast.expr | None) -> bool:
    """True if value is a ``Field(...)`` / ``...Field(...)`` call."""
    return isinstance(value, ast.Call) and (
        (isinstance(value.func, ast.Name) and value.func.id == "Field")
        or (isinstance(value.func, ast.Attribute) and value.func.attr == "Field")
    )


def _nested_container_name(value: ast.expr | None) -> str | None:
    """Class name if the default/default_factory references a config class, else None.

    Handles ``Field(default_factory=X)``, ``Field(default=X())``, and bare ``= X()``.

    Deliberate limitations (no real schema field uses these forms today):
    - ``= SomeConfig`` (bare class reference without a call) is NOT detected.
    - ``Field(default_factory=lambda: X())`` is NOT detected (lambda body not walked).
    Future container fields using those forms would be silently treated as plain leaves.
    """
    if value is None:
        return None
    if isinstance(value, ast.Call) and _is_field_call(value):
        for kw in value.keywords:
            if kw.arg == "default_factory" and isinstance(kw.value, ast.Name):
                return kw.value.id
            if (
                kw.arg == "default"
                and isinstance(kw.value, ast.Call)
                and isinstance(kw.value.func, ast.Name)
            ):
                return kw.value.func.id
        return None
    if isinstance(value, ast.Call) and isinstance(value.func, ast.Name):
        return value.func.id
    return None


def _looks_like_container(name: str | None) -> bool:
    """Name-shape heuristic for an IMPORTED config container (cannot recurse)."""
    return name is not None and name.endswith(("Config", "Overrides", "Weights"))


def _is_required_field(value: ast.expr | None) -> bool:
    """A no-default field: AnnAssign w/o value, or Field(...) w/o default*/factory."""
    if value is None:
        return True
    if isinstance(value, ast.Call) and _is_field_call(value):
        return not any(kw.arg in ("default", "default_factory") for kw in value.keywords)
    return False


def _field_target_name(stmt: ast.stmt) -> str | None:
    if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
        return stmt.target.id
    if isinstance(stmt, ast.Assign):
        names = [t.id for t in stmt.targets if isinstance(t, ast.Name)]
        return names[0] if names else None
    return None


def _field_value(stmt: ast.stmt) -> ast.expr | None:
    if isinstance(stmt, (ast.AnnAssign, ast.Assign)):
        return stmt.value
    return None


def extract_default_surface(file_path: Path) -> set[str]:
    """Return the enforced default-surface symbol keys (amended #192).

    See the design spec's "default surface" definition (authoritative).
    """
    tree = ast.parse(file_path.read_text(encoding="utf-8"))
    classes: dict[str, ast.ClassDef] = {n.name: n for n in tree.body if isinstance(n, ast.ClassDef)}
    surface: set[str] = set()

    # (1) Module-level constants — literal RHS only, with exclusions.
    for node in tree.body:
        targets_values: list[tuple[str, ast.expr | None]] = []
        if isinstance(node, ast.Assign):
            targets_values = [(t.id, node.value) for t in node.targets if isinstance(t, ast.Name)]
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            targets_values = [(node.target.id, node.value)]
        for name, value in targets_values:
            if name.startswith("__") and name.endswith("__"):
                continue
            if name in ("pytestmark", "model_config"):
                continue
            if _is_type_alias_rhs(value):
                continue
            if not _is_literal_value(value):
                continue  # excludes Call/BinOp/Name RHS (loggers, computed, derived)
            surface.add(name)

    # (2) Class-body field defaults — suppress nested containers + recurse;
    #     defining-class keying; required + override-mirror handling.
    def recurse(class_node: ast.ClassDef, path: frozenset[str]) -> None:
        for stmt in class_node.body:
            target = _field_target_name(stmt)
            if target is None or target == "model_config":
                continue
            value = _field_value(stmt)
            if _is_required_field(value):
                continue
            nested = _nested_container_name(value)
            if nested in classes:
                if nested not in path:  # cycle guard: skip if already on descent path
                    recurse(classes[nested], path | {nested})
                continue  # suppress leaf regardless (container, cyclic or not)
            if _looks_like_container(nested):
                continue  # imported container -> suppress (leaves in its own section)
            if class_node.name in OVERRIDE_MIRROR_CLASSES:
                continue
            surface.add(f"{class_node.name}.{target}")

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            recurse(node, frozenset({node.name}))
    return surface


def module_assign_names(file_path: Path) -> set[str]:
    """Every module-level assigned name (any RHS) — for the module-constant exemption."""
    tree = ast.parse(file_path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


# Outer-rooted doc rows re-keyed to defining-class form (single keying rule).
# Phase-1→Phase-4 bridge: this map exists only because the current doc rows use
# 3-level outer-path keying (e.g. ``TrainHyperparams.early_stop.enabled``).
# Once Phase-4 Task 4.6 re-keys those rows to defining-class form
# (``EarlyStopConfig.enabled``), delete this map — or replace it with an
# assertion that fails if any surviving doc row still contains a 3-level path.
_REKEY_PREFIXES: dict[str, str] = {
    "TrainHyperparams.early_stop.": "EarlyStopConfig.",
}


def _rekey_to_defining_class(symbol: str) -> str:
    for prefix, repl in _REKEY_PREFIXES.items():
        if symbol.startswith(prefix):
            return repl + symbol[len(prefix) :]
    return symbol


def check_prose_section(section: Section, file_path: Path) -> list[ProvenanceViolation]:
    """Assertion 1: symbol<->row bijection over the file's default surface (amended #192).

    code->doc: every surface symbol must have a doc row (keyed by defining class).
    doc->code: every doc row naming a *surface* symbol must still resolve, EXCEPT
        in-function-literal rows, required-field rows (Value cell says ``required``),
        subscript/call-path rows (symbol has ``[`` / ``(``), and module-constant rows
        (bare symbol is any module-level assigned name still present, any RHS).
    """
    file_disp = _unescape_md(section.header).strip()
    surface = extract_default_surface(file_path)
    module_names = module_assign_names(file_path)
    rows = parse_prose_rows(section.body, section.header)

    documented: set[str] = set()
    for row in rows:
        if row.is_in_function_literal or row.is_subscript_key:
            continue
        documented.add(_rekey_to_defining_class(row.symbol))

    violations: list[ProvenanceViolation] = []

    # code->doc
    for symbol in sorted(surface - documented):
        violations.append(
            ProvenanceViolation(
                location=f"{file_disp}:{symbol}",
                problem="new undocumented default",
                remediation=(
                    f"add a row to the `## {file_disp}` section of docs/defaults-provenance.md"
                ),
            )
        )

    # doc->code (apply all four exemptions)
    for row in rows:
        if row.is_in_function_literal or row.is_subscript_key:
            continue
        if row.value.lower().startswith("required"):
            continue  # required-field exemption
        symbol = _rekey_to_defining_class(row.symbol)
        if symbol in surface:
            continue
        if "." not in symbol and symbol in module_names:
            continue  # module-constant exemption (documented Call/BinOp constants)
        violations.append(
            ProvenanceViolation(
                location=f"{file_disp}:{row.symbol}",
                problem="stale/orphaned provenance row",
                remediation=("remove or update the row in docs/defaults-provenance.md"),
            )
        )
    return violations


@dataclass(frozen=True)
class CellLine:
    """A preset-table value line: source text + 1-based line number."""

    text: str
    lineno: int


# A value line inside a dict literal: ``"key": <value>,`` (optionally tagged).
# Assumes PRESET_TABLE/_LEGACY_DEFAULTS keys at nesting level 1 are tuples (not
# strings), so only true value cells match _DICT_VALUE_LINE — not intermediate
# string-keyed sub-dict openers.
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
        if any(re.match(rf"^{re.escape(name)}\b.*=\s*\{{\s*$", line) for name in dict_names):
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
        if (idx in in_span and _DICT_VALUE_LINE.match(line)) or _ALIAS_LINE.match(line):
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
        if m is not None:
            letters.add(m.group(1))
    return letters


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


_TEMPLATE_BARE_LINE = re.compile(r"^\s*\$[A-Za-z_][A-Za-z0-9_]*\s*$")
_TEMPLATE_INLINE_TOKEN = re.compile(r"\$[A-Za-z_][A-Za-z0-9_]*")


def _neutralize_template_tokens(text: str) -> str:
    """Strip bare block-expansion lines and replace inline $tokens with a placeholder.

    ``string.Template`` files contain two token shapes:
    - Bare block-expansion lines (stripped == ``$identifier``): these expand to
      whole sub-blocks of user content and are NOT scalar leaves — drop the line
      entirely so ``yaml.safe_load`` does not see a bare scalar where a key is
      expected.
    - Inline value placeholders (``key: $name``): only the path matters, not the
      value — replace the token with the literal string ``__TEMPLATE__`` so the
      line parses and its dotted path is still emitted.
    """
    out: list[str] = []
    for line in text.splitlines(keepends=True):
        if _TEMPLATE_BARE_LINE.match(line):
            continue  # drop: bare block-expansion slot, emits no path
        out.append(_TEMPLATE_INLINE_TOKEN.sub("__TEMPLATE__", line))
    return "".join(out)


def yaml_scalar_dotted_paths(yaml_path: Path) -> set[str]:
    """Dotted paths of every scalar leaf in a YAML file (lists treated as leaves)."""
    raw = yaml_path.read_text(encoding="utf-8")
    data = yaml.safe_load(_neutralize_template_tokens(raw)) or {}
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
                    "add a `cross-link` row to the `## cli/templates/config_full.yaml` section"
                ),
            )
        )
    return violations


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


DOC_REL_PATH = "docs/defaults-provenance.md"


def schema_default_dotted_paths(repo_root: Path) -> set[str]:
    """Dotted YAML paths for every pydantic field that has a default, from the schema.

    Walks ``custom_sam_peft.config.schema.TrainConfig`` (the root config model) and
    yields the dotted path of each field whose model field has a default. Used
    only for Assertion 3; imported lazily so the unit tests that pass explicit
    schema-default sets do not need the real schema importable.

    Recursion into nested BaseModel fields is unconditional, so a nested
    sub-field's default requires a cross-link even when its parent field is
    itself required (has no default).
    """
    from pydantic import BaseModel

    from custom_sam_peft.config.schema import TrainConfig

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
                # A pydantic field "has a default" iff it is not required. Do not
                # test ``field.default is not None``: a required field's default is
                # the ``PydanticUndefined`` sentinel (not ``None``), so that test
                # would wrongly count required fields (e.g. ``run.name``) as
                # defaulted and demand spurious cross-links for their template slots.
                if not field.is_required():
                    paths.add(dotted)

    walk(TrainConfig, "")
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
        if path is not None:
            if kind == "prose":
                violations.extend(check_prose_section(section, path))
            elif kind == "table":
                violations.extend(check_table_section(section, path))
            elif kind == "yaml":
                if schema_paths is None:
                    schema_paths = schema_default_dotted_paths(repo_root)
                violations.extend(check_yaml_section(section, path, schema_paths))
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
