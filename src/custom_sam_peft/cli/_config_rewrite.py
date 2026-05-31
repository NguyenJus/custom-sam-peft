"""In-place, comment-preserving config rewrite helpers.

Provides `_rewrite_sizing_block` — a line-surgery helper that updates a config's
sizing fields (peft.method, peft.r, train.batch_size, train.grad_accum_steps,
model.dtype) without stripping surrounding comments or altering unrelated lines.

Shared by `calibrate` (annotation: '# calibrated <date>') and `init` Task 9
(annotation: '# formula-derived'). pyyaml only — no ruamel.yaml.

Spec: docs/superpowers/specs/2026-05-28-vram-calibration-reassess-design.md §5.3.
"""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path

import yaml


def _rewrite_sizing_block(
    config_path: Path,
    *,
    method: str,
    r: int,
    batch_size: int,
    grad_accum_steps: int,
    classes_per_forward: int,
    dtype: str,
    annotation: str,
) -> None:
    """In-place, comment-preserving rewrite of a config's sizing fields.

    Locates peft.method/peft.r, train.batch_size/train.grad_accum_steps, model.dtype
    and substitutes their values by LINE SURGERY (not yaml.safe_dump, which would
    strip comments/formatting). Prepends `annotation` as a comment line above the
    first touched line. Rewrites DIRECT children of the target section (zero-indent
    section → first child-indent level); deeper-nested keys with the same name are
    left untouched — EXCEPT the nested train.multiplex.classes_per_forward target,
    which is rewritten by a dedicated pass. If any of the 5 direct (section, key)
    targets are absent, raises ValueError naming the missing keys; a missing
    train.multiplex.classes_per_forward likewise raises ValueError. Strips any
    immediately preceding annotation line (starting with '# calibrated' or
    '# formula-derived') before inserting the new one, so exactly one annotation
    remains across repeated runs.

    Note: inline comments on a rewritten line are preserved verbatim and may become
    stale relative to the new value.

    Validates the rewritten file still parses as valid YAML. pyyaml ONLY — do NOT
    add ruamel.yaml or any new dep.
    """
    original = config_path.read_text(encoding="utf-8")

    # Validate the original parses as a YAML mapping.
    parsed_before = yaml.safe_load(original)
    if not isinstance(parsed_before, dict):
        raise ValueError(f"config root is not a mapping: {config_path}")

    # Perform line-surgery: replace each targeted scalar value.
    # Targets: model.dtype, peft.method, peft.r, train.batch_size, train.grad_accum_steps.
    # Each line is matched as: <indent><key>: <value> [# optional comment]
    # We replace the value portion while preserving indentation and inline comments.
    lines = original.splitlines(keepends=True)

    # Track context: which top-level section we're in, and the child indent level
    # for that section (set on the first non-blank, non-comment child line).
    current_section: str | None = None
    child_indent: str | None = None  # exact indent string of direct children
    annotation_inserted = False

    # Pre-compile patterns.
    _top_section_pat = re.compile(r"^([a-zA-Z_][a-zA-Z0-9_]*):\s*$")
    _top_section_with_value_pat = re.compile(r"^([a-zA-Z_][a-zA-Z0-9_]*):\s+\S")
    # Nested target: train.multiplex.classes_per_forward lives one indent level
    # deeper than the direct children rewritten above.
    _cpf_pat = re.compile(r"^(\s+)classes_per_forward:\s+(\S+)(.*)$")

    # Map: (section, key) -> new_value (as YAML scalar string)
    replacements: dict[tuple[str, str], str] = {
        ("model", "dtype"): dtype,
        ("peft", "method"): method,
        ("peft", "r"): str(r),
        ("train", "batch_size"): str(batch_size),
        ("train", "grad_accum_steps"): str(grad_accum_steps),
    }
    done: set[tuple[str, str]] = set()

    # Pattern to match an indented key: value line
    _kv_pat = re.compile(r"^(\s+)(\S+?):\s+(\S+)(.*)")
    # Pattern to detect an annotation comment line (for idempotency stripping)
    _annotation_pat = re.compile(r"^(\s*)#\s*(calibrated|formula-derived)")

    touched_indices: list[int] = []
    staged: list[tuple[int, str]] = []  # (lineno, new_line)

    for i, line in enumerate(lines):
        stripped = line.rstrip("\n")

        # Detect top-level section changes (no leading whitespace, ends with ':')
        top_m = _top_section_pat.match(stripped)
        top_with_val_m = _top_section_with_value_pat.match(stripped) if not top_m else None
        if top_m or top_with_val_m:
            section_name = (top_m or top_with_val_m).group(1)  # type: ignore[union-attr]
            current_section = section_name
            child_indent = None  # reset child-indent for the new section
            continue

        # Try to match a key: value inside a target section
        if current_section in ("model", "peft", "train"):
            m = _kv_pat.match(stripped)
            if m:
                indent, key, _val, tail = m.groups()

                # Establish child indent on the first indented KV line in this section.
                if child_indent is None:
                    child_indent = indent

                # Only act on DIRECT children: skip lines at a deeper indent level.
                if indent != child_indent:
                    continue

                tgt = (current_section, key)
                if tgt in replacements and tgt not in done:
                    new_val = replacements[tgt]
                    # Reconstruct: preserve trailing comment/whitespace
                    new_line = f"{indent}{key}: {new_val}{tail}\n"
                    staged.append((i, new_line))
                    touched_indices.append(i)
                    done.add(tgt)

    # Nested target: train.multiplex.classes_per_forward (one extra indent level
    # under train). Match the first (and expected-unique — `classes_per_forward`
    # only appears under MultiplexConfig) indented `classes_per_forward:` line and
    # rewrite its value; if absent, raise so the caller knows the config predates
    # the multiplex block. Spec §3.
    cpf_done = False
    for i, line in enumerate(lines):
        m = _cpf_pat.match(line.rstrip("\n"))
        if m:
            indent, _old, tail = m.groups()
            staged.append((i, f"{indent}classes_per_forward: {classes_per_forward}{tail}\n"))
            touched_indices.append(i)
            cpf_done = True
            break
    if not cpf_done:
        raise ValueError(
            "_rewrite_sizing_block: config missing train.multiplex.classes_per_forward"
        )

    # Validate that all 5 expected targets were found.
    missing = set(replacements.keys()) - done
    if missing:
        missing_keys = ", ".join(f"{sec}.{key}" for sec, key in sorted(missing))
        raise ValueError(
            f"_rewrite_sizing_block: config is missing expected sizing keys: {missing_keys}"
        )

    # Apply annotation: insert it as a comment line before the first touched line.
    # Strip any immediately preceding existing annotation (idempotency).
    first_touch = min(touched_indices)
    staged_map = dict(staged)

    result_lines: list[str] = []

    for i, line in enumerate(lines):
        # When we reach the first touched line, insert the annotation before it,
        # but first strip any existing annotation on the immediately preceding line.
        if i == first_touch and not annotation_inserted:
            # Remove the last line of result_lines if it is an annotation comment.
            if result_lines and _annotation_pat.match(result_lines[-1].rstrip("\n")):
                result_lines.pop()
            # Determine indent of the first touched line to match annotation alignment
            indent_m = re.match(r"^(\s*)", lines[i])
            indent_str = indent_m.group(1) if indent_m else ""
            result_lines.append(f"{indent_str}{annotation}\n")
            annotation_inserted = True

        if i in staged_map:
            result_lines.append(staged_map[i])
        else:
            result_lines.append(line)

    new_text = "".join(result_lines)

    # Validate the rewritten text still parses as valid YAML.
    parsed_after = yaml.safe_load(new_text)
    if not isinstance(parsed_after, dict):
        raise ValueError(f"rewritten config is no longer a valid YAML mapping: {config_path}")

    # Atomic write: tmp + os.replace.
    fd, tmp = tempfile.mkstemp(prefix=".tmp_rewrite_", dir=str(config_path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(new_text)
        os.replace(tmp, config_path)
    except OSError:
        Path(tmp).unlink(missing_ok=True)
        raise
