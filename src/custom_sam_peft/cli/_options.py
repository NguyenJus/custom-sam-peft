"""Single source of truth for the shared CLI flag vocabulary.

Exposes Annotated[T, typer.Option(...)] aliases, the Progress/Split enums, the
merge_cli_overrides conflict-checking helper, and the shared discover_config
tree-walk. Every command imports its shared parameters from here so the surface
cannot drift the way `predict` once drifted from `train`.

Spec: docs/superpowers/specs/2026-06-02-cli-flag-audit-design.md §4.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer


class Progress(StrEnum):
    """Progress display mode (--progress). Values match the legacy bare-str flag."""

    auto = "auto"
    on = "on"
    off = "off"
    plain = "plain"


class Split(StrEnum):
    """Dataset split for `eval --split`. Only val/test are supported by eval.runner."""

    val = "val"
    test = "test"


def discover_config(checkpoint: Path) -> Path:
    """Walk up from *checkpoint* to the nearest sibling/ancestor config.yaml.

    Verbatim lift of the former export_cmd._discover_config tree-walk. Issue #249
    will later upgrade this single helper for self-describing checkpoints.
    """
    current = checkpoint.resolve()
    for parent in (current, *current.parents):
        candidate = parent / "config.yaml"
        if candidate.is_file():
            return candidate
    raise typer.BadParameter(
        f"could not auto-discover config.yaml above {checkpoint}; pass --config",
        param_hint="--config",
    )


def merge_cli_overrides(
    explicit_overrides: list[str],
    *,
    name: str | None,
    output_dir: Path | None,
) -> list[str]:
    """Append synthesized run.name / run.output_dir overrides for convenience flags.

    The merged list is fed unchanged into load_config(config, overrides=...) ->
    apply_overrides. Error-on-conflict: if a convenience flag and an explicit
    --override target the same dotted key, raise typer.BadParameter rather than
    silently choosing a precedence.
    """
    explicit_keys = {ov.partition("=")[0] for ov in explicit_overrides if "=" in ov}
    merged = list(explicit_overrides)
    if name is not None:
        if "run.name" in explicit_keys:
            raise typer.BadParameter(
                "conflict: --name and --override run.name= both set run.name; pass only one",
                param_hint="--name",
            )
        merged.append(f"run.name={name}")
    if output_dir is not None:
        if "run.output_dir" in explicit_keys:
            raise typer.BadParameter(
                "conflict: --output-dir and --override run.output_dir= both set "
                "run.output_dir; pass only one",
                param_hint="--output-dir",
            )
        merged.append(f"run.output_dir={output_dir}")
    return merged


VerboseOpt = Annotated[bool, typer.Option("-v", "--verbose", help="Enable DEBUG logging.")]
OverrideOpt = Annotated[
    list[str],
    typer.Option("--override", help="Override config keys: dotted.key=value."),
]
ProgressOpt = Annotated[
    Progress,
    typer.Option("--progress", help="Progress display mode: auto|on|off|plain.", metavar="MODE"),
]
DryRunOpt = Annotated[
    bool,
    typer.Option("--dry-run", help="Preview resolved inputs/config; do not run."),
]
NameOpt = Annotated[
    str | None,
    typer.Option("--name", help="Convenience for run.name (synthesizes an --override)."),
]
OutputDirOpt = Annotated[
    Path | None,
    typer.Option(
        "--output-dir",
        help="Convenience for run.output_dir, a run directory (synthesizes an --override).",
    ),
]
ConfigOpt = Annotated[
    Path | None,
    typer.Option("--config", help="Path to config YAML."),
]
ConfigArg = Annotated[
    Path | None,
    typer.Argument(help="Path to config YAML (the launch input)."),
]
