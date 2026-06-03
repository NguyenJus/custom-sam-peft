"""`custom-sam-peft profile` — run a representative profiled eval and dump a bucket breakdown.

For train and predict profiling, run those commands normally with CSP_PROFILE=1:

    CSP_PROFILE=1 csp train --config <cfg>
    CSP_PROFILE=1 csp predict --config <cfg> --input <dir>

Each command self-dumps ``profile_snapshot.json`` to its output_dir when the env
var is set, because ``profiling.bucket()`` / ``profiling.note()`` / ``profiling.incr()``
are permanent, always-compiled-in, and strictly no-op when disabled.

This command enables profiling programmatically (no env var needed) so a single
invocation drives one representative eval and prints the bucket breakdown.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, cast

import typer

from custom_sam_peft import profiling
from custom_sam_peft.cli._logging import configure_logging
from custom_sam_peft.cli._options import (
    DryRunOpt,
    Split,
    VerboseOpt,
    discover_config,
)
from custom_sam_peft.config.loader import load_config
from custom_sam_peft.eval.runner import run_eval

# Ordered bucket display — mirrors the spike runner's display order, extended to
# the full dot-namespaced set so any eval bucket appears in the table.
_EVAL_BUCKET_ORDER = [
    "eval.dataset_load",
    "eval.forward",
    "eval.mask_upsample",
    "eval.transfer_binarize",
    "eval.rle_encode",
    "eval.coco_aggregate",
    "eval.total",
]


def profile(
    config: Path | None = typer.Option(None, "--config", help="Path to config YAML."),
    checkpoint: Path | None = typer.Option(
        None,
        "--checkpoint",
        help="Adapter checkpoint to evaluate.  Omit for zero-shot baseline.",
    ),
    split: Split = typer.Option(Split.val, "--split", help="Dataset split: val | test."),
    output: Path | None = typer.Option(
        None,
        "--output",
        help="Output dir for profile_snapshot.json.  Defaults to cfg.run.output_dir.",
    ),
    dry_run: DryRunOpt = False,
    verbose: VerboseOpt = False,
) -> None:
    """Run a representative profiled eval and dump a JSON bucket breakdown.

    Enables profiling in-process (no CSP_PROFILE env var needed).  Writes
    ``profile_snapshot.json`` to the run output dir, then prints a bucket table
    sorted by seconds descending.

    For train/predict profiling run those commands with CSP_PROFILE=1 — the
    instrumentation is permanent and self-dumps profile_snapshot.json when set.
    """
    configure_logging(verbose)

    if config is None:
        if checkpoint is not None:
            config = discover_config(checkpoint)
        else:
            raise typer.BadParameter(
                "--config is required (or pass --checkpoint and config will be auto-discovered)",
                param_hint="--config",
            )

    cfg = load_config(config)
    split_lit = cast(Literal["val", "test"], split)
    out_dir = output or Path(cfg.run.output_dir)

    if dry_run:
        typer.echo(f"dry-run  config={config}  run.name={cfg.run.name}  output_dir={out_dir}")
        return

    # Enable and reset before the run so even a process that already ran something
    # starts clean.
    profiling.enable()
    profiling.reset()

    try:
        run_eval(cfg, checkpoint=checkpoint, split=split_lit)
    except ValueError as e:
        raise typer.BadParameter(str(e), param_hint="--checkpoint") from e

    buckets, meta = profiling.snapshot()

    # Durable dump FIRST — print() is block-buffered under redirection; a
    # killed/throttled process would otherwise lose the data (#250 gotcha).
    snap_path = profiling.dump(out_dir / "profile_snapshot.json")
    typer.echo(f"\nwrote {snap_path}", err=False)

    # Human-readable bucket table.  ``eval.total`` is a PARENT bucket wrapping
    # the child leaf spans (forward, rle_encode, dataset_load, ...), so it must
    # not be summed alongside them.  When present it IS the wall-time
    # denominator; otherwise fall back to the sum of the leaf buckets.  The gap
    # between eval.total and the summed leaves is the unbucketed residual
    # (per-chunk to_device, GT build).
    leaf_total = sum(s for b, s in buckets.items() if b != "eval.total")
    total = buckets.get("eval.total") or leaf_total or 1.0

    print(f"\n=== csp profile — {cfg.run.name} ===")  # noqa: T201
    print(f"metadata: {meta}")  # noqa: T201

    # Build display order: known buckets first (in canonical order), then any
    # extra buckets sorted alphabetically, then the total row.
    known = [b for b in _EVAL_BUCKET_ORDER if b in buckets]
    extra = sorted(b for b in buckets if b not in _EVAL_BUCKET_ORDER)
    order = known + extra

    print(f"\n{'bucket':<30}{'seconds':>12}{'% of total':>14}")  # noqa: T201
    print("-" * 56)  # noqa: T201
    for name in order:
        s = buckets.get(name, 0.0)
        print(f"{name:<30}{s:>12.4f}{100 * s / total:>13.1f}%")  # noqa: T201
    print("-" * 56)  # noqa: T201
    residual = total - leaf_total if "eval.total" in buckets else 0.0
    if residual > 0:
        pct = 100 * residual / total
        print(f"{'(residual = total - leaves)':<30}{residual:>12.4f}{pct:>13.1f}%")  # noqa: T201
    print(f"{'TOTAL(wall)':<30}{total:>12.4f}{100.0:>13.1f}%")  # noqa: T201
