"""`custom-sam-peft train` — thin CLI shell over custom_sam_peft.train.runner.run_train."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import typer
from rich import print as rprint
from rich.console import Console

from custom_sam_peft.cli._host_ram import format_host_ram_message
from custom_sam_peft.cli._logging import configure_logging
from custom_sam_peft.cli._options import (
    DryRunOpt,
    NameOpt,
    OutputDirOpt,
    OverrideOpt,
    Progress,
    ProgressOpt,
    VerboseOpt,
    merge_cli_overrides,
)
from custom_sam_peft.cli._progress import ProgressKind, progress_session, resolve_mode
from custom_sam_peft.cli._time_limit import format_time_limit_message
from custom_sam_peft.config._duration import parse_duration_to_seconds
from custom_sam_peft.config.loader import load_config
from custom_sam_peft.errors import CheckpointError
from custom_sam_peft.eval.runner import run_eval
from custom_sam_peft.runs.bundle import run_export
from custom_sam_peft.train.checkpoint import find_latest_checkpoint
from custom_sam_peft.train.runner import run_train

_LATEST_SENTINEL = "__latest__"


def train(
    config: Path = typer.Option(..., "--config", help="Path to training config YAML."),
    override: OverrideOpt = [],  # noqa: B006 — Typer creates a new list per invocation
    name: NameOpt = None,
    output_dir: OutputDirOpt = None,
    resume: str | None = typer.Option(
        None,
        "--resume",
        help=(
            "Resume checkpoint. Pass a path, or omit value for the latest "
            "checkpoint matching cfg.run.name."
        ),
    ),
    time_limit: str | None = typer.Option(
        None,
        "--time-limit",
        help=(
            'Wall-clock budget for this run (e.g. "2h30m", "90m", "3600s", or bare '
            "seconds). Overrides train.time_limit. The budget is per-run; --resume "
            "restarts the clock."
        ),
        metavar="DURATION",
    ),
    do_eval: bool = typer.Option(
        False,
        "--eval",
        help="After training, run evaluation against the same config's eval section.",
    ),
    do_export: bool = typer.Option(
        False,
        "--export",
        help="After training (and eval, if --eval), export a run bundle.",
    ),
    dry_run: DryRunOpt = False,
    verbose: VerboseOpt = False,
    progress: ProgressOpt = Progress.auto,
) -> None:
    """Run a finetune. The order is fixed: train → eval → export. Flags only toggle inclusion."""
    configure_logging(verbose)
    merged = merge_cli_overrides(override, name=name, output_dir=output_dir)
    cfg = load_config(config, overrides=merged)

    if time_limit is not None:
        try:
            parse_duration_to_seconds(time_limit)
        except ValueError as e:
            rprint(f"[red]error[/red] invalid --time-limit: {e}")
            raise typer.Exit(code=1) from e
        cfg = cfg.model_copy(
            update={"train": cfg.train.model_copy(update={"time_limit": time_limit})}
        )

    if dry_run:
        rprint(
            f"[cyan]dry-run[/cyan] config={config} run.name={cfg.run.name} "
            f"output_dir={cfg.run.output_dir}"
        )
        return

    mode = resolve_mode(
        None if progress is Progress.auto else progress.value,
        os.environ,
        sys.stdout.isatty(),
        Console().is_jupyter,
    )

    resume_path: Path | None
    if resume == _LATEST_SENTINEL:
        try:
            resume_path = find_latest_checkpoint(cfg)
        except CheckpointError as e:
            rprint(f"[red]error[/red] {e}")
            raise typer.Exit(code=1) from e
    elif resume is not None:
        resume_path = Path(resume)
    else:
        resume_path = None

    try:
        with progress_session(
            kind=ProgressKind.TRAIN,
            total_epochs=cfg.train.epochs,
            total_batches_per_epoch=0,  # Trainer updates dynamically via reset_inner
            mode=mode,
        ):
            result = run_train(cfg, resume_from=resume_path)
    except (ValueError, NotImplementedError) as e:
        rprint(f"[red]error[/red] {e}")
        raise typer.Exit(code=1) from e

    if result.time_limit_stop is not None:
        rprint(
            format_time_limit_message(
                result.time_limit_stop, subcommand="train", config_path=config
            )
        )
        return

    if result.host_ram_stop is not None:
        rprint(
            format_host_ram_message(result.host_ram_stop, subcommand="train", config_path=config)
        )
        return

    rprint(f"[green]done[/green] run_dir={result.run_dir} adapter={result.checkpoint_path}")
    if do_eval:
        try:
            run_eval(cfg, artifacts=result)
        except Exception as e:
            rprint(f"[red]eval error[/red] {e}")
            raise typer.Exit(code=1) from e
    if do_export:
        try:
            run_export(cfg, result.checkpoint_path)
        except Exception as e:
            rprint(f"[red]export error[/red] {e}")
            raise typer.Exit(code=1) from e
