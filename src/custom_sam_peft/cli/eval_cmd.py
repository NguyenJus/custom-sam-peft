"""`custom-sam-peft eval` — thin CLI shell over custom_sam_peft.eval.runner.run_eval."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Literal, cast

import typer
from rich import print as rprint
from rich.console import Console

from custom_sam_peft.cli._logging import configure_logging
from custom_sam_peft.cli._progress import ProgressKind, progress_session, resolve_mode
from custom_sam_peft.config.loader import load_config
from custom_sam_peft.eval.runner import run_eval
from custom_sam_peft.runs.bundle import run_export


def evaluate(
    config: Path | None = typer.Option(None, "--config", help="Path to config YAML."),
    checkpoint: Path | None = typer.Option(
        None,
        "--checkpoint",
        help="Path to adapter checkpoint. Omit to evaluate baseline (zero-shot) SAM.",
    ),
    split: str = typer.Option("val", "--split", help="Dataset split: val | test."),
    output: Path | None = typer.Option(
        None,
        "--output",
        help="Output dir; defaults to checkpoint.parent, else cfg.run.output_dir.",
    ),
    save_predictions: bool | None = typer.Option(
        None,
        "--save-predictions/--no-save-predictions",
        help="Override cfg.eval.save_predictions.",
    ),
    visualize: bool | None = typer.Option(
        None,
        "--visualize/--no-visualize",
        help="Override cfg.eval.visualize (write GT-vs-Pred composite panels).",
    ),
    do_export: bool = typer.Option(
        False,
        "--export",
        help="After evaluation, export a run bundle.",
    ),
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Enable DEBUG logging."),
    progress_flag: str = typer.Option(
        "auto",
        "--progress",
        help="Progress display mode: auto|on|off|plain.",
        metavar="MODE",
    ),
    interactive: bool = typer.Option(
        False,
        "--interactive",
        "-i",
        help=("Build an eval command (reuse a trained adapter) or a baseline eval config."),
    ),
) -> None:
    """Evaluate a checkpoint on the val or test split."""
    configure_logging(verbose)
    if interactive:
        from custom_sam_peft.cli import _interactive

        _interactive.require_tty()
        _interactive.run_eval_interactive(output=output, force=False)
        return
    if config is None:
        raise typer.BadParameter("--config is required", param_hint="--config")
    if split not in ("val", "test"):
        raise typer.BadParameter(f"--split must be val|test; got {split!r}", param_hint="--split")
    cfg = load_config(config)
    split_lit = cast(Literal["val", "test"], split)

    mode = resolve_mode(
        progress_flag if progress_flag != "auto" else None,
        os.environ,
        sys.stdout.isatty(),
        Console().is_jupyter,
    )

    try:
        with progress_session(
            kind=ProgressKind.EVAL,
            total_batches_per_epoch=0,  # Evaluator owns its progress via push_subtask
            mode=mode,
            # total_epochs intentionally omitted — no outer epoch bar for eval (planner decision)
        ):
            report = run_eval(
                cfg,
                checkpoint=checkpoint,
                split=split_lit,
                output_dir=output,
                save_predictions=save_predictions,
                visualize=visualize,
            )
    except ValueError as e:
        raise typer.BadParameter(str(e), param_hint="--checkpoint") from e

    rprint(f"[green]eval complete[/green] — {report.overall}")
    if do_export:
        if checkpoint is None:
            raise typer.BadParameter(
                "--export requires a checkpoint; omit --export for baseline eval",
                param_hint="--checkpoint",
            )
        try:
            run_export(cfg, checkpoint)
        except Exception as e:
            rprint(f"[red]export error[/red] {e}")
            raise typer.Exit(code=1) from e
