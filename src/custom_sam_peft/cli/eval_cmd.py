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


def evaluate(
    config: Path = typer.Option(..., "--config", help="Path to config YAML."),
    checkpoint: Path = typer.Option(..., "--checkpoint", help="Path to adapter checkpoint."),
    split: str = typer.Option("val", "--split", help="Dataset split: val | test."),
    output: Path | None = typer.Option(
        None, "--output", help="Output dir; defaults to checkpoint.parent."
    ),
    save_predictions: bool | None = typer.Option(
        None,
        "--save-predictions/--no-save-predictions",
        help="Override cfg.eval.save_predictions.",
    ),
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Enable DEBUG logging."),
    progress_flag: str = typer.Option(
        "auto",
        "--progress",
        help="Progress display mode: auto|on|off|plain.",
        metavar="MODE",
    ),
) -> None:
    """Evaluate a checkpoint on the val or test split."""
    configure_logging(verbose)
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
            total_batches_per_epoch=0,  # Evaluator updates via P.advance_inner
            mode=mode,
            # total_epochs intentionally omitted — no outer epoch bar for eval (planner decision)
        ):
            report = run_eval(
                cfg,
                checkpoint=checkpoint,
                split=split_lit,
                output_dir=output,
                save_predictions=save_predictions,
            )
    except ValueError as e:
        raise typer.BadParameter(str(e), param_hint="--checkpoint") from e

    rprint(f"[green]eval complete[/green] — {report.overall}")
