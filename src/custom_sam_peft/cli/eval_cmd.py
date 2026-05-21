"""`custom-sam-peft eval` — thin CLI shell over custom_sam_peft.eval.runner.run_eval."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, cast

import typer
from rich import print as rprint

from custom_sam_peft.cli._logging import configure_logging
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
) -> None:
    """Evaluate a checkpoint on the val or test split."""
    configure_logging(verbose)
    if split not in ("val", "test"):
        raise typer.BadParameter(f"--split must be val|test; got {split!r}", param_hint="--split")
    cfg = load_config(config)
    split_lit = cast(Literal["val", "test"], split)
    try:
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
