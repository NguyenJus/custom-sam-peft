"""`custom-sam-peft export` — thin CLI shell over custom_sam_peft.runs.bundle.run_export."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import typer
from rich import print as rprint
from rich.console import Console

from custom_sam_peft.cli._logging import configure_logging
from custom_sam_peft.cli._options import Progress, ProgressOpt, VerboseOpt, discover_config
from custom_sam_peft.cli._progress import ProgressKind, progress_session, resolve_mode
from custom_sam_peft.config.loader import load_config
from custom_sam_peft.runs.bundle import run_export


def export(
    checkpoint: Path = typer.Option(..., "--checkpoint", help="Path to adapter checkpoint."),
    merge: bool = typer.Option(False, "--merge", help="Also export merged full-model weights."),
    output: Path = typer.Option(..., "--output", help="Output directory (created if missing)."),
    config: Path | None = typer.Option(None, "--config", help="Explicit config path."),
    verbose: VerboseOpt = False,
    progress: ProgressOpt = Progress.auto,
) -> None:
    """Export adapter or merged model."""
    configure_logging(verbose)

    mode = resolve_mode(
        None if progress is Progress.auto else progress.value,
        os.environ,
        sys.stdout.isatty(),
        Console().is_jupyter,
    )

    config_path = config if config is not None else discover_config(checkpoint)
    cfg = load_config(config_path)

    with progress_session(kind=ProgressKind.EXPORT_MERGE, total_batches_per_epoch=0, mode=mode):
        out = run_export(cfg, checkpoint, merge=merge, output=output)

    if merge:
        rprint(f"[green]merged[/green] {out}")
    else:
        rprint(f"[green]adapter[/green] {out}")
