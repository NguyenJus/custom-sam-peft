"""`esam3 eval` — Body deferred to spec/cli."""

from __future__ import annotations

from pathlib import Path

import typer
from rich import print as rprint


def evaluate(
    config: Path = typer.Option(..., "--config", help="Path to config YAML."),
    checkpoint: Path = typer.Option(..., "--checkpoint", help="Path to adapter checkpoint."),
    split: str = typer.Option("val", "--split", help="Dataset split: val | test."),
) -> None:
    """Evaluate a checkpoint."""
    rprint(
        f"[yellow]not yet implemented[/yellow] — would eval {checkpoint} "
        f"on {split} split of {config}"
    )
