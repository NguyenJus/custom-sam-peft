"""`esam3 export` — Body deferred to spec/cli."""

from __future__ import annotations

from pathlib import Path

import typer
from rich import print as rprint


def export(
    checkpoint: Path = typer.Option(..., "--checkpoint", help="Path to adapter checkpoint."),
    merge: bool = typer.Option(False, "--merge", help="Also export merged full-model weights."),
    output: Path | None = typer.Option(None, "--output", help="Output directory."),
) -> None:
    """Export adapter or merged model."""
    rprint(
        f"[yellow]not yet implemented[/yellow] — would export {checkpoint} "
        f"(merge={merge}) to {output}"
    )
