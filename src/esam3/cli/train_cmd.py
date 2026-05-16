"""`esam3 train` — parses config, hands off to library. Body deferred to spec/cli."""

from __future__ import annotations

from pathlib import Path

import typer
from rich import print as rprint


def train(
    config: Path = typer.Option(..., "--config", help="Path to training config YAML."),
    override: list[str] = typer.Option(
        [], "--override", help="Override config keys: dotted.key=value."
    ),
    resume: Path | None = typer.Option(None, "--resume", help="Path to resume checkpoint."),
) -> None:
    """Run a finetune."""
    from esam3.config.loader import load_config

    cfg = load_config(config, overrides=override)
    rprint(f"[yellow]not yet implemented[/yellow] — would train run '{cfg.run.name}'")
    if resume is not None:
        rprint(f"  resume: {resume}")
