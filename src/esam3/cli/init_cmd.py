"""`esam3 init` — Body deferred to spec/cli."""

from __future__ import annotations

import typer
from rich import print as rprint

VALID_TEMPLATES = ("coco-text", "coco-bbox", "hf-text")


def init(
    template: str = typer.Option(
        "coco-bbox",
        "--template",
        help=f"Starter config template. One of: {', '.join(VALID_TEMPLATES)}.",
    ),
) -> None:
    """Write a starter config to ./config.yaml."""
    if template not in VALID_TEMPLATES:
        raise typer.BadParameter(f"unknown template '{template}'")
    rprint(f"[yellow]not yet implemented[/yellow] — would write {template} starter config")
