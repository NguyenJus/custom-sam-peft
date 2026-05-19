"""`esam3 init` — write a starter config from a packaged template."""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path

import typer
from rich import print as rprint

TEMPLATES: dict[str, str] = {
    "coco-text-lora": "coco_text_lora.yaml",
    "coco-text-qlora": "coco_text_qlora.yaml",
}


def init(
    template: str = typer.Option(
        "coco-text-lora",
        "--template",
        help=f"Starter config template. One of: {', '.join(TEMPLATES)}.",
    ),
    output: Path = typer.Option(Path("config.yaml"), "--output", help="Destination path."),
    force: bool = typer.Option(False, "--force", help="Overwrite if output exists."),
) -> None:
    """Write a starter config."""
    if template not in TEMPLATES:
        raise typer.BadParameter(
            f"unknown template '{template}'. Available: {', '.join(TEMPLATES)}",
            param_hint="--template",
        )
    if output.exists() and not force:
        raise typer.BadParameter(
            f"refusing to overwrite existing {output}; pass --force",
            param_hint="--output",
        )
    body = (files("esam3.cli.templates") / TEMPLATES[template]).read_text()
    output.write_text(body)
    rprint(f"[green]wrote[/green] {output}")
