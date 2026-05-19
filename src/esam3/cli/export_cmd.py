"""`esam3 export` — export adapter or merged model."""

from __future__ import annotations

from pathlib import Path

import typer
from rich import print as rprint

from esam3.cli._logging import configure_logging
from esam3.config.loader import load_config
from esam3.models.sam3 import load_sam31
from esam3.train.checkpoint import load_adapter, save_adapter, save_merged


def _discover_config(checkpoint: Path) -> Path:
    """Walk up from checkpoint looking for a sibling config.yaml."""
    current = checkpoint.resolve()
    for parent in (current, *current.parents):
        candidate = parent / "config.yaml"
        if candidate.is_file():
            return candidate
    raise typer.BadParameter(
        f"could not auto-discover config.yaml above {checkpoint}; pass --config",
        param_hint="--config",
    )


def export(
    checkpoint: Path = typer.Option(..., "--checkpoint", help="Path to adapter checkpoint."),
    merge: bool = typer.Option(False, "--merge", help="Also export merged full-model weights."),
    output: Path | None = typer.Option(None, "--output", help="Output directory."),
    config: Path | None = typer.Option(None, "--config", help="Explicit config path."),
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Enable DEBUG logging."),
) -> None:
    """Export adapter or merged model."""
    configure_logging(verbose)

    config_path = config if config is not None else _discover_config(checkpoint)
    cfg = load_config(config_path)
    run_dir = config_path.parent

    if merge:
        out = output if output is not None else (run_dir / "merged")
    else:
        if output is None:
            raise typer.BadParameter(
                "--output is required when not using --merge (refusing to overwrite source)",
                param_hint="--output",
            )
        out = output

    wrapper = load_sam31(cfg.model)
    load_adapter(wrapper, checkpoint)
    if merge:
        save_merged(wrapper, out)
        rprint(f"[green]merged[/green] {out}")
    else:
        save_adapter(wrapper, out)
        rprint(f"[green]adapter[/green] {out}")
