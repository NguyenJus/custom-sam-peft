"""`esam3 train` — thin CLI shell over esam3.train.runner.run_training."""

from __future__ import annotations

from pathlib import Path

import typer
from rich import print as rprint

from esam3.cli._logging import configure_logging
from esam3.config.loader import load_config
from esam3.train.runner import run_training


def train(
    config: Path = typer.Option(..., "--config", help="Path to training config YAML."),
    override: list[str] = typer.Option(
        [], "--override", help="Override config keys: dotted.key=value."
    ),
    resume: Path | None = typer.Option(None, "--resume", help="Path to resume checkpoint."),
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Enable DEBUG logging."),
) -> None:
    """Run a finetune."""
    configure_logging(verbose)
    cfg = load_config(config, overrides=override)
    if cfg.data.prompt_mode == "bbox":
        raise typer.BadParameter(
            "prompt_mode='bbox' is not supported for training in v0; "
            "see docs/superpowers/specs/2026-05-15-esam3-architecture-design.md §1.",
            param_hint="--config",
        )
    try:
        result = run_training(cfg, resume_from=resume)
    except (ValueError, NotImplementedError) as e:
        rprint(f"[red]error[/red] {e}")
        raise typer.Exit(code=1) from e
    rprint(f"[green]done[/green] run_dir={result.run_dir} adapter={result.adapter_path}")
    if result.final_metrics is not None:
        rprint(f"  mAP={result.final_metrics.overall.get('mAP', float('nan')):.4f}")
