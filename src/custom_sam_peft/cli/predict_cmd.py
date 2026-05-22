"""`custom-sam-peft predict` — thin CLI shell over custom_sam_peft.predict.run_predict."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, cast

import click
import typer
from rich import print as rprint

from custom_sam_peft.cli._logging import configure_logging
from custom_sam_peft.predict.runner import PredictOptions, PredictReport, run_predict

# ---------------------------------------------------------------------------
# Validation callbacks (spec §8, §10)
# ---------------------------------------------------------------------------


def _validate_unit_interval(value: float) -> float:
    if not (0.0 <= value <= 1.0):
        raise typer.BadParameter(f"--score-threshold must be in [0.0, 1.0], got {value}")
    return value


def _validate_positive_int(value: int) -> int:
    if value < 1:
        raise typer.BadParameter(f"must be >= 1, got {value}")
    return value


def _validate_checkpoint(value: Path | None) -> Path | None:
    if value is None:
        return None
    if not value.exists() or not (value / "adapter_config.json").is_file():
        raise typer.BadParameter(
            f"--checkpoint must contain adapter_config.json; got {value}",
            param_hint="--checkpoint",
        )
    return value


# ---------------------------------------------------------------------------
# Command
# ---------------------------------------------------------------------------


def predict(
    images: Path = typer.Option(..., "--images", help="Dir / glob / manifest / single file."),
    prompts: str = typer.Option(
        ..., "--prompts", help="Comma-separated class names or path to one-per-line file."
    ),
    output: Path = typer.Option(..., "--output", help="Output directory (created if missing)."),
    checkpoint: Path | None = typer.Option(
        None, "--checkpoint", callback=_validate_checkpoint, help="Adapter checkpoint directory."
    ),
    merge_adapter: bool = typer.Option(
        True, "--merge-adapter/--no-merge-adapter", help="Merge adapter weights before inference."
    ),
    config: Path | None = typer.Option(None, "--config", help="Path to config YAML."),
    score_threshold: float = typer.Option(
        0.3,
        "--score-threshold",
        callback=_validate_unit_interval,
        help="Minimum score to keep a prediction [0.0, 1.0].",
    ),
    top_k: int = typer.Option(
        100,
        "--top-k",
        callback=_validate_positive_int,
        help="Max predictions per (image, class) pair.",
    ),
    save_masks: str = typer.Option(
        "rle",
        "--save-masks",
        click_type=click.Choice(["rle", "png", "none"]),
        help="Mask output format: rle | png | none.",
    ),
    visualize: bool = typer.Option(False, "--visualize", help="Write per-image overlay PNGs."),
    device: str = typer.Option(
        "auto",
        "--device",
        click_type=click.Choice(["auto", "cuda", "cpu"]),
        help="Compute device: auto | cuda | cpu.",
    ),
    dtype: str = typer.Option(
        "auto",
        "--dtype",
        click_type=click.Choice(["auto", "bfloat16", "float32"]),
        help="Model dtype: auto | bfloat16 | float32.",
    ),
    batch_size: int = typer.Option(
        1,
        "--batch-size",
        callback=_validate_positive_int,
        help="Images per forward pass.",
    ),
    seed: int = typer.Option(0, "--seed", help="Random seed for reproducibility."),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Preview resolved inputs; skip model load and inference."
    ),
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Enable DEBUG logging."),
) -> None:
    """Run inference on images with optional adapter."""
    configure_logging(verbose)
    opts = PredictOptions(
        images=images,
        prompts=prompts,
        output=output,
        checkpoint=checkpoint,
        merge_adapter=merge_adapter,
        config=config,
        score_threshold=score_threshold,
        top_k=top_k,
        save_masks=cast(Literal["rle", "png", "none"], save_masks),
        visualize=visualize,
        device=cast(Literal["auto", "cuda", "cpu"], device),
        dtype=cast(Literal["auto", "bfloat16", "float32"], dtype),
        batch_size=batch_size,
        seed=seed,
        dry_run=dry_run,
        verbose=verbose,
    )
    try:
        report: PredictReport = run_predict(opts)
    except RuntimeError as exc:
        if "all images failed" in str(exc).lower():
            rprint(f"[red]error[/red] {exc}")
            raise typer.Exit(code=1) from exc
        raise
    rprint(
        f"[green]predict complete[/green] — "
        f"images={report.n_images} predictions={report.n_predictions} "
        f"elapsed={report.elapsed_sec:.2f}s"
    )
