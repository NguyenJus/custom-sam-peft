"""`custom-sam-peft export` — thin CLI shell over custom_sam_peft.runs.bundle.run_export."""

from __future__ import annotations

import logging
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

logger = logging.getLogger(__name__)


def export(
    checkpoint: Path = typer.Option(..., "--checkpoint", help="Path to adapter checkpoint."),
    merge: bool = typer.Option(False, "--merge", help="Also export merged full-model weights."),
    to: str = typer.Option("pytorch", "--to", help="Export format: pytorch (default) or onnx."),
    opset: int = typer.Option(17, "--opset", help="ONNX opset version (floor 17)."),
    fp16: bool = typer.Option(False, "--fp16", help="Export weights in fp16 (required for QLoRA)."),
    include: str = typer.Option("all", "--include", help="ONNX bundle parts: encoder|decoder|all."),
    dynamic_axes: bool = typer.Option(
        True,
        "--dynamic-axes/--no-dynamic-axes",
        help="Dynamic batch dim (spatial stays pinned to the model image size).",
    ),
    check: bool = typer.Option(
        False, "--check", help="Verify torch-vs-ORT parity after export; fail on drift."
    ),
    quantize: str = typer.Option(
        "none",
        "--quantize",
        help="Quantization: none|int8-dynamic (int8-dynamic is RESERVED, not implemented).",
    ),
    output: Path = typer.Option(..., "--output", help="Output directory (created if missing)."),
    config: Path | None = typer.Option(None, "--config", help="Explicit config path."),
    verbose: VerboseOpt = False,
    progress: ProgressOpt = Progress.auto,
) -> None:
    """Export adapter or merged model."""
    configure_logging(verbose)

    if to not in ("pytorch", "onnx"):
        raise typer.BadParameter("--to must be one of: pytorch, onnx.")
    if include not in ("encoder", "decoder", "all"):
        raise typer.BadParameter("--include must be one of: encoder, decoder, all.")
    if opset < 17:
        raise typer.BadParameter("--opset floor is 17.")
    if quantize != "none":
        raise typer.BadParameter("--quantize int8-dynamic is reserved and not yet implemented.")
    if to == "pytorch" and (opset != 17 or fp16 or include != "all" or not dynamic_axes or check):
        logger.info("ONNX-only flag(s) ignored for --to pytorch.")

    mode = resolve_mode(
        None if progress is Progress.auto else progress.value,
        os.environ,
        sys.stdout.isatty(),
        Console().is_jupyter,
    )

    config_path = config if config is not None else discover_config(checkpoint)
    cfg = load_config(config_path)

    with progress_session(kind=ProgressKind.EXPORT_MERGE, total_batches_per_epoch=0, mode=mode):
        if to == "pytorch":
            out = run_export(cfg, checkpoint, merge=merge, output=output)  # UNCHANGED path
        else:  # to == "onnx"
            from custom_sam_peft.export.onnx import run_export_onnx

            out = run_export_onnx(
                cfg,
                checkpoint,
                output=output,
                opset=opset,
                fp16=fp16,
                include=include,
                dynamic_axes=dynamic_axes,
                check=check,
            )

    if to == "pytorch":
        rprint(f"[green]{'merged' if merge else 'adapter'}[/green] {out}")
    else:
        rprint(f"[green]onnx bundle[/green] {out}")
