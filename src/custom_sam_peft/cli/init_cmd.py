"""`custom-sam-peft init` — write a starter config from a packaged template.

After writing the config, optionally download SAM 3.1 weights via the
``--download-weights`` flag, or prompt interactively when stdout is a TTY.
"""

from __future__ import annotations

import sys
from importlib.resources import files
from pathlib import Path

import typer
from rich import print as rprint

from custom_sam_peft.config.loader import load_config
from custom_sam_peft.utils.huggingface import download_model

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
    download_weights: bool | None = typer.Option(
        None,
        "--download-weights/--no-download-weights",
        help=(
            "Download SAM 3.1 weights after writing the config. "
            "Default: prompt interactively when stdin is a TTY."
        ),
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        help=(
            "Skip the interactive prompt; assume yes. Implies "
            "--download-weights when --no-download-weights is not passed."
        ),
    ),
) -> None:
    """Write a starter config, then optionally download weights."""
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
    body = (files("custom_sam_peft.cli.templates") / TEMPLATES[template]).read_text()
    output.write_text(body)
    rprint(f"[green]wrote[/green] {output}")

    _maybe_download_weights(output, download_weights=download_weights, yes=yes)


def _maybe_download_weights(output: Path, *, download_weights: bool | None, yes: bool) -> None:
    """Drive the spec §4.1 decision matrix after the config has been written."""
    cfg = load_config(output)
    if cfg.model.local_dir is None:
        rprint(
            "[dim]model.local_dir is None in the rendered config; skipping weight download.[/dim]"
        )
        return

    local_dir = Path(cfg.model.local_dir)
    ckpt = local_dir / cfg.model.checkpoint_file
    if ckpt.exists():
        rprint(f"[dim]weights already present at {ckpt}; skipping download[/dim]")
        return

    if download_weights is False:
        rprint(
            f"[dim]skipping weights download; weights will be fetched on first "
            f"`custom-sam-peft train`. Re-run `custom-sam-peft init --download-weights` (or "
            f"`huggingface-cli download {cfg.model.name} --local-dir {local_dir}`) "
            f"to fetch them now.[/dim]"
        )
        return

    if download_weights is True or yes:
        proceed = True
    elif sys.stdin.isatty():
        proceed = typer.confirm(
            f"Download {cfg.model.name} weights into {local_dir}? (this can be several GB)",
            default=True,
        )
    else:
        rprint(
            "[dim]non-interactive shell and no --download-weights flag; "
            "skipping. Weights will be fetched on first `custom-sam-peft train`; re-run "
            "with `--download-weights` to fetch them now.[/dim]"
        )
        return

    if not proceed:
        return

    try:
        download_model(cfg.model.name, local_dir, revision=cfg.model.revision)
    except RuntimeError as e:
        rprint(f"[red]error[/red] {e}")
        raise typer.Exit(code=1) from e
