"""`custom-sam-peft init` — write a starter config from a packaged template.

After writing the config, optionally download SAM 3.1 weights via the
``--download-weights`` flag, or prompt interactively when stdout is a TTY.
"""

from __future__ import annotations

import string
import sys
from importlib.resources import files
from pathlib import Path
from typing import get_args

import typer
from rich import print as rprint

from custom_sam_peft.config.loader import load_config
from custom_sam_peft.config.schema import Intensity, Preset
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
    preset: str = typer.Option(
        "natural",
        "--preset",
        case_sensitive=False,
        help=(
            "Augmentation domain preset. One of: natural, medical, satellite, "
            "microscopy, none, custom."
        ),
    ),
    intensity: str = typer.Option(
        "medium",
        "--intensity",
        case_sensitive=False,
        help="Augmentation intensity tier. One of: safe, medium, aggressive.",
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
    preset_lc = preset.lower()
    intensity_lc = intensity.lower()
    valid_presets = set(get_args(Preset))
    valid_intensities = set(get_args(Intensity))
    if preset_lc not in valid_presets:
        raise typer.BadParameter(
            f"unknown preset '{preset}'. Available: {sorted(valid_presets)}",
            param_hint="--preset",
        )
    if intensity_lc not in valid_intensities:
        raise typer.BadParameter(
            f"unknown intensity '{intensity}'. Available: {sorted(valid_intensities)}",
            param_hint="--intensity",
        )
    if output.exists() and not force:
        raise typer.BadParameter(
            f"refusing to overwrite existing {output}; pass --force",
            param_hint="--output",
        )

    if preset_lc == "custom":
        overrides_block = (
            "overrides: {}  # fill in knobs: hflip, vflip, rotate90, "
            "rotate_arbitrary, color_jitter, stain_jitter, blur, gauss_noise"
        )
    else:
        overrides_block = (
            "# Override individual knobs here; unset keys inherit from (preset, intensity).\n"
            "    # overrides:\n"
            "    #   hflip: false\n"
            "    #   color_jitter: 0.15"
        )

    raw = (files("custom_sam_peft.cli.templates") / TEMPLATES[template]).read_text()
    body = string.Template(raw).substitute(
        preset=preset_lc,
        intensity=intensity_lc,
        overrides_block=overrides_block,
    )
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
