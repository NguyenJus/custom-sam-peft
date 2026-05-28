"""`custom-sam-peft init` — thin CLI shell over run_init.

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
from custom_sam_peft.config.schema import ClassImbalance, Intensity, Preset
from custom_sam_peft.utils.huggingface import download_model

TEMPLATES: dict[str, str] = {
    "coco-text-lora": "lora",
    "coco-text-qlora": "qlora",
}

UNIFIED_TEMPLATE = "config_full.yaml"


def _build_loss_overrides_block(preset: str) -> str:
    """Spec §10.1.1 — render the loss-overrides scaffold under `train.loss:`."""
    if preset == "custom":
        return (
            "overrides: {}  # fill in knobs: mask_family, box_family, obj_family, "
            "presence_family, w_mask, w_box, w_obj, w_presence, "
            "focal_gamma, focal_alpha, tversky_alpha, tversky_gamma, "
            "boundary_weight, matcher_weights"
        )
    return (
        "# Override individual loss knobs here; unset keys inherit from "
        "(preset, class_imbalance).\n"
        "    # overrides:\n"
        "    #   mask_family: focal_dice\n"
        "    #   focal_gamma: 2.5\n"
        "    #   tversky_alpha: 0.7"
    )


def _flag_driven_blocks(peft_method: str) -> dict[str, str]:
    """The block-placeholder strings the flag-driven path substitutes.

    Wizard-only branches (dataset format, validation mode, model path) get
    their flag-driven defaults here: local-COCO dataset, explicit-val, default
    model dir.
    """
    model_block = (
        "  name: facebook/sam3.1\n"
        "  local_dir: models/sam3.1\n"
        "  checkpoint_file: sam3.1_multiplex.pt"
    )
    dataset_block = (
        "  format: coco\n"
        "  train:\n"
        "    annotations: data/train.json\n"
        "    images: data/train/\n"
        "  # HuggingFace alternative — set format: hf and uncomment:\n"
        "  # hf:\n"
        "  #   name: org/dataset\n"
        "  #   split_train: train\n"
        "  #   split_val: validation"
    )
    validation_block = (
        "  val:\n"
        "    annotations: data/val.json\n"
        "    images: data/val/\n"
        "  # Auto-split alternative (carve data.train into train+val):\n"
        "  # val_split:\n"
        "  #   fraction: 0.1\n"
        "  #   seed: null\n"
        "  # No-val alternative: omit both val: and val_split:."
    )
    qlora_block = (
        "  qlora:\n    quant_type: nf4\n    compute_dtype: bfloat16"
        if peft_method == "qlora"
        else ""
    )
    # No flag drives data.limit; emit the commented discoverable block (mirrors
    # setup_wizard._limit_block's unset branch) so the knob is visible to edit.
    limit_block = (
        "  # Limit dataset size for quick/smoke runs (int = count, float in (0,1] = fraction):\n"
        "  # limit:\n"
        "  #   train: 100\n"
        "  #   val: 50\n"
        "  #   strategy: random   # random | stratified | first_n\n"
        "  #   seed: 42"
    )
    return {
        "model_block": model_block,
        "dataset_block": dataset_block,
        "validation_block": validation_block,
        "qlora_block": qlora_block,
        "limit_block": limit_block,
    }


def run_init(
    template: str,
    output: Path,
    *,
    preset: str = "natural",
    intensity: str = "medium",
    class_imbalance: str = "balanced",
    force: bool = False,
) -> None:
    """Write a starter config (unified config_full.yaml) to *output*.

    Raises:
        ValueError: unknown template name, preset, intensity, or class_imbalance.
        FileExistsError: output exists and *force* is False.
    """
    if template not in TEMPLATES:
        raise ValueError(f"unknown template '{template}'. Available: {', '.join(TEMPLATES)}")
    valid_presets = set(get_args(Preset))
    valid_intensities = set(get_args(Intensity))
    _CLASS_IMBALANCES = get_args(ClassImbalance)
    if preset not in valid_presets:
        raise ValueError(f"unknown preset '{preset}'. Available: {sorted(valid_presets)}")
    if intensity not in valid_intensities:
        raise ValueError(f"unknown intensity '{intensity}'. Available: {sorted(valid_intensities)}")
    if class_imbalance not in _CLASS_IMBALANCES:
        raise typer.BadParameter(
            f"--class-imbalance must be one of {list(_CLASS_IMBALANCES)}; got {class_imbalance!r}",
            param_hint="--class-imbalance",
        )
    if output.exists() and not force:
        raise FileExistsError(f"refusing to overwrite existing {output}; pass force=True")

    if preset == "custom":
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

    loss_overrides_block = _build_loss_overrides_block(preset)

    peft_method = TEMPLATES[template]
    blocks = _flag_driven_blocks(peft_method)
    raw = (files("custom_sam_peft.cli.templates") / UNIFIED_TEMPLATE).read_text()
    body = string.Template(raw).substitute(
        run_name="my-run",
        peft_method=peft_method,
        epochs=10,
        aug_preset=preset,
        loss_preset=preset,
        aug_intensity=intensity,
        class_imbalance=class_imbalance,
        overrides_block=overrides_block,
        loss_overrides_block=loss_overrides_block,
        **blocks,
    )
    output.write_text(body)


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
    class_imbalance: str = typer.Option(
        "balanced",
        "--class-imbalance",
        case_sensitive=False,
        help="Loss-bundle class-imbalance tier. One of: balanced, moderate, severe.",
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
    interactive: bool = typer.Option(
        False,
        "--interactive",
        "-i",
        help=(
            "Run the interactive setup wizard. Ignores --template/--preset/"
            "--intensity/--class-imbalance (collected interactively)."
        ),
    ),
) -> None:
    """Write a starter config, then optionally download weights."""
    if interactive:
        import torch

        from custom_sam_peft.cli import setup_wizard

        if not sys.stdin.isatty():
            raise typer.BadParameter(
                "interactive setup needs a TTY; use the flag-driven "
                "`custom-sam-peft init …` instead"
            )
        if output.exists() and not force:
            raise typer.BadParameter(
                f"refusing to overwrite existing {output}; pass --force",
                param_hint="--output",
            )
        setup_wizard.generate_config(output, force=force, cuda_available=torch.cuda.is_available())
        _maybe_download_weights(output, download_weights=download_weights, yes=yes)
        return

    try:
        run_init(
            template,
            output,
            preset=preset.lower(),
            intensity=intensity.lower(),
            class_imbalance=class_imbalance.lower(),
            force=force,
        )
    except ValueError as e:
        raise typer.BadParameter(str(e), param_hint="--template") from e
    except FileExistsError as e:
        raise typer.BadParameter(str(e), param_hint="--output") from e
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
