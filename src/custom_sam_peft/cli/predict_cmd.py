"""`custom-sam-peft predict` — thin CLI shell over custom_sam_peft.predict.run_predict."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import click
import typer
from rich import print as rprint
from rich.console import Console

from custom_sam_peft.cli._logging import configure_logging
from custom_sam_peft.cli._options import Progress, ProgressOpt, VerboseOpt
from custom_sam_peft.cli._progress import ProgressKind, progress_session, resolve_mode
from custom_sam_peft.predict.adapter_load import detect_adapter_kind
from custom_sam_peft.predict.runner import PredictOptions, PredictReport, run_predict

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Option defaults — single source of truth for both typer.Option() and the
# instance-only-flags guard.  If a default changes, update here only.
# ---------------------------------------------------------------------------

_DEFAULT_SCORE_THRESHOLD: float = 0.3
_DEFAULT_TOP_K: int = 100
_DEFAULT_SAVE_MASKS: str = "rle"

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


def _validate_batch_size(value: str) -> int | str:
    if value == "auto":
        return "auto"
    try:
        n = int(value)
    except ValueError as exc:
        raise typer.BadParameter("--batch-size must be 'auto' or a positive int") from exc
    if n < 1:
        raise typer.BadParameter(f"must be >= 1, got {n}")
    return n


def _validate_onnx_bundle(bundle: Path) -> None:
    """Validate an ONNX bundle dir has the required sidecars + graphs (spec §4.3)."""
    import json

    if not bundle.is_dir():
        raise typer.BadParameter(
            f"--use-onnx must be an existing bundle directory; got {bundle}",
            param_hint="--use-onnx",
        )
    required = ["decoder.onnx", "preprocessor.json", "model_card.json", "prompts.txt"]
    # image_encoder.onnx is required unless the card declares include == "decoder".
    include = "all"
    card_path = bundle / "model_card.json"
    if card_path.is_file():
        try:
            include = str(json.loads(card_path.read_text(encoding="utf-8")).get("include", "all"))
        except Exception:  # malformed card -> fall back to requiring the encoder
            include = "all"
    if include != "decoder":
        required.append("image_encoder.onnx")

    missing = [name for name in required if not (bundle / name).is_file()]
    if missing:
        raise typer.BadParameter(
            f"--use-onnx bundle {bundle} is missing required file(s): {', '.join(sorted(missing))}",
            param_hint="--use-onnx",
        )


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
    prompts: str | None = typer.Option(
        None,
        "--prompts",
        help=(
            "Comma-separated class names or path to one-per-line file. "
            "Required for instance task; defaults to class_map names under task: semantic."
        ),
    ),
    output: Path = typer.Option(..., "--output", help="Output directory (created if missing)."),
    checkpoint: Path | None = typer.Option(
        None, "--checkpoint", callback=_validate_checkpoint, help="Adapter checkpoint directory."
    ),
    config: Path | None = typer.Option(None, "--config", help="Path to config YAML."),
    use_onnx: Path | None = typer.Option(
        None,
        "--use-onnx",
        help="Run inference from an exported ONNX bundle dir (image_encoder.onnx + decoder.onnx "
        "+ sidecars) instead of the PyTorch model. Mutually exclusive with --checkpoint/--merge.",
    ),
    score_threshold: float = typer.Option(
        _DEFAULT_SCORE_THRESHOLD,
        "--score-threshold",
        callback=_validate_unit_interval,
        help="Minimum score to keep a prediction [0.0, 1.0].",
    ),
    top_k: int = typer.Option(
        _DEFAULT_TOP_K,
        "--top-k",
        callback=_validate_positive_int,
        help="Max predictions per (image, class) pair.",
    ),
    save_masks: str = typer.Option(
        _DEFAULT_SAVE_MASKS,
        "--save-masks",
        click_type=click.Choice(["rle", "png", "none"]),
        help="Mask output format: rle | png | none. Ignored under task: semantic.",
    ),
    visualize: bool = typer.Option(False, "--visualize", help="Write per-image overlay PNGs."),
    batch_size: str = typer.Option(
        "auto",
        "--batch-size",
        callback=_validate_batch_size,
        help="Images per forward pass: 'auto' or a positive int.",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Preview resolved inputs; skip model load and inference."
    ),
    verbose: VerboseOpt = False,
    progress: ProgressOpt = Progress.auto,
    interactive: bool = typer.Option(
        False,
        "--interactive",
        "-i",
        help="Build a runnable predict command interactively (prompts for all inputs).",
    ),
) -> None:
    """Run inference on images with optional adapter."""
    configure_logging(verbose)
    if interactive:
        from custom_sam_peft.cli import _interactive

        _interactive.require_tty()
        _interactive.run_predict_interactive(force=False)
        return

    # --- ONNX bundle validation (spec §4.3): mutual exclusion + completeness ---
    if use_onnx is not None:
        if checkpoint is not None:
            raise typer.BadParameter(
                "--use-onnx and --checkpoint are mutually exclusive; the bundle already "
                "has the adapter merged in.",
                param_hint="--use-onnx",
            )
        _validate_onnx_bundle(use_onnx)

    # Derive merge: LoRA folds deltas into base weights (forward-equivalent, speed/memory only);
    # QLoRA merge dequantizes 4-bit→compute_dtype (memory blowup), so unmerged is safe default.
    merge_adapter = detect_adapter_kind(checkpoint) == "lora" if checkpoint is not None else False

    # --- Resolve task from config (if present); parse YAML once and reuse ---
    resolved_task = "instance"
    raw_cfg: dict[str, object] = {}
    if config is not None:
        try:
            import yaml

            raw_cfg = yaml.safe_load(config.read_text(encoding="utf-8")) or {}
            resolved_task = str(raw_cfg.get("task", "instance"))
        except Exception as _exc:
            logger.debug("predict CLI: could not read task from config: %s", _exc)

    # --- Prompt defaulting under semantic task ---
    resolved_prompts: str
    if prompts is not None:
        resolved_prompts = prompts
    elif resolved_task == "semantic" and config is not None:
        # Derive class_names from data.semantic.class_map (reuse already-parsed raw_cfg)
        try:
            from custom_sam_peft.data._semantic_encode import build_value_to_label

            data_section = raw_cfg.get("data", {})
            sem_section = data_section.get("semantic", {}) if isinstance(data_section, dict) else {}
            class_map_path = sem_section.get("class_map") if isinstance(sem_section, dict) else None
            ignore_index = (
                int(sem_section.get("ignore_index", 255)) if isinstance(sem_section, dict) else 255
            )
            if class_map_path is None:
                raise typer.BadParameter(
                    "--prompts is required: no data.semantic.class_map in config",
                    param_hint="--prompts",
                )
            class_names, _, _ = build_value_to_label(
                class_map_path,
                ignore_index=ignore_index,
                background_class_name=None,
            )
            resolved_prompts = ",".join(class_names)
        except typer.BadParameter:
            raise
        except Exception as exc:
            raise typer.BadParameter(
                f"--prompts omitted and could not derive class_names from config: {exc}",
                param_hint="--prompts",
            ) from exc
    else:
        # Instance path or no config: --prompts is required
        raise typer.BadParameter(
            "--prompts is required for instance task (or provide --config with task: semantic)",
            param_hint="--prompts",
        )

    # --- Instance-only flags under semantic: emit ONE INFO and ignore ---
    _INSTANCE_ONLY_FLAGS_SET = (
        score_threshold != _DEFAULT_SCORE_THRESHOLD
        or top_k != _DEFAULT_TOP_K
        or save_masks != _DEFAULT_SAVE_MASKS
    )
    if resolved_task == "semantic" and _INSTANCE_ONLY_FLAGS_SET:
        logger.info(
            "predict: --score-threshold, --top-k, and --save-masks are instance-only "
            "and are ignored under task: semantic."
        )

    opts = PredictOptions(
        images=images,
        prompts=resolved_prompts,
        output=output,
        checkpoint=checkpoint,
        merge_adapter=merge_adapter,
        config=config,
        score_threshold=score_threshold,
        top_k=top_k,
        save_masks=save_masks,  # type: ignore[arg-type]
        visualize=visualize,
        device="auto",
        dtype="auto",
        batch_size=batch_size,  # type: ignore[arg-type]
        seed=0,
        dry_run=dry_run,
        verbose=verbose,
        use_onnx=use_onnx,
    )

    mode = resolve_mode(
        None if progress is Progress.auto else progress.value,
        os.environ,
        sys.stdout.isatty(),
        Console().is_jupyter,
    )

    try:
        with progress_session(
            kind=ProgressKind.PREDICT,
            total_batches_per_epoch=0,  # runner updates via P.reset_inner(total=len(image_paths))
            mode=mode,
            # total_epochs omitted — no outer bar for predict (spec §5.4)
        ):
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
