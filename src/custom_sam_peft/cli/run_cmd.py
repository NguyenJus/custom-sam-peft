"""`custom-sam-peft run` — train + eval + (optional) export + bundle in one shot.

Body is ≤ 30 lines per the cli-design boundary rule. Phase composition and
context assembly live in `_orchestrate` so the Typer command stays a thin shell.

Spec: docs/superpowers/specs/2026-05-18-simplify-ux-design.md §3.
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import typer
from rich import print as rprint
from rich.console import Console

from custom_sam_peft._registry import lookup
from custom_sam_peft.cli._logging import configure_logging
from custom_sam_peft.cli._progress import ProgressKind, ProgressMode, progress_session, resolve_mode
from custom_sam_peft.config.loader import load_config
from custom_sam_peft.config.schema import TrainConfig
from custom_sam_peft.data.base import Dataset
from custom_sam_peft.eval.runner import run_eval
from custom_sam_peft.models.sam3 import load_sam31
from custom_sam_peft.presets import PresetDecision, decide_preset
from custom_sam_peft.runs.bundle import BundleContext, write_bundle
from custom_sam_peft.train.checkpoint import load_adapter, save_merged
from custom_sam_peft.train.runner import run_training

if TYPE_CHECKING:
    from custom_sam_peft.data.val_source import ValSource

_LOG = logging.getLogger(__name__)


def _fallback_preset(cfg: TrainConfig) -> PresetDecision:
    """No sidecar — synthesize one from cfg + decide_preset(). Spec §11.4."""
    return decide_preset(image_size=cfg.data.image_size)


def _load_preset_or_fallback(cfg: TrainConfig) -> PresetDecision:
    sidecar = Path("preset.json")
    if sidecar.is_file():
        return PresetDecision.from_json(sidecar.read_text())
    return _fallback_preset(cfg)


def _build_val_dataset(cfg: TrainConfig, vs: ValSource) -> Dataset:
    """Build the val dataset using the same image ids the trainer used.

    Spec: docs/superpowers/specs/2026-05-22-data-no-val-auto-split-design.md §7.6.
    """
    data_cfg_dict = cfg.data.model_dump()
    if vs.mode == "auto_split":
        assert vs.val_ids is not None  # noqa: S101 — mode invariant for type narrowing
        data_cfg_dict["_resolved_image_ids"] = {"eval": list(vs.val_ids)}
    builder = lookup("dataset", cfg.data.format)
    return cast(Dataset, builder(data_cfg_dict, model_name=cfg.model.name, pipeline="eval"))


def _orchestrate(cfg: TrainConfig, resume: Path | None, mode: ProgressMode) -> int:
    from custom_sam_peft.data.val_source import load_val_source

    start_ts = datetime.now(UTC)

    # Phase: train.
    try:
        with progress_session(
            kind=ProgressKind.TRAIN,
            total_epochs=cfg.train.epochs,
            total_batches_per_epoch=0,  # Trainer updates dynamically via reset_inner
            mode=mode,
        ):
            train_result = run_training(cfg, resume_from=resume)
    except Exception as exc:
        rprint(f"[red]train failed[/red] {exc}")
        raise typer.Exit(code=1) from exc
    run_dir = train_result.run_dir
    adapter_path = train_result.checkpoint_path

    # Decide val mode from the saved record — same source of truth the trainer used.
    vs = load_val_source(run_dir)
    if vs is None:
        raise RuntimeError(f"runner did not save val_source.json in {run_dir}")

    wrapper: Any = load_sam31(cfg.model)
    load_adapter(wrapper, adapter_path)

    val_dataset: Dataset | None = None
    report: Any = None
    per_example_iou: list[float] = []
    if vs.mode != "none":
        val_dataset = _build_val_dataset(cfg, vs)

        # Phase: eval.
        try:
            with progress_session(
                kind=ProgressKind.EVAL,
                total_batches_per_epoch=0,  # Evaluator updates via P.advance_inner
                mode=mode,
            ):
                report, per_example_iou = cast(
                    tuple[Any, list[float]],
                    run_eval(
                        cfg,
                        checkpoint=adapter_path,
                        output_dir=run_dir,
                        val_dataset=val_dataset,
                        model=wrapper,
                        return_per_example_iou=True,
                    ),
                )
        except Exception as exc:
            rprint(f"[red]eval failed[/red] run_dir={run_dir} — {exc}")
            raise typer.Exit(code=1) from exc

    end_ts = datetime.now(UTC)

    # Phase: export-merge (conditional, soft-fail).
    merged_dir: Path | None = None
    merged_export_error: str | None = None
    if cfg.export.merge:
        target = run_dir / "merged"
        try:
            with progress_session(
                kind=ProgressKind.EXPORT_MERGE,
                total_batches_per_epoch=0,
                mode=mode,
            ):
                save_merged(wrapper, target)
            merged_dir = target
        except Exception as exc:
            _LOG.warning("export-merge failed: %s", exc)
            merged_export_error = str(exc)

    # Phase: bundle.
    preset = _load_preset_or_fallback(cfg)
    ctx = BundleContext(
        run_dir=run_dir,
        config_path=run_dir / "config.yaml",
        start_ts=start_ts,
        end_ts=end_ts,
        preset=preset,
        per_example_iou=per_example_iou,
        merged_dir=merged_dir,
        merged_export_error=merged_export_error,
        oom_events=train_result.oom_events,
    )
    try:
        write_bundle(ctx, report, val_dataset=val_dataset, model_wrapper=wrapper)
    except Exception as exc:
        rprint(f"[red]bundle failed[/red] run_dir={run_dir} — {exc}")
        raise typer.Exit(code=1) from exc

    mAP_str = (
        f"{report.overall.get('mAP', float('nan')):.4f}" if report is not None else "n/a (no val)"
    )
    rprint(
        f"[green]done[/green] run_dir={run_dir} adapter={adapter_path} "
        f"merged={(merged_dir or merged_export_error or 'skipped')} "
        f"summary={run_dir / 'summary.md'} mAP={mAP_str}"
    )
    return 0


def run(
    config: Path = typer.Option(..., "--config", help="Path to config YAML."),
    resume: Path | None = typer.Option(None, "--resume", help="Path to resume checkpoint."),
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Enable DEBUG logging."),
    progress_flag: str = typer.Option(
        "auto",
        "--progress",
        help="Progress display mode: auto|on|off|plain.",
        metavar="MODE",
    ),
) -> None:
    """Alias for `train --eval --export`.

    Use this when you want the full pipeline in one command. The Colab
    notebook uses `run` for the canonical end-to-end flow.
    """
    configure_logging(verbose)
    cfg = load_config(config)
    if cfg.data.prompt_mode == "bbox":
        raise typer.BadParameter(
            "prompt_mode='bbox' is not supported for training in v0.",
            param_hint="--config",
        )
    mode = resolve_mode(
        progress_flag if progress_flag != "auto" else None,
        os.environ,
        sys.stdout.isatty(),
        Console().is_jupyter,
    )
    _orchestrate(cfg, resume, mode)
