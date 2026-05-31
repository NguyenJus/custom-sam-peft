"""`custom-sam-peft run` — train + eval + (optional) export + bundle in one shot.

Body is ≤ 30 lines per the cli-design boundary rule. Phase composition and
context assembly live in `_orchestrate` so the Typer command stays a thin shell.

Spec: docs/superpowers/specs/2026-05-18-simplify-ux-design.md §3.
"""

from __future__ import annotations

import json
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
from custom_sam_peft.cli._time_limit import format_time_limit_message
from custom_sam_peft.cli.init_cmd import run_init
from custom_sam_peft.config._duration import parse_duration_to_seconds
from custom_sam_peft.config.loader import load_config
from custom_sam_peft.config.schema import TrainConfig
from custom_sam_peft.data.base import Dataset
from custom_sam_peft.errors import CheckpointError
from custom_sam_peft.models.sam3 import load_sam31
from custom_sam_peft.presets import PresetDecision, decide_preset
from custom_sam_peft.runs.bundle import BundleContext, write_bundle
from custom_sam_peft.train.checkpoint import find_latest_checkpoint, load_adapter
from custom_sam_peft.train.close_out import close_out
from custom_sam_peft.train.runner import run_training

_LATEST_SENTINEL = "__latest__"

if TYPE_CHECKING:
    from custom_sam_peft.data.val_source import ValSource

_LOG = logging.getLogger(__name__)


def _fallback_preset(cfg: TrainConfig) -> PresetDecision:
    """No sidecar — synthesize one from cfg + decide_preset(). Spec §11.4.

    Passes the config's classes_per_forward as the K upper bound (spec §3).
    """
    return decide_preset(k=cfg.train.multiplex.classes_per_forward)


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


def _orchestrate(
    cfg: TrainConfig, resume: Path | None, mode: ProgressMode, *, config_path: Path
) -> int:
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
    if train_result.time_limit_stop is not None:
        rprint(
            format_time_limit_message(
                train_result.time_limit_stop, subcommand="run", config_path=config_path
            )
        )
        return 0
    run_dir = train_result.run_dir
    adapter_path = train_result.checkpoint_path  # run_dir/adapter (best weights)

    # Decide val mode from the saved record — same source of truth the trainer used.
    vs = load_val_source(run_dir)
    if vs is None:
        raise RuntimeError(f"runner did not save val_source.json in {run_dir}")

    # close_out (inside fit()) already ran the single eval + export-merge on the
    # best weights; reuse its results — no second eval, no second merge.
    report = train_result.final_metrics
    per_example_iou = train_result.per_example_iou or []

    val_dataset: Dataset | None = None
    wrapper: Any = None
    if vs.mode != "none":
        # Rebuild the model + val dataset only for bundle re-inference (sample panels).
        wrapper = load_sam31(
            cfg.model, channels=cfg.data.channels, channel_semantics=cfg.data.channel_semantics
        )
        load_adapter(wrapper, adapter_path)
        val_dataset = _build_val_dataset(cfg, vs)

    end_ts = datetime.now(UTC)

    # merged/ was written by close_out when cfg.export.merge (soft-fail: error
    # is surfaced via EvalArtifacts.merged_export_error, run still continues).
    merged_export_error = train_result.merged_export_error
    merged_dir = (
        (run_dir / "merged")
        if (cfg.export.merge and merged_export_error is None and (run_dir / "merged").is_dir())
        else None
    )

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
        ladder_events=train_result.ladder_events,  # from EvalArtifacts, not metrics.json
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


def _read_final_step_epoch(run_dir: Path, resume: Path) -> tuple[int, int]:
    """Read (global_step, epoch) from best.json or the checkpoint's training_state."""
    best_json = run_dir / "best" / "best.json"
    if best_json.is_file():
        try:
            data = json.loads(best_json.read_text())
            return int(data["global_step"]), 0
        except Exception:  # noqa: S110
            pass
    state_file = resume / "training_state.pt"
    if state_file.is_file():
        import torch

        state = torch.load(state_file, weights_only=False)
        return int(state.get("global_step", 0)), int(state.get("epoch", 0))
    return 0, 0


def _finalize(
    cfg: TrainConfig,
    resume: Path,
    *,
    config_path: Path,
) -> int:
    """Productionize a paused run: rebuild + close_out, NO training (spec §11)."""
    from custom_sam_peft.data.val_source import load_val_source

    start_ts = datetime.now(UTC)
    run_dir = resume.parent.parent  # checkpoints/step_N -> run_dir

    # The run's OWN config governs model/eval/export shape (spec §11.2, A6).
    saved_cfg_path = run_dir / "config.yaml"
    saved_cfg = load_config(saved_cfg_path) if saved_cfg_path.is_file() else cfg
    if saved_cfg_path.is_file():
        _LOG.info("finalize: using the run's saved config.yaml (not --config) for fidelity.")

    # Load the adapter into the base model BEFORE close_out: close_out only
    # restores best/, falling back to these in-memory weights when best/ is
    # absent — so this pre-load is the no-best fallback (the resumed adapter).
    # Prefer best/adapter, else the resumed checkpoint's adapter.
    wrapper: Any = load_sam31(
        saved_cfg.model,
        channels=saved_cfg.data.channels,
        channel_semantics=saved_cfg.data.channel_semantics,
    )
    best_adapter = run_dir / "best" / "adapter"
    adapter = best_adapter if best_adapter.is_dir() else resume / "adapter"
    load_adapter(wrapper, adapter)

    # Rebuild val dataset from the saved record.
    vs = load_val_source(run_dir)
    val_ds: Dataset | None = (
        _build_val_dataset(saved_cfg, vs) if (vs is not None and vs.mode != "none") else None
    )

    final_step, final_epoch = _read_final_step_epoch(run_dir, resume)

    artifacts = close_out(
        run_dir,
        wrapper,
        saved_cfg,
        evaluator_val_ds=val_ds,
        oom_state=None,
        final_step=final_step,
        final_epoch=final_epoch,
        ladder_events=None,
    )

    end_ts = datetime.now(UTC)

    # Thread close_out's soft-fail merge error; set merged_dir from disk
    # ONLY when there was no merge error (mirror _orchestrate exactly).
    merged_export_error = artifacts.merged_export_error
    merged_dir = (
        (run_dir / "merged")
        if (
            saved_cfg.export.merge and merged_export_error is None and (run_dir / "merged").is_dir()
        )
        else None
    )

    ctx = BundleContext(
        run_dir=run_dir,
        config_path=run_dir / "config.yaml",
        start_ts=start_ts,
        end_ts=end_ts,
        preset=_load_preset_or_fallback(saved_cfg),
        per_example_iou=artifacts.per_example_iou or [],
        merged_dir=merged_dir,
        merged_export_error=merged_export_error,
        oom_events=artifacts.oom_events,
        ladder_events=artifacts.ladder_events,  # None for finalize — expected
    )
    try:
        write_bundle(ctx, artifacts.final_metrics, val_dataset=val_ds, model_wrapper=wrapper)
    except Exception as exc:
        rprint(f"[red]bundle failed[/red] run_dir={run_dir} — {exc}")
        raise typer.Exit(code=1) from exc

    mAP_str = (
        f"{artifacts.final_metrics.overall.get('mAP', float('nan')):.4f}"
        if artifacts.final_metrics is not None
        else "n/a (no val)"
    )
    rprint(
        f"[green]finalized[/green] run_dir={run_dir} adapter={run_dir / 'adapter'} "
        f"merged={(merged_dir or merged_export_error or 'skipped')} "
        f"summary={run_dir / 'summary.md'} mAP={mAP_str}"
    )
    return 0


def run(
    config: Path = typer.Option(..., "--config", help="Path to config YAML."),
    resume: str | None = typer.Option(
        None,
        "--resume",
        help=(
            "Resume checkpoint. Pass a path, or omit value for the latest "
            "checkpoint matching cfg.run.name."
        ),
    ),
    time_limit: str | None = typer.Option(
        None,
        "--time-limit",
        help=(
            'Wall-clock budget for this run (e.g. "2h30m", "90m", "3600s", or bare '
            "seconds). Overrides train.time_limit. The budget is per-run; --resume "
            "restarts the clock."
        ),
        metavar="DURATION",
    ),
    finalize: bool = typer.Option(
        False,
        "--finalize",
        help=(
            "Finalize a paused (time-limited) run: rebuild the model from --resume's "
            "checkpoint, restore the best weights, run eval, and write adapter/merged/"
            "metrics/bundle. Runs NO training. Requires --resume; rejects --time-limit."
        ),
    ),
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Enable DEBUG logging."),
    progress_flag: str = typer.Option(
        "auto",
        "--progress",
        help="Progress display mode: auto|on|off|plain.",
        metavar="MODE",
    ),
    visualize: bool = typer.Option(
        True,
        "--visualize/--no-visualize",
        help="Write GT-vs-Pred composite panels in the eval phase.",
    ),
) -> None:
    """Alias for `train --eval --export`.

    Use this when you want the full pipeline in one command. The Colab
    notebook uses `run` for the canonical end-to-end flow.
    """
    configure_logging(verbose)
    if not config.is_file():
        rprint(
            f"[yellow]{config} not initialized — auto-init (formula, no probe) then run.[/yellow]"
        )
        run_init("coco-text-lora", config, force=False)
    cfg = load_config(config)
    cfg = cfg.model_copy(update={"eval": cfg.eval.model_copy(update={"visualize": visualize})})
    if finalize:
        if resume is None:
            rprint("[red]error[/red] --finalize requires --resume (a checkpoint or __latest__).")
            raise typer.Exit(code=1)
        if time_limit is not None:
            rprint(
                "[red]error[/red] --finalize cannot be combined with --time-limit (no training)."
            )
            raise typer.Exit(code=1)
    if time_limit is not None:
        try:
            parse_duration_to_seconds(time_limit)
        except ValueError as e:
            rprint(f"[red]error[/red] invalid --time-limit: {e}")
            raise typer.Exit(code=1) from e
        cfg = cfg.model_copy(
            update={"train": cfg.train.model_copy(update={"time_limit": time_limit})}
        )
    mode = resolve_mode(
        progress_flag if progress_flag != "auto" else None,
        os.environ,
        sys.stdout.isatty(),
        Console().is_jupyter,
    )

    resume_path: Path | None
    if resume == _LATEST_SENTINEL:
        try:
            resume_path = find_latest_checkpoint(cfg)
        except CheckpointError as e:
            rprint(f"[red]error[/red] {e}")
            raise typer.Exit(code=1) from e
    elif resume is not None:
        resume_path = Path(resume)
    else:
        resume_path = None

    if finalize:
        assert resume_path is not None  # guaranteed by the validation above (mypy)  # noqa: S101
        raise typer.Exit(code=_finalize(cfg, resume_path, config_path=config))
    _orchestrate(cfg, resume_path, mode, config_path=config)
