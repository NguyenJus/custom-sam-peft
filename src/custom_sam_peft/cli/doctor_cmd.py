"""`custom-sam-peft doctor` — environment diagnostics formatter."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from custom_sam_peft.config.loader import ConfigError, load_config
from custom_sam_peft.config.schema import TrainConfig
from custom_sam_peft.data.aug_presets import (
    _STEP_NAMES_FOR,
    dump_augmentation_pipeline,
    resolve,
)
from custom_sam_peft.data.transforms import resolve_normalization_with_path
from custom_sam_peft.diagnostics import DoctorReport, run_doctor
from custom_sam_peft.models.losses import dump_loss_bundle
from custom_sam_peft.models.losses import resolve as resolve_losses
from custom_sam_peft.models.losses.presets import _TERM_CLASS_NAMES


def _render_table(report: DoctorReport) -> None:
    console = Console()

    runtime = Table(title="Runtime", show_header=False, box=None)
    runtime.add_row("python", report.python_version)
    runtime.add_row("platform", report.platform)
    runtime.add_row("torch", report.torch_version)
    runtime.add_row("cuda build", report.cuda_build or "(none)")
    runtime.add_row("cuda available", str(report.cuda_available))
    console.print(runtime)

    if report.gpus:
        gpu = Table(title="GPU")
        gpu.add_column("idx")
        gpu.add_column("name")
        gpu.add_column("cap")
        gpu.add_column("free MiB", justify="right")
        gpu.add_column("total MiB", justify="right")
        for g in report.gpus:
            gpu.add_row(
                str(g.index),
                g.name,
                f"{g.capability[0]}.{g.capability[1]}",
                str(g.free_mib),
                str(g.total_mib),
            )
        console.print(gpu)

    opt = Table(title="Optional deps", show_header=False, box=None)
    for name, ver in report.optional_deps.items():
        opt.add_row(name, ver or "(missing)")
    console.print(opt)

    core = Table(title="Core versions", show_header=False, box=None)
    for name, ver in report.core_versions.items():
        core.add_row(name, ver)
    console.print(core)

    w = report.sam3_weights
    weights = Table(title="SAM 3.1 weights", show_header=False, box=None)
    weights.add_row("path", str(w.path))
    weights.add_row("exists", str(w.exists))
    weights.add_row("size", f"{w.size_bytes:,}" if w.size_bytes is not None else "(n/a)")
    console.print(weights)

    hf = report.hf_auth
    auth = Table(title="HuggingFace auth", show_header=False, box=None)
    auth.add_row("token source", hf.token_source)
    auth.add_row("has token", str(hf.has_token))
    console.print(auth)

    if report.dataset is not None:
        ds = report.dataset
        tbl = Table(title="Dataset", show_header=False, box=None)
        tbl.add_row("format", ds.format)
        tbl.add_row("train", f"{ds.train_kept}/{ds.train_total}")
        tbl.add_row("val", f"{ds.val_kept}/{ds.val_total}")
        tbl.add_row("limit.strategy", ds.limit_strategy)
        tbl.add_row("limit.seed", str(ds.limit_seed))
        tbl.add_row("limit.train", str(ds.limit_train))
        tbl.add_row("limit.val", str(ds.limit_val))
        console.print(tbl)

    if report.issues:
        issues = Table(title="Issues", show_header=False, box=None)
        for msg in report.issues:
            issues.add_row("•", msg)
        console.print(issues)


def _render_resolved_config_tables(cfg: TrainConfig) -> None:
    """Spec §11.2.1 + §11.2.2 — two additional tables when --config is set."""
    console = Console()

    resolved = resolve(cfg.data.augmentations)
    aug = Table(title="Resolved augmentations", show_header=False, box=None)
    aug.add_row("preset", cfg.data.augmentations.preset)
    aug.add_row("intensity", cfg.data.augmentations.intensity)
    aug.add_row("hflip", str(resolved.hflip))
    aug.add_row("vflip", str(resolved.vflip))
    aug.add_row("rotate90", str(resolved.rotate90))
    aug.add_row("rotate_arbitrary", str(resolved.rotate_arbitrary))
    aug.add_row("color_jitter", str(resolved.color_jitter))
    aug.add_row("stain_jitter", str(resolved.stain_jitter))
    aug.add_row("blur", str(resolved.blur))
    aug.add_row("gauss_noise", str(resolved.gauss_noise))
    aug.add_row("steps", ", ".join(_STEP_NAMES_FOR(resolved)))
    console.print(aug)

    mean, std, path = resolve_normalization_with_path(cfg.model.name, cfg.data.normalize)
    norm = Table(title="Normalization", show_header=False, box=None)
    norm.add_row("model.name", cfg.model.name)
    norm.add_row("mean", str(mean))
    norm.add_row("std", str(std))
    norm.add_row("resolution path", path)
    console.print(norm)

    losses_resolved = resolve_losses(cfg.train.loss)
    loss_table = Table(title="Resolved losses", show_header=False, box=None)
    loss_table.add_column("knob")
    loss_table.add_column("value")
    loss_table.add_row("preset", cfg.train.loss.preset)
    loss_table.add_row("class_imbalance", cfg.train.loss.class_imbalance)
    for fname in (
        "mask_family",
        "box_family",
        "obj_family",
        "presence_family",
        "w_mask",
        "w_box",
        "w_obj",
        "w_presence",
        "focal_gamma",
        "focal_alpha",
        "tversky_alpha",
        "tversky_gamma",
        "boundary_weight",
    ):
        loss_table.add_row(fname, str(getattr(losses_resolved, fname)))
    term_classes = {
        "mask": _TERM_CLASS_NAMES["mask"][losses_resolved.mask_family],
        "box": _TERM_CLASS_NAMES["box"][losses_resolved.box_family],
        "obj": _TERM_CLASS_NAMES["obj"][losses_resolved.obj_family],
        "presence": _TERM_CLASS_NAMES["presence"][losses_resolved.presence_family],
    }
    loss_table.add_row(
        "term_classes",
        ", ".join(f"{k}={v}" for k, v in term_classes.items()),
    )
    console.print(loss_table)


def _build_resolved_config_json(cfg: TrainConfig) -> dict[str, object]:
    """Spec §11.2.3 — additive `resolved_config` block injected into --json."""
    aug_dump = dump_augmentation_pipeline(cfg.data.augmentations)
    mean, std, path = resolve_normalization_with_path(cfg.model.name, cfg.data.normalize)
    return {
        "augmentations": {
            "preset": aug_dump["preset"],
            "intensity": aug_dump["intensity"],
            "resolved": aug_dump["resolved"],
            "steps": aug_dump["steps"],
        },
        "normalize": {
            "model_name": cfg.model.name,
            "mean": mean,
            "std": std,
            "resolution_path": path,
        },
        "loss": {
            k: v for k, v in dump_loss_bundle(cfg.train.loss).items() if k != "library_version"
        },
    }


def doctor(
    weights_path: Path | None = typer.Option(
        None, "--weights-path", help="Override SAM 3.1 weights file path."
    ),
    json_output: bool = typer.Option(False, "--json", help="Emit JSON instead of a table."),
    config_path: Path | None = typer.Option(
        None,
        "--config",
        help=(
            "Load + validate a config YAML. Reports resolved dataset sizes (may "
            "import pycocotools or trigger datasets.load_dataset) and renders the "
            "resolved augmentations and normalization derived from the config."
        ),
    ),
) -> None:
    """Report environment + dependency status."""
    report = run_doctor(weights_path=weights_path, config_path=config_path)
    # run_doctor already swallows ConfigError / ValidationError into report.issues;
    # we mirror that here so a bad --config still exits 0 with the rendered report.
    cfg = None
    if config_path is not None:
        try:
            cfg = load_config(config_path)
        except (ConfigError, ValueError):
            cfg = None

    if json_output:
        blob = dataclasses.asdict(report)
        if cfg is not None:
            blob["resolved_config"] = _build_resolved_config_json(cfg)
        print(json.dumps(blob, default=str, indent=2))  # noqa: T201
    else:
        _render_table(report)
        if cfg is not None:
            _render_resolved_config_tables(cfg)
