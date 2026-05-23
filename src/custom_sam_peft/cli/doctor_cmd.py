"""`custom-sam-peft doctor` — environment diagnostics formatter."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from custom_sam_peft.diagnostics import DoctorReport, run_doctor


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

    if report.data is not None:
        d = Table(title="Data", show_header=False, box=None)
        d.add_row("val mode", report.data.val_mode)
        if report.data.val_path is not None:
            d.add_row("val path", report.data.val_path)
        if report.data.val_split_fraction is not None:
            d.add_row("val_split.fraction", f"{report.data.val_split_fraction:.3f}")
            d.add_row("val_split.seed", str(report.data.val_split_seed))
        console.print(d)


def doctor(
    weights_path: Path | None = typer.Option(
        None, "--weights-path", help="Override SAM 3.1 weights file path."
    ),
    config_path: Path | None = typer.Option(
        None,
        "--config",
        help=(
            "Load + validate a config YAML; enables the Data table and reports "
            "resolved dataset sizes. Heavy: may import pycocotools or trigger "
            "datasets.load_dataset."
        ),
    ),
    json_output: bool = typer.Option(False, "--json", help="Emit JSON instead of a table."),
) -> None:
    """Report environment + dependency status."""
    report = run_doctor(weights_path=weights_path, config_path=config_path)
    if json_output:
        print(json.dumps(dataclasses.asdict(report), default=str, indent=2))  # noqa: T201
    else:
        _render_table(report)
