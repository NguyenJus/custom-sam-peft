"""`esam3 doctor` — environment diagnostics formatter."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from esam3.diagnostics import DoctorReport, run_doctor


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

    if report.issues:
        issues = Table(title="Issues", show_header=False, box=None)
        for msg in report.issues:
            issues.add_row("•", msg)
        console.print(issues)


def doctor(
    weights_path: Path | None = typer.Option(
        None, "--weights-path", help="Override SAM 3.1 weights file path."
    ),
    json_output: bool = typer.Option(False, "--json", help="Emit JSON instead of a table."),
) -> None:
    """Report environment + dependency status."""
    report = run_doctor(weights_path=weights_path)
    if json_output:
        print(json.dumps(dataclasses.asdict(report), default=str, indent=2))
    else:
        _render_table(report)
