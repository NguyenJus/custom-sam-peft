"""Shared exit-message formatter for a time-limited stop (spec §4.8).

Pure string builder — no Typer, no I/O — so it is unit-testable directly.
Both `train` and `run` call it and print via rprint, then exit 0.
"""

from __future__ import annotations

from pathlib import Path

from custom_sam_peft.eval._artifacts import TimeLimitStop


def _rel(path: Path) -> str:
    """Render relative to cwd when under it, else absolute (matches `done run_dir=` style)."""
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def format_time_limit_message(stop: TimeLimitStop, *, subcommand: str, config_path: Path) -> str:
    """Build the resume message for a time-limited stop.

    subcommand is "train" or "run"; config_path is the actual --config the user
    passed. The duration label comes from stop.duration_label (format_seconds),
    so a 9000s and a "2h30m" stop render identically. Best lines appear only
    when both stop.best_dir and stop.best_map are set (the trainer sets them
    together).
    """
    lines = [
        f"⏱  Time limit ({stop.duration_label}) reached at step {stop.stop_step} "
        f"(epoch {stop.stop_epoch + 1}/{stop.total_epochs}).",
        f"   Checkpoint saved: {_rel(stop.checkpoint_dir)}/",
    ]
    if stop.best_dir is not None and stop.best_map is not None:
        lines.append(f"   Best so far:      {_rel(stop.best_dir)}/ (mAP {stop.best_map:.3f})")
    lines.append("")
    resume = f"custom-sam-peft {subcommand} --config {config_path} --resume __latest__"
    lines.append(f"   • Resume:            {resume}")
    if stop.best_dir is not None and stop.best_map is not None:
        lines.append(f"   • Use best as-is:    {_rel(stop.best_dir)}/adapter/")
    return "\n".join(lines)
