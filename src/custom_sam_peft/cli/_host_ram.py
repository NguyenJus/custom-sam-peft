"""Shared exit-message formatter for a host-RAM-floor stop.

Pure string builder — no Typer, no I/O — so it is unit-testable directly.
Both `train` and `run` call it and print via rprint, then exit 0.
"""

from __future__ import annotations

from pathlib import Path

from custom_sam_peft.eval._artifacts import HostRamStop


def _rel(path: Path) -> str:
    """Render relative to cwd when under it, else absolute (matches `done run_dir=` style)."""
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def format_host_ram_message(stop: HostRamStop, *, subcommand: str, config_path: Path) -> str:
    """Build the resume message for a host-RAM-floor stop.

    subcommand is "train" or "run"; config_path is the actual --config the user
    passed. The message states that training stopped because available host RAM
    fell below the configured floor, names the checkpoint directory, and gives
    resume guidance. Best lines appear only when both stop.best_dir and
    stop.best_map are set (the trainer sets them together).
    """
    lines = [
        f"🛑  Host RAM floor ({stop.floor_gb:.1f} GB) reached at step {stop.stop_step} "
        f"(epoch {stop.stop_epoch + 1}/{stop.total_epochs}). "
        f"Available: {stop.available_gb:.2f} GB.",
        f"   Checkpoint saved: {_rel(stop.checkpoint_dir)}/",
    ]
    if stop.best_dir is not None and stop.best_map is not None:
        lines.append(f"   Best so far:      {_rel(stop.best_dir)}/ (mAP {stop.best_map:.3f})")
    lines.append("")
    resume = f"custom-sam-peft {subcommand} --config {config_path} --resume __latest__"
    lines.append(f"   • Resume:            {resume}")
    lines.append(
        "   • Lower memory:      edit config "
        "(e.g. train.num_workers, train.batch_size), then resume."
    )
    if stop.best_dir is not None and stop.best_map is not None:
        lines.append(f"   • Use best as-is:    {_rel(stop.best_dir)}/adapter/")
    return "\n".join(lines)
