"""`custom-sam-peft` CLI entry point — wires subcommands into a Typer app."""

from __future__ import annotations

import typer

import custom_sam_peft._bootstrap  # noqa: F401  # populate plugin registry before subcommand imports
from custom_sam_peft.cli import (
    calibrate_cmd,
    doctor_cmd,
    eval_cmd,
    export_cmd,
    init_cmd,
    predict_cmd,
    run_cmd,
    train_cmd,
)
from custom_sam_peft.cli._progress import _silence_third_party_progress

# Suppress HF / datasets progress bars once at app entry, unconditionally.
# progress_session also calls this defensively on entry — the double-call is safe.
_silence_third_party_progress()

app = typer.Typer(
    name="custom-sam-peft",
    help="Closed-vocab finetuning of SAM-family models with LoRA / QLoRA.",
    no_args_is_help=True,
    add_completion=False,
)

app.command("train", help="Run a finetune.")(train_cmd.train)
app.command("eval", help="Evaluate a checkpoint.")(eval_cmd.evaluate)
app.command("predict", help="Run inference on images with optional adapter.")(predict_cmd.predict)
app.command("export", help="Export adapter or merged model.")(export_cmd.export)
app.command("init", help="Write a starter config.")(init_cmd.init)
app.command("doctor", help="Report environment + dependency status.")(doctor_cmd.doctor)
app.command("calibrate", help="Probe peak VRAM and cache for tighter preset packing.")(
    calibrate_cmd.calibrate
)
app.command("run", help="Train + eval + (optional) export + bundle, in one shot.")(run_cmd.run)


if __name__ == "__main__":  # pragma: no cover
    app()
