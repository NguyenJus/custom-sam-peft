"""`custom-sam-peft` CLI entry point — wires subcommands into a Typer app."""

from __future__ import annotations

import sys

import typer

from custom_sam_peft._bootstrap import bootstrap

bootstrap()  # populate plugin registry + configure logging before subcommand imports

from custom_sam_peft.cli import (  # noqa: E402
    calibrate_cmd,
    doctor_cmd,
    eval_cmd,
    export_cmd,
    init_cmd,
    predict_cmd,
    run_cmd,
    train_cmd,
)
from custom_sam_peft.cli._progress import _silence_third_party_progress  # noqa: E402
from custom_sam_peft.errors import CustomSamPeftError  # noqa: E402

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
app.command(
    "run", help="Train + eval + (optional) export + bundle. Alias for train --eval --export."
)(run_cmd.run)

# Module-level flag: set to True by main() when -v / --verbose appears in sys.argv
# so that the CustomSamPeftError handler can decide whether to render or re-raise.
_verbose: bool = False


def _render_error(e: CustomSamPeftError) -> str:
    """Format a CustomSamPeftError into the four-part user-facing message."""
    parts = [str(e)]
    if e.expected:
        parts.append(f"Expected: {e.expected}")
    if e.found:
        parts.append(f"Found: {e.found}")
    if e.fix:
        parts.append(f"Fix: {e.fix}")
    parts.append("Rerun with -v for full traceback.")
    return "\n".join(parts)


def main() -> None:
    """Entry point that wraps app() with CustomSamPeftError handling."""
    global _verbose
    _verbose = "-v" in sys.argv or "--verbose" in sys.argv
    try:
        app()
    except CustomSamPeftError as e:
        if _verbose:
            raise
        typer.secho(_render_error(e), err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1) from None


if __name__ == "__main__":  # pragma: no cover
    main()
