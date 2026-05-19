"""`esam3` CLI entry point — wires subcommands into a Typer app."""

from __future__ import annotations

import typer

import esam3._bootstrap  # noqa: F401  # populate plugin registry before subcommand imports
from esam3.cli import (
    doctor_cmd,
    eval_cmd,
    export_cmd,
    init_cmd,
    train_cmd,
)

app = typer.Typer(
    name="esam3",
    help="Parameter-efficient finetuning of SAM3.1.",
    no_args_is_help=True,
    add_completion=False,
)

app.command("train", help="Run a finetune.")(train_cmd.train)
app.command("eval", help="Evaluate a checkpoint.")(eval_cmd.evaluate)
app.command("export", help="Export adapter or merged model.")(export_cmd.export)
app.command("init", help="Write a starter config.")(init_cmd.init)
app.command("doctor", help="Report environment + dependency status.")(doctor_cmd.doctor)


if __name__ == "__main__":  # pragma: no cover
    app()
