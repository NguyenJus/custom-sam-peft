"""Shared interactive-CLI machinery for `init -i`, `eval -i`, and `predict -i`.

Prompt primitives, the WizardStep/Ctx/run_wizard registry-driver, reusable
steps (dataset_source, validation, model_weights), the TTY guard, adapter-peek,
small validators, and the per-command interactive helpers.

Import discipline (spec §2): this module imports only typer, stdlib,
config.loader/config.schema, and — LAZILY, inside function bodies — the
peft_adapters seam and errors. It MUST NOT import init_cmd / setup_wizard /
eval_cmd / predict_cmd at module scope.
"""

from __future__ import annotations

import shlex
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Literal

import typer

from custom_sam_peft.config.loader import load_config

RunMode = Literal["train", "run", "eval"]  # superset; init -i narrows to train|run


@dataclass
class Ctx:
    answers: dict[str, Any]
    cuda_available: bool
    run_mode: RunMode = "train"
    categories: list[str] | None = None
    category_counts: dict[str, int] | None = None


@dataclass(frozen=True)
class WizardStep:
    id: str
    ask: Callable[[Ctx], dict[str, Any]]
    when: Callable[[Ctx], bool] = field(default=lambda ctx: True)


def _deep_merge(dst: dict[str, Any], src: dict[str, Any]) -> None:
    """Recursively merge src into dst. Nested dicts merge; scalars/lists overwrite."""
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_merge(dst[k], v)
        else:
            dst[k] = v


def ask_text(
    prompt: str,
    *,
    default: str | None = None,
    validate: Callable[[str], str | None] | None = None,
) -> str:
    """Free-text prompt; re-asks on validate failure. validate returns an error string or None."""
    while True:
        value = (
            typer.prompt(prompt, default=default) if default is not None else typer.prompt(prompt)
        )
        value = str(value).strip()
        if validate is not None:
            err = validate(value)
            if err is not None:
                typer.echo(err)
                continue
        return value


def ask_choice(prompt: str, choices: list[str], *, default: str | None = None) -> str:
    """Membership-checked choice; re-asks on invalid."""
    rendered = f"{prompt} [{'/'.join(choices)}]"
    while True:
        value = (
            typer.prompt(rendered, default=default)
            if default is not None
            else typer.prompt(rendered)
        )
        value = str(value).strip()
        if value in choices:
            return value
        typer.echo(f"choose one of: {', '.join(choices)}")


def ask_confirm(prompt: str, *, default: bool = True) -> bool:
    return typer.confirm(prompt, default=default)


def _detect_json_candidates(data_dir: Path = Path("data")) -> list[Path]:
    """Return sorted .json files found directly in data_dir, or empty list."""
    if not data_dir.is_dir():
        return []
    return sorted(data_dir.glob("*.json"))


def _detect_dir_candidates(subdirs: list[str], data_dir: Path = Path("data")) -> list[Path]:
    """Return the first matching subdir under data_dir that exists, or empty list."""
    return [data_dir / s for s in subdirs if (data_dir / s).is_dir()]


def _auto_detect_path(
    label: str,
    prompt: str,
    candidates: list[Path],
    *,
    validate: Callable[[str], str | None] | None = None,
) -> str:
    """Propose the first candidate (if any) and confirm; fall back to ask_text."""
    if candidates:
        typer.echo(f"Detected {label}: {candidates[0]}")
        if ask_confirm("Use this path?", default=True):
            return str(candidates[0])
    return ask_text(prompt, validate=validate)


def _ask_dataset_source(ctx: Ctx) -> dict[str, Any]:
    fmt = ask_choice("Dataset format?", ["coco", "hf"], default="coco")
    if fmt == "coco":
        ann = _auto_detect_path(
            "train annotations",
            "Path to COCO train annotations (.json)?",
            _detect_json_candidates(),
        )
        imgs = _auto_detect_path(
            "train images dir",
            "Path to COCO train images dir?",
            _detect_dir_candidates(["train"]),
        )
        return {"data": {"format": "coco", "train": {"annotations": ann, "images": imgs}}}
    name = ask_text("HuggingFace dataset name (org/dataset)?")
    return {"data": {"format": "hf", "hf": {"name": name}}}


def _ask_validation(ctx: Ctx) -> dict[str, Any]:
    fmt = ctx.answers.get("data", {}).get("format", "coco")
    mode = ask_choice("Validation?", ["explicit", "auto-split", "none"], default="auto-split")
    if mode == "none":
        if ctx.run_mode in {"eval", "run"}:
            typer.echo(
                "note: eval/run needs a validation set to score against; "
                "selecting none means eval will have nothing to evaluate."
            )
        return {}
    if mode == "auto-split":

        def _fraction(s: str) -> str | None:
            try:
                f = float(s)
            except ValueError:
                return "fraction must be a number"
            return None if 0.0 < f <= 0.5 else "fraction must be in (0, 0.5]"

        frac = ask_text("Auto-split fraction (0<f<=0.5)?", default="0.1", validate=_fraction)
        return {"data": {"val_split": {"fraction": float(frac)}}}
    if fmt == "hf":
        split = ask_text("HF validation split name?", default="validation")
        return {"data": {"hf": {"split_val": split}}}
    json_candidates = _detect_json_candidates()
    val_candidates = [p for p in json_candidates if "val" in p.name.lower()] or json_candidates
    ann = _auto_detect_path(
        "val annotations",
        "Path to COCO val annotations (.json)?",
        val_candidates,
    )
    imgs = _auto_detect_path(
        "val images dir",
        "Path to COCO val images dir?",
        _detect_dir_candidates(["val"]),
    )
    return {"data": {"val": {"annotations": ann, "images": imgs}}}


def _ask_model_weights(ctx: Ctx) -> dict[str, Any]:
    def _is_file_or_blank(s: str) -> str | None:
        if s == "":
            return None
        return None if Path(s).is_file() else f"no file at {s}"

    raw = ask_text(
        "Path to an existing SAM 3.1 checkpoint (.pt)? Leave blank to use "
        "`models/sam3.1` and download if missing.",
        default="",
        validate=_is_file_or_blank,
    )
    if raw:
        p = Path(raw)
        return {"model": {"local_dir": str(p.parent), "checkpoint_file": p.name}}
    hits = sorted(Path("models").glob("**/sam3.1_multiplex.pt")) if Path("models").is_dir() else []
    if hits:
        return {"model": {"local_dir": str(hits[0].parent)}}
    return {}


def run_wizard(ctx: Ctx, steps: list[WizardStep]) -> dict[str, Any]:
    for step in steps:
        if step.when(ctx):
            fragment = step.ask(ctx)
            _deep_merge(ctx.answers, fragment)
    return ctx.answers


# ---------------------------------------------------------------------------
# validate + emit
# ---------------------------------------------------------------------------

_LAUNCH_VERB = {"train": "train", "run": "run", "eval": "eval"}


def validate(rendered: str) -> None:
    """Validate the exact bytes via load_config by round-tripping through a temp file."""
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write(rendered)
        tmp = Path(f.name)
    try:
        load_config(tmp)
    finally:
        tmp.unlink(missing_ok=True)


def _launch_command(output: Path, run_mode: RunMode) -> str:
    return f"custom-sam-peft {_LAUNCH_VERB[run_mode]} --config {output}"


def _header(launch: str, generating_command: str = "custom-sam-peft init --interactive") -> str:
    return (
        f"# Generated by `{generating_command}` on {date.today().isoformat()}\n"
        f"# Launch: {launch}\n\n"
    )


# ---------------------------------------------------------------------------
# TTY guard, validators, adapter-peek
# ---------------------------------------------------------------------------


def require_tty() -> None:
    """Raise typer.BadParameter if stdin is not a TTY. Call BEFORE any prompt."""
    if not sys.stdin.isatty():
        raise typer.BadParameter(
            "interactive mode needs a TTY; use the flag-driven command instead"
        )


def validate_checkpoint_dir(s: str) -> str | None:
    """ask_text validator: None unless s is a dir containing adapter_config.json."""
    p = Path(s)
    if p.is_dir() and (p / "adapter_config.json").is_file():
        return None
    return f"{s} is not an adapter checkpoint dir (missing adapter_config.json)"


def validate_config_with_eval_split(s: str) -> str | None:
    """ask_text validator for eval-reuse: None when s load_config's AND carries a
    val / val_split / hf.split_val / test source; else an error string."""
    from custom_sam_peft.errors import ConfigError

    try:
        cfg = load_config(Path(s))
    except ConfigError as exc:
        return str(exc)
    hf_has_split_val = (
        cfg.data.format == "hf" and cfg.data.hf is not None and cfg.data.hf.split_val is not None
    )
    has_split = (
        cfg.data.val is not None
        or cfg.data.val_split is not None
        or hf_has_split_val
        or cfg.data.test is not None
    )
    if has_split:
        return None
    return "config has no val/test split to evaluate; pick a config with one"


def peek_adapter(checkpoint_dir: Path) -> tuple[str, str | None]:
    """Return (pretty_method_name, base_model_name) for a known-good adapter dir.

    The caller validates dir existence + adapter_config.json presence (via
    validate_checkpoint_dir) BEFORE calling. Lazy-imports the peft_adapters seam.
    """
    from custom_sam_peft.peft_adapters import (
        discover_method_from_checkpoint,
        method_pretty_name,
        read_adapter_base_model_name,
    )

    method = discover_method_from_checkpoint(checkpoint_dir)
    return method_pretty_name(method), read_adapter_base_model_name(checkpoint_dir)


def _echo_peek(checkpoint_dir: Path) -> None:
    """Echo the detected adapter method + base model for a known-good checkpoint dir."""
    pretty, base = peek_adapter(checkpoint_dir)
    typer.echo(f"detected adapter: {pretty}, base model: {base or '(unspecified)'}")


# ---------------------------------------------------------------------------
# eval --interactive helper
# ---------------------------------------------------------------------------


def _eval_reuse() -> None:
    """Interactive reuse path: print a runnable eval command; write nothing."""
    config_path = ask_text(
        "Path to your existing training config (.yaml)?",
        validate=validate_config_with_eval_split,
    )
    checkpoint_dir = ask_text(
        "Path to the adapter checkpoint directory?",
        validate=validate_checkpoint_dir,
    )
    _echo_peek(Path(checkpoint_dir))
    split = ask_choice("Which split?", ["val", "test"], default="val")
    typer.echo(
        f"custom-sam-peft eval --config {config_path} --checkpoint {checkpoint_dir} --split {split}"
    )


def _eval_baseline(*, output: Path, force: bool) -> None:
    """Interactive baseline path: run shared wizard steps, write a config, print command."""
    from custom_sam_peft.cli.setup_wizard import emit, render

    ctx = Ctx(answers={"run": {"name": "baseline-eval"}}, cuda_available=False, run_mode="eval")
    steps = [
        WizardStep("dataset_source", _ask_dataset_source),
        WizardStep("validation", _ask_validation),
        WizardStep("model_weights", _ask_model_weights),
    ]
    answers = run_wizard(ctx, steps)
    rendered = render(answers, run_mode="eval")
    validate(rendered)
    emit(rendered, output, force, run_mode="eval")
    typer.echo(f"custom-sam-peft eval --config {output} --split val")


def run_eval_interactive(*, output: Path | None, force: bool) -> None:
    """Entry point for `csp eval --interactive`.

    Prompts the user to choose between evaluating a trained adapter (reuse) or
    running a zero-shot baseline (baseline). The reuse path prints a runnable
    command and writes nothing; the baseline path drives the shared wizard steps,
    validates and writes a config, then prints a runnable command.
    """
    mode = ask_choice(
        "Evaluate a trained adapter, or baseline zero-shot SAM?",
        ["reuse", "baseline"],
        default="reuse",
    )
    if mode == "reuse":
        _eval_reuse()
    else:
        out = output if output is not None else Path("baseline-eval.yaml")
        _eval_baseline(output=out, force=force)


# ---------------------------------------------------------------------------
# predict --interactive helper
# ---------------------------------------------------------------------------

_PREDICT_DEFAULT_MODEL = "facebook/sam3.1"
_THIN_CONFIG_NAME = "predict-config.yaml"


def run_predict_interactive(*, force: bool) -> None:
    """Entry point for `csp predict --interactive`.

    Prompts for all predict inputs, optionally writes a thin config when the
    channel count or semantics differ from the RGB default, then prints a
    fully-assembled, runnable predict command.
    """
    from custom_sam_peft.data.channel_semantics import CHANNEL_SEMANTIC_NAMES

    # --- validators ---
    def _positive_int(s: str) -> str | None:
        try:
            n = int(s)
        except ValueError:
            return "must be a positive integer"
        return None if n >= 1 else "must be >= 1"

    def _unit(s: str) -> str | None:
        try:
            f = float(s)
        except ValueError:
            return "must be a number in [0.0, 1.0]"
        return None if 0.0 <= f <= 1.0 else "must be in [0.0, 1.0]"

    # P1: checkpoint
    checkpoint = ask_text(
        "Adapter checkpoint directory? Leave blank for baseline (no adapter).",
        default="",
        validate=lambda s: None if s == "" else validate_checkpoint_dir(s),
    )
    if checkpoint:
        _echo_peek(Path(checkpoint))

    # P2: channels
    channels = int(
        ask_text(
            "Number of input image channels?",
            default="3",
            validate=_positive_int,
        )
    )

    # P3: semantics
    semantics = ask_choice(
        "Channel semantics?",
        list(CHANNEL_SEMANTIC_NAMES),
        default="rgb",
    )

    # P4: merge (only when checkpoint given)
    if checkpoint:
        merge_adapter = ask_confirm("Merge adapter weights before inference?", default=True)
    else:
        merge_adapter = True  # unused

    # P5: threshold
    threshold = float(
        ask_text(
            "Minimum score to keep a prediction [0.0-1.0]?",
            default="0.3",
            validate=_unit,
        )
    )

    # P6: save_masks
    save_masks = ask_choice(
        "Mask output format?",
        ["rle", "png", "none"],
        default="rle",
    )

    # P7: visualize
    visualize = ask_confirm("Write per-image overlay PNGs?", default=False)

    # P8: images
    images = ask_text("Images: dir / glob / manifest / single file?")

    # P9: prompts
    prompts = ask_text("Class prompts (comma-separated) or path to a one-per-line file?")

    # P10: output
    output = ask_text("Output directory?")

    # --- thin config ---
    thin_path: Path | None = None
    needs_thin = channels != 3 or semantics != "rgb"
    if needs_thin:
        thin = Path(_THIN_CONFIG_NAME)
        if thin.exists() and not force:
            raise typer.BadParameter(
                f"refusing to overwrite existing {thin}; pass --force",
                param_hint="--config",
            )
        content = (
            f"# Generated by `custom-sam-peft predict --interactive`"
            f" on {date.today().isoformat()}\n"
            f"model:\n"
            f"  name: {_PREDICT_DEFAULT_MODEL}\n"
            f"data:\n"
            f"  channels: {channels}\n"
            f"  channel_semantics: {semantics}\n"
        )
        thin.write_text(content)
        thin_path = thin

    # --- command assembly ---
    parts: list[str] = [
        "custom-sam-peft predict",
        f"--images {shlex.quote(images)}",
        f"--prompts {shlex.quote(prompts)}",
        f"--output {shlex.quote(output)}",
    ]
    if checkpoint:
        parts.append(f"--checkpoint {shlex.quote(checkpoint)}")
        parts.append("--merge-adapter" if merge_adapter else "--no-merge-adapter")
    parts.append(f"--score-threshold {threshold}")
    parts.append(f"--save-masks {save_masks}")
    if visualize:
        parts.append("--visualize")
    if thin_path is not None:
        parts.append(f"--config {shlex.quote(str(thin_path))}")

    typer.echo(" ".join(parts))
    typer.echo(
        "note: --top-k, --device, --dtype, --batch-size, --seed stay at defaults;"
        " append them as flags if you need to override them."
    )
