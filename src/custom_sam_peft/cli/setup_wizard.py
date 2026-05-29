"""Interactive `csp init --interactive` wizard.

Declarative WizardStep registry → answers dict → render config_full.yaml →
validate via load_config → emit. See
docs/superpowers/specs/2026-05-26-interactive-setup-wizard-design.md.
"""

from __future__ import annotations

import string
from importlib.resources import files
from pathlib import Path
from typing import Any

import typer

from custom_sam_peft.cli._interactive import (
    Ctx,
    RunMode,
    WizardStep,
    _ask_dataset_source,
    _ask_model_weights,
    _ask_validation,
    _deep_merge,  # noqa: F401  # re-exported: tests access sw._deep_merge
    _header,
    _launch_command,
    ask_choice,
    ask_confirm,
    ask_text,
    run_wizard,
    validate,
)
from custom_sam_peft.cli.init_cmd import UNIFIED_TEMPLATE, _build_loss_overrides_block
from custom_sam_peft.config.schema import ClassImbalance
from custom_sam_peft.errors import ConfigError

IMBALANCE_MODERATE_RATIO = 3.0  # R < 3 → balanced
IMBALANCE_SEVERE_RATIO = 10.0  # 3 <= R < 10 → moderate; R >= 10 → severe


def measure_class_imbalance_ratio(annotations: str) -> float | None:
    """Compute the most-to-least-frequent per-category instance-count ratio.

    Mirrors data/subset.py per-class frequency; uses the pycocotools-backed
    primitives in data/coco.py. On ANY failure (missing/unreadable file, zero
    present categories) returns None.
    """
    try:
        from custom_sam_peft.data.coco import _build_category_remap, _load_coco_index

        coco = _load_coco_index(annotations)
        _sparse_ids, remap, _names = _build_category_remap(coco)
        counts: dict[int, int] = {}
        for img_id in coco.getImgIds():
            anns = coco.loadAnns(coco.getAnnIds(imgIds=[img_id]))
            for a in anns:
                if int(a.get("iscrowd", 0)) != 0:
                    continue
                dense = remap.get(int(a["category_id"]))
                if dense is None:
                    continue
                counts[dense] = counts.get(dense, 0) + 1
        present = [c for c in counts.values() if c > 0]
        if not present:
            return None
        return max(present) / min(present)
    except Exception:
        return None


def ratio_to_tier(ratio: float) -> ClassImbalance:
    """Map a most-to-least-frequent ratio to a class-imbalance tier."""
    if ratio < IMBALANCE_MODERATE_RATIO:
        return "balanced"
    if ratio < IMBALANCE_SEVERE_RATIO:
        return "moderate"
    return "severe"


def infer_class_imbalance(annotations: str) -> ClassImbalance:
    """Detect a class-imbalance tier from per-category instance counts.

    On ANY failure (missing/unreadable file, zero present categories) returns
    "balanced".
    """
    ratio = measure_class_imbalance_ratio(annotations)
    if ratio is None:
        return "balanced"
    return ratio_to_tier(ratio)


# ---------------------------------------------------------------------------
# render helpers
# ---------------------------------------------------------------------------


def _model_block(answers: dict[str, Any]) -> str:
    m = answers.get("model", {})
    local_dir = m.get("local_dir", "models/sam3.1")
    ckpt = m.get("checkpoint_file", "sam3.1_multiplex.pt")
    return f"  name: facebook/sam3.1\n  local_dir: {local_dir}\n  checkpoint_file: {ckpt}"


def _dataset_block(answers: dict[str, Any]) -> str:
    data = answers.get("data", {})
    if data.get("format") == "hf":
        hf = data.get("hf", {})
        name = hf["name"]
        lines = [
            "  format: hf",
            "  hf:",
            f"    name: {name}",
            "    split_train: train",
        ]
        if hf.get("split_val") is not None:
            lines.append(f"    split_val: {hf['split_val']}")
        lines += [
            "  # Required stub — not used by the HF loader (set format: coco to use it):",
            "  train:",
            "    annotations: data/train.json",
            "    images: data/train/",
        ]
        return "\n".join(lines)
    train = data.get("train", {})
    ann = train.get("annotations", "data/train.json")
    imgs = train.get("images", "data/train/")
    return (
        "  format: coco\n"
        "  train:\n"
        f"    annotations: {ann}\n"
        f"    images: {imgs}\n"
        "  # HuggingFace alternative — set format: hf and uncomment:\n"
        "  # hf:\n"
        "  #   name: org/dataset\n"
        "  #   split_train: train\n"
        "  #   split_val: validation"
    )


def _validation_block(answers: dict[str, Any]) -> str:
    data = answers.get("data", {})
    hf = data.get("hf", {})
    hf_explicit = data.get("format") == "hf" and hf.get("split_val") is not None
    explicit_active = auto_active = noval_active = False
    if data.get("val") is not None:
        explicit_active = True
        v = data["val"]
        active = f"  val:\n    annotations: {v['annotations']}\n    images: {v['images']}"
    elif data.get("val_split") is not None:
        auto_active = True
        active = f"  val_split:\n    fraction: {data['val_split']['fraction']}\n    seed: null"
    elif hf_explicit:
        # Validation comes from data.hf.split_val (rendered in the dataset block).
        active = "  # validation: HF split set via data.hf.split_val above is used as the val set."
    else:
        noval_active = True
        active = "  # no-val mode: neither val: nor val_split: is set."
    alts = []
    if not explicit_active:
        alts.append(
            "  # Explicit-val alternative (COCO):\n"
            "  # val:\n"
            "  #   annotations: data/val.json\n"
            "  #   images: data/val/"
        )
    if not auto_active:
        alts.append(
            "  # Auto-split alternative:\n  # val_split:\n  #   fraction: 0.1\n  #   seed: null"
        )
    if not noval_active:
        alts.append("  # No-val alternative: omit val:, val_split:, and hf.split_val.")
    return "\n".join([active, *alts])


def _qlora_block(answers: dict[str, Any]) -> str:
    if answers.get("peft", {}).get("method") == "qlora":
        return "  qlora:\n    quant_type: nf4\n    compute_dtype: bfloat16"
    return ""


def _aug_overrides_block() -> str:
    return (
        "# Override individual knobs here; unset keys inherit from (preset, intensity).\n"
        "    # overrides:\n"
        "    #   hflip: false\n"
        "    #   color_jitter: 0.15"
    )


def _limit_validate(s: str) -> str | None:
    """Validate a limit field: blank | int >= 1 | float in (0.0, 1.0]."""
    if s == "":
        return None
    if "." in s:
        try:
            v = float(s)
        except ValueError:
            return "enter a float in (0.0, 1.0], an integer >= 1, or leave blank"
        if 0.0 < v <= 1.0:
            return None
        return "float limit must be in (0.0, 1.0]"
    try:
        v = int(s)
    except ValueError:
        return "enter an integer >= 1, a float in (0.0, 1.0], or leave blank"
    if v >= 1:
        return None
    return "integer limit must be >= 1"


def _parse_limit_value(s: str) -> int | float:
    """Parse a validated non-blank limit string into int or float."""
    if "." in s:
        return float(s)
    return int(s)


def _limit_block(answers: dict[str, Any]) -> str:
    """Render the data.limit YAML block (active when set, commented otherwise)."""
    limit = answers.get("data", {}).get("limit", {})
    train = limit.get("train")
    val = limit.get("val")
    if train is not None or val is not None:
        lines = ["  limit:"]
        if train is not None:
            lines.append(f"    train: {train}")
        if val is not None:
            lines.append(f"    val: {val}")
        lines.append("    # Advanced knobs (uncomment to override defaults):")
        lines.append("    # strategy: random   # random | stratified | first_n")
        lines.append("    # seed: 42")
        return "\n".join(lines)
    return (
        "  # Limit dataset size for quick/smoke runs (int = count, float in (0,1] = fraction):\n"
        "  # limit:\n"
        "  #   train: 100\n"
        "  #   val: 50\n"
        "  #   strategy: random   # random | stratified | first_n\n"
        "  #   seed: 42"
    )


def render(answers: dict[str, Any], *, run_mode: RunMode) -> str:
    """Render the YAML config string from collected answers."""
    data = answers.get("data", {})
    aug = data.get("augmentations", {})
    loss = answers.get("train", {}).get("loss", {})
    epochs = answers.get("train", {}).get("epochs", 1)  # eval defaults to 1
    preset = aug.get("preset", "natural")
    raw = (files("custom_sam_peft.cli.templates") / UNIFIED_TEMPLATE).read_text()
    return string.Template(raw).substitute(
        run_name=answers.get("run", {}).get("name", "my-run"),
        peft_method=answers.get("peft", {}).get("method", "lora"),
        epochs=epochs,
        aug_preset=preset,
        loss_preset=loss.get("preset", "natural"),
        aug_intensity=aug.get("intensity", "medium"),
        class_imbalance=loss.get("class_imbalance", "balanced"),
        overrides_block=_aug_overrides_block(),
        loss_overrides_block=_build_loss_overrides_block(preset),
        model_block=_model_block(answers),
        dataset_block=_dataset_block(answers),
        validation_block=_validation_block(answers),
        qlora_block=_qlora_block(answers),
        limit_block=_limit_block(answers),
    )


# ---------------------------------------------------------------------------
# Step ask-functions
# ---------------------------------------------------------------------------


def _ask_run_mode(ctx: Ctx) -> dict[str, Any]:
    ctx.run_mode = ask_choice("Run mode?", ["train", "run"], default="train")  # type: ignore[assignment]
    return {}


def _ask_run_name(ctx: Ctx) -> dict[str, Any]:
    name = ask_text("Run name?", default="my-run")
    return {"run": {"name": name}}


def _ask_domain(ctx: Ctx) -> dict[str, Any]:
    domain = ask_choice(
        "Domain?",
        ["natural", "medical", "satellite", "microscopy", "none"],
        default="natural",
    )
    intensity = ask_choice(
        "Augmentation intensity?", ["safe", "medium", "aggressive"], default="medium"
    )
    return {
        "data": {"augmentations": {"preset": domain, "intensity": intensity}},
        "train": {"loss": {"preset": domain}},
    }


def _coco_train_annotations(ctx: Ctx) -> str | None:
    data = ctx.answers.get("data", {})
    if data.get("format") != "coco":
        return None
    ann = data.get("train", {}).get("annotations")
    return str(ann) if ann is not None else None


def _ask_class_imbalance(ctx: Ctx) -> dict[str, Any]:
    balanced: dict[str, Any] = {"train": {"loss": {"class_imbalance": "balanced"}}}
    ann = _coco_train_annotations(ctx)
    if ann is None:
        typer.echo(
            "Could not auto-detect class imbalance (non-COCO/no annotations); "
            "defaulting to balanced loss weighting."
        )
        return balanced
    ratio = measure_class_imbalance_ratio(ann)
    if ratio is None:
        typer.echo(
            "Could not measure class imbalance (unreadable annotations/no categories); "
            "defaulting to balanced loss weighting."
        )
        return balanced
    tier = ratio_to_tier(ratio)
    if tier == "balanced":
        typer.echo(
            f"Detected class imbalance: most-to-least frequent class ratio is {ratio:.1f}x "
            "(balanced); no meaningful imbalance to handle."
        )
        return balanced
    typer.echo(
        f"Detected class imbalance: most-to-least frequent class ratio is {ratio:.1f}x ({tier})."
    )
    if ask_confirm("Let the wizard handle this class imbalance automatically?", default=True):
        return {"train": {"loss": {"class_imbalance": tier}}}
    return balanced


def _ask_peft_sizing(ctx: Ctx) -> dict[str, Any]:
    from custom_sam_peft.presets import decide_preset

    if ctx.cuda_available and ask_confirm(
        "Auto-size the PEFT config to your GPU's VRAM?", default=True
    ):
        try:
            decision = decide_preset()
        except RuntimeError as exc:
            typer.echo(f"could not auto-size: {exc}; falling back to manual")
        else:
            typer.echo(decision.label())
            return decision.config_patch
    method = ask_choice("PEFT method?", ["lora", "qlora"], default="lora")
    return {"peft": {"method": method}}


def _ask_epochs(ctx: Ctx) -> dict[str, Any]:
    def _positive_int(s: str) -> str | None:
        try:
            return None if int(s) > 0 else "epochs must be a positive integer"
        except ValueError:
            return "epochs must be a positive integer"

    epochs = ask_text("Number of epochs?", default="10", validate=_positive_int)
    return {"train": {"epochs": int(epochs)}}


def _ask_limit(ctx: Ctx) -> dict[str, Any]:
    if not ask_confirm("Limit dataset size for a quick/smoke run?", default=False):
        return {}
    limit: dict[str, int | float] = {}
    train_raw = ask_text(
        "Train limit (int count or float fraction, blank = no limit)?",
        default="",
        validate=_limit_validate,
    )
    if train_raw:
        limit["train"] = _parse_limit_value(train_raw)
    val_raw = ask_text(
        "Val limit (int count or float fraction, blank = no limit)?",
        default="",
        validate=_limit_validate,
    )
    if val_raw:
        limit["val"] = _parse_limit_value(val_raw)
    if not limit:
        return {}
    return {"data": {"limit": limit}}


STEPS: list[WizardStep] = [
    WizardStep("run_mode", _ask_run_mode),
    WizardStep("run_name", _ask_run_name),
    WizardStep("dataset_source", _ask_dataset_source),
    WizardStep("validation", _ask_validation),
    WizardStep("limit", _ask_limit),
    WizardStep("domain", _ask_domain),
    WizardStep("class_imbalance", _ask_class_imbalance),
    WizardStep("peft_sizing", _ask_peft_sizing),
    WizardStep("epochs", _ask_epochs),
    WizardStep("model_weights", _ask_model_weights),
]


# ---------------------------------------------------------------------------
# validate + emit
# ---------------------------------------------------------------------------


def emit(rendered: str, output: Path, force: bool, *, run_mode: RunMode) -> str:
    """Write header + rendered config to output. Returns the launch command."""
    if output.exists() and not force:
        raise typer.BadParameter(
            f"refusing to overwrite existing {output}; pass --force",
            param_hint="--output",
        )
    launch = _launch_command(output, run_mode)
    output.write_text(_header(launch) + rendered)
    return launch


def _invoke_calibrate(output: Path) -> None:
    """Run the opt-in config-aware calibration probe on the just-written config."""
    from custom_sam_peft.cli.calibrate_cmd import calibrate
    from custom_sam_peft.presets import CACHE_FILENAME

    # Call the Typer command function directly with ALL options passed explicitly,
    # so Typer's OptionInfo defaults are never relied upon. Keep passing every
    # option explicitly if `calibrate` gains new ones.
    calibrate(config=output, output=Path(CACHE_FILENAME), force=False)


def generate_config(output: Path, *, force: bool, cuda_available: bool) -> tuple[str, RunMode]:
    """Run the wizard, validate, write. Returns (launch_command, run_mode).

    Raises:
      typer.Exit(1): final validation backstop fired (no file written).
      KeyboardInterrupt: propagates; no file written.
    """
    ctx = Ctx(answers={}, cuda_available=cuda_available)
    answers = run_wizard(ctx, STEPS)  # KeyboardInterrupt propagates out untouched
    rendered = render(answers, run_mode=ctx.run_mode)
    launch = _launch_command(output, ctx.run_mode)
    body = _header(launch) + rendered
    try:
        validate(body)
    except ConfigError as exc:
        typer.echo(f"error: generated config failed validation:\n{exc}", err=True)
        typer.echo(f"answers: {answers}", err=True)
        raise typer.Exit(code=1) from exc
    if output.exists() and not force:
        raise typer.BadParameter(
            f"refusing to overwrite existing {output}; pass --force",
            param_hint="--output",
        )
    output.write_text(body)
    typer.echo(f"wrote {output}")
    typer.echo(launch)
    if cuda_available and ask_confirm(
        "Run `csp calibrate` now to tighten the VRAM sizing with a live GPU probe? "
        "(opt-in; loads the model and runs one forward+backward)",
        default=False,
    ):
        try:
            _invoke_calibrate(output)
        except typer.Exit:
            typer.echo("calibration did not complete; keeping the formula-derived config", err=True)
    return launch, ctx.run_mode
