"""Interactive `csp init --interactive` wizard.

Declarative WizardStep registry → answers dict → render config_full.yaml →
validate via load_config → emit. See
docs/superpowers/specs/2026-05-26-interactive-setup-wizard-design.md.
"""

from __future__ import annotations

import string
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from importlib.resources import files
from pathlib import Path
from typing import Any, Literal, cast

import typer

from custom_sam_peft.cli.init_cmd import UNIFIED_TEMPLATE, _build_loss_overrides_block
from custom_sam_peft.config.loader import load_config
from custom_sam_peft.config.schema import ClassImbalance
from custom_sam_peft.errors import ConfigError

IMBALANCE_MODERATE_RATIO = 3.0  # R < 3 → balanced
IMBALANCE_SEVERE_RATIO = 10.0  # 3 <= R < 10 → moderate; R >= 10 → severe

RunMode = Literal["train", "run", "eval"]


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


def infer_class_imbalance(annotations: str) -> ClassImbalance:
    """Detect a class-imbalance tier from per-category instance counts.

    Mirrors data/subset.py per-class frequency; uses the pycocotools-backed
    primitives in data/coco.py. On ANY failure (missing/unreadable file, zero
    present categories) returns "balanced".
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
            raise ValueError("no present categories")
        ratio = max(present) / min(present)
    except Exception:
        return "balanced"

    if ratio < IMBALANCE_MODERATE_RATIO:
        return "balanced"
    if ratio < IMBALANCE_SEVERE_RATIO:
        return "moderate"
    return "severe"


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
    )


# ---------------------------------------------------------------------------
# Step ask-functions
# ---------------------------------------------------------------------------


def _ask_run_mode(ctx: Ctx) -> dict[str, Any]:
    ctx.run_mode = ask_choice("Run mode?", ["train", "run", "eval"], default="train")  # type: ignore[assignment]
    return {}


def _ask_run_name(ctx: Ctx) -> dict[str, Any]:
    name = ask_text("Run name?", default="my-run")
    return {"run": {"name": name}}


def _ask_dataset_source(ctx: Ctx) -> dict[str, Any]:
    fmt = ask_choice("Dataset format?", ["coco", "hf"], default="coco")
    if fmt == "coco":
        ann = ask_text("Path to COCO train annotations (.json)?")
        imgs = ask_text("Path to COCO train images dir?")
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
    ann = ask_text("Path to COCO val annotations (.json)?")
    imgs = ask_text("Path to COCO val images dir?")
    return {"data": {"val": {"annotations": ann, "images": imgs}}}


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
    ann = _coco_train_annotations(ctx)
    if ann is None:
        typer.echo(
            "could not auto-detect class imbalance (non-COCO/no annotations); "
            "defaulting to balanced"
        )
        detected: ClassImbalance = "balanced"
    else:
        detected = infer_class_imbalance(ann)
        typer.echo(f"detected class imbalance: {detected}")
    tier = cast(
        ClassImbalance,
        ask_choice("Class imbalance tier?", ["balanced", "moderate", "severe"], default=detected),
    )
    return {"train": {"loss": {"class_imbalance": tier}}}


def _ask_peft_sizing(ctx: Ctx) -> dict[str, Any]:
    from custom_sam_peft.presets import decide_preset

    if ctx.cuda_available and ask_confirm(
        "Auto-size the PEFT config to your GPU's VRAM?", default=True
    ):
        image_size = ctx.answers.get("data", {}).get("image_size", 1008)
        try:
            decision = decide_preset(image_size)
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


STEPS: list[WizardStep] = [
    WizardStep("run_mode", _ask_run_mode),
    WizardStep("run_name", _ask_run_name),
    WizardStep("dataset_source", _ask_dataset_source),
    WizardStep("validation", _ask_validation),
    WizardStep("domain", _ask_domain),
    WizardStep(
        "class_imbalance",
        _ask_class_imbalance,
        when=lambda ctx: ctx.run_mode in {"train", "run"},
    ),
    WizardStep("peft_sizing", _ask_peft_sizing),
    WizardStep("epochs", _ask_epochs, when=lambda ctx: ctx.run_mode != "eval"),
    WizardStep("model_weights", _ask_model_weights),
]


def run_wizard(ctx: Ctx) -> dict[str, Any]:
    for step in STEPS:
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


def _header(launch: str) -> str:
    return (
        f"# Generated by `custom-sam-peft init --interactive` on {date.today().isoformat()}\n"
        f"# Launch: {launch}\n\n"
    )


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


def generate_config(output: Path, *, force: bool, cuda_available: bool) -> tuple[str, RunMode]:
    """Run the wizard, validate, write. Returns (launch_command, run_mode).

    Raises:
      typer.Exit(1): final validation backstop fired (no file written).
      KeyboardInterrupt: propagates; no file written.
    """
    ctx = Ctx(answers={}, cuda_available=cuda_available)
    answers = run_wizard(ctx)  # KeyboardInterrupt propagates out untouched
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
    return launch, ctx.run_mode
