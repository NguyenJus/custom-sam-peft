"""`esam3 eval` — load a config + adapter checkpoint and run the Evaluator."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import typer
from rich import print as rprint

from esam3.config.loader import load_config
from esam3.config.schema import DataSplit, TrainConfig


def _build_dataset(cfg: TrainConfig, split: DataSplit) -> Any:
    """Build a dataset for the given split config.

    Only COCO format is supported today; HF and other formats are a TODO.
    """
    if cfg.data.format == "coco":
        from esam3.data.coco import COCODataset
        from esam3.data.transforms import build_eval_transforms

        transforms = build_eval_transforms(
            cfg.data.image_size,
            model_name=cfg.model.name,
            normalize=cfg.data.normalize,
        )

        return COCODataset(
            annotations=split.annotations,
            images=split.images,
            prompt_mode=cfg.data.prompt_mode,
            transforms=transforms,
            text_prompt=cfg.data.text_prompt,
        )
    # TODO: add HF dataset support (out of scope for Task 6)
    raise NotImplementedError(
        f"Dataset format {cfg.data.format!r} is not yet supported by the eval CLI. "
        "Only 'coco' is currently implemented."
    )


def _run_eval(
    *,
    config: Path,
    checkpoint: Path,
    split: str,
    output: Path | None,
    save_predictions: bool | None,
) -> None:
    cfg = load_config(config)

    if cfg.peft.method != "lora":
        raise typer.BadParameter(
            f"--checkpoint loading currently supports only LoRA adapters; "
            f"got peft.method={cfg.peft.method!r}",
            param_hint="--checkpoint",
        )

    if split == "val":
        data_split = cfg.data.val
    elif split == "test":
        if cfg.data.test is None:
            raise typer.BadParameter(
                "--split test requires data.test in config; got None for data.test",
                param_hint="--split",
            )
        data_split = cfg.data.test
    else:
        raise typer.BadParameter(
            f"--split must be val|test; got {split!r}",
            param_hint="--split",
        )

    dataset = _build_dataset(cfg, data_split)

    from esam3.models.sam3 import load_sam31
    from esam3.peft_adapters.lora import load_lora

    model = load_sam31(cfg.model)
    load_lora(model, checkpoint)

    eval_cfg = cfg.eval
    if save_predictions is not None:
        eval_cfg = eval_cfg.model_copy(update={"save_predictions": save_predictions})

    output_dir = output if output is not None else checkpoint.parent

    from esam3.eval.evaluator import Evaluator

    report = Evaluator(eval_cfg).evaluate_and_save(model, dataset, output_dir)
    rprint(f"[green]eval complete[/green] — {report.overall}")


def evaluate(
    config: Path = typer.Option(..., "--config", help="Path to config YAML."),
    checkpoint: Path = typer.Option(..., "--checkpoint", help="Path to adapter checkpoint."),
    split: str = typer.Option("val", "--split", help="Dataset split: val | test."),
    output: Path | None = typer.Option(
        None, "--output", help="Output dir; defaults to checkpoint.parent."
    ),
    save_predictions: bool | None = typer.Option(
        None,
        "--save-predictions/--no-save-predictions",
        help="Override cfg.eval.save_predictions.",
    ),
) -> None:
    """Evaluate a checkpoint on the val or test split."""
    _run_eval(
        config=config,
        checkpoint=checkpoint,
        split=split,
        output=output,
        save_predictions=save_predictions,
    )
