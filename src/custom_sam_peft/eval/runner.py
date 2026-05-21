"""End-to-end eval pipeline.

The CLI (`custom_sam_peft eval`) is a thin wrapper over `run_eval`. `custom_sam_peft run`
calls it with `val_dataset` / `model` / `return_per_example_iou=True` so
it can re-use a single dataset+wrapper across the eval and bundle phases.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, cast, overload

from custom_sam_peft._registry import lookup
from custom_sam_peft.config.schema import TrainConfig
from custom_sam_peft.data.base import Dataset
from custom_sam_peft.eval.evaluator import Evaluator
from custom_sam_peft.eval.metrics import MetricsReport
from custom_sam_peft.models.sam3 import load_sam31
from custom_sam_peft.peft_adapters.lora import load_lora


@overload
def run_eval(
    cfg: TrainConfig,
    *,
    checkpoint: Path,
    split: Literal["val", "test"] = "val",
    output_dir: Path | None = None,
    save_predictions: bool | None = None,
    val_dataset: Dataset | None = None,
    model: Any | None = None,
    return_per_example_iou: Literal[False] = False,
) -> MetricsReport: ...


@overload
def run_eval(
    cfg: TrainConfig,
    *,
    checkpoint: Path,
    split: Literal["val", "test"] = "val",
    output_dir: Path | None = None,
    save_predictions: bool | None = None,
    val_dataset: Dataset | None = None,
    model: Any | None = None,
    return_per_example_iou: Literal[True],
) -> tuple[MetricsReport, list[float]]: ...


def run_eval(
    cfg: TrainConfig,
    *,
    checkpoint: Path,
    split: Literal["val", "test"] = "val",
    output_dir: Path | None = None,
    save_predictions: bool | None = None,
    val_dataset: Dataset | None = None,
    model: Any | None = None,
    return_per_example_iou: bool = False,
) -> MetricsReport | tuple[MetricsReport, list[float]]:
    """Load model + adapter, build dataset, run Evaluator.

    Optional additive kwargs (used by `custom_sam_peft run`):
      - ``val_dataset``: pre-built dataset; skips registry lookup + transform setup.
      - ``model``: pre-loaded + adapted wrapper; skips ``load_sam31`` + ``load_lora``.
      - ``return_per_example_iou``: when True, returns ``(MetricsReport, list[float])``.

    Backward-compat: defaults preserve the previous behavior (rebuild
    dataset, load model + LoRA, return ``MetricsReport``).

    Raises:
        ValueError: cfg.peft.method != 'lora' AND model is None (QLoRA load
            from disk is not yet supported; pre-loaded wrappers bypass this).
        ValueError: split == 'test' and cfg.data.test is None.
    """
    if model is None and cfg.peft.method != "lora":
        raise ValueError(
            f"checkpoint loading currently supports only LoRA adapters; "
            f"got peft.method={cfg.peft.method!r}"
        )
    if split == "test" and cfg.data.test is None:
        raise ValueError("--split test requires data.test in config; got None for data.test")

    if val_dataset is None:
        cfg_dict = cfg.data.model_dump()
        if split == "test":
            cfg_dict["val"] = cfg_dict["test"]
        builder = lookup("dataset", cfg.data.format)
        dataset = cast(Dataset, builder(cfg_dict, model_name=cfg.model.name, pipeline="eval"))
    else:
        dataset = val_dataset

    if model is None:
        wrapper = load_sam31(cfg.model)
        load_lora(wrapper, checkpoint)
    else:
        wrapper = model

    eval_cfg = cfg.eval
    if save_predictions is not None:
        eval_cfg = eval_cfg.model_copy(update={"save_predictions": save_predictions})

    evaluator = Evaluator(eval_cfg)
    out = output_dir if output_dir is not None else checkpoint.parent

    if return_per_example_iou:
        # We need both the metrics report (and metrics.json on disk) AND the
        # per-example IoUs. `evaluate_and_save` only persists; call `evaluate`
        # for the data we need and then mirror the persistence the CLI path does.
        out.mkdir(parents=True, exist_ok=True)
        report, per_example_iou = evaluator.evaluate(wrapper, dataset, return_per_example_iou=True)
        import json

        (out / "metrics.json").write_text(
            json.dumps(
                {
                    "overall": report.overall,
                    "per_class": report.per_class,
                    "n_images": report.n_images,
                    "n_predictions": report.n_predictions,
                },
                indent=2,
            )
        )
        if eval_cfg.save_predictions and eval_cfg.mode == "full":
            (out / "predictions.json").write_text(json.dumps(evaluator._last_predictions))
        return report, per_example_iou

    return evaluator.evaluate_and_save(wrapper, dataset, out)
