"""End-to-end eval pipeline.

The CLI (`custom_sam_peft eval`) is a thin wrapper over `run_eval`. `custom_sam_peft run`
calls it with `val_dataset` / `model` / `return_per_example_iou=True` so
it can re-use a single dataset+wrapper across the eval and bundle phases.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, cast, overload

from custom_sam_peft._registry import lookup
from custom_sam_peft.config.schema import TrainConfig
from custom_sam_peft.data.base import Dataset
from custom_sam_peft.data.val_source import resolve_val_source
from custom_sam_peft.eval._artifacts import EvalArtifacts
from custom_sam_peft.eval.evaluator import Evaluator
from custom_sam_peft.eval.metrics import MetricsReport
from custom_sam_peft.models.sam3 import MULTIPLEX_CAP, load_sam31
from custom_sam_peft.peft_adapters import make_peft_method
from custom_sam_peft.train.checkpoint import _load_channel_adapter


@overload
def run_eval(
    cfg: TrainConfig,
    *,
    checkpoint: Path | None = None,
    artifacts: EvalArtifacts | None = None,
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
    checkpoint: Path | None = None,
    artifacts: EvalArtifacts | None = None,
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
    checkpoint: Path | None = None,
    artifacts: EvalArtifacts | None = None,
    split: Literal["val", "test"] = "val",
    output_dir: Path | None = None,
    save_predictions: bool | None = None,
    val_dataset: Dataset | None = None,
    model: Any | None = None,
    return_per_example_iou: bool = False,
) -> MetricsReport | tuple[MetricsReport, list[float]]:
    """Load model + adapter, build dataset, run Evaluator.

    When ``artifacts`` is provided (the EvalArtifacts seam), the evaluator reads
    ``checkpoint_path``, ``peft_method``, and ``run_dir`` from it and does NOT
    reach into trainer internals. ``checkpoint`` is ignored when ``artifacts`` is
    given.

    When ``artifacts`` is None, the existing standalone-eval-from-config path
    remains (for ``custom-sam-peft eval cfg.yaml``), and ``checkpoint`` must be
    supplied.

    Optional additive kwargs (used by `custom_sam_peft run`):
      - ``val_dataset``: pre-built dataset; skips registry lookup + transform setup.
      - ``model``: pre-loaded + adapted wrapper; skips ``load_sam31`` + adapter load.
      - ``return_per_example_iou``: when True, returns ``(MetricsReport, list[float])``.

    Backward-compat: defaults preserve the previous behavior (rebuild
    dataset, load model + LoRA, return ``MetricsReport``).

    Raises:
        ValueError: split == 'test' and cfg.data.test is None.
        ValueError: neither ``checkpoint`` nor ``artifacts`` provided.
    """
    # Resolve checkpoint and peft_method from artifacts when provided.
    if artifacts is not None:
        resolved_checkpoint = artifacts.checkpoint_path
        resolved_peft_method = artifacts.peft_method
        resolved_run_dir = artifacts.run_dir
    else:
        if checkpoint is None:
            raise ValueError("run_eval requires either 'checkpoint' or 'artifacts' to be provided.")
        resolved_checkpoint = checkpoint
        resolved_peft_method = cfg.peft.method
        resolved_run_dir = None

    _peft_method = make_peft_method(resolved_peft_method)
    _hf_val = (
        cfg.data.format == "hf" and cfg.data.hf is not None and cfg.data.hf.split_val is not None
    )
    if split == "val" and cfg.data.val is None and cfg.data.val_split is None and not _hf_val:
        raise ValueError(
            "--split val requires data.val, data.val_split, or data.hf.split_val in config; "
            "got none."
        )
    if split == "test" and cfg.data.test is None:
        raise ValueError("--split test requires data.test in config; got None for data.test")

    if val_dataset is None:
        cfg_dict = cfg.data.model_dump()
        if split == "test":
            cfg_dict["val"] = cfg_dict["test"]
        elif split == "val" and cfg.data.val_split is not None:
            vs = resolve_val_source(cfg, run_dir=None)
            assert vs.val_ids is not None  # noqa: S101 — auto_split mode invariant
            cfg_dict["_resolved_image_ids"] = {"eval": list(vs.val_ids)}
        builder = lookup("dataset", cfg.data.format)
        dataset = cast(Dataset, builder(cfg_dict, model_name=cfg.model.name, pipeline="eval"))
    else:
        dataset = val_dataset

    if model is None:
        wrapper = load_sam31(
            cfg.model, channels=cfg.data.channels, channel_semantics=cfg.data.channel_semantics
        )
        _peft_method.load_from_disk(wrapper, resolved_checkpoint)
        _load_channel_adapter(wrapper, resolved_checkpoint)
    else:
        wrapper = model

    eval_cfg = cfg.eval
    if save_predictions is not None:
        eval_cfg = eval_cfg.model_copy(update={"save_predictions": save_predictions})

    if eval_cfg.batch_size == "auto":
        from custom_sam_peft.presets import decide_eval_batch_size

        bs, _, _ = decide_eval_batch_size(classes_per_forward=MULTIPLEX_CAP)
        eval_cfg = eval_cfg.model_copy(update={"batch_size": bs})

    evaluator = Evaluator(eval_cfg)
    # Output dir: prefer explicit, then artifacts.run_dir, then checkpoint parent.
    out = (
        output_dir
        if output_dir is not None
        else (resolved_run_dir if resolved_run_dir is not None else resolved_checkpoint.parent)
    )

    if return_per_example_iou:
        # We need both the metrics report (and metrics.json on disk) AND the
        # per-example IoUs. `evaluate_and_save` only persists; call `evaluate`
        # for the data we need and then mirror the persistence the CLI path does.
        out.mkdir(parents=True, exist_ok=True)
        report, per_example_iou = evaluator.evaluate(wrapper, dataset, return_per_example_iou=True)

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
