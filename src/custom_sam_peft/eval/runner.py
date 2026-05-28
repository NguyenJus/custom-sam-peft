"""End-to-end eval pipeline.

The CLI (`custom_sam_peft eval`) is a thin wrapper over `run_eval`. `custom_sam_peft run`
calls it with `val_dataset` / `model` / `return_per_example_iou=True` so
it can re-use a single dataset+wrapper across the eval and bundle phases.
"""

from __future__ import annotations

import json
import logging
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
from custom_sam_peft.train.checkpoint import load_adapter

_LOG = logging.getLogger(__name__)


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
    ``checkpoint_path`` and ``run_dir`` from it and does NOT reach into trainer
    internals. ``checkpoint`` is ignored when ``artifacts`` is given.

    When ``artifacts`` is None (standalone eval path), ``checkpoint`` is optional:
      - ``checkpoint`` supplied: loads the adapter from disk, dispatching LoRA vs
        QLoRA via the canonical sentinel-based discovery seam
        (``train.checkpoint.load_adapter``). Emits a WARNING when the detected
        method disagrees with ``cfg.peft.method`` (config value is ignored for
        dispatch; the checkpoint's sentinel wins).
      - ``checkpoint=None``: evaluates the baseline (zero-shot) SAM — no adapter
        is loaded and no channel-adapter restore is attempted.

    Optional additive kwargs (used by `custom_sam_peft run`):
      - ``val_dataset``: pre-built dataset; skips registry lookup + transform setup.
      - ``model``: pre-loaded + adapted wrapper; skips ``load_sam31`` + adapter load.
      - ``return_per_example_iou``: when True, returns ``(MetricsReport, list[float])``.

    Backward-compat: defaults preserve the previous behavior (rebuild
    dataset, load model + LoRA, return ``MetricsReport``).

    Raises:
        ValueError: split == 'test' and cfg.data.test is None.
    """
    # Resolve checkpoint and run_dir from artifacts when provided.
    resolved_checkpoint: Path | None
    if artifacts is not None:
        resolved_checkpoint = artifacts.checkpoint_path
        resolved_run_dir = artifacts.run_dir
    else:
        resolved_checkpoint = checkpoint  # may be None → baseline
        resolved_run_dir = None
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
        if resolved_checkpoint is not None:
            from custom_sam_peft.peft_adapters import discover_method_from_checkpoint

            detected = discover_method_from_checkpoint(resolved_checkpoint)
            if cfg.peft.method != detected:
                _LOG.warning(
                    "cfg.peft.method=%r but the checkpoint at %s is %r; loading the "
                    "checkpoint's method (config value ignored for eval dispatch).",
                    cfg.peft.method,
                    resolved_checkpoint,
                    detected,
                )
            load_adapter(wrapper, resolved_checkpoint)
        # else: baseline — no adapter load, no channel-adapter restore.
    else:
        wrapper = model

    eval_cfg = cfg.eval
    if save_predictions is not None:
        eval_cfg = eval_cfg.model_copy(update={"save_predictions": save_predictions})

    if eval_cfg.batch_size == "auto":
        from custom_sam_peft.presets import decide_eval_batch_size

        bs, _, _ = decide_eval_batch_size(classes_per_forward=MULTIPLEX_CAP)
        # Cap by the configured train batch size to avoid eval OOM.
        train_cap = cfg.train.batch_size
        if bs > train_cap:
            _LOG.info(
                "eval auto-batch capped at train batch (%d) — predictor picked %d",
                train_cap,
                bs,
            )
            bs = min(bs, train_cap)
        eval_cfg = eval_cfg.model_copy(update={"batch_size": bs})

    evaluator = Evaluator(eval_cfg)
    # Output dir: prefer explicit arg, then artifacts.run_dir, then checkpoint parent,
    # then cfg.run.output_dir (baseline path where checkpoint is None).
    if output_dir is not None:
        out = output_dir
    elif resolved_run_dir is not None:
        out = resolved_run_dir
    elif resolved_checkpoint is not None:
        out = resolved_checkpoint.parent
    else:
        out = Path(cfg.run.output_dir) if cfg.run.output_dir else Path.cwd()

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
