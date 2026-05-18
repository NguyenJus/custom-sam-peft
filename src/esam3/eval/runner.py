"""End-to-end eval pipeline — extracted from cli/eval_cmd.py.

The CLI layer is now a thin wrapper over `run_eval`; notebooks can import
this module directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from esam3._registry import lookup
from esam3.config.schema import TrainConfig
from esam3.eval.evaluator import Evaluator
from esam3.eval.metrics import MetricsReport
from esam3.models.sam3 import load_sam31
from esam3.peft_adapters.lora import load_lora


def run_eval(
    cfg: TrainConfig,
    *,
    checkpoint: Path,
    split: Literal["val", "test"] = "val",
    output_dir: Path | None = None,
    save_predictions: bool | None = None,
) -> MetricsReport:
    """Load model + adapter, build dataset, run Evaluator.evaluate_and_save.

    Raises:
        ValueError: cfg.peft.method != 'lora' (QLoRA load is not yet supported).
        ValueError: split == 'test' and cfg.data.test is None.
    """
    if cfg.peft.method != "lora":
        raise ValueError(
            f"checkpoint loading currently supports only LoRA adapters; "
            f"got peft.method={cfg.peft.method!r}"
        )
    if split == "test" and cfg.data.test is None:
        raise ValueError("--split test requires data.test in config; got None for data.test")

    cfg_dict = cfg.data.model_dump()
    if split == "test":
        cfg_dict["val"] = cfg_dict["test"]
    builder = lookup("dataset", cfg.data.format)
    dataset = builder(cfg_dict, model_name=cfg.model.name, pipeline="eval")

    wrapper = load_sam31(cfg.model)
    load_lora(wrapper, checkpoint)

    eval_cfg = cfg.eval
    if save_predictions is not None:
        eval_cfg = eval_cfg.model_copy(update={"save_predictions": save_predictions})

    out = output_dir if output_dir is not None else checkpoint.parent
    return Evaluator(eval_cfg).evaluate_and_save(wrapper, dataset, out)
