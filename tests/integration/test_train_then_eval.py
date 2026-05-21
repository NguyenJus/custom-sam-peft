"""Train-then-eval integration: Trainer.fit() produces a real MetricsReport."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

from custom_sam_peft.config.schema import (
    DataConfig,
    DataSplit,
    EvalConfig,
    ModelConfig,
    PEFTConfig,
    RunConfig,
    TrainConfig,
    TrainHyperparams,
)
from custom_sam_peft.eval.metrics import MetricsReport
from custom_sam_peft.peft_adapters.lora import apply_lora
from custom_sam_peft.train.trainer import Trainer
from tests.fixtures.tiny_sam3_lora_stub import FIXTURE_SCOPE_PATTERNS, make_stub_wrapper

pytestmark = pytest.mark.integration


class _RecordingTracker:
    """Minimal tracker that records every log_scalars call."""

    def __init__(self) -> None:
        self.scalars: list[tuple[int, dict[str, float]]] = []

    def start_run(self, run_dir: Any, config: Any, resume_from: Any = None) -> None:
        pass

    def log_scalars(self, step: int, values: dict[str, float]) -> None:
        self.scalars.append((step, values))

    def log_images(self, step: int, images: dict[str, np.ndarray[Any, Any]]) -> None:
        pass

    def close(self) -> None:
        pass


def _make_cfg(tmp_path: Path) -> TrainConfig:
    return TrainConfig(
        run=RunConfig(name="t", output_dir=str(tmp_path / "runs"), seed=0),
        model=ModelConfig(),
        data=DataConfig(
            format="coco",
            train=DataSplit(annotations="x", images="x"),
            val=DataSplit(annotations="x", images="x"),
            prompt_mode="text",
        ),
        peft=PEFTConfig(
            method="lora",
            target_modules=FIXTURE_SCOPE_PATTERNS["vision"],
        ),
        train=TrainHyperparams(
            epochs=1,
            batch_size=1,
            grad_accum_steps=1,
            lr=1e-4,
            warmup_steps=0,
            eval_every=1,
            save_every=1000,
            log_every=1,
            num_workers=0,
        ),
        eval=EvalConfig(mode="full", iou_thresholds=[0.5], lite_max_images=1),
    )


def test_trainer_fit_runs_lite_eval_and_final_full_eval(
    tiny_text_dataset, noop_tracker, tmp_path: Path
):
    wrapper = make_stub_wrapper(dim=8, working=True)
    cfg = _make_cfg(tmp_path)

    apply_lora(wrapper, cfg.peft)
    tracker = _RecordingTracker()
    trainer = Trainer(
        model=wrapper,
        train_ds=tiny_text_dataset,
        val_ds=tiny_text_dataset,
        tracker=tracker,
        cfg=cfg,
    )
    result = trainer.fit(run_dir=tmp_path / "train-then-eval")
    assert isinstance(result.final_metrics, MetricsReport)
    assert "mAP" in result.final_metrics.overall

    metrics_path = result.run_dir / "metrics.json"
    assert metrics_path.exists()
    data = json.loads(metrics_path.read_text())
    assert "overall" in data
    assert "global_step" in data
    assert data["overall"]["mAP"] == result.final_metrics.overall["mAP"]

    # Tracker must have received eval scalars (from on_eval mid-run callback).
    eval_calls = [values for _step, values in tracker.scalars if "mAP" in values]
    assert eval_calls, (
        "Expected tracker to receive at least one log_scalars call with 'mAP' key "
        f"(from lite mid-run eval); got scalars={tracker.scalars}"
    )


def test_trainer_fit_propagates_evaluator_exception(tiny_text_dataset, tmp_path: Path):
    """C1 regression: RuntimeError from Evaluator.evaluate must propagate as-is,
    not be shadowed by UnboundLocalError on full_report/merged_path."""
    wrapper = make_stub_wrapper(dim=8, working=True)
    cfg = _make_cfg(tmp_path)
    apply_lora(wrapper, cfg.peft)

    trainer = Trainer(
        model=wrapper,
        train_ds=tiny_text_dataset,
        val_ds=tiny_text_dataset,
        tracker=_RecordingTracker(),
        cfg=cfg,
    )

    with (
        patch(
            "custom_sam_peft.train.trainer.Evaluator.evaluate",
            side_effect=RuntimeError("injected eval failure"),
        ),
        pytest.raises(RuntimeError, match="injected eval failure"),
    ):
        trainer.fit(run_dir=tmp_path / "fit-propagates")
