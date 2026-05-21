"""Verify every stub raises NotImplementedError with a spec: reference."""

from __future__ import annotations

import pytest

from custom_sam_peft.config.schema import (
    EvalConfig,
)
from custom_sam_peft.eval.evaluator import Evaluator


def test_eval_stubs() -> None:
    # compute_coco_map is implemented (Task 3); Evaluator.evaluate is implemented (Task 4).
    # Nothing left to stub-check in this module — placeholder to keep test collection happy.
    ev = Evaluator(EvalConfig())
    assert ev is not None


def test_trainer_fit_stub() -> None:
    # Trainer is now implemented (Task 8). Verify it raises ValueError for
    # bbox prompt_mode (the v0 guard), not NotImplementedError.
    from custom_sam_peft.config.schema import (
        DataConfig,
        DataSplit,
        PEFTConfig,
        RunConfig,
        TrainConfig,
        TrainHyperparams,
    )
    from custom_sam_peft.tracking.noop import NoopTracker
    from custom_sam_peft.train.trainer import Trainer

    cfg = TrainConfig(
        run=RunConfig(name="t"),
        data=DataConfig(
            format="coco",
            train=DataSplit(annotations="a", images="b"),
            val=DataSplit(annotations="a", images="b"),
            prompt_mode="bbox",
        ),
        peft=PEFTConfig(method="lora"),
        train=TrainHyperparams(epochs=1),
    )
    with pytest.raises(ValueError, match="prompt_mode='bbox'"):
        Trainer(
            model=object(),
            train_ds=object(),  # type: ignore[arg-type]
            val_ds=object(),  # type: ignore[arg-type]
            tracker=NoopTracker(),
            cfg=cfg,
        )
