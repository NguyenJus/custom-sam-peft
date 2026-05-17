"""Verify every stub raises NotImplementedError with a spec: reference."""

from __future__ import annotations

import pytest

from esam3.config.schema import (
    EvalConfig,
)
from esam3.eval.evaluator import Evaluator
from esam3.eval.metrics import compute_coco_map


def _assert_stub(call: object) -> None:
    with pytest.raises(NotImplementedError, match="filled in by spec:"):
        call()  # type: ignore[operator]


def test_eval_stubs() -> None:
    _assert_stub(lambda: compute_coco_map(object(), object(), [0.5]))
    ev = Evaluator(EvalConfig())
    _assert_stub(lambda: ev.evaluate(object(), object()))  # type: ignore[arg-type]


def test_trainer_fit_stub() -> None:
    # Trainer is now implemented (Task 8). Verify it raises ValueError for
    # bbox prompt_mode (the v0 guard), not NotImplementedError.
    from esam3.config.schema import (
        DataConfig,
        DataSplit,
        PEFTConfig,
        RunConfig,
        TrainConfig,
        TrainHyperparams,
    )
    from esam3.tracking.noop import NoopTracker
    from esam3.train.trainer import Trainer

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
