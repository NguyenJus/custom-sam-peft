"""Verify every stub raises NotImplementedError with a spec: reference."""

from __future__ import annotations

import pytest

from esam3.config.schema import (
    EvalConfig,
    PEFTConfig,
)
from esam3.eval.evaluator import Evaluator
from esam3.eval.metrics import compute_coco_map
from esam3.peft_adapters.qlora import apply_qlora
from esam3.train.checkpoint import load_adapter, save_adapter, save_merged
from esam3.train.loop import run_epoch


def _assert_stub(call: object) -> None:
    with pytest.raises(NotImplementedError, match="filled in by spec:"):
        call()  # type: ignore[operator]


def test_peft_stubs() -> None:
    qcfg = PEFTConfig(method="qlora")
    _assert_stub(lambda: apply_qlora(object(), qcfg))


def test_eval_stubs() -> None:
    _assert_stub(lambda: compute_coco_map(object(), object(), [0.5]))
    ev = Evaluator(EvalConfig())
    _assert_stub(lambda: ev.evaluate(object(), object()))  # type: ignore[arg-type]


def test_train_stubs(tmp_path: object) -> None:
    from pathlib import Path

    p = Path(str(tmp_path)) / "x"  # type: ignore[arg-type]
    _assert_stub(lambda: save_adapter(object(), p))
    _assert_stub(lambda: save_merged(object(), p))
    _assert_stub(lambda: load_adapter(object(), p))
    from esam3.config.schema import TrainHyperparams

    _assert_stub(lambda: run_epoch(object(), object(), object(), TrainHyperparams(epochs=1), 0))


def test_trainer_fit_stub() -> None:
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
    trainer = Trainer(
        model=object(),
        train_ds=object(),  # type: ignore[arg-type]
        val_ds=object(),  # type: ignore[arg-type]
        tracker=NoopTracker(),
        cfg=cfg,
    )
    _assert_stub(trainer.fit)
