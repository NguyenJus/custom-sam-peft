"""Verify every stub raises NotImplementedError with a spec: reference."""

from __future__ import annotations

import pytest
import torch

from esam3.config.schema import (
    AugmentationsConfig,
    EvalConfig,
    ModelConfig,
    PEFTConfig,
)
from esam3.data.coco import COCODataset
from esam3.data.collate import collate_batch
from esam3.data.hf import HFDataset
from esam3.data.transforms import build_eval_transforms, build_train_transforms
from esam3.eval.evaluator import Evaluator
from esam3.eval.metrics import compute_coco_map
from esam3.models.losses import box_loss, mask_loss, objectness_loss, total_loss
from esam3.models.sam3 import load_sam31
from esam3.peft_adapters.lora import apply_lora
from esam3.peft_adapters.qlora import apply_qlora
from esam3.train.checkpoint import load_adapter, save_adapter, save_merged
from esam3.train.loop import run_epoch


def _assert_stub(call: object) -> None:
    with pytest.raises(NotImplementedError, match="filled in by spec:"):
        call()  # type: ignore[operator]


def test_data_stubs() -> None:
    coco = COCODataset("a", "b", "bbox")
    _assert_stub(lambda: coco.__len__())
    _assert_stub(lambda: coco.__getitem__(0))
    _assert_stub(lambda: coco.class_names)
    hf = HFDataset("a", "train", "text")
    _assert_stub(lambda: hf.__len__())
    _assert_stub(lambda: hf.__getitem__(0))
    _assert_stub(lambda: hf.class_names)
    _assert_stub(lambda: build_train_transforms(AugmentationsConfig(), 1024))
    _assert_stub(lambda: build_eval_transforms(1024))
    _assert_stub(lambda: collate_batch([]))


def test_model_stubs() -> None:
    _assert_stub(lambda: load_sam31(ModelConfig()))
    t = torch.zeros((1,))
    _assert_stub(lambda: mask_loss(t, t))
    _assert_stub(lambda: box_loss(t, t))
    _assert_stub(lambda: objectness_loss(t, t))
    _assert_stub(lambda: total_loss({}, {}))


def test_peft_stubs() -> None:
    cfg = PEFTConfig(method="lora")
    _assert_stub(lambda: apply_lora(object(), cfg))
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
