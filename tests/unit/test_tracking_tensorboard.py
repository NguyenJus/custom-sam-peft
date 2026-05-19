"""TensorBoardTracker — round-trip event-file read-back."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from esam3.config.schema import (
    DataConfig,
    DataSplit,
    PEFTConfig,
    RunConfig,
    TrainConfig,
    TrainHyperparams,
)

pytest.importorskip("tensorboard")

from esam3.tracking.tensorboard import TensorBoardTracker


def _cfg(tmp_path: Path) -> TrainConfig:
    return TrainConfig(
        run=RunConfig(name="t", output_dir=str(tmp_path), seed=0),
        data=DataConfig(
            format="coco",
            train=DataSplit(annotations="x.json", images="x/"),
            val=DataSplit(annotations="x.json", images="x/"),
            prompt_mode="text",
            image_size=32,
        ),
        peft=PEFTConfig(method="lora", scope="vision"),
        train=TrainHyperparams(
            epochs=1,
            batch_size=1,
            grad_accum_steps=1,
            save_every=2,
            log_every=1,
            warmup_steps=0,
            num_workers=0,
        ),
    )


def _read_back(log_dir: Path):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    acc = EventAccumulator(str(log_dir))
    acc.Reload()
    return acc


def test_tb_round_trip(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    t = TensorBoardTracker(cfg)
    t.start_run(run_dir, {"a": 1, "b": "two"})
    t.log_scalars(0, {"loss": 0.5})
    t.log_scalars(1, {"loss": 0.25})
    t.log_images(0, {"panel": np.zeros((4, 4, 3), dtype=np.uint8)})
    t.close()

    acc = _read_back(run_dir)
    tags = acc.Tags()
    assert "loss" in tags["scalars"]
    assert "panel" in tags["images"]
    assert any(t.startswith("config") for t in tags["tensors"])
    scalars = acc.Scalars("loss")
    assert [(s.step, s.value) for s in scalars] == [(0, 0.5), (1, 0.25)]


def test_tb_drops_non_finite_scalars(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    t = TensorBoardTracker(cfg)
    t.start_run(run_dir, {})
    t.log_scalars(0, {"loss": 0.5, "nan": math.nan, "pinf": math.inf, "ninf": -math.inf})
    t.close()

    acc = _read_back(run_dir)
    assert acc.Tags()["scalars"] == ["loss"]


def test_tb_rejects_wrong_image_shape(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    t = TensorBoardTracker(cfg)
    t.start_run(run_dir, {})
    with pytest.raises(ValueError, match=r"panel.*uint8"):
        t.log_images(0, {"panel": np.zeros((4, 4), dtype=np.uint8)})
    with pytest.raises(ValueError, match=r"panel.*uint8"):
        t.log_images(0, {"panel": np.zeros((4, 4, 3), dtype=np.float32)})
    t.close()


def test_tb_close_is_idempotent(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    t = TensorBoardTracker(cfg)
    t.start_run(run_dir, {})
    t.close()
    t.close()  # must not raise


def test_tb_log_before_start_run_raises(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    t = TensorBoardTracker(cfg)
    with pytest.raises(RuntimeError, match="start_run"):
        t.log_scalars(0, {"loss": 1.0})
