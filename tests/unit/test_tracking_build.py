"""Tests for tracking.build_tracker — backend dispatch + missing-extra surface."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from custom_sam_peft.config.schema import (
    DataConfig,
    DataSplit,
    PEFTConfig,
    RunConfig,
    TrackingConfig,
    TrainConfig,
    TrainHyperparams,
    WandbConfig,
)


def _cfg(tmp_path: Path, backend: str = "none") -> TrainConfig:
    return TrainConfig(
        run=RunConfig(name="t", output_dir=str(tmp_path), seed=0),
        data=DataConfig(
            format="coco",
            train=DataSplit(annotations="x.json", images="x/"),
            val=DataSplit(annotations="x.json", images="x/"),
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
        tracking=TrackingConfig(
            backend=backend,  # type: ignore[arg-type]
            wandb=WandbConfig(project="p", entity=None),
        ),
    )


def test_build_tracker_returns_noop(tmp_path: Path) -> None:
    from custom_sam_peft.tracking import build_tracker

    t = build_tracker(_cfg(tmp_path, "none"))
    assert type(t).__name__ == "NoopTracker"
    assert t.wants_images is False


def test_build_tracker_returns_tensorboard(tmp_path: Path) -> None:
    pytest.importorskip("tensorboard")
    from custom_sam_peft.tracking import build_tracker

    t = build_tracker(_cfg(tmp_path, "tensorboard"))
    assert type(t).__name__ == "TensorBoardTracker"
    assert t.wants_images is True


def test_build_tracker_returns_wandb(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_run = MagicMock()
    fake_run.id = "id"
    fake = types.ModuleType("wandb")
    fake.init = MagicMock(return_value=fake_run)  # type: ignore[attr-defined]
    fake.Image = lambda arr: arr  # type: ignore[attr-defined]
    # The WandBTracker constructor does `import wandb` lazily, so installing
    # the fake in sys.modules is sufficient; no need to reload custom_sam_peft.tracking.wandb.
    monkeypatch.setitem(sys.modules, "wandb", fake)

    from custom_sam_peft.tracking import build_tracker

    t = build_tracker(_cfg(tmp_path, "wandb"))
    assert type(t).__name__ == "WandBTracker"
    assert t.wants_images is True


def test_build_tracker_raises_when_wandb_extra_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "wandb", None)

    from custom_sam_peft.tracking import build_tracker

    with pytest.raises(ImportError, match=r"\[wandb\]"):
        build_tracker(_cfg(tmp_path, "wandb"))


def test_build_tracker_returns_local(tmp_path: Path) -> None:
    from custom_sam_peft.tracking import build_tracker

    t = build_tracker(_cfg(tmp_path, "local"))
    assert type(t).__name__ == "LocalTracker"
    assert t.wants_images is False


def test_tracking_config_default_is_local() -> None:
    from custom_sam_peft.config.schema import TrackingConfig

    assert TrackingConfig().backend == "local"


def test_build_tracker_raises_when_tensorboard_extra_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Force the SummaryWriter import to fail at construction time.
    monkeypatch.setitem(sys.modules, "torch.utils.tensorboard", None)

    from custom_sam_peft.tracking import build_tracker

    with pytest.raises(ImportError, match=r"\[tensorboard\]"):
        build_tracker(_cfg(tmp_path, "tensorboard"))
