"""WandBTracker — fully mocked SDK, including resume-id continuation."""

from __future__ import annotations

import math
import sys
import types
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from esam3.config.schema import (
    DataConfig,
    DataSplit,
    PEFTConfig,
    RunConfig,
    TrackingConfig,
    TrainConfig,
    TrainHyperparams,
    WandbConfig,
)


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
        tracking=TrackingConfig(
            backend="wandb", wandb=WandbConfig(project="esam3-test", entity=None)
        ),
    )


@pytest.fixture
def mock_wandb(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """Install a fake `wandb` module exposing init/Image and a fake Run.

    The constructor and log_images do ``import wandb`` lazily, so simply
    setting sys.modules['wandb'] is enough — no need to reload
    esam3.tracking.wandb (which would re-fire @register and collide).
    """
    fake_run = MagicMock()
    fake_run.id = "fake-run-id-123"
    fake = types.ModuleType("wandb")
    fake.init = MagicMock(return_value=fake_run)  # type: ignore[attr-defined]
    fake.Image = lambda arr: ("WandbImage", arr)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "wandb", fake)
    return fake


def _import_tracker() -> Any:
    from esam3.tracking.wandb import WandBTracker

    return WandBTracker


def test_wb_start_run_calls_init_and_writes_id(tmp_path: Path, mock_wandb: Any) -> None:
    cfg = _cfg(tmp_path)
    run_dir = tmp_path / "myrun"
    run_dir.mkdir()
    WandBTracker = _import_tracker()  # safe to import after the fixture installs the mock
    t = WandBTracker(cfg)
    t.start_run(run_dir, {"x": 1})

    mock_wandb.init.assert_called_once()
    kwargs = mock_wandb.init.call_args.kwargs
    assert kwargs["project"] == "esam3-test"
    assert kwargs["entity"] is None
    assert kwargs["name"] == "myrun"
    assert kwargs["dir"] == str(run_dir)
    assert kwargs["config"] == {"x": 1}
    assert kwargs["id"] is None
    assert kwargs["resume"] is None
    assert (run_dir / "wandb_run_id.txt").read_text() == "fake-run-id-123"


def test_wb_log_scalars_drops_non_finite(tmp_path: Path, mock_wandb: Any) -> None:
    cfg = _cfg(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    WandBTracker = _import_tracker()
    t = WandBTracker(cfg)
    t.start_run(run_dir, {})
    fake_run = mock_wandb.init.return_value
    fake_run.log.reset_mock()
    t.log_scalars(5, {"loss": 0.5, "nan": math.nan, "pinf": math.inf})
    fake_run.log.assert_called_once_with({"loss": 0.5}, step=5)


def test_wb_log_scalars_skips_empty_after_filtering(tmp_path: Path, mock_wandb: Any) -> None:
    cfg = _cfg(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    WandBTracker = _import_tracker()
    t = WandBTracker(cfg)
    t.start_run(run_dir, {})
    fake_run = mock_wandb.init.return_value
    fake_run.log.reset_mock()
    t.log_scalars(5, {"nan": math.nan})
    fake_run.log.assert_not_called()


def test_wb_log_images_wraps_each(tmp_path: Path, mock_wandb: Any) -> None:
    cfg = _cfg(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    WandBTracker = _import_tracker()
    t = WandBTracker(cfg)
    t.start_run(run_dir, {})
    fake_run = mock_wandb.init.return_value
    fake_run.log.reset_mock()
    arr = np.zeros((4, 4, 3), dtype=np.uint8)
    t.log_images(0, {"panel": arr})
    call = fake_run.log.call_args
    assert call.kwargs == {"step": 0}
    payload = call.args[0]
    assert list(payload.keys()) == ["panel"]
    tag, wrapped = payload["panel"]
    assert tag == "WandbImage"
    assert wrapped is arr


def test_wb_log_images_rejects_wrong_dtype(tmp_path: Path, mock_wandb: Any) -> None:
    cfg = _cfg(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    WandBTracker = _import_tracker()
    t = WandBTracker(cfg)
    t.start_run(run_dir, {})
    with pytest.raises(ValueError, match=r"panel.*uint8"):
        t.log_images(0, {"panel": np.zeros((4, 4, 3), dtype=np.float32)})


def test_wb_resumes_when_id_file_present(tmp_path: Path, mock_wandb: Any) -> None:
    cfg = _cfg(tmp_path)
    prior_run_dir = tmp_path / "prior"
    prior_ckpt = prior_run_dir / "checkpoints" / "step_100"
    prior_ckpt.mkdir(parents=True)
    (prior_run_dir / "wandb_run_id.txt").write_text("old-run-id")

    new_run_dir = tmp_path / "new"
    new_run_dir.mkdir()

    WandBTracker = _import_tracker()
    t = WandBTracker(cfg)
    t.start_run(new_run_dir, {}, resume_from=prior_ckpt)

    kwargs = mock_wandb.init.call_args.kwargs
    assert kwargs["id"] == "old-run-id"
    assert kwargs["resume"] == "allow"
    assert (new_run_dir / "wandb_run_id.txt").read_text() == "fake-run-id-123"


def test_wb_starts_fresh_when_id_file_missing(tmp_path: Path, mock_wandb: Any) -> None:
    cfg = _cfg(tmp_path)
    prior_ckpt = tmp_path / "prior" / "checkpoints" / "step_100"
    prior_ckpt.mkdir(parents=True)
    # No wandb_run_id.txt written anywhere.

    new_run_dir = tmp_path / "new"
    new_run_dir.mkdir()

    WandBTracker = _import_tracker()
    t = WandBTracker(cfg)
    t.start_run(new_run_dir, {}, resume_from=prior_ckpt)

    kwargs = mock_wandb.init.call_args.kwargs
    assert kwargs["id"] is None
    assert kwargs["resume"] is None


def test_wb_close_is_idempotent(tmp_path: Path, mock_wandb: Any) -> None:
    cfg = _cfg(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    WandBTracker = _import_tracker()
    t = WandBTracker(cfg)
    t.start_run(run_dir, {})
    t.close()
    t.close()  # must not raise
    fake_run = mock_wandb.init.return_value
    fake_run.finish.assert_called_once()
