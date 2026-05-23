"""_build_dataset limit wrapping + subset.json write."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from custom_sam_peft.data.subset import SubsetDataset
from custom_sam_peft.train.runner import _build_dataset, run_training


def _make_stub_inner(size: int = 10) -> MagicMock:
    inner = MagicMock()
    inner.__len__ = MagicMock(return_value=size)
    inner.class_names = ["a", "b"]
    return inner


def _make_cfg(tmp_path: Path, train_limit=None, val_limit=None, strategy="random") -> MagicMock:
    cfg = MagicMock()
    cfg.run.output_dir = str(tmp_path)
    cfg.run.name = "smoke"
    cfg.run.seed = 0
    cfg.data.format = "coco"
    cfg.data.model_dump.return_value = {"format": "coco"}
    cfg.model.name = "facebook/sam3.1"
    cfg.peft.method = "lora"
    cfg.tracking.backend = "none"
    cfg.tracking.wandb.project = "custom_sam_peft"
    cfg.tracking.wandb.entity = None
    cfg.data.limit.train = train_limit
    cfg.data.limit.val = val_limit
    cfg.data.limit.seed = 42
    cfg.data.limit.strategy = strategy
    # spec/data-no-val-auto-split (#71): the runner consults the val source resolver
    # before building datasets. Force the "explicit" mode (cfg.data.val present,
    # val_split absent) so MagicMock auto-attrs don't trip the auto-split path.
    cfg.data.val = MagicMock()
    cfg.data.val_split = None
    return cfg


def test_no_limit_returns_inner_dataset_directly() -> None:
    inner = _make_stub_inner(10)

    cfg = MagicMock()
    cfg.data.format = "coco"
    cfg.data.model_dump.return_value = {}
    cfg.model.name = "n/a"
    cfg.data.limit.train = None
    cfg.data.limit.val = None

    with patch("custom_sam_peft.train.runner.lookup", return_value=lambda *a, **kw: inner):
        ds = _build_dataset(cfg, "train")

    assert ds is inner  # no wrapping


def test_train_limit_returns_subset_dataset() -> None:
    inner = _make_stub_inner(10)

    cfg = MagicMock()
    cfg.data.format = "coco"
    cfg.data.model_dump.return_value = {}
    cfg.model.name = "n/a"
    cfg.data.limit.train = 3
    cfg.data.limit.val = None
    cfg.data.limit.seed = 0
    cfg.data.limit.strategy = "random"

    with patch("custom_sam_peft.train.runner.lookup", return_value=lambda *a, **kw: inner):
        ds = _build_dataset(cfg, "train")

    assert isinstance(ds, SubsetDataset)
    assert len(ds) == 3


def test_val_limit_only_wraps_val() -> None:
    inner = _make_stub_inner(10)

    cfg = MagicMock()
    cfg.data.format = "coco"
    cfg.data.model_dump.return_value = {}
    cfg.model.name = "n/a"
    cfg.data.limit.train = None
    cfg.data.limit.val = 4
    cfg.data.limit.seed = 0
    cfg.data.limit.strategy = "first_n"

    with patch("custom_sam_peft.train.runner.lookup", return_value=lambda *a, **kw: inner):
        train_ds = _build_dataset(cfg, "train")
        val_ds = _build_dataset(cfg, "eval")

    assert train_ds is inner
    assert isinstance(val_ds, SubsetDataset)
    assert len(val_ds) == 4


def test_info_log_emitted_when_limit_applied(
    caplog: pytest.LogCaptureFixture,
) -> None:
    inner = _make_stub_inner(10)

    cfg = MagicMock()
    cfg.data.format = "coco"
    cfg.data.model_dump.return_value = {}
    cfg.model.name = "n/a"
    cfg.data.limit.train = 3
    cfg.data.limit.val = None
    cfg.data.limit.seed = 42
    cfg.data.limit.strategy = "random"

    with (
        patch("custom_sam_peft.train.runner.lookup", return_value=lambda *a, **kw: inner),
        caplog.at_level(logging.INFO, logger="custom_sam_peft.train.runner"),
    ):
        _build_dataset(cfg, "train")

    assert any("data.limit applied" in r.message for r in caplog.records)


def test_stratified_strategy_accesses_image_class_labels() -> None:
    inner = _make_stub_inner(10)
    accessed = []
    type(inner).image_class_labels = property(
        lambda self: accessed.append(True) or [frozenset([0])] * 10
    )

    cfg = MagicMock()
    cfg.data.format = "coco"
    cfg.data.model_dump.return_value = {}
    cfg.model.name = "n/a"
    cfg.data.limit.train = 3
    cfg.data.limit.val = None
    cfg.data.limit.seed = 0
    cfg.data.limit.strategy = "stratified"

    with patch("custom_sam_peft.train.runner.lookup", return_value=lambda *a, **kw: inner):
        _build_dataset(cfg, "train")

    assert accessed  # image_class_labels was accessed


def test_random_strategy_does_not_access_image_class_labels() -> None:
    inner = _make_stub_inner(10)
    accessed = []
    type(inner).image_class_labels = property(
        lambda self: accessed.append(True) or [frozenset([0])] * 10
    )

    cfg = MagicMock()
    cfg.data.format = "coco"
    cfg.data.model_dump.return_value = {}
    cfg.model.name = "n/a"
    cfg.data.limit.train = 3
    cfg.data.limit.val = None
    cfg.data.limit.seed = 0
    cfg.data.limit.strategy = "random"

    with patch("custom_sam_peft.train.runner.lookup", return_value=lambda *a, **kw: inner):
        _build_dataset(cfg, "train")

    assert accessed == []


def test_subset_json_written_when_at_least_one_limit_set(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path, train_limit=3, val_limit=2)

    def fake_lookup(kind: str, name: str):
        if kind == "peft":
            return lambda wrapper, _peft_cfg: wrapper
        return lambda *a, **kw: _make_stub_inner(10)

    with (
        patch("custom_sam_peft.train.runner.lookup", side_effect=fake_lookup),
        patch("custom_sam_peft.train.runner.load_sam31", return_value=MagicMock()),
        patch(
            "custom_sam_peft.train.runner.build_tracker",
            return_value=MagicMock(close=MagicMock(), start_run=MagicMock()),
        ),
        patch("custom_sam_peft.train.runner.Trainer.fit", return_value=MagicMock()),
    ):
        run_training(cfg)

    # Find the run_dir that was created
    run_dirs = list(tmp_path.iterdir())
    assert len(run_dirs) == 1
    subset_json = run_dirs[0] / "subset.json"
    assert subset_json.exists()
    data = json.loads(subset_json.read_text())
    assert "limit" in data
    assert "train" in data
    assert "val" in data
    assert data["train"]["n_kept"] == 3
    assert data["val"]["n_kept"] == 2


def test_subset_json_not_written_when_both_limits_none(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path, train_limit=None, val_limit=None)

    def fake_lookup(kind: str, name: str):
        if kind == "peft":
            return lambda wrapper, _peft_cfg: wrapper
        return lambda *a, **kw: _make_stub_inner(10)

    with (
        patch("custom_sam_peft.train.runner.lookup", side_effect=fake_lookup),
        patch("custom_sam_peft.train.runner.load_sam31", return_value=MagicMock()),
        patch(
            "custom_sam_peft.train.runner.build_tracker",
            return_value=MagicMock(close=MagicMock(), start_run=MagicMock()),
        ),
        patch("custom_sam_peft.train.runner.Trainer.fit", return_value=MagicMock()),
    ):
        run_training(cfg)

    run_dirs = list(tmp_path.iterdir())
    assert len(run_dirs) == 1
    assert not (run_dirs[0] / "subset.json").exists()
