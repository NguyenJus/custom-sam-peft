"""run_training composes registry dataset/peft/tracker calls and Trainer.fit."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from custom_sam_peft.train.runner import make_run_dir, run_training


def _make_cfg(tmp_path: Path) -> MagicMock:
    cfg = MagicMock()
    cfg.run.output_dir = str(tmp_path)
    cfg.run.name = "smoke"
    cfg.run.seed = 0
    cfg.data.format = "coco"
    cfg.data.prompt_mode = "text"
    cfg.data.model_dump.return_value = {"format": "coco"}
    cfg.model.name = "facebook/sam3.1"
    cfg.peft.method = "lora"
    cfg.tracking.backend = "none"
    cfg.tracking.wandb.project = "custom_sam_peft"
    cfg.tracking.wandb.entity = None
    return cfg


def test_make_run_dir_creates_timestamped_subdir(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path)
    run_dir = make_run_dir(cfg)
    assert run_dir.parent == tmp_path
    assert run_dir.name.startswith("smoke-")
    assert run_dir.exists()
    stamp = run_dir.name.split("-", 1)[1]
    datetime.strptime(stamp, "%Y%m%d-%H%M%S").replace(tzinfo=UTC)


def test_run_training_dispatches_via_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _make_cfg(tmp_path)
    calls: list[tuple[str, str]] = []

    def fake_lookup(kind: str, name: str) -> object:
        calls.append((kind, name))
        if kind == "peft":
            return lambda wrapper, _peft_cfg: wrapper
        return lambda *a, **kw: MagicMock(__len__=lambda self: 0, class_names=[])

    monkeypatch.setattr("custom_sam_peft.train.runner.lookup", fake_lookup)
    monkeypatch.setattr("custom_sam_peft.train.runner.load_sam31", lambda _m: MagicMock())
    monkeypatch.setattr(
        "custom_sam_peft.train.runner.build_tracker",
        lambda _cfg: MagicMock(close=MagicMock(), start_run=MagicMock()),
    )

    fake_result = MagicMock()

    def fake_fit(self, *, run_dir, resume_from=None):
        return fake_result

    monkeypatch.setattr("custom_sam_peft.train.runner.Trainer.fit", fake_fit)

    result = run_training(cfg)
    assert result is fake_result
    kinds = {k for k, _ in calls}
    assert kinds == {"dataset", "peft"}
