"""Tests for the `csp run` CLI option surface (CPU-only)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

import custom_sam_peft.cli.run_cmd as run_cmd


def test_orchestrate_threads_visualize_into_eval_run_eval(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """_orchestrate forwards its visualize kwarg to the eval-phase run_eval call."""
    captured: dict[str, object] = {}

    # Stub run_training
    fake_result = MagicMock()
    fake_result.run_dir = tmp_path
    fake_result.checkpoint_path = tmp_path / "adapter"
    fake_result.oom_events = []
    fake_result.time_limit_stop = None
    monkeypatch.setattr(run_cmd, "run_training", lambda cfg, resume_from=None: fake_result)

    # Stub load_val_source (imported inside _orchestrate via local import)
    vs = MagicMock()
    vs.mode = "auto_split"
    vs.val_ids = [1, 2]
    monkeypatch.setattr(
        "custom_sam_peft.data.val_source.load_val_source",
        lambda rd: vs,
    )

    # Stub load_sam31, load_adapter
    monkeypatch.setattr(run_cmd, "load_sam31", lambda *a, **k: MagicMock())
    monkeypatch.setattr(run_cmd, "load_adapter", lambda *a, **k: None)

    # Stub _build_val_dataset
    monkeypatch.setattr(run_cmd, "_build_val_dataset", lambda cfg, vs: MagicMock())

    # Stub progress_session (context manager)
    from contextlib import contextmanager

    @contextmanager
    def _noop_progress(**kw):
        yield

    monkeypatch.setattr(run_cmd, "progress_session", _noop_progress)

    # Stub write_bundle
    monkeypatch.setattr(run_cmd, "write_bundle", lambda *a, **k: None)

    # Stub _load_preset_or_fallback
    monkeypatch.setattr(run_cmd, "_load_preset_or_fallback", lambda cfg: MagicMock())

    # Stub rprint to silence output
    monkeypatch.setattr(run_cmd, "rprint", lambda *a, **k: None)

    # Stub cfg.export.merge = False to skip export-merge phase
    cfg = MagicMock()
    cfg.train.epochs = 1
    cfg.export.merge = False

    # The key stub: capture kwargs passed to run_eval
    def _fake_run_eval(cfg, **kw):
        captured.update(kw)
        return MagicMock(overall={}), [0.5]

    monkeypatch.setattr(run_cmd, "run_eval", _fake_run_eval)

    run_cmd._orchestrate(
        cfg, None, run_cmd.ProgressMode.OFF, visualize=False, config_path=tmp_path / "config.yaml"
    )

    assert captured.get("visualize") is False
