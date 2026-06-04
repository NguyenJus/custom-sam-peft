"""Tests for keep-last-N checkpoint retention (#316).

Covers the _prune_old_checkpoints helper directly (fast, no real model needed)
plus one integration assertion via save_full_state to verify the full write path.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest
import torch

from custom_sam_peft.config.schema import TrainConfig
from custom_sam_peft.train.checkpoint import (
    _prune_old_checkpoints,
    find_latest_checkpoint,
    save_full_state,
)
from tests.fixtures.tiny_sam3_lora_stub import make_stub_wrapper
from tests.integration.test_trainer_evaluator_seam import _make_cfg

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_step_dirs(base: Path, steps: list[int]) -> list[Path]:
    """Create step_<N> directories under *base* and return their paths."""
    dirs: list[Path] = []
    for s in steps:
        d = base / f"step_{s}"
        d.mkdir(parents=True, exist_ok=True)
        dirs.append(d)
    return dirs


# ---------------------------------------------------------------------------
# Unit tests for _prune_old_checkpoints
# ---------------------------------------------------------------------------


def test_keep_last_1_leaves_only_newest(tmp_path: Path) -> None:
    """keep_last_checkpoints=1 -> only the highest-numbered step_N survives."""
    _make_step_dirs(tmp_path, [10, 20, 30])
    _prune_old_checkpoints(tmp_path, keep=1)
    remaining = {d.name for d in tmp_path.iterdir() if d.is_dir()}
    assert remaining == {"step_30"}


def test_keep_last_2_leaves_two_newest(tmp_path: Path) -> None:
    """keep_last_checkpoints=2 -> the two highest-numbered step_N dirs survive."""
    _make_step_dirs(tmp_path, [5, 10, 15, 20])
    _prune_old_checkpoints(tmp_path, keep=2)
    remaining = {d.name for d in tmp_path.iterdir() if d.is_dir()}
    assert remaining == {"step_15", "step_20"}


def test_keep_last_none_skips_prune(tmp_path: Path) -> None:
    """When N is None, _prune_old_checkpoints is never called (caller responsibility),
    but we can also verify that passing a large keep value leaves everything intact."""
    # Simulate the None path: the caller skips _prune_old_checkpoints entirely.
    # Here we confirm the helper itself is idempotent when keep >= total dirs.
    _make_step_dirs(tmp_path, [1, 2, 3])
    _prune_old_checkpoints(tmp_path, keep=100)
    remaining = {d.name for d in tmp_path.iterdir() if d.is_dir()}
    assert remaining == {"step_1", "step_2", "step_3"}


def test_non_step_dirs_are_not_pruned(tmp_path: Path) -> None:
    """Directories that don't parse as step_<int> (including best/) are never removed."""
    _make_step_dirs(tmp_path, [10, 20])
    (tmp_path / "best").mkdir()
    (tmp_path / "garbage").mkdir()
    _prune_old_checkpoints(tmp_path, keep=1)
    remaining = {d.name for d in tmp_path.iterdir() if d.is_dir()}
    # step_20 survives (newest); step_10 pruned; best + garbage untouched.
    assert "best" in remaining
    assert "garbage" in remaining
    assert "step_20" in remaining
    assert "step_10" not in remaining


def test_malformed_step_dirs_are_not_pruned(tmp_path: Path) -> None:
    """Dirs named step_, step_abc, or bare garbage are left untouched."""
    _make_step_dirs(tmp_path, [100])
    (tmp_path / "step_").mkdir()
    (tmp_path / "step_abc").mkdir()
    _prune_old_checkpoints(tmp_path, keep=1)
    remaining = {d.name for d in tmp_path.iterdir() if d.is_dir()}
    assert "step_" in remaining
    assert "step_abc" in remaining
    assert "step_100" in remaining  # only valid step dir, so it's kept


def test_rmtree_failure_warns_and_does_not_raise(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A single rmtree failure emits a warning and does NOT propagate an exception."""
    _make_step_dirs(tmp_path, [10, 20, 30])

    import shutil

    original_rmtree = shutil.rmtree
    removed: list[str] = []

    def _selective_rmtree(path: Any, *args: Any, **kwargs: Any) -> None:
        p = Path(path)
        if p.name == "step_10":
            raise OSError("simulated disk error")
        original_rmtree(p, *args, **kwargs)
        removed.append(p.name)

    monkeypatch.setattr(shutil, "rmtree", _selective_rmtree)

    with caplog.at_level(logging.WARNING, logger="custom_sam_peft.train.checkpoint"):
        _prune_old_checkpoints(tmp_path, keep=1)  # must not raise

    # The warning must mention the failing path.
    assert any("step_10" in rec.message for rec in caplog.records)
    # step_30 survived (kept); step_20 was pruned successfully.
    remaining = {d.name for d in tmp_path.iterdir() if d.is_dir()}
    assert "step_30" in remaining
    assert "step_20" not in remaining
    # step_10 was NOT deleted due to the simulated error.
    assert "step_10" in remaining


# ---------------------------------------------------------------------------
# Integration: save_full_state respects keep_last_checkpoints
# ---------------------------------------------------------------------------


def _make_checkpoint_cfg(tmp_path: Path, keep: int | None) -> TrainConfig:
    """Build a minimal TrainConfig with the given keep_last_checkpoints value."""
    cfg = _make_cfg(tmp_path)
    return cfg.model_copy(
        update={"train": cfg.train.model_copy(update={"keep_last_checkpoints": keep})}
    )


def test_save_full_state_prunes_via_keep_last(tmp_path: Path) -> None:
    """save_full_state with keep_last_checkpoints=1 leaves only the newest step dir."""
    from custom_sam_peft.peft_adapters.lora import apply_lora

    cfg = _make_checkpoint_cfg(tmp_path, keep=1)
    wrapper = make_stub_wrapper(dim=8, working=True)
    apply_lora(wrapper, cfg.peft)
    opt: torch.optim.Optimizer = torch.optim.AdamW(
        [p for p in wrapper.parameters() if p.requires_grad], lr=1e-4
    )
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max")

    checkpoints_dir = tmp_path / "checkpoints"

    # Write step_10 first (simulates an earlier checkpoint sitting on disk).
    earlier = checkpoints_dir / "step_10"
    earlier.mkdir(parents=True, exist_ok=True)
    (earlier / "dummy").touch()

    # Write step_20 via save_full_state -> should prune step_10.
    save_full_state(
        state_dir=checkpoints_dir / "step_20",
        wrapper=wrapper,
        optimizer=opt,
        scheduler=sched,
        global_step=20,
        epoch=0,
        nan_streak=0,
        cfg=cfg,
    )

    remaining = {d.name for d in checkpoints_dir.iterdir() if d.is_dir()}
    assert "step_20" in remaining
    assert "step_10" not in remaining


def test_save_full_state_none_keep_does_not_prune(tmp_path: Path) -> None:
    """save_full_state with keep_last_checkpoints=None leaves all step dirs intact."""
    from custom_sam_peft.peft_adapters.lora import apply_lora

    cfg = _make_checkpoint_cfg(tmp_path, keep=None)
    wrapper = make_stub_wrapper(dim=8, working=True)
    apply_lora(wrapper, cfg.peft)
    opt: torch.optim.Optimizer = torch.optim.AdamW(
        [p for p in wrapper.parameters() if p.requires_grad], lr=1e-4
    )
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max")

    checkpoints_dir = tmp_path / "checkpoints"

    earlier = checkpoints_dir / "step_5"
    earlier.mkdir(parents=True, exist_ok=True)
    (earlier / "dummy").touch()

    save_full_state(
        state_dir=checkpoints_dir / "step_10",
        wrapper=wrapper,
        optimizer=opt,
        scheduler=sched,
        global_step=10,
        epoch=0,
        nan_streak=0,
        cfg=cfg,
    )

    remaining = {d.name for d in checkpoints_dir.iterdir() if d.is_dir()}
    assert "step_5" in remaining
    assert "step_10" in remaining


def test_find_latest_checkpoint_after_pruning(tmp_path: Path) -> None:
    """After pruning, find_latest_checkpoint still returns the newest surviving step."""
    from custom_sam_peft.config.schema import RunConfig

    cfg = _make_cfg(tmp_path)
    run_name = cfg.run.name
    run_dir = Path(cfg.run.output_dir) / f"{run_name}-2026-01-01T000000"
    checkpoints_dir = run_dir / "checkpoints"

    # Create step_10, step_20, step_30 then prune to keep=1.
    _make_step_dirs(checkpoints_dir, [10, 20, 30])
    # Manually add a dummy training_state.pt so find_latest_checkpoint does not
    # care about it — the function only checks for dir existence.
    _prune_old_checkpoints(checkpoints_dir, keep=1)

    # find_latest_checkpoint needs the config's output_dir + run name to match.
    cfg2 = cfg.model_copy(
        update={
            "run": RunConfig(
                name=run_name,
                output_dir=str(Path(cfg.run.output_dir)),
                seed=0,
            )
        }
    )
    latest = find_latest_checkpoint(cfg2)
    assert latest.name == "step_30"
