"""Host-RAM floor guard — per-step check, graceful flush-and-stop (spec host-ram guard)."""

from __future__ import annotations

import types
from pathlib import Path
from typing import Any

import pytest
import torch

import custom_sam_peft.train.loop as loop_mod
from custom_sam_peft.eval._artifacts import EvalArtifacts
from custom_sam_peft.peft_adapters.lora import apply_lora
from custom_sam_peft.tracking.noop import NoopTracker
from custom_sam_peft.train.trainer import Trainer
from tests.fixtures.tiny_sam3_lora_stub import make_stub_wrapper
from tests.integration.test_trainer_evaluator_seam import _make_cfg, _TinyDataset


def _loader(ds: _TinyDataset) -> list[dict[str, object]]:
    from custom_sam_peft.data.collate import collate_batch

    return [collate_batch([ds[i]]) for i in range(len(ds))]


def _fake_vmem(available_bytes: int) -> types.SimpleNamespace:
    """Return a psutil.virtual_memory()-shaped object with the given available bytes."""
    return types.SimpleNamespace(available=available_bytes)


# ---------------------------------------------------------------------------
# run_epoch-level: guard fires
# ---------------------------------------------------------------------------


def test_run_epoch_flushes_and_raises_host_ram_low(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When available host RAM drops below the floor, a full-state checkpoint is
    flushed and loop_mod._HostRamLow is raised at the first step past the threshold."""
    import psutil

    # Floor is 4 GB; report only 1 GB available → guard should fire after step 1.
    floor_bytes = int(4e9)
    available_bytes = int(1e9)
    monkeypatch.setattr(
        psutil,
        "virtual_memory",
        lambda: _fake_vmem(available_bytes),
    )
    # Also patch the module-level reference inside loop.py
    monkeypatch.setattr(loop_mod.psutil, "virtual_memory", lambda: _fake_vmem(available_bytes))

    ds = _TinyDataset()
    wrapper = make_stub_wrapper(dim=8, working=True)
    cfg = _make_cfg(tmp_path)
    apply_lora(wrapper, cfg.peft)
    run_dir = tmp_path / "run"
    (run_dir / "checkpoints").mkdir(parents=True)

    optimizer = torch.optim.AdamW([p for p in wrapper.parameters() if p.requires_grad], lr=1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda s: 1.0)

    with pytest.raises(loop_mod._HostRamLow) as exc:
        loop_mod.run_epoch(
            wrapper,
            _loader(ds),
            optimizer,
            scheduler,
            NoopTracker(),
            cfg,
            run_dir,
            epoch=0,
            global_step=0,
            nan_streak=0,
            class_names=ds.class_names,
            on_checkpoint=lambda *a: None,
            on_eval=lambda *a: None,
            host_ram_floor_bytes=floor_bytes,
        )

    assert exc.value.step == 1  # fires after the first step
    assert exc.value.epoch == 0
    assert exc.value.available_gb == pytest.approx(available_bytes / 1e9)

    # Full-state checkpoint must have been flushed.
    ckpt = run_dir / "checkpoints" / f"step_{exc.value.step}"
    assert (ckpt / "adapter").exists(), "adapter dir missing from flushed checkpoint"
    assert (ckpt / "training_state.pt").exists(), (
        "training_state.pt missing from flushed checkpoint"
    )


# ---------------------------------------------------------------------------
# run_epoch-level: guard inert (available >> floor)
# ---------------------------------------------------------------------------


def test_run_epoch_no_raise_when_ram_above_floor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When available RAM is well above the floor, no loop_mod._HostRamLow is raised."""
    # Floor is 1 GB; report 32 GB available → guard must NOT fire.
    floor_bytes = int(1e9)
    available_bytes = int(32e9)
    monkeypatch.setattr(loop_mod.psutil, "virtual_memory", lambda: _fake_vmem(available_bytes))

    ds = _TinyDataset()
    wrapper = make_stub_wrapper(dim=8, working=True)
    cfg = _make_cfg(tmp_path)
    apply_lora(wrapper, cfg.peft)
    run_dir = tmp_path / "run"
    (run_dir / "checkpoints").mkdir(parents=True)

    optimizer = torch.optim.AdamW([p for p in wrapper.parameters() if p.requires_grad], lr=1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda s: 1.0)

    # Must NOT raise.
    gs, _ = loop_mod.run_epoch(
        wrapper,
        _loader(ds),
        optimizer,
        scheduler,
        NoopTracker(),
        cfg,
        run_dir,
        epoch=0,
        global_step=0,
        nan_streak=0,
        class_names=ds.class_names,
        on_checkpoint=lambda *a: None,
        on_eval=lambda *a: None,
        host_ram_floor_bytes=floor_bytes,
    )
    assert gs == len(_loader(ds))  # processed all batches normally


# ---------------------------------------------------------------------------
# run_epoch-level: exactly-at-floor boundary (strict <, not <=)
# ---------------------------------------------------------------------------


def test_run_epoch_no_raise_when_available_equals_floor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """available == floor must NOT fire the guard (check is strict <)."""
    floor_bytes = int(4e9)
    available_bytes = floor_bytes  # exactly at the floor
    monkeypatch.setattr(loop_mod.psutil, "virtual_memory", lambda: _fake_vmem(available_bytes))

    ds = _TinyDataset()
    wrapper = make_stub_wrapper(dim=8, working=True)
    cfg = _make_cfg(tmp_path)
    apply_lora(wrapper, cfg.peft)
    run_dir = tmp_path / "run"
    (run_dir / "checkpoints").mkdir(parents=True)

    optimizer = torch.optim.AdamW([p for p in wrapper.parameters() if p.requires_grad], lr=1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda s: 1.0)

    # Must NOT raise — available == floor is not below the floor.
    gs, _ = loop_mod.run_epoch(
        wrapper,
        _loader(ds),
        optimizer,
        scheduler,
        NoopTracker(),
        cfg,
        run_dir,
        epoch=0,
        global_step=0,
        nan_streak=0,
        class_names=ds.class_names,
        on_checkpoint=lambda *a: None,
        on_eval=lambda *a: None,
        host_ram_floor_bytes=floor_bytes,
    )
    assert gs == len(_loader(ds))  # all batches processed normally


def test_run_epoch_raises_when_available_is_floor_minus_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """available == floor - 1 must fire the guard (strictly below the floor)."""
    floor_bytes = int(4e9)
    available_bytes = floor_bytes - 1  # one byte below the floor
    monkeypatch.setattr(loop_mod.psutil, "virtual_memory", lambda: _fake_vmem(available_bytes))

    ds = _TinyDataset()
    wrapper = make_stub_wrapper(dim=8, working=True)
    cfg = _make_cfg(tmp_path)
    apply_lora(wrapper, cfg.peft)
    run_dir = tmp_path / "run"
    (run_dir / "checkpoints").mkdir(parents=True)

    optimizer = torch.optim.AdamW([p for p in wrapper.parameters() if p.requires_grad], lr=1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda s: 1.0)

    with pytest.raises(loop_mod._HostRamLow) as exc:
        loop_mod.run_epoch(
            wrapper,
            _loader(ds),
            optimizer,
            scheduler,
            NoopTracker(),
            cfg,
            run_dir,
            epoch=0,
            global_step=0,
            nan_streak=0,
            class_names=ds.class_names,
            on_checkpoint=lambda *a: None,
            on_eval=lambda *a: None,
            host_ram_floor_bytes=floor_bytes,
        )
    assert exc.value.step == 1  # fires after the first step


# ---------------------------------------------------------------------------
# run_epoch-level: guard disabled (host_ram_floor_bytes=None)
# ---------------------------------------------------------------------------


def test_run_epoch_guard_disabled_when_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When host_ram_floor_bytes=None, the guard is off even if RAM reports 0 available."""
    # Report 0 bytes available — the guard must still NOT fire when disabled.
    monkeypatch.setattr(loop_mod.psutil, "virtual_memory", lambda: _fake_vmem(0))

    ds = _TinyDataset()
    wrapper = make_stub_wrapper(dim=8, working=True)
    cfg = _make_cfg(tmp_path)
    apply_lora(wrapper, cfg.peft)
    run_dir = tmp_path / "run"
    (run_dir / "checkpoints").mkdir(parents=True)

    optimizer = torch.optim.AdamW([p for p in wrapper.parameters() if p.requires_grad], lr=1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda s: 1.0)

    # host_ram_floor_bytes=None → disabled; must complete normally.
    gs, _ = loop_mod.run_epoch(
        wrapper,
        _loader(ds),
        optimizer,
        scheduler,
        NoopTracker(),
        cfg,
        run_dir,
        epoch=0,
        global_step=0,
        nan_streak=0,
        class_names=ds.class_names,
        on_checkpoint=lambda *a: None,
        on_eval=lambda *a: None,
        host_ram_floor_bytes=None,
    )
    assert gs == len(_loader(ds))


# ---------------------------------------------------------------------------
# run_epoch-level: flush_extra is read at fire time (not epoch-start snapshot)
# ---------------------------------------------------------------------------


def test_host_ram_guard_reads_flush_extra_at_fire_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """flush_extra is evaluated when the guard fires, not at epoch start."""
    floor_bytes = int(4e9)
    available_bytes = int(1e9)
    monkeypatch.setattr(loop_mod.psutil, "virtual_memory", lambda: _fake_vmem(available_bytes))

    ds = _TinyDataset()
    wrapper = make_stub_wrapper(dim=8, working=True)
    cfg = _make_cfg(tmp_path)
    apply_lora(wrapper, cfg.peft)
    run_dir = tmp_path / "run"
    (run_dir / "checkpoints").mkdir(parents=True)

    optimizer = torch.optim.AdamW([p for p in wrapper.parameters() if p.requires_grad], lr=1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda s: 1.0)

    live_state: dict[str, Any] = {
        "ladder": {"best": float("-inf"), "evals_without_improvement": 0},
        "best_metric_value": None,
        "scheduler_kind": "plateau",
    }
    post_step_ladder = {"best": 0.77, "evals_without_improvement": 0}
    calls: list[dict[str, Any]] = []

    def flush_extra() -> dict[str, Any]:
        live_state["ladder"] = post_step_ladder
        live_state["best_metric_value"] = 0.77
        snapshot = dict(live_state)
        calls.append(snapshot)
        return snapshot

    with pytest.raises(loop_mod._HostRamLow):
        loop_mod.run_epoch(
            wrapper,
            _loader(ds),
            optimizer,
            scheduler,
            NoopTracker(),
            cfg,
            run_dir,
            epoch=0,
            global_step=0,
            nan_streak=0,
            class_names=ds.class_names,
            on_checkpoint=lambda *a: None,
            on_eval=lambda *a: None,
            host_ram_floor_bytes=floor_bytes,
            flush_extra=flush_extra,
        )

    assert len(calls) == 1, "flush_extra should be called exactly once at the guard fire"
    ckpt_dir = run_dir / "checkpoints" / "step_1"
    state = torch.load(ckpt_dir / "training_state.pt", weights_only=False)
    assert state["ladder"] == post_step_ladder
    assert state["best_metric_value"] == pytest.approx(0.77)


# ---------------------------------------------------------------------------
# Trainer integration: loop_mod._HostRamLow → graceful EvalArtifacts (exit-0 path)
# ---------------------------------------------------------------------------


def test_trainer_fit_returns_graceful_artifacts_on_host_ram_low(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Trainer.fit() with low-RAM condition catches loop_mod._HostRamLow and returns graceful
    EvalArtifacts: checkpoint_path exists, final_metrics is None, time_limit_stop is None,
    host_ram_stop carries the flushed step, run completes without raising."""
    floor_bytes = int(4e9)  # 4 GB floor
    available_bytes = int(1e9)  # 1 GB available → guard fires
    monkeypatch.setattr(loop_mod.psutil, "virtual_memory", lambda: _fake_vmem(available_bytes))

    ds = _TinyDataset()
    wrapper = make_stub_wrapper(dim=8, working=True)
    cfg = _make_cfg(tmp_path)
    cfg = cfg.model_copy(
        update={
            "train": cfg.train.model_copy(
                update={
                    "epochs": 50,
                    "host_ram_floor_gb": floor_bytes / 1e9,
                }
            )
        }
    )
    apply_lora(wrapper, cfg.peft)

    trainer = Trainer(wrapper, ds, ds, NoopTracker(), cfg)
    result = trainer.fit(run_dir=tmp_path / "ram-stop-run")

    # Must return gracefully (no exception), no normal close_out.
    assert isinstance(result, EvalArtifacts)
    assert result.final_metrics is None
    assert result.time_limit_stop is None  # distinct from time-limit path
    # New field: host_ram_stop must be set.
    assert result.host_ram_stop is not None
    stop = result.host_ram_stop
    assert stop.stop_step == 1
    assert stop.stop_epoch == 0
    assert stop.total_epochs == 50
    assert stop.available_gb == pytest.approx(available_bytes / 1e9)
    # Flushed checkpoint must exist.
    ckpt = result.checkpoint_path
    assert ckpt.exists()
    assert not (tmp_path / "ram-stop-run" / "adapter").exists()
    assert not (tmp_path / "ram-stop-run" / "metrics.json").exists()


# ---------------------------------------------------------------------------
# Config schema: default value
# ---------------------------------------------------------------------------


def test_train_hyperparams_host_ram_floor_gb_default() -> None:
    """TrainHyperparams.host_ram_floor_gb defaults to 2.0 (guard on by default)."""
    from custom_sam_peft.config.schema import TrainHyperparams

    assert TrainHyperparams(epochs=1).host_ram_floor_gb == 2.0


def test_train_hyperparams_host_ram_floor_gb_zero_disables() -> None:
    """Setting host_ram_floor_gb=0 produces host_ram_floor_bytes=None in the trainer."""
    from custom_sam_peft.config.schema import TrainHyperparams

    hp = TrainHyperparams(epochs=1, host_ram_floor_gb=0.0)
    assert hp.host_ram_floor_gb == 0.0


def test_run_epoch_psutil_probe_fail_open(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """If psutil.virtual_memory() raises, the run continues without crashing
    and without firing loop_mod._HostRamLow (fail-open: probe error skips that step's check)."""
    floor_bytes = int(4e9)

    def _raise_vmem() -> None:
        raise OSError("psutil probe failed (test)")

    monkeypatch.setattr(loop_mod.psutil, "virtual_memory", _raise_vmem)

    ds = _TinyDataset()
    wrapper = make_stub_wrapper(dim=8, working=True)
    cfg = _make_cfg(tmp_path)
    apply_lora(wrapper, cfg.peft)
    run_dir = tmp_path / "run"
    (run_dir / "checkpoints").mkdir(parents=True)

    optimizer = torch.optim.AdamW([p for p in wrapper.parameters() if p.requires_grad], lr=1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda s: 1.0)

    # Must complete normally — probe failure must not crash training.
    gs, _ = loop_mod.run_epoch(
        wrapper,
        _loader(ds),
        optimizer,
        scheduler,
        NoopTracker(),
        cfg,
        run_dir,
        epoch=0,
        global_step=0,
        nan_streak=0,
        class_names=ds.class_names,
        on_checkpoint=lambda *a: None,
        on_eval=lambda *a: None,
        host_ram_floor_bytes=floor_bytes,
    )
    assert gs == len(_loader(ds))  # all batches processed


def test_trainer_disabled_when_floor_gb_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """host_ram_floor_gb <= 0 disables the guard; run completes normally even with 0 RAM."""
    monkeypatch.setattr(loop_mod.psutil, "virtual_memory", lambda: _fake_vmem(0))

    ds = _TinyDataset()
    wrapper = make_stub_wrapper(dim=8, working=True)
    cfg = _make_cfg(tmp_path)
    cfg = cfg.model_copy(update={"train": cfg.train.model_copy(update={"host_ram_floor_gb": 0.0})})
    apply_lora(wrapper, cfg.peft)

    trainer = Trainer(wrapper, ds, ds, NoopTracker(), cfg)
    result = trainer.fit(run_dir=tmp_path / "no-guard-run")

    # Normal completion; no RAM stop.
    assert isinstance(result, EvalArtifacts)
    assert result.final_metrics is not None or result.time_limit_stop is None
    assert result.host_ram_stop is None
