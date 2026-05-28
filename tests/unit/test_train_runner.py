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

    cfg.data.model_dump.return_value = {"format": "coco"}
    cfg.model.name = "facebook/sam3.1"
    cfg.peft.method = "lora"
    cfg.tracking.backend = "none"
    cfg.tracking.wandb.project = "custom_sam_peft"
    cfg.tracking.wandb.entity = None
    cfg.data.limit.train = None
    cfg.data.limit.val = None
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
    monkeypatch.setattr("custom_sam_peft.train.runner.load_sam31", lambda _m, **_kw: MagicMock())
    monkeypatch.setattr(
        "custom_sam_peft.train.runner.build_tracker",
        lambda _cfg: MagicMock(close=MagicMock(), start_run=MagicMock()),
    )
    # Bypass val_source resolution + persistence — this test is purely about
    # registry dispatch wiring with a MagicMock cfg (no real schema).
    from custom_sam_peft.data.val_source import ValSource

    monkeypatch.setattr(
        "custom_sam_peft.train.runner.resolve_val_source",
        lambda _cfg, run_dir=None: ValSource(
            mode="explicit",
            train_ids=None,
            val_ids=None,
            realized_fraction=None,
            per_class_counts=None,
            missing_in_val=None,
            fraction_requested=None,
            seed_used=None,
        ),
    )
    monkeypatch.setattr("custom_sam_peft.train.runner.save_val_source", lambda _vs, _run_dir: None)
    monkeypatch.setattr("custom_sam_peft.train.runner._log_val_source", lambda _vs: None)

    fake_result = MagicMock()

    def fake_fit(self, *, run_dir, resume_from=None):
        return fake_result

    monkeypatch.setattr("custom_sam_peft.train.runner.Trainer.fit", fake_fit)

    result = run_training(cfg)
    assert result is fake_result
    kinds = {k for k, _ in calls}
    assert kinds == {"dataset", "peft"}


# ---------------------------------------------------------------------------
# spec/data-no-val-auto-split (#71): val_source orchestration
# ---------------------------------------------------------------------------


def test_run_training_writes_val_source_json_on_auto_split(
    tmp_path: Path, tiny_coco_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Spec §6.4 + §9.7.1: end-to-end auto-split writes <run_dir>/val_source.json.

    Uses tiny_coco + LoRA stub to keep this CPU-bound.
    """
    from custom_sam_peft.config.schema import (
        DataConfig,
        DataSplit,
        PEFTConfig,
        RunConfig,
        TrackingConfig,
        TrainConfig,
        TrainHyperparams,
        ValSplitConfig,
    )
    from custom_sam_peft.data.val_source import load_val_source
    from tests.fixtures.tiny_sam3_lora_stub import FIXTURE_SCOPE_PATTERNS, make_stub_wrapper

    cfg = TrainConfig(
        run=RunConfig(name="autosplit", output_dir=str(tmp_path), seed=0),
        data=DataConfig(
            format="coco",
            train=DataSplit(
                annotations=str(tiny_coco_dir / "annotations.json"),
                images=str(tiny_coco_dir / "images"),
            ),
            val=None,
            val_split=ValSplitConfig(fraction=0.5, seed=None),
        ),
        peft=PEFTConfig(
            method="lora", scope="vision", target_modules=FIXTURE_SCOPE_PATTERNS["vision"]
        ),
        train=TrainHyperparams(
            epochs=1,
            batch_size=1,
            grad_accum_steps=1,
            save_every=2,
            log_every=1,
            warmup_steps=0,
            num_workers=0,
        ),
        tracking=TrackingConfig(backend="none"),  # tensorboard not in dev deps
    )

    monkeypatch.setattr(
        "custom_sam_peft.train.runner.load_sam31",
        lambda _m, **_kw: make_stub_wrapper(dim=8, working=True),
    )
    from custom_sam_peft import train as _train_pkg  # noqa: F401

    # peft_factory must accept (wrapper, cfg.peft) and apply lora; reuse real.
    from custom_sam_peft.train.runner import run_training

    result = run_training(cfg)
    assert (result.run_dir / "val_source.json").is_file()
    vs = load_val_source(result.run_dir)
    assert vs is not None
    assert vs.mode == "auto_split"
    assert vs.train_ids is not None and vs.val_ids is not None


def test_run_training_resume_reuses_saved_val_source(
    tmp_path: Path, tiny_coco_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Spec §8.2 + §9.7.2: resume reuses the saved partition; splitter not re-called."""
    from custom_sam_peft.config.schema import (
        DataConfig,
        DataSplit,
        PEFTConfig,
        RunConfig,
        TrackingConfig,
        TrainConfig,
        TrainHyperparams,
        ValSplitConfig,
    )
    from custom_sam_peft.data.val_source import load_val_source
    from tests.fixtures.tiny_sam3_lora_stub import FIXTURE_SCOPE_PATTERNS, make_stub_wrapper

    def _cfg() -> TrainConfig:
        return TrainConfig(
            run=RunConfig(name="resume", output_dir=str(tmp_path), seed=0),
            data=DataConfig(
                format="coco",
                train=DataSplit(
                    annotations=str(tiny_coco_dir / "annotations.json"),
                    images=str(tiny_coco_dir / "images"),
                ),
                val=None,
                val_split=ValSplitConfig(fraction=0.5, seed=None),
            ),
            peft=PEFTConfig(
                method="lora", scope="vision", target_modules=FIXTURE_SCOPE_PATTERNS["vision"]
            ),
            train=TrainHyperparams(
                epochs=1,
                batch_size=1,
                grad_accum_steps=1,
                save_every=1,
                log_every=1,
                warmup_steps=0,
                num_workers=0,
            ),
            tracking=TrackingConfig(backend="none"),  # tensorboard not in dev deps
        )

    monkeypatch.setattr(
        "custom_sam_peft.train.runner.load_sam31",
        lambda _m, **_kw: make_stub_wrapper(dim=8, working=True),
    )
    from custom_sam_peft.train.runner import run_training

    # First run.
    r1 = run_training(_cfg())
    vs1 = load_val_source(r1.run_dir)
    assert vs1 is not None
    saved_train = vs1.train_ids
    saved_val = vs1.val_ids
    ckpts = sorted((r1.run_dir / "checkpoints").glob("step_*"))
    assert ckpts, "first run produced no checkpoint"

    # Second run with resume_from set; if splitter is invoked the test fails.
    def _splitter_must_not_run(*a: object, **kw: object) -> object:
        raise AssertionError("splitter must not be re-called on resume")

    monkeypatch.setattr("custom_sam_peft.data.val_source.stratified_split", _splitter_must_not_run)
    r2 = run_training(_cfg(), resume_from=ckpts[0])
    vs2 = load_val_source(r2.run_dir)
    assert vs2 is not None
    assert vs2.train_ids == saved_train
    assert vs2.val_ids == saved_val
