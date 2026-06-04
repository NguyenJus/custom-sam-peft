"""Trainer.no_val mode tests — val_ds=None short-circuits eval/panel/end-of-run eval.

Spec: docs/superpowers/specs/2026-05-22-data-no-val-auto-split-design.md §7.1, §9.5.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from custom_sam_peft.config.schema import (
    DataConfig,
    DataSplit,
    PEFTConfig,
    RunConfig,
    TrackingConfig,
    TrainConfig,
    TrainHyperparams,
)
from custom_sam_peft.data.coco import COCODataset
from custom_sam_peft.data.transforms import build_train_transforms
from custom_sam_peft.peft_adapters.lora import apply_lora
from custom_sam_peft.tracking import build_tracker
from custom_sam_peft.train.trainer import Trainer
from tests.fixtures.tiny_sam3_lora_stub import FIXTURE_SCOPE_PATTERNS, make_stub_wrapper


def _ds_train(tiny_coco_dir: Path) -> COCODataset:
    from custom_sam_peft.config.schema import (
        AugmentationsConfig,
        NormalizeConfig,
        TextPromptConfig,
    )

    transforms = build_train_transforms(
        AugmentationsConfig(preset="none"),
        32,
        model_name="facebook/sam3.1",
        normalize=NormalizeConfig(),
    )
    return COCODataset(
        annotations=str(tiny_coco_dir / "annotations.json"),
        images=str(tiny_coco_dir / "images"),
        transforms=transforms,
        text_prompt=TextPromptConfig(),
    )


def _cfg(tmp_path: Path, tiny_coco_dir: Path) -> TrainConfig:
    return TrainConfig(
        run=RunConfig(name="no-val", output_dir=str(tmp_path), seed=0),
        data=DataConfig(
            format="coco",
            train=DataSplit(
                annotations=str(tiny_coco_dir / "annotations.json"),
                images=str(tiny_coco_dir / "images"),
            ),
            val=None,
            split=None,
        ),
        peft=PEFTConfig(
            method="lora", scope="vision", target_modules=FIXTURE_SCOPE_PATTERNS["vision"]
        ),
        train=TrainHyperparams(
            epochs=1,
            batch_size=1,
            grad_accum_steps=1,
            save_every=2,
            eval_every=1,
            log_every=1,
            warmup_steps=0,
            num_workers=0,
        ),
        tracking=TrackingConfig(backend="none"),
    )


def test_fit_with_val_ds_none_completes_and_writes_no_val_metrics(
    tmp_path: Path, tiny_coco_dir: Path
) -> None:
    """Trainer(val_ds=None).fit() completes; metrics.json carries the no-val note."""
    cfg = _cfg(tmp_path, tiny_coco_dir)
    ds_train = _ds_train(tiny_coco_dir)
    wrapper = make_stub_wrapper(dim=8, working=True)
    apply_lora(wrapper, cfg.peft)
    # Pre-save a split_source.json so the trainer's tracker hparams reader sees it.
    run_dir = tmp_path / f"{cfg.run.name}-test"
    run_dir.mkdir(parents=True)
    (run_dir / "split_source.json").write_text(
        json.dumps(
            {
                "mode": "none",
                "val_fraction_requested": None,
                "test_fraction_requested": None,
                "seed_used": None,
                "realized_fraction": None,
                "n_train": None,
                "n_val": None,
                "n_test": None,
                "per_class_counts": None,
                "missing_in_val": None,
                "missing_in_test": None,
                "train_ids": None,
                "val_ids": None,
                "test_ids": None,
            }
        )
    )
    trainer = Trainer(wrapper, ds_train, None, build_tracker(cfg), cfg)
    result = trainer.fit(run_dir=run_dir)
    assert result.final_metrics is None
    payload = json.loads((result.run_dir / "metrics.json").read_text())
    assert payload.get("note") == "no validation set provided"
    assert "global_step" in payload


def test_fit_with_val_ds_none_does_not_invoke_evaluator(
    tmp_path: Path, tiny_coco_dir: Path
) -> None:
    """Evaluator must not be constructed/called when val_ds is None."""
    cfg = _cfg(tmp_path, tiny_coco_dir)
    ds_train = _ds_train(tiny_coco_dir)
    wrapper = make_stub_wrapper(dim=8, working=True)
    apply_lora(wrapper, cfg.peft)
    run_dir = tmp_path / f"{cfg.run.name}-test2"
    run_dir.mkdir(parents=True)
    (run_dir / "split_source.json").write_text(json.dumps({"mode": "none"}))

    mock_evaluator = MagicMock()
    with patch("custom_sam_peft.train.trainer.Evaluator", return_value=mock_evaluator):
        trainer = Trainer(wrapper, ds_train, None, build_tracker(cfg), cfg)
        trainer.fit(run_dir=run_dir)
    mock_evaluator.evaluate.assert_not_called()


def test_fit_with_val_ds_none_does_not_log_image_panel(tmp_path: Path, tiny_coco_dir: Path) -> None:
    """The image-panel writer never fires when val_ds is None."""
    cfg = _cfg(tmp_path, tiny_coco_dir)
    ds_train = _ds_train(tiny_coco_dir)
    wrapper = make_stub_wrapper(dim=8, working=True)
    apply_lora(wrapper, cfg.peft)
    run_dir = tmp_path / f"{cfg.run.name}-test3"
    run_dir.mkdir(parents=True)
    (run_dir / "split_source.json").write_text(json.dumps({"mode": "none"}))

    tracker = build_tracker(cfg)
    tracker.log_images = MagicMock()  # type: ignore[method-assign]

    trainer = Trainer(wrapper, ds_train, None, tracker, cfg)
    trainer.fit(run_dir=run_dir)
    tracker.log_images.assert_not_called()


def test_config_yaml_round_trips_through_load_config(tmp_path: Path, tiny_coco_dir: Path) -> None:
    """config.yaml written by the trainer must reload cleanly via load_config.

    Regression for the bug where val_source provenance was injected into
    cfg_dict BEFORE writing config.yaml, causing TrainConfig (extra="forbid")
    to reject the extra key on finalize/resume reload.
    """
    from custom_sam_peft.config.loader import load_config

    cfg = _cfg(tmp_path, tiny_coco_dir)
    ds_train = _ds_train(tiny_coco_dir)
    wrapper = make_stub_wrapper(dim=8, working=True)
    apply_lora(wrapper, cfg.peft)
    run_dir = tmp_path / f"{cfg.run.name}-roundtrip"
    run_dir.mkdir(parents=True)
    # Place a split_source.json so the trainer's provenance-injection branch fires.
    (run_dir / "split_source.json").write_text(
        json.dumps(
            {
                "mode": "none",
                "val_fraction_requested": None,
                "test_fraction_requested": None,
                "seed_used": None,
                "realized_fraction": None,
                "n_train": None,
                "n_val": None,
                "n_test": None,
                "per_class_counts": None,
                "missing_in_val": None,
                "missing_in_test": None,
                "train_ids": None,
                "val_ids": None,
                "test_ids": None,
            }
        )
    )
    trainer = Trainer(wrapper, ds_train, None, build_tracker(cfg), cfg)
    trainer.fit(run_dir=run_dir)

    config_path = run_dir / "config.yaml"
    assert config_path.exists(), "trainer must write config.yaml"

    import yaml as _yaml

    written = _yaml.safe_load(config_path.read_text())
    assert "split_source" not in written, (
        "config.yaml must not contain split_source — it breaks TrainConfig round-trip"
    )

    # Must not raise — this is the exact call path used by --finalize/--resume.
    reloaded = load_config(config_path)
    assert isinstance(reloaded, type(cfg))


def test_fit_test_only_split_mode_produces_no_val_metrics(
    tmp_path: Path, tiny_coco_dir: Path
) -> None:
    """§10.6: test-only split (mode='none') still produces the no-val metrics note.

    Validates that the rename of split_source.json (was val_source.json) in
    assertions is wired correctly and that mode='none' with test_ids populates
    trainer hparams.
    """
    from custom_sam_peft.config.schema import (
        DataConfig,
        DataSplit,
        PEFTConfig,
        RunConfig,
        SplitConfig,
        TrackingConfig,
        TrainConfig,
        TrainHyperparams,
    )

    cfg = TrainConfig(
        run=RunConfig(name="test-only", output_dir=str(tmp_path), seed=0),
        data=DataConfig(
            format="coco",
            train=DataSplit(
                annotations=str(tiny_coco_dir / "annotations.json"),
                images=str(tiny_coco_dir / "images"),
            ),
            val=None,
            split=SplitConfig(test=0.3),
        ),
        peft=PEFTConfig(
            method="lora", scope="vision", target_modules=FIXTURE_SCOPE_PATTERNS["vision"]
        ),
        train=TrainHyperparams(
            epochs=1,
            batch_size=1,
            grad_accum_steps=1,
            save_every=2,
            eval_every=1,
            log_every=1,
            warmup_steps=0,
            num_workers=0,
        ),
        tracking=TrackingConfig(backend="none"),
    )
    ds_train = _ds_train(tiny_coco_dir)
    wrapper = make_stub_wrapper(dim=8, working=True)
    apply_lora(wrapper, cfg.peft)
    run_dir = tmp_path / f"{cfg.run.name}-test-only"
    run_dir.mkdir(parents=True)
    # Pre-save a split_source.json with mode='none' + test_ids populated.
    (run_dir / "split_source.json").write_text(
        json.dumps(
            {
                "mode": "none",
                "val_fraction_requested": None,
                "test_fraction_requested": 0.3,
                "seed_used": 0,
                "realized_fraction": [0.0, 0.3],
                "n_train": 7,
                "n_val": 0,
                "n_test": 3,
                "per_class_counts": None,
                "missing_in_val": None,
                "missing_in_test": [],
                "train_ids": None,
                "val_ids": [],
                "test_ids": ["1", "2", "3"],
            }
        )
    )
    trainer = Trainer(wrapper, ds_train, None, build_tracker(cfg), cfg)
    result = trainer.fit(run_dir=run_dir)
    # mode='none' → no val → final_metrics is None, metrics.json has the note
    assert result.final_metrics is None
    payload = json.loads((result.run_dir / "metrics.json").read_text())
    assert payload.get("note") == "no validation set provided"
    # The split_source.json must be present (not accidentally deleted)
    assert (run_dir / "split_source.json").is_file()
