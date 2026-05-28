"""Pin the current trainer NaN-loss behavior.

C3 per spec §6.2. Documents what src/custom_sam_peft/train/loop.py does
TODAY: skip the micro-step + increment nan_streak; raise RuntimeError
after cfg.train.nan_abort_after consecutive non-finite micro-steps.

This file does NOT modify src/custom_sam_peft/train/loop.py. If a test
fails because the current behavior diverges from what's pinned here
(e.g., per-step vs. per-micro-step threshold semantics differ from the
reading), the reviewer files a follow-up issue; the fix is out of scope
for this PR per spec §6.2 C3.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest
import torch

from custom_sam_peft.config.schema import (
    AugmentationsConfig,
    DataConfig,
    DataSplit,
    NormalizeConfig,
    PEFTConfig,
    RunConfig,
    TextPromptConfig,
    TrainConfig,
    TrainHyperparams,
)
from custom_sam_peft.data.coco import COCODataset
from custom_sam_peft.data.transforms import build_train_transforms
from custom_sam_peft.peft_adapters.lora import apply_lora
from custom_sam_peft.tracking.noop import NoopTracker
from custom_sam_peft.train.trainer import Trainer
from tests.fixtures.tiny_sam3_lora_stub import FIXTURE_SCOPE_PATTERNS, make_stub_wrapper

# --- Local helpers (duplicated from tests/integration/test_train_resume.py
# per spec §6.2 C3 "no shared-helper extraction this PR"). ---


def _ds(tiny_coco_dir: Path) -> COCODataset:
    transforms = build_train_transforms(
        AugmentationsConfig(preset="none"),
        32,
        model_name="facebook/sam3.1",
        normalize=NormalizeConfig(),
    )
    return COCODataset(
        annotations=str(tiny_coco_dir / "annotations.json"),
        images=str(tiny_coco_dir / "images"),
        prompt_mode="text",
        transforms=transforms,
        text_prompt=TextPromptConfig(),
    )


def _cfg(
    tmp_path: Path,
    tiny_coco_dir: Path,
    *,
    nan_abort_after: int,
    epochs: int,
) -> TrainConfig:
    cfg = TrainConfig(
        run=RunConfig(name="nan", output_dir=str(tmp_path), seed=42),
        data=DataConfig(
            format="coco",
            train=DataSplit(
                annotations=str(tiny_coco_dir / "annotations.json"),
                images=str(tiny_coco_dir / "images"),
            ),
            val=DataSplit(
                annotations=str(tiny_coco_dir / "annotations.json"),
                images=str(tiny_coco_dir / "images"),
            ),
            prompt_mode="text",
        ),
        peft=PEFTConfig(
            method="lora",
            scope="vision",
            target_modules=FIXTURE_SCOPE_PATTERNS["vision"],
        ),
        train=TrainHyperparams(
            epochs=epochs,
            batch_size=1,
            grad_accum_steps=1,
            save_every=2,
            log_every=1,
            warmup_steps=0,
            num_workers=0,
            nan_abort_after=nan_abort_after,
        ),
    )
    return cfg


def _nan_loss_dict() -> dict[str, torch.Tensor]:
    """Dict mirroring total_loss()'s return shape, with NaN total."""
    return {
        "total": torch.tensor(float("nan"), requires_grad=True),
        "mask": torch.tensor(0.0),
        "box": torch.tensor(0.0),
        "obj": torch.tensor(0.0),
        "presence": torch.tensor(0.0),
    }


# --- Tests ---


def test_nan_loss_below_threshold_does_not_abort(
    tmp_path: Path, tiny_coco_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """3 NaN micro-steps under nan_abort_after=5 → trainer continues."""
    from custom_sam_peft.train import loop as loop_mod

    real_total_loss = loop_mod.total_loss
    counter = {"n": 0}

    def fake_total_loss(*args: Any, **kwargs: Any) -> dict[str, torch.Tensor]:
        counter["n"] += 1
        if counter["n"] <= 3:
            return _nan_loss_dict()
        return real_total_loss(*args, **kwargs)

    monkeypatch.setattr("custom_sam_peft.train.loop.total_loss", fake_total_loss)

    ds = _ds(tiny_coco_dir)
    cfg = _cfg(tmp_path, tiny_coco_dir, nan_abort_after=5, epochs=2)
    w = make_stub_wrapper(dim=8, working=True)
    apply_lora(w, cfg.peft)
    trainer = Trainer(w, ds, ds, NoopTracker(), cfg)

    # Must NOT raise — 3 < 5.
    result = trainer.fit(run_dir=tmp_path / "run-below")
    assert result.run_dir.exists()


def test_nan_loss_at_threshold_raises_runtime_error(
    tmp_path: Path, tiny_coco_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Persistent NaN under nan_abort_after=3 → RuntimeError('non-finite ...')."""
    monkeypatch.setattr(
        "custom_sam_peft.train.loop.total_loss",
        lambda *a, **kw: _nan_loss_dict(),
    )

    ds = _ds(tiny_coco_dir)
    cfg = _cfg(tmp_path, tiny_coco_dir, nan_abort_after=3, epochs=2)
    w = make_stub_wrapper(dim=8, working=True)
    apply_lora(w, cfg.peft)
    trainer = Trainer(w, ds, ds, NoopTracker(), cfg)

    with pytest.raises(RuntimeError, match="non-finite"):
        trainer.fit(run_dir=tmp_path / "run-abort")


def test_nan_loss_logs_warning_on_value_error_path(
    tmp_path: Path,
    tiny_coco_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """ValueError from total_loss surfaces as the 'treating as non-finite' WARNING."""
    monkeypatch.setattr(
        "custom_sam_peft.train.loop.total_loss",
        lambda *a, **kw: (_ for _ in ()).throw(ValueError("non-finite cost matrix")),
    )

    ds = _ds(tiny_coco_dir)
    # nan_abort_after=2: 2 ValueErrors → abort. We want the WARNING to fire on
    # the first one before the RuntimeError on the second.
    cfg = _cfg(tmp_path, tiny_coco_dir, nan_abort_after=2, epochs=2)
    w = make_stub_wrapper(dim=8, working=True)
    apply_lora(w, cfg.peft)
    trainer = Trainer(w, ds, ds, NoopTracker(), cfg)

    with (
        caplog.at_level(logging.WARNING, logger="custom_sam_peft.train.loop"),
        pytest.raises(RuntimeError, match="non-finite"),
    ):
        trainer.fit(run_dir=tmp_path / "run-warn")

    warning_msgs = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("treating as non-finite" in m for m in warning_msgs), (
        f"expected 'treating as non-finite' warning; got: {warning_msgs}"
    )
