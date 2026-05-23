"""Tracker swap-in/swap-out integration test (spec §9.1 / plan criterion 13).

Parameterizes Trainer.fit() over three tracker backends and asserts that the
same Tracker protocol calls are made regardless of which concrete class is
injected.  The Trainer must never inspect the tracker's type; it must interact
solely through the four-method protocol:

    start_run / log_scalars / log_images / close

Backends covered:
  1. ``NoopTracker``        — always available (built-in; no extras)
  2. ``_RecordingTracker``  — in-test fake that records all calls
  3. ``WandBTracker``       — mocked SDK (skipped if wandb extra absent)
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from custom_sam_peft.config.schema import (
    DataConfig,
    DataSplit,
    EvalConfig,
    PEFTConfig,
    RunConfig,
    TrackingConfig,
    TrainConfig,
    TrainHyperparams,
    WandbConfig,
)
from custom_sam_peft.peft_adapters.lora import apply_lora
from custom_sam_peft.tracking.base import Tracker
from custom_sam_peft.tracking.noop import NoopTracker
from custom_sam_peft.train.trainer import Trainer
from tests.fixtures.tiny_sam3_lora_stub import FIXTURE_SCOPE_PATTERNS, make_stub_wrapper

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Recording tracker (in-test fake)
# ---------------------------------------------------------------------------


class _RecordingTracker:
    """Tracker that records every protocol call for post-run assertion.

    Satisfies the ``Tracker`` runtime-checkable Protocol so ``isinstance``
    checks pass.  All four methods are implemented; ``log_scalars`` and
    ``log_images`` append their arguments to public lists for inspection.
    """

    def __init__(self) -> None:
        self.start_run_calls: list[tuple[Path, dict[str, Any], Path | None]] = []
        self.log_scalars_calls: list[tuple[int, dict[str, float]]] = []
        self.log_images_calls: list[tuple[int, dict[str, np.ndarray[Any, Any]]]] = []
        self.close_call_count: int = 0

    def start_run(
        self,
        run_dir: Path,
        config: dict[str, Any],
        resume_from: Path | None = None,
    ) -> None:
        self.start_run_calls.append((run_dir, config, resume_from))

    def log_scalars(self, step: int, values: dict[str, float]) -> None:
        self.log_scalars_calls.append((step, values))

    def log_images(self, step: int, images: dict[str, np.ndarray[Any, Any]]) -> None:
        self.log_images_calls.append((step, images))

    def close(self) -> None:
        self.close_call_count += 1


# Sanity check: _RecordingTracker satisfies the protocol at import time.
assert isinstance(_RecordingTracker(), Tracker), (
    "_RecordingTracker must satisfy the Tracker runtime-checkable Protocol"
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_cfg(tmp_path: Path) -> TrainConfig:
    """Minimal TrainConfig wired to the tiny in-memory dataset fixture shape."""
    return TrainConfig(
        run=RunConfig(name="swap-test", output_dir=str(tmp_path / "runs"), seed=0),
        data=DataConfig(
            format="coco",
            train=DataSplit(annotations="x", images="x"),
            val=DataSplit(annotations="x", images="x"),
            prompt_mode="text",
        ),
        peft=PEFTConfig(
            method="lora",
            target_modules=FIXTURE_SCOPE_PATTERNS["vision"],
        ),
        train=TrainHyperparams(
            epochs=1,
            batch_size=1,
            grad_accum_steps=1,
            learning_rate=1e-4,
            warmup_steps=0,
            eval_every=1,
            save_every=1000,
            log_every=1,
            num_workers=0,
        ),
        eval=EvalConfig(mode="full", iou_thresholds=[0.5], lite_max_images=1),
        tracking=TrackingConfig(
            backend="none",  # Trainer ignores this; tracker is injected directly.
            wandb=WandbConfig(project="custom_sam_peft-swap-test", entity=None),
        ),
    )


def _make_tiny_dataset() -> Any:
    """Two-image, two-class in-memory dataset (same shape as conftest fixture)."""
    import torch

    from custom_sam_peft.data.base import Example, Instance, TextPrompts

    _class_names = ["cat", "dog"]

    def _make_ex(image_id: str, class_id: int) -> Example:
        h = w = 8
        image = torch.zeros(3, h, w)
        mask = torch.zeros(h, w, dtype=torch.bool)
        mask[:4, :4] = True
        return Example(
            image=image,
            image_id=image_id,
            prompts=TextPrompts(classes=_class_names),
            instances=[
                Instance(
                    mask=mask,
                    class_id=class_id,
                    box=torch.tensor([0.0, 0.0, 4.0, 4.0]),
                )
            ],
        )

    examples = [_make_ex("img_0", 0), _make_ex("img_1", 1)]

    class _InMemDs:
        class_names = _class_names

        def __len__(self) -> int:
            return len(examples)

        def __getitem__(self, i: int) -> Example:
            return examples[i]

    return _InMemDs()


def _run_fit(tracker: Any, run_dir: Path) -> None:
    """Execute ``Trainer.fit()`` with *tracker* injected directly.

    Uses the tiny in-memory dataset and a working stub wrapper so no real
    SAM 3.1 checkpoint or GPU is required.
    """
    ds = _make_tiny_dataset()
    cfg = _make_cfg(run_dir.parent)
    wrapper = make_stub_wrapper(dim=8, working=True)
    apply_lora(wrapper, cfg.peft)

    trainer = Trainer(
        model=wrapper,
        train_ds=ds,
        val_ds=ds,
        tracker=tracker,
        cfg=cfg,
    )
    trainer.fit(run_dir=run_dir)


# ---------------------------------------------------------------------------
# Individual backend tests
# ---------------------------------------------------------------------------


def test_trainer_protocol_calls_noop_tracker(tmp_path: Path) -> None:
    """NoopTracker: Trainer completes fit() without raising."""
    _run_fit(NoopTracker(), run_dir=tmp_path / "run")
    # Completion without exception is the only assertion needed for a no-op sink.


def test_trainer_protocol_calls_recording_tracker(tmp_path: Path) -> None:
    """_RecordingTracker: Trainer calls start_run, log_scalars, and close."""
    tracker = _RecordingTracker()
    _run_fit(tracker, run_dir=tmp_path / "run")

    # --- start_run -----------------------------------------------------------
    assert len(tracker.start_run_calls) == 1, (
        f"Expected exactly 1 start_run call; got {len(tracker.start_run_calls)}"
    )
    start_run_dir, start_cfg, start_resume = tracker.start_run_calls[0]
    assert start_run_dir == tmp_path / "run", (
        f"start_run called with unexpected run_dir={start_run_dir!r}"
    )
    assert isinstance(start_cfg, dict), "start_run config arg must be a dict"
    assert start_resume is None, "no resume_from expected for a fresh run"

    # --- log_scalars ---------------------------------------------------------
    # The lite eval callback (on_eval) fires at least once per epoch.
    assert tracker.log_scalars_calls, (
        "Expected at least one log_scalars call (from lite mid-run eval); "
        f"got log_scalars_calls={tracker.log_scalars_calls!r}"
    )
    for step, values in tracker.log_scalars_calls:
        assert isinstance(step, int) and step >= 0, (
            f"log_scalars step must be a non-negative int; got {step!r}"
        )
        assert isinstance(values, dict), f"log_scalars values must be a dict; got {type(values)!r}"
        for k, v in values.items():
            assert isinstance(k, str), f"scalar key must be str; got {k!r}"
            assert isinstance(v, float), f"scalar value must be float; got {v!r}"

    # --- close ---------------------------------------------------------------
    assert tracker.close_call_count == 1, (
        f"Expected close() called exactly once; got {tracker.close_call_count}"
    )


def test_trainer_protocol_calls_wandb_tracker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """WandBTracker (offline / mocked SDK): Trainer makes the same protocol calls.

    The ``wandb`` package is replaced with a minimal fake so this test passes
    without a real W&B account or network connection.

    Skipped when: ``WandBTracker.__init__`` raises ``ImportError`` even after
    the fake module is injected (indicates the wandb extra is not installed in
    a way that satisfies the guard).
    """
    # Install minimal fake wandb into sys.modules before WandBTracker is used.
    fake_run = MagicMock()
    fake_run.id = "fake-run-id-swap"
    fake_wandb = types.ModuleType("wandb")
    fake_wandb.init = MagicMock(return_value=fake_run)  # type: ignore[attr-defined]
    fake_wandb.Image = lambda arr: ("WandbImage", arr)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    try:
        from custom_sam_peft.tracking.wandb import WandBTracker

        cfg = _make_cfg(tmp_path)
        tracker = WandBTracker(cfg)
    except ImportError as exc:
        pytest.skip(f"WandBTracker not usable (wandb extra absent): {exc}")
        return  # unreachable; pytest.skip raises — satisfies CodeQL definite-init

    _run_fit(tracker, run_dir=tmp_path / "run")

    # wandb.init must be called exactly once (by start_run)
    fake_wandb.init.assert_called_once()

    # run.log must be called at least once (scalars from lite eval callback)
    assert fake_run.log.called, (
        "Expected wandb run.log() to be called at least once via log_scalars; "
        f"call_args_list={fake_run.log.call_args_list!r}"
    )

    # run.finish must be called exactly once (by close)
    fake_run.finish.assert_called_once()
