"""Tests for the Tracker protocol and the noop implementation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from custom_sam_peft._registry import RegistryError, list_registered, lookup
from custom_sam_peft.config.schema import (
    DataConfig,
    DataSplit,
    PEFTConfig,
    RunConfig,
    TrainConfig,
    TrainHyperparams,
)
from custom_sam_peft.tracking.base import Tracker
from custom_sam_peft.tracking.noop import NoopTracker, build_noop  # noqa: F401


def _minimal_cfg(tmp_path: Path) -> TrainConfig:
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
    )


@pytest.fixture(autouse=True)
def _ensure_noop_registered() -> None:
    """Re-register the noop factory if a sibling test file cleared the registry."""
    import contextlib
    import importlib

    try:
        lookup("tracker", "none")
    except RegistryError:
        from custom_sam_peft.tracking import noop as _noop_mod

        with contextlib.suppress(RegistryError):
            importlib.reload(_noop_mod)


def test_noop_tracker_conforms_to_protocol(tmp_path: Path) -> None:
    t: Tracker = NoopTracker()
    t.start_run(tmp_path, {"k": 1}, resume_from=None)
    t.log_scalars(0, {"loss": 1.0})
    t.log_images(0, {"sample": np.zeros((4, 4, 3), dtype=np.uint8)})
    t.close()


def test_noop_start_run_ignores_resume_from(tmp_path: Path) -> None:
    t = NoopTracker()
    t.start_run(tmp_path, {}, resume_from=tmp_path / "fake-checkpoint")  # must not raise


def test_noop_registered_under_tracker_kind(tmp_path: Path) -> None:
    assert "none" in list_registered("tracker")
    factory = lookup("tracker", "none")
    instance = factory(_minimal_cfg(tmp_path))
    assert type(instance).__name__ == "NoopTracker"
    assert type(instance).__module__ == "custom_sam_peft.tracking.noop"
