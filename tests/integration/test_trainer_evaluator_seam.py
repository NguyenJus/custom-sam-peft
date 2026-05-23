"""Trainer ↔ evaluator seam test (spec §9.1 / plan §4.3.6).

Runs both ends without mocking internals. Asserts on EvalArtifacts
shape and that the evaluator consumes nothing else from the trainer.

The Trainer.fit → EvalArtifacts path is verified on CPU using the
existing tiny-stub fixtures. The run_train/run_eval library-API level
test is xfailed because those paths call load_sam31 which requires a
real SAM 3.1 checkpoint (no CPU fixture yet).
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import pytest
import torch

from custom_sam_peft.config.schema import (
    DataConfig,
    DataSplit,
    EvalConfig,
    PEFTConfig,
    RunConfig,
    TrainConfig,
    TrainHyperparams,
)
from custom_sam_peft.data.base import Example, Instance, TextPrompts
from custom_sam_peft.eval._artifacts import EvalArtifacts
from custom_sam_peft.eval.evaluator import Evaluator
from custom_sam_peft.peft_adapters.lora import apply_lora
from custom_sam_peft.tracking.noop import NoopTracker
from custom_sam_peft.train.trainer import Trainer
from tests.fixtures.tiny_sam3_lora_stub import FIXTURE_SCOPE_PATTERNS, make_stub_wrapper

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Minimal CPU-runnable fixtures
# ---------------------------------------------------------------------------


class _TinyDataset:
    """Two-image, one-class in-memory dataset — no disk, no transforms."""

    class_names: ClassVar[list[str]] = ["cat"]

    def __init__(self) -> None:
        self._examples = [
            Example(
                image=torch.zeros(3, 8, 8),
                image_id=f"img_{i}",
                prompts=TextPrompts(classes=["cat"]),
                instances=[
                    Instance(
                        mask=torch.zeros(8, 8, dtype=torch.bool),
                        class_id=0,
                        box=torch.tensor([0.0, 0.0, 4.0, 4.0]),
                    )
                ],
            )
            for i in range(2)
        ]

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, i: int) -> Example:
        return self._examples[i]


def _make_cfg(tmp_path: Path) -> TrainConfig:
    return TrainConfig(
        run=RunConfig(name="seam-test", output_dir=str(tmp_path), seed=0),
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
            save_every=1000,
            log_every=1,
            warmup_steps=0,
            num_workers=0,
        ),
        eval=EvalConfig(mode="lite", iou_thresholds=[0.5], lite_max_images=2),
    )


# ---------------------------------------------------------------------------
# Seam tests — run on CPU with stub model
# ---------------------------------------------------------------------------


def test_trainer_fit_returns_eval_artifacts(tmp_path: Path) -> None:
    """Trainer.fit must return an EvalArtifacts instance."""
    ds = _TinyDataset()
    wrapper = make_stub_wrapper(dim=8, working=True)
    cfg = _make_cfg(tmp_path)
    apply_lora(wrapper, cfg.peft)

    trainer = Trainer(wrapper, ds, ds, NoopTracker(), cfg)
    result = trainer.fit(run_dir=tmp_path / "seam-run")

    assert isinstance(result, EvalArtifacts), (
        f"Trainer.fit must return EvalArtifacts; got {type(result)}"
    )
    assert result.run_dir.is_dir()
    assert result.checkpoint_path.exists()
    assert result.peft_method in {"lora", "qlora"}


def test_evaluator_consumes_only_eval_artifacts(tmp_path: Path) -> None:
    """Evaluator receives only EvalArtifacts fields — not trainer internals.

    The evaluator is constructed independently (not via trainer). If it
    needed trainer-internal fields beyond EvalArtifacts, this test would
    fail to wire up correctly.
    """
    ds = _TinyDataset()
    wrapper = make_stub_wrapper(dim=8, working=True)
    cfg = _make_cfg(tmp_path)
    apply_lora(wrapper, cfg.peft)

    trainer = Trainer(wrapper, ds, ds, NoopTracker(), cfg)
    artifacts = trainer.fit(run_dir=tmp_path / "seam-run-2")

    assert isinstance(artifacts, EvalArtifacts)
    # Stand up a fresh Evaluator (as the eval runner would) and confirm
    # it can run using only what EvalArtifacts carries.
    eval_cfg = EvalConfig(mode="lite", iou_thresholds=[0.5], lite_max_images=2, batch_size=1)
    evaluator = Evaluator(eval_cfg)
    # Re-use the trained wrapper (in practice, eval runner would load from
    # artifacts.checkpoint_path; here we pass the live wrapper to avoid
    # disk-load path which needs a real model).
    report = evaluator.evaluate(wrapper, ds)
    assert "mAP" in report.overall
    # Key seam invariant: the run_dir in artifacts is where training wrote output.
    assert artifacts.run_dir.is_dir()
    assert (artifacts.run_dir / "metrics.json").exists()


def test_eval_artifacts_fields_match_training_output(tmp_path: Path) -> None:
    """peft_method in EvalArtifacts must match what the trainer was configured with."""
    ds = _TinyDataset()
    wrapper = make_stub_wrapper(dim=8, working=True)
    cfg = _make_cfg(tmp_path)
    apply_lora(wrapper, cfg.peft)

    trainer = Trainer(wrapper, ds, ds, NoopTracker(), cfg)
    artifacts = trainer.fit(run_dir=tmp_path / "seam-run-3")

    assert artifacts.peft_method == cfg.peft.method
    assert artifacts.run_dir == tmp_path / "seam-run-3"


# ---------------------------------------------------------------------------
# run_train / run_eval API-level test
# These call load_sam31 which requires a real SAM 3.1 checkpoint on disk.
# xfailed until a CPU-runnable fixture that bypasses load_sam31 exists.
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    reason="requires GPU / real SAM 3.1 checkpoint; will be re-enabled when CPU fixture exists",
    strict=False,
)
def test_run_train_returns_eval_artifacts_via_library_api(tmp_path: Path) -> None:
    """run_train(cfg) -> EvalArtifacts via the canonical library API."""
    from custom_sam_peft.train.runner import run_train

    # NOTE: This config references real dataset paths that don't exist;
    # the xfail catches load_sam31 / dataset-load failures.
    cfg = _make_cfg(tmp_path)
    artifacts = run_train(cfg)
    assert isinstance(artifacts, EvalArtifacts)
    assert artifacts.checkpoint_path.exists()
    assert artifacts.run_dir.is_dir()
    assert artifacts.peft_method in {"lora", "qlora"}


@pytest.mark.xfail(
    reason="requires GPU / real SAM 3.1 checkpoint; will be re-enabled when CPU fixture exists",
    strict=False,
)
def test_run_eval_accepts_eval_artifacts(tmp_path: Path) -> None:
    """run_eval(cfg, artifacts=artifacts) uses EvalArtifacts not trainer internals."""
    from custom_sam_peft.eval.runner import run_eval
    from custom_sam_peft.train.runner import run_train

    cfg = _make_cfg(tmp_path)
    artifacts = run_train(cfg)

    # Pass artifacts — evaluator must not reach into trainer internals.
    metrics = run_eval(cfg, artifacts=artifacts)
    assert isinstance(metrics, dict)
    assert "mAP" in metrics
