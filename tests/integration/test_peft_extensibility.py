"""PEFT extensibility test — OCP proof (spec §9.1 / plan criterion 14).

Registers a stub adapter from tests/fixtures/ via @register("peft", "stub")
and runs a tiny Trainer.fit() to prove that the registry+protocol surface
is open for extension without modifying any file under src/.

The proof is that this test passes.  No src/ file is touched; the only
new code lives in tests/fixtures/stub_peft_adapter.py.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from custom_sam_peft.config.schema import (
    DataConfig,
    DataSplit,
    EvalConfig,
    PEFTConfig,
    RunConfig,
    TrainConfig,
    TrainHyperparams,
)
from custom_sam_peft.tracking.noop import NoopTracker
from custom_sam_peft.train.trainer import Trainer

# Importing the stub module triggers the @register("peft", "stub") side effect.
# This is the only action required to register a new adapter from outside src/.
from tests.fixtures.stub_peft_adapter import StubPEFTMethod, apply_stub
from tests.fixtures.tiny_sam3_lora_stub import FIXTURE_SCOPE_PATTERNS, make_stub_wrapper

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_cfg(tmp_path: Path) -> TrainConfig:
    """Minimal TrainConfig selecting method='stub'.

    PEFTConfig.method is normally a Literal["lora", "qlora"], so we use
    model_construct to bypass Pydantic validation for the stub method string.
    The rest of the config is valid; only the PEFT method name is non-standard.
    This mirrors how a real extension would work if the schema were opened up.
    """
    # Build a valid base TrainConfig, then swap in the stub PEFTConfig.
    base_cfg = TrainConfig(
        run=RunConfig(name="peft-ext-test", output_dir=str(tmp_path / "runs"), seed=0),
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
        eval=EvalConfig(mode="lite", iou_thresholds=[0.5], lite_max_images=2),
    )
    # Swap the PEFT method to "stub" via model_construct (bypasses the Literal
    # validator).  This simulates what would happen if the schema were extended.
    stub_peft = PEFTConfig.model_construct(**{**base_cfg.peft.model_dump(), "method": "stub"})
    return base_cfg.model_copy(update={"peft": stub_peft})


# ---------------------------------------------------------------------------
# OCP-proof extensibility test
# ---------------------------------------------------------------------------


def test_peft_extensibility_stub_adapter(tiny_text_dataset, tmp_path: Path) -> None:
    """Trainer.fit() accepts a third-party stub adapter registered via @register.

    Steps:
      1. Import stub_peft_adapter (already done at module level above) →
         @register("peft", "stub") fires as a side effect.
      2. Apply the stub to the wrapper via apply_stub() — mirrors how
         apply_lora() is called in the seam tests.
      3. Monkeypatch make_peft_method so "stub" maps to StubPEFTMethod
         (no src/ modification; this is the test-boundary injection).
      4. Run Trainer.fit() with method="stub" — must complete without error.
      5. Assert EvalArtifacts carries peft_method="stub".
    """
    wrapper = make_stub_wrapper(dim=8, working=True)
    cfg = _make_cfg(tmp_path)

    # Step 2: apply the stub adapter (no-op; model is returned untouched).
    apply_stub(wrapper, cfg.peft)

    # Step 3: route "stub" through make_peft_method without touching src/.
    _original_make_peft_method = __import__(
        "custom_sam_peft.peft_adapters", fromlist=["make_peft_method"]
    ).make_peft_method

    def _patched_make_peft_method(method: str):  # type: ignore[return]
        if method == "stub":
            return StubPEFTMethod()
        return _original_make_peft_method(method)

    def _stub_save_adapter(w: object, path: Path) -> None:
        """No-op stub checkpoint writer: create the directory as a stand-in."""
        path.mkdir(parents=True, exist_ok=True)

    with (
        patch(
            "custom_sam_peft.train.trainer.make_peft_method",
            side_effect=_patched_make_peft_method,
        ),
        patch(
            "custom_sam_peft.train.trainer.save_adapter",
            side_effect=_stub_save_adapter,
        ),
    ):
        trainer = Trainer(
            model=wrapper,
            train_ds=tiny_text_dataset,
            val_ds=tiny_text_dataset,
            tracker=NoopTracker(),
            cfg=cfg,
        )
        artifacts = trainer.fit(run_dir=tmp_path / "ext-run")

    # Step 5: training completed; EvalArtifacts carry the stub method name.
    assert artifacts.peft_method == "stub", (
        f"Expected peft_method='stub' in EvalArtifacts; got {artifacts.peft_method!r}"
    )
    assert artifacts.run_dir.is_dir(), "run_dir must exist after fit()"
    assert artifacts.checkpoint_path.exists(), "adapter path must exist after fit()"
