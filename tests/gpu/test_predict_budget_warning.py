"""GPU smoke test for the predict-budget warning wired in Trainer._probe_predict_budget.

Gated by ``@pytest.mark.gpu_t4``, ``@requires_compatible_gpu``, and
``@requires_checkpoint``.  Not in CI by default.  Run with::

    pytest -m gpu_t4 tests/gpu/test_predict_budget_warning.py -v

Two branches are exercised:

* **no-warn** — the real probe runs against the tiny LoRA config; the measured
  VRAM footprint is far below the 7 GB budget, so no warning is emitted.
* **warn** — ``decide_predict_budget_warning`` is patched in the *consumer*
  namespace (``custom_sam_peft.train.trainer``) to unconditionally return
  ``(True, <msg>)``; the test asserts both that the real empirical measurement
  was taken (``seen["measured_bytes"]`` is a positive int) and that the WARNING
  record containing "may not be usable" is emitted.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

import custom_sam_peft.train.trainer as trainer_mod
from custom_sam_peft.config.loader import load_config
from custom_sam_peft.config.schema import TrainConfig
from custom_sam_peft.train.runner import run_training
from tests.gpu.conftest import _RecordingTracker

pytestmark = [
    pytest.mark.gpu_t4,
    pytest.mark.requires_compatible_gpu,
    pytest.mark.requires_checkpoint,
]

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "examples" / "gpu_smoke_lora.yaml"

_LOGGER_NAME = "custom_sam_peft.train.trainer"
_WARN_SUBSTRING = "may not be usable"


def _make_cfg(tmp_path: Path, tiny_coco_dir: Path) -> TrainConfig:
    return load_config(
        CONFIG_PATH,
        overrides=[
            f"data.train.annotations={tiny_coco_dir / 'annotations.json'}",
            f"data.train.images={tiny_coco_dir / 'images'}",
            f"data.val.annotations={tiny_coco_dir / 'annotations.json'}",
            f"data.val.images={tiny_coco_dir / 'images'}",
            f"run.output_dir={tmp_path}",
        ],
    )


def test_no_warning_for_small_config(
    tmp_path: Path,
    tiny_coco_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Real unpatched probe: tiny LoRA forward is far under 7 GB — no warning fired."""
    cfg = _make_cfg(tmp_path, tiny_coco_dir)
    tracker = _RecordingTracker()
    monkeypatch.setattr("custom_sam_peft.train.runner.build_tracker", lambda *_a, **_kw: tracker)

    seen: dict[str, object] = {}
    real_decide = trainer_mod.decide_predict_budget_warning

    def _spy(measured_bytes: int, *a: object, **k: object) -> tuple[bool, str]:
        seen["measured_bytes"] = measured_bytes
        return real_decide(measured_bytes, *a, **k)

    monkeypatch.setattr(trainer_mod, "decide_predict_budget_warning", _spy)

    with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
        run_training(cfg)

    # Confirm the probe actually executed (not just silently skipped).
    assert "measured_bytes" in seen, (
        "decide_predict_budget_warning was never called — probe did not execute"
    )
    assert isinstance(seen["measured_bytes"], int) and seen["measured_bytes"] > 0, (
        f"expected a positive measured_bytes, got {seen['measured_bytes']!r}"
    )

    # Confirm no over-budget warning was emitted.
    offending = [r for r in caplog.records if _WARN_SUBSTRING in r.getMessage()]
    assert not offending, (
        f"expected no '{_WARN_SUBSTRING}' warning for small config, "
        f"but got: {[r.getMessage() for r in offending]}"
    )


def test_warning_fires_when_over_budget(
    tmp_path: Path,
    tiny_coco_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Force-warn via seam patch; asserts real empirical measurement ran and warning fires.

    ``decide_predict_budget_warning`` is patched in the *consumer* namespace
    (``custom_sam_peft.train.trainer``) rather than the producer
    (``custom_sam_peft.predict.budget``), because trainer.py binds the name at
    import time via ``from ... import decide_predict_budget_warning``.
    This mirrors the ``build_tracker`` patch rationale in test_real_train_overfits.py.

    The ``seen`` dict captures the ``measured_bytes`` argument that the probe
    passes to the decision function, proving the real forward + memory measurement
    executed before the seam was consulted.
    """
    cfg = _make_cfg(tmp_path, tiny_coco_dir)
    tracker = _RecordingTracker()
    monkeypatch.setattr("custom_sam_peft.train.runner.build_tracker", lambda *_a, **_kw: tracker)

    seen: dict[str, object] = {}

    def _force_warn(measured_bytes: int, *a: object, **k: object) -> tuple[bool, str]:
        seen["measured_bytes"] = measured_bytes
        return (
            True,
            "the trained model's predict footprint is ~9.9 GB; "
            "it may not be usable for prediction on 8 GB / CC 7.5 GPUs "
            "(budget ~7.0 GB).",
        )

    monkeypatch.setattr(trainer_mod, "decide_predict_budget_warning", _force_warn)

    with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
        run_training(cfg)

    # (1) Prove the real empirical forward + memory measurement ran.
    assert "measured_bytes" in seen, (
        "decide_predict_budget_warning was never called — probe did not execute"
    )
    assert isinstance(seen["measured_bytes"], int) and seen["measured_bytes"] > 0, (
        f"expected a positive int for measured_bytes, got {seen['measured_bytes']!r}"
    )

    # (2) Prove the WARNING record was emitted by the trainer logger.
    matching = [
        r
        for r in caplog.records
        if r.name == _LOGGER_NAME
        and r.levelno == logging.WARNING
        and _WARN_SUBSTRING in r.getMessage()
    ]
    assert matching, (
        f"expected a WARNING record containing '{_WARN_SUBSTRING}' from "
        f"logger '{_LOGGER_NAME}', but none found. "
        f"Records: {[(r.name, r.levelname, r.getMessage()) for r in caplog.records]}"
    )
