# tests/unit/test_train_loop_keys.py
"""StepResult/_ScalarWindow parametrized loss-key set (§10.1)."""

from __future__ import annotations

from custom_sam_peft.train.loop import StepResult, _ScalarWindow

INSTANCE_KEYS = ("mask", "box", "obj", "presence", "total")
SEMANTIC_KEYS = ("ce", "region", "total")


def test_step_result_empty_defaults_to_instance_keys():
    r = StepResult.empty()
    assert set(r.losses.keys()) == set(INSTANCE_KEYS)  # byte-identical instance default


def test_step_result_empty_accepts_semantic_keys():
    r = StepResult.empty(loss_keys=SEMANTIC_KEYS)
    assert set(r.losses.keys()) == set(SEMANTIC_KEYS)


def test_scalar_window_instance_keys_unchanged():
    w = _ScalarWindow()  # default
    out = w.flush()
    # instance loss/* keys present exactly as today
    assert "loss/mask" in out and "loss/total" in out


def test_scalar_window_semantic_keys():
    w = _ScalarWindow(loss_keys=SEMANTIC_KEYS)
    r = StepResult(
        losses={"ce": 1.0, "region": 2.0, "total": 3.0},
        n_classes=3,
        grad_norm=0.5,
        skipped=False,
        nan_streak=0,
        images_processed=2,
    )
    w.update(r, lr=1e-4)
    out = w.flush()
    assert "loss/ce" in out and "loss/region" in out and "loss/total" in out
    assert "loss/mask" not in out


def test_scalar_window_reset_preserves_keys_after_flush():
    # flush() resets via __init__; the reset MUST re-seed the SAME loss_keys.
    w = _ScalarWindow(loss_keys=SEMANTIC_KEYS)
    w.flush()
    out = w.flush()  # second flush after reset
    assert "loss/ce" in out and "loss/region" in out and "loss/total" in out
    assert "loss/mask" not in out
