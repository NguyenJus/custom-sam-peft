"""Unit tests for LocalTracker — stdlib-only metrics-to-disk tracker."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from custom_sam_peft.tracking.local import LocalTracker


def _read_rows(run_dir: Path) -> list[dict]:
    text = (run_dir / "metrics.jsonl").read_text()
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _tracker() -> LocalTracker:
    return LocalTracker(MagicMock())


def test_start_run_fresh_creates_metrics_jsonl(tmp_path: Path) -> None:
    t = _tracker()
    t.start_run(tmp_path, {"some": "config"})
    assert (tmp_path / "metrics.jsonl").is_file()
    assert _read_rows(tmp_path) == []
    t.close()


def test_log_scalars_appends_one_json_line_per_call(tmp_path: Path) -> None:
    t = _tracker()
    t.start_run(tmp_path, {})
    t.log_scalars(0, {"loss": 1.5})
    t.log_scalars(1, {"loss": 1.0, "lr": 0.001})
    t.close()
    rows = _read_rows(tmp_path)
    assert len(rows) == 2
    assert rows[0]["step"] == 0
    assert "wall_time" in rows[0]
    assert rows[0]["loss"] == 1.5
    assert rows[1]["step"] == 1
    assert rows[1]["loss"] == 1.0
    assert rows[1]["lr"] == 0.001


def test_log_scalars_filters_non_finite(tmp_path: Path) -> None:
    t = _tracker()
    t.start_run(tmp_path, {})
    t.log_scalars(0, {"loss": float("inf"), "bad": float("nan"), "good": 2.0})
    t.close()
    rows = _read_rows(tmp_path)
    assert rows == [{"step": 0, "wall_time": rows[0]["wall_time"], "good": 2.0}] or (
        rows[0]["good"] == 2.0 and "loss" not in rows[0] and "bad" not in rows[0]
    )


def test_log_scalars_before_start_run_raises(tmp_path: Path) -> None:
    t = _tracker()
    with pytest.raises(RuntimeError, match=r"start_run\(\) must be called before log_scalars\(\)"):
        t.log_scalars(0, {"loss": 1.0})


def test_close_is_idempotent(tmp_path: Path) -> None:
    t = _tracker()
    t.start_run(tmp_path, {})
    t.close()
    t.close()  # must not raise


def test_log_images_is_noop(tmp_path: Path) -> None:
    t = _tracker()
    t.start_run(tmp_path, {})
    t.log_images(0, {"panel": np.zeros((4, 4, 3), dtype=np.uint8)})
    t.close()
    # metrics-only: no scalar rows, no extra files written
    assert _read_rows(tmp_path) == []
    assert not (tmp_path / "panels").exists()


def test_wants_images_is_false() -> None:
    assert LocalTracker.wants_images is False


def _write_rows(run_dir: Path, steps: list[int]) -> None:
    lines = [json.dumps({"step": s, "wall_time": 1.0, "loss": float(s)}) for s in steps]
    (run_dir / "metrics.jsonl").write_text("\n".join(lines) + "\n")


def test_resume_keeps_only_rows_before_resume_step(tmp_path: Path) -> None:
    # Interrupted run logged steps 0..5; last checkpoint was step_4.
    _write_rows(tmp_path, [0, 1, 2, 3, 4, 5])
    ckpt = tmp_path / "checkpoints" / "step_4"
    ckpt.mkdir(parents=True)

    t = _tracker()
    t.start_run(tmp_path, {}, resume_from=ckpt)
    # Re-walk re-logs step 4 onward; appends must not duplicate prior steps.
    t.log_scalars(4, {"loss": 4.0})
    t.log_scalars(5, {"loss": 5.0})
    t.close()

    steps = [r["step"] for r in _read_rows(tmp_path)]
    assert steps == [0, 1, 2, 3, 4, 5], f"expected dedup to keep <4 then re-append; got {steps}"


def test_resume_fallback_when_name_not_step_n(tmp_path: Path) -> None:
    _write_rows(tmp_path, [0, 1, 2])
    ckpt = tmp_path / "checkpoints" / "latest"  # does NOT match step_<N>
    ckpt.mkdir(parents=True)

    t = _tracker()
    t.start_run(tmp_path, {}, resume_from=ckpt)  # warns, appends without dedup
    t.log_scalars(3, {"loss": 3.0})
    t.close()

    steps = [r["step"] for r in _read_rows(tmp_path)]
    assert steps == [0, 1, 2, 3], f"fallback must retain all existing rows; got {steps}"
