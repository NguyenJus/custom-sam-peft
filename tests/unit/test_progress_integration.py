"""Tests I-J: integration smoke test and ruff T201 lint rule guard (spec s9).

All CPU-only. No GPU markers.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

from custom_sam_peft.cli._progress import (
    ProgressKind,
    ProgressMode,
    _NoOpHandle,
    _state,
    progress_session,
)

# ---------------------------------------------------------------------------
# Test I: fake trainer smoke test
# ---------------------------------------------------------------------------


def test_fake_trainer_smoke() -> None:
    """Test I: dummy train loop with progress_session(kind=TRAIN, mode=ON).

    Verifies the real data flow contract: advance_outer twice, advance_inner
    six times total (3 batches x 2 epochs), update_postfix at epoch end.
    """
    total_epochs = 2
    batches_per_epoch = 3

    outer_advances = 0
    inner_advances = 0

    with progress_session(
        kind=ProgressKind.TRAIN,
        total_epochs=total_epochs,
        total_batches_per_epoch=batches_per_epoch,
        mode=ProgressMode.ON,
    ):
        handle = _state.handle
        assert not isinstance(handle, _NoOpHandle), "expected live handle inside session"

        for epoch in range(total_epochs):
            handle.advance_outer()
            outer_advances += 1
            handle.reset_inner()

            for _ in range(batches_per_epoch):
                handle.advance_inner()
                inner_advances += 1

            handle.update_postfix(loss=0.5 - epoch * 0.1, it_s=2.3)

    assert isinstance(_state.handle, _NoOpHandle), "expected _NoOpHandle after session exits"
    assert outer_advances == total_epochs, f"expected {total_epochs} outer advances"
    assert inner_advances == total_epochs * batches_per_epoch, (
        f"expected {total_epochs * batches_per_epoch} inner advances"
    )


# ---------------------------------------------------------------------------
# Test J: ruff T201 lint rule guard
# ---------------------------------------------------------------------------


def test_ruff_t201_lint_rule() -> None:
    """Test J: a file with bare print() fails T201; one with # noqa: T201 passes.

    Guards the lint-config change against accidental reversion.
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        failing = tmp_path / "bad.py"
        failing.write_text("print('x')\n")

        result_fail = subprocess.run(  # noqa: S603
            [sys.executable, "-m", "ruff", "check", "--select", "T201", str(failing)],
            capture_output=True,
            text=True,
        )
        assert result_fail.returncode != 0, (
            f"Expected ruff T201 to fail on bare print(), got returncode=0.\n"
            f"stdout: {result_fail.stdout}\nstderr: {result_fail.stderr}"
        )

        passing = tmp_path / "good.py"
        passing.write_text(
            textwrap.dedent("""\
                import json
                x = {"a": 1}
                print(json.dumps(x))  # noqa: T201
            """)
        )

        result_pass = subprocess.run(  # noqa: S603
            [sys.executable, "-m", "ruff", "check", "--select", "T201", str(passing)],
            capture_output=True,
            text=True,
        )
        assert result_pass.returncode == 0, (
            f"Expected ruff T201 to pass with # noqa: T201, "
            f"got returncode={result_pass.returncode}.\n"
            f"stdout: {result_pass.stdout}\nstderr: {result_pass.stderr}"
        )
