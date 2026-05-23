"""Tests B-H: _progress.py module lifecycle, routing, and env handling (spec s9).

All CPU-only. No GPU markers.
"""

from __future__ import annotations

import io
import logging
import os
import re
import signal
import time as _time
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from custom_sam_peft.cli._progress import (
    ProgressKind,
    ProgressMode,
    _NoOpHandle,
    _silence_third_party_progress,
    _state,
    progress,
    progress_session,
)

# ---------------------------------------------------------------------------
# Test B: no-op default
# ---------------------------------------------------------------------------


def test_no_op_default() -> None:
    """Test B: P.* calls outside any session are no-ops -- no exception, no terminal writes."""
    P = progress
    assert isinstance(_state.handle, _NoOpHandle), (
        "expected _NoOpHandle underlying the proxy when no session is active"
    )

    P.advance_outer()
    P.advance_inner()
    P.advance_inner(n=5)
    P.update_postfix(loss=0.5, lr=1e-4)

    # console property returns a plain Console without writing anything.
    con = P.console
    buf = io.StringIO()
    con.file = buf
    con.log("hello")
    # No assertion on content -- just no exception.

    # push_subtask is a no-op context manager.
    with P.push_subtask("test", total=10):
        pass


# ---------------------------------------------------------------------------
# Test C: session lifecycle + nesting raises RuntimeError
# ---------------------------------------------------------------------------


def test_session_lifecycle(caplog: pytest.LogCaptureFixture) -> None:
    """Test C: entering a session attaches RichHandler; exiting restores prior handlers.

    Opening a second session before the first exits raises RuntimeError.
    """
    root = logging.getLogger()
    prior_handler_ids = [id(h) for h in root.handlers]

    with progress_session(  # noqa: SIM117
        kind=ProgressKind.TRAIN,
        total_epochs=1,
        total_batches_per_epoch=10,
        mode=ProgressMode.OFF,  # OFF: no rich.Progress -- just session bookkeeping
    ):
        with pytest.raises(RuntimeError, match="nested session"):
            with progress_session(
                kind=ProgressKind.EVAL,
                total_batches_per_epoch=5,
                mode=ProgressMode.OFF,
            ):
                pass  # unreachable

    restored_ids = [id(h) for h in root.handlers]
    assert restored_ids == prior_handler_ids, (
        f"handlers after session: {restored_ids} != prior {prior_handler_ids}"
    )


# ---------------------------------------------------------------------------
# Test D: log routing through Live (ON mode)
# ---------------------------------------------------------------------------


def test_log_through_live(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test D: logger.info() inside an ON session writes exactly one line to the progress console.

    Verifies that RichHandler is properly attached and routes log output above
    the pinned bar without duplication.
    """
    from rich.console import Console as _RichConsole

    captured = io.StringIO()

    def _fake_console(*args: Any, **kwargs: Any) -> _RichConsole:
        kwargs.pop("stderr", None)
        return _RichConsole(file=captured, force_terminal=True, no_color=True, width=120)

    monkeypatch.setattr("custom_sam_peft.cli._progress.Console", _fake_console)
    # Ensure the root logger passes INFO through to the RichHandler.
    # The test runner may leave the root logger at WARNING (level=30), which would
    # swallow INFO records before they reach any handler.
    root = logging.getLogger()
    monkeypatch.setattr(root, "level", logging.INFO)

    with progress_session(
        kind=ProgressKind.TRAIN,
        total_epochs=1,
        total_batches_per_epoch=10,
        mode=ProgressMode.ON,
    ):
        logging.getLogger("test.d").info("test-D unique-marker-12345")

    # Rich may insert ANSI bold escapes inside log content (e.g. around numeric
    # suffixes). Strip all ANSI CSI sequences before asserting on plain text.
    raw = captured.getvalue()
    output = re.sub(r"\x1b\[[0-9;?]*[mKlhABCDH]", "", raw)
    assert output.count("test-D unique-marker-12345") == 1, (
        f"Expected log line to appear exactly once in progress console output, "
        f"got count={output.count('test-D unique-marker-12345')}.\nFull output:\n{output}"
    )


# ---------------------------------------------------------------------------
# Test E: push_subtask lifecycle
# ---------------------------------------------------------------------------


def test_push_subtask_lifecycle() -> None:
    """Test E: push_subtask adds a task inside the block; the task is removed on exit."""
    P = progress

    # No session: push_subtask is a no-op -- just confirm no exception.
    with P.push_subtask("lite-eval", total=10):
        pass

    # With a session in ON mode, verify the subtask is added then removed.
    with progress_session(
        kind=ProgressKind.TRAIN,
        total_epochs=2,
        total_batches_per_epoch=3,
        mode=ProgressMode.ON,
    ):
        live_handle = _state.handle
        assert not isinstance(live_handle, _NoOpHandle), "expected _ProgressHandle inside session"
        task_count_before = len(
            [t for t in live_handle._progress.tasks if not t.finished]  # type: ignore[union-attr]
        )
        with P.push_subtask("lite-eval", total=10):
            task_count_during = len(live_handle._progress.tasks)  # type: ignore[union-attr]
            assert task_count_during > task_count_before, "subtask should be added during block"
        task_count_after = len(live_handle._progress.tasks)  # type: ignore[union-attr]
        assert task_count_after == task_count_before, "subtask should be removed on exit"


# ---------------------------------------------------------------------------
# Test F: plain mode line snapshot
# ---------------------------------------------------------------------------


def test_plain_line_snapshot(caplog: pytest.LogCaptureFixture) -> None:
    """Test F: frozen snapshot of the plain-mode progress line format.

    Calls into _PlainHandle._emit directly with fixed inputs and asserts the
    captured log line exactly matches the spec s4 contract (with startswith
    for the ETA portion -- time.monotonic() makes sub-second component
    non-deterministic).

    Step field uses the global step (planner decision s4 PLAIN-mode step semantics):
    epoch 2 (0-indexed) x 4530 batches/epoch + 1240 local step = 10300 global step.
    """
    from custom_sam_peft.cli._progress import _PlainHandle

    handle = _PlainHandle(
        kind=ProgressKind.TRAIN,
        total_batches_per_epoch=4530,
        total_epochs=10,
        log_every=50,
    )
    handle._epoch = 2  # epoch 3 of 10 (0-indexed internally)
    handle._step = 1240
    handle._postfix = {"loss": 0.812, "it_s": 2.3}
    # eta = elapsed * (45300 - 10300) / 10300 = elapsed * 35000 / 10300
    # for eta approx 2530s = 0:42:10 -> elapsed approx 744.43 seconds
    handle._start_time = _time.monotonic() - 744.43

    with caplog.at_level(logging.INFO, logger="custom_sam_peft.progress"):
        handle._emit()

    matching = [r for r in caplog.records if "progress: train" in r.getMessage()]
    assert len(matching) == 1, f"expected exactly one progress line, got {len(matching)}"
    msg = matching[0].getMessage()

    expected_prefix = "progress: train epoch=3/10 step=10300/45300 loss=0.812 it/s=2.3 eta=0:42:"
    assert msg.startswith(expected_prefix), (
        f"plain format snapshot mismatch:\n"
        f"  got:      {msg!r}\n"
        f"  expected prefix: {expected_prefix!r}"
    )


# ---------------------------------------------------------------------------
# Test G: _silence_third_party_progress
# ---------------------------------------------------------------------------


def test_silence_third_party_progress(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test G: _silence_third_party_progress sets env vars and calls datasets.disable_progress_bar.

    Verifies that both environment variables are set and that datasets.disable_progress_bar
    is called on each invocation (the function is idempotent but not de-bounced).
    """
    monkeypatch.delenv("TRANSFORMERS_VERBOSITY", raising=False)
    monkeypatch.delenv("HF_HUB_DISABLE_PROGRESS_BARS", raising=False)

    mock_datasets = MagicMock()
    with patch.dict("sys.modules", {"datasets": mock_datasets}):
        _silence_third_party_progress()

    assert os.environ["TRANSFORMERS_VERBOSITY"] == "warning"
    assert os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] == "1"
    mock_datasets.disable_progress_bar.assert_called_once()

    with patch.dict("sys.modules", {"datasets": mock_datasets}):
        _silence_third_party_progress()

    assert mock_datasets.disable_progress_bar.call_count == 2


# ---------------------------------------------------------------------------
# Test H: SIGINT handler -- clean-exit path
# ---------------------------------------------------------------------------


def test_sigint_handler() -> None:
    """Test H (clean-exit path): after progress_session exits normally,
    signal.getsignal(SIGINT) equals the handler registered before the session opened.
    """
    prior_handler = signal.getsignal(signal.SIGINT)

    with progress_session(
        kind=ProgressKind.TRAIN,
        total_epochs=1,
        total_batches_per_epoch=5,
        mode=ProgressMode.OFF,
    ):
        pass

    restored = signal.getsignal(signal.SIGINT)
    assert restored == prior_handler, (
        f"SIGINT handler not restored after session: {restored!r} != {prior_handler!r}"
    )
