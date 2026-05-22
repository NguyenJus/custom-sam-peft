"""Process-global progress handle with rich/plain/off modes.

Public API: ``progress``, ``progress_session``, ``resolve_mode``,
``ProgressMode``, ``ProgressKind``.
"""

from __future__ import annotations

import logging
import os
import signal
import time
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from enum import StrEnum
from typing import Any

from rich.console import Console
from rich.logging import RichHandler
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeRemainingColumn,
)


class ProgressMode(StrEnum):
    ON = "on"
    OFF = "off"
    PLAIN = "plain"


class ProgressKind(StrEnum):
    TRAIN = "train"
    EVAL = "eval"
    PREDICT = "predict"
    EXPORT_MERGE = "export-merge"


def resolve_mode(
    cli_flag: str | None,
    env: Mapping[str, str],
    stdout_isatty: bool,
    is_jupyter: bool,
) -> ProgressMode:
    """Resolve the effective ProgressMode. Pure function; no side effects.

    Precedence: explicit --progress flag > CSP_NO_PROGRESS env var > auto fallback.
    """
    if cli_flag is not None and cli_flag != "auto":
        return ProgressMode(cli_flag)
    if env.get("CSP_NO_PROGRESS") == "1":
        return ProgressMode.OFF
    if is_jupyter:
        return ProgressMode.PLAIN
    if not stdout_isatty:
        return ProgressMode.PLAIN
    return ProgressMode.ON


class _NoOpHandle:
    """No-op progress handle used when no session is active (default)."""

    @property
    def console(self) -> Console:
        return Console()

    def advance_outer(self, n: int = 1) -> None:
        pass

    def advance_inner(self, n: int = 1) -> None:
        pass

    def reset_inner(self, total: int | None = None) -> None:
        pass

    def update_postfix(self, **kwargs: Any) -> None:
        pass

    @contextmanager
    def push_subtask(self, label: str, total: int) -> Generator[None, None, None]:
        yield


class _ProgressHandle:
    """Live progress handle backed by rich.Progress."""

    def __init__(
        self,
        rich_progress: Progress,
        outer_task_id: TaskID | None,
        inner_task_id: TaskID,
        kind: ProgressKind,
        total_batches_per_epoch: int,
        log_every: int = 50,
    ) -> None:
        self._progress = rich_progress
        self._outer = outer_task_id
        self._inner = inner_task_id
        self._kind = kind
        self._total_batches = total_batches_per_epoch
        self._log_every = log_every
        self._step = 0
        self._epoch = 0
        self._plain_postfix: dict[str, Any] = {}

    @property
    def console(self) -> Console:
        return self._progress.console

    def advance_outer(self, n: int = 1) -> None:
        if self._outer is not None:
            self._progress.advance(self._outer, n)
        self._epoch += n

    def reset_inner(self, total: int | None = None) -> None:
        update_kwargs: dict[str, Any] = {"completed": 0}
        if total is not None:
            update_kwargs["total"] = total
            self._total_batches = total
        self._progress.update(self._inner, **update_kwargs)
        self._step = 0

    def advance_inner(self, n: int = 1) -> None:
        self._progress.advance(self._inner, n)
        self._step += n

    def update_postfix(self, **kwargs: Any) -> None:
        self._plain_postfix.update(kwargs)
        # Build description string for rich display.
        desc = " ".join(f"{k}={v}" for k, v in kwargs.items())
        self._progress.update(self._inner, description=desc)

    @contextmanager
    def push_subtask(self, label: str, total: int) -> Generator[None, None, None]:
        task_id = self._progress.add_task(label, total=total)
        try:
            yield
        finally:
            self._progress.remove_task(task_id)


def format_eta(seconds: float) -> str:
    """Format seconds as H:MM:SS."""
    seconds = int(seconds)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    return f"{hours}:{minutes:02d}:{secs:02d}"


class _PlainHandle:
    """Plain-mode progress handle — emits one log line per log_every window.

    Used when mode == ProgressMode.PLAIN. No rich.Progress; no ANSI; pure
    stdlib logging output. The line format is the spec §4 contract:
        progress: <kind> epoch=E/TE step=S/TS loss=L it/s=R eta=ETA
    """

    def __init__(
        self,
        kind: ProgressKind,
        total_batches_per_epoch: int,
        total_epochs: int | None,
        log_every: int = 50,
    ) -> None:
        self._kind = kind
        self._total_batches = total_batches_per_epoch
        self._total_epochs = total_epochs
        self._log_every = log_every
        self._step = 0
        self._epoch = 0
        self._postfix: dict[str, Any] = {}
        self._logger = logging.getLogger("custom_sam_peft.progress")
        self._start_time = time.monotonic()

    @property
    def console(self) -> Console:
        return Console()  # plain console; logs route through stdlib logger

    def advance_outer(self, n: int = 1) -> None:
        self._epoch += n

    def reset_inner(self, total: int | None = None) -> None:
        if total is not None:
            self._total_batches = total
        self._step = 0

    def advance_inner(self, n: int = 1) -> None:
        self._step += n
        if self._step % self._log_every == 0 or self._step == self._total_batches:
            self._emit()

    def update_postfix(self, **kwargs: Any) -> None:
        self._postfix.update(kwargs)

    @contextmanager
    def push_subtask(self, label: str, total: int) -> Generator[None, None, None]:
        self._logger.info("progress: %s subtask=%s start total=%d", self._kind.value, label, total)
        try:
            yield
        finally:
            self._logger.info("progress: %s subtask=%s end", self._kind.value, label)

    def _emit(self) -> None:
        """Emit one progress line in the spec §4 format (global step)."""
        if self._total_epochs is not None and self._total_batches > 0:
            global_step = self._epoch * self._total_batches + self._step
            global_total = self._total_epochs * self._total_batches
            epoch_str = f"epoch={self._epoch + 1}/{self._total_epochs}"
            step_str = f"step={global_step}/{global_total}"
        else:
            epoch_str = ""
            step_str = f"step={self._step}/{self._total_batches}"

        postfix_parts = []
        if "loss" in self._postfix:
            postfix_parts.append(f"loss={self._postfix['loss']:.3f}")
        if "it_s" in self._postfix:
            postfix_parts.append(f"it/s={self._postfix['it_s']:.1f}")
        elapsed = time.monotonic() - self._start_time
        if self._step > 0:
            total_for_eta = (
                self._total_epochs * self._total_batches
                if self._total_epochs is not None and self._total_batches > 0
                else self._total_batches
            )
            current = (
                self._epoch * self._total_batches + self._step
                if self._total_epochs is not None and self._total_batches > 0
                else self._step
            )
            eta_seconds = elapsed * (total_for_eta - current) / max(current, 1)
            eta = format_eta(eta_seconds)
            postfix_parts.append(f"eta={eta}")

        parts = [f"progress: {self._kind.value}"]
        if epoch_str:
            parts.append(epoch_str)
        parts.append(step_str)
        parts.extend(postfix_parts)
        self._logger.info(" ".join(parts))


def _silence_third_party_progress() -> None:
    """Suppress HF / datasets progress output. Idempotent — safe to call twice."""
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "warning")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    try:
        import datasets as _ds

        _ds.disable_progress_bar()
    except ImportError:
        # ``datasets`` is an optional dependency; absence is a no-op for this helper.
        pass


class _State:
    """Module-level mutable state for the process-global progress singleton.

    Holding state on an object (rather than module-level ``global`` rebinds)
    means ``from _progress import progress as P`` callers see live updates —
    the proxy below dereferences ``_state.handle`` on every call.
    """

    def __init__(self) -> None:
        self.handle: _NoOpHandle | _ProgressHandle | _PlainHandle = _NoOpHandle()
        self.session_active: bool = False


_state = _State()


class _ProgressProxy:
    """Stable user-facing handle that forwards every call to ``_state.handle``.

    Call sites import this once (``from ... import progress as P``); the
    binding is stable, so swapping ``_state.handle`` inside
    ``progress_session`` is visible to every caller without re-importing.
    """

    @property
    def console(self) -> Console:
        return _state.handle.console

    def advance_outer(self, n: int = 1) -> None:
        _state.handle.advance_outer(n)

    def advance_inner(self, n: int = 1) -> None:
        _state.handle.advance_inner(n)

    def reset_inner(self, total: int | None = None) -> None:
        _state.handle.reset_inner(total)

    def update_postfix(self, **kwargs: Any) -> None:
        _state.handle.update_postfix(**kwargs)

    @contextmanager
    def push_subtask(self, label: str, total: int) -> Generator[None, None, None]:
        with _state.handle.push_subtask(label, total):
            yield


progress = _ProgressProxy()


@contextmanager
def progress_session(
    kind: ProgressKind,
    total_batches_per_epoch: int,
    mode: ProgressMode,
    total_epochs: int | None = None,
    log_every: int = 50,
) -> Generator[None, None, None]:
    """Context manager that activates the process-global progress handle.

    Opens a rich.Progress live display for mode=ON; emits plain progress lines
    for mode=PLAIN; suppresses progress output for mode=OFF.

    Raises RuntimeError if called while a session is already active (nesting
    is not supported).
    """
    if _state.session_active:
        raise RuntimeError("nested session: a progress_session is already active in this process")
    _state.session_active = True
    _silence_third_party_progress()

    root_logger = logging.getLogger()
    prior_handlers = list(root_logger.handlers)

    rich_prog: Progress | None = None
    handle: _ProgressHandle | _PlainHandle | _NoOpHandle

    if mode == ProgressMode.ON:
        rich_prog = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeRemainingColumn(),
            console=Console(stderr=False),
            transient=False,
        )
        rich_prog.start()

        # Attach RichHandler so logs scroll above the pinned bar.
        root_logger.handlers.clear()
        root_logger.addHandler(
            RichHandler(
                console=rich_prog.console,
                rich_tracebacks=True,
                show_path=False,
            )
        )
        logging.captureWarnings(True)

        outer_id: TaskID | None = None
        if total_epochs is not None:
            outer_id = rich_prog.add_task(
                f"{kind.value} epoch",
                total=total_epochs,
            )
        inner_id = rich_prog.add_task(
            f"{kind.value} step",
            total=total_batches_per_epoch,
        )
        handle = _ProgressHandle(
            rich_progress=rich_prog,
            outer_task_id=outer_id,
            inner_task_id=inner_id,
            kind=kind,
            total_batches_per_epoch=total_batches_per_epoch,
            log_every=log_every,
        )
    elif mode == ProgressMode.PLAIN:
        handle = _PlainHandle(
            kind=kind,
            total_batches_per_epoch=total_batches_per_epoch,
            total_epochs=total_epochs,
            log_every=log_every,
        )
    else:  # ProgressMode.OFF
        handle = _NoOpHandle()

    prior_sigint = signal.getsignal(signal.SIGINT)

    def _sigint_handler(signum: int, frame: Any) -> None:
        if rich_prog is not None:
            rich_prog.stop()
        if callable(prior_sigint):
            prior_sigint(signum, frame)
        else:
            raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _sigint_handler)

    _state.handle = handle
    try:
        yield
    finally:
        _state.handle = _NoOpHandle()
        _state.session_active = False
        signal.signal(signal.SIGINT, prior_sigint)
        if rich_prog is not None:
            rich_prog.stop()
        root_logger.handlers.clear()
        for h in prior_handlers:
            root_logger.addHandler(h)
        logging.captureWarnings(False)
