"""Shared CLI logging setup. Idempotent — safe to call from every command."""

from __future__ import annotations

import logging

from rich.console import Console
from rich.logging import RichHandler


def configure_logging(verbose: bool, console: Console | None = None) -> None:
    """Configure root logging for a custom-sam-peft CLI invocation.

    When ``console`` is provided, attaches a ``RichHandler`` backed by that
    console so log output flows through an existing rich Live display. The
    default (``console=None``) uses a plain ``basicConfig`` format — unchanged
    from before.

    Note: ``progress_session`` uses its own handler-attachment path (spec §7.2)
    and does not call this function with ``console``. This kwarg exists for
    library callers or custom CLI wrappers that inject a console without a full
    session.
    """
    level = logging.DEBUG if verbose else logging.INFO
    if console is not None:
        handler = RichHandler(console=console, rich_tracebacks=True, show_path=False)
        logging.basicConfig(level=level, handlers=[handler], force=True)
    else:
        logging.basicConfig(
            level=level,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
            force=True,  # Override pytest/dev-tool prior config.
        )
