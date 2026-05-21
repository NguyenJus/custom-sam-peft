"""Shared CLI logging setup. Idempotent — safe to call from every command."""

from __future__ import annotations

import logging


def configure_logging(verbose: bool) -> None:
    """Configure root logging for a custom-sam-peft CLI invocation."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,  # Override pytest/dev-tool prior config.
    )
