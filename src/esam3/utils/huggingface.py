"""HuggingFace Hub helpers for token resolution and model download.

This module is a thin wrapper around ``huggingface_hub``: it never calls
``login()`` (no token persistence), and it never logs the resolved token.

Verified against ``huggingface_hub==1.15.0``: real-file materialization is
the default when ``local_dir=`` is supplied to ``snapshot_download``; the
older ``local_dir_use_symlinks`` kwarg has been removed and is not needed.
"""

from __future__ import annotations

import logging
import os

import huggingface_hub

logger = logging.getLogger(__name__)


def resolve_hf_token(token: str | None = None) -> str | None:
    """Resolve an HF token from explicit arg → ``HF_TOKEN`` env → cached creds.

    Returns the token string, or ``None`` if none is available. Never persists
    the token; never logs its value; never calls ``huggingface_hub.login()``.

    Probe order (returns the first non-empty value; never falls through after
    a hit):
      1. ``token`` argument, if truthy.
      2. ``os.environ.get("HF_TOKEN")``, if truthy.
      3. ``huggingface_hub.get_token()``, which reads
         ``~/.cache/huggingface/token`` written by ``huggingface-cli login``.
    """
    if token:
        return token
    env = os.environ.get("HF_TOKEN")
    if env:
        return env
    return huggingface_hub.get_token() or None
