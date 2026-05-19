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
from pathlib import Path

import huggingface_hub

logger = logging.getLogger(__name__)


def resolve_hf_token(token: str | None = None) -> str | None:
    """Resolve an HF token from explicit arg → ``HF_TOKEN`` env → cached creds.

    Returns the token string, or ``None`` if none is available. Never persists
    the token; never logs its value; never calls ``huggingface_hub.login()``.
    """
    if token:
        return token
    env = os.environ.get("HF_TOKEN")
    if env:
        return env
    return huggingface_hub.get_token() or None


def download_model(
    repo_id: str,
    local_dir: Path,
    *,
    token: str | None = None,
    revision: str | None = None,
    force: bool = False,
) -> Path:
    """Snapshot-download ``repo_id`` into ``local_dir`` if not already present.

    Idempotent unless ``force=True``: when ``local_dir`` exists and is
    non-empty, returns immediately without contacting the Hub.

    The consumer who knows the expected filename should re-check file-level
    presence after this returns — the "non-empty" skip condition is
    intentionally weak.

    Returns ``local_dir`` on success.

    Error mapping (verified against ``huggingface_hub==1.15.0``):
      - ``huggingface_hub.errors.GatedRepoError``        → ``RuntimeError``
      - ``huggingface_hub.errors.RepositoryNotFoundError`` → ``RuntimeError``
      - ``huggingface_hub.errors.HfHubHTTPError`` (generic) → ``RuntimeError``
    Other exception types (including ``OSError`` for network timeouts) are
    NOT wrapped. The resolved token is never embedded in any mapped message.
    """
    from huggingface_hub.errors import (
        GatedRepoError,
        HfHubHTTPError,
        RepositoryNotFoundError,
    )

    if not force and local_dir.exists() and any(local_dir.iterdir()):
        return local_dir

    resolved = resolve_hf_token(token)
    logger.info("fetching %s → %s", repo_id, local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    try:
        huggingface_hub.snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_dir),
            revision=revision,
            token=resolved,
        )
    except GatedRepoError as e:
        raise RuntimeError(
            f"could not download '{repo_id}': the repo is gated. "
            f"Accept the license at https://huggingface.co/{repo_id} and then "
            f"`export HF_TOKEN=<your-token>`."
        ) from e
    except RepositoryNotFoundError as e:
        raise RuntimeError(
            f"could not download '{repo_id}': repo not found, or your token "
            f"lacks access. Check the repo id and verify your token."
        ) from e
    except HfHubHTTPError as e:
        status: str
        resp = getattr(e, "response", None)
        status = str(getattr(resp, "status_code", "?"))
        raise RuntimeError(
            f"could not download '{repo_id}': Hub request failed ({status}). "
            f"Check network and `export HF_TOKEN=...` if the repo is private/gated."
        ) from e
    return local_dir
