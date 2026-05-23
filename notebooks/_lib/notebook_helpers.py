"""Helpers used by `notebooks/custom_sam_peft_train.ipynb` for env detection,
local-checkpoint short-circuit, and HF-token resolution.

CLI never imports this module. Tests and the notebook do.

Note: ``utils/huggingface.py::resolve_hf_token`` is the silent best-effort
resolver used by ``download_model`` — it returns the token or ``None`` and
never raises. ``notebook_helpers.py::resolve_hf_token_for_notebook`` (below)
is an env-aware resolver for notebook contexts: it short-circuits when a
local checkpoint is present and raises ``RuntimeError`` with Colab- or
RunPod-specific instructions when the token is missing. The two are
deliberately not merged; their failure semantics differ.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Literal

_LOG = logging.getLogger(__name__)

Env = Literal["colab", "runpod", "unknown"]

_COLAB_ERR = "Set HF_TOKEN in Colab Secrets (left sidebar → 🔑)."
_RUNPOD_ERR = (
    "Set HF_TOKEN in your pod's Environment Variables, "
    "or mount a network volume containing models/sam3.1/sam3.1_multiplex.pt."
)
_UNKNOWN_ERR = "Set HF_TOKEN in your shell environment (export HF_TOKEN=…)."


def detect_env() -> Env:
    """Best-effort environment detection from env vars.

    - 'colab' if os.environ.get('COLAB_GPU') is set (any value).
    - 'runpod' elif os.environ.get('RUNPOD_POD_ID') is set.
    - 'unknown' otherwise.
    """
    if os.environ.get("COLAB_GPU") is not None:
        return "colab"
    if os.environ.get("RUNPOD_POD_ID") is not None:
        return "runpod"
    return "unknown"


def check_local_checkpoint(local_dir: Path, checkpoint_file: str) -> bool:
    """Return True iff `(local_dir / checkpoint_file).is_file()`."""
    return (Path(local_dir) / checkpoint_file).is_file()


def _resolve_colab_token() -> str | None:
    try:
        from google.colab import userdata
    except ImportError:
        return None
    result: str | None = userdata.get("HF_TOKEN")
    return result


def resolve_hf_token_for_notebook(env: Env, local_present: bool) -> str | None:
    """Resolve the HF token according to environment and local-checkpoint state.

    - If `local_present` is True: log 'local checkpoint detected — skipping HF
      auth' and return None.
    - Else, fetch the token from the env-appropriate source. Missing token
      raises RuntimeError with an env-specific friendly message.
    """
    if local_present:
        _LOG.info("local checkpoint detected — skipping HF auth")
        return None

    if env == "colab":
        token = _resolve_colab_token()
        if not token:
            raise RuntimeError(_COLAB_ERR)
        return token

    if env == "runpod":
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise RuntimeError(_RUNPOD_ERR)
        return token

    # unknown
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError(_UNKNOWN_ERR)
    return token
