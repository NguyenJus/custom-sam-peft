"""Checkpoint sweep orchestration for the proxy-vs-exact gate (#277).

Discovers step_*.pt checkpoint directories under a checkpoints/ directory,
runs lite eval twice per checkpoint (exact and proxy mode) via an injectable
``eval_fn``, and returns a list of SweepRecord objects ready for
:func:`~custom_sam_peft.eval.proxy_gate.evaluate_gate`.

The eval_fn signature is::

    def eval_fn(checkpoint: Path, exact: bool) -> float | tuple[float, float]

It returns either:
  - a single float: the mAP value, or
  - a tuple ``(mAP, mAP_50)``: both headline metrics.

The default implementation (``default_eval_fn``) is imported lazily so the
module is importable without triggering the full model-loading chain.

CSP_LITE_EXACT_MAP toggling:
  - For the exact pass: ``os.environ["CSP_LITE_EXACT_MAP"] = "1"``
  - For the proxy pass: ``del os.environ["CSP_LITE_EXACT_MAP"]`` (or ensure
    it is absent).
  - The original env value is ALWAYS restored after the sweep, whether or not
    an exception occurs.

Checkpoint discovery:
  Scans for directories matching ``step_XXXXXXXX.pt`` under ``checkpoints_dir``.
  The step number is the zero-padded integer in the name.  Directories are
  returned sorted by step number ascending (numerically, not lexicographically).
  Non-matching entries and plain files are ignored.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable
from pathlib import Path

from custom_sam_peft.eval.proxy_gate import SweepRecord

# Pattern for checkpoint directory names: step_XXXXXXXX.pt
_STEP_DIR_RE = re.compile(r"^step_(\d+)\.pt$")


def discover_checkpoints(checkpoints_dir: Path) -> list[Path]:
    """Return step_*.pt checkpoint directories sorted by step number (ascending).

    Args:
        checkpoints_dir: directory that directly contains step_*.pt subdirs.

    Returns:
        List of Path objects (directories), sorted by step number.

    Raises:
        FileNotFoundError: if checkpoints_dir does not exist.
    """
    if not checkpoints_dir.exists():
        raise FileNotFoundError(
            f"discover_checkpoints: checkpoints directory not found: {checkpoints_dir}"
        )

    matches: list[tuple[int, Path]] = []
    for entry in checkpoints_dir.iterdir():
        if not entry.is_dir():
            continue
        m = _STEP_DIR_RE.match(entry.name)
        if m:
            step_num = int(m.group(1))
            matches.append((step_num, entry))

    matches.sort(key=lambda t: t[0])
    return [p for _, p in matches]


# Type alias for the injectable eval function.
EvalFn = Callable[[Path, bool], "float | tuple[float, float]"]


def _toggle_exact_env(exact: bool, saved: str | None) -> None:
    """Set or restore CSP_LITE_EXACT_MAP based on the exact flag.

    - exact=True  → os.environ["CSP_LITE_EXACT_MAP"] = "1"
    - exact=False → remove CSP_LITE_EXACT_MAP from os.environ (if present)
    """
    if exact:
        os.environ["CSP_LITE_EXACT_MAP"] = "1"
    else:
        os.environ.pop("CSP_LITE_EXACT_MAP", None)


def run_sweep(
    checkpoints_dir: Path,
    eval_fn: EvalFn,
) -> list[SweepRecord]:
    """Run lite eval twice per checkpoint, collecting exact and proxy mAP.

    For each checkpoint (sorted by step ascending):
      1. Set ``CSP_LITE_EXACT_MAP=1``, call ``eval_fn(ckpt, exact=True)``.
      2. Unset ``CSP_LITE_EXACT_MAP``, call ``eval_fn(ckpt, exact=False)``.

    The original value of ``CSP_LITE_EXACT_MAP`` is saved before the sweep
    and restored afterwards (via a finally block).

    The eval_fn may return a plain float (mAP only) or a ``(mAP, mAP_50)``
    tuple; the record stores both when available.

    Args:
        checkpoints_dir: directory containing step_*.pt checkpoint subdirs.
        eval_fn: callable accepting (checkpoint_path, exact: bool) and
            returning mAP or (mAP, mAP_50).  Default: ``run_eval`` wrapper.

    Returns:
        List of SweepRecord objects, one per checkpoint, ordered by step.
    """
    checkpoints = discover_checkpoints(checkpoints_dir)
    saved_env = os.environ.get("CSP_LITE_EXACT_MAP")

    records: list[SweepRecord] = []
    try:
        for ckpt in checkpoints:
            step_num = int(_STEP_DIR_RE.match(ckpt.name).group(1))  # type: ignore[union-attr]

            # --- Exact pass ---
            _toggle_exact_env(exact=True, saved=saved_env)
            exact_result = eval_fn(ckpt, True)
            if isinstance(exact_result, tuple):
                exact_map, exact_map_50 = exact_result
            else:
                exact_map = float(exact_result)
                exact_map_50 = None

            # --- Proxy pass ---
            _toggle_exact_env(exact=False, saved=saved_env)
            proxy_result = eval_fn(ckpt, False)
            if isinstance(proxy_result, tuple):
                proxy_map, proxy_map_50 = proxy_result
            else:
                proxy_map = float(proxy_result)
                proxy_map_50 = None

            records.append(
                SweepRecord(
                    checkpoint=ckpt,
                    step=step_num,
                    exact_map=float(exact_map),
                    proxy_map=float(proxy_map),
                    exact_map_50=float(exact_map_50) if exact_map_50 is not None else None,
                    proxy_map_50=float(proxy_map_50) if proxy_map_50 is not None else None,
                )
            )
    finally:
        # Restore the original env state unconditionally.
        if saved_env is None:
            os.environ.pop("CSP_LITE_EXACT_MAP", None)
        else:
            os.environ["CSP_LITE_EXACT_MAP"] = saved_env

    return records
