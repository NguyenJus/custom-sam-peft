"""LocalTracker — stdlib-only metrics-to-disk tracker. Backend "local".

Persists the per-step scalar time-series to ``run_dir/metrics.jsonl`` (one
JSON object per line) using only the standard library. Metrics-only by owner
decision: ``log_images`` is a no-op and ``wants_images`` is False, so the
trainer skips panel-render compute for this backend (see Change 3).
"""

from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, TextIO

from custom_sam_peft._registry import register
from custom_sam_peft.config.schema import TrainConfig

if TYPE_CHECKING:
    import numpy as np

_LOG = logging.getLogger(__name__)

_METRICS_FILENAME = "metrics.jsonl"


class LocalTracker:
    """Tracker backend writing scalar rows to run_dir/metrics.jsonl."""

    wants_images = False

    def __init__(self, cfg: TrainConfig) -> None:
        self._cfg = cfg
        self._run_dir: Path | None = None
        self._fh: TextIO | None = None
        self._closed = False

    def start_run(
        self,
        run_dir: Path,
        config: dict[str, Any],
        resume_from: Path | None = None,
    ) -> None:
        self._run_dir = run_dir
        metrics_path = run_dir / _METRICS_FILENAME
        if resume_from is None:
            # Fresh run: create/truncate, then open for append.
            self._fh = metrics_path.open("w")
            return
        # Resume: run_dir is the old run dir (Change 1), so metrics.jsonl
        # already exists. Drop rows the interrupted run logged AFTER its last
        # checkpoint (step >= resume_step) so resume does not duplicate them.
        resume_step = self._parse_resume_step(resume_from)
        if resume_step is not None and metrics_path.is_file():
            kept: list[str] = []
            for line in metrics_path.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    if int(row["step"]) < resume_step:
                        kept.append(line)
                except (json.JSONDecodeError, KeyError, ValueError, TypeError):
                    # Preserve unparseable lines defensively.
                    kept.append(line)
            metrics_path.write_text("\n".join(kept) + ("\n" if kept else ""))
        self._fh = metrics_path.open("a")

    @staticmethod
    def _parse_resume_step(resume_from: Path) -> int | None:
        """Parse N from a checkpoint dir name of the form ``step_<N>``.

        Returns None (warn + plain append, no dedup) when the name does not
        match — defensive; never crashes the run.
        """
        name = resume_from.name
        prefix = "step_"
        if name.startswith(prefix):
            suffix = name[len(prefix) :]
            if suffix.isdigit():
                return int(suffix)
        _LOG.warning(
            "LocalTracker: resume_from name %r does not match 'step_<N>'; "
            "appending to metrics.jsonl without dedup.",
            name,
        )
        return None

    def log_scalars(self, step: int, values: dict[str, float]) -> None:
        if self._fh is None:
            raise RuntimeError("start_run() must be called before log_scalars()")
        finite = {k: v for k, v in values.items() if math.isfinite(v)}
        row = {"step": step, "wall_time": time.time(), **finite}
        self._fh.write(json.dumps(row) + "\n")
        self._fh.flush()

    def log_images(self, step: int, images: dict[str, np.ndarray[Any, Any]]) -> None:
        # Metrics-only: no-op. Never called for "local" because of the
        # wants_images gate in the trainer (Change 3).
        return None

    def close(self) -> None:
        if self._closed:
            return
        if self._fh is not None:
            self._fh.flush()
            self._fh.close()
        self._closed = True


@register("tracker", "local")
def build_local(cfg: TrainConfig) -> LocalTracker:
    """Factory called by build_tracker for backend='local'."""
    return LocalTracker(cfg)
