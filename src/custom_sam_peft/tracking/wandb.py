"""WandBTracker — wraps wandb.init/log/finish.

Requires the [wandb] optional extra. Eager ImportError surfaces the missing
dependency at construction. Resume continuity is achieved by persisting the
wandb run id into ``run_dir/wandb_run_id.txt`` and re-reading it on resume.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Literal

import numpy as np

from custom_sam_peft._registry import register
from custom_sam_peft.config.schema import TrainConfig
from custom_sam_peft.tracking.base import _validate_image

_WANDB_ID_FILENAME = "wandb_run_id.txt"


class WandBTracker:
    """Tracker backend writing to Weights & Biases."""

    def __init__(self, cfg: TrainConfig) -> None:
        try:
            import wandb  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "tracking.backend='wandb' requires the [wandb] extra. "
                "Install with: pip install 'custom-sam-peft[wandb]'"
            ) from e
        self._cfg = cfg
        self._run: Any | None = None  # wandb.sdk.wandb_run.Run
        self._closed = False

    def start_run(
        self,
        run_dir: Path,
        config: dict[str, Any],
        resume_from: Path | None = None,
    ) -> None:
        import wandb

        run_id, resume_mode = self._maybe_resume_id(resume_from)
        self._run = wandb.init(
            project=self._cfg.tracking.wandb.project,
            entity=self._cfg.tracking.wandb.entity,
            name=run_dir.name,
            dir=str(run_dir),
            config=config,
            id=run_id,
            resume=resume_mode,
        )
        (run_dir / _WANDB_ID_FILENAME).write_text(self._run.id)

    @staticmethod
    def _maybe_resume_id(
        resume_from: Path | None,
    ) -> tuple[str | None, Literal["allow"] | None]:
        """Walk up from ``resume_from`` looking for wandb_run_id.txt.

        Checks ``resume_from`` itself plus up to 3 ancestors, so a path like
        ``runs/<old>/checkpoints/step_100`` finds ``runs/<old>/wandb_run_id.txt``
        (2 levels up). Returns (None, None) when no id file is found.
        """
        if resume_from is None:
            return None, None
        candidate_dir = Path(resume_from)
        for _ in range(4):
            candidate = candidate_dir / _WANDB_ID_FILENAME
            if candidate.is_file():
                return candidate.read_text().strip(), "allow"
            if candidate_dir.parent == candidate_dir:
                break
            candidate_dir = candidate_dir.parent
        return None, None

    def log_scalars(self, step: int, values: dict[str, float]) -> None:
        if self._run is None:
            raise RuntimeError("start_run() must be called before log_scalars()")
        finite = {k: v for k, v in values.items() if math.isfinite(v)}
        if finite:
            self._run.log(finite, step=step)

    def log_images(self, step: int, images: dict[str, np.ndarray[Any, Any]]) -> None:
        import wandb

        if self._run is None:
            raise RuntimeError("start_run() must be called before log_images()")
        payload: dict[str, Any] = {}
        for tag, arr in images.items():
            _validate_image(tag, arr)
            payload[tag] = wandb.Image(arr)
        if payload:
            self._run.log(payload, step=step)

    def close(self) -> None:
        if self._closed or self._run is None:
            return
        self._run.finish()
        self._closed = True


@register("tracker", "wandb")
def build_wandb(cfg: TrainConfig) -> WandBTracker:
    return WandBTracker(cfg)
