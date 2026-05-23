"""Single source of truth for run-directory layout.

Layout:
    runs/<run_id>/
        checkpoints/
        artifacts/
        logs/
        bundle/

Never string-join checkpoint paths anywhere else. The §9.2 static
guard test enforces this.
"""

from __future__ import annotations

from pathlib import Path

CHECKPOINTS_SUBDIR = "checkpoints"
ARTIFACTS_SUBDIR = "artifacts"
LOGS_SUBDIR = "logs"
BUNDLE_SUBDIR = "bundle"


def checkpoint_path(run_dir: Path, *, step: int) -> Path:
    """Return the canonical path for the checkpoint at the given global step."""
    return run_dir / CHECKPOINTS_SUBDIR / f"step_{step:08d}.pt"


def artifact_path(run_dir: Path, *, name: str) -> Path:
    """Return the path for a named artifact (metrics.json, schema.json, ...)."""
    return run_dir / ARTIFACTS_SUBDIR / name


def predictions_path(run_dir: Path, *, split: str) -> Path:
    """Return the path for serialized predictions for a given split."""
    return run_dir / ARTIFACTS_SUBDIR / f"predictions_{split}.jsonl"


def bundle_path(run_dir: Path) -> Path:
    """Return the path for the exported run bundle (zip)."""
    return run_dir / BUNDLE_SUBDIR / "bundle.zip"
