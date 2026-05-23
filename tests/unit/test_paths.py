from pathlib import Path

from custom_sam_peft.paths import (
    artifact_path,
    bundle_path,
    checkpoint_path,
    predictions_path,
)


def test_checkpoint_path(tmp_path):
    run_dir = tmp_path / "runs" / "abc"
    p = checkpoint_path(run_dir, step=42)
    assert p.parent == run_dir / "checkpoints"
    assert "42" in p.name
    assert isinstance(p, Path)


def test_artifact_path(tmp_path):
    run_dir = tmp_path / "runs" / "abc"
    p = artifact_path(run_dir, name="metrics.json")
    assert p == run_dir / "artifacts" / "metrics.json"


def test_predictions_path(tmp_path):
    run_dir = tmp_path / "runs" / "abc"
    p = predictions_path(run_dir, split="val")
    assert p.parent == run_dir / "artifacts"
    assert "val" in p.name


def test_bundle_path(tmp_path):
    run_dir = tmp_path / "runs" / "abc"
    p = bundle_path(run_dir)
    assert p.parent == run_dir / "bundle"


def test_run_dir_layout_constants_exposed():
    from custom_sam_peft.paths import (
        ARTIFACTS_SUBDIR,
        BUNDLE_SUBDIR,
        CHECKPOINTS_SUBDIR,
        LOGS_SUBDIR,
    )

    assert CHECKPOINTS_SUBDIR == "checkpoints"
    assert ARTIFACTS_SUBDIR == "artifacts"
    assert LOGS_SUBDIR == "logs"
    assert BUNDLE_SUBDIR == "bundle"
