from pathlib import Path

from custom_sam_peft.eval._artifacts import EvalArtifacts


def test_eval_artifacts_fields():
    art = EvalArtifacts(
        checkpoint_path=Path("/tmp/runs/x/checkpoints/step_00000100.pt"),  # noqa: S108
        peft_method="lora",
        run_dir=Path("/tmp/runs/x"),  # noqa: S108
    )
    assert art.checkpoint_path.name == "step_00000100.pt"
    assert art.peft_method == "lora"
    assert art.run_dir == Path("/tmp/runs/x")  # noqa: S108


def test_eval_artifacts_is_frozen():
    import pytest

    art = EvalArtifacts(
        checkpoint_path=Path("/x"),
        peft_method="lora",
        run_dir=Path("/y"),
    )
    with pytest.raises((AttributeError, Exception)):
        art.peft_method = "qlora"  # type: ignore[misc]
