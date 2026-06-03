"""Unit tests for EvalArtifacts field defaults and round-trips."""

from pathlib import Path


def test_eval_artifacts_ladder_events_field() -> None:
    """EvalArtifacts.ladder_events defaults to None and round-trips a passed value."""
    from custom_sam_peft.eval._artifacts import EvalArtifacts
    from custom_sam_peft.train.ladder import LadderEvents

    art = EvalArtifacts(
        checkpoint_path=Path("/tmp/adapter"),  # noqa: S108
        peft_method="lora",
        run_dir=Path("/tmp"),  # noqa: S108
        final_metrics=None,
    )
    assert art.ladder_events is None  # default

    events = LadderEvents(stop_reason="early_stop: 10 evals")
    art2 = EvalArtifacts(
        checkpoint_path=Path("/tmp/adapter"),  # noqa: S108
        peft_method="lora",
        run_dir=Path("/tmp"),  # noqa: S108
        final_metrics=None,
        ladder_events=events,
    )
    assert art2.ladder_events == events
