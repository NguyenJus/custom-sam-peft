"""init tier flags reject bad values at the parser (Phase 2).

The enum-typed parameters must produce a Typer/Click parser error (exit 2)
before run_init is called.  We verify parser-level rejection by checking that
the error message mentions the correct flag rather than '--template' (which is
what the post-parse ValueError handler in init() mis-attributes today).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from custom_sam_peft.cli.main import app

runner = CliRunner()


def test_init_bad_preset_rejected() -> None:
    result = runner.invoke(app, ["init", "--preset", "bogus", "--output", "x.yaml"])
    assert result.exit_code != 0
    # Parser-level rejection names the offending flag; post-parse path misattributes to --template.
    assert "--preset" in result.output or "preset" in result.output.lower()
    assert "--template" not in result.output


def test_init_bad_intensity_rejected() -> None:
    result = runner.invoke(app, ["init", "--intensity", "nope", "--output", "x.yaml"])
    assert result.exit_code != 0
    assert "--intensity" in result.output or "intensity" in result.output.lower()
    assert "--template" not in result.output


def test_init_bad_class_imbalance_rejected() -> None:
    result = runner.invoke(app, ["init", "--class-imbalance", "nope", "--output", "x.yaml"])
    assert result.exit_code != 0
    # class-imbalance already raises BadParameter with the correct hint, but not at parser level
    assert "--class-imbalance" in result.output or "class" in result.output.lower()


def test_init_threads_peft_scope_into_decide_preset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """init must size against the run's cfg.peft.scope, not the default (#310).

    A non-default scope ("all") must reach decide_preset and flow into the
    rewritten YAML's batch. Pre-fix, init called decide_preset without scope, so
    it sized against the default "decoder_concept" and the scope-derived batch
    never appeared. We inject a non-default scope via load_config and stub
    decide_preset to encode the received scope in batch_size, so the assertion
    fails if scope is dropped or defaulted on the way through.
    """
    from custom_sam_peft.cli import init_cmd
    from custom_sam_peft.config.loader import load_config as real_load_config
    from custom_sam_peft.presets import PresetDecision

    captured: dict[str, str] = {}

    def fake_load_config(path: Path):  # type: ignore[no-untyped-def]
        cfg = real_load_config(path)
        cfg.peft.scope = "all"  # non-default scope (~20x the default adapter bytes)
        return cfg

    def fake_decide_preset(*, scope: str, **_kwargs: object) -> PresetDecision:
        captured["scope"] = scope
        # Encode scope in batch_size so we can assert it survived end-to-end.
        return PresetDecision(
            method="lora",
            r=16,
            batch_size=7 if scope == "all" else 1,
            grad_accum_steps=1,
            classes_per_forward=4,
            dtype="bfloat16",
            headroom_bytes=0,
            predicted_bytes=0,
            budget_bytes=0,
            gpu_name="stub",
            provenance="analytic",
            cache_path=None,
            calibrated_at=None,
            alpha=32,
        )

    monkeypatch.setattr(init_cmd.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(init_cmd, "load_config", fake_load_config)
    monkeypatch.setattr(init_cmd, "decide_preset", fake_decide_preset)
    monkeypatch.setattr(init_cmd, "infer_num_classes", lambda _data: None)

    out = tmp_path / "cfg.yaml"
    result = runner.invoke(app, ["init", "--output", str(out)])
    assert result.exit_code == 0, result.output

    assert captured["scope"] == "all"
    written = real_load_config(out)
    assert written.train.batch_size == 7
