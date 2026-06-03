"""export --output is always required; merge lands at --output (Phase 3, §6.2)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from custom_sam_peft.cli.main import app

runner = CliRunner()


def _ckpt_with_config(tmp_path: Path) -> Path:
    run_dir = tmp_path / "run"
    ckpt = run_dir / "adapter"
    ckpt.mkdir(parents=True)
    (ckpt / "adapter_config.json").write_text("{}")
    (run_dir / "config.yaml").write_text(
        "run:\n  name: ex\ndata:\n  format: coco\n"
        "  train:\n    annotations: a\n    images: i\n"
        "  val:\n    annotations: a\n    images: i\n"
        "peft:\n  method: lora\ntrain:\n  epochs: 1\n"
    )
    return ckpt


def test_export_requires_output_non_merge(tmp_path: Path) -> None:
    ckpt = _ckpt_with_config(tmp_path)
    result = runner.invoke(app, ["export", "--checkpoint", str(ckpt)])
    assert result.exit_code != 0
    assert "output" in result.output.lower()


def test_export_requires_output_with_merge(tmp_path: Path) -> None:
    ckpt = _ckpt_with_config(tmp_path)
    result = runner.invoke(app, ["export", "--checkpoint", str(ckpt), "--merge"])
    assert result.exit_code != 0
    assert "output" in result.output.lower()


def test_run_export_merge_lands_at_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Library-level: merged weights land at the given output path itself.
    import custom_sam_peft.models.sam3 as sam3_mod
    import custom_sam_peft.train.checkpoint as ckpt_mod
    from custom_sam_peft.runs import bundle

    out = tmp_path / "merged-out"
    captured: dict[str, Any] = {}

    def fake_load_sam31(*a: Any, **k: Any) -> Any:
        return object()

    # Patch where run_export resolves them: inline imports inside the function
    # bind against the module objects, so patch there.
    monkeypatch.setattr(sam3_mod, "load_sam31", fake_load_sam31)
    monkeypatch.setattr(ckpt_mod, "load_adapter", lambda *a, **k: None)
    monkeypatch.setattr(
        ckpt_mod,
        "save_merged",
        lambda wrapper, path: captured.__setitem__("path", path),
    )

    class _Cfg:
        class model: ...

        class data:
            channels = 3
            channel_semantics = "rgb"

    result = bundle.run_export(_Cfg(), tmp_path / "adapter", merge=True, output=out)
    assert captured["path"] == out
    assert result == out


def test_run_export_adapter_default_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Library-level: non-merge path with output=None lands at checkpoint.parent / "exported".
    import custom_sam_peft.models.sam3 as sam3_mod
    import custom_sam_peft.train.checkpoint as ckpt_mod
    from custom_sam_peft.runs import bundle

    checkpoint = tmp_path / "run" / "adapter"
    checkpoint.mkdir(parents=True)
    captured: dict[str, Any] = {}

    def fake_load_sam31(*a: Any, **k: Any) -> Any:
        return object()

    monkeypatch.setattr(sam3_mod, "load_sam31", fake_load_sam31)
    monkeypatch.setattr(ckpt_mod, "load_adapter", lambda *a, **k: None)
    monkeypatch.setattr(
        ckpt_mod,
        "save_adapter",
        lambda wrapper, path: captured.__setitem__("path", path),
    )

    class _Cfg:
        class model: ...

        class data:
            channels = 3
            channel_semantics = "rgb"

    result = bundle.run_export(_Cfg(), checkpoint, merge=False, output=None)
    expected = checkpoint.parent / "exported"
    assert captured["path"] == expected
    assert result == expected
