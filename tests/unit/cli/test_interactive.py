"""Tests for the shared interactive module (CPU-only; prompts monkeypatched)."""

from __future__ import annotations

from pathlib import Path

import pytest
import typer

from custom_sam_peft.cli import _interactive as itv


def test_prompt_primitives_importable() -> None:
    assert callable(itv.ask_text)
    assert callable(itv.ask_choice)
    assert callable(itv.ask_confirm)
    assert callable(itv.run_wizard)
    assert hasattr(itv, "WizardStep")
    assert hasattr(itv, "Ctx")


def test_ask_choice_reasks_on_invalid(monkeypatch: pytest.MonkeyPatch) -> None:
    answers = iter(["bogus", "coco"])
    monkeypatch.setattr(itv.typer, "prompt", lambda *a, **k: next(answers))
    out: list[str] = []
    monkeypatch.setattr(itv.typer, "echo", lambda msg="", *a, **k: out.append(str(msg)))
    assert itv.ask_choice("Format?", ["coco", "hf"], default="coco") == "coco"
    assert any("choose one of" in line for line in out)


def test_deep_merge_nested() -> None:
    dst = {"data": {"format": "coco"}}
    itv._deep_merge(dst, {"data": {"split": {"val": 0.1}}})
    assert dst == {"data": {"format": "coco", "split": {"val": 0.1}}}


def test_shared_steps_return_fragments(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(itv, "ask_choice", lambda *a, **k: "coco")
    answers = iter(["ann.json", "imgs/"])
    monkeypatch.setattr(itv, "ask_text", lambda *a, **k: next(answers))
    ctx = itv.Ctx(answers={}, cuda_available=False)
    frag = itv._ask_dataset_source(ctx)
    assert frag == {
        "data": {"format": "coco", "train": {"annotations": "ann.json", "images": "imgs/"}}
    }


def test_auto_detect_path_accepts_candidate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(itv, "ask_confirm", lambda *a, **k: True)
    result = itv._auto_detect_path("train annotations", "Path?", [tmp_path / "train.json"])
    assert result == str(tmp_path / "train.json")


def test_auto_detect_path_override_candidate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(itv, "ask_confirm", lambda *a, **k: False)
    monkeypatch.setattr(itv, "ask_text", lambda *a, **k: "custom.json")
    result = itv._auto_detect_path("train annotations", "Path?", [tmp_path / "train.json"])
    assert result == "custom.json"


def test_auto_detect_path_no_candidates(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(itv, "ask_text", lambda *a, **k: "manual.json")
    result = itv._auto_detect_path("train annotations", "Path?", [])
    assert result == "manual.json"


def test_ask_dataset_source_detects_json(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "train.json").write_text("{}")
    train_img = data_dir / "train"
    train_img.mkdir()

    monkeypatch.setattr(itv, "ask_choice", lambda *a, **k: "coco")
    monkeypatch.setattr(itv, "ask_confirm", lambda *a, **k: True)
    monkeypatch.setattr(itv, "_detect_json_candidates", lambda **kw: [data_dir / "train.json"])
    monkeypatch.setattr(itv, "_detect_dir_candidates", lambda subdirs, **kw: [train_img])

    ctx = itv.Ctx(answers={}, cuda_available=False)
    frag = itv._ask_dataset_source(ctx)
    assert frag["data"]["train"]["annotations"] == str(data_dir / "train.json")
    assert frag["data"]["train"]["images"] == str(train_img)


def test_detect_json_candidates_empty_when_no_data_dir(tmp_path: Path) -> None:
    result = itv._detect_json_candidates(tmp_path / "nonexistent")
    assert result == []


def test_detect_json_candidates_finds_jsons(tmp_path: Path) -> None:
    (tmp_path / "train.json").write_text("{}")
    (tmp_path / "val.json").write_text("{}")
    (tmp_path / "other.txt").write_text("")
    result = itv._detect_json_candidates(tmp_path)
    assert {p.name for p in result} == {"train.json", "val.json"}


def test_detect_dir_candidates_finds_subdir(tmp_path: Path) -> None:
    (tmp_path / "train").mkdir()
    result = itv._detect_dir_candidates(["train", "val"], tmp_path)
    assert len(result) == 1
    assert result[0].name == "train"


def test_detect_json_candidates_finds_nested_json(tmp_path: Path) -> None:
    nested = tmp_path / "DataFusionContest"
    nested.mkdir()
    (nested / "train.json").write_text("{}")
    result = itv._detect_json_candidates(tmp_path)
    assert len(result) == 1
    assert result[0].name == "train.json"


def test_detect_dir_candidates_finds_nested_subdir(tmp_path: Path) -> None:
    (tmp_path / "DataFusionContest" / "train").mkdir(parents=True)
    result = itv._detect_dir_candidates(["train"], tmp_path)
    assert len(result) == 1
    assert result[0].name == "train"


def test_detect_dir_candidates_empty_when_no_data_dir(tmp_path: Path) -> None:
    result = itv._detect_dir_candidates(["train"], tmp_path / "nonexistent")
    assert result == []


def test_auto_detect_path_single_candidate_suggests_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(itv, "ask_confirm", lambda *a, **k: True)
    result = itv._auto_detect_path("train annotations", "Path?", [tmp_path / "train.json"])
    assert result == str(tmp_path / "train.json")


def test_auto_detect_path_multiple_candidates_skips_confirm(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    confirm_calls: list[bool] = []
    monkeypatch.setattr(itv, "ask_confirm", lambda *a, **k: confirm_calls.append(True) or True)
    got_default: list[str | None] = []

    def _capture_ask_text(*args, default: str | None = None, **kwargs: object) -> str:
        got_default.append(default)
        return "manual.json"

    monkeypatch.setattr(itv, "ask_text", _capture_ask_text)
    result = itv._auto_detect_path(
        "train annotations", "Path?", [tmp_path / "a.json", tmp_path / "b.json"]
    )
    assert confirm_calls == []
    assert got_default == [None]
    assert result == "manual.json"


def test_ask_dataset_source_coco_validates_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    """_ask_dataset_source must wire up file/dir validators for COCO paths."""
    monkeypatch.setattr(itv, "ask_choice", lambda *a, **k: "coco")
    captured: list[tuple[str, object]] = []

    def _capture(label: str, prompt: str, candidates: list, *, validate=None, **kw: object) -> str:
        captured.append((label, validate))
        return "dummy"

    monkeypatch.setattr(itv, "_auto_detect_path", _capture)
    ctx = itv.Ctx(answers={}, cuda_available=False)
    itv._ask_dataset_source(ctx)
    labels = [label for label, _ in captured]
    validators = [v for _, v in captured]
    assert "train annotations" in labels
    assert "train images dir" in labels
    for v in validators:
        assert v is not None, "all COCO path prompts must carry a validate= callback"


def test_ask_validation_explicit_coco_validates_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    """Explicit-val COCO path prompts must carry file/dir validators."""
    monkeypatch.setattr(itv, "ask_choice", lambda *a, **k: "explicit")
    captured: list[tuple[str, object]] = []

    def _capture(label: str, prompt: str, candidates: list, *, validate=None, **kw: object) -> str:
        captured.append((label, validate))
        return "dummy"

    monkeypatch.setattr(itv, "_auto_detect_path", _capture)
    ctx = itv.Ctx(answers={"data": {"format": "coco"}}, cuda_available=False)
    itv._ask_validation(ctx)
    labels = [label for label, _ in captured]
    validators = [v for _, v in captured]
    assert "val annotations" in labels
    assert "val images dir" in labels
    for v in validators:
        assert v is not None, "all COCO val path prompts must carry a validate= callback"


def test_validate_is_file_accepts_existing_file(tmp_path: Path) -> None:
    f = tmp_path / "foo.json"
    f.write_text("{}")
    assert itv._validate_is_file(str(f)) is None


def test_validate_is_file_rejects_missing(tmp_path: Path) -> None:
    assert itv._validate_is_file(str(tmp_path / "missing.json")) is not None


def test_validate_is_dir_accepts_existing_dir(tmp_path: Path) -> None:
    assert itv._validate_is_dir(str(tmp_path)) is None


def test_validate_is_dir_rejects_missing(tmp_path: Path) -> None:
    assert itv._validate_is_dir(str(tmp_path / "missing")) is not None


def test_require_tty_non_tty_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(itv.sys.stdin, "isatty", lambda: False)
    with pytest.raises(typer.BadParameter, match="TTY"):
        itv.require_tty()


def test_require_tty_tty_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(itv.sys.stdin, "isatty", lambda: True)
    assert itv.require_tty() is None


def test_validate_checkpoint_dir(tmp_path: Path) -> None:
    good = tmp_path / "ckpt"
    good.mkdir()
    (good / "adapter_config.json").write_text("{}")
    assert itv.validate_checkpoint_dir(str(good)) is None
    bad = tmp_path / "empty"
    bad.mkdir()
    assert itv.validate_checkpoint_dir(str(bad)) is not None
    assert itv.validate_checkpoint_dir(str(tmp_path / "missing")) is not None


def test_validate_config_with_eval_split(tmp_path: Path) -> None:
    import textwrap

    def _write(body: str) -> Path:
        p = tmp_path / f"{abs(hash(body))}.yaml"
        p.write_text(textwrap.dedent(body))
        return p

    base = """
    run: {name: r}
    model: {name: facebook/sam3.1, local_dir: models/sam3.1, checkpoint_file: c.pt}
    data:
      format: coco
      train: {annotations: t.json, images: t/}
      VAL_BLOCK
    peft: {method: lora, r: 16, alpha: 32, dropout: 0.05}
    train:
      epochs: 1
      loss: {preset: natural, class_imbalance: balanced}
    """
    with_val = _write(base.replace("VAL_BLOCK", "val: {annotations: v.json, images: v/}"))
    assert itv.validate_config_with_eval_split(str(with_val)) is None
    no_val = _write(base.replace("      VAL_BLOCK\n", ""))
    assert itv.validate_config_with_eval_split(str(no_val)) is not None
    assert itv.validate_config_with_eval_split(str(tmp_path / "nope.yaml")) is not None


# ---------------------------------------------------------------------------
# §10.5 — emitter migration: auto-split prompt and emitted config format
# ---------------------------------------------------------------------------


def test_ask_validation_auto_split_yields_split_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """_ask_validation auto-split branch must yield {"data": {"split": {"val": …}}},
    not the old {"data": {"val_split": …}} shape.
    Spec: docs/superpowers/specs/2026-06-04-train-val-test-split-design.md §10.5.
    """
    monkeypatch.setattr(itv, "ask_choice", lambda *a, **k: "auto-split")
    monkeypatch.setattr(itv, "ask_text", lambda *a, **k: "0.2")
    ctx = itv.Ctx(answers={"data": {"format": "coco"}}, cuda_available=False)
    frag = itv._ask_validation(ctx)
    # Must use the new data.split.val shape, not data.val_split.fraction
    assert frag == {"data": {"split": {"val": 0.2}}}
    assert "val_split" not in frag.get("data", {})


def test_peek_adapter_lora(tmp_path: Path) -> None:
    import json

    (tmp_path / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "facebook/sam3.1"})
    )
    pretty, base = itv.peek_adapter(tmp_path)
    assert pretty == "LoRA"
    assert base == "facebook/sam3.1"


def test_peek_adapter_qlora(tmp_path: Path) -> None:
    import json

    (tmp_path / "adapter_config.json").write_text(json.dumps({}))
    (tmp_path / "custom_sam_peft_qlora.json").write_text("{}")
    pretty, base = itv.peek_adapter(tmp_path)
    assert pretty == "QLoRA"
    assert base is None
