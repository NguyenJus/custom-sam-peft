"""Tests for the interactive setup wizard (CPU-only; prompt primitives monkeypatched)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner

from custom_sam_peft.cli import setup_wizard as sw
from custom_sam_peft.cli.main import app
from custom_sam_peft.config.loader import load_config


def test_deep_merge_nested_dicts() -> None:
    dst = {"data": {"format": "coco"}}
    sw._deep_merge(dst, {"data": {"val_split": {"fraction": 0.1}}})
    assert dst == {"data": {"format": "coco", "val_split": {"fraction": 0.1}}}


def test_deep_merge_scalar_overwrites() -> None:
    dst = {"peft": {"method": "lora"}}
    sw._deep_merge(dst, {"peft": {"method": "qlora"}})
    assert dst["peft"]["method"] == "qlora"


def test_ctx_constructs_with_cuda_flag_and_run_mode() -> None:
    ctx = sw.Ctx(answers={}, cuda_available=False)
    assert ctx.answers == {}
    assert ctx.cuda_available is False
    assert ctx.run_mode == "train"  # default
    assert ctx.categories is None


def _write_coco(path: Path, per_cat_counts: dict[int, int], *, iscrowd_extra: int = 0) -> None:
    categories = [{"id": cid, "name": f"c{cid}"} for cid in per_cat_counts]
    images, annotations = [], []
    img_id, ann_id = 0, 0
    for cid, count in per_cat_counts.items():
        for _ in range(count):
            images.append({"id": img_id, "file_name": f"{img_id}.jpg", "height": 4, "width": 4})
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": cid,
                    "bbox": [0, 0, 2, 2],
                    "area": 4,
                    "iscrowd": 0,
                }
            )
            img_id += 1
            ann_id += 1
    for _ in range(iscrowd_extra):
        images.append({"id": img_id, "file_name": f"{img_id}.jpg", "height": 4, "width": 4})
        annotations.append(
            {
                "id": ann_id,
                "image_id": img_id,
                "category_id": next(iter(per_cat_counts)),
                "bbox": [0, 0, 2, 2],
                "area": 4,
                "iscrowd": 1,
            }
        )
        img_id += 1
        ann_id += 1
    path.write_text(
        json.dumps({"images": images, "annotations": annotations, "categories": categories})
    )


def test_infer_balanced_below_3x(tmp_path: Path) -> None:
    p = tmp_path / "a.json"
    _write_coco(p, {1: 10, 2: 10, 3: 12})  # R≈1.2
    assert sw.infer_class_imbalance(str(p)) == "balanced"


def test_infer_moderate_3x_to_10x(tmp_path: Path) -> None:
    p = tmp_path / "a.json"
    _write_coco(p, {1: 10, 2: 40})  # R=4
    assert sw.infer_class_imbalance(str(p)) == "moderate"


def test_infer_severe_at_or_above_10x(tmp_path: Path) -> None:
    p = tmp_path / "a.json"
    _write_coco(p, {1: 5, 2: 100})  # R=20
    assert sw.infer_class_imbalance(str(p)) == "severe"


def test_infer_thresholds_boundary_exact(tmp_path: Path) -> None:
    p3 = tmp_path / "r3.json"
    _write_coco(p3, {1: 10, 2: 30})  # R=3.0 → moderate
    assert sw.infer_class_imbalance(str(p3)) == "moderate"
    p10 = tmp_path / "r10.json"
    _write_coco(p10, {1: 10, 2: 100})  # R=10.0 → severe
    assert sw.infer_class_imbalance(str(p10)) == "severe"


def test_infer_unreadable_defaults_balanced(tmp_path: Path) -> None:
    assert sw.infer_class_imbalance(str(tmp_path / "missing.json")) == "balanced"


def test_infer_iscrowd_excluded(tmp_path: Path) -> None:
    p = tmp_path / "a.json"
    _write_coco(p, {1: 10, 2: 10}, iscrowd_extra=50)
    assert sw.infer_class_imbalance(str(p)) == "balanced"


# ---------------------------------------------------------------------------
# Issue #165: _ask_class_imbalance reports the measured ratio and asks for
# agreement (confirm) rather than forcing a tier choice.
# ---------------------------------------------------------------------------


def _capture_echo(monkeypatch) -> list[str]:
    lines: list[str] = []
    monkeypatch.setattr(sw.typer, "echo", lambda msg="", *a, **k: lines.append(str(msg)))
    return lines


def test_ask_class_imbalance_moderate_confirm_applies_detected_tier(tmp_path, monkeypatch) -> None:
    p = tmp_path / "a.json"
    _write_coco(p, {1: 10, 2: 40})  # R=4.0 → moderate
    echoes = _capture_echo(monkeypatch)
    confirms: list[bool] = []
    monkeypatch.setattr(sw, "ask_confirm", lambda *a, **k: confirms.append(True) or True)
    ctx = sw.Ctx(
        answers={"data": {"format": "coco", "train": {"annotations": str(p)}}},
        cuda_available=False,
    )
    frag = sw._ask_class_imbalance(ctx)
    assert frag == {"train": {"loss": {"class_imbalance": "moderate"}}}
    assert confirms == [True]  # confirm prompt was shown
    assert any("4.0x" in line and "moderate" in line for line in echoes)


def test_ask_class_imbalance_severe_decline_returns_balanced(tmp_path, monkeypatch) -> None:
    p = tmp_path / "a.json"
    _write_coco(p, {1: 5, 2: 100})  # R=20.0 → severe
    echoes = _capture_echo(monkeypatch)
    confirms: list[bool] = []
    monkeypatch.setattr(sw, "ask_confirm", lambda *a, **k: confirms.append(False) or False)
    ctx = sw.Ctx(
        answers={"data": {"format": "coco", "train": {"annotations": str(p)}}},
        cuda_available=False,
    )
    frag = sw._ask_class_imbalance(ctx)
    assert frag == {"train": {"loss": {"class_imbalance": "balanced"}}}
    assert confirms == [False]  # confirm prompt was shown
    assert any("20.0x" in line and "severe" in line for line in echoes)


def test_ask_class_imbalance_measured_balanced_no_confirm(tmp_path, monkeypatch) -> None:
    p = tmp_path / "a.json"
    _write_coco(p, {1: 10, 2: 12})  # R≈1.2 → balanced
    echoes = _capture_echo(monkeypatch)

    def _no_confirm(*a, **k):
        raise AssertionError("ask_confirm must not be called for the balanced path")

    monkeypatch.setattr(sw, "ask_confirm", _no_confirm)
    ctx = sw.Ctx(
        answers={"data": {"format": "coco", "train": {"annotations": str(p)}}},
        cuda_available=False,
    )
    frag = sw._ask_class_imbalance(ctx)
    assert frag == {"train": {"loss": {"class_imbalance": "balanced"}}}
    assert any("balanced" in line for line in echoes)


def test_ask_class_imbalance_non_coco_no_confirm(monkeypatch) -> None:
    echoes = _capture_echo(monkeypatch)

    def _no_confirm(*a, **k):
        raise AssertionError("ask_confirm must not be called for the undetectable path")

    monkeypatch.setattr(sw, "ask_confirm", _no_confirm)
    ctx = sw.Ctx(answers={"data": {"format": "hf", "hf": {"name": "org/ds"}}}, cuda_available=False)
    frag = sw._ask_class_imbalance(ctx)
    assert frag == {"train": {"loss": {"class_imbalance": "balanced"}}}
    assert any("auto-detect" in line.lower() for line in echoes)


def test_ask_class_imbalance_coco_unmeasurable_no_confirm(tmp_path, monkeypatch) -> None:
    echoes = _capture_echo(monkeypatch)

    def _no_confirm(*a, **k):
        raise AssertionError("ask_confirm must not be called when the ratio cannot be measured")

    monkeypatch.setattr(sw, "ask_confirm", _no_confirm)
    missing = tmp_path / "missing.json"  # COCO path set, file unreadable -> ratio is None
    ctx = sw.Ctx(
        answers={"data": {"format": "coco", "train": {"annotations": str(missing)}}},
        cuda_available=False,
    )
    frag = sw._ask_class_imbalance(ctx)
    assert frag == {"train": {"loss": {"class_imbalance": "balanced"}}}
    assert any("measure" in line.lower() for line in echoes)


# ---------------------------------------------------------------------------
# Task 12: STEPS registry + run_wizard
# ---------------------------------------------------------------------------


def _patch_prompts(monkeypatch, *, texts=None, choices=None, confirms=None):
    """Feed scripted answers to the three primitives in call order."""
    t = iter(texts or [])
    c = iter(choices or [])
    cf = iter(confirms or [])
    monkeypatch.setattr(sw, "ask_text", lambda *a, **k: next(t))
    monkeypatch.setattr(sw, "ask_choice", lambda *a, **k: next(c))
    monkeypatch.setattr(sw, "ask_confirm", lambda *a, **k: next(cf))


def test_step_fragment_shapes_are_nested_dicts(monkeypatch) -> None:
    _patch_prompts(
        monkeypatch,
        texts=["my-run", "ann.json", "imgs/", "5", ""],
        choices=["train", "coco", "none", "natural", "medium", "lora"],
        confirms=[False],  # decline limit step
    )
    # Class-imbalance step measures the ratio (no ask_choice). Undetectable
    # annotations → balanced with no confirm prompt.
    monkeypatch.setattr(sw, "measure_class_imbalance_ratio", lambda *a, **k: None)
    ctx = sw.Ctx(answers={}, cuda_available=False)
    answers = sw.run_wizard(ctx)
    assert answers["run"]["name"] == "my-run"
    assert answers["data"]["format"] == "coco"
    assert answers["data"]["train"]["annotations"] == "ann.json"
    assert answers["peft"]["method"] == "lora"
    assert answers["train"]["epochs"] == 5
    assert answers["train"]["loss"]["class_imbalance"] == "balanced"
    assert ctx.run_mode == "train"


def test_when_gating_skips_class_imbalance_in_eval_mode() -> None:
    step = next(s for s in sw.STEPS if s.id == "class_imbalance")
    ctx = sw.Ctx(answers={"data": {"format": "coco"}}, cuda_available=False, run_mode="eval")
    assert step.when(ctx) is False


def test_when_gating_skips_vram_autosize_without_cuda(monkeypatch) -> None:
    _patch_prompts(monkeypatch, choices=["lora"])
    step = next(s for s in sw.STEPS if s.id == "peft_sizing")
    ctx = sw.Ctx(answers={}, cuda_available=False)
    assert step.ask(ctx) == {"peft": {"method": "lora"}}


# ---------------------------------------------------------------------------
# Task 13: render
# ---------------------------------------------------------------------------


def test_render_coco_explicit_val_reloads(tmp_path) -> None:
    answers = {
        "run": {"name": "r"},
        "data": {
            "format": "coco",
            "train": {"annotations": "t.json", "images": "t/"},
            "val": {"annotations": "v.json", "images": "v/"},
            "augmentations": {"preset": "medical", "intensity": "medium"},
        },
        "peft": {"method": "lora"},
        "train": {"epochs": 3, "loss": {"preset": "medical", "class_imbalance": "moderate"}},
    }
    rendered = sw.render(answers, run_mode="train")
    assert "format: coco" in rendered
    assert "# hf:" in rendered
    assert "# val_split:" in rendered
    out = tmp_path / "c.yaml"
    out.write_text(rendered)
    cfg = load_config(out)
    assert cfg.data.val is not None
    assert cfg.peft.method == "lora"
    assert cfg.train.epochs == 3


def test_render_hf_autosplit_qlora_reloads(tmp_path) -> None:
    answers = {
        "run": {"name": "r"},
        "data": {
            "format": "hf",
            "hf": {"name": "org/ds"},
            "val_split": {"fraction": 0.2},
            "augmentations": {"preset": "natural", "intensity": "safe"},
        },
        "peft": {"method": "qlora"},
        "train": {"epochs": 2, "loss": {"preset": "natural", "class_imbalance": "balanced"}},
    }
    rendered = sw.render(answers, run_mode="train")
    assert "name: org/ds" in rendered
    assert "quant_type: nf4" in rendered
    assert "val_split:" in rendered
    out = tmp_path / "c.yaml"
    out.write_text(rendered)
    cfg = load_config(out)
    assert cfg.data.format == "hf"
    assert cfg.peft.method == "qlora"


def test_render_eval_mode_defaults_epochs_to_1(tmp_path) -> None:
    answers = {
        "run": {"name": "r"},
        "data": {
            "format": "coco",
            "train": {"annotations": "t.json", "images": "t/"},
            "augmentations": {"preset": "natural", "intensity": "medium"},
        },
        "peft": {"method": "lora"},
        "train": {"loss": {"preset": "natural", "class_imbalance": "balanced"}},  # no epochs
    }
    rendered = sw.render(answers, run_mode="eval")
    out = tmp_path / "c.yaml"
    out.write_text(rendered)
    cfg = load_config(out)
    assert cfg.train.epochs == 1


# ---------------------------------------------------------------------------
# Task 14: validate + emit
# ---------------------------------------------------------------------------


def test_validate_accepts_good_render() -> None:
    answers = {
        "run": {"name": "r"},
        "data": {
            "format": "coco",
            "train": {"annotations": "t.json", "images": "t/"},
            "augmentations": {"preset": "natural", "intensity": "medium"},
        },
        "peft": {"method": "lora"},
        "train": {"epochs": 1, "loss": {"preset": "natural", "class_imbalance": "balanced"}},
    }
    rendered = sw.render(answers, run_mode="train")
    sw.validate(rendered)  # must not raise


def test_emit_header_and_launch_command(tmp_path) -> None:
    answers = {
        "run": {"name": "r"},
        "data": {
            "format": "coco",
            "train": {"annotations": "t.json", "images": "t/"},
            "augmentations": {"preset": "natural", "intensity": "medium"},
        },
        "peft": {"method": "lora"},
        "train": {"epochs": 1, "loss": {"preset": "natural", "class_imbalance": "balanced"}},
    }
    for mode, verb in [("train", "train"), ("run", "run"), ("eval", "eval")]:
        out = tmp_path / f"{mode}.yaml"
        rendered = sw.render(answers, run_mode=mode)
        sw.emit(rendered, out, force=False, run_mode=mode)
        body = out.read_text()
        lines = body.splitlines()
        assert lines[0].startswith("# Generated by `custom-sam-peft init --interactive`")
        assert lines[1] == f"# Launch: custom-sam-peft {verb} --config {out}"


def test_emit_validated_bytes_reload(tmp_path) -> None:
    answers = {
        "run": {"name": "r"},
        "data": {
            "format": "coco",
            "train": {"annotations": "t.json", "images": "t/"},
            "augmentations": {"preset": "natural", "intensity": "medium"},
        },
        "peft": {"method": "lora"},
        "train": {"epochs": 1, "loss": {"preset": "natural", "class_imbalance": "balanced"}},
    }
    out = tmp_path / "c.yaml"
    rendered = sw.render(answers, run_mode="train")
    sw.emit(rendered, out, force=False, run_mode="train")
    cfg = load_config(out)
    assert cfg.run.name == "r"


# ---------------------------------------------------------------------------
# Task 15: generate_config orchestration + VRAM-step tests
# ---------------------------------------------------------------------------


def test_generate_config_happy_path_local_coco_autosplit(tmp_path, monkeypatch) -> None:
    _patch_prompts(
        monkeypatch,
        texts=["my-run", "ann.json", "imgs/", "0.1", "7", ""],
        choices=["train", "coco", "auto-split", "natural", "medium", "lora"],
        confirms=[False],  # decline limit step
    )
    monkeypatch.setattr(sw, "measure_class_imbalance_ratio", lambda *a, **k: None)
    out = tmp_path / "c.yaml"
    sw.generate_config(out, force=False, cuda_available=False)
    cfg = load_config(out)
    assert cfg.run.name == "my-run"
    assert cfg.data.val_split is not None
    assert cfg.train.epochs == 7


def test_validate_backstop_exits_nonzero_no_file(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        sw,
        "run_wizard",
        lambda ctx: {
            "run": {"name": "r"},
            "data": {
                "format": "coco",
                "train": {"annotations": "t.json", "images": "t/"},
                "augmentations": {"preset": "natural", "intensity": "medium"},
            },
            "peft": {"method": "lora"},
            "train": {"epochs": 0, "loss": {"preset": "natural", "class_imbalance": "balanced"}},
        },
    )
    out = tmp_path / "c.yaml"
    with pytest.raises(typer.Exit):
        sw.generate_config(out, force=False, cuda_available=False)
    assert not out.exists()


def test_ctrl_c_writes_nothing(tmp_path, monkeypatch) -> None:
    def _boom(ctx):
        raise KeyboardInterrupt

    monkeypatch.setattr(sw, "run_wizard", _boom)
    out = tmp_path / "c.yaml"
    with pytest.raises(KeyboardInterrupt):
        sw.generate_config(out, force=False, cuda_available=False)
    assert not out.exists()


def test_vram_autosize_applies_config_patch(monkeypatch) -> None:
    from custom_sam_peft.presets import PresetDecision

    decision = PresetDecision(
        method="qlora",
        r=16,
        batch_size=2,
        grad_accum_steps=8,
        dtype="bfloat16",
        headroom_bytes=0,
        predicted_bytes=0,
        budget_bytes=0,
        gpu_name="StubGPU",
        provenance="analytic",
        cache_path=None,
        calibrated_at=None,
    )
    monkeypatch.setattr("custom_sam_peft.presets.decide_preset", lambda: decision)
    monkeypatch.setattr(sw, "ask_confirm", lambda *a, **k: True)
    ctx = sw.Ctx(answers={}, cuda_available=True)
    frag = sw._ask_peft_sizing(ctx)
    assert frag == decision.config_patch
    assert "gradient_checkpointing" not in frag["model"]


def test_vram_autosize_runtime_error_falls_back_to_manual(monkeypatch) -> None:
    def _boom():
        raise RuntimeError("nothing fits")

    monkeypatch.setattr("custom_sam_peft.presets.decide_preset", _boom)
    monkeypatch.setattr(sw, "ask_confirm", lambda *a, **k: True)
    monkeypatch.setattr(sw, "ask_choice", lambda *a, **k: "qlora")
    ctx = sw.Ctx(answers={}, cuda_available=True)
    frag = sw._ask_peft_sizing(ctx)
    assert frag == {"peft": {"method": "qlora"}}


# ---------------------------------------------------------------------------
# Task 16: --interactive/-i flag + TTY/output pre-flight
# ---------------------------------------------------------------------------

runner = CliRunner()


def test_non_tty_hard_errors(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("custom_sam_peft.cli.init_cmd.sys.stdin.isatty", lambda: False)
    called: list[int] = []
    monkeypatch.setattr(
        "custom_sam_peft.cli.setup_wizard.run_wizard",
        lambda ctx: called.append(1) or {},
    )
    out = tmp_path / "c.yaml"
    result = runner.invoke(app, ["init", "--interactive", "--output", str(out)])
    assert result.exit_code != 0
    assert "TTY" in result.output or "tty" in result.output.lower()
    assert called == []
    assert not out.exists()


def test_output_exists_without_force_errors_before_prompting(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("custom_sam_peft.cli.init_cmd.sys.stdin.isatty", lambda: True)
    called: list[int] = []
    monkeypatch.setattr(
        "custom_sam_peft.cli.setup_wizard.run_wizard",
        lambda ctx: called.append(1) or {},
    )
    out = tmp_path / "c.yaml"
    out.write_text("existing\n")
    result = runner.invoke(app, ["init", "--interactive", "--output", str(out)])
    assert result.exit_code != 0
    assert called == []
    assert out.read_text() == "existing\n"


# ---------------------------------------------------------------------------
# Task 21: render — HF-explicit split_val + no spurious no-val line
# ---------------------------------------------------------------------------


def test_render_hf_explicit_emits_split_val(tmp_path) -> None:
    answers = {
        "run": {"name": "r"},
        "data": {
            "format": "hf",
            "hf": {"name": "org/ds", "split_val": "myval"},
            "augmentations": {"preset": "natural", "intensity": "medium"},
        },
        "peft": {"method": "lora"},
        "train": {"epochs": 2, "loss": {"preset": "natural", "class_imbalance": "balanced"}},
    }
    rendered = sw.render(answers, run_mode="train")
    assert "split_val: myval" in rendered
    # No spurious COCO train: block, and no active no-val claim:
    assert "  # no-val mode:" not in rendered  # the active no-val line must not appear
    out = tmp_path / "c.yaml"
    out.write_text(rendered)
    cfg = load_config(out)
    assert cfg.data.format == "hf"
    assert cfg.data.hf is not None
    assert cfg.data.hf.split_val == "myval"


def test_render_hf_none_emits_no_split_val(tmp_path) -> None:
    answers = {
        "run": {"name": "r"},
        "data": {
            "format": "hf",
            "hf": {"name": "org/ds"},
            "augmentations": {"preset": "natural", "intensity": "medium"},
        },
        "peft": {"method": "lora"},
        "train": {"epochs": 1, "loss": {"preset": "natural", "class_imbalance": "balanced"}},
    }
    rendered = sw.render(answers, run_mode="train")
    assert "split_val:" not in rendered.replace("#   split_val", "")  # no ACTIVE split_val line
    out = tmp_path / "c.yaml"
    out.write_text(rendered)
    cfg = load_config(out)
    assert cfg.data.hf is not None
    assert cfg.data.hf.split_val is None


# ---------------------------------------------------------------------------
# Extra Fix A: auto-split fraction input validation
# ---------------------------------------------------------------------------


def test_fraction_validator_rejects_non_numeric(monkeypatch) -> None:
    """_ask_validation's auto-split branch must validate the fraction input."""
    calls: list[tuple] = []

    def _capture_ask_text(*args, validate=None, **kwargs):
        calls.append((args, kwargs, validate))
        # return a valid value so the function completes
        return "0.1"

    monkeypatch.setattr(sw, "ask_choice", lambda *a, **k: "auto-split")
    monkeypatch.setattr(sw, "ask_text", _capture_ask_text)
    ctx = sw.Ctx(answers={"data": {"format": "coco"}}, cuda_available=False)
    result = sw._ask_validation(ctx)
    assert result == {"data": {"val_split": {"fraction": 0.1}}}
    # The ask_text call must have received a validate= callback
    assert len(calls) == 1
    _args, _kwargs, validate = calls[0]
    assert validate is not None
    # validate rejects non-numeric
    assert validate("abc") is not None
    # validate rejects out-of-range
    assert validate("0") is not None
    assert validate("0.6") is not None
    # validate accepts valid fractions
    assert validate("0.1") is None
    assert validate("0.5") is None


# ---------------------------------------------------------------------------
# Issue #164: _ask_limit step + _limit_block render helper
# ---------------------------------------------------------------------------


def test_limit_validator_accepts_blank() -> None:
    """_ask_limit's validate callback must accept blank (= no limit)."""
    from custom_sam_peft.cli.setup_wizard import _limit_validate

    assert _limit_validate("") is None


def test_limit_validator_accepts_int_count() -> None:
    from custom_sam_peft.cli.setup_wizard import _limit_validate

    assert _limit_validate("1") is None
    assert _limit_validate("100") is None
    assert _limit_validate("9999") is None


def test_limit_validator_accepts_float_fraction() -> None:
    from custom_sam_peft.cli.setup_wizard import _limit_validate

    assert _limit_validate("0.1") is None
    assert _limit_validate("0.5") is None
    assert _limit_validate("1.0") is None


def test_limit_validator_rejects_zero() -> None:
    from custom_sam_peft.cli.setup_wizard import _limit_validate

    assert _limit_validate("0") is not None


def test_limit_validator_rejects_negative_int() -> None:
    from custom_sam_peft.cli.setup_wizard import _limit_validate

    assert _limit_validate("-1") is not None
    assert _limit_validate("-100") is not None


def test_limit_validator_rejects_out_of_range_float() -> None:
    from custom_sam_peft.cli.setup_wizard import _limit_validate

    assert _limit_validate("0.0") is not None
    assert _limit_validate("1.1") is not None
    assert _limit_validate("2.5") is not None


def test_limit_validator_rejects_bool_like_strings() -> None:
    """Strings 'true'/'false'/'True'/'False' must be rejected (not numeric)."""
    from custom_sam_peft.cli.setup_wizard import _limit_validate

    assert _limit_validate("true") is not None
    assert _limit_validate("false") is not None
    assert _limit_validate("True") is not None
    assert _limit_validate("False") is not None


def test_limit_validator_rejects_non_numeric() -> None:
    from custom_sam_peft.cli.setup_wizard import _limit_validate

    assert _limit_validate("abc") is not None
    assert _limit_validate("nan") is not None


def test_limit_scientific_notation_with_dot_accepted() -> None:
    """'1.5e-1' has a dot → float branch → 0.15 is in (0.0, 1.0] → accepted."""
    from custom_sam_peft.cli.setup_wizard import _limit_validate, _parse_limit_value

    assert _limit_validate("1.5e-1") is None
    assert _parse_limit_value("1.5e-1") == pytest.approx(0.15)


def test_limit_scientific_notation_without_dot_rejected() -> None:
    """'1e2' has no dot → int branch → int('1e2') raises → rejected."""
    from custom_sam_peft.cli.setup_wizard import _limit_validate

    assert _limit_validate("1e2") is not None


def test_ask_limit_declined_returns_empty(monkeypatch) -> None:
    """Declining the confirm must yield {} (no data.limit in fragment)."""
    monkeypatch.setattr(sw, "ask_confirm", lambda *a, **k: False)
    ctx = sw.Ctx(answers={}, cuda_available=False)
    frag = sw._ask_limit(ctx)
    assert frag == {}


def test_ask_limit_train_only_fragment(monkeypatch) -> None:
    """Accepting and providing only a train limit produces correct nested shape."""
    cf_iter = iter([True])
    tx_iter = iter(["100", ""])  # train=100, val=blank
    monkeypatch.setattr(sw, "ask_confirm", lambda *a, **k: next(cf_iter))
    monkeypatch.setattr(sw, "ask_text", lambda *a, **k: next(tx_iter))
    ctx = sw.Ctx(answers={}, cuda_available=False)
    frag = sw._ask_limit(ctx)
    assert frag == {"data": {"limit": {"train": 100}}}


def test_ask_limit_both_set_fragment(monkeypatch) -> None:
    """Providing both train and val produces correct nested shape with parsed types."""
    cf_iter = iter([True])
    tx_iter = iter(["200", "0.2"])  # train=200 (int), val=0.2 (float)
    monkeypatch.setattr(sw, "ask_confirm", lambda *a, **k: next(cf_iter))
    monkeypatch.setattr(sw, "ask_text", lambda *a, **k: next(tx_iter))
    ctx = sw.Ctx(answers={}, cuda_available=False)
    frag = sw._ask_limit(ctx)
    assert frag == {"data": {"limit": {"train": 200, "val": 0.2}}}


def test_ask_limit_both_blank_returns_empty(monkeypatch) -> None:
    """Accepting but leaving both fields blank should return {}."""
    cf_iter = iter([True])
    tx_iter = iter(["", ""])
    monkeypatch.setattr(sw, "ask_confirm", lambda *a, **k: next(cf_iter))
    monkeypatch.setattr(sw, "ask_text", lambda *a, **k: next(tx_iter))
    ctx = sw.Ctx(answers={}, cuda_available=False)
    frag = sw._ask_limit(ctx)
    assert frag == {}


def test_limit_block_active_train_only() -> None:
    """_limit_block with only train set emits an active YAML block."""
    block = sw._limit_block({"data": {"limit": {"train": 100}}})
    assert "limit:" in block
    assert "train: 100" in block
    # val should NOT appear as an active (uncommented) key
    lines = block.splitlines()
    active_val_lines = [ln for ln in lines if "val:" in ln and not ln.lstrip().startswith("#")]
    assert active_val_lines == []


def test_limit_block_active_both() -> None:
    """_limit_block with train and val set emits both keys as active lines."""
    block = sw._limit_block({"data": {"limit": {"train": 50, "val": 0.1}}})
    assert "train: 50" in block
    assert "val: 0.1" in block
    # No leading # before these active lines
    for line in block.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("train: 50") or stripped.startswith("val: 0.1"):
            assert not stripped.startswith("#"), f"Active line must not be commented: {line!r}"


def test_limit_block_commented_when_unset() -> None:
    """_limit_block with no limit set emits a fully-commented discovery block."""
    block = sw._limit_block({})
    # Every non-blank line must be commented
    for line in block.splitlines():
        if line.strip():
            assert line.lstrip().startswith("#"), f"Expected comment, got: {line!r}"
    # Key knobs should appear in comments
    assert "limit:" in block
    assert "strategy" in block
    assert "seed" in block


def test_render_with_limit_round_trips(tmp_path) -> None:
    """render() with data.limit produces a config that round-trips through load_config."""
    answers = {
        "run": {"name": "r"},
        "data": {
            "format": "coco",
            "train": {"annotations": "t.json", "images": "t/"},
            "augmentations": {"preset": "natural", "intensity": "medium"},
            "limit": {"train": 100, "val": 0.1},
        },
        "peft": {"method": "lora"},
        "train": {"epochs": 1, "loss": {"preset": "natural", "class_imbalance": "balanced"}},
    }
    rendered = sw.render(answers, run_mode="train")
    assert "train: 100" in rendered
    assert "val: 0.1" in rendered
    out = tmp_path / "c.yaml"
    out.write_text(rendered)
    cfg = load_config(out)
    assert cfg.data.limit.train == 100
    assert cfg.data.limit.val == pytest.approx(0.1)


def test_render_without_limit_has_commented_block(tmp_path) -> None:
    """render() with no data.limit emits commented discovery block and still loads."""
    answers = {
        "run": {"name": "r"},
        "data": {
            "format": "coco",
            "train": {"annotations": "t.json", "images": "t/"},
            "augmentations": {"preset": "natural", "intensity": "medium"},
        },
        "peft": {"method": "lora"},
        "train": {"epochs": 1, "loss": {"preset": "natural", "class_imbalance": "balanced"}},
    }
    rendered = sw.render(answers, run_mode="train")
    # The discovery comment block must appear
    assert "# limit:" in rendered
    out = tmp_path / "c.yaml"
    out.write_text(rendered)
    cfg = load_config(out)
    # No limit set => schema defaults (train/val both None)
    assert cfg.data.limit.train is None
    assert cfg.data.limit.val is None


def test_limit_step_in_steps_list_after_validation() -> None:
    """STEPS must contain a 'limit' step positioned after 'validation'."""
    ids = [s.id for s in sw.STEPS]
    assert "limit" in ids
    val_idx = ids.index("validation")
    limit_idx = ids.index("limit")
    assert limit_idx > val_idx, "limit step must come after validation step"


def test_step_fragment_shapes_updated_for_limit_step(monkeypatch) -> None:
    """Full run_wizard with limit step declining the confirm still works end-to-end."""
    # The limit step adds one ask_confirm call (decline) — include False in confirms
    _patch_prompts(
        monkeypatch,
        texts=["my-run", "ann.json", "imgs/", "5", ""],
        choices=["train", "coco", "none", "natural", "medium", "balanced", "lora"],
        confirms=[False],  # decline limit
    )
    monkeypatch.setattr(sw, "infer_class_imbalance", lambda *a, **k: "balanced")
    ctx = sw.Ctx(answers={}, cuda_available=False)
    answers = sw.run_wizard(ctx)
    assert answers["run"]["name"] == "my-run"
    assert answers["data"]["format"] == "coco"
    assert answers["train"]["epochs"] == 5
    assert "limit" not in answers.get("data", {})


def test_generate_config_happy_path_with_limit_step(tmp_path, monkeypatch) -> None:
    """generate_config works end-to-end when the limit step is answered (decline)."""
    _patch_prompts(
        monkeypatch,
        texts=["my-run", "ann.json", "imgs/", "0.1", "7", ""],
        choices=["train", "coco", "auto-split", "natural", "medium", "lora"],
        confirms=[False],  # decline limit
    )
    monkeypatch.setattr(sw, "measure_class_imbalance_ratio", lambda *a, **k: None)
    out = tmp_path / "c.yaml"
    sw.generate_config(out, force=False, cuda_available=False)
    cfg = load_config(out)
    assert cfg.run.name == "my-run"
    assert cfg.data.val_split is not None
    assert cfg.train.epochs == 7
