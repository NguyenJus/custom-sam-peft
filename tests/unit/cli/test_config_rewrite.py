"""Tests for src/custom_sam_peft/cli/_config_rewrite.py — in-place line-surgery helper."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from custom_sam_peft.cli._config_rewrite import _rewrite_sizing_block
from custom_sam_peft.config.loader import load_config


def _write_config_with_comments(path: Path) -> None:
    """Write a config YAML that includes unrelated comments in multiple sections."""
    content = """\
# This is the top-level comment — must survive rewrite
run:
  name: test-run
  output_dir: ./runs
  seed: 42

# model section comment
model:
  name: facebook/sam3.1
  local_dir: models/sam3.1
  checkpoint_file: sam3.1_multiplex.pt
  dtype: bfloat16  # original dtype comment

data:
  format: coco
  train:
    annotations: data/train.json
    images: data/train/

peft:
  # peft section comment
  method: lora  # original method comment
  r: 16  # original r comment
  alpha: 32
  dropout: 0.05

train:
  epochs: 10
  batch_size: 1  # original batch_size comment
  grad_accum_steps: 8  # original grad_accum_steps comment
  optimizer: auto
  learning_rate: 1.0e-4
  multiplex:
    classes_per_forward: 16

tracking:
  backend: none
"""
    path.write_text(content)


def test_rewrite_sizing_block_annotation_present(tmp_path: Path) -> None:
    """The annotation comment appears in the rewritten file."""
    cfg_path = tmp_path / "config.yaml"
    _write_config_with_comments(cfg_path)

    _rewrite_sizing_block(
        cfg_path,
        method="qlora",
        r=8,
        batch_size=2,
        grad_accum_steps=4,
        dtype="float16",
        annotation="# calibrated 2026-05-28",
    )

    body = cfg_path.read_text()
    assert "# calibrated 2026-05-28" in body


def test_rewrite_sizing_block_values_changed(tmp_path: Path) -> None:
    """The sized fields are updated to the new values."""
    cfg_path = tmp_path / "config.yaml"
    _write_config_with_comments(cfg_path)

    _rewrite_sizing_block(
        cfg_path,
        method="qlora",
        r=8,
        batch_size=2,
        grad_accum_steps=4,
        dtype="float16",
        annotation="# calibrated 2026-05-28",
    )

    parsed = yaml.safe_load(cfg_path.read_text())
    assert parsed["peft"]["method"] == "qlora"
    assert parsed["peft"]["r"] == 8
    assert parsed["train"]["batch_size"] == 2
    assert parsed["train"]["grad_accum_steps"] == 4
    assert parsed["model"]["dtype"] == "float16"


def test_rewrite_sizing_block_unrelated_lines_survive(tmp_path: Path) -> None:
    """Comments and lines unrelated to the sized fields are preserved."""
    cfg_path = tmp_path / "config.yaml"
    _write_config_with_comments(cfg_path)

    _rewrite_sizing_block(
        cfg_path,
        method="qlora",
        r=8,
        batch_size=2,
        grad_accum_steps=4,
        dtype="float16",
        annotation="# calibrated 2026-05-28",
    )

    body = cfg_path.read_text()
    # Top-level comment must survive
    assert "# This is the top-level comment" in body
    # model section comment must survive
    assert "# model section comment" in body
    # peft section comment must survive
    assert "# peft section comment" in body
    # Unrelated fields must survive
    assert "alpha: 32" in body
    assert "dropout: 0.05" in body
    assert "learning_rate: 1.0e-4" in body
    assert "epochs: 10" in body


def test_rewrite_sizing_block_still_parses_via_load_config(tmp_path: Path) -> None:
    """After rewrite, the file is still a valid TrainConfig."""
    cfg_path = tmp_path / "config.yaml"
    _write_config_with_comments(cfg_path)

    _rewrite_sizing_block(
        cfg_path,
        method="qlora",
        r=8,
        batch_size=2,
        grad_accum_steps=4,
        dtype="float16",
        annotation="# calibrated 2026-05-28",
    )

    cfg = load_config(cfg_path)
    assert cfg is not None
    assert cfg.peft.method == "qlora"
    assert cfg.peft.r == 8
    assert cfg.train.batch_size == 2
    assert cfg.train.grad_accum_steps == 4
    assert cfg.model.dtype == "float16"


# ---------------------------------------------------------------------------
# C1: depth-aware section matching
# ---------------------------------------------------------------------------


def _write_config_nested_same_key(path: Path) -> None:
    """Write a config where nested sub-keys share names with direct peft children.

    peft.extra.r = 99 must NOT be rewritten.
    peft.r = 16 is the REAL target and MUST be rewritten.
    """
    content = """\
run:
  name: test-run
  output_dir: ./runs
  seed: 42

model:
  name: facebook/sam3.1
  local_dir: models/sam3.1
  checkpoint_file: sam3.1_multiplex.pt
  dtype: bfloat16

data:
  format: coco
  train:
    annotations: data/train.json
    images: data/train/

peft:
  method: lora
  extra:
    r: 99
  r: 16
  alpha: 32
  dropout: 0.05

train:
  epochs: 10
  batch_size: 1
  grad_accum_steps: 8
  optimizer: auto
  learning_rate: 1.0e-4
  multiplex:
    classes_per_forward: 16

tracking:
  backend: none
"""
    path.write_text(content)


def test_rewrite_depth_aware_nested_key_untouched(tmp_path: Path) -> None:
    """C1: a nested key (peft.extra.r) must not be rewritten; only direct child peft.r is."""
    cfg_path = tmp_path / "config.yaml"
    _write_config_nested_same_key(cfg_path)

    _rewrite_sizing_block(
        cfg_path,
        method="qlora",
        r=8,
        batch_size=2,
        grad_accum_steps=4,
        dtype="float16",
        annotation="# calibrated 2026-05-28",
    )

    body = cfg_path.read_text()
    # The nested key must remain at 99.
    assert "r: 99" in body, "nested peft.extra.r was wrongly overwritten"

    # The real direct-child peft.r must be updated.
    parsed = yaml.safe_load(body)
    assert parsed["peft"]["r"] == 8, "real peft.r was not updated"
    assert parsed["peft"]["extra"]["r"] == 99, "peft.extra.r was corrupted"


# ---------------------------------------------------------------------------
# C2: missing keys raise ValueError
# ---------------------------------------------------------------------------


def _write_config_missing_grad_accum(path: Path) -> None:
    """Write a config that is missing grad_accum_steps under train."""
    content = """\
run:
  name: test-run
  output_dir: ./runs
  seed: 42

model:
  name: facebook/sam3.1
  local_dir: models/sam3.1
  checkpoint_file: sam3.1_multiplex.pt
  dtype: bfloat16

data:
  format: coco
  train:
    annotations: data/train.json
    images: data/train/

peft:
  method: lora
  r: 16
  alpha: 32
  dropout: 0.05

train:
  epochs: 10
  batch_size: 1
  optimizer: auto
  learning_rate: 1.0e-4
  multiplex:
    classes_per_forward: 16

tracking:
  backend: none
"""
    path.write_text(content)


def test_rewrite_missing_key_raises_value_error(tmp_path: Path) -> None:
    """C2: when grad_accum_steps is absent, _rewrite_sizing_block raises ValueError naming it."""
    cfg_path = tmp_path / "config.yaml"
    _write_config_missing_grad_accum(cfg_path)

    with pytest.raises(ValueError, match="grad_accum_steps"):
        _rewrite_sizing_block(
            cfg_path,
            method="qlora",
            r=8,
            batch_size=2,
            grad_accum_steps=4,
            dtype="float16",
            annotation="# calibrated 2026-05-28",
        )


# ---------------------------------------------------------------------------
# I1: annotation idempotency
# ---------------------------------------------------------------------------


def test_rewrite_annotation_idempotent(tmp_path: Path) -> None:
    """I1: running _rewrite_sizing_block twice leaves exactly one annotation line."""
    cfg_path = tmp_path / "config.yaml"
    _write_config_with_comments(cfg_path)

    for _ in range(2):
        _rewrite_sizing_block(
            cfg_path,
            method="qlora",
            r=8,
            batch_size=2,
            grad_accum_steps=4,
            dtype="float16",
            annotation="# calibrated 2026-05-28",
        )

    body = cfg_path.read_text()
    count = body.count("# calibrated 2026-05-28")
    assert count == 1, f"expected 1 annotation line after 2 rewrites, got {count}"


def test_rewrite_against_real_rendered_template(tmp_path: Path) -> None:
    """_rewrite_sizing_block must work against the ACTUAL rendered config_full.yaml
    template (comment scaffolding, blank lines, $qlora_block) — not only the
    hand-written minimal configs the other tests use. Locks the init/calibrate
    template contract that is otherwise exercised only on the GPU bake path.
    """
    from custom_sam_peft.cli.init_cmd import run_init

    out = tmp_path / "config.yaml"
    # CPU-safe: run_init renders the real template defaults (no GPU bake on CPU).
    run_init("coco-text-lora", out, force=True)

    _rewrite_sizing_block(
        out,
        method="qlora",
        r=32,
        batch_size=4,
        grad_accum_steps=2,
        dtype="float16",
        annotation="# formula-derived",
    )
    cfg = load_config(out)
    assert cfg.peft.method == "qlora"
    assert cfg.peft.r == 32
    assert cfg.train.batch_size == 4
    assert cfg.train.grad_accum_steps == 2
    assert cfg.model.dtype == "float16"
    assert "# formula-derived" in out.read_text()

    # Cross-tool idempotency: a second rewrite (as calibrate would do) replaces the
    # prior annotation rather than stacking it, and updates every sizing value.
    _rewrite_sizing_block(
        out,
        method="lora",
        r=8,
        batch_size=1,
        grad_accum_steps=8,
        dtype="bfloat16",
        annotation="# calibrated",
    )
    text = out.read_text()
    assert "# formula-derived" not in text
    assert "# calibrated" in text
    cfg2 = load_config(out)
    assert cfg2.peft.method == "lora"
    assert cfg2.peft.r == 8
    assert cfg2.train.batch_size == 1
    assert cfg2.model.dtype == "bfloat16"
