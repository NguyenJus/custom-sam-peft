# tests/unit/test_semantic_presets.py
"""SEMANTIC_PRESET_TABLE completeness + resolve + override-WARN + sidecar (§7.3)."""

from __future__ import annotations

import ast
import logging
from pathlib import Path

from custom_sam_peft.config.schema import SemanticLossConfig
from custom_sam_peft.models.losses.semantic_presets import (
    SEMANTIC_PRESET_TABLE,
    dump_semantic_loss_bundle,
    resolve,
)

_REAL_PRESETS = ("natural", "medical", "satellite", "microscopy")
_IMBALANCE = ("balanced", "moderate", "severe")


def test_table_is_complete_all_cells():
    # natural/medical/satellite x 3 imbalance = 9 stored, + 3 microscopy alias keys = 12 total.
    # Every (preset, ci) including microscopy must be present.
    for p in _REAL_PRESETS:
        for ci in _IMBALANCE:
            assert (p, ci) in SEMANTIC_PRESET_TABLE, f"missing cell ({p}, {ci})"
    assert len(SEMANTIC_PRESET_TABLE) == 12  # 9 stored (3 presets x 3 ci) + 3 microscopy alias


def test_microscopy_is_alias_of_medical():
    for ci in _IMBALANCE:
        assert SEMANTIC_PRESET_TABLE[("microscopy", ci)] == SEMANTIC_PRESET_TABLE[("medical", ci)]


def test_every_cell_has_sem_family_and_weights():
    for key, cell in SEMANTIC_PRESET_TABLE.items():
        assert "sem_family" in cell, key
        assert "w_ce" in cell and "w_region" in cell, key


def test_resolve_natural_balanced_is_ce_dice_samed_weights():
    r = resolve(SemanticLossConfig(preset="natural", class_imbalance="balanced"))
    assert r.sem_family == "ce_dice"
    assert r.w_ce == 0.2 and r.w_region == 0.8  # SAMed (S)


def test_resolve_override_wins():
    r = resolve(
        SemanticLossConfig(
            preset="natural",
            class_imbalance="balanced",
            overrides={"sem_family": "boundary", "boundary_weight": 0.3},
        )
    )
    assert r.sem_family == "boundary"
    assert r.boundary_weight == 0.3


def test_locked_off_warns(caplog):
    # natural preset overriding to focal_tversky/boundary WARNs (§7.3).
    with caplog.at_level(logging.WARNING):
        resolve(
            SemanticLossConfig(
                preset="natural",
                class_imbalance="balanced",
                overrides={"sem_family": "focal_tversky"},
            )
        )
    assert any("override" in r.message.lower() for r in caplog.records)


def test_dump_sidecar_shape():
    bundle = dump_semantic_loss_bundle(
        SemanticLossConfig(preset="medical", class_imbalance="moderate")
    )
    assert bundle["preset"] == "medical"
    assert bundle["class_imbalance"] == "moderate"
    assert "resolved" in bundle and "sem_family" in bundle["resolved"]
    assert "term_classes" in bundle
    assert "library_version" in bundle


def test_semantic_presets_is_torch_free():
    # The resolver must be importable without torch (so `csp doctor` + schema tests
    # don't drag torch in). Verified structurally via AST — the module source must
    # not import torch. Mirrors tests/unit/test_data_import_boundary.py's approach.
    src = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "custom_sam_peft"
        / "models"
        / "losses"
        / "semantic_presets.py"
    )
    tree = ast.parse(src.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith("torch"), f"imports {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            assert not mod.startswith("torch"), f"imports from {mod}"
