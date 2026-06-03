"""Regression: export round-trips task: semantic in the bundled config.yaml.

Spec §10.4: export is task-agnostic at the artifact level — it ships the LoRA
adapter + the run's config.yaml without rewriting it.  This test confirms that
a semantic TrainConfig serialised via the trainer's mechanism
(cfg.model_dump(mode="json") → yaml.safe_dump → config.yaml) reloads intact
through load_config, so the export command never loses task / semantic_loss.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from custom_sam_peft.cli.main import app
from custom_sam_peft.config.loader import load_config
from custom_sam_peft.config.schema import TrainConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_semantic_config(class_map_path: Path) -> TrainConfig:
    """Minimal valid semantic TrainConfig (mask_png, task=semantic)."""
    return TrainConfig.model_validate(
        {
            "task": "semantic",
            "run": {"name": "sem-test", "output_dir": "./runs", "seed": 0},
            "data": {
                "format": "mask_png",
                "train": {"annotations": str(class_map_path), "images": "./images/"},
                "val": {"annotations": str(class_map_path), "images": "./images/"},
                "semantic": {"class_map": str(class_map_path), "ignore_index": 255},
            },
            "peft": {"method": "lora"},
            "train": {
                "epochs": 1,
                "semantic_loss": {
                    "preset": "natural",
                    "source": "marginalize",
                    "query_reduce": "max",
                    "background_logit": 0.5,
                },
            },
        }
    )


def _write_config_yaml(cfg: TrainConfig, dest: Path) -> None:
    """Serialise exactly the way the trainer writes config.yaml (trainer.py:666-675)."""
    cfg_dict = cfg.model_dump(mode="json")
    dest.write_text(yaml.safe_dump(cfg_dict))


# ---------------------------------------------------------------------------
# (a) Config round-trip — core regression
# ---------------------------------------------------------------------------


def test_semantic_config_round_trips_through_yaml(tmp_path: Path) -> None:
    """task and semantic_loss survive model_dump → yaml.safe_dump → load_config."""
    class_map = tmp_path / "classes.json"
    class_map.write_text('{"0": "background", "1": "road"}')

    cfg = _build_semantic_config(class_map)
    config_path = tmp_path / "config.yaml"
    _write_config_yaml(cfg, config_path)

    reloaded = load_config(config_path)

    assert reloaded.task == "semantic"
    assert reloaded.train.semantic_loss.preset == "natural"
    assert reloaded.train.semantic_loss.source == "marginalize"
    assert reloaded.train.semantic_loss.query_reduce == "max"
    assert abs(reloaded.train.semantic_loss.background_logit - 0.5) < 1e-9
    assert reloaded.data.semantic is not None
    assert reloaded.data.semantic.ignore_index == 255


# ---------------------------------------------------------------------------
# (b) Export discovery — CLI loads the config without stripping task
# ---------------------------------------------------------------------------


def _patch_export(monkeypatch: pytest.MonkeyPatch, captured: dict[str, object]) -> None:
    """Replace run_export so the CLI test stays CPU-only / model-free."""
    import custom_sam_peft.cli.export_cmd as export_cmd

    def _fake_run_export(
        cfg: object,
        checkpoint: object,
        *,
        merge: bool = False,
        output: object = None,
    ) -> object:
        captured["cfg"] = cfg
        if merge:
            out = output if output is not None else (checkpoint.parent / "merged")  # type: ignore[union-attr]
            captured["saved_merged_to"] = out
        else:
            if output is None:
                raise ValueError(
                    "output is required when not merging (refusing to overwrite source checkpoint)"
                )
            out = output
            captured["saved_adapter_to"] = out
        return out

    monkeypatch.setattr(export_cmd, "run_export", _fake_run_export)


@pytest.fixture()
def semantic_run_dir(tmp_path: Path) -> Path:
    """A run-dir-shaped tree with a semantic config.yaml beside the adapter."""
    class_map = tmp_path / "classes.json"
    class_map.write_text('{"0": "background", "1": "road"}')

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "adapter").mkdir()

    cfg = _build_semantic_config(class_map)
    _write_config_yaml(cfg, run_dir / "config.yaml")
    return run_dir


def test_export_discovers_and_loads_semantic_config(
    semantic_run_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CLI export auto-discovers a semantic config.yaml and loads it intact."""
    from typer.testing import CliRunner

    captured: dict[str, object] = {}
    _patch_export(monkeypatch, captured)

    out = semantic_run_dir.parent / "exported_adapter"
    result = CliRunner().invoke(
        app,
        ["export", "--checkpoint", str(semantic_run_dir / "adapter"), "--output", str(out)],
    )
    assert result.exit_code == 0, result.output
    assert captured["saved_adapter_to"] == out

    loaded_cfg = captured["cfg"]
    assert isinstance(loaded_cfg, TrainConfig)
    assert loaded_cfg.task == "semantic"
    assert loaded_cfg.train.semantic_loss.source == "marginalize"
