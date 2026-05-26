"""The eval-runner --split val gate accepts HF + hf.split_val (spec §12.5)."""

from __future__ import annotations

import pytest

from custom_sam_peft.config.schema import TrainConfig


def _hf_cfg(split_val: str | None) -> TrainConfig:
    hf: dict[str, object] = {"name": "tiny/ds"}
    if split_val is not None:
        hf["split_val"] = split_val
    return TrainConfig.model_validate(
        {
            "run": {"name": "r"},
            "model": {},
            "data": {
                "format": "hf",
                "train": {"annotations": "unused", "images": "unused"},
                "val": None,
                "prompt_mode": "text",
                "hf": hf,
            },
            "peft": {"method": "lora"},
            "train": {"epochs": 1},
        }
    )


def _gate_only(cfg: TrainConfig, split: str) -> None:
    """Replicate eval/runner's --split val gate in isolation (no model/data load)."""
    _hf_val = (
        cfg.data.format == "hf" and cfg.data.hf is not None and cfg.data.hf.split_val is not None
    )
    if split == "val" and cfg.data.val is None and cfg.data.val_split is None and not _hf_val:
        raise ValueError(
            "--split val requires data.val, data.val_split, or data.hf.split_val in config; "
            "got none."
        )


def test_gate_accepts_hf_split_val() -> None:
    _gate_only(_hf_cfg("myval"), "val")  # must not raise


def test_gate_rejects_hf_without_split_val() -> None:
    with pytest.raises(ValueError, match=r"data\.hf\.split_val"):
        _gate_only(_hf_cfg(None), "val")
