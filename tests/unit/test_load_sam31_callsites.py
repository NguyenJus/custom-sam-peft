"""Task 15: train runner threads cfg.data.channels / channel_semantics into load_sam31.

The substantive contract: a production call site (run_training) passes the REAL
configured channels + channel_semantics, not the defaults (3, "rgb").
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest


def _make_cfg(tmp_path: Path) -> MagicMock:
    """Build a MagicMock TrainConfig with non-default channels so we can detect threading."""
    cfg = MagicMock()
    cfg.run.output_dir = str(tmp_path)
    cfg.run.name = "ch-test"
    cfg.run.seed = 0
    cfg.data.format = "coco"
    cfg.data.prompt_mode = "text"
    cfg.data.model_dump.return_value = {"format": "coco"}
    cfg.data.channels = 4
    cfg.data.channel_semantics = "rgba"
    cfg.model.name = "facebook/sam3.1"
    cfg.peft.method = "lora"
    cfg.tracking.backend = "none"
    cfg.tracking.wandb.project = "custom_sam_peft"
    cfg.tracking.wandb.entity = None
    cfg.data.limit.train = None
    cfg.data.limit.val = None
    return cfg


def test_train_runner_passes_data_channels(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """run_training threads cfg.data.channels / cfg.data.channel_semantics into load_sam31."""
    import custom_sam_peft.train.runner as R

    captured: dict[str, object] = {}

    def fake_load_sam31(
        model_cfg: object, *, channels: int = 3, channel_semantics: str = "rgb"
    ) -> object:
        captured["channels"] = channels
        captured["semantics"] = channel_semantics
        raise SystemExit("stop after load")  # short-circuit the heavy path

    monkeypatch.setattr(R, "load_sam31", fake_load_sam31)

    # Patch away everything that runs before load_sam31 to avoid heavy I/O.
    from custom_sam_peft.data.val_source import ValSource

    monkeypatch.setattr(
        R,
        "resolve_val_source",
        lambda _cfg, run_dir=None: ValSource(
            mode="explicit",
            train_ids=None,
            val_ids=None,
            realized_fraction=None,
            per_class_counts=None,
            missing_in_val=None,
            fraction_requested=None,
            seed_used=None,
        ),
    )
    monkeypatch.setattr(R, "save_val_source", lambda _vs, _run_dir: None)
    monkeypatch.setattr(R, "_log_val_source", lambda _vs: None)

    def fake_lookup(kind: str, name: str) -> object:
        # dataset builder — returns a zero-length mock dataset
        return lambda *a, **kw: MagicMock(__len__=lambda self: 0, class_names=[])

    monkeypatch.setattr(R, "lookup", fake_lookup)

    cfg = _make_cfg(tmp_path)

    with pytest.raises(SystemExit, match="stop after load"):
        R.run_training(cfg)

    assert captured == {"channels": 4, "semantics": "rgba"}, (
        f"Expected channels=4/rgba but got {captured!r}. "
        "The train runner is NOT threading cfg.data.channels/channel_semantics into load_sam31."
    )
