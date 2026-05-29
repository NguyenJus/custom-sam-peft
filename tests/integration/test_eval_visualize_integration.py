"""CPU integration smoke for eval --visualize (tiny stub + 2-class dataset)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from PIL import Image

from custom_sam_peft.config.schema import EvalConfig
from custom_sam_peft.eval.evaluator import Evaluator
from custom_sam_peft.eval.runner import run_eval


def _cfg(tmp_path: Path, *, visualize: bool = True) -> MagicMock:
    """A run_eval-compatible cfg mock backed by a real EvalConfig for the viz knobs."""
    cfg = MagicMock()
    cfg.data.format = "coco"
    cfg.data.val = MagicMock()
    cfg.data.val_split = None
    cfg.data.test = None
    cfg.data.normalize = None
    cfg.data.channel_semantics = "rgb"
    cfg.model.name = "facebook/sam3.1"
    cfg.peft.method = "lora"
    cfg.run.output_dir = str(tmp_path)
    eval_cfg = EvalConfig(
        mode="full", iou_thresholds=[0.5], batch_size=1, visualize=visualize, visualize_count=10
    )
    cfg.eval = eval_cfg
    return cfg


def test_run_eval_writes_composites(tmp_path, stub_model, tiny_text_dataset) -> None:
    cfg = _cfg(tmp_path, visualize=True)
    run_eval(
        cfg,
        checkpoint=None,
        split="val",
        output_dir=tmp_path,
        val_dataset=tiny_text_dataset,
        model=stub_model,
    )
    vis_dir = tmp_path / "visualizations"
    assert vis_dir.is_dir()
    pngs = sorted(p.name for p in vis_dir.glob("*.png"))
    assert pngs == ["img_0.png", "img_1.png"]  # both GT-bearing, capped at candidate count
    for p in vis_dir.glob("*.png"):
        Image.open(p).verify()
    assert (tmp_path / "metrics.json").exists()


def test_no_visualize_writes_no_composites(tmp_path, stub_model, tiny_text_dataset) -> None:
    cfg = _cfg(tmp_path, visualize=True)  # cfg on, flag off -> off.
    run_eval(
        cfg,
        checkpoint=None,
        split="val",
        output_dir=tmp_path,
        val_dataset=tiny_text_dataset,
        model=stub_model,
        visualize=False,
    )
    assert not (tmp_path / "visualizations").exists()
    assert (tmp_path / "metrics.json").exists()


def test_in_loop_evaluate_writes_no_composites(tmp_path, stub_model, tiny_text_dataset) -> None:
    """Calling Evaluator.evaluate directly (the in-loop path) writes NO visualizations/."""
    cfg = EvalConfig(mode="full", iou_thresholds=[0.5], batch_size=1, visualize=True)
    Evaluator(cfg).evaluate(stub_model, tiny_text_dataset)
    # The in-loop path does no disk I/O at all; no visualizations dir anywhere under tmp.
    assert not (tmp_path / "visualizations").exists()
