"""Regression test: load_qlora must reload into an already-attached PeftModel.

Gated by gpu_t4 + requires_compatible_gpu + requires_checkpoint + requires_bnb.

This exercises the path that close_out uses: load_adapter(model, best_adapter)
is called while the trained QLoRA PeftModel is still attached. Before the fix,
load_qlora raised RuntimeError; after the fix it reloads adapter weights in
place (mirroring load_lora's resume branch).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

import custom_sam_peft.train.runner as _runner_mod
from custom_sam_peft.config.loader import load_config
from custom_sam_peft.train.runner import run_training
from tests.gpu.conftest import _bnb_available, _RecordingTracker

pytestmark = [
    pytest.mark.gpu_t4,
    pytest.mark.requires_compatible_gpu,
    pytest.mark.requires_checkpoint,
]

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "examples" / "gpu_smoke_qlora.yaml"


@pytest.mark.requires_bnb
@pytest.mark.skipif(not _bnb_available(), reason="bitsandbytes not installed")
def test_qlora_load_into_attached_peft_model(
    tmp_path: Path,
    tiny_coco_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """load_qlora reloads into an already-attached PeftModel (mirrors load_lora resume branch).

    Reproduces the close_out best-restore bug: load_adapter is called while the
    trained QLoRA PeftModel is still attached. Pre-fix: RuntimeError. Post-fix:
    adapter weights are reloaded, model remains usable.
    """
    cfg = load_config(
        CONFIG_PATH,
        overrides=[
            f"data.train.annotations={tiny_coco_dir / 'annotations.json'}",
            f"data.train.images={tiny_coco_dir / 'images'}",
            f"data.val.annotations={tiny_coco_dir / 'annotations.json'}",
            f"data.val.images={tiny_coco_dir / 'images'}",
            f"run.output_dir={tmp_path}",
            "train.epochs=2",
            "train.log_every=1",
        ],
    )

    # Capture the live wrapper after apply_qlora attaches the PeftModel.
    # We monkeypatch the peft factory call inside run_training by intercepting
    # load_sam31 to stash the wrapper reference after PEFT is applied.
    captured: dict[str, Any] = {}

    from custom_sam_peft.models.sam3 import load_sam31 as _real_load_sam31

    def _capturing_load_sam31(*args: Any, **kwargs: Any) -> Any:
        wrapper = _real_load_sam31(*args, **kwargs)
        captured["wrapper"] = wrapper
        return wrapper

    monkeypatch.setattr(_runner_mod, "load_sam31", _capturing_load_sam31)

    tracker = _RecordingTracker()
    monkeypatch.setattr("custom_sam_peft.train.runner.build_tracker", lambda *_a, **_kw: tracker)
    run_training(cfg)

    # Locate the adapter written by close_out (run_dir/adapter).
    runs = sorted(tmp_path.glob("gpu-smoke-qlora-*"))
    assert runs, f"no run dir under {tmp_path}"
    adapter_dir = runs[-1] / "adapter"
    assert adapter_dir.is_dir(), f"close_out wrote no adapter dir under {runs[-1]}"

    wrapper = captured.get("wrapper")
    assert wrapper is not None, "failed to capture wrapper from run_training"
    assert wrapper.peft_model is not None, "wrapper.peft_model is None after training"

    # Simulate the close_out best-restore: call load_adapter on the wrapper
    # that ALREADY has the QLoRA PeftModel attached. Pre-fix: RuntimeError.
    from custom_sam_peft.train.checkpoint import load_adapter

    load_adapter(wrapper, adapter_dir)  # must NOT raise

    # Model must remain usable: at least one lora_ param, all finite.
    assert wrapper.peft_model is not None, "peft_model should still be attached after reload"
    lora_params = [(n, p) for n, p in wrapper.peft_model.named_parameters() if "lora_" in n]
    assert lora_params, "no lora_ parameters found in peft_model after reload"
    for name, param in lora_params:
        assert torch.isfinite(param).all(), f"non-finite lora param after reload: {name}"
