"""QLoRA resume smoke: split the 50-step QLoRA overfit budget across a save/load
boundary to pin that bnb 4-bit quant_state survives the resume seam.

Gated by `@pytest.mark.gpu_t4`, `@requires_compatible_gpu`, `@requires_checkpoint`,
plus per-test `@pytest.mark.requires_bnb` and `@pytest.mark.skipif(not _bnb_available())`.
Not in CI by default. Run with:
    pytest -m gpu_t4 tests/gpu/test_real_train_qlora_resume.py -v

Phase A trains ~26 grad steps (epochs=13 against 2-image tiny_coco, batch=1,
just past save_every=25 to land a checkpoint). Phase B resumes from that
checkpoint and completes the test's own 2-image overfit smoke budget
(epochs=25, i.e. 50 forward steps over the 2-image set) — this is NOT the
production training default (which is 160 epochs, set in init_cmd.py /
setup_wizard.py; see also #195 tracking this smoke budget's speed and
convergence). Net GPU time is roughly one extra
`test_qlora_overfits_in_50_steps`; the resume seam can only be exercised
end-to-end against real bnb 4-bit weights (CPU stub at
tiny_sam3_lora_stub.py cannot replicate Linear4bit). See spec §6.1 T1
and §7.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from custom_sam_peft.config.loader import load_config
from custom_sam_peft.train.runner import run_training
from tests.gpu.conftest import _bnb_available, _RecordingTracker

pytestmark = [
    pytest.mark.gpu_t4,
    pytest.mark.requires_compatible_gpu,
    pytest.mark.requires_checkpoint,
]

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "examples" / "gpu_smoke_qlora.yaml"
VRAM_CEIL_GB = 10.0  # Same ceiling as test_real_train_qlora.py.


def _load_cfg(tmp_path: Path, tiny_coco_dir: Path, *, epochs: int) -> object:
    return load_config(
        CONFIG_PATH,
        overrides=[
            f"data.train.annotations={tiny_coco_dir / 'annotations.json'}",
            f"data.train.images={tiny_coco_dir / 'images'}",
            f"data.val.annotations={tiny_coco_dir / 'annotations.json'}",
            f"data.val.images={tiny_coco_dir / 'images'}",
            f"run.output_dir={tmp_path}",
            f"train.epochs={epochs}",
            # save_every=25 lands one checkpoint at step 25, midway through
            # the ~50-step total budget. log_every=1 so every step's scalar
            # is captured for finiteness checks.
            "train.save_every=25",
            "train.log_every=1",
        ],
    )


@pytest.mark.requires_bnb
@pytest.mark.skipif(not _bnb_available(), reason="bitsandbytes not installed")
def test_qlora_resume_survives_quant_state_roundtrip(
    tmp_path: Path,
    tiny_coco_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.cuda.reset_peak_memory_stats()

    # --- Phase A: train ~26 grad steps so save_every=25 fires. ---
    tracker_a = _RecordingTracker()
    monkeypatch.setattr("custom_sam_peft.train.runner.build_tracker", lambda *_a, **_kw: tracker_a)
    cfg_short = _load_cfg(tmp_path, tiny_coco_dir, epochs=13)
    run_training(cfg_short)

    # Locate the checkpoint produced by phase A.
    runs = sorted(tmp_path.glob("gpu-smoke-qlora-*"))
    assert runs, f"phase A wrote no run dir under {tmp_path}"
    ckpts_a = sorted((runs[-1] / "checkpoints").glob("step_*"))
    assert ckpts_a, f"phase A wrote no checkpoint under {runs[-1] / 'checkpoints'}"
    resume_dir = ckpts_a[-1]

    losses_a = [s["loss/total"] for _, s in tracker_a.scalars if s["loss/total"] > 0]
    assert losses_a, "phase A logged no loss scalars"
    assert all(math.isfinite(v) for _, s in tracker_a.scalars for v in s.values()), (
        "phase A logged a non-finite scalar"
    )

    # --- Phase B: resume from phase A's checkpoint, complete the 50-step budget. ---
    tracker_b = _RecordingTracker()
    monkeypatch.setattr("custom_sam_peft.train.runner.build_tracker", lambda *_a, **_kw: tracker_b)
    cfg_full = _load_cfg(tmp_path, tiny_coco_dir, epochs=25)
    run_training(cfg_full, resume_from=resume_dir)

    losses_b = [s["loss/total"] for _, s in tracker_b.scalars if s["loss/total"] > 0]
    assert losses_b, "phase B (resumed) logged no loss scalars"
    assert all(math.isfinite(v) for _, s in tracker_b.scalars for v in s.values()), (
        "phase B logged a non-finite scalar"
    )
    assert math.isfinite(losses_b[-1]), f"phase B final loss not finite: {losses_b[-1]}"

    # Final adapter state: at least one lora_ param, every lora_ param finite.
    # Locate the run dir produced by phase B (most recent under tmp_path).
    runs_b = sorted(tmp_path.glob("gpu-smoke-qlora-*"))
    assert len(runs_b) >= 2, f"phase B did not create a fresh run dir: {runs_b}"
    final_ckpts = sorted((runs_b[-1] / "checkpoints").glob("step_*"))
    assert final_ckpts, "phase B wrote no final checkpoint"
    adapter_dir = final_ckpts[-1] / "adapter"
    assert adapter_dir.exists(), f"phase B wrote no adapter dir under {final_ckpts[-1]}"
    import safetensors.torch  # peft's default adapter format

    adapter_safetensors = adapter_dir / "adapter_model.safetensors"
    adapter_bin = adapter_dir / "adapter_model.bin"
    if adapter_safetensors.exists():
        adapter_state = safetensors.torch.load_file(str(adapter_safetensors))
    elif adapter_bin.exists():
        adapter_state = torch.load(adapter_bin, map_location="cpu")
    else:
        raise AssertionError(
            f"adapter_dir {adapter_dir} contains neither adapter_model.safetensors "
            f"nor adapter_model.bin (peft naming variant?): {list(adapter_dir.iterdir())}"
        )
    lora_params = {k: v for k, v in adapter_state.items() if "lora_" in k}
    assert lora_params, "no lora_ params in final adapter state"
    for name, t in lora_params.items():
        assert torch.isfinite(t).all(), f"non-finite lora param: {name}"

    peak_vram_gb = torch.cuda.max_memory_allocated() / 1e9
    assert peak_vram_gb <= VRAM_CEIL_GB, (
        f"peak VRAM {peak_vram_gb:.2f}GB exceeded ceiling {VRAM_CEIL_GB}GB"
    )
