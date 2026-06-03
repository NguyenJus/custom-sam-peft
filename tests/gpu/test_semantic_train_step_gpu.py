"""GPU smoke: one semantic train_step on the real SAM3.1 wrapper (K=3, K=16).

Gated by ``@pytest.mark.gpu_bf16``, ``@requires_compatible_gpu``, and
``@requires_checkpoint``.  Not in CI by default.  Run with::

    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
      pytest -v --tb=short --no-cov \\
        tests/gpu/test_semantic_train_step_gpu.py

Parametrized over K ∈ {3, 16} (classes_per_forward) to exercise the
assembled-stack memory headroom (spec §14: K=16 multiplex + the held
graph-connected (B,K+1,H,W) slices) on the 16 GB card.  ONE train_step
per case to stay within the real-model freeze threshold.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

import custom_sam_peft.data.mask_png  # noqa: F401 — triggers @register("dataset","mask_png") side-effect
from custom_sam_peft._registry import lookup
from custom_sam_peft.config.schema import (
    DataConfig,
    DataSplit,
    ModelConfig,
    MultiplexConfig,
    PEFTConfig,
    RunConfig,
    SemanticDataConfig,
    SemanticLossConfig,
    TrainConfig,
    TrainHyperparams,
)
from custom_sam_peft.data.collate import collate_batch
from custom_sam_peft.models.sam3 import MULTIPLEX_CAP, load_sam31
from custom_sam_peft.peft_adapters.lora import apply_lora
from custom_sam_peft.runtime import Runtime
from custom_sam_peft.train.loop import OomState, train_step

pytestmark = [
    pytest.mark.gpu_bf16,
    pytest.mark.requires_compatible_gpu,
    pytest.mark.requires_checkpoint,
]


# ---------------------------------------------------------------------------
# Synthetic mask_png tree helpers
# ---------------------------------------------------------------------------


def _make_mask_png_tree(tmp_path: Path, k: int) -> tuple[Path, Path, Path]:
    """Build a minimal mask_png dataset tree with K concept classes.

    Layout::

        tmp_path/
          img/  a.png, b.png           -- B=2 RGB images
          lbl/  a.png, b.png           -- paired label PNGs
          class_map.json               -- {"0":"background","1":"c1",...,"K":"cK"}

    Label PNGs: pixel values 0..K (0=background, 1..K=concepts) plus one 255
    void pixel at (0,0).  Image size is 64x64 (small; transforms resize to 1008).
    """
    img_dir = tmp_path / "img"
    lbl_dir = tmp_path / "lbl"
    img_dir.mkdir()
    lbl_dir.mkdir()

    # K concept classes; place each class in a distinct tile of the label map.
    h = w = 64
    for stem in ("a", "b"):
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        Image.fromarray(rgb).save(img_dir / f"{stem}.png")

        lbl = np.zeros((h, w), dtype=np.uint8)
        # Distribute class ids 1..K across equal-height horizontal bands.
        band = max(1, h // (k + 1))
        for c in range(1, k + 1):
            row_start = (c - 1) * band
            row_end = min(c * band, h)
            lbl[row_start:row_end, :] = c
        lbl[0, 0] = 255  # void pixel
        Image.fromarray(lbl, mode="L").save(lbl_dir / f"{stem}.png")

    # class_map: "0" -> "background", "1" -> "c1", ..., "K" -> "cK"
    class_map: dict[str, str] = {"0": "background"}
    for c in range(1, k + 1):
        class_map[str(c)] = f"c{c}"
    cm_path = tmp_path / "class_map.json"
    cm_path.write_text(json.dumps(class_map))

    return img_dir, lbl_dir, cm_path


# ---------------------------------------------------------------------------
# Core parametrized test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("k", [3, 16])
def test_semantic_train_step_real_model(tmp_path: Path, k: int) -> None:
    """One semantic train_step on the real SAM3.1 wrapper — finite loss + real grad flow.

    Assertions (spec §11 GPU + §14 Risk):
    - ``r.losses["total"]`` is finite (math.isfinite)
    - ``set(r.losses) == {"ce", "region", "total"}`` (semantic loss keys)
    - ``not r.skipped`` (step was not NaN-skipped)
    - ``r.grad_norm is not None and r.grad_norm > 0.0`` (real backward + optimizer step)
    """
    # ------------------------------------------------------------------
    # 1. Build synthetic mask_png tree (B=2, K concept classes)
    # ------------------------------------------------------------------
    img_dir, lbl_dir, cm_path = _make_mask_png_tree(tmp_path, k)

    # ------------------------------------------------------------------
    # 2. Build dataset via the registered builder
    # ------------------------------------------------------------------
    cfg_dict = {
        "format": "mask_png",
        "train": {"images": str(img_dir), "annotations": str(lbl_dir)},
        "val": None,
        "semantic": {
            "class_map": str(cm_path),
            "ignore_index": 255,
            "label_suffix": ".png",
        },
        "channels": 3,
        "text_prompt": {"mode": "all"},
    }
    builder = lookup("dataset", "mask_png")
    ds = builder(cfg_dict, model_name="facebook/sam3.1", pipeline="train")

    batch = collate_batch([ds[0], ds[1]])

    # ------------------------------------------------------------------
    # 3. Build TrainConfig (task=semantic, mask_png, K=k)
    # ------------------------------------------------------------------
    train_cfg = TrainConfig(
        run=RunConfig(name="semantic-gpu-smoke"),
        model=ModelConfig(
            name="facebook/sam3.1",
            local_dir="models/sam3.1",
            checkpoint_file="sam3.1_multiplex.pt",
            dtype="bfloat16",
            device="cuda",
        ),
        data=DataConfig(
            format="mask_png",
            train=DataSplit(
                images=str(img_dir),
                annotations=str(lbl_dir),
            ),
            semantic=SemanticDataConfig(
                class_map=str(cm_path),
                ignore_index=255,
                label_suffix=".png",
            ),
        ),
        peft=PEFTConfig(method="lora"),
        train=TrainHyperparams(
            epochs=1,
            batch_size=2,
            grad_accum_steps=1,
            optimizer="adamw",
            learning_rate=1.0e-4,
            lr_schedule="cosine",
            warmup_steps=0,
            log_every=1,
            multiplex=MultiplexConfig(
                classes_per_forward=k,
            ),
            semantic_loss=SemanticLossConfig(
                preset="natural",
                class_imbalance="balanced",
            ),
        ),
        task="semantic",
    )

    # ------------------------------------------------------------------
    # 4. Load real SAM3.1 wrapper + apply LoRA (exact runner.py path)
    # ------------------------------------------------------------------
    model = load_sam31(train_cfg.model, channels=3, channel_semantics="rgb")
    apply_lora(model, train_cfg.peft)
    model.train()

    # ------------------------------------------------------------------
    # 5. Build optimizer, scheduler, Runtime, OomState
    # ------------------------------------------------------------------
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=train_cfg.train.learning_rate)

    total_steps = 1
    warmup = train_cfg.train.warmup_steps

    def _lr_lambda(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(warmup, 1)
        progress = (step - warmup) / max(total_steps - warmup, 1)
        cos_val = math.cos(math.pi * min(progress, 1.0))
        return 0.5 * (1.0 + float(cos_val))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)

    runtime = Runtime.from_config(device="cuda", dtype="bfloat16")

    oom_state = OomState(
        micro_batch_size=train_cfg.train.batch_size,
        effective_K=min(k, MULTIPLEX_CAP),
    )

    # ------------------------------------------------------------------
    # 6. Reset peak VRAM stats before the step
    # ------------------------------------------------------------------
    torch.cuda.reset_peak_memory_stats()

    # ------------------------------------------------------------------
    # 7. ONE train_step
    # ------------------------------------------------------------------
    r = train_step(
        model,
        batch,
        optimizer,
        scheduler,
        train_cfg,
        class_names=ds.class_names,
        global_step=0,
        nan_streak=0,
        runtime=runtime,
        oom_state=oom_state,
    )

    peak_vram_gb = torch.cuda.max_memory_allocated() / 1e9

    # ------------------------------------------------------------------
    # 8. Assertions
    # ------------------------------------------------------------------
    assert not r.skipped, f"K={k}: train_step skipped (non-finite loss); losses={r.losses}"
    assert set(r.losses) == {"ce", "region", "total"}, (
        f"K={k}: unexpected loss keys: {set(r.losses)}"
    )
    assert math.isfinite(r.losses["total"]), f"K={k}: total loss is not finite: {r.losses['total']}"
    assert math.isfinite(r.losses["ce"]), f"K={k}: ce loss is not finite: {r.losses['ce']}"
    assert math.isfinite(r.losses["region"]), (
        f"K={k}: region loss is not finite: {r.losses['region']}"
    )
    assert r.grad_norm is not None, f"K={k}: grad_norm is None — optimizer step did not fire"
    assert r.grad_norm > 0.0, (
        f"K={k}: grad_norm={r.grad_norm} — gradients did not flow (backward not executed)"
    )

    # Log peak VRAM for §14 headroom observability (not a hard assertion — OOM
    # on the 16 GB card for K=16 is a §14 finding, reported via DONE_WITH_CONCERNS).
    import logging

    logging.getLogger(__name__).info(
        "[K=%d] losses=%s grad_norm=%.4f peak_vram=%.2fGB",
        k,
        r.losses,
        r.grad_norm,
        peak_vram_gb,
    )
