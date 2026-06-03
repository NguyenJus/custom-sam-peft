"""predict auto-dtype is routed through coerce_dtype_for_capability (§6.4)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from custom_sam_peft.predict.runner import PredictOptions, _resolve_config


def _opts(**over: Any) -> PredictOptions:
    base = dict(
        images=Path("img"),
        prompts="a",
        output=Path("out"),
        checkpoint=None,
        merge_adapter=True,
        config=None,
        score_threshold=0.3,
        top_k=100,
        save_masks="rle",
        visualize=False,
        device="auto",
        dtype="auto",
        seed=0,
        dry_run=True,
        verbose=False,
        use_onnx=None,
        batch_size="auto",
    )
    base.update(over)
    return PredictOptions(**base)  # type: ignore[arg-type]


def test_auto_dtype_routed_through_coercion(monkeypatch: Any) -> None:
    calls: dict[str, Any] = {}

    def fake_coerce(dtype: torch.dtype, **k: Any) -> torch.dtype:
        calls["dtype"] = dtype
        calls["kwargs"] = k
        return torch.float16  # pretend sub-CC-8.0 coercion

    monkeypatch.setattr(
        "custom_sam_peft.predict.runner.coerce_dtype_for_capability", fake_coerce, raising=False
    )
    # Force the cuda branch deterministically without a GPU.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    rcfg = _resolve_config(_opts(device="cuda", dtype="auto"))
    assert calls["dtype"] is torch.bfloat16
    assert rcfg.dtype is torch.float16
