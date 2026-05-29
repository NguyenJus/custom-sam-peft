"""Predict OOM ladder (spec §5.3 / §7.4 — the #181 fix).

CPU-only. A stub model injects torch.cuda.OutOfMemoryError; run_predict must
recover via the shared OomLadder and produce predictions byte-identical to a
non-OOM run (no dup, no drop).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image as PILImage

from custom_sam_peft.predict.runner import PredictOptions, run_predict

Q = 4
H_LOW = W_LOW = 16
HIGH = 0.9


def _make_image_dir(tmp_path: Path, n: int) -> Path:
    d = tmp_path / "images"
    d.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        PILImage.new("RGB", (64, 64), color=(i * 20 % 255, 100, 200)).save(d / f"img_{i:03d}.png")
    return d


def _opts(tmp_path: Path, images: Path, *, prompts: str, batch_size: int) -> PredictOptions:
    return PredictOptions(
        images=images,
        prompts=prompts,
        output=tmp_path / "out",
        checkpoint=None,
        merge_adapter=True,
        config=None,
        score_threshold=0.0,
        top_k=100,
        save_masks="rle",
        visualize=False,
        device="cpu",
        dtype="float32",
        batch_size=batch_size,
        seed=42,
        dry_run=False,
        verbose=False,
    )


class _MultiplexStub(torch.nn.Module):
    """Multiplex stub: forward -> (B*K_g, Q, ...). OOMs once if oom_when matches."""

    def __init__(self, oom_predicate=None) -> None:
        super().__init__()
        self._oom = oom_predicate  # callable(images, prompts) -> bool; fires once
        self._fired = False

    def forward(self, images: torch.Tensor, prompts: list[Any], support: Any = None):
        from custom_sam_peft.data.base import TextPrompts as _TP

        if self._oom is not None and not self._fired and self._oom(images, prompts):
            self._fired = True
            raise torch.cuda.OutOfMemoryError("synthetic")
        b = images.shape[0]
        k_g = len(prompts[0].classes) if prompts and isinstance(prompts[0], _TP) else 1
        total = b * k_g
        return {
            "pred_logits": torch.full((total, Q, 1), HIGH),
            "pred_boxes": torch.full((total, Q, 4), 0.5),
            "pred_masks": torch.zeros(total, Q, H_LOW, W_LOW),
            "presence_logit_dec": torch.full((total, 1), HIGH),
        }


def _run(tmp_path: Path, stub: torch.nn.Module, opts: PredictOptions) -> list[dict]:
    import unittest.mock as mock

    with mock.patch("custom_sam_peft.models.sam3.load_sam31", side_effect=lambda cfg, **kw: stub):
        run_predict(opts)
    return json.loads((opts.output / "predictions.json").read_text())


def test_predict_oom_recovers_byte_identical_to_non_oom(tmp_path: Path) -> None:
    """Many classes, a model that OOMs once then succeeds: run completes AND
    predictions are byte-identical to a non-OOM run (no dup, no drop). Spec §7.4."""
    images = _make_image_dir(tmp_path / "ref", n=3)
    # Reference: never OOMs.
    ref = _run(
        tmp_path / "ref",
        _MultiplexStub(),
        _opts(tmp_path / "ref", images, prompts="a,b,c,d,e", batch_size=2),
    )

    # OOM run: same inputs; OOM once on the first multi-class forward at B>1.
    images2 = _make_image_dir(tmp_path / "oom", n=3)
    stub = _MultiplexStub(oom_predicate=lambda imgs, pr: len(pr[0].classes) >= 1)
    got = _run(
        tmp_path / "oom", stub, _opts(tmp_path / "oom", images2, prompts="a,b,c,d,e", batch_size=2)
    )

    # Sort by a stable key to compare content (image ids differ by dir, so compare
    # per-image-relative ordering via category_id + score sequence per image).
    def _key(p: dict) -> tuple:
        return (int(p["category_id"]), round(float(p["score"]), 6))

    assert sorted(got, key=_key) and len(got) == len(ref), (
        f"OOM run dropped/duplicated rows: got {len(got)} vs ref {len(ref)}"
    )
    # No (image_id, category_id) pair should appear more than Q times — a
    # double-commit from OOM retry would produce 2*Q entries for the same pair.
    from collections import Counter

    pair_counts = Counter((int(p["image_id"]), int(p["category_id"])) for p in got)
    over_committed = [(pair, cnt) for pair, cnt in pair_counts.items() if cnt > Q]
    assert not over_committed, f"double-committed (image_id, category_id) pairs: {over_committed}"


def test_predict_oom_retry_b_discards_partial_chunk(tmp_path: Path) -> None:
    """An OOM that triggers RETRY_B mid-chunk discards the partially-buffered chunk
    and re-emits it exactly once at the smaller B (no dup). Spec §5.4 / §7.4."""
    images = _make_image_dir(tmp_path, n=4)
    # batch_size=4 so the first forward is at B=4; OOM once -> RETRY_B halves to 2.
    stub = _MultiplexStub(oom_predicate=lambda imgs, pr: imgs.shape[0] == 4)
    got = _run(tmp_path, stub, _opts(tmp_path, images, prompts="a,b", batch_size=4))

    # Every (image, class) pair present exactly once per query — 4 images x 2 classes.
    # A double-commit from OOM retry would produce 2*Q entries for the same pair.
    from collections import Counter

    pair_counts = Counter((int(p["image_id"]), int(p["category_id"])) for p in got)
    by_pair = set(pair_counts.keys())
    assert len(by_pair) == 4 * 2, f"missing (image, class) pairs: {by_pair}"
    over_committed = [(pair, cnt) for pair, cnt in pair_counts.items() if cnt > Q]
    assert not over_committed, (
        f"double-committed (image, class) pairs after RETRY_B: {over_committed}"
    )
