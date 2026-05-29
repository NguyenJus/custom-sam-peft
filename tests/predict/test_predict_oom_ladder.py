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


# ---------------------------------------------------------------------------
# New tests: close review gaps #1, #2, #3
# ---------------------------------------------------------------------------


def test_predict_oom_retry_b_discards_non_empty_buffer(tmp_path: Path, monkeypatch) -> None:
    """RETRY_B with a non-empty chunk_buf (review gap #1).

    Sequence:
      MULTIPLEX_CAP=1 → each class is its own group (j advances 1 per iteration).
      prompts="a,b,c"  → category_ids 1, 2, 3.
      batch_size=4     → B=4 on entry; OOM fires at (B=4, class="b").

      j=0 (class "a") succeeds → 4 rows (category_id=1) land in chunk_buf.
      j=1 (class "b") OOMs    → RETRY_B (B 4→2); chunk_buf DISCARDED; restart_chunk=True.
      Outer while restarts at i=0, B=2 → two chunks of 2 images, all 3 classes each.

    Asserts:
      - No (image_id, category_id) pair appears more than Q times (a failed discard
        would re-emit class-a rows at B=4 then again at B=2 → 2*Q duplicates).
      - All 4 images x 3 classes present (no missing pair).
      - category_ids {1, 2, 3} all represented.
    """
    import unittest.mock as mock
    from collections import Counter

    images = _make_image_dir(tmp_path, n=4)
    opts = _opts(tmp_path, images, prompts="a,b,c", batch_size=4)

    # Patch MULTIPLEX_CAP=1 so each class is its own forward pass (K_g=1 always).
    monkeypatch.setattr("custom_sam_peft.models.sam3.MULTIPLEX_CAP", 1, raising=False)

    # OOM fires exactly once: when B=4 and the class group is "b".
    stub = _MultiplexStub(
        oom_predicate=lambda imgs, pr: imgs.shape[0] == 4 and pr[0].classes == ["b"]
    )

    with mock.patch("custom_sam_peft.models.sam3.load_sam31", side_effect=lambda cfg, **kw: stub):
        run_predict(opts)

    got = __import__("json").loads((opts.output / "predictions.json").read_text())

    pair_counts = Counter((int(p["image_id"]), int(p["category_id"])) for p in got)

    # No pair should exceed Q occurrences (a failed buffer discard would give 2*Q).
    over_committed = [(pair, cnt) for pair, cnt in pair_counts.items() if cnt > Q]
    assert not over_committed, (
        f"RETRY_B did not discard chunk_buf: double-committed pairs {over_committed}"
    )

    # All 4 images x 3 classes must be present.
    assert len(pair_counts) == 4 * 3, (
        f"missing (image, class) pairs; got {len(pair_counts)}, expected {4 * 3}: {pair_counts}"
    )

    # All three category_ids present.
    category_ids = {int(p["category_id"]) for p in got}
    assert category_ids == {1, 2, 3}, f"unexpected category_ids: {category_ids}"


def test_predict_oom_retry_k_multi_group_j_arithmetic(tmp_path: Path, monkeypatch) -> None:
    """RETRY_K with j>0 class groups — exercises the j-advance arithmetic (review gap #2).

    Sequence:
      MULTIPLEX_CAP=4 → all 4 classes start in ONE group (K_g=4, j=0).
      batch_size=1    → B=1 (already at floor); first OOM → RETRY_K (K 4→2).
      OOM fires when len(classes) > 2 (fires at K_g=4, silent at K_g=2).

      After RETRY_K:  j=0, K_g=2 → group ["a","b"] → category_ids 1,2; j advances to 2.
                      j=2, K_g=2 → group ["c","d"] → category_ids 3,4; j advances to 4.

    Asserts:
      - All 4 category_ids {1,2,3,4} present.
      - No (image_id, category_id) pair appears more than Q times.
      - Exactly 1 image x 4 classes pairs present (no drop, no dup).
    """
    import unittest.mock as mock
    from collections import Counter

    images = _make_image_dir(tmp_path, n=1)
    opts = _opts(tmp_path, images, prompts="a,b,c,d", batch_size=1)

    # Patch MULTIPLEX_CAP=4 so all 4 classes start in one group.
    monkeypatch.setattr("custom_sam_peft.models.sam3.MULTIPLEX_CAP", 4, raising=False)

    # OOM fires exactly once: when the class group has more than 2 classes (K_g=4).
    stub = _MultiplexStub(oom_predicate=lambda imgs, pr: len(pr[0].classes) > 2)

    with mock.patch("custom_sam_peft.models.sam3.load_sam31", side_effect=lambda cfg, **kw: stub):
        run_predict(opts)

    got = __import__("json").loads((opts.output / "predictions.json").read_text())

    pair_counts = Counter((int(p["image_id"]), int(p["category_id"])) for p in got)

    # No pair should exceed Q occurrences.
    over_committed = [(pair, cnt) for pair, cnt in pair_counts.items() if cnt > Q]
    assert not over_committed, f"double-committed pairs after RETRY_K: {over_committed}"

    # All 1 image x 4 classes must be present.
    assert len(pair_counts) == 1 * 4, (
        f"missing (image, class) pairs; got {len(pair_counts)}, expected 4: {pair_counts}"
    )

    # All four category_ids present — proves j=2 group's (j+kk)+1 arithmetic is correct.
    category_ids = {int(p["category_id"]) for p in got}
    assert category_ids == {1, 2, 3, 4}, f"unexpected category_ids: {category_ids}"


def test_predict_oom_content_identical_same_image_dir(tmp_path: Path) -> None:
    """OOM run produces content-identical predictions to a non-OOM run (review gap #3).

    Strengthens test_predict_oom_recovers_byte_identical_to_non_oom by using the
    SAME image directory (so image_ids match) and comparing the full sorted list of
    (image_id, category_id, score) tuples — not just row counts.
    """
    import unittest.mock as mock

    # Shared image directory → both runs compute identical image_id hashes.
    images = _make_image_dir(tmp_path / "images", n=3)

    ref_opts = _opts(tmp_path / "ref", images, prompts="a,b,c,d,e", batch_size=2)
    oom_opts = _opts(tmp_path / "oom", images, prompts="a,b,c,d,e", batch_size=2)

    ref_stub = _MultiplexStub()
    oom_stub = _MultiplexStub(oom_predicate=lambda imgs, pr: len(pr[0].classes) >= 1)

    with mock.patch(
        "custom_sam_peft.models.sam3.load_sam31", side_effect=lambda cfg, **kw: ref_stub
    ):
        run_predict(ref_opts)

    with mock.patch(
        "custom_sam_peft.models.sam3.load_sam31", side_effect=lambda cfg, **kw: oom_stub
    ):
        run_predict(oom_opts)

    ref = __import__("json").loads((ref_opts.output / "predictions.json").read_text())
    got = __import__("json").loads((oom_opts.output / "predictions.json").read_text())

    def _key(p: dict) -> tuple:
        return (int(p["image_id"]), int(p["category_id"]), round(float(p["score"]), 6))

    ref_sorted = sorted(ref, key=_key)
    got_sorted = sorted(got, key=_key)

    assert len(got_sorted) == len(ref_sorted), (
        f"row count mismatch: OOM run has {len(got_sorted)}, ref has {len(ref_sorted)}"
    )
    first_diff = next((g for g, r in zip(got_sorted, ref_sorted, strict=True) if g != r), None)
    assert got_sorted == ref_sorted, (
        f"OOM run predictions differ from reference in content (first mismatch: {first_diff})"
    )
