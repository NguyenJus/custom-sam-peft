"""Sechidis 2011 iterative multi-label stratification.

Pure, no IO, no torch. Used by data.val_source to carve a train+val
partition from a list of `SplittableItem`s representing dataset rows.

Spec: docs/superpowers/specs/2026-05-22-data-no-val-auto-split-design.md §4.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class SplittableItem:
    """An item (image, HF row, etc.) eligible for stratification.

    `image_id` is an opaque string id (COCO int_id is str(int_id); HF row index
    is str(row_index)). `class_ids` is the dense (post-remap) class ids present
    in this item.
    """

    image_id: str
    class_ids: frozenset[int]


@dataclass(frozen=True)
class SplitResult:
    train_ids: tuple[str, ...]
    val_ids: tuple[str, ...]
    realized_fraction: float
    per_class_counts: dict[int, tuple[int, int]]
    missing_in_val: tuple[int, ...]


def stratified_split(
    items: Sequence[SplittableItem],
    fraction: float,
    seed: int,
) -> SplitResult:
    """Carve `items` into a train+val partition via Sechidis 2011 iterative
    multi-label stratification.

    Deterministic given `(items, fraction, seed)`: items are sorted by
    `image_id` before processing so caller ordering does not matter.

    See spec §4.2 for the algorithm and §4.3 for edge-case behavior.
    """
    # 1. Sort input.
    items_sorted = sorted(items, key=lambda it: it.image_id)
    n = len(items_sorted)
    if n == 0:
        return SplitResult(
            train_ids=(),
            val_ids=(),
            realized_fraction=0.0,
            per_class_counts={},
            missing_in_val=(),
        )

    # 2. Quotas.
    v_total = round(n * fraction)
    t_total = n - v_total
    class_totals: dict[int, int] = defaultdict(int)
    for it in items_sorted:
        for c in it.class_ids:
            class_totals[c] += 1
    quota_train: dict[int, int] = {c: nc - round(nc * fraction) for c, nc in class_totals.items()}
    quota_val: dict[int, int] = {c: round(nc * fraction) for c, nc in class_totals.items()}
    remaining = {"train": t_total, "val": v_total}

    # 3. Initial ordering: rarest-class items first, RNG tiebreak.
    rng = random.Random(seed)  # noqa: S311 — not cryptographic; seeded stratification

    def _min_class_count(it: SplittableItem) -> float:
        if not it.class_ids:
            return math.inf
        return float(min(class_totals[c] for c in it.class_ids))

    decorated = [(_min_class_count(it), rng.random(), it) for it in items_sorted]
    decorated.sort(key=lambda t: (t[0], t[1]))

    # 4. Greedy placement.
    train_ids: list[str] = []
    val_ids: list[str] = []

    def _score(side: str, it: SplittableItem) -> int:
        quota = quota_train if side == "train" else quota_val
        if not it.class_ids:
            return remaining[side]
        return max(quota[c] for c in it.class_ids)

    for _min_c, _tiebreak, it in decorated:
        if remaining["train"] == 0:
            chosen = "val"
        elif remaining["val"] == 0:
            chosen = "train"
        else:
            s_t = _score("train", it)
            s_v = _score("val", it)
            if s_t > s_v:
                chosen = "train"
            elif s_v > s_t:
                chosen = "val"
            else:
                # Tie on score: prefer side with larger remaining capacity.
                if remaining["train"] > remaining["val"]:
                    chosen = "train"
                elif remaining["val"] > remaining["train"]:
                    chosen = "val"
                else:
                    # Still tied: seeded coin flip.
                    chosen = "train" if rng.random() < 0.5 else "val"
        (train_ids if chosen == "train" else val_ids).append(it.image_id)
        remaining[chosen] -= 1
        quota = quota_train if chosen == "train" else quota_val
        for c in it.class_ids:
            if quota[c] > 0:
                quota[c] -= 1

    # 5. Post-checks.
    realized_fraction = len(val_ids) / max(n, 1)
    per_class_counts: dict[int, tuple[int, int]] = {}
    train_set = set(train_ids)
    val_set = set(val_ids)
    for c, _total in class_totals.items():
        t_count = sum(1 for it in items_sorted if c in it.class_ids and it.image_id in train_set)
        v_count = sum(1 for it in items_sorted if c in it.class_ids and it.image_id in val_set)
        per_class_counts[c] = (t_count, v_count)
    missing_in_val = tuple(
        sorted(c for c, (t, v) in per_class_counts.items() if class_totals[c] >= 2 and v == 0)
    )

    return SplitResult(
        train_ids=tuple(sorted(train_ids)),
        val_ids=tuple(sorted(val_ids)),
        realized_fraction=realized_fraction,
        per_class_counts=per_class_counts,
        missing_in_val=missing_in_val,
    )
