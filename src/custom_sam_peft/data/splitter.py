"""Sechidis 2011 joint multi-label stratification into train / val / test.

Pure, no IO, no torch. Used by data.split_source to carve a train/val/test
partition from a list of `SplittableItem`s representing dataset rows.

Spec: docs/superpowers/specs/2026-06-04-train-val-test-split-design.md §5.
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
    test_ids: tuple[str, ...]  # empty tuple when test_fraction==0
    realized_fraction: tuple[float, float]  # (val, test) realized fractions
    per_class_counts: dict[int, tuple[int, int, int]]  # (train, val, test)
    missing_in_val: tuple[int, ...]
    missing_in_test: tuple[int, ...]  # empty when test bucket absent


def stratified_split(
    items: Sequence[SplittableItem],
    val_fraction: float,  # 0.0 ⇒ val bucket yields empty ids (always structurally present)
    test_fraction: float,  # 0.0 ⇒ test bucket absent
    seed: int,
) -> SplitResult:
    """Sechidis 2011 joint multi-label stratification into train/val/test.

    Deterministic given (items, val_fraction, test_fraction, seed): items are
    sorted by image_id before processing so caller ordering does not matter.
    The (val_fraction=v, test_fraction=0) case reproduces the prior 2-way
    result bit-for-bit.

    Spec: docs/superpowers/specs/2026-06-04-train-val-test-split-design.md §5.
    """
    # Step 1. Sort input.
    items_sorted = sorted(items, key=lambda it: it.image_id)
    n = len(items_sorted)
    if n == 0:
        return SplitResult(
            train_ids=(),
            val_ids=(),
            test_ids=(),
            realized_fraction=(0.0, 0.0),
            per_class_counts={},
            missing_in_val=(),
            missing_in_test=(),
        )

    # Step 2. Present buckets.
    # Val is always structurally in the bucket list so the 2-way (v, t=0) path
    # is identical to the prior implementation (same scoring, same RNG draws).
    # "test" is appended only when test_fraction > 0.
    buckets: list[str] = ["train", "val"]
    test_present = test_fraction > 0.0
    if test_present:
        buckets.append("test")

    fractions: dict[str, float] = {
        "val": val_fraction,
        "test": test_fraction if test_present else 0.0,
    }
    fractions["train"] = 1.0 - val_fraction - fractions["test"]

    # Step 3. Per-bucket quotas.
    # remaining[b] = total item slots for bucket b
    n_val = round(n * val_fraction)
    n_test = round(n * test_fraction) if test_present else 0
    n_train = n - n_val - n_test
    remaining: dict[str, int] = {"train": n_train, "val": n_val, "test": n_test}

    # class_totals[c] = total count of items that contain class c
    class_totals: dict[int, int] = defaultdict(int)
    for it in items_sorted:
        for c in it.class_ids:
            class_totals[c] += 1

    # Per-class quotas for val and test; train gets the remainder.
    quota_val: dict[int, int] = {c: round(nc * val_fraction) for c, nc in class_totals.items()}
    quota_test: dict[int, int] = (
        {c: round(nc * test_fraction) for c, nc in class_totals.items()}
        if test_present
        else {c: 0 for c in class_totals}
    )
    quota_train: dict[int, int] = {
        c: nc - quota_val[c] - quota_test[c] for c, nc in class_totals.items()
    }

    # Map bucket name → quota dict for convenience in the placement loop.
    quota_map: dict[str, dict[int, int]] = {
        "train": quota_train,
        "val": quota_val,
        "test": quota_test,
    }

    # Step 4. Initial ordering: (min_class_count_in_item, rng.random()) ascending.
    # Empty-class items use math.inf as the min count.
    rng = random.Random(seed)  # noqa: S311 — not cryptographic; seeded stratification

    def _min_class_count(it: SplittableItem) -> float:
        if not it.class_ids:
            return math.inf
        return float(min(class_totals[c] for c in it.class_ids))

    decorated = [(_min_class_count(it), rng.random(), it) for it in items_sorted]
    decorated.sort(key=lambda t: (t[0], t[1]))

    # Step 5. Greedy placement.
    placements: dict[str, list[str]] = {b: [] for b in buckets}

    def _score(bucket: str, it: SplittableItem) -> int:
        q = quota_map[bucket]
        if not it.class_ids:
            return remaining[bucket]
        return max(q[c] for c in it.class_ids)

    for _min_c, _tiebreak, it in decorated:
        # Restrict to present buckets with capacity.
        candidates = [b for b in buckets if remaining[b] > 0]
        if not candidates:
            # No bucket has capacity (shouldn't happen when quotas reconcile).
            # Fallback: place in train.
            candidates = ["train"]

        if len(candidates) == 1:
            chosen = candidates[0]
        else:
            scores = {b: _score(b, it) for b in candidates}
            max_score = max(scores.values())
            top = [b for b in candidates if scores[b] == max_score]
            if len(top) == 1:
                chosen = top[0]
            else:
                # Tiebreak: prefer bucket with largest remaining capacity.
                max_remaining = max(remaining[b] for b in top)
                top2 = [b for b in top if remaining[b] == max_remaining]
                if len(top2) == 1:
                    chosen = top2[0]
                else:
                    # Still tied: seeded RNG pick among tied buckets.
                    # Sort for deterministic ordering; use rng.random() (not
                    # rng.choice, which uses getrandbits) so the 2-way fallback
                    # (top2 == ["train","val"]) reproduces the old behavior
                    # bit-for-bit — int(random()*2) is equivalent to random()<0.5.
                    top2_sorted = sorted(top2)
                    chosen = top2_sorted[int(rng.random() * len(top2_sorted))]

        placements[chosen].append(it.image_id)
        remaining[chosen] -= 1
        q = quota_map[chosen]
        for c in it.class_ids:
            if q.get(c, 0) > 0:
                q[c] -= 1

    # Step 6. Post-checks.
    train_ids_list = placements["train"]
    val_ids_list = placements["val"]
    test_ids_list = placements.get("test", [])

    val_realized = len(val_ids_list) / max(n, 1)
    test_realized = len(test_ids_list) / max(n, 1)

    train_set = set(train_ids_list)
    val_set = set(val_ids_list)
    test_set = set(test_ids_list)

    per_class_counts: dict[int, tuple[int, int, int]] = {}
    for c in class_totals:
        t_count = sum(1 for it in items_sorted if c in it.class_ids and it.image_id in train_set)
        v_count = sum(1 for it in items_sorted if c in it.class_ids and it.image_id in val_set)
        te_count = sum(1 for it in items_sorted if c in it.class_ids and it.image_id in test_set)
        per_class_counts[c] = (t_count, v_count, te_count)

    missing_in_val = tuple(
        sorted(c for c, (t, v, te) in per_class_counts.items() if class_totals[c] >= 2 and v == 0)
    )
    # missing_in_test: only when the test bucket is present (test_fraction > 0).
    if test_present:
        missing_in_test: tuple[int, ...] = tuple(
            sorted(
                c for c, (t, v, te) in per_class_counts.items() if class_totals[c] >= 2 and te == 0
            )
        )
    else:
        missing_in_test = ()

    return SplitResult(
        train_ids=tuple(sorted(train_ids_list)),
        val_ids=tuple(sorted(val_ids_list)),
        test_ids=tuple(sorted(test_ids_list)),
        realized_fraction=(val_realized, test_realized),
        per_class_counts=per_class_counts,
        missing_in_val=missing_in_val,
        missing_in_test=missing_in_test,
    )
