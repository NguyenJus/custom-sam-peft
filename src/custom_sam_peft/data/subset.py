"""Dataset subsetting — pure sampling function + transparent wrapper.

Public API:
  resolve_subset_indices(n_total, limit, *, seed, strategy, image_class_labels)
  SubsetDataset(inner, indices)
"""

from __future__ import annotations

import logging
import random as _random
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from custom_sam_peft.data.base import Dataset

_LOG = logging.getLogger(__name__)


def resolve_subset_indices(
    n_total: int,
    limit: int | float,
    *,
    seed: int,
    strategy: str,
    image_class_labels: Sequence[Sequence[int]] | None,
) -> list[int]:
    """Return sorted-ascending unique indices in [0, n_total).

    Cap resolution:
      int   → min(limit, n_total); warns if limit > n_total.
      float → max(1, round(limit * n_total)); 1.0 yields full range.
    """
    if isinstance(limit, bool):
        raise TypeError("limit must not be a bool")

    # Resolve cap
    if isinstance(limit, int):
        cap = min(limit, n_total)
        if limit > n_total:
            _LOG.warning(
                "limit=%d exceeds dataset size %d; using full dataset",
                limit,
                n_total,
            )
    else:
        cap = max(1, round(limit * n_total))

    if cap >= n_total:
        return list(range(n_total))

    if strategy == "first_n":
        return list(range(cap))

    if strategy == "random":
        return _random_indices(n_total, cap, seed=seed)

    if strategy == "stratified":
        if image_class_labels is None:
            _LOG.warning(
                "stratified subset requested but image_class_labels is None; falling back to random"
            )
            return _random_indices(n_total, cap, seed=seed)
        return _stratified_indices(n_total, cap, seed=seed, labels=image_class_labels)

    raise ValueError(f"unknown strategy: {strategy!r}")


def _random_indices(n_total: int, cap: int, *, seed: int) -> list[int]:
    rng = _random.Random(f"{seed}:{n_total}:random")  # noqa: S311
    pool = list(range(n_total))
    rng.shuffle(pool)
    return sorted(pool[:cap])


def _stratified_indices(
    n_total: int,
    cap: int,
    *,
    seed: int,
    labels: Sequence[Sequence[int]],
) -> list[int]:
    """Multi-label proportional sampling (Sechidis et al. 2011 iterative re-weighting).

    Algorithm:
      1. Collect all unique class ids.
      2. Compute desired per-class count: quota[c] = round(cap * freq[c] / n_total).
      3. Greedy: at each step, find the image (not yet selected) whose rarest
         still-needed class has the highest remaining deficit (quota[c] - selected[c]).
         Tie-break by class with smallest current quota, then by image index.
      4. After the greedy pass, if len(selected) < cap, fill from the remaining
         pool using a seeded random draw.
    """
    # Build class → image index mapping
    all_classes: set[int] = set()
    for row in labels:
        all_classes.update(row)
    if not all_classes:
        return _random_indices(n_total, cap, seed=seed)

    class_list = sorted(all_classes)
    class_to_idx = {c: i for i, c in enumerate(class_list)}
    n_classes = len(class_list)

    # Per-class frequency in the full dataset
    freq = [0] * n_classes
    for row in labels:
        for c in row:
            freq[class_to_idx[c]] += 1

    # Desired quota per class
    quota = [max(1, round(cap * freq[i] / n_total)) for i in range(n_classes)]

    selected: list[int] = []
    selected_set: set[int] = set()
    selected_per_class = [0] * n_classes

    remaining = list(range(n_total))

    for _ in range(min(cap, n_total)):
        if not remaining:
            break

        best_img = -1
        best_key: tuple[int, int, int] = (0, 0, 0)  # (deficit, -quota, -img_idx) — max by deficit

        for img_idx in remaining:
            img_classes = [class_to_idx[c] for c in labels[img_idx] if c in class_to_idx]
            if not img_classes:
                img_classes = []

            # Find the rarest still-needed class for this image
            deficits = [
                (quota[c] - selected_per_class[c], -quota[c], -img_idx)
                for c in img_classes
                if quota[c] - selected_per_class[c] > 0
            ]
            key = (0, 0, -img_idx) if not deficits else max(deficits)

            if best_img == -1 or key > best_key:
                best_key = key
                best_img = img_idx

        if best_img == -1:
            break

        selected.append(best_img)
        selected_set.add(best_img)
        for c in labels[best_img]:
            if c in class_to_idx:
                selected_per_class[class_to_idx[c]] += 1
        remaining.remove(best_img)

    # Fill shortfall with random draw from remaining
    if len(selected) < cap and remaining:
        rng = _random.Random(f"{seed}:{n_total}:stratified_fill")  # noqa: S311
        fill_pool = list(remaining)
        rng.shuffle(fill_pool)
        needed = cap - len(selected)
        selected.extend(fill_pool[:needed])

    return sorted(selected)


class SubsetDataset:
    """Transparent index-mapping wrapper that satisfies the Dataset Protocol.

    The inner dataset never sees the subset — all indexing is at this layer.
    """

    def __init__(self, inner: Dataset, indices: list[int]) -> None:
        self._inner = inner
        self._indices = indices

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, i: int) -> Any:
        return self._inner[self._indices[i]]

    @property
    def class_names(self) -> list[str]:
        return self._inner.class_names
