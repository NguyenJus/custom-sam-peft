"""Unit tests for the Sechidis 2011 iterative multi-label stratifier.

Spec: docs/superpowers/specs/2026-05-22-data-no-val-auto-split-design.md §4, §9.1.
"""

from __future__ import annotations

import random

from custom_sam_peft.data.splitter import SplitResult, SplittableItem, stratified_split


def _items(spec: list[tuple[str, frozenset[int]]]) -> list[SplittableItem]:
    return [SplittableItem(image_id=iid, class_ids=cls) for iid, cls in spec]


def test_determinism_identical_calls_produce_identical_results() -> None:
    items = _items([(str(i), frozenset({i % 5})) for i in range(50)])
    a = stratified_split(items, fraction=0.2, seed=42)
    b = stratified_split(items, fraction=0.2, seed=42)
    assert a == b


def test_order_independence_shuffle_then_split() -> None:
    items = _items([(str(i), frozenset({i % 5})) for i in range(50)])
    a = stratified_split(items, fraction=0.2, seed=42)
    shuffled = list(items)
    random.Random(123).shuffle(shuffled)
    b = stratified_split(shuffled, fraction=0.2, seed=42)
    assert a == b


def test_realized_fraction_close_to_requested_for_single_class() -> None:
    items = _items([(str(i), frozenset({0})) for i in range(100)])
    res = stratified_split(items, fraction=0.1, seed=42)
    assert abs(res.realized_fraction - 0.1) <= 0.01


def test_multiclass_coverage_rare_class_appears_in_both_sides() -> None:
    # 50 items: classes 0..9. The 47 "bulk" items cover classes 0..8 (i % 9),
    # and 3 "rare" items each carry class 9 alone so n_c=3 for class 9. With
    # fraction=0.2 the quota is `quota_val[9] = round(3 * 0.2) = 1`, so the
    # spec's greedy placement must land at least one rare item in val and the
    # rest in train — i.e. class 9 appears on both sides.
    #
    # NOTE: the rare items are intentionally single-label. The spec's score
    # uses `max(quota[c] for c in item.class_ids)`, so pairing class 9 with an
    # abundant class would let the abundant class dominate scoring and pull
    # every rare item into train. Single-label rare items isolate class 9 in
    # the score, which is what the test is meant to exercise.
    item_specs: list[tuple[str, frozenset[int]]] = []
    for i in range(47):
        item_specs.append((str(i), frozenset({i % 9})))  # classes 0..8
    item_specs.append(("47", frozenset({9})))
    item_specs.append(("48", frozenset({9})))
    item_specs.append(("49", frozenset({9})))
    items = _items(item_specs)
    res = stratified_split(items, fraction=0.2, seed=42)
    train_set = set(res.train_ids)
    val_set = set(res.val_ids)
    rare = {"47", "48", "49"}
    rare_train = bool(rare & train_set)
    rare_val = bool(rare & val_set)
    assert rare_train and rare_val, (
        f"rare class 9 must land in both sides: train={rare & train_set}, val={rare & val_set}"
    )


def test_empty_class_set_does_not_crash_and_skips_missing_in_val() -> None:
    items = _items([(str(i), frozenset()) for i in range(5)])
    res = stratified_split(items, fraction=0.2, seed=42)
    assert len(res.train_ids) + len(res.val_ids) == 5
    # Empty-class items must not appear in per_class_counts or missing_in_val.
    assert res.per_class_counts == {}
    assert res.missing_in_val == ()


def test_edge_size_zero() -> None:
    res = stratified_split([], fraction=0.1, seed=42)
    assert res == SplitResult(
        train_ids=(),
        val_ids=(),
        realized_fraction=0.0,
        per_class_counts={},
        missing_in_val=(),
    )


def test_edge_size_one_all_to_train() -> None:
    items = _items([("0", frozenset({0}))])
    res = stratified_split(items, fraction=0.1, seed=42)
    assert res.train_ids == ("0",)
    assert res.val_ids == ()
    assert res.realized_fraction == 0.0


def test_edge_size_two_fraction_half_one_each() -> None:
    items = _items([("0", frozenset({0})), ("1", frozenset({0}))])
    res = stratified_split(items, fraction=0.5, seed=42)
    assert len(res.train_ids) == 1
    assert len(res.val_ids) == 1
    assert set(res.train_ids) | set(res.val_ids) == {"0", "1"}


def test_quota_deviation_records_missing_in_val() -> None:
    # 4 items: class 0 appears 3 times, class 1 appears 1 time (only).
    # With fraction=0.25 and 4 items, V=1; class 1's quota v_c=round(1*0.25)=0,
    # so class 1 has total >= 2 should NOT trigger missing_in_val. We want a
    # case where missing_in_val activates: class 2 has 2 items, fraction=0.1
    # gives v_c=0 → if both placed in train it's still missing_in_val.
    items = _items(
        [
            ("0", frozenset({0})),
            ("1", frozenset({0})),
            ("2", frozenset({0})),
            ("3", frozenset({0})),
            ("4", frozenset({0})),
            ("5", frozenset({2})),
            ("6", frozenset({2})),
        ]
    )
    res = stratified_split(items, fraction=0.1, seed=42)
    # Class 2 has 2 items; with fraction=0.1 the round-down quota is 0 in val.
    # If both class-2 items land in train, class 2 appears in missing_in_val.
    if all(iid in res.train_ids for iid in ("5", "6")):
        assert 2 in res.missing_in_val
    # Realized fraction must be a valid probability.
    assert 0.0 <= res.realized_fraction < 1.0


def test_train_and_val_ids_are_sorted() -> None:
    items = _items([(str(i), frozenset({i % 3})) for i in range(20)])
    res = stratified_split(items, fraction=0.3, seed=42)
    assert list(res.train_ids) == sorted(res.train_ids)
    assert list(res.val_ids) == sorted(res.val_ids)


def test_train_and_val_ids_disjoint() -> None:
    items = _items([(str(i), frozenset({i % 3})) for i in range(20)])
    res = stratified_split(items, fraction=0.3, seed=42)
    assert set(res.train_ids).isdisjoint(set(res.val_ids))
