"""Unit tests for the Sechidis 2011 iterative multi-label stratifier.

Spec: docs/superpowers/specs/2026-06-04-train-val-test-split-design.md §5, §10.1.
"""

from __future__ import annotations

import random

from custom_sam_peft.data.splitter import SplitResult, SplittableItem, stratified_split


def _items(spec: list[tuple[str, frozenset[int]]]) -> list[SplittableItem]:
    return [SplittableItem(image_id=iid, class_ids=cls) for iid, cls in spec]


# ---------------------------------------------------------------------------
# 3-way tests (§10.1)
# ---------------------------------------------------------------------------


def test_3way_determinism_identical_calls_produce_identical_results() -> None:
    items = _items([(str(i), frozenset({i % 5})) for i in range(50)])
    a = stratified_split(items, val_fraction=0.2, test_fraction=0.2, seed=42)
    b = stratified_split(items, val_fraction=0.2, test_fraction=0.2, seed=42)
    assert a == b
    # Both must include test_ids and missing_in_test
    assert hasattr(a, "test_ids")
    assert hasattr(a, "missing_in_test")


def test_3way_order_independence_shuffle_then_split() -> None:
    items = _items([(str(i), frozenset({i % 5})) for i in range(50)])
    a = stratified_split(items, val_fraction=0.2, test_fraction=0.2, seed=42)
    shuffled = list(items)
    random.Random(123).shuffle(shuffled)
    b = stratified_split(shuffled, val_fraction=0.2, test_fraction=0.2, seed=42)
    assert a == b


def test_3way_disjointness_and_complete_coverage() -> None:
    items = _items([(str(i), frozenset({i % 5})) for i in range(60)])
    res = stratified_split(items, val_fraction=0.2, test_fraction=0.2, seed=42)
    train_set = set(res.train_ids)
    val_set = set(res.val_ids)
    test_set = set(res.test_ids)
    all_ids = {it.image_id for it in items}

    assert train_set & val_set == set(), "train and val must be disjoint"
    assert train_set & test_set == set(), "train and test must be disjoint"
    assert val_set & test_set == set(), "val and test must be disjoint"
    assert train_set | val_set | test_set == all_ids, "union must cover all ids"


def test_3way_realized_fractions_close_to_requested() -> None:
    # 100 single-class items; (0.2, 0.2) → realized within ±0.02 of each target.
    items = _items([(str(i), frozenset({0})) for i in range(100)])
    res = stratified_split(items, val_fraction=0.2, test_fraction=0.2, seed=42)
    val_realized, test_realized = res.realized_fraction
    assert abs(val_realized - 0.2) <= 0.02, f"val realized={val_realized}, expected ≈0.2"
    assert abs(test_realized - 0.2) <= 0.02, f"test realized={test_realized}, expected ≈0.2"


def test_3way_per_class_counts_are_3_tuple_summing_to_total() -> None:
    items = _items([(str(i), frozenset({i % 5})) for i in range(60)])
    res = stratified_split(items, val_fraction=0.2, test_fraction=0.2, seed=42)
    # Each class appears 12 times (60 / 5)
    for cls, (t, v, te) in res.per_class_counts.items():
        total = sum(1 for it in items if cls in it.class_ids)
        assert t + v + te == total, f"class {cls}: train={t} + val={v} + test={te} != total={total}"
        assert t >= 0 and v >= 0 and te >= 0


def test_3way_rare_class_missing_in_buckets() -> None:
    """Rare class n_c==2 over 3 buckets → recorded in the absent bucket's missing_in_*."""
    # Build: 30 bulk items with class 0, 2 rare items with class 1 only.
    item_specs: list[tuple[str, frozenset[int]]] = []
    for i in range(30):
        item_specs.append((str(i), frozenset({0})))
    item_specs.append(("30", frozenset({1})))
    item_specs.append(("31", frozenset({1})))
    items = _items(item_specs)

    # With 3 buckets and n_c=2 for class 1, at most 2 buckets can hold class 1.
    res = stratified_split(items, val_fraction=0.2, test_fraction=0.2, seed=42)

    train_count, val_count, test_count = res.per_class_counts[1]
    assert train_count + val_count + test_count == 2

    # Counts must be non-negative and plausible
    missing_val = 1 in res.missing_in_val
    missing_test = 1 in res.missing_in_test

    if val_count == 0:
        assert missing_val, "class 1 (n_c=2) must be in missing_in_val when val_count==0"
    else:
        assert not missing_val, "class 1 should not be in missing_in_val when val_count>0"

    if test_count == 0:
        assert missing_test, "class 1 (n_c=2) must be in missing_in_test when test_count==0"
    else:
        assert not missing_test, "class 1 should not be in missing_in_test when test_count>0"


# ---------------------------------------------------------------------------
# 2-way fallback regression lock (§10.1, §5.2)
# ---------------------------------------------------------------------------

# Pinned output from the PRIOR 2-arg stratified_split(items, fraction=0.2, seed=7)
# captured before this PR. Items: [(str(i), frozenset({i%5})) for i in range(50)].
_FALLBACK_FIXTURE_ITEMS = _items([(str(i), frozenset({i % 5})) for i in range(50)])
_FALLBACK_SEED = 7
_FALLBACK_VAL_FRACTION = 0.2

_PINNED_TRAIN_IDS: tuple[str, ...] = (
    "0",
    "1",
    "10",
    "11",
    "12",
    "13",
    "14",
    "15",
    "16",
    "17",
    "18",
    "19",
    "2",
    "21",
    "22",
    "26",
    "27",
    "28",
    "29",
    "3",
    "30",
    "31",
    "32",
    "33",
    "34",
    "35",
    "37",
    "38",
    "39",
    "4",
    "40",
    "41",
    "42",
    "43",
    "45",
    "46",
    "48",
    "49",
    "5",
    "9",
)
_PINNED_VAL_IDS: tuple[str, ...] = (
    "20",
    "23",
    "24",
    "25",
    "36",
    "44",
    "47",
    "6",
    "7",
    "8",
)


def test_2way_fallback_test_fraction_zero_matches_pinned_fixture() -> None:
    """stratified_split(items, v, 0.0, seed) must reproduce the prior 2-arg result bit-for-bit."""
    res = stratified_split(
        _FALLBACK_FIXTURE_ITEMS,
        val_fraction=_FALLBACK_VAL_FRACTION,
        test_fraction=0.0,
        seed=_FALLBACK_SEED,
    )
    assert res.train_ids == _PINNED_TRAIN_IDS, (
        f"train_ids differ from pinned fixture:\n  got: {res.train_ids}"
    )
    assert res.val_ids == _PINNED_VAL_IDS, (
        f"val_ids differ from pinned fixture:\n  got: {res.val_ids}"
    )
    assert res.test_ids == (), f"expected test_ids==(), got {res.test_ids}"
    assert res.missing_in_test == (), f"expected missing_in_test==(), got {res.missing_in_test}"
    assert res.realized_fraction[1] == 0.0, (
        f"expected realized_fraction[1]==0.0, got {res.realized_fraction[1]}"
    )


# ---------------------------------------------------------------------------
# Edge-size tests (§10.1, §5.3)
# ---------------------------------------------------------------------------


def test_edge_size_zero() -> None:
    res = stratified_split([], val_fraction=0.2, test_fraction=0.2, seed=42)
    assert res == SplitResult(
        train_ids=(),
        val_ids=(),
        test_ids=(),
        realized_fraction=(0.0, 0.0),
        per_class_counts={},
        missing_in_val=(),
        missing_in_test=(),
    )


def test_edge_size_one_all_to_train() -> None:
    items = _items([("0", frozenset({0}))])
    res = stratified_split(items, val_fraction=0.2, test_fraction=0.2, seed=42)
    assert res.train_ids == ("0",)
    assert res.val_ids == ()
    assert res.test_ids == ()
    assert res.realized_fraction == (0.0, 0.0)


def test_edge_size_three_one_each_bucket() -> None:
    """N=3, (1/3, 1/3) → 1 item per bucket (each round(3 * 1/3) = 1)."""
    items = _items([("0", frozenset({0})), ("1", frozenset({0})), ("2", frozenset({0}))])
    res = stratified_split(items, val_fraction=1 / 3, test_fraction=1 / 3, seed=42)
    assert len(res.train_ids) == 1
    assert len(res.val_ids) == 1
    assert len(res.test_ids) == 1
    assert set(res.train_ids) | set(res.val_ids) | set(res.test_ids) == {"0", "1", "2"}


# ---------------------------------------------------------------------------
# Legacy 2-way tests (preserve existing coverage; adapted to new signature)
# ---------------------------------------------------------------------------


def test_determinism_identical_calls_produce_identical_results() -> None:
    items = _items([(str(i), frozenset({i % 5})) for i in range(50)])
    a = stratified_split(items, val_fraction=0.2, test_fraction=0.0, seed=42)
    b = stratified_split(items, val_fraction=0.2, test_fraction=0.0, seed=42)
    assert a == b


def test_order_independence_shuffle_then_split() -> None:
    items = _items([(str(i), frozenset({i % 5})) for i in range(50)])
    a = stratified_split(items, val_fraction=0.2, test_fraction=0.0, seed=42)
    shuffled = list(items)
    random.Random(123).shuffle(shuffled)
    b = stratified_split(shuffled, val_fraction=0.2, test_fraction=0.0, seed=42)
    assert a == b


def test_realized_fraction_close_to_requested_for_single_class() -> None:
    items = _items([(str(i), frozenset({0})) for i in range(100)])
    res = stratified_split(items, val_fraction=0.1, test_fraction=0.0, seed=42)
    val_realized, _test_realized = res.realized_fraction
    assert abs(val_realized - 0.1) <= 0.01


def test_multiclass_coverage_rare_class_appears_in_both_sides() -> None:
    item_specs: list[tuple[str, frozenset[int]]] = []
    for i in range(47):
        item_specs.append((str(i), frozenset({i % 9})))
    item_specs.append(("47", frozenset({9})))
    item_specs.append(("48", frozenset({9})))
    item_specs.append(("49", frozenset({9})))
    items = _items(item_specs)
    res = stratified_split(items, val_fraction=0.2, test_fraction=0.0, seed=42)
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
    res = stratified_split(items, val_fraction=0.2, test_fraction=0.0, seed=42)
    assert len(res.train_ids) + len(res.val_ids) == 5
    assert res.per_class_counts == {}
    assert res.missing_in_val == ()


def test_edge_size_two_fraction_half_one_each() -> None:
    items = _items([("0", frozenset({0})), ("1", frozenset({0}))])
    res = stratified_split(items, val_fraction=0.5, test_fraction=0.0, seed=42)
    assert len(res.train_ids) == 1
    assert len(res.val_ids) == 1
    assert set(res.train_ids) | set(res.val_ids) == {"0", "1"}


def test_quota_deviation_records_missing_in_val() -> None:
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
    res = stratified_split(items, val_fraction=0.1, test_fraction=0.0, seed=42)
    if all(iid in res.train_ids for iid in ("5", "6")):
        assert 2 in res.missing_in_val
    val_realized, _test = res.realized_fraction
    assert 0.0 <= val_realized < 1.0


def test_train_and_val_ids_are_sorted() -> None:
    items = _items([(str(i), frozenset({i % 3})) for i in range(20)])
    res = stratified_split(items, val_fraction=0.3, test_fraction=0.0, seed=42)
    assert list(res.train_ids) == sorted(res.train_ids)
    assert list(res.val_ids) == sorted(res.val_ids)


def test_train_and_val_ids_disjoint() -> None:
    items = _items([(str(i), frozenset({i % 3})) for i in range(20)])
    res = stratified_split(items, val_fraction=0.3, test_fraction=0.0, seed=42)
    assert set(res.train_ids).isdisjoint(set(res.val_ids))
