"""Tests for data/subset.py — schema validation, resolve_subset_indices, SubsetDataset."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from custom_sam_peft.config.schema import LimitConfig

# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "field,value",
    [
        ("train", None),
        ("val", None),
        ("train", 1),
        ("train", 64),
        ("val", 100),
        ("train", 0.5),
        ("train", 1.0),
        ("val", 0.01),
    ],
)
def test_limit_config_valid(field: str, value: object) -> None:
    cfg = LimitConfig(**{field: value})
    assert getattr(cfg, field) == value


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("train", True, "bool"),
        ("train", False, "bool"),
        ("val", True, "bool"),
        ("train", 0, "int"),
        ("train", -1, "int"),
        ("val", 0, "int"),
        ("train", 0.0, "float"),
        ("train", -0.1, "float"),
        ("train", 1.1, "float"),
        ("val", 1.5, "float"),
    ],
)
def test_limit_config_invalid(field: str, value: object, match: str) -> None:
    with pytest.raises(ValidationError):
        LimitConfig(**{field: value})


def test_limit_config_defaults() -> None:
    cfg = LimitConfig()
    assert cfg.train is None
    assert cfg.val is None
    assert cfg.seed == 42
    assert cfg.strategy == "random"


def test_limit_config_strategy_valid() -> None:
    for s in ("random", "stratified", "first_n"):
        cfg = LimitConfig(strategy=s)  # type: ignore[arg-type]
        assert cfg.strategy == s


# ---------------------------------------------------------------------------
# resolve_subset_indices
# ---------------------------------------------------------------------------

import logging

from custom_sam_peft.data.subset import SubsetDataset, resolve_subset_indices


def test_first_n_ascending_range() -> None:
    idx = resolve_subset_indices(10, 4, seed=0, strategy="first_n", image_class_labels=None)
    assert idx == [0, 1, 2, 3]


def test_first_n_clips_to_n_total() -> None:
    idx = resolve_subset_indices(5, 10, seed=0, strategy="first_n", image_class_labels=None)
    assert idx == [0, 1, 2, 3, 4]


def test_first_n_ignores_seed_and_labels() -> None:
    a = resolve_subset_indices(10, 3, seed=0, strategy="first_n", image_class_labels=[[0], [1]])
    b = resolve_subset_indices(10, 3, seed=99, strategy="first_n", image_class_labels=None)
    assert a == b == [0, 1, 2]


def test_random_correct_count_sorted_unique() -> None:
    idx = resolve_subset_indices(100, 20, seed=42, strategy="random", image_class_labels=None)
    assert len(idx) == 20
    assert idx == sorted(idx)
    assert len(set(idx)) == 20
    assert all(0 <= i < 100 for i in idx)


def test_random_deterministic_same_seed_n_total() -> None:
    a = resolve_subset_indices(100, 20, seed=42, strategy="random", image_class_labels=None)
    b = resolve_subset_indices(100, 20, seed=42, strategy="random", image_class_labels=None)
    assert a == b


def test_random_different_n_total_gives_different_subset() -> None:
    a = resolve_subset_indices(100, 20, seed=42, strategy="random", image_class_labels=None)
    b = resolve_subset_indices(101, 20, seed=42, strategy="random", image_class_labels=None)
    assert a != b


def test_random_float_limit_rounds() -> None:
    # 0.25 * 20 = 5
    idx = resolve_subset_indices(20, 0.25, seed=0, strategy="random", image_class_labels=None)
    assert len(idx) == 5


def test_random_float_1_0_returns_all() -> None:
    idx = resolve_subset_indices(10, 1.0, seed=0, strategy="random", image_class_labels=None)
    assert idx == list(range(10))


def test_cap_exceeds_n_total_warns_and_returns_all(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger="custom_sam_peft.data.subset"):
        idx = resolve_subset_indices(5, 100, seed=0, strategy="random", image_class_labels=None)
    assert idx == [0, 1, 2, 3, 4]
    assert any("exceeds" in r.message for r in caplog.records)


def test_stratified_correct_count() -> None:
    # 20 images, 4 classes: indices 0-4 class {0}, 5-9 class {1}, 10-14 class {2}, 15-19 class {3}
    labels = [frozenset([i // 5]) for i in range(20)]
    idx = resolve_subset_indices(20, 8, seed=0, strategy="stratified", image_class_labels=labels)
    assert len(idx) == 8
    assert idx == sorted(idx)
    assert len(set(idx)) == 8


def test_stratified_preserves_all_classes() -> None:
    labels = [frozenset([i // 5]) for i in range(20)]
    idx = resolve_subset_indices(20, 8, seed=0, strategy="stratified", image_class_labels=labels)
    classes_present = set()
    for i in idx:
        classes_present.update(labels[i])
    assert classes_present == {0, 1, 2, 3}


def test_stratified_none_labels_falls_back_to_random(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger="custom_sam_peft.data.subset"):
        idx = resolve_subset_indices(10, 4, seed=0, strategy="stratified", image_class_labels=None)
    assert len(idx) == 4
    assert idx == sorted(idx)
    assert any("stratified" in r.message.lower() for r in caplog.records)


# ---------------------------------------------------------------------------
# SubsetDataset
# ---------------------------------------------------------------------------


class _StubDataset:
    class_names: list[str] = ["a", "b"]  # noqa: RUF012

    def __init__(self, size: int = 10) -> None:
        self._size = size

    def __len__(self) -> int:
        return self._size

    def __getitem__(self, i: int) -> int:  # returns int for simplicity
        return i * 10


def test_subset_dataset_len() -> None:
    inner = _StubDataset(10)
    ds = SubsetDataset(inner, [0, 2, 4])  # type: ignore[arg-type]
    assert len(ds) == 3


def test_subset_dataset_getitem_delegates() -> None:
    inner = _StubDataset(10)
    ds = SubsetDataset(inner, [2, 5, 7])  # type: ignore[arg-type]
    assert ds[0] == 20  # inner[2]
    assert ds[1] == 50  # inner[5]
    assert ds[2] == 70  # inner[7]


def test_subset_dataset_class_names_delegates() -> None:
    inner = _StubDataset()
    ds = SubsetDataset(inner, [0, 1])  # type: ignore[arg-type]
    assert ds.class_names == ["a", "b"]


def test_subset_dataset_satisfies_protocol() -> None:
    import torch

    from custom_sam_peft.data.base import Example, TextPrompts, is_dataset

    class _ExDataset:
        class_names: list[str] = ["x"]  # noqa: RUF012

        def __len__(self) -> int:
            return 2

        def __getitem__(self, i: int) -> Example:
            return Example(
                image=torch.zeros(3, 8, 8),
                image_id=str(i),
                prompts=TextPrompts(classes=["x"]),
                instances=[],
            )

    ds = SubsetDataset(_ExDataset(), [0])  # type: ignore[arg-type]
    assert is_dataset(ds)
