"""Regression test: _build_val_dataset must respect cfg.data.limit.val.

Bug: the bundle's val dataset was built without applying _apply_limit, so when
cfg.data.limit.val=10 was set for a smoke run, len(val_dataset)=355 while
len(per_example_iou)=10, causing pick_samples to raise an alignment invariant.

Fix: route through _build_dataset_from_dict (which calls _apply_limit) instead
of calling the registry builder directly.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from custom_sam_peft.cli.run_cmd import _build_val_dataset
from custom_sam_peft.data.split_source import SplitSource
from custom_sam_peft.data.subset import SubsetDataset
from custom_sam_peft.train.runner import _build_dataset_from_dict


def _make_stub_inner(size: int) -> MagicMock:
    inner = MagicMock()
    inner.__len__ = MagicMock(return_value=size)
    inner.image_class_labels = None
    return inner


def _make_cfg(val_limit: int | None) -> MagicMock:
    cfg = MagicMock()
    cfg.data.format = "coco"
    cfg.data.model_dump.return_value = {"format": "coco"}
    cfg.model.name = "facebook/sam3.1"
    cfg.data.limit.val = val_limit
    cfg.data.limit.train = None
    cfg.data.limit.seed = 42
    cfg.data.limit.strategy = "random"
    return cfg


def _explicit_split_source() -> SplitSource:
    return SplitSource(
        mode="explicit",
        train_ids=None,
        val_ids=None,
        test_ids=None,
        realized_fraction=None,
        per_class_counts=None,
        missing_in_val=None,
        missing_in_test=None,
        val_fraction_requested=None,
        test_fraction_requested=None,
        seed_used=None,
    )


def _auto_split_source(n_val: int) -> SplitSource:
    val_ids = tuple(f"img_{i}" for i in range(n_val))
    train_ids = tuple(f"trn_{i}" for i in range(100))
    return SplitSource(
        mode="auto_split",
        train_ids=train_ids,
        val_ids=val_ids,
        test_ids=None,
        realized_fraction=(n_val / (len(train_ids) + n_val), 0.0),
        per_class_counts={0: (80, n_val, 0)},
        missing_in_val=(),
        missing_in_test=None,
        val_fraction_requested=0.2,
        test_fraction_requested=None,
        seed_used=42,
    )


# ---------------------------------------------------------------------------
# Helpers: both paths use the same lookup stub pointing at runner, where the
# real _apply_limit / SubsetDataset wrapping happens.
# ---------------------------------------------------------------------------


def _make_stub_builder(inner: MagicMock):  # type: ignore[no-untyped-def]
    """Return a builder callable that always returns `inner` (ignores args)."""
    return lambda *a, **kw: inner


# ---------------------------------------------------------------------------
# The regression: without the fix, _build_val_dataset returns the full inner
# dataset (length = full_val_size) instead of the capped subset.
# ---------------------------------------------------------------------------


def test_build_val_dataset_explicit_mode_respects_limit() -> None:
    """_build_val_dataset must return a SubsetDataset of size val_limit.

    Before fix: len(ds) == 355 (full size), assertion fails.
    After fix:  len(ds) == 10 (capped).
    """
    full_val_size = 355
    val_limit = 10
    cfg = _make_cfg(val_limit=val_limit)
    vs = _explicit_split_source()
    inner = _make_stub_inner(full_val_size)

    # Patch the lookup used by runner._build_dataset_from_dict (runner imports lookup).
    with patch("custom_sam_peft.train.runner.lookup", return_value=_make_stub_builder(inner)):
        ds = _build_val_dataset(cfg, vs)

    assert len(ds) == val_limit, (
        f"Expected _build_val_dataset to cap to val_limit={val_limit}, "
        f"but got len={len(ds)} (full_val_size={full_val_size}). "
        f"This is the bundle/per_example_iou length-mismatch bug."
    )
    assert isinstance(ds, SubsetDataset)


def test_build_val_dataset_auto_split_mode_respects_limit() -> None:
    """Same regression in auto_split mode: val_ids injection + limit must both apply."""
    full_val_size = 50
    val_limit = 8
    cfg = _make_cfg(val_limit=val_limit)
    vs = _auto_split_source(n_val=full_val_size)
    inner = _make_stub_inner(full_val_size)

    with patch("custom_sam_peft.train.runner.lookup", return_value=_make_stub_builder(inner)):
        ds = _build_val_dataset(cfg, vs)

    assert len(ds) == val_limit, (
        f"Expected _build_val_dataset to cap to val_limit={val_limit} in auto_split mode, "
        f"but got len={len(ds)}."
    )
    assert isinstance(ds, SubsetDataset)


def test_build_val_dataset_no_limit_returns_full_dataset() -> None:
    """When limit.val is None, the full dataset is returned unchanged."""
    full_val_size = 355
    cfg = _make_cfg(val_limit=None)
    vs = _explicit_split_source()
    inner = _make_stub_inner(full_val_size)

    with patch("custom_sam_peft.train.runner.lookup", return_value=_make_stub_builder(inner)):
        ds = _build_val_dataset(cfg, vs)

    assert len(ds) == full_val_size
    assert not isinstance(ds, SubsetDataset)


def test_build_val_dataset_index_alignment_matches_trainer_eval_path() -> None:
    """Proves index alignment: _build_val_dataset and the trainer's eval path
    produce identical SubsetDataset._indices for the same cfg, so per_example_iou
    indices are always in sync with the bundle's val dataset.
    """
    full_val_size = 30
    val_limit = 7
    cfg = _make_cfg(val_limit=val_limit)
    vs = _explicit_split_source()
    inner = _make_stub_inner(full_val_size)

    data_cfg_dict = cfg.data.model_dump()

    with patch("custom_sam_peft.train.runner.lookup", return_value=_make_stub_builder(inner)):
        bundle_ds = _build_val_dataset(cfg, vs)
        trainer_ds = _build_dataset_from_dict(data_cfg_dict, cfg, "eval")

    assert isinstance(bundle_ds, SubsetDataset)
    assert isinstance(trainer_ds, SubsetDataset)
    assert bundle_ds._indices == trainer_ds._indices, (
        "Index mismatch: bundle val dataset and trainer eval dataset must select "
        "identical indices so per_example_iou aligns with the bundle's dataset."
    )
