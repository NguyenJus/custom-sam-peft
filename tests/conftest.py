"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

from esam3.data.coco import COCODataset
from esam3.tracking.noop import NoopTracker
from tests.fixtures.tiny_sam3_stub import TinySam3Stub

FIXTURES = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture
def tiny_coco_dir() -> Path:
    return FIXTURES / "tiny_coco"


@pytest.fixture
def tiny_coco_dataset(tiny_coco_dir: Path) -> COCODataset:
    """A COCODataset pointing at the tiny_coco fixture.

    Methods raise NotImplementedError until spec/data-loading lands; this
    fixture exists so future integration tests have a typed Dataset handle.
    """
    return COCODataset(
        annotations=str(tiny_coco_dir / "annotations.json"),
        images=str(tiny_coco_dir / "images"),
        prompt_mode="bbox",
    )


@pytest.fixture
def tmp_run_dir(tmp_path: Path) -> Path:
    d = tmp_path / "run"
    d.mkdir()
    return d


@pytest.fixture
def stub_model() -> TinySam3Stub:
    return TinySam3Stub()


@pytest.fixture
def noop_tracker() -> NoopTracker:
    return NoopTracker()
