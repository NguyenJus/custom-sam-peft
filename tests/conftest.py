"""Project-level pytest hooks (markers, autoskips) and shared fixtures."""

from __future__ import annotations

import pathlib
from pathlib import Path

import pytest
import torch

from esam3.data.coco import COCODataset
from esam3.tracking.noop import NoopTracker
from tests.fixtures.tiny_sam3_stub import TinySam3Stub

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "requires_checkpoint: skip unless models/sam3.1/sam3.1_multiplex.pt exists",
    )
    config.addinivalue_line(
        "markers",
        "requires_compatible_gpu: skip unless a CUDA device with compute capability "
        ">= 7.5 is available",
    )


def _has_compatible_gpu() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        major, minor = torch.cuda.get_device_capability()
    except RuntimeError:
        return False
    return (major, minor) >= (7, 5)


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    ckpt = pathlib.Path("models/sam3.1/sam3.1_multiplex.pt")
    skip_no_ckpt = pytest.mark.skip(reason="real SAM 3.1 checkpoint not present locally")
    skip_no_gpu = pytest.mark.skip(reason="real SAM 3.1 forward requires a CUDA GPU with CC >= 7.5")
    have_gpu = _has_compatible_gpu()
    for item in items:
        if "requires_checkpoint" in item.keywords and not ckpt.exists():
            item.add_marker(skip_no_ckpt)
        if "requires_compatible_gpu" in item.keywords and not have_gpu:
            item.add_marker(skip_no_gpu)


@pytest.fixture
def tiny_coco_dir() -> Path:
    return FIXTURES / "tiny_coco"


@pytest.fixture
def tiny_coco_dataset(tiny_coco_dir: Path) -> COCODataset:
    """A COCODataset pointing at the tiny_coco fixture (bbox prompt mode)."""
    from esam3.config.schema import NormalizeConfig, TextPromptConfig
    from esam3.data.transforms import build_eval_transforms

    transforms = build_eval_transforms(
        32, model_name="facebook/sam3.1", normalize=NormalizeConfig()
    )
    return COCODataset(
        annotations=str(tiny_coco_dir / "annotations.json"),
        images=str(tiny_coco_dir / "images"),
        prompt_mode="bbox",
        transforms=transforms,
        text_prompt=TextPromptConfig(),
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
