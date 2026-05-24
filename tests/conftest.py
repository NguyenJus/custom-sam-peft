"""Project-level pytest hooks (markers, autoskips) and shared fixtures."""

from __future__ import annotations

import pathlib
from pathlib import Path

import pytest
import torch

from custom_sam_peft.data.coco import COCODataset
from custom_sam_peft.tracking.noop import NoopTracker
from tests.fixtures.tiny_sam3_stub import TinySam3Stub

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "requires_checkpoint: skip unless models/sam3.1/sam3.1_multiplex.pt exists",
    )
    config.addinivalue_line(
        "markers",
        "requires_compatible_gpu: skip unless a CUDA device with compute "
        "capability >= 6.0 is available (NF4 QLoRA + LoRA work from CC 6.0 / "
        "Pascal; only LLM.int8() needs CC 7.5 and is unused here)",
    )
    config.addinivalue_line(
        "markers",
        "requires_bnb: skip unless bitsandbytes is importable",
    )
    config.addinivalue_line(
        "markers",
        "gpu_inspection: cheap GPU-gated structural/forward tests (Tier 1); "
        "see docs/testing/gpu-test-policy.md",
    )


def _torch_can_launch_kernel() -> bool:
    """Whether the *installed* torch build can run a kernel on the current CUDA device.

    CC >= 6.0 is necessary but not sufficient: the default cu130 wheel ships no
    sm_61 cubin/PTX, so on a GTX 1080 a kernel launch raises "no kernel image is
    available". The opt-in gpu-pascal (cu118) build JITs compute_60 -> sm_61 and
    runs. Probing an actual launch is the only reliable signal. Separated out so
    unit tests can monkeypatch it without a real GPU.
    """
    try:
        torch.zeros(8, device="cuda").add_(1.0).cpu()
        return True
    except Exception:
        return False


def _has_compatible_gpu() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        major, minor = torch.cuda.get_device_capability()
    except RuntimeError:
        return False
    if (major, minor) < (6, 0):
        return False
    return _torch_can_launch_kernel()


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    ckpt = pathlib.Path("models/sam3.1/sam3.1_multiplex.pt")
    skip_no_ckpt = pytest.mark.skip(reason="real SAM 3.1 checkpoint not present locally")
    skip_no_gpu = pytest.mark.skip(
        reason="requires a CUDA GPU with CC >= 6.0 (NF4 QLoRA + LoRA; "
        "LLM.int8() would need CC 7.5 but is unused here)"
    )
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
    from custom_sam_peft.config.schema import NormalizeConfig, TextPromptConfig
    from custom_sam_peft.data.transforms import build_eval_transforms

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


@pytest.fixture
def tiny_text_dataset():
    """A 2-image, 2-class in-memory Dataset that yields text prompts and GT masks.

    Designed for evaluator unit tests: predictable image_ids, predictable mask
    geometry, no transforms, no albumentations.
    """
    from custom_sam_peft.data.base import Example, Instance, TextPrompts

    _class_names = ["cat", "dog"]

    def make_ex(image_id: str, class_id: int) -> Example:
        h = w = 8
        image = torch.zeros(3, h, w)
        mask = torch.zeros(h, w, dtype=torch.bool)
        mask[:4, :4] = True
        return Example(
            image=image,
            image_id=image_id,
            prompts=TextPrompts(classes=_class_names),
            instances=[
                Instance(
                    mask=mask,
                    class_id=class_id,
                    box=torch.tensor([0.0, 0.0, 4.0, 4.0]),
                ),
            ],
        )

    examples = [make_ex("img_0", 0), make_ex("img_1", 1)]

    class _InMemDataset:
        class_names = _class_names

        def __len__(self) -> int:
            return len(examples)

        def __getitem__(self, i: int) -> Example:
            return examples[i]

    return _InMemDataset()
