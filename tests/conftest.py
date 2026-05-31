"""Project-level pytest hooks (markers, autoskips) and shared fixtures.

GPU tier taxonomy (Tesla T4 floor, CC 7.5 / RTX 5070 Ti primary dev card):
- gpu_t4:   CC >= 7.5 AND total VRAM <= 16 GB (fp16 band; bf16 coerced below CC 8.0)
- gpu_bf16: CC >= 8.0 AND total VRAM <= 16 GB (native bf16, RTX 5070 Ti)
- gpu_xl:   total VRAM > 16 GB (deferred — no tests assigned yet)

Bands are NOT linearly ordered; the skip predicate is a capability-subset check.
See docs/testing/gpu-test-policy.md for the full policy.
"""

from __future__ import annotations

import gc
import pathlib
from collections.abc import Iterator
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
        "requires_compatible_gpu: skip unless a CUDA GPU with CC >= 7.5 is available "
        "(Tesla T4 floor; RTX 5070 Ti primary dev card). "
        "See docs/testing/gpu-test-policy.md.",
    )
    config.addinivalue_line(
        "markers",
        "requires_bnb: skip unless bitsandbytes is importable",
    )
    config.addinivalue_line(
        "markers",
        "gpu_t4: requires a CUDA GPU with CC >= 7.5 AND total VRAM <= 16 GB "
        "(Tesla T4 floor and RTX 5070 Ti). fp16 band (bf16 is coerced below CC 8.0). "
        "See docs/testing/gpu-test-policy.md.",
    )
    config.addinivalue_line(
        "markers",
        "gpu_bf16: requires a CUDA GPU with CC >= 8.0 AND total VRAM <= 16 GB "
        "(RTX 5070 Ti). Native, non-coerced bf16 numerics. "
        "See docs/testing/gpu-test-policy.md.",
    )
    config.addinivalue_line(
        "markers",
        "gpu_xl: requires a CUDA GPU with total VRAM > 16 GB. "
        "Empty in this PR; populated only via the gpu_xl follow-up issue. "
        "See docs/testing/gpu-test-policy.md.",
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
    if (major, minor) < (7, 5):
        return False
    return _torch_can_launch_kernel()


_GB = 1024**3


def _satisfied_tiers() -> set[str]:
    """Return the SET of GPU tiers the live card satisfies.

    Bands are NOT linearly ordered: gpu_t4/gpu_bf16 are <=16 GB; gpu_xl is >16 GB.
    The 16 GB band is a CLOSED upper bound (<= 16 * _GB) so a card reporting
    slightly under a marketing "16 GB" (driver-reserved) still counts as gpu_t4/gpu_bf16.
    A >16 GB card satisfies only gpu_xl and is intentionally NOT auto-run for the
    <=16 GB ceiling assertions (running them on a bigger card could mask a small-card OOM).
    """
    import torch

    if not _has_compatible_gpu():
        return set()
    cc = torch.cuda.get_device_capability()
    total = torch.cuda.get_device_properties(0).total_memory
    tiers: set[str] = set()
    if cc >= (7, 5) and total <= 16 * _GB:
        tiers.add("gpu_t4")
    if cc >= (8, 0) and total <= 16 * _GB:
        tiers.add("gpu_bf16")
    if total > 16 * _GB:
        tiers.add("gpu_xl")
    return tiers


_KNOWN_TIERS = frozenset({"gpu_t4", "gpu_bf16", "gpu_xl"})

# Human-readable gate descriptions for skip reasons — one per tier.
_TIER_GATES = {
    "gpu_t4": "CC >= 7.5 AND total VRAM <= 16 GB",
    "gpu_bf16": "CC >= 8.0 AND total VRAM <= 16 GB",
    "gpu_xl": "total VRAM > 16 GB",
}


def _should_skip(marker_tier: str, satisfied: set[str]) -> str | None:
    """Return a skip-reason string if *marker_tier* is not in *satisfied*, else None.

    Pure function — no I/O.  The caller (pytest_collection_modifyitems) supplies
    the live *satisfied* set so unit tests can monkeypatch just the probe.
    """
    if marker_tier in satisfied:
        return None
    gate = _TIER_GATES.get(marker_tier, marker_tier)
    return f"requires {marker_tier} ({gate}); not satisfied on this runner"


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    ckpt = pathlib.Path("models/sam3.1/sam3.1_multiplex.pt")
    skip_no_ckpt = pytest.mark.skip(reason="real SAM 3.1 checkpoint not present locally")
    skip_no_gpu = pytest.mark.skip(
        reason="requires a CUDA GPU with CC >= 7.5 (Tesla T4 floor; RTX 5070 Ti primary)"
    )
    have_gpu = _has_compatible_gpu()
    active_tiers = _satisfied_tiers()
    for item in items:
        if "requires_checkpoint" in item.keywords and not ckpt.exists():
            item.add_marker(skip_no_ckpt)
        if "requires_compatible_gpu" in item.keywords and not have_gpu:
            item.add_marker(skip_no_gpu)

    for item in items:
        item_tier = next((t for t in _KNOWN_TIERS if t in item.keywords), None)
        if item_tier is None:
            continue
        reason = _should_skip(item_tier, active_tiers)
        if reason is not None:
            item.add_marker(pytest.mark.skip(reason=reason))


@pytest.fixture(autouse=True)
def _free_cuda_after_gpu_test(request: pytest.FixtureRequest) -> Iterator[None]:
    """Release CUDA cache after each GPU-gated test so the local tier doesn't OOM.

    Real-model GPU tests each load the full SAM 3.1 checkpoint (and some load it
    twice, e.g. an export/reload round-trip). Without freeing between tests, the
    caching allocator accumulates and a ~7 GB card (GTX 1080, gpu_local tier)
    OOMs partway through a file. Gated on ``requires_compatible_gpu`` so this is
    a no-op for the CPU suite (the only marker every GPU test carries, stable
    across the tier-marker swap).
    """
    yield
    if request.node.get_closest_marker("requires_compatible_gpu") is None:
        return
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@pytest.fixture
def tiny_coco_dir() -> Path:
    return FIXTURES / "tiny_coco"


@pytest.fixture
def tiny_coco_dataset(tiny_coco_dir: Path) -> COCODataset:
    """A COCODataset pointing at the tiny_coco fixture."""
    from custom_sam_peft.config.schema import NormalizeConfig, TextPromptConfig
    from custom_sam_peft.data.transforms import build_eval_transforms

    transforms = build_eval_transforms(
        32, model_name="facebook/sam3.1", normalize=NormalizeConfig()
    )
    return COCODataset(
        annotations=str(tiny_coco_dir / "annotations.json"),
        images=str(tiny_coco_dir / "images"),
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
