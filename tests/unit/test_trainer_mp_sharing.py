"""Behavior of `_maybe_use_file_system_sharing` — the EMFILE mitigation."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
import torch.multiprocessing as torch_mp

from custom_sam_peft.train.trainer import _maybe_use_file_system_sharing

pytestmark = pytest.mark.skipif(
    "file_descriptor" not in torch_mp.get_all_sharing_strategies(),
    reason="platform without file_descriptor sharing strategy (e.g., macOS)",
)


@pytest.fixture(autouse=True)
def _restore_sharing_strategy() -> Iterator[None]:
    saved = torch_mp.get_sharing_strategy()
    try:
        yield
    finally:
        torch_mp.set_sharing_strategy(saved)


def test_skips_when_num_workers_zero() -> None:
    torch_mp.set_sharing_strategy("file_descriptor")
    assert _maybe_use_file_system_sharing(0) is None
    assert torch_mp.get_sharing_strategy() == "file_descriptor"


def test_switches_when_default_is_file_descriptor() -> None:
    torch_mp.set_sharing_strategy("file_descriptor")
    assert _maybe_use_file_system_sharing(4) == "file_system"
    assert torch_mp.get_sharing_strategy() == "file_system"


def test_noop_when_already_file_system() -> None:
    torch_mp.set_sharing_strategy("file_system")
    assert _maybe_use_file_system_sharing(4) is None
    assert torch_mp.get_sharing_strategy() == "file_system"
