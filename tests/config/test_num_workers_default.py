"""Schema tests for TrainHyperparams.num_workers RAM-tiered default.

16GB-class boxes default to 3 workers; larger boxes to 4. The result is still
capped by ``os.cpu_count()``. An explicit value always overrides the default.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from custom_sam_peft.config import schema
from custom_sam_peft.config.schema import TrainHyperparams


def _patch_host(monkeypatch: pytest.MonkeyPatch, *, total_gib: float, cpus: int) -> None:
    monkeypatch.setattr(
        schema.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(total=int(total_gib * 1024**3)),
    )
    monkeypatch.setattr(schema.os, "cpu_count", lambda: cpus)


@pytest.mark.parametrize(
    ("total_gib", "cpus", "expected"),
    [
        (16.0, 16, 3),  # 16GB-class -> 3
        (15.6, 16, 3),  # WSL/firmware-slack 16GB -> 3
        (18.0, 16, 3),  # boundary is inclusive (<= 18 GiB)
        (18.1, 16, 4),  # just above the boundary -> 4
        (32.0, 16, 4),  # larger box -> 4
        (64.0, 2, 2),  # cpu_count caps below the RAM tier
        (16.0, 1, 1),  # single-core box capped to 1
    ],
)
def test_num_workers_ram_tiered_default(
    monkeypatch: pytest.MonkeyPatch, total_gib: float, cpus: int, expected: int
) -> None:
    _patch_host(monkeypatch, total_gib=total_gib, cpus=cpus)
    assert TrainHyperparams(epochs=1).num_workers == expected


def test_num_workers_explicit_overrides_default(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_host(monkeypatch, total_gib=16.0, cpus=16)
    assert TrainHyperparams(epochs=1, num_workers=8).num_workers == 8
    assert TrainHyperparams(epochs=1, num_workers=0).num_workers == 0
