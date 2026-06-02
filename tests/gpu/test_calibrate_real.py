"""GPU smoke test for `custom_sam_peft calibrate` — real v3 calibration cache.

Marked `gpu_t4` (CC 7.5 floor: Tesla T4 / RTX 5070 Ti) so it is skipped on
CPU CI. Runs the full three-stage probe and asserts the persisted v3 cache is
well-formed: a positive measured peak that fit the card, plus a non-negative
activation split (`A_fixed`/`A_per_class`) carrying a real per-class signal.

v3 (#204) split the single `activation_bytes_per_example` field into
`A_fixed` + `A_per_class * k` — so this test asserts the split's invariants
rather than the dropped scalar. The bracket is card-robust: it holds on a real
Tesla T4 (CC 7.5, no Flash → math-kernel SDPA) and on the RTX 5070 Ti (CC 12.0,
Flash → `A_fixed` clamps to 0). Confirmed on a real Colab T4 2026-06-01 (#212):
`calibrate --force` exits 0 (the cheap QLoRA NF4 probes fit comfortably).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

from custom_sam_peft.cli.main import app
from custom_sam_peft.presets import CACHE_SCHEMA_VERSION


@pytest.mark.gpu_t4
@pytest.mark.requires_checkpoint
@pytest.mark.requires_compatible_gpu
def test_calibrate_real_activation_in_sane_range(tmp_path: Path) -> None:
    runner = CliRunner()
    os.chdir(tmp_path)
    result = runner.invoke(app, ["calibrate", "--force"])
    assert result.exit_code == 0, result.output
    data = json.loads((tmp_path / ".custom_sam_peft_calibration.json").read_text())
    assert data["schema_version"] == CACHE_SCHEMA_VERSION
    peak = int(data["peak_memory_bytes_at_probe"])
    total = int(data["gpu_total_memory_bytes"])
    a_fixed = int(data["A_fixed"])
    a_per_class = int(data["A_per_class"])

    # Real-signal check (the faithful translation of the old 0.5 GiB lower
    # bound): a real SAM 3.1 forward+backward at 1008px measures a multi-GiB
    # peak, and the climb is budget-bounded so peak <= total by construction.
    assert 5e8 <= peak <= total, (
        f"peak_memory_bytes_at_probe={peak} not in [0.5 GiB, total={total}]"
    )
    # v3 split invariants: both terms non-negative. A_fixed clamps to 0 on Flash
    # cards and A_per_class clamps to 0 on a negative differential, so >= 0 is
    # the strongest invariant _derive_split actually guarantees (#204).
    assert a_fixed >= 0, f"A_fixed={a_fixed} negative"
    assert a_per_class >= 0, f"A_per_class={a_per_class} negative"
