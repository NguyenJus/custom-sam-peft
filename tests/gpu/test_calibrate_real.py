"""GPU smoke test for `custom_sam_peft calibrate` — real activation byte range.

Marked `gpu_t4` (CC 7.5 floor: Tesla T4 / RTX 5070 Ti) so it is skipped on
CPU CI. Runs the full probe and asserts the measured per-example activation
cost (`A_per_class` in the v3 cache schema) lands in a sane order-of-magnitude
bracket — 0.5 GiB to 10 GiB per example at image_size=1008 on a compatible GPU.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

from custom_sam_peft.cli.main import app


@pytest.mark.gpu_t4
@pytest.mark.requires_checkpoint
@pytest.mark.requires_compatible_gpu
def test_calibrate_real_activation_in_sane_range(tmp_path: Path) -> None:
    runner = CliRunner()
    os.chdir(tmp_path)
    result = runner.invoke(app, ["calibrate", "--force"])
    assert result.exit_code == 0, result.output
    data = json.loads((tmp_path / ".custom_sam_peft_calibration.json").read_text())
    # v3 cache stores the per-example activation cost as `A_per_class`
    # (the legacy `activation_bytes_per_example` key was removed at schema v3).
    activation = int(data["A_per_class"])
    assert 5e8 <= activation <= 1e10, f"A_per_class={activation} outside [0.5 GiB, 10 GiB]"
