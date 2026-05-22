"""GPU smoke test for `custom_sam_peft calibrate` — real activation byte range.

Marked `gpu` so it is skipped on CPU CI. Runs the full probe and asserts the
measured `activation_bytes_per_example` lands in a sane order-of-magnitude
bracket — 0.5 GiB to 10 GiB per example at image_size=1008 on a modern GPU.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

from custom_sam_peft.cli.main import app


@pytest.mark.gpu
@pytest.mark.requires_checkpoint
@pytest.mark.requires_compatible_gpu
def test_calibrate_real_activation_in_sane_range(tmp_path: Path) -> None:
    runner = CliRunner()
    os.chdir(tmp_path)
    result = runner.invoke(app, ["calibrate", "--image-size", "1008", "--force"])
    assert result.exit_code == 0, result.output
    data = json.loads((tmp_path / ".custom_sam_peft_calibration.json").read_text())
    activation = int(data["activation_bytes_per_example"])
    assert 5e8 <= activation <= 1e10, (
        f"activation_bytes_per_example={activation} outside [0.5 GiB, 10 GiB]"
    )
