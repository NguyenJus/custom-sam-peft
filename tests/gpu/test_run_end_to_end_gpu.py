"""End-to-end `custom_sam_peft run` GPU smoke test.

Drives `custom_sam_peft run` (Typer entry) against the same `configs/examples/gpu_smoke_lora.yaml`
fixture used by the other GPU smoke tests. Asserts on the artefacts on disk.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from typer.testing import CliRunner

from custom_sam_peft.cli.main import app

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.requires_compatible_gpu,
    pytest.mark.requires_checkpoint,
]

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "examples" / "gpu_smoke_lora.yaml"


def test_run_end_to_end_writes_bundle(tmp_path: Path, tiny_coco_dir: Path) -> None:
    from custom_sam_peft.config.loader import load_config

    # Materialize a copy of the smoke config pointing at tmp_path output, tiny_coco data.
    cfg = load_config(
        CONFIG_PATH,
        overrides=[
            f"data.train.annotations={tiny_coco_dir / 'annotations.json'}",
            f"data.train.images={tiny_coco_dir / 'images'}",
            f"data.val.annotations={tiny_coco_dir / 'annotations.json'}",
            f"data.val.images={tiny_coco_dir / 'images'}",
            f"run.output_dir={tmp_path}",
        ],
    )
    cfg_path = tmp_path / "smoke.yaml"
    import yaml

    cfg_path.write_text(yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False))

    result = CliRunner().invoke(app, ["run", "--config", str(cfg_path)])
    assert result.exit_code == 0, result.output

    runs = sorted(tmp_path.glob("gpu-smoke-lora-*"))
    assert runs, f"no run dir under {tmp_path}"
    run_dir = runs[-1]

    # Adapter present and non-empty.
    adapter_files = list((run_dir / "adapter").iterdir())
    assert adapter_files, f"adapter dir empty: {run_dir / 'adapter'}"

    # metrics.json parses; has overall.mAP numeric.
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert "overall" in metrics
    mAP = metrics["overall"].get("mAP")
    assert isinstance(mAP, (int, float)) and math.isfinite(mAP) and mAP >= 0.0, (
        f"overall.mAP not finite/non-negative: {mAP}"
    )

    # summary.md exists and mentions mAP.
    summary = (run_dir / "summary.md").read_text()
    assert "mAP" in summary or "0." in summary  # headline embeds the float

    # samples/ has ≤ 6 PNGs.
    pngs = sorted((run_dir / "samples").glob("*.png"))
    assert 0 <= len(pngs) <= 6

    # cfg.export.merge=False in the smoke YAML → no merged/ dir.
    assert not (run_dir / "merged").exists()
