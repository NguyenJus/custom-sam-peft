"""GPU-gated test for N-channel predict: G4 real-model N-channel predict forward.

Requires:
  - A CUDA device with compute capability >= 6.0  (requires_compatible_gpu)
  - The real SAM 3.1 checkpoint at models/sam3.1/sam3.1_multiplex.pt  (requires_checkpoint)

Run explicitly:
    pytest -m gpu_local tests/gpu/test_predict_nchannel_gpu.py -v
"""

from __future__ import annotations

import pytest

pytestmark = [
    pytest.mark.gpu_local,
    pytest.mark.requires_compatible_gpu,
    pytest.mark.requires_checkpoint,
]


def test_G4_real_nchannel_predict(tmp_path):
    """run_predict on a non-rgb multi-channel image produces predictions without error."""
    import numpy as np

    from custom_sam_peft.predict.runner import PredictOptions, PredictReport, run_predict

    # Write a synthetic 4-channel .npy array — exercises the array-read predict path
    # now that resolve_images discovers .npy inputs (spec §6/§11 parity).
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    img = np.random.rand(256, 256, 4).astype(np.float32)
    img_path = img_dir / "img.npy"
    np.save(img_path, img)

    # Minimal predict config: model name, image_size, channels, channel_semantics.
    # No 'format' or 'train' keys needed — runner.py raw-parses only model/data sections.
    cfg_path = tmp_path / "predict_rgba.yaml"
    cfg_path.write_text(
        "model:\n  name: facebook/sam3.1\ndata:\n  channels: 4\n  channel_semantics: rgba\n"
    )

    opts = PredictOptions(
        images=img_dir,
        prompts="thing",
        output=tmp_path / "out",
        checkpoint=None,
        merge_adapter=True,
        config=cfg_path,
        score_threshold=0.0,
        top_k=100,
        save_masks="rle",
        visualize=False,
        device="cuda",
        dtype="bfloat16",
        seed=42,
        dry_run=False,
        verbose=False,
    )
    report = run_predict(opts)
    assert isinstance(report, PredictReport)
    assert report.n_images >= 1
