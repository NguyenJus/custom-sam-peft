"""ONNX --use-onnx predict integration tests (spec §10.6-§10.9).

All CPU-only. Tiny bundles are built by tracing ``TinySam3Stub`` submodules
through ``run_export_onnx`` with a monkeypatched ``_merge_and_cast`` (the same
seam ``tests/export`` uses), so no real SAM 3.1 checkpoint is needed.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import torch
from PIL import Image as PILImage

from custom_sam_peft.config.schema import TrainConfig
from custom_sam_peft.models.sam3 import Sam3Wrapper, _Sam3ImageAdapter
from custom_sam_peft.predict.runner import PredictOptions, run_predict
from tests.fixtures.tiny_sam3_stub import TinySam3Stub

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"

# Bundle-decoder spatial mask size (TinySam3Stub mask_size); kept tiny for CPU speed.
_MASK_SIZE = 8
_NUM_QUERIES = 4


# ---------------------------------------------------------------------------
# Bundle-building helpers (trace TinySam3Stub via run_export_onnx)
# ---------------------------------------------------------------------------


def _coco_cfg(tmp_path: Path) -> TrainConfig:
    """Minimal coco TrainConfig pointing at the tiny_coco fixture (classes thing_a/thing_b)."""
    coco = FIXTURES / "tiny_coco"
    data: dict[str, Any] = {
        "format": "coco",
        "train": {"annotations": str(coco / "annotations.json"), "images": str(coco / "images")},
        "val": {"annotations": str(coco / "annotations.json"), "images": str(coco / "images")},
        "channels": 3,
        "channel_semantics": "rgb",
    }
    return TrainConfig.model_validate(
        {
            "run": {"name": "onnx-predict-test", "output_dir": str(tmp_path / "runs"), "seed": 0},
            "data": data,
            "peft": {"method": "lora"},
            "train": {"epochs": 1},
        }
    )


def _stub_wrapper() -> Sam3Wrapper:
    """A Sam3Wrapper whose merged inner model is a traceable TinySam3Stub."""
    stub = TinySam3Stub(num_queries=_NUM_QUERIES, mask_size=_MASK_SIZE)
    adapter = _Sam3ImageAdapter(stub, channels=3, channel_semantics="rgb")
    return Sam3Wrapper(adapter, mask_size=_MASK_SIZE, channels=3, channel_semantics="rgb")


def _patch_merge(monkeypatch: pytest.MonkeyPatch, wrapper: Sam3Wrapper) -> None:
    """Replace _merge_and_cast so the bundle build never touches a real adapter."""
    from custom_sam_peft.export import onnx as onnx_mod

    def _fake(cfg: Any, checkpoint: Any, *, fp16: bool) -> tuple[Sam3Wrapper, str, torch.dtype]:
        dtype = torch.float16 if fp16 else torch.float32
        wrapper.model.model = wrapper.model.model.to(dtype)
        if wrapper.model.channel_adapter is not None:
            wrapper.model.channel_adapter = wrapper.model.channel_adapter.to(dtype)
        return wrapper, "lora", dtype

    monkeypatch.setattr(onnx_mod, "_merge_and_cast", _fake)


def _build_bundle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Trace a TinySam3Stub into a full ONNX bundle and return the bundle dir."""
    from custom_sam_peft.export.onnx import run_export_onnx

    cfg = _coco_cfg(tmp_path)
    _patch_merge(monkeypatch, _stub_wrapper())
    out = tmp_path / "bundle"
    return run_export_onnx(
        cfg,
        tmp_path / "ckpt",
        output=out,
        opset=17,
        fp16=False,
        include="all",
        dynamic_axes=True,
        check=False,
    )


def _make_image_dir(tmp_path: Path, n: int = 1) -> Path:
    """Write ``n`` small RGB PNGs and return their directory."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    for i in range(n):
        PILImage.new("RGB", (32, 32), color=(i * 20, 80, 160)).save(img_dir / f"img_{i:03d}.png")
    return img_dir


def _onnx_opts(
    tmp_path: Path,
    bundle: Path,
    *,
    images: Path,
    prompts: str,
    out_name: str = "out",
    config: Path | None = None,
    batch_size: int | str = 1,
) -> PredictOptions:
    """A PredictOptions wired for the --use-onnx CPU path."""
    return PredictOptions(
        images=images,
        prompts=prompts,
        output=tmp_path / out_name,
        checkpoint=None,
        merge_adapter=False,
        config=config,
        score_threshold=0.0,
        top_k=100,
        save_masks="rle",
        visualize=False,
        device="cpu",
        dtype="float32",
        batch_size=batch_size,
        seed=0,
        dry_run=False,
        verbose=False,
        use_onnx=bundle,
    )


# ---------------------------------------------------------------------------
# §10.6 separate-process ORT-only load (no torch import)
# ---------------------------------------------------------------------------


def test_subprocess_ort_only_session_load_no_torch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A child process opens the bundle's InferenceSession with onnxruntime only; no torch."""
    bundle = _build_bundle(tmp_path, monkeypatch)
    child = f"""
import sys, onnxruntime
sess = onnxruntime.InferenceSession(
    {str(bundle / "decoder.onnx")!r}, providers=["CPUExecutionProvider"]
)
assert sess is not None
assert "torch" not in sys.modules, sorted(k for k in sys.modules if "torch" in k)
print("OK")
"""
    result = subprocess.run(  # noqa: S603 - trusted: sys.executable with a literal probe script
        [sys.executable, "-c", child],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


# ---------------------------------------------------------------------------
# §10.7 --use-onnx round-trip (CPU)
# ---------------------------------------------------------------------------


def test_use_onnx_round_trip_writes_run_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """run_predict with use_onnx writes predictions.json/run.json; model_source == 'onnx'."""
    bundle = _build_bundle(tmp_path, monkeypatch)
    images = _make_image_dir(tmp_path)
    opts = _onnx_opts(tmp_path, bundle, images=images, prompts="thing_a,thing_b")

    report = run_predict(opts)

    out = tmp_path / "out"
    assert (out / "predictions.json").exists()
    assert (out / "run.json").exists()
    run_json = json.loads((out / "run.json").read_text())
    assert run_json["model_source"] == "onnx"
    assert run_json["onnx_bundle"] == str(bundle)
    assert "onnx_opset" in run_json
    assert report.n_images == 1


# ---------------------------------------------------------------------------
# §10.8 parity vs torch (CPU)
# ---------------------------------------------------------------------------


def test_use_onnx_parity_vs_torch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Same image+prompts through torch and ORT yield matching entry counts + score/bbox."""
    bundle = _build_bundle(tmp_path, monkeypatch)
    images = _make_image_dir(tmp_path)

    # --- ORT path ---
    onnx_opts = _onnx_opts(
        tmp_path, bundle, images=images, prompts="thing_a,thing_b", out_name="onnx_out"
    )
    run_predict(onnx_opts)
    onnx_preds = json.loads((tmp_path / "onnx_out" / "predictions.json").read_text())

    # --- Torch path (same merged stub graph, loaded via load_sam31 monkeypatch) ---
    torch_wrapper = _stub_wrapper()
    monkeypatch.setattr(
        "custom_sam_peft.models.sam3.load_sam31",
        lambda cfg, **kwargs: torch_wrapper,
    )
    torch_opts = PredictOptions(
        images=images,
        prompts="thing_a,thing_b",
        output=tmp_path / "torch_out",
        checkpoint=None,
        merge_adapter=False,
        config=None,
        score_threshold=0.0,
        top_k=100,
        save_masks="rle",
        visualize=False,
        device="cpu",
        dtype="float32",
        seed=0,
        dry_run=False,
        verbose=False,
        use_onnx=None,
    )
    run_predict(torch_opts)
    torch_preds = json.loads((tmp_path / "torch_out" / "predictions.json").read_text())

    assert len(onnx_preds) == len(torch_preds)
    # Compare score/bbox per matched entry (same image_id+category_id ordering).
    onnx_sorted = sorted(onnx_preds, key=lambda e: (e["image_id"], e["category_id"], e["score"]))
    torch_sorted = sorted(torch_preds, key=lambda e: (e["image_id"], e["category_id"], e["score"]))
    for o, t in zip(onnx_sorted, torch_sorted, strict=True):
        assert o["category_id"] == t["category_id"]
        assert abs(o["score"] - t["score"]) <= 1e-3
        for ob, tb in zip(o["bbox"], t["bbox"], strict=True):
            assert abs(ob - tb) <= 1e-3 + 1e-3 * abs(tb)


# ---------------------------------------------------------------------------
# §10.9 semantic --use-onnx (CPU)
# ---------------------------------------------------------------------------


def _semantic_config(tmp_path: Path) -> Path:
    """A predict --config YAML declaring task: semantic with rgb channels."""
    cfg_path = tmp_path / "semantic.yaml"
    cfg_path.write_text(
        "task: semantic\nmodel:\n  name: facebook/sam3.1\ndata:\n"
        "  channels: 3\n  channel_semantics: rgb\n",
        encoding="utf-8",
    )
    return cfg_path


def test_semantic_use_onnx_matches_torch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Semantic task + use_onnx writes a label map matching the torch semantic path."""
    bundle = _build_bundle(tmp_path, monkeypatch)
    # The stub decoder bakes R = TRACE_B(=2) * K rows at trace time, so feed 2 images
    # (b == TRACE_B) to keep the marginalize-group reshape (b, k, H, W) consistent.
    images = _make_image_dir(tmp_path, n=2)
    sem_cfg = _semantic_config(tmp_path)

    # --- ORT semantic path ---
    onnx_opts = _onnx_opts(
        tmp_path,
        bundle,
        images=images,
        prompts="thing_a,thing_b",
        out_name="onnx_sem",
        config=sem_cfg,
        batch_size=2,  # both images in one forward → b == TRACE_B baked into the decoder
    )
    run_predict(onnx_opts)
    onnx_sem = tmp_path / "onnx_sem"
    onnx_label_maps = sorted((onnx_sem / "label_maps").glob("*.png"))
    assert onnx_label_maps, "ONNX semantic path produced no label maps"

    # --- Torch semantic path on the same stub graph ---
    torch_wrapper = _stub_wrapper()
    monkeypatch.setattr(
        "custom_sam_peft.models.sam3.load_sam31",
        lambda cfg, **kwargs: torch_wrapper,
    )
    torch_opts = PredictOptions(
        images=images,
        prompts="thing_a,thing_b",
        output=tmp_path / "torch_sem",
        checkpoint=None,
        merge_adapter=False,
        config=sem_cfg,
        score_threshold=0.0,
        top_k=100,
        save_masks="rle",
        visualize=False,
        device="cpu",
        dtype="float32",
        batch_size=2,
        seed=0,
        dry_run=False,
        verbose=False,
        use_onnx=None,
    )
    run_predict(torch_opts)
    torch_sem = tmp_path / "torch_sem"
    torch_label_maps = sorted((torch_sem / "label_maps").glob("*.png"))
    assert torch_label_maps, "torch semantic path produced no label maps"

    # The index label maps must match within tolerance (same graph + same reduction).
    import numpy as np

    onnx_idx = sorted((onnx_sem / "label_maps").glob("*_index.png"))
    torch_idx = sorted((torch_sem / "label_maps").glob("*_index.png"))
    assert len(onnx_idx) == len(torch_idx) >= 1
    for o, t in zip(onnx_idx, torch_idx, strict=True):
        oa = np.asarray(PILImage.open(o))
        ta = np.asarray(PILImage.open(t))
        assert oa.shape == ta.shape
        # Argmax label maps over identical logits must be element-wise identical.
        assert np.array_equal(oa, ta)
