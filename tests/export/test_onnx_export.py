"""CPU round-trip + sidecar tests for the ONNX export module (spec §10.1, §10.5).

These exercise the export orchestrator, tracers, and sidecar writers end-to-end
against ``TinySam3Stub`` (no real SAM 3.1 checkpoint). The real-model trace is
validated separately on GPU; here we prove the shims trace at tiny scale and
the sidecars carry exactly what the transform pipeline resolves.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from custom_sam_peft.config.schema import NormalizeConfig, TrainConfig
from custom_sam_peft.data.transforms import resolve_normalization_with_path
from custom_sam_peft.models.sam3 import SAM3_IMAGE_SIZE, Sam3Wrapper, _Sam3ImageAdapter
from custom_sam_peft.train.checkpoint import _hash_cfg
from tests.fixtures.tiny_sam3_stub import TinySam3Stub

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def _coco_cfg(
    tmp_path: Path, *, channels: int = 3, semantics: str = "rgb", normalize: dict | None = None
) -> TrainConfig:
    """Minimal coco TrainConfig pointing at the tiny_coco fixture."""
    coco = FIXTURES / "tiny_coco"
    data: dict = {
        "format": "coco",
        "train": {"annotations": str(coco / "annotations.json"), "images": str(coco / "images")},
        "val": {"annotations": str(coco / "annotations.json"), "images": str(coco / "images")},
        "channels": channels,
        "channel_semantics": semantics,
    }
    if normalize is not None:
        data["normalize"] = normalize
    return TrainConfig.model_validate(
        {
            "run": {"name": "onnx-test", "output_dir": str(tmp_path / "runs"), "seed": 0},
            "data": data,
            "peft": {"method": "lora"},
            "train": {"epochs": 1},
        }
    )


def _stub_wrapper(*, channels: int = 3, semantics: str = "rgb") -> Sam3Wrapper:
    """A Sam3Wrapper whose merged inner model is a traceable TinySam3Stub."""
    stub = TinySam3Stub(num_queries=4, mask_size=8)
    adapter = _Sam3ImageAdapter(stub, channels=channels, channel_semantics=semantics)
    return Sam3Wrapper(adapter, mask_size=8, channels=channels, channel_semantics=semantics)


def _patch_merge(monkeypatch: pytest.MonkeyPatch, wrapper: Sam3Wrapper) -> None:
    """Replace _merge_and_cast so tests never touch a real adapter/checkpoint."""
    import torch

    from custom_sam_peft.export import onnx as onnx_mod

    def _fake(cfg, checkpoint, *, fp16):
        # Mirror the real _merge_and_cast precision normalization so the traced
        # stub graph is single-dtype (channel adapter + inner model both cast).
        dtype = torch.float16 if fp16 else torch.float32
        wrapper.model.model = wrapper.model.model.to(dtype)
        if wrapper.model.channel_adapter is not None:
            wrapper.model.channel_adapter = wrapper.model.channel_adapter.to(dtype)
        return wrapper, "lora", dtype

    monkeypatch.setattr(onnx_mod, "_merge_and_cast", _fake)


# ---------------------------------------------------------------------------
# §10.1 round-trip, parametrized by --include
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("include", "encoder_present", "decoder_present"),
    [("all", True, True), ("encoder", True, False), ("decoder", False, True)],
)
def test_round_trip_file_set(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    include: str,
    encoder_present: bool,
    decoder_present: bool,
) -> None:
    """Bundle contains EXACTLY the expected files per --include; sidecars match sources."""
    from custom_sam_peft.export.onnx import (
        DECODER_FILE,
        ENCODER_FILE,
        MODEL_CARD_FILE,
        PREPROCESSOR_FILE,
        PROMPTS_FILE,
        run_export_onnx,
    )

    cfg = _coco_cfg(tmp_path)
    _patch_merge(monkeypatch, _stub_wrapper())
    out = tmp_path / "bundle"

    result = run_export_onnx(
        cfg,
        tmp_path / "ckpt",
        output=out,
        opset=17,
        fp16=False,
        include=include,
        dynamic_axes=True,
        check=False,
    )
    assert result == out

    names = {p.name for p in out.iterdir()}
    assert (ENCODER_FILE in names) == encoder_present
    assert (DECODER_FILE in names) == decoder_present
    # Sidecars always present; never a README.
    assert {PREPROCESSOR_FILE, PROMPTS_FILE, MODEL_CARD_FILE} <= names
    assert "README.md" not in names

    pp = json.loads((out / PREPROCESSOR_FILE).read_text())
    assert pp["image_size"] == SAM3_IMAGE_SIZE
    mean, std, path = resolve_normalization_with_path(
        cfg.model.name,
        cfg.data.normalize or NormalizeConfig(),
        channel_semantics=cfg.data.channel_semantics,
    )
    assert pp["mean"] == mean
    assert pp["std"] == std
    assert pp["normalization_path"] == path

    card = json.loads((out / MODEL_CARD_FILE).read_text())
    assert card["training_config_hash"] == _hash_cfg(cfg)
    assert card["include"] == include
    assert card["parity_checked"] is False

    # prompts.txt == dataset.class_names in order (tiny_coco => thing_a, thing_b).
    from custom_sam_peft.train.runner import _build_dataset

    expected = list(_build_dataset(cfg, "train").class_names)
    assert (out / PROMPTS_FILE).read_text().splitlines() == expected


def test_check_passes_promotes_bundle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """--check passes (torch == ORT on the stub): no raise, bundle promoted, card parity True."""
    from custom_sam_peft.export.onnx import MODEL_CARD_FILE, run_export_onnx

    cfg = _coco_cfg(tmp_path)
    _patch_merge(monkeypatch, _stub_wrapper())
    out = tmp_path / "bundle"
    result = run_export_onnx(
        cfg,
        tmp_path / "ckpt",
        output=out,
        opset=17,
        fp16=False,
        include="all",
        dynamic_axes=True,
        check=True,
    )
    assert result.exists()
    card = json.loads((out / MODEL_CARD_FILE).read_text())
    assert card["parity_checked"] is True


@pytest.mark.parametrize(("fp16", "band"), [(False, "fp32"), (True, "fp16")])
def test_check_fails_on_drift_no_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fp16: bool, band: str
) -> None:
    """--check drift: ExportParityError names the key + band; NO dir at output (staging removed)."""
    from custom_sam_peft.export.onnx import ExportParityError, run_export_onnx
    from custom_sam_peft.predict import onnx_session as ort_mod

    real_run_decoder = ort_mod._OrtCore.run_decoder

    def _perturbed(self, vision_feats, classes):  # type: ignore[no-untyped-def]
        out = real_run_decoder(self, vision_feats, classes)
        out = dict(out)
        out["pred_masks"] = out["pred_masks"] + 0.5  # > both fp32 and fp16 bands
        return out

    monkeypatch.setattr(ort_mod._OrtCore, "run_decoder", _perturbed)

    cfg = _coco_cfg(tmp_path)
    _patch_merge(monkeypatch, _stub_wrapper())
    out = tmp_path / "bundle"
    with pytest.raises(ExportParityError) as ei:
        run_export_onnx(
            cfg,
            tmp_path / "ckpt",
            output=out,
            opset=17,
            fp16=fp16,
            include="all",
            dynamic_axes=True,
            check=True,
        )
    msg = str(ei.value)
    assert "pred_masks" in msg
    assert band in msg
    assert not out.exists()
    assert not out.with_name(out.name + ".tmp-onnx").exists()


def test_fp16_export_dtype(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """fp16 export traces a graph and writes the bundle (model_card.fp16 True)."""
    from custom_sam_peft.export.onnx import MODEL_CARD_FILE, run_export_onnx

    cfg = _coco_cfg(tmp_path)
    _patch_merge(monkeypatch, _stub_wrapper())
    out = tmp_path / "bundle"
    run_export_onnx(
        cfg,
        tmp_path / "ckpt",
        output=out,
        opset=17,
        fp16=True,
        include="all",
        dynamic_axes=True,
        check=False,
    )
    card = json.loads((out / MODEL_CARD_FILE).read_text())
    assert card["fp16"] is True


# ---------------------------------------------------------------------------
# §5.1 merge guards (no monkeypatch — exercise _merge_and_cast directly)
# ---------------------------------------------------------------------------


def test_qlora_fp16_off_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """QLoRA + --fp16 off raises ValueError with the --fp16 hint."""
    from custom_sam_peft.export import onnx as onnx_mod

    monkeypatch.setattr(onnx_mod, "discover_method_from_checkpoint", lambda p: "qlora")
    cfg = _coco_cfg(tmp_path)
    with pytest.raises(ValueError, match="--fp16"):
        onnx_mod._merge_and_cast(cfg, tmp_path / "ckpt", fp16=False)


def test_qlora_no_cuda_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """QLoRA + no CUDA raises RuntimeError about a required CUDA device."""
    import torch

    from custom_sam_peft.export import onnx as onnx_mod

    monkeypatch.setattr(onnx_mod, "discover_method_from_checkpoint", lambda p: "qlora")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    cfg = _coco_cfg(tmp_path)
    with pytest.raises(RuntimeError, match="CUDA"):
        onnx_mod._merge_and_cast(cfg, tmp_path / "ckpt", fp16=True)


def test_qlora_dequant_export_fp16(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """QLoRA marker present + fp16 + forced CUDA: merged module dequantizes to fp16."""
    import torch

    from custom_sam_peft.export import onnx as onnx_mod
    from custom_sam_peft.peft_adapters import _QLORA_META_FILENAME

    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / _QLORA_META_FILENAME).write_text("{}")  # marker => discover => "qlora"

    wrapper = _stub_wrapper()
    # No real CUDA on CI; force the QLoRA-device branch + neutralize the GPU moves.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(onnx_mod, "load_sam31", lambda *a, **k: wrapper)
    monkeypatch.setattr(onnx_mod, "load_adapter", lambda w, c: w)
    monkeypatch.setattr(torch.nn.Module, "to", lambda self, *a, **k: self)

    import custom_sam_peft.peft_adapters.lora as lora_mod

    monkeypatch.setattr(lora_mod, "merge_lora", lambda w: None)

    out_wrapper, method, export_dtype = onnx_mod._merge_and_cast(
        cfg=_coco_cfg(tmp_path), checkpoint=ckpt, fp16=True
    )
    assert method == "qlora"
    assert export_dtype is torch.float16
    merged = out_wrapper.model.model
    assert all(p.dtype is torch.float16 for p in merged.parameters())


# ---------------------------------------------------------------------------
# §10.5 non-rgb export
# ---------------------------------------------------------------------------


def test_non_rgb_export_folds_channel_adapter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Non-rgb: N->3 Conv2d folds into the encoder; preprocessor carries freeform stats."""
    import onnx

    from custom_sam_peft.export.onnx import ENCODER_FILE, PREPROCESSOR_FILE, run_export_onnx

    norm = {"mean": [0.1, 0.2, 0.3, 0.4], "std": [0.5, 0.6, 0.7, 0.8], "max_pixel_value": 1.0}
    cfg = _coco_cfg(tmp_path, channels=4, semantics="rgba", normalize=norm)
    _patch_merge(monkeypatch, _stub_wrapper(channels=4, semantics="rgba"))
    out = tmp_path / "bundle"
    run_export_onnx(
        cfg,
        tmp_path / "ckpt",
        output=out,
        opset=17,
        fp16=False,
        include="all",
        dynamic_axes=True,
        check=False,
    )

    model = onnx.load(str(out / ENCODER_FILE))
    in_shape = model.graph.input[0].type.tensor_type.shape.dim
    assert in_shape[1].dim_value == 4  # folded N=4 -> 3 channel adapter input

    pp = json.loads((out / PREPROCESSOR_FILE).read_text())
    assert pp["channels"] == 4
    assert pp["channel_semantics"] == "rgba"
    assert pp["mean"] == [0.1, 0.2, 0.3, 0.4]
    assert pp["std"] == [0.5, 0.6, 0.7, 0.8]


def test_non_rgb_no_normalize_warns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Non-rgb config with normalize unset emits the §6.1 WARNING from the writer."""
    import logging

    from custom_sam_peft.export.onnx import _write_preprocessor

    # grayscale has a profile default normalize, but we force normalize=None to
    # trigger the writer's warning branch directly (config-fallback path).
    cfg = _coco_cfg(
        tmp_path,
        channels=4,
        semantics="rgba",
        normalize={"mean": [0.1, 0.2, 0.3, 0.4], "std": [0.5, 0.6, 0.7, 0.8]},
    )
    object.__setattr__(cfg.data, "normalize", None)
    with caplog.at_level(logging.WARNING):
        _write_preprocessor(tmp_path, cfg)
    assert any("normalize" in r.message.lower() for r in caplog.records)
