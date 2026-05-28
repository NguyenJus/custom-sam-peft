"""Tests for predict/adapter_load.py — adapter detection and load dispatch."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import typer

from custom_sam_peft.predict.adapter_load import (
    AdapterKind,
    detect_adapter_kind,
    load_adapter,
    maybe_merge_adapter,
    read_adapter_base_model_name,
)

# Resolve fixture paths relative to this file so tests are location-independent.
_FIXTURES = Path(__file__).parent / "fixtures"
_LORA_DIR = _FIXTURES / "lora_adapter"
_QLORA_DIR = _FIXTURES / "qlora_adapter"
_BAD_DIR = _FIXTURES / "bad_adapter"


# ---------------------------------------------------------------------------
# detect_adapter_kind
# ---------------------------------------------------------------------------


def test_detect_adapter_kind_lora() -> None:
    """A dir with only adapter_config.json (no custom_sam_peft_qlora.json) → "lora"."""
    kind: AdapterKind = detect_adapter_kind(_LORA_DIR)
    assert kind == "lora"


def test_detect_adapter_kind_qlora() -> None:
    """A dir with custom_sam_peft_qlora.json present → "qlora" (sentinel wins)."""
    kind: AdapterKind = detect_adapter_kind(_QLORA_DIR)
    assert kind == "qlora"


def test_detect_adapter_kind_missing_adapter_config_raises() -> None:
    """A dir with neither sentinel nor adapter_config.json → typer.BadParameter."""
    with pytest.raises(typer.BadParameter, match=r"adapter_config\.json"):
        detect_adapter_kind(_BAD_DIR)


def test_detect_adapter_kind_qlora_only_no_adapter_config(tmp_path: Path) -> None:
    """A dir with ONLY custom_sam_peft_qlora.json (no adapter_config.json) → "qlora", no raise."""
    (tmp_path / "custom_sam_peft_qlora.json").write_text("{}", encoding="utf-8")
    kind: AdapterKind = detect_adapter_kind(tmp_path)
    assert kind == "qlora"


# ---------------------------------------------------------------------------
# load_adapter dispatch
# ---------------------------------------------------------------------------


def test_load_adapter_dispatches_to_lora() -> None:
    """load_adapter with kind="lora" calls peft_adapters.lora.load_lora, not load_qlora."""
    fake_model = MagicMock()
    fake_result = MagicMock()

    with (
        patch(
            "custom_sam_peft.peft_adapters.lora.load_lora",
            return_value=fake_result,
        ) as mock_lora,
        patch(
            "custom_sam_peft.peft_adapters.qlora.load_qlora",
        ) as mock_qlora,
    ):
        result = load_adapter(fake_model, _LORA_DIR, "lora")

    mock_lora.assert_called_once_with(fake_model, _LORA_DIR)
    mock_qlora.assert_not_called()
    assert result is fake_result


def test_load_adapter_dispatches_to_qlora() -> None:
    """load_adapter with kind="qlora" calls peft_adapters.qlora.load_qlora, not load_lora."""
    fake_model = MagicMock()
    fake_result = MagicMock()

    with (
        patch(
            "custom_sam_peft.peft_adapters.qlora.load_qlora",
            return_value=fake_result,
        ) as mock_qlora,
        patch(
            "custom_sam_peft.peft_adapters.lora.load_lora",
        ) as mock_lora,
    ):
        result = load_adapter(fake_model, _QLORA_DIR, "qlora")

    mock_qlora.assert_called_once_with(fake_model, _QLORA_DIR)
    mock_lora.assert_not_called()
    assert result is fake_result


# ---------------------------------------------------------------------------
# maybe_merge_adapter toggle
# ---------------------------------------------------------------------------


def test_merge_adapter_toggle_off_skips_merge_lora() -> None:
    """merge=False → merge_lora is NOT called; the original model is returned."""
    fake_model = MagicMock()

    with patch("custom_sam_peft.peft_adapters.lora.merge_lora") as mock_merge:
        result = maybe_merge_adapter(fake_model, merge=False)

    mock_merge.assert_not_called()
    assert result is fake_model


def test_merge_adapter_toggle_on_calls_merge_lora() -> None:
    """merge=True → merge_lora IS called with the model; merged result is returned."""
    fake_model = MagicMock()
    merged_model = MagicMock()

    with patch(
        "custom_sam_peft.peft_adapters.lora.merge_lora",
        return_value=merged_model,
    ) as mock_merge:
        result = maybe_merge_adapter(fake_model, merge=True)

    mock_merge.assert_called_once_with(fake_model)
    assert result is merged_model


# ---------------------------------------------------------------------------
# read_adapter_base_model_name
# ---------------------------------------------------------------------------


def test_read_adapter_base_model_name_lora() -> None:
    """Reads base_model_name_or_path from a LoRA adapter_config.json."""
    name = read_adapter_base_model_name(_LORA_DIR)
    assert name == "facebook/sam3.1"


def test_read_adapter_base_model_name_absent_returns_none(tmp_path: Path) -> None:
    """Returns None when base_model_name_or_path is absent from the JSON."""
    cfg = tmp_path / "adapter_config.json"
    cfg.write_text('{"peft_type": "LORA"}', encoding="utf-8")
    assert read_adapter_base_model_name(tmp_path) is None


def test_read_adapter_base_model_name_no_file_returns_none(tmp_path: Path) -> None:
    """Returns None when adapter_config.json does not exist at all."""
    assert read_adapter_base_model_name(tmp_path) is None


def test_read_base_model_name_delegates() -> None:
    """The predict delegator and the relocated peft_adapters impl agree."""
    from custom_sam_peft.peft_adapters import (
        read_adapter_base_model_name as _impl,
    )

    assert read_adapter_base_model_name(_LORA_DIR) == _impl(_LORA_DIR)


def test_detect_adapter_kind_delegates_and_still_validates() -> None:
    """detect_adapter_kind agrees with the canonical seam for lora/qlora dirs
    AND still raises typer.BadParameter on a dir missing adapter_config.json."""
    from custom_sam_peft.peft_adapters import discover_method_from_checkpoint

    assert detect_adapter_kind(_LORA_DIR) == discover_method_from_checkpoint(_LORA_DIR)
    assert detect_adapter_kind(_QLORA_DIR) == discover_method_from_checkpoint(_QLORA_DIR)
    with pytest.raises(typer.BadParameter, match=r"adapter_config\.json"):
        detect_adapter_kind(_BAD_DIR)
