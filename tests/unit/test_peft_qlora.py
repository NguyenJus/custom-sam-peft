"""CPU smoke tests for esam3.peft_adapters.qlora.

The real work (4-bit module swap + LoRA on quantized base) is GPU-only and
lives in tests/integration/test_peft_qlora_real.py. These tests cover:
  - registry wiring
  - schema parse for PEFTConfig(method="qlora") and QLoRAConfig
  - lazy import of bitsandbytes (module import must succeed without bnb)
  - ImportError surface when apply_qlora is called without bnb
"""

from __future__ import annotations

import sys
from typing import Any

import pytest

from esam3._registry import lookup
from esam3.config.schema import PEFTConfig, QLoRAConfig
from tests.fixtures.tiny_sam3_lora_stub import make_stub_wrapper


def test_registry_lookup() -> None:
    """apply_qlora is registered under ('peft', 'qlora')."""
    from esam3.peft_adapters.qlora import apply_qlora

    assert lookup("peft", "qlora") is apply_qlora


def test_schema_qlora_method_defaults() -> None:
    """PEFTConfig(method='qlora') validates with default QLoRAConfig."""
    cfg = PEFTConfig(method="qlora")
    assert cfg.method == "qlora"
    assert cfg.qlora.quant_type == "nf4"
    assert cfg.qlora.compute_dtype == "bfloat16"


def test_schema_rejects_bogus_quant_type() -> None:
    """quant_type must be one of the literal values."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        QLoRAConfig(quant_type="bogus")  # type: ignore[arg-type]


def test_schema_rejects_bogus_compute_dtype() -> None:
    """compute_dtype must be one of the literal values."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        QLoRAConfig(compute_dtype="float32")  # type: ignore[arg-type]


def test_import_does_not_require_bitsandbytes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Importing esam3.peft_adapters.qlora must succeed without bitsandbytes installed."""
    # Force re-import after hiding bnb.
    monkeypatch.setitem(sys.modules, "bitsandbytes", None)
    monkeypatch.delitem(sys.modules, "esam3.peft_adapters.qlora", raising=False)

    import esam3.peft_adapters.qlora as qlora  # imports OK without bnb

    # Sanity: the public symbols are reachable.
    assert hasattr(qlora, "apply_qlora")
    assert hasattr(qlora, "save_qlora")
    assert hasattr(qlora, "load_qlora")


def test_apply_qlora_raises_helpful_importerror_when_bnb_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Calling apply_qlora without bnb raises ImportError naming the [qlora] extra."""
    monkeypatch.setitem(sys.modules, "bitsandbytes", None)
    monkeypatch.delitem(sys.modules, "esam3.peft_adapters.qlora", raising=False)

    from esam3.peft_adapters.qlora import apply_qlora

    w = make_stub_wrapper()
    cfg = PEFTConfig(method="qlora")
    with pytest.raises(ImportError, match=r"\[qlora\]"):
        apply_qlora(w, cfg)


def test_save_qlora_raises_when_no_peft_model(tmp_path: Any) -> None:
    """save_qlora requires apply_qlora to have run first."""
    from esam3.peft_adapters.qlora import save_qlora

    w = make_stub_wrapper()
    assert w.peft_model is None
    with pytest.raises(RuntimeError, match="no PeftModel"):
        save_qlora(w, tmp_path)


def test_load_qlora_raises_when_peft_model_already_set(tmp_path: Any) -> None:
    """load_qlora refuses to overwrite a wrapper that already has an adapter."""
    from esam3.peft_adapters.qlora import load_qlora

    w = make_stub_wrapper()
    # Fake a previously-applied state. The real type is PeftModel; for this
    # guard test any non-None object suffices.
    w.peft_model = object()  # type: ignore[assignment]
    with pytest.raises(RuntimeError, match="already has a PeftModel"):
        load_qlora(w, tmp_path)
