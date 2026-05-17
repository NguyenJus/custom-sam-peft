"""CPU smoke tests for esam3.peft_adapters.qlora.

The real work (4-bit module swap + LoRA on quantized base) is GPU-only and
lives in tests/integration/test_peft_qlora_real.py. These tests cover:
  - registry wiring
  - schema parse for PEFTConfig(method="qlora") and QLoRAConfig
  - lazy import of bitsandbytes (module import must succeed without bnb)
  - ImportError surface when apply_qlora is called without bnb

TDD-red status: this file is committed BEFORE the implementation lands in
Task 3. Three tests are expected to fail against the current
NotImplementedError stub and will go green when Task 3 lands:
  * test_apply_qlora_raises_helpful_importerror_when_bnb_missing
  * test_save_qlora_raises_when_no_peft_model
  * test_load_qlora_raises_when_peft_model_already_set
"""

from __future__ import annotations

import sys
from pathlib import Path

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
        QLoRAConfig(quant_type="bogus")


def test_schema_rejects_bogus_compute_dtype() -> None:
    """compute_dtype must be one of the literal values."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        QLoRAConfig(compute_dtype="float32")


def test_import_does_not_require_bitsandbytes() -> None:
    """qlora.py must not import bitsandbytes at module scope (lazy-import contract).

    Verified via AST inspection rather than monkeypatched re-import: re-importing
    the module triggers the @register("peft", "qlora") decorator a second time,
    which the registry rejects as a duplicate. AST inspection is more direct
    anyway — it pins the static contract.
    """
    import ast
    from pathlib import Path

    import esam3.peft_adapters.qlora as qlora_module

    # The module is already importable (this very import succeeded) — that
    # alone proves importing qlora does not require bitsandbytes at module
    # scope, since bitsandbytes is not installed in the CPU test environment.
    # The AST check below pins this as a static property of the source file,
    # not just a runtime artifact of the current sys.path.
    src = Path(qlora_module.__file__).read_text()
    tree = ast.parse(src)
    for node in tree.body:  # tree.body = top-level only; nested imports ignored
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name != "bitsandbytes", (
                    "qlora.py must not import bitsandbytes at module scope; "
                    "use a lazy import inside apply_qlora/load_qlora instead."
                )
        elif isinstance(node, ast.ImportFrom):
            assert node.module != "bitsandbytes", (
                "qlora.py must not `from bitsandbytes import ...` at module "
                "scope; use a lazy import inside apply_qlora/load_qlora."
            )

    # apply_qlora is reachable now; save_qlora/load_qlora land in Task 3.
    assert hasattr(qlora_module, "apply_qlora")


def test_apply_qlora_raises_helpful_importerror_when_bnb_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Calling apply_qlora without bnb raises ImportError naming the [qlora] extra.

    Setting sys.modules["bitsandbytes"] = None makes a fresh `import
    bitsandbytes` inside apply_qlora's lazy-import helper fail with
    ImportError. We do NOT evict esam3.peft_adapters.qlora from sys.modules
    here — re-importing it would re-fire the @register("peft", "qlora")
    decorator, which the registry rejects as a duplicate.
    """
    from esam3.peft_adapters.qlora import apply_qlora

    monkeypatch.setitem(sys.modules, "bitsandbytes", None)

    w = make_stub_wrapper()
    cfg = PEFTConfig(method="qlora")
    with pytest.raises(ImportError, match=r"\[qlora\]"):
        apply_qlora(w, cfg)


def test_save_qlora_raises_when_no_peft_model(tmp_path: Path) -> None:
    """save_qlora requires apply_qlora to have run first."""
    from esam3.peft_adapters.qlora import save_qlora

    w = make_stub_wrapper()
    assert w.peft_model is None
    with pytest.raises(RuntimeError, match="no PeftModel"):
        save_qlora(w, tmp_path)


def test_load_qlora_raises_when_peft_model_already_set(tmp_path: Path) -> None:
    """load_qlora refuses to overwrite a wrapper that already has an adapter."""
    from esam3.peft_adapters.qlora import load_qlora

    w = make_stub_wrapper()
    # Fake a previously-applied state. The real type is PeftModel; for this
    # guard test any non-None object suffices.
    w.peft_model = object()
    with pytest.raises(RuntimeError, match="already has a PeftModel"):
        load_qlora(w, tmp_path)
