"""CPU smoke tests for esam3.peft_adapters.qlora.

The real work (4-bit module swap + LoRA on quantized base) is GPU-only and
lives in tests/integration/test_peft_qlora_real.py. These tests cover:
  - registry wiring
  - schema parse for PEFTConfig(method="qlora") and QLoRAConfig
  - lazy import of bitsandbytes (module import must succeed without bnb)
  - ImportError surface when apply_qlora is called without bnb
  - _infer_quant_type_from_wrapper fallback chain (primary + legacy + error)

TDD-red status: this file is committed BEFORE the implementation lands in
Task 3. Three tests are expected to fail against the current
NotImplementedError stub and will go green when Task 3 lands:
  * test_apply_qlora_raises_helpful_importerror_when_bnb_missing
  * test_save_qlora_raises_when_no_peft_model
  * test_load_qlora_raises_when_peft_model_already_set
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import pytest
from torch import nn

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


# ---------------------------------------------------------------------------
# CPU unit tests for _infer_quant_type_from_wrapper fallback chain
#
# These tests do NOT import bitsandbytes; they use a lightweight fake that
# mimics the `Linear4bit` / `Params4bit` shape just enough to exercise the
# attribute-read fallbacks in `_infer_quant_type_from_wrapper`.
# ---------------------------------------------------------------------------


@pytest.fixture()
def fake_bnb(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """Install a tiny fake `bitsandbytes` module exposing `bnb.nn.Linear4bit`."""
    fake = types.ModuleType("bitsandbytes")
    fake_nn = types.ModuleType("bitsandbytes.nn")

    class _FakeLinear4bit(nn.Module):
        def __init__(
            self,
            *,
            weight_quant_type: str | None = None,
            module_quant_type: str | None = None,
        ) -> None:
            super().__init__()
            # Mimic a Params4bit weight with `.quant_type` directly on the weight.
            weight = nn.Parameter(nn.functional.normalize(nn.Linear(2, 2).weight))
            if weight_quant_type is not None:
                weight.quant_type = weight_quant_type  # type: ignore[attr-defined]
            self.weight = weight
            if module_quant_type is not None:
                self.quant_type = module_quant_type  # type: ignore[attr-defined]

    fake_nn.Linear4bit = _FakeLinear4bit  # type: ignore[attr-defined]
    fake.nn = fake_nn  # type: ignore[attr-defined]
    fake.__version__ = "0.fake.0"  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "bitsandbytes", fake)
    monkeypatch.setitem(sys.modules, "bitsandbytes.nn", fake_nn)
    return fake


class _FakeWrapper:
    """Stand-in for Sam3Wrapper holding a `peft_model` attribute."""

    def __init__(self, peft_model: nn.Module) -> None:
        self.peft_model = peft_model


def test_infer_quant_type_primary_path(fake_bnb: types.ModuleType) -> None:
    from esam3.peft_adapters.qlora import _infer_quant_type_from_wrapper

    fake_linear4bit = fake_bnb.nn.Linear4bit(weight_quant_type="nf4")  # type: ignore[attr-defined]
    model = nn.Sequential(fake_linear4bit)
    wrapper: Any = _FakeWrapper(model)
    assert _infer_quant_type_from_wrapper(wrapper) == "nf4"


def test_infer_quant_type_legacy_fallback(fake_bnb: types.ModuleType) -> None:
    from esam3.peft_adapters.qlora import _infer_quant_type_from_wrapper

    fake_linear4bit = fake_bnb.nn.Linear4bit(module_quant_type="fp4")  # type: ignore[attr-defined]
    model = nn.Sequential(fake_linear4bit)
    wrapper: Any = _FakeWrapper(model)
    assert _infer_quant_type_from_wrapper(wrapper) == "fp4"


def test_infer_quant_type_raises_when_both_paths_missing(
    fake_bnb: types.ModuleType,
) -> None:
    from esam3.peft_adapters.qlora import _infer_quant_type_from_wrapper

    fake_linear4bit = fake_bnb.nn.Linear4bit()  # no quant_type set anywhere
    model = nn.Sequential(fake_linear4bit)
    wrapper: Any = _FakeWrapper(model)
    with pytest.raises(RuntimeError, match="could not infer quant_type"):
        _infer_quant_type_from_wrapper(wrapper)


def test_has_plain_nn_linear_ignores_lora_adapter_children() -> None:
    """The tightened predicate must ignore lora_A/lora_B nn.Linears but flag base leaks."""
    from tests.helpers.lora_predicates import has_plain_nn_linear as _has_plain_nn_linear

    # Fake LoRA adapter wrapper: holds a Linear4bit-shape sentinel as base, plus
    # full-precision lora_A / lora_B nn.Linear adapters (mimicking peft.tuners.lora.bnb.Linear4bit).
    class _Linear4bitSentinel(nn.Linear):
        """Subclass of nn.Linear (mimics bnb.nn.Linear4bit subclassing)."""

    class _FakeLoraWrapper(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.base_layer = _Linear4bitSentinel(4, 4)
            self.lora_A = nn.ModuleDict({"default": nn.Linear(4, 2, bias=False)})
            self.lora_B = nn.ModuleDict({"default": nn.Linear(2, 4, bias=False)})

    # Case 1: all base Linears already swapped (only Linear4bitSentinel + lora adapters).
    # Expected: predicate returns False.
    clean = nn.Sequential(_FakeLoraWrapper(), _FakeLoraWrapper())
    assert not _has_plain_nn_linear(clean), (
        "predicate must not flag lora_A/lora_B adapter Linears as base leaks"
    )

    # Case 2: introduce a real base-Linear leak alongside the LoRA-wrapped layers.
    # Expected: predicate returns True (the leaked plain nn.Linear is NOT under a lora_* path).
    leaked = nn.Sequential(_FakeLoraWrapper(), nn.Linear(4, 4))
    assert _has_plain_nn_linear(leaked), "predicate must still flag a true base nn.Linear leak"
