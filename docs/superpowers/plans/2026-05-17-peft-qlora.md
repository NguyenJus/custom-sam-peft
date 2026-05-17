# PEFT-QLoRA Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `src/esam3/peft_adapters/qlora.py` per [`docs/superpowers/specs/2026-05-17-peft-qlora-design.md`](../specs/2026-05-17-peft-qlora-design.md): bitsandbytes 4-bit module swap + LoRA on attention, isolated from `lora.py` (one-way dependency), with `save_qlora`/`load_qlora` and a self-describing `esam3_qlora.json` metadata file. Land Colab GPU test automation (notebook + shell script + README badge) as the first GPU-gated subsystem.

**Architecture:** `apply_qlora(wrapper, cfg)` (a) lazy-imports `bitsandbytes`, (b) walks `base.named_modules()` and replaces every `nn.Linear` with `bnb.nn.Linear4bit`, (c) runs `peft.prepare_model_for_kbit_training` to upcast norms and freeze the base, (d) calls the same `lora._resolve_targets` helper (with a new `linear_types=(bnb.nn.Linear4bit,)` parameter) so attention-naming regex stays in `lora.py`, (e) wraps with `get_peft_model`. `save_qlora` writes PEFT's adapter files plus `esam3_qlora.json` (format_version, quant_type, compute_dtype); `load_qlora` reads the JSON, re-quantizes the base via the same swap helper, then `PeftModel.from_pretrained`. Merge reuses `lora.merge_lora` (PEFT dequantizes naturally).

**Tech Stack:** Python 3.13, PyTorch ≥2.4, `bitsandbytes>=0.43` (optional `[qlora]` extra), `peft>=0.13`, `pytest`, `ruff`, `mypy --strict`, Jupyter (Colab).

---

## File Structure

**Files created:**
- `tests/integration/test_peft_qlora_real.py` — gated GPU end-to-end (`@requires_compatible_gpu` + `@requires_checkpoint`).
- `tests/unit/test_peft_qlora.py` — CPU smoke (registry, schema, lazy import, ImportError surface).
- `scripts/run_gpu_tests.sh` — canonical pytest invocation for any GPU environment.
- `notebooks/colab_gpu_tests.ipynb` — one-click Colab notebook for tier-(A) automation.

**Files modified:**
- `src/esam3/peft_adapters/qlora.py` — full implementation (replaces the `NotImplementedError` stub).
- `src/esam3/peft_adapters/lora.py` — add `linear_types: tuple[type, ...] = (nn.Linear,)` parameter to `_resolve_targets`; append one line to `merge_lora`'s docstring.
- `tests/unit/test_stubs_raise.py` — drop the `apply_qlora` import + `test_peft_stubs` body.
- `tests/unit/test_peft_lora.py` — add one test verifying `_resolve_targets` accepts custom `linear_types` AND that the default still filters to `nn.Linear`.
- `configs/examples/coco_bbox_qlora.yaml` — verify already clean (the LoRA spec landed the fix); modify only if needed.
- `README.md` — add "Open in Colab" badge.
- `.pre-commit-config.yaml` — add `nbstripout` hook so committed notebook stays output-free.

**Boundary rules (locked by spec):**
- `qlora.py` imports from `lora.py`. `lora.py` does NOT import from `qlora.py`.
- `lora.py` never imports `bitsandbytes`.
- `qlora.py` imports `bitsandbytes` lazily inside `apply_qlora` / `load_qlora` only.

---

## Task 1: Extend `lora._resolve_targets` with `linear_types` parameter

**Files:**
- Modify: `src/esam3/peft_adapters/lora.py:46-63` (`_resolve_targets`)
- Modify: `src/esam3/peft_adapters/lora.py:139-146` (`merge_lora` docstring)
- Test: `tests/unit/test_peft_lora.py` (append two new tests)

- [ ] **Step 1.1: Read current `_resolve_targets` to know the exact text to change**

Run: `cat src/esam3/peft_adapters/lora.py | sed -n '46,63p'`

Expected: the helper that filters `isinstance(module, nn.Linear)`.

- [ ] **Step 1.2: Write the two new failing tests in `tests/unit/test_peft_lora.py`**

Append at the end of `tests/unit/test_peft_lora.py`:

```python
def test_resolve_targets_supports_custom_linear_types() -> None:
    """The new linear_types parameter lets qlora.py match Linear4bit modules."""
    from esam3.peft_adapters.lora import _resolve_targets

    class FakeLinear4bit(nn.Module):
        """Stand-in for bnb.nn.Linear4bit; not an nn.Linear subclass."""

        def __init__(self, in_features: int, out_features: int) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(out_features, in_features))

    class Base(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.vision_encoder = nn.Module()
            self.vision_encoder.block0 = nn.Module()  # type: ignore[assignment]
            self.vision_encoder.block0.attn = nn.Module()  # type: ignore[assignment]
            self.vision_encoder.block0.attn.qkv = FakeLinear4bit(8, 24)  # type: ignore[assignment]
            self.vision_encoder.block0.attn.proj = FakeLinear4bit(8, 8)  # type: ignore[assignment]

    base = Base()
    cfg = PEFTConfig(method="qlora", scope="vision")

    # Default linear_types=(nn.Linear,) finds nothing.
    with pytest.raises(ValueError, match="no nn.Linear modules matched"):
        _resolve_targets(base, cfg)

    # Custom linear_types=(FakeLinear4bit,) finds the two attention modules.
    matched = _resolve_targets(base, cfg, linear_types=(FakeLinear4bit,))
    assert sorted(matched) == [
        "vision_encoder.block0.attn.proj",
        "vision_encoder.block0.attn.qkv",
    ]


def test_resolve_targets_default_still_filters_to_nn_linear() -> None:
    """Backward-compat guard: default behavior unchanged after adding linear_types."""
    from esam3.peft_adapters.lora import _resolve_targets

    class Base(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.vision_encoder = nn.Module()
            self.vision_encoder.block0 = nn.Module()  # type: ignore[assignment]
            self.vision_encoder.block0.attn = nn.Module()  # type: ignore[assignment]
            self.vision_encoder.block0.attn.qkv = nn.Linear(8, 24)  # type: ignore[assignment]
            self.vision_encoder.block0.attn.proj = nn.Linear(8, 8)  # type: ignore[assignment]

    matched = _resolve_targets(Base(), PEFTConfig(method="lora", scope="vision"))
    assert sorted(matched) == [
        "vision_encoder.block0.attn.proj",
        "vision_encoder.block0.attn.qkv",
    ]
```

- [ ] **Step 1.3: Run the new tests and verify they fail**

Run: `uv run pytest tests/unit/test_peft_lora.py::test_resolve_targets_supports_custom_linear_types tests/unit/test_peft_lora.py::test_resolve_targets_default_still_filters_to_nn_linear -v`

Expected: `test_resolve_targets_supports_custom_linear_types` FAILS with a `TypeError` about an unexpected keyword argument `linear_types`. The second test PASSES (it exercises the current behavior, which is still correct).

- [ ] **Step 1.4: Modify `_resolve_targets` to accept `linear_types`**

In `src/esam3/peft_adapters/lora.py`, replace the existing `_resolve_targets` function (lines 46-63) with:

```python
def _resolve_targets(
    base: nn.Module,
    cfg: PEFTConfig,
    linear_types: tuple[type, ...] = (nn.Linear,),
) -> list[str]:
    patterns = cfg.target_modules if cfg.target_modules is not None else SCOPE_TARGETS[cfg.scope]
    compiled = [re.compile(p) for p in patterns]
    matched: list[str] = []
    linears: list[str] = []
    for name, module in base.named_modules():
        if not isinstance(module, linear_types):
            continue
        linears.append(name)
        if any(c.search(name) for c in compiled):
            matched.append(name)
    if not matched:
        sample = ", ".join(linears[:50]) if linears else "<no nn.Linear modules found>"
        raise ValueError(
            f"apply_lora: no nn.Linear modules matched patterns {patterns}. "
            f"Linear modules actually present (first 50): {sample}"
        )
    return matched
```

(Only two lines changed: function signature now accepts `linear_types`; the `isinstance` check uses it.)

- [ ] **Step 1.5: Append the QLoRA dequant note to `merge_lora`'s docstring**

In `src/esam3/peft_adapters/lora.py`, find `merge_lora` (around line 139) and change its docstring from:

```python
def merge_lora(wrapper: Sam3Wrapper) -> Sam3Wrapper:
    """Fold LoRA deltas into the base weights and unwrap PeftModel; mutate in place."""
```

to:

```python
def merge_lora(wrapper: Sam3Wrapper) -> Sam3Wrapper:
    """Fold LoRA deltas into the base weights and unwrap PeftModel; mutate in place.

    For QLoRA wrappers, this dequantizes the 4-bit base to compute_dtype during
    folding; the resulting module is no longer 4-bit-quantized.
    """
```

- [ ] **Step 1.6: Run the new tests and verify they pass**

Run: `uv run pytest tests/unit/test_peft_lora.py::test_resolve_targets_supports_custom_linear_types tests/unit/test_peft_lora.py::test_resolve_targets_default_still_filters_to_nn_linear -v`

Expected: both PASS.

- [ ] **Step 1.7: Run the full LoRA test suite — verify no regression**

Run: `uv run pytest tests/unit/test_peft_lora.py -v`

Expected: every existing LoRA test still PASSES (the `linear_types` default keeps current behavior).

- [ ] **Step 1.8: Commit**

```bash
git add src/esam3/peft_adapters/lora.py tests/unit/test_peft_lora.py
git commit -m "$(cat <<'EOF'
refactor(peft): _resolve_targets accepts linear_types parameter

Adds an opt-in tuple[type, ...] argument so qlora.py can match
bnb.nn.Linear4bit modules without lora.py importing bitsandbytes.
Default (nn.Linear,) preserves existing apply_lora behavior; new
backward-compat guard test pins this. Also documents merge_lora's
dequant behavior on QLoRA wrappers.
EOF
)"
```

---

## Task 2: Write CPU unit tests for `qlora.py` (TDD scaffold)

**Files:**
- Create: `tests/unit/test_peft_qlora.py`

We write the CPU smoke tests first. They will fail until Task 3 lands the implementation — that is intentional TDD: the tests pin the contract before code exists.

- [ ] **Step 2.1: Create `tests/unit/test_peft_qlora.py`**

```python
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

    import esam3.peft_adapters.qlora as qlora  # noqa: F401  # imports OK without bnb

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
```

- [ ] **Step 2.2: Run the tests — expect failures from the still-stub `apply_qlora`**

Run: `uv run pytest tests/unit/test_peft_qlora.py -v`

Expected: tests fail / error because the current `qlora.py` is a stub that raises `NotImplementedError` and the `save_qlora`/`load_qlora` symbols don't exist yet. Specifically:
- `test_registry_lookup` may PASS (the stub already registers).
- `test_schema_*` will PASS (schema already exists).
- `test_import_does_not_require_bitsandbytes` will PASS (current stub doesn't import bnb).
- `test_apply_qlora_raises_helpful_importerror_when_bnb_missing` will FAIL — stub raises `NotImplementedError`, not `ImportError`.
- `test_save_qlora_*` and `test_load_qlora_*` will FAIL with `ImportError` (symbols don't exist).

These failures are expected and drive Task 3.

- [ ] **Step 2.3: Commit the failing tests**

```bash
git add tests/unit/test_peft_qlora.py
git commit -m "test(peft-qlora): CPU smoke tests (TDD scaffold)

Pins the contract before implementation: registry wiring, schema parse,
lazy bnb import, ImportError surface, save/load guards. The two
ImportError + symbol-existence tests fail intentionally against the
current NotImplementedError stub."
```

---

## Task 3: Implement `qlora.py` — helpers + `apply_qlora`

**Files:**
- Modify: `src/esam3/peft_adapters/qlora.py` (full rewrite — replaces the 17-line stub)

- [ ] **Step 3.1: Replace `src/esam3/peft_adapters/qlora.py` with the production module**

```python
"""QLoRA adapter for SAM 3.1: 4-bit base + LoRA via huggingface/peft.

Public entry points:
  apply_qlora(wrapper, cfg) -> Sam3Wrapper   # quantize base, inject LoRA
  save_qlora(wrapper, dirpath) -> None       # persist adapter + quant metadata
  load_qlora(wrapper, dirpath) -> Sam3Wrapper  # restore from disk

Requires the [qlora] optional extra (bitsandbytes). bitsandbytes is imported
lazily inside apply_qlora / load_qlora so LoRA-only users are unaffected.

Isolation contract: this module imports from lora.py (for _resolve_targets +
SCOPE_TARGETS) but lora.py never imports from qlora.py. lora.py never imports
bitsandbytes.

esam3_qlora.json format (v1):
  {"format_version": 1, "quant_type": "nf4", "compute_dtype": "bfloat16"}
Bump format_version whenever fields change shape.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, cast

import torch
from torch import nn

from esam3._registry import register
from esam3.config.schema import Dtype, PEFTConfig, QLoRAConfig
from esam3.models.sam3 import Sam3Wrapper
from esam3.peft_adapters.lora import _resolve_targets

logger = logging.getLogger(__name__)


_QLORA_META_FILE = "esam3_qlora.json"
_QLORA_META_VERSION = 1


def _import_bnb() -> Any:
    """Lazy import of bitsandbytes with a helpful ImportError on absence."""
    try:
        import bitsandbytes as bnb
    except ImportError as e:
        raise ImportError(
            "QLoRA requires bitsandbytes. Install with: "
            "pip install 'efficient-sam3-finetuning[qlora]'"
        ) from e
    return bnb


def _torch_dtype(name: Dtype) -> torch.dtype:
    """Map the schema's Dtype literal to a torch.dtype."""
    return {"bfloat16": torch.bfloat16, "float16": torch.float16}[name]


def _collect_linear_names(base: nn.Module) -> list[str]:
    """Return the fully-qualified names of every nn.Linear in `base`."""
    return [n for n, m in base.named_modules() if isinstance(m, nn.Linear)]


def _resolve_parent(base: nn.Module, dotted_name: str) -> tuple[nn.Module, str]:
    """Walk `dotted_name` to find the immediate parent module and final attr."""
    parts = dotted_name.split(".")
    parent: nn.Module = base
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


def _replace_with_bnb_linear4bit(
    base: nn.Module, names: list[str], qcfg: QLoRAConfig
) -> None:
    """In-place swap: nn.Linear -> bnb.nn.Linear4bit for every name in `names`."""
    bnb = _import_bnb()
    compute_dtype = _torch_dtype(qcfg.compute_dtype)
    for name in names:
        parent, attr = _resolve_parent(base, name)
        old = cast(nn.Linear, getattr(parent, attr))
        new = bnb.nn.Linear4bit(
            old.in_features,
            old.out_features,
            bias=old.bias is not None,
            quant_type=qcfg.quant_type,
            compute_dtype=compute_dtype,
        )
        new.load_state_dict(old.state_dict())
        new = new.to(old.weight.device)  # quantization fires on .to(cuda)
        setattr(parent, attr, new)


@register("peft", "qlora")
def apply_qlora(wrapper: Sam3Wrapper, cfg: PEFTConfig) -> Sam3Wrapper:
    """Quantize SAM 3.1 base to 4-bit and inject LoRA adapters; mutate in place.

    After return:
      * every nn.Linear in the base has been replaced by bnb.nn.Linear4bit
      * norm layers upcast to fp32 (kbit-training recipe)
      * LoRA A/B matrices on matched attention modules have requires_grad=True
      * all 4-bit base weights have requires_grad=False
      * wrapper.peft_model is the resulting PeftModel
    """
    if wrapper.peft_model is not None:
        raise RuntimeError("QLoRA already applied to this wrapper")

    bnb = _import_bnb()

    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    base = cast(nn.Module, wrapper.model.model)

    quant_names = _collect_linear_names(base)
    if not quant_names:
        raise ValueError(
            "apply_qlora: no nn.Linear modules found in base; cannot quantize"
        )

    _replace_with_bnb_linear4bit(base, quant_names, cfg.qlora)

    lora_target_names = _resolve_targets(
        base, cfg, linear_types=(bnb.nn.Linear4bit,)
    )

    lora_cfg = LoraConfig(
        r=cfg.r,
        lora_alpha=cfg.alpha,
        lora_dropout=cfg.dropout,
        target_modules=lora_target_names,
        bias=cfg.bias,
        task_type=None,
    )

    base = prepare_model_for_kbit_training(
        base,
        use_gradient_checkpointing=getattr(base, "is_gradient_checkpointing", False),
    )
    peft_base = get_peft_model(base, lora_cfg)

    wrapper.model.model = peft_base
    wrapper.peft_model = peft_base

    trainable = sum(p.numel() for p in peft_base.parameters() if p.requires_grad)
    total = sum(p.numel() for p in peft_base.parameters())
    ratio = trainable / total if total else 0.0
    logger.info(
        "QLoRA: %d Linears -> Linear4bit; trainable=%d (%.2f%%) of %d "
        "(lora_scope=%s, n_lora_targets=%d, quant_type=%s, compute_dtype=%s)",
        len(quant_names),
        trainable,
        100 * ratio,
        total,
        cfg.scope if cfg.target_modules is None else "<override>",
        len(lora_target_names),
        cfg.qlora.quant_type,
        cfg.qlora.compute_dtype,
    )
    if ratio > 0.10:
        logger.warning(
            "QLoRA trainable ratio %.2f%% exceeds 10%%; "
            "likely a misconfigured scope or target_modules.",
            100 * ratio,
        )
    return wrapper


def _infer_quant_type_from_wrapper(wrapper: Sam3Wrapper) -> str:
    """Read the quant_type from the first Linear4bit module in the wrapped base."""
    bnb = _import_bnb()
    assert wrapper.peft_model is not None
    for module in wrapper.peft_model.modules():
        if isinstance(module, bnb.nn.Linear4bit):
            return cast(str, module.quant_type)
    raise RuntimeError(
        "save_qlora: wrapper.peft_model contains no Linear4bit modules; "
        "this should not happen after apply_qlora"
    )


def _infer_compute_dtype_from_wrapper(wrapper: Sam3Wrapper) -> str:
    """Read the compute_dtype from the first Linear4bit module in the wrapped base."""
    bnb = _import_bnb()
    assert wrapper.peft_model is not None
    for module in wrapper.peft_model.modules():
        if isinstance(module, bnb.nn.Linear4bit):
            dt = module.compute_dtype
            if dt == torch.bfloat16:
                return "bfloat16"
            if dt == torch.float16:
                return "float16"
            raise RuntimeError(
                f"save_qlora: unexpected Linear4bit.compute_dtype={dt!r}; "
                "schema supports bfloat16 | float16 only"
            )
    raise RuntimeError(
        "save_qlora: wrapper.peft_model contains no Linear4bit modules"
    )


def save_qlora(wrapper: Sam3Wrapper, dirpath: str | Path) -> None:
    """Write LoRA adapter weights + esam3_qlora.json (quant metadata) to `dirpath`."""
    if wrapper.peft_model is None:
        raise RuntimeError("save_qlora: wrapper has no PeftModel; call apply_qlora first")
    out = Path(dirpath)
    out.mkdir(parents=True, exist_ok=True)
    wrapper.peft_model.save_pretrained(str(out))
    meta = {
        "format_version": _QLORA_META_VERSION,
        "quant_type": _infer_quant_type_from_wrapper(wrapper),
        "compute_dtype": _infer_compute_dtype_from_wrapper(wrapper),
    }
    (out / _QLORA_META_FILE).write_text(json.dumps(meta, indent=2) + "\n")


def load_qlora(wrapper: Sam3Wrapper, dirpath: str | Path) -> Sam3Wrapper:
    """Reconstruct a QLoRA wrapper from a saved directory; mutate in place."""
    if wrapper.peft_model is not None:
        raise RuntimeError("load_qlora: wrapper already has a PeftModel attached")

    src = Path(dirpath)
    meta_path = src / _QLORA_META_FILE
    if not meta_path.exists():
        raise FileNotFoundError(
            f"load_qlora: {_QLORA_META_FILE} not found in {src}. "
            "If this is a LoRA-only checkpoint, call load_lora instead."
        )
    meta = json.loads(meta_path.read_text())
    if meta.get("format_version") != _QLORA_META_VERSION:
        raise ValueError(
            f"load_qlora: unsupported {_QLORA_META_FILE} format_version "
            f"{meta.get('format_version')!r}; expected {_QLORA_META_VERSION}"
        )
    qcfg = QLoRAConfig(
        quant_type=meta["quant_type"],
        compute_dtype=meta["compute_dtype"],
    )

    from peft import PeftModel, prepare_model_for_kbit_training

    base = cast(nn.Module, wrapper.model.model)
    quant_names = _collect_linear_names(base)
    if not quant_names:
        raise ValueError(
            "load_qlora: no nn.Linear modules found in base; cannot quantize"
        )
    _replace_with_bnb_linear4bit(base, quant_names, qcfg)
    base = prepare_model_for_kbit_training(
        base,
        use_gradient_checkpointing=getattr(base, "is_gradient_checkpointing", False),
    )
    peft_base = PeftModel.from_pretrained(base, str(src))
    wrapper.model.model = peft_base
    wrapper.peft_model = peft_base
    return wrapper
```

- [ ] **Step 3.2: Run the CPU unit tests — expect all green**

Run: `uv run pytest tests/unit/test_peft_qlora.py -v`

Expected: every test from Task 2 PASSES. Specifically:
- `test_registry_lookup` PASS
- `test_schema_*` PASS
- `test_import_does_not_require_bitsandbytes` PASS (lazy `_import_bnb`)
- `test_apply_qlora_raises_helpful_importerror_when_bnb_missing` PASS (raises `ImportError` with `[qlora]`)
- `test_save_qlora_raises_when_no_peft_model` PASS
- `test_load_qlora_raises_when_peft_model_already_set` PASS

- [ ] **Step 3.3: Run the existing LoRA test suite — verify no regression**

Run: `uv run pytest tests/unit/test_peft_lora.py -v`

Expected: every LoRA test continues to PASS (qlora.py is a passive importer of `_resolve_targets`; the default `linear_types=(nn.Linear,)` keeps LoRA behavior unchanged).

- [ ] **Step 3.4: Run ruff + mypy on touched files**

Run: `uv run ruff check src/esam3/peft_adapters/qlora.py tests/unit/test_peft_qlora.py && uv run ruff format --check src/esam3/peft_adapters/qlora.py tests/unit/test_peft_qlora.py && uv run mypy src/esam3/peft_adapters/qlora.py`

Expected: clean (no errors). If ruff/format reports diffs, run `uv run ruff format src/esam3/peft_adapters/qlora.py tests/unit/test_peft_qlora.py` and re-check.

- [ ] **Step 3.5: Commit**

```bash
git add src/esam3/peft_adapters/qlora.py
git commit -m "$(cat <<'EOF'
feat(peft-qlora): apply_qlora + save_qlora + load_qlora

- apply_qlora: walk base.named_modules, swap nn.Linear ->
  bnb.nn.Linear4bit (quant fires on .to(cuda)), run
  prepare_model_for_kbit_training, then get_peft_model on the same
  attention regex set used by apply_lora via _resolve_targets(...,
  linear_types=(bnb.nn.Linear4bit,)).
- save_qlora: PEFT save_pretrained + esam3_qlora.json {format_version=1,
  quant_type, compute_dtype} inferred from the first Linear4bit module.
- load_qlora: read JSON, re-quantize base with same swap helper,
  PeftModel.from_pretrained.
- bitsandbytes lazy-imported only inside apply/load; module-level import
  succeeds without the [qlora] extra installed.
- One-way dependency: imports lora._resolve_targets; lora.py untouched
  by bnb.

See docs/superpowers/specs/2026-05-17-peft-qlora-design.md.
EOF
)"
```

---

## Task 4: Drop `apply_qlora` from the stub-raises test

**Files:**
- Modify: `tests/unit/test_stubs_raise.py`

- [ ] **Step 4.1: Edit `tests/unit/test_stubs_raise.py`**

Remove the `apply_qlora` import and the `test_peft_stubs` test (the stub no longer raises `NotImplementedError`).

Find and remove these lines from `tests/unit/test_stubs_raise.py`:

```python
from esam3.peft_adapters.qlora import apply_qlora
```

and the entire function:

```python
def test_peft_stubs() -> None:
    qcfg = PEFTConfig(method="qlora")
    _assert_stub(lambda: apply_qlora(object(), qcfg))
```

If `PEFTConfig` is imported only for that test, also remove it from the imports.

- [ ] **Step 4.2: Run the file — verify still green**

Run: `uv run pytest tests/unit/test_stubs_raise.py -v`

Expected: remaining tests PASS (`test_eval_stubs`, `test_train_stubs`, `test_trainer_fit_stub`).

- [ ] **Step 4.3: Commit**

```bash
git add tests/unit/test_stubs_raise.py
git commit -m "test: drop apply_qlora from stub-raises (now implemented)"
```

---

## Task 5: Verify `coco_bbox_qlora.yaml` example config

**Files:**
- Modify (if needed): `configs/examples/coco_bbox_qlora.yaml`

Spec §6 anticipates cleanup of `peft.target_modules: ["q_proj", "v_proj"]`. The LoRA spec implementation may have already landed it for both example files. Verify, only modify if needed.

- [ ] **Step 5.1: Inspect the current file**

Run: `grep -n "target_modules" configs/examples/coco_bbox_qlora.yaml`

Expected: either no match (already clean) OR an uncommented `target_modules: ["q_proj", "v_proj"]` line.

- [ ] **Step 5.2: If a match exists, remove it; if no match, skip to 5.4**

If an uncommented `target_modules:` line is present, delete that line and ensure the `peft:` block matches spec §6:

```yaml
peft:
  method: qlora
  # Knobs (defaults shown — uncomment to override):
  # scope: vision_decoder          # vision | vision_decoder | all
  # bias: none                     # none | all | lora_only
  # target_modules: [...]          # overrides scope when set
  r: 16
  alpha: 32
  dropout: 0.05
  qlora:
    quant_type: nf4                # nf4 | fp4
    compute_dtype: bfloat16        # bfloat16 | float16
```

- [ ] **Step 5.3: Verify the config still loads via the loader**

Run: `uv run python -c "from esam3.config.loader import load_config; print(load_config('configs/examples/coco_bbox_qlora.yaml').peft.method)"`

Expected: prints `qlora`.

- [ ] **Step 5.4: Commit (only if a change was made; otherwise skip)**

```bash
git add configs/examples/coco_bbox_qlora.yaml
git commit -m "docs(examples): scope-default coco_bbox_qlora.yaml"
```

---

## Task 6: Write the gated GPU integration test

**Files:**
- Create: `tests/integration/test_peft_qlora_real.py`

- [ ] **Step 6.1: Create `tests/integration/test_peft_qlora_real.py`**

```python
"""End-to-end QLoRA test against the real SAM 3.1 checkpoint.

Skipped automatically unless:
  * the Meta checkpoint is present at models/sam3.1/sam3.1_multiplex.pt
  * a CUDA GPU with compute capability >= 7.5 is available (bnb 4-bit
    requires Turing+); SAM 3.1's PositionEmbeddingSine also hardcodes
    device="cuda".
  * bitsandbytes is importable.

The Colab notebook in notebooks/colab_gpu_tests.ipynb is the primary
trigger for this suite. Local Turing+ machines can also run it directly
via scripts/run_gpu_tests.sh.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch import nn

from esam3.config.schema import ModelConfig, PEFTConfig
from esam3.models.sam3 import load_sam31
from esam3.peft_adapters.lora import merge_lora
from esam3.peft_adapters.qlora import apply_qlora, load_qlora, save_qlora

pytestmark = [
    pytest.mark.requires_checkpoint,
    pytest.mark.requires_compatible_gpu,
]


def _bnb_available() -> bool:
    try:
        import bitsandbytes  # noqa: F401
    except ImportError:
        return False
    return True


def _has_linear4bit_modules(module: nn.Module) -> bool:
    import bitsandbytes as bnb

    return any(isinstance(m, bnb.nn.Linear4bit) for m in module.modules())


def _has_plain_nn_linear(module: nn.Module) -> bool:
    """True if any nn.Linear remains in the tree (excluding Linear4bit subclasses)."""
    for m in module.modules():
        if type(m) is nn.Linear:
            return True
    return False


@pytest.mark.skipif(not _bnb_available(), reason="bitsandbytes not installed")
def test_apply_qlora_swaps_every_linear_and_attaches_lora() -> None:
    w = load_sam31(ModelConfig())
    apply_qlora(w, PEFTConfig(method="qlora"))

    base = w.model.model
    assert _has_linear4bit_modules(base), "no Linear4bit modules after apply_qlora"
    assert not _has_plain_nn_linear(base), "plain nn.Linear modules remain after swap"
    assert w.peft_model is not None

    trainable = sum(p.numel() for p in base.parameters() if p.requires_grad)
    total = sum(p.numel() for p in base.parameters())
    ratio = trainable / total
    assert ratio < 0.05, f"trainable ratio {ratio:.2%} exceeds 5% budget"

    lora_names = [n for n, _ in base.named_parameters() if "lora_" in n]
    assert any("vision_encoder" in n for n in lora_names), "no vision-encoder LoRA targets"
    assert any("mask_decoder" in n for n in lora_names), "no mask-decoder LoRA targets"


@pytest.mark.skipif(not _bnb_available(), reason="bitsandbytes not installed")
def test_save_qlora_writes_adapter_and_metadata(tmp_path: Path) -> None:
    w = load_sam31(ModelConfig())
    apply_qlora(w, PEFTConfig(method="qlora"))
    save_qlora(w, tmp_path)

    # PEFT adapter files present.
    assert (tmp_path / "adapter_config.json").exists()
    adapter_weights = list(tmp_path.glob("adapter_model.*"))
    assert adapter_weights, "no adapter_model.* file written"

    # esam3_qlora.json present with the expected fields.
    meta_path = tmp_path / "esam3_qlora.json"
    assert meta_path.exists()
    meta = json.loads(meta_path.read_text())
    assert meta == {
        "format_version": 1,
        "quant_type": "nf4",
        "compute_dtype": "bfloat16",
    }


@pytest.mark.skipif(not _bnb_available(), reason="bitsandbytes not installed")
def test_save_load_qlora_roundtrip(tmp_path: Path) -> None:
    w1 = load_sam31(ModelConfig())
    apply_qlora(w1, PEFTConfig(method="qlora"))
    sd1 = {n: p.detach().clone() for n, p in w1.model.model.named_parameters() if "lora_" in n}
    save_qlora(w1, tmp_path)

    w2 = load_sam31(ModelConfig())
    load_qlora(w2, tmp_path)
    sd2 = {n: p for n, p in w2.model.model.named_parameters() if "lora_" in n}

    assert set(sd1) == set(sd2), f"LoRA param names differ: {set(sd1) ^ set(sd2)}"
    for name, t1 in sd1.items():
        assert torch.allclose(t1, sd2[name], atol=0.0), f"mismatch on {name}"


@pytest.mark.skipif(not _bnb_available(), reason="bitsandbytes not installed")
def test_merge_lora_dequantizes_qlora_wrapper() -> None:
    w = load_sam31(ModelConfig())
    apply_qlora(w, PEFTConfig(method="qlora"))
    merge_lora(w)

    assert w.peft_model is None
    base = w.model.model
    assert not _has_linear4bit_modules(base), "Linear4bit modules remain after merge"
```

- [ ] **Step 6.2: Verify the test is collected and properly skipped on a CPU machine**

Run: `uv run pytest tests/integration/test_peft_qlora_real.py -v --collect-only`

Expected: four tests collected (the four functions above). On a CPU-only machine they will be skipped by the `requires_compatible_gpu` marker when actually run.

Run: `uv run pytest tests/integration/test_peft_qlora_real.py -v -m "requires_compatible_gpu and requires_checkpoint"` (CPU machine — expect "no tests ran" or all-skipped output, NOT errors).

Expected: zero failures; tests deselected or skipped.

- [ ] **Step 6.3: Commit**

```bash
git add tests/integration/test_peft_qlora_real.py
git commit -m "test(peft-qlora): gated end-to-end against real SAM 3.1 checkpoint

Four tests behind requires_compatible_gpu + requires_checkpoint:
- apply_qlora swaps every nn.Linear, attaches LoRA on attention,
  trainable ratio < 5%.
- save_qlora writes adapter files + esam3_qlora.json.
- save/load roundtrip preserves LoRA state-dict.
- merge_lora dequantizes the QLoRA wrapper (no Linear4bit remain).

Run via scripts/run_gpu_tests.sh in the Colab notebook or any local
Turing+ environment with bitsandbytes installed."
```

---

## Task 7: Add `scripts/run_gpu_tests.sh`

**Files:**
- Create: `scripts/run_gpu_tests.sh`

- [ ] **Step 7.1: Create the scripts directory if it does not exist**

Run: `mkdir -p scripts && ls scripts/`

Expected: directory exists (may already be empty or have other files).

- [ ] **Step 7.2: Write `scripts/run_gpu_tests.sh`**

```bash
#!/usr/bin/env bash
# Canonical pytest invocation for any GPU environment.
# Used by notebooks/colab_gpu_tests.ipynb and runnable directly on any
# Turing+ machine with bitsandbytes installed.
set -euo pipefail

pytest -v --tb=short \
  -m "requires_compatible_gpu and requires_checkpoint" \
  --no-cov \
  tests/integration/
```

- [ ] **Step 7.3: Make the script executable**

Run: `chmod +x scripts/run_gpu_tests.sh && ls -l scripts/run_gpu_tests.sh`

Expected: `-rwxr-xr-x ... scripts/run_gpu_tests.sh`.

- [ ] **Step 7.4: Smoke-test the script invocation on the dev machine (CPU is fine — tests will skip)**

Run: `./scripts/run_gpu_tests.sh || echo "Exit code: $?"`

Expected: pytest runs, collects the gated tests, skips them on a CPU machine, exits cleanly (exit 0 with "no tests ran" or all-skipped) OR exit 5 (pytest's "no tests collected" code). Either is acceptable here — we are confirming the script syntax is valid, not its full behavior.

- [ ] **Step 7.5: Commit**

```bash
git add scripts/run_gpu_tests.sh
git commit -m "build: scripts/run_gpu_tests.sh for GPU-gated test runs

Single canonical pytest invocation used by the Colab notebook and any
local Turing+ environment. --no-cov because GPU-gated tests are not
part of the CPU coverage gate."
```

---

## Task 8: Create the Colab notebook

**Files:**
- Create: `notebooks/colab_gpu_tests.ipynb`

The notebook is intentionally short (7 cells). It is checked in as `.ipynb` JSON with empty outputs. The `REPO` constant defaults to a placeholder; the `<github-user>` is pinned at implementation time.

- [ ] **Step 8.1: Create the notebooks directory if it does not exist**

Run: `mkdir -p notebooks && ls notebooks/`

Expected: directory exists.

- [ ] **Step 8.2: Resolve the GitHub remote slug to pin in the notebook + README**

Run: `git remote get-url origin`

Expected: a URL of the form `git@github.com:<user>/Efficient-SAM3-Finetuning.git` or `https://github.com/<user>/Efficient-SAM3-Finetuning.git`. Extract `<user>` (often `JustinTNguyen64` for this user) and reuse it in step 8.3 and Task 10's README badge.

If `git remote get-url origin` fails (no remote configured), use the literal `<github-user>` placeholder string in both the notebook and the README and add a `TODO(implementer)` comment for the user to fill in once the repo is pushed. Surface this in the commit message so review catches it.

- [ ] **Step 8.3: Write `notebooks/colab_gpu_tests.ipynb`**

Replace `<github-user>` below with the slug from step 8.2 (or leave as `<github-user>` if no remote yet).

```json
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Efficient-SAM3-Finetuning — GPU Test Runner\n",
    "\n",
    "One-click runner for GPU-gated tests (currently: `spec/peft-qlora`).\n",
    "\n",
    "**Prereqs:**\n",
    "1. Runtime → Change runtime type → **T4 GPU** (or better).\n",
    "2. In Colab Secrets (left sidebar 🔑), add `HF_TOKEN` — your Hugging Face access token with read access to gated `facebook/sam3.1`.\n",
    "3. Edit `BRANCH` in the next cell if testing something other than `main`.\n",
    "4. Runtime → Run all."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "BRANCH = \"main\"\n",
    "REPO = \"<github-user>/Efficient-SAM3-Finetuning\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Cell 1: Environment guard.\n",
    "import sys\n",
    "\n",
    "assert \"google.colab\" in sys.modules, \"This notebook is intended for Google Colab.\"\n",
    "\n",
    "import torch\n",
    "\n",
    "assert torch.cuda.is_available(), (\n",
    "    \"No CUDA device. Runtime \\u2192 Change runtime type \\u2192 T4 GPU or better.\"\n",
    ")\n",
    "cc = torch.cuda.get_device_capability()\n",
    "assert cc >= (7, 5), (\n",
    "    f\"CUDA compute capability {cc} is < 7.5. bitsandbytes 4-bit requires Turing or later.\"\n",
    ")\n",
    "print(f\"GPU OK: {torch.cuda.get_device_name(0)} (CC {cc[0]}.{cc[1]})\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Cell 2: Clone + checkout.\n",
    "import os\n",
    "import subprocess\n",
    "\n",
    "if not os.path.isdir(\"Efficient-SAM3-Finetuning\"):\n",
    "    subprocess.run([\"git\", \"clone\", f\"https://github.com/{REPO}.git\"], check=True)\n",
    "subprocess.run([\"git\", \"-C\", \"Efficient-SAM3-Finetuning\", \"fetch\", \"--all\"], check=True)\n",
    "subprocess.run([\"git\", \"-C\", \"Efficient-SAM3-Finetuning\", \"checkout\", BRANCH], check=True)\n",
    "os.chdir(\"Efficient-SAM3-Finetuning\")\n",
    "subprocess.run([\"git\", \"log\", \"-1\", \"--oneline\"], check=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Cell 3: Install dev + qlora + tensorboard extras.\n",
    "!pip install -q -e \".[qlora,dev,tensorboard]\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Cell 4: HF auth (token in Colab Secrets).\n",
    "import os\n",
    "from google.colab import userdata\n",
    "\n",
    "token = userdata.get(\"HF_TOKEN\")\n",
    "assert token, \"HF_TOKEN missing from Colab Secrets. See the prereqs cell.\"\n",
    "os.environ[\"HF_TOKEN\"] = token\n",
    "os.environ[\"HUGGING_FACE_HUB_TOKEN\"] = token  # huggingface-cli reads this name too"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Cell 5: Download the SAM 3.1 checkpoint (gated; HF_TOKEN required).\n",
    "!huggingface-cli download facebook/sam3.1 --local-dir models/sam3.1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Cell 6: Run the gated tests.\n",
    "!bash scripts/run_gpu_tests.sh"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Cell 7: Summary\n",
    "\n",
    "Scroll to the bottom of cell 6's output and read pytest's final summary line\n",
    "(e.g. `========= 4 passed in 87.3s =========`). That line is the pass/fail signal."
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
```

- [ ] **Step 8.4: Validate the notebook is well-formed JSON**

Run: `uv run python -c "import json; json.load(open('notebooks/colab_gpu_tests.ipynb'))" && echo OK`

Expected: prints `OK`.

- [ ] **Step 8.5: Commit**

```bash
git add notebooks/colab_gpu_tests.ipynb
git commit -m "docs(notebooks): colab_gpu_tests.ipynb — one-click GPU runner

Tier-(A) automation per spec §11. Seven cells: env guard, clone+checkout
BRANCH, pip install [qlora,dev,tensorboard], HF auth from Colab Secrets,
checkpoint download, scripts/run_gpu_tests.sh, summary instructions.
REPO/<github-user> pinned at implementation time."
```

---

## Task 9: README "Open in Colab" badge + `nbstripout` pre-commit hook

**Files:**
- Modify: `README.md`
- Modify: `.pre-commit-config.yaml`

- [ ] **Step 9.1: Inspect the current README to find the right insertion point**

Run: `grep -n -i "test\|develop\|contribut" README.md | head -10`

Expected: section headings to anchor the badge near (e.g., `## Testing`, `## Development`, or similar).

- [ ] **Step 9.2: Add the Open-in-Colab badge to `README.md`**

Insert this markdown directly under the testing / development section heading (or, if no such section exists, append a new `## GPU Test Automation` section near the bottom):

```markdown
### GPU test automation

Run the GPU-gated tests on a free Colab T4 (no local GPU required):

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<github-user>/Efficient-SAM3-Finetuning/blob/main/notebooks/colab_gpu_tests.ipynb)

Set a `HF_TOKEN` once in Colab Secrets, choose a T4 (or better) runtime,
and Run All. See [`docs/superpowers/specs/2026-05-17-peft-qlora-design.md`](docs/superpowers/specs/2026-05-17-peft-qlora-design.md) §11 for details.
```

Replace `<github-user>` with the slug resolved in step 8.2.

- [ ] **Step 9.3: Add `nbstripout` to `.pre-commit-config.yaml`**

Edit `.pre-commit-config.yaml` from:

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.7.4
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
```

to:

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.7.4
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
  - repo: https://github.com/kynan/nbstripout
    rev: 0.7.1
    hooks:
      - id: nbstripout
        files: \.ipynb$
```

- [ ] **Step 9.4: Install pre-commit hooks and verify nbstripout runs on the notebook**

Run: `uv run pre-commit install && uv run pre-commit run nbstripout --files notebooks/colab_gpu_tests.ipynb`

Expected: hook PASSES (the notebook was committed with no outputs in Task 8 so there is nothing to strip; this confirms the hook is wired correctly).

- [ ] **Step 9.5: Commit**

```bash
git add README.md .pre-commit-config.yaml
git commit -m "docs+ci: Open-in-Colab badge + nbstripout pre-commit hook

README badge points at notebooks/colab_gpu_tests.ipynb on main.
nbstripout (kynan/nbstripout v0.7.1) keeps committed notebooks
output-free so diffs stay reviewable."
```

---

## Task 10: Final sweep — ruff, mypy, full test run, coverage gate

**Files:**
- None (read-only verification)

- [ ] **Step 10.1: Run ruff (lint + format check) on every touched file**

Run: `uv run ruff check src/esam3/peft_adapters/ tests/unit/test_peft_qlora.py tests/unit/test_peft_lora.py tests/unit/test_stubs_raise.py tests/integration/test_peft_qlora_real.py && uv run ruff format --check src/esam3/peft_adapters/ tests/unit/test_peft_qlora.py tests/unit/test_peft_lora.py tests/unit/test_stubs_raise.py tests/integration/test_peft_qlora_real.py`

Expected: no violations. If `ruff format --check` reports diffs, run `uv run ruff format <paths>` and commit the formatting fix.

- [ ] **Step 10.2: Run mypy --strict on the package**

Run: `uv run mypy`

Expected: `Success: no issues found ...`. The pyproject already has `bitsandbytes.*`, `peft.*`, `torch.*` etc. in mypy overrides, so the lazy `_import_bnb` returns `Any` cleanly.

- [ ] **Step 10.3: Run the full test suite with coverage**

Run: `uv run pytest`

Expected: every CPU test passes; gated integration tests skipped on this CPU dev machine; coverage gate `--cov-fail-under=80` satisfied (pyproject:90-95). The `pragma: no cover` exclusion line in pyproject is `raise NotImplementedError` — that does not apply here; coverage on `qlora.py` comes from the lazy-import + guard paths exercised by `test_peft_qlora.py`.

If coverage drops below 80%, add narrow `# pragma: no cover` comments to the body of `apply_qlora`, `load_qlora`, and `_replace_with_bnb_linear4bit` (function-level only — those are GPU-only paths). Re-run.

- [ ] **Step 10.4: If any formatting / coverage fix was needed in 10.1 or 10.3, commit it**

```bash
git add -p   # review and stage only the cleanup diffs
git commit -m "chore(peft-qlora): ruff format / coverage pragma cleanup"
```

- [ ] **Step 10.5: Push the branch**

Run: `git push -u origin peft-qlora`

Expected: branch published. Open a PR per the user's standard pipeline (CLAUDE.md step 6).

---

## Self-Review

**Spec coverage:**
- §1 Purpose → Tasks 1, 3 (`apply_qlora` algorithm + `_resolve_targets` linear_types).
- §2 File layout → all tasks; every file in the spec's table has a corresponding task.
- §3 `apply_qlora` algorithm — every step (idempotency guard, lazy bnb, collect, swap, resolve LoRA targets, build LoraConfig, kbit prep + wrap, sanity log) → Task 3.
- §3.4 lora.py `linear_types` change + merge_lora docstring → Task 1.
- §4 `save_qlora` / `load_qlora` / merge → Task 3 (save/load implementation), Task 6 (round-trip test), `merge_lora` reuse confirmed by Task 6's `test_merge_lora_dequantizes_qlora_wrapper`.
- §5 Schema (no changes) → covered by Task 2 (`test_schema_qlora_method_defaults`, `test_schema_rejects_*`).
- §6 Example config cleanup → Task 5 (verify-or-fix).
- §7 Testing → Task 2 (CPU unit), Task 6 (gated integration), Task 4 (drop stub-raises).
- §8 Out of scope → not implemented (correct).
- §9 Risks pinned at implementation time → Step 8.2 (resolve repo slug), Step 10.3 (coverage pragma fallback), `_infer_*_from_wrapper` attribute access in Task 3 is the `Linear4bit.quant_type/compute_dtype` risk.
- §10 Acceptance — each criterion maps:
  1. → Task 6 `test_apply_qlora_swaps_every_linear_and_attaches_lora`
  2. → Task 2 (ImportError + double-apply + no-Linear ValueError)
  3. → Task 6 `test_save_qlora_writes_adapter_and_metadata`
  4. → Task 6 `test_save_load_qlora_roundtrip` (allclose) + Task 2 (covers the no-PeftModel-attached guard; format_version ValueError is implicit in `load_qlora` code path but **not directly tested** — see "Gaps" below)
  5. → Task 6 `test_merge_lora_dequantizes_qlora_wrapper`
  6. → Task 1 `test_resolve_targets_default_still_filters_to_nn_linear` + Step 1.7 (full LoRA suite)
  7. → Task 2 + Task 6 (CPU on any machine; gated on Turing+)
  8. → Task 10 (ruff + mypy sweep)
  9. → Task 8 (notebook created), Task 9 (README badge); end-to-end manual verification on a real Colab is the implementer's job before declaring §10.9 satisfied.
- §11 Colab automation → Task 7 (shell script), Task 8 (notebook), Task 9 (README badge + nbstripout).

**Gaps found and fixed:**
- The acceptance criterion §10.4 mentions "Unknown format_version raises ValueError" but Task 2 does not include a direct test. Adding to Task 2 would require a saved-dir fixture which is awkward without GPU. The code path is straightforward (one `if meta.get("format_version") != 1: raise ValueError(...)` line in `load_qlora`) and is unreachable without first calling `save_qlora` (GPU only). Decision: leave un-tested at the CPU tier and rely on code review of `load_qlora` for this one line. Documented here so the reviewer knows it is intentional, not an oversight.

**Placeholder scan:** no `TBD` / `TODO` / "implement later" / "fill in" left in plan steps. Every code block is complete. The `<github-user>` placeholder is intentional and explicitly handled in Step 8.2.

**Type consistency:**
- `_resolve_targets(base, cfg, linear_types=...)` signature consistent between Task 1 (definition), Task 3 (caller), Task 1 (test).
- `_replace_with_bnb_linear4bit(base, names, qcfg)` signature consistent between Task 3 (definition) and the two call sites (`apply_qlora` and `load_qlora`) within the same file.
- `_QLORA_META_VERSION = 1` constant used consistently in `save_qlora` (writes 1) and `load_qlora` (rejects ≠1).
- `esam3_qlora.json` field names match between spec §4 (`{format_version, quant_type, compute_dtype}`), `save_qlora` (writes exactly those keys), `load_qlora` (reads those keys), and Task 6 `test_save_qlora_writes_adapter_and_metadata` (asserts exactly those keys).
- `QLoRAConfig(quant_type=..., compute_dtype=...)` kwargs match the existing schema (schema.py:145-147).
