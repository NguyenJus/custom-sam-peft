# PEFT-QLoRA Design (spec/peft-qlora)

**Status:** approved (brainstorming)
**Parent spec:** [`2026-05-15-esam3-architecture-design.md`](2026-05-15-esam3-architecture-design.md) §11 step 4
**Sibling specs:** [`2026-05-17-peft-lora-design.md`](2026-05-17-peft-lora-design.md), [`2026-05-16-model-loading-design.md`](2026-05-16-model-loading-design.md)
**Scope:** `src/esam3/peft_adapters/qlora.py`, a one-line docstring + one-parameter signature change in `src/esam3/peft_adapters/lora.py`, example-config cleanup, CPU unit tests, one gated GPU integration test, and a committed Colab notebook + shell script for one-click GPU validation.

---

## 1. Purpose

Provide the single entry point that the trainer (future `spec/training-loop`) and the
exporter (future `spec/cli`) call to turn a freshly-loaded `Sam3Wrapper` into a
4-bit-quantized, LoRA-adapted finetuning target:

```python
wrapper = load_sam31(cfg.model)
wrapper = apply_qlora(wrapper, cfg.peft)   # mutates in place, returns same wrapper
```

`apply_qlora` must:

- Replace **every** `nn.Linear` in the SAM 3.1 base with `bitsandbytes.nn.Linear4bit`
  (standard QLoRA recipe — maximizes the VRAM win that justifies the 12 GB recipe in
  the architecture's §6).
- Run `peft.prepare_model_for_kbit_training` to upcast norm layers to fp32, freeze the
  base, and rewire any active gradient checkpointing with `use_reentrant=False`.
- Attach LoRA adapters to the **same** attention-name regex set used by `apply_lora`
  — `lora.SCOPE_TARGETS` and `lora._resolve_targets` are the single source of truth
  for attention naming.
- Preserve the `Sam3Wrapper(images, prompts)` forward signature so no downstream
  trainer code changes when the wrapper goes from "full-finetune-shaped" to
  "QLoRA-wrapped".
- Expose enough surface (`save_qlora` / `load_qlora`) that `train/checkpoint.py` and
  `cli/export_cmd.py` can persist and resume a QLoRA run without ever importing
  `bitsandbytes` themselves. Merge reuses `lora.merge_lora` — no new helper.

Bitsandbytes is isolated to this module. The dependency points one-way (`qlora.py`
→ `lora.py`); `lora.py` never imports `bitsandbytes`.

## 2. File layout

| File | Change |
| --- | --- |
| `src/esam3/peft_adapters/qlora.py` | Implement `apply_qlora`, `save_qlora`, `load_qlora`. Add private helpers `_collect_linear_names(base)`, `_replace_with_bnb_linear4bit(base, names, qcfg)`, `_resolve_parent(base, dotted_name)`, `_torch_dtype(name)`. Lazy import of `bitsandbytes` inside `apply_qlora` / `load_qlora` (~5 LOC `ImportError` shim with `[qlora]` install hint). Imports `_resolve_targets`, `SCOPE_TARGETS` from `lora.py`. |
| `src/esam3/peft_adapters/lora.py` | (a) Add `linear_types: tuple[type, ...] = (nn.Linear,)` parameter to `_resolve_targets`; replace hard-coded `isinstance(m, nn.Linear)` with `isinstance(m, linear_types)`. (b) Append a one-line note to `merge_lora`'s docstring describing the QLoRA dequant behavior. No other behavioral change. |
| `tests/unit/test_peft_qlora.py` | **New** — CPU smoke tests: registry lookup, schema parse, ImportError surface when bnb missing, `_resolve_targets` accepts custom `linear_types`. No real swap. |
| `tests/integration/test_peft_qlora_real.py` | **New**, `@pytest.mark.requires_compatible_gpu` + `@pytest.mark.requires_checkpoint` — gated end-to-end against the real Meta checkpoint with bnb installed. |
| `tests/unit/test_stubs_raise.py` | Remove the assertion that `apply_qlora` raises `NotImplementedError`. |
| `configs/examples/coco_bbox_qlora.yaml` | Drop SAM-3.1-incorrect `peft.target_modules: ["q_proj","v_proj"]` so library defaults apply; add commented `peft.scope` and `peft.qlora.*` knobs (mirrors LoRA spec §7). |
| `notebooks/colab_gpu_tests.ipynb` | **New** — parameterized notebook for one-click Colab validation (§11). |
| `scripts/run_gpu_tests.sh` | **New** — shared pytest invocation for any GPU environment. |
| `README.md` | Add an "Open in Colab" badge in the testing section pointing at `notebooks/colab_gpu_tests.ipynb`. |

LOC budget for `qlora.py`: roughly 130 lines including docstrings (three public entry
points + four private helpers + lazy-import shim).

## 3. `apply_qlora` — algorithm and contract

### 3.1 Signature

```python
@register("peft", "qlora")
def apply_qlora(wrapper: Sam3Wrapper, cfg: PEFTConfig) -> Sam3Wrapper:
    """Quantize SAM 3.1 base to 4-bit and inject LoRA adapters.

    Mutates `wrapper` in place and returns the same instance. After return:
      - every nn.Linear in the base has been replaced by bnb.nn.Linear4bit
      - norm layers are upcast to fp32 (kbit-training recipe)
      - LoRA A/B matrices on attention modules have requires_grad=True
      - all 4-bit base weights have requires_grad=False
      - wrapper.peft_model is the resulting PeftModel
      - wrapper.model.model is the PeftModel-wrapped (quantized) base
    """
```

Narrows the existing stub's `Any -> Any` signature. The `@register("peft","qlora")`
registration is preserved unchanged.

### 3.2 Algorithm

1. **Idempotency guard.** If `wrapper.peft_model is not None`, raise
   `RuntimeError("QLoRA already applied to this wrapper")`. Same guard, same rationale
   as `apply_lora`.

2. **Lazy-import bnb.** Inside the function body:
   ```python
   try:
       import bitsandbytes as bnb
   except ImportError as e:
       raise ImportError(
           "QLoRA requires bitsandbytes. Install with: "
           "pip install 'efficient-sam3-finetuning[qlora]'"
       ) from e
   ```
   Lazy because `bitsandbytes` lives in the `[qlora]` optional extra; a top-level
   import would break LoRA-only users on machines without bnb.

3. **Locate the inner base.** `base = cast(nn.Module, wrapper.model.model)` — the
   `Sam3Image` instance held by `_Sam3ImageAdapter`. Same path as `apply_lora`.

4. **Collect quantization targets.** `quant_names = _collect_linear_names(base)` —
   every fully-qualified module name where `isinstance(m, nn.Linear)`. No scope
   filtering for quantization in v0 (see §8 for the deferred `quantize_scope` knob).
   If `quant_names` is empty, raise
   `ValueError("apply_qlora: no nn.Linear modules found in base; cannot quantize")`.
   Same loud-failure principle as `apply_lora`'s empty-match guard.

5. **In-place module swap.** `_replace_with_bnb_linear4bit(base, quant_names, cfg.qlora)`.
   For each name:
   ```python
   parent, attr = _resolve_parent(base, name)
   old: nn.Linear = getattr(parent, attr)
   new = bnb.nn.Linear4bit(
       old.in_features,
       old.out_features,
       bias=old.bias is not None,
       quant_type=cfg.qlora.quant_type,                       # "nf4" | "fp4"
       compute_dtype=_torch_dtype(cfg.qlora.compute_dtype),   # bf16 | fp16
   )
   new.load_state_dict(old.state_dict())   # copy pre-trained weights pre-quantization
   new = new.to(old.weight.device)         # quantization fires on .to(cuda)
   setattr(parent, attr, new)
   ```
   The `.to(device)` call is what triggers bnb's 4-bit quantization of the pre-trained
   weight. `_resolve_parent` walks the dotted name to find the immediate parent module
   and the attribute slot to overwrite. `_torch_dtype` maps the schema's `Dtype`
   literal (`"bfloat16"` | `"float16"`) to the corresponding `torch.dtype`.

6. **Resolve LoRA targets via `lora.py`.**
   ```python
   from esam3.peft_adapters.lora import _resolve_targets
   lora_target_names = _resolve_targets(
       base, cfg, linear_types=(bnb.nn.Linear4bit,)
   )
   ```
   After step 5, the attention-projection modules are `Linear4bit` instances — they
   are *not* `nn.Linear` subclasses. `_resolve_targets` gains a new
   `linear_types: tuple[type, ...] = (nn.Linear,)` parameter (see §3.4 for the
   `lora.py` change). `qlora.py` passes `(bnb.nn.Linear4bit,)`. This keeps the
   attention-naming regex set (`SCOPE_TARGETS`) as the single source of truth and
   does not leak `bnb` into `lora.py`.

7. **Build LoraConfig.** Same construction as `apply_lora`:
   ```python
   from peft import LoraConfig
   lora_cfg = LoraConfig(
       r=cfg.r,
       lora_alpha=cfg.alpha,
       lora_dropout=cfg.dropout,
       target_modules=lora_target_names,     # resolved full names, not regex
       bias=cfg.bias,
       task_type=None,
   )
   ```

8. **kbit prep + PEFT wrap.**
   ```python
   from peft import get_peft_model, prepare_model_for_kbit_training
   base = prepare_model_for_kbit_training(
       base,
       use_gradient_checkpointing=getattr(base, "is_gradient_checkpointing", False),
   )
   peft_base = get_peft_model(base, lora_cfg)
   wrapper.model.model = peft_base
   wrapper.peft_model = peft_base
   ```
   `prepare_model_for_kbit_training` (a) upcasts non-`Linear4bit` params to fp32 for
   training stability (norm layers, biases), (b) sets `use_reentrant=False` for any
   active gradient checkpointing, (c) freezes all base params. The explicit
   `requires_grad=False` loop from `apply_lora` step 4 is therefore unnecessary here
   — PEFT does it inside `prepare_model_for_kbit_training`.

   The `is_gradient_checkpointing` attribute check is best-effort; if Meta's SAM 3.1
   build doesn't expose such an attribute, the call still works and `kbit_prep`
   leaves checkpointing untouched on a model that has none active.

9. **Sanity log + warn.**
   ```python
   trainable = sum(p.numel() for p in peft_base.parameters() if p.requires_grad)
   total     = sum(p.numel() for p in peft_base.parameters())
   ratio     = trainable / total if total else 0.0
   logger.info(
       "QLoRA: %d Linears -> Linear4bit; trainable=%d (%.2f%%) of %d "
       "(lora_scope=%s, n_lora_targets=%d, quant_type=%s, compute_dtype=%s)",
       len(quant_names), trainable, 100 * ratio, total,
       cfg.scope if cfg.target_modules is None else "<override>",
       len(lora_target_names), cfg.qlora.quant_type, cfg.qlora.compute_dtype,
   )
   if ratio > 0.10:
       logger.warning(
           "QLoRA trainable ratio %.2f%% exceeds 10%%; "
           "likely a misconfigured scope or target_modules.",
           100 * ratio,
       )
   ```

10. **Return** the same `wrapper` instance for fluent style.

### 3.3 Error surface

| Condition | Behavior |
| --- | --- |
| `wrapper.peft_model is not None` | `RuntimeError("QLoRA already applied to this wrapper")` |
| `bitsandbytes` not installed | `ImportError` naming the `[qlora]` extra and the install command |
| Base contains no `nn.Linear` | `ValueError("apply_qlora: no nn.Linear modules found in base; cannot quantize")` |
| No `Linear4bit` matches the LoRA scope after swap | `ValueError` from `lora._resolve_targets` listing patterns tried and the first 50 `Linear4bit` names found |
| Module-swap `.to(device)` fails because device is CPU | `RuntimeError` from bnb (surfaced unmodified — bnb's error message is clear; callers must move the wrapper to CUDA before `apply_qlora`, same "device before adapter" rule as `apply_lora`) |
| Trainable ratio > 10% | Warning log; not raised |

### 3.4 Required change in `lora.py`

`lora._resolve_targets` currently hard-codes `isinstance(m, nn.Linear)` (lora.py:52).
This spec changes its signature to:

```python
def _resolve_targets(
    base: nn.Module,
    cfg: PEFTConfig,
    linear_types: tuple[type, ...] = (nn.Linear,),
) -> list[str]:
    ...
    for name, module in base.named_modules():
        if not isinstance(module, linear_types):
            continue
        ...
```

The default keeps every existing call site (`apply_lora`) green. `qlora.py` is the
only caller that overrides the default. This is the one minimal touch on `lora.py`
that the architecture's "isolated from `lora.py`" wording permits — the dependency
direction stays `qlora → lora`; bnb never leaks back.

`merge_lora`'s docstring gains one line:

> *"For QLoRA wrappers, this dequantizes the 4-bit base to `compute_dtype` during
> folding; the resulting module is no longer 4-bit-quantized."*

No behavioral change to `merge_lora`.

### 3.5 Interactions with `load_sam31` and `apply_lora`

- **dtype.** Base comes off `load_sam31` in bf16. The swap-then-`.to(cuda)` flow
  quantizes those bf16 weights into nf4. `compute_dtype=bf16` keeps dequantized
  compute in bf16. LoRA `A`/`B` stay in fp32 (PEFT default + `kbit_prep` upcast),
  which is correct for training stability.
- **Gradient checkpointing.** `load_sam31` enables it on the encoder.
  `prepare_model_for_kbit_training` re-wires it to `use_reentrant=False`. We pass
  `use_gradient_checkpointing=True` when the base already has it on (best-effort
  detection via `getattr(base, "is_gradient_checkpointing", False)`); if Meta's
  encoder uses a non-standard hook name, the kbit-prep call is a benign no-op for
  that wiring and `load_sam31`'s setting stands.
- **Device.** `get_peft_model` keeps params on their current device. Callers must
  apply QLoRA after the wrapper is on CUDA (CPU is unsupported by `Linear4bit`).
  Same convention as `apply_lora`'s "device before adapter" rule, stricter here
  because bnb has no CPU fallback.
- **Frozen-base policy.** Discharges the responsibility the model-loading spec §10
  explicitly delegated to `peft_adapters/` for the QLoRA path. After return, only
  LoRA `A`/`B` matrices have `requires_grad=True`.
- **Shared `_resolve_targets`.** `qlora.py` reuses `lora._resolve_targets` and
  `lora.SCOPE_TARGETS`. The minimal `linear_types` parameter addition is the
  single point of coupling.

## 4. `save_qlora` / `load_qlora` / merge

Three small shims. They live in `qlora.py` so any module that needs to persist a
QLoRA run can call them by name without importing `bitsandbytes`.

```python
def save_qlora(wrapper: Sam3Wrapper, dirpath: str | Path) -> None:
    """Write LoRA adapter weights, LoraConfig, and esam3_qlora.json to `dirpath`.

    Raises RuntimeError if wrapper.peft_model is None.
    """

def load_qlora(wrapper: Sam3Wrapper, dirpath: str | Path) -> Sam3Wrapper:
    """Reconstruct a QLoRA wrapper from a saved directory.

    Reads esam3_qlora.json, quantizes wrapper's base via the same swap helper
    apply_qlora uses, runs prepare_model_for_kbit_training, then
    PeftModel.from_pretrained(base, dirpath). Mutates wrapper in place.
    Raises RuntimeError if wrapper.peft_model is already set.
    Raises ValueError on unknown esam3_qlora.json format_version.
    """
```

**`esam3_qlora.json` format** (tiny, ~3 fields, written next to PEFT's `adapter_model.*`
+ `adapter_config.json`):

```json
{
  "format_version": 1,
  "quant_type": "nf4",
  "compute_dtype": "bfloat16"
}
```

`format_version: 1` so a future schema bump can be detected without ambiguity.
`load_qlora` raises `ValueError` on unknown versions. The rationale for bumping
`format_version` is documented in the module docstring.

`save_qlora` body:

```python
if wrapper.peft_model is None:
    raise RuntimeError("save_qlora: wrapper has no PeftModel; call apply_qlora first")
Path(dirpath).mkdir(parents=True, exist_ok=True)
wrapper.peft_model.save_pretrained(str(dirpath))
meta = {
    "format_version": 1,
    "quant_type": _infer_quant_type_from_wrapper(wrapper),
    "compute_dtype": _infer_compute_dtype_from_wrapper(wrapper),
}
(Path(dirpath) / "esam3_qlora.json").write_text(json.dumps(meta, indent=2))
```

`_infer_quant_type_from_wrapper` / `_infer_compute_dtype_from_wrapper` read the
first `Linear4bit` module in the base and return its `.quant_type` /
`.compute_dtype` (the standard `Linear4bit` attribute names). This avoids needing
to thread the original `QLoRAConfig` through `Sam3Wrapper` — the post-apply model
already carries the answer.

`load_qlora` body:

```python
if wrapper.peft_model is not None:
    raise RuntimeError("load_qlora: wrapper already has a PeftModel attached")
meta = json.loads((Path(dirpath) / "esam3_qlora.json").read_text())
if meta.get("format_version") != 1:
    raise ValueError(
        f"load_qlora: unsupported esam3_qlora.json format_version "
        f"{meta.get('format_version')!r}; expected 1"
    )
qcfg = QLoRAConfig(
    quant_type=meta["quant_type"],
    compute_dtype=meta["compute_dtype"],
)
base = cast(nn.Module, wrapper.model.model)
quant_names = _collect_linear_names(base)
_replace_with_bnb_linear4bit(base, quant_names, qcfg)
from peft import PeftModel, prepare_model_for_kbit_training
base = prepare_model_for_kbit_training(
    base,
    use_gradient_checkpointing=getattr(base, "is_gradient_checkpointing", False),
)
peft_base = PeftModel.from_pretrained(base, str(dirpath))
wrapper.model.model = peft_base
wrapper.peft_model = peft_base
return wrapper
```

**Merge.** No new code in `qlora.py`. `lora.merge_lora(wrapper)` works on QLoRA
wrappers because `peft.merge_and_unload()` dequantizes 4-bit weights to
`compute_dtype` and folds in LoRA deltas. The resulting module is no longer
4-bit-quantized. `merge_lora`'s docstring gains one line noting this; the
training-loop and CLI export paths call `merge_lora` regardless of whether the
wrapper is LoRA or QLoRA.

`train/checkpoint.py` (future spec) decides which of `save_lora` / `save_qlora` to
call based on `cfg.peft.method`. `cli/export_cmd.py` decides which of `load_lora` /
`load_qlora` to call by checking for the presence of `esam3_qlora.json` in the
adapter directory.

## 5. Schema — no changes

`QLoRAConfig` (`quant_type`, `compute_dtype`) and `PEFTConfig.method="qlora"` already
exist (`src/esam3/config/schema.py:145-164`). `PEFTConfig.scope`,
`PEFTConfig.target_modules`, `PEFTConfig.bias` are reused unchanged for the LoRA
portion of QLoRA. No new fields, no validator changes. The existing CPU schema tests
(`tests/unit/test_config_schema.py`) already cover `PEFTConfig(method="qlora")`.

## 6. Example config cleanup

`configs/examples/coco_bbox_qlora.yaml` currently carries `peft.target_modules:
["q_proj", "v_proj"]` from scaffolding (those names are wrong for SAM 3.1). Mirrors
the LoRA spec §7 cleanup:

- Remove `target_modules: ["q_proj", "v_proj"]` so the scope-based defaults apply.
- Add commented knobs:
  ```yaml
  peft:
    method: qlora
    # scope: vision_decoder       # vision | vision_decoder (default) | all
    # bias: none                  # none | all | lora_only
    # target_modules: [...]       # overrides scope when set
    r: 16
    alpha: 32
    dropout: 0.05
    qlora:
      quant_type: nf4             # nf4 | fp4
      compute_dtype: bfloat16     # bfloat16 | float16
  ```
  Users see every relevant lever without reading the schema source.

## 7. Testing

### 7.1 CPU unit tests — `tests/unit/test_peft_qlora.py`

Fast, no GPU, no bnb required. CPU CI exercises only what can run without a
compatible GPU.

| Test | Asserts |
| --- | --- |
| `test_registry_lookup` | `lookup("peft", "qlora")` returns the `apply_qlora` callable. |
| `test_schema_qlora_method` | `PEFTConfig(method="qlora")` validates with default `QLoRAConfig`; `quant_type="bogus"` raises `ValidationError`; `compute_dtype="bogus"` raises. |
| `test_import_does_not_require_bnb` | `import esam3.peft_adapters.qlora` succeeds when bnb is absent (verifies lazy import). Uses `monkeypatch.setitem(sys.modules, "bitsandbytes", None)`. |
| `test_apply_qlora_raises_when_bnb_missing` | Same monkeypatch; calling `apply_qlora(stub_wrapper, cfg)` raises `ImportError` whose message names `[qlora]` and the pip install command. |
| `test_resolve_targets_supports_custom_linear_types` | The new `linear_types=` parameter on `lora._resolve_targets` matches a synthetic non-`nn.Linear` class (a tiny `class FakeLinear4bit(nn.Module)` defined in the test). Direct unit test for the small `lora.py` change. |
| `test_resolve_targets_default_still_filters_to_nn_linear` | Calling `_resolve_targets(base, cfg)` (no `linear_types`) still ignores non-`nn.Linear` modules. Guards against accidental broadening of the default. |

These tests do not exercise the module swap or any GPU code path. Coverage gate
(`--cov-fail-under=80`) is satisfied via narrow `# pragma: no cover` comments on the
bodies of `apply_qlora`, `load_qlora`, and `_replace_with_bnb_linear4bit` —
function-level only, never whole-file.

### 7.2 Gated integration test — `tests/integration/test_peft_qlora_real.py`

Marked `@pytest.mark.requires_compatible_gpu` AND `@pytest.mark.requires_checkpoint`.
The combined skip condition:

```python
pytestmark = [
    pytest.mark.requires_compatible_gpu,
    pytest.mark.requires_checkpoint,
]

def _gpu_and_bnb_ready() -> bool:
    if not torch.cuda.is_available():
        return False
    cc = torch.cuda.get_device_capability()
    if cc < (7, 5):
        return False
    try:
        import bitsandbytes  # noqa: F401
    except ImportError:
        return False
    return True

pytest.skipif(not _gpu_and_bnb_ready(), reason="needs CUDA CC >= 7.5 + bnb")
```

(Checkpoint presence is checked the same way the LoRA gated test does it.)

Asserts:

- `load_sam31(ModelConfig(device="cuda"))` then
  `apply_qlora(wrapper, PEFTConfig(method="qlora"))` succeeds without error.
- After apply: at least one module in `base.named_modules()` is a
  `bnb.nn.Linear4bit`; **no** `nn.Linear` modules remain in the base subtree
  (every Linear was swapped).
- `wrapper.peft_model` is a `PeftModel` instance.
- Trainable parameter ratio is `< 0.05`.
- At least one matched LoRA-target name contains `"vision_encoder"`; at least one
  contains `"mask_decoder"` (default `scope="vision_decoder"`).
- `save_qlora(w, tmp_path)` writes `adapter_model.*`, `adapter_config.json`,
  **and** `esam3_qlora.json` with `format_version=1` and the expected
  `quant_type` / `compute_dtype`.
- Fresh wrapper from `load_sam31(...)` + `load_qlora(fresh, tmp_path)` produces
  a state-dict matching the saved LoRA weights within `torch.allclose` tolerance.
- `lora.merge_lora(w)` clears `w.peft_model`; the resulting `w.model.model`
  contains no `Linear4bit` modules (verified by walking `named_modules()`).

This test is excluded from default `pytest` runs. The Colab notebook in §11 is the
primary trigger for it.

### 7.3 Cleanup

- `tests/unit/test_stubs_raise.py`: remove the assertion that `apply_qlora` raises
  `NotImplementedError`.
- `tests/unit/test_imports.py`: should already cover `peft_adapters/qlora.py` via
  its import sweep; verify and add it explicitly if not.
- Coverage gate (`--cov-fail-under=80`) must continue to pass with the new module.
  Use narrow `# pragma: no cover` comments on the function bodies that can only
  run on GPU.

### 7.4 Not tested

- `bitsandbytes` internals.
- The QLoRA paper's numerical fidelity claims (out of scope).
- The actual 4-bit dequant compute path beyond "doesn't crash" — convergence is
  covered indirectly by the future `spec/smoke-test`.

## 8. Out of scope / deferred

| Item | Deferred to |
| --- | --- |
| `quantize_scope` / `exclude_modules` knob (skip quantization on selected modules, e.g. mask-decoder output projection for precision-sensitive layers). | post-v0 — YAGNI until a real precision issue surfaces. |
| Double quantization (`bnb_4bit_use_double_quant=True`). Small additional VRAM win, adds a config field. | post-v0 |
| `paged_adamw_8bit` / paged-optimizer integration. The trainer's optimizer choice is `spec/training-loop`'s concern; `qlora.py` does not touch optimizer state. | `spec/training-loop` |
| `bnb_4bit_quant_storage` dtype tuning. | post-v0 |
| Direct quantization at load time (`load_in_4bit=True` flag in `load_sam31`). Architecture intentionally keeps quantization in `peft_adapters/` (model-loading spec §10). | not planned |
| Multi-adapter composition / hot-swap / routing. | post-v0 |
| AWQ / GPTQ / other quantization formats. | post-v0 |
| Tier-(B) scheduled headless Colab execution and tier-(C) self-hosted GPU runner in GitHub Actions (see §11). | future infra spec — likely landed alongside `spec/smoke-test`. |

## 9. Risks / open items pinned at implementation time

These do not block spec approval; each has a defined resolution path during
implementation:

- **bnb minimum version.** `bitsandbytes>=0.43` is in pyproject `[qlora]` extra.
  `Linear4bit` API + `prepare_model_for_kbit_training` are stable across that range.
  If a future bump renames `compute_dtype` or quant kwargs, blast radius is the
  `_replace_with_bnb_linear4bit` helper.
- **CUDA compute capability requirement.** bnb 4-bit requires CC ≥ 7.5 (Turing or
  later). Documented in the integration test's skip reason and in the Colab notebook
  guard. CONTRIBUTING gains a one-paragraph note that QLoRA needs Turing+ at minimum.
- **Meta's `set_grad_checkpointing` attribute name.**
  `prepare_model_for_kbit_training` toggles checkpointing on its known hooks; if
  Meta uses a non-standard name, the kbit-prep call is a no-op for that wiring and
  `load_sam31`'s setting stands. Pinned at implementation time by inspecting the
  loaded base.
- **`bnb.nn.Linear4bit.load_state_dict` semantics on pre-trained weights.** Verified
  at implementation time on a single-layer GPU fixture; if `load_state_dict` does
  not accept pre-quantization weights cleanly, fall back to
  `new.weight = bnb.nn.Params4bit(old.weight.data, requires_grad=False)` then bias
  copy. Blast radius is the `_replace_with_bnb_linear4bit` helper.
- **`Linear4bit.quant_type` / `compute_dtype` attribute names** (used in
  `save_qlora` to infer metadata). Pinned at implementation time by inspecting
  a real instance. If the names differ in the installed bnb version, update the
  inference helpers.
- **`lora._resolve_targets` signature change.** Adding
  `linear_types: tuple[type, ...] = (nn.Linear,)` is backward-compatible at every
  existing call site. Existing LoRA tests must continue to pass with no changes —
  guarded by the new `test_resolve_targets_default_still_filters_to_nn_linear` test
  in §7.1.

## 10. Acceptance criteria

A correct implementation of this spec satisfies:

1. `apply_qlora(wrapper, PEFTConfig(method="qlora"))` on a CUDA-CC≥7.5 machine with
   bnb installed mutates the wrapper such that every `nn.Linear` in the base is
   replaced with `bnb.nn.Linear4bit`, norm layers are upcast to fp32, only LoRA
   `A`/`B` params are trainable, and `wrapper.peft_model` is a `PeftModel`.
2. `apply_qlora` raises `ImportError` (with `[qlora]` install hint) when bnb is
   missing; `RuntimeError` on double-apply; `ValueError` when no `nn.Linear` modules
   exist in the base.
3. `save_qlora(w, dir)` writes PEFT's adapter files plus `esam3_qlora.json` with
   `{format_version, quant_type, compute_dtype}`.
4. `load_qlora(fresh_wrapper, dir)` reproduces the saved state: every `nn.Linear`
   becomes `Linear4bit`, LoRA weights match the saved adapter within
   `torch.allclose` tolerance. Unknown `format_version` raises `ValueError`.
5. `lora.merge_lora(qlora_wrapper)` clears `peft_model` and leaves a fully
   dequantized base (no `Linear4bit` modules remain).
6. The new `linear_types` parameter on `lora._resolve_targets` does not break any
   existing LoRA test.
7. CPU unit tests in §7.1 pass on any dev machine; the gated integration test in
   §7.2 passes on a Turing+ GPU with bnb installed and the Meta checkpoint present.
8. `ruff check`, `ruff format --check`, and `mypy --strict` pass on every touched
   file.
9. `notebooks/colab_gpu_tests.ipynb` runs end-to-end on a fresh Colab T4 (or better)
   runtime with a valid `HF_TOKEN` secret, producing a green pytest run for the
   gated `requires_compatible_gpu and requires_checkpoint` selection.

## 11. Colab GPU test automation

QLoRA is the first subsystem whose meaningful tests live behind a GPU gate. The
infrastructure landed here is reusable by future GPU-dependent specs (notably
`spec/smoke-test`, architecture step 9).

### 11.1 Artifacts

**`notebooks/colab_gpu_tests.ipynb`** — committed, parameterized notebook.

Single editable input cell at the top:

```python
BRANCH = "peft-qlora"   # change to test a different branch; default "main"
REPO   = "<github-user>/Efficient-SAM3-Finetuning"  # pinned at implementation time
```

Cells in order:

1. **Environment guard.** Asserts running on Colab (presence of `google.colab`),
   `torch.cuda.is_available()`, and `torch.cuda.get_device_capability() >= (7, 5)`.
   Prints the detected GPU name. Fails fast with a clear message if the runtime
   type is wrong ("Runtime → Change runtime type → T4 GPU or better").
2. **Clone + checkout.** `! git clone https://github.com/{REPO}.git` then
   `! cd Efficient-SAM3-Finetuning && git checkout {BRANCH}`.
3. **Install.** `! pip install -e ".[qlora,dev,tensorboard]"` from the cloned
   directory. Pinned versions come from `pyproject.toml` + `uv.lock` (already in
   the repo).
4. **HF auth.** `from google.colab import userdata; os.environ["HF_TOKEN"] =
   userdata.get("HF_TOKEN")`. The notebook docstring instructs users to set
   `HF_TOKEN` once in Colab Secrets (paid-tier-independent feature).
5. **Checkpoint download.**
   `! huggingface-cli download facebook/sam3.1 --local-dir models/sam3.1` — the
   checkpoint is gated; auth from step 4 is required. The first run on a new
   Colab session takes a few minutes; subsequent runs in the same session reuse
   the cached files.
6. **Run gated tests.** `! bash scripts/run_gpu_tests.sh` — single line. The
   shell script (§11.2) is the canonical pytest invocation.
7. **Summary.** A small markdown cell with instructions to read the pytest
   summary line printed at the bottom of cell 6 for at-a-glance pass/fail.
   (Sharing exit codes between `!`-shell cells and Python cells in Jupyter is
   awkward enough that pretending to automate it would be more confusing than
   helpful.)

The notebook is intentionally short (~7 cells). It is checked into git as a
notebook file (not generated). Outputs are stripped before commit via
`nbstripout` (added as a pre-commit hook in `.pre-commit-config.yaml`) so
diffs stay reviewable.

**`scripts/run_gpu_tests.sh`** — committed shell script. The canonical pytest
invocation for any GPU environment (Colab, local Turing+ machine, future
self-hosted runner):

```bash
#!/usr/bin/env bash
set -euo pipefail
pytest -v --tb=short \
  -m "requires_compatible_gpu and requires_checkpoint" \
  --no-cov \
  tests/integration/
```

`--no-cov` because GPU-gated tests are not expected to participate in the
CPU coverage gate. Marked executable in git (`chmod +x` before commit).

**`README.md`** — add an "Open in Colab" badge in the testing section pointing at
`notebooks/colab_gpu_tests.ipynb` on the default branch. The `<github-user>` slug
in the badge URL is pinned at implementation time (same convention as the `REPO`
constant in cell 1 of the notebook):

```markdown
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<github-user>/Efficient-SAM3-Finetuning/blob/main/notebooks/colab_gpu_tests.ipynb)
```

The badge URL is parameterized by branch in Colab's UI — reviewers open it on
whatever branch the PR is on.

### 11.2 Automation tier

This spec lands tier (A) only:

- (A) **One-click reproducible.** Notebook committed, badge in README, single
  pytest invocation via `scripts/run_gpu_tests.sh`. Human clicks "Run all" in
  Colab. **In scope for this spec.**
- (B) Scheduled headless execution via `papermill` + a small VM. Deferred — Colab
  sessions are fragile and the value doesn't pay off until there are multiple
  GPU-gated subsystems.
- (C) Self-hosted GPU runner in GitHub Actions. Deferred — needs a dedicated infra
  spec, likely landed alongside `spec/smoke-test` when there is more than one
  GPU-dependent test suite to amortize the cost over.

The shared `scripts/run_gpu_tests.sh` is the substrate both (B) and (C) would
build on later, so the work here is not throwaway.
