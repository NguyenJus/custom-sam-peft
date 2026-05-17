# PEFT-LoRA Design (spec/peft-lora)

**Status:** approved (brainstorming)
**Parent spec:** [`2026-05-15-esam3-architecture-design.md`](2026-05-15-esam3-architecture-design.md) §11 step 3
**Sibling specs:** [`2026-05-16-model-loading-design.md`](2026-05-16-model-loading-design.md), [`2026-05-16-data-loading-design.md`](2026-05-16-data-loading-design.md)
**Scope:** `src/esam3/peft_adapters/lora.py`, additions to `src/esam3/config/schema.py`, a `peft_model` slot on `Sam3Wrapper`, example-config tweaks, and a new unit-test module plus a gated integration test.

---

## 1. Purpose

Provide the single entry point that the trainer (future `spec/training-loop`) and the
exporter (future `spec/cli`) will call to turn a freshly-loaded `Sam3Wrapper` into a
parameter-efficient finetuning target:

```python
wrapper = load_sam31(cfg.model)
wrapper = apply_lora(wrapper, cfg.peft)   # mutates in place, returns same wrapper
```

`apply_lora` must:

- Freeze every parameter of the SAM 3.1 base.
- Inject LoRA adapters into a scope-determined set of `nn.Linear` modules using
  HuggingFace `peft`.
- Preserve the `Sam3Wrapper(images, prompts)` forward signature so no downstream
  trainer code changes when the wrapper goes from "full-finetune-shaped" to
  "LoRA-wrapped".
- Expose enough surface (`save_lora` / `load_lora` / `merge_lora`) that
  `train/checkpoint.py` and `cli/export_cmd.py` can persist, resume, and merge
  adapters without ever importing `peft` themselves.

QLoRA (4-bit base + LoRA) is intentionally not in scope; it gets its own
`spec/peft-qlora` so the bitsandbytes-specific code stays isolated.

## 2. File layout

| File | Change |
| --- | --- |
| `src/esam3/peft_adapters/lora.py` | Implement `apply_lora`, `save_lora`, `load_lora`, `merge_lora`. Define module-level `SCOPE_TARGETS` dict (regex patterns pinned at implementation time). |
| `src/esam3/config/schema.py` | Add `LoraScope` literal; add `PEFTConfig.scope` and `PEFTConfig.bias`; change `PEFTConfig.target_modules` default to `None`. |
| `src/esam3/models/sam3.py` | Add `Sam3Wrapper.peft_model: PeftModel \| None = None` attribute (initialized in `__init__`). No behavioral change to existing methods. |
| `tests/fixtures/tiny_sam3_lora_stub.py` | **New** — `nn.Module` whose `named_modules()` tree mimics SAM 3.1's vision-encoder + mask-decoder attention naming so unit tests can exercise scope/target resolution without real weights. |
| `tests/unit/test_peft_lora.py` | **New** — eleven CPU-only tests covering apply/scope/precedence/freeze/save+load/merge/idempotency. |
| `tests/unit/test_peft_lora_real.py` | **New**, `@pytest.mark.requires_checkpoint` — gated end-to-end against the real Meta checkpoint. |
| `tests/unit/test_stubs_raise.py` | Remove the assertion that `apply_lora` raises `NotImplementedError`. |
| `tests/unit/test_config_schema.py` | Extend to cover new `PEFTConfig` fields and `target_modules=None` precedence rule. |
| `configs/examples/coco_text_lora.yaml`, `configs/examples/coco_bbox_qlora.yaml` | Drop the SAM-3.1-incorrect default `target_modules: ["q_proj","v_proj"]` so library defaults apply; add a commented `peft.scope: vision_decoder` knob so users see the lever. |

LOC budget for the production module (`lora.py`): roughly 120 lines including
docstrings, including the four helpers and the `SCOPE_TARGETS` dict.

## 3. Schema additions (`src/esam3/config/schema.py`)

```python
LoraScope = Literal["vision", "vision_decoder", "all"]

class PEFTConfig(_Strict):
    method: PEFTMethod
    r: PositiveInt = 16
    alpha: PositiveInt = 32
    dropout: float = Field(default=0.05, ge=0.0, lt=1.0)
    scope: LoraScope = "vision_decoder"                     # NEW
    target_modules: list[str] | None = None                 # CHANGED (was ["q_proj","v_proj"])
    bias: Literal["none", "all", "lora_only"] = "none"      # NEW
    qlora: QLoRAConfig = Field(default_factory=QLoRAConfig)
```

**Default rationale.** The expected user has a niche, non-natural-image dataset
with a novel class taxonomy. `vision_decoder` adapts both the visual-feature path
(needed for distribution-shifted pixels) and the mask-decoder cross-attention
(needed to bind new class names to mask predictions), while leaving the text
encoder frozen since class words are still English in the common case. `vision`
is the cheaper-VRAM fallback; `all` adds text-encoder adaptation for genuinely
novel vocabulary.

**Precedence rule** (documented on `target_modules` and enforced in `apply_lora`,
not in pydantic):

- `target_modules is None` → use `SCOPE_TARGETS[scope]`.
- `target_modules` is a list → use it verbatim, **ignore `scope`**.

No mixing. The pydantic layer accepts any combination so users get a single
validation error surface at config-load time; `apply_lora` makes the precedence
explicit at apply time.

`bias` is exposed because LoRA-paper bias handling sometimes helps on small
datasets; default `"none"` matches the LoRA paper and PEFT lib defaults.

## 4. `Sam3Wrapper.peft_model` slot

Add a single attribute to `src/esam3/models/sam3.py`:

```python
class Sam3Wrapper(nn.Module):
    def __init__(self, model: nn.Module, image_size: int = 1008, mask_size: int = 288) -> None:
        super().__init__()
        self.model = model
        self.image_size = image_size
        self.mask_size = mask_size
        self.peft_model: PeftModel | None = None     # NEW
```

`peft_model` is `None` after `load_sam31` and set by `apply_lora` /
`load_lora`; cleared by `merge_lora`. Save/load helpers consult this slot to
decide whether a wrapper is in a LoRA-applied state.

`from peft import PeftModel` is imported under `TYPE_CHECKING` in `sam3.py` to
keep import-time cost zero when LoRA isn't used.

## 5. `apply_lora` — algorithm and contract

### 5.1 Signature

```python
@register("peft", "lora")
def apply_lora(wrapper: Sam3Wrapper, cfg: PEFTConfig) -> Sam3Wrapper:
    """Freeze SAM 3.1 base and inject LoRA adapters.

    Mutates `wrapper` in place and returns the same instance. After return:
      - every base parameter has requires_grad=False
      - LoRA A/B matrices in matched modules have requires_grad=True
      - wrapper.peft_model is the resulting PeftModel handle
      - wrapper.model.model is the PeftModel-wrapped base
    """
```

The Any -> Any signature on the existing stub narrows to `Sam3Wrapper -> Sam3Wrapper`.
The `@register("peft","lora")` registration is preserved unchanged.

### 5.2 Algorithm

1. **Locate the inner base.** `base = wrapper.model.model` — the `Sam3Image`
   instance held by `_Sam3ImageAdapter`. If `wrapper.peft_model is not None`,
   raise `RuntimeError("LoRA already applied to this wrapper")`. Idempotent
   guard prevents double-wrapping that would silently break save/load.

2. **Resolve target modules.**
   - If `cfg.target_modules is None`: `patterns = SCOPE_TARGETS[cfg.scope]`.
   - Else: `patterns = cfg.target_modules`.
   - Walk `base.named_modules()`. A module name `n` matches if **any** pattern
     in `patterns` is a regex match against `n` AND `isinstance(module, nn.Linear)`.
     Collect the full names into `matched_names: list[str]`.

3. **Validate at least one match.** If `matched_names` is empty, raise
   `ValueError` whose message lists the patterns tried and the first 50
   `nn.Linear` names actually present in `base`. Silent zero-match is the
   worst failure mode (training proceeds with zero trainable params); we
   surface it loudly here.

4. **Freeze the base.** `for p in base.parameters(): p.requires_grad = False`.
   `get_peft_model` does this internally too, but explicit freeze first makes
   the trainable-param-count assertion in step 7 meaningful and reduces the
   blast radius if a future PEFT version changes its freezing default.

5. **Build `LoraConfig`.**
   ```python
   lora_cfg = LoraConfig(
       r=cfg.r,
       lora_alpha=cfg.alpha,
       lora_dropout=cfg.dropout,
       target_modules=matched_names,     # resolved full names, not regex
       bias=cfg.bias,
       task_type=None,
   )
   ```
   We pass **resolved full module names**, not regex, so the validation in
   step 3 is the single source of truth for what gets adapted. `task_type=None`
   because SAM 3.1 doesn't fit `peft`'s preset task types (no seq-cls, no LM
   head).

6. **Wrap and re-slot.**
   ```python
   peft_base = get_peft_model(base, lora_cfg)
   wrapper.model.model = peft_base
   wrapper.peft_model = peft_base
   ```
   The `Sam3Wrapper.forward` path becomes
   `wrapper(images, prompts)` → `_Sam3ImageAdapter.forward(images, prompts)` →
   `self.model(...)` where `self.model` is now `PeftModel`. `PeftModel.forward`
   delegates to `base_model.forward` with `*args, **kwargs`, so the existing
   wrapper signature continues to work without a forward shim.

7. **Sanity log + warn.**
   ```python
   trainable = sum(p.numel() for p in peft_base.parameters() if p.requires_grad)
   total     = sum(p.numel() for p in peft_base.parameters())
   ratio     = trainable / total
   logger.info("LoRA: trainable=%d (%.2f%%) of %d", trainable, 100 * ratio, total)
   if ratio > 0.10:
       logger.warning("LoRA trainable ratio %.2f%% exceeds 10%%; likely a misconfigured scope or target_modules.", 100 * ratio)
   ```

8. **Return** the same `wrapper` instance for fluent style.

### 5.3 `SCOPE_TARGETS` (pinned at implementation time)

Reserved placeholders. The implementer pins exact patterns by running

```python
for n, m in base.named_modules():
    if isinstance(m, nn.Linear):
        print(n)
```

once against a real SAM 3.1 build and grouping the output by subtree.

```python
SCOPE_TARGETS: dict[str, list[str]] = {
    "vision":         [r"vision_encoder\..*\.attn\.(qkv|proj)$"],
    "vision_decoder": [r"vision_encoder\..*\.attn\.(qkv|proj)$",
                       r"mask_decoder\..*\.(self_attn|cross_attn)\.(q|k|v|out)_proj$"],
    "all":            [r".*\.(qkv|q_proj|k_proj|v_proj|out_proj|proj)$"],
}
```

This is the **single point** in the codebase that encodes SAM 3.1's attention
naming. Like `meta_to_canonical` in the model-loading spec, if Meta renames
attention modules, only `SCOPE_TARGETS` (and a few stub-test fixtures) needs
to change.

### 5.4 Error surface

| Condition | Behavior |
| --- | --- |
| `wrapper.peft_model is not None` | `RuntimeError("LoRA already applied to this wrapper")` |
| No `nn.Linear` matches any pattern | `ValueError` listing patterns tried + first 50 Linear names present |
| `cfg.scope` not in `SCOPE_TARGETS` and `cfg.target_modules is None` | `KeyError`, but pydantic blocks this upstream because `scope` is a `Literal` |
| Trainable ratio > 10% | Warning log; not raised |
| `peft` not installed | `ImportError` from the import at top of `lora.py`; surfaces the pyproject dep that should already be present |

### 5.5 Interactions with the model-loading spec

- **dtype.** Base is bf16 after `load_sam31`. PEFT's default places LoRA `A`
  and `B` matrices in fp32, which is correct for training stability. We do not
  override; the base stays bf16, the adapters stay fp32.
- **gradient checkpointing.** `load_sam31` enables it on the base when
  `cfg.model.gradient_checkpointing=True`. `get_peft_model` does not disturb
  it; the swapped attention `Linear`s remain inside the checkpointed encoder
  blocks.
- **device.** `get_peft_model` keeps params on their current device. Callers
  must apply LoRA after the wrapper is on its final device (CPU in unit
  tests, CUDA in integration).
- **frozen-base policy.** The model-loading spec §10 explicitly delegates
  freezing to this subsystem. `apply_lora` discharges that responsibility for
  the LoRA path; `apply_qlora` (next spec) discharges it for the QLoRA path.

## 6. Save / load / merge helpers

Three small shims over `peft` API calls. They live in `lora.py` so any module
that needs to persist or fold adapters can call them by name without importing
`peft`.

```python
def save_lora(wrapper: Sam3Wrapper, dirpath: str | Path) -> None:
    """Write adapter weights and LoraConfig to `dirpath`.

    Raises RuntimeError if wrapper.peft_model is None.
    """

def load_lora(wrapper: Sam3Wrapper, dirpath: str | Path) -> Sam3Wrapper:
    """Apply LoRA from a saved adapter directory.

    Reads LoraConfig from dirpath, rebuilds the PeftModel via
    PeftModel.from_pretrained(base, dirpath), re-slots into `wrapper`, and
    returns the same wrapper. Raises RuntimeError if wrapper.peft_model is
    already set.
    """

def merge_lora(wrapper: Sam3Wrapper) -> Sam3Wrapper:
    """Fold LoRA deltas into the base weights in place and unwrap PeftModel.

    Post-condition: wrapper.peft_model is None; wrapper.model.model is the
    bare base again with merged weights. requires_grad on the resulting base
    is left as whatever peft.merge_and_unload returns; the export caller does
    not need to train, and a training caller would re-apply LoRA rather than
    use the merged module directly. Raises RuntimeError if
    wrapper.peft_model is None.
    """
```

Each helper is roughly 3–6 lines. `merge_lora` uses
`peft_base.merge_and_unload()` and assigns the result back into
`wrapper.model.model`.

`train/checkpoint.py` (future spec) wraps these with run-directory layout
policy. `cli/export_cmd.py` calls `load_lora` then `merge_lora` for
`esam3 export --merge`.

## 7. Example-config fixes

`configs/examples/coco_text_lora.yaml` and `configs/examples/coco_bbox_qlora.yaml`
currently carry `target_modules: ["q_proj", "v_proj"]` from the scaffolding
phase. Those names are wrong for SAM 3.1.

Changes per file:

- Remove `target_modules: ["q_proj", "v_proj"]` so the scope-based defaults
  apply.
- Add a commented knob:
  ```yaml
  peft:
    method: lora
    # scope: vision_decoder       # vision | vision_decoder (default) | all
    # bias: none                  # none | all | lora_only
    # target_modules: [...]       # overrides scope when set
    r: 16
    alpha: 32
    dropout: 0.05
  ```
  Users see all three levers without reading the schema source.

The QLoRA example keeps `peft.method: qlora` and its `qlora:` sub-block; this
spec only edits the `target_modules` line and adds the commented knobs above.

## 8. Testing

### 8.1 Unit tests (CPU, no real checkpoint) — `tests/unit/test_peft_lora.py`

Backed by `tests/fixtures/tiny_sam3_lora_stub.py` (see §8.3).

| Test | Asserts |
| --- | --- |
| `test_apply_lora_default_scope_freezes_base` | After `apply_lora(stub_wrapper, PEFTConfig(method="lora"))`: every non-LoRA param has `requires_grad=False`; LoRA `A`/`B` params have `requires_grad=True`. |
| `test_apply_lora_vision_scope_matches_only_vision` | `scope="vision"`: only modules under `vision_encoder.*` are adapted; no `mask_decoder.*` params are trainable. |
| `test_apply_lora_vision_decoder_scope` | `scope="vision_decoder"`: both `vision_encoder.*` and `mask_decoder.*` attention Linears adapted; "negative-control" Linears outside either subtree stay frozen and un-adapted. |
| `test_apply_lora_all_scope` | `scope="all"`: every Linear in the stub (including negative controls) is adapted. |
| `test_target_modules_overrides_scope` | `target_modules=["vision_encoder.block0.attn.qkv"]` with `scope="all"`: only that one module is adapted. Verifies the precedence rule. |
| `test_apply_lora_no_match_raises` | `target_modules=["nonexistent.module"]` raises `ValueError`; message contains the patterns tried and at least one real Linear name. |
| `test_apply_lora_idempotent_guard` | Calling `apply_lora` twice on the same wrapper raises `RuntimeError("LoRA already applied …")`. |
| `test_apply_lora_trainable_ratio` | Trainable / total ratio under `scope="vision_decoder"` is `< 0.05` on the stub (sanity bound, not a hard claim about the real model). |
| `test_apply_lora_preserves_forward_signature` | `inspect.signature(wrapper.forward)` after apply still has `(images, prompts)`. No forward execution (the model-loading spec's `_Sam3ImageAdapter.forward` still raises `NotImplementedError`; this test is signature-only). |
| `test_save_load_lora_roundtrip` | `save_lora(w, tmp_path)` writes `adapter_model.*` + `adapter_config.json`. Fresh `load_lora(new_wrapper, tmp_path)` produces a wrapper whose LoRA `state_dict` matches the saved one within `torch.allclose` tolerance. |
| `test_merge_lora_unwraps_and_clears_handle` | After `merge_lora(w)`: `w.peft_model is None`; `type(w.model.model).__name__` no longer contains "Peft"; at least one base weight differs from its pre-merge value (deltas folded). |

### 8.2 Schema tests — extend `tests/unit/test_config_schema.py`

- `PEFTConfig(method="lora")` yields `scope="vision_decoder"`, `target_modules=None`, `bias="none"`.
- `PEFTConfig(method="lora", scope="encoder")` raises `ValidationError`.
- `PEFTConfig(method="lora", bias="some")` raises `ValidationError`.
- `PEFTConfig(method="lora", scope="all", target_modules=["foo"])` validates;
  pydantic does not enforce precedence (that's `apply_lora`'s job). A separate
  unit test (above) verifies the runtime precedence.

### 8.3 Test fixture — `tests/fixtures/tiny_sam3_lora_stub.py`

An `nn.Module` mirroring SAM 3.1's attention naming, small enough to run
hundreds of times in CPU CI:

```
TinyStub
├── vision_encoder
│   ├── block0.attn.qkv      (nn.Linear 8 → 24)
│   ├── block0.attn.proj     (nn.Linear 8 → 8)
│   ├── block1.attn.qkv
│   └── block1.attn.proj
├── mask_decoder
│   └── layer0
│       ├── self_attn.q_proj  (nn.Linear 8 → 8)
│       ├── self_attn.k_proj
│       ├── self_attn.v_proj
│       ├── self_attn.out_proj
│       ├── cross_attn.q_proj
│       ├── cross_attn.k_proj
│       ├── cross_attn.v_proj
│       └── cross_attn.out_proj
├── neg_control_a              (nn.Linear, outside any scope)
└── neg_control_b              (nn.Linear, outside any scope)
```

Wrapped in the existing `_Sam3ImageAdapter` and `Sam3Wrapper` so tests
exercise the real wrapping path. Forward of the inner stub raises
`NotImplementedError` (matches `_Sam3ImageAdapter`'s current state); tests
in §8.1 do not invoke forward.

### 8.4 Integration test (gated) — `tests/unit/test_peft_lora_real.py`

`@pytest.mark.requires_checkpoint`, skipped unless
`models/sam3.1/sam3.1_multiplex.pt` is present. Asserts:

- `load_sam31(ModelConfig())` then `apply_lora(wrapper, PEFTConfig(method="lora"))`
  succeeds.
- Trainable ratio is `< 0.05`.
- At least one matched module path contains `"vision_encoder"`; at least one
  contains `"mask_decoder"`.
- `save_lora(w, tmp_path)` then `load_lora(fresh_wrapper, tmp_path)`
  round-trips state-dict within tolerance.
- `merge_lora(w)` returns a wrapper with `peft_model is None`.

Excluded from default `pytest`; opt-in via the existing
`requires_checkpoint` marker.

### 8.5 Cleanup

- `tests/unit/test_stubs_raise.py`: remove the line asserting `apply_lora`
  raises `NotImplementedError`.
- `tests/unit/test_imports.py`: should already cover `peft_adapters/lora.py`
  via its import sweep; verify and add it explicitly if not.
- Coverage gate (`--cov-fail-under=80`) must continue to pass with the new
  module.

### 8.6 Not tested

- HuggingFace `peft` internals (third-party library).
- The QLoRA path (next spec).
- Training-loop integration (later spec).
- bf16-base + fp32-adapter numerical stability (would need a real GPU run;
  covered indirectly by `spec/smoke-test`).

## 9. Out of scope / deferred

| Item | Deferred to |
| --- | --- |
| `peft_adapters/qlora.py` implementation (bitsandbytes-dependent module swap + LoRA on top). | `spec/peft-qlora` |
| `peft_adapters/none.py` (full-finetune no-op). Architecture v0 PEFT methods are LoRA + QLoRA only. | post-v0 |
| `train/checkpoint.py` policy (run-dir layout, resume orchestration, best-model selection). | `spec/training-loop` |
| `cli/export_cmd.py` wiring (`esam3 export --merge` calls `load_lora` then `merge_lora`). | `spec/cli` |
| `modules_to_save` config (PEFT-lib feature for fully-trained layers alongside LoRA, e.g. mask-decoder output projection). YAGNI for v0; add when a real workload demands it. | post-v0 |
| Adapter composition / hot-swap / multi-adapter routing. | post-v0 |
| LoRA rank-search / NAS. | post-v0 |

## 10. Risks / open items pinned at implementation time

These do not block spec approval; each has a defined resolution path during
implementation:

- **Exact attention module names in SAM 3.1.** Resolved by inspecting
  `base.named_modules()` once and pinning the regex patterns in
  `SCOPE_TARGETS`. Same convention as the model-loading spec.
- **PeftModel forward kwarg passthrough.** PEFT delegates forward via
  `*args, **kwargs`; should pass our `(images, prompts)` cleanly. Verified
  during the unit `test_apply_lora_preserves_forward_signature` test
  (signature-level) and the gated integration test (runtime).
- **PEFT lib API stability across `>=0.13`.** `LoraConfig` /
  `get_peft_model` / `PeftModel.from_pretrained` / `merge_and_unload` are
  stable surfaces in this version range. If a future bump breaks one, the
  blast radius is confined to `lora.py`.

## 11. Acceptance criteria

A correct implementation of this spec satisfies:

1. `apply_lora(wrapper, PEFTConfig(method="lora"))` mutates the wrapper such
   that base params are frozen, LoRA params are trainable, and
   `wrapper.peft_model` is a `PeftModel`.
2. Scope precedence works as specified: `target_modules=None` picks
   `SCOPE_TARGETS[scope]`; non-None overrides scope entirely.
3. Empty matched-name list raises a `ValueError` whose message identifies the
   miscondition (no silent zero-trainable-param training).
4. `save_lora` → `load_lora` round-trips state-dict to within `torch.allclose`
   tolerance; `merge_lora` clears the handle and folds deltas into the base.
5. All unit tests in §8.1 and §8.2 pass on CPU without the real checkpoint;
   coverage stays ≥ 80%.
6. The integration test in §8.4 passes when run with the real checkpoint.
7. Example configs in §7 load via the existing config loader without errors
   and no longer carry the SAM-3.1-incorrect `target_modules` default.
8. `ruff check`, `ruff format --check`, and `mypy --strict` pass on every
   touched file.
