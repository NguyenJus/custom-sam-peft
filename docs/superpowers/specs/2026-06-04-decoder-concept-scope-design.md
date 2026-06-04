# Add `decoder_concept` LoRA scope (trunk-frozen) as the new default, with per-scope adapter sizing

Date: 2026-06-04
Status: Locked design — implementation-ready spec (do not implement from this document alone; it captures the agreed design only)

## Summary / Goal

Today every LoRA scope in the project (`vision`, `vision_decoder`,
`vision_decoder_concept`, `all`) includes the ViT vision trunk in its target
patterns. This design adds a fifth scope, **`decoder_concept`**, defined as
`vision_decoder_concept` **minus the trunk pattern**, and makes it the new
default `peft.scope`.

The motivation is to **freeze the ViT vision trunk by default** while still
adapting the full decoder in concept-style (cross-attention out-projection,
FFN linears, and the `self_attn` / `ca_text` MultiheadAttention modules with
their in-projection surface). Freezing the trunk means it carries no LoRA
adapters and all of its base parameters keep `requires_grad=False`; autograd
then skips the trunk subgraph during the backward pass automatically. The
forward pass still runs through the trunk — cross-epoch caching of frozen
trunk features is a separate throughput follow-up (issue #300, out of scope).

This change also replaces the single hardcoded adapter-layer-count constant in
`presets.py` with a **per-scope adapter-dimension-sum mapping**, because a
single count cannot be correct across five scopes with very different target
sets (the trunk-frozen default has far fewer adapter layers than the
trunk-dominated old count of 96).

## Background — current scope model

`SCOPE_TARGETS` and `SCOPE_MHA_MODULES` in
`src/custom_sam_peft/peft_adapters/lora.py` together encode SAM 3.1's attention
naming. `SCOPE_TARGETS` maps each `LoraScope` literal to a list of regexes
matched against `nn.Linear` modules; `SCOPE_MHA_MODULES` maps the literal to
regexes matched against `nn.MultiheadAttention` modules, whose names are unioned
into `target_modules` so peft dispatches them to `lora.MultiheadAttention`
(adapting both `in_proj_weight` and `out_proj`).

The four current scopes:

| Scope                     | Generic (`nn.Linear`) targets | MHA targets | Includes trunk? |
| ------------------------- | ----------------------------- | ----------- | --------------- |
| `vision`                  | trunk `attn.(qkv\|proj)`      | none        | yes             |
| `vision_decoder`          | trunk `attn.(qkv\|proj)`; decoder `(self_attn\|cross_attn\|ca_text).out_proj`; decoder `linear[12]` | none | yes |
| `vision_decoder_concept`  | trunk `attn.(qkv\|proj)`; decoder `cross_attn.out_proj`; decoder `linear[12]` | decoder `ca_text`, `self_attn` | yes |
| `all`                     | `.*` (every `nn.Linear`)      | none        | yes             |

All four include the trunk pattern
`backbone\.vision_backbone\.trunk\.blocks\.\d+\.attn\.(qkv|proj)$`. There is no
scope today that adapts the decoder while leaving the trunk frozen.

The resolution logic (`_resolve_targets`, `_resolve_mha_modules`) is
scope-agnostic: it reads whichever pattern list the scope maps to. `apply_lora`
sets `requires_grad=False` on every base parameter before injecting LoRA, so any
module not matched by a scope pattern stays frozen.

## Design

### 1. New scope `decoder_concept` (`peft_adapters/lora.py`)

Add a fifth entry to **both** maps, identical to `vision_decoder_concept` minus
the trunk line
`r"backbone\.vision_backbone\.trunk\.blocks\.\d+\.attn\.(qkv|proj)$"`.

```python
SCOPE_TARGETS["decoder_concept"] = [
    r"transformer\.decoder\.layers\.\d+\.cross_attn\.out_proj$",
    r"transformer\.decoder\.layers\.\d+\.linear[12]$",
]

SCOPE_MHA_MODULES["decoder_concept"] = [
    r"transformer\.decoder\.layers\.\d+\.ca_text$",
    r"transformer\.decoder\.layers\.\d+\.self_attn$",
]
```

No change to `_resolve_targets` or `_resolve_mha_modules` logic — they already
consume these maps generically. The new scope works for **both LoRA and QLoRA**,
since both paths consume the same maps.

QLoRA note to call out in the implementation comment: `cross_attn.out_proj` is a
genuine `nn.Linear` (the `cross_attn` is a `RoPEAttention`, not an
`nn.MultiheadAttention` wrapper), and `linear1`/`linear2` are bare `nn.Linear`
FFN modules. Therefore QLoRA behavior for `decoder_concept` is identical to
`vision_decoder_concept` minus the trunk: the decoder FFN linears quantize under
QLoRA, the `cross_attn.out_proj` stays targetable, and the `self_attn`/`ca_text`
MHA modules are adapted via `SCOPE_MHA_MODULES` exactly as in
`vision_decoder_concept`.

### 2. Schema (`config/schema.py`)

- Add `"decoder_concept"` to the `LoraScope` literal (currently line ~106):

  ```python
  LoraScope = Literal[
      "vision", "vision_decoder", "vision_decoder_concept", "decoder_concept", "all"
  ]
  ```

- Flip the default in `PEFTConfig` (currently line ~583):

  ```python
  scope: LoraScope = "decoder_concept"
  ```

- Rewrite the existing `#230` reproducibility comment (lines ~584-589) to
  document the new trunk-frozen default and the migration described in the
  Reproducibility & Migration section below.

### 3. Per-scope adapter sizing (`presets.py` + `scripts/_derive_preset_constants.py`)

**Replace** the single `LORA_LAYERS = 96` constant and the `D_IN = 768` /
`D_OUT = 768` averages (lines ~45-47) with a per-scope mapping:

```python
ADAPTER_DIM_SUM_BY_SCOPE: dict[str, int] = {
    # <scope>: sum over every injected LoRA adapter of
    #          (lora_A.in_features + lora_B.out_features)
    # derived offline via scripts/_derive_preset_constants.py (see §3 mechanism)
    "vision": ...,                    # tbd: derive run
    "vision_decoder": ...,            # tbd: derive run
    "vision_decoder_concept": ...,    # tbd: derive run
    "decoder_concept": ...,           # tbd: derive run
    "all": ...,                       # tbd: derive run
}
```

Keyed by the `LoraScope` literal **value** (string), so every scope has an entry.

**Change** `_adapter_bytes(r)` (lines ~166-168) to `_adapter_bytes(r, scope)`:

```python
def _adapter_bytes(r: int, scope: str) -> int:
    # ADAPTER_DIM_SUM_BY_SCOPE[scope] * r * 2 bytes (bf16 adapter weights, 2 B/param).
    return ADAPTER_DIM_SUM_BY_SCOPE[scope] * r * 2
```

`_optimizer_bytes` (line ~171) is defined as `_adapter_bytes(...) * 4`, so it
must also take and forward `scope`:

```python
def _optimizer_bytes(r: int, scope: str) -> int:
    return _adapter_bytes(r, scope) * 4
```

**Thread `scope` through the call sites:**

- `presets.py` — the train branch of `_predicted_bytes` (lines ~243-246) calls
  `_adapter_bytes(r)` and `_optimizer_bytes(r)`. `_predicted_bytes` must accept a
  `scope` argument and forward it; `decide_preset` / `decide_eval_batch_size`
  have the run config (a `PEFTConfig`) reachable and supply the scope. The eval
  branch does not add adapter/optimizer bytes, so it is unaffected by the value
  but still passes `scope` through the shared signature.
- `cli/calibrate_cmd.py` (line ~220) — the QLoRA static floor:
  `static = _model_bytes("qlora") + _adapter_bytes(4) + _optimizer_bytes(4) + WORKSPACE_BYTES`
  must become `_adapter_bytes(4, scope)` / `_optimizer_bytes(4, scope)`, with
  `scope` taken from the in-scope `PEFTConfig`.
- `scripts/_derive_preset_constants.py` — the `static` term (lines ~88-93) calls
  `_adapter_bytes(args.r)` / `_optimizer_bytes(args.r)`; thread the scope it is
  probing.

**Why per-scope, and why a dimension sum (not count × average):**

- A single hardcoded layer count cannot be correct for five scopes with very
  different target sets; the trunk-frozen default adapts far fewer layers than
  the trunk-dominated old count of 96.
- Summing exact `in + out` dims (rather than `count × avg-dim`) is **exact** for
  the concept scopes, where peft's `lora.MultiheadAttention` adapts **both**
  `in_proj` (E → 3E) and `out_proj` (E → E). A count × average heuristic cannot
  capture that asymmetry.

**How the numbers are produced — chosen mechanism (offline-derived, "option B"):**

Extend `scripts/_derive_preset_constants.py` so that, for each `LoraScope`
value, it calls `apply_lora` on the already-built **real** model
(via `load_sam31`) and sums the **actual injected peft adapter dims**: for every
LoRA adapter present after injection, add
`lora_A.in_features + lora_B.out_features`. This is exact by construction and
captures peft's real MHA treatment (both `in_proj` and `out_proj`) with zero
hand-derivation. The script prints a copy-paste-ready
`ADAPTER_DIM_SUM_BY_SCOPE = {...}` block for `presets.py`.

**Why NOT a live count at estimate time (record this so it is not re-attempted):**
A live "build SAM 3.1 on `torch.device('meta')` and count" approach was spiked
and is **infeasible**:

- The vendored ViT `__init__` calls `.item()` on a meta tensor —
  `sam3/model/vitdet.py:878`: `dpr = [x.item() for x in torch.linspace(...)]` →
  "Tensor.item() cannot be called on meta tensors".
- Complex-RoPE initialization is a further likely blocker on meta.
- A real CPU build of the ~5B-parameter model does not fit the 16 GB box.

Hence the offline derivation via the maintainer GPU script is the chosen path.

**Population is a gated step.** The real `ADAPTER_DIM_SUM_BY_SCOPE` values must
be populated by **running the extended derive script against the real model**.
This requires a GPU / real-model run and therefore the user's explicit
go-ahead (project rule: ask before any GPU / real-model run). Until populated,
each value must carry an explicit `# tbd:` tag; once filled from the derive run,
each must carry a citation comment naming the derive script and the run date —
never a silent guess (project rule: every new / changed default needs a citation
or an explicit `# tbd:` tag).

### 4. Tests

- **Coverage guard** at `tests/unit/test_peft_lora.py` (~line 177): update the
  expected set to

  ```python
  assert set(SCOPE_TARGETS) == {
      "vision", "vision_decoder", "vision_decoder_concept", "decoder_concept", "all"
  }
  ```

- **Fixtures** in `tests/fixtures/tiny_sam3_lora_stub.py`: add `decoder_concept`
  entries to **both** `FIXTURE_SCOPE_PATTERNS` (~line 141) and
  `FIXTURE_SCOPE_MHA_MODULES` (~line 156), mirroring the real `decoder_concept`
  patterns minus the trunk, using the fixture's truncated `transformer_decoder`
  prefix:

  ```python
  FIXTURE_SCOPE_PATTERNS["decoder_concept"] = [
      r"transformer_decoder\.layers\.\d+\.cross_attn\.out_proj$",
  ]
  FIXTURE_SCOPE_MHA_MODULES["decoder_concept"] = [
      r"transformer_decoder\.layers\.\d+\.ca_text$",
      r"transformer_decoder\.layers\.\d+\.self_attn$",
  ]
  ```

- **New behavioral tests (LoRA, on the stub; plus a QLoRA mirror in the
  real-integration tests):** with `decoder_concept`,
  - LoRA attaches to the decoder `cross_attn.out_proj`, FFN `linear1`/`linear2`,
    and the `self_attn` / `ca_text` MHA modules;
  - the **trunk is frozen** — zero resolved LoRA generic targets and zero
    resolved MHA matches whose name contains `trunk.blocks` (assert no trunk
    module is in the matched/MHA target lists);
  - `PEFTConfig().scope == "decoder_concept"` (the new default).

- **New deterministic coverage guard (no real model):**
  `ADAPTER_DIM_SUM_BY_SCOPE` has an entry for **every** value of the `LoraScope`
  literal, enumerated via `typing.get_args(LoraScope)`. This ensures a future
  scope addition cannot silently miss a sizing entry.

- **New deterministic smoke (no real model):** `_adapter_bytes(r, scope)`
  returns the expected product for a known `(scope, dim_sum)` entry —
  e.g. for a chosen scope with dim sum `S`, `_adapter_bytes(r, scope) == S * r * 2`.

### 5. Templates / docs

- `cli/templates/config_full.yaml` (~line 46): update the commented `scope`
  example and any `LoraScope` option-list comment to include `decoder_concept`,
  note it is the **new default**, and note that it **freezes the ViT trunk**.
- Update any other inline comment enumerating the scope options to include
  `decoder_concept` with the same note.

## Reproducibility & Migration

- Configs that pin an explicit `peft.scope` of `vision`, `vision_decoder`,
  `vision_decoder_concept`, or `all`, **or** that set `peft.target_modules`
  (which overrides the scope entirely), are **unaffected** — their behavior is
  byte-for-byte unchanged.
- Configs with **no explicit `peft.scope`** now resolve to `decoder_concept`
  instead of `vision_decoder_concept`, which means they **stop adapting the ViT
  trunk** (the trunk is frozen). This is the only behavioral change.
- **No in-repo config relies on the default**, so there is no silent in-repo
  behavior change: `overfit_debug.yaml` pins `vision_decoder_concept`, and the
  example configs pin `vision_decoder`. The change matters only for
  **external / user configs that omit `peft.scope`**.
- The `#230` schema comment must be rewritten to state the above migration and
  the new trunk-frozen default rationale.

## Out of scope / follow-ups

- **Issue #300 — "Cache frozen ViT-trunk features across epochs."** With the
  trunk frozen, its forward output is deterministic per input across epochs, so
  it can be computed once and cached — a throughput lever. This is explicitly
  **out of scope** for this spec and tracked in #300.

## Risks & notes

- **Meta-build infeasibility:** a live meta-device count at estimate time is
  blocked by `sam3/model/vitdet.py:878` (`.item()` on a meta tensor) and likely
  complex-RoPE init; a real CPU build of the ~5B-param model exceeds the 16 GB
  box. Do not re-attempt; use the offline derive script.
- **Gated derive run:** populating `ADAPTER_DIM_SUM_BY_SCOPE` with real values
  requires a GPU / real-model run; obtain the user's go-ahead before running the
  extended `scripts/_derive_preset_constants.py`.
- **Citation requirement:** every populated value must carry a citation comment
  (derive script name + date) or an explicit `# tbd:` tag until the derive run
  lands. No silent guesses.
- **Sizing only affects the VRAM preset estimate**, not training correctness —
  the adapter bytes feed `_predicted_bytes` / the QLoRA static floor. An
  inaccurate sum yields a conservative or optimistic memory estimate, not a
  wrong model; but per-scope exactness is the point of the change.
