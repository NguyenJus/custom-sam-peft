# PEFT in_proj Concept Scope Design (spec/peft-in-proj-concept-scope)

**Status:** draft (brainstorming)
**Issue:** [#230 — LoRA: adapt more layers for niche text concepts](https://github.com/NguyenJus/custom-sam-peft/issues/230)
**Research note (source of truth):** [`docs/research/2026-06-01-issue-230-peft-adaptation-surface-lit-review.md`](../../research/2026-06-01-issue-230-peft-adaptation-surface-lit-review.md)
**Sibling specs:** [`2026-05-17-peft-lora-design.md`](2026-05-17-peft-lora-design.md), [`2026-05-17-peft-qlora-design.md`](2026-05-17-peft-qlora-design.md)
**Scope:** `src/custom_sam_peft/peft_adapters/lora.py` (new `target_parameters` resolution axis + one new scope), `src/custom_sam_peft/peft_adapters/qlora.py` (wire the same axis), `src/custom_sam_peft/config/schema.py` (`LoraScope` literal, default `scope`, new `target_parameters` field), `tests/fixtures/tiny_sam3_lora_stub.py` (expose two `nn.MultiheadAttention` decoder children), `src/custom_sam_peft/cli/calibrate_cmd.py` (VRAM-autosize alpha co-scaling + WARNING; extend `PresetDecision` / v3 `chosen_*` cache keys with `alpha`), `src/custom_sam_peft/cli/_config_rewrite.py` (write `alpha` alongside `r` in the sizing block), `tests/unit/test_calibrate_cmd.py` (alpha co-scale + warning + cache round-trip tests), and new/extended unit tests. A feasibility spike against the real SAM 3.1 decoder gates the in_proj surface.

---

## 1. Purpose and goal

Support a fixed set of niche classes — invoked by **text / concept prompts** — on
**instance and semantic** segmentation with SAM 3.1 + LoRA. The shipped default scope
`vision_decoder` adapts only `ca_text.out_proj`, leaving the text-mixing q/k/v
(`in_proj_weight`) **frozen**, so the model can re-weight the *output* of text
cross-attention but cannot re-learn *how* text attends to image features — it cannot learn
niche text concepts. This is the central finding of the research note (§4).

This spec closes exactly that gap, and only that gap:

- Reach the decoder's two genuine `nn.MultiheadAttention` blocks — `ca_text` (concept
  injection) and `self_attn` (DETR-style duplicate-removal / instance separation) — at
  their `in_proj_weight` (packed q/k/v), via `peft`'s `target_parameters` API.
- Ship this as **one new LoRA scope** that becomes the default, without mutating the
  existing `vision` / `vision_decoder` / `all` scopes (reproducibility).
- Add a parallel **parameter-name resolution axis** (`target_parameters`) alongside the
  existing module-name axis (`target_modules`), wired through both the plain-LoRA and the
  QLoRA apply paths.

Priority order, per the research note and project memory: **final accuracy > user-facing
simplicity >> training speed**. The design must stay robust from small to medium datasets
with an overfit-safe default; it therefore adds a *low-parameter* surface and keeps
`r=16`/`alpha=32` unchanged.

This is a single, coherent feature — a new scope plus the one resolution axis it needs.
The broader "adapt more of the network" framing in issue #230 (trunk MLP/FFN, neck,
mask-decoder) is explicitly **deferred** to a future medium tier (§9), per the
maintainer decision recorded in the research note (§7).

## 2. Background: what is and is not reachable today

Confirmed against the installed SAM 3.1 source
(`/.venv/lib/python3.12/site-packages/sam3/model/decoder.py`,
`TransformerDecoderLayer`):

- `self.ca_text = nn.MultiheadAttention(...)` — line 54. **Genuine torch MHA.**
- `self.self_attn = nn.MultiheadAttention(...)` — line 59. **Genuine torch MHA.**
- `self.cross_attn = cross_attention` — line 47. The injected image cross-attention is a
  `RoPEAttention` (from `sam3.sam.transformer`), **not** MHA; its q/k/v are its own
  `nn.Linear` projections reachable by a `target_modules` regex. **Out of scope here**
  (medium tier, §9).

PyTorch `nn.MultiheadAttention` packs q/k/v into a single `in_proj_weight`
(`nn.Parameter`) and exposes only `out_proj` as an `nn.Linear`. The current
`SCOPE_TARGETS["vision_decoder"]` (lora.py:52-56) matches:

```text
backbone\.vision_backbone\.trunk\.blocks\.\d+\.attn\.(qkv|proj)$
transformer\.decoder\.layers\.\d+\.(self_attn|cross_attn|ca_text)\.out_proj$
transformer\.decoder\.layers\.\d+\.linear[12]$
```

So `ca_text.out_proj` and `self_attn.out_proj` are adapted under plain LoRA, but the
`in_proj_weight` of both — the text-mixing and query-mixing q/k/v — is frozen and
unreachable by the module-name matcher. The `transformer\.decoder\.layers\.\d+\.` prefix
used by the existing out_proj regex is exactly the prefix the new in_proj parameter
patterns will use; the named-parameter paths are:

```text
transformer.decoder.layers.<N>.ca_text.in_proj_weight
transformer.decoder.layers.<N>.self_attn.in_proj_weight
```

`peft` 0.19.1 (installed; verified by introspection) exposes both `target_modules` and
`target_parameters` on `LoraConfig`. `target_parameters` is the API built specifically to
LoRA-adapt bare `nn.Parameter`s such as `nn.MultiheadAttention.in_proj_weight`. So the gap
closes at config level — **no module rewrite, no new layers** — consistent with the
project's "stay true to SAM" constraint. It does carry known MHA sharp edges, which the
gating spike (§7) must resolve before the surface is committed.

## 3. Design overview

Four coordinated changes for the in_proj surface, in dependency order — plus one
orthogonal calibrate fix (§7a) that ships in the same issue:

1. **Feasibility spike (§7, FIRST work item):** confirm `target_parameters` LoRA on a real
   SAM 3.1 decoder layer's `ca_text` / `self_attn` `in_proj_weight` attaches, forwards, and
   merges in **both** plain-LoRA and QLoRA modes. The spike result gates whether the
   in_proj surface ships enabled (full design) or gated (fallback, §7.3). The rest of the
   spec is written assuming the spike passes; §7.3 defines the contingency.

2. **New resolution axis (§5):** add a per-scope **parameter-name** target set alongside
   the existing module-name set, resolve it against `base.named_parameters()`, and pass the
   resolved names to `LoraConfig(target_parameters=...)` in both `apply_lora` and the QLoRA
   apply path. Add a `PEFTConfig.target_parameters: list[str] | None` override mirroring
   `target_modules`.

3. **New scope + new default (§4, §6):** add one `LoraScope` literal that equals
   `vision_decoder`'s module patterns **plus** the two in_proj parameter patterns; make it
   the new `PEFTConfig.scope` default; leave `vision` / `vision_decoder` / `all` untouched.

4. **Fixtures + tests (§10):** expose `ca_text` / `self_attn` as `nn.MultiheadAttention`
   (with `in_proj_weight`) under `transformer.decoder.layers.<N>` in the LoRA stub so the
   CPU predicate tests resolve the new parameters under both LoRA and QLoRA.

5. **Calibrate alpha co-scale (§7a, orthogonal):** make the pre-flight VRAM calibrate
   autosize co-scale `alpha` whenever it reduces LoRA rank `r` below the configured value
   (preserving the configured `alpha:r` ratio), persist `alpha` alongside `r`, and warn.
   This is independent of the in_proj surface (§4-§7) — it has no dependency on the spike
   and touches only `calibrate_cmd.py` / `_config_rewrite.py` and their tests. It is bundled
   here because the new default trains more surface, making rank-fit reductions (and thus
   the stale-alpha bug it fixes) more likely. The runtime `oom.py` ladder is out of scope
   (§7a.4).

## 4. The new scope

### 4.1 Name

Add **one** literal to `LoraScope`: **`vision_decoder_concept`**.

Rationale (brief, per the "do not proliferate scopes" constraint): the name reads as a
superset of `vision_decoder` (which it literally is, at the module level) with the added
intent — adapting the **concept** (text cross-attention) and instance-separation pathways.
It signals "everything `vision_decoder` does, plus the text/concept-mixing q/k/v" without
inventing an opaque tier word like `aggressive`. Exactly one new scope is added; no others.

### 4.2 Contents

`vision_decoder_concept` carries **both** target sets:

- **Module patterns** (identical to `SCOPE_TARGETS["vision_decoder"]`, matched against
  `named_modules()`):

  ```text
  backbone\.vision_backbone\.trunk\.blocks\.\d+\.attn\.(qkv|proj)$
  transformer\.decoder\.layers\.\d+\.(self_attn|cross_attn|ca_text)\.out_proj$
  transformer\.decoder\.layers\.\d+\.linear[12]$
  ```

- **Parameter patterns** (new, matched against `named_parameters()`):

  ```text
  transformer\.decoder\.layers\.\d+\.ca_text\.in_proj_weight$
  transformer\.decoder\.layers\.\d+\.self_attn\.in_proj_weight$
  ```

The two parameter patterns deliberately target only `ca_text` and `self_attn`. The third
decoder attention (`cross_attn`, a `RoPEAttention`) is *not* MHA and is not an in_proj
target here; its q/k/v go to the medium tier (§9).

### 4.3 Scopes left unchanged

`vision`, `vision_decoder`, and `all` keep their exact current module patterns and carry an
**empty** parameter target set (no `target_parameters`). This preserves byte-for-byte
reproducibility for existing configs that pin those scopes — the issue's explicit
reproducibility concern. The change is purely additive: a new scope literal, not a
mutation of shipped ones.

## 5. The `target_parameters` resolution axis

### 5.1 Data structure

Today `SCOPE_TARGETS: dict[str, list[str]]` maps scope → list of module-name regexes
(lora.py:36-60), and `_resolve_targets` matches them against `nn.Linear` (or `Linear4bit`)
modules. The new axis needs a parallel mapping of scope → list of **parameter-name**
regexes.

**Chosen representation: a second module-level mapping, `SCOPE_TARGET_PARAMETERS`**, kept
adjacent to `SCOPE_TARGETS` in `lora.py`. Rationale:

- It mirrors the existing "single point of contact for SAM naming" comment block exactly:
  the docstring already calls `SCOPE_TARGETS` (with `meta_to_canonical`) the two
  single-points-of-contact for SAM surface naming. A sibling dict extends that contract
  with the parameter axis in the same place and the same style, rather than introducing a
  new per-scope dataclass that would force every existing scope entry to be rewritten.
- It keeps the additive, reproducibility-preserving shape: only `vision_decoder_concept`
  appears in the new dict; scopes absent from it resolve to "no parameter targets."

```python
# Parallel to SCOPE_TARGETS: scope -> regexes matched against named_parameters().
# Reaches bare nn.Parameter q/k/v packed in nn.MultiheadAttention.in_proj_weight,
# which target_modules cannot see. Only the concept scope populates it; absent
# scopes carry no parameter targets (reproducibility for vision/vision_decoder/all).
SCOPE_TARGET_PARAMETERS: dict[str, list[str]] = {
    "vision_decoder_concept": [
        r"transformer\.decoder\.layers\.\d+\.ca_text\.in_proj_weight$",
        r"transformer\.decoder\.layers\.\d+\.self_attn\.in_proj_weight$",
    ],
}
```

`SCOPE_TARGETS["vision_decoder_concept"]` is added in the same edit and equals the
`vision_decoder` module list (§4.2). Lookups into `SCOPE_TARGET_PARAMETERS` use
`.get(scope, [])` so the three legacy scopes return an empty list without a `KeyError`.

### 5.2 New resolver: `_resolve_target_parameters`

Add a sibling to `_resolve_targets`, matching parameter-name patterns against
`base.named_parameters()`:

```python
def _resolve_target_parameters(base: nn.Module, cfg: PEFTConfig) -> list[str]:
    """Resolve scope/override parameter-name patterns against named_parameters().

    Precedence mirrors _resolve_targets:
      * cfg.target_parameters is not None -> use it verbatim (overrides scope).
      * else -> SCOPE_TARGET_PARAMETERS.get(cfg.scope, []).
    Returns the full matched parameter names (e.g.
    'transformer.decoder.layers.0.ca_text.in_proj_weight') to pass to
    LoraConfig(target_parameters=...). Returns [] when the resolved pattern
    list is empty (legacy scopes). Raises ValueError only when a NON-EMPTY
    pattern list matches zero parameters.
    """
```

Behavior:

- `patterns = cfg.target_parameters if cfg.target_parameters is not None else
  SCOPE_TARGET_PARAMETERS.get(cfg.scope, [])`.
- If `patterns` is empty → return `[]` (no parameter targets; the common case for legacy
  scopes). This is **not** an error — it is the documented "this scope has no in_proj
  surface" state.
- Else compile and search each pattern against every `name` from
  `base.named_parameters()`; collect full matched names.
- If `patterns` is non-empty but **zero** names match → raise `ValueError` listing the
  patterns tried and the first 50 parameter names actually present. This mirrors the
  existing `_resolve_targets` no-match `ValueError` (lora.py:78-84) so a typo'd or
  SAM-renamed parameter path fails loudly instead of silently training nothing on the
  in_proj surface (§8).

This asymmetry — empty-list is fine, non-empty-no-match is an error — is the key
distinction from `_resolve_targets`, where an empty module set is always an error. The
difference exists because parameter targets are an *optional* second axis that most scopes
legitimately do not use, whereas every scope must match at least one module.

### 5.3 `apply_lora` wiring (lora.py)

In `apply_lora` (lora.py:88-137), after resolving `matched_names`:

1. `matched_params = _resolve_target_parameters(base, cfg)`.
2. Build `LoraConfig` passing **both** axes:

   ```python
   lora_cfg = LoraConfig(
       r=cfg.r,
       lora_alpha=cfg.alpha,
       lora_dropout=cfg.dropout,
       target_modules=matched_names,
       target_parameters=(matched_params or None),
       bias=cfg.bias,
       task_type=None,
   )
   ```

   Pass `None` (not `[]`) when there are no parameter targets, so the LoRA config for the
   three legacy scopes is byte-identical to today's (reproducibility — `target_parameters`
   defaults to `None` in peft).
3. The trainable-ratio log line (lora.py:123-130) gains the in_proj count for visibility;
   e.g. extend the existing `n_targets=%d` with an `n_param_targets=%d` field. The >10 %
   warning logic (lora.py:131-136) is unchanged in structure (see §8.3 on the threshold).

`get_peft_model` then attaches LoRA to both the matched `nn.Linear` modules and the matched
in_proj parameters in one `PeftModel`. No change to `save_lora` / `load_lora` /
`merge_lora`: peft persists and merges `target_parameters` adapters through the same
`save_pretrained` / `from_pretrained` / `merge_and_unload` surface the module adapters use.

### 5.4 QLoRA wiring (qlora.py)

In `_inject_lora_adapters` (qlora.py:224-249), the LoRA config is built after
`_resolve_targets(model, cfg, linear_types=(bnb.nn.Linear4bit,))`. Add the parameter axis:

1. `lora_param_names = _resolve_target_parameters(model, cfg)` (imported from `lora.py`
   alongside `_resolve_targets`; the one-way `qlora.py -> lora.py` import contract is
   preserved — lora.py still never imports qlora.py or bitsandbytes).
2. Pass `target_parameters=(lora_param_names or None)` into the `LoraConfig` here too.

**Why this is correct under QLoRA (the coexistence guarantee, §7.2):** `_mha_exclusion_types`
(qlora.py:59-96) deliberately keeps `nn.MultiheadAttention` children **unquantized**, and
`in_proj_weight` is a bare `Parameter` that is never quantized regardless. So under QLoRA:

- The `Linear4bit` LoRA attaches to FFN/trunk linears via `target_modules` (as today).
- The in_proj LoRA attaches to the **unquantized bf16** `in_proj_weight` parameters via
  `target_parameters` — it is *plain* LoRA-on-bf16 even in QLoRA mode.
- Both adapters live in **one** `PeftModel` and must merge together via the existing
  `merge_lora` (`merge_and_unload`) path.

The `_resolve_target_parameters` call resolves against the **same** parameter names in
both modes (in_proj is never swapped to `Linear4bit`), so the parameter set is identical
LoRA vs QLoRA — only the module set differs (out_proj drops out under QLoRA, as already
documented in qlora.py:113-120). The QLoRA log line (qlora.py:278-286) gains the same
`n_param_targets` field for parity.

### 5.5 Control / data flow summary

```text
PEFTConfig(scope=..., target_modules=..., target_parameters=...)
        |
        v
apply_lora / apply_qlora
        |
   +----+--------------------------+
   |                               |
_resolve_targets               _resolve_target_parameters
(named_modules, nn.Linear      (named_parameters, by name regex;
 or Linear4bit; ValueError      [] allowed; ValueError only on
 on zero match)                 non-empty-no-match)
   |                               |
   v                               v
matched_names                  matched_params
   |                               |
   +-------------+-----------------+
                 v
   LoraConfig(target_modules=matched_names,
             target_parameters=(matched_params or None), ...)
                 v
        get_peft_model -> one PeftModel
                 v
   forward / merge_and_unload (both axes folded together)
```

## 6. Schema changes (`src/custom_sam_peft/config/schema.py`)

### 6.1 `LoraScope` literal (schema.py:99)

```python
LoraScope = Literal["vision", "vision_decoder", "vision_decoder_concept", "all"]
```

### 6.2 New default `scope` (schema.py:496)

```python
scope: LoraScope = "vision_decoder_concept"
# tbd: #230 (project-chosen SAM 3.1 concept scope; default flipped from
#      vision_decoder so the shipped default can learn niche TEXT concepts —
#      vision_decoder freezes ca_text/self_attn in_proj. See research note §4, §7.)
```

This **replaces** the prior `scope: LoraScope = "vision_decoder"  # tbd: #191 ...`
annotation. Update the inline `# tbd:` to reference #230 and state the rationale, per the
cite/tbd discipline (every changed default needs a tag).

**Reproducibility implication (must be documented in the spec and in a code comment):**
flipping the default changes what an existing config *without* an explicit `peft.scope`
adapts — it now additionally adapts `ca_text` / `self_attn` in_proj. Configs that pin
`scope: vision_decoder` (or `vision` / `all`) are entirely unaffected. This is the
intended behavior change (the shipped default could not previously learn text concepts),
but it is a behavior change and is called out as such in §11 acceptance and in the example
configs (§6.4).

### 6.3 New `target_parameters` override field (schema.py PEFTConfig, alongside
`target_modules` at schema.py:498-504)

```python
target_parameters: list[str] | None = Field(
    default=None,
    description=(
        "Explicit list of parameter-name patterns to adapt via LoRA "
        "target_parameters (e.g. nn.MultiheadAttention in_proj_weight). When "
        "None, apply_lora uses SCOPE_TARGET_PARAMETERS.get(scope, []). When set, "
        "overrides the scope's parameter patterns; independent of target_modules."
    ),
)
```

**Override interaction (documented on the field and enforced in the resolvers, not in
pydantic):**

- `target_modules` and `target_parameters` are **independent** axes. Each overrides only
  its own axis when set; neither affects the other.
- `target_parameters is None` → use `SCOPE_TARGET_PARAMETERS.get(scope, [])`.
- `target_parameters` is a list → use it verbatim, ignore the scope's parameter patterns.
  (Same precedence shape as `target_modules` vs `SCOPE_TARGETS[scope]`.)
- A user may set `target_modules` alone (override modules, keep scope parameters),
  `target_parameters` alone (override parameters, keep scope modules), both, or neither.
  Pydantic accepts any combination; the two resolvers apply precedence at apply time.
- Edge case to document: setting `target_parameters` to a non-empty list whose patterns
  match nothing raises `ValueError` (§5.2); setting it to `[]` explicitly means "no
  parameter targets" and is valid (returns `[]`, no error).

### 6.4 Example configs

Update the LoRA/text example config(s) under `configs/examples/` that carry commented PEFT
knobs (the LoRA spec §7 added a commented block) to:

- List `vision_decoder_concept` as the default in the `# scope:` comment line and note it
  is the new shipped default.
- Add a commented `# target_parameters: [...]  # overrides scope's in_proj patterns`
  knob mirroring the existing commented `# target_modules:` knob.

No uncommented value changes — defaults already apply; this is documentation of the new
lever only.

## 7. Gating feasibility spike (FIRST work item)

This is a **distinct, first** piece of work. The in_proj surface is not committed until the
spike passes; everything in §4–§6 is contingent on it.

### 7.1 What the spike must confirm

`peft` + `nn.MultiheadAttention` has known sharp edges: MHA's `forward` calls
`F.linear(x, in_proj_weight, ...)` and accesses `self.out_proj.weight` directly rather than
dispatching through a child module's `forward`, so peft needs MHA-specific handling and
integrations have hit `AttributeError` / merge-path issues (research note §4). peft 0.19.1
*does* ship a dedicated `peft.tuners.lora.layer.MultiheadAttention` support path — note it
as the **primary route** for adapting MHA, and treat `target_parameters` **correctness** as
the thing to confirm, not assume.

The spike runs on the **real** SAM 3.1 decoder (a single `TransformerDecoderLayer`, or the
full model under the gated GPU markers — see §7.4) and confirms, for `ca_text` and
`self_attn` `in_proj_weight`:

1. **Attach:** `apply_lora` with `scope="vision_decoder_concept"` produces a `PeftModel`
   whose `named_parameters()` contains LoRA params for both in_proj parameters (in addition
   to the module targets), with no `AttributeError` during `get_peft_model`.
2. **Forward:** a forward pass through the wrapped layer/model runs without error and the
   in_proj LoRA params receive gradients (finite `lora_A.grad`).
3. **Merge:** `merge_lora` (`merge_and_unload`) folds **both** the module and the
   parameter adapters back into the base without raising, and the merged base differs from
   the pre-merge base on the in_proj weights.
4. **Both modes:** items 1–3 hold under **plain LoRA** and under **QLoRA** (§7.2).

The spike's output is a written go/no-go on the in_proj surface, recorded in the PR. The
spec must **not** assume the spike passes silently — §7.3 is mandatory contingency.

### 7.2 QLoRA coexistence (hard requirement)

Under QLoRA the in_proj LoRA is plain LoRA-on-bf16 (in_proj stays unquantized) while the
rest is `Linear4bit` LoRA. The spike must verify **both adapters attach and merge in ONE
`PeftModel`** under QLoRA: `apply_qlora` with `scope="vision_decoder_concept"` attaches the
`Linear4bit` module LoRA *and* the bf16 in_proj parameter LoRA; forward runs; `merge_lora`
folds both (dequantizing the 4-bit base per `merge_lora`'s documented behavior) without a
dtype/packed-weight error. This is a hard requirement, not a nice-to-have.

### 7.3 Fallback / contingency (mandatory)

If the spike shows peft cannot cleanly attach + forward + merge MHA in_proj in this stack:

- **Option (a) — peft MHA-specific support:** route through peft's dedicated
  `lora.layer.MultiheadAttention` support path instead of (or in addition to)
  `target_parameters`. If peft's MHA path adapts in_proj correctly when the MHA module
  itself is named as a `target_module`, the scope can express the surface that way and
  `target_parameters` becomes an internal detail or is dropped. Re-run §7.1 items 1–4
  against this route.
- **Option (b) — ship infrastructure, gate the surface:** land the full scope +
  `target_parameters` config axis + resolvers + fixtures + tests (everything except a
  *working* in_proj attach), but keep `SCOPE_TARGET_PARAMETERS["vision_decoder_concept"]`
  **empty** (or the scope's parameter patterns commented out) so the new scope is, for now,
  behaviorally equal to `vision_decoder` and the in_proj surface is *gated off* behind a
  documented `# tbd:` pending a peft fix. In this fallback the default still flips to
  `vision_decoder_concept` (so re-enabling later is a one-line change), and the
  reproducibility note (§6.2) is adjusted to say the default is currently equivalent to
  `vision_decoder` until the in_proj surface is enabled.

The choice between (a) and (b) is made from the spike result and recorded in the PR. The
planner should sequence the spike as the first phase so the rest of the work proceeds with
a known outcome.

### 7.4 Where the spike lives

The real-decoder verification runs under the existing GPU/checkpoint markers
(`requires_checkpoint`, `requires_compatible_gpu`) used by
`tests/integration/test_peft_{lora,qlora}_real.py` — SAM 3.1's `PositionEmbeddingSine`
hardcodes `device="cuda"`, so a real-model forward needs a compatible GPU. The CPU stub
tests (§10) cover the resolution logic; the GPU integration tests cover attach + forward +
merge on real module names. The spike extends those real tests with in_proj assertions
rather than adding a separate harness.

## 7a. VRAM-autosize rank/alpha handling

This requirement is orthogonal to the in_proj surface (§4-§7) but ships in the same
issue (#230). It governs the **pre-flight VRAM calibrate autosize** only — the path that
already reduces LoRA rank `r` to fit the GPU — and makes that path co-scale `alpha`
instead of leaving it stale.

### 7a.1 Pinning reaffirmed (no dynamic rank/alpha)

`r=16` / `alpha=32` remain the **pinned cited defaults** (LoRA Hu 2021 §4.1; the
`alpha = 2r` convention). #230 does **not** make rank or alpha dynamic, scope-dependent,
or otherwise reactive to the chosen scope. The new `vision_decoder_concept` scope uses the
same `r` / `alpha` as every other scope. **The only place rank ever changes is the
pre-flight VRAM calibrate autosize** (`calibrate_cmd.py`), exactly as today — this
requirement does not add any new rank-changing site.

### 7a.2 The problem: calibrate reduces `r` but never touches `alpha`

`run_calibration` → `_confirm_and_climb` (calibrate_cmd.py) probes the configured
`(method, r, batch, k)` and, on OOM / over-budget, shrinks down the documented sacrifice
order `batch -> K -> r -> method (LoRA->QLoRA)`. The final LoRA rank `r_final` can land
**below** the user's configured `cfg.peft.r` from any of three reductions:

1. the analytic Stage-2 aim already choosing a smaller `r`,
2. the OOM shrink walk stepping `r` down `_RS[i] -> _RS[i-1]`
   (`_confirm_and_climb` step 3), or
3. the `LoRA -> QLoRA` flip, which resets `r = _RS[-1]` and then lets the loop shrink it
   (`_confirm_and_climb` step 4).

Today the calibrate path reads only `cfg.peft.r` (calibrate_cmd.py ~L432); it never reads
`cfg.peft.alpha`. It persists `r` (via `PresetDecision.r`, the v3 `chosen_r` cache key, and
`_rewrite_sizing_block`'s `peft.r` line) but **never `alpha`**. So a user who configured
`r=16` / `alpha=32` and gets autosized to `r_final=8` is left with `alpha=32` against
`r=8` — an `alpha:r` ratio of 4:1, double the cited `alpha = 2r` convention, silently
changing the effective LoRA scaling `alpha / r`. This is a latent correctness bug the
in_proj work surfaces (the new default trains more surface, making rank-fit reductions more
likely), and #230 fixes it.

### 7a.3 Requirement: co-scale `alpha`, persist it, and warn

When `_confirm_and_climb` selects a final LoRA rank `r_final` that is **less than** the
user's configured `cfg.peft.r`, the calibrate path MUST:

- **(a) Co-scale alpha to preserve the configured `alpha:r` ratio.** Compute
  `alpha_final = round(cfg.peft.alpha * r_final / cfg.peft.r)`.
  - For the cited default (`alpha = 2r`, i.e. `alpha=32` at `r=16`) this is exactly
    `alpha_final = 2 * r_final` (e.g. `r_final=8 -> alpha_final=16`).
  - A user who configured a **non-2r** ratio keeps **their** ratio — the formula scales
    whatever `cfg.peft.alpha / cfg.peft.r` the user set. Do **not** force `alpha = 2r`.
  - Use integer `round(...)`; `alpha_final` must stay a positive int (matches the
    `PositiveInt` schema field). For the cited 2r default the multiply is exact, so no
    rounding loss occurs there.

- **(b) Persist both `r` and `alpha`.** Extend the calibrate persistence chain so `alpha`
  rides alongside `r` end-to-end:
  - `PresetDecision` gains an `alpha: int` field (placed adjacent to `r`).
  - The v3 cache gains a `chosen_alpha` key, written by `_write_cache_v3` (additive, same
    optional shape as the other `chosen_*` keys) and read back by `_decision_from_cache`.
  - `_rewrite_sizing_block` gains a `peft.alpha` target in its `replacements` map, so the
    config's `peft.alpha:` line is rewritten alongside `peft.r:` (today it is not). This is
    a 6th direct (section, key) target; mirror the existing `peft.r` handling exactly
    (line surgery, preserve inline comments, idempotent annotation).
  - `_apply_config_rewrite` passes `decision.alpha` through to `_rewrite_sizing_block`.

- **(c) Emit a user-facing WARNING** via `typer.echo(..., err=True)` (matching the existing
  calibrate WARNING style), naming the change in both rank and alpha:

  ```text
  WARNING: VRAM autosize reduced LoRA rank r {cfg.r}->{r_final} to fit {gpu_name};
  alpha co-scaled {cfg.alpha}->{alpha_final} to preserve alpha/r scaling.
  ```

  Fire it once, when the reduction is finalized (after `_confirm_and_climb` returns the
  empirical tuple and `r_final < cfg.peft.r` is detected), on the probe path. The
  cache-fresh early-return path that reconstructs a prior decision from `chosen_*` keys does
  not re-warn (the decision — and its already-co-scaled alpha — was finalized on the
  original probe run; re-warning on every cache-fresh re-run would be noise).

- **(d) No-op when not reduced.** If `r_final == cfg.peft.r` (autosize left rank alone),
  do **not** warn and leave `alpha` exactly as configured — `alpha_final = cfg.peft.alpha`.
  The cache write, config rewrite, and emitted output for this case must be **byte-identical
  to today's** for `alpha` (the new `chosen_alpha` key simply records the unchanged
  configured alpha; no WARNING line is added). Autosize that does not reduce rank is
  observably unchanged.

`r_final > cfg.peft.r` cannot occur: calibrate's climb never raises `r` above the
configured aim (it only grows K then batch — `_confirm_and_climb` climb phase, and the
analytic aim is bounded by the configured `r`). The co-scale is therefore strictly a
reduction-time concern; treat any `r_final >= cfg.peft.r` as the no-op (d) case.

### 7a.4 Runtime OOM ladder is explicitly OUT of this requirement

`oom.py::OomLadder` is **unaffected** by this requirement and MUST NOT be touched for it.
The runtime ladder sacrifices micro-batch `B` then multiplex `K` only (actions
`microbatch_halved` / `multiplex_halved`), already emits warnings via `_LOG.warning`, and
**never changes LoRA rank** (and therefore never needs to co-scale alpha). This requirement
is solely the **pre-flight calibrate autosize** in `calibrate_cmd.py` /
`_config_rewrite.py`. An implementer must not add rank/alpha handling to the runtime ladder.

### 7a.5 Cite / tbd discipline for the co-scale

Co-scaling `alpha` to preserve the configured ratio (`alpha = 2r` for the default) is
**justified by the existing `alpha = 2r` citation** (LoRA Hu 2021 §4.1) — it requires **no
new `# tbd:` tag**. The co-scale *restores* the cited convention at the autosized rank;
the status-quo behavior (leaving `alpha` fixed while `r` drops) is what *violates* the
cited convention. No new hyperparameter is introduced — `alpha_final` is a deterministic
function of the already-cited `cfg.alpha`, `cfg.r`, and the probe-chosen `r_final`. This is
recorded explicitly in the §12 table.

## 8. Error handling

| Condition | Behavior |
| --- | --- |
| `cfg.target_parameters is None`, scope has no parameter patterns (legacy scopes) | `_resolve_target_parameters` returns `[]`; `LoraConfig` gets `target_parameters=None`. No error — this is the normal legacy path. |
| Non-empty parameter pattern list matches zero `named_parameters()` | `ValueError` listing patterns tried + first 50 parameter names present (mirrors `_resolve_targets` lora.py:78-84). Surfaces a typo or SAM rename loudly; never silently trains nothing on in_proj. |
| `cfg.target_parameters = []` (explicit empty) | Returns `[]`; no error (explicit "no parameter targets"). |
| `cfg.target_modules` matches zero modules | Unchanged: existing `_resolve_targets` `ValueError`. |
| Scope not a known literal | Unchanged: pydantic blocks it upstream (`LoraScope` is a `Literal`); `SCOPE_TARGET_PARAMETERS.get(scope, [])` would also not `KeyError`. |
| peft cannot attach/merge MHA in_proj | Spike fallback §7.3 — not a runtime error path; resolved before ship. |

### 8.1 No-match parity

The existing `_resolve_targets` raising a helpful `ValueError` on zero matches is preserved
verbatim for modules and **replicated** for parameters (with the empty-pattern-list
exception of §5.2). The error message format is the same shape: patterns tried + sample of
real names present.

### 8.2 QLoRA dtype safety

No new dtype handling is required: in_proj is never quantized (it is a bare bf16
`Parameter`, and `_mha_exclusion_types` keeps the whole MHA unquantized), so the
`target_parameters` LoRA is plain bf16 LoRA. The dtype-collision footguns documented in
qlora.py (`_freeze_non_adapter`, the deliberate skip of `prepare_model_for_kbit_training`)
are unchanged by this spec. The spike (§7.2) is the place this is *verified*, not assumed.

### 8.3 Trainable-ratio guard

The >10 % warning (lora.py:131-136, qlora.py:287-292) stays. The two in_proj parameters per
decoder layer are low-parameter (a LoRA pair on each `in_proj_weight`), so the
post-change ratio should remain well under 10 %. The implementation must **empirically
confirm** the post-change trainable ratio on the real model (the GPU integration test
already asserts `ratio < 0.05`, lora.py real test) and only adjust the 10 % threshold or
its comment **if reality demands it** — the default assumption is no change. A new
hyperparameter (a changed threshold) would require a `# cite:`/`# tbd:` tag; leaving it at
10 % needs none.

## 9. Out of scope / future medium tier (document, do not build)

Recorded per the research note §7 as a documented future "medium" tier (separate issue),
**not** built here:

- **ViT trunk MLP/FFN** `backbone...blocks.N.mlp.{fc1,fc2}` — niche appearance capacity;
  FFN is ~2/3 of transformer params, so it raises overfitting risk on small data and fights
  the accuracy-on-small-data priority. Add only with data scale.
- **Image `cross_attn` (RoPEAttention) q/k/v** via `target_modules` regex — the
  small-object **localization** lever (DETR cross-attention "locks queries onto regions").
  A different mechanism (module regex, not `target_parameters`) since RoPEAttention exposes
  its own `nn.Linear` projections.
- **Conv2d neck adaptation** (`Sam3DualViTDetNeck` / `Sam3TriViTDetNeck`) — the multi-scale
  small-object axis; requires extending the matcher's `linear_types` to LoRA-on-Conv2d.
- **SAMed / Conv-LoRA-style full-fine-tune of the small mask-decoder head** — a full-FT
  mechanism, not a scope pattern.

Also explicitly **excluded** (out of "stay true to SAM"): Conv-LoRA conv-experts, VPT
concept tokens, text-encoder adaptation, and the non-PEFT resolution / feature-fusion
levers (research note §5, §7).

## 10. Testing and fixtures

### 10.1 Fixture changes — `tests/fixtures/tiny_sam3_lora_stub.py`

The current `_DecoderLayer` (tiny_sam3_lora_stub.py:43-47) uses `_DecoderAttn`
(`q/k/v/out_proj` as separate `nn.Linear`s) for `self_attn` and `cross_attn`. The CPU
predicate tests must resolve the new `target_parameters` patterns, which require real
`nn.MultiheadAttention` children exposing `in_proj_weight`.

**Required fixture change:** under the decoder layer subtree, expose `ca_text` and
`self_attn` as `nn.MultiheadAttention` modules (each has a real `in_proj_weight`
`nn.Parameter`), at paths the new parameter patterns match. Because the stub uses truncated
prefixes (`transformer_decoder` rather than the real `transformer.decoder`) and the tests
drive resolution via `FIXTURE_SCOPE_PATTERNS` (tiny_sam3_lora_stub.py:131-138) rather than
the production `SCOPE_TARGETS`, add a parallel `FIXTURE_SCOPE_TARGET_PARAMETERS` mapping
with the fixture-prefixed parameter patterns, e.g.:

```text
transformer_decoder\.layers\.\d+\.ca_text\.in_proj_weight$
transformer_decoder\.layers\.\d+\.self_attn\.in_proj_weight$
```

Specifics the implementer must honor:

- `ca_text` and `self_attn` become `nn.MultiheadAttention(dim, n_heads)` so `in_proj_weight`
  exists under `transformer_decoder.layers.0.{ca_text,self_attn}.in_proj_weight`.
- Keep `cross_attn` as a non-MHA attention (it must **not** match the in_proj parameter
  patterns — it is the negative control for the parameter axis).
- The `working=True` forward path must still exercise at least one LoRA-targeted module so
  the existing forward/backward grad test (test_peft_scope_coverage.py:95-129) keeps
  working; if the MHA forward is awkward to wire into the stub's forward, the in_proj grad
  check can be a separate structural+grad test using a minimal MHA forward, consistent with
  how the stub already separates structural (`working=False`) from forward (`working=True`)
  modes.
- `FIXTURE_SCOPE_PATTERNS["vision_decoder_concept"]` is added (equal to the
  `vision_decoder` fixture module patterns) so module-axis tests can drive the new scope.

The fixture is the second of the two single-points-of-contact for SAM naming (the
production `SCOPE_TARGETS` / `SCOPE_TARGET_PARAMETERS` is the first); the parameter axis is
mirrored in both.

### 10.2 CPU unit tests (extend `tests/unit/test_peft_scope_coverage.py` and/or a new
`tests/unit/test_peft_target_parameters.py`)

Drive resolution on the stub via the fixture pattern mappings (the existing tests pass
`target_modules=FIXTURE_SCOPE_PATTERNS[scope]`; the new ones additionally pass
`target_parameters=FIXTURE_SCOPE_TARGET_PARAMETERS[scope]`). Enumerated cases:

| Test | Asserts |
| --- | --- |
| New scope resolves expected modules **and** parameters | `scope="vision_decoder_concept"` attaches LoRA to the vision-trunk + decoder out_proj/FFN modules **and** to `ca_text.in_proj_weight` + `self_attn.in_proj_weight`; `cross_attn.in_proj_weight` is absent (it is not MHA / not targeted). |
| Existing scopes unchanged | `vision` / `vision_decoder` / `all` produce **no** `target_parameters` (empty resolution) and the same module set as before — byte-identical `LoraConfig` shape (reproducibility). |
| Default is the new scope | `PEFTConfig(method="lora").scope == "vision_decoder_concept"`. |
| `target_parameters` override | Setting `cfg.target_parameters=[<one pattern>]` adapts exactly that parameter and ignores the scope's parameter patterns; setting `target_modules` does **not** change the resolved parameter set (axis independence). |
| Non-empty no-match raises | `target_parameters=["nonexistent.param"]` raises `ValueError` whose message lists patterns tried + a real parameter name. |
| Empty override is valid | `target_parameters=[]` resolves to no parameter targets without error. |
| Trainable ratio sane | Ratio under `vision_decoder_concept` on the stub stays a small bound (consistent with the existing `< 0.05` style assertion). |
| Forward/backward grad on in_proj LoRA | Under `working=True`, the in_proj `lora_A` params are in the gradient graph with finite grads (wiring assertion, mirroring test_peft_scope_coverage.py:95-129). |

### 10.3 QLoRA CPU coverage

Mirror the parameter-resolution assertions for the QLoRA path where they can run on CPU
(resolution is pure name-matching and does not need bitsandbytes): assert
`_resolve_target_parameters` returns the **same** parameter set for the QLoRA config as for
the LoRA config on the same stub (the parameter axis is mode-independent; only the module
axis differs). The real attach + coexist + merge under QLoRA is GPU-gated (§10.4).

### 10.4 GPU integration tests (gated)

Extend `tests/integration/test_peft_lora_real.py` and `test_peft_qlora_real.py` (under the
existing `requires_checkpoint` + `requires_compatible_gpu` markers) to assert, on the real
SAM 3.1 model with `scope="vision_decoder_concept"`:

- LoRA params exist for `ca_text.in_proj_weight` and `self_attn.in_proj_weight` (both
  modes).
- Forward runs; `merge_lora` folds both axes without error (both modes — the QLoRA
  coexistence requirement, §7.2).
- Trainable ratio stays under the existing budget (`< 0.05`), empirically confirming §8.3.

These tests are the productionized form of the §7 spike.

### 10.5 Calibrate alpha co-scale + WARNING (CPU, extend `tests/unit/test_calibrate_cmd.py`)

These cover §7a. They run on **CPU** the same way the existing calibrate tests do — the
probe is mocked (`monkeypatch.setattr(calibrate_cmd, "_run_probe", ...)`) and
`torch.cuda.*` is stubbed (`is_available`, `get_device_*`, `max_memory_allocated`,
`reset_peak_memory_stats`), so no GPU and no real model load are required. Assert on the
emitted WARNING via `capsys` / captured stderr (the existing tests already capture
`typer.echo(..., err=True)` output) and on the written cache / rewritten config.

| Test | Asserts |
| --- | --- |
| Reduction co-scales alpha + warns | A config with `r=16` / `alpha=32` whose probe forces `r_final < 16` (e.g. mocked peaks that fit only at a smaller `r`, or the LoRA→QLoRA flip) yields `alpha_final = 2 * r_final`, and the WARNING fires naming `r {16}->{r_final}` and `alpha {32}->{alpha_final}`. |
| No reduction → no warn, alpha untouched | When the mocked probe fits at the configured `r` (`r_final == 16`), **no** WARNING line is emitted and the persisted/rewritten `alpha` equals the configured `32` (byte-identical-to-today behavior for alpha). |
| Config rewrite + v3 cache round-trip persist co-scaled alpha | After a reducing run, the rewritten config's `peft.alpha:` line and the v3 cache `chosen_alpha` key both hold `alpha_final`; a **fresh-cache re-run** (`_decision_from_cache`) reproduces both `r_final` **and** `alpha_final` (and does not re-warn). |
| Custom (non-2r) ratio preserved | A config with `r=16` / `alpha=16` (1:1, non-2r) reduced to `r_final=8` yields `alpha_final=8` (preserving the user's 1:1 ratio), **not** `16` (it is not forced to `alpha = 2r`). |

## 11. Acceptance criteria

A correct implementation satisfies:

1. **Spike resolved first.** The in_proj feasibility spike (§7) has a recorded go/no-go;
   the shipped behavior matches its outcome (full surface, or fallback §7.3 with the
   surface gated and the reproducibility note adjusted).
2. **New scope.** `LoraScope` gains exactly one literal, `vision_decoder_concept`, equal to
   `vision_decoder`'s module patterns plus the two in_proj parameter patterns. `vision`,
   `vision_decoder`, `all` are byte-for-byte unchanged.
3. **New default.** `PEFTConfig.scope` defaults to `vision_decoder_concept` with an updated
   `# tbd: #230` annotation; the reproducibility implication for default-scope configs is
   documented in code and spec.
4. **Resolution axis.** `SCOPE_TARGET_PARAMETERS` + `_resolve_target_parameters` exist;
   both `apply_lora` and the QLoRA apply path pass resolved parameter names to
   `LoraConfig(target_parameters=...)`; legacy scopes pass `target_parameters=None`.
5. **Override field.** `PEFTConfig.target_parameters: list[str] | None = None` exists,
   mirrors `target_modules` precedence, is axis-independent from `target_modules`, and is
   documented (including the empty-list vs non-empty-no-match distinction).
6. **QLoRA coexistence.** In both LoRA and QLoRA modes the in_proj parameter LoRA and the
   module LoRA attach, forward, and `merge_and_unload` together in one `PeftModel` (or, in
   fallback §7.3(b), the surface is gated and this is deferred with the in_proj patterns
   empty).
7. **Error parity.** Non-empty parameter pattern lists that match nothing raise a helpful
   `ValueError`; empty resolution is not an error.
8. **Trainable-ratio guard.** The post-change ratio is empirically confirmed under 10 % (in
   practice under the test's 5 % budget); the threshold/comment is changed only if reality
   demands it, with a tag if so.
9. **Calibrate alpha co-scale (§7a).** When the pre-flight VRAM calibrate autosize selects
   `r_final < cfg.peft.r` (from the analytic aim, the OOM shrink walk, or the LoRA→QLoRA
   flip), the calibrate path co-scales `alpha_final = round(cfg.peft.alpha * r_final /
   cfg.peft.r)` (preserving the configured ratio, **not** forcing `alpha = 2r`), persists
   `alpha` alongside `r` through `PresetDecision`, the v3 `chosen_alpha` cache key, and
   `_rewrite_sizing_block`'s `peft.alpha` line, and emits a single user-facing WARNING
   (`typer.echo(..., err=True)`) naming both the `r` and the `alpha` change. When
   `r_final == cfg.peft.r` it does **not** warn and leaves `alpha` untouched
   (byte-identical to today). `oom.py::OomLadder` is **untouched** (it changes B/K only,
   never rank/alpha). The co-scale needs **no** new `# tbd:` — it is justified by the
   existing `alpha = 2r` citation. The §10.5 CPU tests (mocked probe) pass.
10. **Tests/fixtures.** The LoRA stub exposes `ca_text` / `self_attn` as
    `nn.MultiheadAttention`; the CPU predicate tests (§10.2-10.3, §10.5) and the gated GPU
    tests (§10.4) pass; coverage stays >= 80 %.
11. **Lint/type.** `ruff check`, `ruff format --check`, and `mypy --strict` pass on every
    touched file; the spec passes the repo markdownlint gate.

## 12. Chosen defaults (cite / tbd discipline)

Every new or changed default carries a `# cite:` or `# tbd:` tag (repo-enforced).

| Knob | Value | Tag / basis |
| --- | --- | --- |
| `r` | 16 (unchanged) | `# cite:` LoRA (Hu 2021) arXiv:2106.09685 §4.1 — **do not change** (only the pre-flight calibrate autosize reduces it, §7a) |
| `alpha` | 32 (unchanged) | `# cite:` LoRA (Hu 2021) §4.1 (`alpha = 2r`) — **do not change**; co-scaled by calibrate autosize to preserve the cited ratio (§7a) |
| calibrate autosize `alpha` co-scale | `alpha_final = round(cfg.alpha * r_final / cfg.r)` (preserve configured ratio; default `= 2 * r_final`) | **no new tag** — justified by the existing `alpha = 2r` citation (Hu 2021 §4.1); the co-scale *restores* the cited convention at the autosized rank, leaving alpha fixed would *violate* it (§7a.5). Not a new hyperparameter — a deterministic function of cited inputs. |
| `scope` default | `vision_decoder_concept` (changed from `vision_decoder`) | `# tbd: #230` — project-chosen concept scope; flip rationale = shipped default must learn text concepts (research note §4, §7) |
| `target_parameters` field default | `None` | framework parity with `target_modules` (`# cite:` PEFT LoraConfig default `target_parameters=None`) |
| small-data `r` guidance | 8 (note only, **not** a default change) | `# tbd:` within cited r≈8-16 small-data range (Unsloth / Raschka; research note §6). Documented as a comment/reference near the `r` default, not applied. |
| trainable-ratio threshold | 10 % (unchanged unless reality demands) | no tag needed if unchanged; a change requires `# tbd:` + empirical basis (§8.3) |
| new scope literal name | `vision_decoder_concept` | design choice (settled here; not a hyperparameter) |

**Small-data guidance is documentation, not a default change:** the research note (§6)
cites r≈8 for small datasets, but the default stays `r=16` (cited, conservative for
small/medium). The guidance is recorded as a `# tbd:`-tagged comment/reference near the `r`
field so a small-data user knows the lever exists, without silently changing the cited
default.
