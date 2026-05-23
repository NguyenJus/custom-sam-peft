# spec/hardening-followup-91-100 — Sweep PR for hardening follow-ups (#91–#100, except #98)

**Status:** Draft (2026-05-22)
**Tracking:** [#91](https://github.com/NguyenJus/custom-sam-peft/issues/91), [#92](https://github.com/NguyenJus/custom-sam-peft/issues/92), [#93](https://github.com/NguyenJus/custom-sam-peft/issues/93), [#94](https://github.com/NguyenJus/custom-sam-peft/issues/94), [#95](https://github.com/NguyenJus/custom-sam-peft/issues/95), [#96](https://github.com/NguyenJus/custom-sam-peft/issues/96), [#97](https://github.com/NguyenJus/custom-sam-peft/issues/97), [#99](https://github.com/NguyenJus/custom-sam-peft/issues/99), [#100](https://github.com/NguyenJus/custom-sam-peft/issues/100) — all labeled `hardening-followup`.
**Scope:** A single sweep PR on branch `hardening-followup-91-100` that resolves 9 of the 10 follow-up issues opened off of Section J of the [2026-05-21 hardening audit inventory](2026-05-21-hardening-audit-inventory.md). Two issues (#91, #94) close tracked-only via PR description; the other seven land code or doc changes. **#98 (QLoRA checkpoint disk-load) is out of scope** — it gets its own spec and PR.

**Builds on:** [`2026-05-21-hardening-pass-design.md`](2026-05-21-hardening-pass-design.md) (v0.7.0 hardening pass; merged as `bc36e7d`); [`2026-05-21-hardening-audit-inventory.md`](2026-05-21-hardening-audit-inventory.md) (Section J catalogues these items).

---

## 1. Context

PR #90 merged the v0.7.0 hardening pass. Section J of its audit inventory enumerated 9 items deferred to follow-up issues; a 10th (#100, `make_peft_method` if/elif) was discovered later. All 10 are tagged `hardening-followup` on GitHub. This spec resolves 9 of them in one PR. The dispositions below are the locked outcome of brainstorming on 2026-05-22.

**Out of scope (split):** [#98](https://github.com/NguyenJus/custom-sam-peft/issues/98) — *QLoRA checkpoint disk-load via `supports_checkpoint_load_from_disk()` (currently `False` at `src/custom_sam_peft/peft_adapters/__init__.py:110`; gated in `eval/runner.py:106`)*. That work has design depth (state-dict reconstruction, validation, eval integration) and is being split to a separate spec/PR.

---

## 2. Per-Issue Disposition

Each row below is a contract: action + concrete change + acceptance criterion. Implementer should be able to land the change without consulting the issue thread.

### 2.1 #91 — `EvalConfig.metrics`: close as tracked-only

**Action:** No code. Issue closes via PR description footer.

**Status check:** `EvalConfig` at `src/custom_sam_peft/config/schema.py:412` has no `metrics` field (verified). The field was already removed during the v0.7.0 hardening pass.

**Acceptance:** PR description closes #91 with the note: *"Field is already removed; reopen when a metrics-selector feature is designed."*

### 2.2 #92 — `MatcherWeights.lambda_l1`/`lambda_giou`: inline as constants

**Action:** Hardcode `lambda_l1=0.0`, `lambda_giou=0.0` at the `HungarianMatcher` construction site in `src/custom_sam_peft/models/losses.py` (currently lines 170–171 read them from `cfg.matcher_weights`). Drop the two fields from `MatcherWeights` in `src/custom_sam_peft/config/_internal.py:28-29`. `MatcherWeights` itself stays — it still holds `lambda_mask` (line 30) consumed at `losses.py:172`.

**Before** (`losses.py:169-173`):
```python
matcher = HungarianMatcher(
    lambda_l1=cfg.matcher_weights.lambda_l1,
    lambda_giou=cfg.matcher_weights.lambda_giou,
    lambda_mask=cfg.matcher_weights.lambda_mask,
)
```

**After:**
```python
matcher = HungarianMatcher(
    lambda_l1=0.0,
    lambda_giou=0.0,
    lambda_mask=cfg.matcher_weights.lambda_mask,
)
```

`HungarianMatcher.__init__` (`models/matching.py:112-118`) is unchanged — it still receives all three as floats.

**Acceptance:**
- `MatcherWeights` (`config/_internal.py`) has exactly one field: `lambda_mask: float = 5.0`.
- `losses.py` line that constructs `HungarianMatcher` passes literal `0.0` for `lambda_l1` and `lambda_giou`.
- `grep -n "lambda_l1\|lambda_giou" src/` returns only `models/matching.py` (the constructor + scoring at line 165–166) and the literal `0.0` in `losses.py`.
- Existing matcher tests (`tests/unit/test_matching.py` if present, or whichever exercises the matcher) pass without modification, or are updated minimally if they accessed the dataclass field directly.

### 2.3 #93 — `LossConfig.focal_gamma`/`focal_alpha`: inline as module constants

**Action:** Add module-level constants `_FOCAL_GAMMA = 2.0`, `_FOCAL_ALPHA = 0.25` near the top of `src/custom_sam_peft/models/losses.py`. Replace the two `cfg.focal_*` references at `losses.py:189-190` with the constants. Drop `focal_gamma` and `focal_alpha` from `LossConfig` in `config/_internal.py:56-57`. `LossConfig` stays — it still holds `w_mask`, `w_obj`, `w_presence`, `w_box`, and `matcher_weights`.

**Before** (`losses.py:186-191`):
```python
"obj": objectness_loss(
    canonical.obj_logits,
    matched_mask,
    gamma=cfg.focal_gamma,
    alpha=cfg.focal_alpha,
),
```

**After:**
```python
"obj": objectness_loss(
    canonical.obj_logits,
    matched_mask,
    gamma=_FOCAL_GAMMA,
    alpha=_FOCAL_ALPHA,
),
```

**Acceptance:**
- `_FOCAL_GAMMA` and `_FOCAL_ALPHA` defined once at module scope in `losses.py`, each with a short comment naming them as demoted-from-config constants (per audit Section E).
- `LossConfig` no longer has `focal_gamma` or `focal_alpha` attributes.
- A regression-guard assertion lives in the losses test (see §4): asserts the focal call site uses `gamma == 2.0` (e.g., by mocking `objectness_loss` and inspecting kwargs).

### 2.4 #94 — `early_stop_p_threshold`: close as tracked-only

**Action:** No code. Issue closes via PR description footer.

**Status check:** `BoxHintSchedule` at `src/custom_sam_peft/config/schema.py:359-378` does not expose `early_stop_p_threshold`; the field is documented in the docstring (line 367) as future-mechanism scaffolding but is not a settable attribute. `tests/unit/test_box_hint_schedule.py:25` asserts `not hasattr(s, "early_stop_p_threshold")` (verified).

**Acceptance:** PR description closes #94 with the note: *"Field already removed from schema; `test_box_hint_schedule.py` regression-asserts `not hasattr`."*

### 2.5 #95 — `flatten_metrics_report`: delete

**Action:** Delete `flatten_metrics_report` from `src/custom_sam_peft/tracking/__init__.py` (currently defined at lines 40–55, exported in `__all__` at line 17). Delete the corresponding test file `tests/unit/test_tracking_flatten.py`. Also drop the `TYPE_CHECKING` import of `MetricsReport` at lines 11–15 if no other reference remains in the file (verify post-edit; this is the only consumer).

**Rationale:** The audit (Section J5) flagged the helper as dead — zero `src/` callers. The "wire it in" alternative (call from the tracker on `evaluate` completion) is YAGNI: no caller has needed it in the year since it was written, and the eval reporter writes its own JSON shape; adding a hidden flatten-and-log side effect would be surprising. Delete now; reintroduce only when a concrete caller materializes.

**Acceptance:**
- `rg -n "flatten_metrics_report" src/ tests/` returns no matches.
- `tracking/__init__.py:__all__` is `["Tracker", "build_tracker"]`.
- `tests/unit/test_tracking_flatten.py` does not exist.
- `uv run pytest tests/unit/test_tracking_*` is green (no orphan references).

### 2.6 #96 — `models/_patches/`: SAM-3 bump checklist README

**Action:** Add `src/custom_sam_peft/models/_patches/README.md`. Doc-only — no Python changes. The README lists each patch file with a one-liner explaining what it patches and why, then provides a "When SAM-3 bumps" checklist.

**Existing patch files** (verified — 8 files plus `__init__.py`):

| File | One-liner |
| --- | --- |
| `addmm_act_grad_safe.py` | Guards `addmm` autograd path against an upstream activation-grad shape mismatch. |
| `encode_prompt_dtype.py` | Forces prompt-encoder activations to the wrapper's compute dtype to prevent fp16/bf16 cast mismatches. |
| `forward_grounding_skip_matching.py` | Skips the upstream grounding matcher path that we replace with our own Hungarian matcher. |
| `mha_input_dtype.py` | Casts MHA inputs to a consistent dtype across Q/K/V projections. |
| `module_input_dtype.py` | Generic input-dtype harmonizer for modules that drop kwargs through. |
| `pos_enc_dtype.py` | Aligns positional-encoding dtype with the surrounding activation dtype. |
| `roi_align_dtype.py` | Forces ROI-Align inputs to fp32 (kernel only supports fp32; see `2026-05-22-fix-roi-align-dtype-mismatch.md`). |
| `text_pool_dtype.py` | Aligns text-pool projection dtype with the text-encoder output. |

**"When SAM-3 bumps" checklist** (verbatim shape — implementer fills in exact wording):
1. Re-run `tests/gpu/` against the new SAM-3 checkpoint.
2. For each patch in this directory: open the corresponding upstream source file (`vendor/sam3/...` or the pinned pip dep), confirm the line numbers / function signatures the patch targets still exist.
3. If a target moved: update the patch's line / signature reference; if a target was removed: open an issue tagged `sam3-bump` to delete the patch.
4. Confirm `models/sam3.py::load_sam31` still wires each patch into the wrapper's `_apply_patches` step.
5. Update the SAM-3 checkpoint SHA pin in `src/custom_sam_peft/presets.py::_current_sam3_checkpoint_sha` (the analytic VRAM cache uses this to invalidate prior calibrations).

**Acceptance:**
- `src/custom_sam_peft/models/_patches/README.md` exists.
- Lists all 8 patch files with one-liners.
- Includes the 5-item bump checklist.
- No source-code changes in the commit.

### 2.7 #97 — `resolve_hf_token` duplication: rename + document

**Action:** Rename `resolve_hf_token` in `src/custom_sam_peft/notebook_helpers.py` (currently at line 54) to `resolve_hf_token_for_notebook`. Update the two callers:
- `tests/unit/test_notebook_helpers.py:15` import.
- `notebooks/custom_sam_peft_train.ipynb` cell that does `from custom_sam_peft.notebook_helpers import (... resolve_hf_token, ...)` (line 31 of the JSON) and the invocation `token = resolve_hf_token(env, local_present)` (line 42).

Update the test function names accordingly (`test_resolve_hf_token_*` → `test_resolve_hf_token_for_notebook_*` is acceptable, or shorter helper names — implementer choice).

Add a module docstring at the top of `notebook_helpers.py` explicitly contrasting the two `resolve_hf_token*` functions:

> Note: `utils/huggingface.py::resolve_hf_token` is the silent best-effort resolver used by `download_model` (returns the token or `None`, never raises). `notebook_helpers.py::resolve_hf_token_for_notebook` is an env-aware resolver for notebook contexts — it short-circuits when a local checkpoint is present and raises `RuntimeError` with Colab/RunPod-specific instructions when the token is missing. The two are deliberately not merged; their failure semantics differ.

The note about `diagnostics.py::_hf_auth_info` deliberately *not* delegating (it discriminates `env` vs `cache` rather than returning a token) is already in source (`diagnostics.py:127-133`) — no change needed there.

**Acceptance:**
- `rg -n "def resolve_hf_token" src/custom_sam_peft/` returns two distinct names: `resolve_hf_token` (in `utils/huggingface.py`) and `resolve_hf_token_for_notebook` (in `notebook_helpers.py`).
- `tests/unit/test_notebook_helpers.py` imports the renamed function and tests pass.
- The notebook JSON's import and call sites use `resolve_hf_token_for_notebook`.
- `notebook_helpers.py` module docstring contains the contrast paragraph.
- No semantics change in either function body.

### 2.8 #99 — Move `notebook_helpers.py`; keep `presets.py`

**Action:** Move `src/custom_sam_peft/notebook_helpers.py` → `notebooks/_lib/notebook_helpers.py`. Create `notebooks/_lib/__init__.py` (empty, or one-line docstring). Update the two import sites:
- `tests/unit/test_notebook_helpers.py:12` → from new path.
- `notebooks/custom_sam_peft_train.ipynb:31` → from new path.

The notebook's import is currently `from custom_sam_peft.notebook_helpers import (...)`. The new path requires the notebook to add the `notebooks/` directory to `sys.path` (or use a relative import via `_lib`). The notebook already does `sys.path` manipulation around its setup cells — implementer chooses the cleanest seam (likely: prepend `notebooks/` to `sys.path` and `from _lib.notebook_helpers import ...`).

Also move the test file: `tests/unit/test_notebook_helpers.py` → `tests/unit/notebooks/test_notebook_helpers.py`. Create `tests/unit/notebooks/__init__.py` if needed (project uses pytest's rootdir-relative collection — verify by running the relocated test).

**Do NOT move `presets.py`.** Its Section J premise (no `src/` callers) is stale: it now has three (verified):
- `src/custom_sam_peft/cli/run_cmd.py:30` — `from custom_sam_peft.presets import PresetDecision, decide_preset`
- `src/custom_sam_peft/cli/calibrate_cmd.py:24, 33` — both import from `presets`
- `src/custom_sam_peft/runs/bundle.py:36` — `from custom_sam_peft.presets import PresetDecision`

These were added by the v0.11.0 analytic-VRAM preset work after the audit landed. `presets.py` stays in `src/custom_sam_peft/`.

**Acceptance:**
- `src/custom_sam_peft/notebook_helpers.py` does not exist.
- `notebooks/_lib/notebook_helpers.py` exists with identical body (modulo the §2.7 rename).
- `notebooks/_lib/__init__.py` exists.
- `tests/unit/notebooks/test_notebook_helpers.py` exists and `uv run pytest tests/unit/notebooks/` is green.
- `src/custom_sam_peft/presets.py` still exists; its three `src/` callers still resolve.
- The Colab notebook imports cleanly when opened (smoke check: `python -c "import json; json.load(open('notebooks/custom_sam_peft_train.ipynb'))"` parses; eyeball the import cell).

### 2.9 #100 — `make_peft_method`: drive from registry

**Action:** Replace the if/elif branches at `src/custom_sam_peft/peft_adapters/__init__.py:129-146` with a `lookup()` call against a **new** `"peft_method"` registry namespace, decorate the two adapter classes with `@register("peft_method", ...)`, and re-raise the `RegistryError` (a `KeyError` subclass) as `ValueError` to preserve the error contract.

**Drift from the brief** — important: the brief implied the existing `@register("peft", "lora")` / `@register("peft", "qlora")` decorators (`peft_adapters/lora.py:88`, `peft_adapters/qlora.py:249`) already register what `make_peft_method` needs. They do not. Those decorators register the `apply_lora` / `apply_qlora` **functions** (signature `(wrapper, cfg) -> Sam3Wrapper`), not the `LoraAdapter` / `QloraAdapter` **classes** (constructed with `()` to produce a `PEFTMethod` protocol instance). The `"peft"` namespace is already taken — it's used at `train/runner.py:115` (`peft_factory = lookup("peft", cfg.peft.method)` then `peft_factory(wrapper, cfg.peft)`). Reusing it would conflate two different callables. A new `"peft_method"` namespace keeps the seam clean.

**Current** (`peft_adapters/__init__.py:129-146`):
```python
def make_peft_method(method: str) -> PEFTMethod:
    """Return the PEFTMethod instance for the given peft.method string.
    ...
    """
    if method == "lora":
        return LoraAdapter()
    if method == "qlora":
        return QloraAdapter()
    raise ValueError(
        f"Unknown peft.method {method!r}; expected 'lora' or 'qlora'. "
        "Register additional adapters via @register('peft', '<name>') and "
        "add a branch here."
    )
```

**Target:**

1. Decorate the adapter classes at their definition sites in `peft_adapters/__init__.py:70` and `peft_adapters/__init__.py:92`:

```python
from custom_sam_peft._registry import RegistryError, lookup, register


@register("peft_method", "lora")
class LoraAdapter:
    ...


@register("peft_method", "qlora")
class QloraAdapter:
    ...
```

2. Replace the factory body:

```python
def make_peft_method(method: str) -> PEFTMethod:
    """Return the PEFTMethod instance for the given peft.method string.

    Resolves via the @register("peft_method", ...) registry — adding a new
    adapter requires only a @register decorator on the new class, no edits here.
    """
    try:
        adapter_cls = lookup("peft_method", method)
    except RegistryError as exc:
        raise ValueError(
            f"Unknown peft.method {method!r}; expected 'lora' or 'qlora'. "
            "Register additional adapters via @register('peft_method', '<name>')."
        ) from exc
    return cast(PEFTMethod, adapter_cls())
```

`RegistryError` is a `KeyError` subclass (`_registry.py:16`), so `except RegistryError` is the precise catch (`except KeyError` would also work — the existing tests assert the `ValueError` shape, not the inner exception type). The re-raise preserves the `ValueError` contract that `tests/unit/test_peft_method_protocol.py:142-144` (`test_make_peft_method_unknown_raises`) verifies via `pytest.raises(ValueError, match=r"Unknown peft\.method")`.

**Why a new namespace, not reusing `"peft"`:**
- The existing `"peft"` bucket holds `apply_lora` / `apply_qlora` — callables with signature `(Sam3Wrapper, PEFTConfig) -> Sam3Wrapper`.
- `make_peft_method` returns a `PEFTMethod` protocol instance constructed with `()`.
- Two different callables with two different signatures and return types — separate registry keys are the right shape.

**Imports to add/adjust in `peft_adapters/__init__.py`:**
- Add `from custom_sam_peft._registry import RegistryError, lookup, register`.
- Add `from typing import cast` (or extend the existing typing import on line 19).
- The existing `Protocol`, `runtime_checkable`, `Path`, `CheckpointError` imports are unchanged.

**Module docstring update:** The package docstring at `peft_adapters/__init__.py:1-14` currently says `lookup("peft", "lora") → apply_lora`. Add a parallel line clarifying the new namespace: `lookup("peft_method", "lora") → LoraAdapter`, etc. The implementer should mirror the existing wording style.

**Acceptance:**
- `make_peft_method` body uses `lookup("peft_method", method)`; no `if method ==` literals.
- `LoraAdapter` and `QloraAdapter` carry `@register("peft_method", "lora")` / `@register("peft_method", "qlora")` decorators.
- `make_peft_method("lora")` returns a `LoraAdapter` instance (`test_make_peft_method_lora` passes unchanged).
- `make_peft_method("qlora")` returns a `QloraAdapter` instance (`test_make_peft_method_qlora` passes unchanged).
- `make_peft_method("unknown")` raises `ValueError` matching `r"Unknown peft\.method"` (`test_make_peft_method_unknown_raises` passes unchanged).
- The existing `lookup("peft", ...)` callers (e.g. `train/runner.py:115`) are untouched and still receive `apply_lora` / `apply_qlora`.
- No new tests required — the three existing tests fully cover the contract.

---

## 3. PR Shape & Commit Sequencing

Single PR on branch `hardening-followup-91-100`, targets `main`. Six commits, each independently revertable, ordered by risk-decay (most-tested code first, then deletions, then renames, then moves, then docs). `#91` and `#94` produce no commits — they close via PR description footer.

| # | Commit message | Touches |
| --- | --- | --- |
| 1 | `chore(config): inline MatcherWeights.lambda_l1/giou and LossConfig.focal_* constants (#92, #93)` | `src/custom_sam_peft/config/_internal.py`, `src/custom_sam_peft/models/losses.py` |
| 2 | `chore(peft): drive make_peft_method from registry (#100)` | `src/custom_sam_peft/peft_adapters/__init__.py` |
| 3 | `chore(tracking): delete unused flatten_metrics_report (#95)` | `src/custom_sam_peft/tracking/__init__.py`, `tests/unit/test_tracking_flatten.py` (deleted) |
| 4 | `chore(hf): rename resolve_hf_token in notebook_helpers, clarify duplication (#97)` | `src/custom_sam_peft/notebook_helpers.py`, `tests/unit/test_notebook_helpers.py`, `notebooks/custom_sam_peft_train.ipynb` |
| 5 | `chore(notebooks): move notebook_helpers.py to notebooks/_lib (#99)` | `src/custom_sam_peft/notebook_helpers.py` (deleted), `notebooks/_lib/__init__.py` (new), `notebooks/_lib/notebook_helpers.py` (new), `tests/unit/test_notebook_helpers.py` → `tests/unit/notebooks/test_notebook_helpers.py`, `notebooks/custom_sam_peft_train.ipynb` (import path) |
| 6 | `docs(models): add _patches/README.md SAM-3 version-bump checklist (#96)` | `src/custom_sam_peft/models/_patches/README.md` (new) |

**Rationale for ordering:**
- Commits 1, 2 touch code with the heaviest existing test coverage (loss/matcher/peft factory) — easiest to verify in isolation.
- Commit 3 deletes dead code — no semantic risk.
- Commit 4 renames before Commit 5 moves, so the move commit is a pure path change with no body diff (clean git history; `git mv` discovers the rename).
- Commit 6 is doc-only — safe at the end.

---

## 4. Test Plan

All CPU-only. The "manual gate" at the bottom is the implementer's pre-PR check.

| Issue | Test action |
| --- | --- |
| #91 | None (PR-description close). |
| #92 | Existing matcher tests cover the inlined values. Update any test that poked `MatcherWeights.lambda_l1` / `lambda_giou` directly (if any — sweep with `rg -n "lambda_l1\|lambda_giou" tests/`). |
| #93 | Existing `tests/unit/test_losses.py` (or whichever test exercises the focal call) covers the constants. **Add one regression-guard assertion**: mock `objectness_loss` (or inspect the call) and assert `gamma == 2.0` and `alpha == 0.25` are still applied at the call site. This protects against a silent constant drift after the field is gone from config. |
| #94 | None (PR-description close); `tests/unit/test_box_hint_schedule.py` already regression-asserts `not hasattr(s, "early_stop_p_threshold")`. |
| #95 | Delete `tests/unit/test_tracking_flatten.py`. No new tests. |
| #96 | None (doc-only). |
| #97 | Update existing test imports in `tests/unit/test_notebook_helpers.py` to the renamed function. Optionally rename test functions for clarity. No new tests. |
| #99 | Move `tests/unit/test_notebook_helpers.py` → `tests/unit/notebooks/test_notebook_helpers.py` so pytest still discovers it. Verify by running `uv run pytest tests/unit/notebooks/`. Smoke-verify the notebook still parses as valid JSON; no notebook-execution test added (no notebook-CI tooling in scope). |
| #100 | Existing `test_make_peft_method_lora`, `test_make_peft_method_qlora`, `test_make_peft_method_unknown_raises` in `tests/unit/test_peft_method_protocol.py:130-144` fully cover the contract. No new tests. Confirm `_bootstrap.bootstrap()` (or any path that imports `peft_adapters`) fires the new `@register("peft_method", ...)` decorators — the adapter classes are defined in `peft_adapters/__init__.py`, so importing the package is sufficient (no `_bootstrap.py` edit needed). |

**Manual gate** (run before opening the PR):
- `uv run pytest -m "not gpu"` — green.
- `uv run ruff check src/ tests/` — clean.
- `uv run ruff format --check src/ tests/` — clean.
- `rg -n "flatten_metrics_report\|notebook_helpers" src/` — only the `notebooks/_lib/` path remains.
- `python -c "import json; json.load(open('notebooks/custom_sam_peft_train.ipynb'))"` — parses.

---

## 5. PR Description Footer

The PR description ends with a verbatim `Closes` block:

```
Closes #91 — EvalConfig.metrics: tracked-only; field already removed.
Closes #92 — MatcherWeights.lambda_l1/giou inlined.
Closes #93 — LossConfig.focal_* inlined.
Closes #94 — early_stop_p_threshold: tracked-only; field already removed.
Closes #95 — flatten_metrics_report deleted.
Closes #96 — _patches/README.md SAM-3 bump checklist added.
Closes #97 — resolve_hf_token duplication renamed + documented.
Closes #99 — notebook_helpers.py moved to notebooks/_lib/ (presets.py kept — premise stale).
Closes #100 — make_peft_method driven from registry.
```

`#98` stays open; PR description includes a one-line note: *"#98 (QLoRA checkpoint disk-load) is split out for a separate spec/PR — follow-up to come."*

---

## 6. Non-Goals (explicit)

- **No removal of `LossConfig` or `MatcherWeights` classes.** Both retain remaining fields (`w_mask`, `w_obj`, `w_presence`, `w_box`, `matcher_weights` on `LossConfig`; `lambda_mask` on `MatcherWeights`).
- **No `presets.py` move.** Section J's "no `src/` callers" premise is stale (now 3 callers); the audit predates v0.11.0 work.
- **No merge of the two `resolve_hf_token` implementations.** Their failure semantics differ deliberately (silent best-effort vs. raise-with-friendly-message); merging would conflate cases the call sites need to discriminate.
- **No #98 work in this PR.** QLoRA disk-load is split to its own spec.
- **No notebook-execution CI tooling.** Smoke-parsing the JSON is the only notebook validation in this PR.
- **No registry refactor beyond `make_peft_method`.** The dataset/peft/tracker registries are not touched; only the one consumer named in #100 is migrated.
- **No new public API.** Every change is either an internal-constant inline, a deletion, a rename, a move, or a docs addition.

---

## 7. Implementation Plan (numbered)

This section is the seam consumed by `superpowers:writing-plans`.

**Step 1.** Inline `MatcherWeights.lambda_l1/giou` and `LossConfig.focal_*` per §2.2 and §2.3. Add the `_FOCAL_*` module constants in `models/losses.py`. Drop four fields from `config/_internal.py`. Add the gamma-applied regression assertion to the losses test per §4. Commit message per §3 row 1.

**Step 2.** Replace `make_peft_method` if/elif with `lookup("peft_method", method)` per §2.9. Decorate `LoraAdapter` and `QloraAdapter` with `@register("peft_method", ...)`. Import `lookup`, `register`, `RegistryError`, and `cast`; preserve the `RegistryError → ValueError` error contract. Run `test_peft_method_protocol.py::test_make_peft_method_*`. Commit message per §3 row 2.

**Step 3.** Delete `flatten_metrics_report` and its test per §2.5. Confirm no orphan references via `rg`. Commit message per §3 row 3.

**Step 4.** Rename `resolve_hf_token` → `resolve_hf_token_for_notebook` in `notebook_helpers.py` per §2.7. Add the module docstring contrasting the two functions. Update test imports and the notebook JSON's import + call site. Commit message per §3 row 4.

**Step 5.** Move `src/custom_sam_peft/notebook_helpers.py` → `notebooks/_lib/notebook_helpers.py` per §2.8. Create `notebooks/_lib/__init__.py`. Move and re-home the test file to `tests/unit/notebooks/`. Update the notebook's import (likely via `sys.path` prepend). Verify `presets.py` stays put. Commit message per §3 row 5.

**Step 6.** Add `src/custom_sam_peft/models/_patches/README.md` per §2.6: one-liner per existing patch + 5-step bump checklist. Doc-only. Commit message per §3 row 6.

**Step 7.** Run the manual gate per §4. Open the PR with the description footer per §5.
