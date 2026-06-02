# PEFT in_proj Concept Scope Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add one new LoRA scope `vision_decoder_concept` (the new default) that reaches the
decoder's `ca_text` / `self_attn` `in_proj_weight` by naming the two `nn.MultiheadAttention`
**modules** as `target_modules` so peft dispatches them to its `lora.MultiheadAttention`
support layer (adapting both `in_proj_weight` and `out_proj`, with dropout). Also make the
pre-flight VRAM calibrate autosize co-scale `alpha` when it reduces rank `r`.

**Architecture:** The feasibility spike (Phase 1) is **DONE** — it SELECTED the
`lora.MultiheadAttention` route over the `target_parameters` route (which hard-crashes on the
default `dropout=0.05`); see spec §7.3a. Phase 2 **reverts** the already-committed
`target_parameters` mechanism and **reworks** it into an MHA-module resolution axis
(`SCOPE_MHA_MODULES` + `_resolve_mha_modules`) that unions matched `nn.MultiheadAttention`
module names into `target_modules` through both `apply_lora` and the QLoRA apply path, lands
the new scope/default, de-overlaps the concept `SCOPE_TARGETS` entry, updates fixtures, and
ships CPU + GPU tests. An orthogonal, file-disjoint change makes calibrate persist and warn
on a co-scaled `alpha` (Phase 3).

**Tech Stack:** Python 3.12, PyTorch, HuggingFace `peft` 0.19.1, `bitsandbytes` (QLoRA),
pydantic v2, typer, pytest. SAM 3.1 (`sam3`) for the real-decoder GPU tests.

**Spec (source of truth):** `docs/superpowers/specs/2026-06-01-peft-in-proj-concept-scope-design.md`
**Research note:** `docs/research/2026-06-01-issue-230-peft-adaptation-surface-lit-review.md`

---

## Current committed state (Phase 2 is a REVERT-AND-REWORK, not greenfield)

Commits `287386a..b283be7` implemented the **OLD `target_parameters` mechanism**. Phase 2's
implementer is **editing already-committed code**, not starting clean. The pivot
(spec §7.3a, user-approved) is a **full revert** of the `target_parameters` axis, replaced by
the MHA-module axis. The committed surfaces Phase 2 must transform:

- `src/custom_sam_peft/config/schema.py`: a `PEFTConfig.target_parameters: list[str] | None`
  field was added — it is **REVERTED** (no new field; §6.3). The `LoraScope` literal +
  `scope = "vision_decoder_concept"` default flip is **KEPT** (still correct).
- `src/custom_sam_peft/peft_adapters/lora.py`: `SCOPE_TARGET_PARAMETERS`,
  `_resolve_target_parameters`, the `vision_decoder_concept` `SCOPE_TARGETS` entry
  (currently == `vision_decoder`), and the `apply_lora` `target_parameters` wiring — all
  become the MHA-module mechanism (`SCOPE_MHA_MODULES`, `_resolve_mha_modules`, a
  **de-overlapped** concept `SCOPE_TARGETS` entry, union into `target_modules`).
- `src/custom_sam_peft/peft_adapters/qlora.py`: the `target_parameters` import + wiring becomes
  the MHA-module union.
- `tests/fixtures/tiny_sam3_lora_stub.py`: `ca_text` / `self_attn` are already real
  `nn.MultiheadAttention` (**KEEP** — both mechanisms need this). `FIXTURE_SCOPE_TARGET_PARAMETERS`
  becomes `FIXTURE_SCOPE_MHA_MODULES`; the concept `FIXTURE_SCOPE_PATTERNS` entry is
  **de-overlapped**.
- `tests/integration/test_peft_{lora,qlora}_real.py`: Phase-1 spike tests use
  `target_parameters` overrides — reworked to the MHA-module surface (folded into Task 2.7's
  GPU tests, which drive the real `scope="vision_decoder_concept"`).
- `tests/unit/test_peft_target_parameters.py` (created by the old commits): retargeted to the
  MHA axis (rename symbols, de-overlap assertions). The implementer may keep the filename or
  rename it; the plan tasks reference it by its current path.

After Phase 2 completes, **no** reference to `target_parameters`, `SCOPE_TARGET_PARAMETERS`,
or `_resolve_target_parameters` may remain anywhere in `src/` or `tests/` (Task 2.8 greps for
this).

---

## Conventions every task obeys

Bake these into every implementer task — they are non-negotiable project gates.

- **Lint/type gate before each commit:** run all three and fix findings on touched files:
  - `uv run ruff check <touched files>`
  - `uv run ruff format --check <touched files>` (separate from `ruff check`; CI runs both)
  - `uv run mypy --strict <touched files>` (CI scopes mypy to `src/custom_sam_peft`)
- **No `assert isinstance(...)` in `src/`** — ruff S101 / bandit forbids `assert` in
  `src/`. Narrow structurally (`if isinstance(x, T) and ...:`), never with a bare `assert`.
  (Tests under `tests/` may assert freely.)
- **CPU test runs bypass the coverage gate:** run CPU subsets with
  `uv run pytest -o "addopts=" <path>` to bypass the global `--cov-fail-under=80`. Do **not**
  run `pytest --cov` locally (it segfaults torch on this box) — trust CI for coverage; keep
  coverage >= 80%.
- **GPU tests:** never run a bare `pytest tests/` (the real-model GPU suite must not run in
  one process). Use `scripts/run_gpu_tests.sh`. The real-decoder work is gated by the
  `requires_checkpoint` + `requires_compatible_gpu` markers and only executes where a
  compatible GPU + checkpoint exist; the implementer writes/edits these tests but verifies
  them structurally (ruff/mypy/`py_compile`) when no GPU is present, and relies on CI/the
  GPU runner to execute them.
- **Blast-radius rule for schema/`PresetDecision` changes:** before declaring a task done,
  `grep` every constructor / call site of the changed type and run the **full** CPU suite
  (`uv run pytest tests/unit tests/integration -o "addopts=" -q` or the project's standard
  CPU invocation), not just the new test file.
- **Eager-import caveat:** `src/custom_sam_peft/__init__.py` eagerly imports the train chain,
  so an import error anywhere in the package breaks the whole package. After any
  symbol-add/rename/remove, verify with `uv run python -c "import custom_sam_peft"` and
  `uv run python -m py_compile <touched files>`.
- **cite / `# tbd:` discipline:** every new or changed default carries a `# cite:` or
  `# tbd:` tag. The spec §12 already resolves all tags in this plan — copy them verbatim;
  do not invent new ones and do not drop required ones.
- **Reproducibility (hard):** legacy scopes (`vision` / `vision_decoder` / `all`) and any
  config that does not reduce rank must produce **byte-identical** `LoraConfig` / cache /
  rewritten-config output to today. Tests assert this.
- **Per-phase close:** each phase ends with a verification task (run lint/type/tests, grep
  blast radius, confirm package imports) before its commit/handoff.

---

## Phase dependency map

| Phase | Title | Depends on | Parallelizable with |
| --- | --- | --- | --- |
| 1 | in_proj feasibility spike (GATING) | — | 3 |
| 2 | MHA-module axis + scope + schema + tests (revert-and-rework) | **Phase 1 — DONE (§7.3a)** | — |
| 3 | calibrate VRAM-autosize alpha co-scale + WARNING | — (orthogonal) | 1 and 2 |

- **Phase 1 is DONE; its outcome gates Phase 2.** The spike SELECTED the
  `lora.MultiheadAttention` route (spec §7.3a); `target_parameters` is rejected/reverted.
  Phase 2 is written to that decision — there is no remaining mechanism choice.
- **Phase 3 is orthogonal.** It is file-disjoint from Phases 1–2 (touches only
  `calibrate_cmd.py`, `_config_rewrite.py`, `presets.py::PresetDecision`, and
  `tests/unit/test_calibrate_cmd.py`) and does **not** depend on the spike. An orchestrator
  MAY run Phase 3 in parallel with Phase 2 on the same branch/worktree — no shared files, no
  shared symbols. **Serialize commits** (parallel agents committing on one branch can orphan
  a commit).

---

## Phase 1 — in_proj feasibility spike (GATING) — **DONE**

**Status: DONE. Mechanism decided: spec §7.3a.**

The spike ran against the real SAM 3.1 decoder and **SELECTED Option (a)** — peft's
`lora.MultiheadAttention` support path (name the `ca_text` / `self_attn`
`nn.MultiheadAttention` modules in `target_modules`; peft adapts both `in_proj_weight` and
`out_proj`, **with dropout**). The originally-planned `target_parameters` route was
**REJECTED**: peft 0.19.1's `lora.ParamWrapper` (the `target_parameters` LoRA layer)
hard-raises `ValueError: lora.ParamWrapper does not work with lora_dropout != 0`
(`peft/tuners/lora/layer.py:2142`), and `PEFTConfig.dropout` defaults to `0.05`, so the
shipped concept default would crash on construction.

**Empirically established on GPU (spec §7.3a, stated as confirmed):**

- `lora.MultiheadAttention` (named via `target_modules`) adapts in_proj (`<mha>.lora_A` /
  `<mha>.lora_B`; in_proj `lora_B` shape `[3*embed_dim, r]`) **and** out_proj
  (`<mha>.base_layer.out_proj.lora_A` / `.lora_B`), supports `lora_dropout`, uses no
  `ParamWrapper`.
- **QLoRA-MHA coexistence is GPU-confirmed**: the MHA stays unquantized
  (`_mha_exclusion_types`), attaches alongside `Linear4bit` LoRA in one `PeftModel` with
  `dropout=0.05`, forward + grad finite on all surfaces, `merge_and_unload` clean (only a
  benign NF4-rounding `UserWarning`). The §7.2 hard requirement holds on the MHA route.

**What Phase 1 produced (interface contract consumed by Phase 2):**

- Recorded **go/no-go: GO** on the in_proj surface, mechanism = `lora.MultiheadAttention`
  (§7.3a).
- Empirically-observed trainable ratio stays under the existing GPU `< 0.05` budget (feeds
  Phase 2's §8.3 confirmation).

**Leftover the spike committed (Phase 2 reworks it, NOT re-planned here):** the Phase-1
spike tests in `tests/integration/test_peft_{lora,qlora}_real.py` drive resolution via an
explicit `PEFTConfig.target_parameters` override. That override surface is being reverted, so
those spike tests must be **reworked to the MHA-module surface** — folded into **Phase 2,
Task 2.7** (which replaces them with the productionized `scope="vision_decoder_concept"` GPU
tests). Do not re-run or re-plan the spike itself.

---

## Phase 2 — MHA-module axis + new scope + schema + tests (revert-and-rework)

**Feature block:** Revert the committed `target_parameters` axis and rework it into the
production `nn.MultiheadAttention`-module resolution axis (`SCOPE_MHA_MODULES` +
`_resolve_mha_modules`), keep the new `vision_decoder_concept` scope + default flip,
de-overlap the concept `SCOPE_TARGETS` entry, update the fixture, and land the full CPU + GPU
test suite. This is the heart of the feature.

**Consumes from Phase 1:** the GO + the `lora.MultiheadAttention` mechanism (§7.3a). There is
no remaining mechanism branch — implement exactly the MHA-module route below.

**Interface contract this phase PRODUCES (for downstream sessions / future tiers):**

- `SCOPE_MHA_MODULES: dict[str, list[str]]` in `lora.py` — scope → `nn.MultiheadAttention`
  **module-name** regexes (sibling to `SCOPE_TARGETS`). Only `vision_decoder_concept`
  populates it; legacy scopes resolve to `[]` via `.get(scope, [])`.
- `_resolve_mha_modules(base: nn.Module, cfg: PEFTConfig) -> list[str]` in `lora.py` —
  matches `SCOPE_MHA_MODULES[scope]` against the `nn.MultiheadAttention` modules in
  `base.named_modules()`; returns `[]` for empty pattern lists (legacy scopes) **and** when
  `cfg.target_modules` is overridden; raises `ValueError` only on non-empty-no-match.
  Imported by `qlora.py`.
- `SCOPE_TARGETS["vision_decoder_concept"]` = `vision_decoder`'s generic-module set **minus**
  the `self_attn` / `ca_text` `out_proj` alternatives (de-overlap; peft's MHA wrapper adapts
  out_proj internally).
- `LoraScope = Literal["vision", "vision_decoder", "vision_decoder_concept", "all"]` and
  `PEFTConfig.scope` default = `"vision_decoder_concept"` (already committed; kept).
- **No new `PEFTConfig` field** — `target_parameters` is reverted; `target_modules` remains
  the single module-axis override.
- `FIXTURE_SCOPE_MHA_MODULES` + a de-overlapped `FIXTURE_SCOPE_PATTERNS["vision_decoder_concept"]`
  in the stub.
- Both `apply_lora` and the QLoRA apply path build `LoraConfig` with
  `target_modules = matched_names + [n for n in mha_names if n not in matched_names]` and no
  `target_parameters` kwarg.

### Task 2.1: Revert the `PEFTConfig.target_parameters` override field

**Files:**

- Modify: `src/custom_sam_peft/config/schema.py` (remove the committed `target_parameters`
  field added in `287386a`, immediately after the `target_modules` field)
- Test: `tests/unit/test_config_schema.py` (remove the two committed
  `target_parameters` tests; locate via `grep -rn "target_parameters" tests/unit`)

The MHA-module mechanism adds **no** override field (spec §6.3). A user selects the concept
in_proj surface via `scope: vision_decoder_concept`, not a parameter override. The
`target_modules` override remains the single module-axis override.

- [ ] **Step 1: Remove the committed field + its tests**

Delete the `target_parameters: list[str] | None = Field(...)` block from `PEFTConfig` (the
block added after `target_modules`). Delete the two committed schema tests
(`test_peftconfig_target_parameters_defaults_none` / `..._accepts_list`).

- [ ] **Step 2: Confirm the field is gone (negative test)**

Run:

```bash
grep -rn "target_parameters" src/custom_sam_peft/config/schema.py tests/unit/test_config_schema.py
uv run pytest -o "addopts=" tests/unit/test_config_schema.py -v
```

Expected: the grep returns **nothing**; the schema suite PASSES. `PEFTConfig` is `_Strict`,
so passing `target_parameters=...` now raises `ValidationError` (the intended reverted state).

- [ ] **Step 3: Lint/type + import smoke**

Run:

```bash
uv run ruff check src/custom_sam_peft/config/schema.py
uv run ruff format --check src/custom_sam_peft/config/schema.py
uv run mypy --strict src/custom_sam_peft/config/schema.py
uv run python -c "import custom_sam_peft"
```

Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add src/custom_sam_peft/config/schema.py tests/unit/test_config_schema.py
git commit -m "revert(#230): remove PEFTConfig.target_parameters field (MHA-module pivot)"
```

> NOTE: the `LoraScope` literal and `scope = "vision_decoder_concept"` default flip were
> committed in `44b4d31` and are **kept**. They are re-asserted in Task 2.5's tests; this task
> does not touch them. The `# tbd:` annotation already references #230 (spec §6.2, §12).

### Task 2.2: Replace `SCOPE_TARGET_PARAMETERS` with `SCOPE_MHA_MODULES`; de-overlap the concept `SCOPE_TARGETS` entry

**Files:**

- Modify: `src/custom_sam_peft/peft_adapters/lora.py` (the committed `SCOPE_TARGET_PARAMETERS`
  dict + the `vision_decoder_concept` `SCOPE_TARGETS` entry)
- Test: `tests/unit/test_peft_target_parameters.py` (retarget the committed scope/dict tests)

- [ ] **Step 1: Rewrite the failing tests to the MHA axis**

In `tests/unit/test_peft_target_parameters.py`, replace the committed
`SCOPE_TARGET_PARAMETERS` import and the three scope/dict tests with:

```python
"""Unit coverage for the #230 SCOPE_MHA_MODULES resolution axis."""

from __future__ import annotations

from custom_sam_peft.peft_adapters.lora import SCOPE_MHA_MODULES, SCOPE_TARGETS


def test_scope_mha_modules_has_concept_mha_patterns() -> None:
    pats = SCOPE_MHA_MODULES["vision_decoder_concept"]
    assert any("ca_text" in p for p in pats)
    assert any("self_attn" in p for p in pats)
    assert not any("cross_attn" in p for p in pats), "cross_attn is RoPEAttention, not MHA"
    assert not any("in_proj_weight" in p for p in pats), "MHA axis names modules, not params"


def test_concept_scope_modules_de_overlap_vision_decoder() -> None:
    """Concept SCOPE_TARGETS drops the self_attn/ca_text out_proj alternatives (peft's
    MHA wrapper adapts out_proj internally; double-targeting must be avoided)."""
    concept = SCOPE_TARGETS["vision_decoder_concept"]
    # cross_attn.out_proj is kept (RoPEAttention -> genuine nn.Linear).
    assert any("cross_attn" in p and "out_proj" in p for p in concept)
    # self_attn / ca_text out_proj are NOT generic targets under the concept scope.
    assert not any(("self_attn" in p or "ca_text" in p) and "out_proj" in p for p in concept)
    # And it is NOT module-equal to vision_decoder anymore.
    assert SCOPE_TARGETS["vision_decoder_concept"] != SCOPE_TARGETS["vision_decoder"]


def test_legacy_scopes_have_no_mha_targets() -> None:
    for scope in ("vision", "vision_decoder", "all"):
        assert SCOPE_MHA_MODULES.get(scope, []) == []
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `uv run pytest -o "addopts=" tests/unit/test_peft_target_parameters.py -v`
Expected: FAIL (`ImportError` — `SCOPE_MHA_MODULES` does not exist; the committed
`SCOPE_TARGET_PARAMETERS` still does).

- [ ] **Step 3: Replace the dict + de-overlap the concept `SCOPE_TARGETS` entry**

In `src/custom_sam_peft/peft_adapters/lora.py`:

1. **De-overlap** the committed `SCOPE_TARGETS["vision_decoder_concept"]` entry. It currently
   equals the `vision_decoder` list; narrow the
   `(self_attn|cross_attn|ca_text)\.out_proj` alternation to `cross_attn\.out_proj` (spec
   §4.2, §5.1):

   ```python
   # vision_decoder's generic-module set MINUS the self_attn/ca_text out_proj
   # alternatives (peft's lora.MultiheadAttention adapts those out_proj internally when
   # the MHA module is targeted via SCOPE_MHA_MODULES; double-targeting must be avoided).
   # cross_attn is a RoPEAttention (genuine nn.Linear out_proj), so it stays generic.
   # New default scope (schema.py). See spec #230 §4.2, §5.1.
   "vision_decoder_concept": [
       r"backbone\.vision_backbone\.trunk\.blocks\.\d+\.attn\.(qkv|proj)$",
       r"transformer\.decoder\.layers\.\d+\.cross_attn\.out_proj$",
       r"transformer\.decoder\.layers\.\d+\.linear[12]$",
   ],
   ```

2. **Replace** the committed `SCOPE_TARGET_PARAMETERS` dict (and its docstring/comment) with
   `SCOPE_MHA_MODULES`, immediately after the `SCOPE_TARGETS` dict closes (spec §5.1):

   ```python
   # Parallel to SCOPE_TARGETS: scope -> regexes matched against nn.MultiheadAttention
   # modules in named_modules(). Naming an MHA module makes peft dispatch it to its
   # lora.MultiheadAttention layer, which adapts BOTH in_proj_weight and out_proj (with
   # dropout support). Only the concept scope populates it; absent scopes carry no MHA
   # targets (reproducibility for vision/vision_decoder/all). This is the second
   # single-point-of-contact for SAM 3.1 surface naming alongside SCOPE_TARGETS.
   SCOPE_MHA_MODULES: dict[str, list[str]] = {
       "vision_decoder_concept": [
           r"transformer\.decoder\.layers\.\d+\.ca_text$",
           r"transformer\.decoder\.layers\.\d+\.self_attn$",
       ],
   }
   ```

- [ ] **Step 4: Run the test to confirm it passes**

Run: `uv run pytest -o "addopts=" tests/unit/test_peft_target_parameters.py -v`
Expected: the three rewritten tests PASS. (The `_resolve_*` / `apply_lora` tests still
reference the old symbols and will FAIL/ERROR until Tasks 2.3–2.4 — that is expected; run the
`-k` subset above, not the whole file, until then.)

- [ ] **Step 5: Lint/type + import smoke**

Run:

```bash
uv run ruff check src/custom_sam_peft/peft_adapters/lora.py
uv run ruff format --check src/custom_sam_peft/peft_adapters/lora.py
uv run mypy --strict src/custom_sam_peft/peft_adapters/lora.py
uv run python -c "import custom_sam_peft"
```

Expected: lint/format/import pass. (`mypy --strict` on `lora.py` may still flag the not-yet-
reworked `_resolve_target_parameters` usage downstream — that is resolved in Tasks 2.3–2.4;
do not commit until Task 2.4's lint is clean. Commit this task only once `lora.py` lints/types
clean for the symbols it owns.)

- [ ] **Step 6: Commit**

```bash
git add src/custom_sam_peft/peft_adapters/lora.py tests/unit/test_peft_target_parameters.py
git commit -m "feat(#230): replace SCOPE_TARGET_PARAMETERS with SCOPE_MHA_MODULES; de-overlap concept scope"
```

### Task 2.3: Replace `_resolve_target_parameters` with `_resolve_mha_modules`

**Files:**

- Modify: `src/custom_sam_peft/peft_adapters/lora.py` (replace the committed
  `_resolve_target_parameters`, sibling to `_resolve_targets`)
- Test: `tests/unit/test_peft_target_parameters.py`

- [ ] **Step 1: Rewrite the resolver tests to the MHA axis + `_MiniBase`**

In `tests/unit/test_peft_target_parameters.py`, replace the committed
`_resolve_target_parameters` tests with `_resolve_mha_modules` tests. The `_MiniBase` (real
`nn.MultiheadAttention` children at `transformer.decoder.layers.0.{ca_text,self_attn}`) is
reused — it needs **no** monkeypatch because it uses the real `transformer.decoder` prefix:

```python
import pytest
from torch import nn

from custom_sam_peft.config.schema import PEFTConfig
from custom_sam_peft.peft_adapters.lora import _resolve_mha_modules


class _MiniBase(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.transformer = nn.Module()
        self.transformer.decoder = nn.Module()
        layer = nn.Module()
        layer.ca_text = nn.MultiheadAttention(8, 2)
        layer.self_attn = nn.MultiheadAttention(8, 2)
        self.transformer.decoder.layers = nn.ModuleList([layer])


def test_resolve_mha_concept_scope_returns_both_modules() -> None:
    got = _resolve_mha_modules(_MiniBase(), PEFTConfig(method="lora", scope="vision_decoder_concept"))
    assert got == [
        "transformer.decoder.layers.0.ca_text",
        "transformer.decoder.layers.0.self_attn",
    ]


def test_resolve_mha_empty_for_legacy_scope_returns_empty_no_error() -> None:
    assert _resolve_mha_modules(_MiniBase(), PEFTConfig(method="lora", scope="vision_decoder")) == []


def test_resolve_mha_returns_empty_when_target_modules_overridden() -> None:
    cfg = PEFTConfig(
        method="lora",
        scope="vision_decoder_concept",  # has MHA patterns ...
        target_modules=[r"\.ca_text$"],  # ... but override owns the module axis
    )
    assert _resolve_mha_modules(_MiniBase(), cfg) == []


def test_resolve_mha_non_empty_no_match_raises_valueerror(monkeypatch: pytest.MonkeyPatch) -> None:
    import custom_sam_peft.peft_adapters.lora as lora_mod

    # A non-empty pattern list that matches zero MHA modules must raise.
    monkeypatch.setitem(lora_mod.SCOPE_MHA_MODULES, "vision_decoder_concept", [r"\.nonexistent_mha$"])
    cfg = PEFTConfig(method="lora", scope="vision_decoder_concept")
    with pytest.raises(ValueError) as exc:
        _resolve_mha_modules(_MiniBase(), cfg)
    msg = str(exc.value)
    assert "nonexistent_mha" in msg  # patterns tried listed
    assert "ca_text" in msg or "self_attn" in msg  # a real MHA module name sampled
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `uv run pytest -o "addopts=" tests/unit/test_peft_target_parameters.py -k resolve -v`
Expected: FAIL (`ImportError` — `_resolve_mha_modules` not defined; `_resolve_target_parameters`
still is).

- [ ] **Step 3: Replace the resolver**

In `src/custom_sam_peft/peft_adapters/lora.py`, **remove** `_resolve_target_parameters` and
add `_resolve_mha_modules` directly after `_resolve_targets` (spec §5.2). Key asymmetry vs
`_resolve_targets`: empty pattern list is **fine** (returns `[]`); an explicit
`cfg.target_modules` override returns `[]`; only a **non-empty-no-match** raises:

```python
def _resolve_mha_modules(base: nn.Module, cfg: PEFTConfig) -> list[str]:
    """Resolve scope MHA-module patterns against the nn.MultiheadAttention modules.

    Precedence mirrors _resolve_targets:
      * cfg.target_modules is not None -> return [] (the user's explicit module
        override owns the module axis; the scope's MHA patterns do not apply).
      * else -> SCOPE_MHA_MODULES.get(cfg.scope, []) matched against the
        nn.MultiheadAttention modules in base.named_modules().

    Returns the full matched MHA module names (e.g.
    'transformer.decoder.layers.0.ca_text') to union into target_modules so peft
    dispatches them to lora.MultiheadAttention (adapting in_proj_weight + out_proj).
    Returns [] when the resolved pattern list is empty (legacy scopes) or when
    target_modules is overridden -- NOT an error. Raises ValueError only when a
    NON-EMPTY pattern list matches zero MHA modules (a typo or SAM rename), mirroring
    _resolve_targets' no-match error so the in_proj surface never silently trains
    nothing.
    """
    if cfg.target_modules is not None:
        return []
    patterns = SCOPE_MHA_MODULES.get(cfg.scope, [])
    if not patterns:
        return []
    compiled = [re.compile(p) for p in patterns]
    mha_names = [
        name
        for name, module in base.named_modules()
        if isinstance(module, nn.MultiheadAttention)
    ]
    matched = [name for name in mha_names if any(c.search(name) for c in compiled)]
    if not matched:
        sample = ", ".join(mha_names[:50]) if mha_names else "<no nn.MultiheadAttention modules found>"
        raise ValueError(
            f"apply_lora: no nn.MultiheadAttention modules matched SCOPE_MHA_MODULES "
            f"patterns {patterns}. MHA modules actually present (first 50): {sample}"
        )
    return matched
```

- [ ] **Step 4: Run the test to confirm it passes**

Run: `uv run pytest -o "addopts=" tests/unit/test_peft_target_parameters.py -k resolve -v`
Expected: the four resolver tests PASS.

- [ ] **Step 5: Lint/type + import smoke**

Run:

```bash
uv run ruff check src/custom_sam_peft/peft_adapters/lora.py tests/unit/test_peft_target_parameters.py
uv run ruff format --check src/custom_sam_peft/peft_adapters/lora.py tests/unit/test_peft_target_parameters.py
uv run mypy --strict src/custom_sam_peft/peft_adapters/lora.py
uv run python -c "import custom_sam_peft"
```

Expected: pass for the resolver. (If `apply_lora` still calls `_resolve_target_parameters`,
that line is fixed in Task 2.4 in the same `lora.py`; sequence Task 2.4 immediately after and
do not leave the file in a broken-import state across a commit boundary — `apply_lora`'s call
is updated here-or-next. Prefer updating the `apply_lora` call site in this task's Step 3 so
`lora.py` imports clean, then expand the wiring in Task 2.4.)

- [ ] **Step 6: Commit**

```bash
git add src/custom_sam_peft/peft_adapters/lora.py tests/unit/test_peft_target_parameters.py
git commit -m "feat(#230): replace _resolve_target_parameters with _resolve_mha_modules"
```

### Task 2.4: Rework `apply_lora` to union MHA module names into `target_modules`

**Files:**

- Modify: `src/custom_sam_peft/peft_adapters/lora.py` (`apply_lora`)
- Test: `tests/unit/test_peft_target_parameters.py`

- [ ] **Step 1: Rewrite the wiring test (legacy scope stays byte-identical; concept unions)**

In `tests/unit/test_peft_target_parameters.py`, replace the committed
`test_apply_lora_legacy_scope_passes_target_parameters_none` with a `LoraConfig`-spy test
asserting (a) **no** `target_parameters` kwarg is ever passed, and (b) a legacy scope's
`target_modules` is unchanged (no MHA union):

```python
def test_apply_lora_never_passes_target_parameters_and_legacy_unioned_empty() -> None:
    """Reproducibility: legacy scopes build LoraConfig with no MHA union and never a
    target_parameters kwarg (the reverted axis)."""
    import custom_sam_peft.peft_adapters.lora as lora_mod
    from tests.fixtures.tiny_sam3_lora_stub import FIXTURE_SCOPE_PATTERNS, make_stub_wrapper

    captured: dict[str, object] = {}
    real_cfg = lora_mod.LoraConfig

    def _spy(*args: object, **kwargs: object) -> object:
        captured.update(kwargs)
        return real_cfg(*args, **kwargs)

    w = make_stub_wrapper(dim=8, working=False)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(lora_mod, "LoraConfig", _spy)
        lora_mod.apply_lora(
            w,
            PEFTConfig(
                method="lora",
                scope="vision_decoder",
                target_modules=FIXTURE_SCOPE_PATTERNS["vision_decoder"],
            ),
        )
    assert "target_parameters" not in captured  # reverted axis is gone
    assert captured["target_modules"] == FIXTURE_SCOPE_PATTERNS["vision_decoder"]  # no MHA union
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `uv run pytest -o "addopts=" tests/unit/test_peft_target_parameters.py -k never_passes_target_parameters -v`
Expected: FAIL (`apply_lora` still builds `LoraConfig(..., target_parameters=...)` from the
committed wiring, so the kwarg is present in `captured`).

- [ ] **Step 3: Rework the `apply_lora` wiring**

In `src/custom_sam_peft/peft_adapters/lora.py`, in `apply_lora`, after
`matched_names = _resolve_targets(base, cfg)`:

1. Resolve the MHA axis and union it (generic-first, deduped — spec §5.3):

   ```python
       mha_names = _resolve_mha_modules(base, cfg)
       target_modules = matched_names + [n for n in mha_names if n not in matched_names]
   ```

2. Rework the `LoraConfig(...)` construction to use the union and **remove** the
   `target_parameters=...` kwarg (the reverted axis):

   ```python
       lora_cfg = LoraConfig(
           r=cfg.r,
           lora_alpha=cfg.alpha,
           lora_dropout=cfg.dropout,
           target_modules=target_modules,
           bias=cfg.bias,
           task_type=None,
       )
   ```

   For the three legacy scopes `mha_names == []`, so `target_modules == matched_names` and the
   `LoraConfig` is byte-identical to today's (reproducibility). `lora_dropout=cfg.dropout`
   (default `0.05`) is passed unchanged — `lora.MultiheadAttention` supports dropout (spec
   §5.3, §7.3a).

3. Rework the info-log count field from the reverted `n_param_targets` to `n_mha_targets`
   (spec §5.3 step 4):

   ```python
       logger.info(
           "LoRA: trainable=%d (%.2f%%) of %d (scope=%s, n_targets=%d, n_mha_targets=%d)",
           trainable,
           100 * ratio,
           total,
           cfg.scope if cfg.target_modules is None else "<override>",
           len(matched_names),
           len(mha_names),
       )
   ```

   (The `> 0.10` warning is unchanged in structure — spec §8.3.)

- [ ] **Step 2.4 verification: run the wiring test + reproducibility guard**

Run:

```bash
uv run pytest -o "addopts=" tests/unit/test_peft_target_parameters.py tests/unit/test_peft_scope_coverage.py -v
```

Expected: PASS. (`test_peft_scope_coverage.py` confirms legacy scopes still attach exactly as
before — the reproducibility guard.)

- [ ] **Step 4: Lint/type + import smoke**

Run:

```bash
uv run ruff check src/custom_sam_peft/peft_adapters/lora.py
uv run ruff format --check src/custom_sam_peft/peft_adapters/lora.py
uv run mypy --strict src/custom_sam_peft/peft_adapters/lora.py
uv run python -c "import custom_sam_peft"
```

Expected: all pass — `lora.py` is now fully on the MHA-module axis (no `target_parameters`
references remain in it).

- [ ] **Step 5: Commit**

```bash
git add src/custom_sam_peft/peft_adapters/lora.py tests/unit/test_peft_target_parameters.py
git commit -m "feat(#230): union MHA module names into target_modules in apply_lora"
```

### Task 2.5: Rework the QLoRA apply path to union MHA module names

**Files:**

- Modify: `src/custom_sam_peft/peft_adapters/qlora.py` (the committed `target_parameters`
  import + `_inject_lora_adapters` wiring + log line)
- Test: `tests/unit/test_peft_target_parameters.py`

- [ ] **Step 1: Rewrite the QLoRA-parity test (MHA axis is mode-independent)**

In `tests/unit/test_peft_target_parameters.py`, replace the committed
`test_qlora_and_lora_resolve_same_parameter_set` with the MHA-axis equivalent. Also add the
schema-default re-assertions (kept from `44b4d31`) so the default flip stays covered:

```python
def test_qlora_and_lora_resolve_same_mha_set() -> None:
    """§10.3: the MHA axis is mode-independent — same module names for LoRA and QLoRA."""
    base = _MiniBase()
    lora_cfg = PEFTConfig(method="lora", scope="vision_decoder_concept")
    qlora_cfg = PEFTConfig(method="qlora", scope="vision_decoder_concept")
    assert _resolve_mha_modules(base, lora_cfg) == _resolve_mha_modules(base, qlora_cfg)
    got = _resolve_mha_modules(base, lora_cfg)
    assert any(n.endswith("ca_text") for n in got)
    assert any(n.endswith("self_attn") for n in got)


def test_peftconfig_default_scope_is_concept() -> None:
    assert PEFTConfig(method="lora").scope == "vision_decoder_concept"


def test_lorascope_literal_includes_concept() -> None:
    import typing

    from custom_sam_peft.config.schema import LoraScope

    assert set(typing.get_args(LoraScope)) == {
        "vision",
        "vision_decoder",
        "vision_decoder_concept",
        "all",
    }
```

- [ ] **Step 2: Run it to confirm state**

Run: `uv run pytest -o "addopts=" tests/unit/test_peft_target_parameters.py -k "same_mha_set or default_scope or lorascope_literal" -v`
Expected: the resolver/schema assertions PASS already (they assert properties of
`_resolve_mha_modules` + the committed schema). They are the regression guards for the wiring
below.

- [ ] **Step 3: Rework the QLoRA wiring**

In `src/custom_sam_peft/peft_adapters/qlora.py`:

1. Change the committed import from `_resolve_target_parameters` to `_resolve_mha_modules`:

   ```python
   from custom_sam_peft.peft_adapters.lora import _resolve_mha_modules, _resolve_targets
   ```

2. In `_inject_lora_adapters`, after
   `lora_target_names = _resolve_targets(model, cfg, linear_types=(bnb.nn.Linear4bit,))`,
   resolve + union the MHA axis (spec §5.4) and pass the union to `LoraConfig`, removing the
   committed `target_parameters=...` kwarg:

   ```python
       mha_names = _resolve_mha_modules(model, cfg)
       target_modules = lora_target_names + [n for n in mha_names if n not in lora_target_names]
   ```

   and pass `target_modules=target_modules` to the `LoraConfig(...)` (no `target_parameters`).
   The MHA modules stay **unquantized** (`_mha_exclusion_types`), so the MHA LoRA is plain
   bf16 LoRA coexisting with the `Linear4bit` LoRA in one `PeftModel` (spec §5.4, §7.2,
   GPU-confirmed §7.3a). The one-way `qlora.py -> lora.py` import contract is preserved
   (`lora.py` imports neither `qlora.py` nor `bitsandbytes`).

3. Rework the QLoRA log line's count field from the reverted `n_param_targets` to
   `n_mha_targets`:

   ```python
       logger.info(
           "QLoRA: trainable=%d (%.2f%%) of %d "
           "(lora_scope=%s, quant_type=%s, compute_dtype=%s, n_mha_targets=%d)",
           trainable,
           100 * ratio,
           total,
           cfg.scope if cfg.target_modules is None else "<override>",
           cfg.qlora.quant_type,
           cfg.qlora.compute_dtype,
           len(mha_names),
       )
   ```

- [ ] **Step 4: Run the test**

Run: `uv run pytest -o "addopts=" tests/unit/test_peft_target_parameters.py -v`
Expected: PASS.

- [ ] **Step 5: Lint/type + import smoke**

Run:

```bash
uv run ruff check src/custom_sam_peft/peft_adapters/qlora.py
uv run ruff format --check src/custom_sam_peft/peft_adapters/qlora.py
uv run mypy --strict src/custom_sam_peft/peft_adapters/qlora.py
uv run python -c "import custom_sam_peft"
```

Expected: all pass (`mypy --strict` on `qlora.py` exercises the reworked import + union).

- [ ] **Step 6: Commit**

```bash
git add src/custom_sam_peft/peft_adapters/qlora.py tests/unit/test_peft_target_parameters.py
git commit -m "feat(#230): union MHA module names into target_modules in QLoRA apply path"
```

### Task 2.6: Rework the stub fixture — `FIXTURE_SCOPE_MHA_MODULES` + de-overlapped concept patterns

**Files:**

- Modify: `tests/fixtures/tiny_sam3_lora_stub.py` (the committed
  `FIXTURE_SCOPE_TARGET_PARAMETERS` + the `vision_decoder_concept` `FIXTURE_SCOPE_PATTERNS`
  entry)
- Test: `tests/unit/test_peft_target_parameters.py`

`ca_text` / `self_attn` are **already** real `nn.MultiheadAttention` in the stub (committed in
`b283be7`) — **KEEP** that. This task reworks only the fixture **mappings**: rename
`FIXTURE_SCOPE_TARGET_PARAMETERS` → `FIXTURE_SCOPE_MHA_MODULES` (module-name patterns, not
param patterns) and de-overlap the concept `FIXTURE_SCOPE_PATTERNS` entry.

- [ ] **Step 1: Rewrite the fixture test to the MHA mappings**

In `tests/unit/test_peft_target_parameters.py`, replace the committed
`test_fixture_exposes_mha_inproj_and_concept_patterns` with:

```python
def test_fixture_exposes_mha_modules_and_de_overlapped_concept_patterns() -> None:
    from tests.fixtures.tiny_sam3_lora_stub import (
        FIXTURE_SCOPE_MHA_MODULES,
        FIXTURE_SCOPE_PATTERNS,
        make_stub_wrapper,
    )

    w = make_stub_wrapper(dim=8, working=False)
    base = w.model.model
    # ca_text / self_attn are real nn.MultiheadAttention (in_proj_weight exists).
    names = [n for n, _ in base.named_parameters()]
    assert any(n.endswith("ca_text.in_proj_weight") for n in names), names[:10]
    assert any(n.endswith("self_attn.in_proj_weight") for n in names), names[:10]
    # cross_attn must NOT be MHA (negative control for the MHA axis).
    assert not any("cross_attn.in_proj_weight" in n for n in names)

    # The concept fixture mappings exist and are de-overlapped.
    assert "vision_decoder_concept" in FIXTURE_SCOPE_PATTERNS
    assert "vision_decoder_concept" in FIXTURE_SCOPE_MHA_MODULES
    concept_generic = FIXTURE_SCOPE_PATTERNS["vision_decoder_concept"]
    assert not any(("self_attn" in p or "ca_text" in p) and "out_proj" in p for p in concept_generic)
    concept_mha = FIXTURE_SCOPE_MHA_MODULES["vision_decoder_concept"]
    assert any("ca_text" in p for p in concept_mha)
    assert any("self_attn" in p for p in concept_mha)
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `uv run pytest -o "addopts=" tests/unit/test_peft_target_parameters.py -k fixture_exposes -v`
Expected: FAIL (`ImportError` — `FIXTURE_SCOPE_MHA_MODULES` does not exist; the committed
`FIXTURE_SCOPE_TARGET_PARAMETERS` does).

- [ ] **Step 3: Rework the fixture mappings**

In `tests/fixtures/tiny_sam3_lora_stub.py` (keep the committed MHA `_DecoderLayer` children
as-is), de-overlap the `vision_decoder_concept` `FIXTURE_SCOPE_PATTERNS` entry and replace
`FIXTURE_SCOPE_TARGET_PARAMETERS` with `FIXTURE_SCOPE_MHA_MODULES` (fixture-prefixed
**module** patterns; note the truncated `transformer_decoder` prefix). The concept generic
entry drops the `self_attn` out_proj alternative so it does not double-target the MHA wrapper
on the stub:

```python
FIXTURE_SCOPE_PATTERNS: dict[str, list[str]] = {
    "vision": [r"vision_trunk\.blocks\.\d+\.attn\.(qkv|proj)$"],
    "vision_decoder": [
        r"vision_trunk\.blocks\.\d+\.attn\.(qkv|proj)$",
        r"transformer_decoder\.layers\.\d+\.(self_attn|cross_attn)\.out_proj$",
    ],
    # De-overlapped: only cross_attn.out_proj is a generic target; self_attn out_proj
    # is adapted by peft's lora.MultiheadAttention via FIXTURE_SCOPE_MHA_MODULES.
    "vision_decoder_concept": [
        r"vision_trunk\.blocks\.\d+\.attn\.(qkv|proj)$",
        r"transformer_decoder\.layers\.\d+\.cross_attn\.out_proj$",
    ],
    "all": [r".*"],
}

# Parallel to the production SCOPE_MHA_MODULES, but with the fixture's truncated
# `transformer_decoder` prefix. Drives the MHA-module axis on the stub.
FIXTURE_SCOPE_MHA_MODULES: dict[str, list[str]] = {
    "vision_decoder_concept": [
        r"transformer_decoder\.layers\.\d+\.ca_text$",
        r"transformer_decoder\.layers\.\d+\.self_attn$",
    ],
}
```

- [ ] **Step 4: Run the test + the existing stub-driven tests**

Run:

```bash
uv run pytest -o "addopts=" tests/unit/test_peft_target_parameters.py tests/unit/test_peft_scope_coverage.py -v
```

Expected: PASS. `test_peft_scope_coverage.py`'s `working=True` forward/backward test must
still pass — the stub's forward routes through `vision_trunk.blocks[0].attn.qkv`, unaffected
by the fixture-mapping rework.

- [ ] **Step 5: Lint/type + import smoke**

Run:

```bash
uv run ruff check tests/fixtures/tiny_sam3_lora_stub.py tests/unit/test_peft_target_parameters.py
uv run ruff format --check tests/fixtures/tiny_sam3_lora_stub.py tests/unit/test_peft_target_parameters.py
uv run mypy --strict tests/fixtures/tiny_sam3_lora_stub.py
uv run python -c "import custom_sam_peft"
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add tests/fixtures/tiny_sam3_lora_stub.py tests/unit/test_peft_target_parameters.py
git commit -m "test(#230): rework stub to FIXTURE_SCOPE_MHA_MODULES + de-overlapped concept patterns"
```

### Task 2.7: CPU coverage — monkeypatched union path on the stub; example configs; GPU tests

**Files:**

- Modify: `tests/unit/test_peft_target_parameters.py` (monkeypatched `apply_lora` union path)
- Modify: `configs/examples/*` (example-config docs — NO `target_parameters` knob)
- Modify: `tests/integration/test_peft_lora_real.py` and `tests/integration/test_peft_qlora_real.py`
  (rework the Phase-1 `target_parameters`-override spike tests to the real
  `scope="vision_decoder_concept"`)

**CPU-harness note (spec §10.2, must be honored).** Production `SCOPE_MHA_MODULES` and
`SCOPE_TARGETS["vision_decoder_concept"]` use the real `transformer.decoder` prefix, which
does **not** match the stub's truncated `transformer_decoder` prefix. So a stub call to
`apply_lora(scope="vision_decoder_concept")` **without intervention** makes
`_resolve_mha_modules` raise non-empty-no-match. To exercise `apply_lora`'s **real union path**
on the stub, the CPU test must **monkeypatch the production** `SCOPE_MHA_MODULES` **and**
`SCOPE_TARGETS["vision_decoder_concept"]` to the fixture-prefixed patterns (from
`FIXTURE_SCOPE_MHA_MODULES` and `FIXTURE_SCOPE_PATTERNS["vision_decoder_concept"]`), then call
`apply_lora`. (The focused `_resolve_mha_modules` unit test against `_MiniBase` in Task 2.3
needs no monkeypatch — `_MiniBase` uses the real prefix.)

- [ ] **Step 1: Write the monkeypatched union test + the `all`-never-reaches-MHA hard test**

Add to `tests/unit/test_peft_target_parameters.py`:

```python
def _lora_param_names(wrapper: object) -> list[str]:
    return [n for n, _ in wrapper.model.model.named_parameters() if "lora_" in n]


def test_concept_scope_unions_modules_and_mha_on_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    """§10.2: with production scope dicts monkeypatched to fixture prefixes, apply_lora's
    real union path attaches generic-module LoRA + MHA in_proj/out_proj LoRA, and NOT on
    cross_attn (negative control)."""
    import custom_sam_peft.peft_adapters.lora as lora_mod
    from tests.fixtures.tiny_sam3_lora_stub import (
        FIXTURE_SCOPE_MHA_MODULES,
        FIXTURE_SCOPE_PATTERNS,
        make_stub_wrapper,
    )

    monkeypatch.setitem(
        lora_mod.SCOPE_MHA_MODULES, "vision_decoder_concept",
        FIXTURE_SCOPE_MHA_MODULES["vision_decoder_concept"],
    )
    monkeypatch.setitem(
        lora_mod.SCOPE_TARGETS, "vision_decoder_concept",
        FIXTURE_SCOPE_PATTERNS["vision_decoder_concept"],
    )
    w = make_stub_wrapper(dim=8, working=False)
    lora_mod.apply_lora(w, PEFTConfig(method="lora", scope="vision_decoder_concept"))

    names = _lora_param_names(w)
    assert any("vision_trunk.blocks" in n for n in names)
    assert any("ca_text" in n and "lora" in n for n in names), names[:10]
    assert any("self_attn" in n and "lora" in n for n in names), names[:10]
    assert not any("cross_attn" in n and "in_proj" in n for n in names)
    # Trainable ratio sane (stub is tiny; loose bound mirrors existing style).
    base = w.model.model
    trainable = sum(p.numel() for p in base.parameters() if p.requires_grad)
    total = sum(p.numel() for p in base.parameters())
    assert trainable / total < 0.5


def test_all_scope_never_reaches_mha_on_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    """§4.3 HARD: the 'all' scope's .* lives only in _resolve_targets (nn.Linear); it can
    never reach an nn.MultiheadAttention module."""
    from tests.fixtures.tiny_sam3_lora_stub import FIXTURE_SCOPE_PATTERNS, make_stub_wrapper
    from custom_sam_peft.peft_adapters.lora import _resolve_mha_modules

    base = make_stub_wrapper(dim=8, working=False).model.model
    assert _resolve_mha_modules(base, PEFTConfig(method="lora", scope="all")) == []
```

- [ ] **Step 2: Run them to confirm they pass**

Run: `uv run pytest -o "addopts=" tests/unit/test_peft_target_parameters.py -k "unions_modules or all_scope_never" -v`
Expected: PASS (all wiring from Tasks 2.2–2.6 is in place). If `apply_lora` raises a peft error
attaching `lora.MultiheadAttention` to the stub's MHA on CPU, that contradicts the §7.3a
GPU-confirmed result — escalate per the design-ambiguity ladder; do not weaken the test.

- [ ] **Step 3: Update example configs (NO `target_parameters` knob)**

Find every commented PEFT knob block:

```bash
grep -rln "# scope:" configs/examples
```

In each matched file, update only the commented knob block (spec §6.4): list
`vision_decoder_concept` as the default in the `# scope:` line and note it is the new shipped
default (adapts `ca_text` / `self_attn` MHA in_proj + out_proj for text concepts). Leave the
existing commented `# target_modules:` knob as-is. **Do NOT add a `# target_parameters:` knob**
(the field is reverted). No uncommented value changes — defaults already apply. Example block:

```yaml
  # Knobs (defaults shown — uncomment to override):
  # scope: vision_decoder_concept  # vision | vision_decoder | vision_decoder_concept | all
  #                                # (default; adapts ca_text/self_attn MHA in_proj+out_proj
  #                                #  for text concepts)
  # bias: none                     # none | all | lora_only
  # target_modules: [...]          # overrides scope's module patterns when set
```

Validate each edited config still loads:

```bash
uv run python -c "from custom_sam_peft.config.loader import load_config; load_config('configs/examples/coco_text_lora.yaml')"
```

(Repeat for each edited path / loop over the grep list.)

- [ ] **Step 4: Rework the GPU integration tests to the real scope**

In `tests/integration/test_peft_lora_real.py`, **replace** the Phase-1
`test_spike_inproj_lora_*` test (which used `target_parameters` overrides — now reverted) with
the productionized §10.4 test, and delete the spike's `_INPROJ_PARAM_PATTERNS` /
`_VISION_DECODER_MODULE_PATTERNS` helpers (keep the recorded go/no-go in the PR / a comment):

```python
def test_concept_scope_inproj_on_real_sam31() -> None:
    """§10.4: scope='vision_decoder_concept' attaches MHA in_proj+out_proj LoRA, merges,
    ratio<5%."""
    w = load_sam31(ModelConfig())
    apply_lora(w, PEFTConfig(method="lora", scope="vision_decoder_concept"))

    lora_names = [n for n, _ in w.model.model.named_parameters() if "lora_" in n]
    assert any("ca_text" in n for n in lora_names), f"no ca_text MHA LoRA: {lora_names[:8]}"
    assert any("self_attn" in n for n in lora_names), f"no self_attn MHA LoRA: {lora_names[:8]}"

    trainable = sum(p.numel() for p in w.model.model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in w.model.model.parameters())
    assert trainable / total < 0.05, "concept scope exceeds 5% trainable budget"

    merge_lora(w)
    assert w.peft_model is None
```

In `tests/integration/test_peft_qlora_real.py`, mirror this for QLoRA with
`scope="vision_decoder_concept"` (replacing the Phase-1 `target_parameters`-override
`test_spike_inproj_qlora_*` test), asserting the bf16 MHA LoRA coexists with the `Linear4bit`
module LoRA and `merge_lora` folds both in one `PeftModel` (§7.2; GPU-confirmed §7.3a — merge
emits only a benign NF4-rounding `UserWarning`). Remove the spike's local
`_INPROJ_PARAM_PATTERNS` from that file too.

- [ ] **Step 5: Verify compile + lint + type (off-GPU) + run CPU subset**

Run:

```bash
uv run python -m py_compile tests/integration/test_peft_lora_real.py tests/integration/test_peft_qlora_real.py
uv run ruff check tests/integration/test_peft_lora_real.py tests/integration/test_peft_qlora_real.py tests/unit/test_peft_target_parameters.py configs/examples
uv run ruff format --check tests/integration/test_peft_lora_real.py tests/integration/test_peft_qlora_real.py tests/unit/test_peft_target_parameters.py
uv run mypy --strict tests/integration/test_peft_lora_real.py tests/integration/test_peft_qlora_real.py
uv run pytest -o "addopts=" tests/unit/test_peft_target_parameters.py -v
```

Expected: all pass (GPU test bodies remain gated/skipped off-GPU). On the GPU runner,
`scripts/run_gpu_tests.sh local` executes the new concept-scope tests; record the empirical
trainable ratio (confirms §8.3 — leave the 10% guard / `< 0.05` budget unchanged unless
reality demands; a threshold change would need a `# tbd:`).

- [ ] **Step 6: Commit**

```bash
git add tests/unit/test_peft_target_parameters.py configs/examples tests/integration/test_peft_lora_real.py tests/integration/test_peft_qlora_real.py
git commit -m "test(#230): CPU union-path + GPU concept-scope tests; example-config docs (no target_parameters knob)"
```

### Task 2.8: Phase 2 verification before completion

**Files:** none (verification only)

- [ ] **Step 1: Full CPU suite + blast-radius grep (incl. revert completeness)**

Run:

```bash
# New symbols are present and wired:
grep -rn "SCOPE_MHA_MODULES\|_resolve_mha_modules\|vision_decoder_concept" src tests
# REVERT COMPLETENESS: the old axis must be GONE everywhere.
grep -rn "target_parameters\|SCOPE_TARGET_PARAMETERS\|_resolve_target_parameters" src tests
uv run pytest -o "addopts=" tests/unit tests/integration -q
uv run python -c "import custom_sam_peft"
```

Expected: the first grep shows the new symbols; the **second grep returns NOTHING** (no
`target_parameters` / `SCOPE_TARGET_PARAMETERS` / `_resolve_target_parameters` reference
remains in `src/` or `tests/`); the full CPU suite PASSES (GPU tests skipped); the package
imports clean.

- [ ] **Step 2: Lint/type the whole touched set**

Run:

```bash
uv run ruff check src/custom_sam_peft/peft_adapters/lora.py src/custom_sam_peft/peft_adapters/qlora.py src/custom_sam_peft/config/schema.py tests/fixtures/tiny_sam3_lora_stub.py tests/unit/test_peft_target_parameters.py tests/integration/test_peft_lora_real.py tests/integration/test_peft_qlora_real.py
uv run ruff format --check src/custom_sam_peft/peft_adapters/lora.py src/custom_sam_peft/peft_adapters/qlora.py src/custom_sam_peft/config/schema.py tests/fixtures/tiny_sam3_lora_stub.py
uv run mypy --strict src/custom_sam_peft/peft_adapters/lora.py src/custom_sam_peft/peft_adapters/qlora.py src/custom_sam_peft/config/schema.py tests/fixtures/tiny_sam3_lora_stub.py
```

Expected: all pass. Fix any finding, then the phase is complete.

**Phase 2 handoff line (literal):**
`Resume phase. Next: 3. Plan: <this plan path>. Worktree: <full path>.`

---

## Phase 3 — calibrate VRAM-autosize alpha co-scale + WARNING (ORTHOGONAL)

**Feature block:** When the pre-flight VRAM calibrate autosize selects a final LoRA rank
`r_final < cfg.peft.r`, co-scale `alpha` to preserve the configured `alpha:r` ratio, persist
`alpha` through the whole calibrate chain, and emit one WARNING. No-op (byte-identical to
today) when rank is not reduced. `oom.py::OomLadder` is untouched.

**ORTHOGONALITY (call-out for the orchestrator):** This phase is **file-disjoint** from
Phases 1–2 and does **not** depend on the spike or the MHA-module axis. It touches only:
`src/custom_sam_peft/cli/calibrate_cmd.py`, `src/custom_sam_peft/cli/_config_rewrite.py`,
`src/custom_sam_peft/presets.py` (`PresetDecision` only), and
`tests/unit/test_calibrate_cmd.py`. It may run **in parallel** with Phase 2 on the
same branch/worktree. **Serialize commits** with the other phases (parallel agents
committing on one branch can orphan a commit) — but the work itself is independent.

**Interface contract this phase PRODUCES:**

- `PresetDecision.alpha: int` (new field, adjacent to `r`).
- v3 cache key `chosen_alpha` (additive; written by `_write_cache_v3`, read by
  `_decision_from_cache`).
- `_rewrite_sizing_block(..., alpha: int, ...)` — a 6th direct `(peft, alpha)` rewrite
  target.
- Co-scale rule: `alpha_final = round(cfg.peft.alpha * r_final / cfg.peft.r)` when
  `r_final < cfg.peft.r`; else `alpha_final = cfg.peft.alpha`. No new `# tbd:` (justified by
  the existing `alpha = 2r` citation, spec §7a.5).

### Task 3.1: Add `alpha` to `PresetDecision` and thread it through the config_patch

**Files:**

- Modify: `src/custom_sam_peft/presets.py:91-159` (`PresetDecision`)
- Test: `tests/unit/test_presets.py` (locate the existing PresetDecision tests via
  `grep -rln "PresetDecision" tests/unit`)

`PresetDecision` is a `@dataclass(frozen=True)`. Adding a required field is a blast-radius
change — every constructor must pass it. There are constructors in `calibrate_cmd.py`
(`_decision_from_cache:318`, the post-confirm `PresetDecision(...):498`), and possibly in
`presets.py::decide_preset` and tests. Give the field a **default** so existing
constructors that do not yet pass `alpha` stay valid, then update the calibrate constructors
in Tasks 3.4–3.5.

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/test_presets.py`:

```python
def test_presetdecision_has_alpha_field_and_in_config_patch() -> None:
    from custom_sam_peft.presets import PresetDecision

    d = PresetDecision(
        method="lora", r=8, alpha=16, batch_size=1, grad_accum_steps=8,
        classes_per_forward=1, dtype="bfloat16", headroom_bytes=0,
        predicted_bytes=0, budget_bytes=0, gpu_name="X",
        provenance="calibrated", cache_path=None, calibrated_at=None,
    )
    assert d.alpha == 16
    assert d.config_patch["peft"]["alpha"] == 16
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `uv run pytest -o "addopts=" tests/unit/test_presets.py -k has_alpha_field -v`
Expected: FAIL (`TypeError` — no `alpha` field; `config_patch` lacks `peft.alpha`).

- [ ] **Step 3: Add the field + patch entry**

In `src/custom_sam_peft/presets.py`, add `alpha` adjacent to `r` (after `:102`) with a
default so existing constructors keep compiling:

```python
    r: int
    alpha: int = 32  # cite: LoRA (Hu 2021) §4.1 (alpha = 2r); co-scaled by calibrate autosize
```

In `config_patch` (`:116-126`), add `alpha` to the `peft` section:

```python
            "peft": {"method": self.method, "r": self.r, "alpha": self.alpha},
```

- [ ] **Step 4: Run the test + the full presets suite (blast radius)**

Run:

```bash
grep -rn "PresetDecision(" src tests
uv run pytest -o "addopts=" tests/unit/test_presets.py -v
```

Expected: PASS. Confirm no other `PresetDecision(...)` constructor breaks (the default makes
`alpha` optional for now).

- [ ] **Step 5: Lint/type + import smoke**

Run:

```bash
uv run ruff check src/custom_sam_peft/presets.py
uv run ruff format --check src/custom_sam_peft/presets.py
uv run mypy --strict src/custom_sam_peft/presets.py
uv run python -c "import custom_sam_peft"
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/custom_sam_peft/presets.py tests/unit/test_presets.py
git commit -m "feat(#230): add alpha to PresetDecision + config_patch"
```

### Task 3.2: Add `alpha` to `_rewrite_sizing_block`

**Files:**

- Modify: `src/custom_sam_peft/cli/_config_rewrite.py:23-88` (signature + `replacements`
  map + docstring "5 direct targets" -> "6")
- Test: `tests/unit/test_config_rewrite.py` (locate via
  `grep -rln "_rewrite_sizing_block" tests/unit`)

- [ ] **Step 1: Write the failing test**

Add to the config-rewrite test module:

```python
def test_rewrite_sizing_block_rewrites_peft_alpha(tmp_path: Path) -> None:
    import yaml

    from custom_sam_peft.cli._config_rewrite import _rewrite_sizing_block

    cfg = tmp_path / "c.yaml"
    cfg.write_text(
        "model:\n  dtype: bfloat16\n"
        "peft:\n  method: lora\n  r: 16\n  alpha: 32  # keep comment\n"
        "train:\n  batch_size: 1\n  grad_accum_steps: 8\n"
        "  multiplex:\n    classes_per_forward: 16\n"
    )
    _rewrite_sizing_block(
        cfg, method="lora", r=8, alpha=16, batch_size=1, grad_accum_steps=8,
        classes_per_forward=16, dtype="bfloat16", annotation="# calibrated 2026-06-01",
    )
    parsed = yaml.safe_load(cfg.read_text())
    assert parsed["peft"]["r"] == 8
    assert parsed["peft"]["alpha"] == 16
    assert "# keep comment" in cfg.read_text()  # inline comment preserved
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `uv run pytest -o "addopts=" tests/unit/test_config_rewrite.py -k rewrites_peft_alpha -v`
Expected: FAIL (`TypeError` — `_rewrite_sizing_block` has no `alpha` parameter).

- [ ] **Step 3: Add the `alpha` parameter + replacement target**

In `src/custom_sam_peft/cli/_config_rewrite.py`, add `alpha: int,` to the keyword-only
signature (`:23-33`, alongside `r: int`):

```python
    r: int,
    alpha: int,
```

Add the target to the `replacements` map (`:82-88`), directly after `("peft", "r")`:

```python
        ("peft", "r"): str(r),
        ("peft", "alpha"): str(alpha),
```

Update the docstring count references from "5 direct (section, key) targets" to "6" (`:42`
and `:63`, the `# Targets:` comment), adding `peft.alpha` to the listed targets. The existing
line-surgery / annotation / idempotency / missing-key logic handles the new target with no
further change (it is a 6th direct child of the `peft` section, exactly like `peft.r`).

- [ ] **Step 4: Run the test**

Run: `uv run pytest -o "addopts=" tests/unit/test_config_rewrite.py -v`
Expected: PASS (including the new alpha test and all existing rewrite tests).

- [ ] **Step 5: Lint/type + import smoke**

Run:

```bash
uv run ruff check src/custom_sam_peft/cli/_config_rewrite.py
uv run ruff format --check src/custom_sam_peft/cli/_config_rewrite.py
uv run mypy --strict src/custom_sam_peft/cli/_config_rewrite.py
uv run python -c "import custom_sam_peft"
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/custom_sam_peft/cli/_config_rewrite.py tests/unit/test_config_rewrite.py
git commit -m "feat(#230): rewrite peft.alpha alongside peft.r in sizing block"
```

### Task 3.3: Persist `chosen_alpha` in the v3 cache (write + read)

**Files:**

- Modify: `src/custom_sam_peft/cli/calibrate_cmd.py:246-287` (`_write_cache_v3`) and
  `:290-332` (`_decision_from_cache`)
- Test: `tests/unit/test_calibrate_cmd.py`

- [ ] **Step 1: Write the failing test (round-trip)**

Add to `tests/unit/test_calibrate_cmd.py`:

```python
def test_write_and_read_chosen_alpha_round_trip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from custom_sam_peft.cli import calibrate_cmd

    _patch_probe(monkeypatch, tmp_path=tmp_path)
    out = tmp_path / "cache.json"
    calibrate_cmd._write_cache_v3(
        out, gpu_name="X", total=int(16 * _GB), a_fixed=1, a_per_class=1, peak=123,
        method="lora", r=8, alpha=16, batch=1, classes_per_forward=8,
    )
    data = json.loads(out.read_text())
    assert data["chosen_alpha"] == 16
    decision = calibrate_cmd._decision_from_cache(out, k_cap=16)
    assert decision is not None
    assert decision.r == 8
    assert decision.alpha == 16
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `uv run pytest -o "addopts=" tests/unit/test_calibrate_cmd.py -k chosen_alpha_round_trip -v`
Expected: FAIL (`TypeError` — `_write_cache_v3` has no `alpha` kwarg; `decision.alpha`
absent).

- [ ] **Step 3: Add `alpha` to write + read**

In `_write_cache_v3` (`:246-258`), add `alpha: int | None = None,` to the signature (next to
`r`), and the additive payload key (`:276-283`, mirroring `chosen_r`):

```python
    if r is not None:
        payload["chosen_r"] = int(r)
    if alpha is not None:
        payload["chosen_alpha"] = int(alpha)
```

In `_decision_from_cache` (`:290-332`), read it back with a backward-compatible fallback
(legacy caches lack `chosen_alpha`; default to `2 * r` so an old cache still yields the cited
ratio), and pass it to the `PresetDecision(...)` (`:318-332`):

```python
    r = int(data["chosen_r"])
    alpha = int(data.get("chosen_alpha", 2 * r))
```

and add `alpha=alpha,` to the `PresetDecision(...)` constructor.

- [ ] **Step 4: Run the test**

Run: `uv run pytest -o "addopts=" tests/unit/test_calibrate_cmd.py -k chosen_alpha_round_trip -v`
Expected: PASS.

- [ ] **Step 5: Lint/type + import smoke**

Run:

```bash
uv run ruff check src/custom_sam_peft/cli/calibrate_cmd.py
uv run ruff format --check src/custom_sam_peft/cli/calibrate_cmd.py
uv run mypy --strict src/custom_sam_peft/cli/calibrate_cmd.py
uv run python -c "import custom_sam_peft"
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/custom_sam_peft/cli/calibrate_cmd.py tests/unit/test_calibrate_cmd.py
git commit -m "feat(#230): persist chosen_alpha in v3 cache (write + read)"
```

### Task 3.4: Co-scale alpha + WARNING in `run_calibration`; pass alpha through the rewrite

**Files:**

- Modify: `src/custom_sam_peft/cli/calibrate_cmd.py:159-188` (`_apply_config_rewrite` — pass
  `decision.alpha`), `:411-514` (`run_calibration` — compute `alpha_final`, warn, persist)
- Test: `tests/unit/test_calibrate_cmd.py`

- [ ] **Step 1: Write the failing tests (reduce -> co-scale + warn; no-reduce -> no warn)**

Add to `tests/unit/test_calibrate_cmd.py` (the reduce path reuses the `r=64` under-fit
pattern from the existing `test_run_calibration_reduces_r_on_under_fit:452`):

```python
def test_calibrate_reduce_coscales_alpha_and_warns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import yaml

    from custom_sam_peft.cli import calibrate_cmd

    # Configure alpha=2r so co-scale is exact: r=64 -> r_final<=16 -> alpha=2*r_final.
    _patch_probe(monkeypatch, tmp_path=tmp_path, gpu_name="SmallGPU", total=int(16 * _GB))
    _write_config_with_alpha(tmp_path / "config.yaml", method="lora", r=64, alpha=128, k=16)

    def _probe(**kw):
        if kw["r"] > 16 or kw["batch"] > 1 or kw["k_eff"] > 1:
            raise torch.cuda.OutOfMemoryError("synthetic")
        return _synthetic_peak(**kw)

    monkeypatch.setattr(calibrate_cmd, "_run_probe", _probe)
    monkeypatch.chdir(tmp_path)
    out = tmp_path / ".custom_sam_peft_calibration.json"
    decision = calibrate_cmd.run_calibration(config=tmp_path / "config.yaml", output=out, force=True)

    assert decision.r <= 16
    assert decision.alpha == 2 * decision.r  # preserved alpha:r ratio (was 128:64 = 2:1)
    cfg = yaml.safe_load((tmp_path / "config.yaml").read_text())
    assert cfg["peft"]["alpha"] == decision.alpha
    assert json.loads(out.read_text())["chosen_alpha"] == decision.alpha


def test_calibrate_no_reduction_leaves_alpha_untouched_no_warn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import yaml

    from custom_sam_peft.cli import calibrate_cmd

    _patch_probe(monkeypatch, tmp_path=tmp_path)  # A100/40GB: fits at configured r
    _write_config_with_alpha(tmp_path / "config.yaml", method="lora", r=16, alpha=32, k=16)
    monkeypatch.setattr(calibrate_cmd, "_run_probe", lambda **kw: _synthetic_peak(**kw))
    monkeypatch.chdir(tmp_path)
    out = tmp_path / ".custom_sam_peft_calibration.json"
    decision = calibrate_cmd.run_calibration(config=tmp_path / "config.yaml", output=out, force=True)

    assert decision.r == 16
    assert decision.alpha == 32  # untouched
    cfg = yaml.safe_load((tmp_path / "config.yaml").read_text())
    assert cfg["peft"]["alpha"] == 32


def test_calibrate_custom_ratio_preserved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """§7a.3(a): a non-2r ratio is preserved (NOT forced to alpha=2r)."""
    from custom_sam_peft.cli import calibrate_cmd

    _patch_probe(monkeypatch, tmp_path=tmp_path, gpu_name="SmallGPU", total=int(16 * _GB))
    _write_config_with_alpha(tmp_path / "config.yaml", method="lora", r=64, alpha=64, k=16)  # 1:1

    def _probe(**kw):
        if kw["r"] > 8 or kw["batch"] > 1 or kw["k_eff"] > 1:
            raise torch.cuda.OutOfMemoryError("synthetic")
        return _synthetic_peak(**kw)

    monkeypatch.setattr(calibrate_cmd, "_run_probe", _probe)
    monkeypatch.chdir(tmp_path)
    out = tmp_path / ".custom_sam_peft_calibration.json"
    decision = calibrate_cmd.run_calibration(config=tmp_path / "config.yaml", output=out, force=True)
    assert decision.r == 8
    assert decision.alpha == 8  # 1:1 ratio preserved, NOT 16
```

Add the `_write_config_with_alpha` helper near `_write_config` (`:23`) — a copy that also
emits `alpha: {alpha}` in the `peft` block:

```python
def _write_config_with_alpha(path: Path, *, method: str, r: int, alpha: int, k: int) -> None:
    _write_config(path, method=method, r=r, k=k)
    text = path.read_text().replace(f"  r: {r}\n", f"  r: {r}\n  alpha: {alpha}\n")
    path.write_text(text)
```

- [ ] **Step 2: Run them to confirm they fail**

Run: `uv run pytest -o "addopts=" tests/unit/test_calibrate_cmd.py -k "coscales_alpha or leaves_alpha_untouched or custom_ratio" -v`
Expected: FAIL (alpha is never co-scaled / persisted / passed through; the rewrite call has
no `alpha`).

- [ ] **Step 3: Implement co-scale + WARNING + threading**

In `run_calibration`, after the empirical tuple is returned by `_confirm_and_climb`
(`:467-474`) and before the final `_write_cache_v3` (`:479`), compute the co-scaled alpha and
warn once:

```python
    # Co-scale alpha to preserve the configured alpha:r ratio when autosize reduced r.
    # Justified by the existing alpha=2r citation (LoRA Hu 2021 §4.1); no new # tbd:.
    # No-op (byte-identical to today) when r is not reduced (spec §7a.3(d)).
    if r < cfg.peft.r:
        alpha_final = round(cfg.peft.alpha * r / cfg.peft.r)
        alpha_final = max(1, alpha_final)  # PositiveInt invariant
        typer.echo(
            f"WARNING: VRAM autosize reduced LoRA rank r {cfg.peft.r}->{r} to fit {gpu_name}; "
            f"alpha co-scaled {cfg.peft.alpha}->{alpha_final} to preserve alpha/r scaling.",
            err=True,
        )
    else:
        alpha_final = cfg.peft.alpha
```

Pass `alpha=alpha_final` to the final `_write_cache_v3(...)` (`:479-490`) and to the
authoritative `PresetDecision(...)` (`:498-512`).

In `_apply_config_rewrite` (`:173-182`), pass the decision's alpha through to the rewrite:

```python
        _rewrite_sizing_block(
            config,
            method=decision.method,
            r=decision.r,
            alpha=decision.alpha,
            batch_size=decision.batch_size,
            grad_accum_steps=decision.grad_accum_steps,
            classes_per_forward=decision.classes_per_forward,
            dtype=decision.dtype,
            annotation=annotation,
        )
```

Note: `r > cfg.peft.r` cannot occur (the climb never raises `r`); the `else` branch covers
`r == cfg.peft.r` and the impossible `>` as the byte-identical no-op (spec §7a.3, §7a final
para).

- [ ] **Step 4: Run the new + existing calibrate tests**

Run: `uv run pytest -o "addopts=" tests/unit/test_calibrate_cmd.py -v`
Expected: the three new tests PASS and **all existing calibrate tests still pass** — in
particular the no-reduction paths must remain byte-identical for alpha (the cache-fresh
early-return at `:439-450` does not re-warn; it reconstructs alpha from `chosen_alpha`).

- [ ] **Step 5: Lint/type + import smoke**

Run:

```bash
uv run ruff check src/custom_sam_peft/cli/calibrate_cmd.py tests/unit/test_calibrate_cmd.py
uv run ruff format --check src/custom_sam_peft/cli/calibrate_cmd.py tests/unit/test_calibrate_cmd.py
uv run mypy --strict src/custom_sam_peft/cli/calibrate_cmd.py
uv run python -c "import custom_sam_peft"
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/custom_sam_peft/cli/calibrate_cmd.py tests/unit/test_calibrate_cmd.py
git commit -m "feat(#230): co-scale alpha on VRAM-autosize rank reduction + WARNING"
```

### Task 3.5: Assert `oom.py` is untouched + add the WARNING-text test

**Files:**

- Test: `tests/unit/test_calibrate_cmd.py`
- Verify (read-only): `src/custom_sam_peft/oom.py` is unchanged in the diff

- [ ] **Step 1: Add a WARNING-text assertion via captured stderr**

Add to `tests/unit/test_calibrate_cmd.py` (extend the reduce-path test or add a focused one)
asserting the WARNING string is emitted on the reduce path and **absent** on the no-reduce
path. Capture stderr the way the existing tests do (the calibrate WARNING uses
`typer.echo(..., err=True)`; `CliRunner`/`capsys` captures it):

```python
def test_calibrate_reduce_emits_warning_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from custom_sam_peft.cli import calibrate_cmd

    _patch_probe(monkeypatch, tmp_path=tmp_path, gpu_name="SmallGPU", total=int(16 * _GB))
    _write_config_with_alpha(tmp_path / "config.yaml", method="lora", r=64, alpha=128, k=16)

    def _probe(**kw):
        if kw["r"] > 16 or kw["batch"] > 1 or kw["k_eff"] > 1:
            raise torch.cuda.OutOfMemoryError("synthetic")
        return _synthetic_peak(**kw)

    monkeypatch.setattr(calibrate_cmd, "_run_probe", _probe)
    monkeypatch.chdir(tmp_path)
    calibrate_cmd.run_calibration(
        config=tmp_path / "config.yaml", output=tmp_path / "c.json", force=True
    )
    err = capsys.readouterr().err
    assert "VRAM autosize reduced LoRA rank" in err
    assert "alpha co-scaled" in err
```

- [ ] **Step 2: Run it**

Run: `uv run pytest -o "addopts=" tests/unit/test_calibrate_cmd.py -k emits_warning_text -v`
Expected: PASS.

- [ ] **Step 3: Confirm `oom.py` was not modified**

Run:

```bash
git status --porcelain src/custom_sam_peft/oom.py
git diff --stat -- src/custom_sam_peft/oom.py
```

Expected: empty output (the runtime OOM ladder is out of scope per spec §7a.4 — it changes
B/K only, never rank/alpha).

- [ ] **Step 4: Lint/type**

Run:

```bash
uv run ruff check tests/unit/test_calibrate_cmd.py
uv run ruff format --check tests/unit/test_calibrate_cmd.py
uv run mypy --strict tests/unit/test_calibrate_cmd.py
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add tests/unit/test_calibrate_cmd.py
git commit -m "test(#230): assert co-scale WARNING text; oom.py untouched"
```

### Task 3.6: Phase 3 verification before completion

**Files:** none (verification only)

- [ ] **Step 1: Full CPU suite + blast-radius grep**

Run:

```bash
grep -rn "PresetDecision(\|_rewrite_sizing_block(\|_write_cache_v3(\|chosen_alpha" src tests
uv run pytest -o "addopts=" tests/unit -q
uv run python -c "import custom_sam_peft"
```

Expected: full CPU unit suite PASSES; every `PresetDecision(...)` / `_rewrite_sizing_block(...)`
/ `_write_cache_v3(...)` call site compiles with the new `alpha` axis; package imports clean.

- [ ] **Step 2: Lint/type the whole touched set**

Run:

```bash
uv run ruff check src/custom_sam_peft/cli/calibrate_cmd.py src/custom_sam_peft/cli/_config_rewrite.py src/custom_sam_peft/presets.py tests/unit/test_calibrate_cmd.py tests/unit/test_config_rewrite.py tests/unit/test_presets.py
uv run ruff format --check src/custom_sam_peft/cli/calibrate_cmd.py src/custom_sam_peft/cli/_config_rewrite.py src/custom_sam_peft/presets.py
uv run mypy --strict src/custom_sam_peft/cli/calibrate_cmd.py src/custom_sam_peft/cli/_config_rewrite.py src/custom_sam_peft/presets.py
```

Expected: all pass. Fix any finding; the phase is then complete.

---

## Final acceptance cross-check (spec §11)

Before opening the PR, confirm each spec §11 criterion maps to delivered work:

1. **Spike resolved first** — Phase 1 DONE; go/no-go = GO, mechanism = `lora.MultiheadAttention`
   (§7.3a); `target_parameters` reverted.
2. **New scope** — Task 2.2 (`vision_decoder_concept`; `SCOPE_TARGETS` entry =
   `vision_decoder`'s generic set **minus** `self_attn`/`ca_text` out_proj **plus** an
   `SCOPE_MHA_MODULES` entry; legacy scopes byte-identical, asserted in Task 2.2 /
   `test_peft_scope_coverage.py`).
3. **New default** — kept from `44b4d31`, re-asserted in Task 2.5 (`# tbd: #230`;
   reproducibility note in code + spec).
4. **Resolution axis** — Tasks 2.2–2.5 (`SCOPE_MHA_MODULES`, `_resolve_mha_modules`, both
   apply paths union matched MHA module names into `target_modules`; legacy adds no MHA
   targets). `target_parameters` field/resolver/dict/wiring reverted (Tasks 2.1–2.5).
5. **No override field** — Task 2.1 (`target_parameters` reverted; `target_modules` owns the
   module axis and suppresses the scope's MHA union when set, asserted in Task 2.3).
6. **QLoRA coexistence** — Tasks 2.5, 2.7 (one `PeftModel`, attach+forward+merge with
   `dropout=0.05`; GPU-confirmed §7.3a).
7. **Error parity** — Task 2.3 (non-empty-no-match `ValueError`; empty resolution / override
   is fine).
8. **Trainable-ratio guard** — Tasks 2.7 (empirically < 5% budget; 10% guard unchanged).
9. **Calibrate alpha co-scale** — Phase 3 (Tasks 3.1–3.5); `oom.py` untouched (Task 3.5);
   no new `# tbd:`.
10. **Tests/fixtures** — Task 2.6 (stub MHA + `FIXTURE_SCOPE_MHA_MODULES`), Tasks 2.7 / 3.x
    (CPU monkeypatched union + GPU); coverage >= 80% (trust CI).
11. **Lint/type** — every task's lint/type step; the markdown gate for this plan + the spec.

## Self-review notes (placeholders/types checked)

- Every code step shows complete code (no TBD / "add error handling" placeholders).
- Type/name consistency: `_resolve_mha_modules(base, cfg) -> list[str]`, `SCOPE_MHA_MODULES`,
  `FIXTURE_SCOPE_MHA_MODULES`, the de-overlapped `SCOPE_TARGETS["vision_decoder_concept"]`,
  `target_modules = matched_names + [n for n in mha_names if n not in matched_names]`,
  `PresetDecision.alpha`, `_rewrite_sizing_block(..., alpha, ...)`, `chosen_alpha`, and
  `alpha_final = round(cfg.peft.alpha * r / cfg.peft.r)` are used identically everywhere they
  appear. No `target_parameters` / `SCOPE_TARGET_PARAMETERS` / `_resolve_target_parameters`
  reference remains (revert completeness, Task 2.8).
- Phase boundaries each publish an explicit interface contract; Phase 1 is DONE and gates
  Phase 2; Phase 3 is orthogonal and parallelizable (commits serialized).
