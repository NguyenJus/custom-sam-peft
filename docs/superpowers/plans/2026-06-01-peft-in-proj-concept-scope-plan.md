# PEFT in_proj Concept Scope Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add one new LoRA scope `vision_decoder_concept` (the new default) that reaches the
decoder's `ca_text` / `self_attn` `in_proj_weight` via peft's `target_parameters` axis, and
make the pre-flight VRAM calibrate autosize co-scale `alpha` when it reduces rank `r`.

**Architecture:** A feasibility spike against the real SAM 3.1 decoder gates the in_proj
surface (Phase 1). On a go, a parameter-name resolution axis (`SCOPE_TARGET_PARAMETERS` +
`_resolve_target_parameters`) is wired through both `apply_lora` and the QLoRA apply path,
a new scope/default and `target_parameters` override field land in the schema, and fixtures
plus CPU/GPU tests cover both axes (Phase 2). An orthogonal, file-disjoint change makes
calibrate persist and warn on a co-scaled `alpha` (Phase 3).

**Tech Stack:** Python 3.12, PyTorch, HuggingFace `peft` 0.19.1, `bitsandbytes` (QLoRA),
pydantic v2, typer, pytest. SAM 3.1 (`sam3`) for the real-decoder GPU tests.

**Spec (source of truth):** `docs/superpowers/specs/2026-06-01-peft-in-proj-concept-scope-design.md`
**Research note:** `docs/research/2026-06-01-issue-230-peft-adaptation-surface-lit-review.md`

---

## Conventions every task obeys

Bake these into every implementer task — they are non-negotiable project gates.

- **Lint/type gate before each commit:** run all three and fix findings on touched files:
  - `uv run ruff check <touched files>`
  - `uv run ruff format --check <touched files>` (separate from `ruff check`; CI runs both)
  - `uv run mypy --strict <touched files>`
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
  symbol-add/rename, verify with `uv run python -c "import custom_sam_peft"` and
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
| 2 | `target_parameters` axis + scope + schema + tests | **Phase 1 go/no-go + mechanism** | — |
| 3 | calibrate VRAM-autosize alpha co-scale + WARNING | — (orthogonal) | 1 and 2 |

- **Phase 1 gates Phase 2.** Phase 2's resolver/scope wiring depends on the mechanism the
  spike chooses (peft `target_parameters` — the default written into this plan — vs peft's
  dedicated `lora.layer.MultiheadAttention` support path, vs the gated fallback §7.3(b)).
  Do **not** start Phase 2 until the spike's go/no-go and mechanism are recorded.
- **Phase 3 is orthogonal.** It is file-disjoint from Phases 1–2 (touches only
  `calibrate_cmd.py`, `_config_rewrite.py`, `presets.py::PresetDecision`, and
  `tests/unit/test_calibrate_cmd.py`) and does **not** depend on the spike. An orchestrator
  MAY run Phase 3 in parallel with Phase 1 (and Phase 2) on the same branch/worktree — no
  shared files, no shared symbols.

---

## Phase 1 — in_proj feasibility spike (GATING)

**Feature block:** Prove, on the **real** SAM 3.1 decoder, that LoRA can attach + forward +
merge on `ca_text` / `self_attn` `in_proj_weight` in **both** plain-LoRA and QLoRA modes,
and decide the mechanism. This phase ships only GPU-gated test code plus a recorded
go/no-go; it changes no production resolver/schema code.

**Why first:** §7 of the spec — the entire in_proj surface (Phase 2) is contingent on this
result. The spike chooses between (full) `target_parameters`, (fallback a) peft's
`lora.layer.MultiheadAttention` support path, or (fallback b) ship-infra-gate-surface.

**Interface contract this phase PRODUCES (consumed by Phase 2):**

- A recorded **go/no-go** on the in_proj surface (in the PR description and as a comment in
  the spike test file).
- The chosen **mechanism**, one of:
  - `target_parameters` — pass resolved `in_proj_weight` parameter names to
    `LoraConfig(target_parameters=...)` (the design's primary route; Phase 2 as written).
  - `lora.layer.MultiheadAttention` — name the MHA module itself as a `target_module` and
    let peft's MHA support adapt in_proj (Phase 2's resolver still lands, but the scope may
    express the surface via modules; §7.3(a)).
  - **gated** — peft cannot cleanly attach/forward/merge in this stack; Phase 2 lands the
    full infra but keeps `SCOPE_TARGET_PARAMETERS["vision_decoder_concept"]` **empty** and
    adjusts the §6.2 reproducibility note (§7.3(b)).
- The empirically-observed **trainable ratio** under the new scope on the real model (feeds
  Phase 2's §8.3 confirmation that the 10% guard / 5% test budget still holds).

This contract is a short written record (PR text + test-file comment). Phase 2 reads it
cold; it does not need to re-run the spike.

### Task 1.1: Write the gated in_proj spike for plain LoRA

**Files:**

- Modify: `tests/integration/test_peft_lora_real.py` (markers + helpers at top; existing
  tests at `:26-62`)

The spike productionizes §7.1 items 1–3 for plain LoRA. It uses the real model
(`load_sam31(ModelConfig())`) under the existing `requires_checkpoint` +
`requires_compatible_gpu` markers (the `pytestmark` list at `:19-23`). Because no production
scope literal exists yet (Phase 2 adds it), the spike drives resolution through an explicit
`PEFTConfig.target_modules` + `PEFTConfig.target_parameters` override that names the real
in_proj parameter paths directly — this both proves feasibility and exercises the
mechanism Phase 2 will wrap in a scope. (Phase 2's GPU tests, Task 2.10, replace these
overrides with the real `scope="vision_decoder_concept"` once it exists.)

- [ ] **Step 1: Write the spike test (attach + forward grad + merge, plain LoRA)**

Add to `tests/integration/test_peft_lora_real.py`:

```python
# --- #230 in_proj feasibility spike (§7). Drives target_parameters via an
# explicit override since the vision_decoder_concept scope is added in Phase 2.
_INPROJ_PARAM_PATTERNS = [
    r"transformer\.decoder\.layers\.\d+\.ca_text\.in_proj_weight$",
    r"transformer\.decoder\.layers\.\d+\.self_attn\.in_proj_weight$",
]
_VISION_DECODER_MODULE_PATTERNS = [
    r"backbone\.vision_backbone\.trunk\.blocks\.\d+\.attn\.(qkv|proj)$",
    r"transformer\.decoder\.layers\.\d+\.(self_attn|cross_attn|ca_text)\.out_proj$",
    r"transformer\.decoder\.layers\.\d+\.linear[12]$",
]


def test_spike_inproj_lora_attaches_forwards_merges() -> None:
    """#230 §7.1: target_parameters LoRA on ca_text/self_attn in_proj_weight
    attaches, receives gradients on forward, and merges on the real decoder."""
    w = load_sam31(ModelConfig())
    apply_lora(
        w,
        PEFTConfig(
            method="lora",
            target_modules=_VISION_DECODER_MODULE_PATTERNS,
            target_parameters=_INPROJ_PARAM_PATTERNS,
        ),
    )

    # 1) Attach: LoRA params exist for both in_proj parameters.
    lora_names = [n for n, _ in w.model.model.named_parameters() if "lora_" in n]
    assert any("ca_text" in n for n in lora_names), f"no ca_text in_proj LoRA: {lora_names[:8]}"
    assert any("self_attn" in n for n in lora_names), f"no self_attn in_proj LoRA: {lora_names[:8]}"

    # Record the observed trainable ratio for the Phase 2 §8.3 contract.
    trainable = sum(p.numel() for p in w.model.model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in w.model.model.parameters())
    ratio = trainable / total
    assert ratio < 0.05, f"trainable ratio {ratio:.2%} exceeds 5% budget"

    # 3) Merge: folds module + parameter adapters without raising.
    merge_lora(w)
    assert w.peft_model is None
    assert "Peft" not in type(w.model.model).__name__
```

Notes for the implementer:

- `apply_lora` today does **not** accept `target_parameters` on the `LoraConfig` it builds
  (`src/custom_sam_peft/peft_adapters/lora.py:107-114`). The spike **requires** a minimal
  wiring of `target_parameters` into that `LoraConfig` to run. If you prefer to keep Phase 1
  production-code-free, build the `PeftModel` inline in the test via
  `peft.get_peft_model(base, LoraConfig(..., target_parameters=...))` instead of calling
  `apply_lora`. Either is acceptable for the spike; the production wiring lands in Phase 2,
  Task 2.4. Record which route you took in the test docstring.
- The forward-grad check (§7.1 item 2) needs a real forward. If a full-model forward is
  awkward under the markers, assert the in_proj `lora_A` params are present and
  `requires_grad=True` (structural attach proof) and defer the grad-through-forward
  assertion to Phase 2's Task 2.10, noting this in the spike record. Do not silently drop
  item 2 from the recorded go/no-go.

- [ ] **Step 2: Verify the spike compiles and lints**

Run:

```bash
uv run python -m py_compile tests/integration/test_peft_lora_real.py
uv run ruff check tests/integration/test_peft_lora_real.py
uv run ruff format --check tests/integration/test_peft_lora_real.py
uv run mypy --strict tests/integration/test_peft_lora_real.py
```

Expected: all pass. (The test body is GPU-gated and will be **skipped** off-GPU.)

- [ ] **Step 3: Execute on the GPU runner**

Run (on a machine with the checkpoint + compatible GPU):

```bash
scripts/run_gpu_tests.sh local
```

Expected: `test_spike_inproj_lora_attaches_forwards_merges` PASSES, or FAILS with a
captured error. **Record the outcome** — this is the go/no-go for plain LoRA.

### Task 1.2: Write the gated in_proj spike for QLoRA coexistence

**Files:**

- Modify: `tests/integration/test_peft_qlora_real.py` (markers at `:36-37`; existing tests)

The hard requirement (§7.2): the bf16 in_proj parameter LoRA and the `Linear4bit` module
LoRA must attach + forward + **merge together in ONE `PeftModel`** under QLoRA. `apply_qlora`
keeps MHA children unquantized (`_mha_exclusion_types`, `qlora.py:59-96`), so in_proj stays a
bare bf16 `Parameter` and the in_proj LoRA is plain LoRA-on-bf16 even in QLoRA mode.

- [ ] **Step 1: Write the QLoRA coexistence spike**

Add to `tests/integration/test_peft_qlora_real.py` (mirror the override approach from
Task 1.1; reuse the same `_INPROJ_PARAM_PATTERNS`, redefining them locally in this file):

```python
def test_spike_inproj_qlora_coexists_attaches_merges() -> None:
    """#230 §7.2: under QLoRA the bf16 in_proj LoRA and the Linear4bit module
    LoRA attach and merge_and_unload together in one PeftModel."""
    w = load_sam31(ModelConfig())
    apply_qlora(
        w,
        PEFTConfig(
            method="qlora",
            target_parameters=[
                r"transformer\.decoder\.layers\.\d+\.ca_text\.in_proj_weight$",
                r"transformer\.decoder\.layers\.\d+\.self_attn\.in_proj_weight$",
            ],
        ),
    )
    lora_names = [n for n, _ in w.model.model.named_parameters() if "lora_" in n]
    assert any("ca_text" in n for n in lora_names), f"no ca_text in_proj LoRA (qlora): {lora_names[:8]}"
    assert any("self_attn" in n for n in lora_names), f"no self_attn in_proj LoRA (qlora): {lora_names[:8]}"

    merge_lora(w)  # dequantizes 4-bit base; must fold BOTH axes without dtype error
    assert w.peft_model is None
```

Notes:

- `apply_qlora` builds its `LoraConfig` in `_inject_lora_adapters` (`qlora.py:224-249`) and
  today passes only `target_modules`. As in Task 1.1, the spike needs `target_parameters`
  threaded in to run — either temporarily wire it (lands properly in Phase 2, Task 2.5) or
  build the `PeftModel` inline with the `Linear4bit`-targeting module set + the in_proj
  parameter set. Record the route.
- The merge path dequantizes the 4-bit base (`merge_lora` docstring, `lora.py:169-180`); the
  assertion is that no dtype / packed-weight error is raised while folding **both** adapters.

- [ ] **Step 2: Verify compile + lint + type**

Run:

```bash
uv run python -m py_compile tests/integration/test_peft_qlora_real.py
uv run ruff check tests/integration/test_peft_qlora_real.py
uv run ruff format --check tests/integration/test_peft_qlora_real.py
uv run mypy --strict tests/integration/test_peft_qlora_real.py
```

Expected: all pass.

- [ ] **Step 3: Execute on the GPU runner**

Run:

```bash
scripts/run_gpu_tests.sh local
```

Expected: `test_spike_inproj_qlora_coexists_attaches_merges` PASSES or FAILS with a captured
error. **Record the QLoRA go/no-go.**

### Task 1.3: Record the go/no-go and mechanism decision

**Files:**

- Modify: the PR description (the gating record) — and the docstrings of the two spike tests.

- [ ] **Step 1: Write the decision record**

Capture, in the PR body and as a comment block atop the two spike tests:

1. Plain-LoRA result (attach / forward-grad / merge) — pass or the captured failure.
2. QLoRA coexistence result (attach / merge-both) — pass or the captured failure.
3. The observed trainable ratio under the in_proj surface on the real model.
4. **Chosen mechanism** for Phase 2: `target_parameters` (default), `MultiheadAttention`
   support path (§7.3(a)), or `gated` (§7.3(b)).

If the decision is `gated`, also note that Phase 2 must keep
`SCOPE_TARGET_PARAMETERS["vision_decoder_concept"]` empty and adjust the §6.2
reproducibility note so the new default reads as currently equivalent to `vision_decoder`.

- [ ] **Step 2: Phase 1 verification + commit**

Run:

```bash
uv run ruff check tests/integration/test_peft_lora_real.py tests/integration/test_peft_qlora_real.py
uv run ruff format --check tests/integration/test_peft_lora_real.py tests/integration/test_peft_qlora_real.py
uv run mypy --strict tests/integration/test_peft_lora_real.py tests/integration/test_peft_qlora_real.py
uv run python -c "import custom_sam_peft"
```

Expected: all pass.

```bash
git add tests/integration/test_peft_lora_real.py tests/integration/test_peft_qlora_real.py
git commit -m "test(#230): gated in_proj feasibility spike (LoRA + QLoRA) [Phase 1]"
```

**Phase 1 handoff line (literal):**
`Resume phase. Next: 2. Plan: <this plan path>. Worktree: <full path>.`

---

## Phase 2 — `target_parameters` axis + new scope + schema + tests

**Feature block:** Land the production parameter-name resolution axis, the new
`vision_decoder_concept` scope + default flip, the `target_parameters` override field,
fixtures, and the full CPU + GPU test suite. This is the heart of the feature.

**Consumes from Phase 1:** the go/no-go and mechanism. This plan is written for the
`target_parameters` mechanism (the default). If Phase 1 chose:

- **`MultiheadAttention` support path (§7.3(a)):** still land Tasks 2.1–2.9 (the resolver +
  field + fixtures + CPU tests are the infrastructure), but in Task 2.2 the scope may also
  name the MHA modules in `SCOPE_TARGETS["vision_decoder_concept"]`, and Task 2.10's GPU
  assertions follow whichever attachment peft produces. Note the deviation in the PR.
- **`gated` (§7.3(b)):** land all tasks, but set
  `SCOPE_TARGET_PARAMETERS["vision_decoder_concept"] = []` in Task 2.2 (in_proj patterns
  commented out with a `# tbd:` pending a peft fix), adjust Task 2.6's reproducibility
  comment, and relax Task 2.10's in_proj GPU assertions to "no in_proj targets yet".

**Interface contract this phase PRODUCES (for downstream sessions / future tiers):**

- `SCOPE_TARGET_PARAMETERS: dict[str, list[str]]` in `lora.py` — scope → parameter-name
  regexes (sibling to `SCOPE_TARGETS`).
- `_resolve_target_parameters(base: nn.Module, cfg: PEFTConfig) -> list[str]` in `lora.py` —
  matches against `named_parameters()`; returns `[]` for empty pattern lists; raises
  `ValueError` only on non-empty-no-match. Imported by `qlora.py`.
- `LoraScope = Literal["vision", "vision_decoder", "vision_decoder_concept", "all"]` and
  `PEFTConfig.scope` default = `"vision_decoder_concept"`.
- `PEFTConfig.target_parameters: list[str] | None = None` — axis-independent override.
- `FIXTURE_SCOPE_TARGET_PARAMETERS` + `FIXTURE_SCOPE_PATTERNS["vision_decoder_concept"]` in
  the stub.

### Task 2.1: Add the `target_parameters` override field to `PEFTConfig`

**Files:**

- Modify: `src/custom_sam_peft/config/schema.py:498-504` (alongside `target_modules`)
- Test: `tests/unit/test_config_schema.py` (or the nearest existing schema test module —
  `grep -rl "PEFTConfig" tests/unit` to locate; create a focused test if none fits)

- [ ] **Step 1: Write the failing test**

```python
def test_peftconfig_target_parameters_defaults_none() -> None:
    from custom_sam_peft.config.schema import PEFTConfig

    cfg = PEFTConfig(method="lora")
    assert cfg.target_parameters is None


def test_peftconfig_target_parameters_accepts_list() -> None:
    from custom_sam_peft.config.schema import PEFTConfig

    cfg = PEFTConfig(method="lora", target_parameters=[r"x\.in_proj_weight$"])
    assert cfg.target_parameters == [r"x\.in_proj_weight$"]
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `uv run pytest -o "addopts=" tests/unit/test_config_schema.py -k target_parameters -v`
Expected: FAIL (`TypeError`/`ValidationError` — `target_parameters` is not a field yet, and
`PEFTConfig` is `_Strict` so an unknown kwarg is rejected).

- [ ] **Step 3: Add the field**

In `src/custom_sam_peft/config/schema.py`, immediately after the `target_modules` field
(ends at `:504`), add:

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

- [ ] **Step 4: Run the test to confirm it passes**

Run: `uv run pytest -o "addopts=" tests/unit/test_config_schema.py -k target_parameters -v`
Expected: PASS.

- [ ] **Step 5: Lint/type + import smoke**

Run:

```bash
uv run ruff check src/custom_sam_peft/config/schema.py
uv run ruff format --check src/custom_sam_peft/config/schema.py
uv run mypy --strict src/custom_sam_peft/config/schema.py
uv run python -c "import custom_sam_peft"
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/custom_sam_peft/config/schema.py tests/unit/test_config_schema.py
git commit -m "feat(#230): add PEFTConfig.target_parameters override field"
```

### Task 2.2: Add `SCOPE_TARGET_PARAMETERS` and the `vision_decoder_concept` module entry

**Files:**

- Modify: `src/custom_sam_peft/peft_adapters/lora.py:36-60` (after `SCOPE_TARGETS`)
- Test: `tests/unit/test_peft_target_parameters.py` (new)

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_peft_target_parameters.py`:

```python
"""Unit coverage for the #230 target_parameters resolution axis."""

from __future__ import annotations

from custom_sam_peft.peft_adapters.lora import SCOPE_TARGET_PARAMETERS, SCOPE_TARGETS


def test_scope_target_parameters_has_concept_inproj_patterns() -> None:
    pats = SCOPE_TARGET_PARAMETERS["vision_decoder_concept"]
    assert any("ca_text" in p and "in_proj_weight" in p for p in pats)
    assert any("self_attn" in p and "in_proj_weight" in p for p in pats)
    assert not any("cross_attn" in p for p in pats), "cross_attn is RoPEAttention, not MHA"


def test_concept_scope_modules_equal_vision_decoder() -> None:
    assert SCOPE_TARGETS["vision_decoder_concept"] == SCOPE_TARGETS["vision_decoder"]


def test_legacy_scopes_have_no_parameter_targets() -> None:
    for scope in ("vision", "vision_decoder", "all"):
        assert SCOPE_TARGET_PARAMETERS.get(scope, []) == []
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `uv run pytest -o "addopts=" tests/unit/test_peft_target_parameters.py -v`
Expected: FAIL (`ImportError`/`KeyError` — `SCOPE_TARGET_PARAMETERS` and the concept scope
do not exist).

- [ ] **Step 3: Add the dict + scope entry**

In `src/custom_sam_peft/peft_adapters/lora.py`, add a `"vision_decoder_concept"` entry to
`SCOPE_TARGETS` equal to the `"vision_decoder"` list (insert it directly after the
`vision_decoder` entry, before `"all"`):

```python
    # vision_decoder + the two in_proj parameter targets (see SCOPE_TARGET_PARAMETERS).
    # Module set is IDENTICAL to vision_decoder; the concept surface is the in_proj
    # parameter axis. New default scope (schema.py). See spec #230 §4.
    "vision_decoder_concept": [
        r"backbone\.vision_backbone\.trunk\.blocks\.\d+\.attn\.(qkv|proj)$",
        r"transformer\.decoder\.layers\.\d+\.(self_attn|cross_attn|ca_text)\.out_proj$",
        r"transformer\.decoder\.layers\.\d+\.linear[12]$",
    ],
```

Then, immediately after the `SCOPE_TARGETS` dict closes (`:60`), add:

```python
# Parallel to SCOPE_TARGETS: scope -> regexes matched against named_parameters().
# Reaches the bare nn.Parameter q/k/v packed in nn.MultiheadAttention.in_proj_weight,
# which target_modules cannot see. Only the concept scope populates it; absent scopes
# carry no parameter targets (reproducibility for vision/vision_decoder/all). This is
# the second single-point-of-contact for SAM 3.1 surface naming alongside SCOPE_TARGETS.
SCOPE_TARGET_PARAMETERS: dict[str, list[str]] = {
    "vision_decoder_concept": [
        r"transformer\.decoder\.layers\.\d+\.ca_text\.in_proj_weight$",
        r"transformer\.decoder\.layers\.\d+\.self_attn\.in_proj_weight$",
    ],
}
```

(If Phase 1 chose the **gated** fallback §7.3(b): set the list to `[]` with the two patterns
commented out and a `# tbd: #230 — in_proj surface gated pending peft MHA fix` tag.)

- [ ] **Step 4: Run the test to confirm it passes**

Run: `uv run pytest -o "addopts=" tests/unit/test_peft_target_parameters.py -v`
Expected: the three tests PASS.

- [ ] **Step 5: Lint/type + import smoke**

Run:

```bash
uv run ruff check src/custom_sam_peft/peft_adapters/lora.py tests/unit/test_peft_target_parameters.py
uv run ruff format --check src/custom_sam_peft/peft_adapters/lora.py tests/unit/test_peft_target_parameters.py
uv run mypy --strict src/custom_sam_peft/peft_adapters/lora.py
uv run python -c "import custom_sam_peft"
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/custom_sam_peft/peft_adapters/lora.py tests/unit/test_peft_target_parameters.py
git commit -m "feat(#230): add SCOPE_TARGET_PARAMETERS + vision_decoder_concept scope"
```

### Task 2.3: Add `_resolve_target_parameters`

**Files:**

- Modify: `src/custom_sam_peft/peft_adapters/lora.py` (sibling to `_resolve_targets` at
  `:63-85`)
- Test: `tests/unit/test_peft_target_parameters.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/test_peft_target_parameters.py`:

```python
import pytest
import torch
from torch import nn

from custom_sam_peft.config.schema import PEFTConfig
from custom_sam_peft.peft_adapters.lora import _resolve_target_parameters


class _MiniBase(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.transformer = nn.Module()
        self.transformer.decoder = nn.Module()
        layer = nn.Module()
        layer.ca_text = nn.MultiheadAttention(8, 2)
        layer.self_attn = nn.MultiheadAttention(8, 2)
        self.transformer.decoder.layers = nn.ModuleList([layer])


def _real_paths() -> list[str]:
    return [n for n, _ in _MiniBase().named_parameters()]


def test_resolve_empty_for_legacy_scope_returns_empty_no_error() -> None:
    base = _MiniBase()
    got = _resolve_target_parameters(base, PEFTConfig(method="lora", scope="vision_decoder"))
    assert got == []


def test_resolve_override_verbatim_precedence() -> None:
    base = _MiniBase()
    cfg = PEFTConfig(
        method="lora",
        scope="vision_decoder",  # legacy scope (no scope params) ...
        target_parameters=[r"\.ca_text\.in_proj_weight$"],  # ... but override set
    )
    got = _resolve_target_parameters(base, cfg)
    assert got == ["transformer.decoder.layers.0.ca_text.in_proj_weight"]


def test_resolve_empty_list_override_is_valid() -> None:
    base = _MiniBase()
    cfg = PEFTConfig(method="lora", scope="vision_decoder_concept", target_parameters=[])
    assert _resolve_target_parameters(base, cfg) == []


def test_resolve_non_empty_no_match_raises_valueerror() -> None:
    base = _MiniBase()
    cfg = PEFTConfig(method="lora", target_parameters=["nonexistent.param.path$"])
    with pytest.raises(ValueError) as exc:
        _resolve_target_parameters(base, cfg)
    msg = str(exc.value)
    assert "nonexistent.param.path$" in msg  # patterns tried listed
    assert "in_proj_weight" in msg  # a real parameter name sampled
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `uv run pytest -o "addopts=" tests/unit/test_peft_target_parameters.py -k resolve -v`
Expected: FAIL (`ImportError` — `_resolve_target_parameters` not defined).

- [ ] **Step 3: Implement the resolver**

In `src/custom_sam_peft/peft_adapters/lora.py`, add directly after `_resolve_targets`
(after `:85`):

```python
def _resolve_target_parameters(base: nn.Module, cfg: PEFTConfig) -> list[str]:
    """Resolve scope/override parameter-name patterns against named_parameters().

    Precedence mirrors _resolve_targets:
      * cfg.target_parameters is not None -> use it verbatim (overrides scope).
      * else -> SCOPE_TARGET_PARAMETERS.get(cfg.scope, []).

    Returns the full matched parameter names (e.g.
    'transformer.decoder.layers.0.ca_text.in_proj_weight') to pass to
    LoraConfig(target_parameters=...). Returns [] when the resolved pattern list is
    empty (legacy scopes) — NOT an error. Raises ValueError only when a NON-EMPTY
    pattern list matches zero parameters (a typo or SAM rename), mirroring
    _resolve_targets' no-match error so the in_proj surface never silently trains
    nothing.
    """
    patterns = (
        cfg.target_parameters
        if cfg.target_parameters is not None
        else SCOPE_TARGET_PARAMETERS.get(cfg.scope, [])
    )
    if not patterns:
        return []
    compiled = [re.compile(p) for p in patterns]
    param_names = [name for name, _ in base.named_parameters()]
    matched = [name for name in param_names if any(c.search(name) for c in compiled)]
    if not matched:
        sample = ", ".join(param_names[:50]) if param_names else "<no parameters found>"
        raise ValueError(
            f"apply_lora: no parameters matched target_parameters patterns {patterns}. "
            f"Parameters actually present (first 50): {sample}"
        )
    return matched
```

- [ ] **Step 4: Run the test to confirm it passes**

Run: `uv run pytest -o "addopts=" tests/unit/test_peft_target_parameters.py -k resolve -v`
Expected: all four PASS.

- [ ] **Step 5: Lint/type + import smoke**

Run:

```bash
uv run ruff check src/custom_sam_peft/peft_adapters/lora.py tests/unit/test_peft_target_parameters.py
uv run ruff format --check src/custom_sam_peft/peft_adapters/lora.py tests/unit/test_peft_target_parameters.py
uv run mypy --strict src/custom_sam_peft/peft_adapters/lora.py
uv run python -c "import custom_sam_peft"
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/custom_sam_peft/peft_adapters/lora.py tests/unit/test_peft_target_parameters.py
git commit -m "feat(#230): add _resolve_target_parameters (empty-ok, non-empty-no-match raises)"
```

### Task 2.4: Wire `target_parameters` into `apply_lora`

**Files:**

- Modify: `src/custom_sam_peft/peft_adapters/lora.py:88-137` (`apply_lora`)
- Test: `tests/unit/test_peft_target_parameters.py`

- [ ] **Step 1: Write the failing test (legacy scope must stay byte-identical)**

Add to `tests/unit/test_peft_target_parameters.py`:

```python
def test_apply_lora_legacy_scope_passes_target_parameters_none() -> None:
    """Reproducibility: legacy scopes must build LoraConfig with target_parameters=None."""
    import custom_sam_peft.peft_adapters.lora as lora_mod
    from tests.fixtures.tiny_sam3_lora_stub import FIXTURE_SCOPE_PATTERNS, make_stub_wrapper

    captured: dict[str, object] = {}
    real_cfg = lora_mod.LoraConfig

    def _spy(*args, **kwargs):
        captured.update(kwargs)
        return real_cfg(*args, **kwargs)

    w = make_stub_wrapper(dim=8, working=False)
    import pytest as _pytest

    with _pytest.MonkeyPatch.context() as mp:
        mp.setattr(lora_mod, "LoraConfig", _spy)
        lora_mod.apply_lora(
            w,
            PEFTConfig(
                method="lora",
                scope="vision_decoder",
                target_modules=FIXTURE_SCOPE_PATTERNS["vision_decoder"],
            ),
        )
    assert captured.get("target_parameters") is None
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `uv run pytest -o "addopts=" tests/unit/test_peft_target_parameters.py -k legacy_scope_passes -v`
Expected: FAIL (`KeyError`/`AssertionError` — `apply_lora` does not pass `target_parameters`
to `LoraConfig` yet, so the kwarg is absent from `captured`).

- [ ] **Step 3: Wire the axis into `apply_lora`**

In `src/custom_sam_peft/peft_adapters/lora.py`, in `apply_lora`, after
`matched_names = _resolve_targets(base, cfg)` (`:102`) add:

```python
    matched_params = _resolve_target_parameters(base, cfg)
```

Change the `LoraConfig(...)` construction (`:107-114`) to pass the new axis:

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

Extend the info log (`:123-130`) to surface the param-target count:

```python
    logger.info(
        "LoRA: trainable=%d (%.2f%%) of %d (scope=%s, n_targets=%d, n_param_targets=%d)",
        trainable,
        100 * ratio,
        total,
        cfg.scope if cfg.target_modules is None else "<override>",
        len(matched_names),
        len(matched_params),
    )
```

(The `> 0.10` warning at `:131-136` is unchanged.)

- [ ] **Step 4: Run the test + the full Phase-2 unit file**

Run:

```bash
uv run pytest -o "addopts=" tests/unit/test_peft_target_parameters.py tests/unit/test_peft_scope_coverage.py -v
```

Expected: PASS. (`test_peft_scope_coverage.py` confirms legacy scopes still attach exactly
as before — the reproducibility guard.)

- [ ] **Step 5: Lint/type + import smoke**

Run:

```bash
uv run ruff check src/custom_sam_peft/peft_adapters/lora.py
uv run ruff format --check src/custom_sam_peft/peft_adapters/lora.py
uv run mypy --strict src/custom_sam_peft/peft_adapters/lora.py
uv run python -c "import custom_sam_peft"
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/custom_sam_peft/peft_adapters/lora.py tests/unit/test_peft_target_parameters.py
git commit -m "feat(#230): pass target_parameters through apply_lora (None for legacy scopes)"
```

### Task 2.5: Wire `target_parameters` into the QLoRA apply path

**Files:**

- Modify: `src/custom_sam_peft/peft_adapters/qlora.py:33` (import) and `:224-249`
  (`_inject_lora_adapters`); log line at `:278-286`
- Test: `tests/unit/test_peft_target_parameters.py`

- [ ] **Step 1: Write the failing test (parameter axis is mode-independent)**

Add to `tests/unit/test_peft_target_parameters.py`:

```python
def test_qlora_and_lora_resolve_same_parameter_set() -> None:
    """§10.3: the parameter axis is mode-independent — same names for LoRA and QLoRA."""
    from custom_sam_peft.peft_adapters.lora import _resolve_target_parameters

    base = _MiniBase()
    lora_cfg = PEFTConfig(method="lora", scope="vision_decoder_concept")
    qlora_cfg = PEFTConfig(method="qlora", scope="vision_decoder_concept")
    assert _resolve_target_parameters(base, lora_cfg) == _resolve_target_parameters(base, qlora_cfg)
    # And both resolve the two in_proj params on the mini base.
    got = _resolve_target_parameters(base, lora_cfg)
    assert any("ca_text.in_proj_weight" in n for n in got)
    assert any("self_attn.in_proj_weight" in n for n in got)
```

- [ ] **Step 2: Run it to confirm it passes-or-fails appropriately**

Run: `uv run pytest -o "addopts=" tests/unit/test_peft_target_parameters.py -k same_parameter_set -v`
Expected: PASS already (this asserts a property of `_resolve_target_parameters`, which is
mode-agnostic). It is the regression guard for the wiring below; keep it. (If it fails, the
resolver is mode-coupled — fix the resolver, not the test.)

- [ ] **Step 3: Wire the axis into `_inject_lora_adapters`**

In `src/custom_sam_peft/peft_adapters/qlora.py`, extend the import at `:33`:

```python
from custom_sam_peft.peft_adapters.lora import _resolve_target_parameters, _resolve_targets
```

In `_inject_lora_adapters` (`:224-249`), after
`lora_target_names = _resolve_targets(model, cfg, linear_types=(bnb.nn.Linear4bit,))`
(`:239`) add:

```python
    lora_param_names = _resolve_target_parameters(model, cfg)
```

and add `target_parameters=(lora_param_names or None),` to the `LoraConfig(...)` (`:240-247`),
mirroring the LoRA wiring. The one-way `qlora.py -> lora.py` import contract is preserved
(`lora.py` still imports neither `qlora.py` nor `bitsandbytes`).

Extend the QLoRA log line (`:278-286`) with `n_param_targets`:

```python
    logger.info(
        "QLoRA: trainable=%d (%.2f%%) of %d "
        "(lora_scope=%s, quant_type=%s, compute_dtype=%s, n_param_targets=%d)",
        trainable,
        100 * ratio,
        total,
        cfg.scope if cfg.target_modules is None else "<override>",
        cfg.qlora.quant_type,
        cfg.qlora.compute_dtype,
        len(lora_param_names),
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

Expected: all pass. (`mypy --strict` on `qlora.py` exercises the new import + kwarg.)

- [ ] **Step 6: Commit**

```bash
git add src/custom_sam_peft/peft_adapters/qlora.py tests/unit/test_peft_target_parameters.py
git commit -m "feat(#230): wire target_parameters through QLoRA apply path"
```

### Task 2.6: Flip the default scope to `vision_decoder_concept`

**Files:**

- Modify: `src/custom_sam_peft/config/schema.py:99` (`LoraScope` literal) and `:496`
  (`scope` default)
- Test: `tests/unit/test_config_schema.py` (and `tests/unit/test_peft_target_parameters.py`)

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/test_peft_target_parameters.py`:

```python
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

- [ ] **Step 2: Run it to confirm it fails**

Run: `uv run pytest -o "addopts=" tests/unit/test_peft_target_parameters.py -k "default_scope_is_concept or lorascope_literal" -v`
Expected: FAIL (default is still `vision_decoder`; literal lacks the concept member).

- [ ] **Step 3: Update the literal + default**

In `src/custom_sam_peft/config/schema.py:99`:

```python
LoraScope = Literal["vision", "vision_decoder", "vision_decoder_concept", "all"]
```

In `src/custom_sam_peft/config/schema.py:496`, replace the `scope` line:

```python
    scope: LoraScope = "vision_decoder_concept"
    # tbd: #230 (project-chosen SAM 3.1 concept scope; default flipped from
    #      vision_decoder so the shipped default can learn niche TEXT concepts —
    #      vision_decoder freezes ca_text/self_attn in_proj. Reproducibility: a config
    #      without an explicit peft.scope now additionally adapts ca_text/self_attn
    #      in_proj; configs pinning vision/vision_decoder/all are unaffected. See
    #      research note §4, §7.)
```

(If Phase 1 chose **gated** §7.3(b): change the comment to note the new default is currently
behaviorally equivalent to `vision_decoder` until the in_proj surface is enabled.)

- [ ] **Step 4: Run the tests + full schema/config suites (blast radius)**

The default flip changes what every default-scope config adapts — grep and run broadly:

```bash
grep -rn 'scope.*vision_decoder"' src tests configs
uv run pytest -o "addopts=" tests/unit/test_config_schema.py tests/unit/test_peft_target_parameters.py -v
```

Expected: PASS. Inspect any test/config that pins or asserts the old default string and
update only those that asserted the **default** (not those that explicitly pin
`vision_decoder`).

- [ ] **Step 5: Lint/type + import smoke**

Run:

```bash
uv run ruff check src/custom_sam_peft/config/schema.py
uv run ruff format --check src/custom_sam_peft/config/schema.py
uv run mypy --strict src/custom_sam_peft/config/schema.py
uv run python -c "import custom_sam_peft"
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/custom_sam_peft/config/schema.py tests/unit/test_config_schema.py tests/unit/test_peft_target_parameters.py
git commit -m "feat(#230): flip default scope to vision_decoder_concept (#tbd #230)"
```

### Task 2.7: Expose MHA `in_proj_weight` in the LoRA stub fixture

**Files:**

- Modify: `tests/fixtures/tiny_sam3_lora_stub.py` (`_DecoderLayer:43-47`,
  `FIXTURE_SCOPE_PATTERNS:131-138`)
- Test: `tests/unit/test_peft_target_parameters.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/test_peft_target_parameters.py`:

```python
def test_fixture_exposes_mha_inproj_and_concept_patterns() -> None:
    from tests.fixtures.tiny_sam3_lora_stub import (
        FIXTURE_SCOPE_PATTERNS,
        FIXTURE_SCOPE_TARGET_PARAMETERS,
        make_stub_wrapper,
    )

    w = make_stub_wrapper(dim=8, working=False)
    base = w.model.model
    names = [n for n, _ in base.named_parameters()]
    assert any(n.endswith("ca_text.in_proj_weight") for n in names), names[:10]
    assert any(n.endswith("self_attn.in_proj_weight") for n in names), names[:10]
    # cross_attn must NOT be MHA (negative control for the parameter axis).
    assert not any("cross_attn.in_proj_weight" in n for n in names)
    # The concept fixture mappings exist.
    assert "vision_decoder_concept" in FIXTURE_SCOPE_PATTERNS
    assert "vision_decoder_concept" in FIXTURE_SCOPE_TARGET_PARAMETERS
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `uv run pytest -o "addopts=" tests/unit/test_peft_target_parameters.py -k fixture_exposes -v`
Expected: FAIL (`ImportError`/`AssertionError` — no MHA children, no concept mappings).

- [ ] **Step 3: Update the fixture**

In `tests/fixtures/tiny_sam3_lora_stub.py`, change `_DecoderLayer` (`:43-47`) so `ca_text`
and `self_attn` are real `nn.MultiheadAttention` (keep `cross_attn` non-MHA as the negative
control):

```python
class _DecoderLayer(nn.Module):
    def __init__(self, dim: int = 8, n_heads: int = 2) -> None:
        super().__init__()
        # Genuine torch MHA so in_proj_weight (a bare nn.Parameter) exists, mirroring
        # SAM 3.1's decoder ca_text/self_attn. cross_attn stays a non-MHA attention so
        # it is the negative control for the in_proj parameter axis.
        self.ca_text = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.self_attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.cross_attn = _DecoderAttn(dim)
```

Add a `vision_decoder_concept` entry to `FIXTURE_SCOPE_PATTERNS` (equal to the
`vision_decoder` fixture module patterns; note the truncated `transformer_decoder` prefix)
and add the new parallel mapping after it (`:131-138`):

```python
FIXTURE_SCOPE_PATTERNS: dict[str, list[str]] = {
    "vision": [r"vision_trunk\.blocks\.\d+\.attn\.(qkv|proj)$"],
    "vision_decoder": [
        r"vision_trunk\.blocks\.\d+\.attn\.(qkv|proj)$",
        r"transformer_decoder\.layers\.\d+\.(self_attn|cross_attn)\.out_proj$",
    ],
    "vision_decoder_concept": [
        r"vision_trunk\.blocks\.\d+\.attn\.(qkv|proj)$",
        r"transformer_decoder\.layers\.\d+\.(self_attn|cross_attn)\.out_proj$",
    ],
    "all": [r".*"],
}

# Parallel to the production SCOPE_TARGET_PARAMETERS, but with the fixture's truncated
# `transformer_decoder` prefix. Drives the in_proj parameter axis on the stub.
FIXTURE_SCOPE_TARGET_PARAMETERS: dict[str, list[str]] = {
    "vision_decoder_concept": [
        r"transformer_decoder\.layers\.\d+\.ca_text\.in_proj_weight$",
        r"transformer_decoder\.layers\.\d+\.self_attn\.in_proj_weight$",
    ],
}
```

Note: `nn.MultiheadAttention`'s `out_proj` child has name `out_proj` (matches the existing
`(self_attn|cross_attn)` out_proj pattern only for `self_attn` — `ca_text` out_proj is not
in the fixture's `vision_decoder` pattern, matching production where the concept scope's
module list does include `ca_text.out_proj`; the in_proj coverage comes from the parameter
axis). The fixture module patterns intentionally mirror the **existing** `vision_decoder`
fixture shape; the new coverage is the parameter axis. Do not expand the module patterns
beyond the existing two-line `vision_decoder` shape.

- [ ] **Step 4: Run the test + the existing stub-driven tests**

Run:

```bash
uv run pytest -o "addopts=" tests/unit/test_peft_target_parameters.py tests/unit/test_peft_scope_coverage.py -v
```

Expected: PASS. `test_peft_scope_coverage.py`'s `working=True` forward/backward test must
still pass — the stub's forward routes through `vision_trunk.blocks[0].attn.qkv`
(`tiny_sam3_lora_stub.py:92`), unaffected by the decoder-layer MHA swap.

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
git commit -m "test(#230): expose ca_text/self_attn MHA in_proj in LoRA stub fixture"
```

### Task 2.8: CPU coverage — concept scope resolves both axes via `apply_lora` on the stub

**Files:**

- Modify: `tests/unit/test_peft_target_parameters.py`

The production `SCOPE_TARGET_PARAMETERS` patterns use the real `transformer.decoder` prefix
and will not match the stub's truncated `transformer_decoder` prefix; drive the stub via the
`FIXTURE_*` overrides (the same approach `test_peft_scope_coverage.py` uses for modules).

- [ ] **Step 1: Write the failing tests**

Add to `tests/unit/test_peft_target_parameters.py`:

```python
def _lora_param_names(wrapper: object) -> list[str]:
    return [n for n, _ in wrapper.model.model.named_parameters() if "lora_" in n]


def test_concept_scope_attaches_modules_and_inproj_on_stub() -> None:
    from tests.fixtures.tiny_sam3_lora_stub import (
        FIXTURE_SCOPE_PATTERNS,
        FIXTURE_SCOPE_TARGET_PARAMETERS,
        make_stub_wrapper,
    )
    from custom_sam_peft.peft_adapters.lora import apply_lora

    w = make_stub_wrapper(dim=8, working=False)
    apply_lora(
        w,
        PEFTConfig(
            method="lora",
            scope="vision_decoder_concept",
            target_modules=FIXTURE_SCOPE_PATTERNS["vision_decoder_concept"],
            target_parameters=FIXTURE_SCOPE_TARGET_PARAMETERS["vision_decoder_concept"],
        ),
    )
    names = _lora_param_names(w)
    assert any("vision_trunk.blocks" in n for n in names)
    assert any("ca_text" in n and "lora" in n for n in names), names[:10]
    assert any("self_attn" in n and "lora" in n for n in names), names[:10]
    assert not any("cross_attn" in n and "in_proj" in n for n in names)


def test_concept_scope_trainable_ratio_small_on_stub() -> None:
    from tests.fixtures.tiny_sam3_lora_stub import (
        FIXTURE_SCOPE_PATTERNS,
        FIXTURE_SCOPE_TARGET_PARAMETERS,
        make_stub_wrapper,
    )
    from custom_sam_peft.peft_adapters.lora import apply_lora

    w = make_stub_wrapper(dim=8, working=False)
    apply_lora(
        w,
        PEFTConfig(
            method="lora",
            scope="vision_decoder_concept",
            target_modules=FIXTURE_SCOPE_PATTERNS["vision_decoder_concept"],
            target_parameters=FIXTURE_SCOPE_TARGET_PARAMETERS["vision_decoder_concept"],
        ),
    )
    base = w.model.model
    trainable = sum(p.numel() for p in base.parameters() if p.requires_grad)
    total = sum(p.numel() for p in base.parameters())
    assert trainable / total < 0.5  # stub is tiny; loose bound mirrors existing style
```

- [ ] **Step 2: Run them to confirm they pass**

Run: `uv run pytest -o "addopts=" tests/unit/test_peft_target_parameters.py -k "concept_scope_attaches or trainable_ratio_small" -v`
Expected: PASS (all wiring from Tasks 2.2–2.7 is now in place). If `apply_lora` raises a peft
error attaching `target_parameters` to the stub's MHA on CPU, that surfaces a peft/stack
incompatibility the Phase 1 spike should have caught — escalate per the gated fallback,
do not weaken the test.

- [ ] **Step 3: Lint/type**

Run:

```bash
uv run ruff check tests/unit/test_peft_target_parameters.py
uv run ruff format --check tests/unit/test_peft_target_parameters.py
uv run mypy --strict tests/unit/test_peft_target_parameters.py
```

Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add tests/unit/test_peft_target_parameters.py
git commit -m "test(#230): CPU coverage for concept scope two-axis resolution on stub"
```

### Task 2.9: Update example configs to document the new scope + override knob

**Files:**

- Modify: `configs/examples/coco_text_lora.yaml:45-48` (the commented PEFT knob block) and
  the analogous block in the other example configs that carry it:
  `configs/examples/coco_text_qlora.yaml`, `coco_text_auto_split.yaml`,
  `coco_text_no_val.yaml`, `coco_text_lora_subset.yaml`, `min_gpu_qlora.yaml`,
  `gpu_smoke_lora.yaml`, `gpu_smoke_qlora.yaml` (grep each; only those with a `# scope:`
  comment line need editing)

- [ ] **Step 1: Find every commented PEFT knob block**

Run:

```bash
grep -rln "# scope:" configs/examples
```

Expected: the list of example configs carrying the commented `# scope:` knob.

- [ ] **Step 2: Update the comment block in each**

In each matched file, replace the commented knob block (in `coco_text_lora.yaml` it is
`:45-48`) with:

```yaml
  # Knobs (defaults shown — uncomment to override):
  # scope: vision_decoder_concept  # vision | vision_decoder | vision_decoder_concept | all
  #                                # (default; adapts ca_text/self_attn in_proj for text concepts)
  # bias: none                     # none | all | lora_only
  # target_modules: [...]          # overrides scope's module patterns when set
  # target_parameters: [...]       # overrides scope's in_proj patterns when set
```

Only comment lines change; **no uncommented value changes** (defaults already apply — this
documents the new lever only). Preserve each file's existing indentation exactly.

- [ ] **Step 3: Validate the configs still load**

Run (substitute each edited path):

```bash
uv run python -c "from custom_sam_peft.config.loader import load_config; load_config('configs/examples/coco_text_lora.yaml')"
```

Expected: loads without error. Repeat for each edited config (or loop over the grep list).

- [ ] **Step 4: Commit**

```bash
git add configs/examples
git commit -m "docs(#230): document vision_decoder_concept + target_parameters in example configs"
```

### Task 2.10: Productionize the GPU integration tests with the real scope

**Files:**

- Modify: `tests/integration/test_peft_lora_real.py` and `tests/integration/test_peft_qlora_real.py`
  (replace the Phase-1 spike overrides with the real `scope="vision_decoder_concept"`)

- [ ] **Step 1: Replace the spike overrides with the production scope (LoRA)**

In `tests/integration/test_peft_lora_real.py`, update the Phase-1 spike test (or add a
production test alongside it) to drive the real scope and assert §10.4:

```python
def test_concept_scope_inproj_on_real_sam31() -> None:
    """§10.4: scope='vision_decoder_concept' attaches in_proj LoRA, merges, ratio<5%."""
    w = load_sam31(ModelConfig())
    apply_lora(w, PEFTConfig(method="lora", scope="vision_decoder_concept"))

    lora_names = [n for n, _ in w.model.model.named_parameters() if "lora_" in n]
    assert any("ca_text" in n for n in lora_names), f"no ca_text in_proj LoRA: {lora_names[:8]}"
    assert any("self_attn" in n for n in lora_names), f"no self_attn in_proj LoRA: {lora_names[:8]}"

    trainable = sum(p.numel() for p in w.model.model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in w.model.model.parameters())
    assert trainable / total < 0.05, "concept scope exceeds 5% trainable budget"

    merge_lora(w)
    assert w.peft_model is None
```

Remove or fold the Phase-1 `test_spike_inproj_lora_*` test now that the scope exists (keep
its recorded outcome in the PR / spike comment). (If Phase 1 chose **gated** §7.3(b): assert
the concept scope produces **no** in_proj LoRA yet — `not any("ca_text" in n ...)` — and that
it is behaviorally equal to `vision_decoder`.)

- [ ] **Step 2: Replace the spike overrides with the production scope (QLoRA)**

In `tests/integration/test_peft_qlora_real.py`, mirror Step 1 for QLoRA with
`scope="vision_decoder_concept"`, asserting the in_proj LoRA coexists with the `Linear4bit`
module LoRA and `merge_lora` folds both (§7.2). Remove/fold the Phase-1 spike test.

- [ ] **Step 3: Verify compile + lint + type (off-GPU)**

Run:

```bash
uv run python -m py_compile tests/integration/test_peft_lora_real.py tests/integration/test_peft_qlora_real.py
uv run ruff check tests/integration/test_peft_lora_real.py tests/integration/test_peft_qlora_real.py
uv run ruff format --check tests/integration/test_peft_lora_real.py tests/integration/test_peft_qlora_real.py
uv run mypy --strict tests/integration/test_peft_lora_real.py tests/integration/test_peft_qlora_real.py
```

Expected: all pass (the test bodies remain GPU-gated/skipped off-GPU).

- [ ] **Step 4: Execute on the GPU runner**

Run:

```bash
scripts/run_gpu_tests.sh local
```

Expected: the new concept-scope tests PASS on a machine with the checkpoint + compatible
GPU. Record the empirical trainable ratio (confirms §8.3 — leave the 10% guard /
`< 0.05` budget unchanged unless reality demands; a threshold change would need a `# tbd:`).

- [ ] **Step 5: Commit**

```bash
git add tests/integration/test_peft_lora_real.py tests/integration/test_peft_qlora_real.py
git commit -m "test(#230): GPU concept-scope in_proj tests (LoRA + QLoRA coexistence)"
```

### Task 2.11: Phase 2 verification before completion

**Files:** none (verification only)

- [ ] **Step 1: Full CPU suite + blast-radius grep**

Run:

```bash
grep -rn "SCOPE_TARGETS\|SCOPE_TARGET_PARAMETERS\|target_parameters\|vision_decoder_concept" src tests
uv run pytest -o "addopts=" tests/unit tests/integration -q
uv run python -c "import custom_sam_peft"
```

Expected: full CPU suite PASSES (GPU tests skipped); the package imports clean; no stray
reference to the old default that asserted `vision_decoder` as the default.

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
Phases 1–2 and does **not** depend on the spike. It touches only:
`src/custom_sam_peft/cli/calibrate_cmd.py`, `src/custom_sam_peft/cli/_config_rewrite.py`,
`src/custom_sam_peft/presets.py` (`PresetDecision` only), and
`tests/unit/test_calibrate_cmd.py`. It may run **in parallel** with Phases 1 and 2 on the
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

1. **Spike resolved first** — Phase 1 (Tasks 1.1–1.3); go/no-go + mechanism recorded.
2. **New scope** — Task 2.2 (`vision_decoder_concept`; modules == `vision_decoder`; legacy
   scopes byte-identical, asserted in Task 2.2 / `test_peft_scope_coverage.py`).
3. **New default** — Task 2.6 (`# tbd: #230`; reproducibility note in code + spec).
4. **Resolution axis** — Tasks 2.2–2.5 (`SCOPE_TARGET_PARAMETERS`,
   `_resolve_target_parameters`, both apply paths pass the axis; legacy = `None`).
5. **Override field** — Task 2.1 (`target_parameters: list[str] | None = None`; precedence +
   empty-vs-no-match covered in Task 2.3).
6. **QLoRA coexistence** — Tasks 1.2, 2.5, 2.10 (one `PeftModel`, attach+forward+merge; or
   gated per §7.3(b)).
7. **Error parity** — Task 2.3 (non-empty-no-match `ValueError`; empty resolution is fine).
8. **Trainable-ratio guard** — Tasks 1.x / 2.8 / 2.10 (empirically < 5% budget; 10% guard
   unchanged).
9. **Calibrate alpha co-scale** — Phase 3 (Tasks 3.1–3.5); `oom.py` untouched (Task 3.5);
   no new `# tbd:`.
10. **Tests/fixtures** — Task 2.7 (stub MHA), Tasks 2.8 / 2.10 / 3.x (CPU + GPU); coverage
    >= 80% (trust CI).
11. **Lint/type** — every task's lint/type step; the markdown gate for this plan + the spec.

## Self-review notes (placeholders/types checked)

- Every code step shows complete code (no TBD / "add error handling" placeholders).
- Type/name consistency: `_resolve_target_parameters(base, cfg) -> list[str]`,
  `SCOPE_TARGET_PARAMETERS`, `PEFTConfig.target_parameters`,
  `PresetDecision.alpha`, `_rewrite_sizing_block(..., alpha, ...)`,
  `chosen_alpha`, and `alpha_final = round(cfg.peft.alpha * r / cfg.peft.r)` are used
  identically everywhere they appear.
- Phase boundaries each publish an explicit interface contract; Phase 1 gates Phase 2;
  Phase 3 is orthogonal and parallelizable (commits serialized).
