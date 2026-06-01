<!-- markdownlint-disable MD013 -->

# GPU test migration: re-architect testing around the RTX 5070 Ti — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Re-architect the tiered GPU-test policy around the RTX 5070 Ti — drop Pascal, set the Tesla T4 (CC 7.5) as the floor, name tiers by capability with live auto-detection, add an empirical "won't fit a small card" predict warning, harden CPU/stub coverage for escaped GPU-bug classes, add a non-blocking GPU-evidence gate, and close #142/#139/#193/#195/#83.

**Architecture:** Five sequential, independently-reviewable phases. Phase A rewrites the tier taxonomy (markers, set-returning capability probe, capability-subset skip predicate, CC-gate move, Pascal removal). Phase B adds the empirical predict-budget warning (pure decision function + train-entrypoint hook). Phase C adds #142's 8 GB-ceiling train + predict tests and the #83 probe. Phase D adds three bounded CPU/stub regression tests. Phase E rewrites the runner/notebook/policy doc, adds the provable-non-blocking evidence gate, files the gpu_xl issue, and wires the per-issue closures (Colab-dependent closures gated LAST on the user's confirmation).

**Tech Stack:** Python 3.12, pytest (`--strict-markers`), `uv`, ruff, mypy, PyTorch CUDA, bitsandbytes, PEFT, bash + shellcheck, GitHub Actions, markdownlint-cli2.

---

## How to read this plan

- **One phase per orchestrator session.** Each phase ends with an explicit **Interface Contract** (what it EXPOSES / what later phases CONSUME) so a fresh session can build on it by reading only the contract, not the prior phase's code.
- **CPU vs GPU verification.** Every test task is tagged **[verify on CPU/CI]** or **[verify on 5070 Ti]**. Real-GPU assertions can only be confirmed on the 5070 Ti; author them so they auto-skip green on CPU.
- **Spec is source of truth.** Requirement IDs (R1–R33, X1–X3) reference `docs/superpowers/specs/2026-05-31-gpu-test-migration-5070ti-design.md`. A traceability table at the end maps every requirement to a task.
- **`src/` layout.** The package lives at `src/custom_sam_peft/`. There is no top-level `custom_sam_peft/`.
- **Commit discipline.** Implementer commits during implementation are exempt from the lint gate; the FINAL ready-PR state must pass the §X3 gate (see Definition of Done). If a phase runs file-disjoint tasks in parallel, **the orchestrator serializes the git commits** (parallel agents committing on the same branch can orphan a commit).
- **Eager-import caution.** `src/custom_sam_peft/__init__.py` eagerly imports the train chain, so removing/renaming a symbol (`_TIER_ORDER`, `_current_tier`) can un-import the package mid-phase. After any symbol removal, verify with `uv run ruff check` / `python -c "import custom_sam_peft"` (or `py_compile`) and gate behavior at phase end.

---

## ⚠️ Anchor-grounding pre-step (run FIRST in every phase that edits a named file)

The spec cites many `file:line` anchors. Line numbers drift across merges. **Before editing any file named with a line anchor, re-locate the symbol with grep and use the live line.** This plan deliberately references symbols by NAME, not by raw line number, for edit targets. The known anchors to re-verify at execution time:

```bash
# Run from the worktree root. Capture the CURRENT line numbers before editing.
ROOT=/home/justin/projects/custom-sam-peft/.claude/worktrees/gpu-test-migration-5070ti
grep -n "_current_tier\|_satisfied_tiers\|_has_compatible_gpu\|_TIER_ORDER\|_torch_can_launch_kernel\|_free_cuda_after_gpu_test\|requires_compatible_gpu\|pytest_collection_modifyitems" "$ROOT/tests/conftest.py"
grep -n "coerce_dtype_for_capability\|capability >= (8, 0)\|capability >= (6, 0)" "$ROOT/src/custom_sam_peft/runtime/_runtime.py"
grep -n "cc >= (8, 0)\|_headroom_bytes\|flash" "$ROOT/src/custom_sam_peft/presets.py"
grep -n "free VRAM\|batch-size 4\|batch_size == 1\|vram" "$ROOT/src/custom_sam_peft/predict/runner.py"
grep -n "gpu_local\|gpu_t4\|gpu_xl\|markers" "$ROOT/pyproject.toml"
grep -rn "gpu_local\|gpu_t4\|gpu_xl\|gpu-pascal\|cu118\|run_gpu_tests.sh" "$ROOT/tests" "$ROOT/scripts" "$ROOT/docs" "$ROOT/notebooks" "$ROOT/.github" "$ROOT/pyproject.toml"
```

> **Stale-anchor handling:** if any spec anchor no longer matches (e.g. `_current_tier` is on a different line, or `predict/runner.py` upward-hint moved), use the live location and **note the correction in the policy doc's "implementation notes"** — do NOT edit the spec.
>
> **Verified live anchors (grounded 2026-05-31 against this worktree — spec corrections noted, spec NOT edited):**
>
> - `coerce_dtype_for_capability` — `src/custom_sam_peft/runtime/_runtime.py:61` (spec correct); `if capability >= (8, 0): return dtype` at line 83, bf16→fp16 coercion at line 93. R9: leave untouched.
> - `presets.py`: `_headroom_bytes` def at **line 340** (spec `:340` — correct); Flash gate `_flash_attention_available` returns `cc >= (8, 0)` at **line 225** (spec `:225` — correct). Reference by name.
> - `predict/runner.py` upward hint: `free_bytes, _ = torch.cuda.mem_get_info()` (line 356) then `if free_bytes > 12 * 1024**3: logger.info("free VRAM is >12 GB; consider --batch-size 4 or 8.")` (lines 357–358). Spec said `352-358` — the comment "# Step 7: VRAM hint" is at line 352; the active code is 356–358. R18 guards this region. (NOTE: the spec also references a test at `tests/predict/test_gpu_predict.py:226+` — the live upward-hint test is the `gpu_t4`-marked one near line 200; confirm by name during R18 guard.)
> - `tests/conftest.py`: `pytest_configure` registers markers at lines 20–47 (incl. `gpu_local` at 35–39, `requires_compatible_gpu` at 25–30); `_torch_can_launch_kernel` at line 50; `_TIER_ORDER` at line 66; `_has_compatible_gpu` (CC gate `< (6, 0)` at line 76) at 69–78; `_current_tier` at 81–91; `pytest_collection_modifyitems` at 94–124 (the `_TIER_ORDER[item_tier] > _TIER_ORDER[runner_tier]` comparison is at line 116); autouse `_free_cuda_after_gpu_test` at 127–143. **Markers are registered in BOTH `pyproject.toml` (lines 128–135) AND `conftest.py::pytest_configure` (dual-registration — update BOTH sites).**
> - `#193` tbd lives at `docs/defaults-provenance.md:277` (prose in "Reference Training Profile", §ends line 251+) and is cross-linked from the `TrainHyperparams.epochs` row at line 84. There is NO standalone "Step time" table row — the `# tbd: #193` is the per-step wall-clock prose at line 277. Resolve THAT (add the 5070 Ti per-step figure; leave the T4 sample slot for Task E8).
> - **markdownlint CI invocation (CORRECTED — from `.github/workflows/ci.yml:98-99`, there is NO `docs-lint.yml`):** `npx --yes markdownlint-cli2 --config .config/markdownlint-cli2.jsonc "**/*.md" "#node_modules"`. Config file: `.config/markdownlint-cli2.jsonc` (no leading dot on the filename). `gpu-deselect-check` is a separate job at `.github/workflows/ci.yml:104`; CI lint steps: `ruff check` (line 38), `ruff format --check` (41), `mypy src/custom_sam_peft` (44), `uv run pytest` (47), `shellcheck scripts/*.sh` (101–102). NOTE: `shellcheck scripts/*.sh` globs ALL scripts — the new `scripts/check_gpu_evidence.sh` (Phase E) is auto-covered.
> - `test_gpu_predict.py` carries 4 GPU marker decorations: line 153 `@pytest.mark.gpu_local` (base model), 170 `@pytest.mark.gpu_t4`, 200 `@pytest.mark.gpu_t4` (the upward-hint test — R18 leaves it alone), 230 `@pytest.mark.gpu_local`. Module `pytestmark` (line 31) is `requires_compatible_gpu`/`requires_checkpoint`, NOT a tier. Apply the default rule per decoration.
> - **EXISTING tests that WILL BREAK on the taxonomy change (blast-radius consumers — see Task A2b):** `tests/unit/test_marker_autoskip.py` (asserts `gpu_local` runs / `gpu_t4` skipped on a "local" runner via the OLD `_TIER_ORDER` + `_current_tier` model; references `gpu_xl` skip-reason naming #124) and `tests/unit/test_run_gpu_tests_script.py` (`test_local_maps_to_gpu_local_marker` asserts the removed `local` selector). Both must be rewritten in this PR.

---

## File-structure map (what this plan touches)

| File | Responsibility | Phase(s) |
|------|----------------|----------|
| `pyproject.toml` | pytest marker registration; remove `gpu-pascal` extra | A |
| `tests/conftest.py` | `_satisfied_tiers()` probe, capability-subset skip predicate, CC-gate move, marker docstrings | A |
| `tests/integration/test_load_sam31_real.py` | per-test marker reclass (gpu_local→gpu_t4) | A |
| `tests/integration/test_peft_lora_real.py` | module `pytestmark` reclass | A |
| `tests/integration/test_peft_qlora_real.py` | module/per-test marker reclass | A |
| `tests/predict/test_gpu_predict.py` | per-test marker reclass | A |
| `tests/gpu/*.py` (8 files) | module `pytestmark` reclass | A |
| `tests/unit/test_conftest_tiers.py` (new) | CPU unit tests for the probe + skip predicate | A |
| `tests/gpu/test_bf16_faithful.py` (new) | the gpu_bf16 non-coerced bf16 test | A |
| `docs/testing/local-pascal-gpu-testing.md` | DELETE | A |
| `docs/testing/manual-gpu-pass-2026-05-24-gtx1080.md` | add superseded banner | A |
| `src/custom_sam_peft/predict/budget.py` (new) OR a budget module | `PREDICT_8GB_BUDGET_GB`, pure decision function | B |
| `src/custom_sam_peft/train/loop.py` (or `trainer.py`) | model-ready probe hook | B |
| `tests/unit/test_predict_budget.py` (new) | CPU unit test for the pure decision fn | B |
| `tests/gpu/test_predict_budget_warning.py` (new) | GPU warn / no-warn test | B |
| `tests/gpu/test_real_train_qlora.py` or new file | #142 8 GB-ceiling smoke + `QLORA_8GB_CEIL_GB` | C |
| `tests/predict/test_predict_fits_8gb.py` (new) | predict-fits-8GB probe | C |
| `configs/examples/min_gpu_qlora.yaml` | de-Pascal rationale comments | C |
| `tests/gpu/test_peft_scope_coverage_gpu.py` (new, conditional) | #83 all-scope LoRA smoke (branch a) | C |
| `tests/unit/test_channel_adapter_dtype.py` (new) | CPU dtype-consistency contract | D |
| `tests/unit/test_row_outputs_nontensor.py` (new) | CPU non-tensor forward-output contract | D |
| `tests/unit/test_predict_image_size_contract.py` (new) | CPU 1008-RoPE image-size contract | D |
| `scripts/run_gpu_tests.sh` | rewrite selectors to new taxonomy | E |
| `scripts/check_gpu_evidence.sh` (new) | standalone non-blocking evidence check | E |
| `tests/unit/test_gpu_evidence_check.py` (new) | provable-non-blocking test (R33) | E |
| `.github/workflows/*.yml` | non-blocking evidence job; keep `gpu-deselect-check` | E |
| `notebooks/colab_gpu_tests.ipynb` | colab-min subset + #139/#193 capture | E |
| `docs/testing/gpu-test-policy.md` | full rewrite to new taxonomy | E |
| `docs/defaults-provenance.md` | rows for `QLORA_8GB_CEIL_GB`, `PREDICT_8GB_BUDGET_GB`; resolve `# tbd: #193` | C, B, E |

---

## Phase A — Taxonomy, conftest, pyproject, Pascal removal

**Requirements:** R1, R2, R3 (registration only), R4, R5, R6, R7, R8, R9, R-counts, X2 (marker blast radius).

**Goal of phase:** Replace `gpu_local`/`gpu_t4`/`gpu_xl` dev-card tiers with capability-named, auto-detected tiers; move the CC gate 6.0→7.5; delete Pascal artifacts; add the gpu_bf16 faithful-bf16 test. End with the full CPU suite green.

> **TDD note:** the probe and skip predicate are CPU-unit-testable by monkeypatching `torch.cuda` capability + `total_memory` (mirror `tests/unit/test_presets.py`'s `_stub_gpu` style). Write those unit tests FIRST. The marker reclass and Pascal deletions are mechanical; verify via `pytest --collect-only` counts.

---

### Task A1: Register the three capability-named markers; remove `gpu_local`

**Files:**
- Modify: `pyproject.toml` (the `[tool.pytest.ini_options] markers` list, ~lines 128–134 — re-locate with the grep pre-step)
- Modify: `tests/conftest.py` (any `config.addinivalue_line("markers", ...)` registrations, if the dual-registration convention is present)

- [ ] **Step 1: Re-locate the marker block.**

```bash
grep -n "gpu_local\|gpu_t4\|gpu_xl\|markers = \[" pyproject.toml
grep -n "addinivalue_line\|markers" tests/conftest.py
```

- [ ] **Step 2: Replace the marker definitions in `pyproject.toml`.** Remove the `gpu_local` line. Define exactly three GPU tiers, each docstring stating its gate and linking the policy doc:

```toml
    "gpu_t4: requires a CUDA GPU with CC >= 7.5 AND total VRAM <= 16 GB (Tesla T4 floor and RTX 5070 Ti). fp16 band (bf16 is coerced below CC 8.0). See docs/testing/gpu-test-policy.md.",
    "gpu_bf16: requires a CUDA GPU with CC >= 8.0 AND total VRAM <= 16 GB (RTX 5070 Ti). Native, non-coerced bf16 numerics. See docs/testing/gpu-test-policy.md.",
    "gpu_xl: requires a CUDA GPU with total VRAM > 16 GB. Empty in this PR; populated only via the gpu_xl follow-up issue. See docs/testing/gpu-test-policy.md.",
```

- [ ] **Step 3: Mirror in `tests/conftest.py` if the dual-registration convention exists.** If `conftest.py` also registers markers via `addinivalue_line`, update the same three and remove `gpu_local`. If only `pyproject.toml` registers them, skip.

- [ ] **Step 4: Verify strict-markers collection does not error yet (it will error until tests are reclassed — expected).**

Run: `uv run pytest --collect-only -q 2>&1 | grep -i "gpu_local" | head`
Expected: any remaining `gpu_local` usages are the test files (reclassed in A2). The marker registration itself raises no `'gpu_local' not found` until A2 lands. **[verify on CPU/CI]**

- [ ] **Step 5: Commit.**

```bash
git add pyproject.toml tests/conftest.py
git commit -m "feat(tests): register capability-named gpu_t4/gpu_bf16/gpu_xl markers; drop gpu_local"
```

---

### Task A2: Reclassify all 27 GPU tests to the new tiers (R2)

**Files (re-verify each marker site with grep first):**
- Modify: `tests/integration/test_load_sam31_real.py` (per-test: 2 `gpu_local`→`gpu_t4`; K8 multiplex `gpu_t4`→`gpu_t4` unchanged)
- Modify: `tests/integration/test_peft_lora_real.py` (module `pytestmark`: `gpu_local`→`gpu_t4`)
- Modify: `tests/integration/test_peft_qlora_real.py` (`gpu_local`/`gpu_t4`→`gpu_t4`)
- Modify: `tests/predict/test_gpu_predict.py` (per-test: base model + vram_hint `gpu_local`→`gpu_t4`; LoRA + QLoRA predict `gpu_t4`→`gpu_t4`)
- Modify: `tests/gpu/test_calibrate_real.py`, `test_channel_adapter_gpu.py`, `test_multiplex_vram.py`, `test_predict_nchannel_gpu.py`, `test_real_train_overfits.py`, `test_real_train_qlora.py`, `test_real_train_qlora_resume.py`, `test_run_end_to_end_gpu.py` (module `pytestmark`)

- [ ] **Step 1: List every marker usage to confirm against the spec's mapping table (§2 R2).**

```bash
grep -rn "@pytest.mark.gpu_local\|@pytest.mark.gpu_t4\|@pytest.mark.gpu_xl\|pytestmark" tests/
```

- [ ] **Step 2: Apply the default rule per the spec table.** `gpu_local → gpu_t4`; `gpu_t4 → gpu_t4` (unchanged). Edit per-test markers on the two mixed-tier files (`test_load_sam31_real.py`, `test_gpu_predict.py`); replace module `pytestmark` elsewhere. **If grep surfaces a marker NOT in the spec's table, apply the default rule and note it in the policy doc (Phase E).**

- [ ] **Step 3: Verify zero live `gpu_local` references remain in tests.**

Run: `grep -rn "gpu_local" tests/ scripts/ pyproject.toml`
Expected: zero hits. **[verify on CPU/CI]**

- [ ] **Step 4: Verify the collection contract (R-counts).**

Run:
```bash
uv run pytest --collect-only -m gpu_t4 -q 2>&1 | tail -3      # expect 27 reclassed here; final gpu_t4 total is 33 after Phases B–C add tests
uv run pytest --collect-only -m gpu_xl -q 2>&1 | tail -3       # expect 0
uv run pytest --collect-only -q 2>&1 | tail -3                 # CPU collection: no strict-marker error
```
Expected: `-m gpu_t4` collects the 27 reclassed tests; `-m gpu_xl` collects 0; no strict-marker error. **[verify on CPU/CI]**

- [ ] **Step 5: Commit.**

```bash
git add tests/
git commit -m "feat(tests): reclassify 27 GPU tests gpu_local->gpu_t4 per new taxonomy"
```

---

### Task A2b: Rewrite the two EXISTING unit tests that consume the old taxonomy (X2 blast radius)

> **Why this task exists:** grounding (2026-05-31) found two existing CPU unit tests that hard-code the OLD model and WILL FAIL once A4/A5 land. They are blast-radius consumers the spec's X2 sweep demands. Rewrite them to the new model. **Do this BEFORE A4/A5 change behavior would leave them red mid-phase** — but the new assertions only pass once A4/A5 exist, so: rewrite here to the NEW contract, expect RED until A5, GREEN after A5. (Alternatively the orchestrator may fold these rewrites into A5's commit; either way they must be green at phase end.)

**Files:**
- Modify: `tests/unit/test_marker_autoskip.py` — currently builds a `_FakeItem(tier, "requires_compatible_gpu")` and asserts the OLD `_TIER_ORDER`/`_current_tier` behavior: `test_gpu_t4_skipped_on_local_runner`, `test_gpu_local_runs_on_local_runner`, `test_gpu_xl_skip_reason_names_124`. All three reference deleted symbols / the deleted `gpu_local` tier / the #124 skip-reason.
- Modify: `tests/unit/test_run_gpu_tests_script.py` — `test_local_maps_to_gpu_local_marker` asserts the REMOVED `local` selector → `gpu_local` marker mapping.

- [ ] **Step 1: Re-read both files to capture the current `_FakeItem` helper + how they invoke `pytest_collection_modifyitems`.**

```bash
cat tests/unit/test_marker_autoskip.py tests/unit/test_run_gpu_tests_script.py
```

- [ ] **Step 2: Rewrite `test_marker_autoskip.py` to the capability-subset model.** Drive the new skip predicate by monkeypatching `_satisfied_tiers()` (not `_current_tier`). New cases (mirror the existing `_FakeItem` harness):
  - on a stubbed `{gpu_t4}` card: a `gpu_t4` item is NOT skipped; a `gpu_bf16` item IS skipped (skip reason names the unmet gate, e.g. "CC ≥ 8.0").
  - on a stubbed `{gpu_t4, gpu_bf16}` card: both run.
  - on a stubbed `{}` (no compatible GPU): all tier items skip.
  - delete the `gpu_local`-runs and `#124`-skip-reason cases (both retired); replace the `gpu_xl` case with one asserting `gpu_xl` skips on a ≤16 GB card with a reason naming the >16 GB gate (no #124 reference).

- [ ] **Step 3: Rewrite `test_run_gpu_tests_script.py`.** Replace `test_local_maps_to_gpu_local_marker` with assertions for the NEW selectors (default `gpu_t4 or gpu_bf16`, `t4`→`gpu_t4`, `bf16`→`gpu_bf16`, `xl`→`gpu_xl`); a `garbage`-selector case asserting non-zero exit + usage line. (The selector REWRITE itself lands in Phase E Task E1; this test encodes the contract E1 must satisfy — it may stay RED until E1. Mark it `@pytest.mark.xfail(reason="selector rewrite lands in Phase E E1")` here if running A in isolation, and remove the xfail in E1.)

- [ ] **Step 4: Run (expect the new marker-autoskip cases RED until A5; the script cases RED until E1).**

Run: `uv run pytest tests/unit/test_marker_autoskip.py tests/unit/test_run_gpu_tests_script.py -v -o "addopts="`
Expected: marker-autoskip cases PASS after A5; script cases xfail/pending until E1. **[verify on CPU/CI]**

- [ ] **Step 5: Commit.**

```bash
git add tests/unit/test_marker_autoskip.py tests/unit/test_run_gpu_tests_script.py
git commit -m "test(unit): rewrite marker-autoskip + runner-script tests to capability taxonomy"
```

---

### Task A3: CPU unit tests for `_satisfied_tiers()` + the skip predicate (TDD, write FIRST)

**Files:**
- Create: `tests/unit/test_conftest_tiers.py`

- [ ] **Step 1: Write the failing CPU unit tests** (mirror `tests/unit/test_presets.py` `_stub_gpu` style — stub both capability and `total_memory`). These define the contract A4/A5 implement.

```python
import pytest
import custom_sam_peft  # noqa: F401  (eager import sanity)

_GB = 1024**3


def _stub_cuda(monkeypatch, *, available=True, cap=(12, 0), total_gb=16, can_launch=True):
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: available)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: cap)

    class _Props:
        total_memory = int(total_gb * _GB)

    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda *a, **k: _Props())
    # kernel-launch probe stub (re-locate the real name with grep: _torch_can_launch_kernel)
    import tests.conftest as cf
    monkeypatch.setattr(cf, "_torch_can_launch_kernel", lambda *a, **k: can_launch)


@pytest.mark.parametrize(
    "cap,total_gb,expected",
    [
        ((12, 0), 16, {"gpu_t4", "gpu_bf16"}),  # 5070 Ti
        ((7, 5), 16, {"gpu_t4"}),               # T4
        ((6, 1), 8, set()),                     # Pascal -> nothing (gate is CC 7.5)
        ((8, 0), 24, {"gpu_xl"}),               # >16 GB -> xl only
    ],
)
def test_satisfied_tiers(monkeypatch, cap, total_gb, expected):
    _stub_cuda(monkeypatch, cap=cap, total_gb=total_gb)
    from tests.conftest import _satisfied_tiers
    assert _satisfied_tiers() == expected


def test_satisfied_tiers_empty_without_cuda(monkeypatch):
    _stub_cuda(monkeypatch, available=False)
    from tests.conftest import _satisfied_tiers
    assert _satisfied_tiers() == set()


def test_has_compatible_gpu_gate_is_cc_75(monkeypatch):
    from tests.conftest import _has_compatible_gpu
    _stub_cuda(monkeypatch, cap=(6, 1))
    assert _has_compatible_gpu() is False
    _stub_cuda(monkeypatch, cap=(7, 5))
    assert _has_compatible_gpu() is True
    _stub_cuda(monkeypatch, cap=(12, 0))
    assert _has_compatible_gpu() is True
```

- [ ] **Step 2: Run to verify it fails** (the symbol does not exist yet / still named `_current_tier`).

Run: `uv run pytest tests/unit/test_conftest_tiers.py -v -o "addopts="`
Expected: FAIL (ImportError on `_satisfied_tiers`, or wrong return type). **[verify on CPU/CI]**

> Note: `-o "addopts="` bypasses the repo's global `--cov-fail-under=80` for a single-file subset run.

- [ ] **Step 3: Commit the failing test.**

```bash
git add tests/unit/test_conftest_tiers.py
git commit -m "test(tests): failing CPU unit tests for set-returning tier probe + CC 7.5 gate"
```

---

### Task A4: Implement `_satisfied_tiers()` set-returning probe + CC-gate move (R5, R7)

**Files:**
- Modify: `tests/conftest.py` — re-locate `_current_tier` (`~53-69`), `_has_compatible_gpu` (`~45-50`), `requires_compatible_gpu` registration, module docstring

- [ ] **Step 1: Move the CC gate in `_has_compatible_gpu()`** from `capability >= (6, 0)` to `capability >= (7, 5)`. **Preserve** `_torch_can_launch_kernel` and the autouse `_free_cuda_after_gpu_test` fixture unchanged.

- [ ] **Step 2: Replace `_current_tier()` with `_satisfied_tiers()`** returning a SET:

```python
_GB = 1024**3


def _satisfied_tiers() -> set[str]:
    """Return the SET of GPU tiers the live card satisfies.

    Bands are NOT linearly ordered: gpu_t4/gpu_bf16 are <=16 GB; gpu_xl is >16 GB.
    The 16 GB band is a CLOSED upper bound (<= 16 * _GB) so a card reporting
    slightly under a marketing "16 GB" (driver-reserved) still counts as gpu_t4/gpu_bf16.
    A >16 GB card satisfies only gpu_xl and is intentionally NOT auto-run for the
    <=16 GB ceiling assertions (running them on a bigger card could mask a small-card OOM).
    """
    import torch

    if not _has_compatible_gpu():
        return set()
    cc = torch.cuda.get_device_capability()
    total = torch.cuda.get_device_properties(0).total_memory
    tiers: set[str] = set()
    if cc >= (7, 5) and total <= 16 * _GB:
        tiers.add("gpu_t4")
    if cc >= (8, 0) and total <= 16 * _GB:
        tiers.add("gpu_bf16")
    if total > 16 * _GB:
        tiers.add("gpu_xl")
    return tiers
```

- [ ] **Step 3: Update the `requires_compatible_gpu` marker docstring** (CC ≥ 6.0 → CC ≥ 7.5) at its registration site, and the **conftest module docstring** (replace "GTX 1080 dev box"/"Colab T4 only smoke tier" framing with "T4-floor / 5070 Ti-primary").

- [ ] **Step 4: Run the A3 unit tests — expect PASS.**

Run: `uv run pytest tests/unit/test_conftest_tiers.py -v -o "addopts="`
Expected: all PASS. **[verify on CPU/CI]**

- [ ] **Step 5: Eager-import sanity** (symbol rename can un-import the package).

Run: `uv run python -c "import custom_sam_peft; import tests.conftest" && uv run ruff check tests/conftest.py`
Expected: no ImportError; ruff clean. **[verify on CPU/CI]**

- [ ] **Step 6: Commit.**

```bash
git add tests/conftest.py
git commit -m "feat(tests): set-returning _satisfied_tiers probe + move CC gate 6.0->7.5"
```

---

### Task A5: Capability-subset skip predicate; delete `_TIER_ORDER` (R6)

**Files:**
- Modify: `tests/conftest.py` — `pytest_collection_modifyitems`, delete `_TIER_ORDER`

- [ ] **Step 1: Extend the A3 unit test file** with skip-predicate cases (monkeypatch the probe directly):

```python
def test_skip_predicate_stub_t4_only(monkeypatch):
    """On a {gpu_t4} card, gpu_bf16 tests skip and gpu_t4 run."""
    import tests.conftest as cf
    monkeypatch.setattr(cf, "_satisfied_tiers", lambda: {"gpu_t4"})
    # exercise via a tiny in-process pytest or via the helper the predicate calls;
    # assert the active tier set drives selection: gpu_bf16 not in {gpu_t4} -> skip.
    assert "gpu_bf16" not in cf._satisfied_tiers()
    assert "gpu_t4" in cf._satisfied_tiers()
```

(If the predicate factors a pure helper `_should_skip(marker_tier, satisfied) -> str | None`, unit-test THAT directly: returns a skip-reason string when `marker_tier not in satisfied`, else `None`.)

- [ ] **Step 2: Rewrite `pytest_collection_modifyitems`** so a test marked tier `T` runs iff `T ∈ active_tiers`, where `active_tiers` is the runner's forced tier (env/CLI) when set, else `_satisfied_tiers()`. **Delete `_TIER_ORDER`** and every `>=`/index comparison over tiers. Skip reason names the unmet gate, e.g. `f"requires {T} (CC >= 8.0, <=16 GB); have CC {cc}"`.

- [ ] **Step 3: Run the unit tests — expect PASS.**

Run: `uv run pytest tests/unit/test_conftest_tiers.py -v -o "addopts="`
Expected: PASS. **[verify on CPU/CI]**

- [ ] **Step 4: Confirm `_TIER_ORDER` is gone everywhere.**

Run: `grep -rn "_TIER_ORDER" tests/ scripts/ src/`
Expected: zero hits. **[verify on CPU/CI]**

- [ ] **Step 5: Commit.**

```bash
git add tests/conftest.py tests/unit/test_conftest_tiers.py
git commit -m "feat(tests): capability-subset skip predicate; remove _TIER_ORDER"
```

---

### Task A6: New gpu_bf16 faithful-bf16 test (R4)

**Files:**
- Create: `tests/gpu/test_bf16_faithful.py`

- [ ] **Step 1: Write the test (1–2 cases), marked `gpu_bf16`.** It must auto-skip on CPU and on CC < 8.0.

```python
import pytest
import torch

pytestmark = pytest.mark.gpu_bf16


def test_bf16_not_coerced_on_cc_ge_80():
    """On CC >= 8.0 the dtype is NOT coerced; a real bf16 tensor stays bf16."""
    from custom_sam_peft.runtime._runtime import coerce_dtype_for_capability
    cap = torch.cuda.get_device_capability()
    assert cap >= (8, 0)
    assert coerce_dtype_for_capability(torch.bfloat16, cap) == torch.bfloat16
    x = torch.randn(8, 8, device="cuda", dtype=torch.bfloat16)
    y = x @ x  # a real bf16 kernel path
    assert y.dtype == torch.bfloat16
    assert torch.isfinite(y).all()
```

(Optionally extend to a 1–2 step train or a model forward built with `dtype=bfloat16`, asserting a representative parameter/activation has `.dtype == torch.bfloat16` — keep it small.)

- [ ] **Step 2: Verify collection (CPU).**

Run: `uv run pytest --collect-only -m gpu_bf16 -q 2>&1 | tail -3`
Expected: collects exactly the new test(s). On CPU it auto-skips at the gate. **[verify on CPU/CI]**

- [ ] **Step 3: Verify it runs and asserts non-coerced bf16.** **[verify on 5070 Ti]** — runs on the 5070 Ti; skips on a T4 (CC 7.5 < 8.0).

- [ ] **Step 4: Commit.**

```bash
git add tests/gpu/test_bf16_faithful.py
git commit -m "test(gpu): faithful non-coerced bf16 test for gpu_bf16 tier (#139)"
```

---

### Task A7: Delete Pascal artifacts; banner the gtx1080 doc (R8)

**Files:**
- Modify: `pyproject.toml` (remove `gpu-pascal` extra + its index/source routing)
- Delete: `docs/testing/local-pascal-gpu-testing.md`
- Modify: `docs/testing/manual-gpu-pass-2026-05-24-gtx1080.md` (add banner)

- [ ] **Step 1: Locate and remove the `gpu-pascal` (cu118) uv extra and its `[tool.uv.sources]` / index routing.**

```bash
grep -n "gpu-pascal\|cu118\|pascal" pyproject.toml
```

Remove the extra and any cu118 index/source entry. The default cu130 wheel covers both T4 and 5070 Ti.

- [ ] **Step 2: Delete the Pascal doc.**

```bash
git rm docs/testing/local-pascal-gpu-testing.md
```

- [ ] **Step 3: Add the banner to the gtx1080 doc** (top of file):

```markdown
> **Superseded by the RTX 5070 Ti; Pascal is no longer supported as of this PR (min supported GPU: Tesla T4, CC 7.5).**
```

- [ ] **Step 4: Confirm no live Pascal references remain.**

```bash
grep -rn "gpu-pascal\|cu118" pyproject.toml docs/ .github/ uv.lock
```
Expected: zero live references in config/CI/docs (dated history excepted). **[verify on CPU/CI]**

- [ ] **Step 5: Confirm uv still resolves on the default extra.**

Run: `uv sync --frozen 2>&1 | tail -5 || uv sync 2>&1 | tail -5`
Expected: resolves cleanly on the default extra. **[verify on CPU/CI]**

- [ ] **Step 6: Commit.**

```bash
git add pyproject.toml docs/testing/manual-gpu-pass-2026-05-24-gtx1080.md
git rm --cached docs/testing/local-pascal-gpu-testing.md 2>/dev/null; true
git commit -m "chore: drop gpu-pascal extra + Pascal doc; banner gtx1080 record (T4 floor)"
```

---

### Task A8: R9 guard + X2 blast-radius full-suite gate

**Files:** none new (verification task)

- [ ] **Step 1: Confirm `coerce_dtype_for_capability` is untouched (R9).**

Run: `git diff --stat -- src/custom_sam_peft/runtime/_runtime.py`
Expected: no diff to `_runtime.py`. **[verify on CPU/CI]**

- [ ] **Step 2: Blast-radius grep sweep (X2).**

```bash
grep -rn "gpu_local\|@pytest.mark.gpu_local" tests/ scripts/ src/   # 0
grep -rn "_TIER_ORDER\|_current_tier" tests/ scripts/ src/           # 0 (renamed)
grep -rn "capability >= (6\|CC >= 6\|sm_61\|Pascal" src/ tests/ scripts/  # only dated history
```

- [ ] **Step 3: Run the FULL CPU suite (X2).**

Run: `uv run pytest`
Expected: green; CPU collection clean; GPU tiers auto-skip. **[verify on CPU/CI]**

- [ ] **Step 4: Commit (if any cleanup edits were needed).**

```bash
git add -A && git commit -m "chore(tests): blast-radius sweep — full CPU suite green after taxonomy change" || true
```

---

## Interface Contract — END OF PHASE A

**EXPOSES (later phases CONSUME these):**

- **Markers (registered, `--strict-markers`-safe):** `gpu_t4` (CC ≥ 7.5 ∧ ≤16 GB), `gpu_bf16` (CC ≥ 8.0 ∧ ≤16 GB), `gpu_xl` (>16 GB, empty). `gpu_local` is **deleted** — do not use it.
  - New GPU tests in Phase B/C tag `@pytest.mark.gpu_t4` (the ≤16 GB fp16 band).
- **`tests/conftest.py::_satisfied_tiers() -> set[str]`** — set-returning live capability probe; monkeypatchable for CPU unit tests.
- **`tests/conftest.py::_has_compatible_gpu()`** — gate is now **CC ≥ 7.5** + kernel-launch probe. Preserved helpers: `_torch_can_launch_kernel`, autouse `_free_cuda_after_gpu_test`. `requires_compatible_gpu` docstring says CC ≥ 7.5.
- **Skip predicate:** `pytest_collection_modifyitems` runs a test iff its tier ∈ active tiers (forced-tier env/CLI else `_satisfied_tiers()`). **`_TIER_ORDER` is deleted.** Phase E's runner sets the forced tier via the same env/CLI hook Phase A reads — **Phase E must read how A consumes the forced tier** (env var name or `-m` filter) and match it.
- **Collection counts:** `-m gpu_t4` → 27 at end of Phase A (grows to 33 as Phases B–C add the predict-budget-warning, 8 GB-ceiling, predict-fits-8GB, all-scope-LoRA, and qlora-load-attached gpu_t4 tests); `-m gpu_bf16` → the new bf16 test(s); `-m gpu_xl` → 0; CPU full suite green.
- **Runtime unchanged:** `coerce_dtype_for_capability` in `src/custom_sam_peft/runtime/_runtime.py` is identical to pre-PR.

**CONSUMES:** nothing prior.

---

## Phase B — Empirical predict-budget warning

**Requirements:** R13, R14, R15, R16, R17, R18 (guard), X1 (cite-or-tbd), R12 (provenance row for `PREDICT_8GB_BUDGET_GB`).

**Goal of phase:** Add a downward "trained model may not fit a small card" warning to the train path — **empirical only, never analytic**. A single batch=1/K=1 predict-path probe at model-ready (pre-loop) measures peak VRAM; a **pure decision function** compares it to `PREDICT_8GB_BUDGET_GB` and returns (warn, message). CPU no-op. The upward hint at `predict/runner.py` is untouched.

> **Parallelizable with Phase A** (no shared code) — but both touch `docs/defaults-provenance.md`; serialize that file's commits.

---

### Task B1: Define `PREDICT_8GB_BUDGET_GB` + pure decision function (TDD)

**Files:**
- Create: `src/custom_sam_peft/predict/budget.py` (or co-locate in an existing predict/train sizing module — implementer confirms; this plan assumes a new small module)
- Create: `tests/unit/test_predict_budget.py`

- [ ] **Step 1: Write the failing CPU unit test FIRST.**

```python
def test_decide_predict_budget_warning_over_budget():
    from custom_sam_peft.predict.budget import decide_predict_budget_warning, PREDICT_8GB_BUDGET_GB
    over = int((PREDICT_8GB_BUDGET_GB + 1.0) * 1024**3)
    warn, msg = decide_predict_budget_warning(measured_bytes=over, budget_gb=PREDICT_8GB_BUDGET_GB)
    assert warn is True
    assert "may not be usable" in msg
    assert "8 GB" in msg or "7.0" in msg


def test_decide_predict_budget_warning_under_budget():
    from custom_sam_peft.predict.budget import decide_predict_budget_warning, PREDICT_8GB_BUDGET_GB
    under = int((PREDICT_8GB_BUDGET_GB - 1.0) * 1024**3)
    warn, msg = decide_predict_budget_warning(measured_bytes=under, budget_gb=PREDICT_8GB_BUDGET_GB)
    assert warn is False
```

- [ ] **Step 2: Run — expect FAIL.**

Run: `uv run pytest tests/unit/test_predict_budget.py -v -o "addopts="`
Expected: FAIL (ImportError). **[verify on CPU/CI]**

- [ ] **Step 3: Implement the constant + pure function.**

```python
"""Empirical predict-footprint budget for the 8 GB / CC 7.5 minimum-supported card."""

# cite: a CC 7.5 / 8 GB card has ~8.0 GB nominal; subtract ~1.0 GB driver/CUDA-context
# reservation (consistent with the ~1.0 GiB headroom convention in presets.py::_headroom_bytes)
# to get the usable predict budget. Date: 2026-05-31.
# tbd: #142 — replace the ~1.0 GB reservation with a measured figure from a real 8 GB card.
PREDICT_8GB_BUDGET_GB: float = 7.0


def decide_predict_budget_warning(measured_bytes: int, budget_gb: float = PREDICT_8GB_BUDGET_GB) -> tuple[bool, str]:
    """Pure decision: warn iff the measured predict peak exceeds the small-card budget."""
    measured_gb = measured_bytes / (1024**3)
    if measured_gb > budget_gb:
        return True, (
            f"the trained model's predict footprint is ~{measured_gb:.1f} GB; it may not be "
            f"usable for prediction on 8 GB / CC 7.5 GPUs (budget ~{budget_gb:.1f} GB)."
        )
    return False, ""
```

- [ ] **Step 4: Run — expect PASS.**

Run: `uv run pytest tests/unit/test_predict_budget.py -v -o "addopts="`
Expected: PASS. **[verify on CPU/CI]**

- [ ] **Step 5: Add the `docs/defaults-provenance.md` row (R12/X1).** Add a row: constant `PREDICT_8GB_BUDGET_GB`, value `7.0`, source (8 GB nominal − ~1.0 GB reservation; `presets.py::_headroom_bytes` convention), date `2026-05-31`, context (CC 7.5 / 8 GB predict). Markdown-lint the file (Phase E gate also covers it).

- [ ] **Step 6: Commit.**

```bash
git add src/custom_sam_peft/predict/budget.py tests/unit/test_predict_budget.py docs/defaults-provenance.md
git commit -m "feat(predict): PREDICT_8GB_BUDGET_GB + pure budget-warning decision fn (cited)"
```

---

### Task B2: Wire the empirical probe + warning into the train entrypoint (R14, R15, R16)

**Files:**
- Modify: `src/custom_sam_peft/train/loop.py` or `src/custom_sam_peft/train/trainer.py` — re-locate the post-construction / pre-loop seam:

```bash
grep -rn "def train\|model.*ready\|for epoch\|training loop\|max_memory_allocated\|reset_peak_memory_stats" src/custom_sam_peft/train/
```

- [ ] **Step 1: Insert the probe at model-ready, BEFORE the training loop.** Clean no-op on CPU.

```python
import torch
from custom_sam_peft.predict.budget import PREDICT_8GB_BUDGET_GB, decide_predict_budget_warning

# ... after model + adapter are built/loaded and ready, before the loop:
if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()
    _run_minimal_predict_probe(model, batch_size=1, k=1)  # one forward; NO training step
    measured_predict_peak = torch.cuda.max_memory_allocated()
    warn, msg = decide_predict_budget_warning(measured_predict_peak)
    if warn:
        logger.warning(msg)  # warn-not-block; training continues
```

The `_run_minimal_predict_probe` helper runs exactly one batch=1/K=1 forward (cheap). **No analytic model is consulted.**

- [ ] **Step 2: CPU no-op verification** — the probe block is skipped entirely when `not torch.cuda.is_available()`.

Run: `uv run python -c "import custom_sam_peft" && uv run ruff check src/custom_sam_peft/train/`
Expected: imports clean; ruff clean. **[verify on CPU/CI]**

- [ ] **Step 3: CPU integration check** — run an existing CPU train smoke (if one exists) or a `pytest -k train` CPU subset; confirm no CUDA call / no error on CPU.

Run: `uv run pytest tests/ -k "train and not gpu" -q -o "addopts="`
Expected: green; no CUDA error. **[verify on CPU/CI]**

- [ ] **Step 4: Commit.**

```bash
git add src/custom_sam_peft/train/
git commit -m "feat(train): empirical model-ready predict-budget probe + warn-not-block (CPU no-op)"
```

---

### Task B3: GPU test for the warning — warn + no-warn branches (R17)

**Files:**
- Create: `tests/gpu/test_predict_budget_warning.py`

- [ ] **Step 1: Write the `gpu_t4` test** (auto-skips on CPU). Covers (a) over-budget → warns (drive an over-budget config or inject a measured value through the seam), and (b) the small config's real predict peak ≤ budget → no warning (consistent with R11).

```python
import pytest

pytestmark = pytest.mark.gpu_t4


def test_warning_fires_when_over_budget(caplog):
    # build a config/path whose predict probe exceeds PREDICT_8GB_BUDGET_GB (or inject),
    # run the train entrypoint up to the probe, assert logger.warning fired with the message.
    ...


def test_no_warning_for_small_config(caplog):
    # min_gpu_qlora-class config: real predict peak <= budget -> no warning.
    ...
```

- [ ] **Step 2: Collection check (CPU).**

Run: `uv run pytest --collect-only -m gpu_t4 tests/gpu/test_predict_budget_warning.py -q 2>&1 | tail -3`
Expected: collects; auto-skips on CPU. **[verify on CPU/CI]**

- [ ] **Step 3: Real-measurement check.** **[verify on 5070 Ti]** — both warn and no-warn branches pass against real measurements.

- [ ] **Step 4: R18 guard — upward hint untouched.**

Run: `git diff --stat -- src/custom_sam_peft/predict/runner.py tests/predict/test_gpu_predict.py`
Expected: no diff to the upward-hint code at `predict/runner.py` / its test. **[verify on CPU/CI]**

- [ ] **Step 5: Commit.**

```bash
git add tests/gpu/test_predict_budget_warning.py
git commit -m "test(gpu): predict-budget warning warn/no-warn branches (#142)"
```

---

## Interface Contract — END OF PHASE B

**EXPOSES:**

- **`src/custom_sam_peft/predict/budget.py`:**
  - `PREDICT_8GB_BUDGET_GB: float = 7.0` (cited; `# tbd: #142` on the reservation line). **Phase C's R11 predict-fits-8GB test imports and asserts against THIS constant.**
  - `decide_predict_budget_warning(measured_bytes: int, budget_gb: float = PREDICT_8GB_BUDGET_GB) -> tuple[bool, str]` — pure, CPU-testable.
- **Train-entrypoint hook:** an empirical batch=1/K=1 predict probe at model-ready (pre-loop), CPU no-op, warn-not-block. Phase E's policy doc documents this.
- **`docs/defaults-provenance.md`** has a `PREDICT_8GB_BUDGET_GB` row.

**CONSUMES:** none of Phase A's code (shares only `docs/defaults-provenance.md`).

---

## Phase C — #142 8 GB-ceiling train + predict; #195 budgets; #83 probe

**Requirements:** R10, R11, R12 (`QLORA_8GB_CEIL_GB` row), R30 for #142/#83/#195, X1.

**Goal of phase:** Prove an 8 GB / CC 7.5 card supports BOTH train and predict of the small config (validated on the 16 GB 5070 Ti via 8 GB-ceiling/budget assertions); de-Pascal `min_gpu_qlora.yaml`; confirm-or-retune #195's step budgets; run the #83 all-scope LoRA peak probe and branch.

> **CONSUMES Phase A** (tags new tests `gpu_t4`) and **Phase B** (`PREDICT_8GB_BUDGET_GB`).

---

### Task C1: #142 8 GB-ceiling QLoRA training smoke + `QLORA_8GB_CEIL_GB` (R10, R12)

**Files:**
- Modify or Create: a `gpu_t4` test (e.g. extend `tests/gpu/test_real_train_qlora.py` or new `tests/gpu/test_qlora_8gb_ceiling.py`)
- Reference config: `configs/examples/min_gpu_qlora.yaml`

- [ ] **Step 1: Add the ceiling constant with provenance** (near the test or in a shared test constants module):

```python
# cite: measured ~5.0 GB peak (GTX 1080, fp16) in
# docs/research/2026-05-24-issue-137-qlora-8gb-feasibility.md. 8.0 GB = target
# minimum-card envelope with ~3 GB margin over the measured peak. Date 2026-05-31.
# tbd: #142 — record the 5070 Ti measured peak here after the §9 run (keep the 8.0 envelope).
QLORA_8GB_CEIL_GB: float = 8.0
```

- [ ] **Step 2: Write the smoke** — `gpu_t4`, runs the 2-image `tests/fixtures/tiny_coco/` overfit (epochs=25, batch=1, grad_accum=1 → 50 updates), measures peak via `reset_peak_memory_stats()` + `max_memory_allocated()`, asserts `peak <= QLORA_8GB_CEIL_GB * 1024**3` AND a loss-drop assertion consistent with the existing smokes.

- [ ] **Step 3: Collection check (CPU).**

Run: `uv run pytest --collect-only -m gpu_t4 -q 2>&1 | tail -3`
Expected: this smoke adds 1 gpu_t4 test; the final Phase-C gpu_t4 total is **33** (27 reclassed + 6 net-new across Phases B–C). **[verify on CPU/CI]**

- [ ] **Step 4: Real run.** **[verify on 5070 Ti]** — overfits, `peak <= 8.0 GB`; record the measured peak in the constant comment (resolves the `# tbd: #142` measured-peak line) and the evidence artifact.

- [ ] **Step 5: Provenance row (R12).** Add `QLORA_8GB_CEIL_GB` to `docs/defaults-provenance.md` (name, 8.0, source = issue-137 feasibility doc + 3 GB margin, date, GPU/config context).

- [ ] **Step 6: Commit.**

```bash
git add tests/gpu/ docs/defaults-provenance.md
git commit -m "feat(gpu): #142 8GB-ceiling QLoRA train smoke + QLORA_8GB_CEIL_GB (cited)"
```

---

### Task C2: De-Pascal `min_gpu_qlora.yaml` rationale (R10)

**Files:**
- Modify: `configs/examples/min_gpu_qlora.yaml`

- [ ] **Step 1: Re-locate the Pascal/sm_61 rationale comments.**

```bash
grep -n "Pascal\|sm_61\|float16\|dtype" configs/examples/min_gpu_qlora.yaml
```

- [ ] **Step 2: Keep `dtype: float16`; rewrite the rationale.** Replace "Pascal-required / sm_61" wording with:

```yaml
# fp16 to model the CC 7.5 / 8 GB minimum-supported card (bf16 is coerced to fp16
# below CC 8.0 — see src/custom_sam_peft/runtime/_runtime.py coerce_dtype_for_capability);
# fp16 keeps the 8 GB-envelope assertion honest.
```

- [ ] **Step 3: Verify no Pascal rationale remains.**

Run: `grep -n "Pascal\|sm_61" configs/examples/min_gpu_qlora.yaml`
Expected: zero hits. **[verify on CPU/CI]**

- [ ] **Step 4: Commit.**

```bash
git add configs/examples/min_gpu_qlora.yaml
git commit -m "docs(config): de-Pascal min_gpu_qlora rationale (CC 7.5 / 8 GB framing)"
```

---

### Task C3: predict-fits-8GB validation (R11)

**Files:**
- Create: `tests/predict/test_predict_fits_8gb.py`

- [ ] **Step 1: Write the `gpu_t4` test** — builds the `min_gpu_qlora`-class adapter/model (reuse `tests/predict/fixtures/qlora_adapter` where possible), runs a batch=1/K=1 predict path, measures peak, asserts `peak <= PREDICT_8GB_BUDGET_GB * 1024**3` (imports the Phase B constant).

```python
import pytest
from custom_sam_peft.predict.budget import PREDICT_8GB_BUDGET_GB

pytestmark = pytest.mark.gpu_t4
```

- [ ] **Step 2: Collection check (CPU).**

Run: `uv run pytest --collect-only -m gpu_t4 tests/predict/test_predict_fits_8gb.py -q 2>&1 | tail -3`
Expected: collects; auto-skips on CPU. **[verify on CPU/CI]**

- [ ] **Step 3: Real run.** **[verify on 5070 Ti]** — small-config predict peak ≤ `PREDICT_8GB_BUDGET_GB`; record in evidence artifact.

- [ ] **Step 4: Commit.**

```bash
git add tests/predict/test_predict_fits_8gb.py
git commit -m "test(predict): #142 predict-fits-8GB validation against PREDICT_8GB_BUDGET_GB"
```

---

### Task C4: #195 — confirm-or-retune the 25/50-step budgets (R30 for #195)

**Files:**
- Modify (only if retune needed): `tests/gpu/test_real_train_qlora_resume.py`, `tests/gpu/test_real_train_qlora.py` (the `test_qlora_overfits_in_50_steps` docstring/claims)

- [ ] **Step 1: Re-read the current step budgets + docstring claims.**

```bash
grep -n "50\|25\|step\|epoch\|overfit\|min" tests/gpu/test_real_train_qlora_resume.py tests/gpu/test_real_train_qlora.py
```

- [ ] **Step 2: Run the 2-image overfit on the 5070 Ti** and record speed/convergence. **[verify on 5070 Ti]**

- [ ] **Step 3: Confirm or retune.** If the 25/50-step budgets hold on the 5070 Ti, confirm in the evidence artifact (no code change). If a retune is needed, change the budget + docstring claims **and carry provenance** (measured figure + GPU + date). **[verify on 5070 Ti]**

- [ ] **Step 4: Commit (only if retuned).**

```bash
git add tests/gpu/ && git commit -m "test(gpu): #195 confirm/retune 25/50-step budgets vs 5070 Ti measurement" || true
```

---

### Task C5: #83 — all-scope LoRA peak probe + branch (R30 for #83)

**Files:**
- Create (branch a only): `tests/gpu/test_peft_scope_coverage_gpu.py` (a `gpu_t4` all-scope LoRA smoke)
- (CPU wiring already covered by `tests/unit/test_peft_scope_coverage.py` — do NOT duplicate.)

- [ ] **Step 1: Probe all-scope (regex `.*`) LoRA peak VRAM on the 5070 Ti.** **[verify on 5070 Ti]** Record the measured peak in the evidence artifact.

- [ ] **Step 2: Branch on the measurement.**
  - **(a) ≤ ~15 GB with margin →** add a `gpu_t4` all-scope LoRA smoke; close #83 **DONE**.
  - **(b) > 16 GB →** do NOT add a `gpu_t4` test; record the measured number for the gpu_xl issue (Phase E, R31); close #83 as **superseded**.

- [ ] **Step 3: Collection check if branch (a) (CPU).**

Run: `uv run pytest --collect-only -m gpu_t4 tests/gpu/test_peft_scope_coverage_gpu.py -q 2>&1 | tail -3`
Expected: collects (branch a only); auto-skips on CPU. **[verify on CPU/CI]**

- [ ] **Step 4: Commit (branch a) or record-only (branch b).**

```bash
git add tests/gpu/test_peft_scope_coverage_gpu.py 2>/dev/null && git commit -m "test(gpu): #83 all-scope LoRA smoke (peak fits gpu_t4 band)" || true
```

---

## Interface Contract — END OF PHASE C

**EXPOSES:**

- **`QLORA_8GB_CEIL_GB = 8.0`** (cited; `# tbd: #142` measured-peak line resolved after the 5070 Ti run) — provenance row in `docs/defaults-provenance.md`.
- **#142 deliverables:** 8 GB-ceiling QLoRA train smoke (`gpu_t4`, total gpu_t4 collection count now **33**) + predict-fits-8GB test, both asserting against `QLORA_8GB_CEIL_GB` / `PREDICT_8GB_BUDGET_GB`.
- **`configs/examples/min_gpu_qlora.yaml`** carries CC 7.5 / 8 GB rationale (no Pascal/sm_61).
- **#195 status:** step budgets confirmed-or-retuned vs 5070 Ti (recorded in evidence artifact).
- **#83 branch decision** + measured all-scope LoRA peak (feeds Phase E: either a landed `gpu_t4` smoke, or a number for the gpu_xl issue body).

**CONSUMES:** Phase A markers (`gpu_t4`); Phase B `PREDICT_8GB_BUDGET_GB`.

---

## Phase D — CPU integration audit + bounded regression tests

**Requirements:** R19, R20, R21, R22 (R22 doc text lands fully in Phase E; the principle is drafted here for the audit subsection).

**Goal of phase:** Add three bounded CPU/stub regression tests for the GPU-bug classes that escaped (per `docs/testing/gpu-audit-2026-05-24.md`), using `tests/fixtures/tiny_sam3_stub.py::TinySam3Stub` where possible. Record the audit findings (consumed by Phase E's policy-doc rewrite).

> **Pure CPU; parallelizable** with A/B/C (no GPU, no shared source with the train/predict hooks). Serialize commits if run in parallel.

---

### Task D1: dtype-consistency contract test (R20, bug class 1)

**Files:**
- Create: `tests/unit/test_channel_adapter_dtype.py`
- Reference: `tests/fixtures/tiny_sam3_stub.py::TinySam3Stub`, bug in `src/custom_sam_peft/models/sam3.py`

- [ ] **Step 1: Write the CPU test** — drive a stub forward with a mismatched input dtype through the channel_adapter Conv2d path; assert the contract the GPU bug violated (clear error OR coercion). Reference the audit bug class in a docstring.

- [ ] **Step 2: Run — expect PASS (and that it would FAIL against the pre-fix behavior).**

Run: `uv run pytest tests/unit/test_channel_adapter_dtype.py -v -o "addopts="`
Expected: PASS. NOT GPU-gated (runs on `ubuntu-latest`). **[verify on CPU/CI]**

- [ ] **Step 3: Commit.**

```bash
git add tests/unit/test_channel_adapter_dtype.py
git commit -m "test(cpu): channel_adapter dtype-consistency contract (audit bug class 1)"
```

---

### Task D2: non-tensor forward-output contract test (R20, bug class 2)

**Files:**
- Create: `tests/unit/test_row_outputs_nontensor.py`
- Reference: `_row_outputs` in `src/custom_sam_peft/eval/evaluator.py`

- [ ] **Step 1: Write the CPU test** — feed a stub `forward_grounding` output containing a non-tensor entry through `_row_outputs`; assert it no longer KeyErrors (handles or skips non-tensor entries). Reference the audit bug class.

- [ ] **Step 2: Run — expect PASS.**

Run: `uv run pytest tests/unit/test_row_outputs_nontensor.py -v -o "addopts="`
Expected: PASS; NOT GPU-gated. **[verify on CPU/CI]**

- [ ] **Step 3: Commit.**

```bash
git add tests/unit/test_row_outputs_nontensor.py
git commit -m "test(cpu): _row_outputs non-tensor forward-output contract (audit bug class 2)"
```

---

### Task D3: image-size / 1008-RoPE contract test (R20, bug class 3)

**Files:**
- Create: `tests/unit/test_predict_image_size_contract.py`
- Reference: `_BUILTIN_DEFAULT_IMAGE_SIZE` and `load_sam31` in `src/custom_sam_peft/predict/runner.py`

- [ ] **Step 1: Write the CPU test** — assert the default image size the predict path uses is consistent with `load_sam31`'s 1008 RoPE expectation (catches the 1024-vs-1008 class on CPU). Reference the audit bug class.

- [ ] **Step 2: Run — expect PASS.**

Run: `uv run pytest tests/unit/test_predict_image_size_contract.py -v -o "addopts="`
Expected: PASS; NOT GPU-gated. **[verify on CPU/CI]**

- [ ] **Step 3: Commit.**

```bash
git add tests/unit/test_predict_image_size_contract.py
git commit -m "test(cpu): predict default-image-size matches 1008 RoPE contract (audit bug class 3)"
```

---

### Task D4: Audit record + R21 deferral check

**Files:**
- Create/append: a short audit note (a scratch markdown or a section staged for Phase E's policy doc — the audit subsection lands in `docs/testing/gpu-test-policy.md` in Phase E). Capture here: for each of the 3 bug classes, the guarding CPU test path.

- [ ] **Step 1: Record the audit mapping** (bug class → guarding CPU test):
  - channel_adapter Conv2d dtype → `tests/unit/test_channel_adapter_dtype.py`
  - `_row_outputs` non-tensor → `tests/unit/test_row_outputs_nontensor.py`
  - image-size 1024-vs-1008 → `tests/unit/test_predict_image_size_contract.py`

- [ ] **Step 2: R21 — file a follow-up issue ONLY if the audit surfaces a large net-new area beyond the 3 tests.**

```bash
# only if needed:
gh issue create --assignee @me --label testing --title "<area>" --body "Surfaced by the GPU-test-migration audit; out of scope for that PR."
```

If nothing surfaces, the audit states "no further coverage gaps." Record the issue number (if any) for Phase E's policy doc.

- [ ] **Step 3: Run the full CPU suite to confirm the 3 new tests integrate.**

Run: `uv run pytest`
Expected: green. **[verify on CPU/CI]**

- [ ] **Step 4: Commit any staged audit note.**

```bash
git add -A && git commit -m "docs(testing): record integration-audit bug-class -> CPU-test mapping" || true
```

---

## Interface Contract — END OF PHASE D

**EXPOSES:**

- Three CPU/stub regression tests (NOT GPU-gated; run in CI on `ubuntu-latest`):
  - `tests/unit/test_channel_adapter_dtype.py` (bug class 1)
  - `tests/unit/test_row_outputs_nontensor.py` (bug class 2)
  - `tests/unit/test_predict_image_size_contract.py` (bug class 3)
- **Audit mapping** (bug class → guarding test) + any R21 follow-up issue number — **Phase E's policy-doc audit subsection (R19) consumes this verbatim.**

**CONSUMES:** nothing GPU (pure CPU).

---

## Phase E — Runner, notebook, policy doc, evidence gate, issue closures

**Requirements:** R23, R24, R25, R26, R27, R28, R29, R30 (PR body), R31, R32, R33, X3 (final lint gate). Consumes Phases A–D.

**Goal of phase:** Rewrite the runner selectors to the new taxonomy; build the minimal Colab surface; rewrite the policy doc; add the provable-non-blocking evidence gate + its test; keep `gpu-deselect-check`; file the gpu_xl issue; wire per-issue closures with the Colab-dependent closures (#139, #193) **LAST, gated on the user's confirmation**. This phase opens the PR (final phase).

> **CONSUMES Phases A–D.** Read each prior phase's Interface Contract; you should not need to re-read their code.

---

### Task E1: Rewrite `scripts/run_gpu_tests.sh` selectors (R26, R27)

**Files:**
- Modify: `scripts/run_gpu_tests.sh`

- [ ] **Step 1: Re-read the current runner** (selectors, the `local` one-process-per-file loop, the stateful `--deselect` convention).

```bash
grep -n "local\|t4\|xl\|deselect\|-m \|case " scripts/run_gpu_tests.sh
```

- [ ] **Step 2: Replace the `local` selector; implement the new selector→marker map.** Preserve the **one-pytest-process-per-file** behavior (checkpoint memory release) and the `--deselect` convention on the default path. **Match Phase A's forced-tier mechanism** (the env var / `-m` filter `pytest_collection_modifyitems` reads).

| Selector | Marker filter |
|----------|---------------|
| (default) | `gpu_t4 or gpu_bf16` |
| `t4` | `gpu_t4` |
| `bf16` | `gpu_bf16` |
| `xl` | `gpu_xl` |
| `colab-min` | the R28 curated subset (load+forward + one short smoke) |
| `light` | the R25 subset |

Add a usage line; `run_gpu_tests.sh garbage` exits non-zero.

- [ ] **Step 3: Update every caller of the removed `local` selector.**

```bash
grep -rn "run_gpu_tests.sh" docs/ notebooks/ .github/
```
Update each to a new selector. No caller still passes `local`.

- [ ] **Step 4: Verify on a CPU box every selector autoskips clean.**

Run:
```bash
for s in "" t4 bf16 xl colab-min light; do bash scripts/run_gpu_tests.sh $s; echo "exit=$? selector=$s"; done
bash scripts/run_gpu_tests.sh garbage; echo "garbage exit=$?"   # expect non-zero + usage
shellcheck scripts/run_gpu_tests.sh                              # clean
```
Expected: each valid selector exits 0 (clean autoskip on CPU); `garbage` exits non-zero with a usage line; shellcheck clean. **[verify on CPU/CI]**

- [ ] **Step 5: Commit.**

```bash
git add scripts/run_gpu_tests.sh docs/ notebooks/ .github/
git commit -m "feat(scripts): rewrite run_gpu_tests.sh selectors to capability taxonomy"
```

---

### Task E2: Standalone non-blocking evidence-check script + its test (R23, R33)

**Files:**
- Create: `scripts/check_gpu_evidence.sh` (standalone, unit-testable off-CI; **always exits 0**)
- Create: `tests/unit/test_gpu_evidence_check.py`
- Modify: a CI workflow under `.github/workflows/` (non-blocking job; keep `gpu-deselect-check`)

- [ ] **Step 1: Write the failing test FIRST (R33 — the provable-non-blocking lock).**

```python
import subprocess
from pathlib import Path

SCRIPT = "scripts/check_gpu_evidence.sh"


def _run(args):
    return subprocess.run(["bash", SCRIPT, *args], capture_output=True, text=True)


def test_exit_zero_when_artifact_missing(tmp_path):
    r = _run([str(tmp_path / "nonexistent.md"), "deadbeef"])
    assert r.returncode == 0          # non-blocking even when missing


def test_exit_zero_when_artifact_stale(tmp_path):
    art = tmp_path / "evidence.md"
    art.write_text("evidence for commit OLDSHA\n")
    r = _run([str(art), "NEWSHA"])
    assert r.returncode == 0          # non-blocking even when stale
    assert "stale" in (r.stdout + r.stderr).lower()


def test_exit_zero_and_green_when_current(tmp_path):
    art = tmp_path / "evidence.md"
    art.write_text("evidence for commit GOODSHA\n")
    r = _run([str(art), "GOODSHA"])
    assert r.returncode == 0
    out = (r.stdout + r.stderr).lower()
    assert "ok" in out or "current" in out or "green" in out


def test_workflow_declares_job_non_required():
    # assert the workflow YAML for the evidence job is not in any required-status set
    wf = Path(".github/workflows")
    text = "\n".join(p.read_text() for p in wf.glob("*.yml"))
    assert "gpu-evidence" in text
    # crude guard: the evidence job/step must not be marked required/blocking
    assert "required: true" not in text or "gpu-evidence" not in text.split("required: true")[0][-200:]
```

- [ ] **Step 2: Run — expect FAIL** (script absent).

Run: `uv run pytest tests/unit/test_gpu_evidence_check.py -v -o "addopts="`
Expected: FAIL. **[verify on CPU/CI]**

- [ ] **Step 3: Implement `scripts/check_gpu_evidence.sh`** — takes `<artifact_path> <head_sha>`; **always `exit 0`**. Reports: `missing` (no artifact), `stale` (artifact does not reference the head sha / fails the documented freshness signal), or `ok/current` (references head sha). Emits only a warning annotation / neutral message for missing/stale; green only for current. Document the freshness signal in a comment.

- [ ] **Step 4: Run — expect PASS.**

Run: `uv run pytest tests/unit/test_gpu_evidence_check.py -v -o "addopts=" && shellcheck scripts/check_gpu_evidence.sh`
Expected: PASS; shellcheck clean. **[verify on CPU/CI]**

- [ ] **Step 5: Add the non-blocking CI job.** Add a `gpu-evidence` job/step invoking the script; conclusion is **neutral/warning, never failing/required**. Do NOT add it to any branch-protection required-status set; do NOT mark it `required`. **Keep `gpu-deselect-check` unchanged** (additive — both run).

- [ ] **Step 6: Commit.**

```bash
git add scripts/check_gpu_evidence.sh tests/unit/test_gpu_evidence_check.py .github/workflows/
git commit -m "feat(ci): non-blocking GPU-evidence check (always exit 0) + provable-non-blocking test"
```

---

### Task E3: Minimal Colab T4 surface (R28)

**Files:**
- Modify: `notebooks/colab_gpu_tests.ipynb`

- [ ] **Step 1: Re-read the notebook's runner cell.**

```bash
grep -n "run_gpu_tests\|local\|t4\|pip install\|forward\|smoke" notebooks/colab_gpu_tests.ipynb
```

- [ ] **Step 2: Update the runner cell to call `colab-min`.** The `colab-min` selector (E1) maps to exactly: `test_load_sam31_real.py::test_load_sam31_forward_to_canonical` + one short smoke (`test_real_train_qlora.py::test_qlora_overfits_in_50_steps`). Keep install + load real SAM 3.1. **NOT the full suite.**

- [ ] **Step 3: Add a cell that captures a cheap few-step T4 timing sample** (for #193) and **document that bf16 is coerced on the T4** (for #139's finding).

- [ ] **Step 4: Verify the notebook JSON is valid.**

Run: `uv run python -c "import json,sys; json.load(open('notebooks/colab_gpu_tests.ipynb'))" && echo OK`
Expected: OK (valid JSON). **[verify on CPU/CI]** (Execution itself is **[verify on Colab T4 — user-triggered]**.)

- [ ] **Step 5: Commit.**

```bash
git add notebooks/colab_gpu_tests.ipynb
git commit -m "feat(notebooks): minimal colab-min T4 surface + #139 coercion note + #193 timing sample"
```

---

### Task E4: Rewrite the policy doc (R22, R19, R25, R23, R29)

**Files:**
- Modify: `docs/testing/gpu-test-policy.md`

- [ ] **Step 1: Rewrite to the new taxonomy.** Include: the three capability-named tiers + gates; the auto-detection model (`_satisfied_tiers()`); the T4-floor / Pascal-dropped decision; per-tier counts + runner selectors (the E1 table); the **CPU-first review gate (R22)** stated verbatim — "Test on CPU by default; a test earns a GPU tier ONLY when it needs real weights / kernels / quant that a stub cannot reproduce." — listed in an "adding a new GPU test" checklist; the **integration-audit subsection (R19)** from Phase D's mapping; the **light-subset definition (R25)**; and the **non-blocking evidence gate (R23)**. Update the per-tier counts/runtimes table. Record any stale-anchor corrections and any reclass entries beyond the spec's table.

- [ ] **Step 2: No live `gpu_local`/Pascal as a tier.**

Run: `grep -n "gpu_local\|Pascal\|sm_61" docs/testing/gpu-test-policy.md`
Expected: only superseded-history mentions, no live tier. **[verify on CPU/CI]**

- [ ] **Step 3: Markdown-lint (CI's exact invocation — discover from the workflow).**

```bash
# CI's exact invocation (from .github/workflows/ci.yml:98-99; config .config/markdownlint-cli2.jsonc):
npx --yes markdownlint-cli2 --config .config/markdownlint-cli2.jsonc "**/*.md" "#node_modules"
# If no system node: use uvx + nodejs-bin to supply node (per the project's markdown-lint memo):
#   uvx --from markdownlint-cli2 --with nodejs-bin markdownlint-cli2 --config .config/markdownlint-cli2.jsonc "**/*.md" "#node_modules"
```
Expected: clean. **[verify on CPU/CI]**

- [ ] **Step 4: Commit.**

```bash
git add docs/testing/gpu-test-policy.md
git commit -m "docs(testing): rewrite GPU test policy to capability taxonomy (T4 floor)"
```

---

### Task E5: File the gpu_xl issue (R31)

**Files:** none (GitHub)

- [ ] **Step 1: Ensure the `testing` label exists.**

```bash
gh label list | grep -i testing || gh label create testing --description "Testing / CI" --color "0e8a16"
```

- [ ] **Step 2: Create the issue.**

```bash
gh issue create --assignee @me --label testing \
  --title "gpu_xl tier: GPU tests requiring > 16 GB VRAM" \
  --body "Holds GPU tests requiring >16 GB VRAM. Cross-refs #125 (cloud auto-provision, OPEN). Will hold any #83 all-scope overflow (measured peak: <record from Phase C C5 if branch (b)>) + future >16 GB tests."
```

- [ ] **Step 3: If #83 took branch (b),** put the measured all-scope LoRA peak number in this issue body. Record the new issue number for the PR body.

---

### Task E6: Resolve #193 `# tbd:` in defaults-provenance (R30 for #193, partial)

**Files:**
- Modify: `docs/defaults-provenance.md`

- [ ] **Step 1: Re-locate the `# tbd: #193` per-step figure** in the "Reference Training Profile".

```bash
grep -n "tbd: #193\|per-step\|Reference Training Profile" docs/defaults-provenance.md
```

- [ ] **Step 2: Replace with the 5070 Ti per-step figure** (measured in-session; GPU + date + command cited). **Leave the T4 sample slot pending the user's Colab run** (note it as "T4 sample: pending user Colab confirmation" — resolved in Task E8). **[verify on 5070 Ti for the 5070 Ti figure]**

- [ ] **Step 3: Markdown-lint + commit.**

```bash
# npx --yes markdownlint-cli2 --config .config/markdownlint-cli2.jsonc "docs/defaults-provenance.md"  (CI ci.yml:98-99)
git add docs/defaults-provenance.md
git commit -m "docs: resolve #193 per-step figure (5070 Ti); T4 sample pending Colab"
```

---

### Task E7: Open the PR (R24, R30, R32) — final lint gate FIRST

**Files:** none (GitHub)

- [ ] **Step 1: Run the FULL final verification gate (X3) — must be green before opening a ready PR.**

```bash
ruff check
uv run ruff format --check
uv run mypy src/custom_sam_peft
uv run pytest
shellcheck scripts/run_gpu_tests.sh scripts/check_gpu_evidence.sh
npx --yes markdownlint-cli2 --config .config/markdownlint-cli2.jsonc "**/*.md" "#node_modules"   # CI ci.yml:98-99
```
Expected: all green. **[verify on CPU/CI]**

- [ ] **Step 2: Open the PR** linking the spec + this plan, with the R24 checklist item and the R32 split:

```bash
gh pr create --assignee @me --label testing \
  --title "GPU test migration: re-architect testing around the RTX 5070 Ti" \
  --body "$(cat <<'EOF'
Closes #142, #195, #83 (5070 Ti-evidenced, landed). Closes #139, #193 pending one user Colab run.
Spec: docs/superpowers/specs/2026-05-31-gpu-test-migration-5070ti-design.md
Plan: docs/superpowers/plans/2026-05-31-gpu-test-migration-5070ti.md

## 5070 Ti-evidenced (landed)
- #142: 8GB-ceiling QLoRA smoke + predict-fits-8GB (peak <= ceiling/budget); min_gpu_qlora de-Pascal'd.
- #195: 25/50-step budgets confirmed/retuned vs 5070 Ti.
- #83: all-scope LoRA peak = <measured>; branch (a) DONE / (b) superseded -> gpu_xl issue #<n>.

## Colab-confirmation-pending (do NOT block merge)
- #139: minimal Colab T4 surface wired; bf16-coercion finding documented; faithful bf16 in gpu_bf16. Closes on user Colab confirmation.
- #193: 5070 Ti per-step figure resolved; T4 sample pending user Colab run.

## Checklist
- [ ] ran the light GPU subset on the 5070 Ti (see evidence artifact)

New issue filed: gpu_xl tier #<n> (xrefs #125).
EOF
)"
```

- [ ] **Step 3: Notify the user; end the session** per the orchestrator pipeline (the evidence run + #139/#193 closures happen in a later session, gated on user permission/confirmation).

---

### Task E8: Colab-gated closures (#139, #193) — LAST, user-triggered (R30, R32)

> **This task runs ONLY after the user confirms the one Colab run.** It does NOT block the PR or the 5070 Ti deliverables.

- [ ] **Step 1: User triggers the Colab `colab-min` run** (install + load + forward + one smoke), capturing the bf16-coercion observation and a cheap few-step T4 timing sample.

- [ ] **Step 2: Record the T4 timing sample** in `docs/defaults-provenance.md` (resolves the remaining #193 T4 slot from E6), cited (GPU + date + command). Commit + markdown-lint. **[verify on Colab T4 — user-triggered]**

- [ ] **Step 3: Confirm #139 deliverables** — Colab surface ran; policy doc records the coercion finding; `gpu_bf16` has the faithful test (A6).

- [ ] **Step 4: Close #139 and #193** (close-out handles `gh issue close` per the orchestrator pipeline) referencing the Colab evidence.

---

## Interface Contract — END OF PHASE E (project complete)

**EXPOSES:**

- `scripts/run_gpu_tests.sh` with selectors: (default) `gpu_t4 or gpu_bf16`, `t4`, `bf16`, `xl`, `colab-min`, `light`; usage line; non-zero on garbage; CPU autoskip clean.
- `scripts/check_gpu_evidence.sh` — always exits 0; warns on missing/stale; green only on current. Locked by `tests/unit/test_gpu_evidence_check.py`.
- A non-blocking `gpu-evidence` CI job (additive to `gpu-deselect-check`; not required).
- `notebooks/colab_gpu_tests.ipynb` — `colab-min` surface + #139/#193 capture.
- `docs/testing/gpu-test-policy.md` — full new-taxonomy policy (CPU-first gate, audit subsection, light subset, evidence gate).
- gpu_xl follow-up issue (xrefs #125).
- PR with the R32 "landed vs Colab-pending" split.

**CONSUMES:** Phases A (markers + forced-tier mechanism), B (probe/budget docs), C (#142/#83/#195 evidence), D (audit mapping).

---

## Definition of Done / Final verification checklist (X3)

Run from the worktree root. ALL must be green before the PR is marked ready:

- [ ] `ruff check` — clean
- [ ] `uv run ruff format --check` — clean (separate from `ruff check`; CI runs format-check)
- [ ] `uv run mypy src/custom_sam_peft` — clean
- [ ] `uv run pytest` — full CPU suite green (collection clean after marker changes)
- [ ] `shellcheck scripts/run_gpu_tests.sh scripts/check_gpu_evidence.sh` — clean
- [ ] CI's exact `markdownlint-cli2` (discover from `.github/workflows/`) — clean on every touched `.md` (this plan, the spec, the policy doc, defaults-provenance, the gtx1080 banner)
- [ ] `grep -rn "gpu_local\|_TIER_ORDER\|gpu-pascal\|cu118" src/ tests/ scripts/ pyproject.toml .github/` — zero live hits (dated history excepted)
- [ ] Collection counts: `-m gpu_t4` → 33 (27 reclassed + 6 net-new across Phases B–C); `-m gpu_bf16` → new bf16 test(s); `-m gpu_xl` → 0
- [ ] 5070 Ti evidence artifact committed (light subset run) with: #142 peak, predict peak, #195 step confirmation, #83 measured all-scope peak + branch
- [ ] PR body separates "5070 Ti-evidenced (landed)" from "Colab-confirmation-pending (#139, #193)"
- [ ] gpu_xl issue filed (xrefs #125)
- [ ] #139/#193 closures held until the user confirms the Colab run

---

## Requirement → Phase/Task traceability

| Req | Phase/Task |
|-----|-----------|
| R1 (marker defs) | A1 |
| R2 (reclass mapping, 27 tests) | A2 |
| R3 (#142 smoke → gpu_t4, registration) | A1 (marker) + C1 (test) |
| R4 (gpu_bf16 faithful test) | A6 |
| R-counts (collection contract) | A2, A6, C1 |
| R5 (`_satisfied_tiers` set probe) | A3 (test), A4 (impl) |
| R6 (capability-subset skip predicate; delete `_TIER_ORDER`) | A5 |
| R7 (CC gate 6.0→7.5) | A3 (test), A4 (impl) |
| R8 (delete Pascal artifacts) | A7 |
| R9 (runtime coercion untouched) | A8 (guard) |
| R10 (#142 8GB-ceiling smoke + `QLORA_8GB_CEIL_GB` + de-Pascal yaml) | C1, C2 |
| R11 (predict-fits-8GB) | C3 |
| R12 (provenance rows) | B1 (`PREDICT_8GB_BUDGET_GB`), C1 (`QLORA_8GB_CEIL_GB`) |
| R13 (`PREDICT_8GB_BUDGET_GB`) | B1 |
| R14 (empirical model-ready probe) | B2 |
| R15 (warn-not-block) | B2 |
| R16 (hook location, CPU no-op, pure decision fn) | B1 (pure fn), B2 (hook) |
| R17 (GPU warning test) | B3 |
| R18 (upward hint untouched) | B3 (guard) |
| R19 (integration audit subsection) | D4 (record) + E4 (doc) |
| R20 (3 bounded CPU regression tests) | D1, D2, D3 |
| R21 (defer large areas to issue) | D4 |
| R22 (CPU-first review gate) | E4 (doc) |
| R23 (non-blocking evidence CI check) | E2 |
| R24 (PR-checklist tracking, approval-gated merge) | E7 |
| R25 (define light subset) | E1 (selector) + E4 (doc) |
| R26 (primary env = 5070 Ti local) | E1 |
| R27 (runner selector rewrite) | E1 |
| R28 (minimal Colab surface) | E3 |
| R29 (policy-doc rewrite) | E4 (final); A (doc-stub references) |
| R30 (#142/#139/#193/#195/#83 closures in PR body) | C1/C3 (#142), E3/E8 (#139), E6/E8 (#193), C4 (#195), C5 (#83), E7 (PR body) |
| R31 (file gpu_xl issue) | E5 |
| R32 (manual dependency + sequencing) | E7 (PR split), E8 (Colab closures last) |
| R33 (provable non-blocking + approval-gated merge) | E2 |
| X1 (cite-or-tbd) | B1, C1 (each new constant), + Definition of Done |
| X2 (blast-radius discipline) | A8 |
| X3 (lint gates) | E7 + Definition of Done |

---

## Self-review notes

- **Coverage:** every R1–R33 and X1–X3 maps to at least one task (table above). No requirement is unplaced.
- **Phasing rationale (vs spec §12):** follows the spec's A–E grouping exactly. Phases B and D are parallelizable with A (no shared source); both B and C/E touch `docs/defaults-provenance.md`, so the orchestrator serializes commits to that file. Phase E is last (it documents/wires everything and contains the Colab-gated closures).
- **Manual dependency (R32):** #139 and #193's final closures live in Task E8, explicitly gated on the user's one Colab run, after the PR is opened (E7) — so they never block the PR or the 5070 Ti deliverables.
- **GPU vs CPU:** real-GPU assertions (A6, B3, C1/C3/C4/C5 runs) are tagged **[verify on 5070 Ti]** and authored to auto-skip green on CPU; everything else is **[verify on CPU/CI]**. The Colab surface is **[verify on Colab T4 — user-triggered]**.
- **Eager-import caution:** symbol renames (`_current_tier`→`_satisfied_tiers`, deleting `_TIER_ORDER`) carry explicit `import custom_sam_peft` / ruff sanity steps (A4 step 5, A5 step 4).
- **Anchor grounding:** line numbers are deliberately NOT used as edit targets; every edit task re-locates its symbol with grep (anchor-grounding pre-step) and notes any stale anchor in the policy doc.
