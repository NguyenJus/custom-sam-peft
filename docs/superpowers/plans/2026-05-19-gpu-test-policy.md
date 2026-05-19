# spec/gpu-test-policy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement [`spec/gpu-test-policy`](../specs/2026-05-19-gpu-test-policy-design.md) — split the 11 GPU-gated tests into a cheap `gpu_inspection` tier (9 tests in `tests/integration/`) and an expensive `gpu` release tier (2 tests in `tests/gpu/`), register the new marker, rewrite `scripts/run_gpu_tests.sh` to take a `inspection | release | all` argument, and publish the policy at `docs/testing/gpu-test-policy.md`.

**Architecture:** Pure docs + test-metadata + shell-script diff — no `src/esam3/` change, no CI workflow change, no new dependencies. The new `gpu_inspection` marker is registered first (step A) because `--strict-markers` is globally enabled and any test referencing the marker fails collection otherwise. Three integration test files then opt into the new tier at module level (steps B1–B3). The runner script is rewritten to a tier-driven dispatcher (step C). The policy doc is the largest single chunk — its 11-row inventory table requires reading each GPU-gated test once (step D). Final verification (step E) reuses the spec §7 exit-criteria commands verbatim.

**Tech Stack:** Python 3.12, pytest (markers + `addinivalue_line`), bash (strict mode), Markdown (markdownlint default rules minus `MD013`). No new library code, no new dependency.

---

## File Map

**New files:**

```
docs/testing/
  gpu-test-policy.md                       # six numbered sections per spec §5; 250–350 lines
```

**Modified files:**

```
scripts/
  run_gpu_tests.sh                         # rewritten per spec §4.1: accept inspection|release|all

tests/
  conftest.py                              # register `gpu_inspection` marker
  integration/
    test_load_sam31_real.py                # per-test decorators → module-level pytestmark
    test_peft_lora_real.py                 # append gpu_inspection to existing pytestmark
    test_peft_qlora_real.py                # append gpu_inspection to existing pytestmark
```

No file under `src/esam3/` is modified. No CI workflow file is modified. No new YAML, no new Python module, no new dependency.

---

## Pre-flight checks

- [ ] **Step 0a: Confirm worktree and branch**

```bash
pwd && git rev-parse --abbrev-ref HEAD
```
Expected: `/home/justin/projects/Efficient-SAM3-Finetuning/.worktrees/spec-gpu-test-policy` and `spec/gpu-test-policy`.

- [ ] **Step 0b: Confirm working tree clean**

```bash
git status
```
Expected: `nothing to commit, working tree clean` (the spec is already committed in an earlier brainstormer-planner commit, and so is this plan once it lands).

- [ ] **Step 0c: Confirm baseline CI is green on this branch**

```bash
gh run list --branch spec/gpu-test-policy --workflow CI --limit 3
```
Expected: most recent run is `completed` / `success`. If not, halt and investigate — this plan should land on a green baseline so any new red is attributable to our changes.

- [ ] **Step 0d: Confirm the eleven GPU-gated tests collect cleanly today**

```bash
uv run pytest tests/gpu tests/integration --collect-only -q 2>&1 | tail -n 20
```
Expected: all tests collect (most will skip at runtime on CPU via the autoskip in `tests/conftest.py:44-53`). No collection errors. Record the count — both `tests/gpu/` files plus the three `*_real.py` files should appear.

---

## Task A: Register the `gpu_inspection` marker

**Files:**
- Modify: `tests/conftest.py:18-31`

The marker MUST be registered before any test file references it, otherwise `--strict-markers` (set in `pyproject.toml` `addopts`) fails collection with `'gpu_inspection' not found in markers configuration option`. This task lands first so the test-file edits in Task B do not break collection.

- [ ] **Step A1: Edit `tests/conftest.py` to register the marker**

Open `tests/conftest.py` and find the `pytest_configure` function (lines 18–31). Append one more `addinivalue_line` call so the function reads:

```python
def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "requires_checkpoint: skip unless models/sam3.1/sam3.1_multiplex.pt exists",
    )
    config.addinivalue_line(
        "markers",
        "requires_compatible_gpu: skip unless a CUDA device with compute capability "
        ">= 7.5 is available",
    )
    config.addinivalue_line(
        "markers",
        "requires_bnb: skip unless bitsandbytes is importable",
    )
    config.addinivalue_line(
        "markers",
        "gpu_inspection: cheap GPU-gated structural/forward tests (Tier 1); "
        "see docs/testing/gpu-test-policy.md",
    )
```

No other change to this file.

- [ ] **Step A2: Confirm the marker is registered**

```bash
uv run pytest --markers | grep gpu_inspection
```
Expected: a single line `@pytest.mark.gpu_inspection: cheap GPU-gated structural/forward tests (Tier 1); see docs/testing/gpu-test-policy.md`.

- [ ] **Step A3: Confirm collection is still clean (no test uses the marker yet)**

```bash
uv run pytest --collect-only -q 2>&1 | tail -n 5
```
Expected: same collection summary as Step 0d. No marker warnings, no collection errors.

- [ ] **Step A4: Commit**

```bash
git add tests/conftest.py
git commit -m "test(markers): register gpu_inspection pytest marker"
```

**Done when:** `pytest --markers` lists `gpu_inspection`; `pytest --collect-only` still exits clean; one commit added.

---

## Task B1: Convert `test_load_sam31_real.py` to module-level `pytestmark`

**Files:**
- Modify: `tests/integration/test_load_sam31_real.py:18-27`

Today this file uses per-test decorators on each of the two test functions (`:18-19` and `:26-27`). The spec §6.2 requires converting to a module-level `pytestmark` list — matching the convention already in the other two `*_real.py` files — and adding `pytest.mark.gpu_inspection` to the list. The four per-test decorator lines are removed.

- [ ] **Step B1a: Edit `tests/integration/test_load_sam31_real.py`**

Replace the test-function decorator pairs. Today:

```python
@pytest.mark.requires_checkpoint
@pytest.mark.requires_compatible_gpu
def test_load_sam31_returns_wrapper() -> None:
    cfg = ModelConfig(device="cuda", gradient_checkpointing=False, dtype="bfloat16")
    wrapper = load_sam31(cfg)
    assert isinstance(wrapper, Sam3Wrapper)


@pytest.mark.requires_checkpoint
@pytest.mark.requires_compatible_gpu
def test_load_sam31_forward_to_canonical() -> None:
```

Target — drop both pairs of decorators and add a module-level `pytestmark` block after the imports:

```python
pytestmark = [
    pytest.mark.requires_checkpoint,
    pytest.mark.requires_compatible_gpu,
    pytest.mark.gpu_inspection,
]


def test_load_sam31_returns_wrapper() -> None:
    cfg = ModelConfig(device="cuda", gradient_checkpointing=False, dtype="bfloat16")
    wrapper = load_sam31(cfg)
    assert isinstance(wrapper, Sam3Wrapper)


def test_load_sam31_forward_to_canonical() -> None:
```

The two test bodies are otherwise unchanged. The file-level docstring (lines 1–6) and imports (lines 7–15) are untouched.

- [ ] **Step B1b: Confirm collection and skip behavior**

```bash
uv run pytest tests/integration/test_load_sam31_real.py --collect-only -q
```
Expected: `2 tests collected`. No marker warnings, no collection errors.

```bash
uv run pytest tests/integration/test_load_sam31_real.py -v
```
Expected: `2 skipped` (autoskip from `requires_compatible_gpu` / `requires_checkpoint` in `tests/conftest.py:44-53`). No failures.

- [ ] **Step B1c: ruff + mypy on the edited file**

```bash
uv run ruff check tests/integration/test_load_sam31_real.py && uv run mypy tests/integration/test_load_sam31_real.py
```
Expected: both clean.

- [ ] **Step B1d: Commit**

```bash
git add tests/integration/test_load_sam31_real.py
git commit -m "test(gpu_inspection): convert test_load_sam31_real to module-level pytestmark + add marker"
```

**Done when:** the file has no per-test `requires_checkpoint`/`requires_compatible_gpu` decorators, the module-level `pytestmark` list contains all three markers, collection still produces 2 tests, both tests skip cleanly on CPU.

---

## Task B2: Append `gpu_inspection` to `test_peft_lora_real.py`

**Files:**
- Modify: `tests/integration/test_peft_lora_real.py:19-22`

The existing `pytestmark` list at lines 19–22 has two entries; append the new marker. No other change.

- [ ] **Step B2a: Edit `tests/integration/test_peft_lora_real.py`**

Replace:

```python
pytestmark = [
    pytest.mark.requires_checkpoint,
    pytest.mark.requires_compatible_gpu,
]
```

with:

```python
pytestmark = [
    pytest.mark.requires_checkpoint,
    pytest.mark.requires_compatible_gpu,
    pytest.mark.gpu_inspection,
]
```

The three test functions (`test_apply_lora_on_real_sam31_under_trainable_budget`, `test_save_load_roundtrip_on_real_sam31`, `test_merge_lora_on_real_sam31`) are unchanged.

- [ ] **Step B2b: Confirm collection and skip behavior**

```bash
uv run pytest tests/integration/test_peft_lora_real.py --collect-only -q
```
Expected: `3 tests collected`. No marker warnings.

```bash
uv run pytest tests/integration/test_peft_lora_real.py -v
```
Expected: `3 skipped`. No failures.

- [ ] **Step B2c: ruff + mypy on the edited file**

```bash
uv run ruff check tests/integration/test_peft_lora_real.py && uv run mypy tests/integration/test_peft_lora_real.py
```
Expected: both clean.

- [ ] **Step B2d: Commit**

```bash
git add tests/integration/test_peft_lora_real.py
git commit -m "test(gpu_inspection): tag test_peft_lora_real with gpu_inspection marker"
```

**Done when:** the `pytestmark` list contains `pytest.mark.gpu_inspection`, the three tests still collect, all three skip cleanly on CPU.

---

## Task B3: Append `gpu_inspection` to `test_peft_qlora_real.py`

**Files:**
- Modify: `tests/integration/test_peft_qlora_real.py:30-33`

Same shape as Task B2. The per-test `@pytest.mark.skipif(not _bnb_available(), reason="bitsandbytes not installed")` decorators on each of the four test functions are NOT touched.

- [ ] **Step B3a: Edit `tests/integration/test_peft_qlora_real.py`**

Replace:

```python
pytestmark = [
    pytest.mark.requires_checkpoint,
    pytest.mark.requires_compatible_gpu,
]
```

with:

```python
pytestmark = [
    pytest.mark.requires_checkpoint,
    pytest.mark.requires_compatible_gpu,
    pytest.mark.gpu_inspection,
]
```

The module-level `_bnb_available()` and `_has_linear4bit_modules()` helpers, the per-test `@pytest.mark.skipif(not _bnb_available(), ...)` decorators, and the four test bodies are all unchanged.

- [ ] **Step B3b: Confirm collection and skip behavior**

```bash
uv run pytest tests/integration/test_peft_qlora_real.py --collect-only -q
```
Expected: `4 tests collected`. No marker warnings.

```bash
uv run pytest tests/integration/test_peft_qlora_real.py -v
```
Expected: `4 skipped`. No failures.

- [ ] **Step B3c: ruff + mypy on the edited file**

```bash
uv run ruff check tests/integration/test_peft_qlora_real.py && uv run mypy tests/integration/test_peft_qlora_real.py
```
Expected: both clean.

- [ ] **Step B3d: Commit**

```bash
git add tests/integration/test_peft_qlora_real.py
git commit -m "test(gpu_inspection): tag test_peft_qlora_real with gpu_inspection marker"
```

**Done when:** the `pytestmark` list contains `pytest.mark.gpu_inspection`, the per-test `skipif(_bnb_available)` decorators are still present, the four tests still collect, all four skip cleanly on CPU.

---

## Task B-sweep: Verify total `gpu_inspection` collection across all three files

After Tasks B1, B2, B3 are all committed, confirm the spec §7 collect-only counts.

- [ ] **Step B-sweep1: Marker `gpu_inspection` collects exactly 9 tests**

```bash
uv run pytest --collect-only -m gpu_inspection -q 2>&1 | tail -n 3
```
Expected: the collection summary line reports `9 tests collected` (or `9/N tests collected` if a deselect count is shown). On a CPU box those 9 also report as deselected/skipped at runtime via the existing autoskip — but here we only care about the collection count.

- [ ] **Step B-sweep2: Marker `gpu` is unchanged (still 2)**

```bash
uv run pytest --collect-only -m gpu -q 2>&1 | tail -n 3
```
Expected: `2 tests collected` — `tests/gpu/test_real_train_overfits.py::test_overfits_in_50_steps` and `tests/gpu/test_real_train_qlora.py::test_qlora_overfits_in_50_steps`.

- [ ] **Step B-sweep3: Union `gpu or gpu_inspection` collects 11**

```bash
uv run pytest --collect-only -m "gpu or gpu_inspection" -q 2>&1 | tail -n 3
```
Expected: `11 tests collected`.

**Done when:** the three collect-only counts are exactly 9 / 2 / 11. If any count is wrong, return to the corresponding Task B step and audit the marker placement; do NOT proceed to Task C until the counts match.

---

## Task C: Rewrite `scripts/run_gpu_tests.sh`

**Files:**
- Rewrite: `scripts/run_gpu_tests.sh`

Spec §4.1. The script becomes a tier dispatcher; the no-arg default is `all` (runs all 11 tests, fixing the latent bug that today's script silently skips the 2 `tests/gpu/` tests). Preserve `set -euo pipefail`, the `"${PYTHON:-python}" -m pytest` invocation, `--no-cov`, and the `-v --tb=short` flags. The existing header comment about bare-`pytest`-on-PATH is preserved (re-stated below).

- [ ] **Step C1: Replace the entire script with the rewritten version**

Replace `scripts/run_gpu_tests.sh` with:

```bash
#!/usr/bin/env bash
# Canonical pytest invocation for any GPU environment.
# Used by notebooks/colab_gpu_tests.ipynb and runnable directly on any
# Turing+ machine with bitsandbytes installed.
#
# Usage:
#   scripts/run_gpu_tests.sh [inspection|release|all]
#
# Tiers (see docs/testing/gpu-test-policy.md):
#   inspection — cheap structural/forward tests in tests/integration/ (9 tests)
#   release    — expensive training-smoke tests in tests/gpu/ (2 tests)
#   all        — both tiers; this is the default (11 tests)
set -euo pipefail
TIER="${1:-all}"

case "$TIER" in
  inspection) MARKER_EXPR="gpu_inspection" ; PATHS="tests/integration/" ;;
  release)    MARKER_EXPR="gpu"            ; PATHS="tests/gpu/" ;;
  all)        MARKER_EXPR="gpu or gpu_inspection" ; PATHS="tests/gpu/ tests/integration/" ;;
  *) echo "usage: $0 [inspection|release|all]" >&2; exit 2 ;;
esac

# Use `python -m pytest` (not bare `pytest`) so the test runner picks the
# same interpreter that `pip install -e .` populated. Bare `pytest` on
# PATH can resolve to a different Python (common in Colab) and trigger
# `ModuleNotFoundError: No module named 'esam3'`.
"${PYTHON:-python}" -m pytest -v --tb=short \
  -m "$MARKER_EXPR" --no-cov $PATHS
```

Keep the file executable: `chmod +x scripts/run_gpu_tests.sh` (likely already executable from before; verify with `ls -l scripts/run_gpu_tests.sh`).

- [ ] **Step C2: shellcheck on the rewritten script**

```bash
shellcheck scripts/run_gpu_tests.sh
```
Expected: clean exit, no output. (The `$PATHS` unquoted expansion is intentional — `PATHS` is a space-separated list and must word-split. If shellcheck flags `SC2086`, leave the code as-is; the spec mandates this shape and the words are tightly controlled.)

> **If `SC2086` fires:** add `# shellcheck disable=SC2086` immediately above the `"${PYTHON:-python}" -m pytest` line with a one-line rationale comment (`# PATHS is a controlled space-separated list of paths; intentional word split.`). Re-run shellcheck and confirm clean.

- [ ] **Step C3: Each tier dispatches cleanly on a CPU box**

```bash
bash scripts/run_gpu_tests.sh inspection
```
Expected: exit 0. Pytest runs against `tests/integration/` with `-m gpu_inspection`, collects 9 tests, autoskips all 9 (no CUDA + no checkpoint on this box), prints `9 skipped`.

```bash
bash scripts/run_gpu_tests.sh release
```
Expected: exit 0. Pytest runs against `tests/gpu/` with `-m gpu`, collects 2 tests, autoskips both, prints `2 skipped`.

```bash
bash scripts/run_gpu_tests.sh all
```
Expected: exit 0. Pytest runs against both paths with `-m "gpu or gpu_inspection"`, collects 11 tests, autoskips all 11, prints `11 skipped`.

```bash
bash scripts/run_gpu_tests.sh
```
Expected: identical to `bash scripts/run_gpu_tests.sh all` — exit 0, 11 skipped.

- [ ] **Step C4: Unknown tier prints usage and exits 2**

```bash
bash scripts/run_gpu_tests.sh garbage; echo "exit=$?"
```
Expected: `usage: scripts/run_gpu_tests.sh [inspection|release|all]` on stderr, then `exit=2`.

- [ ] **Step C5: Commit**

```bash
git add scripts/run_gpu_tests.sh
git commit -m "scripts(gpu): rewrite run_gpu_tests.sh to accept inspection|release|all (default all)"
```

**Done when:** all four invocations (`inspection`, `release`, `all`, no-arg) exit 0 with the expected skip counts; `garbage` exits 2 with the usage line on stderr; shellcheck is clean.

---

## Task D: Write `docs/testing/gpu-test-policy.md`

**Files:**
- Create: `docs/testing/gpu-test-policy.md`

Spec §5. Six numbered sections, ~250–350 lines total. The implementer writes the doc; the spec specifies WHAT each section contains, not WORD-FOR-WORD what to write.

**Important:** the inventory table in §3 has 11 rows, one per GPU-gated test. The plan does NOT pre-fill those rows — the implementer fills them at this step by reading each of the 11 test files and applying the CPU-stub viability rubric from spec §5.3.1.

- [ ] **Step D1: Create the file and add the six section scaffold**

Create `docs/testing/gpu-test-policy.md` with this top-level shape (the section bodies are filled in below):

```markdown
# GPU Test Policy

<!-- Two-tier policy for GPU-gated tests. See spec/gpu-test-policy (#37). -->

## 1. Why this exists

…

## 2. Tier definitions

### Tier 1 — `gpu_inspection`

…

### Tier 2 — `gpu`

…

## 3. Inventory

| `file::test` | Tier | What it covers | Why GPU is required | CPU-stub viability |
| --- | --- | --- | --- | --- |
| … | … | … | … | … |

## 4. T4 validation policy

…

## 5. Data-size policy

…

## 6. Adding a new GPU test — criteria checklist

1. …
```

- [ ] **Step D2: Fill Section 1 — "Why this exists" (~3 sentences)**

Per spec §5.1. Name three cost axes explicitly: wall time, paid minutes (Colab free-tier consumed per PR; any future paid runner would amplify), and flake surface (driver mismatch, transient OOM, NaN under non-deterministic kernels). Link to issue [#37](https://github.com/NguyenJus/Efficient-SAM3-Finetuning/issues/37).

- [ ] **Step D3: Fill Section 2 — "Tier definitions" (two sub-sections + closing paragraph)**

Per spec §5.2. For each of the two sub-sections, name:

- **Marker** — `gpu_inspection` (Tier 1) or `gpu` (Tier 2).
- **Cadence guidance** — verbatim phrasing:
  - Tier 1: "run on Colab notebook for any PR touching GPU paths."
  - Tier 2: "run before a tagged release, or when training-loop / tracker / optimizer code changes land."
- **Runner invocation** — `bash scripts/run_gpu_tests.sh inspection` or `bash scripts/run_gpu_tests.sh release`.
- **What's in the tier** — one-line summary of the test file(s).

Close with a paragraph noting:
- No hosted GPU CI runner exists today; both tiers run via the same `notebooks/colab_gpu_tests.ipynb` notebook.
- The notebook's Cell 6 invokes `bash scripts/run_gpu_tests.sh` with no argument, which defaults to `all` — both tiers in one session.
- A contributor can override by editing Cell 6 to pass `inspection` or `release`, or by running the script locally on a Turing+ machine.
- The tier policy is trigger-agnostic.

- [ ] **Step D4: Fill Section 3 — "Inventory table" (11 rows + two §5.3.2 example rows preserved)**

Per spec §5.3. The table has 11 rows total, one per GPU-gated test. Columns:

| Column | Contents |
| --- | --- |
| `file::test` | e.g. `tests/integration/test_peft_lora_real.py::test_apply_lora_on_real_sam31_under_trainable_budget` |
| Tier | `inspection` or `release`. |
| What it covers | ≤ 15 words. |
| Why GPU is required | Free-text, derived from the file-level docstring + imports. |
| CPU-stub viability | One of three buckets per spec §5.3.1. |

**The 11 test rows the implementer fills:**

```
tests/integration/test_load_sam31_real.py::test_load_sam31_returns_wrapper             # inspection
tests/integration/test_load_sam31_real.py::test_load_sam31_forward_to_canonical        # inspection
tests/integration/test_peft_lora_real.py::test_apply_lora_on_real_sam31_under_trainable_budget   # inspection
tests/integration/test_peft_lora_real.py::test_save_load_roundtrip_on_real_sam31       # inspection
tests/integration/test_peft_lora_real.py::test_merge_lora_on_real_sam31                # inspection
tests/integration/test_peft_qlora_real.py::test_apply_qlora_swaps_every_linear_and_attaches_lora   # inspection
tests/integration/test_peft_qlora_real.py::test_save_qlora_writes_adapter_and_metadata  # inspection
tests/integration/test_peft_qlora_real.py::test_save_load_qlora_roundtrip               # inspection
tests/integration/test_peft_qlora_real.py::test_merge_lora_unloads_qlora_wrapper        # inspection
tests/gpu/test_real_train_overfits.py::test_overfits_in_50_steps                        # release
tests/gpu/test_real_train_qlora.py::test_qlora_overfits_in_50_steps                     # release
```

**CPU-stub viability rubric (verbatim from spec §5.3.1):**

- **`none — needs real SAM3.1 weights`** — the test asserts on SAM3.1-specific named parameters (e.g. `"vision_backbone" in n`, `"transformer.decoder" in n`), or on a forward output shape that only the real model produces, or on bitsandbytes quant kernels (`Linear4bit`). CPU stub cannot replicate these.
- **`partial — could cover X but loses Y`** — a structural assertion (e.g. "trainable ratio < 5%", "Linear modules swapped") can be covered by a CPU stub that mimics the same module-tree shape, but the test would lose its end-to-end signal. Name X and Y in the cell.
- **`viable — see follow-up #N`** — the test makes assertions a `TinySam3Stub` (or equivalent CPU mock) can satisfy with no real-model dependency. Implementer files a GitHub issue (`gh issue create --assignee @me`) and puts the issue number in the cell. Title pattern: `CPU-stub replacement for <file>::<test>`.

For each row: open the test, read its docstring + assertions, classify it, fill the cell. If the verdict is `viable`, file the follow-up issue *before* writing the cell so the cell can name the issue number.

The two §5.3.2 example rows are part of the final table, not placeholders — include them verbatim:

```markdown
| `tests/integration/test_peft_lora_real.py::test_apply_lora_on_real_sam31_under_trainable_budget` | inspection | LoRA trainable-ratio < 5% and presence of `vision_backbone` + `transformer.decoder` LoRA params after `apply_lora` on real SAM3.1. | Asserts on SAM3.1-specific named parameters; CPU stub does not have these modules. | none — needs real SAM3.1 weights |
| `tests/gpu/test_real_train_overfits.py::test_overfits_in_50_steps` | release | 50-step LoRA overfit on tiny_coco via `run_training(gpu_smoke_lora.yaml)`; asserts loss drops ≥ 30%, peak VRAM ≤ 14 GB, all logged scalars finite. | End-to-end real training: real SAM3.1 weights, real PEFT, real CUDA kernels, real optimizer step. | none — needs real SAM3.1 weights |
```

- [ ] **Step D5: Fill Section 4 — "T4 validation policy" (three short paragraphs)**

Per spec §5.4. Three paragraphs, in order:

1. T4 (Tesla T4, 16 GB VRAM, Turing) is the only validated VRAM tier. The Colab notebook's prereqs cell already pins T4 as the minimum runtime; VRAM ceilings in the smoke YAMLs are pinned to T4: 14 GB LoRA (`configs/examples/gpu_smoke_lora.yaml`, asserted at `tests/gpu/test_real_train_overfits.py:32`) and 10 GB QLoRA (`configs/examples/gpu_smoke_qlora.yaml`, asserted at `tests/gpu/test_real_train_qlora.py:34`).
2. Larger GPUs (A100, L4, H100) MAY run the suite. A green run on a larger GPU does NOT substitute for a green T4 run for release validation — the VRAM headroom asymmetry means an A100 can mask a T4 OOM.
3. The VRAM ceilings MUST NOT be raised to accommodate larger GPUs. If a future training-loop change pushes T4 above the ceiling, the fix is to reduce VRAM usage (gradient checkpointing knobs, micro-batch shape, etc.), not to raise the ceiling.

- [ ] **Step D6: Fill Section 5 — "Data-size policy" (two short paragraphs)**

Per spec §5.5. Two paragraphs:

1. Smoke configs already use the `tests/fixtures/tiny_coco/` fixture (2 images) and 50 grad updates (epochs=25, batch_size=1, grad_accum_steps=1). This is the minimal end-to-end overfit shape. Confirmed against `configs/examples/gpu_smoke_lora.yaml` and `gpu_smoke_qlora.yaml`.
2. New GPU tests MUST reuse `tiny_coco` (or smaller). A test that needs more data has slipped into integration-suite territory and belongs out-of-tier or on CPU with stubs. Enforced by code review against the §6 checklist, not by tooling.

- [ ] **Step D7: Fill Section 6 — "Adding a new GPU test — criteria checklist" (numbered list of 5)**

Per spec §5.6. Render as a numbered ordered list (NOT bullets), so a reviewer pasting them into a PR comment gets actionable numbered items. Verbatim:

1. Does the test exercise behavior unique to real SAM3.1 weights (matched named params, real forward shapes) or quant kernels (`Linear4bit`, 4-bit base + LoRA delta paths)?
2. Is the CPU+stub variant insufficient, AND is that explicitly documented in the test docstring?
3. Is the data fixture `tiny_coco` (or smaller)?
4. Does the test fit within the assigned tier's cost envelope? **Inspection:** ~load time per test (single forward pass at most). **Release:** ≤ 2 minutes on T4.
5. Is the test tagged with exactly one of `gpu` or `gpu_inspection`?

Closing line: "If any answer is no, the test belongs out-of-tier (move marker) or needs redesign (move work to CPU with a stub; file an issue if the work itself is worth keeping)."

- [ ] **Step D8: Apply Section conventions (spec §5.7)**

Sweep the doc once for consistency:

- Tier 1 = `gpu_inspection` everywhere. Tier 2 = `gpu`. No alternate spellings.
- "Tiny COCO" or "the tiny_coco fixture" — match the source-tree spelling `tiny_coco`.
- "T4" used without expansion after the first mention in Section 4.
- No per-test runtime numbers anywhere.
- Section 6 renders the five questions as a numbered ordered list.

- [ ] **Step D9: Verify line count is in range**

```bash
wc -l docs/testing/gpu-test-policy.md
```
Expected: between 250 and 350 lines (inclusive). If short, sections 1–6 likely lost detail compared to the spec's outline; expand. If over, prune commentary — the inventory table accounts for ~15 lines (header + separator + 11 rows + spacing), the rest is prose.

- [ ] **Step D10: Markdown lint clean**

```bash
npx --yes markdownlint-cli2 docs/testing/gpu-test-policy.md
```
Expected: clean exit. The repo-root `.markdownlint.json` rules apply (per the ci-hardening spec §5.5): the doc is a **new live doc** under `docs/testing/`, NOT under `docs/superpowers/`, so the `docs/superpowers/.markdownlint.json` directory-scoped relaxation does NOT apply — all default markdownlint rules except `MD013` (line length) are in force. Common fixes if findings appear:

| Rule | Fix |
| --- | --- |
| `MD022` headings need blank lines around | Add blank line before/after each `#` heading. |
| `MD031` fenced code blocks need blank lines around | Same, around each ```` ``` ```` fence. |
| `MD034` bare URL | Wrap as `<https://...>` or `[text](url)`. |
| `MD040` fenced code block language | Add a language tag (e.g. `` ```bash ``). |
| `MD041` first line should be a top-level heading | The file already starts with `# GPU Test Policy`; should pass. |

- [ ] **Step D11: Self-review the inventory table against the rubric**

```bash
grep -c '^|' docs/testing/gpu-test-policy.md
```
Expected: at least 13 — table header row + separator row + 11 data rows. If fewer, a row is missing.

For each row, verify the CPU-stub viability cell is one of the three rubric buckets and is non-empty. If any cell is empty or reads "TBD"/"TODO", the implementer did not finish the rubric pass — go back to Step D4.

- [ ] **Step D12: Commit**

```bash
git add docs/testing/gpu-test-policy.md
git commit -m "docs(testing): publish GPU test policy (two-tier, T4-pinned, tiny_coco floor)"
```

**Done when:** the file exists with six numbered sections, 11 inventory rows with non-empty CPU-stub viability cells, line count in [250, 350], `markdownlint` clean.

---

## Task E: Final verification sweep (spec §7 exit criteria)

The exit gate is CPU-only. Real-GPU runs of either tier are post-merge follow-up per spec §7 ("explicitly NOT a gate"). Each command below maps to a spec §7 bullet.

- [ ] **Step E1: `ruff check` clean**

```bash
uv run ruff check
```
Expected: `All checks passed!`. (Maps to spec §7 — full lint/test gate.)

- [ ] **Step E2: `ruff format --check` clean**

```bash
uv run ruff format --check
```
Expected: clean. (Maps to spec §7 — full lint/test gate.)

- [ ] **Step E3: `mypy src/esam3` clean**

```bash
uv run mypy src/esam3
```
Expected: `Success: no issues found`. (Maps to spec §7 — full lint/test gate. Note: this plan does not touch `src/esam3/`, so mypy should be unchanged from baseline.)

- [ ] **Step E4: Full `pytest` green**

```bash
uv run pytest
```
Expected: full pass on CPU. The 11 GPU-gated tests all autoskip via `tests/conftest.py:44-53`. (Maps to spec §7 — full lint/test gate.)

- [ ] **Step E5: `gpu_inspection` collects exactly 9**

```bash
uv run pytest --collect-only -m gpu_inspection
```
Expected: `9 tests collected`. (Maps to spec §7 bullet 2.) On a CPU box those 9 then autoskip at collection time via the existing `requires_compatible_gpu` autoskip.

- [ ] **Step E6: `gpu` collects exactly 2**

```bash
uv run pytest --collect-only -m gpu
```
Expected: `2 tests collected` — `test_overfits_in_50_steps` and `test_qlora_overfits_in_50_steps`. (Maps to spec §7 bullet 3.)

- [ ] **Step E7: `gpu or gpu_inspection` collects exactly 11**

```bash
uv run pytest --collect-only -m "gpu or gpu_inspection"
```
Expected: `11 tests collected`. (Maps to spec §7 bullet 4.)

- [ ] **Step E8: All four runner-script invocations exit clean on CPU**

```bash
bash scripts/run_gpu_tests.sh                  ; echo "no-arg exit=$?"
bash scripts/run_gpu_tests.sh inspection       ; echo "inspection exit=$?"
bash scripts/run_gpu_tests.sh release          ; echo "release exit=$?"
bash scripts/run_gpu_tests.sh all              ; echo "all exit=$?"
```
Expected: each prints `exit=0`. Pytest treats all-skipped as success. No collection error, no marker warning. (Maps to spec §7 bullet 5.)

- [ ] **Step E9: Unknown tier exits 2 with the usage line on stderr**

```bash
bash scripts/run_gpu_tests.sh garbage 2>/tmp/usage.txt; echo "exit=$?"
cat /tmp/usage.txt
```
Expected: `exit=2` and the file contains `usage: scripts/run_gpu_tests.sh [inspection|release|all]`. (Maps to spec §7 bullet 6.)

- [ ] **Step E10: `markdownlint` clean on the policy doc**

```bash
npx --yes markdownlint-cli2 docs/testing/gpu-test-policy.md
```
Expected: clean exit. (Maps to spec §7 bullet 7.)

- [ ] **Step E11: `shellcheck` clean on the runner script**

```bash
shellcheck scripts/run_gpu_tests.sh
```
Expected: clean exit. (Maps to spec §7 bullet 8.)

- [ ] **Step E12: Push and watch CI**

```bash
git push
gh run watch --exit-status
```
Expected: all CI jobs pass (`test`, `lock-check`, `lint-hygiene`, `pip-audit`, `gitleaks`). Cross-reference: `gh pr checks --watch`.

**Done when:** every step E1–E12 returns the expected output. If any fails, return to the corresponding task (A/B/C/D) and audit the change.

---

## Drive-by decisions (optional, at implementer's discretion)

These are explicit decision points the spec leaves open. Implementer makes a yes/no call and records it in the PR description.

- [ ] **DB1: Mirror `gpu_inspection` registration in `pyproject.toml`?**

Spec §6.1 closing paragraph: today's repo registers some markers in `pyproject.toml` (`integration`, `gpu`, `requires_checkpoint`, `requires_compatible_gpu`) and others via `addinivalue_line` in `tests/conftest.py` (`requires_checkpoint`, `requires_compatible_gpu`, `requires_bnb`). The implementer MAY mirror `gpu_inspection` in `pyproject.toml`'s `[tool.pytest.ini_options].markers` list for consistency with `gpu`. **This is optional and not gated by exit criteria.**

Decision points:
- **Yes (mirror):** edit `pyproject.toml` to add `"gpu_inspection: cheap GPU-gated structural/forward tests (Tier 1); see docs/testing/gpu-test-policy.md"` alongside the other four entries. Single-line addition. Commit as `chore(pytest): mirror gpu_inspection marker in pyproject.toml`. Pros: consistency with `gpu`. Cons: two sources of truth for the same marker (low cost — `addinivalue_line` wins at collection time).
- **No (do not mirror):** leave `tests/conftest.py` as the sole source. Pros: one source of truth, matches `requires_bnb`. Cons: minor inconsistency with `gpu`.

Either decision is fine. Record the choice in the PR description. No further action required.

---

## Rollback

Each step is small enough that a single `git restore <file>` reverts it.

| Task | Rollback command | Notes |
| --- | --- | --- |
| Task A (marker register) | `git restore tests/conftest.py` | Reverts the `addinivalue_line` addition. Run before Task B or `pytest --collect-only` will fail under `--strict-markers` because the test files reference an unregistered marker. |
| Task B1 (test_load_sam31_real) | `git restore tests/integration/test_load_sam31_real.py` | Restores the per-test decorators. |
| Task B2 (test_peft_lora_real) | `git restore tests/integration/test_peft_lora_real.py` | Restores the two-entry `pytestmark`. |
| Task B3 (test_peft_qlora_real) | `git restore tests/integration/test_peft_qlora_real.py` | Restores the two-entry `pytestmark`. The per-test `skipif(_bnb_available)` decorators were untouched and so are unaffected by the restore. |
| Task C (runner script) | `git restore scripts/run_gpu_tests.sh` | Restores the no-arg, hard-coded invocation. |
| Task D (policy doc) | `rm docs/testing/gpu-test-policy.md && rmdir docs/testing 2>/dev/null || true` | The directory is new; remove it if empty. |

If a rollback to an earlier task is needed mid-rollout, restore in reverse-dependency order: Task C and D are independent and can be rolled back individually; Task B depends on Task A (the marker registration) — roll back any B tasks **before** rolling back A. After rollback, re-run Step E5–E7 to confirm collection counts are sane.

For a full revert of the branch's changes (e.g. abandoning the PR):

```bash
git reset --hard origin/main
```

**Do not** run `git reset --hard` without explicit user confirmation — destructive.

---

## Spec coverage map

| Spec section | Task(s) |
| --- | --- |
| §1 Current State — `gpu_inspection` marker registration | A |
| §1 Current State — `tests/integration/test_load_sam31_real.py` convert + tag | B1 |
| §1 Current State — `tests/integration/test_peft_lora_real.py` tag | B2 |
| §1 Current State — `tests/integration/test_peft_qlora_real.py` tag | B3 |
| §1 Current State — `scripts/run_gpu_tests.sh` rewrite | C |
| §1 Current State — `docs/testing/gpu-test-policy.md` new file | D |
| §2 Goals — tier split into Tier 1 / Tier 2 | A, B1, B2, B3 |
| §2 Goals — fix latent bug in runner script | C |
| §2 Goals — T4 + tiny_coco policy doc + reviewer checklist | D5, D6, D7 |
| §3 File Map | matches Task A–D file lists |
| §4.1 Runner script shape (tier dispatcher, default `all`) | C |
| §4.2 Strict-markers compatibility (marker registered before collection) | A (lands before B) |
| §4.3 Tier → marker → path mapping table | C, verified by E5–E9 |
| §4.4 Colab notebook impact (no edit; default behavior changes) | C (script default), D3 (doc note) |
| §5.1 Section 1 "Why this exists" | D2 |
| §5.2 Section 2 "Tier definitions" | D3 |
| §5.3 Section 3 "Inventory table" | D4 |
| §5.3.1 CPU-stub viability rubric | D4 (instructions to implementer) |
| §5.3.2 Example rows preserved verbatim | D4 |
| §5.4 Section 4 "T4 validation policy" | D5 |
| §5.5 Section 5 "Data-size policy" | D6 |
| §5.6 Section 6 "Adding a new GPU test — criteria checklist" | D7 |
| §5.7 Section conventions | D8 |
| §6.1 `tests/conftest.py` marker registration | A |
| §6.1 Optional `pyproject.toml` mirror | DB1 |
| §6.2 `test_load_sam31_real.py` convert + add | B1 |
| §6.3 `test_peft_lora_real.py` append | B2 |
| §6.4 `test_peft_qlora_real.py` append | B3 |
| §6.5 What is NOT edited | enforced by plan scope (no tasks touch `tests/gpu/`, `tests/unit/`, `tests/cli/`, `src/`, CI workflows) |
| §7 Exit Criteria — code/doc bullets | enforced by Done-when lines on A–D and verified in E |
| §7 Exit Criteria — CPU tests green | E1–E4 |
| §7 Exit Criteria — collect-only counts (9/2/11) | E5, E6, E7 |
| §7 Exit Criteria — runner invocations | E8, E9 |
| §7 Exit Criteria — `markdownlint` clean on doc | E10 |
| §7 Exit Criteria — `shellcheck` clean on script | E11 |
| §8 Deferred items | not implemented (CPU-stub replacements filed as separate issues during D4 when verdict is `viable`) |
