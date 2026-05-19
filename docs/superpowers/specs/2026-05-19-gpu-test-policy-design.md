# spec/gpu-test-policy — Minimize the GPU-Gated Test Surface

**Status:** Draft (2026-05-19)
**Tracking issue:** #37
**Scope:** Audit the 12 GPU-gated tests across `tests/gpu/` and `tests/integration/`, tier them into `gpu_inspection` (cheap structural) and `gpu` (expensive training-smoke), publish a short policy doc at `docs/testing/gpu-test-policy.md`, register the new marker, and rewrite `scripts/run_gpu_tests.sh` to take a tier argument. Pure docs + test-metadata + shell-script diff; no `src/esam3/` change, no CI workflow change, no new dependencies.

---

## 1. Current State

| Surface | State today | This spec |
| --- | --- | --- |
| GPU-gated tests in `tests/gpu/` | 3 tests in `test_real_train_overfits.py`, `test_real_train_qlora.py`, and `test_run_end_to_end_gpu.py` (the last added by #46 after the original audit), all marked `@pytest.mark.gpu`, `@requires_compatible_gpu`, `@requires_checkpoint`. Added by #28 and #46. Expensive: minutes of real training each. | **Tier 2 — Release.** Selector unchanged (`-m gpu`). Files untouched. |
| GPU-gated tests in `tests/integration/` | 9 tests total: `test_load_sam31_real.py` (2 tests, per-test decorators), `test_peft_lora_real.py` (3 tests, module-level `pytestmark`), `test_peft_qlora_real.py` (4 tests, module-level `pytestmark` + per-test `skipif`). All carry `requires_compatible_gpu` + `requires_checkpoint`. None carry `gpu`. Cheap: load + structural inspection only, no training. | **Tier 1 — Inspection.** New selector `-m gpu_inspection` added at module level on all three files. `test_load_sam31_real.py` is converted from per-test decorators to module-level `pytestmark` as a drive-by. |
| `@pytest.mark.gpu_inspection` marker | Not registered. | Registered in `tests/conftest.py` alongside the existing three markers, with description pointing at the policy doc. |
| `@pytest.mark.gpu` marker | Registered in the `markers` list in `pyproject.toml` (line ~94), alongside `integration`, `requires_checkpoint`, and `requires_compatible_gpu`. The repo also enables `--strict-markers` globally via `addopts` (line ~100). The existing semantic is "release-tier real-GPU training smoke." | Semantic preserved — `gpu` continues to mean "release-tier training smoke." No retag of existing usage. The new `gpu_inspection` marker is registered in `tests/conftest.py` (see §6.1); whether to also add it to `pyproject.toml` is the implementer's call (see §6.1 update below). |
| `scripts/run_gpu_tests.sh` | Hard-coded to `pytest -m "requires_compatible_gpu and requires_checkpoint" --no-cov tests/integration/`. Silently skips the 3 `tests/gpu/` tests — latent bug surfaced by this audit. | **Rewritten** to accept `inspection | release | all` (default `all`). New default matches user expectation: "run everything GPU-gated." See §4. |
| `notebooks/colab_gpu_tests.ipynb` Cell 6 | Invokes `bash scripts/run_gpu_tests.sh` with no argument. | Unchanged — the no-arg default in the rewritten script keeps Cell 6 behavior backwards-compatible. The notebook now runs all 12 tests instead of just 9 (the 3 release-tier tests in `tests/gpu/` + the 9 inspection-tier tests in `tests/integration/`). |
| Policy doc | None. The audit, tiering rationale, T4 policy, data-size policy, and "should I add a GPU test?" criteria are not written down anywhere. | **New** `docs/testing/gpu-test-policy.md` — six sections, 250–350 lines, inventory table covering all 12 tests. |
| CI workflow | `ci.yml` runs `uv run pytest` on `ubuntu-latest`. `tests/conftest.py:44-53` autoskips anything marked `requires_compatible_gpu` (no GPU on the runner) or `requires_checkpoint` (no `models/sam3.1/sam3.1_multiplex.pt` on the runner). | Unchanged. There is no hosted GPU CI runner; both tiers run via the same Colab notebook today. |

The CI autoskip mechanism at `tests/conftest.py:44-53` reads `requires_compatible_gpu` and `requires_checkpoint` keywords on items at collection time and applies a `pytest.mark.skip` to each match. Both markers are also registered with descriptions at `tests/conftest.py:18-31`. This spec adds a fourth marker (`gpu_inspection`) at the same registration site; it does **not** add a new autoskip rule — autoskip remains driven by `requires_compatible_gpu` + `requires_checkpoint`, which every GPU-gated test in both tiers carries. (Updated 2026-05-19: a third release-tier test, `tests/gpu/test_run_end_to_end_gpu.py::test_run_end_to_end_writes_bundle`, was added by PR #46 after the original audit; counts have been bumped throughout this spec to reflect the post-merge state.)

---

## 2. Goals & Non-Goals

**Goals.**

- Make the cost of running GPU tests legible: split the suite into Tier 1 (cheap, ~load time per test) and Tier 2 (expensive, ~minutes per test) by marker so a contributor can choose the right tier for the change they made.
- Fix the latent bug in `scripts/run_gpu_tests.sh` — its current invocation silently skips the 3 `tests/gpu/` tests because the path filter excludes them.
- Write down the T4-only validation policy and the `tiny_coco` data-size floor so future GPU tests don't drift.
- Give reviewers a five-question checklist they can paste into a PR to gate new GPU tests against scope creep.
- Keep the rollout trigger-agnostic: both tiers run via the same Colab notebook today; the runner's CLI arg picks the tier. No new CI infra.

**Non-goals.**

- Building any hosted GPU CI runner, self-hosted runner, or scheduled workflow. There is no hosted GPU CI today and this spec does not change that.
- Removing any existing GPU test. The audit's CPU-stub viability column flags candidates for follow-up; actually building CPU-stub replacements is out of scope (would become separate issues).
- Retagging the existing `gpu` marker, renaming it, or changing its semantic. `gpu` continues to mean "release-tier training smoke."
- Editing `tests/gpu/test_real_train_overfits.py` or `tests/gpu/test_real_train_qlora.py`. They already carry `gpu` and stay as-is.
- Editing `src/esam3/`. This is pure docs + test-metadata + shell-script diff.
- Adding a new dependency. The runner script stays bash-only; the policy doc is plain Markdown.
- Raising the VRAM ceilings in `configs/examples/gpu_smoke_lora.yaml` (14 GB) or `gpu_smoke_qlora.yaml` (10 GB) to accommodate larger GPUs. T4 (16 GB) is the validation tier; the ceilings are pinned to T4 and stay there.
- Raising the dataset size beyond `tiny_coco` (2 images) or the step count beyond 50 grad updates for any future GPU test.

---

## 3. Files Touched / Module Layout

```text
docs/testing/
  gpu-test-policy.md                       # NEW — the policy doc (six sections, 250–350 lines)

scripts/
  run_gpu_tests.sh                         # CHANGED — accept inspection|release|all (default all)

tests/
  conftest.py                              # CHANGED — register `gpu_inspection` marker
  integration/
    test_load_sam31_real.py                # CHANGED — convert per-test decorators to module-level
                                           # `pytestmark`; add gpu_inspection
    test_peft_lora_real.py                 # CHANGED — append gpu_inspection to existing pytestmark
    test_peft_qlora_real.py                # CHANGED — append gpu_inspection to existing pytestmark
```

No file under `src/esam3/` is modified. No CI workflow file is modified. No new YAML, no new Python module, no new dependency.

---

## 4. Runner Script Design

### 4.1 Shape

`scripts/run_gpu_tests.sh` is rewritten verbatim to:

```bash
#!/usr/bin/env bash
set -euo pipefail
TIER="${1:-all}"

case "$TIER" in
  inspection) MARKER_EXPR="gpu_inspection" ; PATHS="tests/integration/" ;;
  release)    MARKER_EXPR="gpu"            ; PATHS="tests/gpu/" ;;
  all)        MARKER_EXPR="gpu or gpu_inspection" ; PATHS="tests/gpu/ tests/integration/" ;;
  *) echo "usage: $0 [inspection|release|all]" >&2; exit 2 ;;
esac

"${PYTHON:-python}" -m pytest -v --tb=short \
  -m "$MARKER_EXPR" --no-cov $PATHS
```

The script preserves three behaviors from the existing version:

- `set -euo pipefail` — strict bash mode.
- `"${PYTHON:-python}" -m pytest` — uses the interpreter that `pip install -e .` populated, avoids the bare-`pytest`-on-PATH trap documented in the existing script's header comment.
- `--no-cov` — GPU tests should not contribute to or be penalized by the 80% coverage gate.

The script gains three behaviors:

- A single positional argument `TIER`, defaulting to `all`.
- Tier-driven marker expression and path filter, so each tier runs exactly its tests and nothing else.
- A usage line on unknown tier, exit code 2 (distinct from pytest's exit codes).

### 4.2 Strict-markers compatibility

The repo enables `--strict-markers` globally via `addopts` in `pyproject.toml` (line 100). The new `gpu_inspection` marker MUST be registered before any test that uses it is collected, or `pytest` will fail with a strict-marker error. Registration happens in `tests/conftest.py` (§6.1) and runs at collection time, which is early enough for both `pytest` and the runner script. No additional flag is needed in `scripts/run_gpu_tests.sh`.

### 4.3 Tier → marker → path mapping (verbatim invariant)

| Tier | Marker expression | Path filter | Tests collected on a Turing+ box |
| --- | --- | --- | --- |
| `inspection` | `gpu_inspection` | `tests/integration/` | 9 — the three `*_real.py` files |
| `release` | `gpu` | `tests/gpu/` | 3 — the two `test_real_train_*.py` files plus `test_run_end_to_end_gpu.py` |
| `all` (default, no-arg) | `gpu or gpu_inspection` | `tests/gpu/ tests/integration/` | 12 — all of the above |
| anything else | — | — | exit 2, usage line on stderr |

This table is the contract; §6 exit criteria pin it.

### 4.4 Colab notebook impact

`notebooks/colab_gpu_tests.ipynb` Cell 6 invokes `bash scripts/run_gpu_tests.sh` with no argument. Under the new script, no-arg defaults to `all`, which runs all 12 tests instead of the 9 the current script runs. This is a deliberate fix of the latent bug §1 calls out — the 3 `tests/gpu/` tests should always have been part of the Colab run when triggered against a release-relevant change, and contributors deliberately invoking the notebook for a routine PR can switch to `bash scripts/run_gpu_tests.sh inspection` to skip the expensive training smokes.

The notebook itself is not edited by this spec — the behavior change comes from the runner script. The policy doc's Section 2 (see §5.2 below) calls out the no-arg default explicitly so a contributor reading the policy understands what Cell 6 runs.

---

## 5. Policy Doc Design

`docs/testing/gpu-test-policy.md` is a new file with six numbered sections. Target length: 250–350 lines total. The implementer writes the doc; this spec specifies WHAT each section contains, not WORD-FOR-WORD what to write.

### 5.1 Section 1: "Why this exists"

~3 sentences. Three cost axes named explicitly: wall time (real SAM3.1 forward passes are seconds not milliseconds; training smokes are minutes), paid minutes (Colab free-tier consumed per PR that touches GPU paths; any future paid runner would amplify this), and flake surface (GPU tests fail for reasons unrelated to the code: driver mismatch, transient OOM, NaN under non-deterministic kernels). Link to issue #37.

### 5.2 Section 2: "Tier definitions"

Two sub-sections, one per tier. Each sub-section names:

- **Marker** — `gpu_inspection` (Tier 1) or `gpu` (Tier 2).
- **Cadence guidance** — the exact phrasing from the brainstorm:
  - Tier 1: "run on Colab notebook for any PR touching GPU paths."
  - Tier 2: "run before a tagged release, or when training-loop / tracker / optimizer code changes land."
- **Runner invocation** — `bash scripts/run_gpu_tests.sh inspection` or `bash scripts/run_gpu_tests.sh release`.
- **What's in the tier** — one-line summary of the test file(s).

Followed by a single paragraph noting:

- No hosted GPU CI runner exists today; both tiers run via the same `notebooks/colab_gpu_tests.ipynb` notebook.
- The notebook's Cell 6 invokes `bash scripts/run_gpu_tests.sh` with no argument, which defaults to `all` — running both tiers in one session.
- A contributor can override by editing Cell 6 to pass `inspection` or `release`, or by running the script locally on a Turing+ machine.
- The tier policy is trigger-agnostic — the same tests, the same markers, the same script, regardless of trigger.

### 5.3 Section 3: "Inventory table"

A single Markdown table, one row per GPU-gated test, 12 rows total. Columns:

| Column | Contents | Source |
| --- | --- | --- |
| `file::test` | e.g. `tests/integration/test_peft_lora_real.py::test_apply_lora_on_real_sam31_under_trainable_budget` | Read each file. |
| Tier | `inspection` or `release`. | From the tiering in §1. |
| What it covers | ≤15 words. E.g. "PEFT-LoRA budget + named-param coverage on real SAM3.1." | Read the test docstring + assertions, summarize. |
| Why GPU is required | Free-text. E.g. "real SAM3.1 weights; PositionEmbeddingSine hardcodes `device='cuda'`." | Read the file-level docstring + imports. |
| CPU-stub viability | One of three: `none — needs real SAM3.1 weights` / `partial — could cover X but loses Y` / `viable — see follow-up #N`. | Judgment call per §5.3.1. |

#### 5.3.1 How to judge CPU-stub viability (instructions for the implementer)

For each test, classify into one of three buckets:

- **`none — needs real SAM3.1 weights`** — the test asserts on SAM3.1-specific named parameters (e.g. `"vision_backbone" in n`, `"transformer.decoder" in n`), or on a forward output shape that only the real model produces, or on bitsandbytes quant kernels (`Linear4bit`). CPU stub cannot replicate these.
- **`partial — could cover X but loses Y`** — a structural assertion (e.g. "trainable ratio < 5%", "Linear modules swapped") can be covered by a CPU stub that mimics the same module-tree shape, but the test would lose its end-to-end signal (e.g. "the real model under real PEFT actually produces a finite forward pass"). Name X and Y in the cell.
- **`viable — see follow-up #N`** — the test makes assertions a `TinySam3Stub` (or equivalent CPU mock) can satisfy with no real-model dependency. Implementer files a GitHub issue (`gh issue create --assignee @me`) and puts the issue number in the cell. The follow-up issue title pattern: "CPU-stub replacement for `<file>::<test>`."

The implementer must read each of the 12 tests once to fill the row. No per-test runtime numbers — they would rot the moment hardware or model versions shift.

#### 5.3.2 Example rows (template for the implementer)

The doc includes two example rows verbatim — one inspection, one release — so the implementer has a template:

```markdown
| `tests/integration/test_peft_lora_real.py::test_apply_lora_on_real_sam31_under_trainable_budget` | inspection | LoRA trainable-ratio < 5% and presence of `vision_backbone` + `transformer.decoder` LoRA params after `apply_lora` on real SAM3.1. | Asserts on SAM3.1-specific named parameters; CPU stub does not have these modules. | none — needs real SAM3.1 weights |
| `tests/gpu/test_real_train_overfits.py::test_overfits_in_50_steps` | release | 50-step LoRA overfit on tiny_coco via `run_training(gpu_smoke_lora.yaml)`; asserts loss drops ≥ 30%, peak VRAM ≤ 14 GB, all logged scalars finite. | End-to-end real training: real SAM3.1 weights, real PEFT, real CUDA kernels, real optimizer step. | none — needs real SAM3.1 weights |
```

These two example rows are part of the final doc — they are also valid rows in the 12-row inventory, not placeholders. The implementer fills the remaining 10 rows by reading the corresponding tests.

### 5.4 Section 4: "T4 validation policy"

Three short paragraphs:

1. T4 (Tesla T4, 16 GB VRAM, Turing) is the only validated VRAM tier. The Colab notebook's prereqs cell already pins T4 as the minimum runtime, and the VRAM ceilings in the smoke YAMLs are pinned to T4: 14 GB LoRA (`configs/examples/gpu_smoke_lora.yaml`, asserted at `tests/gpu/test_real_train_overfits.py:32`) and 10 GB QLoRA (`configs/examples/gpu_smoke_qlora.yaml`, asserted at `tests/gpu/test_real_train_qlora.py:34`).
2. Larger GPUs (A100, L4, H100) MAY run the suite. A green run on a larger GPU does NOT substitute for a green T4 run for release validation — the VRAM headroom asymmetry means an A100 can mask a T4 OOM that a release-tier run must catch.
3. The VRAM ceilings in the smoke YAMLs MUST NOT be raised to accommodate larger GPUs. If a future training-loop change pushes T4 above the ceiling, the fix is to reduce VRAM usage (gradient checkpointing knobs, micro-batch shape, etc.), not to raise the ceiling.

### 5.5 Section 5: "Data-size policy"

Two short paragraphs:

1. Smoke configs already use the `tests/fixtures/tiny_coco/` fixture (2 images) and 50 grad updates (epochs=25, batch_size=1, grad_accum_steps=1). This is the minimal end-to-end overfit shape. Confirmed against `configs/examples/gpu_smoke_lora.yaml` and `gpu_smoke_qlora.yaml`.
2. New GPU tests MUST reuse `tiny_coco` (or smaller). A test that needs more data is a test that has slipped into integration-suite territory and belongs out-of-tier or on CPU with stubs. The policy is enforced by code review against the §5.6 checklist, not by tooling.

### 5.6 Section 6: "Adding a new GPU test — criteria checklist"

A five-question checklist a reviewer can paste verbatim into a PR comment. Listed exactly:

1. Does the test exercise behavior unique to real SAM3.1 weights (matched named params, real forward shapes) or quant kernels (`Linear4bit`, 4-bit base + LoRA delta paths)?
2. Is the CPU+stub variant insufficient, AND is that explicitly documented in the test docstring?
3. Is the data fixture `tiny_coco` (or smaller)?
4. Does the test fit within the assigned tier's cost envelope? **Inspection:** ~load time per test (single forward pass at most). **Release:** ≤ 2 minutes on T4.
5. Is the test tagged with exactly one of `gpu` or `gpu_inspection`?

If any answer is no, the test belongs out-of-tier (move marker) or needs redesign (move work to CPU with a stub; file an issue if the work itself is worth keeping).

### 5.7 Section conventions

- Tier 1 = `gpu_inspection` everywhere in the doc. Tier 2 = `gpu`. No alternate spellings.
- "Tiny COCO" or "the tiny_coco fixture" — match the source-tree spelling `tiny_coco`.
- "T4" used without expansion after the first mention in §5.4.
- No per-test runtime numbers anywhere — they rot.
- Section 6 renders the five questions as a numbered ordered list (not bullets) so a reviewer pasting them into a PR comment gets actionable numbered items.

---

## 6. Marker Registration and Test File Edits

### 6.1 `tests/conftest.py` — marker registration

Add a fourth `addinivalue_line` block to the existing `pytest_configure` (`tests/conftest.py:18-31`), placed alphabetically or at the end of the existing block:

```python
config.addinivalue_line(
    "markers",
    "gpu_inspection: cheap GPU-gated structural/forward tests (Tier 1); "
    "see docs/testing/gpu-test-policy.md",
)
```

No new autoskip logic. The existing autoskip at `tests/conftest.py:44-53` already handles every `gpu_inspection` test because each one also carries `requires_compatible_gpu` and `requires_checkpoint`.

The repo today registers some markers in `pyproject.toml` (`integration`, `gpu`, `requires_checkpoint`, `requires_compatible_gpu`) and others via `addinivalue_line` in `tests/conftest.py` (`requires_checkpoint`, `requires_compatible_gpu`, `requires_bnb` — note: the first two are registered in both places). Registering `gpu_inspection` in `tests/conftest.py` keeps it next to `requires_bnb`, which is the same kind of test-tier-policy marker. The implementer MAY mirror the registration in `pyproject.toml` for consistency with `gpu`; this is optional and not required by the exit criteria.

### 6.2 `tests/integration/test_load_sam31_real.py` — convert + add marker

Today this file has per-test decorators (`test_load_sam31_real.py:18-19` and `:26-27`):

```python
@pytest.mark.requires_checkpoint
@pytest.mark.requires_compatible_gpu
def test_load_sam31_returns_wrapper() -> None: ...

@pytest.mark.requires_checkpoint
@pytest.mark.requires_compatible_gpu
def test_load_sam31_forward_to_canonical() -> None: ...
```

Replace with module-level `pytestmark`, matching the convention in the other two `*_real.py` files:

```python
pytestmark = [
    pytest.mark.requires_checkpoint,
    pytest.mark.requires_compatible_gpu,
    pytest.mark.gpu_inspection,
]
```

Remove the four per-test decorator lines. The two test functions are otherwise unchanged.

### 6.3 `tests/integration/test_peft_lora_real.py` — append marker

Existing module-level block at `test_peft_lora_real.py:19-22`:

```python
pytestmark = [
    pytest.mark.requires_checkpoint,
    pytest.mark.requires_compatible_gpu,
]
```

Append:

```python
pytestmark = [
    pytest.mark.requires_checkpoint,
    pytest.mark.requires_compatible_gpu,
    pytest.mark.gpu_inspection,
]
```

The three existing test functions are unchanged.

### 6.4 `tests/integration/test_peft_qlora_real.py` — append marker

Existing module-level block at `test_peft_qlora_real.py:30-33`:

```python
pytestmark = [
    pytest.mark.requires_checkpoint,
    pytest.mark.requires_compatible_gpu,
]
```

Append:

```python
pytestmark = [
    pytest.mark.requires_checkpoint,
    pytest.mark.requires_compatible_gpu,
    pytest.mark.gpu_inspection,
]
```

The four existing test functions, the module-level `_bnb_available()` and `_has_linear4bit_modules` helpers, and the per-test `@pytest.mark.skipif(not _bnb_available(), ...)` decorators are unchanged.

### 6.5 What is not edited

- `tests/gpu/test_real_train_overfits.py` — already carries `gpu`, no edit.
- `tests/gpu/test_real_train_qlora.py` — already carries `gpu`, no edit.
- `tests/gpu/conftest.py` — already exists with shared helpers, no edit.
- Any `tests/unit/`, `tests/cli/`, `tests/integration/test_train_*.py` — not GPU-gated, no edit.

---

## 7. Exit Criteria

**Code / docs.**

- [ ] `docs/testing/gpu-test-policy.md` exists. Six sections per §5. Inventory table covers all 12 GPU-gated tests with tier + CPU-stub viability. T4 validation policy and `tiny_coco` policy stated. Five-question reviewer checklist present and pasteable. Length 250–350 lines.
- [ ] `tests/conftest.py` registers the `gpu_inspection` marker with a description that mentions `docs/testing/gpu-test-policy.md`.
- [ ] `tests/integration/test_load_sam31_real.py` uses module-level `pytestmark = [requires_checkpoint, requires_compatible_gpu, gpu_inspection]`. Per-test `@pytest.mark.requires_checkpoint` and `@pytest.mark.requires_compatible_gpu` decorators are removed.
- [ ] `tests/integration/test_peft_lora_real.py` `pytestmark` list contains `pytest.mark.gpu_inspection` in addition to the two existing markers.
- [ ] `tests/integration/test_peft_qlora_real.py` `pytestmark` list contains `pytest.mark.gpu_inspection` in addition to the two existing markers.
- [ ] `scripts/run_gpu_tests.sh` accepts `inspection | release | all` (default `all`) and dispatches per the table in §4.3. Bash strict mode preserved. `"${PYTHON:-python}" -m pytest`, `--no-cov`, and `-v --tb=short` preserved from the existing script.

**Tests (CPU-only — what CI runs on `ubuntu-latest`).**

- [ ] `ruff check && uv run ruff format --check && uv run mypy src/esam3 && uv run pytest` green.
- [ ] `pytest --collect-only -m gpu_inspection` collects exactly 9 tests (the three `*_real.py` files). On a CPU box, those 9 are then auto-skipped at collection time by the `requires_compatible_gpu` autoskip in `tests/conftest.py:44-53`.
- [ ] `pytest --collect-only -m gpu` collects exactly 3 tests (`test_real_train_overfits.py::test_overfits_in_50_steps`, `test_real_train_qlora.py::test_qlora_overfits_in_50_steps`, and `test_run_end_to_end_gpu.py::test_run_end_to_end_writes_bundle`).
- [ ] `pytest --collect-only -m "gpu or gpu_inspection"` collects exactly 12 tests.
- [ ] On a CPU box, each of `bash scripts/run_gpu_tests.sh`, `bash scripts/run_gpu_tests.sh inspection`, `bash scripts/run_gpu_tests.sh release`, `bash scripts/run_gpu_tests.sh all` runs cleanly to autoskip (no collection error, no marker-warning, exit code 0 because pytest treats all-skipped as success).
- [ ] `bash scripts/run_gpu_tests.sh garbage` exits 2 with the usage line on stderr.
- [ ] `markdownlint` clean on `docs/testing/gpu-test-policy.md`. The repo-root `.markdownlint.json` rules (per `2026-05-18-ci-hardening-design.md` §5.5) apply — the doc is a new live doc, NOT under `docs/superpowers/`, so the directory-scoped relaxation does NOT apply; all default markdownlint rules except `MD013` are in force.
- [ ] `shellcheck scripts/run_gpu_tests.sh` clean.

**Explicitly NOT a gate.**

- Real-GPU runs of either tier on the PR. There is no hosted GPU CI; real-GPU validation happens manually via the Colab notebook post-merge.
- Removing any existing GPU test or implementing CPU-stub replacements. The audit's "viable" entries become follow-up issues; this PR does not write the replacements.
- Editing `notebooks/colab_gpu_tests.ipynb`. The runner-script default change preserves Cell 6's no-arg invocation.
- Editing `tests/gpu/test_real_train_*.py` or `tests/gpu/conftest.py`.
- Editing any `src/esam3/` file.
- Editing any `.github/workflows/` file.

---

## 8. Deferred (Out of Scope, Tracked Elsewhere)

- **CPU-stub replacements for "viable" inventory rows.** Each `viable — see follow-up #N` cell in the §5.3 inventory becomes a separate GitHub issue filed by the implementer with `gh issue create --assignee @me`. The replacement work (writing the CPU stub assertions, removing the GPU test) is a separate PR per issue, not this PR.
- **Hosted GPU CI.** No self-hosted runner, no scheduled workflow, no paid cloud runner. If introduced later, it would consume the same `scripts/run_gpu_tests.sh inspection|release|all` interface this spec defines, so the tier policy is forward-compatible.
- **Runtime budgets per test.** No per-test wall-clock budget is encoded anywhere. The §5.6 checklist gives qualitative envelopes ("~load time" / "≤ 2 minutes on T4") but does not enforce them in code. Enforcement would require running on T4 in CI, which is the deferred hosted-GPU-CI item above.
