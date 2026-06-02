# GPU Test Policy

<!-- Capability-taxonomy policy (CC ≥ 7.5 floor, bf16-band, xl-band). Supersedes the old gpu_local/sub-CC7.5 three-tier scheme. -->

## 1. Why this exists

GPU-gated tests impose costs that must be managed deliberately. **Wall time**:
loading real SAM 3.1 weights takes seconds even for a structural inspection,
and a training-smoke test that runs 50 gradient updates on a Colab T4 takes
many minutes. **Paid minutes**: Colab free-tier quota is consumed each time a
GPU notebook run is triggered, and any move to a paid cloud runner amplifies
that cost linearly with PR volume. **Flake surface**: GPU tests fail for
reasons unrelated to the code — driver version mismatches, transient OOM from
other sessions sharing a host, and non-deterministic CUDA kernel scheduling.

These costs motivate *tiering*, but the **primary axis is hardware capability**
(VRAM / compute-capability architecture), not cost. Cost guidance is secondary:
it follows naturally from hardware constraints, and tests that require a T4 are
structurally different from tests that fit an entry-level card — not merely
"more expensive". The CPU-only CI workflow (`ci.yml`, `ubuntu-latest`) never
runs GPU-gated tests; the conftest autoskip handles that automatically for any
test carrying `requires_compatible_gpu` or `requires_checkpoint`. See
[#37](https://github.com/NguyenJus/custom-sam-peft/issues/37) for the
original audit that prompted this policy.

## 2. Tier definitions

The taxonomy uses **three named tiers**, but they are **not linearly ordered**.
`gpu_t4` and `gpu_bf16` form the ≤ 16 GB band (closed upper bound: a card
reporting slightly under the marketing-16 GB still counts); `gpu_xl` is the
disjoint > 16 GB band. A card satisfies whichever tiers match its capability —
it does not automatically satisfy lower tiers.

### `gpu_t4` — fp16 band (CC ≥ 7.5, VRAM ≤ 16 GB)

- **Marker:** `@pytest.mark.gpu_t4`
- **Hardware gate:** CUDA GPU with **CC ≥ 7.5 AND total VRAM ≤ 16 GB** (Tesla
  T4 floor; also satisfied by RTX 5070 Ti at CC 12.0 / 16 GB).
- **Dtype note:** **bf16 is coerced to fp16 below CC 8.0**. Tests in this tier
  run in fp16 on a real T4 (CC 7.5) and in fp16-compatible mode on a 5070 Ti.
  They must not assert bf16-faithful numerics (that is `gpu_bf16`'s role). The
  coercion is confirmed on a real Colab T4 (2026-06-01, finding #139) — see
  [`gpu-evidence-colab-t4.md`](gpu-evidence-colab-t4.md).
- **Flash-attention note:** below CC 8.0 there is no Flash attention, so SAM 3.1
  self-attn falls back to the math kernel, which **materializes the full H·N²
  score matrix**. At 1008px this is ~12.8 GiB in one allocation — fine for a
  single-class (B=1/K=1) forward but it OOMs the T4 for **multiplex (K ≥ 8)
  forward**. Multiplex forward is therefore *not guaranteed on a real T4*; it is
  a `gpu_bf16` (Flash-card) capability. The **guaranteed T4 floor is the
  B=1/K=1 single-class forward** plus LoRA/QLoRA training. Confirmed on Colab
  2026-06-01 (#212) — see [`gpu-evidence-colab-t4.md`](gpu-evidence-colab-t4.md).
- **Count:** **31 tests** across `tests/integration/`, `tests/predict/`, and
  `tests/gpu/`.
- **Cadence guidance:** run before a tagged release, or when the training
  runner / tracker / optimizer code changes land.
- **Runner invocation:** `bash scripts/run_gpu_tests.sh t4`

### `gpu_bf16` — faithful bf16 band (CC ≥ 8.0, VRAM ≤ 16 GB)

- **Marker:** `@pytest.mark.gpu_bf16`
- **Hardware gate:** **CC ≥ 8.0 AND total VRAM ≤ 16 GB** (RTX 5070 Ti at CC
  12.0 satisfies this; a T4 at CC 7.5 does NOT).
- **Dtype note:** these tests exercise **non-coerced, faithful bf16 numerics**.
  They are inappropriate on any card that silently downcasts bf16 to fp16.
- **Flash-attention note:** this tier also carries the **multiplex (K ≥ 8)
  forward** tests, which need Flash attention (CC ≥ 8.0) to avoid materializing
  a ~12.8 GiB self-attn score matrix at 1008px — they OOM a real T4 (#212).
- **Count:** **3 tests** (`tests/gpu/test_bf16_faithful.py`;
  `tests/gpu/test_multiplex_vram.py`; the K=8 multiplex forward in
  `tests/integration/test_load_sam31_real.py`).
- **Runner invocation:** `bash scripts/run_gpu_tests.sh bf16`

### `gpu_xl` — large-VRAM band (VRAM > 16 GB)

- **Marker:** `@pytest.mark.gpu_xl`
- **Hardware gate:** **total VRAM > 16 GB** (e.g. A100, L4, H100).
- **Isolation note:** a > 16 GB card satisfies **only** `gpu_xl` and is
  intentionally NOT auto-run for the ≤ 16 GB ceiling assertions. Running
  `gpu_t4` tests on a > 16 GB card would silently pass VRAM assertions even if
  actual usage exceeds 16 GB.
- **Count:** **0 tests** (empty in this PR; populated only via the gpu_xl
  follow-up issue).
- **Cadence guidance:** cloud auto-provision via
  [#124](https://github.com/NguyenJus/custom-sam-peft/issues/124).
- **Runner invocation:** `bash scripts/run_gpu_tests.sh xl`

### Superseded tier: `gpu_local` (CC 6.x / pre-T4)

`gpu_local` and pre-T4 GPU (CC 6.1 and below) support were **removed** in the
capability-taxonomy migration. Minimum supported GPU is now the **Tesla T4
(CC 7.5)**. See [§ 3](#3-t4-floor--sub-cc75-dropped-decision) for the full
rationale. The old `gpu_local` marker and the `gpu-pascal` (cu118) uv extra no
longer exist.

## 3. T4-floor / sub-CC7.5 dropped decision

GPUs below CC 7.5 (e.g. CC 6.1) are **no longer supported** as of this PR.

- The `gpu-pascal` (cu118) uv extra was removed; the default `cu130` torch
  wheel covers both T4 and the 5070 Ti.
- The old `docs/testing/local-pascal-gpu-testing.md` was deleted.
- `docs/testing/manual-gpu-pass-2026-05-24-gtx1080.md` carries a **superseded
  banner** noting that CC 6.1 is no longer a tested target.
- The historical GTX 1080 audit (`gpu-audit-2026-05-24.md`) remains as dated
  history.

The minimum supported GPU is the **Tesla T4 (CC 7.5)**. The autoskip gate
changed accordingly (see [§ 4](#4-auto-detection-model)).

## 4. Auto-detection model

`tests/conftest.py` contains two capability functions:

- **`_has_compatible_gpu() -> bool`** — returns `True` when a CUDA GPU with
  **CC ≥ 7.5** is present and a kernel-launch probe succeeds.
- **`_satisfied_tiers() -> set[str]`** — a **live capability probe** that
  returns the set of tier names the running card satisfies. Examples:
  - RTX 5070 Ti (CC 12.0, 16 GB): `{"gpu_t4", "gpu_bf16"}`
  - Tesla T4 (CC 7.5, 16 GB): `{"gpu_t4"}`
  - A100 (CC 8.0, 40 GB): `{"gpu_xl"}`
  - CPU / no GPU: `set()`

**Skip predicate** (`pytest_collection_modifyitems`): a test marked tier `T`
runs iff `T ∈ active_tiers`, where `active_tiers` is:

1. the runner's forced tier (env/CLI `-m` filter) when set, **or**
2. `_satisfied_tiers()` otherwise.

The old `_TIER_ORDER` linear comparison is **deleted**. Tiers are a set
membership query, not a numeric comparison.

## 5. Runner selectors

The runner script (`scripts/run_gpu_tests.sh`) keeps one pytest process **per
file/node** (so the ~3.3 GB checkpoint is freed between files). An unknown
selector exits non-zero with a usage line on stderr.

| Selector | Marker filter / subset | Purpose |
|----------|------------------------|---------|
| (default / `local`) | `gpu_t4 or gpu_bf16` | 5070 Ti local bulk run |
| `t4` | `gpu_t4` | Real T4 / ≤ 16 GB fp16 band |
| `bf16` | `gpu_bf16` | 5070 Ti bf16-faithful tests |
| `xl` | `gpu_xl` | > 16 GB band (empty here) |
| `colab-min` | `test_load_sam31_real.py::test_load_sam31_forward_to_canonical` + `test_real_train_qlora.py::test_qlora_overfits_in_50_steps` | Minimal Colab T4 surface |
| `light` | Post-PR permissioned evidence subset (see [§ 9](#9-light-gpu-subset)) | Evidence run after PR opens |

Per-tier test counts and typical runtimes:

| Tier | Tests collected | Typical runtime |
|------|-----------------|-----------------|
| `gpu_t4` | 31 | ~15–25 min on 5070 Ti / ~30–40 min on T4 |
| `gpu_bf16` | 3 | < 5 min on 5070 Ti |
| `gpu_xl` | 0 (currently empty) | — |

The script uses `"${PYTHON:-python}" -m pytest` rather than bare `pytest` so
the runner picks the same interpreter that `pip install -e .` populated — bare
`pytest` on PATH can resolve to a different Python in Colab and trigger
`ModuleNotFoundError: No module named 'custom_sam_peft'`.

## 6. T4 validation policy

T4 (Tesla T4, 16 GB VRAM, Turing architecture, CC 7.5) is the validated
ceiling for training-smoke tests. The VRAM ceilings in the two smoke configs
are calibrated to T4: 14 GB for LoRA (`configs/examples/gpu_smoke_lora.yaml`,
asserted at `tests/gpu/test_real_train_overfits.py:44`) and 10 GB for QLoRA
(`configs/examples/gpu_smoke_qlora.yaml`, asserted at
`tests/gpu/test_real_train_qlora.py:34`). These ceilings are not aspirational
targets — they are the measured peak usage under the current training
configuration on a T4, rounded up by a small margin to absorb minor
non-determinism across Colab sessions.

Larger GPUs (A100, L4, H100) MAY run the suite and are useful for development
iteration when a T4 is not available. However, a green run on a larger GPU does
**NOT** substitute for a green T4 run when validating a `gpu_t4` release gate.
The asymmetry is fundamental: an A100 with 40 GB will silently pass the VRAM
assertion (`peak_vram_gb <= 14.0`) even if actual usage is 20 GB; a T4 with 16
GB will fail in that scenario. T4 validation MUST be confirmed before merging
any PR that carries `gpu_t4` tests as a release gate.

The VRAM ceilings MUST NOT be raised to accommodate larger GPUs or to paper
over a VRAM regression. If a future change pushes T4 peak usage above a
ceiling, the correct response is to reduce usage — reduce micro-batch size,
reduce gradient accumulation steps, shrink sequence length — not to raise the
assertion.

## 7. Data-size policy

The smoke configs drive training with `tests/fixtures/tiny_coco/`: 2 images,
50 gradient updates total (epochs=25, batch\_size=1, grad\_accum\_steps=1;
25 × 2 = 50 gradient updates). This is the minimal end-to-end overfit shape
that exercises the full `run_training` call graph without growing the GPU run
time beyond a single Colab session slot. The fixture size and step count are
confirmed against both `configs/examples/gpu_smoke_lora.yaml` and
`configs/examples/gpu_smoke_qlora.yaml`. The 50-step window is long enough to
confirm the loss curve is moving (the loss-ratio assertion verifies it drops by
at least 25–30% depending on tier) and short enough that the entire `gpu_t4`
suite completes in a reasonable window on a T4. The two-image fixture
intentionally repeats the same images because the goal is overfit verification
— not generalization.

New GPU tests MUST reuse `tiny_coco` (or a smaller fixture). A test that
requires more images is testing data variety, not the training machinery; it
belongs either in a separate tier with explicit cost approval or on CPU with
stubs and mocked data loaders. This policy is enforced by code review against
the [§ 8 checklist](#8-adding-a-new-gpu-test--criteria-checklist).

## 8. Adding a new GPU test — criteria checklist

> Test on CPU by default; a test earns a GPU tier ONLY when it needs real
> weights / kernels / quant that a stub cannot reproduce.

Any test that needs a real GPU must carry at least one of
`@pytest.mark.requires_compatible_gpu` and `@pytest.mark.requires_checkpoint`
so that the autoskip in `tests/conftest.py` (gate: **CC ≥ 7.5** plus a
kernel-launch probe) suppresses it on CPU-only CI runners. It must also carry
**exactly one tier marker** (`gpu_t4`, `gpu_bf16`, or `gpu_xl`) so the runner
script can filter by tier. The CPU-only CI workflow on `ubuntu-latest` never
runs tests marked `requires_compatible_gpu` or `requires_checkpoint` — the
conftest autoskip applies at collection time.

Before merging a new GPU-gated test, confirm all five of the following. A
reviewer can paste this list verbatim into a PR comment:

1. Does the test exercise behavior unique to real SAM 3.1 weights (matched
   named params, real forward shapes) or quant kernels (`Linear4bit`, 4-bit
   base + LoRA delta paths)?
2. Is the CPU+stub variant insufficient, AND is that explicitly documented in
   the test docstring?
3. Is the data fixture `tiny_coco` (or smaller)?
4. Does the test fit within the assigned tier's cost envelope?
   **`gpu_t4`:** ≤ ~20 minutes wall on T4.
   **`gpu_bf16`:** bf16-faithful assertion only; typically < 1 min.
   **`gpu_xl`:** requires explicit cost approval.
5. Is the test tagged with exactly one of `gpu_t4`, `gpu_bf16`, or `gpu_xl`?

If any answer is no, the test belongs out-of-tier (move marker) or needs
redesign (move work to CPU with a stub; file an issue if the work itself is
worth keeping).

### Choosing the right tier

The primary question is hardware capability:

- **`gpu_t4`:** needs real weights, CUDA kernels, or a training loop, and fits
  within 16 GB VRAM. Does NOT require bf16-faithful numerics.
- **`gpu_bf16`:** must assert non-coerced bf16 precision (a CC ≥ 8.0 / RTX
  5070 Ti or Ampere+ card is required; a T4 at CC 7.5 would silently coerce).
- **`gpu_xl`:** exceeds T4's 16 GB VRAM budget.

When in doubt, prefer `gpu_t4` and document the rationale in the test
docstring. The [audit table](gpu-audit-2026-05-24.md) shows the historical
split; only tests that explicitly require bf16 fidelity belong in `gpu_bf16`.

## 9. Light GPU subset

The `light` subset is run **after the PR is opened**, only with the user's
explicit permission. It is tracked as a PR-description checklist item, NOT a
hard merge gate. The subset is:

1. `test_load_sam31_real.py::test_load_sam31_forward_to_canonical` — load +
   forward canonical check.
2. `tests/gpu/test_qlora_8gb_ceiling.py` — the #142 8 GB-ceiling QLoRA smoke.
3. `tests/predict/test_predict_fits_8gb.py` — predict-fits-8GB probe.
4. `tests/gpu/test_predict_budget_warning.py` — predict-budget warning GPU test.

Runner: `bash scripts/run_gpu_tests.sh light`

## 10. Non-blocking GPU evidence gate

A standalone script `scripts/check_gpu_evidence.sh` (takes
`<artifact_path> <head_sha>`) provides a **non-blocking** evidence report:

- **Always exits 0** regardless of finding.
- Reports `missing` / `stale` / `current`; green only when the evidence
  artifact references HEAD.
- Wired as a CI job `gpu-evidence` that is **additive to `gpu-deselect-check`,
  never required, never blocking**.
- Its non-blocking guarantee is locked by
  `tests/unit/test_gpu_evidence_check.py`.
- The evidence artifact lives at `docs/testing/gpu-evidence-5070ti.md`.

Merge proceeds solely on the user's explicit approval.

## 11. Integration-audit coverage (R19)

The GPU integration audit (`docs/testing/gpu-audit-2026-05-24.md`) enumerated
exactly three bug classes. All are now guarded by CPU tests that run in normal
CI on `ubuntu-latest` (NOT GPU-gated):

| Audit bug class | Guarding CPU test | Status |
|---|---|---|
| 1 — `channel_adapter` Conv2d dtype mismatch (`models/sam3.py`) | `tests/unit/test_channel_adapter_dtype.py` | pre-existing (#138) |
| 2 — `_row_outputs` non-tensor entry → `KeyError(slice)` (`eval/evaluator.py`) | `tests/unit/test_row_outputs_nontensor.py` | new |
| 3 — predict default image-size 1024-vs-1008 RoPE (`predict/runner.py`) | `tests/unit/test_predict_image_size_contract.py` | new |

R21 finding: the audit enumerates exactly these 3 bug classes, all now guarded
on CPU, so **no further coverage gaps** and no follow-up issue was filed.

## 12. Implementation notes (stale-anchor corrections)

These notes reconcile the spec/plan with the actual code state so the record
matches the implementation:

- The spec/plan said to CREATE `tests/unit/test_channel_adapter_dtype.py`, but
  it already existed from PR #138 and covers bug class 1 — it was not
  duplicated.
- `_BUILTIN_DEFAULT_IMAGE_SIZE` no longer exists in `predict/runner.py`
  (refactored away). The live wiring is `image_size = SAM3_IMAGE_SIZE`
  (`SAM3_IMAGE_SIZE == 1008`), which `tests/unit/test_predict_image_size_contract.py`
  asserts.
