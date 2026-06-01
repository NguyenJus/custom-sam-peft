<!-- markdownlint-disable MD013 -->

# GPU evidence — RTX 5070 Ti

This is the GPU-run evidence artifact checked (non-blocking) by
`scripts/check_gpu_evidence.sh` and the `gpu-evidence` CI job. It records the
real-hardware measurements that back this PR's `gpu_t4` ceiling/budget
assertions.

**Hardware:** NVIDIA GeForce RTX 5070 Ti — CC 12.0, 16 GB VRAM, WSL2.
**Runner:** `scripts/run_gpu_tests.sh` (one pytest process per file; checkpoint
freed between files).
**Captured:** Phase C real-GPU validation, 2026-05-31; full `gpu_t4 or gpu_bf16`
per-test re-validation, 2026-06-01 (32/34 pass — see "Full-sweep re-validation").

> **Freshness / non-blocking contract.** The `gpu-evidence` check reports this
> artifact `current` only when it contains the workflow's HEAD commit SHA, and
> `stale` otherwise — but it **always exits 0** (it is additive to
> `gpu-deselect-check` and never gates merge; see R23/R33 and
> `docs/testing/gpu-test-policy.md`). The authoritative refresh is the **light
> GPU subset** (R25) run after this PR is opened, with the user's explicit
> permission; that run updates this file against the then-current HEAD. Until
> then this file records the Phase C measurements below.

## #142 — 8 GB-ceiling QLoRA train + predict (5070 Ti, fp16, `min_gpu_qlora`)

| Test | Assertion | Measured | Result |
|------|-----------|----------|--------|
| `tests/gpu/test_qlora_8gb_ceiling.py::test_qlora_8gb_ceiling` | peak ≤ `QLORA_8GB_CEIL_GB` (8.0 GB) | **2.348 GB** | green |
| `tests/predict/test_predict_fits_8gb.py::test_predict_fits_8gb` | predict peak ≤ `PREDICT_8GB_BUDGET_GB` (7.0 GB) | ≤ 7.0 GB | green |

The 2.348 GB QLoRA-train peak sits well inside the 8.0 GB minimum-card
envelope, so the small-card claim validates on the 16 GB 5070 Ti.

## Training smokes (50-step `tiny_coco` overfit; #195 step budgets)

| Test | Loss ratio (ceil) | Peak VRAM (ceil) | Wall |
|------|-------------------|------------------|------|
| LoRA `test_real_train_overfits.py::test_overfits_in_50_steps` | 0.590 (≤ 0.70) | 4.49 GB (≤ 14) | 55.0 s |
| QLoRA `test_real_train_qlora.py::test_qlora_overfits_in_50_steps` | 0.626 (≤ 0.75) | 3.13 GB (≤ 10) | 37.6 s |

**#195:** the 25-epoch / 50-update step budgets hold on the 5070 Ti (both
smokes overfit inside the window); confirmed, no retune needed. Per-step
wall-clock samples (≈ 1.10 s/step LoRA, ≈ 0.75 s/step QLoRA) are recorded in
`docs/defaults-provenance.md` (#193; T4 sample still pending a user Colab run).

## #83 — all-scope LoRA peak

`tests/gpu/test_peft_scope_coverage_gpu.py` — all-scope (regex `.*`) LoRA peak
VRAM = **3.926 GB**, well inside the ≤ 16 GB `gpu_t4` band. **Branch (a): DONE**
— the all-scope smoke fits `gpu_t4`; no `gpu_xl` overflow.

## Full-sweep re-validation (2026-06-01, per-test process isolation)

Re-ran the entire `gpu_t4`/`gpu_bf16` surface (34 tests) one process per test on
the 5070 Ti (durable per-test transcript). **32 passed, 2 known out-of-scope reds
(#208, #209).** Every Phase C deliverable above is green, and #207 now passes
(see below).

## #207 — FIXED (1008px input contract)

`test_peft_qlora_real::test_save_load_qlora_roundtrip` previously tripped a
`freqs_cis` RoPE shape at `vitdet.py:110` because the test forwarded a raw
**1024²** tensor straight to the model, which only accepts SAM 3.1's native
1008px (`SAM3_IMAGE_SIZE`). The predict/train paths rescale internally, so this
was a test-only contract gap, not a model bug. Corrected to `SAM3_IMAGE_SIZE`
(part of the codebase-wide 1024px→1008 purge); the roundtrip now passes.

## Out-of-scope GPU reds (triaged, NOT blockers)

Filed as separate issues during Phase C; not regressions introduced by this PR
(both re-confirmed identical in the 2026-06-01 sweep):

- **#208** — `calibrate` VRAM probe `device not ready` (likely environmental).
- **#209** — vram-hint not logged for < 12 GB free on a 16 GB card (card-specific).
