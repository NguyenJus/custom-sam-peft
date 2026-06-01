<!-- markdownlint-disable MD013 -->

# Phase C — Resume Handoff (GPU test migration to RTX 5070 Ti)

**Branch:** `worktree-gpu-test-migration-5070ti` · **Worktree:** `/home/justin/projects/custom-sam-peft/.claude/worktrees/gpu-test-migration-5070ti`
**Plan:** `docs/superpowers/plans/2026-05-31-gpu-test-migration-5070ti.md` (§ Phase C, lines 714–883)
**Spec:** `docs/superpowers/specs/2026-05-31-gpu-test-migration-5070ti-design.md`
**As of:** 2026-05-31. Tree clean; 19 commits ahead of `origin`, unpushed.

## Context a fresh session needs

- Phases **A and B are DONE and green** (verified against their interface contracts). Phase C CONSUMES: Phase A markers (`gpu_t4`/`gpu_bf16`) and Phase B's `PREDICT_8GB_BUDGET_GB = 7.0` (`src/custom_sam_peft/predict/budget.py`).
- The dev card is an RTX 5070 Ti: CC 12.0, 16 GB. It satisfies BOTH `gpu_t4` and `gpu_bf16`. The 8 GB / CC 7.5 small-card claims are validated *on the 16 GB card* via ≤8 GB ceiling assertions — the card never needs to be 8 GB.
- **Crash-safety fix already landed (commit `fd2d562`).** `scripts/run_gpu_tests.sh` now runs **one pytest process per file for every tier** (OS reclaims all GPU+host memory between files — sequential, non-overlapping), and the autouse `_free_cuda_after_gpu_test` teardown synchronizes before/after gc+empty_cache. A single-process `pytest -m gpu_t4` across all files previously froze the whole machine on WSL2. **Always run GPU tests via the script, never a bare multi-file `pytest -m`.**

## Task status

| Task | What | Status |
|------|------|--------|
| **C1** | #142 8 GB-ceiling QLoRA train smoke + `QLORA_8GB_CEIL_GB = 8.0` | ✅ DONE — `tests/gpu/test_qlora_8gb_ceiling.py`; provenance row `docs/defaults-provenance.md:261`; **5070 Ti peak 2.348 GB recorded**. The residual `# tbd: #142` (line 36) is intentional per spec (confirm on a real 8 GB card; 5070 Ti can't discharge it). |
| **C2** | De-Pascal `min_gpu_qlora.yaml` rationale | ✅ DONE — fp16 / CC 7.5 framing; zero Pascal/sm_61 hits. |
| **C3** | predict-fits-8GB validation | ✅ DONE — `tests/predict/test_predict_fits_8gb.py` imports `PREDICT_8GB_BUDGET_GB`; 5070 Ti run recorded in docstring. |
| **C5** | #83 all-scope LoRA peak probe + branch | ✅ DONE — branch (a): `tests/gpu/test_peft_scope_coverage_gpu.py` landed; **measured 3.926 GB**, fits gpu_t4 band; #83 → DONE in docstring. |
| **C4** | #195 confirm-or-retune the 25/50-step budgets | ⚠️ **OPEN — the only incomplete Phase C task.** |

## Remaining work to close Phase C

### 1. C4 — confirm-or-retune #195 step budgets (the open task)

The 50-step QLoRA overfit budget lives in `tests/gpu/test_real_train_qlora.py::test_qlora_overfits_in_50_steps` and is split across save/load in `tests/gpu/test_real_train_qlora_resume.py`. `gpu_smoke_lora.yaml` sets `epochs: 25, batch_size: 1`.

- [ ] Run the 2-image overfit on the 5070 Ti; record convergence (loss drop) + speed. **[verify on 5070 Ti]**
- [ ] **Confirm** (budgets hold → no code change, just record the figures) **or retune** (change budget + docstring claims, carry provenance: measured figure + GPU + date).
- [ ] Commit only if retuned: `test(gpu): #195 confirm/retune 25/50-step budgets vs 5070 Ti measurement`.

### 2. Re-validate all Phase C GPU tests AFTER the safety fix (critical)

The recorded peaks predate commit `fd2d562`, and the prior run is what crashed the machine. Re-run to confirm green + stable under the new per-file isolation, ideally one heavy file at a time first:

```bash
# single heavy file first, to confirm stability:
.venv/bin/python -m pytest -v --no-cov -m "gpu_t4 or gpu_bf16" tests/gpu/test_peft_scope_coverage_gpu.py
# then the safe full sweep (one process per file):
scripts/run_gpu_tests.sh local
```

- [ ] Confirm C1/C3/C5 tests pass on the 5070 Ti and peaks still match the recorded numbers (2.348 / predict / 3.926 GB).

### 3. Evidence artifact (note for the boundary with Phase E)

C1/C3/C4/C5 each say "record in the evidence artifact." That consolidated **5070 Ti evidence artifact is a Phase E Definition-of-Done deliverable** (plan line 1335) and does not exist yet — measured numbers currently live only in docstrings/provenance. Phase C does not need to create it, but C4's confirmation figure and the re-validation results must be captured somewhere Phase E can fold in (drop them in this handoff or a scratch note).

## Interface Contract Phase C must EXPOSE (plan lines 871–881) — checklist

- [x] `QLORA_8GB_CEIL_GB = 8.0` cited + provenance row.
- [x] #142 8 GB-ceiling train smoke + predict-fits-8GB test, asserting vs the constants.
- [x] `min_gpu_qlora.yaml` carries CC 7.5 / 8 GB rationale (no Pascal).
- [ ] **#195 status: step budgets confirmed-or-retuned vs 5070 Ti** ← C4, outstanding.
- [x] #83 branch decision + measured all-scope LoRA peak (branch a, 3.926 GB).

## When Phase C is closed

Halt and hand off to Phase D with the literal line:
`Resume phase. Next: D. Plan: <abs path to plan>. Worktree: <abs worktree path>.`
(Phase D = three bounded CPU regression tests; D1 `test_channel_adapter_dtype.py` already exists, **D2 `test_row_outputs_nontensor.py` and D3 `test_predict_image_size_contract.py` are missing.**)
