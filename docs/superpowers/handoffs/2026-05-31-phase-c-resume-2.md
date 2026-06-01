<!-- markdownlint-disable MD013 -->

# Phase C — Resume Handoff #2 (RAM guard landed; full GPU re-validation done)

**Branch:** `worktree-gpu-test-migration-5070ti` · **Worktree:** `/home/justin/projects/custom-sam-peft/.claude/worktrees/gpu-test-migration-5070ti`
**Plan:** `docs/superpowers/plans/2026-05-31-gpu-test-migration-5070ti.md` (§ Phase C)
**Spec:** `docs/superpowers/specs/2026-05-31-gpu-test-migration-5070ti-design.md`
**Supersedes:** `docs/superpowers/handoffs/2026-05-31-phase-c-resume.md` (still accurate for the interface contract; this file carries the newer state).
**As of:** 2026-05-31. HEAD `0446e98`. Tree clean.

## ⚠️ Read first — session-crash context (not a memory problem)

Two prior sessions crashed on WSL2. **Root cause is a Claude Code / Bun runtime segfault, NOT host-RAM/VRAM OOM** (crash #1 printed an explicit Bun panic; crash #2 had no memory spike, box healthy after). They correlate with **long, orchestration-heavy sessions**. Mitigations for the next session:

- **Update Claude Code** (was 2.1.159 / Bun 1.3.14) before resuming.
- **Keep the session light:** avoid persistent Monitors and wide parallel-subagent fan-out; bound context; do one block of work then halt.
- **Never run `pytest tests/` bare** — `pyproject.toml` `addopts` has NO GPU deselection, so it runs the full real-model GPU suite in ONE process (~3.3 GB checkpoint per file) and can freeze the 16 GB box. Use `scripts/run_gpu_tests.sh local` (per-file isolation). `-o "addopts="` is safe ONLY on CPU dirs (`tests/unit`, `tests/train`).

## What this session did

1. **Re-validated the full GPU suite** post-`fd2d562` via `scripts/run_gpu_tests.sh local` (the crashed-session sweep that survived as a detached process; `SWEEP_EXIT=1`). Result below.
2. **Built the host-RAM-floor graceful-stop guard** (user-directed, on this branch) — commit **`0446e98`**. This is an END-USER training feature; it is unrelated to the session crashes above.

## GPU re-validation result (SWEEP_EXIT=1)

**Phase C named targets — all GREEN on the 5070 Ti:**

- C1 `tests/gpu/test_qlora_8gb_ceiling.py::test_qlora_8gb_ceiling` — PASSED
- C3 `tests/predict/test_predict_fits_8gb.py::test_predict_fits_8gb` — PASSED
- C5 `tests/gpu/test_peft_scope_coverage_gpu.py::test_all_scope_lora_fits_gpu_t4_band` — PASSED

**4 failures** (rest passed; CPU-tier files deselected):

| Test | Cause | Scope |
|------|-------|-------|
| `tests/gpu/test_real_train_overfits.py::test_overfits_in_50_steps` | `KeyError: 'loss/total'` at line 62 — unguarded `s["loss/total"]` in filter+comprehension | **C4 (in-scope) — FIX THIS** |
| `tests/gpu/test_calibrate_real.py::...activation_in_sane_range` | CLI exit≠0; live probe `CUDA driver error: device not ready` | out (#148/#179) |
| `tests/integration/test_peft_qlora_real.py::test_save_load_qlora_roundtrip` | bare `AssertionError` in SAM forward: `vitdet.py:110 reshape_for_broadcast` — `freqs_cis` shape mismatch on 1024² input (RoPE table vs seq dims) | out (D3-ish) |
| `tests/predict/test_gpu_predict.py::test_predict_vram_hint_log` | predict ran but never logged `"free VRAM is >12 GB"` hint | out (predict) |

Note: there was **no known-green full-suite baseline on the 5070 Ti** before this (prior run froze the box), so the 3 out-of-scope reds may be longstanding T4→5070Ti deltas (native bf16, CC 12.0), not `fd2d562` regressions.

## Remaining work

### 0. FIRST: code-quality review of the RAM guard (`0446e98`) — pending

The RAM guard passed spec-compliance review but the code-quality review (last gate of `superpowers:subagent-driven-development`) was deferred to keep this crashed session short. Run it: `superpowers:requesting-code-review` over **BASE `b70f4c2` → HEAD `0446e98`** (min sonnet/high; opus/xhigh — it's training control-flow, must never crash a real run). What it added: `_HostRamLow` (loop.py) + per-step `psutil.virtual_memory().available < floor` check beside the deadline block (shared `_flush_full_state` helper); `TrainHyperparams.host_ram_floor_gb = 2.0` (`# tbd:`-tagged); `HostRamStop` + `EvalArtifacts.host_ram_stop` (parallels `TimeLimitStop`, default `None`, zero blast radius); trainer wiring + actionable warning; `psutil>=5.9` dep. Tests: `tests/train/test_host_ram_guard.py` (8). Fix any findings, then the guard is done.

### 1. C4 — the one in-scope GPU fix + figures (#195)

- **Fix** `tests/gpu/test_real_train_overfits.py:62`: change `s["loss/total"]` → `s.get("loss/total", 0)` in the filter (mirror `tests/gpu/test_real_train_qlora.py:64` and commit `3e4b5e5`). Verify the `.get()` fix yields a NON-empty `losses` (some logged dicts DO carry `loss/total`) — not an empty-list pass.
- **Capture C4 figures** by running both 50-step overfits on the 5070 Ti **via the isolation runner / single-file pytest** (NOT bare `pytest tests/`): LoRA `test_real_train_overfits` (ceil 0.70 ratio / 14 GB) and QLoRA `test_real_train_qlora` (0.75 / 10 GB). Record loss[0], loss[-1], drop ratio, step count, peak VRAM, wall time.
- **Confirm-or-retune:** budgets hold → record figures only (no code change); else retune budget + docstring carrying provenance (figure + GPU + date). Both QLoRA overfit + resume already PASSED in the sweep, so the QLoRA budget is confirmed; the LoRA one needs the KeyError fix then a green run.

### 2. Triage the 3 out-of-scope reds → GitHub issue(s)

Do NOT expand Phase C. Classify each (transient vs real; regression vs longstanding) and `gh issue create --assignee @me --label ...`. Re-run `test_calibrate_real` once in isolation to classify the `device not ready` probe failure.

## Interface contract Phase C must EXPOSE — checklist

- [x] `QLORA_8GB_CEIL_GB = 8.0` cited + provenance row.
- [x] #142 8 GB-ceiling train smoke + predict-fits-8GB test.
- [x] `min_gpu_qlora.yaml` CC 7.5 / 8 GB rationale (no Pascal).
- [ ] **#195 step budgets confirmed-or-retuned vs 5070 Ti** ← C4: QLoRA confirmed; LoRA needs KeyError fix + green run + figures.
- [x] #83 branch decision + measured all-scope LoRA peak (3.926 GB).

## When Phase C is closed

Halt and hand to Phase D (literal line):
`Resume phase. Next: D. Plan: <abs path to plan>. Worktree: <abs worktree path>.`
(Phase D = three CPU regression tests; D1 `test_channel_adapter_dtype.py` exists; **D2 `test_row_outputs_nontensor.py` and D3 `test_predict_image_size_contract.py` are missing** — and D3 may relate to the `freqs_cis` roundtrip red above.)
