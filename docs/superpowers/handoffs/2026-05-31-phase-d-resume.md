<!-- markdownlint-disable MD013 -->

# Phase D — Resume Handoff (Phase C CLOSED)

**Branch:** `worktree-gpu-test-migration-5070ti` · **Worktree:** `/home/justin/projects/custom-sam-peft/.claude/worktrees/gpu-test-migration-5070ti`
**Plan:** `docs/superpowers/plans/2026-05-31-gpu-test-migration-5070ti.md` (§ Phase D, line ~885)
**Spec:** `docs/superpowers/specs/2026-05-31-gpu-test-migration-5070ti-design.md`
**Supersedes:** `docs/superpowers/handoffs/2026-05-31-phase-c-resume-2.md` (Phase C now fully closed).
**As of:** 2026-05-31. Tree clean.

## ⚠️ Read first — session-crash context (not a memory problem)

Prior sessions crashed on WSL2 from a **Claude Code / Bun runtime segfault, NOT host-RAM/VRAM OOM**. They correlate with long, orchestration-heavy sessions. Mitigations:

- **Update Claude Code** before resuming (was Bun 1.3.14).
- **Keep the session light:** no persistent Monitors, no wide parallel fan-out; one block of work, then halt.
- **Never run `pytest tests/` bare** — `pyproject.toml` `addopts` has NO GPU deselection, so it runs the full real-model GPU suite in ONE process (~3.3 GB checkpoint per file) and can freeze the 16 GB box. Use `scripts/run_gpu_tests.sh local`, or single-file `python -m pytest <file>` (each in its own process). `-o "addopts="` is safe ONLY on CPU dirs (`tests/unit`, `tests/train`).

## What this session did (Phase C close-out)

1. **RAM-guard code-quality review (the deferred gate on `0446e98`) — DONE.** Reviewed `b70f4c2..0446e98` (opus). Found a **Critical** bug: the CLI never consumed `host_ram_stop`, so a host-RAM stop fell through to model reload + eval/export/bundle on a near-OOM box — defeating the guard. Fixes landed in **`884a7e0`**:
   - `train`/`run` CLI now early-return on `host_ram_stop` before any reload/eval/export/bundle; new `cli/_host_ram.py` formatter mirrors `_time_limit.py`.
   - `loop.py` per-step `psutil.virtual_memory()` probe wrapped **fail-open** (probe error logs-and-skips; flush/raise stay outside the try) so it can never crash a real run.
   - Tests: CLI consumption (train/run short-circuit), exactly-at-floor boundary (strict `<`), psutil-probe-raises fail-open; dead `hasattr` simplified.
   - `uv.lock` updated to record `psutil>=5.9` (added to `pyproject` in `0446e98` but the lock was missed).
   - Verified: 20 CPU tests green (`tests/train/test_host_ram_guard.py` + `tests/cli/test_host_ram_cli.py`), ruff check + format clean. **RAM guard is fully done.**

2. **C4 (#195) — KeyError fixed + budgets confirmed on the 5070 Ti — DONE.** Commit **`94e09a6`**:
   - `tests/gpu/test_real_train_overfits.py:62` — `s["loss/total"]` → `s.get("loss/total", 0)` in the filter (mirrors `test_real_train_qlora.py` / `3e4b5e5`). Verified **non-empty** losses (`n=2`), not an empty-list pass.
   - Ran both 50-step overfits in isolation on the **RTX 5070 Ti (sm_120)**, 2026-05-31 — both GREEN:

     | Test | loss[0]→loss[-1] | ratio (ceil) | peak VRAM (ceil) | wall |
     |------|------------------|--------------|------------------|------|
     | LoRA `test_overfits_in_50_steps` | 0.5222→0.3081 | **0.590** (≤0.70) | **4.49 GB** (≤14) | 55.0 s |
     | QLoRA `test_qlora_overfits_in_50_steps` | 0.6221→0.3893 | **0.626** (≤0.75) | **3.13 GB** (≤10) | 37.6 s |

   - Both budgets hold with headroom → **confirmed, no retune.** Provenance recorded in the ceiling comments of both test files. **These figures are the #195 input for Phase E's evidence artifact (plan line ~1335).**

3. **Triaged the 3 out-of-scope GPU reds → GitHub issues** (classified from the prior sweep's evidence; did NOT re-run on GPU — user needed the card):
   - **#207** — `test_peft_qlora_real::test_save_load_qlora_roundtrip`: `freqs_cis` (RoPE) shape mismatch at `vitdet.py:110 reshape_for_broadcast` on 1024² input. **Real bug, longstanding** (shape logic is device-independent, not a 5070 Ti delta). `bug`/`testing`/`priority:medium`. **Likely related to Phase D's D3** (`test_predict_image_size_contract.py`).
   - **#208** — `test_calibrate_real::...activation_in_sane_range`: live VRAM probe `CUDA driver error: device not ready`. **Likely transient/environmental** (opt-in probe from #148/#179, no green baseline yet). `testing`/`priority:low`.
   - **#209** — `test_gpu_predict::test_predict_vram_hint_log`: `">12 GB free VRAM"` hint not logged. **Likely card-specific expectation** (16 GB card → free VRAM may legitimately sit <12 GB after load). `testing`/`priority:low`.

## Phase C interface contract — final status (all EXPOSED items satisfied)

- [x] `QLORA_8GB_CEIL_GB = 8.0` cited + provenance.
- [x] #142 8 GB-ceiling train smoke + predict-fits-8GB test (C1/C3 green).
- [x] `min_gpu_qlora.yaml` CC 7.5 / 8 GB rationale (no Pascal).
- [x] **#195 budgets confirmed vs 5070 Ti** — LoRA + QLoRA figures above; no retune.
- [x] #83 branch decision + measured all-scope LoRA peak (3.926 GB; C5 green).

Note: the plan's per-step `- [ ]` checkboxes are left as-is across ALL phases — this project's convention tracks state in these handoff docs, not by ticking the plan. Don't read unticked plan boxes as undone work.

## Phase D scope (next session)

Phase D = three bounded CPU/stub regression tests (plan § Phase D, ~line 885). All `[verify on CPU/CI]` — no GPU needed.

- **D1 `tests/.../test_channel_adapter_dtype.py`** — already EXISTS (per prior handoff). Verify it still passes; confirm it covers the intended regression.
- **D2 `test_row_outputs_nontensor.py`** — MISSING, must be written.
- **D3 `test_predict_image_size_contract.py`** — MISSING, must be written. **May relate to #207** (the 1024² `freqs_cis`/RoPE shape contract) — read #207 before authoring D3; decide whether D3 should encode the contract that #207's fix must satisfy, or stay CPU-only and leave the GPU bug to #207.

Read the exact D1/D2/D3 specs in the plan's Phase D section + the matching spec requirements before implementing. Use `superpowers:subagent-driven-development` (TDD; implementers sonnet/high). Gate every implementer on `ruff check` AND `ruff format --check`.

## When Phase D is closed

Phase E is the FINAL phase and opens the PR (per the Orchestrator pipeline). Phase E also builds the **5070 Ti evidence artifact** — feed it the #195 figures table above, the #142/#83 peaks, and the predict peak. Hand to Phase E (literal line):
`Resume phase. Next: E. Plan: <abs path to plan>. Worktree: <abs worktree path>.`
