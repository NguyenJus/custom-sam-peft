<!-- markdownlint-disable MD013 -->

# Phase E — Resume Handoff (Phase D CLOSED)

**Branch:** `worktree-gpu-test-migration-5070ti` · **Worktree:** `/home/justin/projects/custom-sam-peft/.claude/worktrees/gpu-test-migration-5070ti`
**Plan:** `docs/superpowers/plans/2026-05-31-gpu-test-migration-5070ti.md` (§ Phase E, line ~1007)
**Spec:** `docs/superpowers/specs/2026-05-31-gpu-test-migration-5070ti-design.md`
**Supersedes:** `docs/superpowers/handoffs/2026-05-31-phase-d-resume.md` (Phase D now closed).
**As of:** 2026-06-01. Tree clean. **Phase E is the FINAL phase and opens the PR.**

## ⚠️ Read first — session-crash + GPU-safety context

- WSL2 session crashes are a **Claude Code / Bun runtime segfault, NOT host-RAM/VRAM OOM**. Keep sessions light: no persistent Monitors, no wide parallel fan-out.
- **This box HAS the RTX 5070 Ti.** `_has_compatible_gpu()` returns True here, so **never run bare `uv run pytest` / `pytest tests/`** — the GPU tests would actually run (3.3 GB checkpoint per file, one process) and freeze the box. Use `scripts/run_gpu_tests.sh`, or single-file `python -m pytest <file>`. `-o "addopts="` (bypasses `--cov-fail-under=80`) is safe ONLY on CPU dirs.
- **User constraint:** pause to ask the user before invoking the GPU.

## What this session did (Phase D close-out)

Phase D = three bounded CPU/stub regression tests for the GPU-bug classes in `docs/testing/gpu-audit-2026-05-24.md`. **All pure CPU; no GPU invoked.**

1. **D1 — already satisfied (no new file).** `tests/unit/test_channel_adapter_dtype.py` already exists (landed in PR #138, commit `2f5958e`, on `origin/main`) and comprehensively covers bug class 1 (channel_adapter Conv2d dtype). The plan said "Create" but the file pre-dated this PR — **did NOT duplicate it.**
2. **D2 — NEW.** `tests/unit/test_row_outputs_nontensor.py` (commit `56828aa`). 7 tests; guards bug class 2 (`_row_outputs` slicing non-tensor entries → `KeyError(slice)`). The fix (`isinstance(v, torch.Tensor)` filter) was already in `eval/evaluator.py`; this locks in the contract on CPU.
3. **D3 — NEW.** `tests/unit/test_predict_image_size_contract.py` (commit `b4db62d`). 4 tests; guards bug class 3 (predict default image-size 1024-vs-1008 RoPE) by driving `_resolve_config` and asserting `image_size == SAM3_IMAGE_SIZE == 1008`.
4. **Blast-radius regression fix (pre-existing, NOT Phase D scope) — `c8320ab`.** Full-CPU-suite verification surfaced **4 pre-existing reds** in `tests/unit/test_cli.py` (`test_train_invokes_runner`, `test_train_resume_*`). Root cause: Phase C's `884a7e0` added `host_ram_stop` consumption + `format_host_ram_message(...)` to the train/run CLI, but the tests' `fake_result` MagicMocks didn't stub `host_ram_stop` → auto-mock is truthy, enters the format branch, and `MagicMock.__format__` raises `TypeError`. Production code is correct; fix added `host_ram_stop=None` to the 5 `fake_result` MagicMocks (mirrors `time_limit_stop=None`). Confirmed present at `HEAD~2` (before D2/D3) — it was a Phase C escape (that session only ran the targeted host-ram tests, not the full `test_cli.py`).

**CPU suite status (this session):** `tests/unit tests/cli tests/config tests/eval tests/train` → **1535 passed** (`-o "addopts="`). ruff check + format clean on all touched files. **GPU dirs (`tests/gpu`, `tests/integration`, `tests/predict`) were NOT run this session** (user constraint + crash-safety); they were validated on the 5070 Ti in Phase C.

## Phase D Interface Contract — final status (R19 consumes the mapping below VERBATIM)

**Audit mapping (bug class → guarding CPU test):**

| Audit bug class (`docs/testing/gpu-audit-2026-05-24.md`) | Guarding CPU test | Status |
|---|---|---|
| 1 — `channel_adapter` Conv2d dtype mismatch (`models/sam3.py`) | `tests/unit/test_channel_adapter_dtype.py` | pre-existing (#138) |
| 2 — `_row_outputs` non-tensor entry → `KeyError(slice)` (`eval/evaluator.py`) | `tests/unit/test_row_outputs_nontensor.py` | NEW (`56828aa`) |
| 3 — predict default image-size 1024-vs-1008 RoPE (`predict/runner.py`) | `tests/unit/test_predict_image_size_contract.py` | NEW (`b4db62d`) |

**R21 — follow-up issue check: NONE filed.** The audit enumerates exactly these 3 bug classes; all three are now guarded on CPU. No large net-new coverage area surfaced → **"no further coverage gaps."**

## Stale-anchor corrections (record in Phase E policy-doc "implementation notes", R19; do NOT edit the spec)

- **D1 file pre-existed.** Plan/spec said "Create `tests/unit/test_channel_adapter_dtype.py`"; it already existed from #138 and covers the contract.
- **`_BUILTIN_DEFAULT_IMAGE_SIZE` no longer exists.** Plan/spec reference it in `predict/runner.py`; it was refactored away. The live wiring is `image_size = SAM3_IMAGE_SIZE` (imported from `custom_sam_peft.models.sam3`, `SAM3_IMAGE_SIZE == 1008`), resolved via `_resolve_config` → `_ResolvedConfig`. D3 asserts the live resolution path.

## D3 vs #207 decision (the handoff-in flagged this)

D3 **stays CPU-only**: it guards the *predict* default-image-size contract (config-less predict resolves to 1008, not 1024). **#207 is a separate GPU manifestation** — `test_peft_qlora_real::test_save_load_qlora_roundtrip` trips `freqs_cis` RoPE shape at `vitdet.py:110` on a 1024² input — and remains the tracked GPU bug. D3 does not close #207; leave #207 to its own fix.

## Phase E scope (next session) — opens the PR

Phase E (plan § Phase E, ~line 1007) consumes Phases A–D via their Interface Contracts. Tasks E1–E8: rewrite `scripts/run_gpu_tests.sh` selectors; non-blocking `check_gpu_evidence.sh` + test (R33); minimal Colab `colab-min` surface; **rewrite `docs/testing/gpu-test-policy.md`** (fold in the audit mapping + stale-anchor notes above, R19); file the `gpu_xl` issue (R31); resolve `# tbd: #193` per-step figure; run the X3 final gate; open the PR (R30 body: closes #142/#195/#83 landed; #139/#193 pending one user Colab run); close issues (Colab-dependent #139/#193 closures LAST, gated on user confirmation).

**5070 Ti evidence-artifact inputs** (from the Phase C handoff `2026-05-31-phase-d-resume.md`, still valid):

| Test | loss ratio (ceil) | peak VRAM (ceil) | wall |
|------|-------------------|------------------|------|
| LoRA `test_overfits_in_50_steps` | 0.590 (≤0.70) | 4.49 GB (≤14) | 55.0 s |
| QLoRA `test_qlora_overfits_in_50_steps` | 0.626 (≤0.75) | 3.13 GB (≤10) | 37.6 s |

Plus #83 all-scope LoRA peak = 3.926 GB (branch a, green). Predict-fits-8GB + 8 GB-ceiling peaks from C1/C3 (read those commits/test comments).

**Note on triaged reds #207/#208/#209** (from Phase C): out-of-scope GPU reds, already filed as issues. #207 (RoPE 1024² in qlora roundtrip) is real/longstanding; #208 (calibrate VRAM probe `device not ready`) and #209 (vram-hint <12 GB on a 16 GB card) are likely environmental/card-specific. Mention in PR body if relevant; they are NOT blockers.

Hand to Phase E (literal line):
`Resume phase. Next: E. Plan: /home/justin/projects/custom-sam-peft/.claude/worktrees/gpu-test-migration-5070ti/docs/superpowers/plans/2026-05-31-gpu-test-migration-5070ti.md. Worktree: /home/justin/projects/custom-sam-peft/.claude/worktrees/gpu-test-migration-5070ti.`
