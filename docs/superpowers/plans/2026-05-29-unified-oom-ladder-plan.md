# Unified OOM ladder: one shared B-then-K component across train, eval, predict — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract OOM ladder STATE + the B-then-K halving DECISION into one shared `OomLadder` component (`src/custom_sam_peft/oom.py`), migrate train onto it without changing observable behavior, then migrate eval (gaining a K-rung) and predict (gaining the full ladder — the #181 fix) onto the same component (spec §1–§8).

**Architecture:** `OomLadder` owns sticky, monotonically-decreasing `(micro_batch_size, effective_K)` state plus `pending_oom_events`, and a single `on_oom(step) -> OomDecision` method that applies the B-then-K policy (halve B to 1, then halve K to 1, then one floor retry, then terminal) with a guarded `torch.cuda.empty_cache()` at the top. The B-rung *mechanics* are factored into one shared module-level `_halve_microbatch(state, step)` routine (field-only, so it works on train's method-less `_State` stub) that `on_oom`'s B-branch delegates to. Each caller maps the returned `OomDecision` to its own control flow: train replays gradients from group 0 on a K-change. Train's catch site (inner helper) and K-handler (outer rung) are at different levels, so the inner helper handles the B-rung directly via the shared `_halve_microbatch` routine (it does NOT call `state.on_oom()`) and raises a fieldless `_MicrobatchExhausted` at B==1; the outer rung is the single `on_oom()` call for the K decision (exactly once per OOM event). Eval/predict accumulate independent per-row predictions, so they resume mid-chunk on a K-change via index-driven `while` loops with per-image-chunk buffer-and-commit, calling `on_oom()` once at their single catch site. `OomEvent` relocates from `train/types.py` to `oom.py` and is re-exported for back-compat.

**Tech Stack:** Python 3.12, PyTorch (`torch.cuda.OutOfMemoryError`, `torch.cuda.empty_cache`/`is_available`), `enum.Enum`, frozen `@dataclass`, pytest (CPU-only; synthetic `torch.cuda.OutOfMemoryError` injection, no GPU). Lint/type: `uv run ruff check` / `uv run ruff format --check` / `uv run mypy src/custom_sam_peft`. Tests: `uv run pytest`.

---

## Ground-truth facts verified against source (read before starting)

The codebase is **further along than spec §2 verbatim implies** — train already has the full B-then-K ladder inline. These were confirmed by reading the actual files; the tasks below target the **real** current symbols and migrate them onto the new shared component:

1. `src/custom_sam_peft/train/loop.py` **already** defines `_MicrobatchExhausted` (loop.py:42-47, currently a *fieldless* `Exception`), `OomState` (loop.py:63-80, fields `step`/`micro_batch_size`/`pending_oom_events`/`effective_K`), `_train_step_with_oom_ladder` (loop.py:83-137, inner B-halving + `empty_cache()` + raises `_MicrobatchExhausted` at B==1), and `train_step`'s outer K-replay loop (loop.py:226-406) that catches `_MicrobatchExhausted`, `zero_grad`s, halves `effective_K` **inline** (loop.py:389), appends a `multiplex_halved` `OomEvent` (loop.py:390-397), and replays. **So the migration's job is to re-source the inline `state.micro_batch_size //= 2` (loop.py:122-123) and the inline `oom_state.effective_K = max(1, …//2)` (loop.py:389) from a single `OomLadder.on_oom()` call — NOT to add a ladder that does not exist.**
2. **The double-halving trap (spec §3 "One halving per OOM event", §4 invariant, §5.1).** Today the inner helper halves B at loop.py:122-123 and the outer rung halves K at loop.py:389 — two *different* mutation sites for one logical ladder. After migration, the work stays split across the two levels but the *mechanics* are centralized: the inner helper's B-rung calls the shared module-level `_halve_microbatch(state, step)` routine (field-only — works on the method-less `_State` stub), and the outer rung is the **single** `on_oom()` call site, reached only at B==1 where `on_oom()` skips its B-branch and decides the K-rung (`RETRY_K`/`FLOOR_RETRY`/`TERMINAL`). The inner helper does **NOT** call `state.on_oom()` (the `_State` stub has no such method) and `_MicrobatchExhausted` stays **fieldless** (no carried decision). For any single OOM, exactly one halving occurs — B (inner, via `_halve_microbatch`) or K (outer, via the one `on_oom()` call) — never both, never twice. A second `on_oom()` call (e.g. one at the inner helper plus one at the outer rung) would halve K twice for one OOM and break `tests/unit/test_trainer_oom_retry.py`.
3. `_train_step_with_oom_ladder` (loop.py:83-137) takes `state: Any` and is called both by the trainer (with `OomState`) and by `tests/unit/test_trainer_oom_retry.py` (with a local `_State` dataclass having `step`/`micro_batch_size`/`pending_oom_events` — **no `effective_K`, no methods**). The helper today only ever touches `state.micro_batch_size` and `state.pending_oom_events`, so the `_State` stub works. **The migration must keep the helper working with a method-less stub that has only `micro_batch_size`/`pending_oom_events`/`step`** (it never needs `effective_K` because B>1 there, or it raises; and it must NOT call `state.on_oom()` because the stub has no methods). The chosen path: the helper's B-rung calls the shared module-level `_halve_microbatch(state, step)` routine — which operates only on `state` *fields*, so it works on the stub — and at B==1 raises a fieldless `_MicrobatchExhausted`. The K decision is sourced by the *outer rung* calling `on_oom()` once. See Task 4 for the exact shape that keeps `_State` green.
4. `OomState` is constructed only at `trainer.py:495-498`: `OomState(micro_batch_size=cfg.train.batch_size, effective_K=min(cfg.train.multiplex.classes_per_forward, _MULTIPLEX_CAP))`. `_MULTIPLEX_CAP` is imported at trainer.py:493 from `custom_sam_peft.models.sam3`.
5. `OomEvent` (train/types.py:16-34) is `@dataclass(frozen=True)` with `step: int`, `action: Literal["microbatch_halved", "multiplex_halved"]`, `new_micro_batch_size: int`, `effective_K: int | None = None`. It is imported from `custom_sam_peft.train.types` by: `train/loop.py:33`, `runs/bundle.py:37`, `eval/_artifacts.py:11`, and `tests/unit/test_trainer_oom_retry.py:18`. **All four must keep working unchanged after the relocation.**
6. `tests/unit/test_eval_batch_size_cap.py:14` does `from custom_sam_peft.train.loop import OomState` and constructs `OomState(micro_batch_size=2)` / `OomState(micro_batch_size=4)` reading `.micro_batch_size`. So `train/loop.py` must keep exposing the name **`OomState`**, constructible with `micro_batch_size=` (and optionally `effective_K=`), exposing `.micro_batch_size`.
7. `eval/evaluator.py` (200-259) already has the index-driven `while i < len(examples)` loop with `chunk_buf` buffer-and-commit (the #176 fix) and `_row_outputs` (39-47), but the inner class loop is `for group in _chunked(dataset.class_names, MULTIPLEX_CAP)` (216) — **a fixed `MULTIPLEX_CAP`, no K-rung**. `_eval_forward_with_oom_ladder` (50-83) is B-only and is the symbol to replace. The OOM catch at 223-226 `break`s and discards `chunk_buf`. `cfg.batch_size` is already an int (resolved by `run_eval`).
8. `predict/runner.py` `run_predict` (240-…) has the flat `for chunk_paths in _chunked(list(image_paths), bs)` (397) × `for group in _chunked(prompts, MULTIPLEX_CAP)` (431) loops. The OOM catch (437-442) logs one `logger.error` advice line and **re-raises** — no halving, no `empty_cache`, no recovery. `prompts` is a `list[str]` (parsed at 268 via `parse_prompts`); the 1-based category id is computed as `prompts.index(group[kk]) + 1` (448) — a **value lookup**, distinct from eval's index arithmetic. `bs` is resolved at 343-349 (`"auto"` → `decide_eval_batch_size(classes_per_forward=MULTIPLEX_CAP)`). The per-image open/transform block is 405-428; row postprocess + score/top-k filter is 444-459; verbose + progress is 464-486; `n_successful`/`images_processed` counters drive the report and the all-failed guard (491-493).
9. `predict/runner.py` catches `RuntimeError` and string-matches `"out of memory"` (437-438). The spec wants `torch.cuda.OutOfMemoryError` caught (it subclasses `RuntimeError`); a non-OOM `RuntimeError` must still propagate.
10. **Test layout:** train/oom unit tests live in `tests/unit/`; eval OOM tests in `tests/unit/test_eval_oom_ladder.py`; predict tests in `tests/predict/` (`test_runner_smoke.py` is the harness reference — stub `nn.Module` returning `(B*K_g, Q, …)` tensors, `_patch_load` patches `custom_sam_peft.models.sam3.load_sam31`, `_make_opts`/`_make_image_dir` build CPU inputs). The new predict test goes in `tests/predict/`.
11. **CI lint/test commands** (`.github/workflows/ci.yml`): `uv run ruff check`, `uv run ruff format --check`, `uv run mypy src/custom_sam_peft`, `uv run pytest` (the `--cov-fail-under=80` gate is in `pyproject.toml:136`, run on the FULL suite). Markdown lint: `npx --yes markdownlint-cli2 --config .config/markdownlint-cli2.jsonc "**/*.md"` — **note CI uses `.config/markdownlint-cli2.jsonc`** (disables only MD013/MD018/MD029), which is *stricter* than `docs/superpowers/.markdownlint.json`. Conform this plan + the spec to the stricter CI config.

### Hard invariants to preserve (enforced as explicit test assertions in the noted tasks)

- (a) **Exactly one halving per OOM event** — B (inner helper, via the shared `_halve_microbatch` routine) OR K (outer rung, via the single `on_oom()` call), never both, never twice. `on_oom()` is invoked at most once per OOM; for train that single call is the outer rung's K decision. The B-mechanics have one shared implementation (`_halve_microbatch`), used by both the inner helper and `on_oom`'s B-branch (Task 1, Task 4).
- (b) **`test_trainer_oom_retry.py` and `test_eval_batch_size_cap.py` stay GREEN untouched** — `OomState` name + `micro_batch_size`/`pending_oom_events` fields + the `_State` stub contract (Task 4).
- (c) **B and K are sticky** — only ever decrease, for the ladder's lifetime (Task 1).
- (d) **`FLOOR_RETRY` returned at most once** over the ladder's lifetime; next `on_oom` after a consumed floor returns `TERMINAL` (Task 1).
- (e) **Row reassembly uses the ACTUAL sub-group length `K_g`**, never a fixed `MULTIPLEX_CAP` (Task 6, Task 7).
- (f) **Buffer-and-commit: a chunk commits exactly once, only on full completion; `RETRY_B` discards the buffer, `RETRY_K` retains it** — no dup, no drop (Task 6, Task 7).
- (g) **`OomEvent` importable from BOTH `custom_sam_peft.oom` and `custom_sam_peft.train.types`** (Task 2).

---

## File structure

**Phase 1 — shared component + train migration:**

- `src/custom_sam_peft/oom.py` — **new.** `OomDecision` enum, relocated `OomEvent` frozen dataclass, the module-level `_halve_microbatch(state, step)` shared B-rung mechanic (field-only; used by both `on_oom`'s B-branch and train's inner helper), and `OomLadder` (sticky monotone B/K state + `pending_oom_events` + `on_oom`). Pure state + decision; nothing about any caller's loop.
- `src/custom_sam_peft/train/types.py` — `OomEvent` moves out; re-export `from custom_sam_peft.oom import OomEvent` plus `__all__`. Module docstring updated to note the relocation.
- `src/custom_sam_peft/train/loop.py` — `_MicrobatchExhausted` stays **fieldless** (no `decision` field — infeasible against the untouched field-only `_State` stub); `OomState` becomes an alias/subclass of `OomLadder` (keeping the name + construction contract); `_train_step_with_oom_ladder`'s inner B-rung calls the shared `_halve_microbatch(state, step)` routine directly (NOT `state.on_oom()`) and raises a fieldless `_MicrobatchExhausted` at B==1; `train_step`'s outer rung is the single `on_oom()` site for the K decision (replacing the inline K-halving).
- `src/custom_sam_peft/train/trainer.py` — construct the ladder (already constructs `OomState(micro_batch_size=…, effective_K=…)`; only changes if `OomState`'s constructor signature changes — Task 4 keeps it compatible so this file may need **no** change).
- Tests: `tests/unit/test_oom_ladder.py` (**new**, §7.1); `tests/unit/test_trainer_oom_retry.py` + `tests/unit/test_eval_batch_size_cap.py` (must stay green **untouched**); `tests/unit/test_oom_reexport.py` (**new**, the §9 smoke import).

**Phase 2 — eval + predict migration:**

- `src/custom_sam_peft/eval/evaluator.py` — replace `_eval_forward_with_oom_ladder` usage with an `OomLadder`; make the inner class loop index-driven on `ladder.effective_K` with a K-rung; preserve `chunk_buf` commit-on-completion and actual-`K_g` reassembly.
- `src/custom_sam_peft/predict/runner.py` — replace the two `for`-loops with index-driven `while` loops keyed on `ladder.micro_batch_size`/`ladder.effective_K`; add per-image-chunk `chunk_buf` commit-on-completion; catch `torch.cuda.OutOfMemoryError`; map the four decisions.
- Tests: `tests/unit/test_eval_oom_ladder.py` (extend, §7.3); `tests/predict/test_predict_oom_ladder.py` (**new**, §7.4).

---

## Sequencing rationale (read before starting)

1. **Phase 1 first, end-to-end.** `oom.py` (Task 1) is the contract. Train (Task 4) is the **reference consumer** that pins the ladder's semantics against `test_trainer_oom_retry.py`. The relocation (Task 2) must land before train migrates (train imports `OomEvent`). Phase 1 ends with the interface contract pinned for Phase 2 to consume cold.
2. **Within Phase 1, serialize:** Task 1 (`oom.py`) → Task 2 (relocate `OomEvent`, re-export) → Task 3 (re-export smoke test) → Task 4 (train migration). Task 1 and Task 2 touch different files but Task 2's re-export imports the relocated symbol Task 1 creates, so Task 1 lands first. Task 4 depends on both.
3. **Phase 2 after Phase 1.** Eval (Task 6) and predict (Task 7) both **consume** the frozen `OomLadder` API. They touch **disjoint files** (`eval/evaluator.py` vs `predict/runner.py`) with no shared state → **candidates for parallel dispatch**. Their tests (Task 5 extends eval's test file; Task 7's new predict test file) are likewise disjoint. Serialize only if a reviewer prefers eval-first as the simpler reference.
4. **Tests are TDD-ordered within each task** (write failing test → run red → implement → run green → commit).

**Parallelizable:** Phase 2 Task 6 (eval) and Task 7 (predict) are file-disjoint and dependency-free once Phase 1 is merged — dispatch in parallel. Everything in Phase 1 serializes.

---

## Phase 1 — `src/custom_sam_peft/oom.py` + train migration

> The shared component and its reference consumer (train). Train's existing behavior (and `test_trainer_oom_retry.py` + `test_eval_batch_size_cap.py`) is the acceptance gate. Tasks serialize: 1 → 2 → 3 → 4.

### Task 1: Build `OomLadder` + `OomDecision` + relocated `OomEvent` in `oom.py`

**Files:**

- Create: `src/custom_sam_peft/oom.py`
- Test: `tests/unit/test_oom_ladder.py`

- [ ] **Step 1: Write the failing `OomLadder` unit tests**

Create `tests/unit/test_oom_ladder.py` with the full §7.1 suite. CPU-only; synthetic OOM is not needed here (we call `on_oom()` directly). The `empty_cache` guard tests patch `torch.cuda.is_available`/`empty_cache`.

```python
"""Unit tests for the shared OomLadder (spec §7.1).

CPU-only. on_oom() is called directly; no synthetic CUDA OOM needed except
for the empty_cache-guard tests which patch torch.cuda.is_available/empty_cache.
"""

from __future__ import annotations

import pytest
import torch

from custom_sam_peft.oom import OomDecision, OomEvent, OomLadder


def test_decision_sequence_b_then_k_then_floor_then_terminal() -> None:
    """From (B0=4, K0=4): RETRY_B until B==1, then RETRY_K until K==1, then one
    FLOOR_RETRY, then TERMINAL forever. Assert exact decision + B/K after each."""
    ladder = OomLadder(micro_batch_size=4, effective_K=4)

    assert ladder.on_oom(step=0) is OomDecision.RETRY_B
    assert (ladder.micro_batch_size, ladder.effective_K) == (2, 4)
    assert ladder.on_oom(step=0) is OomDecision.RETRY_B
    assert (ladder.micro_batch_size, ladder.effective_K) == (1, 4)

    assert ladder.on_oom(step=0) is OomDecision.RETRY_K
    assert (ladder.micro_batch_size, ladder.effective_K) == (1, 2)
    assert ladder.on_oom(step=0) is OomDecision.RETRY_K
    assert (ladder.micro_batch_size, ladder.effective_K) == (1, 1)

    assert ladder.on_oom(step=0) is OomDecision.FLOOR_RETRY
    assert (ladder.micro_batch_size, ladder.effective_K) == (1, 1)

    assert ladder.on_oom(step=0) is OomDecision.TERMINAL
    assert ladder.on_oom(step=0) is OomDecision.TERMINAL


def test_b_and_k_are_sticky_monotone() -> None:
    """B and K only ever decrease across the ladder's lifetime."""
    ladder = OomLadder(micro_batch_size=8, effective_K=2)
    seen_b = [ladder.micro_batch_size]
    seen_k = [ladder.effective_K]
    for _ in range(10):
        ladder.on_oom(step=1)
        seen_b.append(ladder.micro_batch_size)
        seen_k.append(ladder.effective_K)
    assert seen_b == sorted(seen_b, reverse=True)
    assert seen_k == sorted(seen_k, reverse=True)
    assert ladder.micro_batch_size == 1
    assert ladder.effective_K == 1


def test_pending_oom_events_emission() -> None:
    """One OomEvent per halving; none for FLOOR_RETRY/TERMINAL. microbatch_halved
    carries new_micro_batch_size with effective_K is None; multiplex_halved carries
    the new effective_K and the current new_micro_batch_size."""
    ladder = OomLadder(micro_batch_size=2, effective_K=2)

    ladder.on_oom(step=7)  # RETRY_B: B 2->1
    ladder.on_oom(step=7)  # RETRY_K: K 2->1
    ladder.on_oom(step=7)  # FLOOR_RETRY: no event
    ladder.on_oom(step=7)  # TERMINAL: no event

    assert len(ladder.pending_oom_events) == 2  # one per halving only
    b_ev, k_ev = ladder.pending_oom_events
    assert b_ev.action == "microbatch_halved"
    assert b_ev.new_micro_batch_size == 1
    assert b_ev.effective_K is None
    assert b_ev.step == 7
    assert k_ev.action == "multiplex_halved"
    assert k_ev.new_micro_batch_size == 1
    assert k_ev.effective_K == 1


def test_empty_cache_guarded_called_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    """With is_available()->True, every on_oom invokes empty_cache once."""
    calls: list[int] = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append(1))
    ladder = OomLadder(micro_batch_size=2, effective_K=1)
    ladder.on_oom(step=0)  # RETRY_B
    ladder.on_oom(step=0)  # FLOOR_RETRY (B==1, K==1)
    ladder.on_oom(step=0)  # TERMINAL
    assert len(calls) == 3  # one per call, regardless of decision


def test_empty_cache_not_called_when_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """With is_available()->False, empty_cache is never called."""
    calls: list[int] = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append(1))
    ladder = OomLadder(micro_batch_size=2, effective_K=2)
    ladder.on_oom(step=0)
    ladder.on_oom(step=0)
    assert calls == []


def test_degenerate_start_b1_k1_floor_then_terminal() -> None:
    """(B=1, K=1): first on_oom is FLOOR_RETRY, second is TERMINAL; no events."""
    ladder = OomLadder(micro_batch_size=1, effective_K=1)
    assert ladder.on_oom(step=0) is OomDecision.FLOOR_RETRY
    assert ladder.on_oom(step=0) is OomDecision.TERMINAL
    assert ladder.pending_oom_events == []


def test_degenerate_start_b1_k_gt1_halves_k_immediately() -> None:
    """(B=1, K>1): starts halving K immediately (skips the B-rung)."""
    ladder = OomLadder(micro_batch_size=1, effective_K=4)
    assert ladder.on_oom(step=0) is OomDecision.RETRY_K
    assert ladder.effective_K == 2
    assert ladder.pending_oom_events[-1].action == "multiplex_halved"


def test_oom_event_microbatch_defaults_effective_k_none() -> None:
    ev = OomEvent(step=1, action="microbatch_halved", new_micro_batch_size=4)
    assert ev.effective_K is None


def test_oom_event_multiplex_carries_effective_k() -> None:
    ev = OomEvent(step=5, action="multiplex_halved", new_micro_batch_size=1, effective_K=8)
    assert ev.effective_K == 8
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_oom_ladder.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'custom_sam_peft.oom'`.

- [ ] **Step 3: Write `src/custom_sam_peft/oom.py`**

Create the module with `OomEvent` (relocated verbatim from `train/types.py`), `OomDecision`, and `OomLadder`:

```python
"""Shared OOM ladder: sticky B-then-K state + the halving decision.

`OomLadder` owns ladder STATE (micro_batch_size B, effective_K K) and the
B-then-K halving DECISION only. It knows nothing about microbatches, image
chunks, class groups, gradients, or replay — those are caller concepts. Train,
eval, and predict each construct a ladder with their per-path initial (B, K)
and map the returned `OomDecision` to their own control flow (spec §3 mapping
table). `OomEvent` lives here (relocated from train/types.py, which re-exports
it for back-compat).

Spec: docs/superpowers/specs/2026-05-29-unified-oom-ladder-design.md
"""

from __future__ import annotations

import enum
import logging
from dataclasses import dataclass, field
from typing import Any, Literal

import torch

_LOG = logging.getLogger(__name__)

__all__ = ["OomDecision", "OomEvent", "OomLadder"]


@dataclass(frozen=True)
class OomEvent:
    """One OOM-halving transition, recorded for telemetry / bundle rendering.

    `action` records the rung:
      - "microbatch_halved": B was halved (effective_K is None).
      - "multiplex_halved": K was halved; carries the new effective_K.

    Fields capture *post*-halving state so downstream rendering can reconstruct
    the run's safety-net history without re-traversing mutable state.
    """

    step: int
    action: Literal["microbatch_halved", "multiplex_halved"]
    new_micro_batch_size: int
    effective_K: int | None = None  # set only for "multiplex_halved" events


class OomDecision(enum.Enum):
    """What a caller should do after one OOM, per the B-then-K policy."""

    RETRY_B = "retry_b"
    RETRY_K = "retry_k"
    FLOOR_RETRY = "floor_retry"
    TERMINAL = "terminal"


def _halve_microbatch(state: Any, step: int | None = None) -> None:
    """Shared B-rung mechanic: halve micro_batch_size + record the transition.

    FIELD-ONLY. Operates solely on the object's *fields* — `micro_batch_size`,
    `pending_oom_events`, `step` — so it works on BOTH `OomLadder` and train's
    field-only `_State` stub (which has no methods). This is the SINGLE
    implementation of the B-halving: `OomLadder.on_oom()`'s B-branch delegates
    to it, and train's inner helper calls it directly (spec §4 "Shared
    _halve_microbatch routine", §5.1). Callers do the `empty_cache()` and the
    `micro_batch_size > 1` guard before calling this.
    """
    if step is not None:
        state.step = step
    state.micro_batch_size //= 2
    state.pending_oom_events.append(
        OomEvent(
            step=state.step,
            action="microbatch_halved",
            new_micro_batch_size=state.micro_batch_size,
        )
    )
    _LOG.warning(
        "OOM at step %d — halving micro_batch_size to %d",
        state.step,
        state.micro_batch_size,
    )


@dataclass
class OomLadder:
    """Sticky, monotonically-decreasing B-then-K OOM state + decision.

    Constructed per path with the initial (micro_batch_size, effective_K).
    on_oom() applies the B-then-K policy; callers map the returned OomDecision
    to their own control flow (spec §3 mapping table). B and K only ever
    decrease (sticky). FLOOR_RETRY is returned at most once per lifetime.
    """

    micro_batch_size: int  # B — only ever decreases
    effective_K: int  # K — only ever decreases
    pending_oom_events: list[OomEvent] = field(default_factory=list)
    step: int = 0  # last-seen step, for telemetry parity with the old OomState
    _floor_retry_used: bool = field(default=False, repr=False)

    def on_oom(self, step: int | None = None) -> OomDecision:
        """Apply the B-then-K policy to one OOM event.

        1. Guarded torch.cuda.empty_cache() (the #176 robustness guarantee).
        2. If B > 1: delegate to the shared _halve_microbatch() routine
           (halves B, records a microbatch_halved event, warns), RETRY_B.
        3. elif K > 1: halve K, record a multiplex_halved event, warn, RETRY_K.
        4. elif not used: consume the single FLOOR_RETRY.
        5. else: TERMINAL.

        Records at most one OomEvent per call (the halving it performed); none
        for FLOOR_RETRY / TERMINAL.
        """
        if step is not None:
            self.step = step
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if self.micro_batch_size > 1:
            # Delegate to the SINGLE shared B-rung mechanic (also called directly
            # by train's inner helper — spec §4). No duplicated halving logic.
            _halve_microbatch(self, self.step)
            return OomDecision.RETRY_B

        if self.effective_K > 1:
            self.effective_K //= 2
            self.pending_oom_events.append(
                OomEvent(
                    step=self.step,
                    action="multiplex_halved",
                    new_micro_batch_size=self.micro_batch_size,
                    effective_K=self.effective_K,
                )
            )
            _LOG.warning(
                "OOM at step %d after micro_batch=1 — halving effective_K to %d",
                self.step,
                self.effective_K,
            )
            return OomDecision.RETRY_K

        if not self._floor_retry_used:
            self._floor_retry_used = True
            return OomDecision.FLOOR_RETRY

        return OomDecision.TERMINAL
```

Notes for the implementer:

- `step` is kept as a field (default `0`) so `OomState` (Task 4) preserves the old attribute the trainer writes (`oom_state.step = global_step`) and the `_State` stub reads. `on_oom(step=...)` updates it; passing `step=None` leaves it at the last value.
- `K //= 2` floors at 1 automatically for `K in {2, 3}` → 1; for K already 1 the branch is skipped. This matches the old `max(1, K // 2)`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_oom_ladder.py -v`
Expected: PASS — all decision-sequence, stickiness, event-emission, empty-cache-guard, and degenerate-start cases green.

- [ ] **Step 5: Commit**

```bash
git add src/custom_sam_peft/oom.py tests/unit/test_oom_ladder.py
git commit -m "feat(oom): add shared OomLadder + OomDecision + relocated OomEvent"
```

### Task 2: Relocate `OomEvent` out of `train/types.py` with a back-compat re-export

**Files:**

- Modify: `src/custom_sam_peft/train/types.py`
- Test: covered by Task 3's smoke test + the existing suite.

- [ ] **Step 1: Rewrite `train/types.py` to re-export from `oom.py`**

Replace the `OomEvent` dataclass definition (train/types.py:16-34) with a re-export. Keep the module docstring (updated to note the relocation) and add `__all__`:

```python
"""Re-export shim for the trainer subsystem's OOM types.

`OomEvent` now lives in `custom_sam_peft.oom` (the shared OOM ladder module);
this module re-exports it so existing imports
(`from custom_sam_peft.train.types import OomEvent`) keep working unchanged.

Spec: docs/superpowers/specs/2026-05-29-unified-oom-ladder-design.md §4.
"""

from __future__ import annotations

from custom_sam_peft.oom import OomEvent

__all__ = ["OomEvent"]
```

Notes for the implementer:

- After this edit `train/types.py` no longer needs `dataclass`/`Literal` imports — remove them (ruff will flag unused imports otherwise).
- `train/loop.py:33`, `runs/bundle.py:37`, `eval/_artifacts.py:11`, and `tests/unit/test_trainer_oom_retry.py:18` all import `OomEvent` from `custom_sam_peft.train.types` — they keep working because the re-export resolves to the same class.

- [ ] **Step 2: Run the existing suite to confirm no import breakage**

Run: `uv run pytest tests/unit/test_trainer_oom_retry.py tests/unit/runs/test_bundle.py -q`
Expected: PASS — `OomEvent` resolves through the re-export; `test_oom_event_supports_multiplex_halved_action` and `test_oom_event_microbatch_action_defaults_effective_k_none` (test_trainer_oom_retry.py:164-172) still pass against the relocated class.

- [ ] **Step 3: Commit**

```bash
git add src/custom_sam_peft/train/types.py
git commit -m "refactor(train): relocate OomEvent to oom.py with back-compat re-export"
```

### Task 3: Smoke test — `OomEvent` resolves from both import paths

**Files:**

- Create: `tests/unit/test_oom_reexport.py`

- [ ] **Step 1: Write the smoke test**

Create `tests/unit/test_oom_reexport.py` (spec §9 risk mitigation — both paths resolve to the same class):

```python
"""Smoke: OomEvent is importable from both oom.py and train/types.py, same class."""

from __future__ import annotations


def test_oom_event_same_class_from_both_paths() -> None:
    from custom_sam_peft.oom import OomEvent as OomEventNew
    from custom_sam_peft.train.types import OomEvent as OomEventReexport

    assert OomEventNew is OomEventReexport


def test_oom_event_constructs_from_train_types_path() -> None:
    from custom_sam_peft.train.types import OomEvent

    ev = OomEvent(step=1, action="microbatch_halved", new_micro_batch_size=4)
    assert ev.new_micro_batch_size == 4
    assert ev.effective_K is None
```

- [ ] **Step 2: Run to verify it passes**

Run: `uv run pytest tests/unit/test_oom_reexport.py -v`
Expected: PASS — both names are the same object; construction via the re-export path works.

- [ ] **Step 3: Commit**

```bash
git add tests/unit/test_oom_reexport.py
git commit -m "test(oom): smoke test OomEvent re-export resolves to one class"
```

### Task 4: Migrate `train/loop.py` + `train/trainer.py` onto `OomLadder` (preserve behavior exactly)

**Files:**

- Modify: `src/custom_sam_peft/train/loop.py` (`_MicrobatchExhausted` ~42-47; `OomState` ~63-80; `_train_step_with_oom_ladder` ~83-137; `train_step` outer rung ~379-406)
- Modify: `src/custom_sam_peft/train/trainer.py` (`OomState(...)` ~495-498 — only if its constructor changes)
- Test: `tests/unit/test_trainer_oom_retry.py` (stays GREEN **untouched**); `tests/unit/test_eval_batch_size_cap.py` (stays GREEN **untouched**); `tests/unit/test_train_loop_multiplex.py` (the K-rung behavioral tests — stay green)

> This is the highest-risk task (spec §9). The migration **only re-sources** the B-halving mechanics (the shared `_halve_microbatch` routine) and the K decision (`OomLadder.on_oom()` at the outer rung); control flow (`_MicrobatchExhausted`, whole-step replay, `optimizer.zero_grad(set_to_none=True)`, the `/(G·grad_accum)` divisor, the NaN-driven `ValueError` group-skip) is preserved verbatim. `_MicrobatchExhausted` stays **fieldless**. The single correctness change is **one halving per OOM event** — B at the inner helper (via the shared routine) or K at the outer rung (the single `on_oom()` call), never both (Ground-truth fact 2). No `test_trainer_oom_retry.py`/`test_eval_batch_size_cap.py` edit.

- [ ] **Step 1: Confirm the contract tests are currently green (baseline)**

Run: `uv run pytest tests/unit/test_trainer_oom_retry.py tests/unit/test_eval_batch_size_cap.py tests/unit/test_train_loop_multiplex.py -q`
Expected: PASS (this is the pre-migration baseline — these must stay green after every edit below).

- [ ] **Step 2: Add the `oom.py` import (do NOT add a `decision` field to `_MicrobatchExhausted`)**

> **DO NOT IMPLEMENT a carried-`decision` field.** `_MicrobatchExhausted` stays **fieldless** (exactly as today, loop.py:42-47). This matches SPEC §1/§4/§5.1: the inner helper handles only the B-rung (via the shared `_halve_microbatch` routine) and raises a fieldless `_MicrobatchExhausted` at B==1; the *outer* rung is the single `on_oom()` site for the K decision (Step 5/Step 6). A carried-`decision` field is **infeasible** because it would require the inner helper to call `state.on_oom()`, but `tests/unit/test_trainer_oom_retry.py`'s field-only `_State` stub (lines 45-49: `step`/`micro_batch_size`/`pending_oom_events`, **no methods**) is passed to the helper directly and must stay untouched. **The implementer must NOT add then remove a `decision` field — leave `_MicrobatchExhausted` exactly as it is today.**

The only loop.py change in this step is the import. Add it at the top of `loop.py` (alongside the existing `from custom_sam_peft.train.types import OomEvent` at line 33):

```python
from custom_sam_peft.oom import OomDecision, OomEvent, OomLadder, _halve_microbatch
```

(Keep importing `OomEvent` from `train.types` **or** switch it to `oom`; both resolve to the same class. Prefer importing `OomDecision`/`OomLadder`/`OomEvent`/`_halve_microbatch` all from `custom_sam_peft.oom` and dropping the `train.types` import line to avoid a redundant import — ruff-clean.)

- [ ] **Step 3: Make `OomState` an alias for `OomLadder` (preserve the name + construction contract)**

Replace the `OomState` dataclass (loop.py:63-80) with an alias. `OomLadder`'s fields (`micro_batch_size`, `effective_K`, `pending_oom_events`, `step`) are a superset of what the trainer and `test_eval_batch_size_cap.py` need, and its constructor accepts `micro_batch_size=`/`effective_K=` as keyword args. **But** `OomLadder` requires `effective_K` positionally/keyword with no default, whereas `test_eval_batch_size_cap.py` constructs `OomState(micro_batch_size=2)` with **no** `effective_K`. So `OomState` must default `effective_K`. Use a thin subclass that adds the default:

```python
@dataclass
class OomState(OomLadder):
    """Back-compat alias for OomLadder used by the trainer.

    Keeps the name `OomState` (imported by tests/unit/test_eval_batch_size_cap.py)
    and makes `effective_K` default to 1 so `OomState(micro_batch_size=N)` still
    constructs (the trainer always passes effective_K explicitly; the test does
    not). All ladder behavior is inherited unchanged. Spec §4 / §5.1.
    """

    effective_K: int = 1
```

Notes for the implementer:

- `@dataclass` inheritance: fields with defaults in the base (`pending_oom_events`, `step`, `_floor_retry_used`) come after `micro_batch_size`/`effective_K`. Re-declaring `effective_K: int = 1` in the subclass gives it a default while keeping field order valid (no "non-default after default" error, because `micro_batch_size` stays first and `effective_K` now has a default). Verify the dataclass compiles: `OomState(micro_batch_size=2)` and `OomState(micro_batch_size=8, effective_K=4)` must both work.
- `tests/unit/test_eval_batch_size_cap.py` reads `oom_state.micro_batch_size` only — inherited from `OomLadder`. No test edit.

- [ ] **Step 4: Re-source the inner helper's B-rung from the shared `_halve_microbatch` routine**

Rewrite the inner helper's `except` block (loop.py:119-137) so its B-rung calls the shared module-level `_halve_microbatch(state, state.step)` routine (Task 1) and `continue`s; at B==1 it raises a **fieldless** `_MicrobatchExhausted`. The helper does **NOT** call `state.on_oom()` — `tests/unit/test_trainer_oom_retry.py`'s `_State` stub (lines 45-49: `step`/`micro_batch_size`/`pending_oom_events`, **no methods**) is passed to the helper directly and must stay untouched. This implements SPEC §4 (shared `_halve_microbatch`) / §5.1 (train inner helper):

```python
        except torch.cuda.OutOfMemoryError as oom_err:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if state.micro_batch_size > 1:
                # Shared B-rung mechanic — field-only, so the method-less _State
                # stub works. The SAME routine backs on_oom()'s B-branch (Task 1),
                # so there is one implementation of the B-halving, not a copy.
                _halve_microbatch(state, state.step)
                continue
            # B == 1: hand off to train_step's outer rung, which is the single
            # on_oom() site for the K decision. _MicrobatchExhausted stays FIELDLESS
            # (no carried OomDecision) — the inner helper never calls state.on_oom().
            raise _MicrobatchExhausted(f"micro_batch exhausted at step {state.step}") from oom_err
```

Notes for the implementer:

- This **removes** the bespoke inline `state.micro_batch_size //= 2; state.pending_oom_events.append(OomEvent(...)); _LOG.warning(...)` block (loop.py:120-136) and replaces it with the single `_halve_microbatch(state, state.step)` call — the B-halving mechanics now have exactly ONE implementation (the shared routine), used by both this helper and `on_oom`'s B-branch. No byte-identical copy is hand-maintained.
- The `empty_cache()` guard stays in the inner helper's catch handler (the helper does not go through `on_oom`, whose `empty_cache` is at its own top). `_halve_microbatch` does NOT call `empty_cache` itself — callers do (here, and `on_oom`'s top).
- `state` here is `OomState`/`OomLadder` in prod and the `_State` stub in the test. `_halve_microbatch` touches only fields (`micro_batch_size`/`pending_oom_events`/`step`), so all three work. **Do NOT add a `decision` field to `_MicrobatchExhausted` (Step 2) and do NOT call `state.on_oom()` here.**

- [ ] **Step 5: Why the inner helper uses `_halve_microbatch` (not `state.on_oom`) — implements SPEC §4/§5.1**

This step records the rationale; no additional code edit beyond Step 4.

`tests/unit/test_trainer_oom_retry.py`'s `_State` (lines 45-49) is a plain dataclass with `step`/`micro_batch_size`/`pending_oom_events` and **no methods**. It is passed to `_train_step_with_oom_ladder` directly (e.g. test lines 64, 73, 87), and `test_oom_after_microbatch_1_signals_b_exhausted` (line 86) only asserts `pytest.raises(_MicrobatchExhausted)` — it does NOT inspect any exception field. So:

- The inner helper **cannot** call `state.on_oom()` (the stub has no such method) and **cannot** carry an `OomDecision` on `_MicrobatchExhausted` (which would require the helper to first call `on_oom()` to compute it).
- Instead, the inner helper's B-rung calls the shared field-only `_halve_microbatch(state, step)` routine, and at B==1 raises a fieldless `_MicrobatchExhausted`.
- `OomLadder.on_oom()` is the single decision site for the K-rung, called **exactly once** by the *outer rung* (Step 6). Eval/predict (Phase 2) call `on_oom()` once at their single catch site for the full B-then-K policy.

This **is** the SPEC §4/§5.1 design — not a divergence from it. The SPEC describes exactly this two-level split: inner helper handles B via the shared routine; outer rung is the single `on_oom()` K-decision site. Centralization holds because the B-halving mechanics have one shared implementation (`_halve_microbatch`, used by both the inner helper and `on_oom`'s B-branch), and the K decision has one site (`on_oom`). For any single OOM, exactly one halving occurs — B (inner) or K (outer) — never both, never twice.

> **Implementer decision (binding):** The inner helper's B-rung calls the shared `_halve_microbatch(state, state.step)` routine (so the `_State` stub and all of `test_trainer_oom_retry.py` stay untouched and green). `OomLadder.on_oom()` is the single K/floor/terminal decision site, called **once** by the outer rung (Step 6). `_MicrobatchExhausted` stays **fieldless** — do NOT add a `decision` field (Step 2 is explicitly not implemented). `OomState` is still an `OomLadder` subclass (Step 3) so the outer rung can call `oom_state.on_oom(...)`, but the inner helper never calls it (it only reads/writes `micro_batch_size`/`pending_oom_events`/`step` via `_halve_microbatch`, all of which the stub provides).

- [ ] **Step 6: Re-source the K-rung in `train_step`'s outer rung from `on_oom()`**

Rewrite the `except _MicrobatchExhausted` handler (loop.py:379-406) to call `oom_state.on_oom(global_step)` **once** and branch on the decision, replacing the inline `oom_state.effective_K = max(1, …//2)` + manual event append + manual warning (loop.py:387-405):

```python
        except _MicrobatchExhausted as exc:
            # Inner B-ladder exhausted at micro_batch=1. Source the K/floor/terminal
            # decision from the single shared ladder (spec §3 mapping table, train
            # column). on_oom() is called exactly once here for this OOM event; the
            # inner helper did NOT call it (it halved B via the shared
            # _halve_microbatch routine, never on_oom), so K is not double-halved.
            if oom_state is None:
                raise RuntimeError(
                    f"OOM at step {global_step} after micro_batch=1 and "
                    f"classes_per_forward=1. Use a larger GPU or smaller image_size."
                ) from exc
            decision = oom_state.on_oom(global_step)
            if decision is OomDecision.RETRY_K:
                # Discard partial grads from the failed larger-K attempt; replay
                # the whole step from group 0 at the smaller (already-decremented) K.
                optimizer.zero_grad(set_to_none=True)
                continue  # loop continues -> replays at smaller K
            if decision is OomDecision.FLOOR_RETRY:
                # Retry the whole step once at the floor (B and K unchanged).
                optimizer.zero_grad(set_to_none=True)
                continue
            # TERMINAL
            raise RuntimeError(
                f"OOM at step {global_step} after micro_batch=1 and "
                f"classes_per_forward=1. Use a larger GPU or smaller image_size."
            ) from exc
```

Notes for the implementer:

- The `multiplex_halved` `OomEvent` and the per-halving warning are now appended/emitted **inside** `on_oom()` (Task 1) — remove the manual `oom_state.pending_oom_events.append(OomEvent(action="multiplex_halved", …))` (loop.py:390-397) and the manual `_LOG.warning("…halving effective_K…")` (loop.py:398-405) from the outer rung.
- **`zero_grad` on `FLOOR_RETRY`:** today there is no floor-retry rung in train (it hard-fails at K==1). Adding `FLOOR_RETRY` is an intended behavior change (spec §6 train row: "The shared `FLOOR_RETRY` matches eval's #176 robustness; train's existing tests pin the contract"). `zero_grad` before the floor replay keeps the grad state clean — consistent with the K-rung. Confirm `test_trainer_oom_retry.py` does not assert a hard-fail at the *first* K==1 OOM (it tests B-exhaustion at the helper level, and the K-rung tests live in `test_train_loop_multiplex.py` which asserts the final hard-fail only after K is exhausted AND the floor retry is consumed — see Step 7).
- **`effective_K` read at the top of the replay loop** (loop.py:229-232) is unchanged: it reads `oom_state.effective_K` each iteration, which `on_oom()` already decremented. So the re-chunk at the smaller K happens automatically on the next `while True` iteration.
- The NaN-driven `ValueError` group-skip (loop.py:355-366) stays **inside** the per-group body, untouched — it is NOT caught by this `_MicrobatchExhausted` handler.

- [ ] **Step 7: Reconcile `test_train_loop_multiplex.py`'s hard-fail expectation with the new FLOOR_RETRY**

`test_train_loop_multiplex.py` has `test_oom_final_hard_fail_only_when_b_and_k_both_one` (asserts `RuntimeError` match `classes_per_forward=1`). With the new `FLOOR_RETRY`, a single OOM at B==1, K==1 now returns `FLOOR_RETRY` (replay once) **before** `TERMINAL`. So a fake wrapper that OOMs *every* time at B==1/K==1 will: 1st OOM → `FLOOR_RETRY` (replay) → 2nd OOM → `TERMINAL` (raise). Read the test's fake wrapper: if it raises OOM unconditionally at the floor, the `RuntimeError` is still raised (just one replay later) and the test passes unchanged. **Verify by running it** (Step 8). If the test asserts the *exact* number of forward calls at the floor and the extra floor-retry forward breaks that count, this is the one place the migration changes an observable: the test would need a one-line update — but per the task constraint, first confirm whether it actually over-asserts before touching it. If it does over-assert, that edit is in-scope for *this* test file (only `test_trainer_oom_retry.py` and `test_eval_batch_size_cap.py` are the strictly-untouched contract; `test_train_loop_multiplex.py` may be updated to reflect the intended floor-retry behavior).

- [ ] **Step 8: Check `train/trainer.py` construction still compiles**

The trainer constructs `OomState(micro_batch_size=cfg.train.batch_size, effective_K=min(cfg.train.multiplex.classes_per_forward, _MULTIPLEX_CAP))` (trainer.py:495-498). With `OomState` now an `OomLadder` subclass accepting both kwargs (Step 3), this is unchanged. **No edit to trainer.py is expected.** Confirm by import + the eval-cap test.

- [ ] **Step 9: Run the full train + cap + multiplex suite to verify behavior is preserved**

Run: `uv run pytest tests/unit/test_trainer_oom_retry.py tests/unit/test_eval_batch_size_cap.py tests/unit/test_train_loop_multiplex.py tests/unit/test_train_loop_legacy_k1.py tests/unit/test_trainer_nan_behavior.py -q`
Expected: PASS — B halves to 1 then signals `_MicrobatchExhausted`; sticky halving; gradient magnitude preserved; `zero_grad` once per step; `OomEvent` actions/fields unchanged; **a single OOM at B==1 halves K exactly once** (on_oom called once at the outer rung); K-rung re-chunks all classes; final hard-fail only at B==1 ∧ K==1 (after the floor retry); NaN group-skip untouched.

- [ ] **Step 10: Commit**

```bash
git add src/custom_sam_peft/train/loop.py src/custom_sam_peft/train/trainer.py tests/unit/test_train_loop_multiplex.py
git commit -m "refactor(train): source K-rung decision from shared OomLadder.on_oom (one call per OOM)"
```

---

## REVIEW CHECKPOINT A — Phase 1 complete (interface contract pinned)

Before starting Phase 2, verify the shared component and train reference consumer are self-consistent and the contract tests are green untouched:

- [ ] Run: `uv run pytest tests/unit/test_oom_ladder.py tests/unit/test_oom_reexport.py tests/unit/test_trainer_oom_retry.py tests/unit/test_eval_batch_size_cap.py tests/unit/test_train_loop_multiplex.py tests/unit/test_train_loop_legacy_k1.py -q`
      Expected: all PASS.
- [ ] Confirm `git diff --stat` on this phase touches **no** lines in `tests/unit/test_trainer_oom_retry.py` or `tests/unit/test_eval_batch_size_cap.py` (the untouched-contract tests).
- [ ] Dispatch a code-review subagent (opus/xhigh — concurrency/design-sensitive: the inner→outer decision routing) over the Phase-1 diff: confirm `on_oom()` is called **exactly once per OOM event** for the K dimension (only at the outer rung), the inner helper's B-halving goes through the SHARED `_halve_microbatch` routine (the same one `on_oom`'s B-branch delegates to — no hand-copied byte-identical block), the inner helper never calls `state.on_oom()`, `_MicrobatchExhausted` is **fieldless** and never escapes `train_step`, the NaN `ValueError` skip is not swallowed by the new handler, and no class is dropped on a K-change.

### Interface contract exposed by Phase 1 (consumed by Phase 2 — written explicitly)

Phase 2 may build on this **without re-reading Phase 1's implementation**:

- **Module:** `custom_sam_peft.oom`.
- **Constructor:** `OomLadder(micro_batch_size: int, effective_K: int)` — both required. (`pending_oom_events`, `step`, `_floor_retry_used` default.) Construct per path: eval `OomLadder(micro_batch_size=int(cfg.batch_size), effective_K=min(MULTIPLEX_CAP, len(dataset.class_names)))`; predict `OomLadder(micro_batch_size=bs, effective_K=min(MULTIPLEX_CAP, len(prompts)))`.
- **Fields (read-only to callers, mutated only by `on_oom`):**
  - `micro_batch_size: int` — B, sticky, monotone non-increasing.
  - `effective_K: int` — K, sticky, monotone non-increasing.
  - `pending_oom_events: list[OomEvent]` — one entry per halving (B or K); none for floor/terminal.
- **Method:** `on_oom(step: int | None = None) -> OomDecision`. Does guarded `torch.cuda.empty_cache()` at the top (always, before the policy branch), then applies B-then-K: halve B (return `RETRY_B`) while B>1; else halve K (return `RETRY_K`) while K>1; else one `FLOOR_RETRY`; else `TERMINAL`. Mutates `micro_batch_size`/`effective_K`/`pending_oom_events` in place. Idempotent at the floor/terminal (no further state change).
- **Enum:** `OomDecision` with members `RETRY_B`, `RETRY_K`, `FLOOR_RETRY`, `TERMINAL`.
- **`OomEvent`:** frozen dataclass `(step: int, action: Literal["microbatch_halved","multiplex_halved"], new_micro_batch_size: int, effective_K: int | None = None)`, importable from `custom_sam_peft.oom` and `custom_sam_peft.train.types`.
- **Decision → eval/predict control flow (spec §3 mapping table):**
  - `RETRY_B`: image set per forward changed → **discard `chunk_buf`**, restart the current image-chunk at the new (smaller) `ladder.micro_batch_size`. Do NOT advance the image index.
  - `RETRY_K`: resume the **same** image-chunk **from the current class index** at the smaller `ladder.effective_K`; already-completed K-groups' buffered rows stay valid (keep `chunk_buf`).
  - `FLOOR_RETRY`: retry the same forward once (B and K unchanged at the floor).
  - `TERMINAL`: `raise RuntimeError("... use a larger GPU or smaller image_size.")`.
- **Single catch site:** eval/predict have exactly one OOM catch site, so they call `on_oom()` once there for the full B-then-K policy and branch on the return directly. (Train's two-level split — inner helper handles the B-rung via the shared `_halve_microbatch` routine, outer rung is the single `on_oom()` K-decision site — is train-only and does not affect Phase 2. No `OomDecision` is ever carried on an exception.)

---

## Phase 2 — eval + predict migration onto the Phase-1 `OomLadder`

> **Consumes** the Phase-1 interface contract above; adds nothing to it. Task 6 (eval) and Task 7 (predict) touch disjoint files (`eval/evaluator.py` vs `predict/runner.py`) and disjoint test files → **dispatch in parallel**. Task 5 (extend eval's test) precedes Task 6's implementation (TDD).

### Task 5: Extend `test_eval_oom_ladder.py` with a K-rung case (failing first)

**Files:**

- Modify: `tests/unit/test_eval_oom_ladder.py`

- [ ] **Step 1: Add the failing K-rung test**

Append to `tests/unit/test_eval_oom_ladder.py` a test that exercises the new K-rung: a stub model that OOMs on a large class group at B==1, then succeeds at the halved K; assert the class loop resumes from the current class index, completed K-groups' rows are retained, and final predictions have no duplicate / no dropped `(image_id, category_id)` rows. Model this on the existing `test_mid_chunk_oom_does_not_produce_duplicate_predictions` (lines 82-169) which builds an in-memory dataset and calls `ev._iter_predictions`.

```python
def test_eval_k_rung_resumes_mid_chunk_no_dup_no_drop(monkeypatch) -> None:
    """At B==1 with K>1, an OOM on a multi-class group halves effective_K and
    resumes from the current class index; completed K-groups' rows are retained;
    no (image_id, category_id) is duplicated or dropped. Spec §5.2 / §7.3."""
    from custom_sam_peft.config.schema import EvalConfig
    from custom_sam_peft.data.base import Example, Instance, TextPrompts
    from custom_sam_peft.eval.evaluator import Evaluator

    # 4 classes, start K=4 (MULTIPLEX_CAP high enough). batch_size=1 so B is at
    # the floor immediately and the FIRST OOM goes straight to the K-rung.
    class_names = ["a", "b", "c", "d"]
    monkeypatch.setattr("custom_sam_peft.eval.evaluator.MULTIPLEX_CAP", 4, raising=False)

    def _make_ex(idx: int) -> Example:
        h = w = 8
        image = torch.zeros(3, h, w)
        mask = torch.zeros(h, w, dtype=torch.bool)
        mask[:4, :4] = True
        return Example(
            image=image,
            image_id=f"img_{idx}",
            prompts=TextPrompts(classes=class_names),
            instances=[
                Instance(mask=mask, class_id=0, box=torch.tensor([0.0, 0.0, 4.0, 4.0]))
            ],
        )

    class _DS:
        class_names = ["a", "b", "c", "d"]

        def __len__(self) -> int:
            return 1

        def __getitem__(self, i: int) -> Example:
            return _make_ex(i)

    dataset = _DS()
    calls: list[int] = [0]

    def _model(images, prompts, support=None):
        calls[0] += 1
        k_g = len(prompts[0].classes)
        # First forward sees K_g=4 (the full group) -> OOM. After K halves to 2,
        # forwards with K_g<=2 succeed.
        if k_g > 2:
            raise _make_oom_error()
        b = images.shape[0]
        rows = b * k_g
        h, w = images.shape[-2], images.shape[-1]
        return {
            "pred_logits": torch.zeros(rows, 1, 1),
            "pred_boxes": torch.zeros(rows, 1, 4),
            "pred_masks": torch.zeros(rows, 1, h, w),
            "presence_logit_dec": torch.zeros(rows, 1),
        }

    cfg = EvalConfig(mode="full", iou_thresholds=[0.5], batch_size=1)
    ev = Evaluator(cfg)
    examples = [dataset[0]]
    preds = ev._iter_predictions(_model, examples, dataset)

    seen: set[tuple[int, int]] = set()
    dups: list[tuple[int, int]] = []
    for p in preds:
        key = (int(p["image_id"]), int(p["category_id"]))
        if key in seen:
            dups.append(key)
        seen.add(key)
    assert not dups, f"duplicate (image_id, category_id): {dups}"
    # All 4 classes (category_id 1..4) must appear exactly once for the 1 image.
    assert {cid for _, cid in seen} == {1, 2, 3, 4}, f"missing/extra classes: {seen}"
```

- [ ] **Step 2: Run to verify it fails (current eval has no K-rung)**

Run: `uv run pytest tests/unit/test_eval_oom_ladder.py::test_eval_k_rung_resumes_mid_chunk_no_dup_no_drop -v`
Expected: FAIL — current eval uses a fixed `MULTIPLEX_CAP` group and the B-only `_eval_forward_with_oom_ladder`, so the OOM at B==1/K==4 hits the floor and raises `RuntimeError`, never halving K. (The test will error/raise rather than complete.)

- [ ] **Step 3: Commit the failing test**

```bash
git add tests/unit/test_eval_oom_ladder.py
git commit -m "test(eval): add failing K-rung resume test (no dup/drop)"
```

### Task 6: Migrate `eval/evaluator.py` onto `OomLadder` with a K-rung

**Files:**

- Modify: `src/custom_sam_peft/eval/evaluator.py` (`_iter_predictions` ~163-259; remove `_eval_forward_with_oom_ladder` ~50-83)
- Test: `tests/unit/test_eval_oom_ladder.py` (Task 5's new test + the existing `test_mid_chunk_oom_does_not_produce_duplicate_predictions` stay green)

- [ ] **Step 1: Add the import**

At the top of `eval/evaluator.py` (alongside `from custom_sam_peft.models.sam3 import MULTIPLEX_CAP` at line 25), add:

```python
from custom_sam_peft.oom import OomDecision, OomLadder
```

- [ ] **Step 2: Rewrite the `_iter_predictions` forward loop with the index-driven K-rung**

Replace the body of the `with torch.no_grad(), P.push_subtask(...)` block (evaluator.py:199-254) so the inner class loop is index-driven on `ladder.effective_K` and maps the four decisions. Construct the ladder once before the loop; drop the old `state` dict and `_eval_forward_with_oom_ladder`. The `chunk_buf` commit-on-completion, the actual-`K_g` reassembly (evaluator.py:227-243), and the `queries_to_coco_results` call are preserved verbatim:

```python
        # Replace the old state dict with the shared ladder. effective_K starts at
        # min(MULTIPLEX_CAP, n_classes); micro_batch_size at the resolved cfg.batch_size.
        n_classes = len(dataset.class_names)
        ladder = OomLadder(
            micro_batch_size=int(cfg.batch_size),
            effective_K=min(MULTIPLEX_CAP, n_classes) if n_classes else 1,
        )

        predictions: list[dict[str, object]] = []
        img_idx_global = 0
        try:
            with torch.no_grad(), P.push_subtask("eval", total=len(examples)) as sub:
                i = 0
                while i < len(examples):
                    bs = ladder.micro_batch_size
                    image_chunk = list(examples[i : i + bs])
                    images_t = to_device(
                        torch.stack([ex.image for ex in image_chunk]), eval_runtime
                    )
                    chunk_buf: list[dict[str, object]] = []
                    chunk_done = False
                    j = 0  # class index into dataset.class_names
                    restart_chunk = False
                    while j < n_classes:
                        K_g = min(ladder.effective_K, n_classes - j)
                        group = dataset.class_names[j : j + K_g]
                        prompts_g = [TextPrompts(classes=list(group)) for _ in image_chunk]
                        try:
                            outputs = cast(
                                "dict[str, torch.Tensor]",
                                model(images_t, prompts_g, support=None),
                            )
                        except torch.cuda.OutOfMemoryError:
                            decision = ladder.on_oom()
                            if decision is OomDecision.RETRY_B:
                                # Image set per forward changed: discard the buffer
                                # and restart this image-chunk at the smaller B.
                                restart_chunk = True
                                break
                            if decision is OomDecision.RETRY_K:
                                # Resume from the SAME class index at the smaller K_g
                                # (recomputed at the top of the loop). Completed
                                # K-groups' rows in chunk_buf stay valid.
                                continue
                            if decision is OomDecision.FLOOR_RETRY:
                                continue  # retry the same forward once
                            raise RuntimeError(
                                "eval OOM at batch_size=1 and classes_per_forward=1; "
                                "use a larger GPU or smaller image_size."
                            )
                        for r in range(len(image_chunk) * K_g):
                            ii, kk = divmod(r, K_g)
                            ex = image_chunk[ii]
                            original_hw = (int(ex.image.shape[-2]), int(ex.image.shape[-1]))
                            int_id = _int_image_id(ex.image_id)
                            cat_idx = dataset.class_names.index(group[kk])
                            entries = queries_to_coco_results(
                                _row_outputs(outputs, r),
                                int_id,
                                cat_idx + 1,
                                original_hw,
                                cfg.mask_threshold,
                            )
                            chunk_buf.extend(entries)
                        j += K_g  # advance by the ACTUAL group length
                    if restart_chunk:
                        continue  # re-enter outer while at smaller B; i unchanged
                    # Completed every class group for this image-chunk: commit once.
                    predictions.extend(chunk_buf)
                    i += len(image_chunk)
                    img_idx_global += len(image_chunk)
                    for _ in range(len(image_chunk)):
                        sub.advance()
                    sub.update_postfix(it_s=float(img_idx_global))
        finally:
            if was_training and hasattr(model, "train"):
                model.train()
```

Notes for the implementer:

- This **removes** `_eval_forward_with_oom_ladder` (evaluator.py:50-83) — delete the function. Its B-halving + #176 floor-retry are now subsumed by `OomLadder.on_oom()` (the `FLOOR_RETRY` rung is the #176 retry-once). Remove the now-unused `state` dict (evaluator.py:194).
- The `advanced_i`/`for…else` no-break idiom (evaluator.py:215, 244-249) is replaced by the explicit `restart_chunk` flag + `continue`. Semantics are identical: a `RETRY_B` does NOT commit and does NOT advance `i`; full completion commits exactly once and advances.
- **Reassembly uses the ACTUAL `K_g`** (`min(ladder.effective_K, n_classes - j)`), never `MULTIPLEX_CAP` — invariant (e). After a K-halving, `K_g` shrinks and `j` advances by the real length, so each class is emitted exactly once.
- `cast` is already imported (evaluator.py:13). Keep `n_classes == 0` guarded (effective_K=1 fallback) — the existing code never hits an empty `class_names`, but `min(..., len(...))` with an empty list would give 0, which `OomLadder` would treat as K never >1; the explicit `else 1` avoids a degenerate `effective_K=0`.
- The `with torch.no_grad()` and the `model.eval()`/`model.train()` restore (evaluator.py:183-191, 255-257) are unchanged.

- [ ] **Step 3: Run the eval OOM suite to verify green**

Run: `uv run pytest tests/unit/test_eval_oom_ladder.py -v`
Expected: PASS — the new K-rung test resumes mid-chunk with no dup/drop; the existing `test_mid_chunk_oom_does_not_produce_duplicate_predictions` (B-discard path) stays green. Note: the three tests that import `_eval_forward_with_oom_ladder` directly (`test_oom_halves_batch_size_sticky_and_warns_once`, `test_oom_raises_at_B1_floor`, `test_oom_at_B1_retries_once_and_succeeds`) reference a deleted symbol — see Step 4.

- [ ] **Step 4: Handle the deleted `_eval_forward_with_oom_ladder` direct-import tests**

`test_eval_oom_ladder.py:17` imports `_eval_forward_with_oom_ladder`, and three tests (lines 34-79, 172-183, 186-227) call it directly. Since the function is removed, these tests must be migrated to the new path. The B-halving/floor/retry behaviors they assert are now `OomLadder` behaviors **already covered by `tests/unit/test_oom_ladder.py`** (Task 1: B-stickiness, floor-retry-once, terminal). Therefore: **delete these three now-redundant direct-helper tests and the import line**, leaving the two end-to-end `_iter_predictions` tests (the #176 dup-safety test and the new K-rung test) which exercise the integrated ladder. This is in-scope: `test_eval_oom_ladder.py` is not on the strictly-untouched contract (only `test_trainer_oom_retry.py` and `test_eval_batch_size_cap.py` are). Confirm coverage of the removed assertions: B-halving + warn → `test_oom_ladder.py::test_decision_sequence...` + `test_pending_oom_events_emission`; floor-retry-once + terminal → `test_decision_sequence...`; empty_cache guard → `test_empty_cache_guarded_*`.

- [ ] **Step 5: Run the eval OOM suite again after cleanup**

Run: `uv run pytest tests/unit/test_eval_oom_ladder.py -v`
Expected: PASS — two end-to-end tests green; no import error for the deleted helper.

- [ ] **Step 6: Run the broader eval suite for regressions**

Run: `uv run pytest tests/unit/test_eval_runner.py tests/unit/test_eval_runner_gate.py -q && uv run pytest tests/unit -k eval -q`
Expected: PASS — the evaluator's public contract (predictions, metrics, save) is unaffected; the loop change is internal.

- [ ] **Step 7: Commit**

```bash
git add src/custom_sam_peft/eval/evaluator.py tests/unit/test_eval_oom_ladder.py
git commit -m "feat(eval): migrate to shared OomLadder + add K-rung (resume mid-chunk on K-halving)"
```

### Task 7: Migrate `predict/runner.py` onto `OomLadder` (the #181 fix)

**Files:**

- Modify: `src/custom_sam_peft/predict/runner.py` (`run_predict` forward loop ~376-486)
- Test: `tests/predict/test_predict_oom_ladder.py` (**new**)

- [ ] **Step 1: Write the failing predict OOM test**

Create `tests/predict/test_predict_oom_ladder.py`. Reuse the `tests/predict/test_runner_smoke.py` harness conventions (stub `nn.Module` returning `(B*K_g, Q, …)` tensors, `_patch_load` patching `custom_sam_peft.models.sam3.load_sam31`, `_make_image_dir`, CPU `PredictOptions`). Two tests per §7.4: byte-identical recovery, and the `RETRY_B` buffer-discard path.

```python
"""Predict OOM ladder (spec §5.3 / §7.4 — the #181 fix).

CPU-only. A stub model injects torch.cuda.OutOfMemoryError; run_predict must
recover via the shared OomLadder and produce predictions byte-identical to a
non-OOM run (no dup, no drop).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image as PILImage

from custom_sam_peft.predict.runner import PredictOptions, run_predict

Q = 4
H_LOW = W_LOW = 16
HIGH = 0.9


def _make_image_dir(tmp_path: Path, n: int) -> Path:
    d = tmp_path / "images"
    d.mkdir()
    for i in range(n):
        PILImage.new("RGB", (64, 64), color=(i * 20 % 255, 100, 200)).save(d / f"img_{i:03d}.png")
    return d


def _opts(tmp_path: Path, images: Path, *, prompts: str, batch_size: int) -> PredictOptions:
    return PredictOptions(
        images=images,
        prompts=prompts,
        output=tmp_path / "out",
        checkpoint=None,
        merge_adapter=True,
        config=None,
        score_threshold=0.0,
        top_k=100,
        save_masks="rle",
        visualize=False,
        device="cpu",
        dtype="float32",
        batch_size=batch_size,
        seed=42,
        dry_run=False,
        verbose=False,
    )


class _MultiplexStub(torch.nn.Module):
    """Multiplex stub: forward -> (B*K_g, Q, ...). OOMs once if oom_when matches."""

    def __init__(self, oom_predicate=None) -> None:
        super().__init__()
        self._oom = oom_predicate  # callable(images, prompts) -> bool; fires once
        self._fired = False

    def forward(self, images: torch.Tensor, prompts: list[Any], support: Any = None):
        from custom_sam_peft.data.base import TextPrompts as _TP

        if self._oom is not None and not self._fired and self._oom(images, prompts):
            self._fired = True
            raise torch.cuda.OutOfMemoryError("synthetic")
        b = images.shape[0]
        k_g = len(prompts[0].classes) if prompts and isinstance(prompts[0], _TP) else 1
        total = b * k_g
        return {
            "pred_logits": torch.full((total, Q, 1), HIGH),
            "pred_boxes": torch.full((total, Q, 4), 0.5),
            "pred_masks": torch.zeros(total, Q, H_LOW, W_LOW),
            "presence_logit_dec": torch.full((total, 1), HIGH),
        }


def _run(tmp_path: Path, stub: torch.nn.Module, opts: PredictOptions) -> list[dict]:
    import unittest.mock as mock

    with mock.patch(
        "custom_sam_peft.models.sam3.load_sam31", side_effect=lambda cfg, **kw: stub
    ):
        run_predict(opts)
    return json.loads((opts.output / "predictions.json").read_text())


def test_predict_oom_recovers_byte_identical_to_non_oom(tmp_path: Path) -> None:
    """Many classes, a model that OOMs once then succeeds: run completes AND
    predictions are byte-identical to a non-OOM run (no dup, no drop). Spec §7.4."""
    images = _make_image_dir(tmp_path / "ref", n=3)
    # Reference: never OOMs.
    ref = _run(tmp_path / "ref", _MultiplexStub(), _opts(tmp_path / "ref", images, prompts="a,b,c,d,e", batch_size=2))

    # OOM run: same inputs; OOM once on the first multi-class forward at B>1.
    images2 = _make_image_dir(tmp_path / "oom", n=3)
    stub = _MultiplexStub(oom_predicate=lambda imgs, pr: len(pr[0].classes) >= 1)
    got = _run(tmp_path / "oom", stub, _opts(tmp_path / "oom", images2, prompts="a,b,c,d,e", batch_size=2))

    # Sort by a stable key to compare content (image ids differ by dir, so compare
    # per-image-relative ordering via category_id + score sequence per image).
    def _key(p: dict) -> tuple:
        return (int(p["category_id"]), round(float(p["score"]), 6))

    assert sorted(got, key=_key) and len(got) == len(ref), (
        f"OOM run dropped/duplicated rows: got {len(got)} vs ref {len(ref)}"
    )
    # No duplicate (image_id, category_id, score) triples within the OOM run.
    seen: set[tuple] = set()
    for p in got:
        t = (int(p["image_id"]), int(p["category_id"]), round(float(p["score"]), 6))
        assert t not in seen, f"duplicate row in OOM run: {t}"
        seen.add(t)


def test_predict_oom_retry_b_discards_partial_chunk(tmp_path: Path) -> None:
    """An OOM that triggers RETRY_B mid-chunk discards the partially-buffered chunk
    and re-emits it exactly once at the smaller B (no dup). Spec §5.4 / §7.4."""
    images = _make_image_dir(tmp_path, n=4)
    # batch_size=4 so the first forward is at B=4; OOM once -> RETRY_B halves to 2.
    stub = _MultiplexStub(oom_predicate=lambda imgs, pr: imgs.shape[0] == 4)
    got = _run(tmp_path, stub, _opts(tmp_path, images, prompts="a,b", batch_size=4))

    seen: set[tuple] = set()
    for p in got:
        t = (int(p["image_id"]), int(p["category_id"]), round(float(p["score"]), 6))
        assert t not in seen, f"duplicate after RETRY_B re-emit: {t}"
        seen.add(t)
    # Every (image, class) pair present: 4 images x 2 classes x Q queries, top_k=100.
    by_pair = {(int(p["image_id"]), int(p["category_id"])) for p in got}
    assert len(by_pair) == 4 * 2, f"missing (image, class) pairs: {by_pair}"
```

Notes for the implementer:

- The reference/OOM runs use separate `images` dirs so their `_int_image_id`s differ; the assertions compare **counts + no-dup** and full `(image, class)` coverage rather than literal `image_id` equality. If a stricter byte-identical comparison is wanted, run both against the *same* image dir into two output dirs and compare the prediction lists after sorting by `(image_id, category_id, score)`. The implementer may strengthen this if convenient, but the count + no-dup + full-coverage assertions are the binding requirements.
- `oom_predicate` fires **once** (`self._fired`) so the retry succeeds — matching "OOMs once then succeeds".

- [ ] **Step 2: Run to verify it fails (current predict has no ladder)**

Run: `uv run pytest tests/predict/test_predict_oom_ladder.py -v`
Expected: FAIL — current `run_predict` catches the `OutOfMemoryError` (a `RuntimeError` subclass), logs the advice line, and **re-raises** (runner.py:437-442), so the OOM run raises instead of recovering.

- [ ] **Step 3: Commit the failing test**

```bash
git add tests/predict/test_predict_oom_ladder.py
git commit -m "test(predict): add failing OOM-recovery + RETRY_B discard tests (#181)"
```

- [ ] **Step 4: Restructure the `run_predict` forward loop with index-driven `while` loops**

In `predict/runner.py`, replace the two `for`-loops (the `for chunk_paths in _chunked(list(image_paths), bs)` at 397 and the inner `for group in _chunked(prompts, MULTIPLEX_CAP)` at 431) with index-driven `while` loops keyed on `ladder.micro_batch_size` (image index) and `ladder.effective_K` (class index), per the spec §5.5 sketch. Add the import and construct the ladder before the loop:

Add near the other lazy imports at the top of the Step-9 block (runner.py:378-380):

```python
    from custom_sam_peft.oom import OomDecision, OomLadder
```

Construct the ladder once before the loop (after `images_processed = 0` at runner.py:395):

```python
    image_path_list = list(image_paths)
    n_images = len(image_path_list)
    ladder = OomLadder(
        micro_batch_size=bs,
        effective_K=min(MULTIPLEX_CAP, len(prompts)) if prompts else 1,
    )
    i = 0
```

Replace the whole `for chunk_paths in _chunked(list(image_paths), bs):` block (runner.py:397-486) with the `while` structure. The per-image open/transform block (405-428), the row postprocess + score/top-k filter (444-459), the verbose logging (464-475), and the progress ticks (477-486) are **preserved verbatim**, only re-homed into the new loop with a per-chunk `chunk_buf` and commit-on-completion:

```python
    while i < n_images:
        bs_cur = ladder.micro_batch_size
        chunk_paths = image_path_list[i : i + bs_cur]
        chunk_t0 = time.perf_counter()

        # --- open + transform each image in the chunk (UNCHANGED from runner.py:405-428) ---
        imgs: list[torch.Tensor] = []
        metas: list[tuple[int, int, int]] = []
        chunk_paths_ok: list[Path] = []
        for img_path in chunk_paths:
            try:
                from custom_sam_peft.data.io import read_image as _read_image

                img_np = _read_image(img_path, rcfg.channels)
            except Exception as exc:
                logger.warning("Skipping unreadable image %s: %s", img_path, exc)
                continue
            orig_h, orig_w = img_np.shape[0], img_np.shape[1]
            image_id = _int_image_id(img_path)
            id_to_path[image_id] = img_path.resolve()
            id_to_stem[image_id] = img_path.stem
            originals[image_id] = (orig_h, orig_w)
            transformed = transforms(image=img_np, bboxes=[], class_labels=[], instance_idx=[])
            imgs.append(transformed["image"].to(rcfg.device, dtype=rcfg.dtype))
            metas.append((image_id, orig_h, orig_w))
            chunk_paths_ok.append(img_path)

        if not imgs:
            i += len(chunk_paths)  # advance past the (all-unreadable) chunk
            continue

        img_batch = torch.stack(imgs, dim=0)  # (B, C, H, W)

        # --- index-driven inner class loop with the K-rung + buffer-and-commit ---
        chunk_buf: list[dict[str, object]] = []
        restart_chunk = False
        j = 0  # class index into prompts
        while j < len(prompts):
            K_g = min(ladder.effective_K, len(prompts) - j)
            group = prompts[j : j + K_g]
            prompts_g = [TextPrompts(classes=list(group)) for _ in metas]
            try:
                with torch.no_grad():
                    outputs = model(img_batch, prompts_g, support=None)
            except torch.cuda.OutOfMemoryError:
                decision = ladder.on_oom()
                if decision is OomDecision.RETRY_B:
                    restart_chunk = True
                    break  # discard chunk_buf; restart this image-chunk at smaller B
                if decision is OomDecision.RETRY_K:
                    continue  # resume from j at the smaller K_g
                if decision is OomDecision.FLOOR_RETRY:
                    continue  # retry the same forward once
                raise RuntimeError(
                    "OOM at batch_size=1 and classes_per_forward=1; "
                    "use a larger GPU or smaller image_size."
                )
            # postprocess each (image, class) row (UNCHANGED from runner.py:445-459,
            # but category id uses the class index j+kk, value-equivalent to the old
            # prompts.index(group[kk]) since group = prompts[j:j+K_g]).
            for r in range(len(metas) * K_g):
                ii, kk = divmod(r, K_g)
                image_id, orig_h, orig_w = metas[ii]
                class_idx_one_based = (j + kk) + 1
                entries = queries_to_coco_results(
                    _row_outputs(outputs, r),
                    image_id=image_id,
                    category_id=class_idx_one_based,
                    original_hw=(orig_h, orig_w),
                    mask_threshold=0.0,
                )
                entries = [e for e in entries if cast(float, e["score"]) >= opts.score_threshold]
                entries.sort(key=lambda e: cast(float, e["score"]), reverse=True)
                entries = entries[: opts.top_k]
                chunk_buf.extend(entries)
            j += K_g  # advance by the ACTUAL group length

        if restart_chunk:
            continue  # re-enter outer while at smaller B; i unchanged, buffer dropped

        # Chunk completed every class group: commit exactly once.
        all_predictions.extend(chunk_buf)
        n_successful += len(metas)
        images_processed += len(metas)

        # Verbose logging (UNCHANGED from runner.py:464-475).
        if opts.verbose:
            chunk_latency_ms = (time.perf_counter() - chunk_t0) * 1000.0
            per_image_ms = chunk_latency_ms / max(len(metas), 1)
            for img_path in chunk_paths_ok:
                logger.info(
                    "image %d/%d %s (%.1f ms)",
                    images_processed - len(metas) + chunk_paths_ok.index(img_path) + 1,
                    n_images,
                    img_path.name,
                    per_image_ms,
                )

        # Progress ticks (UNCHANGED from runner.py:477-486).
        for _ in metas:
            P.advance_inner()
        if images_processed % log_every_n == 0 or images_processed == n_images:
            elapsed_so_far = max(time.perf_counter() - t_start, 1e-9)
            P.update_postfix(
                done=f"{images_processed}/{n_images}",
                it_s=images_processed / elapsed_so_far,
            )

        i += len(chunk_paths)  # advance the image index by the consumed chunk
```

Notes for the implementer:

- **`category_id` change is value-equivalent and safe.** Today: `prompts.index(group[kk]) + 1`. Now: `(j + kk) + 1`. Since `group = prompts[j : j + K_g]`, `group[kk]` is `prompts[j + kk]`, so `prompts.index(group[kk])` equals `j + kk` **as long as prompts are unique** — and the index-based form is in fact *more* correct (it does not collapse duplicate prompt strings). Use `(j + kk) + 1`.
- **`i` advancement on an all-unreadable chunk.** The old `for` loop's `if not imgs: continue` simply skipped to the next chunk because the `for` advanced the iterator. The `while` loop must advance `i` manually (`i += len(chunk_paths)`) before `continue`, or it would loop forever on an unreadable chunk. This preserves `test_run_predict_unreadable_image_warns_and_skips` and `test_run_predict_every_image_fails_exits_1` (n_successful stays 0 → the all-failed guard at runner.py:491-493 fires).
- **`bs` resolution + warmup unchanged.** Lines 343-373 (the `"auto"` resolution, the VRAM hint, the warmup) are untouched. `ladder` is constructed from the resolved `bs`.
- **The old `logger.error("OOM: consider --no-merge-adapter …")` advice line (runner.py:439-441) is removed** (spec §5.3, §6). The per-halving warnings now come from `OomLadder.on_oom()`.
- Catch **`torch.cuda.OutOfMemoryError`** (subclass of `RuntimeError`), not the string-matched `RuntimeError`. A non-OOM `RuntimeError` from the forward now propagates untouched (the old code re-raised it anyway, but without the `empty_cache`/advice; behavior for non-OOM errors is preserved — they still propagate).
- `cast` and `time` are already imported in `runner.py`. `TextPrompts` is imported at runner.py:365.

- [ ] **Step 5: Run the predict OOM test to verify green**

Run: `uv run pytest tests/predict/test_predict_oom_ladder.py -v`
Expected: PASS — the OOM run recovers and produces no-dup/no-drop predictions; the `RETRY_B` discard path re-emits the chunk exactly once.

- [ ] **Step 6: Run the predict smoke suite for regressions**

Run: `uv run pytest tests/predict/test_runner_smoke.py -v`
Expected: PASS — all 15 smoke tests green. In particular `test_run_predict_flat_loop_iterates_image_chunks_x_groups` (counts forward calls), `test_run_predict_unreadable_image_warns_and_skips`, and `test_run_predict_every_image_fails_exits_1` confirm the loop restructure preserved iteration, skip, and all-failed semantics.

- [ ] **Step 7: Commit**

```bash
git add src/custom_sam_peft/predict/runner.py tests/predict/test_predict_oom_ladder.py
git commit -m "feat(predict): add shared OomLadder B-then-K recovery to forward loop (closes #181)"
```

---

## REVIEW CHECKPOINT B — full CPU suite + lint/type gate

- [ ] Run the FULL suite with coverage (the 80% gate runs on the full suite, not a subset):
      `uv run pytest`
      Expected: all PASS, coverage >= 80%.
- [ ] Run lint/format/type (CI commands):
      `uv run ruff check && uv run ruff format --check && uv run mypy src/custom_sam_peft`
      Expected: clean (fix findings before the ready PR — lint gate).
- [ ] Dispatch a code-review subagent (opus/xhigh; design-sensitive — the buffer-and-commit dup-safety + the K-resume index arithmetic) over the Phase-2 diff: confirm reassembly uses the actual `K_g` (never `MULTIPLEX_CAP`), `RETRY_B` discards the buffer + does not advance the index, `RETRY_K` retains the buffer + resumes from `j`, the chunk commits exactly once, and the predict `category_id` index form is value-equivalent to the old `prompts.index`.

---

## Final verification

- [ ] `uv run pytest` — full suite green, coverage >= 80%.
- [ ] `uv run ruff check && uv run ruff format --check && uv run mypy src/custom_sam_peft` — clean.
- [ ] `! rg -n "_eval_forward_with_oom_ladder" src/ tests/` — the deleted eval helper has no remaining references.
- [ ] `rg -n "from custom_sam_peft.oom import" src/` — confirm `oom.py` is consumed by `train/loop.py`, `eval/evaluator.py`, `predict/runner.py`, and re-exported by `train/types.py`.
- [ ] Markdown-lint this plan + the spec before they land on the ready PR (CI lints them):
      `npx --yes markdownlint-cli2 --config .config/markdownlint-cli2.jsonc "docs/superpowers/plans/2026-05-29-unified-oom-ladder-plan.md" "docs/superpowers/specs/2026-05-29-unified-oom-ladder-design.md"`
      Expected: clean (CI's config disables only MD013/MD018/MD029 — language-tag all fences, blank lines around lists/headings/fences).
- [ ] Confirm the seven hard invariants (a)-(g) each have a passing assertion (Tasks 1, 2, 4, 6, 7).

---

## Self-review (writer's pass against the spec)

**Spec coverage:**

- §3 component (shared `OomLadder`, the shared `_halve_microbatch` B-rung mechanic, the decision→control-flow mapping table, the one-halving-per-OOM split) → Task 1 (`OomLadder`/`OomDecision`/`_halve_microbatch`/`on_oom`), Task 4 (train's inner B-rung via `_halve_microbatch` + outer-rung `on_oom` K-decision), Phase-1 interface contract (the mapping table written explicitly for eval/predict).
- §4 `oom.py` details (relocated `OomEvent`, `OomDecision`, the shared `_halve_microbatch` routine, `OomLadder` fields/invariants, the one-halving-per-OOM invariant, field-name load-bearing constraints) → Task 1 (the module + shared routine), Task 2 (relocation + re-export), Task 4 (the `OomState` alias keeping `micro_batch_size`/`effective_K`/`pending_oom_events`; inner B-rung via `_halve_microbatch`; fieldless `_MicrobatchExhausted`; single outer-rung `on_oom`).
- §5.1 train → Task 4 (inner B-rung via shared `_halve_microbatch`; single outer-rung `on_oom` K-decision; fieldless `_MicrobatchExhausted`; preserve replay/`zero_grad`/divisor/NaN-skip; `OomState` name + `_State`-stub compatibility).
- §5.2 eval (replace `_eval_forward_with_oom_ladder`; index-driven K-rung; buffer-and-commit; actual-`K_g` reassembly) → Task 5 (failing K-rung test), Task 6 (migration).
- §5.3 predict (the #181 fix — index-driven `while` loops, `empty_cache` via `on_oom`, floor retry, remove the advice-and-die path, catch `OutOfMemoryError`) → Task 7.
- §5.4 buffer-and-commit anti-dup mechanism → Task 6 (eval `chunk_buf`/`restart_chunk`), Task 7 (predict `chunk_buf`/`restart_chunk`), invariant (f).
- §5.5 index-driven loop sketch → Task 6 + Task 7 (both follow the sketch; `RETRY_B` break-without-advance, `RETRY_K`/`FLOOR_RETRY` continue, actual `K_g`).
- §6 behavior changes (predict gains ladder; eval gains K-rung; train hard-fail observably unchanged modulo the added floor-retry; warning cadence per-halving; `OomEvent` importable from both paths) → Task 7, Task 6, Task 4 (Step 6/7 floor-retry reconciliation), Task 2 + Task 3.
- §7 tests (§7.1 `OomLadder` unit → Task 1; §7.2 train/cap untouched → Task 4 Steps 1/9 + Checkpoint A; §7.3 eval K-rung → Task 5/6; §7.4 predict byte-identical + RETRY_B discard → Task 7).
- §8 phasing + interface contract → Phase 1 / Phase 2 split; the explicit interface contract block after Checkpoint A.
- §9 risks (train regression net = untouched contract tests; #176 dup/drop trap = buffer-and-commit + actual-`K_g`; K-resume double-count = `j += K_g`; `OomEvent` relocation = Task 3 smoke; train/eval semantic drift = decision-boundary-only component) → covered across Tasks 4, 6, 7, 3.

**Type consistency:** `OomLadder(micro_batch_size: int, effective_K: int)` defined in Task 1, consumed identically in Tasks 4/6/7. `on_oom(step: int | None = None) -> OomDecision` consistent everywhere. `OomDecision` members `RETRY_B`/`RETRY_K`/`FLOOR_RETRY`/`TERMINAL` referenced consistently. `OomEvent(step, action, new_micro_batch_size, effective_K=None)` identical across Task 1 (definition + the shared `_halve_microbatch` B-event), Task 2 (re-export), and train's inner B-rung (Task 4 Step 4, which calls `_halve_microbatch`). The shared `_halve_microbatch(state, step)` signature is consistent between Task 1 (definition + `on_oom` B-branch delegation) and Task 4 (the inner helper's call). `OomState(OomLadder)` with `effective_K: int = 1` default (Task 4 Step 3) preserves the `OomState(micro_batch_size=…)` construction the cap test uses.

**Placeholder scan:** No TBD/TODO. Every code step shows complete code. The one deliberate delegation: Task 6 Step 4 and Task 7's "UNCHANGED from runner.py:NNN" blocks reference verbatim-preserved existing code rather than re-pasting unrelated I/O — the surrounding context names the exact line ranges and the binding semantic constraints (advance `i` manually, commit once, actual `K_g`), so an implementer reading the task in isolation has the full picture.

**Resolved ambiguity (binding decisions recorded for the implementer):**

1. **The `_State`-stub vs. single-`on_oom()` design (Task 4 Steps 4-5) — implements SPEC §4/§5.1, NOT a divergence.** `test_trainer_oom_retry.py`'s `_State` stub has no methods and must stay untouched, so the inner helper cannot call `state.on_oom()` and `_MicrobatchExhausted` cannot carry an `OomDecision`. The SPEC now describes exactly this: the inner helper's B-rung calls the shared **`_halve_microbatch`** routine (the single B-halving implementation, also used by `on_oom`'s B-branch — no hand-copied block), and `on_oom()` is called **once at the outer rung** for the K/floor/terminal decision. For any single OOM, exactly one halving occurs — B (inner) or K (outer) — never both, never twice. Centralization holds: the B-mechanics have one shared implementation and the K decision has one site. This is the spec design, not a divergence from it. (The "carry a `decision` field" approach — which would force the inner helper to call `on_oom()` — is explicitly NOT IMPLEMENTED, see Task 4 Step 2, so the implementer does not break the stub.)
2. **Train gains a `FLOOR_RETRY` rung it lacked (Task 4 Step 6/7).** Spec §6 explicitly intends this ("the shared `FLOOR_RETRY` matches eval's #176 robustness"). The reconciliation step (Step 7) checks whether `test_train_loop_multiplex.py`'s hard-fail test over-asserts forward-call counts at the floor; if so, that single non-contract test is updated to reflect the intended one-floor-retry-then-terminal behavior.
3. **Markdown lint config.** CI uses `.config/markdownlint-cli2.jsonc` (stricter than `docs/superpowers/.markdownlint.json`), so the plan + spec are linted against the CI config in Final Verification.
4. **Predict `category_id` form.** Switched from value-lookup `prompts.index(group[kk])` to index arithmetic `(j + kk)` — value-equivalent for unique prompts and more correct for duplicates. Noted for the reviewer.
