# Unified OOM ladder: one shared B-then-K component across train, eval, predict

**Issue:** [#181 — predict: OOM with multiple classes — no microbatch/K-halving ladder in prediction path](https://github.com/NguyenJus/custom-sam-peft/issues/181)
**Release:** pre-1.0 minor bump (predict gains a recovery ladder; eval gains a K-rung; an internal symbol relocates with a back-compat re-export → MINOR).
**Status:** locked design, single feature, two implementation phases.

Multiplex prediction OOMs on a 24 GiB GPU when run with multiple classes; dropping the class count to **1** makes it succeed. SAM 3.1's multiplex forward materializes per-class mask/box decoder activations sized `~per_example × B × K`, so peak memory scales with the per-group class count **K**. The prediction path has **zero** OOM mitigation today: `run_predict`'s forward loop catches the `RuntimeError`, logs one advice line, and re-raises (`src/custom_sam_peft/predict/runner.py:437-442`). Training already has a two-rung B-then-K ladder; eval has a B-only ladder; prediction has none.

This spec unifies OOM handling across all three paths behind **one shared component** that owns ladder STATE and the halving DECISION, with **B-first-then-K** ordering applied uniformly. The change fixes predict (no ladder today), completes eval (gains the missing K-rung), and centralizes train's inline halving onto the shared component without altering train's externally observable behavior.

---

## §1 Scope and non-goals

### In scope

| File | Phase | Change |
|------|-------|--------|
| `src/custom_sam_peft/oom.py` | 1 | **New.** Owns `OomLadder` (sticky monotonic B/K state), `OomDecision` enum, and the relocated `OomEvent`. Pure state + decision; nothing about any caller's loop. |
| `src/custom_sam_peft/train/types.py` | 1 | `OomEvent` moves to `oom.py`; this module **re-exports** it for back-compat (existing `from custom_sam_peft.train.types import OomEvent` keeps working). |
| `src/custom_sam_peft/train/loop.py` | 1 | Train's B-halving and K-halving SOURCE their mechanics from the shared `oom.py` routines instead of bespoke inline code: the inner helper's B-rung calls the shared module-level `_halve_microbatch(state, step)` (operates only on `state` *fields*, so the field-only `_State` stub keeps working — no `state.on_oom()` call), and the outer rung's K-rung is the single `OomLadder.on_oom()` call. Control flow (replay, `zero_grad`, `_MicrobatchExhausted`) is preserved; `_MicrobatchExhausted` stays **fieldless** (it carries no `OomDecision` — the carried-decision idea was dropped as infeasible against the untouched field-only `_State` stub; §5.1, §4 invariant). |
| `src/custom_sam_peft/train/trainer.py` | 1 | Constructs the ladder from `train.batch_size` and `train.multiplex.classes_per_forward` (today builds `OomState` — preserved as a name; see §4); reads `pending_oom_events` for telemetry; the eval-batch cap reads the ladder's `micro_batch_size`. |
| `src/custom_sam_peft/eval/evaluator.py` | 2 | `_eval_forward_with_oom_ladder` is replaced by the shared `OomLadder`; the index-driven `while` loop gains a **K-rung** (resume mid-chunk from the current class index on a K-halving). The existing buffer-and-commit anti-dup mechanism is preserved and extended to the K dimension. |
| `src/custom_sam_peft/predict/runner.py` | 2 | The bare `RuntimeError` re-raise (lines 437-442) is replaced by the shared `OomLadder` driving an index-driven `while` loop with per-image-chunk prediction buffering and the row-reassembly fix. |
| Tests (see §7) | 1, 2 | New `OomLadder` unit tests; extend `test_eval_oom_ladder.py` for the K-rung; new predict OOM test (byte-identical predictions vs a non-OOM run, including the buffer-discard-on-B-change path). `test_trainer_oom_retry.py` stays green untouched. |

### Out of scope (one line each)

- **No CLI or config surface changes beyond what already exists.** Predict keeps `--batch-size`/`--prompts`; predict gains **no** `classes_per_forward` config (it remains train-only). The ladder's initial K for eval/predict is derived at runtime, not configured.
- **No change to the model forward.** `_Sam3ImageAdapter.forward`, `MULTIPLEX_CAP`, and the multiplex activation shape are untouched. The ladder shrinks the per-group K *caller-side* by re-chunking; it never alters how the model packs a group.
- **No GPU-required tests.** All new tests inject a synthetic `torch.cuda.OutOfMemoryError` and run on CPU, matching the existing `test_trainer_oom_retry.py` / `test_eval_oom_ladder.py` pattern.
- **No change to train's numerics or control flow.** Train keeps `_MicrobatchExhausted`, whole-step replay, `optimizer.zero_grad(set_to_none=True)`, the `/(G · grad_accum_steps)` divisor, the NaN-driven group-skip, and the `multiplex_halved` whole-step replay semantics. `test_trainer_oom_retry.py` is the behavioral contract.
- **No new env vars.** The VRAM headroom override and `decide_eval_batch_size` are unchanged.
- **No lifting of `MULTIPLEX_CAP = 16`.** The ladder only ever shrinks K below the cap, never above it.

---

## §2 Current state (verified) and the gap

The codebase is further along than issue #181's body implies. Verified by reading the source:

- **Train (`train/loop.py`)** already has the full two-rung, gradient-aware ladder: `_MicrobatchExhausted` (42-47), `OomState` (63-80) with both `micro_batch_size` and `effective_K`, `_train_step_with_oom_ladder` (83-137) with inner B-halving and `empty_cache()`, and the outer K-rung in `train_step` (379-406) that `zero_grad`s and replays the whole step. Both B and K are sticky (only decrease). It emits `OomEvent(action="microbatch_halved" | "multiplex_halved")`.
- **Eval (`eval/evaluator.py`)** already has an index-driven `while i < len(examples)` loop (201-254), per-image-chunk buffering (`chunk_buf`, committed only in the `for…else` no-break branch, 214-247), and the `_row_outputs` row-reassembly helper (39-47). But `_eval_forward_with_oom_ladder` (50-83) is **B-only** — there is no K-rung. The #176 dup/drop regression test already exists (`test_eval_oom_ladder.py::test_mid_chunk_oom_does_not_produce_duplicate_predictions`).
- **Predict (`predict/runner.py`)** has the flat `(image-chunk × class-group)` loop (397-487) and per-row reassembly using the **actual** sub-group length `K_g` (445-459), but the OOM catch (437-442) only logs `logger.error(...)` advice and re-raises — **no halving, no `empty_cache`, no recovery**.

So the genuine deltas this spec lands are:

1. **Centralization.** Train, eval, and predict each inline their own halving today. Extract STATE + DECISION into one `OomLadder` so the three callers share one tested policy.
2. **Eval K-rung.** Eval's latent K-OOM (a single large class group at B=1) currently hard-fails; it gains a K-rung.
3. **Predict ladder.** Predict gets the full B-then-K ladder with `empty_cache()` and a floor retry, replacing the advice-and-die path. This is the headline fix for #181.

---

## §3 Architecture

### The shared component (`src/custom_sam_peft/oom.py`)

`OomLadder` owns **state and the halving decision only**. It knows nothing about microbatches, image chunks, class groups, gradients, or replay — those are caller concepts. It exposes:

- A constructor taking the per-path initial `(micro_batch_size, effective_K)`.
- Sticky, monotonically decreasing fields `micro_batch_size` (B) and `effective_K` (K).
- A `pending_oom_events` list of `OomEvent` recording each halving for telemetry (the name `train/trainer.py` and `test_trainer_oom_retry.py` already read).
- A single method `on_oom(step) -> OomDecision` that applies the B-then-K policy.

```text
┌──────────────────────────── oom.py (Phase 1) ─────────────────────────────┐
│  OomEvent          — relocated from train/types.py; re-exported there      │
│  OomDecision       — enum: RETRY_B | RETRY_K | FLOOR_RETRY | TERMINAL       │
│  _halve_microbatch — shared field-only B-rung mechanic (on_oom + train use)│
│  OomLadder         — sticky monotone state (B, K), pending_oom_events, on_oom│
└────────────────────────────────────────────────────────────────────────────┘
        ▲                         ▲                          ▲
        │ sources B/K + decision  │ sources B/K + decision   │ sources B/K + decision
        │                         │                          │
   train/loop.py            eval/evaluator.py          predict/runner.py
   (replay + zero_grad)     (while-loop + buffer)      (while-loop + buffer)
   Phase 1 reference        Phase 2                    Phase 2
```

### `OomLadder.on_oom()` policy (the single source of truth)

`on_oom(step: int | None = None) -> OomDecision`:

1. Call `torch.cuda.empty_cache()`, guarded by `torch.cuda.is_available()`.
2. **B-then-K**, in order:
   1. `if B > 1`: call the shared `_halve_microbatch(self, step)` routine (the single B-rung mechanic — halves `B`, appends `OomEvent(step, action="microbatch_halved", new_micro_batch_size=B)`, logs one warning); return `RETRY_B`.
   2. `elif K > 1`: `K //= 2`; append `OomEvent(step, action="multiplex_halved", new_micro_batch_size=B, effective_K=K)`; log one warning; return `RETRY_K`.
   3. `elif not _floor_retry_used`: set `_floor_retry_used = True`; return `FLOOR_RETRY`.
   4. `else`: return `TERMINAL`.

The ladder is agnostic to control-flow meaning. The four decisions are mapped to per-path control flow by the callers, per the table below.

### Decision → control-flow mapping

| Decision | Train | Eval / Predict |
|----------|-------|----------------|
| `RETRY_B` | Redo microbatches at the new (smaller) `B` via the existing replay path. | Discard the current image-chunk's **buffered** predictions and redo the chunk at the smaller `B` (the image set per forward changed). |
| `RETRY_K` | `optimizer.zero_grad(set_to_none=True)` + replay the whole step at the smaller K-group size (existing). | Resume the **same** image-chunk at the smaller K-group size **from the current class index**; already-completed K-groups stay valid (their buffered rows remain). |
| `FLOOR_RETRY` | Retry the forward once. | Retry the forward once. |
| `TERMINAL` | `raise RuntimeError(...)` advising a larger GPU / smaller `image_size`. | `raise RuntimeError(...)` advising a larger GPU / smaller `image_size`. |

The two `RETRY_*` rows differ between train and eval/predict because train accumulates **gradients** (a K-change must discard partial grads and replay from group 0, since groups share adapter buffers), whereas eval/predict accumulate **independent per-row predictions** (a K-change can resume mid-chunk because completed K-groups produced final, independent rows). This asymmetry is the crux of why the shared component stops at the decision boundary and leaves control flow to the callers.

**One halving per OOM event — shared B-mechanics, single K decision site.** The B-rung *mechanics* (the `state.micro_batch_size //= 2` + `OomEvent(action="microbatch_halved", …)` append + one warning) have a **single shared implementation**: the module-level `_halve_microbatch(state, step)` routine in `oom.py` (§4). `OomLadder.on_oom()`'s B-branch delegates to it, and train's inner helper calls it directly — there is no second, hand-copied B-halving. The routine operates only on the object's *fields* (`micro_batch_size`, `pending_oom_events`, `step`), so it works on train's field-only `_State` stub without any method call on `state`.

Train is the only path whose catch site and decision-handler live at different levels: the inner helper (`_train_step_with_oom_ladder`) catches the exception and handles the B-rung; a K-rung (`RETRY_K`/`FLOOR_RETRY`/`TERMINAL`) is acted on by the outer rung in `train_step`. So train splits the work across the two levels:

- **Inner helper (B-rung):** on OOM, if `micro_batch_size > 1` it calls the shared `_halve_microbatch(state, step)` and `continue`s the microbatch loop. It does **not** call `state.on_oom()` (the `_State` stub has no such method). At `micro_batch_size == 1` it raises a **fieldless** `_MicrobatchExhausted` (no carried `OomDecision`).
- **Outer rung (K-rung):** the `except _MicrobatchExhausted` handler calls `oom_state.on_oom(step)` **exactly once** for this OOM event. At B == 1 `on_oom()` skips its B-branch and advances the K-rung, returning `RETRY_K` / `FLOOR_RETRY` / `TERMINAL`.

For any single OOM, exactly one halving occurs — B (inner helper, via the shared routine) **or** K (outer rung, via the single `on_oom()` call) — never both, never twice. `on_oom()` is invoked **at most once per OOM event**; for train that single call is the outer rung's K decision. (A second `on_oom()` call would halve K twice for one OOM, breaking train's one-halving-per-OOM contract and `test_trainer_oom_retry.py`.) Eval/predict have a single catch site, so they call `on_oom()` once there for the full B-then-K policy and branch on its return directly (§5.5).

---

## §4 Detailed design — `oom.py`

### `OomEvent` (relocated)

Move the frozen dataclass verbatim from `train/types.py` to `oom.py`:

```python
@dataclass(frozen=True)
class OomEvent:
    step: int
    action: Literal["microbatch_halved", "multiplex_halved"]
    new_micro_batch_size: int
    effective_K: int | None = None  # set only for "multiplex_halved"
```

`train/types.py` re-exports it (`from custom_sam_peft.oom import OomEvent` plus `__all__` entry) so the existing imports in `train/loop.py`, `train/trainer.py`, `runs/bundle.py`, and tests keep working unchanged. The bundle edge-note and trainer telemetry semantics are untouched.

### `OomDecision`

```python
class OomDecision(enum.Enum):
    RETRY_B = "retry_b"
    RETRY_K = "retry_k"
    FLOOR_RETRY = "floor_retry"
    TERMINAL = "terminal"
```

### `OomLadder`

```python
class OomLadder:
    """Sticky, monotonically-decreasing B-then-K OOM state + decision.

    Constructed per path with the initial (micro_batch_size, effective_K).
    on_oom() applies the B-then-K policy; callers map the returned
    OomDecision to their own control flow (see spec §3 mapping table).
    """

    def __init__(self, micro_batch_size: int, effective_K: int) -> None: ...

    micro_batch_size: int                   # B — only ever decreases
    effective_K: int                        # K — only ever decreases
    pending_oom_events: list[OomEvent]      # one entry per halving transition

    def on_oom(self, step: int | None = None) -> OomDecision: ...
```

**Field names are load-bearing for the untouched-tests contract (§5.1, §7.2).** `tests/unit/test_eval_batch_size_cap.py` imports and constructs `OomState` by name (`from custom_sam_peft.train.loop import OomState`; `OomState(micro_batch_size=…)`) and reads `oom_state.micro_batch_size`; `tests/unit/test_trainer_oom_retry.py`'s `_State` stub reads `micro_batch_size` and `pending_oom_events`. So the ladder keeps the field name **`pending_oom_events`** (not `events`), keeps **`micro_batch_size`** / **`effective_K`**, and `train/loop.py` keeps **`OomState`** as a name for that test's import/construction — either as the `OomLadder` class itself, an alias (`OomState = OomLadder`), or a thin subclass. Whichever form is chosen, `OomState(micro_batch_size=…, effective_K=…)` must construct and expose `micro_batch_size` / `effective_K` / `pending_oom_events` so both tests stay green untouched.

Internal `_floor_retry_used: bool` (default `False`) gates the single `FLOOR_RETRY`. Logging is one warning per halving transition (B or K), matching train's current one-line-per-halving behavior. `empty_cache()` is always attempted (guarded) at the top of `on_oom`, before the policy branch — this is the #176 robustness guarantee, applied uniformly.

### Shared `_halve_microbatch` routine (the single B-rung mechanic)

The B-rung *mechanics* are factored into one module-level helper in `oom.py` so there is exactly one implementation of the B-halving, used by both `OomLadder.on_oom()`'s B-branch and train's inner helper:

```python
def _halve_microbatch(state, step: int | None = None) -> None:
    """Halve micro_batch_size and record the transition. FIELD-ONLY.

    Operates solely on the object's *fields* — `micro_batch_size`,
    `pending_oom_events`, `step` — so it works on both OomLadder and train's
    field-only `_State` stub (which has no methods). Callers do the
    `empty_cache()` (on_oom's top; the inner helper's catch handler) and the
    `B > 1` guard before calling this.
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
```

`OomLadder.on_oom()`'s B-branch (step 2.i) is `self._floor`-free and simply calls `_halve_microbatch(self, step)` then returns `RETRY_B` — no duplicated halving logic. Train's inner helper calls the **same** `_halve_microbatch(state, step)` on its field-only `_State`/`OomState` argument (§5.1). Because the routine touches only fields, the inner helper never needs `state.on_oom()`, and `test_trainer_oom_retry.py`'s method-less `_State` stub stays untouched.

**Invariants:**

- B and K never increase. Once halved they stay halved for the ladder's lifetime (sticky), matching train's existing `OomState` semantics and eval's sticky-B behavior.
- `on_oom` records at most one `OomEvent` per call (the halving it performed) onto `pending_oom_events`; `FLOOR_RETRY` and `TERMINAL` record none.
- `FLOOR_RETRY` is returned at most once over the ladder's lifetime; the next `on_oom` after a consumed floor retry returns `TERMINAL`.
- **For any single OOM event, exactly one halving occurs — B (inner helper, via the shared `_halve_microbatch` routine) OR K (outer rung, via the single `on_oom()` call) — never both, never twice.** `on_oom()` is invoked **at most once per OOM event**; for train that single call happens at the outer rung (the K decision). The B-halving *mechanics* have a single shared implementation (`_halve_microbatch`), so centralization holds even though train's inner B-rung does not go through the `on_oom()` *method* (it must accept the method-less `_State` stub and so calls `_halve_microbatch` directly). Eval/predict have a single catch site and call `on_oom()` once for the full B-then-K policy.

---

## §5 Detailed design per path

### §5.1 Train (Phase 1 — reference consumer)

Train's control flow is preserved exactly; only the *source* of B/K values and the halving decision changes.

- `OomState` is replaced by / wraps `OomLadder` (the name `OomState` is preserved per §4). The trainer constructs `OomLadder(micro_batch_size=cfg.train.batch_size, effective_K=min(cfg.train.multiplex.classes_per_forward, MULTIPLEX_CAP))` where it builds `OomState` today (`trainer.py:495-498`).
- **`on_oom()` is called exactly once per OOM event — at train's *outer* rung (the K decision).** Train's catch site (inner helper) and its K-handler (outer rung) are at different levels, so the work is split: the inner helper handles the B-rung directly (via the shared `_halve_microbatch` routine), and only the outer rung calls `on_oom()`. The inner helper does **not** call `state.on_oom()` — `test_trainer_oom_retry.py`'s field-only `_State` stub has no such method and must stay untouched (§4). A second `on_oom()` would halve K twice for one OOM and break that test.
- `_train_step_with_oom_ladder`'s inner OOM `except` block (`loop.py:119-137`) handles only the B-rung:
  - `if state.micro_batch_size > 1` → call the shared `_halve_microbatch(state, state.step)` routine (§4) and `continue` the inner microbatch loop at the new `state.micro_batch_size`. This is byte-identical to today's inline `state.micro_batch_size //= 2; …; continue`, but the mechanics now live in the one shared routine that `on_oom()`'s B-branch also uses. The helper never calls `state.on_oom()`.
  - `else` (at `state.micro_batch_size == 1`) → raise a **fieldless** `_MicrobatchExhausted` (exactly as today — it carries no `OomDecision`) so `train_step`'s outer rung can decide the K-rung.
- `train_step`'s outer rung (`loop.py:379-406`) catches `_MicrobatchExhausted` and calls `oom_state.on_oom(global_step)` **exactly once** for this OOM event. At B == 1 `on_oom()` skips its B-branch and advances the K-rung, returning one of:
  - `RETRY_K` → `optimizer.zero_grad(set_to_none=True)`, then `continue` the outer K-replay loop, which re-chunks all classes at the already-decremented `ladder.effective_K` and replays from group 0 (today the outer rung itself did `oom_state.effective_K = max(1, oom_state.effective_K // 2)` then looped; now the K-decrement happened inside the single `on_oom()` call).
  - `FLOOR_RETRY` → replay the whole step once (re-enter the outer rung with B and K unchanged at the floor).
  - `TERMINAL` → raise the existing `RuntimeError("OOM at step … after micro_batch=1 and classes_per_forward=1. Use a larger GPU or smaller image_size.")`.
- `pending_oom_events` is read directly off the ladder (it *is* the ladder's `pending_oom_events` list — the trainer reads it for telemetry / bundle edge-note exactly as it reads `OomState.pending_oom_events` today; no separate copy). The inner helper's `_halve_microbatch` appends `microbatch_halved` events to that same list; the outer rung's `on_oom()` appends `multiplex_halved` events.

**Behavioral contract:** every assertion in `test_trainer_oom_retry.py` must remain green untouched — B halves to 1 then signals exhaustion (via a fieldless `_MicrobatchExhausted`), halving is sticky across steps, gradient magnitude is preserved by the helper's `/n_micro`, `zero_grad` fires once per step, **a single OOM at B == 1 halves K exactly once** (the outer rung's single `on_oom()` call — never two), and `OomEvent` actions/fields are as before. The test's `_State` stub reads `micro_batch_size` and `pending_oom_events` and has no methods, so the inner helper touches only fields (via `_halve_microbatch`) and those field names are preserved (§4).

### §5.2 Eval (Phase 2)

Eval already uses an index-driven `while i < len(examples)` loop with per-image-chunk buffering (`evaluator.py:201-254`). Two changes:

1. **Replace `_eval_forward_with_oom_ladder` with the shared `OomLadder`.** Construct one ladder per `_iter_predictions` call: `OomLadder(micro_batch_size=int(cfg.batch_size), effective_K=min(MULTIPLEX_CAP, len(dataset.class_names)))`. (`cfg.batch_size` is already resolved to an int by `run_eval`.)
2. **Add the K-rung.** The inner class-group loop becomes index-driven on `ladder.effective_K`. On `OutOfMemoryError` from the forward, call `ladder.on_oom()` and branch:
   - `RETRY_B`: discard `chunk_buf`, break to the outer `while`, which re-chunks images from `i` at the new `ladder.micro_batch_size`.
   - `RETRY_K`: keep the rows already committed to `chunk_buf` from completed K-groups; resume the class loop **from the current class index** at the smaller K-group size.
   - `FLOOR_RETRY`: retry the same forward once.
   - `TERMINAL`: raise `RuntimeError`.

The existing buffer-and-commit invariant holds: `chunk_buf` is committed to `predictions` only when the image-chunk's class loop completes without a pending B-change. A B-change discards the buffer (image set changed); a K-change keeps it (completed K-groups produced final rows).

Row reassembly uses the **actual** sub-group length: `for r in range(len(image_chunk) * K_g)` with `ii, kk = divmod(r, K_g)` and `cat_idx = dataset.class_names.index(group[kk])` — this is unchanged from today and is the #176 dup/drop-safe form. After a K-halving, `K_g` shrinks; the reassembly must use the current group's actual length, never a fixed `MULTIPLEX_CAP`.

### §5.3 Predict (Phase 2 — the #181 fix)

`run_predict`'s forward loop (`predict/runner.py:397-487`) is restructured to mirror eval:

- Replace the two `for`-loops (`for chunk_paths in _chunked(image_paths, bs)` and `for group in _chunked(prompts, MULTIPLEX_CAP)`) with **index-driven `while` loops** keyed on `ladder.micro_batch_size` (image index) and `ladder.effective_K` (class index).
- Construct `OomLadder(micro_batch_size=bs, effective_K=min(MULTIPLEX_CAP, len(prompts)))` once, before the loop. `bs` is the already-resolved per-forward image batch size (the `"auto"` → `decide_eval_batch_size` resolution at lines 343-349 is unchanged).
- **Buffer per image-chunk.** Collect a chunk's prediction entries into a local `chunk_buf` and commit to `all_predictions` only when all the chunk's K-groups complete — identical to eval's mechanism. This is the anti-duplication guarantee that makes a B-restart safe.
- Replace the OOM catch (437-442) with: on `OutOfMemoryError`, call `ladder.on_oom()` and apply the eval/predict column of the §3 mapping table (`RETRY_B` discards `chunk_buf` and restarts the image-chunk; `RETRY_K` resumes from the current class index keeping completed groups; `FLOOR_RETRY` retries once; `TERMINAL` raises `RuntimeError` with the larger-GPU / smaller-`image_size` guidance). The old single `logger.error(...)` advice line is removed.
- Catch `torch.cuda.OutOfMemoryError` (a subclass of `RuntimeError`); preserve a guard so non-OOM `RuntimeError`s still propagate untouched.

The per-image-chunk image-open/transform block (405-428), the per-row postprocess and score/top-k filtering (445-459), progress ticks, verbose logging, and all sidecar writing (run.json, image_id_map, predictions) are otherwise unchanged. The warmup call (367-373) is unchanged.

### §5.4 The buffer-and-commit anti-dup mechanism (correctness-critical)

The shared rule for both eval and predict:

1. Each image-chunk has a private `chunk_buf`.
2. A completed K-group appends its rows to `chunk_buf`.
3. `chunk_buf` is committed to the global predictions list **only** when every K-group of the chunk finishes (the `for…else` no-break branch in eval; the equivalent completion flag in predict).
4. **`RETRY_B` discards `chunk_buf`** and restarts the chunk: the image set per forward changed, so any buffered rows are stale.
5. **`RETRY_K` retains `chunk_buf`** and resumes from the current class index: completed K-groups produced final per-`(image, class)` rows that are independent of the K-group size.

This guarantees byte-identical output to a non-OOM run: no row is dropped (the index-driven loops cover every `(image, class)` pair) and no row is duplicated (a chunk commits exactly once, and only on full completion).

### §5.5 Index-driven loop sketch (eval/predict)

```text
ladder = OomLadder(micro_batch_size=bs, effective_K=min(MULTIPLEX_CAP, n_classes))
i = 0
while i < n_images:
    B = ladder.micro_batch_size
    image_chunk = items[i : i + B]
    chunk_buf = []
    j = 0                       # class index into the prompt/class list
    chunk_done = False
    while j < n_classes:
        K_g = min(ladder.effective_K, n_classes - j)
        group = classes[j : j + K_g]
        try:
            outputs = model(stack(image_chunk), prompts_for(group))
        except OutOfMemoryError:
            decision = ladder.on_oom(step=None)
            if decision is RETRY_B:        # image set changed → restart chunk
                chunk_buf = []; break       # re-enter outer while at smaller B
            if decision is RETRY_K:        # resume from current class index
                continue                    # smaller K_g recomputed at top of inner loop
            if decision is FLOOR_RETRY:
                continue                    # retry same forward once (K_g unchanged at floor)
            raise RuntimeError("OOM at B=1, K=1; use a larger GPU or smaller image_size.")
        for r in range(len(image_chunk) * K_g):   # ACTUAL K_g, never MULTIPLEX_CAP
            ii, kk = divmod(r, K_g)
            chunk_buf.extend(postprocess_row(outputs, r, image_chunk[ii], group[kk]))
        j += K_g
    else:
        chunk_done = True
    if chunk_done:
        predictions.extend(chunk_buf)       # commit exactly once, on full completion
        i += len(image_chunk)
```

(At the K=1 floor `FLOOR_RETRY` retries the same forward; `K_g` cannot shrink further. `RETRY_B` uses an inner-loop `break` whose `else` is skipped, so `chunk_done` stays `False` and `i` does not advance — the chunk re-runs at the smaller B.)

---

## §6 Behavior changes (all intended)

| Change | Path | Notes |
|--------|------|-------|
| Gains the full B-then-K ladder + `empty_cache()` + a floor retry; the old single `logger.error` advice line is removed. | Predict | The #181 fix. Multiplex predict with many classes now recovers instead of crashing. |
| Gains a **K-rung** (was B-only). | Eval | Fixes a latent eval K-OOM: a single large class group at B=1 previously hit the floor and hard-failed; it now halves K and resumes mid-chunk. |
| Train's hard-fail path is unchanged in observable behavior. | Train | Train already does `empty_cache()` + retries within the ladder; it already reaches `TERMINAL` only at B=1, K=1. The shared `FLOOR_RETRY` (`empty_cache` + one retry before erroring) matches eval's #176 robustness; train's existing tests pin the contract. |
| Warning cadence is **one line per halving transition** (bounded by ~`log2(B_0) + log2(K_0)` lines per ladder), for all three paths. | All | Eval previously emitted "warn once total"; it now emits one line per B-halving and one per K-halving — consistent with train's existing per-halving warnings. Predict previously emitted one error line; it now emits per-halving warnings. |
| `OomEvent` is importable from `custom_sam_peft.oom`; `train/types.py` re-exports it. | All | Back-compat re-export keeps existing imports green. |

---

## §7 Testing strategy

CPU-only. Every test injects a synthetic `torch.cuda.OutOfMemoryError` (constructible without CUDA, as the existing tests do) and patches `torch.cuda.is_available` / `empty_cache` where needed.

### §7.1 `OomLadder` unit tests (Phase 1) — new

- **Decision sequence.** Starting from `(B0, K0)` with `B0, K0 > 1`: repeated `on_oom()` returns `RETRY_B` until `B == 1` (each call halves B), then `RETRY_K` until `K == 1` (each call halves K), then `FLOOR_RETRY` exactly once, then `TERMINAL` on every subsequent call. Assert the exact `OomDecision` ordering and the `B`/`K` values after each step.
- **Stickiness / monotonicity.** `B` and `K` only ever decrease; a later `on_oom` never raises either.
- **`pending_oom_events` emission.** Exactly one `OomEvent` appended to `pending_oom_events` per halving call; none for `FLOOR_RETRY`/`TERMINAL`. `microbatch_halved` events carry the new `new_micro_batch_size` and `effective_K is None`; `multiplex_halved` events carry the new `effective_K` and the current `new_micro_batch_size`. The event count equals the number of halvings.
- **`empty_cache` guard.** With `is_available()` patched `True` and `empty_cache` stubbed, every `on_oom` call invokes `empty_cache` once; with `is_available()` `False`, it is never called.
- **Degenerate starts.** `(B=1, K=1)` → first `on_oom` is `FLOOR_RETRY`, second is `TERMINAL`. `(B=1, K>1)` → starts halving K immediately.

### §7.2 Train (Phase 1) — unchanged contract

- `tests/unit/test_trainer_oom_retry.py` runs **green untouched**. It is the behavioral contract for the train migration: B halves to 1 then signals `_MicrobatchExhausted`, sticky halving, gradient-magnitude preservation, `zero_grad` once per step, `OomEvent` actions/fields, bundle edge-note linkage.
- `tests/unit/test_eval_batch_size_cap.py` stays green: the eval-batch cap reads the ladder's `micro_batch_size` (it reads `OomState.micro_batch_size` today; the field name/value is preserved through the migration).

### §7.3 Eval (Phase 2) — extend

- Extend `tests/unit/test_eval_oom_ladder.py` with a **K-rung** case: a stub model that OOMs on a large class group at B=1, then succeeds at the halved K. Assert the class loop resumes from the current class index, completed K-groups' rows are retained, and final predictions have no duplicate / no dropped `(image_id, category_id)` rows.
- The existing `test_mid_chunk_oom_does_not_produce_duplicate_predictions` (B-change discard path) must stay green.

### §7.4 Predict (Phase 2) — new

- **Byte-identical recovery.** Run `run_predict` (or its forward loop, factored for testability) with **many classes** against a stub model that OOMs once then succeeds, and a reference non-OOM run with the same inputs. Assert: (a) the OOM run **completes**, and (b) its prediction entries are **byte-identical** to the non-OOM run — same rows, in the same order, no duplicates, no drops.
- **Buffer-discard-on-B-change.** Explicitly exercise an OOM that triggers `RETRY_B` mid-chunk: assert the partially-buffered chunk is discarded and re-emitted exactly once at the smaller B (no duplicate `(image_id, category_id)`), mirroring the eval regression.

---

## §8 Phasing and interface contract

Two sequential phases. The boundary is the `OomLadder` public API.

### Phase 1 — `oom.py` + train migration (the reference consumer)

Build `src/custom_sam_peft/oom.py` (`OomLadder`, `OomDecision`, relocated `OomEvent` with the `train/types.py` re-export) and migrate `train/loop.py` + `train/trainer.py` onto it. Train is the reference consumer that **pins the contract**: its existing behavior (and `test_trainer_oom_retry.py`) is the acceptance gate for the ladder's semantics. Ship the `OomLadder` unit tests (§7.1) here.

**Interface exposed (consumed by Phase 2):**

- `OomLadder(micro_batch_size: int, effective_K: int)` — constructor. `train/loop.py` also exposes the name `OomState` for it (class, alias, or thin subclass) so `test_eval_batch_size_cap.py`'s `from custom_sam_peft.train.loop import OomState` / `OomState(micro_batch_size=…)` stays green untouched (see §4).
- Fields: `micro_batch_size: int`, `effective_K: int` (sticky, monotone non-increasing), `pending_oom_events: list[OomEvent]`.
- `on_oom(step: int | None = None) -> OomDecision` — `empty_cache()` (guarded) + B-then-K policy.
- `OomDecision` enum: `RETRY_B`, `RETRY_K`, `FLOOR_RETRY`, `TERMINAL`.
- `OomEvent` importable from `custom_sam_peft.oom` (and, for back-compat, `custom_sam_peft.train.types`).

### Phase 2 — eval + predict migration

Migrate `eval/evaluator.py` and `predict/runner.py` onto the Phase-1 `OomLadder`: index-driven `while` loops keyed on `ladder.micro_batch_size` / `ladder.effective_K`, per-image-chunk prediction buffering with the commit-on-full-completion rule, the actual-`K_g` row-reassembly form, and the §7.3/§7.4 tests. Eval gains its K-rung; predict gains the full ladder. **Consumes** the Phase-1 interface above; adds nothing to it.

---

## §9 Risks and mitigations

| Risk | Mitigation |
|------|------------|
| **Train migration regresses behavior** (highest risk — train's ladder is gradient-aware with whole-step replay and `zero_grad` coupling). | `test_trainer_oom_retry.py` is the net and must stay green untouched. The migration only re-sources B/K values and the halving decision from `OomLadder`; control flow (`_MicrobatchExhausted`, replay, `zero_grad`) is preserved verbatim. Phase 1 lands train as the reference consumer precisely so the contract is pinned before eval/predict build on it. |
| **The #176 dup/drop trap** — row reassembly with a stale group length, or committing a buffer twice on retry. | Eval already encodes the fix (commit-only-on-completion + actual-`K_g` reassembly) and has a regression test; predict adopts the identical mechanism. §7.3 / §7.4 add dedicated tests for both the B-discard and K-resume paths. Reassembly must use the current group's actual length, never `MULTIPLEX_CAP`. |
| **K-resume mid-chunk double-counts a class.** | The inner class index `j` only advances by the *actual* `K_g` after a successful forward; a `RETRY_K` recomputes `K_g` at the smaller size from the same `j`, so each class is emitted exactly once. Covered by §7.3. |
| **`OomEvent` relocation breaks existing imports.** | `train/types.py` re-exports `OomEvent` from `oom.py`; a smoke import in the test suite confirms both paths resolve to the same class. |
| **Train/eval semantic drift in the shared component** (train needs replay-from-0 on K-change; eval/predict resume mid-chunk). | The shared component deliberately stops at the *decision* boundary (`OomDecision`); the divergent control flow lives in each caller and is documented in the §3 mapping table. The ladder never implies a control-flow shape. |
