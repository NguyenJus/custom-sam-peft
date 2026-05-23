# SAM 3.1 multiplex forward for train, eval, and predict

**Issue:** [#22 — feat: use SAM 3.1's multiplex forward (≤16 classes per pass)](https://github.com/NguyenJus/custom-sam-peft/issues/22)
**Release:** `v0.8.0` (pre-1.0 minor bump — single user-visible knob added, per-step loss magnitudes shift)
**Status:** locked design, single PR, no back-compat shims.

We ship the released `sam3.1_multiplex.pt` checkpoint but currently run one forward per class prompt. This spec replaces that serialized loop with **SAM 3.1's native multiplex forward**: one forward per ≤16-class group, in training, eval, and predict. The wrapper, training loop, evaluator, and predict CLI change together, gated by a single `MULTIPLEX_CAP = 16` constant and a configurable `classes_per_forward` knob that defaults to 16.

---

## §1 Motivation

Per issue #22, SAM 3.1's released `sam3.1_multiplex.pt` checkpoint is named after its ability to accept up to 16 class prompts in a single forward and emit per-(image, class) detections in parallel. We don't use that capability. `Sam3Wrapper._validate_inputs` (`src/custom_sam_peft/models/sam3.py:229-234`) raises whenever `TextPrompts` carries more than one class, and `train_step` (`src/custom_sam_peft/train/loop.py:210-297`) loops over `classes_in_batch` calling `model(...)` once per class. Eval (`src/custom_sam_peft/eval/evaluator.py:147-162`) and `predict` (`src/custom_sam_peft/predict/runner.py:377-410`) do the same.

The waste compounds with class count: COCO at 80 classes incurs ~80× the image-encoder cost per micro-step beyond what multiplex would charge, since SAM 3.1's image encoder is shared across class prompts in a multiplex call. The multiplex variant was designed precisely to amortize that.

Today's serialized loop also runs the loss at K=1 per call, which is **off-distribution** relative to SAM 3.1's pretraining (which jointly supervises ≤16 classes per forward). Restoring the multiplex regime brings the loss signal back onto the trained distribution — see §10 R5.

---

## §2 Goals and non-goals

### Goals

- One forward per ≤16-class group in **training, eval, and predict** — single multiplex code path.
- Trainer and Evaluator/Predict own **chunking** (`classes_in_batch` sliced into ⌈K_total / 16⌉ groups); the wrapper accepts ≤ `MULTIPLEX_CAP` classes per call and validates that all images carry the same class list.
- New `eval.batch_size: PositiveInt | Literal["auto"]` knob (default `"auto"`) routed through a forward-only sibling of `decide_preset` (`presets.decide_eval_batch_size`); same knob added to the `predict` CLI.
- **Aggregate-only** per-step loss attribution. The losses dict (`mask`, `box`, `obj`, `presence`, `total`) is computed once per group on the flattened `(B·K_g, …)` outputs; no per-class slicing.
- Configurable `train.multiplex.classes_per_forward: int = 16` (range 1..16). K=1 is the **degenerate case of the same code path**, not a separate path; at K=1, per-step RNG order and gradient computation are seed-bit-equivalent to today's main (see §10 R3).

### Non-goals (one line each)

- **Loss-weight ablation.** Keep `w_mask = w_obj = w_presence = 1` defaults; multiplex restores pretraining distribution, so re-tuning is deferred (§12 follow-up).
- **Image-size bucketing.** Eval images keep their existing one-resize-fits-all preprocessing; `batch_size` is a single scalar, not per-`original_hw`.
- **Lifting the 16-class cap.** SAM 3.1's multiplex head is trained at K ≤ 16; we adopt that as a hard cap.
- **Per-class TensorBoard loss attribution.** The per-row loss decomposition is not surfaced — only aggregated tensorboard scalars (matches §4 locked decision).
- **Multi-dataset mixing.** Orthogonal; the original issue body used "multiplex" colloquially for that — see issue #22 rewrite note.
- **Joint (B, K_eval) VRAM tuning.** K_eval is fixed at 16 in `decide_eval_batch_size`; jointly searching over both dims is a §12 follow-up.

### Forward-compat note

Single-class semantic-segmentation users land on K=1 automatically (one class in the vocabulary → one group of one class per group). The single code path means they get the same numerics as today's main.

---

## §3 Architecture and data flow

The key conceptual shift: a forward's output is no longer one prediction per image; it is one prediction per **(image, class) row**. A "row" indexes into the flattened `(B·K_g, …)` output of one multiplex forward.

Four layers change:

| Layer | File | Change |
|-------|------|--------|
| Wrapper + adapter | `src/custom_sam_peft/models/sam3.py` | Validation lifts the K=1 raise; adapter builds B·K_g rows via `FindStage(img_ids, text_ids)`; geometric prompt parameterized on column count. |
| Losses | `src/custom_sam_peft/models/losses.py` | No interface change. Inputs grow first dim from `B` to `B·K_g`; targets list grows from `[B]` to `[B·K_g]` where row r belongs to `(image = r // K_g, class = r % K_g)`. |
| Train loop | `src/custom_sam_peft/train/loop.py` | Replace per-class loop with `_chunked(classes_in_batch, classes_per_forward)`; per-(image, class) Bernoulli box hints; one `total_loss` per group; backward `total / (G · grad_accum_steps)`. |
| Eval + predict | `src/custom_sam_peft/eval/evaluator.py`, `src/custom_sam_peft/eval/runner.py`, `src/custom_sam_peft/predict/runner.py` | Flat loop over `(image_chunk, class_group)`; per-row postprocess; `_eval_forward_with_oom_ladder` mirroring train's. |

### Data flow per training step (pseudocode)

```
classes_in_batch = sorted(unique class names across batch's prompts)
B = images.shape[0]
G = ceil(len(classes_in_batch) / classes_per_forward)
if len(classes_in_batch) > MULTIPLEX_CAP:
    log.info("multiplex auto-chunk: K_total=%d -> G=%d groups", ...)  # once per run

for group in _chunked(classes_in_batch, classes_per_forward):
    K_g = len(group)
    # one TextPrompts per image, all carrying group's K_g class names in the same order
    prompts_g = [TextPrompts(classes=list(group)) for _ in range(B)]
    # per-(image, class) Bernoulli for box hints
    hints_g = [maybe_box_hint(image=i, class=c, p=p_t) for i in range(B) for c in group]
    # one multiplex forward — output rows are B·K_g
    out = model(images, prompts_g, box_hints=as_per_image_lists(hints_g))
    # flat targets list of length B·K_g (image-major, class-minor)
    targets_g = [filter_by_class(targets[i], c_dense=class_names.index(c))
                 for i in range(B) for c in group]
    losses = total_loss(out, targets_g, cfg.train.loss)  # one call, aggregate
    (losses["total"] / (G * grad_accum_steps)).backward()
```

The flattening order is **image-major, class-minor** to match the adapter's `(img_ids, text_ids)` layout (§4).

---

## §4 Wrapper and adapter changes

**File:** `src/custom_sam_peft/models/sam3.py`.

### `MULTIPLEX_CAP` constant

A new module-level constant `MULTIPLEX_CAP = 16` co-located with `Sam3Wrapper`. Imported by trainer and evaluator for chunking; exported from `models/sam3.py` (not re-exported via top-level `__init__`).

### `Sam3Wrapper._validate_inputs`

Replace the existing `len(p.classes) != 1` raise (lines 229-234) with two checks:

1. `1 <= len(p.classes) <= MULTIPLEX_CAP` per prompt — raise on out-of-range with a clear message naming `MULTIPLEX_CAP` and pointing at `train.multiplex.classes_per_forward`.
2. **Shared class list**: for `TextPrompts` batches, every prompt must carry the same `tuple(classes)` (same order, same names). Raise on mismatch — this is the locked decision in §9 of the brainstorming summary: the `(img_ids, text_ids)` layout assumes a shared K-prompt vocabulary across the batch. The trainer and evaluator construct prompts that satisfy this; the check guards against caller bugs.

### `_Sam3ImageAdapter.forward`

Replace the current single-class assembly (lines 315-360) with multiplex assembly:

- `K = len(prompts[0].classes)` (validated equal across the batch).
- `forward_text` is called **once** with the K-element list of class names; result is broadcast to all rows via `text_ids` indexing inside `forward_grounding`.
- Build `FindStage` with:
  - `img_ids = arange(B, device=device).repeat_interleave(K)` — length `B·K`.
  - `text_ids = arange(K, device=device).repeat(B)` — length `B·K`.
- Geometric prompt (`_build_geometric_prompt`) is parameterized on **column count** `B·K` (today it's `B`). The wrapper accepts a flat list of length `B·K` of per-row box hints, ordered image-major / class-minor.
- The dummy zero-row `Prompt` when `gp is None` becomes `(0, B·K, 4)` and `(B·K, 0)` to match.

### `_build_geometric_prompt`

Signature changes from `(box_hints, image_size, device)` to `(box_hints, n_cols, image_size, device)`. Internals already iterate per-column; we replace the hard-coded `B = len(box_hints)` with the explicit `n_cols` and require `len(box_hints) == n_cols`. The trainer/evaluator passes `n_cols = B · K_g` and a flat list of `B·K_g` hint tensors-or-None.

### Output shape

Today `outputs["pred_logits"]` is `(B, Q, 1)`; under multiplex it becomes `(B·K, Q, 1)`. Same for `pred_boxes` (`(B·K, Q, 4)`), `pred_masks` (`(B·K, Q, H, W)`), and `presence_logit_dec` (`(B·K, 1)`). `meta_to_canonical` is unchanged — it just sees a bigger leading dim. **Row r in the output corresponds to (image = r // K, class index in group = r % K).**

---

## §5 Training loop changes

**File:** `src/custom_sam_peft/train/loop.py`.

### Replace the per-class loop

Lines 210-297 (the `for c in classes_in_batch:` loop) become:

```
groups = _chunked(classes_in_batch, cfg.train.multiplex.classes_per_forward)
G = len(groups)
for group in groups:
    K_g = len(group)
    prompts_g = [TextPrompts(classes=list(group)) for _ in range(B)]
    # Per-(image, class) Bernoulli for box hints
    hints_g = []
    for i in range(B):
        for c in group:
            row_targets = [inst for inst in targets[i] if inst.class_id == class_names.index(c)]
            if row_targets and random.random() < p_t:
                hints_g.append(stack_boxes(row_targets).to(runtime))
                n_hint_applied += 1
            else:
                hints_g.append(None)
    targets_g = [
        [inst for inst in targets[i] if inst.class_id == class_names.index(c)]
        for i in range(B) for c in group
    ]
    # one forward + one loss call per group
    out = model(images, prompts_g, box_hints=hints_g)
    group_losses = total_loss(out, targets_g, cfg.train.loss)
    group_scaled = group_losses["total"] / (G * cfg.train.grad_accum_steps)
    if torch.isfinite(group_scaled):
        group_scaled.backward()
        finite_group_count += 1
        for k in accum:
            accum[k] += float(group_losses[k].detach())
```

`_chunked(seq, n)` is a tiny local helper (3-line itertools-style); placed near the top of `train/loop.py`.

### `MULTIPLEX_CAP` enforcement

`cfg.train.multiplex.classes_per_forward` is validated to `1..16` at the schema layer (§7). The trainer additionally clamps: `effective_K = min(cfg.train.multiplex.classes_per_forward, MULTIPLEX_CAP)`. When `len(classes_in_batch) > MULTIPLEX_CAP`, **always auto-chunk** — no error path on cap exceedance (locked decision §1). The trainer logs once per run (at the first chunk-trigger step) at INFO level: `"multiplex auto-chunk: classes_in_batch=%d > MULTIPLEX_CAP=%d -> %d groups"`.

### OOM ladder

`_train_step_with_oom_ladder` (lines 53-127) is untouched in core mechanics — its job is to halve the **image dimension** within a group. The closure built for it now operates on the flat row layout: given a microbatch of image indices `[i0, i1, …]`, it constructs `prompts_g_micro`, `hints_g_micro` (a length-`|micro| · K_g` slice in image-major order), and `targets_g_micro` accordingly. The closure returns `group_losses["total"] / (G * grad_accum_steps)` (not further divided by n_micro — the ladder applies `/ n_micro` itself, per lines 86-88).

### NaN policy

Today the trainer counts a step as "skipped" only when every class was non-finite. Under multiplex the policy becomes: a step is skipped only when **every group** is non-finite. `nan_streak` increments and `nan_abort_after` semantics are unchanged.

Per-row non-finiteness inside a finite group is **not** detected: `total_loss` reduces by `.mean()`, so any row's NaN contaminates the whole group's total. This is intentional — finer-grained row-level recovery is a §12 follow-up if it proves necessary in practice (R4 in §10).

### Accumulator denominator

Lines 316 use `max(finite_class_count, 1)` as the per-key average denominator for `StepResult.losses`. Replace with `max(finite_group_count, 1)`. The `box_hint/applied` denominator in `_ScalarWindow.update` (line 359, `r.n_classes * max(r.images_processed, 1)`) keeps the same total `B · K_total` denominator — it's expressed via `n_classes` (which still tracks `len(classes_in_batch)`) and `images_processed`. No change there.

`StepResult.n_classes` continues to mean `len(classes_in_batch)`; that field's contract is unchanged.

---

## §6 Eval and predict

### `Evaluator._iter_predictions`

**File:** `src/custom_sam_peft/eval/evaluator.py:108-169`.

Today's loop is image-major, class-minor with `B=1` per call. Replace with a flat loop over `(image_chunk, class_group)`:

```
for image_chunk in _chunked(examples, cfg.batch_size):
    images_t = stack_and_to_device([ex.image for ex in image_chunk])
    for group in _chunked(dataset.class_names, MULTIPLEX_CAP):
        K_g = len(group)
        prompts_g = [TextPrompts(classes=list(group)) for _ in image_chunk]
        outputs = _eval_forward_with_oom_ladder(model, images_t, prompts_g)
        # Row r in outputs maps to (image_chunk[r // K_g], group[r % K_g]).
        for r in range(len(image_chunk) * K_g):
            i, k = divmod(r, K_g)
            cat_idx = dataset.class_names.index(group[k])
            entries = queries_to_coco_results(
                _row_outputs(outputs, r),
                _int_image_id(image_chunk[i].image_id),
                cat_idx + 1,
                (image_chunk[i].image.shape[-2], image_chunk[i].image.shape[-1]),
                cfg.mask_threshold,
            )
            predictions.extend(entries)
```

`_row_outputs(outputs, r)` is a small helper that unrolls one multiplex row back into the per-image dict shape that `queries_to_coco_results` expects today — i.e. it indexes each tensor on dim 0 at `[r:r+1]`, preserving `queries_to_coco_results`'s `batch == 1` hard-assert. This keeps the postprocess module unchanged.

### `_eval_forward_with_oom_ladder`

New module-private helper in `src/custom_sam_peft/eval/evaluator.py` (or pulled out to `src/custom_sam_peft/eval/_oom.py` if the planner prefers). Mirrors `_train_step_with_oom_ladder` (§4 of the algo-vram-preset spec) in shape, but with:

- **No grad-checkpointing rung.** Eval runs under `torch.no_grad()`; there is no backward pass to checkpoint. The ladder has exactly one rung: halve `cfg.batch_size`.
- **No microbatch backward.** Just `model(...)`; on OOM, halve B and replay.
- **Sticky.** Once B is halved, it stays halved for the rest of the eval call.
- **Single warn.** Emit `_LOG.warning("eval OOM at image_idx=%d — halving batch_size to %d", ...)` at most once per `evaluate` call.

### `EvalConfig.batch_size`

New field on `EvalConfig` (`src/custom_sam_peft/config/schema.py:412-420`):
`batch_size: PositiveInt | Literal["auto"] = "auto"`.

### `run_eval` resolves `"auto"`

`src/custom_sam_peft/eval/runner.py:56-169` resolves `"auto"` via `presets.decide_eval_batch_size(cfg.data.image_size, classes_per_forward=16)` once at the top of `run_eval`, then constructs `Evaluator` with the resolved `EvalConfig`. The resolution happens before `evaluate()` is called.

Resolution caveat: when CUDA is unavailable, `decide_eval_batch_size` falls back to `B = 1` and logs `_LOG.info("eval.batch_size=auto on CPU -> falling back to 1")`. When CUDA is available but no calibration cache exists, the analytic estimate runs (§8).

### Predict CLI

**File:** `src/custom_sam_peft/predict/runner.py:214-503`.

The current loop (lines 352-410) is per-image, per-class. Restructure to mirror the evaluator:

- `PredictOptions.batch_size` field is `int | Literal["auto"]` with dataclass default `"auto"`. The CLI shell (`cli/predict_cmd.py`) parses `--batch-size auto` into the sentinel and forwards; when the user omits the flag the dataclass default applies. `run_predict` resolves `"auto"` once at entry.
- `run_predict` resolves `"auto"` via `decide_eval_batch_size` (same sibling).
- The forward loop becomes `for image_chunk in _chunked(image_paths, batch_size): for group in _chunked(prompts, MULTIPLEX_CAP): ...`.
- Postprocess remains per-row (via the same `_row_outputs` helper), so `queries_to_coco_results`'s `batch == 1` hard-assert holds.
- The warmup call (lines 327-333) is unchanged — single image, single class is fine for warmup and avoids touching the rest of the dry-run path.

---

## §7 Configuration

### New: `MultiplexConfig`

```python
class MultiplexConfig(_Strict):
    """Multiplex forward knobs.

    classes_per_forward: number of class prompts per multiplex forward pass.
    Capped at SAM 3.1's MULTIPLEX_CAP=16. Setting 1 reduces to the legacy
    per-class regime within the same code path.
    """
    classes_per_forward: int = Field(default=16, ge=1, le=16)
```

Placed in `src/custom_sam_peft/config/schema.py` near `BoxHintSchedule`.

### `TrainHyperparams.multiplex`

Add `multiplex: MultiplexConfig = Field(default_factory=MultiplexConfig)` to `TrainHyperparams` (`schema.py:389-409`). Listed in advanced section.

### `EvalConfig.batch_size`

New field (§6). Default `"auto"`. Existing user fields preserved.

### `PredictOptions.batch_size`

Today's `PredictOptions.batch_size: int` (`predict/runner.py:67`) becomes `int | Literal["auto"]` with default `"auto"` set directly on the dataclass field. Symmetric with `EvalConfig.batch_size`. CLI shell forwards the user's value verbatim (or omits the kwarg to take the dataclass default); the `"auto"` sentinel is resolved inside `run_predict`, the same way `run_eval` resolves it. Notebook users importing `PredictOptions` see and can override the auto path directly without needing to know about CLI-layer translation.

### `LossConfig` docstring

Update `src/custom_sam_peft/config/_internal.py:34-57`: the line *"No `w_cls`: discrimination across classes comes from running one forward pass per class prompt."* changes to *"No `w_cls`: SAM 3.1's multiplex forward provides open-vocabulary discrimination directly via per-text-embedding queries; per-class `w_cls` is unneeded."* Defaults unchanged.

### `MULTIPLEX_CAP`

Module-level constant in `models/sam3.py` (§4). No corresponding YAML knob — the cap is a model property, not user-tunable.

### No env vars

No new env vars added by this PR. The existing `CUSTOM_SAM_PEFT_VRAM_HEADROOM_GIB` is reused unchanged by `decide_eval_batch_size`.

### Example configs

No edits required to `configs/examples/*.yaml`. The shipped defaults (`text_prompt.mode=present_plus_negatives` with `negatives_per_image=4`, dataset class vocab provides positives) already produce ≤16 distinct classes per typical COCO batch, so the auto-chunk path is exercised only when datasets have very many distinct classes per batch (LVIS-1203 or aggressive `negatives_per_image`).

---

## §8 VRAM math

**File:** `src/custom_sam_peft/presets.py`.

### `decide_eval_batch_size(image_size, classes_per_forward=16)`

Sibling of `decide_preset` (`presets.py:271-334`). Returns `(batch_size: int, predicted_bytes: int, provenance: Literal["calibrated", "analytic"])`. Used by `run_eval` and `run_predict` to resolve `"auto"`.

Key differences from `decide_preset`:

- **No optimizer state.** `_optimizer_bytes` is excluded.
- **No backward activations.** Activations are scaled by a `forward_only_factor = 0.25` constant: forward-only memory is roughly a quarter of the train-step probe (train captures forward + backward + retained graph; eval captures only forward, no graph).
- **K_eval fixed at 16.** Multiplex packs K class prompts per forward; activations per row don't scale linearly with K (image-encoder cost is shared), so the activation estimate is keyed off `B` only, with K folded into `forward_only_factor`'s empirical calibration.
- **Search space:** `B ∈ [1, 64]`. The largest feasible B wins. (`decide_preset` searches over `method`, `r`, `batch`, `ckpt`; here only `batch`.) The upper bound is set high enough that no current GPU caps out on it — at K_eval=16, decoder activation dominates and `B > 16` is only feasible on 80 GB+ devices. The analytic prediction is a best-effort heuristic; the eval OOM ladder (`_eval_forward_with_oom_ladder`, §6) is the authoritative safety net that catches over-prediction at runtime.

### Reuse

- `_load_cache(image_size, gpu_name)` is reused as-is. The cached `activation_bytes_per_example` (probed at LoRA r=4 with one forward+backward) is multiplied by `forward_only_factor` for the eval estimate.
- `_headroom_bytes()` is reused as-is — same `CUSTOM_SAM_PEFT_VRAM_HEADROOM_GIB` env var, same default `1 GiB`.
- `_predicted_bytes` is **extended** to take a mode parameter (`Literal["train", "eval"]`); the eval mode skips `_optimizer_bytes` and `_adapter_bytes`, and multiplies `_activation_bytes` by `forward_only_factor`. Train-mode callers are unchanged.

### Fallback

When no calibration cache is present, the analytic path uses `BASE_ACTIVATION_AT_1024 * (image_size / 1024)**2 * forward_only_factor` per example. When CUDA is unavailable, returns `(1, predicted_bytes=0, provenance="analytic")` and logs once.

### Constants

`forward_only_factor = 0.25` is a calibration constant. It is conservative (under-estimating forward-only memory pushes us toward smaller B; the runtime OOM ladder catches a too-large B). May need adjustment if §9's GPU test proves too tight or too loose; see §9 for the threshold-as-knob note.

---

## §9 Testing strategy

CPU-heavy, GPU-minimal (per [CLAUDE.md memory](feedback_gpu_vs_cpu_testing.md)). Two GPU tests; everything else lands on CPU.

### CPU unit tests (new or updated)

| Test file | Coverage |
|-----------|----------|
| `tests/unit/test_sam3_wrapper.py` | `_validate_inputs` accepts 1..16 classes per prompt; rejects 0, 17, mismatched class lists across the batch. The K=1 case still passes through. |
| `tests/unit/test_sam3_adapter.py` | New. Mock `forward_grounding`; assert `find_input.img_ids = arange(B).repeat_interleave(K)` and `find_input.text_ids = arange(K).repeat(B)` for `(B, K) ∈ {(1,1), (2,3), (4,16)}`. |
| `tests/unit/test_build_geometric_prompt.py` | Existing test extended: `_build_geometric_prompt` accepts `n_cols ≠ len(box_hints)` → raises; produces correctly-shaped `(N_max, n_cols, 4)` and `(n_cols, N_max)` tensors. |
| `tests/unit/test_train_loop.py` | Update mock to return `(B·K_g, Q, ...)` shapes; assert one `total_loss` call per group; assert `(loss / (G * grad_accum_steps)).backward()`; assert OOM-ladder closure builds a length-`|micro|·K_g` flat hint list. |
| `tests/unit/test_train_loop_legacy_k1.py` | **New.** Run train_step with `classes_per_forward=1` on the tiny-fixture dataset; assert per-step RNG draw order (collected via a seeded Bernoulli probe) matches today's main; loss values numerically equal main's. **Locked-decision regression guard.** |
| `tests/unit/test_train_loop_multiplex.py` | **New.** K=4 case on tiny fixture; assert auto-chunk triggers when K_total > MULTIPLEX_CAP via `caplog.records` containing the INFO message; assert `n_classes` in `StepResult` still equals `len(classes_in_batch)`. |
| `tests/unit/test_evaluator.py` | Update to assert flat `(image_chunk × group)` iteration order; `_row_outputs` helper produces per-row dicts that `queries_to_coco_results` accepts. |
| `tests/unit/test_eval_oom_ladder.py` | **New.** Synthetic OOM raised on the 2nd image-chunk; assert B halves once and replay completes; `_LOG.warning` emitted exactly once. |
| `tests/unit/test_decide_eval_batch_size.py` | **New.** Mocked CUDA props + cache: assert chosen B given probed activation; assert analytic fallback when cache misses; assert CPU fallback returns `B=1`. |
| `tests/unit/test_predict_runner.py` | Update existing predict tests: assert flat `(image_chunk × group)` iteration; warmup call still single-image / single-class. |

### GPU tests (the only two)

| Test | Coverage |
|------|----------|
| `tests/gpu/test_sam3_real_load.py` (extend) | Add ONE assertion that a real K=8 multiplex forward produces `pred_logits.shape[0] == B * 8` and finite outputs. Same fixture, no new GPU minutes. |
| `tests/gpu/test_multiplex_vram.py` | **New, marked `requires_compatible_gpu` + `requires_checkpoint`** (the existing project markers; see `tests/conftest.py:21-25`). Run real `decide_eval_batch_size` at `image_size=1008`; assert the chosen `B` runs a real K=16 forward to completion without OOM; assert `torch.cuda.max_memory_allocated()` ≤ 4× the predicted bytes. The 4× ceiling is a conservative regression guard, not a tightness check. |

### Benchmark (not in CI)

`scripts/bench_multiplex_throughput.py` — wall-clock comparison of K=1 vs K=16 on COCO-80 mini-fixture at `image_size=1008`. Reported in the PR description; not gated.

### Calibration constants

- `forward_only_factor = 0.25` in `presets.decide_eval_batch_size`.
- `4×` predicted-bytes ceiling in `test_multiplex_vram.py`.

Both are conservative regression guards. **They may need adjustment if the test proves flaky across GPU SKUs**, but neither becomes a user-facing config knob. Flakiness should trigger a tightening review, not parameterization.

---

## §10 Risks

### R1 — Per-step loss magnitude shift

The per-step loss the user sees on TensorBoard moves under multiplex. Today the trainer accumulates `accum[k] += float(class_losses[k])` across `finite_class_count` classes and divides by it; under multiplex, the per-group `total_loss` already reduces across the `(B·K_g, ...)` axis via `.mean()`, so per-step magnitudes will look smoother but shift by a constant factor at K=1 → K=16. Documented in the changelog; not a correctness issue.

### R2 — Hungarian matcher per-row cost

`HungarianMatcher` (`models/matching.py:120-176`) is already per-row (iterates `for i in range(b)` where `b = outputs.obj_logits.shape[0]`). Under multiplex, `b = B·K_g`, so the matcher runs `B·K_g` times per group — same total per-row cost as today's `B · K` calls. No matcher change needed.

### R3 — Seed-bit equivalence (tightened from locked decision §5, §7)

- **At K=1:** Per-step RNG draws (Bernoulli for box hints, matcher's `linear_sum_assignment` non-determinism) preserve order vs today's main. The `_chunked([c], 1)` produces single-class groups in sorted class order; iteration is image-minor within each (only one class per group); per-image Bernoulli order matches today's `for c: for i:` order at K=1 (single class collapses both orders). `test_train_loop_legacy_k1.py` (§9) is the regression guard.
- **At K>1:** Order shifts. Today the loop is `for c: for i:` (B Bernoulli draws per class, K times). Under multiplex K=16, it's one group of `B·16` draws in image-major order. Total draws per step are identical (`B · K_total`) but ordering differs, so seed reproducibility against today's main is not guaranteed for K>1. Documented in changelog.

### R4 — Per-row NaN contamination

`total_loss` reduces by `.mean()` across the `(B·K_g, ...)` dim. A single non-finite row in a finite group contaminates the whole group's total and causes the group's contribution to be dropped via the existing `torch.isfinite` check. We lose finer-grained recovery at high K. Mitigation: ship as-is; if observed in practice, follow up with per-row finiteness masking (§12).

### R5 — Loss-weight calibration

`w_mask=w_obj=w_presence=1` defaults were chosen at the K=1 off-distribution regime. Under multiplex (K=16), the loss signal returns to the trained distribution, so the defaults are more likely correct, not less — but unverified. §12 flags a future ablation; defaults stay put for this PR.

---

## §11 Rollout

Single PR. **Defaults change to new behavior.** No back-compat shims, no feature flag.

### Changelog entries

- **feat:** SAM 3.1 multiplex forward — one forward per ≤16-class group in train, eval, and predict. New `train.multiplex.classes_per_forward` (1..16, default 16), new `eval.batch_size: int | "auto"` (default `"auto"`).
- **perf:** Multi-class training/eval workloads (COCO ≥80 classes, LVIS) see significantly higher throughput; see PR description for `scripts/bench_multiplex_throughput.py` numbers.
- **breaking (numeric):** Per-step loss magnitudes shift vs prior versions. The `LossConfig` defaults are unchanged; if you've manually tuned `w_mask`/`w_obj`/`w_presence`, re-validate.
- **breaking (numeric):** Per-step RNG draw order shifts at K>1; runs are not seed-bit-equivalent to <0.8.0 for K>1. Bit-equivalence holds at `train.multiplex.classes_per_forward=1`.
- **escape hatch:** Set `train.multiplex.classes_per_forward: 1` to recover today's per-class iteration (single code path).

---

## §12 Out-of-scope follow-ups

Files at PR-open time, one issue each.

| Title | Why deferred |
|-------|--------------|
| Joint `(B, K_eval)` VRAM tuning | K_eval is fixed at 16 in `decide_eval_batch_size`; jointly searching adds a dimension we haven't validated empirically. |
| Eval image-size bucketing | Out of scope per §2; would invalidate the single-`batch_size` knob. |
| Loss-weight ablation under multiplex | Defaults are unchanged in this PR; ablation is a separate calibration study. |
| Predict CLI image-size / `original_hw` bucketing | Same shape as eval bucketing. |
| Per-class loss attribution for TensorBoard | Locked decision §4 — aggregate-only this PR. |
| Cost estimator (#109) update for multiplex throughput | The steps-per-second lookup table needs new constants under multiplex; orthogonal. |
| Per-row NaN finiteness masking | R4 mitigation if it occurs in practice. |

---

## §13 Acceptance criteria

Concrete and checkable. Lifted and refined from issue #22's acceptance list.

1. `Sam3Wrapper._validate_inputs` accepts `1..MULTIPLEX_CAP` classes per `TextPrompts` and rejects out-of-range plus mismatched-across-batch class lists; the `len(p.classes) != 1` raise is gone.
2. `MULTIPLEX_CAP = 16` constant exists in `src/custom_sam_peft/models/sam3.py` and is the single source of truth (no second literal `16` in trainer/evaluator/predict; all cite the constant).
3. `_Sam3ImageAdapter.forward` builds `FindStage` with `img_ids = arange(B).repeat_interleave(K)` and `text_ids = arange(K).repeat(B)`; the geometric prompt has `(N_boxes_max, B·K, 4)` shape.
4. `train_step` runs one forward and one `total_loss` call per `_chunked` group; backward divides by `G * grad_accum_steps`; the per-class `for c in classes_in_batch` loop is gone.
5. `_train_step_with_oom_ladder` still halves the image dim within a group; closure operates on the flat row layout.
6. NaN policy: a step is skipped only when every group is non-finite. `nan_streak` increments unchanged.
7. `Evaluator._iter_predictions` iterates `(image_chunk, class_group)` flat; `_row_outputs` helper drives `queries_to_coco_results` per row.
8. `EvalConfig.batch_size: PositiveInt | Literal["auto"]` schema field exists (default `"auto"`); `run_eval` resolves `"auto"` via `decide_eval_batch_size` before constructing `Evaluator`.
9. `_eval_forward_with_oom_ladder` exists; mirrors train's ladder with no grad-ckpt rung and sticky B halving; emits ≤1 warn per `evaluate` call.
10. `MultiplexConfig` schema validates `classes_per_forward ∈ [1, 16]`; `TrainHyperparams.multiplex` exists with `MultiplexConfig` default factory.
11. `LossConfig` docstring in `config/_internal.py:34-57` no longer claims "one forward pass per class prompt."
12. `presets.decide_eval_batch_size(image_size, classes_per_forward=16)` exists; returns `(batch_size, predicted_bytes, provenance)`; reuses `_load_cache`, `_headroom_bytes`; folds `forward_only_factor = 0.25` into activation estimate; falls back to analytic on cache miss and `B=1` on CPU.
13. Predict CLI: `PredictOptions.batch_size` accepts `int | "auto"`; `run_predict` resolves `"auto"` and iterates `(image_chunk, class_group)`; warmup is unchanged.
14. CPU unit tests in §9 exist and pass; in particular `tests/unit/test_train_loop_legacy_k1.py` proves K=1 numerical and RNG-order equivalence to today's main.
15. GPU `tests/gpu/test_multiplex_vram.py` exists, is marked `requires_compatible_gpu` + `requires_checkpoint` (matching `tests/conftest.py:21-25`), runs a real K=16 forward at the chosen B without OOM, and asserts `max_memory_allocated ≤ 4 × predicted_bytes`.
16. The existing `tests/gpu/test_sam3_real_load.py` carries one extra assertion that a real K=8 multiplex forward produces `pred_logits.shape[0] == B * 8` and finite outputs.
17. Changelog (§11) entries are present in the PR description.
