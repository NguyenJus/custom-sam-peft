# spec/gpu-coverage-assessment — Close the GPU-Coverage Gaps Uniquely Owned by #65

**Status:** Draft (2026-05-21)
**Tracking issue:** #65
**Scope:** Audit the post-`v0.6.1` GPU/CPU test suite against issue #65's eight coverage dimensions, categorize each dimension as covered well / covered weakly / not covered, cross-reference dimensions now owned by dedicated follow-up issues (#23, #64, #68, #74, #78, #79), and land the test additions for the gaps that remain uniquely owned by #65. Adds one new release-tier GPU test (`tests/gpu/test_real_train_qlora_resume.py`), extends three existing GPU tests with eval-metric and PEFT-scope assertions, adds four new CPU unit tests (`tests/unit/test_evaluator_schema.py`, `tests/unit/test_peft_scope_coverage.py`, `tests/unit/test_trainer_nan_behavior.py`, `tests/unit/test_checkpoint_roundtrip.py`), and extends two existing CPU/integration tests (`tests/integration/test_train_resume.py`, `tests/integration/test_train_end_to_end.py`). No change under `src/custom_sam_peft/`, no new YAML config, no new dependency. Closing #65 with this PR also satisfies the `GPU test gate decision shipped — #65` line on issue #70's pre-v1.0 checklist.

---

## 1. Context

### 1.1 Why this exists

Issue #65 was opened the day the GPU smoke gate stabilized (PR #58, merged from branch `manual-gpu-pass-44`). It is an audit question, not a feature request: with three release-tier `gpu`-marked tests and nine inspection-tier `gpu_inspection`-marked tests now green on T4, does the suite actually catch the regressions we care about, and what is it blind to? The issue body enumerates eight dimensions worth examining and asks for "a short writeup that categorizes the above into covered well / covered weakly / not covered, picks the top 2–3 gaps worth filling before v1, files concrete follow-up issues for each."

The issue's framing has aged. Two of the eight dimensions have been resolved in-place (closed issues #60 activation checkpointing, #44 manual GPU pass). Five others were filed as dedicated issues after #65 was opened — #23 (model variants), #64 (headless GPU CI), #68 (VRAM-floor tightening), #74 (`csp predict`), #78 (video brainstorm), #79 (older CUDA) — and each is the right home for the work the corresponding dimension demands. The audit's job is to point at those owners, not re-litigate them.

What remains uniquely owned by #65, after the cross-references settle, are four gaps: `vision` / `all` PEFT scopes (no per-scope wiring assertion on the real module tree), mid-training NaN abort behavior (no pin), eval-metric assertions (Evaluator runs in the 50-step tests but its output is never gated; the e2e test asserts mAP numeric but not finite), and GPU resume (CPU stub-only today; bnb 4-bit quant-state survival across save/load is untested). The user's brainstorm-locked scope is the assessment doc **plus** the test additions for these four gaps — and equally, **deferring** to CPU anything that doesn't need a real GPU to fail.

### 1.2 v1.0 gate dependency

Issue #70 maintains the pre-v1.0 checklist. One line reads `GPU test gate decision shipped — #65`. Despite #65 carrying `priority:low`, closing it with a landed PR is therefore on the v1.0 critical path. The spec's exit criteria (§5) deliberately produce a closure-ready state: doc filed, gaps closed or owned, follow-up issues opened, #65 close comment templated in §9.

---

## 2. Coverage Map

The eight #65 dimensions, post-cross-reference, categorize as follows. "Covered well" means the existing suite already asserts the contract end-to-end; "covered weakly" means the contract is partially asserted but the assertion is too loose to catch a real regression; "not covered" means nothing in the suite would notice if the behavior broke.

| # | Dimension | Bucket | Owner |
| - | --------- | ------ | ----- |
| 1 | `vision_decoder` PEFT scope forward+backward | covered well | release tier already (`test_real_train_overfits.py`, `test_real_train_qlora.py`) |
| 2 | `vision` / `all` PEFT scopes (forward+backward, real module tree) | covered weakly | **#65 (this spec)** — inspection-tier asserts target-name match at default scope only; no per-scope wiring assertion on real ViT-Det |
| 3 | Mid-training finite-scalar logging | covered well | release-tier 50-step tests already assert `math.isfinite(v)` for every tracker scalar |
| 4 | Mid-training NaN abort behavior | not covered | **#65 (this spec)** — `train/loop.py:137` raises `RuntimeError` after `nan_abort_after` consecutive non-finite micro-steps; behavior is unpinned by any test |
| 5 | OOM recovery | not covered | doc-only concern, not a test gap — see §4 below |
| 6 | Missing-checkpoint error path | covered well | `tests/unit/test_load_sam31_missing_keys_filter.py` and CLI doctor unit tests cover the canonical missing-file message |
| 7 | Malformed-dataset error path | covered weakly | **#65 (this spec)** — CPU concern; no GPU dependency; covered by extending `tests/integration/test_train_end_to_end.py` |
| 8 | Eval-metric assertion on 50-step tests | covered weakly | **#65 (this spec)** — Evaluator runs but `metrics.json` is never inspected |
| 9 | Eval-metric assertion on e2e test | covered weakly | **#65 (this spec)** — `isinstance(mAP, (int, float))` allows `NaN`, `inf`, and negative values |
| 10 | GPU resume (save → load → continue) | not covered | **#65 (this spec)** — CPU stub resume test covers LoRA logic; QLoRA bnb 4-bit quant-state continuity is untested |
| 11 | Model variants (`facebook/sam3` alongside `facebook/sam3.1`) | not covered | **#23** owns |
| 12 | VRAM-floor / hardware-tier variance | not covered | **#68** owns |
| 13 | Older-CUDA / driver-version variance | not covered | **#79** owns |
| 14 | Video tracking smoke | not covered | **#78** owns |
| 15 | Headless `pytest -m gpu` CI | not covered | **#64** owns |
| 16 | `csp predict` GPU smoke | not covered | **#74** adds when the command lands |

Bucket commentary, one paragraph each:

**Covered well.** The three release-tier tests (`test_real_train_overfits.py`, `test_real_train_qlora.py::test_qlora_overfits_in_50_steps`, `test_qlora_smoke_fast`) and the e2e CLI test give real end-to-end signal for the default `vision_decoder` scope under both LoRA and QLoRA, plus finite-scalar logging through the tracker. Missing-checkpoint coverage rides on CPU unit tests. These dimensions need no new work.

**Covered weakly.** Four dimensions assert *something* but the assertion is loose enough that a real regression could slip past: `vision` / `all` scopes never get exercised against real ViT-Det module names; the 50-step tests run `Evaluator.evaluate()` but never inspect the returned `MetricsReport`; the e2e test only asserts mAP is a number (not finite, not `>= 0`); malformed-dataset error paths are exercised in passing but not pinned with specific message contracts. §3 lists per-gap interventions.

**Not covered.** Mid-training NaN abort is the only behavioral gap with no existing assertion at all — `train/loop.py`'s `nan_abort_after` policy is undocumented by any test. GPU resume is also fully open for QLoRA: the CPU stub resume test asserts adapter-tensor equality on a LoRA stub, but bnb 4-bit `quant_state` survival across `save_full_state` → `load_full_state` is a real-only failure mode. OOM recovery is intentionally NOT a test gap — the project's stance (echoed in `docs/testing/gpu-test-policy.md` §5.5) is that OOM is a configuration problem caught by the VRAM ceiling assertion, not a behavior to recover from in-loop.

---

## 3. Owner Map

The five dimensions now owned by dedicated issues are deferred to those issues' specs. This spec does not redesign their decisions; it only confirms they are the right home so the audit does not duplicate.

| Dimension | Owner | Why this spec defers |
| --------- | ----- | -------------------- |
| Model variants (`facebook/sam3`) | **#23** | The variant matrix is a loader / weight-resolution problem, not a test-shape problem. #23's spec will decide which variants ship and which get GPU smokes. |
| Headless `pytest -m gpu` CI | **#64** | #53 / #62 already produced the GCP / Lambda Labs / Modal evaluation; #64 owns the implementation. Spinning a runner is out of scope for an audit. |
| VRAM-floor tightening (QLoRA on 8 GB, LoRA on 16 GB) | **#68** | The 14 GB / 10 GB ceilings in `gpu_smoke_lora.yaml` / `gpu_smoke_qlora.yaml` are pinned to T4 per `2026-05-19-gpu-test-policy-design.md` §5.4. Lowering them is a tuning exercise on its own branch. |
| `csp predict` GPU smoke | **#74** | The CLI command does not exist yet. The smoke test will be authored alongside the command. |
| Video tracking | **#78** | Per the closed-but-influential #60, activation checkpointing was deferred as a v1+ video prerequisite. The video test plan rides on #78's brainstorm. |
| Older-CUDA hardware variance | **#79** | Driver / CUDA-version compatibility is a matrix problem distinct from the audit. #79 owns whether (and how) to test against pre-12.x CUDA. |

If a reviewer reads this spec and asks "what about X-from-#65?", the answer is in this table: either X is an owned issue (defer) or X appears in §6 (closed in this PR).

---

## 4. Goals & Non-Goals

**Goals.**

- Produce a self-contained assessment doc (this file) that a cold reader can land on in three months and understand what #65 closed, what it deferred, and why.
- Close the four uniquely-#65-owned gaps via concrete test additions: GPU resume for QLoRA (T1), eval-metric finiteness on the three release-tier tests (T2/T3/T4), per-scope wiring assertions on real ViT-Det at inspection tier (T5/T6).
- Defer everything CPU-testable to CPU: Evaluator schema (C1), PEFT scope wiring on a stub (C2), trainer NaN-loss behavior (C3), resume-state continuity on optimizer / scheduler / RNG (C4), malformed-dataset error contracts (C5).
- Satisfy the `GPU test gate decision shipped — #65` line in #70's pre-v1.0 checklist.
- File two follow-up issues to track work that was raised by the audit but consciously deferred (§8).

**Non-goals.**

- No threshold calibration on the eval metrics. T2/T3/T4 assert finiteness and non-negativity only. A real mAP / IoU floor requires a calibration GPU run (no contributor has the data to set a non-trivial floor today) and is deferred via the issue stub in §8.
- No new `csp predict` test. Owner: #74.
- No retag of existing markers. The `gpu` / `gpu_inspection` semantics from `2026-05-19-gpu-test-policy-design.md` are preserved verbatim.
- No new YAML config. T1 reuses `configs/examples/gpu_smoke_qlora.yaml` with a per-test override (`train.save_every=25`, `train.epochs=2` → ~50 steps total, halt at step 25).
- No `src/custom_sam_peft/` change. This is purely test additions. If the implementer discovers that C3 surfaces an actual bug in `train/loop.py`'s NaN behavior, the fix is a separate issue, not this PR.
- No VRAM-ceiling change in either smoke YAML. T1's QLoRA resume reuses the existing 10 GB ceiling.
- No additional `all`-scope GPU smoke. The `all` scope is regex `.*`, would dominate VRAM, and is unlikely to fit T4 under the existing ceilings. Deferred via the §8 stub cross-referencing #68.
- No OOM recovery test, no multi-GPU test, no MIG-slice test. Outside the audit's defined scope.

---

## 5. Files Touched / Module Layout

```text
docs/superpowers/specs/
  2026-05-21-gpu-coverage-assessment-design.md     # THIS FILE — assessment + intervention spec

tests/gpu/
  test_real_train_qlora_resume.py                  # NEW — T1, QLoRA resume smoke
  test_real_train_overfits.py                      # CHANGED — T2, assert metrics.json mAP finite
  test_real_train_qlora.py                         # CHANGED — T3, assert metrics.json mAP finite
                                                   #          (test_qlora_overfits_in_50_steps only)
  test_run_end_to_end_gpu.py                       # CHANGED — T4, tighten mAP to math.isfinite + >= 0

tests/integration/
  test_peft_lora_real.py                           # CHANGED — T5, add apply_lora(scope="vision") subtest
  test_peft_qlora_real.py                          # CHANGED — T6, add apply_qlora(scope="vision") subtest
  test_train_resume.py                             # CHANGED — C4, monotone optimizer/scheduler ≥ assertions
  test_train_end_to_end.py                         # CHANGED — C5, malformed-dataset error contracts
                                                   #          (extend in place; do NOT create a sibling)

tests/unit/
  test_evaluator_schema.py                         # NEW — C1, MetricsReport schema + edge cases
  test_peft_scope_coverage.py                      # NEW — C2, scope→trainable-set wiring on stub
  test_trainer_nan_behavior.py                     # NEW — C3, pin NaN-abort-after-N policy
  test_checkpoint_roundtrip.py                     # NEW — C4, save_full_state ↔ load_full_state bit-equality
```

No file under `src/custom_sam_peft/` is modified. No CI workflow is modified. No new YAML, no new Python module under `src/`, no new dependency.

---

## 6. Interventions

This section gives the per-file contract for each addition / extension. The implementer writes the tests; the spec specifies **what** each test asserts, **what** is monkeypatched, and **which fixtures** it uses.

### 6.1 GPU additions

#### T1 — `tests/gpu/test_real_train_qlora_resume.py` (NEW, release tier)

**Markers:** module-level `pytestmark = [pytest.mark.gpu, pytest.mark.requires_compatible_gpu, pytest.mark.requires_checkpoint]`, plus per-test `@pytest.mark.requires_bnb` and `@pytest.mark.skipif(not _bnb_available(), reason=...)` matching the existing `test_real_train_qlora.py` pattern.

**Config:** Reuse `configs/examples/gpu_smoke_qlora.yaml`. The shipped YAML has `epochs: 25, batch_size: 1` → with 2-image `tiny_coco` → ~50 grad steps. Apply overrides at load time via the existing `load_config(CONFIG_PATH, overrides=[...])` seam:
- `data.train.*` and `data.val.*` → `tiny_coco_dir`.
- `run.output_dir` → `tmp_path`.
- `train.save_every=25` (lands one checkpoint at step 25, midway through the 50-step run).
- `train.log_every=1` (so every step's scalar is captured for finiteness checks).

The total grad-step budget across both phases is the same 50 steps that `test_qlora_overfits_in_50_steps` already pays — T1 splits the existing budget across a save/load boundary rather than adding net new training time.

**Flow:** Two phases in one test, both under one `torch.cuda.reset_peak_memory_stats()` window.

Phase A: build `cfg_short` (`train.epochs=13` → ~26 grad steps, just past the `save_every=25` cutoff to ensure the checkpoint lands and the post-checkpoint step writes); call `run_training(cfg_short)` with the existing `_RecordingTracker` monkeypatch; locate the most recent checkpoint at `run_dir / "checkpoints" / "step_25"`; capture `losses_a = [s["loss/total"] for ...]`.

Phase B: build `cfg_full` (`train.epochs=25` → the shipped 50-step budget); call `run_training(cfg_full, resume_from=<phase A checkpoint>)`. `run_training` already exposes `resume_from: Path | None` (`src/custom_sam_peft/train/runner.py:34`), which threads through to `Trainer.fit(..., resume_from=...)`. Capture `losses_b`.

**Assertions.**

- `losses_a`, `losses_b` non-empty.
- Every scalar in both phases is `math.isfinite`.
- `losses_b[-1]` is finite (specifically: not `NaN`, not `inf`).
- Final adapter state contains at least one `lora_` parameter, and every `lora_` parameter passes `torch.isfinite(...).all()`.
- Peak VRAM across both phases ≤ 10 GB (the existing `test_real_train_qlora.py` ceiling — `VRAM_CEIL_GB = 10.0`).

**Explicitly NOT asserted.** No comparison to an uninterrupted reference run. No bit-equality. No loss-ratio. The justification: the test's purpose is to prove bnb 4-bit `quant_state` survives the save/load cycle without corrupting subsequent gradients — finiteness is the binary signal, and a parallel reference would double the GPU time without adding signal (the LoRA-stub CPU resume test in `tests/integration/test_train_resume.py` already covers tensor-equality for the deterministic path).

#### T2 — `tests/gpu/test_real_train_overfits.py` (CHANGED)

After the existing `peak_vram_gb` assertion, locate the run directory written by `run_training(cfg)` and add:

```python
import json
runs = sorted(tmp_path.glob("gpu-smoke-lora-*"))
assert runs, f"no run dir under {tmp_path}"
metrics = json.loads((runs[-1] / "metrics.json").read_text())
assert "overall" in metrics, f"metrics.json missing 'overall': {metrics}"
mAP = metrics["overall"].get("mAP")
assert isinstance(mAP, (int, float)) and math.isfinite(mAP) and mAP >= 0.0, (
    f"overall.mAP not finite/non-negative: {mAP}"
)
```

The run-dir glob pattern matches the existing `cfg.run.name` from `gpu_smoke_lora.yaml` (`gpu-smoke-lora`). The `math` import is already present. No other change.

#### T3 — `tests/gpu/test_real_train_qlora.py::test_qlora_overfits_in_50_steps` (CHANGED)

Same intervention as T2, with run-dir glob `gpu-smoke-qlora-*`. Apply ONLY to `test_qlora_overfits_in_50_steps`. Do NOT touch `test_qlora_smoke_fast`: that test deliberately monkeypatches `Evaluator` to a no-op (cf. `_SkipEvaluator` at lines 124–131), so it never writes a `metrics.json` body that could be inspected — and its existing finite-scalar assertion already covers the contract it exists to pin.

#### T4 — `tests/gpu/test_run_end_to_end_gpu.py` (CHANGED)

Tighten the existing mAP assertion. Current (line 59):

```python
assert isinstance(metrics["overall"].get("mAP"), (int, float))
```

Replace with:

```python
import math
mAP = metrics["overall"].get("mAP")
assert isinstance(mAP, (int, float)) and math.isfinite(mAP) and mAP >= 0.0, (
    f"overall.mAP not finite/non-negative: {mAP}"
)
```

`math` is not currently imported in the file; add the import.

#### T5 — `tests/integration/test_peft_lora_real.py` (CHANGED, inspection tier)

Append a new test function `test_apply_lora_vision_scope_targets_only_vision_backbone`. No new file, no new fixture. Module-level `pytestmark` already includes the inspection marker.

**Flow:** Load the real wrapper via `load_sam31(ModelConfig())`. Call `apply_lora(w, PEFTConfig(method="lora", scope="vision"))` — note the explicit non-default scope. Inspect `w.model.model.named_parameters()`.

**Assertions:**

- `lora_names = [n for n, _ in w.model.model.named_parameters() if "lora_" in n]` is non-empty.
- `any("vision_backbone" in n for n in lora_names)` — vision-trunk targets present.
- `all("transformer.decoder" not in n for n in lora_names)` — decoder targets EXCLUDED at `vision` scope.
- `all("mask_decoder" not in n for n in lora_names)` — mask-decoder targets EXCLUDED.

The test does NOT call forward — inspection tier remains forward-free per `2026-05-19-gpu-test-policy-design.md` §5.6. The cost is dominated by `load_sam31`, which is already paid by the file's other tests.

#### T6 — `tests/integration/test_peft_qlora_real.py` (CHANGED, inspection tier)

Append a new test function `test_apply_qlora_vision_scope_targets_only_vision_backbone` with the same `skipif(not _bnb_available(), ...)` per-test decorator that the existing QLoRA tests use. Mirror T5's contract:

- Call `apply_qlora(w, PEFTConfig(method="qlora", scope="vision"))`.
- Assert `vision_backbone` present, `transformer.decoder` absent, `mask_decoder` absent in `lora_` parameter names.

### 6.2 CPU additions

#### C1 — `tests/unit/test_evaluator_schema.py` (NEW)

**Markers:** none beyond pytest defaults. Standard CPU unit test.

**Target API:** `custom_sam_peft.eval.metrics.MetricsReport`, `compute_coco_map`, and `custom_sam_peft.eval.evaluator.Evaluator`. The implementer reads `src/custom_sam_peft/eval/metrics.py` (already studied while writing this spec — see schema below) and `src/custom_sam_peft/eval/evaluator.py` for the call site.

**Schema invariant (from `metrics.py`):** `MetricsReport(overall: dict[str, float], per_class: dict[str, dict[str, float]], n_images: int, n_predictions: int)`. `overall` contains `mAP` always; `mAP_50` only when `0.5 in iou_thresholds`; `mAP_75` only when `0.75 in iou_thresholds`. `per_class` maps class name → `{"AP": float, "AP_50": float?}`; classes with no GT are skipped (cf. `metrics.py:93`). Empty predictions return a zeroed `MetricsReport` with `n_predictions=0` (cf. `metrics.py:54-61`).

**Test cases (one function per case):**

- `test_empty_predictions_returns_zeroed_report` — call `compute_coco_map(predictions=[], ground_truth=<tiny COCO loaded via pycocotools>, iou_thresholds=[0.5, 0.75, 0.95], include_per_class=True)`. Assert `report.overall == {"mAP": 0.0, "mAP_50": 0.0, "mAP_75": 0.0}`, `report.per_class == {}`, `report.n_predictions == 0`, `report.n_images == 2` (tiny_coco has 2 images).
- `test_iou_thresholds_pick_slices` — pass `iou_thresholds=[0.5]`. Assert `"mAP_50" in report.overall` and `"mAP_75" not in report.overall`. Pass `iou_thresholds=[0.75]`; assert the inverse.
- `test_overall_keys_finite` — pass a single synthetic prediction (one RLE mask, one image, one category) generated from the GT mask itself (perfect prediction). Assert every value in `report.overall` is `math.isfinite(...)` and `>= 0.0` and `<= 1.0`.
- `test_per_class_skips_classes_without_gt` — construct a 2-image GT with one category. Pass `include_per_class=True` and confirm `per_class` is keyed by the category name (string), and each row has `"AP"` finite.
- `test_include_per_class_false_returns_empty_per_class` — same input, `include_per_class=False`; assert `report.per_class == {}` and `report.overall["mAP"]` still populated.

**Fixtures:** Use `tiny_coco_dir` (already exposed in `tests/conftest.py` per the existing GPU smoke pattern). The implementer loads it via `pycocotools.coco.COCO(str(tiny_coco_dir / "annotations.json"))`.

#### C2 — `tests/unit/test_peft_scope_coverage.py` (NEW)

**Markers:** none.

**Target API:** `custom_sam_peft.peft_adapters.lora.apply_lora`, `SCOPE_TARGETS`, plus the stub at `tests/fixtures/tiny_sam3_lora_stub.py::make_stub_wrapper` (with `working=True` for the forward-pass cases). The `FIXTURE_SCOPE_PATTERNS` dict in that same fixture provides the regex set that mirrors `SCOPE_TARGETS` against the renamed stub subtrees — pass these via `PEFTConfig(target_modules=FIXTURE_SCOPE_PATTERNS[scope])` to drive the stub.

**Test cases:**

- `test_scope_vision_targets_only_vision_subtree` — `scope="vision"`. Apply to a `working=False` stub (no forward needed). Assert that `lora_` parameters appear under `vision_trunk.blocks.*` and NOT under `transformer_decoder.*` or the `neg_control_*` linears.
- `test_scope_vision_decoder_targets_vision_and_decoder` — `scope="vision_decoder"`. Assert `lora_` params under both `vision_trunk.blocks.*` AND `transformer_decoder.layers.*.{self_attn,cross_attn}.out_proj`. Assert `neg_control_*` linears have no `lora_` params.
- `test_scope_all_targets_every_linear` — `scope="all"`. Assert every `nn.Linear` in the stub picks up `lora_` params, including the negative controls.
- `test_scope_vision_forward_backward_smoke` — `scope="vision"`, `working=True`. Run one forward (`wrapper(images=torch.randn(1, 3, 8, 8))`), compute a dummy loss (`out["pred_masks"].sum()`), call `.backward()`. Assert at least one `lora_A` parameter has non-`None` `.grad` and the gradient is finite. This is the wiring assertion — proves the scope actually plugged LoRA into the gradient path, not just renamed parameters.
- Repeat the forward+backward smoke for `scope="vision_decoder"` and `scope="all"` via `@pytest.mark.parametrize("scope", ["vision", "vision_decoder", "all"])` on a single `test_scope_forward_backward_finite_grad` function.

This is the test C2 is named for: it covers the per-scope wiring contract on CPU so the GPU tier doesn't have to. Combined with T5/T6 (which cover real-module-name matching on GPU), the `vision` / `all` scope dimension shifts from "covered weakly" to "covered well."

#### C3 — `tests/unit/test_trainer_nan_behavior.py` (NEW)

**Markers:** none.

**Goal:** Pin whatever the current trainer does when a step yields a non-finite loss. The spec does NOT prescribe the behavior — it documents what `src/custom_sam_peft/train/loop.py` does today (skip + increment `nan_streak`; raise `RuntimeError` after `cfg.train.nan_abort_after` consecutive non-finite micro-steps; cf. `loop.py:137`) and asks the implementer to pin that exact behavior. If the implementer discovers the behavior is undefined or silently bad, the test fails and surfaces the gap as a follow-up; do not patch `train/loop.py` from this branch.

**Setup:** Use `make_stub_wrapper(dim=8, working=True)` and the `COCODataset` / `Trainer` wiring from `tests/integration/test_train_resume.py::_ds` / `_cfg` — duplicate the helpers locally inside the test file (no shared-helper extraction this PR; if the duplication becomes painful, factor in a follow-up). Reduce `cfg.train.epochs` to enough to drive `nan_abort_after + 2` micro-steps. The simplest NaN injection: monkeypatch `custom_sam_peft.train.loop.total_loss` to return a dict whose `"total"` value is `torch.tensor(float("nan"), requires_grad=True)`.

**Test cases:**

- `test_nan_loss_below_threshold_does_not_abort` — set `cfg.train.nan_abort_after = 5`; inject NaN for 3 steps then stop the monkeypatch (let subsequent steps return a finite loss). Call `Trainer(...).fit(run_dir=tmp_path)`. Assert no exception, `nan_streak` resets to 0 after the next finite step (inspectable via the tracker's logged scalars if `nan_streak` is logged; otherwise inspect `result.run_dir / "checkpoints"` for at least one checkpoint written post-recovery).
- `test_nan_loss_at_threshold_raises_runtime_error` — set `cfg.train.nan_abort_after = 3`; inject NaN persistently. Call `Trainer(...).fit(...)`. Assert `pytest.raises(RuntimeError, match="non-finite")`.
- `test_nan_loss_logs_warning` — verify (via `caplog`) the warning at `loop.py:126` (`"train_step: class %r raised %s; treating as non-finite."`) fires when the matcher raises `ValueError` on a non-finite cost matrix. This is a separate path from the loss-side NaN; implementer wires the ValueError injection via monkeypatching `custom_sam_peft.train.loop.total_loss` to raise `ValueError("non-finite cost")` instead of returning NaN.

If the current trainer behavior diverges from what's pinned here (e.g., `nan_abort_after` defaults to a value that makes case 2 hard to trigger, or the threshold is per-step vs. per-micro-step), the implementer pins what the code actually does and leaves a follow-up issue.

#### C4 — `tests/unit/test_checkpoint_roundtrip.py` (NEW) + `tests/integration/test_train_resume.py` (CHANGED)

**Amendment (2026-05-21, mid-implementation, tier-2):** the original C4 spec asserted bit-identical optimizer `step` counter and `exp_avg`/`exp_avg_sq` continuity across a reference run vs. an interrupted-then-resumed run. That contract is impossible to satisfy given the trainer's documented epoch-boundary resume semantics (`src/custom_sam_peft/train/checkpoint.py:7`): `load_full_state` sets `start_epoch` to the saved epoch, and `Trainer.fit` does `for epoch in range(start_epoch, cfg.train.epochs)` — so the resumed run *re-walks* the interrupted epoch, taking strictly more optimizer steps than the reference run (verified empirically: tiny_coco 2 images, batch_size=1, save_every=2, epochs=2 → reference run `step=4`, resumed run `step=6`). The existing `test_resume_matches_uninterrupted` already calls this out in its comment ("not bit-identical … because the re-walked epoch retreads some examples"). C4 is therefore split into two pieces that, between them, cover the actual contracts cleanly:

**Unit-level (primary C4 coverage): `tests/unit/test_checkpoint_roundtrip.py` (NEW).**

Pins the save→load roundtrip contract directly — the seam C4 was always trying to assert, without the re-walk noise. Build a small `Sam3Wrapper` LoRA stub (reuse `tiny_sam3_lora_stub.make_stub_wrapper` + `apply_lora`); build a real Adam optimizer + a real `LRScheduler` (reuse `_build_optimizer` / `_build_scheduler` from `custom_sam_peft.train.trainer`); step the optimizer a handful of times against the stub to populate `exp_avg` / `exp_avg_sq` / `step`; snapshot RNG state; call `save_full_state(...)` to a `tmp_path`; build a *fresh* optimizer + scheduler against the same model; call `load_full_state(...)`; then assert:

- `optimizer.state_dict()["param_groups"]` equals the pre-save value on `lr`, `betas`, `weight_decay`, `eps`.
- For every parameter ID in the pre-save `state`: `step` is exactly equal, and `exp_avg` / `exp_avg_sq` are `torch.equal(...)` (bit-identical — this is a save/load roundtrip, not a training-step comparison).
- `scheduler.state_dict()["last_epoch"]` and `["_step_count"]` are exactly equal.
- Post-load `torch.get_rng_state()` is bit-identical to the pre-save snapshot. (CUDA RNG is left to the GPU resume test T1; this unit test runs CPU-only.)
- `ResumeState` returned by `load_full_state` matches the values passed to `save_full_state` (`start_step == global_step`, `start_epoch == epoch`, `nan_streak`, `box_hint_p`).

Expected runtime: <1 second (one stub forward+backward, three optimizer steps, one save, one load, then asserts).

**Integration-level (smoke retention): `tests/integration/test_train_resume.py` (CHANGED).**

Keep the existing `test_resume_matches_uninterrupted` end-to-end smoke (uninterrupted reference run + truncated-then-resumed run, both via `Trainer.fit`). The current finiteness assertion stays. Add only the assertions that **are** consistent with the re-walk semantics — i.e. monotone "the resumed run did at least as much work as the reference":

- Capture both optimizers via a `monkeypatch.setattr` spy on `custom_sam_peft.train.trainer._build_optimizer` (same closure pattern as the original amendment described). Same for `_build_scheduler`.
- Assert `param_groups` equal on `lr`, `betas`, `weight_decay`, `eps` (these are config-driven, so they should be identical across both runs regardless of step count).
- Assert `int(opt_c.state_dict()["state"][pid]["step"]) >= int(opt_a.state_dict()["state"][pid]["step"])` for every shared parameter ID (the resumed run runs strictly ≥ steps because it re-walks the interrupted epoch). At least one strict-greater is acceptable; do not assert strict inequality (the contract is "≥").
- Assert `sched_c.state_dict()["last_epoch"] >= sched_a.state_dict()["last_epoch"]` and `sched_c.state_dict()["_step_count"] >= sched_a.state_dict()["_step_count"]`.
- Do NOT assert `exp_avg` / `exp_avg_sq` allclose — these diverge legitimately under re-walk.
- Do NOT assert RNG-state equality at the end of `run-c` against `run-a` — the unit test above pins the save/load RNG contract; an integration-level RNG assertion would require a per-step hook (out of scope, see original plan note line 1371).

The integration test continues to catch regressions where the resume path fails to *function* end-to-end (e.g., a future bug that drops the optimizer state entirely would surface as `step_c < step_a`). The unit test catches regressions in the serialized contract itself (e.g., a future bug that drops `exp_avg` from the payload).

#### C5 — `tests/integration/test_train_end_to_end.py` (CHANGED)

Extend the existing file in place (no sibling file). The current file exercises a happy-path CPU integration; add three parametrized failure-mode tests, one per bad input:

- `test_malformed_coco_json_raises_clear_error` — write a `tmp_path/annotations.json` that is invalid JSON (e.g. `{`). Build a `TrainConfig` pointing at it. Call `run_training(cfg)`. Assert `pytest.raises(json.JSONDecodeError)` OR whatever exception the loader currently raises — the implementer pins the actual exception class and message.
- `test_missing_image_file_raises_clear_error` — write a valid COCO JSON referencing `images/missing.jpg`, but do not create the file. Call `run_training(cfg)`. Assert the failure surfaces with a message naming `missing.jpg` (the dataset loader should fail on the first batch when the dataloader resolves the file). Pin whatever exception type the code actually raises.
- `test_missing_annotation_entry_raises_clear_error` — write a valid COCO JSON, but for one `image_id` reference no entry in `annotations`. Decide based on the loader's actual behavior: either the dataset returns zero-instance items (acceptable; assert training runs without crashing) or raises an error (assert the message names the missing image). The implementer pins what happens.

Like C3, these tests pin current behavior. If the loader's error messages are bad, the test surfaces that as a separate concern; this PR does not improve them.

---

## 7. Why Each New GPU Test Cannot Be CPU-Substituted

This section is the audit's load-bearing argument: every GPU addition (T1–T6) must defend its place against the project's "minimize GPU surface" policy (`2026-05-19-gpu-test-policy-design.md` §5.6). A reviewer should be able to read this section and either accept each justification or push back specifically.

**T1 — QLoRA resume (NEW).** Bitsandbytes `Linear4bit` modules store quantization state (`absmax`, `quant_state`, `nested_absmax`) that lives on CUDA and is not serializable through the same path as plain LoRA tensors. The `peft` library's save/load roundtrip for QLoRA is well-tested on hot reload, but a *training resume* — where the trainer's `save_full_state` snapshots the adapter alongside optimizer / scheduler / RNG / step counter, then `load_full_state` reconstructs the wrapper from scratch — is a separate seam. The failure mode this test catches: a regression where 4-bit `quant_state` gets dropped or corrupted by the resume path, causing the post-resume forward to silently produce non-finite gradients. The CPU LoRA stub at `tiny_sam3_lora_stub.py` cannot replicate this — `bnb.nn.Linear4bit` is a CUDA-only module by construction.

**T2 / T3 — eval-metric finiteness on the 50-step tests.** The 50-step tests already call `Evaluator.evaluate(...)` after training, which writes `metrics.json` to the run dir. The Evaluator's real-model integration involves running predictions through `custom_sam_peft.eval.postprocess`, RLE-encoding masks, and feeding them to `pycocotools` — every step of which depends on real SAM 3.1 output shapes and the postprocess pipeline's actual CUDA-resident behavior. A CPU stub would either bypass the postprocess (losing the integration signal) or produce synthetic predictions that don't exercise the real shape contracts. C1 covers the metric *schema* on CPU (which doesn't need real predictions); T2 / T3 cover the metric *value* under a real training pass.

**T4 — e2e mAP finite + non-negative.** The current `isinstance(mAP, (int, float))` allows `NaN` (it's a `float`), `-inf`, and negative numbers. The Python `float` type is too permissive to encode "this number is a valid mAP." `pycocotools` can produce negative values when its precision array is all `-1` (sentinel for "no GT") and the mean handling slips through a numerical edge case — a real regression that would land silently today. The reason this can't move to CPU: the test drives the full Typer CLI (`csp run`) end-to-end against real weights, and the contract being asserted is that the CLI's emitted artefact contains a real-valued mAP, not whether `compute_coco_map` itself returns finite numbers (that's C1).

**T5 / T6 — `vision`-scope wiring on real ViT-Det.** The `SCOPE_TARGETS["vision"]` regex is `r"backbone\.vision_backbone\.trunk\.blocks\.\d+\.attn\.(qkv|proj)$"`, which depends on the real SAM 3.1 named-module tree. C2 covers the scope-to-trainable-set logic on a stub whose module names are deliberately renamed (`vision_trunk.blocks.*` vs the real `backbone.vision_backbone.trunk.blocks.*`) — the stub's `FIXTURE_SCOPE_PATTERNS` mirror the production regex against the renamed paths. What T5 / T6 add: the assertion that the real production regex actually matches the real SAM 3.1 names. A regression like "Meta renames `vision_backbone` to `image_encoder` in a future weights release" would slip past C2 (which uses a stub) but T5 / T6 would catch on the next inspection-tier Colab run. The cost is near-zero — `load_sam31` is already paid by the file's other tests.

---

## 8. Test Budget Impact

**Release tier (currently 3 tests, ~25 minutes total on T4).**

- T1 (`test_real_train_qlora_resume.py`) — NEW. Splits the existing 50-step QLoRA budget across a save/load boundary (26 steps + 50 steps continuing from step 25). Net wall-clock is roughly equivalent to one `test_qlora_overfits_in_50_steps` invocation (~14 minutes on T4) plus the one save and one load operation (~few seconds each). Estimate: +14–15 minutes. The full release-tier QLoRA path is intentionally re-paid here because the resume seam can only be exercised end-to-end against real bnb 4-bit weights.
- T2, T3, T4 — extension of existing tests with file-read + numeric checks. ~0 additional time.

Post-change release tier: 4 tests, ~40 minutes total on T4. This stretches but does not exceed the existing Colab session budget (one notebook run completes the full suite in well under a free-tier session). T1 reuses the existing QLoRA YAML so adds no config drift.

**Inspection tier (currently 9 tests, ~3 minutes total on T4).**

- T5, T6 — one new test function each, in files that already pay `load_sam31` once. The added forward-free `apply_lora` / `apply_qlora` call is sub-second.

Post-change inspection tier: 11 tests, ~3.5 minutes total. Negligible.

**CPU suite (currently ~70 unit tests + ~7 integration tests, ~30 seconds in CI).**

- C1 (test_evaluator_schema.py) — 5 small test functions; expected runtime <1 second total (`pycocotools` on 2-image fixtures is fast).
- C2 (test_peft_scope_coverage.py) — 5 small test functions; PEFT apply + one forward+backward on an 8-dim stub; expected runtime <2 seconds.
- C3 (test_trainer_nan_behavior.py) — 3 test functions; each drives ~5–10 trainer micro-steps on the stub; expected runtime ~3–5 seconds total.
- C4 (new `test_checkpoint_roundtrip.py` + extension of `test_train_resume.py`) — unit roundtrip pins save/load bit-equality (<1 second); integration extension adds monotone ≥ assertions with no additional fit() invocations (<1 second delta).
- C5 (extending `test_train_end_to_end.py`) — 3 small failure-mode tests; each drives `run_training` until the first bad-data exception fires; expected runtime ~3–5 seconds.

Post-change CPU suite: +18 test functions, +~10–13 seconds total CI time. Well within the 80% coverage budget and the existing CI runtime envelope (`ci.yml` runs `uv run pytest` on `ubuntu-latest`; current wall-clock <2 minutes).

The net result aligns with the project's stated policy: **all CPU-testable failure modes are pinned on CPU**; only the four genuinely-real-only failures (T1 QLoRA quant-state, T2/T3/T4 real-model eval values, T5/T6 real-module-name regex matches) consume GPU minutes.

---

## 9. Follow-Up Issues Filed by This Branch

The planner subagent (next pipeline step) files these two issues via `gh issue create --assignee @me --label <...>`. Each issue body sketch below is the source of truth; the planner copies the prose verbatim into the issue body.

### 9.1 Benchmark eval-metric thresholds (defer mAP / IoU floors)

**Title:** `Benchmark eval-metric thresholds: set non-trivial mAP / IoU floors once a calibration run is available`

**Labels:** `question`, `priority:low`

**Body sketch:**

> T2, T3, T4 (added by #65's PR — see `docs/superpowers/specs/2026-05-21-gpu-coverage-assessment-design.md` §6.1) assert only that `overall.mAP` is finite and `>= 0.0`. The original #65 question asked whether the 50-step tests should assert specific mAP or IoU floors. We deferred because:
>
> 1. No contributor has run the smoke configs to convergence on tiny_coco to establish what a "healthy" mAP actually looks like at 50 steps.
> 2. tiny_coco (2 images) is small enough that mAP values will be noisy; a tight floor would flake, a loose floor adds no signal.
>
> What we need to make this concrete: one calibration run per smoke YAML (LoRA + QLoRA), capturing `overall.mAP`, `overall.mAP_50`, `overall.mAP_75`, plus the per-class APs, plus the across-run variance. Once we have N≥5 runs of each, we can set a floor at `mean - 2*stddev` or similar.
>
> Cross-ref: #65, `2026-05-21-gpu-coverage-assessment-design.md` §4 (non-goals) and §6.1 (T2/T3/T4 specs).

### 9.2 GPU `all`-scope smoke once VRAM budget allows

**Title:** `GPU "all"-scope smoke once VRAM budget allows`

**Labels:** `enhancement`, `priority:low`

**Body sketch:**

> The `all` PEFT scope (regex `.*`) attaches LoRA to every `nn.Linear` in the SAM 3.1 tree. Its VRAM footprint almost certainly exceeds T4's `gpu_smoke_lora.yaml` (14 GB) and `gpu_smoke_qlora.yaml` (10 GB) ceilings — both pinned to T4 per `2026-05-19-gpu-test-policy-design.md` §5.4. C2 in #65's PR (`tests/unit/test_peft_scope_coverage.py`) covers the `all`-scope wiring contract on a CPU stub, which is sufficient for the typical regression mode (a scope mis-mapping).
>
> What we'd want a GPU smoke for: catching a real-model regression where `all`-scope memory blows up *beyond what the stub can predict* (e.g. a future SAM weights release with substantially more linear layers). The blocker: until #68 (VRAM-floor tightening) reclassifies the VRAM tiers — possibly establishing a "release-tier-only" T4-32 GB / L4 / A10 ceiling for memory-heavy smokes — we have no place to put it that doesn't break the T4 baseline.
>
> Defer until #68 lands. Cross-ref: #65, #68, `2026-05-21-gpu-coverage-assessment-design.md` §4 (non-goals).

---

## 10. Closing #65

The orchestrator posts the following comment on issue #65 when this PR merges, then closes the issue:

> Assessment complete; gaps closed where uniquely owned.
>
> Full audit: `docs/superpowers/specs/2026-05-21-gpu-coverage-assessment-design.md`. PR: `#<pr-number>`.
>
> **Closed in this PR:**
> - `vision` / `all` PEFT scope coverage (T5/T6 GPU inspection + C2 CPU).
> - Mid-training NaN-abort behavior pinned (C3).
> - Eval-metric finiteness asserted on 50-step + e2e tests (T2/T3/T4).
> - GPU resume — QLoRA quant-state continuity (T1); CPU resume continuity for optimizer / scheduler / RNG (C4).
> - Malformed-dataset error contracts pinned (C5).
> - Evaluator schema unit coverage (C1).
>
> **Deferred via follow-up issues:**
> - eval-metric *threshold* floors → see follow-up issue `#<follow-up-1>`.
> - GPU `all`-scope smoke → see follow-up issue `#<follow-up-2>`, blocked on #68.
>
> **Owned elsewhere (no work here):** model variants #23, headless GPU CI #64, VRAM-floor tightening #68, `csp predict` smoke #74, video #78, older CUDA #79.
>
> This closes the `GPU test gate decision shipped — #65` line on the #70 pre-v1.0 checklist.

---

## 11. Exit Criteria

**Doc.**

- [ ] `docs/superpowers/specs/2026-05-21-gpu-coverage-assessment-design.md` lands at the path above with all 11 sections present.

**GPU additions.**

- [ ] `tests/gpu/test_real_train_qlora_resume.py` exists; module-level markers match the existing QLoRA file; per-test `skipif(not _bnb_available())`; reuses `gpu_smoke_qlora.yaml`; asserts finiteness + finite final adapter + peak VRAM ≤ 10 GB; does NOT compare against an uninterrupted reference.
- [ ] `tests/gpu/test_real_train_overfits.py::test_overfits_in_50_steps` reads `metrics.json` and asserts `overall.mAP` is finite and `>= 0.0`.
- [ ] `tests/gpu/test_real_train_qlora.py::test_qlora_overfits_in_50_steps` reads `metrics.json` and asserts `overall.mAP` is finite and `>= 0.0`. `test_qlora_smoke_fast` is unchanged.
- [ ] `tests/gpu/test_run_end_to_end_gpu.py::test_run_end_to_end_writes_bundle` tightens its mAP check to `math.isfinite(mAP) and mAP >= 0.0`.
- [ ] `tests/integration/test_peft_lora_real.py` gains `test_apply_lora_vision_scope_targets_only_vision_backbone` asserting `vision_backbone` present + `transformer.decoder` and `mask_decoder` absent at `scope="vision"`.
- [ ] `tests/integration/test_peft_qlora_real.py` gains the mirror test for QLoRA, guarded by `skipif(not _bnb_available())`.

**CPU additions.**

- [ ] `tests/unit/test_evaluator_schema.py` exists with the five test cases listed in §6.2.
- [ ] `tests/unit/test_peft_scope_coverage.py` exists with per-scope target assertions + per-scope forward+backward smokes.
- [ ] `tests/unit/test_trainer_nan_behavior.py` exists with the three test cases listed in §6.2; pins whatever the trainer currently does without modifying `src/custom_sam_peft/train/loop.py`.
- [ ] `tests/unit/test_checkpoint_roundtrip.py` exists; pins `save_full_state` → `load_full_state` bit-identical roundtrip on optimizer state (`step`, `exp_avg`, `exp_avg_sq`, `param_groups`), scheduler state (`last_epoch`, `_step_count`), CPU RNG, and `ResumeState` fields. See §6.2 C4 for the full assertion list.
- [ ] `tests/integration/test_train_resume.py::test_resume_matches_uninterrupted` gains monotone optimizer / scheduler continuity assertions (`step_c >= step_a`, `last_epoch_c >= last_epoch_a`, `_step_count_c >= _step_count_a`) consistent with the trainer's epoch-boundary re-walk semantics. No `exp_avg` allclose; no end-of-run RNG equality (those live in the unit test above).
- [ ] `tests/integration/test_train_end_to_end.py` gains malformed-JSON / missing-image / missing-annotation failure-mode tests.

**Tests + lint (CPU CI gates).**

- [ ] `uv run ruff check` and `uv run ruff format --check` clean.
- [ ] `uv run mypy src/custom_sam_peft` clean.
- [ ] `uv run pytest` green on the full CPU suite. Net new tests pass; existing tests pass.
- [ ] `pytest --collect-only -m gpu` collects 4 tests (the existing 3 plus T1).
- [ ] `pytest --collect-only -m gpu_inspection` collects 11 tests (the existing 9 plus T5, T6).
- [ ] `markdownlint` clean on the spec file (`docs/superpowers/` directory relaxation applies per `2026-05-18-ci-hardening-design.md`).

**Follow-up issues.**

- [ ] Issue "Benchmark eval-metric thresholds" filed via `gh issue create --assignee @me --label question --label priority:low`, body matches §9.1.
- [ ] Issue "GPU all-scope smoke once VRAM budget allows" filed via `gh issue create --assignee @me --label enhancement --label priority:low`, body matches §9.2.

**Close-out (orchestrator only).**

- [ ] Real-GPU run of T1 + T2 + T3 + T4 + T5 + T6 on the Colab notebook is green (manual, post-merge).
- [ ] Comment from §10 posted on #65; #65 closed.
- [ ] #70 checkbox `GPU test gate decision shipped — #65` ticked.

**Explicitly NOT a gate.**

- mAP / IoU threshold tuning (deferred to §9.1's follow-up).
- New `all`-scope GPU test (deferred to §9.2's follow-up, blocked on #68).
- Any `csp predict` test (owner: #74).
- Any change under `src/custom_sam_peft/`. If C3 surfaces a real NaN-behavior bug, that becomes a separate issue.
- Any new dependency, new YAML, or workflow edit.

---

## 12. Deferred (Out of Scope, Tracked Elsewhere)

- **Model variants.** `facebook/sam3` alongside `facebook/sam3.1`. Owner: #23.
- **Headless GPU CI.** Self-hosted runner / cloud GPU CI. Owner: #64.
- **VRAM-floor tightening.** Lowering the 14 GB / 10 GB ceilings; reclassifying VRAM tiers. Owner: #68.
- **`csp predict` GPU smoke.** Predict command does not exist yet. Owner: #74.
- **Video tracking.** Out until activation checkpointing is unblocked. Owner: #78.
- **Older CUDA / driver matrix.** Owner: #79.
- **mAP / IoU threshold floors.** Filed by this branch — see §9.1.
- **`all`-scope GPU smoke.** Filed by this branch — see §9.2.
