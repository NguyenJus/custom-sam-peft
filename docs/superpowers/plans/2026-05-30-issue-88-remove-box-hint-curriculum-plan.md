# Remove the `box_hint` Curriculum (gated on a literature review) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Spec:** [`docs/superpowers/specs/2026-05-30-issue-88-remove-box-hint-curriculum-design.md`](../specs/2026-05-30-issue-88-remove-box-hint-curriculum-design.md)
**Issue:** [#88](https://github.com/NguyenJus/custom-sam-peft/issues/88) — originally "tune the `box_hint` schedule"; rescoped to "decide whether to keep the curriculum at all, then remove it."

**Goal:** Produce a citation-backed deep-research note that recommends keep-or-remove for the `box_hint` localization-hint curriculum, and — on the user's "remove" confirmation — fully delete the curriculum (schema, train loop, checkpoint, trainer, model plumbing, data field, 9 templates, docs) while retaining the `SupportPrompts` extension seam and tolerating pre-removal checkpoints on resume.

**Architecture:** Two phases separated by a **hard user-decision gate**. Phase 1 is a single research deliverable that halts for user confirmation. Phase 2 is one interdependent removal carried end-to-end in a single session (deleting the schema field breaks the loop, which breaks the model, etc., so intermediate states will not import/run in isolation) — grouped into six reviewable task blocks ending in a green gated test suite and a PR.

**Tech Stack:** Python 3.12, Pydantic v2 (`config/schema.py` strict models), frozen dataclasses (`data/base.py` seam), PyTorch + SAM 3.1 (`models/sam3.py`), pytest + pytest-cov (TDD, `--cov-fail-under=80` gate), ruff + mypy + markdownlint-cli2 (CI gates), `gh` CLI (PR + cross-issue comments).

---

## Phase structure & the user-decision gate (read first)

```text
Phase 1  ──►  COMMIT lit-review note  ──►  [USER-DECISION GATE]  ──►  Phase 2
(deep research)   (interface OUT)          (user reads + confirms)   (removal)
```

| Phase | Deliverable | Boundary contract |
|---|---|---|
| 1 | `docs/research/2026-05-30-issue-88-box-hint-curriculum-lit-review.md` — TL;DR + four cited answers + explicit keep/remove recommendation | **OUT:** committed note carrying an explicit recommendation. Session **HALTS**. |
| 2 | `box_hint` removed; `SupportPrompts` seam retained; docs + tests updated; PR closing #88 | **IN:** the user-confirmed "remove" decision from the gate. |

**The gate is non-negotiable.** The session executing Phase 1 commits the note and **halts**. The user reads it and either:

- confirms **"remove"** → Phase 2 runs exactly as written below (primary path); or
- pivots to **"keep + cite"** (spec §5.4 surprise branch — literature shows a strong endpoint/simplicity benefit) → **this plan does NOT cover the keep-path.** That is a planner-amendment trigger: stop, dispatch a planner to re-plan Phase 2 as "keep `box_hint` + cite the chosen defaults," and do not execute the removal tasks below.

**Guiding-principle lens (spec §4) — the single evaluative frame for Phase 1's recommendation and the gate decision:**

> Priority order: (1) endpoint accuracy, (2) user-facing simplicity, (3) — far behind — training speed. `box_hint` decays to `p_end=0.0` over the first ~75% of every run (the final ~25% is already pure text-only) and is never used at inference, so it **cannot move the final optimum** — only the optimization path. A convergence-speed benefit alone does **not** justify the config surface a knob adds. Keep only if the literature shows it improves the text-only **endpoint** or reduces **user-facing complexity**; otherwise remove.

---

## File structure

### Phase 1 — create

- `docs/research/2026-05-30-issue-88-box-hint-curriculum-lit-review.md` — the deep-research note. Header mirrors existing notes (e.g. `docs/research/2026-05-24-issue-137-qlora-8gb-feasibility.md`): title, issue link, date, spec link, plan link; opens with a **TL;DR** stating the recommendation up front.

### Phase 2 — modify (source)

- `src/custom_sam_peft/config/schema.py` — delete `BoxHintSchedule` class + its `_check_monotone` validator; delete `TrainHyperparams.box_hint` field; drop `"BoxHintSchedule"` from `__all__` if present.
- `src/custom_sam_peft/data/base.py` — drop `SupportPrompts.boxes` field; rewrite the `SupportPrompts` docstring to the field-less reserved-seam form.
- `src/custom_sam_peft/models/sam3.py` — delete `_build_geometric_prompt`; drop the now-orphaned `from sam3.model.box_ops import box_xyxy_to_cxcywh` import; strip `box_hints` plumbing from `Sam3Wrapper.forward`, `_validate_inputs`, and `_Sam3ImageAdapter.forward`; keep `support` as a no-op and keep the `SupportPrompts` import and the `Prompt` import (still used by the dummy fallback).
- `src/custom_sam_peft/train/loop.py` — delete `_box_hint_p`, the Bernoulli gate + `hints_g` + `n_hint_applied`, `import random` + its `# noqa: S311`, `StepResult.p_t`/`n_hint_applied`, `_ScalarWindow.last_p_t` + `box_hint/*` logging; drop `box_hints=`/`support=` from the two model-forward call sites (text-only); change the `on_checkpoint` callback signature to drop `p_t`; keep `targets_g`; drop the `BoxHintSchedule` import.
- `src/custom_sam_peft/train/checkpoint.py` — drop `ResumeState.box_hint_p`; drop the `box_hint_p` param + payload key in `save_full_state`; stop reading `box_hint_p` in `load_full_state` (back-compat: tolerate old keys present AND new keys absent); update the module docstring.
- `src/custom_sam_peft/train/trainer.py` — drop `decay_steps` from `resolve_schedule_steps` (keep `save_every`/`eval_every`) + its call site + the log line; drop `resolved_box_hint` + the `box_hint` `model_copy` entry; drop `box_hint_p=` from `_maybe_checkpoint`/`_save_checkpoint` and the `on_checkpoint` closure's `p_t`; drop the no-resume `ResumeState(box_hint_p=...)`; drop `box_hint_p_final` from both `metrics.json` writes; drop the `_box_hint_p` import.

### Phase 2 — modify (templates, 9 files)

- `configs/examples/coco_text_lora.yaml`, `coco_text_qlora.yaml`, `coco_text_lora_subset.yaml`, `coco_text_auto_split.yaml`, `coco_text_no_val.yaml`, `gpu_smoke_lora.yaml`, `gpu_smoke_qlora.yaml`, `min_gpu_qlora.yaml` — remove the `box_hint:` block.
- `src/custom_sam_peft/cli/templates/config_full.yaml` — remove the `box_hint:` block (lines 66–72, including its comment lines).

### Phase 2 — modify (docs)

- `docs/ARCHITECTURE.md` — rewrite the Prompt invariant (line 6); keep the `base.py` symbol line (line 17).
- `docs/config-schema.md` — delete the three `train.box_hint.*` rows (lines 126–128).
- `CHANGELOG.md` — add a `[Unreleased]` entry.

### Phase 2 — delete (tests)

- `tests/unit/test_box_hint_schedule.py`
- `tests/unit/test_geometric_prompt_builder.py`

### Phase 2 — update (tests)

- Spec §6.3 list: `tests/unit/test_schedule_resolution.py`, `tests/unit/test_train_step.py`, `tests/unit/test_sam3_adapter.py`, `tests/unit/test_train_loop_legacy_k1.py`, `tests/unit/test_train_checkpoint.py`, `tests/unit/test_checkpoint_roundtrip.py`, `tests/unit/test_config_schema.py`, `tests/unit/test_trainer_run_dir.py`.
- **Planner-found, missing from spec §6.3 (see "Spec gaps" below):** `tests/fixtures/tiny_sam3_lora_stub.py`, `tests/fixtures/tiny_sam3_stub.py`, `tests/unit/test_static_guards.py`.

---

## Spec gaps & corrections found during planning (read before Phase 2)

These were verified against `main` at planning time. Symbols are authoritative; line numbers may drift.

1. **Orphaned import in `models/sam3.py`.** Deleting `_build_geometric_prompt` removes the **only** use of `from sam3.model.box_ops import box_xyxy_to_cxcywh` (the call is at the line `cxcywh = box_xyxy_to_cxcywh(norm_xyxy)` inside the builder). Spec §6.2 does not mention this import. **The plan drops it** (Task 3) — ruff would otherwise flag F401. The `Prompt` import stays (still used by the zero-length dummy fallback). The `SupportPrompts` import stays (still used in signatures).

2. **`_validate_inputs` box validation lives lower than the spec's line range.** Spec §6.2 cites `_validate_inputs` at "≈ lines 236–241" (the signature). The actual `support.boxes` length/shape validation is a separate block (the `boxes = support.boxes if support is not None else None` block, ≈ lines 276–292). The spec's prose ("remove any `support`/`box_hints` validation logic") covers it; the plan removes that whole block (Task 3).

3. **Three test/fixture files the spec §6.3 list omits.** A repo-wide grep found:
   - `tests/fixtures/tiny_sam3_lora_stub.py` — `_StubAdapter.forward(..., box_hints=None)`. After removal `Sam3Wrapper.forward` calls the inner model text-only, so this kwarg is dead. Drop it for a clean cut (Task 3). Harmless if left (defaulted), so not a green-blocker.
   - `tests/fixtures/tiny_sam3_stub.py` — a comment on the `del kwargs` line mentions `box_hints=`. Cosmetic comment update (Task 3).
   - `tests/unit/test_static_guards.py` — `test_no_to_device_outside_collator_and_runtime` allows `/models/sam3.py` for `_build_geometric_prompt`'s `h.to(device=...)` (the **only** `.to(device` in sam3.py). After the builder is deleted that allowance is dead. The guard still **passes** (the allowlist is permissive, never required), but its docstring references a deleted symbol. Tighten line 59's `allowed_substrings` to drop `/models/sam3.py` and update the docstring (Task 3). Verify no other `.to(device` remains in sam3.py before tightening.

4. **`metrics.json` schema doc artifact.** The eval spec documents `box_hint_p_final` in the `metrics.json` schema. Spec §6.2 notes it is a doc artifact dropped here. No separate eval-spec edit is in scope (specs under `docs/superpowers/specs/` are dated records; only `ARCHITECTURE.md`, `config-schema.md`, `CHANGELOG.md` are touched).

5. **No other production `box_hint` references.** Outside the spec's named surface + the three items above, a repo-wide grep across `src/`, `tests/`, `configs/`, and the three docs found no further `box_hint` / `BoxHintSchedule` / `_box_hint_p` / `_build_geometric_prompt` / `box_hint_p_final` references (historical `docs/superpowers/` records excluded — left untouched per spec §2 non-goals).

---

## Markdown lint gate (applies to every `.md` commit in both phases)

Before committing any tracked `.md` (the research note, `ARCHITECTURE.md`, `config-schema.md`, `CHANGELOG.md`, and this plan + the spec), run CI's exact linter and fix findings. This dev box has no system node, so use the Python-bundled Node path (MEMORY: markdown-lint gate):

```bash
cp .config/markdownlint-cli2.jsonc /tmp/x.markdownlint-cli2.jsonc
uv run --no-project --with nodejs-bin python -c "
from nodejs import node, npx
import os, sys
os.environ['PATH'] = os.path.dirname(node.path) + os.pathsep + os.environ['PATH']
sys.exit(npx.run(['--yes','markdownlint-cli2@0.14.0','--config','/tmp/x.markdownlint-cli2.jsonc', *sys.argv[1:]]).returncode)
" docs/research/2026-05-30-issue-88-box-hint-curriculum-lit-review.md
```

Config disables only MD013/MD018/MD029; all other default rules are active. Expected: clean exit (0).

---

# PHASE 1 — Deep-research literature-review note

**This is one coherent unit that ends at the gate.** It produces the note, lints it, commits it, and **halts**.

## Block — research note

### Task 1: Produce the deep-research literature-review note

**Files:**

- Create: `docs/research/2026-05-30-issue-88-box-hint-curriculum-lit-review.md`

- [ ] **Step 1: Run the deep-research capability on the spec §5.2 questions through the §4 lens**

Use the project's `deep-research` skill. The technique under review is precisely: *feeding decaying ground-truth box hints alongside text/class prompts, weaned to `p=0` so the model is trained toward text-only inference.* The run must be adversarially verified and broadly cited (cross-check claims that bear on the recommendation; do not rely on a single source). The note MUST answer, with citations:

1. **Does the technique appear in published research, and under what name?** Survey at least:
   - Curriculum learning — Bengio et al., 2009.
   - Scheduled sampling — Bengio et al., 2015 — and Huszár's (2015) statistical-inconsistency critique of it.
   - Learning using privileged information (LUPI) — Vapnik & Vashist.
   - Denoising training in DETR-family detectors — DN-DETR (Li et al., 2022) and DINO (Zhang et al., 2022) as the closest analog. State explicitly that their box-denoising is a **parallel branch kept throughout training**, **not annealed to zero** — the opposite of our decayed-to-zero curriculum.
   - Box-as-prompt in promptable / open-vocab models — SAM / SAM-family, GLIP, Grounding-DINO. State explicitly that these use boxes as **inference-time prompts**, not as a decayed **training** curriculum.
2. **Endpoint vs. path.** Does the prior art show such a *removed-by-end* hint changes the **endpoint** (final solution quality) or only the **optimization path** (convergence speed)?
3. **Evidence of benefit.** Is there evidence the technique actually *accelerates convergence* or *improves the text-only endpoint*?
4. **Explicit recommendation (keep / remove)**, reasoned against the guiding principle (§4): endpoint accuracy and user-facing simplicity above speed. A speed-only finding ⇒ recommend remove.

- [ ] **Step 2: Write the note with the house header + TL;DR-first structure**

Mirror `docs/research/2026-05-24-issue-137-qlora-8gb-feasibility.md`: a `#` title, a blockquote header carrying the issue link, the date (`2026-05-30`), a link to the spec, and a link to this plan. Open with a `## TL;DR` that states the keep/remove recommendation up front, then a section per §5.2 question (1–4) with inline citations, ending in the explicit recommendation.

- [ ] **Step 3: Markdown-lint the note**

Run the markdown-lint gate command above on the note. Expected: clean exit (0). Fix any findings.

- [ ] **Step 4: Commit**

```bash
git add docs/superpowers/specs/2026-05-30-issue-88-remove-box-hint-curriculum-design.md \
        docs/superpowers/plans/2026-05-30-issue-88-remove-box-hint-curriculum-plan.md \
        docs/research/2026-05-30-issue-88-box-hint-curriculum-lit-review.md
git commit -m "docs(research): box_hint curriculum literature review (#88)"
```

(The spec + plan are added here only if they are not already committed on this branch; if they are, just add the note.)

- [ ] **Step 5: HALT at the user-decision gate**

**Phase boundary — interface contract OUT:** the committed note carrying an explicit recommendation.

**Stop. Do not begin Phase 2.** Surface the recommendation to the user and wait for confirmation:

- **"remove"** → proceed to Phase 2 (this plan).
- **"keep + cite"** → planner-amendment trigger (spec §5.4). Do NOT run the Phase 2 tasks below; dispatch a planner to re-plan Phase 2 as the keep-path.

---

# PHASE 2 — Remove the curriculum, keep the seam

**Phase boundary — interface contract IN:** the user-confirmed **"remove"** decision from the gate. If the user pivoted to "keep," STOP — this phase does not apply.

**Execute Phase 2 in a single session, end-to-end.** The removal is interdependent: deleting the schema field breaks the loop, which breaks the model and trainer. Intermediate per-task states will not import/run in isolation, so do **not** split this phase across sessions and do **not** expect a green suite between every task — run the **full gated suite once at Task 11**. During iteration, run a focused subset with the coverage gate bypassed:

```bash
uv run pytest -o "addopts=" tests/unit/test_config_schema.py -q   # example: gate bypassed for fast iteration
```

(`-o "addopts="` clears the global `--cov-fail-under=80` in `addopts`; `--no-cov` does NOT work here — MEMORY: pytest subset coverage gate.)

**What is KEPT (do not remove):**

- `SupportPrompts` dataclass (field-less, documented reserved seam — #126 §12).
- `Sam3Wrapper.forward(..., support: SupportPrompts | None = None)` signature — `support` becomes a no-op.
- `_validate_inputs(..., support)` parameter — no-op.
- The `SupportPrompts` import in `models/sam3.py` and the `Prompt` import.
- `targets_g` construction in `train/loop.py` (losses need per-row targets).
- All `save_every` / `eval_every` resolution in `resolve_schedule_steps`.

**Parallelism within Phase 2:** task blocks (a) schema+data, (b) model, (c) train loop, (d) checkpoint+trainer are **code-interdependent and must be serialized in source order** (each depends on symbols the previous removed/changed). Block (e) templates+docs is **file-disjoint from all code blocks** and may be done in parallel with (a)–(d) by a separate agent on the same branch (no shared files). Block (f) verify+reconcile must run **last**, after every other block lands. If dispatching parallel agents for (e) vs (a)–(d), serialize the git commits to avoid the orphaned-commit race (MEMORY: parallel agent commit race).

---

## Block (a) — schema + data

### Task 2: Remove `BoxHintSchedule` and `TrainHyperparams.box_hint` from the schema

**Files:**

- Modify: `src/custom_sam_peft/config/schema.py` — delete `class BoxHintSchedule(_Strict)` (incl. `_check_monotone`); delete the `box_hint: BoxHintSchedule = Field(default_factory=BoxHintSchedule)` line in `TrainHyperparams`; remove `"BoxHintSchedule"` from `__all__` if present.
- Test: `tests/unit/test_config_schema.py` — drop `"BoxHintSchedule"` from the exported-names assertion; add a test that a config carrying `train.box_hint:` is now rejected by the strict model.

- [ ] **Step 1: Update the failing test first**

In `tests/unit/test_config_schema.py`, remove `"BoxHintSchedule"` from the expected-`__all__` list (it appears in the assertion alongside other schema symbols). Add:

```python
def test_box_hint_field_rejected_by_schema() -> None:
    """After #88 removal, train.box_hint is no longer a valid field."""
    import pytest
    from pydantic import ValidationError

    from custom_sam_peft.config.schema import TrainHyperparams

    with pytest.raises(ValidationError):
        TrainHyperparams(epochs=1, box_hint={"p_start": 1.0, "p_end": 0.0})
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest -o "addopts=" tests/unit/test_config_schema.py::test_box_hint_field_rejected_by_schema -q`
Expected: FAIL — `box_hint` is currently accepted, so no `ValidationError` is raised.

- [ ] **Step 3: Delete the schema symbols**

In `src/custom_sam_peft/config/schema.py`: delete the entire `class BoxHintSchedule(_Strict)` block (docstring, `p_start`/`p_end`/`decay_steps` fields, and the `_check_monotone` validator); delete the `box_hint: BoxHintSchedule = Field(default_factory=BoxHintSchedule)` line in `TrainHyperparams`; if `"BoxHintSchedule"` appears in `__all__`, remove it.

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest -o "addopts=" tests/unit/test_config_schema.py -q`
Expected: PASS (both the new test and the updated `__all__` assertion).

- [ ] **Step 5: Commit**

```bash
git add src/custom_sam_peft/config/schema.py tests/unit/test_config_schema.py
git commit -m "refactor(schema): remove BoxHintSchedule + TrainHyperparams.box_hint (#88)"
```

### Task 3: Reduce `SupportPrompts` to a field-less documented seam

**Files:**

- Modify: `src/custom_sam_peft/data/base.py` — drop the `boxes: list[torch.Tensor | None] | None = None` field; rewrite the `SupportPrompts` docstring to the empty-seam form (remove the `boxes` length-convention prose).

- [ ] **Step 1: Rewrite the dataclass**

Replace the `SupportPrompts` dataclass body so it carries **no fields** and documents the reserved seam:

```python
@dataclass(frozen=True)
class SupportPrompts:
    """Reserved extension seam for auxiliary localization prompts.

    Rides alongside ``TextPrompts``; never replaces text; never used at
    inference. Reserved for future hints (masks, positive points, negative
    points) per #126 §12. Currently carries **no fields** — the ``box_hint``
    curriculum that previously populated ``boxes`` was removed in #88.
    """
```

(A frozen dataclass with no fields is valid; `SupportPrompts()` constructs.)

- [ ] **Step 2: Run the data-base tests to verify they still pass**

Run: `uv run pytest -o "addopts=" tests/unit/test_data_base.py -q`
Expected: PASS — no test should assert `SupportPrompts` has a `boxes` field (the wrapper/loop tests that exercised `boxes` are updated in Tasks 4–6). If any `test_data_base.py` test references `SupportPrompts(boxes=...)`, update it to `SupportPrompts()` in this step.

- [ ] **Step 3: Commit**

```bash
git add src/custom_sam_peft/data/base.py tests/unit/test_data_base.py
git commit -m "refactor(data): SupportPrompts becomes a field-less reserved seam (#88)"
```

---

## Block (b) — model

### Task 4: Strip `box_hint` plumbing from `models/sam3.py`, keep `support` as a no-op

**Files:**

- Modify: `src/custom_sam_peft/models/sam3.py` — delete `_build_geometric_prompt`; drop the orphaned `box_xyxy_to_cxcywh` import; strip `box_hints` from `Sam3Wrapper.forward`, `_validate_inputs`, and `_Sam3ImageAdapter.forward`; keep `support` (no-op), the `SupportPrompts` import, and the `Prompt` import.
- Test: `tests/unit/test_sam3_adapter.py` — drop the `box_hints=None` kwarg from the two `adapter(...)` call sites.
- Delete: `tests/unit/test_geometric_prompt_builder.py` (the builder is gone).
- Modify: `tests/fixtures/tiny_sam3_lora_stub.py` — drop `box_hints` from `_StubAdapter.forward`.
- Modify: `tests/fixtures/tiny_sam3_stub.py` — update the `del kwargs` comment.
- Modify: `tests/unit/test_static_guards.py` — tighten the `.to(device` allowlist + docstring.

- [ ] **Step 1: Delete the geometric-prompt builder test, then update the adapter test (failing state)**

Delete `tests/unit/test_geometric_prompt_builder.py` entirely.

In `tests/unit/test_sam3_adapter.py`, change the two adapter calls from `adapter(images, prompts, box_hints=None)` / `adapter(torch.zeros(b, 3, 8, 8), prompts, box_hints=None)` to drop the `box_hints=None` kwarg (the adapter `forward` will no longer accept it).

- [ ] **Step 2: Run the adapter test to verify it fails**

Run: `uv run pytest -o "addopts=" tests/unit/test_sam3_adapter.py -q`
Expected: FAIL — `_Sam3ImageAdapter.forward` still declares `box_hints=...`, so the edited test passes against the OLD code; run it AFTER editing the test to confirm the test still imports, then it will fail once Step 3 removes the kwarg. (If it passes here, that's fine — the binding assertion is Step 4 after the source edit.)

- [ ] **Step 3: Edit `models/sam3.py`**

Apply all of these:

- Delete the entire `def _build_geometric_prompt(...)` function (docstring through its `return Prompt(...)`).
- Delete the now-orphaned import line `from sam3.model.box_ops import box_xyxy_to_cxcywh` (only used inside the deleted builder).
- In `Sam3Wrapper.forward`, delete `box_hints = support.boxes if support is not None else None` and call the inner model text-only:

  ```python
  def forward(
      self,
      images: Tensor,
      prompts: list[Prompts],
      support: SupportPrompts | None = None,
  ) -> dict[str, Any]:
      self._validate_inputs(images, prompts, support)
      out: dict[str, Any] = self.model(images, prompts)
      return out
  ```

- In `_validate_inputs`, keep the `support` parameter but delete the `boxes = support.boxes if support is not None else None` block and its length/shape validation (the block iterating `for i, h in enumerate(boxes)`). The image/prompt validation stays.
- In `_Sam3ImageAdapter.forward`, delete the `box_hints: list[Tensor | None] | None = None` parameter, delete the `gp = _build_geometric_prompt(...)` call, and make the zero-length dummy the **only** path (the existing `if gp is None:` fallback `Prompt(...)` becomes unconditional):

  ```python
  gp = Prompt(
      box_embeddings=torch.zeros(0, n_cols, 4, device=device, dtype=model_dtype),
      box_mask=torch.zeros(n_cols, 0, device=device, dtype=torch.bool),
      point_embeddings=torch.zeros(0, n_cols, 2, device=device, dtype=model_dtype),
      point_mask=torch.zeros(n_cols, 0, device=device, dtype=torch.bool),
  )
  ```

- Update the `_Sam3ImageAdapter` class docstring: drop the `(images, prompts, box_hints)` convention line and the `_build_geometric_prompt` routing paragraph; state the backbone is called text-only.
- Update the `Sam3Wrapper` class docstring's `support`/`boxes` paragraph to say `support` is a reserved no-op (currently ignored; text-only forward).

- [ ] **Step 4: Run the adapter + wrapper tests to verify they pass**

Run: `uv run pytest -o "addopts=" tests/unit/test_sam3_adapter.py -q`
Expected: PASS — adapter forward runs text-only with the zero-length dummy.

- [ ] **Step 5: Update fixtures + the static-guard test**

- `tests/fixtures/tiny_sam3_lora_stub.py`: change `def forward(self, images: Any = None, prompts: Any = None, box_hints: Any = None) -> Any:` to drop `box_hints` → `def forward(self, images: Any = None, prompts: Any = None) -> Any:`.
- `tests/fixtures/tiny_sam3_stub.py`: change the `del kwargs  # box_hints= ... and support= ... are both ignored` comment to `del kwargs  # support= (outer-model path) is ignored`.
- `tests/unit/test_static_guards.py`, in `test_no_to_device_outside_collator_and_runtime`: drop `"/models/sam3.py"` from `allowed_substrings` (there is no longer any `.to(device` in `sam3.py` after the builder is deleted — verify with `rg -n '\.to\(device' src/custom_sam_peft/models/sam3.py` → expect 0 hits), and remove the `/models/sam3.py (_build_geometric_prompt): ...` paragraph from the docstring.

- [ ] **Step 6: Run the static-guard + fixture-dependent tests**

Run: `uv run pytest -o "addopts=" tests/unit/test_static_guards.py tests/unit/test_sam3_adapter.py -q`
Expected: PASS — the `.to(device` guard finds zero offenders and zero in sam3.py.

- [ ] **Step 7: Commit**

```bash
git add src/custom_sam_peft/models/sam3.py tests/unit/test_sam3_adapter.py \
        tests/fixtures/tiny_sam3_lora_stub.py tests/fixtures/tiny_sam3_stub.py \
        tests/unit/test_static_guards.py
git rm tests/unit/test_geometric_prompt_builder.py
git commit -m "refactor(model): remove box_hint geometric-prompt plumbing; support is a no-op (#88)"
```

---

## Block (c) — train loop

### Task 5: Remove the `box_hint` sampler, gate, logging, and `StepResult` fields from `train/loop.py`

**Files:**

- Modify: `src/custom_sam_peft/train/loop.py`.
- Test: `tests/unit/test_train_step.py`, `tests/unit/test_train_loop_legacy_k1.py`.
- Delete: `tests/unit/test_box_hint_schedule.py`.

- [ ] **Step 1: Update the loop tests first (failing state)**

- Delete `tests/unit/test_box_hint_schedule.py` entirely.
- `tests/unit/test_train_step.py`: drop `BoxHintSchedule` from the imports; drop `_box_hint_p` from `from custom_sam_peft.train.loop import _box_hint_p, train_step` (→ `from custom_sam_peft.train.loop import train_step`); delete `test_box_hint_p_endpoints`, `test_box_hint_p_midpoint`, and `test_train_step_box_hint_sampling` (these test the removed sampler and the `cfg.train.box_hint.*` knobs). Keep all other `train_step` tests; if any remaining test reads `result.p_t` or `result.n_hint_applied`, delete those assertions.
- `tests/unit/test_train_loop_legacy_k1.py`: this file's two tests (`...box_hint_applied...` / the draw-order hint-count test) assert on `n_hint_applied` and set `cfg.train.box_hint.*`. Delete those two box-hint-specific tests and any `cfg.train.box_hint.*` setup / `monkeypatch` of `random.random`. Keep any K=1 multiplex assertions that do **not** depend on hints; if the whole file is box-hint-only, delete the file (verify it has no non-hint test before deleting).

- [ ] **Step 2: Run the loop tests to confirm they collect (import-clean)**

Run: `uv run pytest -o "addopts=" tests/unit/test_train_step.py --collect-only -q`
Expected: collection succeeds (no import of `_box_hint_p` / `BoxHintSchedule`). It may still fail at runtime until Step 3.

- [ ] **Step 3: Edit `train/loop.py`**

- Delete `import random` (line ~14) and the `# noqa: S311` on the sampling line.
- Delete `from custom_sam_peft.config.schema import BoxHintSchedule, TrainConfig` → keep only `TrainConfig`.
- Delete the `def _box_hint_p(global_step, cfg)` function.
- In `StepResult`: delete the `p_t: float` and `n_hint_applied: int` fields; in `StepResult.empty`, delete the `p_t` parameter and the `p_t=` / `n_hint_applied=0` kwargs (signature becomes `def empty(cls, nan_streak: int = 0)`).
- In `train_step`: delete `p_t = _box_hint_p(global_step, cfg.train.box_hint)`; change `return StepResult.empty(p_t=p_t, nan_streak=nan_streak)` → `StepResult.empty(nan_streak=nan_streak)`; delete `n_hint_applied = 0`; delete the `if row_targets and random.random() < p_t:` branch and its `box_tensor` / `hints_g.append(...)` / `n_hint_applied += 1` body, the `else: hints_g.append(None)`, and the `hints_g: list[Tensor | None] = []` declaration. **Keep** `targets_g` and its `targets_g.append(row_targets)`.
- In the OOM `_forward_group` closure: drop the `_hints_g` default-arg and the `micro_hints = [...]` slice; change `micro_out = _model(micro_imgs, micro_prompts, support=SupportPrompts(boxes=micro_hints))` → `micro_out = _model(micro_imgs, micro_prompts)`.
- In the non-OOM forward (the `else:` branch): change `out = model(images, prompts_g, support=SupportPrompts(boxes=hints_g))` → `out = model(images, prompts_g)`.
- Drop `SupportPrompts` from `from custom_sam_peft.data.base import Instance, SupportPrompts, TextPrompts` (no longer constructed here) → `from custom_sam_peft.data.base import Instance, TextPrompts`.
- In the final `return StepResult(...)`: delete `p_t=p_t` and `n_hint_applied=n_hint_applied`.
- In `_ScalarWindow`: delete `"box_hint/applied": 0.0` from `sums`; delete `last_p_t: float = 0.0`; in `update`, delete `self.sums["box_hint/applied"] += r.n_hint_applied / denom` and `self.last_p_t = r.p_t` (and the `denom = ...` line if it is now used only for that); in `flush`, delete the `"box_hint/p": self.last_p_t` and `"box_hint/applied": ...` output keys.
- Change `run_epoch`'s `on_checkpoint` type to drop `p_t`: `on_checkpoint: Callable[[int, int, int], None]`; update its docstring (`on_checkpoint(global_step, epoch, nan_streak)`); change the call `on_checkpoint(global_step, epoch, result.p_t, nan_streak)` → `on_checkpoint(global_step, epoch, nan_streak)`.
- Update the module docstring's `box_hint` mention ("Bernoulli box-hint sampling") to drop it.

- [ ] **Step 4: Run the loop tests to verify they pass**

Run: `uv run pytest -o "addopts=" tests/unit/test_train_step.py tests/unit/test_train_loop_legacy_k1.py -q`
Expected: PASS (or, if `test_train_loop_legacy_k1.py` was deleted, just `test_train_step.py` PASS).

- [ ] **Step 5: Commit**

```bash
git add src/custom_sam_peft/train/loop.py tests/unit/test_train_step.py tests/unit/test_train_loop_legacy_k1.py
git rm tests/unit/test_box_hint_schedule.py
git commit -m "refactor(train): remove box_hint sampler/gate/logging + StepResult.p_t (#88)"
```

---

## Block (d) — checkpoint + trainer

### Task 6: Remove `box_hint_p` from checkpoint save/load; add an old-key resume test

**Files:**

- Modify: `src/custom_sam_peft/train/checkpoint.py`.
- Test: `tests/unit/test_train_checkpoint.py`, `tests/unit/test_checkpoint_roundtrip.py`.

- [ ] **Step 1: Write the back-compat (old-key) resume test first**

Add to `tests/unit/test_train_checkpoint.py` a test that proves `load_full_state` resumes cleanly from an OLD-format payload that still carries a `box_hint_p` key (the §6.2 / §6.3 / §8 requirement). Build it by saving a checkpoint, then injecting `box_hint_p` back into the on-disk payload to simulate a pre-removal file:

```python
def test_load_full_state_tolerates_legacy_box_hint_p_key(tmp_path: Path) -> None:
    """Resume must ignore a stale box_hint_p key from a pre-#88 checkpoint."""
    cfg = _make_cfg(tmp_path)
    w_a = make_stub_wrapper(dim=8)
    apply_lora(w_a, cfg.peft)
    opt_a = _trainable_optimizer(w_a)
    sched_a = torch.optim.lr_scheduler.LambdaLR(opt_a, lr_lambda=lambda s: 1.0)
    state_dir = tmp_path / "checkpoints" / "step_7"
    save_full_state(state_dir, w_a, opt_a, sched_a, 7, 0, 0, cfg)  # new signature: no box_hint_p

    # Simulate a pre-#88 checkpoint by re-injecting the legacy key.
    state_file = state_dir / "training_state.pt"
    payload = torch.load(state_file, weights_only=False)
    payload["box_hint_p"] = 0.42
    torch.save(payload, state_file)

    w_b = make_stub_wrapper(dim=8)
    apply_lora(w_b, cfg.peft)
    opt_b = _trainable_optimizer(w_b)
    sched_b = torch.optim.lr_scheduler.LambdaLR(opt_b, lr_lambda=lambda s: 1.0)
    rs = load_full_state(state_dir, w_b, opt_b, sched_b, cfg)
    assert isinstance(rs, ResumeState)
    assert rs.start_step == 7
    assert not hasattr(rs, "box_hint_p")
```

Also update the EXISTING tests in this file: `save_full_state(...)` calls drop the `box_hint_p=0.5` kwarg and the positional `0.8` (the call `save_full_state(state_dir, w_a, opt_a, sched_a, 5, 0, 0, 0.8, cfg)` becomes `save_full_state(state_dir, w_a, opt_a, sched_a, 5, 0, 0, cfg)`); delete `assert state["box_hint_p"] == 0.5` and `assert rs.box_hint_p == 0.8`.

- [ ] **Step 2: Run the new test to verify it fails**

Run: `uv run pytest -o "addopts=" tests/unit/test_train_checkpoint.py::test_load_full_state_tolerates_legacy_box_hint_p_key -q`
Expected: FAIL — `save_full_state` still requires `box_hint_p`, and `load_full_state` still reads `state["box_hint_p"]`, so the new (no-`box_hint_p`) call signature errors.

- [ ] **Step 3: Edit `train/checkpoint.py`**

- In `ResumeState`: delete the `box_hint_p: float` field.
- In `save_full_state`: delete the `box_hint_p: float,` parameter and the `"box_hint_p": float(box_hint_p),` payload key.
- In `load_full_state`: delete the `box_hint_p=float(state["box_hint_p"]),` line from the returned `ResumeState(...)`. (This is the only `box_hint_p` access — a bare `state["box_hint_p"]` — so removing it makes resume tolerate both old payloads that still carry the key (ignored) and new ones that omit it. No `.get()` shim needed.)
- Update the module docstring: drop `box_hint_p` from the "optimizer / scheduler / RNG / step / epoch / box_hint_p" line.

- [ ] **Step 4: Run the checkpoint tests to verify they pass**

Run: `uv run pytest -o "addopts=" tests/unit/test_train_checkpoint.py tests/unit/test_checkpoint_roundtrip.py -q`
Expected: PASS — including the new old-key tolerance test. (Update `test_checkpoint_roundtrip.py` in the same step: drop the `box_hint_p=cfg.train.box_hint.p_start` kwarg from its `save_full_state` call and the `assert rs.box_hint_p == pytest.approx(...)` line.)

- [ ] **Step 5: Commit**

```bash
git add src/custom_sam_peft/train/checkpoint.py tests/unit/test_train_checkpoint.py tests/unit/test_checkpoint_roundtrip.py
git commit -m "refactor(checkpoint): drop box_hint_p; tolerate legacy key on resume (#88)"
```

### Task 7: Remove `decay_steps`, `resolved_box_hint`, `box_hint_p`, and `box_hint_p_final` from `train/trainer.py`

**Files:**

- Modify: `src/custom_sam_peft/train/trainer.py`.
- Test: `tests/unit/test_schedule_resolution.py`, `tests/unit/test_trainer_run_dir.py`.

- [ ] **Step 1: Update the schedule-resolution test first (failing state)**

In `tests/unit/test_schedule_resolution.py`: `resolve_schedule_steps` will return a 2-tuple `(save_every, eval_every)` and no longer accept `decay_steps`. Rewrite the tests:

- Every `resolve(...)` / `resolve_schedule_steps(...)` call drops the `decay_steps=...` kwarg.
- Every `save, eval, decay = resolve(...)` unpack becomes `save, eval = resolve(...)`.
- Delete the decay-specific tests: `test_decay_steps_defaults_to_75_percent_of_run`, `test_decay_steps_formula_rounds`, `test_explicit_decay_steps_unchanged`, `test_tiny_run_decay_steps_floor`, and the `assert decay_steps ...` lines elsewhere.
- Delete the `BoxHintSchedule`-construction integration tests in this file (`test_box_hint_decay_steps_none`, `test_box_hint_decay_steps_explicit_int`, `test_box_hint_monotone_validator_*`, `test_default_box_hint_has_none_for_decay_steps`, and the two `model_copy(update={"decay_steps": ...})` round-trip tests that build `box_hint=BoxHintSchedule(...)`). Keep the `save_every`/`eval_every` resolution tests, with `decay_steps` removed from their calls.

- [ ] **Step 2: Run the schedule-resolution test to verify it fails**

Run: `uv run pytest -o "addopts=" tests/unit/test_schedule_resolution.py --collect-only -q`
Expected: collection fails OR tests fail — `resolve_schedule_steps` still takes/returns `decay_steps`, and `BoxHintSchedule` no longer imports (deleted in Task 2).

- [ ] **Step 3: Edit `train/trainer.py`**

- `resolve_schedule_steps`: delete the `decay_steps: int | None,` parameter, the `resolved_decay = (...)` line, and `resolved_decay` from the returned tuple → `return resolved_save, resolved_eval`. Change the return type to `tuple[int, int]`. Update the docstring (drop the `decay_steps` rule and the `decay_steps` mention in "Returns").
- The call site: `resolved_save, resolved_eval = resolve_schedule_steps(save_every=..., eval_every=..., epochs=..., steps_per_epoch=...)` (drop `decay_steps=cfg.train.box_hint.decay_steps`).
- The `_LOG.info("schedule resolved: ...")` line: drop the `decay_steps=%d` term and the `resolved_decay` arg.
- Delete `resolved_box_hint = cfg.train.box_hint.model_copy(...)` and the `"box_hint": resolved_box_hint,` entry in the `resolved_train` `model_copy`.
- `_maybe_checkpoint`: delete the `p_t: float,` parameter; in its `save_full_state(...)` call, delete `box_hint_p=p_t,`.
- The `on_checkpoint` closure: change `def on_checkpoint(step, epoch, p_t, streak)` → `def on_checkpoint(step, epoch, streak)` and the body `self._maybe_checkpoint(step, epoch, streak, run_dir, ...)`.
- The no-resume default: change `ResumeState(start_step=0, start_epoch=0, nan_streak=0, box_hint_p=cfg.train.box_hint.p_start)` → drop the `box_hint_p=...` kwarg.
- Both `metrics.json` writes: delete the `"box_hint_p_final": _box_hint_p(global_step, cfg.train.box_hint),` line.
- Imports: change `from custom_sam_peft.train.loop import OomState, _box_hint_p, run_epoch` → `from custom_sam_peft.train.loop import OomState, run_epoch`.

- [ ] **Step 4: Update `tests/unit/test_trainer_run_dir.py`**

This test builds a Mock cfg and sets `cfg.train.box_hint.p_start/p_end/decay_steps` and `cfg.train.box_hint.model_copy.return_value = cfg.train.box_hint`. Delete those `cfg.train.box_hint.*` setup lines (the trainer no longer touches `box_hint`). If the test asserts the resolved `config.yaml` contains `box_hint`, delete that assertion.

- [ ] **Step 5: Run the trainer tests to verify they pass**

Run: `uv run pytest -o "addopts=" tests/unit/test_schedule_resolution.py tests/unit/test_trainer_run_dir.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/custom_sam_peft/train/trainer.py tests/unit/test_schedule_resolution.py tests/unit/test_trainer_run_dir.py
git commit -m "refactor(trainer): drop decay_steps/box_hint_p/box_hint_p_final plumbing (#88)"
```

---

## Block (e) — templates + docs (file-disjoint; parallelizable with blocks a–d)

### Task 8: Remove the `box_hint:` block from all nine templates

**Files (modify):**

- `configs/examples/coco_text_lora.yaml`, `coco_text_qlora.yaml`, `coco_text_lora_subset.yaml`, `coco_text_auto_split.yaml`, `coco_text_no_val.yaml`, `gpu_smoke_lora.yaml`, `gpu_smoke_qlora.yaml`, `min_gpu_qlora.yaml`
- `src/custom_sam_peft/cli/templates/config_full.yaml`

- [ ] **Step 1: Remove each `box_hint:` block**

In each of the 8 `configs/examples/*.yaml`, delete the 4-line block:

```yaml
  box_hint:
    p_start: 1.0
    p_end: 0.0
    decay_steps: 5000
```

In `src/custom_sam_peft/cli/templates/config_full.yaml`, delete the full block including its comments (lines ~66–72):

```yaml
  box_hint:
    p_start: 1.0
    p_end: 0.0
    # decay_steps defaults to max(1, round(0.75 * epochs * steps_per_epoch)) when omitted:
    # the box hint decays over the first 75% of the run, then trains at p=0.
    # Uncomment and set an explicit integer to override:
    # decay_steps: 5000
```

Because the schema removed the field and configs use the strict (`_Strict`) model, a leftover `box_hint:` key now raises `extra_forbidden` `ValidationError` — so removal from all nine is mandatory.

- [ ] **Step 2: Verify no `box_hint` remains in any template**

Run: `rg -n 'box_hint' configs/examples/ src/custom_sam_peft/cli/templates/config_full.yaml`
Expected: 0 matches.

- [ ] **Step 3: Verify the templates load under the strict schema**

Run: `uv run pytest -o "addopts=" tests/unit/test_config_examples.py tests/unit/test_cli_init.py -q`
Expected: PASS — the example-config load test and the `config_full.yaml` render/init test both succeed with the field gone. (If `test_cli_init.py` asserts the rendered template contains `box_hint`, delete that assertion.)

- [ ] **Step 4: Commit**

```bash
git add configs/examples/*.yaml src/custom_sam_peft/cli/templates/config_full.yaml \
        tests/unit/test_config_examples.py tests/unit/test_cli_init.py
git commit -m "chore(configs): drop box_hint block from all 9 templates (#88)"
```

### Task 9: Update `ARCHITECTURE.md`, `config-schema.md`, and `CHANGELOG.md`

**Files (modify):**

- `docs/ARCHITECTURE.md`, `docs/config-schema.md`, `CHANGELOG.md`

- [ ] **Step 1: Rewrite the Prompt invariant in `ARCHITECTURE.md` (line 6)**

Replace the current sentence ("Auxiliary localization hints (currently just GT box hints, the `box_hint` curriculum from #14) ride alongside via `SupportPrompts`. …") with text stating that **training is text-only** and `SupportPrompts` is a reserved extension seam (#126 §12) carrying no fields today. Suggested:

```markdown
**Prompt invariant:** Text is the only prompt — the model takes one or more text (class) prompts and segments all matching instances. Training is text-only (the `box_hint` localization-hint curriculum was removed in #88). `SupportPrompts` is retained as a reserved extension seam (see [#126](https://github.com/NguyenJus/custom-sam-peft/issues/126) §12) for future hints (masks / points); it carries no fields today and is never used at inference.
```

Keep line 17 (`base.py ... Prompts (= TextPrompts), SupportPrompts ...`) unchanged.

- [ ] **Step 2: Delete the three `train.box_hint.*` rows in `config-schema.md` (lines 126–128)**

Delete the `train.box_hint.p_start`, `train.box_hint.p_end`, and `train.box_hint.decay_steps` table rows.

- [ ] **Step 3: Add a `[Unreleased]` entry to `CHANGELOG.md`**

Under `## [Unreleased]`, following the Keep-a-Changelog section style already used in the file (e.g. the `### Breaking — text-primary prompt invariant (#126)` section), add:

```markdown
### Removed — box_hint localization-hint curriculum (#88)

- **train**: removed the `box_hint` curriculum and the `BoxHintSchedule`
  config model (`train.box_hint.*`). Training is now text-only.
- **Changed**: `SupportPrompts` is retained as a field-less reserved extension
  seam (#126 §12) for future mask/point hints; `Sam3Wrapper.forward(support=)`
  stays as a no-op. Inference is unchanged (already text-only).
- **Note**: resume tolerates pre-removal checkpoints — a stale `box_hint_p`
  key in an old `training_state.pt` is ignored.
- **Note**: any config carrying `train.box_hint:` now fails to load with a
  Pydantic `extra_forbidden` error; delete the block from your YAML.
```

- [ ] **Step 4: Markdown-lint the changed docs**

Run the markdown-lint gate command (top of this plan) on `docs/ARCHITECTURE.md docs/config-schema.md CHANGELOG.md`. Expected: clean exit (0). Fix findings.

- [ ] **Step 5: Commit**

```bash
git add docs/ARCHITECTURE.md docs/config-schema.md CHANGELOG.md
git commit -m "docs: text-only training; box_hint removed; SupportPrompts seam retained (#88)"
```

---

## Block (f) — verify + cross-issue reconcile (runs last)

### Task 10: Static + type + lint sweep

**Files:** none (verification).

- [ ] **Step 1: Confirm no `box_hint` surface remains in production code/configs/docs**

Run:

```bash
rg -n 'box_hint|BoxHintSchedule|_box_hint_p|_build_geometric_prompt|box_hint_p_final|box_xyxy_to_cxcywh' \
  src/ tests/ configs/ docs/ARCHITECTURE.md docs/config-schema.md CHANGELOG.md
```

Expected: 0 matches (`box_xyxy_to_cxcywh` is included to confirm the orphaned import was removed). Historical `docs/superpowers/` records are intentionally excluded and untouched.

- [ ] **Step 2: ruff + mypy**

Run: `uv run ruff check src/ tests/ && uv run mypy src/`
Expected: clean. (Watch for F401 on the removed `random` / `box_xyxy_to_cxcywh` / `_box_hint_p` / `BoxHintSchedule` / `SupportPrompts` imports — all dropped in the relevant tasks.)

### Task 11: Full gated test suite (the binding final check)

**Files:** none (verification).

- [ ] **Step 1: Run the full non-GPU suite WITH the coverage gate**

Run: `uv run pytest -m "not gpu"`
Expected: PASS, **including** the global `--cov-fail-under=80` gate (do NOT pass `-o "addopts="` here — the final check must run the gate). Removing `box_hint` trims both the coverage numerator (deleted source lines) and denominator together; confirm the gate still holds. If coverage dips below 80%, the shortfall is almost certainly a now-untested branch that the deletions exposed — inspect the coverage report and either remove genuinely dead code or restore a deleted test that covered surviving code.

### Task 12: Open the PR and reconcile cross-issues

**Files:** none (`gh` operations).

- [ ] **Step 1: Push the branch**

```bash
git push -u origin HEAD
```

- [ ] **Step 2: Open the PR closing #88, linking spec + plan**

```bash
gh pr create --assignee @me --label <existing-or-created-label> \
  --title "Remove the box_hint curriculum; keep the SupportPrompts seam (#88)" \
  --body "Closes #88.

Removes the box_hint localization-hint curriculum (schema, train loop, checkpoint, trainer, model plumbing, data field, 9 templates, docs). Training is now text-only; inference unchanged. The SupportPrompts seam is retained field-less (#126 §12). Resume tolerates pre-removal checkpoints.

Gated on the literature review at docs/research/2026-05-30-issue-88-box-hint-curriculum-lit-review.md (user confirmed 'remove').

Spec: docs/superpowers/specs/2026-05-30-issue-88-remove-box-hint-curriculum-design.md
Plan: docs/superpowers/plans/2026-05-30-issue-88-remove-box-hint-curriculum-plan.md"
```

(Run `gh label list`; pick an existing label such as `refactor`/`enhancement` or create one inline per the Labels rule.)

- [ ] **Step 3: Comment on #126 recording the conscious reversal (do NOT reopen)**

```bash
gh issue comment 126 --body "Heads-up: #88 (PR linked) intentionally **reverses** #126's 'KEEP box_hint in full' decision. Rationale: box_hint decays to p_end=0.0 over the first ~75% of every run, so it cannot move the endpoint — only the optimization path — and a speed-only benefit fails the project's guiding principle (endpoint accuracy and user-facing simplicity above training speed). The SupportPrompts seam #126 valued is **retained** field-less for future mask/point hints (§12); only the box-hint curriculum is removed. This is a conscious, recorded reversal — #126 stays closed (not reopened)."
```

- [ ] **Step 4: Comment on #120 that the `decay_steps=5000` cite line-item is moot**

```bash
gh issue comment 120 --body "FYI: #88 removes BoxHintSchedule entirely, so the 'literature-cite the decay_steps=5000 default' line-item is now **moot** — the field no longer exists."
```

- [ ] **Step 5: Final report**

Confirm: PR opened closing #88; #126 and #120 annotated; full gated suite green. Then follow the session's PR/close-out protocol.

---

## Self-review (planner — completed)

**Spec coverage** — every spec §6 surface maps to a task: schema/`BoxHintSchedule`→T2; `SupportPrompts.boxes`→T3; `_build_geometric_prompt` + wrapper/adapter/`_validate_inputs`→T4; loop sampler/gate/logging/`StepResult`/`import random`/`on_checkpoint` sig→T5; checkpoint `box_hint_p` + back-compat test→T6; trainer `decay_steps`/`resolved_box_hint`/`box_hint_p`/`box_hint_p_final`→T7; 9 templates→T8; ARCHITECTURE/config-schema/CHANGELOG→T9; verify+gate→T10/T11; PR + #126/#120 reconcile→T12. The kept seam (`SupportPrompts` field-less, `support=` no-op) is enforced in T3/T4. The §6.3 test-delete/update list is covered (T2,T4,T5,T6,T7,T8) plus the three planner-found extras (T4).

**Placeholder scan** — no TBD/TODO; every code step shows the actual edit; every verify step has a concrete command + expected result.

**Type consistency** — `resolve_schedule_steps` returns a 2-tuple `(save_every, eval_every)` consistently across T7 and its test edits; `StepResult.empty(nan_streak=...)` and `on_checkpoint(step, epoch, streak)` / `on_checkpoint(step, epoch, nan_streak)` signatures match between `loop.py` (T5) and `trainer.py` (T7); `save_full_state(...)` and `ResumeState` drop `box_hint_p` consistently across `checkpoint.py` (T6) and all three checkpoint test files.

**Gaps found in the spec (corrected inline in "Spec gaps & corrections"):** (1) the orphaned `box_xyxy_to_cxcywh` import; (2) `_validate_inputs` box-validation block lives at ~276–292, not the cited 236–241 (signature only); (3) three test/fixture files missing from §6.3 — `tiny_sam3_lora_stub.py`, `tiny_sam3_stub.py`, `test_static_guards.py`. None change scope; all are mechanical consequences of the named removals.
