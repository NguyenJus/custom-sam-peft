# Remove the `box_hint` Curriculum (gated on a literature review) — Design

**Status:** Draft (2026-05-30)
**Issue:** [#88](https://github.com/NguyenJus/custom-sam-peft/issues/88)
**Scope:** A deep-research literature-review note (Phase 1) that gates the
removal of the `box_hint` localization-hint curriculum (Phase 2). Phase 2
deletes `BoxHintSchedule` and all of its plumbing across `config/schema.py`,
`train/loop.py`, `train/checkpoint.py`, `train/trainer.py`,
`models/sam3.py`, and `data/base.py`, **keeping the `SupportPrompts`
extension seam** (per #126 §12). Training becomes text-only; inference is
unchanged (already text-only).
**Reverses:** #126's explicit "KEEP `box_hint` in full" decision — this
reversal is conscious and recorded (see §9).

---

## 1. Background & how #88 changed shape

Issue #88 originally asked to **hyperparameter-tune** the `box_hint`
schedule (`p_start`, `p_end`, `decay_steps`) on an empirical sweep, plus a
short literature review. A brainstorming session changed the scope
decisively. The following are **final inputs** to this spec, not open
questions:

1. **The empirical sweep is dropped — infeasible on the available
   hardware.** Single RTX 5070 Ti (16 GB); COCO 2017 is not downloaded; one
   full fine-tune is ≈100k+ optimizer steps (days of wall-clock); a 3-D
   grid with seeds is far worse. There is no realistic path to the
   sweep evidence the original acceptance criteria demanded.

2. **The real question is whether the curriculum is worth keeping at all.**
   The schedule decays to `p_end=0.0`, and `decay_steps` auto-resolves to
   75% of the run (`max(1, round(0.75 * epochs * steps_per_epoch))`,
   `trainer.resolve_schedule_steps`), so the final ~25% of every run
   already trains pure text-only **regardless of the schedule**. `box_hint`
   is training-only and never used at inference (box-term losses are
   `0.0`). It therefore **cannot move the final optimum** — it can only
   change the *optimization path* (convergence speed). A knob that cannot
   change the endpoint is a candidate for removal.

3. **Guiding principle (the lit-review's evaluative lens).** The project's
   design priority order is:

   1. **Final model accuracy / endpoint quality.**
   2. **Simplicity for user understanding** — a small, comprehensible
      config surface.
   3. *(far behind)* **Training / convergence speed.**

   A speed-only benefit is therefore a **weak** reason to keep a knob that
   adds config surface. The literature review **must** evaluate `box_hint`
   primarily on *"does it improve the endpoint or reduce user-facing
   complexity?"* and treat any convergence-speed benefit as secondary.

4. **Decision: remove the `box_hint` curriculum, but keep the
   `SupportPrompts` extension seam** (#126 §12, reserved for future
   mask/point hints). This reverses #126's "KEEP `box_hint`" decision; the
   reversal is recorded, not silent (§9).

5. **The removal is gated** behind a thorough literature review used as a
   **decision checkpoint** (Phase 1 → user reads the note → confirms).

---

## 2. Goals & non-goals

### Goals

- A committed, citation-backed deep-research note that answers the four
  questions in §5.2 and ends with an **explicit keep/remove
  recommendation** reasoned against the guiding principle (§1.3).
- **On the user's confirmation**, full removal of the `box_hint`
  curriculum: schema, train loop, checkpoint, trainer, model plumbing,
  data field, templates, tests, and docs.
- Retention of the `SupportPrompts` dataclass as a **field-less, documented
  reserved seam** so future mask/point hints (#126 §12) have a home.
- Training is text-only; inference is unchanged; the full test suite stays
  green including the `--cov-fail-under=80` gate; resume tolerates
  pre-removal checkpoints.

### Non-goals (out of scope — stated explicitly)

- **The empirical sweep** and **any sweep / grid harness** — dropped (§1.1).
- **Inference changes** — inference is already text-only; nothing to do.
- **Removing the `SupportPrompts` seam** — it is deliberately retained.
- **Retuning any other defaults** — only `box_hint` is touched.
- **Historical plan docs** under `docs/superpowers/plans/` — left untouched
  (they are dated records).

---

## 3. Two-phase plan with a hard user-decision gate

The work is two phases separated by a **user-decision gate**. The
implementation session executing Phase 1 **halts** after committing the
note; the user reads it and confirms before Phase 2 begins.

```text
Phase 1  ──►  COMMIT lit-review note  ──►  [USER-DECISION GATE]  ──►  Phase 2
(deep research)   (interface OUT)          (user reads + confirms)   (removal)
```

| Phase | Deliverable | Boundary contract |
|---|---|---|
| 1 | `docs/research/2026-05-30-issue-88-box-hint-curriculum-lit-review.md` with citations + explicit recommendation | **OUT:** committed note. Session **halts**; user reads + confirms. |
| 2 | `box_hint` removed; `SupportPrompts` seam retained; docs + tests updated; PR | **IN:** the user-confirmed decision from the gate. |

---

## 4. Guiding principle, restated for the implementer

This is the single lens through which Phase 1's recommendation must be
written and through which the gate decision is made:

> **Priority order: (1) endpoint accuracy, (2) user-facing simplicity,
> (3) — far behind — training speed.** A convergence-speed benefit alone
> does **not** justify keeping a config knob. `box_hint` provably cannot
> change the endpoint (it decays to zero; the final 25% of every run is
> already text-only), so the bar for keeping it is: *does the literature
> show it improves the text-only endpoint or reduces user-facing
> complexity?* If not, remove it.

---

## 5. Phase 1 — deep-research literature-review note

### 5.1 Deliverable & process

- **Path:** `docs/research/2026-05-30-issue-88-box-hint-curriculum-lit-review.md`
  (matches the existing `docs/research/YYYY-MM-DD-issue-N-topic.md`
  convention, e.g. `2026-05-24-issue-137-qlora-8gb-feasibility.md`).
- **Process:** produced via the project's `deep-research` capability — a
  thorough, **adversarially-verified**, broadly-cited deep-research run.
  Claims that bear on the recommendation must be cross-checked, not taken
  from a single source.
- **Header convention:** mirror the existing research notes — a title, an
  issue link, the date, and a link to this spec (and to the Phase 2 plan
  once it exists). Open with a **TL;DR** that states the recommendation up
  front.

### 5.2 The note MUST answer, with citations

1. **Does the technique appear in published research, and under what
   name?** The technique is precisely: *feeding decaying ground-truth box
   hints alongside text/class prompts, weaned to zero so the model is
   trained toward text-only inference.* Survey at least:
   - **Curriculum learning** — Bengio et al., 2009.
   - **Scheduled sampling** — Bengio et al., 2015 — and **Huszár's (2015)
     statistical-inconsistency critique** of it.
   - **Learning using privileged information (LUPI)** — Vapnik & Vashist.
   - **Denoising training in DETR-family detectors** — **DN-DETR**
     (Li et al., 2022) and **DINO** (Zhang et al., 2022) as the *closest
     analog*. Note explicitly that their box-denoising is a **parallel
     branch kept throughout training**, **not annealed to zero** — the
     opposite of our decayed-to-zero curriculum.
   - **Box-as-prompt in promptable / open-vocab models** — SAM / SAM-family,
     GLIP, Grounding-DINO. Note explicitly that these use boxes as
     **inference-time prompts**, not as a decayed **training** curriculum.
2. **Endpoint vs. path.** Does the prior art show such a *removed-by-end*
   hint changes the **endpoint** (final solution quality), or only the
   **optimization path** (convergence speed)?
3. **Evidence of benefit.** Is there evidence the technique actually
   *accelerates convergence* or *improves the text-only endpoint*?
4. **Explicit recommendation (keep / remove),** reasoned **against the
   guiding principle** (§4) — i.e. weighing endpoint accuracy and
   user-facing simplicity above speed. A speed-only finding ⇒ recommend
   remove.

### 5.3 Phase boundary — interface contract OUT

The committed note carrying an explicit recommendation. **The
implementation session then HALTS.** The user reads the note and confirms
the decision before any Phase 2 code change.

### 5.4 Contingency (both branches; removal is primary)

- **Expected (primary path):** the literature shows endpoint-parity, an
  unproven / weak speedup, and that our specific decayed-to-zero variant is
  largely ad-hoc (the closest analog, DN-DETR/DINO denoising, keeps the
  branch rather than annealing it). ⇒ **User confirms removal; proceed to
  Phase 2 as specified below.**
- **Surprise branch:** the literature shows a **strong endpoint or
  simplicity benefit** for the decayed-to-zero variant. ⇒ The gate lets the
  user **pivot Phase 2** to *"keep `box_hint` + cite the chosen defaults"*
  (feeding #120's "literature-cite every default" effort) instead of
  removing. This spec describes the removal path in full; the keep-path is
  a planner amendment triggered at the gate, not pre-specified here.

---

## 6. Phase 2 — remove the curriculum, keep the seam

**Interface contract IN:** the user-confirmed "remove" decision from §5.3.
If the user instead pivots to "keep" (§5.4 surprise branch), Phase 2 is
re-planned and the rest of this section does not apply.

### 6.1 What is KEPT (the `SupportPrompts` seam)

- `SupportPrompts` survives in `data/base.py` as a **field-less, documented
  reserved extension seam** (#126 §12). Its docstring is updated to say it is
  reserved for future hints (masks / positive points / negative points) and
  **currently carries no fields**.
- `Sam3Wrapper.forward` keeps its signature
  `support: SupportPrompts | None = None`, but **ignores `support`** and
  calls the backbone text-only. `_validate_inputs` keeps the `support`
  parameter as a no-op.
- The `SupportPrompts` **import** in `models/sam3.py` is kept.
- Inference is unchanged (already text-only; box-term losses already `0.0`).

### 6.2 What is REMOVED — exact surface

Line numbers are at the spec's writing time (verified against current
`main`); treat the **named symbols** as authoritative if lines drift.

#### Schema — `src/custom_sam_peft/config/schema.py`

- The **`BoxHintSchedule`** class (≈ lines 505–541), including its
  `_check_monotone` validator.
- The **`TrainHyperparams.box_hint`** field
  (≈ line 570: `box_hint: BoxHintSchedule = Field(default_factory=BoxHintSchedule)`).

#### Train loop — `src/custom_sam_peft/train/loop.py`

- **`_box_hint_p`** (≈ lines 151–160) and every call site
  (`p_t = _box_hint_p(...)` at ≈ line 200).
- The **`Bernoulli(p_t)` gate** + **`hints_g` construction** (≈ lines
  251–263): the `if row_targets and random.random() < p_t:` branch, the
  `box_tensor`/`hints_g.append(...)` plumbing, `n_hint_applied`, and the
  `hints_g` list itself. **`targets_g` construction MUST remain** — losses
  still need per-row targets. The forward call into the model drops its
  `box_hints=` argument (see model surface below).
- **Logging / accounting:** `box_hint/p` and `box_hint/applied` keys in
  `_ScalarWindow.sums` (≈ line 433), `_ScalarWindow.last_p_t` (≈ line 439)
  and its assignment (≈ line 460), the `box_hint/applied` accumulation (≈
  lines 454–455), and the `box_hint/p`/`box_hint/applied` emit in
  `flush()` (≈ lines 473–474). `StepResult.p_t` and the `n_hint_applied`
  field are removed; `StepResult.empty(...)` and the `on_checkpoint`
  callback signature drop their `p_t` argument (see trainer surface).
- The now-unused **`import random`** (line 14): its only use is the
  `box_hint` sampling at ≈ line 258 — remove the import **and** its
  `# noqa: S311` on that line.

#### Model — `src/custom_sam_peft/models/sam3.py`

- **`_build_geometric_prompt`** (≈ lines 108–166) — deleted.
- **`Sam3Wrapper.forward`** (≈ lines 225–234): drop
  `box_hints = support.boxes if support is not None else None` and the
  `box_hints=box_hints` kwarg; call the inner model text-only. Keep the
  `support` parameter (ignored).
- **`_validate_inputs`** (≈ lines 236–241): keep the `support` parameter as
  a no-op; remove any `support`/`box_hints` validation logic.
- **`_Sam3ImageAdapter.forward`** (≈ lines 398–459): drop the
  `box_hints: list[Tensor | None] | None = None` parameter (≈ line 402),
  drop the `_build_geometric_prompt(...)` call (≈ lines 440–445), and make
  `geometric_prompt` the **always-zero-length dummy** unconditionally
  (the existing `gp is None` fallback `Prompt(...)` at ≈ lines 447–452
  becomes the only path). The backbone is called text-only.
- Keep the `SupportPrompts` import.

#### Data — `src/custom_sam_peft/data/base.py`

- Drop the **`SupportPrompts.boxes`** field (≈ line 37).
- **Keep the `SupportPrompts` dataclass** as a documented empty seam;
  rewrite its docstring (≈ lines 20–35) to say it is reserved for future
  hints (#126 §12) and currently carries no fields. Remove the now-stale
  `boxes` length-convention prose.

#### Checkpoint — `src/custom_sam_peft/train/checkpoint.py`

- Drop **`ResumeState.box_hint_p`** (≈ line 82).
- Drop the `box_hint_p` parameter + payload key in **`save_full_state`**
  (≈ lines 156, 173).
- **`load_full_state`** (≈ line 244): stop populating `box_hint_p` from the
  state dict. **Back-compat requirement:** resume MUST tolerate **both**
  old checkpoints that still carry a `box_hint_p` key (ignore it) **and**
  new checkpoints that omit it. Reading is via the removed field, so the
  natural outcome (no longer reading the key) already ignores old
  checkpoints — but the implementer must verify no other `load_full_state`
  code path does a bare `state["box_hint_p"]` that would `KeyError` on a new
  checkpoint. The module docstring's `box_hint_p` mention (≈ line 5) is
  updated.

#### Trainer — `src/custom_sam_peft/train/trainer.py`

- **`resolve_schedule_steps`** (≈ lines 124–152): drop the `decay_steps`
  parameter, its resolution rule, and `decay_steps` from the returned
  tuple; **keep `save_every` / `eval_every` resolution**. Update its
  docstring. Update the single call site (≈ lines 433–439) and the
  `_LOG.info("schedule resolved: ...")` line (≈ lines 440–446) to drop the
  `decay_steps` term.
- Drop **`resolved_box_hint`** + the `box_hint` entry in the `resolved_train`
  `model_copy` (≈ lines 447–453).
- Drop `box_hint_p` from the **`_save_checkpoint`** path (≈ lines 387–395)
  and from the **`ResumeState`** construction at the no-resume default
  (≈ lines 480–487, `box_hint_p=cfg.train.box_hint.p_start`).
- Drop the **`box_hint_p_final`** field written into `metrics.json`
  (≈ lines 555, 567) and the `_box_hint_p` import (≈ line 35). Note: the
  eval spec's `metrics.json` schema also lists `box_hint_p_final`; that is
  a documentation artifact and is dropped here.
- Keep all `save_every` / `eval_every` resolution and the `save_full_state`
  / `load_full_state` calls otherwise intact.

#### Templates (9 files) — remove the `box_hint:` block from each

```text
configs/examples/coco_text_lora.yaml
configs/examples/coco_text_qlora.yaml
configs/examples/coco_text_lora_subset.yaml
configs/examples/coco_text_auto_split.yaml
configs/examples/coco_text_no_val.yaml
configs/examples/gpu_smoke_lora.yaml
configs/examples/gpu_smoke_qlora.yaml
configs/examples/min_gpu_qlora.yaml
src/custom_sam_peft/cli/templates/config_full.yaml
```

Each carries a `box_hint:` block (e.g. `coco_text_lora.yaml` lines 63–66:
`p_start: 1.0` / `p_end: 0.0` / `decay_steps: 5000`). Remove the whole
block from every file. Since the schema removes the field and configs use
the strict (`_Strict`) model, leaving a `box_hint:` key would now raise a
validation error — so removal from all nine is mandatory, not cosmetic.

### 6.3 Tests

- **Delete:** `tests/unit/test_box_hint_schedule.py`,
  `tests/unit/test_geometric_prompt_builder.py`.
- **Update** the tests that incidentally exercise `box_hint`:
  `test_schedule_resolution.py`, `test_train_step.py`,
  `test_sam3_adapter.py`, `test_train_loop_legacy_k1.py`,
  `test_train_checkpoint.py`, `test_checkpoint_roundtrip.py`,
  `test_config_schema.py`, `test_trainer_run_dir.py`.
- **Add back-compat coverage:** at least one test in `test_train_checkpoint.py`
  (or `test_checkpoint_roundtrip.py`) asserting that `load_full_state`
  resumes cleanly from a checkpoint payload that still contains a
  `box_hint_p` key (old format) — guarding the §6.2 back-compat
  requirement.
- The full suite (`uv run pytest -m "not gpu"`) MUST stay green **and** the
  global `--cov-fail-under=80` coverage gate MUST still pass. Removing
  `box_hint` removes lines from both the numerator and denominator of
  coverage; the implementer verifies the gate holds after the cut (see
  MEMORY: a test subset can be run with `-o "addopts="` to bypass the gate
  during iteration, but the **final** check must run the full gated suite).

### 6.4 Docs & issue reconciliation

- **`docs/ARCHITECTURE.md`** — the **Prompt invariant** (line 6) currently
  says auxiliary box hints "ride alongside via `SupportPrompts`
  (… the `box_hint` curriculum from #14)". Rewrite it to state that
  **training is text-only** and `SupportPrompts` is a **reserved extension
  seam** (#126 §12) carrying no fields today. Keep the `base.py` symbol
  line (line 17) referencing `SupportPrompts`.
- **`docs/config-schema.md`** — remove the three `train.box_hint.*` rows
  (≈ lines 126–128).
- **`CHANGELOG.md`** — add an entry under `[Unreleased]` describing the
  removal (Removed: `box_hint` curriculum / `BoxHintSchedule`; Changed:
  training is text-only; Note: `SupportPrompts` retained as a reserved
  seam; Note: resume tolerates pre-removal checkpoints). Follow the
  existing Keep-a-Changelog section style.
- **Leave `docs/superpowers/plans/` untouched** (dated records).

### 6.5 PR & cross-issue annotations

- The PR **closes #88**.
- **Comment on #126** recording that its explicit "KEEP `box_hint`"
  decision is **intentionally reversed** by this PR, with the rationale
  (decays to zero ⇒ cannot move the endpoint; speed-only benefit fails the
  guiding principle; seam retained). **Do NOT reopen #126.**
- **Comment on #120** that the `BoxHintSchedule.decay_steps=5000` cite
  line-item is now **moot** (the field no longer exists).

---

## 7. Implementation plan (numbered)

Consumed by the writing-plans skill. **Phase 1 and Phase 2 are separated by
the user-decision gate (§3); the plan must phase them accordingly.**

**Phase 1 — research (one phase; halts at the gate).**

**Step 1.** Run the `deep-research` capability on the §5.2 questions through
the §4 lens. Produce
`docs/research/2026-05-30-issue-88-box-hint-curriculum-lit-review.md` with
TL;DR, the four answered questions (cited), and an explicit keep/remove
recommendation. Markdown-lint it (Markdown lint gate). Commit.

> **GATE.** Session halts. User reads the note and confirms remove (primary)
> or pivots to keep (§5.4). Phase 2 begins only on a "remove" confirmation.

**Phase 2 — removal (gated; primary path).** Group into reviewable blocks:

**Step 2 (schema + data).** Remove `BoxHintSchedule` and
`TrainHyperparams.box_hint`; drop `SupportPrompts.boxes` and rewrite the
dataclass docstring to the empty-seam form. Update `test_config_schema.py`.

**Step 3 (model).** Remove `_build_geometric_prompt`; strip `box_hints`
plumbing from `Sam3Wrapper.forward`, `_validate_inputs`, and
`_Sam3ImageAdapter.forward` (text-only, always-zero-length dummy
`geometric_prompt`); keep `support` as a no-op and keep the `SupportPrompts`
import. Update `test_sam3_adapter.py`; delete
`test_geometric_prompt_builder.py`.

**Step 4 (train loop).** Remove `_box_hint_p`, the Bernoulli gate / `hints_g`
/ `n_hint_applied`, the `box_hint/*` logging + `last_p_t`, `StepResult.p_t`,
and `import random` (+ its `# noqa`). Keep `targets_g`. Update
`test_train_step.py`, `test_train_loop_legacy_k1.py`; delete
`test_box_hint_schedule.py`.

**Step 5 (checkpoint + trainer).** Drop `box_hint_p` from `ResumeState` /
`save_full_state` / `load_full_state` (with back-compat for old keys); drop
`decay_steps` from `resolve_schedule_steps` and callers (keep
`save_every`/`eval_every`); drop `resolved_box_hint`, the `_save_checkpoint`
`box_hint_p`, the no-resume default, and `box_hint_p_final` in
`metrics.json`. Update `test_schedule_resolution.py`,
`test_train_checkpoint.py`, `test_checkpoint_roundtrip.py` (incl. the old-key
resume test), `test_trainer_run_dir.py`.

**Step 6 (templates + docs).** Remove the `box_hint:` block from all nine
templates; update `docs/ARCHITECTURE.md`, `docs/config-schema.md`,
`CHANGELOG.md`. Markdown-lint changed `.md` files.

**Step 7 (verify + reconcile).** Run the full `uv run pytest -m "not gpu"`
suite with the `--cov-fail-under=80` gate; fix any coverage shortfall. Open
the PR closing #88; annotate #126 (reversal recorded, not reopened) and #120
(decay_steps cite moot).

---

## 8. Risk & edge cases

| Risk | Handling |
|---|---|
| Resume from a pre-removal checkpoint `KeyError`s on `box_hint_p`. | `load_full_state` stops reading the key; verify no bare `state["box_hint_p"]` remains. Add an explicit old-key resume test (§6.3). |
| A leftover `box_hint:` key in any template fails `_Strict` validation. | All nine templates de-blocked (§6.2); CI config-load tests catch a miss. |
| Coverage dips below 80% after the cut. | Removal trims numerator and denominator together; verify the gate on the full suite (§6.3); MEMORY note on subset-coverage iteration. |
| `targets_g` accidentally removed with `hints_g`. | Spec calls out explicitly: `targets_g` stays; only the hint branch goes (§6.2). |
| Silent reversal of #126. | §6.5 mandates a #126 comment recording the conscious reversal; do not reopen. |
| Eval spec's documented `metrics.json` lists `box_hint_p_final`. | Field dropped here; it was a doc artifact, noted in §6.2 trainer surface. |

---

## 9. #126 reversal — recorded decision

#126 explicitly decided to **KEEP `box_hint` in full** as the project's
permanent auxiliary-hint mechanism. This spec **reverses** that decision.
The reversal is conscious and recorded here and (at PR time) on #126 itself:

- `box_hint` decays to `p_end=0.0` over the first 75% of every run, so it
  **cannot change the endpoint** — only the optimization path.
- Under the guiding principle (§4), a speed-only benefit does not justify
  the config surface a knob adds.
- The **seam** that #126 valued (`SupportPrompts`, for future mask/point
  hints) is **retained** — only the box-hint *curriculum* is removed.

The PR comments on #126 to record this (not reopen it) and on #120 to mark
the `decay_steps=5000` cite line-item moot.

---

## 10. Acceptance criteria

- [ ] **Phase 1:** Deep-research note committed at
  `docs/research/2026-05-30-issue-88-box-hint-curriculum-lit-review.md` with
  citations + an explicit keep/remove recommendation reasoned against the
  guiding principle (§4); **user-decision gate passed.**
- [ ] **Phase 2 (on confirm):** the `box_hint` curriculum is fully removed
  (schema, train loop, checkpoint, trainer, model plumbing, data field,
  9 templates); the `SupportPrompts` seam is retained and documented;
  training is text-only.
- [ ] Full test suite green (`uv run pytest -m "not gpu"`) **including** the
  `--cov-fail-under=80` coverage gate; resume tolerates pre-removal
  checkpoints (old-key resume test passes).
- [ ] `docs/ARCHITECTURE.md`, `docs/config-schema.md`, and `CHANGELOG.md`
  updated; **#88 closed**; **#126 and #120 annotated** (per §6.5).
