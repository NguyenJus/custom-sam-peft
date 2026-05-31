# Plateau-Response LR Decay & Early Stopping — Literature Review

> Research write-up for [issue #197](https://github.com/NguyenJus/custom-sam-peft/issues/197)
> Date: 2026-05-30
> Method: deep-research harness (5-angle fan-out, 20 sources, 25 claims under 3-vote
> adversarial verification — 22 confirmed, 3 killed) plus manual verification of the
> SAMed reference profile against its paper and reference implementation.

## TL;DR

- **Composition verdict (high confidence): plateau-based LR decay and a warmup+cosine
  schedule are alternatives, not stacked.** Hugging Face makes them mutually exclusive
  (`lr_scheduler_type` selects exactly one of `reduce_lr_on_plateau` / `cosine` /
  `cosine_with_min_lr` / `warmup_stable_decay`). There is no documented pattern for a
  plateau cut multiplying, pausing, or layering on top of a cosine schedule. So #197's
  `plateau` mode **replaces** the per-step cosine schedule rather than composing with it.
- **For this repo's regime — a ~160-epoch convergence run that metric-early-stops —
  `ReduceLROnPlateau` is the reliable, accuracy-oriented choice**, and it becomes the
  default. A horizon-calibrated cosine/poly schedule earns its accuracy in the low-LR
  *endgame*; metric early-stopping fires before the horizon, forfeiting that endgame.
  Plateau decay reacts to the same stall that drives the stop, so a low-LR fine-tuning
  phase always happens before the run gives up — the canonical reason to pair the two.
- **Chosen defaults are all cited or `# tbd:`-tagged** (see [§7](#7--chosen-defaults-for-197)).
  The LoRA-specific *values* for a plateau ladder are a genuine evidence gap, so the
  repo-chosen picks within cited ranges carry `# tbd:` tags.
- **"Other decay methods" (SGDR warm restarts, weight-decay scheduling) are deferred:**
  they fire on a *fixed schedule*, not a plateau/validation trigger, so they do not belong
  in a plateau-*response* rung.

## §1 — Method and scope

The question: ground default hyperparameters for a `ReduceLROnPlateau`-style ladder that
**decays the learning rate before early-stopping**, for PEFT/LoRA fine-tuning of SAM. Five
angles were searched: canonical docs and defaults; composition with an existing
cosine/warmup schedule; LoRA / short-run practice; early-stopping patience and `min_delta`
(including Prechelt 1998); and alternative decay methods stacked on triggers.

A premise correction drove the final design: the issue text (and the first research pass)
assumed **short 1–3 epoch LoRA runs**. The repo's real default is a **~160-epoch
convergence run** anchored to SAMed (see [§4](#4--samed-reference-profile-grounding)). The
core findings below hold regardless; the *application* of the patience/factor values is
re-derived for the long-run regime.

## §2 — `ReduceLROnPlateau` common defaults

| Knob | PyTorch default | Keras | Practitioner range |
| --- | --- | --- | --- |
| `factor` | `0.1` | example `0.2` (docs recommend a 2×–10× cut) | `0.1`–`0.5` |
| `patience` | `10` | example `5` | `5`–`10` |
| `min_lr` | `0` | example `0.001` | repo-chosen floor |
| `threshold` (`min_delta`) | `1e-4` | `1e-4` | — |

Sources: [PyTorch `ReduceLROnPlateau`](https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html),
[Keras `ReduceLROnPlateau`](https://keras.io/api/callbacks/reduce_lr_on_plateau/). Both
count `patience` in scheduler *calls* (per eval), not epochs — which is why an **eval-count**
patience unit is correct, not an epoch- or step-count one.

## §3 — Composition verdict (the key question)

**Plateau decay is normally an alternative to cosine, not stacked on top.** The
[Hugging Face optimizer-schedules docs](https://huggingface.co/docs/transformers/en/main_classes/optimizer_schedules)
expose a single-valued `lr_scheduler_type` enum whose options include `reduce_lr_on_plateau`,
`cosine`, `cosine_with_min_lr`, and `warmup_stable_decay` — you pick exactly one. No source
documents a plateau cut that multiplies, pauses, or overrides a running cosine schedule.

**Caveat (why this is "suggestive, not a ban"):** the single-valued enum is a framework
artifact, not an explicit prohibition. Stacking *is* mechanically possible in PyTorch via
`SequentialLR` or `ChainedScheduler`. But there is no established, evidence-backed pattern
for it, and on a long cosine schedule a late plateau cut does little (cosine is already near
its floor) or compounds unpredictably. #197 therefore treats `plateau` as a schedule mode
that **replaces** cosine.

## §4 — SAMed reference-profile grounding

The 160-epoch default is **SAMed's convergence figure**, not an arbitrary budget:
[SAMed (Zhang & Liu 2023, arXiv:2304.13785)](https://arxiv.org/abs/2304.13785) — LoRA
fine-tuning of SAM (rank 4, AdamW) on the small Synapse dataset — reports *"after finetuning
only 160 epochs ... SAMed achieves 81.88 DSC."* The repo's `docs/defaults-provenance.md`
already anchors `epochs` to this source as a convergence figure under the standing priority
**final accuracy ≫ training speed**.

What SAMed actually does for the learning rate, confirmed against its
[reference implementation](https://github.com/hitachinsk/SAMed):

- **Warmup**, linear: `lr = base_lr * ((iter_num + 1) / warmup_period)`.
- **Decay**, polynomial power 0.9: `lr = base_lr * (1 - shift_iter / max_iterations) ** 0.9`,
  with `max_iterations = max_epochs * len(trainloader)` — so the LR reaches its floor *at*
  the 160-epoch horizon.
- **Stopping**: a pure **epoch-count cap** (`epoch_num >= max_epoch - 1`). There is **no
  validation-metric early stopping** — `best_performance` is initialized but never used.

**Why this flips the LR-method choice for #197.** SAMed's accuracy comes from running a
horizon-calibrated monotonic decay *to completion*, so it always reaches the low-LR endgame.
The repo's current `cosine` default is the same family (horizon-calibrated monotonic decay;
it even cites the cosine *shape* to SGDR, not to SAMed). #197 adds **metric-plateau early
stopping**, which SAMed never had — and a metric stop that fires before the horizon leaves a
horizon-calibrated schedule at a relatively high LR, forfeiting the endgame fine-tuning that
makes it accurate. `ReduceLROnPlateau` is horizon-agnostic: it decays in response to the
same stall that triggers the stop, so the low-LR probe always happens before the run quits.
That is exactly the literature's stated reason to combine early stopping with
reduce-on-plateau.

**Honest weakness of the plateau choice:** on a run that improves monotonically and never
stalls, `ReduceLROnPlateau` never decays and never anneals — there cosine is better. On a
160-epoch convergence run this is unlikely (it plateaus before the horizon, which is why
early stopping is wanted at all), and the `min_lr` floor plus the epoch cap bound the
downside.

## §5 — Early-stopping siblings (Prechelt 1998)

[Prechelt, *Early Stopping — But When?* (1998)](https://page.mi.fu-berlin.de/prechelt/Biblio/stop_tricks1997.pdf)
is the canonical patience reference. It defines generalization-loss (`GL_α`), the
training-progress quotient (`PQ_α`), and the pure-patience strip rule (`UP_s`, where a
"strip" is 5 epochs). Its central empirical result across 1296 runs: **slower stopping buys
only about 4% better generalization at about 4× the training cost** (up to 7× in one case).

Two consequences for #197:

- The priority here is accuracy ≫ speed, so a **generous** `stop_patience` is justified
  (Prechelt: more patience does buy generalization, just at a compute cost the project
  accepts).
- Framework defaults are too aggressive: [Keras `EarlyStopping`](https://keras.io/api/callbacks/early_stopping/)
  ships `patience=0`, `min_delta=0`. Practitioner guidance raises these to roughly `5`–`10`
  and `0.001`–`0.01`. The [HF `EarlyStoppingCallback`](https://huggingface.co/docs/transformers/main_classes/callback)
  counts evaluation calls (in `on_evaluate`), confirming the eval-count unit.

## §6 — "Other decay methods" — deferred with rationale

[SGDR (Loshchilov & Hutter 2017, arXiv:1608.03983)](https://arxiv.org/abs/1608.03983)
warm-restarts the cosine schedule on a **fixed epoch schedule** — it needs no validation set
and is positioned as a replacement for step decay (reaching comparable accuracy in 2–4×
fewer epochs). AdamWR is simply AdamW plus SGDR (a schedule on a schedule). Weight-decay
scheduling is likewise not plateau-triggered. None of these react to a validation plateau,
so they do not belong in a plateau-*response* rung; rung 1 is `LR × factor` only.

## §7 — Chosen defaults for #197

Every value is a documented citation or carries a `# tbd:` tag, per the repo's
cite-new-hyperparams rule. Cross-reference: `config/schema.py:TrainHyperparams` and
`docs/defaults-provenance.md`.

| Knob | Value | Basis |
| --- | --- | --- |
| `lr_schedule` default | `plateau` | flip from `cosine`; `ReduceLROnPlateau` (PyTorch/Keras) + the canonical early-stop pairing + the §4 horizon-mismatch argument; `# tbd: #197` for the default flip |
| `lr_decay_on_plateau.factor` | `0.1` | `# cite:` PyTorch `ReduceLROnPlateau` default `0.1` |
| `lr_decay_on_plateau.patience` | `5` evals | `# cite:` Keras `ReduceLROnPlateau` example `5` (low end of cited `5`–`10`) |
| `lr_decay_on_plateau.min_lr` | `1e-6` | `# cite:` PyTorch default `0`; `# tbd:` floored at `learning_rate / 100` to avoid a dead LR |
| `early_stop.enabled` | `true` | `# issue:` on by default |
| `early_stop.monitor` | `mAP` | existing best-metric key (`trainer.py`) |
| `early_stop.min_delta` | `0.001` | `# cite:` early-stop `min_delta` range `0.001`–`0.01` (Keras / practitioner); `# tbd:` low end for a noisy mAP |
| `early_stop.stop_patience` | `10` evals | `# cite:` patience `5`–`10` (PyTorch default `10`, Prechelt); `# tbd:` high end, accuracy ≫ speed |

**Resulting ladder** (base LR `1e-4`, one eval/epoch): five non-improving evals trigger one
10× cut to `1e-5`; five more non-improving evals halt at eval 10. A single deep endgame drop
with a low-LR probe window before the run gives up — standard reduce-then-stop. Respecting
both cited ranges (`patience ≥ 5`, `stop_patience ≤ 10`) yields exactly one cut before the
stop; a multi-step staircase would require pushing `stop_patience` past the cited range.

## §8 — Evidence gaps, caveats, and refuted claims

- **LoRA-specific plateau values are an evidence gap.** No canonical LoRA guide
  ([Unsloth](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide)
  included) publishes a plateau `factor`, `patience`, or `min_lr`. Unsloth's "1–3 epochs"
  guidance is for *generic* LoRA and does **not** describe this repo's SAMed-anchored
  convergence regime. The repo-chosen picks within cited ranges are therefore `# tbd:`.
- **The composition verdict rests on the HF enum** — suggestive, not an explicit ban (see
  [§3](#3--composition-verdict-the-key-question)).
- **Prechelt studied MLPs**, so the patience/cost tradeoff extends to LoRA by extrapolation.
- **Three claims were killed in verification** (majority-refuted), and are *not* relied on:
  - "Unsloth recommends only linear/cosine for LoRA, implying plateau decay is non-standard"
    (refuted 0–3).
  - "Unsloth's guide not addressing plateau decay is evidence it is absent from LoRA
    practice" (refuted 1–2).
  - "HF ships an `EarlyStoppingCallback` but no plateau-decay callback" (refuted 1–2 — HF
    *does* expose `reduce_lr_on_plateau` via `lr_scheduler_type`).

## Sources

- [PyTorch `ReduceLROnPlateau`](https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html) — primary
- [Keras `ReduceLROnPlateau`](https://keras.io/api/callbacks/reduce_lr_on_plateau/) — primary
- [Keras `EarlyStopping`](https://keras.io/api/callbacks/early_stopping/) — primary
- [Hugging Face optimizer schedules](https://huggingface.co/docs/transformers/en/main_classes/optimizer_schedules) — primary
- [Hugging Face Trainer callbacks](https://huggingface.co/docs/transformers/main_classes/callback) — primary
- [Prechelt, *Early Stopping — But When?* (1998)](https://page.mi.fu-berlin.de/prechelt/Biblio/stop_tricks1997.pdf) — primary
- [SGDR (Loshchilov & Hutter 2017, arXiv:1608.03983)](https://arxiv.org/abs/1608.03983) — primary
- [AdamW (Loshchilov & Hutter 2019, arXiv:1711.05101)](https://arxiv.org/abs/1711.05101) — primary
- [SAMed (Zhang & Liu 2023, arXiv:2304.13785)](https://arxiv.org/abs/2304.13785) and its [reference implementation](https://github.com/hitachinsk/SAMed) — primary
- [Unsloth LoRA hyperparameters guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide) — practitioner
- [MachineLearningMastery — early stopping](https://machinelearningmastery.com/how-to-stop-training-deep-neural-networks-at-the-right-time-using-early-stopping/) — blog
