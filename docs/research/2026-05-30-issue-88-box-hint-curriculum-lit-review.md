# The `box_hint` Curriculum — Literature Review & Keep/Remove Recommendation

> Research write-up for [issue #88](https://github.com/NguyenJus/custom-sam-peft/issues/88)
> Date: 2026-05-30
> Spec: `docs/superpowers/specs/2026-05-30-issue-88-remove-box-hint-curriculum-design.md`
> Plan: `docs/superpowers/plans/2026-05-30-issue-88-remove-box-hint-curriculum-plan.md`

## TL;DR

**Recommendation: REMOVE the `box_hint` curriculum.**

The technique under review — feeding a *decaying* ground-truth bounding-box hint
alongside text prompts during training, annealed from `p_start=1.0` to
`p_end=0.0` over the first ~75% of the run, then training the final ~25%
purely text-only — has **no direct precedent** in the published literature. It
is a hybrid of curriculum learning (Bengio et al., 2009) and scheduled-sampling
-style annealing (Bengio et al., 2015): an auxiliary input weaned toward the
unaided inference regime.

The literature is consistent on the load-bearing point: a hint that is **removed
by the end of training** can, at most, change the **optimization path
(convergence speed)** — it **cannot change the endpoint** (the final converged
solution), because the model finishes training in exactly the no-hint regime it
will be deployed in. The project's own mechanics make this airtight: `box_hint`
decays to `p=0`, the box-term losses are `0.0`, and the hint is never present at
inference, so it is mathematically incapable of moving the final optimum.

The **closest published analog** — query/box denoising in DN-DETR (Li et al.,
2022) and DINO (Zhang et al., 2022) — actually argues *against* our design: those
methods get their convergence-and-accuracy gains by keeping the box-denoising
branch **active throughout the entire run** as a constant parallel auxiliary
task, the **opposite** of annealing it to zero. The promptable-model cluster
(SAM, GLIP, Grounding-DINO) uses boxes as **inference-time prompts or model
outputs**, never as a decayed training curriculum, so it offers no precedent
either.

Against the project's guiding principle — (1) endpoint accuracy, (2) user-facing
simplicity, (3) *far behind* training speed — the best case the literature
supports for `box_hint` is a **speed-only** benefit, and even that is unproven
for the decayed-to-zero variant. A speed-only benefit does not justify the
config surface a knob adds. **Remove it; retain the `SupportPrompts` seam.**

---

## §1 — Does the technique appear in published research, and under what name

The exact mechanism (a GT box hint fed alongside text prompts during training,
decayed to zero so the model is weaned to text-only inference) does **not** appear
as a named, studied technique. It is best understood as a *curriculum / continuation
annealing of an auxiliary input* — a composite of the five literatures below. Each
is a genuine relative, but each differs from our variant in a specific, important
way.

### Curriculum learning — Bengio et al. (2009)

Bengio, Louradour, Collobert & Weston (2009) formalized **curriculum learning**:
presenting training examples in a meaningful easy-to-hard order, framed explicitly
as a **continuation method** (start from a smoothed/easier objective, gradually
deform it into the true harder objective). The paper hypothesizes a *dual* effect:
a **convergence-speed** benefit (the path) and, **for non-convex objectives only**,
a possible **better-basin / better-generalization** benefit (the endpoint).
Critically, the authors note that **for convex problems curriculum cannot change
the optimum** — example order is irrelevant to a unique global minimum.

Our `box_hint` schedule *is* a curriculum: it orders training from "easy" (model
gets a localization crutch) to "hard" (text-only). But the endpoint side of the
curriculum claim is **empirically contested**. The analytical teacher–student
theory of Saglietti, Mannelli & Saxe (2022) finds a clear curriculum speed-up in
the *online (single-pass)* regime that **largely disappears once examples can be
stored and replayed** — i.e. in the standard multi-epoch regime our trainer uses.
The robust, reproducible curriculum benefit is on the **path**; the endpoint
benefit is fragile and regime-dependent.

### Scheduled sampling — Bengio et al. (2015), and Huszár's (2015) critique

Scheduled sampling (Bengio, Vinyals, Jaitly & Shazeer, 2015) is the closest
*train/inference-gap* analog. During RNN training it anneals from teacher forcing
(feed the true previous token) toward feeding the model's **own** prediction, so
the train-time conditioning converges to the inference-time conditioning. The
shared idea with `box_hint` is *annealing a ground-truth crutch toward zero so the
model is weaned to its unaided inference mode*.

The distinction matters for citing it accurately. Scheduled sampling **swaps the
source of an existing input** (true token → self-generated token); the model
always receives a previous-token input, only its provenance changes. `box_hint`
instead **anneals an auxiliary input out of existence** — the box hint is present
(decaying) at train time and fully *absent* at inference. So `box_hint` is closer
to *input-feature dropout / hint-decay* than to scheduled sampling's
self-conditioning.

Huszár (2015) is the essential cautionary citation. He shows scheduled sampling's
training objective is an **inconsistent estimator**: as the self-conditioning
fraction grows, the global optimum drifts toward a degenerate, context-ignoring
factorised solution rather than the true data distribution — so the annealing can
*worsen the endpoint*. **However**, this critique targets losses that condition on
**self-generated values**; it does **not** automatically transfer to our case,
where the prediction target and loss stay fixed to ground-truth masks via a proper
loss and only an auxiliary *input* decays. The honest reading: Huszár refutes the
naive assumption that "annealing a signal to zero is automatically harmless," but
it is not direct evidence *against* `box_hint`. We cite it to bound the analogy,
not to condemn the knob.

### Learning Using Privileged Information — Vapnik & Vashist (2009)

LUPI (Vapnik & Vashist, 2009; Vapnik & Izmailov, 2015) is the paradigm where extra
information `x*` is available **only at training time, never at test** — exactly
the structure of a train-only box hint absent at inference. The canonical
realization is SVM+. The decay-to-zero wrinkle is **not** part of LUPI, though:
LUPI keeps the privileged channel available throughout training and simply never
exposes it at test time.

The LUPI evidence reinforces the path-vs-endpoint split. Pechyony & Vapnik (2010)
prove a **sample-complexity (rate) advantage** — `O(1/n)` vs `O(1/√n)` in a
separable construction — i.e. *faster learning*, not a better asymptotic model.
And the endpoint benefit is theoretically contested: Sharoni & Sabato (2023) show
worst-case guarantees for privileged ERM **cannot improve over standard ERM**
unless the privileged information's capacity is similar to or smaller than the
ordinary features'. LUPI's defensible claim is **efficiency / path**, not endpoint.

### Box-denoising in DETR-family detectors — DN-DETR and DINO (the closest analog)

DN-DETR (Li et al., 2022) and DINO (Zhang et al., 2022) are the **closest
published analog** to "feed GT boxes during training, keep inference box-free."
DN-DETR feeds **noised GT boxes** as auxiliary denoising queries that the model
learns to reconstruct, stabilizing bipartite matching; DINO adds **contrastive
denoising** (positive + negative noised queries). Both **remove the denoising
branch only at inference**.

The decisive contrast: **both keep the box-denoising branch active throughout the
entire training run** as a constant parallel auxiliary task. **Neither anneals or
decays the box signal to zero.** Our `box_hint` does the **opposite** — it weans
the model *off* the hint over training. And the benefit framing is instructive:
DN-DETR is titled "*Accelerate DETR Training*" (primarily a **convergence** claim,
with a secondary +1.9 AP); DINO reports both faster convergence and SOTA accuracy.
Whatever endpoint accuracy these methods gain is attributable to the **persistent**
auxiliary task they retain — precisely the design choice `box_hint`'s decay
discards. So the strongest precedent for box-during-training argues for *keeping a
constant branch*, not for an *annealed-to-zero curriculum*; and if our pipeline is
unwilling to carry a permanent box branch (it isn't — box losses are `0.0` and
inference is text-only), the denoising literature offers `box_hint` no endpoint
support.

### Box-as-prompt in promptable / open-vocab models — SAM, GLIP, Grounding-DINO

In the promptable / open-vocabulary cluster, a box is either an
**inference-time prompt the user supplies** or a **model output / grounding
target** — never a decayed training-only signal:

- **SAM / SAM 2 / SAM 3** (Kirillov et al., 2023; Ravi et al., 2024; Meta FAIR,
  2025): boxes/points/masks are **interactive inference-time prompts**. During
  training SAM *simulates* that interaction by sampling prompts from GT masks; a
  box prompt is **jittered** (noise std = 10% of box side length, capped at 20 px)
  to harden a **permanent** test-time interface — it is **not** annealed to zero.
  This is the opposite of a decay curriculum. (SAM 3 is this project's base model
  and explicitly keeps boxes as an inference-time prompt and as presence-scored
  output candidates.)
- **GLIP** (Li et al., 2022): unifies detection and phrase grounding; the **text is
  the prompt** and **boxes are outputs / grounding targets** (it even bootstraps
  24M grounding boxes via self-training). Never a decayed input hint.
- **Grounding-DINO** (Liu et al., 2023): takes an (image, text) pair, **text is the
  prompt**, and the model **outputs** ~900 candidate boxes scored against the words.
  Boxes are outputs, not a training crutch.

None of these is a precedent for a decayed box-hint curriculum.

---

## §2 — Endpoint vs. path

Across all five literatures the verdict is uniform: a hint that is **removed by the
end of training** governs the **optimization path (convergence speed / stability)**,
not the **endpoint (final converged quality)**.

- **Curriculum learning** can in principle affect the endpoint only for non-convex
  objectives, and that effect is **empirically fragile** — it largely vanishes in
  the multi-epoch replay regime this trainer uses (Saglietti et al., 2022).
- **LUPI** buys a **rate / sample-efficiency** advantage (Pechyony & Vapnik, 2010);
  its endpoint guarantee is contested and capacity-bounded (Sharoni & Sabato, 2023).
- **Scheduled sampling**, the one annealing method *pitched* as an endpoint fix
  (exposure bias), is exactly the one Huszár (2015) shows can **degrade** the
  endpoint — annealing is not automatically benign.
- **DN-DETR / DINO** obtain endpoint gains **only by keeping the box branch
  throughout training**; the moment you anneal the box signal to zero you forgo the
  mechanism that produced those gains.

For `box_hint` specifically the argument is not merely empirical — it is
**mechanical and airtight**. The schedule reaches `p=0` over the first ~75% of the
run, the final ~25% trains pure text-only, the box-term losses are `0.0`, and the
hint is absent at inference. The model therefore **finishes optimization in
precisely the no-hint regime it is deployed in**, so `box_hint` **cannot move the
final optimum**. Its entire possible effect is confined to the path.

---

## §3 — Evidence of benefit

**Is there evidence the decayed-to-zero variant accelerates convergence or improves
the text-only endpoint? No direct evidence for either, and a structural reason to
doubt the endpoint claim.**

- **Endpoint:** ruled out mechanically for `box_hint` (§2). No surveyed work shows a
  *removed-by-end* auxiliary hint improving the final converged solution; the
  techniques that do report endpoint gains (DN-DETR, DINO) **retain** the auxiliary
  branch rather than annealing it, and the curriculum/LUPI endpoint claims are
  contested (Saglietti et al., 2022; Sharoni & Sabato, 2023).
- **Path (speed):** this is where the analogs locate their benefit — curriculum
  speed-ups (online regime), LUPI's faster learning rate, DN-DETR/DINO's faster
  convergence. But every one of those is for a **differently-scheduled** mechanism
  (ordered examples; persistent privileged channel; persistent denoising branch),
  **not** for an annealed-to-zero box hint. There is **no published measurement** of
  the decayed-box-hint variant's convergence benefit, and the project has **no
  empirical sweep** of its own (dropped as infeasible on the available single
  RTX 5070 Ti — spec §1.1). The speed benefit is therefore **unproven** here, not
  merely secondary.

In short: the only category of benefit the literature even plausibly supports for
this design is convergence speed, and for the specific decayed-to-zero variant that
benefit is **unmeasured and unverified**.

---

## §4 — Recommendation

**Remove the `box_hint` curriculum.** Reasoned against the guiding principle
(endpoint accuracy ≫ user-facing simplicity ≫ training speed):

1. **Endpoint (priority 1): no benefit, provably.** `box_hint` decays to `p=0`,
   box losses are `0.0`, and the final ~25% of every run is already pure text-only,
   so it cannot change the final optimum. The literature offers no counter-example
   of a removed-by-end hint improving the endpoint; the methods that do (DN-DETR,
   DINO) keep the branch throughout, which `box_hint` does not.
2. **Simplicity (priority 2): removal is a clear win.** `box_hint` adds a
   three-field `BoxHintSchedule` (`p_start`, `p_end`, `decay_steps`) to the config
   surface plus plumbing across schema, train loop, checkpoint, trainer, and model.
   Deleting it shrinks the user-facing config and the code the user must understand.
3. **Speed (priority 3, far behind): at best a speed-only, unproven benefit.** The
   only benefit the literature plausibly supports is faster convergence, and for the
   decayed-to-zero variant it is unmeasured. Per the guiding principle, a speed-only
   benefit does **not** justify the config surface a knob adds — and an *unproven*
   speed-only benefit justifies it even less.

The retained `SupportPrompts` extension seam (#126 §12) preserves the only durable
value in the design — a home for future mask/point hints — at zero config cost.
This recommendation is the spec's **primary path** (spec §5.4); the surveyed
literature surfaces no strong endpoint-or-simplicity benefit that would trigger the
"keep + cite" surprise branch.

---

## References

- Bengio, Y., Louradour, J., Collobert, R., & Weston, J. (2009). Curriculum
  Learning. *Proceedings of the 26th International Conference on Machine Learning
  (ICML '09)*, 41–48. DOI:
  [10.1145/1553374.1553380](https://dl.acm.org/doi/10.1145/1553374.1553380).
- Bengio, S., Vinyals, O., Jaitly, N., & Shazeer, N. (2015). Scheduled Sampling for
  Sequence Prediction with Recurrent Neural Networks. *Advances in Neural
  Information Processing Systems 28 (NeurIPS 2015)*. arXiv:
  [1506.03099](https://arxiv.org/abs/1506.03099).
- Huszár, F. (2015). How (not) to Train your Generative Model: Scheduled Sampling,
  Likelihood, Adversary? arXiv:
  [1511.05101](https://arxiv.org/abs/1511.05101).
- Saglietti, L., Mannelli, S. S., & Saxe, A. (2022). An Analytical Theory of
  Curriculum Learning. *NeurIPS 2022* (also *PNAS* 2023). PMC:
  [PMC10561397](https://pmc.ncbi.nlm.nih.gov/articles/PMC10561397/).
- Vapnik, V., & Vashist, A. (2009). A New Learning Paradigm: Learning Using
  Privileged Information. *Neural Networks*, 22(5–6), 544–557. DOI:
  [10.1016/j.neunet.2009.06.042](https://dl.acm.org/doi/10.1016/j.neunet.2009.06.042).
- Vapnik, V., & Izmailov, R. (2015). Learning Using Privileged Information:
  Similarity Control and Knowledge Transfer. *Journal of Machine Learning Research*,
  16(61), 2023–2049. [jmlr.org/papers/v16/vapnik15b.html](https://jmlr.org/papers/v16/vapnik15b.html).
- Pechyony, D., & Vapnik, V. (2010). On the Theory of Learning with Privileged
  Information. *Advances in Neural Information Processing Systems 23 (NeurIPS 2010)*.
- Sharoni, O., & Sabato, S. (2023). On the Capacity Limits of Privileged ERM.
  *AISTATS 2023*. arXiv:
  [2303.02658](https://arxiv.org/abs/2303.02658).
- Li, F., Zhang, H., Liu, S., Guo, J., Ni, L. M., & Zhang, L. (2022). DN-DETR:
  Accelerate DETR Training by Introducing Query DeNoising. *CVPR 2022*. arXiv:
  [2203.01305](https://arxiv.org/abs/2203.01305).
- Zhang, H., Li, F., Liu, S., Zhang, L., Su, H., Zhu, J., Ni, L. M., & Shum, H.-Y.
  (2022). DINO: DETR with Improved DeNoising Anchor Boxes for End-to-End Object
  Detection. arXiv:
  [2203.03605](https://arxiv.org/abs/2203.03605) (*ICLR 2023*).
- Kirillov, A., Mintun, E., Ravi, N., Mao, H., Rolland, C., Gustafson, L., Xiao, T.,
  Whitehead, S., Berg, A. C., Lo, W.-Y., Dollár, P., & Girshick, R. (2023). Segment
  Anything. *ICCV 2023*. arXiv:
  [2304.02643](https://arxiv.org/abs/2304.02643).
- Ravi, N., Gabeur, V., Hu, Y.-T., et al. (2024). SAM 2: Segment Anything in Images
  and Videos. arXiv:
  [2408.00714](https://arxiv.org/abs/2408.00714).
- Meta FAIR (2025). SAM 3: Segment Anything with Concepts. arXiv:
  [2511.16719](https://arxiv.org/abs/2511.16719).
- Li, L. H., Zhang, P., Zhang, H., Yang, J., Li, C., Zhong, Y., Wang, L., Yuan, L.,
  Zhang, L., Hwang, J.-N., Chang, K.-W., & Gao, J. (2022). Grounded Language-Image
  Pre-training (GLIP). *CVPR 2022*. arXiv:
  [2112.03857](https://arxiv.org/abs/2112.03857).
- Liu, S., Zeng, Z., Ren, T., Li, F., Zhang, H., Yang, J., Jiang, Q., Li, C., Yang,
  J., Su, H., Zhu, J., & Zhang, L. (2023). Grounding DINO: Marrying DINO with
  Grounded Pre-Training for Open-Set Object Detection. arXiv:
  [2303.05499](https://arxiv.org/abs/2303.05499) (*ECCV 2024*).
