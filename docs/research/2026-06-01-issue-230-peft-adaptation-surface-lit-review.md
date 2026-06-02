# PEFT Adaptation Surface for Niche Classes — Layer-Level Literature Review

> Research write-up for [issue #230](https://github.com/NguyenJus/custom-sam-peft/issues/230)
> Date: 2026-06-01
> Method: two manual literature passes (SAM PEFT family, open-vocabulary/concept
> adaptation, LoRA layer-selection and rank-vs-data) cross-checked against the
> installed SAM 3.1 source (`sam3/model/{decoder,necks,vitdet,vl_combiner}.py`),
> the repo's `lora.py`/`qlora.py`/`schema.py`, and the installed `peft` 0.19.1 API.

## TL;DR

- **The issue is framed as "add more `nn.Linear` patterns to `SCOPE_TARGETS`," but the
  two stated goals decompose into two *different* surfaces.** Learning **niche text
  concepts** lives in the decoder's **text cross-attention (`ca_text`)** and the
  vision–language alignment; segmenting **small objects** lives in **multi-scale /
  fine-detail** machinery (encoder local priors, the neck, the mask-decoder output
  path). "More coverage" is not the same as "the right coverage."
- **Highest-leverage, lowest-cost gap (confirmed): the text-mixing weights are
  currently frozen and unreachable by the shipped scope.** `ca_text` is a raw
  `nn.MultiheadAttention`; its q/k/v live in `in_proj_weight` (a bare `nn.Parameter`).
  Today's `vision_decoder` scope adapts only `ca_text.out_proj` — the *output* of text
  cross-attention, not the projections that inject text into the query stream. Open-vocab
  literature finds that fine-tuning the text↔image attention is precisely what teaches new
  concepts.
- **The issue's "out of scope: in_proj re-architecting" is based on an outdated
  constraint.** `peft` 0.19.1 ships a `target_parameters` API built specifically to
  LoRA-adapt bare `nn.Parameter`s like `nn.MultiheadAttention.in_proj_weight`. Reaching
  `ca_text` q/k/v is a **config-level** change, not a module rewrite — and it stays "true
  to SAM" (no new layers). It does carry known `nn.MultiheadAttention` sharp edges
  (§4) that warrant a validation spike, not a blind flip.
- **Small objects are mostly *not* a "more LoRA" problem.** The canonical fixes are
  architectural and multi-scale (HQ-SAM early+late feature fusion; Conv-LoRA conv-experts),
  and the biggest levers — input resolution and feature-pyramid fusion — sit **outside
  PEFT entirely**. The repo's neck is **`nn.Conv2d`, not `nn.Linear`**, so it is invisible
  to the current `nn.Linear`-only matcher.
- **"Adapt everything" fights the project's #1 priority on small-data users.** Encoder
  MLP/FFN is ≈⅔ of transformer parameters; adding it raises capacity *and* overfitting
  risk. Literature guidance is to keep rank/surface conservative on small data
  (r≈8–16) and widen only with data. The existing >10 % trainable-ratio warning is a real
  signal.
- **Chosen direction (per maintainer, 2026-06-01): stay true to SAM, no new layers,
  adapt low-parameter surfaces first, medium later.** This phase = pure LoRA on SAM's
  existing layers, closing the `ca_text` in_proj gap; trunk MLP/FFN and the Conv2d neck
  become a documented **future "medium" tier** ([§7](#7--chosen-direction--tiering)).

## §1 — Method and scope

The question: *what is the right PEFT adaptation surface to support a fixed set of niche
classes — prompted by text — on instance and semantic segmentation, given a priority of
final accuracy > simplicity >> speed, robust from small to medium datasets?* The shipped
default (`scope="vision_decoder"`) adapts a narrow slice; the maintainer's hypothesis is
that it is insufficient for (1) learning niche text concepts and (2) finding small objects.

Two literature passes were run — the SAM PEFT family (SAMed, HQ-SAM, Conv-LoRA,
SAM-Adapter), open-vocabulary/concept adaptation (text cross-attention, prompt tuning), and
LoRA layer-selection + rank-vs-data — then cross-checked against the actual SAM 3.1 module
graph and the repo's adapter code so every "adapt layer X" claim is grounded in a real
module name and parameter type.

## §2 — Two goals, two surfaces

SAM 3 is a DETR-style detector conditioned on **text / geometric / exemplar** prompts,
with a vision encoder + a ~300 M-parameter text encoder aligned by a Perception Encoder
([SAM 3, arXiv:2511.16719](https://arxiv.org/html/2511.16719v1)). For **text-prompted**
niche classes, the concept signal flows text → vision through the decoder's text
cross-attention. That makes the two goals target different parts of the graph:

- **Niche text concepts** → text↔vision fusion: the decoder `ca_text` block and the VL
  alignment. Empirically, "fine-tuning text-to-image attention layers significantly
  improves" few-shot concept segmentation
  ([The Power of One, arXiv:2503.10779](https://arxiv.org/pdf/2503.10779); see also
  [Adaptive Prompt Tuning, arXiv:2412.14640](https://arxiv.org/html/2412.14640v2)).
- **Small objects** → multi-scale / fine detail: HQ-SAM attributes SAM's thin-structure
  and boundary failures to the decoder, fixing them with an HQ output token + a
  **global-local fusion of early and late ViT features** at <0.5 % added params
  ([HQ-SAM, arXiv:2306.01567](https://arxiv.org/pdf/2306.01567)). It also reports that
  naive fine-tuning does *not* resolve fine-structure errors — a structural limit, not a
  capacity one.

## §3 — Layer map

Every candidate surface, grounded in the installed SAM 3.1 source and the repo matcher:

| Surface | In repo as | Goal | Evidence | Cost / risk |
| --- | --- | --- | --- | --- |
| ViT trunk `blocks.N.attn.{qkv,proj}` | `nn.Linear` (**adapted**) | Domain shift (both) | SAMed: encoder q,v is the key lever, best results | Low — keep |
| Decoder `ca_text` **in_proj** (q/k/v) | `nn.MultiheadAttention.in_proj_weight` — bare Param (**frozen**) | **Niche text concepts** | Text↔image attention tuning teaches concepts | **Low — top gap**, reachable via `target_parameters` |
| Decoder `{ca_text,cross_attn,self_attn}.out_proj` | `nn.Linear` (**adapted**) | Concept + instance separation | Complements in_proj | Low — keep |
| Decoder FFN `layers.N.linear[12]` | `nn.Linear` (**adapted**) | Query reshaping | — | Low — keep |
| ViT trunk MLP/FFN `blocks.N.mlp.{fc1,fc2}` | `nn.Linear` (**frozen**) | Niche appearance | QLoRA: adding MLP raises accuracy; orig. LoRA froze it | **Medium/high** — FFN ≈ ⅔ of params, overfit risk |
| Neck (multi-scale) | **`nn.Conv2d`** (**frozen**) | **Small objects** | Multi-scale fusion is the small-object axis | Medium — needs LoRA-on-Conv2d; `nn.Linear`-only matcher misses it |
| Mask-decoder / output head | `nn.Linear`/`Conv` (**frozen**) | Small-object mask quality | SAMed + Conv-LoRA **full-FT** the small decoder | Low module, but **full-FT mechanism**, not a scope pattern |
| Conv-LoRA MoE conv-experts | n/a (**new module**) | Small objects | Multi-scale local prior for object-size variation | New layer — **excluded** ("stay true to SAM") |
| Concept / text prompt tokens (VPT) | n/a (**new module**) | Niche concepts, low-data-safe | VPT competitive at <1 % params | New layer — **excluded** |

## §4 — The `ca_text` in_proj finding (the central result)

Confirmed in `sam3/model/decoder.py` (`TransformerDecoderLayer`): `self.ca_text =
nn.MultiheadAttention(...)` (line 54), `self.self_attn = nn.MultiheadAttention(...)`
(line 59). PyTorch's `nn.MultiheadAttention` packs q/k/v into a single `in_proj_weight`
`nn.Parameter` and exposes only `out_proj` as an `nn.Linear`. The shipped `vision_decoder`
scope matches `…ca_text.out_proj` only, so the **text-mixing q/k/v are frozen** — the model
can re-weight the *output* of text attention but cannot re-learn *how* text attends to image
features. For text-prompted niche concepts that is the wrong half to leave frozen.

**Only two of the three decoder attentions are `nn.MHA`.** `ca_text` and `self_attn` are
`nn.MultiheadAttention` (bare `in_proj_weight` → need `target_parameters`). The image
cross-attention `self.cross_attn` (line 47) is a **`RoPEAttention`** (from
`sam3.sam.transformer`), *not* MHA — its q/k/v are its own `nn.Linear` projections, reachable
by a `target_modules` **regex**, a different mechanism. By their DETR roles
([DETR](https://medium.com/@vishwajeethogale307/detr-end-to-end-object-detection-with-transformers-e1e4bf19a20a),
[DAC-DETR](https://proceedings.neurips.cc/paper_files/paper/2023/file/edd0d433f8a1a51aa11237a6543fc280-Paper-Conference.pdf)):
`self_attn` (query↔query) does **duplicate removal / inter-object reasoning** — the NMS-free
instance-separation mechanism (→ instance segmentation); image `cross_attn` does
**localization** ("queries lock onto regions") — a small-object lever; `ca_text` does
**concept** injection. Open-vocab PEFT work also warns that fine-tuning *all* parameters
harms open-vocab generalization, favouring a minimal attention surface
([Lightweight Modular PEFT for OVD, arXiv:2408.10787](https://arxiv.org/pdf/2408.10787)).
**Decision (2026-06-01): this phase adapts `ca_text` + `self_attn` in_proj** (concept +
instance separation, one `target_parameters` mechanism). Image `cross_attn` q/k/v
(RoPEAttention, `target_modules`) is the small-object/localization lever and moves to the
future medium tier.

**Feasibility:** `peft` 0.19.1 exposes both `target_modules` and `target_parameters` on
`LoraConfig` (verified by introspection). `target_parameters` exists specifically to adapt
bare parameters such as `nn.MultiheadAttention.in_proj_weight`
([peft LoRA docs](https://huggingface.co/docs/peft/en/package_reference/lora)). So the gap
closes at config level — no module rewrite, no new layers.

**Known sharp edges (→ validation spike, not a blind flip):**

- PyTorch `nn.MultiheadAttention.forward` calls `F.linear(x, in_proj_weight, …)` directly
  and accesses `self.out_proj.weight` rather than dispatching through a child module's
  `forward`. peft therefore needs special handling for MHA, and integrations have hit
  `AttributeError`/merge-path issues. Targeting in_proj must be tested for *forward
  correctness and merge* on a real SAM 3.1 decoder layer.
- **QLoRA interaction (already documented in `qlora.py`):** `_mha_exclusion_types()`
  deliberately keeps `nn.MultiheadAttention` (and `sam3.model.model_misc.MultiheadAttention`)
  **unquantized**, because both implement `forward` via `F.linear(act, in_proj_weight, …)`
  and would bypass `Linear4bit`. Consequence: under QLoRA the in_proj path stays a raw
  bf16 Parameter, so a `target_parameters` LoRA on it is *plain* LoRA-on-bf16 even in QLoRA
  mode — consistent, but it must coexist in one `PeftModel` with the `Linear4bit`
  `target_modules` LoRA. Verify both adapters attach and merge together.
- There are **two** MHA classes in play (torch built-in in the decoder; a custom
  `sam3.model.model_misc.MultiheadAttention`). Any in_proj targeting must be scoped to the
  decoder's actual `ca_text`/`self_attn` instances and verified against both.

## §5 — Small-object reality

Pure LoRA on frozen-feature attention does not add resolution that is not there. The
literature levers, in order of evidence:

- **Feature fusion / HQ token (HQ-SAM):** reuse early + late ViT features for fine
  boundaries and thin structures; architectural, in the decoder.
- **Multi-scale local priors (Conv-LoRA):** inject MoE convolutional experts into encoder
  LoRA, applied at the appropriate feature scale to handle object-size variation; freezes
  the prompt encoder, adds an MLP class branch, and **full-fine-tunes the mask decoder**
  ([Conv-LoRA, arXiv:2401.17868](https://arxiv.org/pdf/2401.17868)).
- **Neck / feature-pyramid adaptation:** the repo neck (`Sam3DualViTDetNeck` /
  `Sam3TriViTDetNeck`) is built from `nn.Conv2d`; adapting it means LoRA-on-Conv2d and
  extending the matcher's `linear_types`.
- **Non-PEFT, highest-impact, out of this scope:** input resolution and the
  feature-pyramid fusion itself. Worth stating plainly so small-object expectations of a
  LoRA-only change stay calibrated.

Conv-LoRA conv-experts and an HQ-style token are **new layers**, excluded by the "stay true
to SAM" constraint; they are recorded here as a future research track, not this issue's work.

## §6 — Rank / capacity vs. data scale

- Original **LoRA** adapted attention only, freezing MLP for parameter efficiency
  ([Hu 2021, arXiv:2106.09685](https://arxiv.org/abs/2106.09685)).
- **QLoRA** found adapting *all* linear layers (attention **and** MLP) matters for matching
  full fine-tuning ([Dettmers 2023, arXiv:2305.14314](https://arxiv.org/abs/2305.14314)).
  The two are reconciled by data scale: MLP capacity helps when there is data to support it.
- **Rank vs. data:** practitioner guidance converges on r≈8–16 for small/domain data,
  r≈32–64 for moderate, higher only for data-rich regimes; *reduce* r (and add dropout /
  early stopping) to curb overfitting on small sets
  ([Unsloth LoRA hyperparameters](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide),
  [Raschka, practical LoRA tips](https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms)).
- **SAMed** is the in-domain anchor: LoRA on encoder **q,v** + fine-tune the small mask
  decoder was sufficient and best; decoder LoRA was *optional*
  ([SAMed, arXiv:2304.13785](https://arxiv.org/pdf/2304.13785)).
- **Caution for specialized domains:** prompt/text tuning can *degrade* generalization on
  novel queries when trained only on base queries
  ([OpenDAS, arXiv:2405.20141](https://arxiv.org/pdf/2405.20141)) — relevant to the
  "robust across users/datasets" requirement.

## §7 — Chosen direction & tiering

Maintainer decision (2026-06-01): **stay true to SAM, add no new layers, adapt
low-parameter surfaces first, medium later.** Concretely:

**Now — low-parameter LoRA tier (this issue):**

- Keep existing `vision` / `vision_decoder` / `all` scopes intact for reproducibility (add
  one new scope literal rather than mutating shipped ones — the issue's reproducibility
  question).
- Close the top gap: reach decoder **`ca_text` + `self_attn` in_proj** (concept learning +
  instance separation; both genuine `nn.MHA`, low-parameter) via `target_parameters`, behind
  a validation spike (§4).
- The new scope **becomes the default** (replacing `vision_decoder`): text-concept learning
  is the stated primary goal and the shipped default cannot do it. Note the reproducibility
  change vs the `# tbd: #191` default.
- Keep rank/alpha conservative and tiered with a safe default; small-data users stay
  overfit-safe.

**Future — medium-parameter tier (separate issue):**

- ViT trunk MLP/FFN `blocks.N.mlp.{fc1,fc2}`.
- Image `cross_attn` (RoPEAttention) q/k/v — localization lever for small objects, via
  `target_modules` regex (a second mechanism).
- Conv2d neck adaptation (requires extending the matcher's `linear_types`).
- Optionally SAMed/Conv-LoRA-style full-fine-tune of the small mask-decoder head.

**Deferred / excluded (out of "stay true to SAM"):** Conv-LoRA conv-experts, VPT concept
tokens, text-encoder adaptation, and the non-PEFT resolution/feature-fusion levers.

### Chosen defaults (cite / tbd discipline)

| Knob | Value | Basis |
| --- | --- | --- |
| `r` | 16 (default) | `# cite:` LoRA (Hu 2021) §4.1; conservative for small/medium |
| `alpha` | 32 | `# cite:` LoRA (Hu 2021) §4.1 (`alpha = 2r`) |
| small-data `r` guidance | 8 | `# tbd:` within cited r≈8–16 small-data range (Unsloth / Raschka) |
| new scope literal name(s) | design choice | settled in spec; not a hyperparameter |

## §8 — Open implementation risks (for the spec/plan)

1. **`target_parameters` × `nn.MultiheadAttention` correctness** — forward + merge on a real
   SAM 3.1 decoder layer; this is the gating spike before committing the in_proj surface.
2. **QLoRA coexistence** — in_proj LoRA (bf16, via `target_parameters`) must attach and
   merge alongside `Linear4bit` LoRA (via `target_modules`) in one `PeftModel`.
3. **`PEFTConfig` shape** — there is no `target_parameters` field today; the scope→targets
   resolution in `_resolve_targets` handles only `target_modules`. A second axis is needed.
4. **Trainable-ratio guard** — adding in_proj is small, but confirm the >10 % warning still
   reflects reality once the surface changes.
5. **Stub fixtures** — `tests/fixtures/` must expose `ca_text`/in_proj so predicate tests
   resolve targets under both LoRA and QLoRA.

## Sources

- [SAMed — Customized SAM for Medical Image Segmentation (arXiv:2304.13785)](https://arxiv.org/pdf/2304.13785)
- [HQ-SAM — Segment Anything in High Quality (arXiv:2306.01567)](https://arxiv.org/pdf/2306.01567)
- [SAM 3 — Segment Anything with Concepts (arXiv:2511.16719)](https://arxiv.org/html/2511.16719v1)
- [Conv-LoRA — Convolution Meets LoRA (arXiv:2401.17868)](https://arxiv.org/pdf/2401.17868)
- [LoRA (Hu 2021, arXiv:2106.09685)](https://arxiv.org/abs/2106.09685)
- [QLoRA (Dettmers 2023, arXiv:2305.14314)](https://arxiv.org/abs/2305.14314)
- [Visual Prompt Tuning (arXiv:2203.12119)](https://arxiv.org/pdf/2203.12119)
- [The Power of One — single-example segmentation in VLMs (arXiv:2503.10779)](https://arxiv.org/pdf/2503.10779)
- [Adaptive Prompt Tuning (arXiv:2412.14640)](https://arxiv.org/html/2412.14640v2)
- [H-CLIP — Hyperspherical PEFT for Open-vocab Segmentation (arXiv:2405.18840)](https://arxiv.org/abs/2405.18840)
- [OpenDAS — Open-Vocabulary Domain Adaptation for Segmentation (arXiv:2405.20141)](https://arxiv.org/pdf/2405.20141)
- [peft LoRA API reference (`target_parameters`)](https://huggingface.co/docs/peft/en/package_reference/lora)
- [Unsloth — LoRA hyperparameters guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide)
- [Raschka — Practical tips for finetuning LLMs with LoRA](https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms)
