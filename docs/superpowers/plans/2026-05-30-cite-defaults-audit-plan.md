# Literature-Cite Every Default Hyperparameter (Issue #120) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Attach a verified, concise provenance tag to every trust-bearing default across seven in-scope files, build the central `docs/defaults-provenance.md` index, firm up the loss-preset `# citation needed` cells, and align the shipped `epochs` default to an analytically-derived reference training profile.

**Architecture:** Two coupled deliverables meeting at one interface seam. Phase 1 establishes the provenance row schema, files the three follow-up issues (so `# tbd:` tags can reference a real umbrella number), then audits each file — producing inline tags **and** matching doc rows together, with the deep-verification quote/DOI captured at tag time. Phase 2 consumes the `epochs` doc row stub from Phase 1, writes the Reference Training Profile section, sets the template `$epochs` default, and adds the CI-truncation comment to the integration tests. This is documentation-and-provenance work; no numeric default changes except the sanctioned `epochs` alignment.

**Tech Stack:** Python 3.12 (Pydantic v2 schema, dataclasses), Markdown (`docs/defaults-provenance.md`), `markdownlint-cli2`, pytest with a `--cov-fail-under=80` gate, `gh` CLI for follow-up issues, web research for primary-source verification.

---

## Orientation (read before any task)

These facts are verified against the audit branch tree. Trust them; do not re-derive.

**In-scope files (Deliverable 1):**

- `src/custom_sam_peft/config/_internal.py` — `MatcherWeights`, `WandbConfig`, `ExportConfig`.
- `src/custom_sam_peft/config/schema.py` — ~60 user-facing Pydantic `Field` defaults.
- `src/custom_sam_peft/data/aug_presets.py` — 12-cell `PRESET_TABLE` (4 presets × 3 intensities, 8 knobs/cell). `# noqa` already exempts `E501`/`RUF003` for this file (pyproject `per-file-ignores`).
- `src/custom_sam_peft/data/channel_semantics.py` — `_IMAGENET_MEAN`/`_IMAGENET_STD`, four `CHANNEL_SEMANTICS` profiles.
- `src/custom_sam_peft/data/transforms.py` — `KNOWN_PROCESSOR_STATS`, `_STATS_DIVERGENCE_ATOL`, `_HED_FROM_RGB_MATRIX`, `_GAUSS_NOISE_MAX_VAR`, `_GAUSS_BLUR_MAX_SIGMA`.
- `src/custom_sam_peft/presets.py` — VRAM-model constants (`MODEL_PARAMS`, `LORA_LAYERS`, `D_IN`, `D_OUT`, `Q_OVERHEAD`, `WORKSPACE_BYTES`, `BASE_ACTIVATION_AT_1024`, `forward_only_factor`, `_SAM3_PATCH`, `_SAM3_HEADS`, `_bytes_per_param_for_method`, `_optimizer_bytes` `*4`, `CACHE_SCHEMA_VERSION`).
- `src/custom_sam_peft/cli/templates/config_full.yaml` — user-facing template; echoes literal defaults and carries the `$epochs` slot.
- FIRM-UP only: `src/custom_sam_peft/models/losses/presets.py` — 9 `# citation needed` cells + 1 docstring mention (line 25). `RUF002`/`RUF003` already exempted for this file.

**Out of scope (index row only, no inline tag) — do not tag inline:** `HFFieldMap.*` field-name maps, `output_dir="./runs"`, `model.name="facebook/sam3.1"`, `local_dir`, `checkpoint_file`, `bbox_format="xyxy"`, `split_train="train"`, `TrackingConfig.backend="tensorboard"`, `WandbConfig.project/entity`, `ExportConfig.merge`, `format` literals. `configs/examples/*.yaml` are entirely out of scope. **No new defaults. No re-tuning.**

**Markdownlint reality (verified):** `.config/markdownlint-cli2.jsonc` sets `MD013: false` (line-length OFF), `MD018: false`, `MD029: false`. Wide provenance tables are fine — no per-block disable needed. CI command (ci.yml line 99):
`npx --yes markdownlint-cli2 --config .config/markdownlint-cli2.jsonc "**/*.md" "#node_modules"`.

**Coverage gate (verified):** `pyproject.toml` `addopts` includes `--cov-fail-under=80`. This plan adds no source code, so it cannot lower coverage; but when running a *subset* of tests, bypass the gate with `pytest -o "addopts="` (memory: `--no-cov` does not work here).

**No provenance doc exists yet.** `docs/defaults-provenance.md` is a new file. The only file currently carrying `# cite:` tags is `models/losses/presets.py`.

**Tag taxonomy (spec §1.1) — the six allowed inline tag classes:**

| Tag form | Meaning |
| --- | --- |
| `# cite: <paper> §/Eq.N` | Value backed by a published paper at an exact location. |
| `# cite: <repo>:<path>` | Value mirrors a reference implementation (file/line). |
| `# cite: empirical (<dataset>, run <id>)` | Value from an internal calibration/training run. |
| `# cite: degenerate-case` | Value is a math identity / limiting case. |
| `# cite: framework default` | Value is the upstream library default (Albumentations / torch / HF). |
| `# tbd: #<sub-issue>` | No external source AND no internal run; tracked by the umbrella sub-issue. |

**Doc row schema (spec §1.2) — frozen contract, every row has exactly these six columns:**

`| Location | Value | Tag | Full reference | Verifying quote | Notes |`

- **Location:** `file:symbol` (e.g. `config/schema.py:TrainHyperparams.learning_rate`).
- **Value:** the literal default.
- **Tag:** the inline tag class applied (mirrors the code), or `index-only` for untagged structural defaults.
- **Full reference:** authors, year, arXiv/DOI, exact Eq./Table/Fig (or upstream-doc URL + pinned version for framework defaults; or repo file/line for reference-impl).
- **Verifying quote:** a short quote from the primary source establishing the value (empty only for `degenerate-case`/`index-only`, which instead state the identity/rationale in Notes).
- **Notes:** caveats, degenerate-case identities, calibration run pointers, cross-links.

**Deep-verification rule (spec §1.5) — apply per value, at tag time:**

- `# cite: <paper>` → capture quote + DOI/arXiv + exact Eq./Table/Fig.
- `# cite: framework default` → confirm it truly is the Albumentations/torch/HF default; link the upstream doc and pin the version/commit observed.
- `# cite: degenerate-case` → state the math identity in Notes.
- `# cite: <repo>:<path>` → cite the specific reference-impl file/line.
- No external source AND no internal run → `# tbd: #<umbrella>`. **Never fabricate.**

---

## Phase 1 — Provenance Audit + Central Index (Deliverable 1)

**Feature block:** the full provenance audit. Files the three follow-up issues, freezes the doc row schema, then audits all seven in-scope files plus the loss-preset firm-up, producing inline tags and matching doc rows together with verification captured at tag time.

**What this phase EXPOSES to Phase 2 (interface contract — Phase 2 consumes only these, without re-reading Phase 1 code):**

1. **The frozen doc row schema** (the six-column table above), established in Task 1.2 as `docs/defaults-provenance.md`'s row format. Every later task in both phases emits rows in exactly this shape.
2. **The `epochs` row stub** in the `config/schema.py` section of `docs/defaults-provenance.md`, written in Task 1.4. Its Location is `config/schema.py:TrainHyperparams.epochs` (required field, **no schema default**); its Value cell reads `required (template `$epochs` slot)`; its Tag is `# cite: empirical`; its Full reference / Notes cells contain the literal text `See "Reference Training Profile" section below (Deliverable 2).` Phase 2 owns that section and the numeric value; Phase 1 owns this linking row. This is the §1.3 seam.
3. **The umbrella issue number** (from Task 1.1), used by every `# tbd: #<umbrella>` tag in Phase 1 AND by the `epochs` empirical-confirmation `# tbd:` in Phase 2. Recorded verbatim at the top of the `docs/defaults-provenance.md` Verification Standard section as `Umbrella # tbd: tracker: #<N>`.
4. **A reserved, empty `## Reference Training Profile` heading** at the end of `docs/defaults-provenance.md` (added in Task 1.2), so Phase 2 appends its content under a known anchor without restructuring the file.

### Task 1.1: File the three follow-up issues (umbrella FIRST)

**Files:** none (GitHub only). This task MUST run before any `# tbd:` tag is written, because tags reference the umbrella issue number.

> **Follow-up issue inventory (recorded numbers — already created; do not re-issue `gh`).** The full inventory is now FOUR issues: **#191** (umbrella `# tbd:` tracker — the `UMBRELLA` number used by every `# tbd:` tag), **#192** (CI no-uncited-default hook), **#193** (empirical T4 confirmation of the reference profile — the `T4_CONFIRM` number used by Phase 2; now reframed as runtime confirmation for the convergence-anchored 160-epoch profile), and **#195** (empirically measure the 2-image overfit GPU smoke-test speed/convergence on `tiny_coco` — added during Phase-2 prep; the empirical home for the GPU-test budget questions). Steps 1–5 below describe how the first three were originally filed; `#195` was created separately during the Phase-2 amendment.

- [ ] **Step 1: Confirm `gh` auth and labels**

Run: `gh auth status && gh label list`
Expected: authenticated; labels include `docs`, `testing`, `enhancement`, `priority:low`. (Verified present at planning time.)

- [ ] **Step 2: Create the umbrella `# tbd:` tracker issue FIRST and capture its number**

```bash
gh issue create \
  --title "Umbrella: source-or-retune every default that resolved to # tbd: (from #120)" \
  --body $'Tracks every default in the #120 provenance audit that resolved to \x60# tbd:\x60 — no published source and no recorded internal run. Each \x60# tbd: #<this-issue>\x60 inline tag and its docs/defaults-provenance.md row points here.\n\nKnown candidates (verify during the audit; some may upgrade to a real cite or # cite: empirical):\n- models/losses/presets.py: focal_gamma 2.5 / 3.0 escalation above Lin et al. gamma=2.0\n- models/losses/presets.py: tversky_alpha=0.8 (no paper value)\n- config/schema.py: learning_rate=1e-4, warmup_steps=100, batch_size=1, grad_accum_steps=8 (repo engineering choices)\n- data/aug_presets.py: domain-tuned magnitudes (rotate_arbitrary, stain_jitter, gauss_noise, blur)\n- data/transforms.py: _GAUSS_NOISE_MAX_VAR=0.05, _GAUSS_BLUR_MAX_SIGMA=3.0, _STATS_DIVERGENCE_ATOL=1e-3\n\nResolution: source each value or re-tune it in a future pass.\n\nParent: #120' \
  --label docs --label priority:low --assignee @me
```

Run the command, then capture the number it prints:
`gh issue list --search "Umbrella: source-or-retune" --state open --json number,title -q '.[0].number'`
Record this number as `UMBRELLA` (e.g. `184`). **You will use `#<UMBRELLA>` in every `# tbd:` tag and in the doc's Verification Standard section.**

- [ ] **Step 3: Create the CI "no-uncited-default" enforcement-hook issue**

```bash
gh issue create \
  --title "CI: enforce no-uncited-default hook (flag new trust-bearing defaults lacking a tag + doc row)" \
  --body $'Follow-up from #120 (explicitly deferred by that issue\x27s out-of-scope list).\n\nAdd a lint/CI check that flags any new trust-bearing default added to an in-scope file that lacks BOTH an inline \x60# cite:\x60/\x60# tbd:\x60 tag AND a matching row in docs/defaults-provenance.md.\n\nParent: #120' \
  --label docs --label testing --label priority:low --assignee @me
```

- [ ] **Step 4: Create the empirical-T4-confirmation issue**

```bash
gh issue create \
  --title "Empirical T4 confirmation of the reference training profile epochs/30-min budget (from #120)" \
  --body $'Follow-up from #120. The shipped \x60epochs\x60 default and the \x22train+eval <=30 min on a 16 GB free-Colab T4\x22 budget are derived ANALYTICALLY (the repo has no runtime/throughput model — only the memory model in presets.py) on an explicit, stated per-step-time assumption.\n\nRun the reference profile on a real free-Colab T4 and replace the analytical budget claim (and its per-step-time assumption) with a measured one. The # tbd: in the docs/defaults-provenance.md \x22Reference Training Profile\x22 section points here.\n\nRelated: #139 (operationalize gpu_t4 tier). Parent: #120' \
  --label docs --label testing --label priority:low --assignee @me
```

Record this number as `T4_CONFIRM` (used by Phase 2 Task 2.2).

- [ ] **Step 5: Verify all three issues exist**

Run: `gh issue list --search "from #120 in:body OR #120 in:body" --state open --json number,title`
Expected: the three new issues appear. Note `UMBRELLA` and `T4_CONFIRM` numbers for later tasks. No commit (GitHub-only task).

### Task 1.2: Scaffold `docs/defaults-provenance.md` (freeze the row schema + reserve the profile anchor)

**Files:**

- Create: `docs/defaults-provenance.md`

This task establishes the frozen interface contract: the Verification Standard section, the row schema, one empty section heading per in-scope file (in audit order), and the reserved `## Reference Training Profile` anchor for Phase 2.

- [ ] **Step 1: Write the doc skeleton**

Create `docs/defaults-provenance.md` with exactly this content (replace `#<UMBRELLA>` with the number from Task 1.1):

```markdown
# Defaults Provenance

This document is the source of truth for the provenance of every trust-bearing
default hyperparameter in `custom-sam-peft`. Inline `# cite:` / `# tbd:` tags in
the code are deliberately terse pointers into the rows below.

Umbrella `# tbd:` tracker: #<UMBRELLA>
(Every `# tbd: #<UMBRELLA>` tag and row points there.)

## Verification Standard

Every literature-backed value is verified against its *primary* source with a
captured quote + URL/DOI + exact equation/table/figure. Framework defaults link
the upstream docs and pin the observed version. Degenerate cases state the math
identity. Reference-implementation values cite the file/line they mirror.
Project numbers with no external source and no internal run are tagged
`# tbd: #<UMBRELLA>` — never fabricated.

Row schema (every section uses these six columns):

| Location | Value | Tag | Full reference | Verifying quote | Notes |
| --- | --- | --- | --- | --- | --- |

- **Location** — `file:symbol`.
- **Value** — the literal default.
- **Tag** — the inline tag class applied (mirrors the code), or `index-only` for
  untagged self-evident structural/string defaults.
- **Full reference** — authors, year, arXiv/DOI, exact Eq./Table/Fig.; or the
  upstream-doc URL + pinned version (framework defaults); or repo file/line
  (reference-impl).
- **Verifying quote** — short quote from the primary source establishing the
  value.
- **Notes** — caveats, degenerate-case identities, calibration run pointers,
  cross-links.

## config/_internal.py

| Location | Value | Tag | Full reference | Verifying quote | Notes |
| --- | --- | --- | --- | --- | --- |

## config/schema.py

| Location | Value | Tag | Full reference | Verifying quote | Notes |
| --- | --- | --- | --- | --- | --- |

## data/aug_presets.py

Legend letters used in the `aug_presets.py` module docstring resolve here.

| Location | Value | Tag | Full reference | Verifying quote | Notes |
| --- | --- | --- | --- | --- | --- |

## data/channel_semantics.py

| Location | Value | Tag | Full reference | Verifying quote | Notes |
| --- | --- | --- | --- | --- | --- |

## data/transforms.py

| Location | Value | Tag | Full reference | Verifying quote | Notes |
| --- | --- | --- | --- | --- | --- |

## presets.py

| Location | Value | Tag | Full reference | Verifying quote | Notes |
| --- | --- | --- | --- | --- | --- |

## cli/templates/config_full.yaml

Template-echoed literals; the authoritative provenance is the schema row for the
same symbol. This section cross-links the template slot to its schema row.

| Location | Value | Tag | Full reference | Verifying quote | Notes |
| --- | --- | --- | --- | --- | --- |

## models/losses/presets.py

### Citation legend (folded in from the module docstring)

| Letter | Source | Establishes |
| --- | --- | --- |

### Preset-table parameters

| Location | Value | Tag | Full reference | Verifying quote | Notes |
| --- | --- | --- | --- | --- | --- |

## Reference Training Profile

<!-- Owned by Deliverable 2 (epochs alignment). Populated in Phase 2. -->
```

- [ ] **Step 2: Run the markdownlint gate on the skeleton**

Run: `npx --yes markdownlint-cli2 --config .config/markdownlint-cli2.jsonc "docs/defaults-provenance.md"`
Expected: PASS (clean). If `npx` cannot reach a node toolchain, use the memory-documented fallback (`uvx --from nodejs-bin --with markdownlint-cli2 markdownlint-cli2 ...`); both run the same config.

- [ ] **Step 3: Commit**

```bash
git add docs/defaults-provenance.md
git commit -m "docs(provenance): scaffold defaults-provenance.md row schema + section anchors"
```

### Task 1.3: Audit `config/_internal.py` (matcher + internal dataclasses)

**Files:**

- Modify: `src/custom_sam_peft/config/_internal.py` (`MatcherWeights` lines 32–34; `WandbConfig` lines 45–46; `ExportConfig` line 57)
- Modify: `docs/defaults-provenance.md` (`## config/_internal.py` table)

- [ ] **Step 1: Verify the matcher-weight provenance (deep verification)**

Research DETR (Carion et al. 2020, arXiv:2005.12872) Hungarian-matcher cost weighting — capture the quote and the exact section/equation for the mask/dice cost weight ratio. Cross-check the Meta SAM3 / DETR reference impl path that the project mirrors for `lambda_mask=5.0`. If the exact `5.0` magnitude is a project ratio with no published value, tag `# tbd: #<UMBRELLA>` rather than over-claiming a DETR number.

- [ ] **Step 2: Add inline tags**

In `MatcherWeights` (lines 32–34) add a terse tag to each weight:

```python
    lambda_l1: float = 0.0  # cite: degenerate-case (text-only v0: box terms disabled)
    lambda_giou: float = 0.0  # cite: degenerate-case (text-only v0: box terms disabled)
    lambda_mask: float = 5.0  # cite: DETR (Carion 2020) matcher cost weighting; see provenance doc
```

(If Step 1 found no published `5.0`, use `# tbd: #<UMBRELLA>` on `lambda_mask` instead and reflect that in the row.)

`WandbConfig.project`/`entity` and `ExportConfig.merge` are self-evident structural/boolean defaults — **do NOT add inline tags**; they get index-only doc rows.

- [ ] **Step 3: Add the matching doc rows**

Under `## config/_internal.py`, add one row per default. Example shape (fill verified content):

```markdown
| `config/_internal.py:MatcherWeights.lambda_mask` | `5.0` | `# cite: DETR (Carion 2020)` | Carion et al. 2020, "End-to-End Object Detection with Transformers", arXiv:2005.12872, §[exact] | "[captured quote]" | Mask/dice cost weight; cross-checked against [SAM3/DETR ref-impl path]. |
| `config/_internal.py:MatcherWeights.lambda_l1` | `0.0` | `# cite: degenerate-case` | — | — | Text-only v0 disables box terms; YAGNI-demoted internal constant. |
| `config/_internal.py:MatcherWeights.lambda_giou` | `0.0` | `# cite: degenerate-case` | — | — | Text-only v0 disables box terms; YAGNI-demoted internal constant. |
| `config/_internal.py:WandbConfig.project` | `"custom_sam_peft"` | `index-only` | — | — | Self-evident project string; not user-trust-bearing. |
| `config/_internal.py:WandbConfig.entity` | `None` | `index-only` | — | — | Optional W&B entity; no default to cite. |
| `config/_internal.py:ExportConfig.merge` | `False` | `index-only` | — | — | Boolean export toggle; off by default. |
```

- [ ] **Step 4: Verify tag/row parity for this file**

Run: `grep -nE '# (cite|tbd):' src/custom_sam_peft/config/_internal.py`
Expected: exactly the three `MatcherWeights` tags. Confirm each tagged symbol has a doc row, and every `config/_internal.py:` doc row names a real symbol.

- [ ] **Step 5: Lint + commit**

Run: `uv run ruff check src/custom_sam_peft/config/_internal.py && uv run ruff format --check src/custom_sam_peft/config/_internal.py && npx --yes markdownlint-cli2 --config .config/markdownlint-cli2.jsonc "docs/defaults-provenance.md"`
Expected: all PASS.

```bash
git add src/custom_sam_peft/config/_internal.py docs/defaults-provenance.md
git commit -m "docs(provenance): cite config/_internal.py matcher + internal defaults"
```

### Task 1.4: Audit `config/schema.py` (~60 user-facing defaults) + write the `epochs` row stub

**Files:**

- Modify: `src/custom_sam_peft/config/schema.py` (defaults across `RunConfig`, `ModelConfig`, `TextPromptConfig`, `NormalizeConfig`, `PEFTConfig`, `QLoRAConfig`, `BoxHintSchedule`, `MultiplexConfig`, `TrainHyperparams`, `EvalConfig`, `ValSplitConfig`, `LimitConfig`, `DataConfig`)
- Modify: `docs/defaults-provenance.md` (`## config/schema.py` table)

This is the largest task. Audit by sub-model in the order below. **The `epochs` row stub (Step 8) is the Phase-2 interface contract — it must be written exactly as specified.**

- [ ] **Step 1: Verify PEFT/QLoRA provenance**

Research LoRA (Hu et al. 2021, arXiv:2106.09685) for rank/alpha/dropout conventions (note the α=2r convention: r=16, α=32). Research QLoRA (Dettmers et al. 2023, arXiv:2305.14314) for NF4 + double-quant. Capture quotes + exact sections. Tag `PEFTConfig.r=16` (line 489), `alpha=32` (line 490), `dropout=0.05` (line 491) → `# cite: LoRA (Hu 2021)`; `QLoRAConfig.quant_type="nf4"` (line 482), `use_double_quant=False` (line 484) → `# cite: QLoRA (Dettmers 2023)`. `scope="vision_decoder"` (line 492), `bias="none"` (line 501) → project/framework convention (`# cite: framework default` or `# tbd:` per verification).

- [ ] **Step 2: Verify optimizer/schedule provenance**

Research cosine/SGDR (Loshchilov & Hutter 2017, arXiv:1608.03983) and AdamW (Loshchilov & Hutter 2019, arXiv:1711.05101). Tag `TrainHyperparams.lr_schedule="cosine"` (line 560) → `# cite: SGDR (Loshchilov 2017)`; `optimizer="auto"` (line 558) → `# cite: AdamW (Loshchilov 2019)` (note `auto` resolves to adamw/adamw8bit at trainer construction, per the module comment lines 93–94). `learning_rate=1.0e-4` (line 559) and `warmup_steps=100` (line 561) are repo-chosen magnitudes → `# tbd: #<UMBRELLA>` unless an internal run justifies them (`# cite: empirical (...)`); related open issue #87 (A/B lr) is a Notes cross-link, not a source.

- [ ] **Step 3: Verify eval/COCO provenance**

Research the COCO detection eval protocol (Lin et al. 2014, arXiv:1405.0312) for the IoU sweep `[0.5:0.05:0.95]`. Tag `EvalConfig.iou_thresholds` (lines 594–596) → `# cite: COCO (Lin 2014)`. `mask_threshold=0.0` (line 599) → `# cite: degenerate-case` (logit-zero decision boundary; state the identity in Notes). `mode="full"` (line 597), `lite_max_images=64` (line 598), `visualize=True` (line 602), `visualize_count=10` (line 603) → `# cite: empirical` / `# tbd: #<UMBRELLA>` per verification.

- [ ] **Step 4: Tag the normalization stats**

`NormalizeConfig.mean` default `[0.485,0.456,0.406]` (lines 266–268) and `std` `[0.229,0.224,0.225]` (lines 269–271) → `# cite:` ImageNet-1k training-set statistics (torchvision provenance; capture the torchvision docs/source URL + version). `max_pixel_value=255.0` (lines 272–281) → `# cite: framework default` (8-bit uint8 max; Albumentations `A.Normalize` convention). Note in the row that these mirror `KNOWN_PROCESSOR_STATS["facebook/sam3.1"]` and the `facebook/sam3.1` processor.

- [ ] **Step 5: Tag box-hint, multiplex, text-prompt, misc**

- `BoxHintSchedule.p_start=1.0` (line 519), `p_end=0.0` (line 520) → `# cite: degenerate-case` (full-hint→no-hint linear decay endpoints; identity in Notes). `decay_steps=None` (line 521) → `# cite: empirical` / index (the 0.75 decay fraction lives in the field description; cross-link open issue #88).
- `MultiplexConfig.classes_per_forward=16` (line 551) and `TextPromptConfig.k=16` (line 241) → `# cite: <repo>:models/sam3.py` SAM 3.1 `MULTIPLEX_CAP=16`.
- `TextPromptConfig.negatives_per_image=0` (line 231) → `# cite: empirical` / index (the description explains the 0 default).
- `RunConfig.seed=42` (line 111), `LimitConfig.seed=42` (line 357) → `# cite: degenerate-case` / convention (arbitrary fixed seed; state it's convention in Notes).
- `ValSplitConfig.fraction=0.1` (line 336) → `# cite: empirical` / `# tbd: #<UMBRELLA>`.
- `TrainHyperparams.nan_abort_after=20` (line 583), `log_every=50` (line 571), `num_workers=min(4,cpu)` (lines 584–588) → `# cite: empirical` / index.
- `DataConfig.channels=3` (line 384) cap-16 rationale and `channel_semantics="rgb"` (line 398) → index-only (rationale is in the field description; default reproduces today's behavior). `ModelConfig.dtype="bfloat16"` (line 117) → `# cite: framework default`.

- [ ] **Step 6: Tag batch/accum (cross-link presets.py)**

`TrainHyperparams.batch_size=1` (line 556), `grad_accum_steps=8` (line 557) → `# cite: empirical` / `# tbd: #<UMBRELLA>` — VRAM-driven engineering choice; Notes cross-link the `presets.py` memory model.

- [ ] **Step 7: Confirm structural/string defaults get index-only rows (no inline tags)**

For `output_dir`, `model.name`, `local_dir`, `checkpoint_file`, `bbox_format`, `split_train`, `TrackingConfig.backend`, `HFFieldMap.*`, `format` literals: add `index-only` doc rows; add NO inline tags.

- [ ] **Step 8: Write the `epochs` row stub (PHASE-2 INTERFACE CONTRACT — verbatim)**

`TrainHyperparams.epochs` (line 555) is a **required** `PositiveInt` field with **no schema default** — do NOT add an inline tag to the field. Add exactly this row under `## config/schema.py`:

```markdown
| `config/schema.py:TrainHyperparams.epochs` | `required (template $epochs slot)` | `# cite: empirical` | See "Reference Training Profile" section below (Deliverable 2). | See "Reference Training Profile" section below (Deliverable 2). | Required field; no schema default. The shipped default lives in the `config_full.yaml` `$epochs` slot, set by the `init` flow. Provenance is the analytical reference profile, not a single citation. |
```

- [ ] **Step 9: Add all remaining `## config/schema.py` doc rows**

One row per audited default, in the schema's declaration order, each with verified Full reference + Verifying quote captured during Steps 1–7.

- [ ] **Step 10: Verify tag/row parity**

Run: `grep -nE '# (cite|tbd):' src/custom_sam_peft/config/schema.py`
Expected: a tag on every trust-bearing default from Steps 1–6; none on `epochs` or structural defaults. Confirm every tag has a doc row and every `config/schema.py:` row names a real symbol. Confirm the `epochs` row text matches Step 8 verbatim.

- [ ] **Step 11: Lint + verify schema still imports + commit**

Run: `uv run ruff check src/custom_sam_peft/config/schema.py && uv run ruff format --check src/custom_sam_peft/config/schema.py && uv run python -c "import custom_sam_peft.config.schema" && npx --yes markdownlint-cli2 --config .config/markdownlint-cli2.jsonc "docs/defaults-provenance.md"`
Expected: all PASS (comments do not change behavior; import sanity-checks no accidental edit).

```bash
git add src/custom_sam_peft/config/schema.py docs/defaults-provenance.md
git commit -m "docs(provenance): cite config/schema.py user-facing defaults + epochs row stub"
```

### Task 1.5: Audit `data/aug_presets.py` (12-cell table via letter-legend)

**Files:**

- Modify: `src/custom_sam_peft/data/aug_presets.py` (module docstring lines 1–16; `PRESET_TABLE` lines 34–155)
- Modify: `docs/defaults-provenance.md` (`## data/aug_presets.py` table)

Highest-priority file (0 cites today). Dense table → use a per-file letter-legend in the docstring (like the loss-preset A–H legend); cells reference legend letters.

- [ ] **Step 1: Verify augmentation-knob provenance**

For each knob class, determine the dominant tag:

- `hflip`/`vflip`/`rotate90` booleans and the `p=0.5` probabilities → `# cite: framework default` (Albumentations `HorizontalFlip`/`VerticalFlip`/`RandomRotate90` default `p=0.5`; capture the Albumentations docs URL + pin the installed version from `uv pip show albumentations`).
- `color_jitter` magnitudes (0.05/0.1/0.2) → verify against Albumentations `ColorJitter` defaults; domain-tuned magnitudes → `# tbd: #<UMBRELLA>`.
- `rotate_arbitrary` (5/10/15°), `blur`, `gauss_noise` magnitudes → domain-tuned project choices → `# tbd: #<UMBRELLA>` (or `# cite: empirical` if a run justifies).
- `stain_jitter` (0.03/0.07) → ties to Ruifrok & Johnston 2001 / Tellez et al. 2018 (the HED basis lives in `transforms.py`); cite those for the H&E rationale, `# tbd:` for the exact magnitude.
- The `LOCKED_OFF` laterality rationale is already documented in-code; reference it in the legend, not re-cited.

- [ ] **Step 2: Add the letter-legend to the module docstring**

Extend the docstring (after line 16, before `from __future__`) with a legend block, e.g.:

```python
Citation legend (full references in docs/defaults-provenance.md §data/aug_presets.py):
  (a) Albumentations framework default — flip/rotate p=0.5; ColorJitter default ranges
  (b) project-empirical magnitude — domain-tuned; no published value (# tbd: #<UMBRELLA>)
  (c) Ruifrok & Johnston 2001 / Tellez et al. 2018 — H&E stain-jitter rationale
  (d) laterality-driven locked-off (see LOCKED_OFF) — clinically/structurally meaningful
```

- [ ] **Step 3: Tag table cells by legend letter**

Append a terse legend-letter tag to the trust-bearing knobs in `PRESET_TABLE` cells, e.g. `"hflip": True,  # (a)`, `"rotate_arbitrary": 10.0,  # (b)`, `"stain_jitter": 0.03,  # (c)`. Zero-valued/disabled knobs (`0.0`/`False`) that are simply "off" need no tag; only tag values that carry trust. (`E501`/`RUF003` are already exempted for this file in pyproject, so wide inline comments are fine.)

- [ ] **Step 4: Add doc rows**

Under `## data/aug_presets.py`, add a row per trust-bearing knob value (group by knob to avoid 96 near-duplicate rows: one row per (knob, distinct-value) with the cells that use it listed in Notes is acceptable and DRY). Each row's Tag mirrors the legend letter; Full reference + Verifying quote carry the verified content.

- [ ] **Step 5: Verify resolver still works + parity**

Run: `uv run python -c "from custom_sam_peft.data.aug_presets import PRESET_TABLE, resolve; print(len(PRESET_TABLE))"`
Expected: prints `12` (3 microscopy aliases are in the *losses* file, not here; `aug_presets` has 12 literal cells). Confirm legend letters used in cells all appear in the docstring legend and in doc rows.

- [ ] **Step 6: Lint + commit**

Run: `uv run ruff check src/custom_sam_peft/data/aug_presets.py && uv run ruff format --check src/custom_sam_peft/data/aug_presets.py && npx --yes markdownlint-cli2 --config .config/markdownlint-cli2.jsonc "docs/defaults-provenance.md"`
Expected: all PASS.

```bash
git add src/custom_sam_peft/data/aug_presets.py docs/defaults-provenance.md
git commit -m "docs(provenance): cite data/aug_presets.py preset table via letter-legend"
```

### Task 1.6: Audit `data/channel_semantics.py` (normalization stats)

**Files:**

- Modify: `src/custom_sam_peft/data/channel_semantics.py` (`_IMAGENET_MEAN`/`_IMAGENET_STD` lines 15–16; `CHANNEL_SEMANTICS` profiles lines 31–60)
- Modify: `docs/defaults-provenance.md` (`## data/channel_semantics.py` table)

- [ ] **Step 1: Verify provenance**

`_IMAGENET_MEAN=(0.485,0.456,0.406)` and `_IMAGENET_STD=(0.229,0.224,0.225)` → ImageNet-1k training-set statistics (reuse the torchvision provenance captured in Task 1.4 Step 4). The `grayscale` profile `((0.449,),(0.226,))` → torchvision grayscale-ImageNet single-channel convention (capture source). The `rgba` extra alpha channel `0.5` → `# cite: degenerate-case` (neutral [0,1]→[-1,1] zero-centering; state the identity). `freeform` `normalize_default=None` → index-only (requires explicit user stats; no default to cite).

- [ ] **Step 2: Add inline tags**

```python
_IMAGENET_MEAN = (0.485, 0.456, 0.406)  # cite: ImageNet-1k stats (torchvision); see provenance doc
_IMAGENET_STD = (0.229, 0.224, 0.225)  # cite: ImageNet-1k stats (torchvision); see provenance doc
```

In the `grayscale` profile, tag `normalize_default=((0.449,), (0.226,))` → `# cite: torchvision grayscale-ImageNet`. In `rgba`, the appended `0.5` → `# cite: degenerate-case (neutral alpha)`.

- [ ] **Step 3: Add doc rows + Step 4: verify import + Step 5: lint + commit**

Add the matching rows under `## data/channel_semantics.py`. Then:

Run: `uv run python -c "import custom_sam_peft.data.channel_semantics" && uv run ruff check src/custom_sam_peft/data/channel_semantics.py && uv run ruff format --check src/custom_sam_peft/data/channel_semantics.py && npx --yes markdownlint-cli2 --config .config/markdownlint-cli2.jsonc "docs/defaults-provenance.md"`
Expected: all PASS.

```bash
git add src/custom_sam_peft/data/channel_semantics.py docs/defaults-provenance.md
git commit -m "docs(provenance): cite data/channel_semantics.py normalization stats"
```

### Task 1.7: Audit `data/transforms.py` (processor stats, HED matrix, projection constants)

**Files:**

- Modify: `src/custom_sam_peft/data/transforms.py` (`KNOWN_PROCESSOR_STATS` line 60; `_STATS_DIVERGENCE_ATOL` line 67; `_HED_FROM_RGB_MATRIX` lines 76–83; `_GAUSS_NOISE_MAX_VAR` line 87; `_GAUSS_BLUR_MAX_SIGMA` line 88)
- Modify: `docs/defaults-provenance.md` (`## data/transforms.py` table)

- [ ] **Step 1: Verify provenance**

- `KNOWN_PROCESSOR_STATS["facebook/sam3.1"]` → ImageNet stats mirroring the HF `Sam3ImageProcessor`; the inline comment (lines 55–59) already ratifies this via the 2026-05-21 audit. Cross-link open issue #86 (empirical processor-stats verification) in Notes.
- `_HED_FROM_RGB_MATRIX` → Ruifrok & Johnston 2001, "Quantification of histochemical staining by color deconvolution", Anal Quant Cytol Histol 23(4):291–299. Capture the published H/E/DAB RGB absorbance vectors as the verifying quote and confirm they match the literal matrix rows.
- `_STATS_DIVERGENCE_ATOL=1e-3` → `# cite: empirical` (tolerance to catch `[0.5,0.5,0.5]` drift; rationale in the inline comment).
- `_GAUSS_NOISE_MAX_VAR=0.05`, `_GAUSS_BLUR_MAX_SIGMA=3.0` → `# tbd: #<UMBRELLA>` (project magnitude→Albumentations projection ceilings; the spec §8.1 they reference is internal, not a publishable source).

- [ ] **Step 2: Add inline tags**

Append tags: `KNOWN_PROCESSOR_STATS` → `# cite: ImageNet stats (HF Sam3ImageProcessor)`; `_STATS_DIVERGENCE_ATOL` → `# cite: empirical`; the `_HED_FROM_RGB_MATRIX` definition → `# cite: Ruifrok & Johnston 2001` (the docstring comment lines 70–75 already names it; add the terse tag at the assignment); `_GAUSS_NOISE_MAX_VAR`/`_GAUSS_BLUR_MAX_SIGMA` → `# tbd: #<UMBRELLA>`.

- [ ] **Step 3: Add doc rows + Step 4: verify + Step 5: lint + commit**

Add rows under `## data/transforms.py`. Then:

Run: `uv run python -c "import custom_sam_peft.data.transforms" && uv run ruff check src/custom_sam_peft/data/transforms.py && uv run ruff format --check src/custom_sam_peft/data/transforms.py && npx --yes markdownlint-cli2 --config .config/markdownlint-cli2.jsonc "docs/defaults-provenance.md"`
Expected: all PASS.

```bash
git add src/custom_sam_peft/data/transforms.py docs/defaults-provenance.md
git commit -m "docs(provenance): cite data/transforms.py HED matrix + processor stats"
```

### Task 1.8: Audit `presets.py` (VRAM-model constants)

**Files:**

- Modify: `src/custom_sam_peft/presets.py` (`MODEL_PARAMS` line 41; `LORA_LAYERS` line 45; `D_IN`/`D_OUT` lines 46–47; `Q_OVERHEAD` line 48; `WORKSPACE_BYTES` line 49; `BASE_ACTIVATION_AT_1024` line 50; `forward_only_factor` line 55; `_SAM3_PATCH`/`_SAM3_HEADS` lines 140–141; `_bytes_per_param_for_method` line 147; `_optimizer_bytes` `*4` lines 160–163; `CACHE_SCHEMA_VERSION` line 60)
- Modify: `docs/defaults-provenance.md` (`## presets.py` table)

- [ ] **Step 1: Verify provenance**

- `MODEL_PARAMS=5e9` → `# cite: <repo>:scripts/_derive_preset_constants.py` (SAM 3.1 checkpoint parameter count; the inline comment lines 41–44 already cite the derivation script). `_SAM3_PATCH=14`, `_SAM3_HEADS=16` → `# cite: <repo>:sam3/model_builder.py` (hiera-large backbone; comment lines 134–140 already note it).
- `_bytes_per_param_for_method` 2.0/0.5 (line 147) and `_optimizer_bytes` `*4` (lines 160–163) → `# cite: framework default` (bf16/fp16 = 2 B/param; NF4 = 0.5 B/param; AdamW fp32 m+v+master = 8 B/param → 4× the bf16 adapter). The `*4` is the AdamW-state cost the issue mislabeled `ADAMW_STATE_MULT` — confirm in Notes that no `ADAMW_STATE_MULT` symbol exists.
- `LORA_LAYERS=96`, `D_IN=768`, `D_OUT=768`, `Q_OVERHEAD`, `WORKSPACE_BYTES`, `BASE_ACTIVATION_AT_1024`, `forward_only_factor=0.25` → `# cite: empirical` — VRAM calibration in #148 / #179 (both CLOSED/merged). Capture the PR/issue links and any run IDs as the provenance; these are empirical-calibration values, not literature.
- `CACHE_SCHEMA_VERSION=2` → index-only (internal cache versioning; not trust-bearing).

- [ ] **Step 2: Add inline tags**

Append terse tags to each trust-bearing constant per Step 1 (e.g. `forward_only_factor: float = 0.25  # cite: empirical (#148/#179 VRAM calibration)`). Tag the `*4` literal in `_optimizer_bytes` with an inline `# cite: framework default (AdamW fp32 m+v+master = 8 B/param)` on the relevant line.

- [ ] **Step 3: Add doc rows + Step 4: verify + Step 5: lint + commit**

Add rows under `## presets.py`. Then:

Run: `uv run python -c "import custom_sam_peft.presets" && uv run ruff check src/custom_sam_peft/presets.py && uv run ruff format --check src/custom_sam_peft/presets.py && npx --yes markdownlint-cli2 --config .config/markdownlint-cli2.jsonc "docs/defaults-provenance.md"`
Expected: all PASS.

```bash
git add src/custom_sam_peft/presets.py docs/defaults-provenance.md
git commit -m "docs(provenance): cite presets.py VRAM-model constants"
```

### Task 1.9: Index `cli/templates/config_full.yaml` (template-echoed literals)

**Files:**

- Modify: `src/custom_sam_peft/cli/templates/config_full.yaml` (only if a terse pointer comment helps; primary work is doc rows)
- Modify: `docs/defaults-provenance.md` (`## cli/templates/config_full.yaml` table)

The template echoes schema defaults (`r: 16`, `alpha: 32`, `dropout: 0.05`, `learning_rate: 1.0e-4`, `lr_schedule: cosine`, `warmup_steps: 100`, `max_grad_norm: 1.0`, `negatives_per_image: 4`, ImageNet `mean`/`std`, `epochs: $epochs`). YAML inline-comment tags are optional and can clutter a user-facing file; the authoritative provenance is the schema row.

- [ ] **Step 1: Add cross-link doc rows**

Under `## cli/templates/config_full.yaml`, add one row per echoed literal whose Tag is `cross-link` and whose Notes points at the authoritative `config/schema.py:<symbol>` row, e.g.:

```markdown
| `config_full.yaml:peft.r` | `16` | cross-link | See `config/schema.py:PEFTConfig.r` row. | — | Template echo of the schema default. |
| `config_full.yaml:train.learning_rate` | `1.0e-4` | cross-link | See `config/schema.py:TrainHyperparams.learning_rate` row. | — | Template echo. |
| `config_full.yaml:train.epochs` | `$epochs` | cross-link | See `config/schema.py:TrainHyperparams.epochs` row + "Reference Training Profile". | — | Placeholder filled by the `init` flow; default set in Phase 2. |
```

Note in the section preamble that `negatives_per_image: 4` here differs from the schema default `0` — record the template's `4` value and cross-link the schema `negatives_per_image` row + its field-description rationale (COCO present-class headroom).

- [ ] **Step 2: Verify + commit**

Run: `npx --yes markdownlint-cli2 --config .config/markdownlint-cli2.jsonc "docs/defaults-provenance.md"`
Expected: PASS.

```bash
git add docs/defaults-provenance.md src/custom_sam_peft/cli/templates/config_full.yaml
git commit -m "docs(provenance): cross-link config_full.yaml template literals to schema rows"
```

### Task 1.10: Firm up `models/losses/presets.py` (zero `# citation needed`)

**Files:**

- Modify: `src/custom_sam_peft/models/losses/presets.py` (docstring line 25; cells: `focal_gamma` lines 116, 131, 162, 177, 208, 223; `tversky_alpha` lines 164, 179, 225)
- Modify: `docs/defaults-provenance.md` (`## models/losses/presets.py` — legend fold-in + preset-table rows)

- [ ] **Step 1: Verify the escalated values (deep verification)**

- `focal_gamma` escalation 2.5 (lines 116, 162, 208) and 3.0 (lines 131, 177, 223) sit above Lin et al. 2017 (focal loss) γ=2.0 (legend C). No direct paper value for 2.5/3.0 → `# tbd: #<UMBRELLA>` unless an internal run justifies (`# cite: empirical`). Confirm there is no recorded run; default to `# tbd:`.
- `tversky_alpha=0.7` (lines 164, 225) → verify against Salehi et al. 2017 (α=0.7 best on MS lesions, legend E). If it matches, convert to `# cite: (A,E)` (drop "citation needed", point at legend E with the captured quote in the doc).
- `tversky_alpha=0.8` (line 179) → no paper value → `# tbd: #<UMBRELLA>`.
- Docstring line 25 mentions cells "lacking a firm cite carry `# citation needed`" — rewrite that sentence to state the firm-up outcome (all cells now resolve to a legend letter or `# tbd: #<UMBRELLA>`) so the literal string `citation needed` no longer appears.

- [ ] **Step 2: Edit each `# citation needed` cell**

For γ=2.5/3.0 cells, replace `# cite: (A)  # citation needed` with `# tbd: #<UMBRELLA>` (or `# cite: (A,C+)` only if an empirical run is found). For `tversky_alpha=0.7` cells, replace `# cite: (A,D)  # citation needed for this exact value` with `# cite: (A,E)` (Salehi 2017). For `tversky_alpha=0.8`, replace `# cite: (A)  # citation needed` with `# tbd: #<UMBRELLA>`. Rewrite the docstring line-25 sentence.

- [ ] **Step 3: Fold the A–H legend into the doc + add preset-table rows**

Reproduce the A–H legend (docstring lines 13–27) in the `### Citation legend` sub-table under `## models/losses/presets.py`, expanded with quotes/DOIs (Lin 2017 focal; Abraham & Khan 2019 focal-Tversky; Salehi 2017 Tversky; Kervadec 2019 boundary). Add `### Preset-table parameters` rows for the firmed values (the γ escalations and the tversky_alpha values), each Tag mirroring the new inline tag. No re-tuning of any loss hyperparameter.

- [ ] **Step 4: Verify zero `citation needed` (acceptance criterion 2)**

Run: `grep -c "citation needed" src/custom_sam_peft/models/losses/presets.py`
Expected: `0`.

- [ ] **Step 5: Verify loss resolver still works + lint + commit**

Run: `uv run python -c "from custom_sam_peft.models.losses.presets import PRESET_TABLE, resolve; print(len(PRESET_TABLE))" && uv run ruff check src/custom_sam_peft/models/losses/presets.py && uv run ruff format --check src/custom_sam_peft/models/losses/presets.py && npx --yes markdownlint-cli2 --config .config/markdownlint-cli2.jsonc "docs/defaults-provenance.md"`
Expected: prints `15` (12 base + 3 microscopy aliases); all lint PASS.

```bash
git add src/custom_sam_peft/models/losses/presets.py docs/defaults-provenance.md
git commit -m "docs(provenance): firm up loss-preset citation-needed cells; fold legend into doc"
```

### Task 1.11: Phase 1 full-suite verification

**Files:** none (verification only).

- [ ] **Step 1: Run the relevant unit suites (bypass coverage gate for the subset)**

Run: `uv run pytest -o "addopts=" tests/unit/test_config_schema.py tests/unit/test_aug_presets.py tests/unit/test_channel_semantics.py tests/unit/test_data_transforms.py tests/unit/test_presets.py tests/unit/test_loss_presets.py -q`
Expected: all PASS (comments + new doc do not change behavior).

- [ ] **Step 2: Full markdownlint gate (exact CI command)**

Run: `npx --yes markdownlint-cli2 --config .config/markdownlint-cli2.jsonc "**/*.md" "#node_modules"`
Expected: PASS (covers the new `docs/defaults-provenance.md` plus all existing `.md`).

- [ ] **Step 3: Confirm no numeric default changed (acceptance criterion 6)**

Run: `git diff main -- src/custom_sam_peft/config src/custom_sam_peft/data src/custom_sam_peft/presets.py src/custom_sam_peft/models/losses/presets.py | grep -E '^\+' | grep -vE '#|^\+\+\+'`
Expected: no added lines that change a numeric/string literal value — only comment additions. (`epochs` is untouched in Phase 1; it changes in Phase 2 via the template, not the schema.) Manually scan the output to confirm every `+` line is a comment or doc, not a value edit.

- [ ] **Step 4: Confirm tag/row parity tree-wide**

Run: `grep -rnE '# (cite|tbd):' src/custom_sam_peft/config src/custom_sam_peft/data src/custom_sam_peft/presets.py | wc -l`
and visually confirm each tagged symbol has a doc row in `docs/defaults-provenance.md` and every doc-row Location names a real symbol. No commit (verification only; prior tasks already committed).

---

## Phase 2 — Reference Training Profile + epochs Alignment + Integration-Test Comment (Deliverable 2)

**Feature block:** the one default whose provenance is a recipe. Consumes the Phase-1 interface contract (the `epochs` row stub, the reserved `## Reference Training Profile` anchor, the umbrella + T4-confirm issue numbers) without re-reading Phase-1 source edits.

**What this phase CONSUMES from Phase 1:**

- The `config/schema.py:TrainHyperparams.epochs` doc row stub (linking row), already present in `docs/defaults-provenance.md`.
- The reserved `## Reference Training Profile` heading at the end of `docs/defaults-provenance.md`.
- `UMBRELLA` and `T4_CONFIRM` issue numbers (recorded in the doc's Verification Standard section / from `gh issue list`).

**What this phase EXPOSES:** the shipped `epochs` numeric default (in the template), the written profile section, and the CI-truncation comment in the integration tests. Final task opens the PR.

### Amendment (2026-05-30): epochs convergence-anchored to SAMed (supersedes the 30-min-budget framing)

During Phase-2 prep two corrections superseded this phase's original framing:

- **Factual correction.** The shipped `epochs` default is **`10`**, not `25`. It lives in `src/custom_sam_peft/cli/init_cmd.py:162` (`string.Template(...).substitute(epochs=10)`) and the wizard `src/custom_sam_peft/cli/setup_wizard.py:362` (`default="10"`). The GPU test's `epochs=25` is an **unrelated 2-image `tiny_coco` overfit smoke budget** (50 forward steps ÷ 2 images = 25 epochs), not the production default — its "shipped 50-step budget" wording means the *test's* budget.
- **Decision: `EPOCHS_VALUE = 160`**, cited to **SAMed** (Zhang & Liu 2023, "Customized Segment Anything Model for Medical Image Segmentation", arXiv:2304.13785) — LoRA fine-tuning of SAM on a small dataset reaching convergence at 160 epochs. The Reference Training Profile becomes **convergence-anchored**, not budget-anchored: the 30-min-T4 budget and reaching convergence are mutually exclusive, and the standing design priority is **final accuracy ≫ training speed**, so the 30-min budget is **dropped** as the driver.
- There is **no citable T4 per-step wall-clock figure** in the literature; any runtime estimate stays `# tbd: #193`.
- **This supersedes the parent spec's 30-min-budget framing for the `epochs` deliverable.** Follow-up inventory is now FOUR issues: **#191** (umbrella `# tbd:` tracker), **#192** (CI no-uncited-default hook), **#193** (empirical T4 confirmation), **#195** (2-image overfit GPU smoke-test speed/convergence). Tasks 2.1–2.5 below are amended accordingly; where older text says "if it confirms 25" or "30-min budget", the amended text governs.

### Task 2.1: Select the reference epochs value (literature-anchored)

**Files:** none (selection; output recorded in Task 2.2).

This task is now a **literature-anchored selection**, not a 30-min-budget arithmetic derivation. `EPOCHS_VALUE = 160`, cited to SAMed. The steps below capture the source locators and the corrected baseline.

- [ ] **Step 1: Read the inputs**

Read `src/custom_sam_peft/presets.py` (`decide_preset` / `_predicted_bytes` — the memory model; confirms a step *fits* a 16 GB T4, not how long it takes) and `tests/gpu/test_real_train_qlora_resume.py` (the `epochs=25` value there is a **2-image overfit smoke budget** — 50 forward steps ÷ 2 images — NOT the production default). Confirm there is **no** runtime/throughput estimator anywhere in `src/` (verified: only the memory model exists).

Run: `grep -rn "estimate_runtime\|estimate_vram\|wall.clock\|seconds_per_step\|throughput" src/custom_sam_peft/ || echo "NONE FOUND (expected)"`
Expected: `NONE FOUND (expected)`. (The grep confirms there is no in-repo runtime model — so any wall-clock claim stays `# tbd: #193`.)

- [ ] **Step 2: Capture the SAMed convergence anchor**

The closest published analog is **SAMed** (Zhang & Liu 2023, "Customized Segment Anything Model for Medical Image Segmentation", arXiv:2304.13785): LoRA fine-tuning of SAM (rank 4, AdamW) on a SMALL dataset reaching convergence at **160 epochs**. Capture these locators/quotes for the doc row in Task 2.2:

- Sec 4.2: "We adopt early stop at 14880 iterations (160 epochs)".
- Sec 4.1: "the training set contains 2212 axial slices" (18 cases — a small dataset, the regime this repo targets).
- Abstract: "After finetuning only 160 epochs on Synapse ... SAMed achieves 81.88 DSC".

This anchors `EPOCHS_VALUE = 160` to a convergence figure, not a budget artifact.

- [ ] **Step 3: Record runtime as unverified (no citable T4 figure)**

A literature review found **no citable T4 per-step wall-clock figure** for a SAM/ViT-scale forward+backward. The 30-min-T4 budget and reaching convergence are mutually exclusive (160 epochs on a real dataset at batch=1×grad_accum=8 takes well over 30 min on a T4), and the standing design priority is **final accuracy ≫ training speed** — so the 30-min budget is dropped as the driver. Record the runtime as **unverified**: `# tbd: #193` (empirical T4 confirmation). Do NOT fabricate a per-step time.

- [ ] **Step 4: Record the corrected baseline + chosen value**

Record the corrected baseline: the **current shipped default is `10`** (`init_cmd.py:162` substitute `epochs=10` + wizard `setup_wizard.py:362` `default="10"`), NOT `25` (the GPU test's `25` is a separate 2-image overfit budget, now tracked by #195). The **chosen value is `160`** (SAMed convergence anchor). Write `EPOCHS_VALUE = 160` down for Tasks 2.2/2.3. The value is honestly **literature-anchored (SAMed)** with runtime **unverified** (`# tbd: #193`) — not budget-derived.

### Task 2.2: Write the Reference Training Profile section

**Files:**

- Modify: `docs/defaults-provenance.md` (`## Reference Training Profile` section — the reserved anchor)

- [ ] **Step 1: Replace the reserved comment with the profile content (convergence-anchored)**

Under `## Reference Training Profile`, write the **convergence-anchored** profile specifying: `batch_size=1` / `grad_accum_steps=8` (the shipped values); `epochs=EPOCHS_VALUE` = **160** (from Task 2.1); eval mode (`full`). Anchor the 160-epoch default to **SAMed** with the quote + locator: arXiv:2304.13785, Sec 4.2 "We adopt early stop at 14880 iterations (160 epochs)" (Sec 4.1: 2212-slice small training set; Abstract: 160 epochs → 81.88 DSC). State the **convergence-vs-runtime tradeoff honestly**: 160 epochs on a real dataset at batch=1×grad_accum=8 takes **well over 30 min on a T4**, so the original 30-min budget is dropped in favor of convergence (accuracy ≫ speed); there is **no citable T4 per-step wall-clock figure**, so the runtime stays `# tbd: #193` (empirical T4 confirmation) — never a completed claim. Add a cross-reference to **#195** (the 2-image overfit GPU smoke-test speed/convergence) as the empirical home for the GPU-test budget questions.

- [ ] **Step 2: Update the schema epochs doc row Tag cell**

In the already-committed `config/schema.py:TrainHyperparams.epochs` doc row (`## config/schema.py` table), change the **Tag** cell from `# cite: empirical` to the literature anchor (e.g. `# cite: SAMed (Zhang 2023)`, with `# tbd: #193` for the unverified runtime). The row's "See Reference Training Profile section below (Deliverable 2)." pointer text in the Full reference / Verifying quote / Notes cells **stays unchanged**.

- [ ] **Step 3: Confirm the epochs row now resolves**

Verify the `config/schema.py:TrainHyperparams.epochs` row's "See Reference Training Profile section below" pointer lands on a populated section (only the Tag cell changed in Step 2; the pointer text is unchanged).

- [ ] **Step 4: Markdownlint + commit**

Run: `npx --yes markdownlint-cli2 --config .config/markdownlint-cli2.jsonc "docs/defaults-provenance.md"`
Expected: PASS.

```bash
git add docs/defaults-provenance.md
git commit -m "docs(provenance): add convergence-anchored Reference Training Profile (epochs=160, SAMed); update schema epochs Tag cell"
```

### Task 2.3: Set the shipped `epochs` default + align the GPU-test docstring

**Files:**

- Modify: `src/custom_sam_peft/cli/init_cmd.py` (line 162: `substitute(... epochs=10 ...)` → `epochs=160`)
- Modify: `src/custom_sam_peft/cli/setup_wizard.py` (line 362: `ask_text("Number of epochs?", default="10", ...)` → `default="160"`)
- Modify: `tests/gpu/test_real_train_qlora_resume.py` docstring (de-conflate the "shipped" wording; do NOT touch the `epochs=13`/`epochs=25` override values)

The shipped default is `10` (NOT the GPU test's `25`). It lives in **both** `init_cmd.py:162` (template `$epochs` substitution) and `setup_wizard.py:362` (interactive wizard default). Both move **`10` → `160`** (`EPOCHS_VALUE`, SAMed-anchored).

- [ ] **Step 1: Locate where `$epochs` is substituted**

Run: `grep -rn '\$epochs\|epochs' src/custom_sam_peft/cli/ | grep -iv 'log_every\|eval_every\|save_every'`
Confirm both sites: `init_cmd.py:162` `string.Template(...).substitute(... epochs=10 ...)` (fills the `config_full.yaml` `$epochs` placeholder — the template keeps `$epochs`) and `setup_wizard.py:362` `ask_text("Number of epochs?", default="10", ...)`. Do NOT hardcode a literal into the template itself; change the substitution default and the wizard default.

- [ ] **Step 2: Apply the epochs default in BOTH sites**

Set `init_cmd.py:162` `epochs=10` → `epochs=160`, and `setup_wizard.py:362` `default="10"` → `default="160"`. After this, both the `init` flow's rendered `config_full.yaml` and the wizard ship `epochs: 160`.

- [ ] **Step 3: Fix the GPU-test docstring (de-conflate; do NOT retune the budget)**

In `tests/gpu/test_real_train_qlora_resume.py`, UPDATE the docstring text to remove the misleading "shipped" conflation: the test's 50-step / `epochs=25` budget is a **2-image overfit** budget (50 forward steps ÷ 2 images), NOT the production default. Reword so the docstring says this is a 2-image overfit smoke budget, cross-reference the real production default (**160**, set in `init_cmd.py`/`setup_wizard.py`) and issue **#195** (which empirically tracks the 2-image overfit test's speed/convergence). **Do NOT change the test's `epochs=13` / `epochs=25` override values** — they are a deliberate 2-image step budget, now tracked by #195; retuning them is **out of scope** for this task.

- [ ] **Step 4: Verify init/wizard rendering + tests**

Run: `uv run pytest -o "addopts=" tests/unit/cli/ tests/unit/test_cli_init.py -q`
Expected: PASS. Confirm `init`/wizard render the new **160** default. If any unit test asserts the old `10` literal (e.g. an `init`-render or wizard-default assertion), update that expectation to `160` as part of this task.

- [ ] **Step 5: Lint + commit**

Run: `uv run ruff check src/custom_sam_peft && uv run ruff format --check src/custom_sam_peft`
Expected: PASS.

```bash
git add src/custom_sam_peft/cli/init_cmd.py src/custom_sam_peft/cli/setup_wizard.py tests/gpu/test_real_train_qlora_resume.py
git commit -m "feat(init): raise shipped epochs default 10->160 (SAMed convergence anchor); de-conflate GPU-test docstring"
```

### Task 2.4: Add the CI-truncation comment to the integration tests (verify default path)

**Files:**

- Modify: `tests/integration/test_train_then_eval.py` (`_make_cfg`, lines 49–74)
- Modify: `tests/integration/test_train_end_to_end.py` (config builders inside `test_fit_end_to_end_on_tiny_coco` ~line 64, `test_end_to_end_writes_loss_bundle_json` ~line 126, `_bad_data_cfg` ~line 178, `test_e2e_auto_split_on_tiny_coco` ~line 368, `test_e2e_no_val_on_tiny_coco` ~line 417)

**Verified override inventory (from the live files at planning time — confirm, don't re-derive):**

- `test_train_then_eval.py:_make_cfg` (spec called this `_build_cfg`; the real symbol is `_make_cfg`) sets `epochs=1`, `learning_rate=1e-4` (equals the schema default — harmless but redundant), `warmup_steps=0` (the schema default is `100` — a **non-default practical override**), plus structural test knobs `grad_accum_steps=1`, `eval_every=1`, `save_every=1000`, `log_every=1`, `num_workers=0`, and `EvalConfig(mode="full", iou_thresholds=[0.5], lite_max_images=1)`. `PEFTConfig(method="lora", target_modules=FIXTURE_SCOPE_PATTERNS["vision"])` leaves `r/alpha/dropout/scope` at schema defaults.
- `test_train_end_to_end.py` builders set `epochs=1`, `warmup_steps=0` (again the **non-default** vs schema `100`), structural `grad_accum_steps=1`, `save_every` (2 or 1), `log_every=1`, `num_workers=0`; `PEFTConfig(method="lora", scope="vision", target_modules=...)`. No `learning_rate`/`lr_schedule`/`optimizer`/`max_grad_norm` pins (those default).

- [ ] **Step 1: Read both tests and audit the configs**

Read both files. For each config builder, confirm it sets `PEFTConfig(method="lora")` and `TrainHyperparams(epochs=1, …)` and **relies on schema defaults** for `learning_rate`, `lr_schedule`, `optimizer`, `max_grad_norm`, and `peft.r/alpha/dropout/scope`. Note the one practical deviation already known: `warmup_steps=0` (schema default `100`). `grad_accum_steps`, `eval_every`, `save_every`, `log_every`, `num_workers`, and the `EvalConfig` knobs are structural test-speed settings, not "practical training hyperparameters" — leave them.

Run: `grep -nE 'learning_rate|lr_schedule|optimizer=|warmup_steps|max_grad_norm|PEFTConfig\(|TrainHyperparams\(|epochs' tests/integration/test_train_then_eval.py tests/integration/test_train_end_to_end.py`
Expected: surfaces every hyperparameter the configs set explicitly. (Per spec, expect only `method="lora"` and `epochs=1` set; the rest defaulted. Confirm against the live files.)

- [ ] **Step 2: Resolve the `warmup_steps=0` deviation**

`warmup_steps=0` is the one practical hyperparameter pinned away from its schema default (`100`). With `epochs=1` on a tiny dataset the run is only a handful of steps, so a 100-step warmup would never finish ramping — `warmup_steps=0` is arguably structurally necessary for a 1-epoch CI run. **Decision for the implementer:** keep `warmup_steps=0` and justify it explicitly in the Step 3 comment (it is a consequence of the CI epoch truncation, not a hidden retune). Do NOT remove the redundant `learning_rate=1e-4` in `_make_cfg` — it equals the schema default and removing it changes nothing; leave the file minimally touched. Do NOT change `epochs` (it stays truncated). If any OTHER practical hyperparameter (`lr_schedule`, `optimizer`, `max_grad_norm`, `peft.r/alpha/dropout/scope`) turns out to be pinned to a non-default, remove that pin so the default path runs.

- [ ] **Step 3: Add the CI-truncation comment**

At each config builder, add a comment stating that `epochs` is truncated to 1 purely for CI runtime while all other hyperparameters (lr / schedule / optimizer / PEFT config) are the real schema defaults, so the green run genuinely guards the default path. Example:

```python
    # Default-path guard: every practical hyperparameter (learning_rate,
    # lr_schedule, optimizer, max_grad_norm, peft.r/alpha/dropout/scope) is left
    # at its schema default so this end-to-end run exercises the real default
    # training path. `epochs` is truncated (to 1) for CI runtime — NOT run to
    # convergence — and `warmup_steps=0` follows from that truncation (a 100-step
    # warmup never completes in a 1-epoch tiny-dataset run). See
    # docs/defaults-provenance.md "Reference Training Profile" for the shipped
    # epochs default.
```

- [ ] **Step 4: Run the integration tests green (acceptance criterion 5)**

Run: `uv run pytest -o "addopts=" tests/integration/test_train_then_eval.py tests/integration/test_train_end_to_end.py -v`
Expected: PASS (all parametrizations; `tensorboard` is `importorskip`-gated). If reverting an override (Step 2) breaks a test, debug per `superpowers:systematic-debugging` — the default path must run green; if a default genuinely cannot run on CPU, restore the minimal override and justify it in the comment.

- [ ] **Step 5: Lint + commit**

Run: `uv run ruff check tests/integration/test_train_then_eval.py tests/integration/test_train_end_to_end.py && uv run ruff format --check tests/integration/test_train_then_eval.py tests/integration/test_train_end_to_end.py`
Expected: PASS.

```bash
git add tests/integration/test_train_then_eval.py tests/integration/test_train_end_to_end.py
git commit -m "test(integration): exercise default hyperparam path; document CI epoch truncation"
```

### Task 2.5: Final acceptance + PR

**Files:** none until the PR.

- [ ] **Step 1: Re-run all six acceptance checks (spec §"Verification & Acceptance")**

```bash
# (1) inline coverage — every trust-bearing default tagged; spot-check parity
grep -rnE '# (cite|tbd):' src/custom_sam_peft/config src/custom_sam_peft/data src/custom_sam_peft/presets.py src/custom_sam_peft/models/losses/presets.py | head
# (2) zero citation-needed
grep -c "citation needed" src/custom_sam_peft/models/losses/presets.py   # -> 0
# (3) markdownlint gate (exact CI command)
npx --yes markdownlint-cli2 --config .config/markdownlint-cli2.jsonc "**/*.md" "#node_modules"
# (5) integration tests green
uv run pytest -o "addopts=" tests/integration/test_train_then_eval.py tests/integration/test_train_end_to_end.py -q
# (6) no numeric default changed except epochs
git diff main -- src/custom_sam_peft | grep -E '^\+' | grep -vE '#|^\+\+\+|\.md'
```

Expected: (2) prints `0`; (3) PASS; (5) PASS; (6) the only value-changing `+` line is the `epochs` default substitution from Task 2.3 (all other `+` lines are comments). Manually confirm (1) tag/row parity and (4) the profile + epochs alignment are documented.

- [ ] **Step 2: Full pytest with the coverage gate (CI parity)**

Run: `uv run pytest`
Expected: PASS at ≥80% coverage (no source logic added; this run mirrors CI's `Test` job).

- [ ] **Step 3: Open the PR**

```bash
gh pr create \
  --title "docs: literature-cite every default + align epochs to reference profile (#120)" \
  --body $'Closes #120.\n\nDeliverable 1: inline # cite:/# tbd: tags on every trust-bearing default across config/_internal.py, config/schema.py, data/aug_presets.py, data/channel_semantics.py, data/transforms.py, presets.py, and the config_full.yaml template; loss-preset # citation needed cells firmed up (grep -c == 0); central docs/defaults-provenance.md index with verified quotes/DOIs.\n\nDeliverable 2: convergence-anchored Reference Training Profile; shipped epochs default raised 10->160 (init_cmd.py + setup_wizard.py), anchored to SAMed (arXiv:2304.13785, 160-epoch convergence on a small LoRA-SAM dataset). Runtime stays unverified (# tbd: #193 — no citable T4 per-step figure; the 30-min budget was dropped in favor of convergence, accuracy >> speed). GPU-test docstring de-conflated (its epochs=25 is a 2-image overfit budget, not the production default). Integration tests document CI epoch truncation while exercising the real default hyperparameter path.\n\nSpec: docs/superpowers/specs/2026-05-30-cite-defaults-audit-design.md\nPlan: docs/superpowers/plans/2026-05-30-cite-defaults-audit-plan.md\n\nFollow-ups filed: #191 (umbrella # tbd: tracker), #192 (CI no-uncited-default hook), #193 (empirical T4 confirmation), #195 (2-image overfit GPU smoke-test speed/convergence).' \
  --label docs --assignee @me
```

Expected: PR created, linking spec + plan and the four follow-up issues (#191/#192/#193/#195).

---

## Self-Review

**Spec coverage:**

- Deliverable 1 inline provenance + central index → Tasks 1.2–1.10 (one task per in-scope file + the central doc scaffolded in 1.2). ✓
- Tag taxonomy §1.1 + doc row schema §1.2 → frozen in Orientation + Task 1.2. ✓
- §1.3 interface contract (row schema + `epochs` row) → Phase-1 EXPOSES list + Task 1.4 Step 8 (verbatim stub) + Phase-2 CONSUMES list. ✓
- §1.4 per-file audit plan → Tasks 1.3–1.10 reference real file:symbol + line numbers + expected tag classes. ✓
- §1.5 deep verification → research steps precede tagging in every audit task; quotes/DOIs captured into rows at tag time. ✓
- Loss-preset firm-up (§1.4 / acceptance 2) → Task 1.10 with `grep -c == 0` check. ✓
- Deliverable 2 (§2.1–§2.4) → Tasks 2.1–2.4 (analytical derivation, profile section, template epochs default, GPU-test docstring, integration-test comment). ✓
- Three follow-up issues (§"Follow-up Issues") with umbrella-FIRST ordering → Task 1.1 (umbrella created first, number captured, used by all `# tbd:` tags). ✓
- Verification & Acceptance (6 criteria) → Task 1.11 + Task 2.5 run each criterion explicitly; markdownlint uses the exact CI command; coverage-gate bypass via `-o "addopts="` documented. ✓
- Non-Goals (no re-tuning, structural defaults index-only, examples out of scope) → enforced in Orientation + acceptance criterion 6 check. ✓

**Placeholder scan:** No "TBD/TODO/handle edge cases" in plan text. `<UMBRELLA>`, `<T4_CONFIRM>`, `EPOCHS_VALUE`, and per-value quotes are *runtime-captured values* (an issue number not yet created; an analytically-derived integer; primary-source quotes the implementer must verify rather than the planner fabricate) — each has an explicit capture step, which is the correct pattern for a provenance audit, not a plan placeholder.

**Type/name consistency:** Symbol names and line numbers match the read source (`MatcherWeights.lambda_mask`, `PEFTConfig.r/alpha/dropout`, `TrainHyperparams.epochs` required field, `_HED_FROM_RGB_MATRIX`, `forward_only_factor`, `_optimizer_bytes`). Doc row schema (six columns) is identical everywhere it appears. The `epochs` row stub text is identical in Task 1.4 Step 8 and the Phase-1 EXPOSES contract.

## Resolved ambiguities / coverage notes (for the orchestrator)

- **Loss PRESET_TABLE length:** `aug_presets.py` has 12 literal cells (verify `== 12`); `models/losses/presets.py` has 15 (12 + 3 microscopy aliases, verify `== 15`). Both verification commands reflect this.
- **`negatives_per_image` mismatch:** the template ships `4` (line 32) while the schema default is `0` (line 231). Task 1.9 records the template's `4` and cross-links the schema row + its field-description rationale, rather than treating it as a discrepancy to "fix" (Non-Goal: no re-tuning).
- **`$epochs` is a substitution, not a literal:** the template keeps the `$epochs` placeholder; the `init` flow supplies the integer. Task 2.3 changes the *substitution default*, not the template text — confirmed against `config_full.yaml` line 52.
- **No `ADAMW_STATE_MULT` symbol:** the AdamW-state cost is the inline `*4` in `presets.py:_optimizer_bytes` (lines 160–163); Task 1.8 tags that literal and notes the issue's mislabel.
- **Markdownlint MD013 is already OFF** (`.config/markdownlint-cli2.jsonc`), so the spec's "confirm MD013 handling before authoring wide tables" risk is resolved: wide tables need no per-block disable.
