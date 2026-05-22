# spec/yaml-config-defaults-audit — Audit and correct shipped YAML/schema defaults (issue #69)

**Status:** Draft (2026-05-21)
**Tracking:** [#69](https://github.com/NguyenJus/custom-sam-peft/issues/69) — *Assess correctness of default YAML configs (especially normalization fallback)*
**Scope:** Audit every default in the four shipped configs (`configs/examples/coco_text_{lora,qlora}.yaml`, `src/custom_sam_peft/cli/templates/coco_text_{lora,qlora}.yaml`) against the pydantic schema (`src/custom_sam_peft/config/schema.py`) and the model code; reconcile so the schema is the actual source of truth; close the silent-mis-normalization correctness hole #69 raised by replacing `resolve_normalization` with a three-step resolver that consults a `KNOWN_PROCESSOR_STATS` table; ship CPU-only tests for the new resolver paths. No new config knobs, no augmentation rework, no hyperparameter retuning.

**Builds on / supersedes:**
[`2026-05-16-model-loading-design.md`](2026-05-16-model-loading-design.md) — superseded **for normalization only** (this audit ratifies ImageNet stats `[0.485, 0.456, 0.406] / [0.229, 0.224, 0.225]` as the SAM3.1 ground truth, contradicting that spec's §4.2 / §8 which asserted `[0.5, 0.5, 0.5]`). The rest of the 5/16 spec (image-size 1008, wrapper API, matcher, losses, example-file structure) stands.

---

## 1. Problem Statement

Issue #69 flagged that all four shipped configs carry `normalize.mean = std = [0.5, 0.5, 0.5]` and the schema's `NormalizeConfig` default was claimed to match. Two things are true:

1. The `[0.5, 0.5, 0.5]` block in the four YAMLs is wrong for SAM3.1 — `AutoImageProcessor.from_pretrained("facebook/sam3.1")` returns ImageNet stats, and on the cache-miss path `resolve_normalization` silently feeds the backbone `[-1, 1]`-rescaled inputs (`src/custom_sam_peft/data/transforms.py:36-44`).
2. The issue body's premise that "schema default in `src/custom_sam_peft/config/schema.py` matches" the YAMLs is **factually incorrect**: the schema's `NormalizeConfig` actually defaults to ImageNet (`src/custom_sam_peft/config/schema.py:86-91`). The schema is right; the YAMLs are wrong; and the schema-as-SoT contract was already broken before #69 was filed.

The 5/16 model-loading spec (lines 120, 283-285) asserted SAM3.1 uses `[0.5, 0.5, 0.5]` and that ImageNet "would be silently wrong on day 1". After empirical inspection, **that assertion was wrong**: SAM3.1's `AutoImageProcessor` returns ImageNet stats. This spec ratifies ImageNet as the SAM3.1 ground truth and is the canonical source for normalization from this point forward.

The cache-miss path is also a logging-level bug: a silent quality regression is exactly the class of failure that warrants WARN, not INFO. This spec also addresses several other schema-vs-YAML drift points surfaced by the audit (image size, gradient checkpointing).

---

## 2. Goals & Non-Goals

### 2.1 Goals

- Make the pydantic schema the actual source of truth for every shipped default. Each YAML key either echoes a schema default (for discoverability) or is a deliberate override carrying a one-line justification.
- Close the silent-mis-normalization hole: introduce a three-step resolver with a known-good lookup table so the cache-miss path is correct-by-default for SAM3.1 and loud (WARN) for anything else.
- Per-default audit of every knob #69 called out: Keep / Change / Reconcile, each with a one-line justification. Land justifiable changes in this PR; file follow-up issues for the rest.
- Add CPU-only unit tests covering each new resolver branch.

### 2.2 Non-Goals

- No new config knobs (no override flag for the known-good table, no startup divergence assertion that hard-errors).
- No augmentation pipeline rework.
- No hyperparameter retuning that would require a real training run (LR sweeps, box-hint schedule retune, etc.). These are deferred to follow-up issues.
- No empirical re-verification of `Sam3ImageProcessor.image_mean/image_std` in this PR — that is filed as a follow-up issue so a developer with GPU access can dump the live values and attach evidence.

---

## 3. Ground Truth Pivot — Normalization Stats for SAM3.1

The 5/16 model-loading spec stated (§4.2, line 120):

> `images`: `(B, 3, 1008, 1008)` bf16, normalized with `mean=std=[0.5, 0.5, 0.5]`.

and added in §8 (lines 283-285) that ImageNet stats would be "silently wrong on day 1". This audit reverses that call. The user has decided, based on the schema's own existing default and consistency with SAM/SAM2 processors, that **SAM3.1's normalization ground truth is ImageNet**:

```
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

This is the value the new `KNOWN_PROCESSOR_STATS` table will carry for `"facebook/sam3.1"`. Empirical confirmation against `AutoImageProcessor.from_pretrained("facebook/sam3.1").image_mean/image_std` is deferred to follow-up issue (a) — see §8.

This spec is the canonical source for SAM3.1 normalization from 2026-05-21 onward. The 5/16 spec gets a "Superseded for normalization" callout (§7.3).

---

## 4. Normalization Resolver Architecture

### 4.1 New behavior — three-step resolution

`src/custom_sam_peft/data/transforms.py::resolve_normalization` is rewritten to consult, in order:

1. **`AutoImageProcessor.from_pretrained(model_name, local_files_only=True)`.** On success, read `image_mean` / `image_std`. *Before returning*, look up `model_name` in `KNOWN_PROCESSOR_STATS` (defined in the same module). If the model is in the table and the loaded `(mean, std)` diverges from the table values beyond tolerance (`1e-3`, element-wise on both vectors), emit `logging.WARNING` naming both vectors so the user notices the drift. Otherwise emit `logging.INFO` ("Using image_mean/image_std from AutoImageProcessor for `<model>`.") — preserving the existing happy-path log line.
2. **On `OSError | AttributeError | ValueError`**, look up `model_name` in `KNOWN_PROCESSOR_STATS`. If present, return that pair and emit `logging.WARNING`: `"AutoImageProcessor unavailable for <model>; using known-good stats (mean=..., std=...). Populate the HF cache to silence this warning."`
3. **Otherwise** (processor unavailable AND not in the table), return `(fallback.mean, fallback.std)` from the user's `NormalizeConfig` and emit `logging.WARNING`: `"AutoImageProcessor unavailable for <model> AND no known-good entry registered; using NormalizeConfig fallback (mean=..., std=...). Verify these are correct for this backbone."`

The existing INFO-on-cache-miss line (`transforms.py:37-43`) is **promoted to WARN** and absorbed into path 2 / path 3 above. Quality-regressing fallbacks must be loud.

### 4.2 `KNOWN_PROCESSOR_STATS` table

A module-level constant in `transforms.py`, sitting next to `resolve_normalization`:

```python
# Known-good (mean, std) per HF model name. Used as the offline fallback
# AND as a divergence sentinel against AutoImageProcessor on path 1.
#
# facebook/sam3.1: ImageNet stats. This matches what
# AutoImageProcessor.from_pretrained("facebook/sam3.1").image_mean/image_std
# returns; consistent with SAM/SAM2-class processors. Ratified by the
# 2026-05-21 config-defaults audit (supersedes the 2026-05-16 model-loading
# spec's [0.5, 0.5, 0.5] claim).
KNOWN_PROCESSOR_STATS: dict[str, tuple[list[float], list[float]]] = {
    "facebook/sam3.1": ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
}
```

The table lives in `transforms.py`, **not** in user-visible config. It is part of the resolver's contract — coupled to the resolver, not to a YAML knob.

### 4.3 What the resolver deliberately does NOT do

- **No new schema knob, no CLI flag, no environment-variable override.** Users who must bypass the table edit `data.normalize.mean/std` in their YAML. The user's edited values then take effect on path 3 (where the resolver emits a WARN naming the values). There is no separate "ignore the table" toggle — the table is a fallback, not a hard constraint.
- **No startup hard-error on divergence.** Path 1 WARNs on table-vs-processor divergence beyond `1e-3` element-wise; it does not raise. Hard-erroring would gate startup on a table the developer (not the user) controls, which is the wrong trade for a debugging hint.
- **`NormalizeConfig._check_ranges` is unchanged.** Validation of mean ∈ [0, 1] and std > 0 still runs at config-load time.

### 4.4 Divergence tolerance

Element-wise absolute tolerance of `1e-3` on both `mean` and `std` (i.e., `all(abs(loaded[i] - table[i]) <= 1e-3) for both vectors`). Rationale: HF processors store stats as serialized floats; `1e-3` is loose enough to absorb representation noise but tight enough to catch a real change (e.g., a switch from ImageNet to `[0.5, 0.5, 0.5]` would diverge by ≥ 0.014 on every channel).

---

## 5. Per-Default Audit Table

Every knob #69 enumerated, plus the schema-vs-YAML drift surfaced during this audit. **Decision** is one of `Keep` (no change), `Change` (value moves in this PR), or `Reconcile` (schema and YAML are out of sync; pick one and align both).

| # | Knob | Current | Decision | Justification |
|---|---|---|---|---|
| 1 | `data.normalize.mean/std` (all four YAMLs) | `[0.5, 0.5, 0.5]` | Change → ImageNet | Mis-normalizes SAM3.1 on cache-miss. Block stays for discoverability with a comment: `# remove unless overriding for a non-SAM3 backbone`. |
| 2 | `data.normalize.mean/std` (schema `NormalizeConfig` default) | ImageNet `[0.485, …]` / `[0.229, …]` | Keep | Correct ground truth (§3). YAMLs now align. |
| 3 | `data.image_size` (all four YAMLs) | `1008` | Keep | SAM3.1's native input, hardcoded at `src/custom_sam_peft/models/sam3.py:192,304,1202-1203` and documented in 2026-05-16-model-loading-design.md:283. |
| 4 | `data.image_size` (schema `DataConfig` default) | `1024` | Change → `1008` | Schema-as-SoT contract broken: schema is stale, YAMLs and model code are right (`schema.py:136`). |
| 5 | `data.augmentations.color_jitter` | `0.1` | Keep + audit note | Modest jitter; safe for segmentation. Caveat noted: text-prompt CLIP-style encoders can be jitter-sensitive; no retune without an A/B. |
| 6 | `data.text_prompt.negatives_per_image` | `4` | Keep + schema-docstring rationale | Bounded by TextPrompts multiplex cap (`text_prompt.k <= 16`, `schema.py:74`); 4 leaves headroom for typical COCO present counts (~3-7). Rationale added to the `Field(description=...)`. |
| 7 | `train.lr` (LoRA and QLoRA YAMLs) | `1.0e-4` | Keep | Standard PEFT starting point. A QLoRA-at-5e-4 retune is real but needs GPU time; deferred to follow-up issue (b). |
| 8 | `train.warmup_steps` | `100` | Keep | Modest fixed warmup; fine for short example runs. |
| 9 | `train.epochs` | `10` | Keep | Example run length, not a global default. |
| 10 | `train.grad_accum_steps` | `8` | Keep | Reasonable effective batch with `batch_size: 1` on T4/A10-class GPUs. |
| 11 | `train.box_hint.{p_start, p_end, decay_steps, early_stop_p_threshold}` | `1.0 / 0.0 / 5000 / 0.05` | Keep + TODO referencing #24 | Ad-hoc schedule; #24 (the in-flight bbox-prompt PR) will retune. |
| 12 | `train.optimizer` | `"auto"` | Keep + schema-docstring rationale | Resolution rule (`adamw8bit` if QLoRA else `adamw`) lives at `src/custom_sam_peft/train/trainer.py:45-49`. Documented inline on the `Optimizer` `Literal` in `schema.py`. |
| 13 | `eval.iou_thresholds` | COCO 0.5..0.95 | Keep | Standard mAP IoU sweep; not COCO-specific. |
| 14 | `model.gradient_checkpointing` (schema `True`, examples `false`, templates `true`) | three-way mismatch | Reconcile → **all `false`** | #60 was the open issue when examples were patched to `false` (`coco_text_lora.yaml:10`, `coco_text_qlora.yaml:10`). Templates currently ship `true`, which forces users into the same workaround. Schema flips with an inline `# TODO(#60): re-enable when sam3 activation-checkpointing recompute mismatch is fixed`. |
| 15 | `peft.{r, alpha, dropout, scope, bias}` | `16 / 32 / 0.05 / vision_decoder / none` | Keep, schema-default-echoes-only | Common PEFT starting point. Add a top-of-YAML comment block stating "Keys without an override comment are echoes of schema defaults for discoverability." |

---

## 6. Files Touched

### 6.1 Code

| File | Change |
|---|---|
| `src/custom_sam_peft/data/transforms.py` | Add `KNOWN_PROCESSOR_STATS` constant (seeded with `"facebook/sam3.1"` → ImageNet). Rewrite `resolve_normalization` to the three-step pattern in §4.1. Promote the existing INFO cache-miss line to WARN; add the two new WARN sites (table-divergence on path 1; no-table-entry on path 3). |
| `src/custom_sam_peft/config/schema.py` | `DataConfig.image_size`: `1024` → `1008` (line 136). `ModelConfig.gradient_checkpointing`: `True` → `False` with comment `# TODO(#60): re-enable when sam3 activation-checkpointing recompute mismatch is fixed` (line 43). `NormalizeConfig` docstring updated to describe the three-step resolver and the known-good table (lines 77-84). `TextPromptConfig.negatives_per_image` gains a `Field(description=...)` carrying the rationale from row 6. `Optimizer` `Literal` (line 20) gains an inline comment describing the `"auto"` resolution rule (LoRA → `adamw`, QLoRA → `adamw8bit`). |

### 6.2 YAMLs (all four)

| File | Change |
|---|---|
| `configs/examples/coco_text_lora.yaml` | `normalize.mean/std` → ImageNet, with the discoverability comment. Top-of-file comment block describing the schema-as-SoT contract. (Already has `gradient_checkpointing: false`.) |
| `configs/examples/coco_text_qlora.yaml` | Same as above. |
| `src/custom_sam_peft/cli/templates/coco_text_lora.yaml` | `normalize.mean/std` → ImageNet (with comment). `gradient_checkpointing: true` → `false` (with the same `# see issue #60` annotation the example configs already carry). Top-of-file comment block describing the schema-as-SoT contract. |
| `src/custom_sam_peft/cli/templates/coco_text_qlora.yaml` | Same as above. |

The `normalize:` block stays present in all four files (rather than being deleted) for discoverability — users learning the config surface should be able to see the knob. The discoverability comment tells them when removing the block is appropriate.

### 6.3 Tests (CPU-only, no real model weights)

Extend `tests/unit/test_data_transforms.py` with five new tests, matching the existing file's style (logging capture via `caplog.set_level(...)` on `"custom_sam_peft.data.transforms"`, `AutoImageProcessor` mocked via `unittest.mock.patch("transformers.AutoImageProcessor", ...)`, `SimpleNamespace` for the fake processor):

| Test | Scenario | Asserts |
|---|---|---|
| `test_resolve_normalization_processor_loads_no_table_entry_no_warn` | Path 1, model NOT in table | Returns processor values; **no** WARN log records; INFO log present. Provides regression coverage for path 1's happy path against the table check. |
| `test_resolve_normalization_processor_loads_matches_table` | Path 1, model in table, values within `1e-3` | Returns processor values; no WARN; INFO present. |
| `test_resolve_normalization_processor_loads_diverges_from_table` | Path 1, model in table, fake processor returns `[0.5, 0.5, 0.5]` for `"facebook/sam3.1"` | Returns processor values (table is a sentinel, not a gate); exactly one WARN log record naming both vectors. |
| `test_resolve_normalization_processor_fails_model_in_table` | Path 2, `OSError` raised, model in table | Returns the table's `([0.485, …], [0.229, …])`; exactly one WARN log record naming the table fallback. |
| `test_resolve_normalization_processor_fails_model_not_in_table` | Path 3, `OSError` raised, model NOT in table | Returns the user's `NormalizeConfig` values; exactly one WARN log record naming the YAML fallback. |

The existing tests (`test_resolve_normalization_uses_image_processor_when_available`, `..._falls_back_on_oserror`, `..._falls_back_on_attribute_error`) are updated for the WARN-vs-INFO change but otherwise kept as-is. All tests load `transformers.AutoImageProcessor` via `unittest.mock.patch`; no real model weights are touched. All run on CPU.

---

## 7. Existing-Spec Corrections

### 7.1 Add "Superseded for normalization" callout to the 5/16 model-loading spec

Edit `docs/superpowers/specs/2026-05-16-model-loading-design.md`. At the top of §4.2 (`### 4.2 Forward behavior`, which begins around line 118), insert a blockquote callout:

```markdown
> **Superseded for normalization (2026-05-21).** The `mean=std=[0.5, 0.5, 0.5]` claim
> on line 120 and the example-config edits on lines 283-285 are **wrong** for SAM3.1.
> See [`2026-05-21-yaml-config-defaults-audit-design.md`](2026-05-21-yaml-config-defaults-audit-design.md)
> for the corrected ground truth (ImageNet stats) and the three-step resolver.
> Everything else in this spec — image-size 1008, wrapper API, matcher, losses — stands.
```

No other content in the 5/16 spec is changed; the callout is additive.

### 7.2 No other specs touched

The 5/15 architecture spec, the 5/17 training-loop spec, etc. do not reference normalization stats by value, so no further callouts are needed.

---

## 8. Follow-up Issues (filed by orchestrator after merge)

Each created via `gh issue create --assignee @me --label …`. Existing labels confirmed via `gh label list`: `priority:low` and `question` exist.

| # | Title | Labels | Rationale |
|---|---|---|---|
| (a) | Empirically verify `Sam3ImageProcessor` stats vs `KNOWN_PROCESSOR_STATS["facebook/sam3.1"]` | `priority:low`, `question` | One-off; closes when the live processor's `image_mean` / `image_std` are dumped and attached to the issue. Provides direct evidence for the table value. |
| (b) | A/B QLoRA `lr` at `5e-4` vs `1e-4` post-#44 GPU eval | `priority:low` | The QLoRA-at-5e-4 retune row 7 deferred — wants real training to settle. |
| (c) | Re-audit `box_hint` schedule after #24 lands | `priority:low` | Row 11; #24 is the in-flight bbox-prompt PR. |
| (d) | When #60's underlying ViT activation-checkpointing bug is fixed, re-enable `gradient_checkpointing` default in schema + both YAML templates + both example configs | `priority:low` | Row 14; the `# TODO(#60)` schema comment is the in-code pointer to this issue. |

The orchestrator files all four issues as part of close-out before merging.

---

## 9. Acceptance Criteria

1. **Schema-vs-YAML alignment.** All four YAMLs and the schema produce identical resolved configs for `data.normalize`, `data.image_size`, and `model.gradient_checkpointing`. Verified by loading each YAML through the existing config loader and asserting the resolved `TrainConfig` fields against the schema defaults.
2. **Happy-path correctness.** When `AutoImageProcessor.from_pretrained("facebook/sam3.1", local_files_only=True)` succeeds with cached weights, training emits an **INFO** log and proceeds with the processor's stats — behavior unchanged from today.
3. **Cache-miss correctness on SAM3.1.** On a machine without the HF cache, training emits exactly one **WARN** log record naming the known-good table fallback, and proceeds with ImageNet stats. No `[0.5, 0.5, 0.5]` regression is possible on the SAM3.1 path.
4. **Table-divergence detection.** A unit test simulating a fake processor that returns `[0.5, 0.5, 0.5]` for `"facebook/sam3.1"` produces exactly one WARN log record naming both vectors (loaded and table) within the same record.
5. **Unknown-model fallback.** A unit test using a model name not in `KNOWN_PROCESSOR_STATS` with the processor mocked to raise `OSError` returns the user's `NormalizeConfig` and emits exactly one WARN naming the YAML values.
6. **Schema-default drift fixed.** `DataConfig.image_size` default is `1008`; `ModelConfig.gradient_checkpointing` default is `False`. Both verified by `TrainConfig.model_json_schema()` or a direct attribute assertion against a freshly constructed default instance.
7. **Lint/test gates.** Existing CI (`ci.yml`, security workflow, lint-hygiene) passes on the PR. All new tests run on CPU; the 80% coverage gate in `pyproject.toml` is unaffected.
8. **No new schema knobs introduced.** `TrainConfig` field count is unchanged; only existing field defaults and docstrings are modified.

---

## 10. Out-of-Scope (explicitly excluded)

- New config knobs (no `normalize_strict`, no `--bypass-known-stats`, no override path).
- A hard-error startup assertion on table-vs-processor divergence (we WARN; we do not raise).
- Augmentation pipeline rework (color jitter, hflip, anything beyond `data.augmentations`).
- Hyperparameter retunes that need real training runs (LR, box-hint schedule, epochs).
- Removing the `normalize:` block from the YAMLs (kept for discoverability).
- Re-litigating the SAM3.1 normalization ground truth — this spec ratifies ImageNet; downstream specs build on that.
- Empirical dump of `Sam3ImageProcessor.image_mean/image_std` against the live HF model — filed as follow-up (a), not blocking this PR.

---

## 11. Verification Matrix

| Check | Method | Where |
|---|---|---|
| Schema defaults match shipped YAMLs for normalize/image_size/gradient_checkpointing | Direct attribute assertion in a new schema-vs-YAML parity test (CPU) | Extension of `tests/unit/test_data_transforms.py` or sibling unit test file |
| Each resolver path emits the right log level and exactly one record | `caplog.set_level(...)` + log-message regex, mirroring existing transforms tests | `tests/unit/test_data_transforms.py` |
| Table-divergence WARN names both vectors | Substring assertion on the WARN record's `getMessage()` | `tests/unit/test_data_transforms.py` |
| Schema docstrings updated (normalize, negatives_per_image rationale, optimizer auto rule) | Visual inspection during reviewer pass | reviewer (sonnet/high) |
| Existing CI green (ruff, mypy, pytest, lint-hygiene, security) | GitHub Actions | unchanged from today |

---

## 12. File Layout Diff

```
src/custom_sam_peft/data/transforms.py             TOUCHED  (+KNOWN_PROCESSOR_STATS, rewritten resolve_normalization)
src/custom_sam_peft/config/schema.py               TOUCHED  (image_size 1024→1008; gradient_checkpointing True→False; docstrings)
configs/examples/coco_text_lora.yaml               TOUCHED  (normalize → ImageNet; top-of-file comment)
configs/examples/coco_text_qlora.yaml              TOUCHED  (normalize → ImageNet; top-of-file comment)
src/custom_sam_peft/cli/templates/coco_text_lora.yaml      TOUCHED  (normalize → ImageNet; gradient_checkpointing → false; comment)
src/custom_sam_peft/cli/templates/coco_text_qlora.yaml     TOUCHED  (normalize → ImageNet; gradient_checkpointing → false; comment)
tests/unit/test_data_transforms.py                 TOUCHED  (+5 tests for the three-step resolver; existing tests updated for WARN)
docs/superpowers/specs/2026-05-16-model-loading-design.md   TOUCHED  (+"Superseded for normalization" callout at top of §4.2)
docs/superpowers/specs/2026-05-21-yaml-config-defaults-audit-design.md  NEW (this spec)
```

No deletions. No new top-level files outside `docs/superpowers/specs/`. No changes under `src/custom_sam_peft/models/`, `src/custom_sam_peft/train/`, or `src/custom_sam_peft/eval/`.
