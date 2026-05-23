# YAML Config Defaults Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Spec:** [`docs/superpowers/specs/2026-05-21-yaml-config-defaults-audit-design.md`](../specs/2026-05-21-yaml-config-defaults-audit-design.md)
**Issue:** [#69](https://github.com/NguyenJus/custom-sam-peft/issues/69) — *Assess correctness of default YAML configs (especially normalization fallback)*
**Branch:** `audit/config-defaults-69` (worktree at `/home/justin/projects/custom-sam-peft/.worktrees/audit-config-defaults-69`)

**Goal:** Reconcile the four shipped YAMLs and the pydantic schema so the schema is the actual source of truth for every default; close the silent-mis-normalization correctness hole #69 raised by rewriting `resolve_normalization` to consult a `KNOWN_PROCESSOR_STATS` table (WARN on cache miss, WARN on table divergence); add CPU-only unit tests for the new resolver paths; insert a "Superseded for normalization" callout into the 5/16 model-loading spec; file four follow-up issues post-merge.

**Architecture:** Three small, file-disjoint code changes (`transforms.py`, `schema.py`, four YAMLs) plus one test-file extension and one spec-callout edit. All changes land on the existing `audit/config-defaults-69` branch in a single PR. The orchestrator files four follow-up GitHub issues *after* the PR merges. No new config knobs, no augmentation rework, no hyperparameter retunes.

**Tech Stack:** Python 3.12, pydantic v2, `transformers.AutoImageProcessor`, pytest + `caplog`, `unittest.mock.patch`, `uv` + `ruff` + `mypy`, `gh` CLI.

---

## File Map

**Modified files:**

```
src/custom_sam_peft/data/transforms.py                           TOUCHED (+KNOWN_PROCESSOR_STATS, rewritten resolve_normalization)
src/custom_sam_peft/config/schema.py                             TOUCHED (image_size 1024→1008; gradient_checkpointing True→False; docstrings; Field on negatives_per_image; Optimizer comment)
configs/examples/coco_text_lora.yaml                             TOUCHED (normalize → ImageNet; SoT comment; discoverability comment)
configs/examples/coco_text_qlora.yaml                            TOUCHED (same as above)
src/custom_sam_peft/cli/templates/coco_text_lora.yaml            TOUCHED (normalize → ImageNet; gradient_checkpointing true→false; SoT comment; discoverability comment)
src/custom_sam_peft/cli/templates/coco_text_qlora.yaml           TOUCHED (same as above)
tests/unit/test_data_transforms.py                               TOUCHED (+5 tests; +1 schema-vs-YAML parity test; 3 existing tests updated for WARN-vs-INFO)
docs/superpowers/specs/2026-05-16-model-loading-design.md        TOUCHED (+"Superseded for normalization" callout at the top of §4.2)
```

No new files. No deletions. No moves. No source under `src/custom_sam_peft/models/`, `src/custom_sam_peft/train/`, or `src/custom_sam_peft/eval/` touched.

---

## Assumptions for the cold reader

1. **Working directory.** Every shell command below runs with `cwd = /home/justin/projects/custom-sam-peft/.worktrees/audit-config-defaults-69`. Use absolute paths when invoking external tools; use repo-relative paths inside the plan text.
2. **Tooling.** `uv` is on PATH. The repo uses `uv run …` for every Python entry point (ruff, mypy, pytest). Do NOT shell out to `python` directly — `uv run python …` ensures the project venv is used.
3. **GitHub labels confirmed.** `gh label list` was run during planning: `priority:low` (`#cccccc`, "Explicitly deferred") and `question` (`#d876e3`) both exist. No new label needs to be created.
4. **Schema source-of-truth contract.** Per spec §5 row 15, YAML keys without an `# override:` comment are echoes of schema defaults for discoverability. Keys with an override comment are deliberate divergence with a one-line justification.
5. **Tolerance.** `1e-3` element-wise on both mean and std vectors. `0.5` vs ImageNet diverges by ≥ 0.014 on every channel — well above the threshold.
6. **CPU-only tests.** `transformers.AutoImageProcessor` is mocked via `unittest.mock.patch("transformers.AutoImageProcessor", …)`. No real model weights are downloaded.
7. **Most recent tag.** `git describe --tags --abbrev=0` returns `v0.7.1` at plan-write time. The orchestrator computes the next version at release time, *not* in this plan. The plan deliberately does not bake a version number in.

---

## Parallelization opportunities (for orchestrator dispatch)

Phase 0 (pre-flight checks, no edits) blocks nothing — it is a guard.

Phases 1, 2, 3, and 4 are file-disjoint and can be fanned out in parallel via `superpowers:dispatching-parallel-agents`:

- **Phase 1** touches only `src/custom_sam_peft/data/transforms.py`.
- **Phase 2** touches only `src/custom_sam_peft/config/schema.py`.
- **Phase 3** touches only the four YAMLs (`configs/examples/coco_text_{lora,qlora}.yaml`, `src/custom_sam_peft/cli/templates/coco_text_{lora,qlora}.yaml`).
- **Phase 4** touches only `docs/superpowers/specs/2026-05-16-model-loading-design.md`.

**Phase 5** (tests) depends on Phases 1 *and* 2 *and* 3 because the new tests import the rewritten `resolve_normalization` (Phase 1), assert the new schema defaults (Phase 2), and parity-check the YAMLs against the schema (Phase 3). Serialize after the parallel cluster.

**Phase 6** (lint/format/type/test gate) depends on Phases 1–5.

**Phase 7** (PR) depends on Phase 6.

**Phase 8** (post-merge follow-up issues) depends on the PR landing.

Dependency graph:

```
Phase 0 (pre-flight)
   ├─→ Phase 1 (transforms.py)  ┐
   ├─→ Phase 2 (schema.py)      ├─→ Phase 5 (tests) → Phase 6 (gates) → Phase 7 (PR) → Phase 8 (issues)
   ├─→ Phase 3 (four YAMLs)     ┘
   └─→ Phase 4 (5/16 spec callout)
```

**Reviewer model floor:** sonnet/high for every phase. **Do not** dispatch any subagent at haiku — the resolver-rewrite and the schema-vs-YAML parity logic require judgment that haiku consistently truncates.

---

## Per-default audit coverage check

Every row of spec §5 maps to an explicit action below. This table is the cold-reader's index.

| Spec §5 row | Decision | Plan phase / step |
|---|---|---|
| 1 — `data.normalize.mean/std` (all four YAMLs) | Change → ImageNet | Phase 3, Steps P3-1..P3-4 |
| 2 — `data.normalize.mean/std` (schema) | Keep | **No code change.** Verified by Phase 5 parity test. |
| 3 — `data.image_size` (all four YAMLs) | Keep | **No code change.** Verified by Phase 5 parity test. |
| 4 — `data.image_size` (schema) | Change → `1008` | Phase 2, Step P2-1 |
| 5 — `data.augmentations.color_jitter` | Keep + audit note | **No code change.** Audit note recorded in spec §5 row 5; no plan action required. |
| 6 — `data.text_prompt.negatives_per_image` | Keep + schema-docstring rationale | Phase 2, Step P2-3 |
| 7 — `train.lr` (LoRA/QLoRA YAMLs) | Keep | **No code change.** Deferred to follow-up issue (b) — Phase 8. |
| 8 — `train.warmup_steps` | Keep | **No code change.** |
| 9 — `train.epochs` | Keep | **No code change.** |
| 10 — `train.grad_accum_steps` | Keep | **No code change.** |
| 11 — `train.box_hint.*` | Keep + TODO referencing #24 | **No code change.** Deferred to follow-up issue (c) — Phase 8. |
| 12 — `train.optimizer` | Keep + schema-docstring rationale | Phase 2, Step P2-4 |
| 13 — `eval.iou_thresholds` | Keep | **No code change.** |
| 14 — `model.gradient_checkpointing` | Reconcile → all `False`/`false` | Phase 2 (schema, Step P2-2) + Phase 3 (templates, Steps P3-3/P3-4). Examples already `false` (verified). |
| 15 — `peft.{r, alpha, dropout, scope, bias}` | Keep, schema-default-echoes only | Comment block added to all four YAMLs (Phase 3, all steps) — no value change. |

---

## Pre-flight (Phase 0)

**Model/effort:** sonnet / medium (one subagent, ~5 minutes).
**Parallel:** No. **Blocks:** all later phases.

- [ ] **Step P0-1: Confirm working tree state**

```bash
git -C /home/justin/projects/custom-sam-peft/.worktrees/audit-config-defaults-69 status
```

Expected: branch `audit/config-defaults-69`. Untracked file `docs/superpowers/specs/2026-05-21-yaml-config-defaults-audit-design.md` (the approved spec) and this plan file. No staged or modified files. If the spec is missing, halt — the plan cannot be executed without it.

- [ ] **Step P0-2: Stage the spec and the plan in a single commit so subsequent diffs are clean**

```bash
git -C /home/justin/projects/custom-sam-peft/.worktrees/audit-config-defaults-69 add \
  docs/superpowers/specs/2026-05-21-yaml-config-defaults-audit-design.md \
  docs/superpowers/plans/2026-05-21-yaml-config-defaults-audit.md
git -C /home/justin/projects/custom-sam-peft/.worktrees/audit-config-defaults-69 commit -m "docs: add yaml-config-defaults-audit spec + plan (#69)"
```

- [ ] **Step P0-3: Baseline unit-test sanity**

```bash
uv run pytest tests/unit/test_data_transforms.py tests/unit/test_config_examples.py tests/unit/test_config_loader.py -q
```

Expected: all green. If anything is red, halt — the baseline is broken and Phase 5 cannot be validated.

- [ ] **Step P0-4: Confirm `gh label list` shows the two labels the spec relies on**

```bash
gh label list --limit 100 | grep -E '^(priority:low|question)\b'
```

Expected: both lines appear. If either label is missing, halt — the orchestrator cannot file the follow-up issues with the labels the spec specifies (Phase 8).

---

## Phase 1: Rewrite `resolve_normalization` and add `KNOWN_PROCESSOR_STATS`

**Model/effort:** sonnet / high.
**Parallel:** Yes, with Phases 2, 3, 4.
**Spec:** §4.1, §4.2, §4.3, §4.4, §6.1 row 1.

**Files:**
- Modify: `src/custom_sam_peft/data/transforms.py` (lines 22-47 today)

**Goal:** Replace the existing two-branch `resolve_normalization` with the three-step resolver from spec §4.1, and add the module-level `KNOWN_PROCESSOR_STATS` constant from spec §4.2. The existing public signature (`resolve_normalization(model_name: str, fallback: NormalizeConfig) -> tuple[list[float], list[float]]`) is preserved. The existing INFO cache-miss log line is removed and replaced with the two new WARN sites described in spec §4.1 paths 2 and 3.

### Task 1a: Add `KNOWN_PROCESSOR_STATS` and rewrite `resolve_normalization`

- [ ] **Step P1-1: Apply the diff**

In `src/custom_sam_peft/data/transforms.py`, replace lines 22-47 (the current `resolve_normalization`) with the block below. Add the `KNOWN_PROCESSOR_STATS` constant just above it (between line 19's `_LOG = …` and the function).

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

# Element-wise absolute tolerance for table-vs-processor divergence detection
# on path 1. Loose enough to absorb float-serialization noise; tight enough
# to catch a real change (e.g. [0.5, 0.5, 0.5] diverges by >=0.014 per channel).
_STATS_DIVERGENCE_ATOL = 1e-3


def _stats_diverge(
    loaded: tuple[list[float], list[float]],
    table: tuple[list[float], list[float]],
) -> bool:
    """True if loaded and table differ on any channel of either vector beyond tolerance."""
    loaded_mean, loaded_std = loaded
    table_mean, table_std = table
    if len(loaded_mean) != len(table_mean) or len(loaded_std) != len(table_std):
        return True
    for lm, tm in zip(loaded_mean, table_mean, strict=True):
        if abs(lm - tm) > _STATS_DIVERGENCE_ATOL:
            return True
    for ls, ts in zip(loaded_std, table_std, strict=True):
        if abs(ls - ts) > _STATS_DIVERGENCE_ATOL:
            return True
    return False


def resolve_normalization(
    model_name: str, fallback: NormalizeConfig
) -> tuple[list[float], list[float]]:
    """Three-step resolution of (mean, std) for image normalization.

    1. Try ``AutoImageProcessor.from_pretrained(model_name, local_files_only=True)``.
       On success, read ``image_mean`` / ``image_std``. Before returning, look up
       ``model_name`` in :data:`KNOWN_PROCESSOR_STATS`. If the model is in the
       table and the loaded stats diverge beyond ``_STATS_DIVERGENCE_ATOL``,
       emit a WARNING naming both vectors. Otherwise emit INFO.
    2. On ``(OSError, AttributeError, ValueError)``, look up ``model_name`` in
       the table. If present, return the table values and emit WARNING.
    3. Otherwise (processor unavailable AND no table entry), return the user's
       ``fallback`` values and emit WARNING.

    Quality-regressing fallbacks must be loud; only path 1's happy path is INFO.
    """
    from transformers import AutoImageProcessor

    table_entry = KNOWN_PROCESSOR_STATS.get(model_name)

    try:
        proc = AutoImageProcessor.from_pretrained(model_name, local_files_only=True)  # type: ignore[no-untyped-call]
        mean = list(proc.image_mean)
        std = list(proc.image_std)
    except (OSError, AttributeError, ValueError):
        # Path 2 / Path 3
        if table_entry is not None:
            table_mean, table_std = table_entry
            _LOG.warning(
                "AutoImageProcessor unavailable for %r; using known-good stats "
                "(mean=%s, std=%s). Populate the HF cache to silence this warning.",
                model_name,
                table_mean,
                table_std,
            )
            return list(table_mean), list(table_std)
        _LOG.warning(
            "AutoImageProcessor unavailable for %r AND no known-good entry registered; "
            "using NormalizeConfig fallback (mean=%s, std=%s). Verify these are correct "
            "for this backbone.",
            model_name,
            fallback.mean,
            fallback.std,
        )
        return list(fallback.mean), list(fallback.std)

    # Path 1: processor loaded.
    if table_entry is not None and _stats_diverge((mean, std), table_entry):
        table_mean, table_std = table_entry
        _LOG.warning(
            "AutoImageProcessor for %r returned stats (mean=%s, std=%s) that diverge "
            "from KNOWN_PROCESSOR_STATS (mean=%s, std=%s) beyond tolerance %g. "
            "Using processor values; update the table if this divergence is expected.",
            model_name,
            mean,
            std,
            table_mean,
            table_std,
            _STATS_DIVERGENCE_ATOL,
        )
    else:
        _LOG.info(
            "Using image_mean/image_std from AutoImageProcessor for %r.", model_name
        )
    return mean, std
```

**Key design notes (do not modify):**
- The table lookup happens *once* up front (`table_entry = KNOWN_PROCESSOR_STATS.get(model_name)`) so paths 2/3 don't repeat the lookup.
- Path 1 logs WARN on divergence but **still returns the processor values** — the table is a sentinel, not a gate. This matches spec §4.3 ("No startup hard-error on divergence").
- The two cache-miss branches (path 2 and path 3) BOTH emit WARN — quality-regressing fallbacks must be loud (spec §4.1 last paragraph).
- The existing INFO line `"Using image_mean/image_std from AutoImageProcessor for %r."` is preserved verbatim for path 1's happy path.
- `NormalizeConfig._check_ranges` is unchanged (spec §4.3 final bullet) — this file does not touch `schema.py`.

### Task 1b: Verify the rewrite compiles and the module imports

- [ ] **Step P1-2: Import check**

```bash
uv run python -c "from custom_sam_peft.data.transforms import resolve_normalization, KNOWN_PROCESSOR_STATS; print(KNOWN_PROCESSOR_STATS)"
```

Expected: prints `{'facebook/sam3.1': ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])}` and exits 0.

- [ ] **Step P1-3: Commit**

```bash
git add src/custom_sam_peft/data/transforms.py
git commit -m "feat(data): three-step resolve_normalization with KNOWN_PROCESSOR_STATS (#69)"
```

**Reviewer focus (sonnet/high):**
- The five logging sites are spelled exactly as in spec §4.1.
- Path 1 returns processor values on divergence (NOT table values).
- No new public symbol beyond `KNOWN_PROCESSOR_STATS` is exported (the helper `_stats_diverge` and constant `_STATS_DIVERGENCE_ATOL` are module-private).

---

## Phase 2: Schema edits (`schema.py`)

**Model/effort:** sonnet / high.
**Parallel:** Yes, with Phases 1, 3, 4.
**Spec:** §5 rows 4, 6, 12, 14; §6.1 row 2.

**Files:**
- Modify: `src/custom_sam_peft/config/schema.py` (lines 20, 43, 73, 77-91, 136)

**Goal:** Apply the four schema edits the spec calls for — `DataConfig.image_size` 1024→1008, `ModelConfig.gradient_checkpointing` True→False, `NormalizeConfig` docstring update, `TextPromptConfig.negatives_per_image` `Field(description=…)`, and the `Optimizer` `Literal` inline comment.

### Task 2a: `DataConfig.image_size` default 1024 → 1008

- [ ] **Step P2-1: Edit line 136**

Current (`src/custom_sam_peft/config/schema.py:136`):

```python
    image_size: PositiveInt = 1024
```

Replace with:

```python
    image_size: PositiveInt = 1008  # SAM3.1's native input; see models/sam3.py:192,304,1202-1203.
```

### Task 2b: `ModelConfig.gradient_checkpointing` default True → False with #60 TODO

- [ ] **Step P2-2: Edit line 43**

Current:

```python
    gradient_checkpointing: bool = True
```

Replace with:

```python
    gradient_checkpointing: bool = False  # TODO(#60): re-enable when sam3 activation-checkpointing recompute mismatch is fixed
```

### Task 2c: `NormalizeConfig` docstring rewrite

- [ ] **Step P2-3a: Edit lines 77-84 (the `NormalizeConfig` class docstring)**

Current docstring (lines 78-84):

```python
    """Normalization stats used when AutoImageProcessor cannot be loaded.

    Resolution order at dataset construction:
      1. AutoImageProcessor.from_pretrained(model.name, local_files_only=True)
         and read image_mean/image_std.
      2. On OSError/AttributeError/ValueError, fall back to (mean, std) here.
    """
```

Replace with:

```python
    """Normalization stats used as a user-controllable fallback for image preprocessing.

    Resolution is delegated to
    :func:`custom_sam_peft.data.transforms.resolve_normalization`, which consults
    three sources in order:

      1. ``AutoImageProcessor.from_pretrained(model.name, local_files_only=True)``
         (succeeds when the HF cache is populated). Emits INFO.
      2. On ``OSError/AttributeError/ValueError``: look up ``model.name`` in
         :data:`custom_sam_peft.data.transforms.KNOWN_PROCESSOR_STATS`. If
         present, return the table values (emits WARNING).
      3. Otherwise, return the (mean, std) here (emits WARNING — verify these
         are correct for the backbone).

    Defaults are ImageNet stats, matching ``facebook/sam3.1``'s
    ``Sam3ImageProcessor`` and the ``KNOWN_PROCESSOR_STATS`` entry. Users with a
    non-SAM3 backbone should override these and the YAML's ``data.normalize``
    block accordingly.
    """
```

### Task 2d: `TextPromptConfig.negatives_per_image` — add `Field(description=…)`

- [ ] **Step P2-3b: Edit line 73**

Current:

```python
    negatives_per_image: int = Field(default=0, ge=0)
```

Replace with:

```python
    negatives_per_image: int = Field(
        default=0,
        ge=0,
        description=(
            "How many randomly-sampled negative class names to add per image when "
            "mode='present_plus_negatives'. Bounded above by TextPrompts' multiplex "
            "cap of 16 (k field). Example configs ship 4, which leaves headroom for "
            "typical COCO present-class counts (~3-7 per image)."
        ),
    )
```

### Task 2e: `Optimizer` `Literal` — inline comment describing the `"auto"` resolution rule

- [ ] **Step P2-4: Edit line 20**

Current:

```python
Optimizer = Literal["adamw", "adamw8bit", "auto"]
```

Replace with:

```python
# "auto" resolves at trainer construction (src/custom_sam_peft/train/trainer.py:45-49):
# adamw8bit if peft.method == "qlora" else adamw.
Optimizer = Literal["adamw", "adamw8bit", "auto"]
```

### Task 2f: Verify and commit

- [ ] **Step P2-5: Import + default-attribute sanity**

```bash
uv run python -c "
from custom_sam_peft.config.schema import DataConfig, ModelConfig, TextPromptConfig
assert DataConfig.model_fields['image_size'].default == 1008, DataConfig.model_fields['image_size'].default
assert ModelConfig.model_fields['gradient_checkpointing'].default is False
assert TextPromptConfig.model_fields['negatives_per_image'].description is not None
print('schema defaults OK')
"
```

Expected: `schema defaults OK`. Exits 0.

- [ ] **Step P2-6: Commit**

```bash
git add src/custom_sam_peft/config/schema.py
git commit -m "fix(schema): align defaults with shipped YAMLs and document resolver (#69)"
```

**Reviewer focus (sonnet/high):**
- `Field(description=…)` on `negatives_per_image` does NOT change the validation constraints (`default=0, ge=0` preserved).
- The `Optimizer` comment is placed *above* the `Literal` line, not inline on the same line (a `# comment` after `Literal[...]` on the same line would be tolerated by ruff but reads awkwardly).
- The `NormalizeConfig` docstring's reference to `KNOWN_PROCESSOR_STATS` uses the fully-qualified path so Sphinx-style autolinks (when/if added) resolve.

---

## Phase 3: YAML edits (four files)

**Model/effort:** sonnet / high.
**Parallel:** Yes, with Phases 1, 2, 4. The four YAMLs are file-disjoint within this phase too; an aggressive orchestrator can subdivide further, but the per-file diff is small enough that one subagent handling all four is cheaper.
**Spec:** §5 rows 1 and 14; §6.2.

**Files:**
- Modify: `configs/examples/coco_text_lora.yaml`
- Modify: `configs/examples/coco_text_qlora.yaml`
- Modify: `src/custom_sam_peft/cli/templates/coco_text_lora.yaml`
- Modify: `src/custom_sam_peft/cli/templates/coco_text_qlora.yaml`

**Goal:** For each file: change the `data.normalize.{mean,std}` block from `[0.5, 0.5, 0.5]` to ImageNet stats with the discoverability comment; add the top-of-file schema-as-SoT comment block. For the two templates only: change `gradient_checkpointing: true` → `false` with the `# see issue #60` annotation the example configs already carry.

### Top-of-file comment block (identical for all four YAMLs)

The block to insert at the **very top** of each file. For `configs/examples/*.yaml` the file currently begins with `run:` on line 1, so the comment block goes above line 1. For `src/custom_sam_peft/cli/templates/*.yaml` the file currently begins with three comment lines (`# custom-sam-peft starter config — …` etc.) — insert the new SoT block as additional lines *between* the existing template-purpose comment and `run:`, preserving the existing purpose lines on top.

Block content:

```yaml
# -----------------------------------------------------------------------------
# Schema is the source of truth.
#
# Keys below without a trailing `# override:` comment are echoes of the
# pydantic schema defaults at src/custom_sam_peft/config/schema.py — they are
# included for discoverability. Keys WITH a `# override:` comment are
# deliberate divergence from the schema default, justified inline.
# -----------------------------------------------------------------------------
```

### `data.normalize` block (identical for all four YAMLs)

The four files currently have:

```yaml
  normalize:
    mean: [0.5, 0.5, 0.5]
    std: [0.5, 0.5, 0.5]
```

Replace with:

```yaml
  # remove the `normalize:` block unless overriding for a non-SAM3 backbone —
  # SAM3.1's Sam3ImageProcessor returns the ImageNet stats below.
  normalize:
    mean: [0.485, 0.456, 0.406]
    std: [0.229, 0.224, 0.225]
```

### Task 3a: `configs/examples/coco_text_lora.yaml`

- [ ] **Step P3-1: Edit**

1. Insert the SoT comment block at the very top of the file (above the current line 1 `run:`).
2. Replace lines 29-31 (the `normalize:` block) with the ImageNet variant above.
3. **No change** to `gradient_checkpointing` — it is already `false` with the issue-#60 annotation on current line 10 (verified at plan-write time).

- [ ] **Step P3-2: Verify the resulting file still validates**

```bash
uv run python -c "
from pathlib import Path
from custom_sam_peft.config.loader import load_config
cfg = load_config('configs/examples/coco_text_lora.yaml')
assert cfg.data.normalize.mean == [0.485, 0.456, 0.406]
assert cfg.data.normalize.std == [0.229, 0.224, 0.225]
assert cfg.model.gradient_checkpointing is False
assert cfg.data.image_size == 1008
print('coco_text_lora.yaml OK')
"
```

Expected: `coco_text_lora.yaml OK`. Exits 0.

### Task 3b: `configs/examples/coco_text_qlora.yaml`

- [ ] **Step P3-3: Edit**

Identical to Task 3a: insert the SoT block at the top, replace the `normalize:` block (currently lines 29-31). `gradient_checkpointing` is already `false`.

- [ ] **Step P3-4: Verify**

```bash
uv run python -c "
from custom_sam_peft.config.loader import load_config
cfg = load_config('configs/examples/coco_text_qlora.yaml')
assert cfg.data.normalize.mean == [0.485, 0.456, 0.406]
assert cfg.model.gradient_checkpointing is False
print('coco_text_qlora.yaml OK')
"
```

### Task 3c: `src/custom_sam_peft/cli/templates/coco_text_lora.yaml`

- [ ] **Step P3-5: Edit**

1. Preserve the existing top-of-file comment block (lines 1-3 — the template-purpose lines). Insert the SoT comment block *below* those existing lines and *above* the `run:` block.
2. Change line 13 from:
   ```yaml
     gradient_checkpointing: true
   ```
   to:
   ```yaml
     gradient_checkpointing: false  # see issue #60 — sam3 activation checkpointing fails under non-reentrant recompute
   ```
   (matches the `configs/examples/*.yaml` annotation verbatim.)
3. Replace lines 32-34 (the `normalize:` block) with the ImageNet variant above.

- [ ] **Step P3-6: Verify via `custom-sam-peft init`**

The CLI `init` command writes the template to a target path; reload it through `load_config` to verify it parses.

```bash
uv run python -c "
import tempfile, pathlib
from typer.testing import CliRunner
from custom_sam_peft.cli.main import app
from custom_sam_peft.config.loader import load_config

with tempfile.TemporaryDirectory() as d:
    d = pathlib.Path(d)
    # Touch the four data paths the template references so load_config validates.
    (d / 'data').mkdir()
    (d / 'data' / 'train.json').write_text('{}')
    (d / 'data' / 'val.json').write_text('{}')
    (d / 'data' / 'train').mkdir()
    (d / 'data' / 'val').mkdir()
    out = d / 'config.yaml'
    r = CliRunner().invoke(app, ['init', '--template', 'coco-text-lora', '--output', str(out)])
    assert r.exit_code == 0, r.output
    cfg = load_config(out)
    assert cfg.model.gradient_checkpointing is False
    assert cfg.data.normalize.mean == [0.485, 0.456, 0.406]
    print('templates/coco_text_lora.yaml OK')
"
```

Expected: `templates/coco_text_lora.yaml OK`.

### Task 3d: `src/custom_sam_peft/cli/templates/coco_text_qlora.yaml`

- [ ] **Step P3-7: Edit**

Identical structure to Task 3c. Lines 1-3 preserved; SoT block inserted after; line 13 `gradient_checkpointing: true` → `false` with the `# see issue #60` annotation; lines 32-34 `normalize:` block replaced.

- [ ] **Step P3-8: Verify via `custom-sam-peft init` (qlora variant)**

```bash
uv run python -c "
import tempfile, pathlib
from typer.testing import CliRunner
from custom_sam_peft.cli.main import app
from custom_sam_peft.config.loader import load_config

with tempfile.TemporaryDirectory() as d:
    d = pathlib.Path(d)
    (d / 'data').mkdir()
    (d / 'data' / 'train.json').write_text('{}')
    (d / 'data' / 'val.json').write_text('{}')
    (d / 'data' / 'train').mkdir()
    (d / 'data' / 'val').mkdir()
    out = d / 'config.yaml'
    r = CliRunner().invoke(app, ['init', '--template', 'coco-text-qlora', '--output', str(out)])
    assert r.exit_code == 0, r.output
    cfg = load_config(out)
    assert cfg.model.gradient_checkpointing is False
    assert cfg.peft.method == 'qlora'
    print('templates/coco_text_qlora.yaml OK')
"
```

### Task 3e: Commit all four YAMLs together

- [ ] **Step P3-9: Commit**

```bash
git add configs/examples/coco_text_lora.yaml \
        configs/examples/coco_text_qlora.yaml \
        src/custom_sam_peft/cli/templates/coco_text_lora.yaml \
        src/custom_sam_peft/cli/templates/coco_text_qlora.yaml
git commit -m "fix(configs): align YAML defaults with schema; ImageNet normalize stats (#69)"
```

**Reviewer focus (sonnet/high):**
- The `normalize:` block is **not deleted** — it stays in the file with the discoverability comment (spec §6.2 last paragraph).
- The `# override:` convention introduced by the SoT comment block is currently used on exactly two keys across the four files: `gradient_checkpointing: false  # see issue #60 …` qualifies as an override comment in spirit; the reviewer should confirm the SoT block's wording (`without a trailing # override: comment`) is consistent with how `# see issue #60` reads.
- The CLI `init` command (Task 3c verification) actually copies one of the two templates into the user's chosen path — make sure the template files (not the example configs) drive that path. Source at `src/custom_sam_peft/cli/init_cmd.py` — do not modify; only the data files change.

---

## Phase 4: Insert "Superseded for normalization" callout in the 5/16 spec

**Model/effort:** sonnet / medium.
**Parallel:** Yes, with Phases 1, 2, 3.
**Spec:** §7.1.

**Files:**
- Modify: `docs/superpowers/specs/2026-05-16-model-loading-design.md` (above current line 120 — i.e. at the very top of §4.2, between line 118 `### 4.2 Forward behavior` and line 120 `- **\`images\`**: \`(B, 3, 1008, 1008)\` bf16, normalized with \`mean=std=[0.5, 0.5, 0.5]\`.`)

**Goal:** Insert the blockquote callout from spec §7.1 verbatim, between the §4.2 heading and its first bullet. No other content in the 5/16 spec is changed.

### Task 4a: Insert the callout

- [ ] **Step P4-1: Edit**

The current file (verified at plan-write time) has, on three consecutive lines:

- Line 118: `### 4.2 Forward behavior`
- Line 119: (blank line)
- Line 120: `- **\`images\`**: \`(B, 3, 1008, 1008)\` bf16, normalized with \`mean=std=[0.5, 0.5, 0.5]\`.`

Insert the following block between line 119 (blank) and line 120 (the first bullet) — i.e. immediately after the existing blank line that follows the §4.2 heading. Preserve a blank line both above and below the inserted block.

```markdown
> **Superseded for normalization (2026-05-21).** The `mean=std=[0.5, 0.5, 0.5]` claim
> on line 120 and the example-config edits on lines 283-285 are **wrong** for SAM3.1.
> See [`2026-05-21-yaml-config-defaults-audit-design.md`](2026-05-21-yaml-config-defaults-audit-design.md)
> for the corrected ground truth (ImageNet stats) and the three-step resolver.
> Everything else in this spec — image-size 1008, wrapper API, matcher, losses — stands.
```

(The exact five-line block text is mandated by spec §7.1 — do **not** rephrase or compress it.)

- [ ] **Step P4-2: Verify the callout landed in the right place**

```bash
sed -n '118,127p' docs/superpowers/specs/2026-05-16-model-loading-design.md
```

Expected first ~10 lines: heading on line 118, blank, then the five-line blockquote, blank, then the original `- **\`images\`**: \`(B, 3, 1008, 1008)\` …` bullet. The line numbers downstream shift by 7 — that is acceptable; nothing else references those line numbers by absolute index.

- [ ] **Step P4-3: Commit**

```bash
git add docs/superpowers/specs/2026-05-16-model-loading-design.md
git commit -m "docs(spec): add 'Superseded for normalization' callout to 5/16 model-loading spec (#69)"
```

**Reviewer focus (sonnet/high):**
- The callout text is *exactly* the five lines spec §7.1 mandates, including the markdown autolink to the new spec by filename (not by full path — the relative link resolves because both specs live under `docs/superpowers/specs/`).
- The "Everything else in this spec — image-size 1008, wrapper API, matcher, losses — stands." sentence is preserved verbatim; do not paraphrase.

---

## Phase 5: Test extensions in `tests/unit/test_data_transforms.py`

**Model/effort:** sonnet / high.
**Parallel:** No (depends on Phases 1, 2, 3 all committed). **Blocks:** Phase 6.
**Spec:** §6.3, §11 rows 2-3.

**Files:**
- Modify: `tests/unit/test_data_transforms.py`

**Goal:** Add **5 new tests** (one per resolver scenario from spec §6.3), update **3 existing tests** for the WARN-vs-INFO log-level changes from Phase 1, and add **1 schema-vs-YAML parity test** to cover acceptance criterion #1 from spec §9. All tests run on CPU; no real model weights are touched.

Per spec §6.3, the new tests must:
- Capture logs via `caplog.set_level(<level>, logger="custom_sam_peft.data.transforms")`.
- Mock `transformers.AutoImageProcessor` via `unittest.mock.patch("transformers.AutoImageProcessor", …)`.
- Use `SimpleNamespace` for the fake processor object.

The existing file already follows this style — see lines 21-27 (`_patch_proc_to_imagenet` helper) and lines 29-46 (the existing happy-path test). Match it exactly.

### Task 5a: Update the three existing resolver tests for WARN-vs-INFO

- [ ] **Step P5-1: `test_resolve_normalization_uses_image_processor_when_available` (lines 29-46)**

This test uses model name `"facebook/sam3.1"` and a fake processor returning `[0.1, 0.2, 0.3] / [0.4, 0.5, 0.6]`. Under the new resolver, `facebook/sam3.1` IS in the table, and the loaded stats DIVERGE from the table — so this test now exercises path 1's *divergence-WARN* branch, not the happy-path INFO branch.

To preserve the test's intent (happy-path INFO), change the model name in the assertion to something **not** in the table. Replace lines 29-46 with:

```python
def test_resolve_normalization_uses_image_processor_when_available(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Path 1, model not in KNOWN_PROCESSOR_STATS: returns processor values, logs INFO."""
    fake_proc = SimpleNamespace(image_mean=[0.1, 0.2, 0.3], image_std=[0.4, 0.5, 0.6])
    mock_aip = MagicMock()
    mock_aip.from_pretrained.return_value = fake_proc

    with patch("transformers.AutoImageProcessor", mock_aip):
        caplog.set_level(logging.INFO, logger="custom_sam_peft.data.transforms")
        mean, std = resolve_normalization("some/other-backbone", NormalizeConfig())

    mock_aip.from_pretrained.assert_called_once_with("some/other-backbone", local_files_only=True)
    assert mean == [0.1, 0.2, 0.3]
    assert std == [0.4, 0.5, 0.6]
    assert any(
        re.search(r"Using image_mean/image_std from AutoImageProcessor", rec.message)
        for rec in caplog.records
    )
    # No WARN records on the happy path.
    assert not any(rec.levelno >= logging.WARNING for rec in caplog.records)
```

- [ ] **Step P5-2: `test_resolve_normalization_falls_back_on_oserror` (lines 49-61)**

The model name is `"facebook/sam3.1"` (in the table) so this test now exercises **path 2** — table fallback with a WARN log. The existing assertion that mean = ImageNet still passes; flip the log-level capture from INFO to WARNING and update the regex.

Replace lines 49-61 with:

```python
def test_resolve_normalization_falls_back_on_oserror(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Path 2: OSError + model in table -> table fallback, exactly one WARN."""
    mock_aip = MagicMock()
    mock_aip.from_pretrained.side_effect = OSError("no cache")

    with patch("transformers.AutoImageProcessor", mock_aip):
        caplog.set_level(logging.WARNING, logger="custom_sam_peft.data.transforms")
        mean, std = resolve_normalization("facebook/sam3.1", NormalizeConfig())

    assert mean == [0.485, 0.456, 0.406]
    assert std == [0.229, 0.224, 0.225]
    warn_records = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warn_records) == 1
    assert "known-good stats" in warn_records[0].getMessage()
```

- [ ] **Step P5-3: `test_resolve_normalization_falls_back_on_attribute_error` (lines 64-71)**

Same logic — `"facebook/sam3.1"` is in the table, so an `AttributeError` (path 2) returns the table values with a WARN.

Replace lines 64-71 with:

```python
def test_resolve_normalization_falls_back_on_attribute_error() -> None:
    """Path 2: AttributeError + model in table -> table fallback."""
    mock_aip = MagicMock()
    mock_aip.from_pretrained.return_value = SimpleNamespace()  # missing image_mean/image_std

    with patch("transformers.AutoImageProcessor", mock_aip):
        mean, _std = resolve_normalization("facebook/sam3.1", NormalizeConfig())

    assert mean == [0.485, 0.456, 0.406]
```

### Task 5b: Add the five new tests from spec §6.3

Insert these tests after `test_resolve_normalization_falls_back_on_attribute_error` (after the updated line 71) and before `test_eval_transforms_resizes_to_square` (currently line 74).

- [ ] **Step P5-4: `test_resolve_normalization_processor_loads_no_table_entry_no_warn`**

```python
def test_resolve_normalization_processor_loads_no_table_entry_no_warn(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Path 1, model NOT in table: returns processor values, no WARN, INFO present."""
    fake_proc = SimpleNamespace(image_mean=[0.7, 0.7, 0.7], image_std=[0.2, 0.2, 0.2])
    mock_aip = MagicMock()
    mock_aip.from_pretrained.return_value = fake_proc

    with patch("transformers.AutoImageProcessor", mock_aip):
        caplog.set_level(logging.INFO, logger="custom_sam_peft.data.transforms")
        mean, std = resolve_normalization("some/unknown-backbone", NormalizeConfig())

    assert mean == [0.7, 0.7, 0.7]
    assert std == [0.2, 0.2, 0.2]
    assert not any(rec.levelno >= logging.WARNING for rec in caplog.records)
    assert any(
        re.search(r"Using image_mean/image_std from AutoImageProcessor", rec.message)
        for rec in caplog.records
    )
```

- [ ] **Step P5-5: `test_resolve_normalization_processor_loads_matches_table`**

```python
def test_resolve_normalization_processor_loads_matches_table(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Path 1, model in table, values within 1e-3 of table entry: no WARN, INFO present."""
    fake_proc = SimpleNamespace(
        image_mean=[0.4855, 0.4555, 0.4055],  # within 1e-3 of [0.485, 0.456, 0.406]
        image_std=[0.2295, 0.2245, 0.2255],   # within 1e-3 of [0.229, 0.224, 0.225]
    )
    mock_aip = MagicMock()
    mock_aip.from_pretrained.return_value = fake_proc

    with patch("transformers.AutoImageProcessor", mock_aip):
        caplog.set_level(logging.INFO, logger="custom_sam_peft.data.transforms")
        mean, std = resolve_normalization("facebook/sam3.1", NormalizeConfig())

    assert mean == [0.4855, 0.4555, 0.4055]
    assert std == [0.2295, 0.2245, 0.2255]
    assert not any(rec.levelno >= logging.WARNING for rec in caplog.records)
    assert any(
        re.search(r"Using image_mean/image_std from AutoImageProcessor", rec.message)
        for rec in caplog.records
    )
```

- [ ] **Step P5-6: `test_resolve_normalization_processor_loads_diverges_from_table`**

```python
def test_resolve_normalization_processor_loads_diverges_from_table(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Path 1, model in table, divergence > 1e-3: returns processor values + exactly one WARN naming both vectors."""
    fake_proc = SimpleNamespace(image_mean=[0.5, 0.5, 0.5], image_std=[0.5, 0.5, 0.5])
    mock_aip = MagicMock()
    mock_aip.from_pretrained.return_value = fake_proc

    with patch("transformers.AutoImageProcessor", mock_aip):
        caplog.set_level(logging.WARNING, logger="custom_sam_peft.data.transforms")
        mean, std = resolve_normalization("facebook/sam3.1", NormalizeConfig())

    # Table is a sentinel, not a gate: processor values are returned.
    assert mean == [0.5, 0.5, 0.5]
    assert std == [0.5, 0.5, 0.5]
    warn_records = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warn_records) == 1
    msg = warn_records[0].getMessage()
    # Both vectors must appear in the single WARN message.
    assert "0.5" in msg
    assert "0.485" in msg
    assert "0.229" in msg
```

- [ ] **Step P5-7: `test_resolve_normalization_processor_fails_model_in_table`**

```python
def test_resolve_normalization_processor_fails_model_in_table(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Path 2: OSError + model in table -> returns table values, exactly one WARN naming the table fallback."""
    mock_aip = MagicMock()
    mock_aip.from_pretrained.side_effect = OSError("no cache")

    with patch("transformers.AutoImageProcessor", mock_aip):
        caplog.set_level(logging.WARNING, logger="custom_sam_peft.data.transforms")
        # User's NormalizeConfig is intentionally distinct from the table to confirm table wins.
        user_norm = NormalizeConfig(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        mean, std = resolve_normalization("facebook/sam3.1", user_norm)

    assert mean == [0.485, 0.456, 0.406]
    assert std == [0.229, 0.224, 0.225]
    warn_records = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warn_records) == 1
    msg = warn_records[0].getMessage()
    assert "known-good stats" in msg
    assert "0.485" in msg
```

- [ ] **Step P5-8: `test_resolve_normalization_processor_fails_model_not_in_table`**

```python
def test_resolve_normalization_processor_fails_model_not_in_table(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Path 3: OSError + model NOT in table -> returns user's fallback, exactly one WARN naming the YAML values."""
    mock_aip = MagicMock()
    mock_aip.from_pretrained.side_effect = OSError("no cache")

    with patch("transformers.AutoImageProcessor", mock_aip):
        caplog.set_level(logging.WARNING, logger="custom_sam_peft.data.transforms")
        user_norm = NormalizeConfig(mean=[0.3, 0.3, 0.3], std=[0.2, 0.2, 0.2])
        mean, std = resolve_normalization("some/unknown-backbone", user_norm)

    assert mean == [0.3, 0.3, 0.3]
    assert std == [0.2, 0.2, 0.2]
    warn_records = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warn_records) == 1
    msg = warn_records[0].getMessage()
    assert "no known-good entry" in msg
    assert "0.3" in msg
```

### Task 5c: Add the schema-vs-YAML parity test (acceptance criterion #1)

This test covers spec §9 acceptance criterion #1 ("All four YAMLs and the schema produce identical resolved configs for `data.normalize`, `data.image_size`, and `model.gradient_checkpointing`").

- [ ] **Step P5-9: `test_shipped_yamls_match_schema_defaults`**

Add this test at the very end of `tests/unit/test_data_transforms.py`, after `test_train_transforms_color_jitter_zero_preserves_color`:

```python
from pathlib import Path

from custom_sam_peft.config.loader import load_config
from custom_sam_peft.config.schema import DataConfig, ModelConfig


def test_shipped_yamls_match_schema_defaults() -> None:
    """All four shipped YAMLs resolve normalize / image_size / gradient_checkpointing
    to the schema's default values — i.e. the YAML echoes are consistent with the
    schema as the source of truth.
    """
    repo_root = Path(__file__).resolve().parents[2]
    yaml_paths = [
        repo_root / "configs" / "examples" / "coco_text_lora.yaml",
        repo_root / "configs" / "examples" / "coco_text_qlora.yaml",
        repo_root / "src" / "custom_sam_peft" / "cli" / "templates" / "coco_text_lora.yaml",
        repo_root / "src" / "custom_sam_peft" / "cli" / "templates" / "coco_text_qlora.yaml",
    ]
    schema_image_size = DataConfig.model_fields["image_size"].default
    schema_grad_ckpt = ModelConfig.model_fields["gradient_checkpointing"].default
    # NormalizeConfig defaults are constructed via default_factory; build an
    # instance to read them.
    from custom_sam_peft.config.schema import NormalizeConfig
    schema_mean = NormalizeConfig().mean
    schema_std = NormalizeConfig().std

    for p in yaml_paths:
        assert p.is_file(), p
        cfg = load_config(p)
        assert cfg.data.image_size == schema_image_size, p
        assert cfg.model.gradient_checkpointing == schema_grad_ckpt, p
        assert cfg.data.normalize.mean == schema_mean, p
        assert cfg.data.normalize.std == schema_std, p
```

**Note for the cold reader:** `load_config` resolves YAML paths relative to the config file's directory but does NOT assert the referenced data files exist (confirmed by reading `src/custom_sam_peft/config/loader.py:99-114`). The four YAMLs reference `data/coco/…` or `data/train.json` etc. that don't exist in the worktree — `load_config` will accept them. No tmp-dir scaffolding needed for this test.

### Task 5d: Run the full transforms test suite

- [ ] **Step P5-10: Run all tests in `test_data_transforms.py`**

```bash
uv run pytest tests/unit/test_data_transforms.py -v
```

Expected: all `test_resolve_normalization_*` (3 updated + 5 new) pass, `test_shipped_yamls_match_schema_defaults` passes, and the existing transforms tests (`test_eval_transforms_*`, `test_train_transforms_*`) still pass. Total: ~12 tests. Exit 0.

Common failure modes:
- "expected exactly one WARN record" failing with 0 records: caplog level is set on the wrong logger name. The logger is `custom_sam_peft.data.transforms` — must match the module path exactly.
- "0.485 not in msg" failing on the divergence test: the WARN message uses `%s` formatting on a Python list, which renders as `[0.485, 0.456, 0.406]`. The substring `0.485` will be present. If the implementation uses `%.3f`-style formatting instead, the assertion still passes because the leading `0.485` digits match.

### Task 5e: Commit

- [ ] **Step P5-11: Commit**

```bash
git add tests/unit/test_data_transforms.py
git commit -m "test(data): cover three-step resolve_normalization paths + schema-vs-YAML parity (#69)"
```

**Reviewer focus (sonnet/high):**
- All new tests use the existing file's mocking pattern (`MagicMock` for the AutoImageProcessor class, `SimpleNamespace` for the processor instance). No live `from_pretrained` call.
- The divergence test (Step P5-6) asserts the WARN message contains both `0.5` (loaded) and `0.485` / `0.229` (table) — substring matches are robust to whatever exact format string the implementation uses.
- The parity test (Step P5-9) is the *only* test that touches the actual shipped YAMLs. If it fails for a YAML, Phase 3 was applied incompletely.

---

## Phase 6: Lint / format / type / test gate

**Model/effort:** sonnet / high.
**Parallel:** No. **Depends on:** Phases 1-5. **Blocks:** Phase 7.

**Goal:** Run every gate `ci.yml` runs (ruff, ruff format, mypy, pytest) and fix anything inline. The repo's CI workflow at `.github/workflows/ci.yml` (lines 1-50) is the authority for this list.

- [ ] **Step P6-1: Auto-fix ruff lint issues**

```bash
uv run ruff check . --fix
```

Expected: no remaining lint errors after auto-fix. If any errors are NOT auto-fixable, fix them by hand and re-run. The repo selects rule sets `E, F, I, B, UP, SIM, RUF, S` (verified in `pyproject.toml:67`).

- [ ] **Step P6-2: Apply ruff format**

```bash
uv run ruff format .
```

Expected: file rewrites are minimal (the existing code is already formatted). The notebook directory is excluded via `pyproject.toml:62-64`.

- [ ] **Step P6-3: Confirm ruff format --check passes**

```bash
uv run ruff format --check .
```

Expected: "X files already formatted" with no diff. If a diff is reported, re-run Step P6-2 and inspect.

- [ ] **Step P6-4: Confirm ruff check passes (no `--fix`)**

```bash
uv run ruff check .
```

Expected: "All checks passed!" Exit 0.

- [ ] **Step P6-5: Run mypy in strict mode on the package**

```bash
uv run mypy src/custom_sam_peft
```

Expected: "Success: no issues found in N source files." The `[tool.mypy]` block at `pyproject.toml:79-98` configures `strict = true` and lists per-module `ignore_missing_imports` overrides for `transformers`-adjacent libs.

- [ ] **Step P6-6: Run the full pytest suite**

```bash
uv run pytest
```

Expected: all tests pass; coverage ≥ 80% (the `--cov-fail-under=80` gate from `pyproject.toml:109` is unaffected by these changes — the new tests increase coverage). Exit 0.

- [ ] **Step P6-7: Commit any fixups**

If ruff or ruff format made changes in Steps P6-1 or P6-2, commit them as a separate fixup commit:

```bash
git add -u
git commit -m "style: ruff lint+format fixups (#69)"
```

If nothing was changed, skip this step.

**Reviewer focus (sonnet/high):**
- The new logging f-strings / `%s` patterns satisfy ruff's `G` rules (if enabled — verify with Step P6-4) — `%s` lazy formatting is preferred over `f""` inside log calls.
- mypy strict mode requires the `KNOWN_PROCESSOR_STATS` annotation `dict[str, tuple[list[float], list[float]]]` — verified explicit in the Phase 1 code.

---

## Phase 7: Open the PR

**Model/effort:** sonnet / medium.
**Parallel:** No. **Depends on:** Phase 6 green.

**Goal:** Push the branch and open a non-draft PR linking spec, plan, and issue #69. Note the four follow-up issues will be filed by the orchestrator *after* merge (Phase 8). The orchestrator owns the semver bump decision separately — this plan does not bake a version number into the PR title or body.

- [ ] **Step P7-1: Final clean-state check**

```bash
git status
git log --oneline origin/main..HEAD
```

Expected: working tree clean; commit log shows the per-phase commits (P0, P1, P2, P3, P4, P5, optional P6 fixup).

- [ ] **Step P7-2: Push the branch**

```bash
git push -u origin audit/config-defaults-69
```

If the upstream already exists from prior work, `git push` alone suffices.

- [ ] **Step P7-3: Open the PR**

```bash
gh pr create \
  --assignee @me \
  --title "fix: audit and align shipped YAML/schema defaults (closes #69)" \
  --body "$(cat <<'EOF'
## Summary

- Rewrites `resolve_normalization` to a three-step resolver (`transformers.AutoImageProcessor` → `KNOWN_PROCESSOR_STATS` table → user's `NormalizeConfig`); cache-miss and table-divergence now WARN (previously INFO).
- Aligns `DataConfig.image_size` (1024 → 1008) and `ModelConfig.gradient_checkpointing` (True → False, with `# TODO(#60)`) with the shipped YAMLs and the SAM 3.1 model code.
- Reconciles the four shipped YAMLs (`configs/examples/coco_text_{lora,qlora}.yaml`, `src/custom_sam_peft/cli/templates/coco_text_{lora,qlora}.yaml`) to ImageNet normalize stats; flips the two CLI templates' `gradient_checkpointing: true` → `false`; adds a top-of-file "schema is the source of truth" comment block to all four.
- Adds schema docstrings: `NormalizeConfig` describes the three-step resolver; `TextPromptConfig.negatives_per_image` gains a `Field(description=...)` carrying the rationale; the `Optimizer` Literal gains an inline comment for the `"auto"` resolution rule.
- Extends `tests/unit/test_data_transforms.py` with 5 new CPU-only tests covering each resolver branch, 3 updates to existing tests for WARN-vs-INFO, and 1 schema-vs-YAML parity test for acceptance criterion #1.
- Inserts a "Superseded for normalization" callout at the top of §4.2 of `docs/superpowers/specs/2026-05-16-model-loading-design.md`.

No new config knobs. No augmentation rework. No hyperparameter retunes.

**Spec:** `docs/superpowers/specs/2026-05-21-yaml-config-defaults-audit-design.md`
**Plan:** `docs/superpowers/plans/2026-05-21-yaml-config-defaults-audit.md`
**Closes:** #69

## Acceptance criteria verification

- [x] Schema-vs-YAML alignment (criterion #1): `tests/unit/test_data_transforms.py::test_shipped_yamls_match_schema_defaults` covers all three keys across all four YAMLs.
- [x] Happy-path correctness (criterion #2): `test_resolve_normalization_processor_loads_matches_table` + `test_resolve_normalization_uses_image_processor_when_available`.
- [x] Cache-miss correctness on SAM3.1 (criterion #3): `test_resolve_normalization_processor_fails_model_in_table`.
- [x] Table-divergence detection (criterion #4): `test_resolve_normalization_processor_loads_diverges_from_table`.
- [x] Unknown-model fallback (criterion #5): `test_resolve_normalization_processor_fails_model_not_in_table`.
- [x] Schema-default drift fixed (criterion #6): see parity test + direct attribute assertion in Phase 2 step P2-5.
- [x] Lint/test gates pass (criterion #7): ruff check, ruff format --check, mypy strict, pytest with 80% coverage gate all green.
- [x] No new schema knobs (criterion #8): `TrainConfig` field count is unchanged — only existing field defaults and docstrings were touched.

## Follow-up issues (filed by orchestrator after merge)

Per spec §8, four issues will be opened post-merge with `priority:low` labels:

1. Empirically verify `Sam3ImageProcessor` stats vs `KNOWN_PROCESSOR_STATS["facebook/sam3.1"]` (`priority:low`, `question`).
2. A/B QLoRA `lr` at `5e-4` vs `1e-4` post-#44 GPU eval (`priority:low`).
3. Re-audit `box_hint` schedule after #24 lands (`priority:low`).
4. Re-enable `gradient_checkpointing` defaults once #60's ViT activation-checkpointing bug is fixed (`priority:low`).

## Test plan

- [ ] `uv run pytest tests/unit/test_data_transforms.py -v` — all 12 tests pass.
- [ ] `uv run pytest` — full suite green; coverage ≥ 80%.
- [ ] `uv run ruff check . && uv run ruff format --check .` — clean.
- [ ] `uv run mypy src/custom_sam_peft` — clean.
- [ ] Spot-check: `uv run python -c "from custom_sam_peft.config.schema import DataConfig, ModelConfig; assert DataConfig.model_fields['image_size'].default == 1008 and ModelConfig.model_fields['gradient_checkpointing'].default is False"` exits 0.
- [ ] Spot-check: `uv run python -c "from custom_sam_peft.data.transforms import KNOWN_PROCESSOR_STATS; assert KNOWN_PROCESSOR_STATS['facebook/sam3.1'] == ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])"` exits 0.
EOF
  )"
```

- [ ] **Step P7-4: Note PR URL and watch CI**

Record the PR URL printed by `gh pr create`. The `ci.yml` workflow (test, lock-check, lint-hygiene, gpu-deselect-check) will run on PR push. The `docker.yml` workflow (semver-tag-only) will NOT run — expected.

**Semver bump:** This PR makes user-visible default changes (`image_size`, `gradient_checkpointing`, normalize stats) — these are technically API-breaking for users who rely on the schema's defaults. The orchestrator decides whether this warrants a `0.8.0` (minor) or `0.7.2` (patch) bump per repo convention; the most recent tag at plan-write time is `v0.7.1`. **This plan does NOT bake a version number** — the orchestrator handles the version bump as a separate commit on `main` after merge per CLAUDE.md / repo convention.

---

## Phase 8: File follow-up issues (orchestrator, post-merge)

**Model/effort:** sonnet / low — purely mechanical `gh issue create` invocations.
**Parallel:** Yes (four independent issue creations).
**Depends on:** PR merged to `main`. Spec §8.

**Goal:** File the four follow-up issues spec §8 mandates. Each uses `gh issue create --assignee @me --label …`. Both labels (`priority:low`, `question`) were confirmed present in Phase 0, Step P0-4.

- [ ] **Step P8-1: File issue (a) — Empirical SAM3.1 stat dump**

```bash
gh issue create \
  --assignee @me \
  --label "priority:low" \
  --label "question" \
  --title "Empirically verify Sam3ImageProcessor stats vs KNOWN_PROCESSOR_STATS['facebook/sam3.1']" \
  --body "$(cat <<'EOF'
Follow-up to #69 / the 2026-05-21 yaml-config-defaults audit.

On a machine with HF cache access to `facebook/sam3.1`, dump the live
processor's `image_mean` and `image_std` and confirm they match the
`KNOWN_PROCESSOR_STATS["facebook/sam3.1"]` entry in
`src/custom_sam_peft/data/transforms.py` (currently
`([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])`).

Attach the live values + the processor type / class name to this issue
as evidence. Closes when verified.

```python
from transformers import AutoImageProcessor
proc = AutoImageProcessor.from_pretrained("facebook/sam3.1")
print(type(proc).__name__, proc.image_mean, proc.image_std)
```

If the live values diverge from the table beyond `1e-3`, update the table
in a follow-up PR.

Refs: `docs/superpowers/specs/2026-05-21-yaml-config-defaults-audit-design.md` §3, §8 row (a).
EOF
)"
```

- [ ] **Step P8-2: File issue (b) — QLoRA lr A/B**

```bash
gh issue create \
  --assignee @me \
  --label "priority:low" \
  --title "A/B QLoRA lr at 5e-4 vs 1e-4 post-#44 GPU eval" \
  --body "$(cat <<'EOF'
Follow-up to #69 / the 2026-05-21 yaml-config-defaults audit.

Row 7 of the audit kept `train.lr = 1.0e-4` for both LoRA and QLoRA
templates pending a real training run. Once #44 (GPU eval infra) lands,
run an A/B at `5e-4` for the QLoRA path and confirm whether the higher
LR is safe or destabilizing. If safe, bump `configs/examples/coco_text_qlora.yaml`
and `src/custom_sam_peft/cli/templates/coco_text_qlora.yaml`.

Refs: `docs/superpowers/specs/2026-05-21-yaml-config-defaults-audit-design.md` §5 row 7, §8 row (b).
EOF
)"
```

- [ ] **Step P8-3: File issue (c) — Box-hint schedule re-audit**

```bash
gh issue create \
  --assignee @me \
  --label "priority:low" \
  --title "Re-audit box_hint schedule after #24 lands" \
  --body "$(cat <<'EOF'
Follow-up to #69 / the 2026-05-21 yaml-config-defaults audit.

Row 11 of the audit kept the ad-hoc box-hint schedule
(`p_start=1.0, p_end=0.0, decay_steps=5000, early_stop_p_threshold=0.05`)
pending #24 (the in-flight bbox-prompt PR). Once #24 lands and the
bbox-prompt training path is exercised, re-audit and retune the schedule.

Refs: `docs/superpowers/specs/2026-05-21-yaml-config-defaults-audit-design.md` §5 row 11, §8 row (c).
EOF
)"
```

- [ ] **Step P8-4: File issue (d) — Re-enable gradient_checkpointing after #60**

```bash
gh issue create \
  --assignee @me \
  --label "priority:low" \
  --title "Re-enable gradient_checkpointing defaults once #60 is fixed" \
  --body "$(cat <<'EOF'
Follow-up to #69 / the 2026-05-21 yaml-config-defaults audit.

Row 14 of the audit flipped `ModelConfig.gradient_checkpointing` default to
`False` because of #60 (sam3 activation-checkpointing recompute mismatch).
A `# TODO(#60)` comment was added inline at
`src/custom_sam_peft/config/schema.py:43` as the in-code pointer to this
follow-up.

When #60's underlying ViT activation-checkpointing bug is fixed:
- Re-enable `gradient_checkpointing: true` in
  `src/custom_sam_peft/cli/templates/coco_text_lora.yaml`,
  `src/custom_sam_peft/cli/templates/coco_text_qlora.yaml`,
  `configs/examples/coco_text_lora.yaml`,
  `configs/examples/coco_text_qlora.yaml`.
- Flip `ModelConfig.gradient_checkpointing` default back to `True` in
  `src/custom_sam_peft/config/schema.py` and remove the `# TODO(#60)` comment.

Refs: `docs/superpowers/specs/2026-05-21-yaml-config-defaults-audit-design.md` §5 row 14, §8 row (d).
EOF
)"
```

- [ ] **Step P8-5: Close issue #69**

```bash
gh issue close 69 --comment "Closed by PR <URL>. Follow-ups filed: (a) processor stat verification, (b) QLoRA lr A/B, (c) box-hint re-audit post-#24, (d) gradient_checkpointing re-enable post-#60."
```

Replace `<URL>` with the merged PR URL from Phase 7, Step P7-4.

---

## Definition of done

All items below must be checked before this plan is considered complete:

- [ ] `src/custom_sam_peft/data/transforms.py` has `KNOWN_PROCESSOR_STATS` (seeded with `facebook/sam3.1` → ImageNet) and the three-step `resolve_normalization`.
- [ ] `src/custom_sam_peft/config/schema.py` has `DataConfig.image_size = 1008`, `ModelConfig.gradient_checkpointing = False` with the `# TODO(#60)` comment, the rewritten `NormalizeConfig` docstring, `Field(description=...)` on `TextPromptConfig.negatives_per_image`, and the `Optimizer` Literal inline comment.
- [ ] All four shipped YAMLs (`configs/examples/coco_text_{lora,qlora}.yaml` and `src/custom_sam_peft/cli/templates/coco_text_{lora,qlora}.yaml`) carry ImageNet normalize stats with the discoverability comment, the top-of-file schema-as-SoT comment block, and (templates only) `gradient_checkpointing: false`.
- [ ] `docs/superpowers/specs/2026-05-16-model-loading-design.md` has the five-line "Superseded for normalization" blockquote at the top of §4.2.
- [ ] `tests/unit/test_data_transforms.py` has the 5 new tests, the 3 updated tests, and the 1 schema-vs-YAML parity test; `uv run pytest tests/unit/test_data_transforms.py -v` exits 0.
- [ ] `uv run ruff check .`, `uv run ruff format --check .`, `uv run mypy src/custom_sam_peft`, and `uv run pytest` all exit 0.
- [ ] PR is open, links spec + plan + issue #69, and lists acceptance criteria.
- [ ] After merge: four follow-up issues filed (Step P8-1..P8-4) and issue #69 closed (Step P8-5).

---

## Self-review

**1. Spec coverage:** Every row of spec §5 is mapped in the "Per-default audit coverage check" table at the top of the plan, each either to a code-change phase/step or marked "No code change" with rationale. Every deliverable in spec §6 (code, YAMLs, tests), §7 (spec callout), §8 (follow-up issues), and §9 (acceptance criteria) has at least one explicit task or verification command.

**2. Placeholder scan:** No "TBD", "TODO", "implement later", or "fill in details" language. The `<URL>` placeholder in Step P8-5 is the merged PR URL, only knowable post-merge — its substitution is mechanical at issue-close time.

**3. Type consistency:** `resolve_normalization` signature is preserved as `(model_name: str, fallback: NormalizeConfig) -> tuple[list[float], list[float]]` from the existing line 22-24. `KNOWN_PROCESSOR_STATS` is annotated `dict[str, tuple[list[float], list[float]]]` everywhere. `_stats_diverge` returns `bool`. `_STATS_DIVERGENCE_ATOL` is `float` (1e-3). All test function names match between the spec §6.3 table and Steps P5-4..P5-8.

**4. Parallelism:** Phases 1, 2, 3, 4 are file-disjoint and called out as parallel-dispatchable. Phase 5 (tests) is serialized after the parallel cluster because the tests import from Phase 1 and assert defaults from Phase 2 and YAML values from Phase 3.

**5. Dependency ordering:** No task references an artifact a later task creates. The schema-vs-YAML parity test (Step P5-9) depends on Phase 2 (schema defaults) AND Phase 3 (YAML values) — both committed before Phase 5 runs, per the dependency graph.

**6. Acceptance criteria → commands:** Every spec §9 criterion has at least one concrete verification command somewhere in the plan (Steps P2-5, P3-2/4/6/8, P5-10, P6-3/4/5/6, P7-3 PR body Test plan).
```
