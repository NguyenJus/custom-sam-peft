# Remove non-text primary prompt pathways (text is always the primary prompt)

Delete the half-built `prompt_mode='bbox'` / `BoxPrompts` primary-prompt surface; introduce a `SupportPrompts` auxiliary-prompt container that carries the existing `box_hint` curriculum (and is the future seam for mask / point hints). After this PR the codebase states one invariant in one voice: **text is the only primary prompt; geometry only ever rides along as an auxiliary localization hint, never at inference.**

**Issues:** [#126 — refactor: remove non-text primary prompt pathways](https://github.com/NguyenJus/custom-sam-peft/issues/126) (primary). Related: [#24](https://github.com/NguyenJus/custom-sam-peft/issues/24) (closed-as-misfiled, the ambiguity that motivated this), [#14](https://github.com/NguyenJus/custom-sam-peft/issues/14) (the `box_hint` curriculum we preserve), [#88](https://github.com/NguyenJus/custom-sam-peft/issues/88) ("re-audit `box_hint` schedule after #24 lands" — now unblocked), [#113](https://github.com/NguyenJus/custom-sam-peft/issues/113) (interactive-pipe architectural note: `enable_inst_interactivity=False` stays).
**Release:** pre-1.0 MINOR bump (breaking schema change — `data.prompt_mode` is removed and any config that carries it fails at load).
**Status:** locked design, single PR, no back-compat shims.

---

## §1 Scope & non-goals

### In scope

| File | Change |
|------|--------|
| `src/custom_sam_peft/config/schema.py` | Delete `PromptMode = Literal["text", "bbox"]` (line 92) and the `"PromptMode"` entry in `__all__` (line 77). Delete `DataConfig.prompt_mode: PromptMode` field (line 386). `_Strict`/`extra="forbid"` (line 106) then rejects any stray `prompt_mode:` key at schema-load time. |
| `src/custom_sam_peft/data/base.py` | Delete `BoxPrompts` dataclass (lines 18–27). Collapse `Prompts = TextPrompts \| BoxPrompts` (line 30) to `Prompts = TextPrompts` (keep the alias). Add a new `SupportPrompts` frozen dataclass next to `TextPrompts` with one optional field, `boxes: list[Tensor \| None] \| None = None`. |
| `src/custom_sam_peft/data/coco.py` | Delete the `prompt_mode` ctor param + `("text","bbox")` validator (lines 128, 136–137), `self._prompt_mode` field (line 140), `BoxPrompts` import inside `_pack_example` (line 274), and the `if self._prompt_mode == "text": ... # bbox mode ...` branch (lines 288–348) — collapse to the text branch, always emitting `TextPrompts`. Drop `prompt_mode=cfg["prompt_mode"]` in `build_coco` (line 412). |
| `src/custom_sam_peft/data/hf.py` | Same pattern: delete `prompt_mode` ctor param + validator (lines 134, 143–144), `self._prompt_mode` (line 148), `BoxPrompts` import (line 308), the text-vs-bbox branch in `_pack_example` (lines 322–381) — collapse to text. Drop `prompt_mode=cfg["prompt_mode"]` in `build_hf` (line 445). |
| `src/custom_sam_peft/models/sam3.py` | Delete `BoxPrompts` from the import on line 30. Update `Sam3Wrapper`'s docstring at lines 178–204 (drop the `BoxPrompts` mentions on lines 185, 188–189). Change `Sam3Wrapper.forward` signature (line 223) from `box_hints: list[Tensor \| None] \| None = None` to `support: SupportPrompts \| None = None`; pass `support.boxes` (or `None`) into `self.model(...)` as `box_hints=...`. Simplify `_validate_inputs` (lines 233–302) — drop the mixed-batch branch and `BoxPrompts` checks (lines 257–259, 278–283); rename the `box_hints` parameter to `support`, validate `support.boxes` length/shape exactly as today's `box_hints`. Add a short anchoring comment at line 616 (`enable_inst_interactivity=False`) citing #126. |
| `src/custom_sam_peft/train/trainer.py` | Delete the bbox guard at lines 135–140. Replace `box_hints=None` on line 485 (the in-trainer eval-panel forward) with `support=None`. Replace `box_hints=...` wiring where the trainer calls `Sam3Wrapper.forward` indirectly via the loop (no change needed; see `train/loop.py`). |
| `src/custom_sam_peft/train/loop.py` | At lines 294 and 313, replace `box_hints=micro_hints` / `box_hints=hints_g` with `support=SupportPrompts(boxes=micro_hints)` / `support=SupportPrompts(boxes=hints_g)`. The `_box_hint_p` schedule (line 155), `hints_g` construction (lines 235–247), and the `box_hint/applied` metric (line 397) are unchanged. |
| `src/custom_sam_peft/cli/train_cmd.py` | Delete the bbox guard at lines 48–52. |
| `src/custom_sam_peft/cli/run_cmd.py` | Delete the bbox guard at lines 191–195. |
| `src/custom_sam_peft/cli/templates/config_full.yaml` | Drop the `prompt_mode: text` line (line 23). |
| `src/custom_sam_peft/cli/setup_wizard.py` | Confirm there is no `prompt_mode` emit logic to remove (verified — the wizard relies on the template). No code change; only the template loses the line. |
| `configs/examples/coco_text_lora.yaml` (line 28) | Drop `prompt_mode: text`. |
| `configs/examples/coco_text_qlora.yaml` (line 28) | Drop `prompt_mode: text`. |
| `configs/examples/coco_text_lora_subset.yaml` (line 22) | Drop `prompt_mode: text`. |
| `configs/examples/coco_text_no_val.yaml` (line 26) | Drop `prompt_mode: text`. |
| `configs/examples/coco_text_auto_split.yaml` (line 27) | Drop `prompt_mode: text`. |
| `configs/examples/min_gpu_qlora.yaml` (line 27) | Drop `prompt_mode: text`. |
| `configs/examples/gpu_smoke_lora.yaml` (line 20) | Drop `prompt_mode: text`. |
| `configs/examples/gpu_smoke_qlora.yaml` (line 20) | Drop `prompt_mode: text`. |
| Tests (see §9) | Remove `BoxPrompts` / `prompt_mode='bbox'` assertions; rewrite the box-hints wrapper test to use `support=SupportPrompts(boxes=...)`; add a schema-rejection test and a `SupportPrompts` shape test. |
| `docs/ARCHITECTURE.md` | State the text-primary invariant up front. Replace the `Prompts (TextPrompts \| BoxPrompts)` description (line 15) with `TextPrompts` + a note that `SupportPrompts` is the auxiliary container. |
| `docs/config-schema.md` | Delete the `data.prompt_mode` row (line 52). |
| `CHANGELOG.md` | Add a breaking-change entry (§10). |

### Out of scope

- **No mask / point fields on `SupportPrompts` in this PR.** Strict YAGNI: we add only `boxes`, the one auxiliary signal we actually plumb today. Mask and point hints land when their plumbing is built (§12).
- **No changes to `BoxHintSchedule`, `_box_hint_p`, the per-image `Bernoulli(p_t)` sampler, `_build_geometric_prompt`, `geometric_prompt` fusion, or the starter templates' `box_hint:` blocks.** Those collectively *are* the curriculum from #14 — exactly the auxiliary signal we want to keep.
- **No flip of `enable_inst_interactivity` (`models/sam3.py:616`).** It stays disabled. SAM 3.1's vendor interactive predictor is the only place point / mask prompts exist as *primary* prompts; we never route to it. A short comment is added so the flag is not flipped on casually.
- **No back-compat shim for the field rename.** `Sam3Wrapper.forward(box_hints=...)` is replaced by `support=...` in one shot. The only in-repo callers (loop.py, trainer.py eval panel) are migrated in the same PR. External callers migrate per §10's breaking-change note.
- **Dated specs under `docs/superpowers/specs/`** (e.g. `2026-05-17-training-loop-design.md`, `2026-05-23-multiplex-forward-design.md`) are point-in-time records and remain untouched.

---

## §2 The text-primary invariant

The invariant: **text is always the primary prompt.** The model takes one or more text (class) prompts and segments all matching instances. Geometry (boxes) may *only* ever ride along as an auxiliary localization hint (the `box_hint` curriculum) to accelerate training. There is no box-primary, point-primary, or mask-primary prompt mode, and there is none at inference.

That invariant got blurred by half-built scaffolding:

- `PromptMode = Literal["text", "bbox"]` (`config/schema.py:92`) and the required `DataConfig.prompt_mode` field (`config/schema.py:386`) advertise a bbox primary mode at the surface.
- `data/coco.py` and `data/hf.py` half-implement it: when `prompt_mode == "bbox"` they emit a `BoxPrompts` example (`data/coco.py:346`, `data/hf.py:379`) — but no training entry point accepts it.
- `train/trainer.py:135`, `cli/train_cmd.py:48`, `cli/run_cmd.py:191` each raise on `prompt_mode == "bbox"` — three hand-rolled runtime guards papering over a path the data layer happily produces.
- `Sam3Wrapper._validate_inputs` carries dead BoxPrompts-vs-`box_hints` branches (`models/sam3.py:258`, `279–285`) that can never fire if no caller can construct a `BoxPrompts` batch.

That ambiguity directly caused [#24](https://github.com/NguyenJus/custom-sam-peft/issues/24) to be misfiled — closed as already-satisfied by the `box_hint` curriculum from #14 — because two different concepts shared the word "bbox": a primary prompt mode (which we don't want) and an auxiliary hint (which we kept). This spec removes the dead primary-prompt surface so #24's concern reads unambiguously next time and #88's deferred re-audit can proceed.

The "keep" list (`BoxHintSchedule`, `_box_hint_p`, `_build_geometric_prompt`, `geometric_prompt` fusion, the `box_hint:` blocks in templates) is the auxiliary curriculum from #14. None of it is touched.

---

## §3 `SupportPrompts` container

Auxiliary localization prompts ride alongside text via a new dataclass in `src/custom_sam_peft/data/base.py`, placed next to `TextPrompts`:

```python
@dataclass(frozen=True)
class SupportPrompts:
    """Auxiliary localization prompts that ride alongside TextPrompts.

    Never replaces text; never used at inference. Today carries only optional
    per-image GT box hints (the `box_hint` curriculum from #14). Future fields
    (masks, positive points, negative points) will be added when their
    plumbing is built — see #126 §12.
    """

    boxes: list[Tensor | None] | None = None
```

**Length convention for `boxes`** — identical to today's `box_hints` kwarg:

- Length is `B*K` (image-major, class-minor), where `K` is the number of class prompts per multiplex forward call.
- Each element is either `None` (no hint for that image/class slot) or a `(M_i, 4)` float tensor of absolute pixel xyxy boxes.
- For the common `K=1` case, length is `B` and the ordering is trivially image-major.

This convention is pinned by `_build_geometric_prompt`'s contract (`models/sam3.py:108–166`) and reused unchanged.

**Why a frozen dataclass.** Matches `TextPrompts` (also a frozen dataclass) and the seam discipline in `data/base.py`: simple, hashable, no Pydantic dependency in the data layer. The container is the stable seam between the trainer (where `boxes` is sampled per the `box_hint` curriculum) and `Sam3Wrapper.forward` (where it is unpacked into `_build_geometric_prompt`).

**Future-additive note (strict YAGNI).** Only `boxes` is added today. Adding new optional fields (e.g. `masks`, `point_positive`, `point_negative`) later is non-breaking because every field defaults to `None` and `Sam3Wrapper.forward` ignores `support=None` end-to-end. We deliberately do not pre-add those fields without the plumbing to consume them — empty fields invite misuse.

---

## §4 `Sam3Wrapper.forward` API change

### Old (today)

```python
def forward(
    self,
    images: Tensor,
    prompts: list[Prompts],
    box_hints: list[Tensor | None] | None = None,
) -> dict[str, Any]:
    self._validate_inputs(images, prompts, box_hints)
    out: dict[str, Any] = self.model(images, prompts, box_hints=box_hints)
    return out
```

### New

```python
def forward(
    self,
    images: Tensor,
    prompts: list[Prompts],
    support: SupportPrompts | None = None,
) -> dict[str, Any]:
    self._validate_inputs(images, prompts, support)
    box_hints = support.boxes if support is not None else None
    out: dict[str, Any] = self.model(images, prompts, box_hints=box_hints)
    return out
```

The inner `_Sam3ImageAdapter.forward` (`models/sam3.py:409–471`) keeps its `box_hints` kwarg unchanged — it is the low-level adapter that talks to Meta's `sam3` package. Only the public `Sam3Wrapper.forward` boundary changes; `Sam3Wrapper` unpacks `support.boxes` and forwards as `box_hints=` exactly as today.

### `_validate_inputs` simplification

After `Prompts == TextPrompts`, several branches go away:

1. **Drop the "mixed batch" branch.** Lines 253–259 currently re-check that every prompt is the same type. With only `TextPrompts`, `type(p) is first` is trivially true for any non-empty batch. The check is removed; the per-prompt `isinstance(p, TextPrompts)` check that validates `len(p.classes) ∈ [1, MULTIPLEX_CAP]` (lines 260–265) stays.
2. **Drop the `BoxPrompts`-vs-`box_hints` branch.** Lines 278–283 raise when callers mix `BoxPrompts` with `box_hints=`. That combination is now unconstructable.
3. **Simplify the length check.** Today's branch (lines 286–295) computes `expected_len = b * k` for `TextPrompts` else `b` for "other prompt types". With only `TextPrompts` there is no "else"; the check is `expected_len = b * len(prompts[0].classes)`.
4. **Rename the parameter from `box_hints` to `support`.** Internally the check pulls `boxes = support.boxes if support is not None else None`; the existing length/shape checks (lines 291–302) run against `boxes` unchanged. Per-element shape check (`h.ndim != 2 or h.shape[-1] != 4`) is preserved verbatim.
5. **Drop the shared-class-list check's `if first is TextPrompts:` gate** (lines 268–276). With only `TextPrompts`, the check is unconditional.

The `BoxPrompts` import on line 30 (`from custom_sam_peft.data.base import BoxPrompts, Prompts, TextPrompts`) is reduced to `from custom_sam_peft.data.base import Prompts, SupportPrompts, TextPrompts`.

### Anchoring comment at `enable_inst_interactivity=False`

In `_construct_raw_model` (`models/sam3.py:610–618`), the call site is:

```python
raw_model = sam3.build_sam3_image_model(
    device=device,
    eval_mode=False,
    checkpoint_path=str(ckpt_path),
    load_from_HF=False,
    enable_segmentation=True,
    enable_inst_interactivity=False,
    compile=False,
)
```

Add a one-line comment immediately above `enable_inst_interactivity=False`:

```python
# Disabled by design: this is SAM3's vendor point/box-primary interactive pipe.
# Our prompt invariant is text-primary (see #126); no code path routes to it.
enable_inst_interactivity=False,
```

This is the only "keep with comment" change in `models/sam3.py` beyond the import / docstring / signature / validator edits above.

---

## §5 Data layer changes

### §5.1 `data/base.py`

Delete `BoxPrompts` (lines 18–27); collapse the union on line 30:

```python
Prompts = TextPrompts  # keep the alias so call sites referring to Prompts still resolve
```

Insert `SupportPrompts` between `TextPrompts` and the `Prompts` alias (§3 for the body).

### §5.2 `data/coco.py`

`COCODataset.__init__` (line 124) loses its `prompt_mode` parameter; the validator at lines 136–137 and the `self._prompt_mode` field at line 140 are removed.

`COCODataset._pack_example` (line 263) simplifies dramatically:

- The local import on line 274 (`from custom_sam_peft.data.base import BoxPrompts, Instance, TextPrompts`) becomes `from custom_sam_peft.data.base import Instance, TextPrompts`.
- The `if self._prompt_mode == "text":` gate (line 288) is removed; the body that follows (lines 289–314) becomes unconditional.
- The entire `# bbox mode` branch (lines 316–348) is deleted.

`build_coco` (line 365) drops the `prompt_mode=cfg["prompt_mode"]` argument on line 412.

### §5.3 `data/hf.py`

Symmetric change:

- `HFDataset.__init__` (line 130) loses `prompt_mode` (line 134) and its validator (lines 143–144) and the `self._prompt_mode` field (line 148).
- `HFDataset._pack_example` (line 294): the local import on line 308 becomes `from custom_sam_peft.data.base import Instance, TextPrompts`. The `if self._prompt_mode == "text":` gate at line 322 is removed; the bbox branch (lines 350–381) is deleted.
- `build_hf` (line 402) drops `prompt_mode=cfg["prompt_mode"]` on line 445.

After both edits, `data/coco.py` and `data/hf.py` always emit `TextPrompts`. The `TextPromptConfig`-driven prompt-string assembly (`_build_text_prompts`, `data/coco.py:84`) is unchanged; it is the only text-prompt building block in either adapter.

---

## §6 Trainer / CLI guard removal

Three hand-rolled `prompt_mode == "bbox"` runtime guards are deleted:

| Location | Lines | Action |
|----------|-------|--------|
| `src/custom_sam_peft/train/trainer.py` | 135–140 | Delete the `if cfg.data.prompt_mode == "bbox": raise ValueError(...)` block at the top of `Trainer.__init__`. |
| `src/custom_sam_peft/cli/train_cmd.py` | 48–52 | Delete the `if cfg.data.prompt_mode == "bbox": raise typer.BadParameter(...)` block. |
| `src/custom_sam_peft/cli/run_cmd.py` | 191–195 | Same. |

The schema is now the sole gate. Because `DataConfig` extends `_Strict` (`extra="forbid"`, `config/schema.py:106`) and `prompt_mode` no longer exists as a field, any YAML carrying `prompt_mode:` (any value, not just `"bbox"`) fails at `load_config` time with a Pydantic v2 `ValidationError` of `type="extra_forbidden"` — the standard error our `ConfigError` wrapper already formats.

### Expected error UX

Today (with the hand-rolled guard, on `prompt_mode: bbox`):

```text
typer.BadParameter: prompt_mode='bbox' is not supported for training in v0.
```

After (any `prompt_mode:` key, including `text`):

```text
custom_sam_peft.config.loader.ConfigError: 1 validation error for TrainConfig
data.prompt_mode
  Extra inputs are not permitted [type=extra_forbidden, ...]
```

This is the same error users already see for any other unknown key (e.g. a typo on `data.image_szie`). It surfaces at `load_config` before any model build, so the failure is fast and the message points to the exact offending key.

---

## §7 Training-loop wiring (`box_hint` curriculum flow)

The `box_hint` curriculum is preserved verbatim; only the kwarg name at the call sites changes.

**Sampling, unchanged** (`train/loop.py:199`, `229–247`):

```python
p_t = _box_hint_p(global_step, cfg.train.box_hint)  # decaying probability
...
for i in range(B):
    for c in group:
        c_dense = class_names.index(c)
        row_targets = [inst for inst in targets[i] if inst.class_id == c_dense]
        targets_g.append(row_targets)
        if row_targets and random.random() < p_t:
            box_tensor = torch.stack([inst.box for inst in row_targets])
            hints_g.append(to_device(box_tensor, runtime))
            n_hint_applied += 1
        else:
            hints_g.append(None)
```

`hints_g` ends up image-major / class-minor of length `B*K_g` — exactly `SupportPrompts.boxes`'s layout.

**Forward call wiring change**, two sites:

- `train/loop.py:294` (OOM-ladder microbatch path):

  ```python
  # before
  micro_out = _model(micro_imgs, micro_prompts, box_hints=micro_hints)
  # after
  micro_out = _model(micro_imgs, micro_prompts, support=SupportPrompts(boxes=micro_hints))
  ```

- `train/loop.py:313` (non-ladder path):

  ```python
  # before
  out = model(images, prompts_g, box_hints=hints_g)
  # after
  out = model(images, prompts_g, support=SupportPrompts(boxes=hints_g))
  ```

- `train/trainer.py:485` (eval-panel forward, always None hints):

  ```python
  # before
  out = self.model(..., box_hints=None)
  # after
  out = self.model(..., support=None)
  ```

`SupportPrompts` is imported from `custom_sam_peft.data.base` at the top of `train/loop.py` (next to the existing `TextPrompts` import — verified via the `prompts_g = [TextPrompts(...) ...]` construction on line 231). `train/trainer.py:485` does not need the import because it passes `None`.

The `box_hint/applied` metric (`train/loop.py:397`) and the `box_hint/p` flush key (line 416) are unchanged — they read `n_hint_applied` and `p_t` from `StepResult`, not from the kwarg name.

---

## §8 Configs, templates, wizard

### §8.1 Example configs (8 files)

All eight example configs under `configs/examples/` carry a `prompt_mode: text` line that must be removed. The lines (verified by `grep`):

| File | Line |
|------|------|
| `configs/examples/coco_text_lora.yaml` | 28 |
| `configs/examples/coco_text_qlora.yaml` | 28 |
| `configs/examples/coco_text_lora_subset.yaml` | 22 |
| `configs/examples/coco_text_no_val.yaml` | 26 |
| `configs/examples/coco_text_auto_split.yaml` | 27 |
| `configs/examples/min_gpu_qlora.yaml` | 27 |
| `configs/examples/gpu_smoke_lora.yaml` | 20 |
| `configs/examples/gpu_smoke_qlora.yaml` | 20 |

Each edit is a single-line delete; no other text on the line.

### §8.2 Unified template

`src/custom_sam_peft/cli/templates/config_full.yaml` line 23 is `  prompt_mode: text` — delete the line. The surrounding `data:` block (lines 20–22 dataset block, 22 validation block, 24 `image_size: 1008`) keeps the existing layout; the deletion does not break any `$` placeholder.

### §8.3 Setup wizard

`src/custom_sam_peft/cli/setup_wizard.py` — verified by `grep`: there is no `prompt_mode` emit logic in the wizard (the prior interactive-setup-wizard spec referenced `prompt_mode: text` as *hardcoded in the template*, not in wizard Python code). No wizard code change is required; the template edit above is sufficient.

### §8.4 Verification

After the eight example edits, the template edit, and the source edits, the project should have **zero** remaining `prompt_mode` occurrences. Run:

```bash
grep -rn 'prompt_mode' configs/ src/ docs/ARCHITECTURE.md docs/config-schema.md CHANGELOG.md
```

Expected: no matches. (The dated specs under `docs/superpowers/specs/` are point-in-time records and may still mention `prompt_mode`; that is intentional.)

---

## §9 Tests

### §9.1 Remove / rewrite

| Test file | Change | Intent of the rewrite |
|-----------|--------|-----------------------|
| `tests/unit/test_data_base.py` | Remove the `BoxPrompts` construction test (lines 8, 19–24). | `BoxPrompts` no longer exists; the test would fail to import. |
| `tests/unit/test_data_coco.py` | Remove `BoxPrompts` import (line 178) and every `prompt_mode="bbox"` test (e.g. lines 203, 255, 434–470). Update remaining `prompt_mode="text"` constructions (lines 268, 282, 326, 368, 417, 512) to drop the kwarg — `COCODataset.__init__` no longer takes it. | After the data-layer collapse, every `COCODataset` call site emits `TextPrompts`; there is no bbox variant left to test. |
| `tests/unit/test_data_hf.py` | Same pattern: remove `BoxPrompts` import (line 120) and `prompt_mode="bbox"` tests (lines 150, 186); strip the kwarg from text-mode constructions. | Symmetric to COCO. |
| `tests/unit/test_data_hf_limit.py` | Remove the `ds._prompt_mode = "bbox"` line (line 28) and any assertions that depend on it. | The attribute no longer exists. |
| `tests/unit/test_data_collate.py` | Remove `BoxPrompts` import (line 8) and the `BoxPrompts(...)` construction at line 38; remove the `isinstance(batch["prompts"][1], BoxPrompts)` assertion (line 48). Keep all other collate assertions. | The collator never had to handle BoxPrompts at the trainer boundary (training rejected them anyway); the test was exercising a dead path. |
| `tests/unit/test_sam3_wrapper.py` | Remove `BoxPrompts` import (line 8) and the `BoxPrompts(boxes=..., class_ids=...)` mixed-batch test (line 42). Keep all `TextPrompts` validation tests. | The mixed-batch branch in `_validate_inputs` is deleted. |
| `tests/unit/test_sam3_wrapper_box_hints.py` → rename to **`tests/unit/test_sam3_wrapper_support.py`** | Rewrite all tests to construct `SupportPrompts(boxes=...)` and call `wrapper(images, prompts, support=...)`. Drop the `BoxPrompts`-cannot-combine-with-`box_hints` test (lines 50–51); that branch is gone. Keep the length-mismatch and shape-mismatch tests, retargeted at `support.boxes`. | Exercises the new auxiliary-prompt API; the old test file's name no longer reflects what it asserts. |
| `tests/unit/test_config_schema.py` | Remove `"prompt_mode": "bbox"` from any fixture (line 19); remove `test_invalid_prompt_mode_rejected` (lines 48–50) — the field no longer exists. Remove `"prompt_mode": "text"` from the remaining valid-config fixture (line 201). | The schema field is deleted; tests that asserted its values are obsolete. |
| `tests/unit/test_trainer_guards.py` | Remove `prompt_mode` from `_cfg`'s signature/body (lines 20, 28) and delete `test_trainer_rejects_bbox_prompt_mode` (lines 50–55). Keep the qlora-optimizer-coercion tests. | The hand-rolled trainer guard is deleted; the schema is the gate. |
| `tests/unit/test_cli.py` | Delete `test_train_rejects_bbox_prompt_mode` (lines 95–115). In the other config YAMLs embedded in this file (lines 130, 175, 209), drop the `prompt_mode: text` line. | The CLI no longer raises on `prompt_mode == "bbox"` (the schema does, with a different error); the bespoke CLI test is replaced by the schema-level test in §9.2. |
| `tests/unit/test_stubs_raise.py` | Delete the bbox-rejection test at lines 22, 40–45. | The trainer guard that the test was exercising is removed. |
| `tests/conftest.py` | Drop the `prompt_mode="bbox"` line in the shared fixture (line 163); if the fixture's purpose was bbox-mode coverage, repurpose it or delete it. | The fixture currently produces an unconstructable config; with the field removed it must be either updated or dropped. |
| `tests/fixtures/tiny_sam3_stub.py` | Update the docstring (line 27) to drop the `BoxPrompts` mention. | Stale reference. |

### §9.2 Add

| Test (new) | Intent |
|------------|--------|
| `tests/unit/test_config_schema.py::test_prompt_mode_rejected_by_schema` | Load a config dict that carries `prompt_mode: "text"` (or any other value); assert that `TrainConfig.model_validate(...)` raises a Pydantic `ValidationError` whose first error has `type == "extra_forbidden"` and whose loc tuple ends with `"prompt_mode"`. This is the canonical proof that the schema is the sole gate — for any value of `prompt_mode`, not just `"bbox"`. |
| `tests/unit/test_data_base.py::test_support_prompts_dataclass` | `s = SupportPrompts(boxes=[torch.zeros(2, 4), None])`; assert `s.boxes[0].shape == (2, 4)` and `s.boxes[1] is None`; assert `SupportPrompts()` constructs with `boxes is None`; assert `SupportPrompts` is frozen (`dataclasses.is_dataclass(s)` and `replace(s, boxes=None)` works while `s.boxes = ...` raises `FrozenInstanceError`). |

### §9.3 Coverage continuity for the `box_hint` curriculum

Existing tests that cover `_box_hint_p`, the OOM-ladder microbatch slicing of `hints_g`, and the `box_hint/applied` metric (under `tests/unit/` for `train/loop.py`) remain green: the curriculum sampling and metric paths do not change. The only surface change is the `support=SupportPrompts(boxes=...)` wrapping at the `Sam3Wrapper.forward` boundary, which is exercised by the rewritten `test_sam3_wrapper_support.py` and by the integration paths the existing trainer tests already touch.

---

## §10 Docs & CHANGELOG

### §10.1 `docs/ARCHITECTURE.md`

Add a short sentence near the top stating the invariant ("Text is the only primary prompt; auxiliary localization hints ride alongside via `SupportPrompts` — never replace text, never used at inference"). Replace line 15 (`base.py            Example, Prompts (TextPrompts | BoxPrompts), Dataset protocol`) with:

```text
  base.py            Example, Prompts (= TextPrompts), SupportPrompts, Dataset protocol
```

### §10.2 `docs/config-schema.md`

Delete the `data.prompt_mode` row (line 52). The surrounding table (lines 47–58) re-flows without other changes; check the row separators and the `data.text_prompt.mode` row's preceding context still reads cleanly.

### §10.3 `CHANGELOG.md`

Add a top-of-file entry under a new `## [Unreleased]` heading (or the next version heading, per the release workflow). Exact phrasing:

```markdown
### Breaking — text-primary prompt invariant (#126)

- **schema**: removed the `data.prompt_mode` field. Any config that carries
  `prompt_mode:` (any value) now fails at load with a Pydantic
  `extra_forbidden` error. Migration: delete the line from your YAML.
- **api**: replaced `Sam3Wrapper.forward(..., box_hints=...)` with
  `Sam3Wrapper.forward(..., support=SupportPrompts(boxes=...))`. Downstream
  callers that pass per-image GT boxes as a training hint must wrap them in
  a `SupportPrompts(boxes=...)` and pass via `support=`. Passing
  `support=None` (the default) is equivalent to today's `box_hints=None`.
- **types**: removed `BoxPrompts` and `PromptMode`. `Prompts` is now an alias
  for `TextPrompts`.
- **trainer/CLI**: removed three hand-rolled `prompt_mode == "bbox"` guards
  (`train/trainer.py`, `cli/train_cmd.py`, `cli/run_cmd.py`) — the schema is
  the sole gate.

The `box_hint` training curriculum (`train.box_hint.*`, `BoxHintSchedule`) is
unchanged — it continues to sample per-image GT boxes alongside text prompts
as an auxiliary localization hint, now flowing through `SupportPrompts`.
```

Dated specs under `docs/superpowers/specs/` are intentionally not edited.

---

## §11 Acceptance criteria

- [ ] `BoxPrompts`, `PromptMode`, `DataConfig.prompt_mode`, and the `box_hints=` kwarg on `Sam3Wrapper.forward` no longer exist anywhere under `src/custom_sam_peft/`.
- [ ] `Prompts` resolves to `TextPrompts` (alias preserved to minimize call-site churn).
- [ ] `SupportPrompts` exists in `src/custom_sam_peft/data/base.py` with a single optional field `boxes: list[Tensor | None] | None = None`, frozen, dataclass.
- [ ] `Sam3Wrapper.forward(..., support=SupportPrompts(boxes=...))` is the auxiliary-prompt API; `Sam3Wrapper.forward` no longer accepts `box_hints=`.
- [ ] Loading a config with `prompt_mode:` (any value) raises a Pydantic `ValidationError` of `type == "extra_forbidden"` (verified by `test_prompt_mode_rejected_by_schema`).
- [ ] The `box_hint` curriculum trains end-to-end via the new path: `_box_hint_p`, the Bernoulli per-image sampling, and the `box_hint/applied` metric are unchanged; both shipped templates (`config_full.yaml` rendered with `lora` and with `qlora`) load and train.
- [ ] `enable_inst_interactivity=False` is retained at `models/sam3.py:616` with the new anchoring comment citing #126.
- [ ] `grep -rn 'prompt_mode' configs/ src/ docs/ARCHITECTURE.md docs/config-schema.md CHANGELOG.md` returns no matches.
- [ ] `docs/ARCHITECTURE.md`, `docs/config-schema.md`, and `CHANGELOG.md` are updated per §10.
- [ ] `pytest` (full suite), `mypy`, and `ruff` are green; coverage gate holds.

---

## §12 Out-of-scope / future

- **`SupportPrompts.masks: list[Tensor | None] | None = None`** — added when a mask-hint curriculum (e.g. partial-mask supervision, prior-mask priming) is plumbed end-to-end. Today there is no source of mask hints in the training loop and no consumer in `_build_geometric_prompt` / Meta's `Prompt` (its `mask_*` fields would need separate wiring).
- **`SupportPrompts.point_positive: list[Tensor | None] | None = None` and `SupportPrompts.point_negative: list[Tensor | None] | None = None`** — added when a point-hint curriculum is built. Meta's `Prompt` has `point_*` slots, but our trainer never samples points today and Meta's vendor interactive predictor (the only place point prompts exist as a *primary* signal in the broader codebase) stays disabled by design.
- **`#88` `box_hint` schedule re-audit** — unblocked by this PR (the misfiling concern from #24 is resolved); proceeds independently.
- **No deprecation shim for `Sam3Wrapper.forward(box_hints=...)`**. Pre-1.0, single PR, no back-compat path. The migration is the one-line wrap shown in §10.3.
