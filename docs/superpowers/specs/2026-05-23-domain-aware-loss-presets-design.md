# spec/domain-aware-loss-presets — Domain-aware loss-function presets with class-imbalance dial (issue #112)

**Status:** Draft (2026-05-23)
**Tracking:** [#112](https://github.com/NguyenJus/custom-sam-peft/issues/112) — *feat(train): domain-aware loss-function presets (natural / medical / satellite / …) with class-imbalance dial*
**Scope:** Replace the internal flat `LossConfig` (`w_mask`, `w_box`, `w_obj`, `w_presence`, `matcher_weights`, `focal_gamma`, `focal_alpha`) with a `(preset, class_imbalance, overrides)` triple that resolves to thirteen typed knobs across four loss axes (mask / box / obj / presence), break `models/losses.py` into a `models/losses/` package with a `terms/` library and a `compose.py` bundle, expose `--preset` / `--class-imbalance` to `csp init` and a "Resolved losses" table to `csp doctor`, and persist a `run_dir/loss_bundle.json` sidecar so a finished run records exactly which loss bundle ran. Clean schema break (no back-compat aliases; pre-1.0 per #70).

**Builds on:**
[`2026-05-22-domain-aware-augmentation-presets-design.md`](2026-05-22-domain-aware-augmentation-presets-design.md) — that spec defined the `Preset` literal, the `(preset, intensity, overrides)` resolver pattern, the locked-off / WARN convention, the `run_dir/*.json` sidecar shape, the `csp init` template-substitution mechanism, and the `csp doctor --config` table+JSON pattern. This spec is the loss-side counterpart: it **reuses** the `Preset` literal verbatim (no duplicate type), mirrors the resolver structure, and slots a second sidecar + a third doctor table into the same wire-up points. The diverging axis name (`class_imbalance` vs `intensity`) is deliberate and is called out in issue #112's body: losses don't have a single "strength" dial — the dominant cross-domain tuning axis is class imbalance / rare-positive emphasis.

---

## 1. Goals

- Replace the demoted-internal flat `LossConfig` (`config/_internal.py`) with a higher-level user-facing `(preset, class_imbalance, overrides)` API so users pick a domain ("medical", "satellite", "microscopy", "natural") and an imbalance tier ("balanced", "moderate", "severe") instead of hand-tuning thirteen knobs.
- Define thirteen typed knobs across four axes — `mask_family`, `box_family`, `obj_family`, `presence_family` (term selection) and `w_mask`, `w_box`, `w_obj`, `w_presence`, `focal_gamma`, `focal_alpha`, `tversky_alpha`, `tversky_gamma`, `boundary_weight` (hyperparameters) — with a frozen lookup table for each `(preset, class_imbalance)` combination.
- Make domain safety explicit: term-family choices that are likely wrong for a domain (e.g. overriding `mask_family` away from the rare-positive-friendly choice under `preset: medical`) emit a `logging.WARNING` naming the knob, the preset, and the reason. **Locked-off overrides are NOT stripped** — the user's explicit override always wins; the warning is the contract (matches #75).
- Ship a fourteen-class **term library** under `models/losses/terms/` — one `nn.Module`-style callable per family — so the resolved configuration can be assembled into a `LossBundle` with no `if/elif` ladders in the trainer.
- Persist the resolved bundle to `run_dir/loss_bundle.json` next to `config.yaml` and `augmentation_pipeline.json` so each run is fully reproducible from the surface fields, and cross-version reproducible by copying `resolved` into `overrides:` under `preset: custom`.
- Expose presets in the CLI: `csp init --preset X --class-imbalance Y` renders the chosen pair into the starter template alongside the existing `--intensity` flag from #75; `csp doctor --config path.yaml` shows a "Resolved losses" table next to "Resolved augmentations" and "Normalization" so users can dry-run their config without launching training.
- 100% CPU-testable. No real model load, no GPU.

## 2. Non-goals

- **User-supplied loss callable** (`train.loss.custom_callable: "pkg.mod:fn"`) — possible v1.1; out of scope for this PR.
- **User-defined presets** (extending `PRESET_TABLE` from outside the package) — v1.1.
- **3D / volumetric loss terms** (surface Dice, 3D Tversky) — #110 territory; family vocabulary extension only when 3D imagery lands.
- **`box_family: none`** — achieved via `w_box=0.0`; no separate family literal needed.
- **Per-knob application-probability overrides** — presets fix term selection; weights tune contribution; no per-step probability dial.
- **Boundary-loss schedule** — Kervadec et al. ramp from 0.01 to ~1.0 across training; v1 uses a constant `boundary_weight`. Schedule support is a follow-up if/when a user files it.
- **Citation pass across the whole codebase** — separately tracked by issue #120; this spec emits `# citation needed` comments at the table sites that need them, but does not block on closing them.

## 3. Current state

The loss-function surface is split across two files:

```python
# src/custom_sam_peft/config/_internal.py
@dataclass
class LossConfig:
    """Internal config — not user-set."""
    w_mask: float = 1.0
    w_obj: float = 1.0
    w_presence: float = 1.0
    w_box: float = 0.0
    matcher_weights: MatcherWeights = field(default_factory=MatcherWeights)
    focal_gamma: float = 2.0
    focal_alpha: float = 0.25
```

```python
# src/custom_sam_peft/models/losses.py
def mask_loss(pred, target):       # fixed 50/50 Dice + BCE
def box_loss(pred, target):        # fixed smoothL1 + (1 - GIoU)
def objectness_loss(obj_logits, matched_mask, gamma, alpha):  # focal BCE
def presence_loss(img_presence, image_has_target):            # plain BCE
def total_loss(outputs, targets, cfg: LossConfig) -> dict[str, Tensor]:
    canonical = meta_to_canonical(outputs)
    matcher = HungarianMatcher(**cfg.matcher_weights)
    indices = matcher(canonical, targets)
    # ... gather matched pairs, call the four term functions, weight+sum into "total"
```

The docstring on the `_internal.py` `LossConfig` calls itself "not user-set", but the surface is **partially user-reachable today**: both shipped templates carry a `train.loss:` block (lines ~79–86 of each) that sets `w_mask`, `w_obj`, `w_presence`, and a nested `matcher_weights.lambda_mask`. So the migration must (a) delete the flat-knob YAML keys, (b) replace them with the new triple, and (c) update the templates. The dataclass `LossConfig` symbol itself is fully replaced — there is no parallel "internal" `LossConfig` after this PR; the internal plumbing for the thirteen resolved knobs lives in `models/losses/presets.py::ResolvedLosses` instead.

`total_loss` is called twice in `src/custom_sam_peft/train/loop.py` (lines 257 and 278, once in the OOM-ladder path and once in the normal path) with `cfg.train.loss` as the third argument. Both call sites continue to work through the migration via a `total_loss` re-exported shim.

The trainer (`src/custom_sam_peft/train/trainer.py::_setup_run_dir`, lines ~188–207) already writes `run_dir/config.yaml` and `run_dir/augmentation_pipeline.json`. The new `run_dir/loss_bundle.json` write slots in immediately after the augmentation-pipeline write, sharing the same atomic-failure semantics.

`csp init` (`src/custom_sam_peft/cli/init_cmd.py`) already does template substitution for `${preset}`, `${intensity}`, and `${overrides_block}` from #75. The new flag adds `${class_imbalance}` and `${loss_overrides_block}` to the substitution dict and the templates gain matching placeholders under their `train.loss:` blocks.

`csp doctor` (`src/custom_sam_peft/cli/doctor_cmd.py`) already renders two `--config`-conditional tables ("Resolved augmentations", "Normalization") and a `resolved_config` JSON sub-block. The new code adds a third table ("Resolved losses") and a third JSON sub-key (`loss`) using the same pattern.

## 4. Schema

New definitions in `src/custom_sam_peft/config/schema.py`. The dataclass `LossConfig` in `config/_internal.py` is **deleted** (not aliased, not deprecated) — clean break under #70's pre-1.0 schema-break allowance. The new `LossConfig` is a Pydantic `_Strict` model living in `schema.py` alongside `AugmentationsConfig`, replacing the prior re-export from `_internal`.

`MatcherWeights` stays in `config/_internal.py` (it remains an internal contract between the resolver and the Hungarian matcher; its three `lambda_*` fields are not exposed at the top level of the user-facing schema — they live under `LossOverrides.matcher_weights`).

```python
# src/custom_sam_peft/config/schema.py

# Reused verbatim from the existing schema — do NOT redefine.
# Preset = Literal["natural", "medical", "satellite", "microscopy", "none", "custom"]

ClassImbalance  = Literal["balanced", "moderate", "severe"]

MaskFamily      = Literal["bce", "dice", "dice_bce",
                          "focal_bce", "focal_dice",
                          "focal_tversky", "boundary"]
BoxFamily       = Literal["l1_giou", "giou_only", "ciou"]
ObjFamily       = Literal["focal_bce", "bce"]
PresenceFamily  = Literal["bce", "focal_bce"]


class LossOverrides(_Strict):
    """Per-knob overrides. All None → inherit from (preset, class_imbalance).

    Setting any field to a non-None value replaces just that field in the
    resolved table. Extra keys are rejected (extra="forbid"); typos surface
    at config-load time.
    """

    # Term selection (4 axes)
    mask_family:     MaskFamily     | None = None
    box_family:      BoxFamily      | None = None
    obj_family:      ObjFamily      | None = None
    presence_family: PresenceFamily | None = None

    # Weights (4)
    w_mask:          PositiveFloat  | None = None
    w_box:           float          | None = Field(default=None, ge=0.0)
    w_obj:           PositiveFloat  | None = None
    w_presence:      PositiveFloat  | None = None

    # Focal params (2)
    focal_gamma:     PositiveFloat  | None = None
    focal_alpha:     float          | None = Field(default=None, ge=0.0, le=1.0)

    # Tversky params (2)
    tversky_alpha:   float          | None = Field(default=None, ge=0.0, le=1.0)
    tversky_gamma:   PositiveFloat  | None = None

    # Boundary blend coefficient (1)
    boundary_weight: float          | None = Field(default=None, ge=0.0, le=1.0)

    # Matcher contract (internal sub-model, optional override)
    matcher_weights: MatcherWeights | None = None


class LossConfig(_Strict):
    preset:          Preset         = "natural"
    class_imbalance: ClassImbalance = "balanced"
    overrides:       LossOverrides  = Field(default_factory=LossOverrides)
```

Type-import notes for the planner:

- `Preset` is the existing `Literal` already defined in `schema.py` (line 114). Do not redefine; do not import from anywhere else.
- `MatcherWeights` is imported from `config._internal` (unchanged dataclass). Pydantic v2 accepts dataclass-typed fields via `arbitrary_types_allowed=True` *or* converts them via a `field_validator`. The planner picks the smallest mechanism that keeps `_Strict`'s `extra="forbid"` honored — preferred: declare `LossOverrides.matcher_weights: MatcherWeights | None` with a `@field_validator` that accepts a dict and routes through `MatcherWeights(**d)`. (If easier, promote `MatcherWeights` to a Pydantic `_Strict` model in `_internal.py` — that is acceptable; it does not alter the internal-vs-user contract because users still only see it under `LossOverrides`.)
- `PositiveFloat` is already imported by `schema.py`.
- `_Strict` (`extra="forbid"`) is the existing base in `schema.py`; both `LossOverrides` and `LossConfig` inherit it.

`TrainHyperparams.loss: LossConfig = Field(default_factory=LossConfig)` — field name unchanged. The only downstream change is that the **dataclass** `LossConfig` is gone; readers of `cfg.train.loss` now hit a Pydantic model whose only top-level fields are `preset`, `class_imbalance`, `overrides`.

### 4.1 Knob semantics

- **Term-family knobs** (`mask_family`, `box_family`, `obj_family`, `presence_family`) are literal selectors that pick which class from `models/losses/terms/` is instantiated for that axis.
- **Weight knobs** (`w_mask`, `w_box`, `w_obj`, `w_presence`) multiply the corresponding axis's loss before the `"total"` sum. `w_box=0.0` is the documented way to disable the box term — equivalent to a hypothetical `box_family: none`.
- **Focal params** (`focal_gamma`, `focal_alpha`) are consumed by any family whose name starts with `focal_` (`focal_bce`, `focal_dice`, `focal_tversky`) and ignored by the others.
- **Tversky params** (`tversky_alpha`, `tversky_gamma`) are consumed by `focal_tversky` and (for `tversky_alpha`) by a plain `tversky` formulation inside `boundary` if `mask_family == "boundary"` uses Tversky as its base; in this v1, `boundary` is `boundary_weight · Kervadec + (1 - boundary_weight) · Dice`, so Tversky params are ignored under `mask_family: boundary`.
- **`boundary_weight`** is in `[0, 1]` and is the blend coefficient for `boundary` family only. At `0.0`, the boundary family degenerates to plain `Dice`; at `1.0`, it is pure Kervadec. Out of `[0, 1]` is rejected at schema-validation time.

Every irrelevant hyperparameter is **still set to a concrete float** by the resolver (e.g. `tversky_alpha=0.5` is in the resolved bundle even when `mask_family: dice_bce`). The term constructors accept all thirteen hyperparameters and silently ignore the irrelevant ones, so the composer's `build_loss_bundle` call is uniform across family selections — no `if/elif` ladders.

### 4.2 Migration (clean break)

There is no back-compat alias. Every callsite is migrated in the same PR:

- Two templates: `cli/templates/coco_text_lora.yaml`, `cli/templates/coco_text_qlora.yaml` — the existing `train.loss:` block (lines ~79–86) is replaced wholesale with the §11.3 `${preset} / ${class_imbalance} / ${loss_overrides_block}` block.
- One internal module: `config/_internal.py` — the dataclass `LossConfig` is deleted. `LossConfig` no longer appears in `_internal.py`'s `__all__`-equivalent or in `schema.py`'s re-export block. `MatcherWeights` stays.
- One module renamed: `models/losses.py` becomes a package `models/losses/` (see §7). The shim `total_loss` symbol is re-exported from `models/losses/__init__.py` so the two trainer call sites in `train/loop.py` continue to work unmodified through the migration.
- One trainer file: `train/trainer.py::_setup_run_dir` gains a `loss_bundle.json` write next to the existing `augmentation_pipeline.json` write.
- `cli/init_cmd.py` gains a `--class-imbalance` flag and a `${loss_overrides_block}` substitution.
- `cli/doctor_cmd.py` gains a third resolved-config table and a third JSON sub-key.

Tests are updated to:
- Stop importing `LossConfig` from `_internal.py` (the symbol is gone).
- Stop using `LossConfig` as a dataclass with positional `w_mask=...` keyword args.
- Use `LossConfig(preset="...", class_imbalance="...", overrides=LossOverrides(...))` instead.
- Tests that wanted "default loss behavior" use `LossConfig(preset="none")` (numerically equivalent to the pre-#112 defaults).

Test files touched: `test_loss_config.py` (rewritten — see §12), `test_box_hint_schedule.py` (drops `LossConfig` import and the `test_loss_config_default_w_box_is_zero` test in favor of the new `test_loss_presets.py`), `test_config_schema.py`, `test_trainer_nan_behavior.py`, `test_trainer_run_dir.py`, `test_data_coco.py`, `test_data_hf.py`, plus the integration tests `test_train_resume.py`, `test_train_end_to_end.py`.

A user with a pre-existing config carrying `train.loss: {w_mask: 1.0, w_obj: 1.0, matcher_weights: {lambda_mask: 5.0}}` will fail `load_config` with a pydantic `ValidationError` ("extra fields not permitted: w_mask, w_obj, matcher_weights"). The PR description includes a one-line migration recipe:

> Before: `{w_mask: X, w_obj: Y, focal_gamma: G, matcher_weights: {…}}`
> After:  `{preset: custom, overrides: {w_mask: X, w_obj: Y, focal_gamma: G, matcher_weights: {…}}}`

## 5. Preset × class_imbalance table

Single frozen source of truth, lives as `PRESET_TABLE` in the new module `src/custom_sam_peft/models/losses/presets.py`. Twelve `(preset, class_imbalance)` cells for the four real domains; `none` and `custom` are handled by short-circuit (§7), not by table lookup. Microscopy is a strict alias of medical for v1 (see §5.2).

| knob | nat/bal | nat/mod | nat/sev | med/bal | med/mod | med/sev | sat/bal | sat/mod | sat/sev | mic/bal | mic/mod | mic/sev |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `mask_family`     | dice_bce  | dice_bce  | focal_dice    | focal_dice | focal_tversky | boundary | dice_bce | focal_dice | focal_tversky | focal_dice | focal_tversky | boundary |
| `box_family`      | l1_giou  | l1_giou  | l1_giou  | l1_giou  | l1_giou  | l1_giou  | l1_giou  | l1_giou  | l1_giou  | l1_giou  | l1_giou  | l1_giou  |
| `obj_family`      | focal_bce | focal_bce | focal_bce | focal_bce | focal_bce | focal_bce | focal_bce | focal_bce | focal_bce | focal_bce | focal_bce | focal_bce |
| `presence_family` | bce      | bce      | bce      | bce      | bce      | bce      | bce      | bce      | bce      | bce      | bce      | bce      |
| `w_mask`          | 1.0      | 1.0      | 1.0      | 1.0      | 1.0      | 1.0      | 1.0      | 1.0      | 1.0      | 1.0      | 1.0      | 1.0      |
| `w_box`           | 0.0      | 0.0      | 0.0      | 0.0      | 0.0      | 0.0      | 0.0      | 0.0      | 0.0      | 0.0      | 0.0      | 0.0      |
| `w_obj`           | 1.0      | 1.0      | 1.0      | 1.0      | 1.0      | 1.0      | 1.0      | 1.0      | 1.0      | 1.0      | 1.0      | 1.0      |
| `w_presence`      | 1.0      | 1.0      | 1.0      | 1.0      | 1.0      | 1.0      | 1.0      | 1.0      | 1.0      | 1.0      | 1.0      | 1.0      |
| `focal_gamma`     | 2.0      | 2.5      | 3.0      | 2.0      | 2.5      | 3.0      | 2.0      | 2.5      | 3.0      | 2.0      | 2.5      | 3.0      |
| `focal_alpha`     | 0.25     | 0.25     | 0.25     | 0.25     | 0.25     | 0.25     | 0.25     | 0.25     | 0.25     | 0.25     | 0.25     | 0.25     |
| `tversky_alpha`   | 0.5      | 0.5      | 0.6      | 0.6      | 0.7      | 0.8      | 0.5      | 0.6      | 0.7      | 0.6      | 0.7      | 0.8      |
| `tversky_gamma`   | 1.0      | 1.0      | 0.75     | 0.75     | 0.75     | 0.75     | 1.0      | 0.75     | 0.75     | 0.75     | 0.75     | 0.75     |
| `boundary_weight` | 0.0      | 0.0      | 0.0      | 0.0      | 0.0      | 0.2      | 0.0      | 0.0      | 0.0      | 0.0      | 0.0      | 0.2      |

The microscopy columns (`mic/bal`, `mic/mod`, `mic/sev`) are **byte-equal** to the medical columns by construction. See §5.2.

`preset: none` → use the **legacy hardcoded values** (preserves pre-#112 trainer behavior). Concretely:
`mask_family=dice_bce`, `box_family=l1_giou`, `obj_family=focal_bce`, `presence_family=bce`,
`w_mask=1.0, w_box=0.0, w_obj=1.0, w_presence=1.0`, `focal_gamma=2.0`, `focal_alpha=0.25`.
`tversky_*` and `boundary_weight` are set to neutral values (`tversky_alpha=0.5`, `tversky_gamma=1.0`, `boundary_weight=0.0`) because the chosen families don't use them; the resolver still populates them so `ResolvedLosses` has every field set. `class_imbalance` is ignored; no warns.

`preset: custom` → seed = `PRESET_TABLE[("natural", "balanced")]`; overrides applied on top; `class_imbalance` is ignored; locked-off checks are skipped (user is fully explicit).

### 5.1 Rationale for the table values

- **All `box_family`, `obj_family`, `presence_family` and the four `w_*` rows are constant across cells.** v0 trains text-only with no box supervision (`w_box=0.0`); changing the box family without enabling its weight is pointless. Objectness and presence are stable per-query / per-image supervisions that don't benefit from family swaps at this rare-positive intensity range. These rows are constant so reviewers can verify "the table only varies what actually changes per domain × imbalance".
- **`mask_family` is the dominant axis.** It walks: `dice_bce` (well-conditioned balanced data) → `focal_dice` (mild imbalance) → `focal_tversky` (FN bias) → `boundary` (rare positives + tight margins). Natural never reaches `boundary` (no clinical-margin requirement); medical does at `severe`; satellite stops at `focal_tversky` (tiny-object precision benefits from focal-Tversky's FN bias but boundary loss is overkill for sparse-instance overhead imagery); microscopy mirrors medical (see §5.2).
- **`focal_gamma`** grows linearly with class-imbalance tier (2.0 → 2.5 → 3.0) per the RetinaNet ablation (Lin et al. 2017, Table 1) — γ=2 is the cited best on balanced data; higher γ down-weights easy negatives more aggressively as positives become rarer.
- **`tversky_alpha`** is the FN/FP tradeoff knob. α=0.5 ≡ Dice (no FN bias); α=0.7 is Salehi et al. 2017's best result on MS lesions; α=0.8 is a progression for `severe`. Natural stays at 0.5 (no bias needed); medical jumps to 0.6 at `balanced` because even balanced medical data tends to have small-object rarity; satellite tracks medical at one tier lower.
- **`tversky_gamma`** = 1.0 reduces Focal-Tversky to plain Tversky (no focal exponent); = 0.75 is Abraham & Khan 2019's best result on ISIC. Cells where the family is not Focal-Tversky still get a `tversky_gamma` value (1.0 or 0.75) but the term ignores it.
- **`boundary_weight` = 0.2** at the two `severe`+boundary cells is a representative Kervadec et al. 2019 blend coefficient. The paper uses a schedule (0.01 → ~1.0); v1 uses a constant — schedule support is a deliberate non-goal (§2).

### 5.2 Microscopy = alias of medical

Per issue #112's body §"Sub-domains": *"Microscopy starts as an alias of medical until we have a justified divergence — e.g., a clDice term once we have a real user."*

`PRESET_TABLE[("microscopy", x)]` is **byte-equal** to `PRESET_TABLE[("medical", x)]` for `x ∈ {balanced, moderate, severe}` in v1. The four `microscopy/*` columns above are the same as the four `medical/*` columns. The resolver does **not** special-case the alias — it just looks up `(microscopy, x)` and returns the dict, which happens to match `(medical, x)`. This keeps the resolver's branch count down and surfaces any future divergence as an explicit table edit.

### 5.3 Citation convention

Each cell value in `PRESET_TABLE` carries a `# cite: <tag>` inline comment in `models/losses/presets.py`. The tags themselves are defined in a module docstring at the top of the file:

```python
"""...
Citation tags:
  (A) #112 issue body            — cell lifted verbatim from the issue's draft table
  (B) preserved pre-#112         — matches today's hardcoded trainer behavior in losses.py
  (C) Lin et al. 2017 (RetinaNet/focal loss)         — γ=2.0, α=0.25 from Table 1
  (D) Abraham & Khan 2019 (focal Tversky)            — γ=0.75 best on ISIC; their γ=4/3 in
                                                       inverse-convention notation maps to
                                                       0.75 in our `(1-TI)^γ` form
  (E) Salehi et al. 2017 (Tversky loss)              — α=0.7 best on MS lesions
  (F) degenerate-case identity                       — α=0.5 reduces Tversky to Dice;
                                                       γ=1.0 reduces Focal-Tversky to Tversky
  (G) alias-of-medical                               — microscopy copies medical (citation
                                                       needed — see #120)
  (H) Kervadec et al. 2019 (boundary loss)           — blend coefficient ~0.2 representative;
                                                       paper uses a schedule (out of scope v1)

Cells lacking a firm cite carry an inline `# citation needed` comment. Per user
direction (recorded in #112 brainstorming summary): silent defaults are fine —
expert users will spot and file issues against the master tracker #120.
"""
```

Cite tag assignments per cell (literal — planner reproduces these exactly):

- **All `mask_family` values** → `# cite: (A)` (lifted from issue body).
- **All `box_family`, `obj_family`, `presence_family`, `w_mask`, `w_box`, `w_obj`, `w_presence` values** → `# cite: (B)` (preserved pre-#112).
- **`focal_gamma`**:
  - `nat/bal`, `med/bal`, `sat/bal`, `mic/bal` (=2.0) → `# cite: (A,C)`.
  - all `mod` and `sev` cells (=2.5, 3.0) → `# cite: (A)  # citation needed`.
- **`focal_alpha`** (all =0.25) → `# cite: (A,C)`.
- **`tversky_alpha`**:
  - `nat/bal`, `nat/mod`, `sat/bal` (=0.5) → `# cite: (F)`.
  - `med/bal`, `sat/mod` (=0.6) → `# cite: (A,E)`.
  - `med/mod`, `sat/sev` (=0.7) → `# cite: (A,D)  # citation needed for this exact value`.
  - `med/sev` (=0.8) → `# cite: (A)  # citation needed`.
  - `mic/*` cells → `# cite: (G)` (alias of medical).
  - `nat/sev` (=0.6) → `# cite: (A,E)` (matches Salehi α=0.6 progression).
- **`tversky_gamma`**:
  - Cells = 1.0 → `# cite: (F)` (degenerate, reduces Focal-Tversky to Tversky).
  - Cells = 0.75 → `# cite: (D)`.
- **`boundary_weight`**:
  - `med/sev`, `mic/sev` (=0.2) → `# cite: (A,H)`.
  - All `=0.0` cells → no cite comment (it's the disable value; `boundary` family is not selected).

## 6. Locked-off rules and warn behavior

For each preset where a family choice is justified by domain knowledge, a fixed set of knobs is **locked off** — they have a fixed family value in every `class_imbalance` tier in the table, and if a user explicitly overrides them via `overrides`, the resolver emits a `logging.WARNING`. The override is **NOT stripped**: per issue #112's "presets are guidance, not law" principle (mirrored from #75), the user's value wins; the warn is the entire contract.

Module-level constant in `models/losses/presets.py`:

```python
LOCKED_OFF: dict[str, dict[str, str]] = {
    "medical": {
        "mask_family": (
            "the medical preset chose focal_dice/focal_tversky/boundary to handle "
            "rare positives; overriding to dice_bce or bce may underweight them"
        ),
    },
    "natural": {
        "mask_family": (
            "the natural preset chose dice_bce/focal_dice; overriding to "
            "focal_tversky or boundary is unusual for balanced natural-image data"
        ),
    },
    # satellite, microscopy: no locked-off entries in v1 (revisit after real users).
}
```

`preset: none` and `preset: custom` are **not** keyed in `LOCKED_OFF` — they bypass the check, by design (none = preserved pre-#112; custom = user is fully explicit).

Initial coverage is intentionally minimal — `mask_family` only, for two presets. Reviewers may add entries during PR review; the structure scales to any knob × preset pair. The planner does not invent new entries beyond what is listed here.

### 6.1 Warning message format

Exactly one `logging.WARNING` per locked-off override, per `resolve()` call. Format:

> `You overrode <knob>=<value> under preset=<preset>; <reason>. The override will be applied as-is.`

Examples:
- `You overrode mask_family=dice_bce under preset=medical; the medical preset chose focal_dice/focal_tversky/boundary to handle rare positives; overriding to dice_bce or bce may underweight them. The override will be applied as-is.`
- `You overrode mask_family=focal_tversky under preset=natural; the natural preset chose dice_bce/focal_dice; overriding to focal_tversky or boundary is unusual for balanced natural-image data. The override will be applied as-is.`

Logger: `logging.getLogger("custom_sam_peft.models.losses.presets")` (module-level via `_LOG = logging.getLogger(__name__)`).

Dedup is automatic: `resolve()` is called exactly once per `build_loss_bundle` call, which is called exactly once per trainer init (the bundle is built once and reused per step — see §8.4). No multi-emission across steps.

### 6.2 What counts as "enabled" for the warn check

`mask_family` is a string literal selector, so the warn-check trigger is *"the override changes the family to a different value"*, not the float-comparison `_is_enabled` used in #75. Concretely:

```python
def _override_triggers_warn(field: str, value: object, preset: str) -> bool:
    if preset not in LOCKED_OFF:
        return False
    if field not in LOCKED_OFF[preset]:
        return False
    # For the v1 LOCKED_OFF table, every entry is a Literal field (mask_family).
    # Any override that sets a non-None value triggers the warn. (We don't compare
    # against the table value; an override that happens to match the table value
    # is still an explicit choice by the user — but the warn message would be
    # misleading, so the resolver compares: if override == table[field], no warn.)
    return value is not None and value != _SEED_FOR_PRESET(preset, field)
```

`_SEED_FOR_PRESET(preset, field)` is the value the resolver would have used absent the override — i.e. `PRESET_TABLE[(preset, class_imbalance)][field]`. The comparison ensures that `overrides={"mask_family": "focal_dice"}` under `preset=medical, class_imbalance=balanced` (where the table already has `focal_dice`) emits no warn (the override is a no-op).

For non-locked-off fields (every field except `mask_family` in v1), no warn check runs.

## 7. Resolution algorithm and module layout

Pseudocode (the real implementation lives in `models/losses/presets.py::resolve`):

```python
@dataclass(frozen=True)
class ResolvedLosses:
    mask_family:     MaskFamily
    box_family:      BoxFamily
    obj_family:      ObjFamily
    presence_family: PresenceFamily
    w_mask:          float
    w_box:           float
    w_obj:           float
    w_presence:      float
    focal_gamma:     float
    focal_alpha:     float
    tversky_alpha:   float
    tversky_gamma:   float
    boundary_weight: float
    matcher_weights: MatcherWeights


def resolve(cfg: LossConfig) -> ResolvedLosses:
    # 1. Seed from the preset table (or short-circuit for none/custom).
    if cfg.preset == "none":
        base = dict(_LEGACY_DEFAULTS)
        seed_matcher = MatcherWeights()
    elif cfg.preset == "custom":
        base = dict(PRESET_TABLE[("natural", "balanced")])
        seed_matcher = MatcherWeights()
    else:
        base = dict(PRESET_TABLE[(cfg.preset, cfg.class_imbalance)])
        seed_matcher = MatcherWeights()

    # 2. Apply overrides; warn if a locked-off knob is overridden.
    ov = cfg.overrides.model_dump(exclude_unset=False)  # all keys present (None for unset)
    for field, override in ov.items():
        if override is None:
            continue
        if field == "matcher_weights":
            seed_matcher = MatcherWeights(**override) if isinstance(override, dict) else override
            continue
        # Warn (locked-off check). Skipped under preset in {none, custom}.
        if cfg.preset not in ("none", "custom") and _override_triggers_warn(
            field, override, cfg.preset, cfg.class_imbalance
        ):
            reason = LOCKED_OFF[cfg.preset][field]
            _LOG.warning(
                "You overrode %s=%s under preset=%s; %s. The override will be applied as-is.",
                field, override, cfg.preset, reason,
            )
        base[field] = override

    # 3. Build the immutable resolved view.
    return ResolvedLosses(**base, matcher_weights=seed_matcher)
```

`_LEGACY_DEFAULTS` is the 13-field dict matching the deleted dataclass's defaults (`mask_family=dice_bce, box_family=l1_giou, obj_family=focal_bce, presence_family=bce, w_mask=1.0, w_box=0.0, w_obj=1.0, w_presence=1.0, focal_gamma=2.0, focal_alpha=0.25, tversky_alpha=0.5, tversky_gamma=1.0, boundary_weight=0.0`). It is defined as a module-level constant in `presets.py`.

### 7.1 Module layout

The current single-file module becomes a package:

```
src/custom_sam_peft/models/
    losses/
        __init__.py        # re-exports total_loss shim (back-compat for train/loop.py)
        presets.py         # PRESET_TABLE, LOCKED_OFF, ResolvedLosses, resolve, dump_loss_bundle
        compose.py         # LossBundle, build_loss_bundle
        terms/
            __init__.py    # re-exports the 14 term classes
            mask.py        # BCE, Dice, DiceBCE, FocalBCE (mask-shape), FocalDice,
                           #   Tversky, FocalTversky, Boundary
            box.py         # L1GIoU, GIoUOnly, CIoU
            obj.py         # FocalBCE (obj-shape), BCE (obj-shape)
            presence.py    # BCE (presence-shape), FocalBCE (presence-shape)
```

`models/losses/__init__.py` content:

```python
"""SAM 3.1 training losses — domain-aware preset-driven loss bundle."""

from custom_sam_peft.models.losses.compose import LossBundle, build_loss_bundle
from custom_sam_peft.models.losses.presets import (
    PRESET_TABLE,
    LOCKED_OFF,
    ResolvedLosses,
    dump_loss_bundle,
    resolve,
)

# Back-compat shim — the trainer in train/loop.py still calls total_loss(outputs, targets, cfg.train.loss).
# This shim builds a fresh bundle per call (cheap) and delegates. A future PR may
# collapse the shim once train/loop.py is updated to hold a long-lived LossBundle
# (see §8.4); for now, the shim keeps the migration PR minimal.
def total_loss(outputs, targets, cfg):
    bundle = build_loss_bundle(resolve(cfg))
    return bundle.forward(outputs, targets)


__all__ = [
    "LossBundle", "build_loss_bundle",
    "PRESET_TABLE", "LOCKED_OFF", "ResolvedLosses",
    "dump_loss_bundle", "resolve",
    "total_loss",
]
```

This shim contract: the two call sites in `train/loop.py` (`total_loss(out, targets, cfg.train.loss)`) work unmodified. The shim builds a new bundle every call — that is a known minor inefficiency. The planner is expected to collapse the shim within the same PR by lifting `bundle = build_loss_bundle(resolve(cfg.train.loss))` into trainer init and passing the bundle into the per-step closures; if that ends up too invasive for one PR, the shim stays and a follow-up issue tracks the inefficiency.

### 7.2 Module placement and import boundary

- `models/losses/presets.py` owns `PRESET_TABLE`, `LOCKED_OFF`, `ResolvedLosses`, `resolve`, `dump_loss_bundle`. **Pure-Python** (no torch import) so the resolver can be imported standalone — matches `data/aug_presets.py`'s isolation in #75. Note: `MatcherWeights` is imported from `config._internal`, which itself does not import torch — so the pure-Python guarantee holds transitively.
- `models/losses/terms/*.py` import `torch` and define `nn.Module` subclasses.
- `models/losses/compose.py` imports `torch`, the term modules, and `presets.ResolvedLosses`. Owns `LossBundle` and `build_loss_bundle`.
- `models/losses/__init__.py` re-exports the public surface and the `total_loss` shim. Importing this package pulls torch (via `compose`) — same as today's `models/losses.py`, no boundary regression.

## 8. Term library (14 classes) and composer

The fourteen classes live under `models/losses/terms/` with per-axis uniform `forward` signatures so the composer can drive them generically.

### 8.1 Mask axis (8 classes; `forward(pred_logits, target)`)

`pred_logits` and `target` are `(N, H, W)`. If spatial shapes differ, `pred_logits` is bilinear-upsampled to the target resolution (matches today's `mask_loss`).

| class | formula |
|---|---|
| `BCELoss`            | `BCE_with_logits(p, t)` |
| `DiceLoss`           | `1 - 2·Σpt / (Σp + Σt + ε)` |
| `DiceBCELoss`        | `0.5·Dice(p, t) + 0.5·BCE(p, t)` *(today's `mask_loss`)* |
| `FocalBCELoss`       | `mean(α(1-p_t)^γ · BCE_per_pixel(p, t))` |
| `FocalDiceLoss`      | `0.5·Dice(p, t) + 0.5·FocalBCE(p, t)` |
| `TverskyLoss`        | `1 - TP / (TP + α·FN + (1-α)·FP + ε)` |
| `FocalTverskyLoss`   | `(1 - TI(p, t; α))^γ` |
| `BoundaryLoss`       | `boundary_weight · Kervadec(p, t) + (1 - boundary_weight) · Dice(p, t)` |

`ε = 1.0` (matches today's `_dice_loss`).

`BoundaryLoss` implementation note (spec, not code): the Kervadec term needs a signed distance transform of the target mask. Two viable backends:

- **scipy.ndimage.distance_transform_edt** — CPU per-image, no new dep (`scipy>=1.10` is already required in `pyproject.toml`). Slower but adds zero deps.
- **kornia.contrib.distance_transform** — GPU-friendly, requires `kornia` as a new dep.

The planner audits `pyproject.toml` (currently has `scipy>=1.10`, not `kornia`) and **uses scipy by default**. If the planner chooses to add `kornia` for GPU speedup, it must (a) gate the import behind a try/except with a scipy fallback so the test suite still passes without kornia, and (b) update the dep audit in the PR description. Default expectation: scipy-only, no new deps. Per-image SDT is computed eagerly inside `forward` from the detached `target` (no autograd through the transform — Kervadec's gradient comes from the `p * SDT` product where the SDT is a constant w.r.t. `p`).

### 8.2 Box axis (3 classes; `forward(pred_cxcywh, target_cxcywh)`)

| class | formula |
|---|---|
| `L1GIoULoss`         | `smooth_l1(p, t) + (1 - GIoU(p, t)).mean()` *(today's `box_loss`)* |
| `GIoUOnlyLoss`       | `(1 - GIoU(p, t)).mean()` |
| `CIoULoss`           | Zheng et al. 2020 (mean over batch) |

`GIoU` reuses today's `_giou_pairwise(b1, b2)` helper (move it from `models/losses.py` to `models/losses/terms/box.py`); `CIoU` is implemented from scratch using the same xyxy conversion utility.

### 8.3 Obj axis (2 classes; `forward(obj_logits, matched_mask)`)

`obj_logits` is `(B, Q)`; `matched_mask` is `(B, Q)` bool — True for queries assigned to some target. Identical input contract to today's `objectness_loss`.

| class | formula |
|---|---|
| `FocalBCELoss`       | sigmoid-focal BCE (mean over `B·Q`) *(today's `objectness_loss`)* |
| `BCELoss`            | plain `BCE_with_logits(obj, matched.float())` |

### 8.4 Presence axis (2 classes; `forward(img_presence, image_has_target)`)

`img_presence` is `(B,)`; `image_has_target` is `(B,)` bool. Identical input contract to today's `presence_loss`.

| class | formula |
|---|---|
| `BCELoss`            | plain `BCE_with_logits(p, t.float())` *(today's `presence_loss`)* |
| `FocalBCELoss`       | sigmoid-focal BCE (mean over `B`) |

### 8.5 Class-sharing scheme

The obj/presence axes share most of the math with the mask axis's BCE/FocalBCE — they just differ in input rank (mask=3D, obj=2D, presence=1D) and reduction axes. The spec leaves the sharing scheme to the planner: either (a) thin subclasses in `terms/obj.py` and `terms/presence.py` that route through the mask-axis primitives, or (b) factory functions that produce per-axis BCELoss/FocalBCELoss closures. Both work and have similar surface; the planner picks the cleanest. **Constraint:** the 14-class public-surface count is preserved in `terms/__init__.py` (each axis exposes a distinct class object so `term_classes` introspection in §10 returns the expected names).

Constructor signature: every term class takes `(focal_gamma=2.0, focal_alpha=0.25, tversky_alpha=0.5, tversky_gamma=1.0, boundary_weight=0.0)` keyword-only kwargs and ignores the irrelevant ones. This lets `build_loss_bundle` instantiate every term with the same hyperparameter pack:

```python
def build_loss_bundle(resolved: ResolvedLosses) -> LossBundle:
    hp = dict(
        focal_gamma=resolved.focal_gamma,
        focal_alpha=resolved.focal_alpha,
        tversky_alpha=resolved.tversky_alpha,
        tversky_gamma=resolved.tversky_gamma,
        boundary_weight=resolved.boundary_weight,
    )
    mask_term     = _MASK_TERMS[resolved.mask_family](**hp)
    box_term      = _BOX_TERMS[resolved.box_family](**hp)
    obj_term      = _OBJ_TERMS[resolved.obj_family](**hp)
    presence_term = _PRESENCE_TERMS[resolved.presence_family](**hp)
    weights = (resolved.w_mask, resolved.w_box, resolved.w_obj, resolved.w_presence)
    return LossBundle(mask_term, box_term, obj_term, presence_term,
                      weights=weights, matcher_weights=resolved.matcher_weights)
```

`_MASK_TERMS`, `_BOX_TERMS`, `_OBJ_TERMS`, `_PRESENCE_TERMS` are module-level `dict[Literal, type]` maps in `compose.py`, keyed by the family literals. Missing keys raise `KeyError` — never reachable because pydantic validates the literal at config-load time.

### 8.6 `LossBundle` and trainer integration

```python
class LossBundle:
    """Pre-instantiated four-term loss bundle. Built once per trainer init."""

    def __init__(self, mask_term, box_term, obj_term, presence_term,
                 *, weights, matcher_weights):
        self.mask_term     = mask_term
        self.box_term      = box_term
        self.obj_term      = obj_term
        self.presence_term = presence_term
        self.w_mask, self.w_box, self.w_obj, self.w_presence = weights
        self.matcher = HungarianMatcher(
            lambda_l1=matcher_weights.lambda_l1,
            lambda_giou=matcher_weights.lambda_giou,
            lambda_mask=matcher_weights.lambda_mask,
        )

    def forward(self, outputs, targets) -> dict[str, Tensor]:
        # Mirrors today's total_loss: matcher once, each term, weighted sum into "total".
        canonical = meta_to_canonical(outputs)
        indices = self.matcher(canonical, targets)
        pred_boxes_m, tgt_boxes_m, pred_masks_m, tgt_masks_m = (
            _gather_matched_boxes_masks(canonical, targets, indices)
        )
        matched_mask = _matched_query_mask(canonical, indices)
        has_target = _image_has_target(targets, canonical.img_presence.device)
        zero = canonical.obj_logits.new_zeros(())
        losses: dict[str, Tensor] = {
            "mask":     self.mask_term(pred_masks_m, tgt_masks_m)
                        if pred_masks_m.numel() > 0 else zero,
            "box":      self.box_term(pred_boxes_m, tgt_boxes_m)
                        if pred_boxes_m.numel() > 0 else zero,
            "obj":      self.obj_term(canonical.obj_logits, matched_mask),
            "presence": self.presence_term(canonical.img_presence, has_target),
        }
        losses["total"] = (
            self.w_mask     * losses["mask"]
            + self.w_box    * losses["box"]
            + self.w_obj    * losses["obj"]
            + self.w_presence * losses["presence"]
        )
        return losses
```

`_gather_matched_boxes_masks`, `_matched_query_mask`, `_image_has_target` are moved from today's `models/losses.py` into `models/losses/compose.py` unchanged.

**Trainer wire-up:** the long-run goal is `bundle = build_loss_bundle(resolve(cfg.train.loss))` constructed **once at trainer init** (`Trainer.__init__`) and stored as `self._loss_bundle`. The two call sites in `train/loop.py` then read `self._loss_bundle.forward(out, targets)` instead of `total_loss(out, targets, cfg.train.loss)`. If the planner cannot land both the package refactor and the trainer wire-up in one PR without ballooning scope, the fallback is to ship the package refactor with the `total_loss` shim in `__init__.py` and leave the trainer call sites untouched; a follow-up issue then collapses the shim. The spec recommends doing both in one PR (the trainer change is two lines per call site).

The shim is the contract that guarantees `preset: none` is numerically identical to today's `total_loss` output (see test §12.3).

## 9. Sidecar: `run_dir/loss_bundle.json`

`src/custom_sam_peft/train/trainer.py` already writes `run_dir/config.yaml` and `run_dir/augmentation_pipeline.json`. After the augmentation-pipeline write, it now also writes `run_dir/loss_bundle.json`:

```json
{
  "preset": "medical",
  "class_imbalance": "moderate",
  "resolved": {
    "mask_family": "focal_tversky",
    "box_family": "l1_giou",
    "obj_family": "focal_bce",
    "presence_family": "bce",
    "w_mask": 1.0,
    "w_box": 0.0,
    "w_obj": 1.0,
    "w_presence": 1.0,
    "focal_gamma": 2.5,
    "focal_alpha": 0.25,
    "tversky_alpha": 0.7,
    "tversky_gamma": 0.75,
    "boundary_weight": 0.0
  },
  "term_classes": {
    "mask":     "FocalTverskyLoss",
    "box":      "L1GIoULoss",
    "obj":      "FocalBCELoss",
    "presence": "BCELoss"
  },
  "library_version": "0.11.0"
}
```

The example above is the literal expected output for `preset: medical, class_imbalance: moderate` with no overrides — internally consistent with §5 (med/mod row: `mask_family=focal_tversky, focal_gamma=2.5, tversky_alpha=0.7, tversky_gamma=0.75, boundary_weight=0.0`) and with §8's term-class map (`focal_tversky → FocalTverskyLoss`, `l1_giou → L1GIoULoss`, `focal_bce` (obj-axis) → `FocalBCELoss`, `bce` (presence-axis) → `BCELoss`).

The `matcher_weights` sub-object is intentionally **not** included in `resolved` — it is an internal contract of the matcher, not a knob the user tunes via the preset table. Users who override `matcher_weights` via `overrides.matcher_weights` still get the override applied (the bundle uses it), but the sidecar omits it to keep `resolved` aligned with `LossOverrides`'s 13 user-tunable knobs. A future revision may add a top-level `matcher_weights` key alongside `resolved` if a use case emerges.

### 9.1 Helper

`dump_loss_bundle(cfg: LossConfig) -> dict` lives in `models/losses/presets.py`. It calls `resolve(cfg)`, instantiates `build_loss_bundle(resolved)` once to introspect the chosen term-class names (via `type(term).__name__`), and assembles the dict above. `library_version` is sourced from `custom_sam_peft.__version__` (already exported from `_version.py` via the package `__init__`). Falls back to `"unknown"` if the import fails (defense-in-depth — should never happen in a real install).

`dump_loss_bundle` builds a bundle just to introspect names — that does instantiate four `nn.Module` instances per call. The call is one-shot per training run (single trainer init), so the cost is negligible. An alternative is a parallel `_TERM_CLASS_NAMES` dict keyed by family literals; the planner picks the cleanest of the two. Spec-recommended: build-and-introspect (one source of truth — the class registry in `compose.py`).

### 9.2 Trainer wire-up

`trainer.py::_setup_run_dir` already does:

```python
(run_dir / "config.yaml").write_text(yaml.safe_dump(cfg.model_dump(mode="json")))
from custom_sam_peft.data.aug_presets import dump_augmentation_pipeline
(run_dir / "augmentation_pipeline.json").write_text(
    json.dumps(dump_augmentation_pipeline(cfg.data.augmentations), indent=2, sort_keys=False)
)
```

The new code appends, in the same function, immediately after the augmentation-pipeline write:

```python
from custom_sam_peft.models.losses import dump_loss_bundle
(run_dir / "loss_bundle.json").write_text(
    json.dumps(dump_loss_bundle(cfg.train.loss), indent=2, sort_keys=False)
)
```

Failure semantics: if `dump_loss_bundle` raises (e.g. transient `cfg` corruption), the run dir is left with `config.yaml` and `augmentation_pipeline.json` but no `loss_bundle.json` — partial but recoverable. This matches #75's "augmentation_pipeline.json fails → no partial run-dir below it" pattern (loss_bundle.json is the next-deepest artifact).

### 9.3 Cross-version reproducibility escape

Documented in the spec and as a docstring on `dump_loss_bundle`: the `resolved` dict is the surface that pins behavior across library versions. A user who needs strict reproducibility against a future preset-table change can copy `resolved` verbatim into `overrides:` under `preset: custom` and the resolver will return the same 13 values — even if the table for, say, `medical/moderate` shifts. Same-library-version reproducibility is automatic: `(preset, class_imbalance, overrides)` is a pure function of the resolved view (modulo `MatcherWeights`, which the user copies separately if they overrode it).

## 10. CLI changes

### 10.1 `csp init --class-imbalance`

`csp init` already has `--preset` and `--intensity` from #75. The new flag is `--class-imbalance`. The scope of `--preset` **expands** to drive both augmentation and loss — it is the same `Preset` literal, no duplicate type, no separate flag.

New Typer option on `cli/init_cmd.py::init`:

```python
class_imbalance: str = typer.Option(
    "balanced",
    "--class-imbalance",
    case_sensitive=False,
    help="Loss-bundle class-imbalance tier. One of: balanced, moderate, severe.",
),
```

Validation: `init_cmd` checks the value against the `ClassImbalance` literal tuple (imported from `config/schema.py`) and raises `typer.BadParameter` on mismatch — matching the existing `--intensity` validation pattern.

Template substitution: `run_init` extends the substitution dict with `class_imbalance=class_imbalance` and `loss_overrides_block=…`. Templates carry `${class_imbalance}` and `${loss_overrides_block}` placeholders inside the `train.loss:` block.

#### 10.1.1 `--preset custom` branch (loss side)

The existing `${overrides_block}` (#75) is for the *augmentation* overrides under `data.augmentations:`. The new `${loss_overrides_block}` is the parallel construct under `train.loss:`. Same shape, different field names:

```python
if preset == "custom":
    loss_overrides_block = (
        "overrides: {}  # fill in knobs: mask_family, box_family, obj_family, presence_family, "
        "w_mask, w_box, w_obj, w_presence, focal_gamma, focal_alpha, "
        "tversky_alpha, tversky_gamma, boundary_weight, matcher_weights"
    )
else:
    loss_overrides_block = (
        "# Override individual loss knobs here; unset keys inherit from (preset, class_imbalance).\n"
        "    # overrides:\n"
        "    #   mask_family: focal_dice\n"
        "    #   focal_gamma: 2.5\n"
        "    #   tversky_alpha: 0.7"
    )
```

The block is constructed with leading 4-space indentation on continuation lines so it slots under the `train.loss:` indent without re-indentation (same pattern as #75's `overrides_block`).

When `--preset custom` is passed on the CLI:
- The augmentation `${overrides_block}` gets the augmentation `overrides: {}` scaffold (existing #75 behavior).
- The loss `${loss_overrides_block}` gets the loss `overrides: {}` scaffold (new).

Both branches keyed on the same `preset == "custom"` check — one source of truth for the "is the user explicit?" decision.

### 10.2 `csp doctor --config` extension

`csp doctor --config X.yaml` currently renders two extra tables ("Resolved augmentations", "Normalization") via `_render_resolved_config_tables` and a `resolved_config: {augmentations, normalize}` JSON sub-block via `_build_resolved_config_json` (`cli/doctor_cmd.py` lines 94–139). The new code adds a third table and a third JSON sub-key.

#### 10.2.1 "Resolved losses" table

Rendered after the "Normalization" table in `_render_resolved_config_tables`, via `rich.table.Table(title="Resolved losses", show_header=False, box=None)`. Rows:

- `preset` → `cfg.train.loss.preset`
- `class_imbalance` → `cfg.train.loss.class_imbalance`
- 13 knob rows, one per `ResolvedLosses` field (in the §7 declaration order, excluding `matcher_weights` which is rendered separately):
  - `mask_family`, `box_family`, `obj_family`, `presence_family`,
  - `w_mask`, `w_box`, `w_obj`, `w_presence`,
  - `focal_gamma`, `focal_alpha`,
  - `tversky_alpha`, `tversky_gamma`,
  - `boundary_weight`
- `matcher_weights` → `lambda_l1=…, lambda_giou=…, lambda_mask=…` (single-row, joined string)
- `term_classes` → comma-joined Python class names of the 4 chosen terms (e.g. `mask=FocalTverskyLoss, box=L1GIoULoss, obj=FocalBCELoss, presence=BCELoss`)

Loading: `cfg = load_config(config_path)` — existing call site, no changes. Resolver: `from custom_sam_peft.models.losses import dump_loss_bundle; loss_dump = dump_loss_bundle(cfg.train.loss)`. The table renders directly off `loss_dump["resolved"]` and `loss_dump["term_classes"]`, so the table and the sidecar are guaranteed to agree (one source of truth).

#### 10.2.2 `--json` mode

When `--config` is unset, JSON output is byte-identical to today + #75's contract.

When `--config` is set, the `resolved_config` sub-dict gains a `loss` key parallel to `augmentations` and `normalize`:

```json
{
  "python_version": "...",
  "...": "existing fields",
  "resolved_config": {
    "augmentations": { "...": "from #75" },
    "normalize":     { "...": "from #75" },
    "loss": {
      "preset": "medical",
      "class_imbalance": "moderate",
      "resolved": { /* 13 knobs */ },
      "term_classes": { "mask": "...", "box": "...", "obj": "...", "presence": "..." }
    }
  }
}
```

Implementation: `_build_resolved_config_json(cfg)` appends `"loss": dump_loss_bundle(cfg.train.loss)` to the returned dict, after dropping `library_version` from that sub-dict (the top-level diagnostics report already carries version info; the per-run sidecar carries it because it's persisted to disk). The `loss` JSON sub-dict therefore has 4 keys: `preset`, `class_imbalance`, `resolved`, `term_classes`. The `DoctorReport` dataclass and `diagnostics.run_doctor` are **not** modified — config-derived data stays a pure `doctor_cmd.py` concern, same as #75.

### 10.3 Template updates

Both `cli/templates/coco_text_lora.yaml` and `cli/templates/coco_text_qlora.yaml`: the existing `train.loss:` block (lines ~79–86 of each) is **replaced** with:

```yaml
  loss:
    preset: ${preset}
    class_imbalance: ${class_imbalance}
    ${loss_overrides_block}
```

The `${preset}` placeholder is shared with #75's augmentation block (one CLI flag drives both); the `${class_imbalance}` and `${loss_overrides_block}` placeholders are new. The block sits at the same indentation as today's `loss:` block (2-space indent under `train:`, with `preset:` / `class_imbalance:` at 4-space indent).

`string.Template`-safe escaping: the audit confirms no literal `$` in either template (still true; #75 already established this). Forward-only protection only.

## 11. Test plan

CPU-only. Mirrors #75 §12 in shape, with naming and module placement adjusted for the loss-side counterpart. New tests use the existing `caplog` convention for log-message assertions and the existing `Typer` runner convention for CLI tests (matching `test_cli_init.py`, `test_cli_doctor.py`).

### 11.1 New: `tests/unit/test_loss_presets.py`

Resolver behavior, sidecar dict shape, locked-off warns.

- `test_resolve_table_exact_values` — parameterized over all 12 `(preset, class_imbalance)` pairs for the four real domains; for each, build `LossConfig(preset=p, class_imbalance=ci)`, call `resolve`, and assert each of the 13 resolved fields matches the §5 table.
- `test_resolve_none_uses_legacy_defaults` — for each `class_imbalance ∈ {balanced, moderate, severe}`, `preset="none"` resolves to the §5 legacy-defaults block; `class_imbalance` ignored.
- `test_resolve_custom_seeds_with_natural_balanced` — `preset="custom"` with empty overrides equals `(natural, balanced)`.
- `test_resolve_override_wins_over_table` — representative cell (`(natural, balanced)` + `overrides={"focal_gamma": 5.0}`): the override replaces `focal_gamma`; other 12 fields keep their table values.
- `test_resolve_override_zero_is_valid` — `overrides={"w_box": 0.0}` is honored (zero is a valid value, not "inherit").
- `test_resolve_override_matcher_weights` — `overrides={"matcher_weights": {"lambda_mask": 10.0}}` is honored; the resolved `matcher_weights.lambda_mask == 10.0` while the other two lambdas keep their defaults.
- `test_resolve_locked_off_warns_medical_mask_family` — `caplog`, level WARNING: `LossConfig(preset="medical", overrides={"mask_family": "dice_bce"})` emits exactly one warning whose message contains `mask_family`, `medical`, and the substring "rare positives".
- `test_resolve_locked_off_warns_natural_mask_family` — same shape: `LossConfig(preset="natural", overrides={"mask_family": "focal_tversky"})` warns containing `mask_family`, `natural`, "unusual".
- `test_resolve_locked_off_no_warn_when_override_matches_table` — `LossConfig(preset="medical", class_imbalance="moderate", overrides={"mask_family": "focal_tversky"})` (which equals the table value) emits **no** warn.
- `test_resolve_locked_off_no_warn_for_satellite` — satellite + any `mask_family` override: no warn (satellite has no `LOCKED_OFF` entry in v1).
- `test_resolve_none_skips_locked_off` — `preset="none"` with `overrides={"mask_family": "boundary"}` emits no warn; `resolved.mask_family == "boundary"`.
- `test_resolve_custom_skips_locked_off` — same with `preset="custom"`.
- `test_resolved_losses_frozen` — `dataclasses.FrozenInstanceError` on direct mutation; `dataclasses.replace(resolved, w_mask=2.0)` works.
- `test_dump_loss_bundle_shape` — `dump_loss_bundle(LossConfig(preset="medical", class_imbalance="moderate"))` returns the §9 dict structure: top-level keys `{"preset", "class_imbalance", "resolved", "term_classes", "library_version"}`; `resolved` has all 13 knob keys; `term_classes` has the 4 axis keys; `library_version` is a non-empty string.
- `test_dump_loss_bundle_for_none` — `LossConfig(preset="none")` → `term_classes == {"mask": "DiceBCELoss", "box": "L1GIoULoss", "obj": "FocalBCELoss", "presence": "BCELoss"}` (the legacy choices).
- `test_dump_loss_bundle_matches_resolve_exactly` — the `resolved` sub-dict equals `dataclasses.asdict(resolve(cfg))` minus `matcher_weights` (the sidecar omits it; §9).
- `test_microscopy_aliases_medical` — for each `class_imbalance`, `resolve(LossConfig(preset="microscopy", class_imbalance=ci))` equals `resolve(LossConfig(preset="medical", class_imbalance=ci))` (byte-equal `ResolvedLosses`).

### 11.2 New: `tests/unit/test_loss_terms.py`

Per term class (14 total): forward on a small synthetic batch — finite output, gradient flows, correct shape. Plus degenerate-case identities.

**Smoke tests (one per class — 14 tests):**
- For each term class, instantiate with default hyperparameters, build a tiny input tensor of the right shape (mask: `(2, 16, 16)`; box: `(4, 4)`; obj: `(2, 8)`; presence: `(2,)`), call `.forward(pred, target)`, assert the result is a 0-dim tensor, finite, and `grad_fn is not None` (gradient flows from inputs).

**Degenerate-case identities:**
- `test_tversky_alpha_half_equals_dice` — `TverskyLoss(tversky_alpha=0.5)(p, t) ≈ DiceLoss()(p, t)` on the same input (atol=1e-6).
- `test_focal_tversky_alpha_half_gamma_one_equals_dice` — `FocalTverskyLoss(tversky_alpha=0.5, tversky_gamma=1.0)(p, t) ≈ DiceLoss()(p, t)` (atol=1e-6).
- `test_focal_bce_gamma_zero_alpha_half_equals_bce` — `FocalBCELoss(focal_gamma=0.0, focal_alpha=0.5)(p, t) ≈ BCELoss()(p, t)` (atol=1e-6).
- `test_dice_bce_equals_half_dice_plus_half_bce` — pin the explicit weighting.

**BoundaryLoss:**
- `test_boundary_loss_identity_when_pred_equals_target` — perfect prediction (pred = target sigmoid-inverse): the loss is bounded below by `(1 - boundary_weight) · Dice(perfect) = 0`; the Kervadec term is finite. Concrete assertion: `BoundaryLoss(boundary_weight=0.5)(logits_for_target, target).item() < 1e-3`.
- `test_boundary_loss_finite_on_extreme_imbalance` — single-positive-pixel target on a `(1, 64, 64)` mask; loss is finite and `> 0` for a random pred.
- `test_boundary_loss_cpu_runs_under_scipy` — the SDT path runs without crashing on CPU under the scipy backend (the only backend in v1 unless the planner opts in to kornia — see §8.1).
- `test_boundary_loss_zero_weight_equals_dice` — `BoundaryLoss(boundary_weight=0.0)(p, t) ≈ DiceLoss()(p, t)` (atol=1e-6).
- `test_boundary_loss_full_weight_is_pure_kervadec` — `BoundaryLoss(boundary_weight=1.0)(p, t)` does not invoke the Dice term (numerically: equal to a direct call into the Kervadec helper with the same inputs).

**Box terms on edge cases:**
- `test_giou_only_loss_disjoint_boxes` — `GIoUOnlyLoss()(pred, target)` is finite and `>= 1.0` when boxes don't overlap.
- `test_giou_only_loss_overlapping_boxes` — finite and `< 1.0` when boxes overlap.
- `test_ciou_loss_disjoint_boxes` — finite, `> 0`.
- `test_ciou_loss_perfect_overlap` — `≈ 0` when pred == target.

**Mask upsampling:**
- `test_mask_terms_auto_upsample_when_shapes_differ` — `pred` shape `(N, 8, 8)`, `target` shape `(N, 16, 16)`: all 7 mask terms run without raising; outputs are finite.

### 11.3 New: `tests/unit/test_loss_compose.py`

Composer wiring, `LossBundle.forward` end-to-end, equivalence under `preset: none`.

- `test_build_loss_bundle_picks_correct_term_classes` — parameterized over a representative subset (`(natural, balanced)`, `(medical, moderate)`, `(medical, severe)`, `(satellite, severe)`); for each, `bundle = build_loss_bundle(resolve(cfg))`; assert `type(bundle.mask_term).__name__` matches the expected name (e.g. `(medical, severe)` → `"BoundaryLoss"`).
- `test_total_loss_equivalence_for_none_preset` — `preset="none"`: build a tiny canonical `outputs` + `targets` via the `tests/fixtures/tiny_sam3_stub.py` helpers; compare `LossBundle.forward(outputs, targets)["total"]` against today's pre-#112 `total_loss(outputs, targets, legacy_dataclass_with_defaults)`. The test imports the deleted dataclass's defaults from a pinned-constants block in the test file (snapshot of the legacy values) so the equivalence check survives the dataclass deletion. Tolerance: `torch.allclose(actual, expected, atol=1e-6)`.
- `test_loss_bundle_built_once` — assert the trainer constructor wires the bundle once. Implementation: monkey-patch `custom_sam_peft.models.losses.build_loss_bundle` with a call-counter spy; build a `Trainer(cfg)` and run two training steps via the existing tiny-stub flow; assert the spy was called exactly once. If the trainer wire-up is deferred to a follow-up (per §8.6 fallback), this test is reframed as `test_total_loss_shim_does_not_warn_per_step` — assert `caplog` contains at most one preset-resolver warning over many steps (warns are deterministic per call but they don't multiply).
- `test_loss_bundle_forward_matches_term_outputs` — build a bundle for `(natural, balanced)`, run `forward` on tiny inputs, and re-run each term independently; assert the `"mask"`, `"box"`, `"obj"`, `"presence"` outputs in the returned dict equal the per-term direct calls (sanity check for the composer math).

### 11.4 Extend: `tests/unit/test_config_schema.py`

- `test_loss_config_defaults` — `LossConfig()` has `preset="natural"`, `class_imbalance="balanced"`, `overrides == LossOverrides()`.
- `test_loss_config_preset_literal_validation` — `LossConfig(preset="mediacl")` raises `pydantic.ValidationError`.
- `test_loss_config_class_imbalance_literal_validation` — `LossConfig(class_imbalance="extreme")` raises (the literal is `balanced|moderate|severe`, not `balanced|imbalanced|extreme`; the issue body draft used a different vocab — assertion guards against drift).
- `test_loss_overrides_rejects_unknown_keys` — `LossOverrides(focal_gma=2.5)` raises (typo).
- `test_loss_overrides_w_box_zero_allowed` — `LossOverrides(w_box=0.0)` validates (no `gt=0` constraint on `w_box`).
- `test_loss_overrides_focal_alpha_out_of_range` — `LossOverrides(focal_alpha=1.5)` raises (constrained to `[0, 1]`).
- `test_loss_overrides_tversky_alpha_out_of_range` — `LossOverrides(tversky_alpha=-0.1)` raises.
- `test_loss_overrides_boundary_weight_out_of_range` — `LossOverrides(boundary_weight=1.5)` raises (constrained to `[0, 1]`).
- `test_loss_overrides_default_factory` — two `LossConfig()` instances do not share the same `overrides` object.
- `test_loss_overrides_all_none_by_default` — `LossConfig().overrides.model_dump()` has every field = `None`.
- `test_loss_overrides_matcher_weights_accepts_dict` — `LossOverrides(matcher_weights={"lambda_mask": 7.0})` validates; the resulting `matcher_weights` is a `MatcherWeights` instance with `lambda_mask=7.0`.

### 11.5 Rewrite: `tests/unit/test_loss_config.py`

The existing file tests the deleted dataclass. Its surviving useful assertions become:

- `test_loss_config_appears_in_train_hyperparams` — `TrainConfig(…).train.loss` is a `LossConfig` (the new Pydantic model).
- `test_matcher_weights_defaults_still_correct` — `MatcherWeights()` defaults unchanged.
- `test_matcher_weights_no_lambda_cls` — `assert not hasattr(MatcherWeights(), "lambda_cls")`.
- `test_no_w_cls_on_loss_overrides` — `assert "w_cls" not in LossOverrides.model_fields`.

The pre-#112 tests that asserted the dataclass's defaults (`test_loss_config_defaults`, `test_loss_config_rejects_extra_fields`) are **deleted** (the dataclass is gone; the new pydantic equivalents live in `test_config_schema.py` per §11.4).

### 11.6 Extend: `tests/unit/test_data_coco.py`, `tests/unit/test_data_hf.py`

These currently exercise YAML/dict shapes for `data.augmentations:` (post-#75). They do **not** currently exercise `train.loss:` — but any fixture YAML that includes a `train:` block with a flat `loss:` shape (e.g. `loss: {w_mask: 1.0}`) breaks. Audit:

- `test_data_coco.py` and `test_data_hf.py` build configs via `cfg.model_validate(...)` from dict fixtures. The fixtures either omit `train` entirely (relying on defaults) or set a minimal `train: {epochs: 1}`. If any fixture sets a flat `loss:` block, migrate it to `loss: {preset: "none"}` (numerically equivalent).
- Add no new tests; this is fixture-shape maintenance only.

### 11.7 Extend: `tests/unit/test_trainer_nan_behavior.py`

This file monkey-patches `total_loss` (the import path `custom_sam_peft.train.loop.total_loss`). The shim path stays unchanged: `total_loss` is re-exported from `models/losses/__init__.py` and re-imported by `train/loop.py`. The existing monkeypatch contract works without modification — but the test should be re-run after the migration to confirm.

- No code change to the test required; the assertions on `_LOG.warning(...)` continue to fire against the pre-existing trainer NaN handling.
- Add a one-line comment in the test docstring noting "monkey-patches the shim from `models/losses/__init__.py`".

### 11.8 Extend: `tests/unit/test_trainer_run_dir.py`

The current `test_trainer_run_dir.py` already validates `augmentation_pipeline.json`. Add the loss-sidecar equivalent:

- `test_run_dir_writes_loss_bundle_json` — after `Trainer(...)` constructs `run_dir`, assert `run_dir/loss_bundle.json` exists, parses as JSON, has top-level keys `{"preset", "class_imbalance", "resolved", "term_classes", "library_version"}`, `resolved` has all 13 knob keys, `term_classes` has the 4 axis keys.
- `test_run_dir_loss_bundle_matches_config` — with `cfg.train.loss = LossConfig(preset="medical", class_imbalance="moderate")`, assert the persisted JSON's `preset == "medical"`, `class_imbalance == "moderate"`, `resolved["mask_family"] == "focal_tversky"`.

### 11.9 Extend: `tests/integration/test_train_resume.py`, `tests/integration/test_train_end_to_end.py`

- Migrate any `LossConfig(...)` constructor with positional/flat kwargs to the new triple shape. Tests that wanted "default loss behavior" use `LossConfig(preset="none")`.
- In `test_train_end_to_end.py`, after the run finishes, assert `run_dir/loss_bundle.json` exists with the expected `preset` and a non-empty `library_version`.

### 11.10 Extend: `tests/unit/test_cli_init.py`

- `test_init_renders_class_imbalance` — invoke `csp init --preset medical --class-imbalance moderate --output tmp.yaml`; `load_config(tmp.yaml)` succeeds; `cfg.train.loss.preset == "medical"` and `cfg.train.loss.class_imbalance == "moderate"`.
- Parameterized version over `4 real presets × 3 class_imbalances = 12 combinations` plus `none × balanced` and `custom × balanced` → 14 cases (mirrors #75's 14-case parameterization; `class_imbalance` is ignored for `none`/`custom` so one representative is enough).
- `test_init_custom_writes_uncommented_loss_overrides_scaffold` — `csp init --preset custom --output tmp.yaml`; the rendered file contains `loss:\n    preset: custom\n    class_imbalance: balanced\n    overrides: {}` (uncommented scaffold) and `load_config(tmp.yaml)` succeeds.
- `test_init_invalid_class_imbalance_rejected` — `csp init --class-imbalance huge` exits non-zero; stderr mentions "class_imbalance" or "class-imbalance".
- `test_init_other_fields_parse_identically` — render with defaults; load config; assert non-loss fields (`run.name`, `model.name`, `train.epochs`) equal the values in today's templates — guards against template-substitution accidentally corrupting unrelated YAML, mirroring #75.

### 11.11 Extend: `tests/unit/test_cli_doctor.py` (and/or `test_cli_doctor_config.py`)

- `test_doctor_with_config_renders_resolved_losses` — `csp doctor --config <good.yaml>`; stdout contains the literal title string `"Resolved losses"`; contains a row labeled `preset`; the value of that row matches the config.
- `test_doctor_with_config_renders_loss_term_classes` — same; stdout contains a row labeled `term_classes` and the comma-joined names match the resolved bundle (e.g. for `medical/moderate` → `mask=FocalTverskyLoss`).
- `test_doctor_json_no_config_no_loss_block` — `csp doctor --json`; parsed JSON has no `"resolved_config"` key (existing #75 contract preserved).
- `test_doctor_json_with_config_has_loss_block` — `csp doctor --config <good.yaml> --json`; parsed JSON has `"resolved_config"` with sub-keys `"augmentations"`, `"normalize"`, **and** `"loss"`; the `loss` sub-dict has keys `{preset, class_imbalance, resolved, term_classes}` and `resolved` has 13 keys.
- `test_doctor_no_config_byte_identical` — existing #75 test continues to pass; no snapshot update required because the new code is gated on `--config`.

## 12. Migration (clean break, pre-1.0 per #70)

No back-compat alias. Single PR migrates everything:

1. **Delete** the dataclass `LossConfig` from `config/_internal.py` (lines ~33–57). Remove the re-export from `config/schema.py` (`__all__` entry "LossConfig" and the `from config._internal import …` line stay only for `ExportConfig`, `MatcherWeights`, `WandbConfig` — `LossConfig` is removed from both).
2. **Add** the new Pydantic `LossConfig`, `LossOverrides`, `ClassImbalance`, `MaskFamily`, `BoxFamily`, `ObjFamily`, `PresenceFamily` literals to `schema.py` (per §4).
3. **Add** `models/losses/presets.py` (`PRESET_TABLE`, `LOCKED_OFF`, `ResolvedLosses`, `_LEGACY_DEFAULTS`, `_SEED_FOR_PRESET`, `_override_triggers_warn`, `resolve`, `dump_loss_bundle`).
4. **Refactor** `models/losses.py` → `models/losses/` package: `__init__.py` (shim + re-exports per §7.1), `compose.py` (`LossBundle`, `build_loss_bundle`, the three private gather helpers moved verbatim from today's `losses.py`), `terms/` subdir with `mask.py`, `box.py`, `obj.py`, `presence.py` per §8.
5. **Update** `train/trainer.py::_setup_run_dir` to write `loss_bundle.json` (per §9.2).
6. **Update** `cli/init_cmd.py` to accept `--class-imbalance`, validate it, and substitute `${class_imbalance}` + `${loss_overrides_block}` into the template (per §10.1).
7. **Update** `cli/doctor_cmd.py` to render the "Resolved losses" table and the `loss` JSON sub-block (per §10.2).
8. **Update** both starter templates to use the new `train.loss:` block (per §10.3).
9. **Migrate** tests per §11.4–§11.11.

Tests that previously did `LossConfig(w_mask=2.0)` (positional dataclass kwargs) migrate to `LossConfig(preset="custom", overrides=LossOverrides(w_mask=2.0))` or `LossConfig(preset="none")` if they just want defaults.

A user with a pre-existing config carrying `train.loss: {w_mask: 1.0, w_obj: 1.0, matcher_weights: {lambda_mask: 5.0}}` will fail `load_config` with a pydantic `ValidationError` ("extra fields not permitted: w_mask, w_obj, matcher_weights"). The PR description includes the one-line migration recipe from §4.2.

#70 (v1.0 criteria) is the umbrella issue tracking acceptable pre-1.0 schema breaks. This PR's `Related issues` section in the PR body adds a one-line note that this break is gated under #70, mirroring #75.

## 13. In / out of scope

### 13.1 In scope (v1)

- New `LossConfig` + `LossOverrides` + family literals in `schema.py` (clean break, no aliases).
- `models/losses/presets.py` module with `PRESET_TABLE`, `LOCKED_OFF`, `ResolvedLosses`, `_LEGACY_DEFAULTS`, `resolve`, `dump_loss_bundle`.
- 14 term classes under `models/losses/terms/` per §8.
- `models/losses/compose.py` with `LossBundle` and `build_loss_bundle`.
- `models/losses/__init__.py` with the `total_loss` shim re-export.
- `csp init --class-imbalance` with template substitution; `${loss_overrides_block}` placeholder; `--preset custom` writes the uncommented loss-overrides scaffold.
- `csp doctor --config` with the third table ("Resolved losses") and the `loss` sub-block in `--json`.
- `trainer.py` writes `run_dir/loss_bundle.json` next to `augmentation_pipeline.json`.
- Both starter templates updated.
- Tests as enumerated in §11.

### 13.2 Out of scope (file as follow-up issues only if explicitly requested)

- User-supplied loss callable (`train.loss.custom_callable: "pkg.mod:fn"`) — v1.1.
- User-defined presets (extending `PRESET_TABLE` from outside the package) — v1.1.
- 3D / volumetric loss terms (surface Dice, 3D Tversky) — #110 territory.
- A `box_family: none` literal (achieved via `w_box=0.0` in v1).
- Per-knob application-probability overrides (presets fix term selection; weights tune contribution).
- Boundary-loss schedule (Kervadec ramps from 0.01 to ~1.0 across training); v1 uses a constant `boundary_weight`.
- Citation-pass across the whole codebase — separately tracked by issue #120; this spec emits `# citation needed` comments at the cells that need them.
- kornia dependency for GPU-friendly distance transform — scipy fallback is the v1 default; planner may opt in only if it audits `pyproject.toml` and gates the import.

## 14. Edge cases

- **`preset="none"`** → 13 resolved knobs equal the §5 legacy-defaults block (matches the deleted dataclass's defaults); `class_imbalance` ignored; no warns even if locked-off knobs are overridden.
- **`preset="custom"`** → seed = `PRESET_TABLE[("natural", "balanced")]`; overrides apply; `class_imbalance` ignored; no warns.
- **Microscopy = medical (alias)** → byte-equal `ResolvedLosses` between `(microscopy, x)` and `(medical, x)` for every `x`. No special-case in the resolver; the table just carries duplicate rows.
- **Override sets a locked-off knob to the same value the table already has** → no warn (the override is a no-op; the resolver compares `value != _SEED_FOR_PRESET(...)` per §6.2).
- **Override sets a non-locked-off knob (e.g. `focal_gamma`) to any value** → no warn (only `mask_family` is locked off in v1).
- **WARN dedup** → automatic; `resolve()` is called once per trainer init (via `build_loss_bundle`).
- **`extra="forbid"` on `LossOverrides`** → catches typos like `focal_gma: 2.5` at `load_config` time with a clear error message naming the bad key.
- **`extra="forbid"` on `LossConfig`** → catches typos like `presset: medical` at `load_config` time.
- **`w_box=0` (default)** → the box term is *still* constructed and *still* runs in `LossBundle.forward`, but its contribution to `"total"` is `0.0 · box_loss = 0`. The bookkeeping `"box"` value in the returned dict is the un-weighted box loss — non-zero on a non-empty matched batch. Users reading `losses["box"]` see the un-weighted value; users reading `losses["total"]` see the weighted sum. This matches today's behavior.
- **`w_box > 0` under any preset** → no warn; this is a recommended way for advanced users to enable box supervision when their dataset has reliable boxes.
- **`mask_family: boundary` under preset where `boundary_weight: 0.0`** → the `BoundaryLoss` term degenerates to plain `Dice` (boundary blend = 0). The bundle still works; the user effectively has Dice. Documented behavior.
- **`mask_family: focal_tversky` with `tversky_gamma: 1.0`** → degenerates to plain Tversky (no focal exponent). Documented behavior; covered by the §11.2 identity tests.
- **All-zero weights (`w_mask=w_box=w_obj=w_presence=0`)** → `losses["total"] == 0`. The training step still runs (autograd has no gradient — backward is a no-op or warns). The schema disallows this for `w_mask`, `w_obj`, `w_presence` (they are `PositiveFloat`), so the only way to reach all-zero is to override `w_box=0` (already default) and override the other three to zero, which fails at schema validation. Documented as "schema-protected; cannot happen via valid config".
- **`scipy` missing at runtime** → `BoundaryLoss` raises `ImportError` on instantiation with a message pointing to `pyproject.toml`'s `scipy>=1.10` requirement. The error is loud and early — bundle construction fails at trainer init, not deep in the first training step. The planner may either let scipy's import fail naturally or wrap with a `_require_scipy()` helper that adds the actionable error message; spec prefers the wrapper.
- **`MatcherWeights` override with extra fields** → `LossOverrides(matcher_weights={"lambda_mask": 5.0, "lambda_cls": 2.0})`: the `lambda_cls` field is rejected by `MatcherWeights`'s dataclass constructor (`TypeError`) per the existing contract pinned in `test_matcher_weights_rejects_extra_fields`. Surfaces at validation time.
- **`library_version` derivation** → reads `custom_sam_peft.__version__` (already wired via `_version.py`); falls back to `"unknown"` if the import fails.
- **`term_classes` introspection** → driven by `type(term).__name__` at bundle-construction time. If a developer accidentally renames a term class, the sidecar's `term_classes` value changes and the §11.1 `test_dump_loss_bundle_for_none` test catches the drift.
- **Order of warns** → multiple overridden locked-off knobs (currently impossible — only `mask_family` is locked off in v1, and a knob can be overridden at most once per `LossOverrides` instance) → each emits its own warn in `LossOverrides`-declaration order. Order is deterministic given the same overrides dict.
- **Concurrent overrides on `mask_family` and a focal/tversky param** → e.g. `LossConfig(preset="medical", overrides={"mask_family": "dice_bce", "focal_gamma": 5.0})`: one warn (for `mask_family`); the `focal_gamma` override applies silently. Correct: only the locked-off knob trips a warn.

## 15. Deliverables-to-issue mapping

Mirrors issue #112's "Deliverables" section.

| Issue item | Where in this spec | Implementation file(s) |
|---|---|---|
| 1. Spec | This document | `docs/superpowers/specs/2026-05-23-domain-aware-loss-presets-design.md` |
| 2. New `LossConfig` in `config/schema.py` | §4 | `src/custom_sam_peft/config/schema.py` + `src/custom_sam_peft/config/_internal.py` (delete the dataclass) |
| 3. Refactor `models/losses.py` | §7 + §8 | `src/custom_sam_peft/models/losses/__init__.py`, `presets.py`, `compose.py`, `terms/*.py` |
| 4. New loss families | §8 | `src/custom_sam_peft/models/losses/terms/mask.py` (focal_dice, focal_tversky, boundary added; dice_bce preserved as the natural default) |
| 5. Update `cli/init_cmd.py` | §10.1 | `src/custom_sam_peft/cli/init_cmd.py` + `src/custom_sam_peft/cli/templates/*.yaml` |
| 6. Update `cli/doctor_cmd.py` + run metadata | §10.2 + §9 | `src/custom_sam_peft/cli/doctor_cmd.py` + `src/custom_sam_peft/train/trainer.py` |
| 7. Update example YAMLs | §10.3 | `src/custom_sam_peft/cli/templates/coco_text_lora.yaml`, `src/custom_sam_peft/cli/templates/coco_text_qlora.yaml` |
| 8. Tests | §11 | `tests/unit/test_loss_presets.py` (new), `tests/unit/test_loss_terms.py` (new), `tests/unit/test_loss_compose.py` (new), and the existing files enumerated in §11.4–§11.11 |

## 16. Related issues

- **#75** — domain-aware augmentation presets (closed). This spec mirrors #75's shape and reuses its `Preset` literal verbatim. The only intentional shape divergence is `class_imbalance` instead of `intensity` (called out in issue #112's body).
- **#70** — v1.0 criteria; this PR's schema break is gated under #70's "breaking changes acceptable before v1.0" allowance. The PR description references #70 so the umbrella's checklist captures this change alongside #75's break.
- **#69** — normalization fallback (closed by the 2026-05-21 audit). Orthogonal to this PR; no normalization code is touched.
- **#110** — 3D imagery: volumetric loss terms would extend this issue's family vocabulary once 3D lands. Out of scope here.
- **#111** — grayscale support: orthogonal; doesn't change loss families directly, but the focal/Tversky defaults for single-channel medical/IR data may want a future revisit.
- **#120** — whole-codebase literature-cite audit (opened from this brainstorming session). Receives the `# citation needed` cells emitted by this spec as in-scope work.
- **#60** — sam3 gradient-checkpointing recompute mismatch. Unrelated; mentioned only because the template comments preserved verbatim by §10.3 reference it.
