# Interactive setup wizard + full gradient-checkpointing removal

**Issues:** [#149 — Interactive CLI setup wizard](https://github.com/NguyenJus/custom-sam-peft/issues/149) (primary). Touches [#148](https://github.com/NguyenJus/custom-sam-peft/issues/148) only as a forward-compat constraint (do not implement). Closes the gradient-checkpointing knob abandoned in [#60](https://github.com/NguyenJus/custom-sam-peft/issues/60) / [#89](https://github.com/NguyenJus/custom-sam-peft/issues/89) / [#127](https://github.com/NguyenJus/custom-sam-peft/issues/127).
**Release:** pre-1.0 minor bump (new feature + a breaking schema change → MINOR).
**Status:** locked design, single PR, no back-compat shims.

This PR ships two bundled workstreams:

1. **Interactive setup wizard** — extend `csp init` (`custom-sam-peft init`) with an `--interactive` / `-i` flag that walks a user through a tight set of prompts (the interactive session stays minimal) and emits a **comprehensive**, schema-validated `config.yaml`, then hands off to the existing weight-download flow. The interactive session is a declarative, extensible step registry — adding/reordering/removing a prompt is a one-line list edit, never surgery. The output is produced by rendering one unified comprehensive template (`config_full.yaml`), so the emitted file shows every section, the chosen branches active with alternatives commented, and advanced knobs as commented scaffolds — a config meant to be tweaked.
2. **Fully remove configurable gradient checkpointing (GC)** — delete the user knob, the runtime lever, and the presets search dimension. GC is abandoned and already a no-op on the current SAM 3.1 revision; main ships it off. This is a clean breaking change: no migration, no deprecation shim.

The two workstreams are bundled because GC removal simplifies the wizard's VRAM-auto-size step (`config_patch` no longer carries a `gradient_checkpointing` key, so the wizard needs no stripping logic), and both touch `presets.py` / templates / schema, so a single PR avoids two rounds of conflict resolution. The template consolidation (one unified `config_full.yaml`, deleting the two legacy templates) belongs to **Workstream 1**, not Workstream 2 — the unified template simply ships without a `gradient_checkpointing` line (see §1 / §8.7).

---

## §1 Scope & non-goals

### In scope

| File | Workstream | Change |
|------|-----------|--------|
| `src/custom_sam_peft/cli/setup_wizard.py` | 1 | **New.** `WizardStep`, `Ctx`, prompt primitives, `STEPS` registry, driver, **render stage** (answers dict → template placeholders, including the branching dataset-format / validation blocks), validate-via-`load_config`, emit, `infer_class_imbalance`. |
| `src/custom_sam_peft/cli/templates/config_full.yaml` | 1 | **New (unified).** One comprehensive `string.Template` that is a superset of the two legacy templates: all top-level sections, commonly-tuned fields echoed, advanced override scaffolds commented, branching blocks (dataset-format, validation) with the chosen branch active and the others commented, `$peft_method` + `qlora:` sub-block, no `gradient_checkpointing` line, `prompt_mode: text` hardcoded. Rendered by BOTH `csp init` and the wizard. |
| `src/custom_sam_peft/cli/templates/coco_text_lora.yaml`, `coco_text_qlora.yaml` | 1 | **DELETED.** Superseded by `config_full.yaml` (consolidation is Workstream 1's template work, not a GC-only edit). |
| `src/custom_sam_peft/cli/init_cmd.py` | 1 | Add `--interactive`/`-i` flag; branch to wizard; reuse `_maybe_download_weights`. **Update `run_init`** to render the unified `config_full.yaml`: grow `TEMPLATES` mapping + the `string.Template` substitution set to cover the new placeholders, mapping the two `--template` values onto `$peft_method` (`coco-text-lora`→`lora`, `coco-text-qlora`→`qlora`) with flag-driven defaults for placeholders the flags do not supply. |
| `src/custom_sam_peft/config/schema.py` | 2 | Remove `ModelConfig.gradient_checkpointing`. |
| `src/custom_sam_peft/presets.py` | 2 | Remove `gradient_checkpointing` from `PresetDecision`/`config_patch`; collapse search space to ckpt-off; delete `CKPT_FACTOR`; drop `ckpt` param from `_activation_bytes`/`_predicted_bytes`; fix `label()`, `_sort_key`, `_candidates`, "nothing fits" message, `to_json`/`from_json`, `decide_eval_batch_size`. |
| `src/custom_sam_peft/models/sam3.py` | 2 | Remove the `if cfg.gradient_checkpointing:` no-op block. |
| `src/custom_sam_peft/train/loop.py` | 2 | Remove the GC rung from the OOM ladder and the `gradient_checkpointing` field from `OomState`; update the final raise message. |
| `src/custom_sam_peft/train/types.py` | 2 | `OomEvent.action` → `Literal["microbatch_halved"]`; drop `new_gradient_checkpointing`; update docstring. |
| `src/custom_sam_peft/runs/bundle.py` | 2 | Stop rendering ckpt state in the `## Preset` block. |
| `docs/config-schema.md` | 2 | Remove the `model.gradient_checkpointing` row. |
| Tests (see §10) | both | New wizard tests; flag-driven init tests retargeted to the unified template; GC-removal test adjustments. |

### Out of scope

- **No new `setup` subcommand and no top-level `--setup` flag.** The wizard is an interactive mode of `init` (issue #149's recommended resolution of the flag-vs-subcommand question). The codebase uses subcommands, and `init` already owns config generation.
- **No exhaustive prompt coverage.** The wizard prompts the minimal set needed for a working run. Advanced knobs (learning rate, lr schedule, eval thresholds, tracking backend, num_workers, normalize, box_hint, text_prompt mode, multiplex, channels, etc.) stay at schema defaults and are left to YAML editing. This is a deliberate "tight prompt set" decision.
- **No #148 implementation.** The wizard depends only on the stable `decide_preset(image_size)` facade; #148's later internal rework (lookup table vs. live probe) does not change that signature or `PresetDecision.config_patch`. The wizard MUST NOT call `calibrate` or any probe path (no OOM risk). See §6.1.
- **No GC migration / deprecation path.** Pre-1.0; users update their YAML. See §11.
- **No change to the flag-driven `init` CLI surface.** `csp init` without `--interactive` keeps the same flags (both `--template` values still accepted), but its OUTPUT changes: it now renders the unified `config_full.yaml` (comprehensive), so its emitted file is no longer byte-for-byte identical to the old per-template output. This template consolidation is in scope (§7, §8.7).

---

## §2 Architectural approach (Workstream 1)

The wizard and the flag-driven path render the SAME unified `string.Template` file (`config_full.yaml`). The wizard is a schema-first generation pipeline that collects answers interactively, maps them onto the template's placeholders, renders, validates the rendered output, then writes:

```
csp init --interactive
        │
        ▼
  init_cmd.init():
    pre-flight (BEFORE any prompt):
      - require TTY (sys.stdin.isatty()) ............... else BadParameter
      - --output exists & not --force? ................. else BadParameter (same as today)
        │
        ▼
  setup_wizard.run_wizard(ctx) → answers: dict
    driver iterates STEPS:
      for step in STEPS:
        if step.when(ctx):
          fragment = step.ask(ctx)        # returns a nested config-dict FRAGMENT
          deep_merge(ctx.answers, fragment)
        │
        ▼
  setup_wizard.render(answers) → rendered: str
    map answers onto config_full.yaml placeholders (computing the
    dataset-format and validation BLOCK placeholders: chosen branch
    active, the alternatives commented), then string.Template.substitute
        │
        ▼
  setup_wizard.validate(rendered):
    load_config(<rendered string>)        # the "emitted config always loads" guarantee
        │  (validates the EXACT bytes about to be written)
        ▼
  setup_wizard.emit(rendered, output, force):
    header comment (2 lines) + exact launch command + rendered template
    write to --output (respect --force)
        │
        ▼
  init_cmd._maybe_download_weights(output, ...)   # SHARED, unchanged
```

Genuine reuse — the wizard renders the same unified template the flag path uses:

- **Single validity source:** `TrainConfig` (Pydantic). The wizard maps answers onto `config_full.yaml`, renders to a string, and validates *those exact bytes* via `load_config` before writing. The file that lands on disk is the file that was validated, so it re-loads by construction.
- **Single output template:** both `csp init` and the wizard render the one `config_full.yaml`. There is no separate wizard-only emit format — the answers-dict/registry is the extensibility mechanism, the template is the output mechanism.
- **Shared weight download:** `_maybe_download_weights` is called identically by both `init` paths.
- **Shared preset semantics:** loss/augmentation presets are set by mapping the answers onto `data.augmentations.preset`, `train.loss.preset`, `data.augmentations.intensity`, and `train.loss.class_imbalance` placeholders — the same schema fields the flag path substitutes. The wizard does not duplicate the preset tables.

### Comprehensive-output rationale

A wizard-generated config is meant to be *tweaked*, not just run — so the emitted file is comprehensive, not minimal. Rendering the unified `config_full.yaml` yields every top-level section, commonly-tuned fields echoed (wizard answers filled in, the rest at schema defaults), advanced override knobs present as commented scaffolds, and the alternative branches preserved as commented-out blocks: the chosen validation mode active with the other two commented (explicit `val:` / `val_split:` / no-val), and the chosen dataset format active with the other commented (coco / hf). The user sees the full surface area and can uncomment or edit any knob without consulting the schema docs. This is why the wizard renders the same comprehensive template the flag path renders rather than dumping only the chosen keys.

---

## §3 Component model: `WizardStep` registry (Workstream 1)

The flow is a declarative list of steps. Adding a prompt = append/insert a `WizardStep`; reorder = move a list entry; remove = delete one. No driver edits.

### `Ctx` — accumulating state + cached facts

```python
@dataclass
class Ctx:
    answers: dict[str, Any]          # the accumulating nested config dict
    cuda_available: bool             # torch.cuda.is_available(), cached once at construction
    # Cached lazily by the steps that need them:
    categories: list[str] | None = None       # dataset class names, if loadable
    category_counts: dict[str, int] | None = None  # per-category instance counts, if loadable
```

`Ctx` is constructed once, seeded with `cuda_available` (and `answers={}`). Steps read prior answers from `ctx.answers` (e.g. the class-imbalance step reads `ctx.answers["data"]` to find the annotations path) and may populate the cached-facts fields so later steps reuse them.

### `WizardStep` — one prompt (or composite of prompts)

```python
@dataclass(frozen=True)
class WizardStep:
    id: str
    ask: Callable[[Ctx], dict[str, Any]]   # runs the prompt(s); returns a config-dict FRAGMENT
    when: Callable[[Ctx], bool] = lambda ctx: True   # gate; default always-true
```

Contract:

- `ask(ctx)` ALWAYS returns a **nested config-dict fragment** keyed by schema path (e.g. `{"run": {"name": "my-run"}}`), never a bare scalar. Simple steps (one field) and composite steps (multiple fields, e.g. dataset source → `data.format` + `data.train.annotations` + `data.train.images`) share this one contract. This is what lets the driver deep-merge uniformly.
- `ask(ctx)` may read `ctx.answers` and may write to `ctx`'s cached-facts fields, but MUST NOT mutate `ctx.answers` directly — the driver owns the merge.
- `when(ctx)` is a pure predicate over `ctx`. Default is always-true.

### Driver

```python
def run_wizard(ctx: Ctx) -> dict[str, Any]:
    for step in STEPS:
        if step.when(ctx):
            fragment = step.ask(ctx)
            _deep_merge(ctx.answers, fragment)
    return ctx.answers
```

`_deep_merge(dst, src)` recursively merges nested dicts (later steps refining earlier sub-dicts, e.g. the dataset step seeds `data.format` and the validation step adds `data.val_split`). Scalars and lists overwrite. This is a small local helper (the existing `apply_overrides` in `config/loader.py` is dotted-string-based and not reused here; the wizard works with native nested dicts).

### Prompt primitives

Three thin wrappers over `rich`/`typer`, used by every `ask` so look-and-feel is consistent:

- `ask_text(prompt, *, default=None, validate=None) -> str` — free text; echoes the default; re-asks on `validate` failure.
- `ask_choice(prompt, choices, *, default=None) -> str` — membership-checked choice; re-asks on invalid; echoes the default.
- `ask_confirm(prompt, *, default=True) -> bool` — yes/no; wraps `typer.confirm`.

Each re-asks on invalid input rather than raising (except non-recoverable conditions, which surface as the final-validate backstop). All accept a default that is echoed in the prompt. `Ctrl-C` propagates out of any primitive as `KeyboardInterrupt` → the wizard writes nothing (file is emitted only at the very end). See §9.

---

## §4 Prompt flow (Workstream 1)

Ordered. "Required" means the schema field has **no default**, so it is always asked. Anything not in this table inherits its schema default silently. Result: ~7 prompts for a working train config.

Schema fields with no default (must be collected or otherwise satisfied): `run.name`, `data.format`, `data.train` (`annotations` + `images`, when COCO), `data.prompt_mode`, `peft.method`, `train.epochs`. (`data.hf.name` is required-when-HF via the `_check_format_specific` validator.) `data.prompt_mode` has no schema default but is **not prompted** — the template hardcodes `prompt_mode: text` (per #126's text-primary invariant; see §1 file table), so the wizard always emits `text` to satisfy the required field.

| # | Step `id` | Prompt | Schema target(s) | Required / Defaulted |
|---|-----------|--------|------------------|----------------------|
| 1 | `run_mode` | Run mode: `train` / `run` / `eval`. Drives the printed launch command and whether validation is needed. **Not persisted to YAML** — `run`/`train`/`eval` are CLI subcommands, not config fields. Stored in `ctx` (see note). | (none — CLI selector) | Always asked |
| 2 | `run_name` | `run.name`? | `run.name` | **Required** |
| 3 | `dataset_source` | Local COCO or HuggingFace? Then path(s). | COCO → `data.format="coco"`, `data.train.annotations`, `data.train.images`; HF → `data.format="hf"`, `data.hf.name` | **Required** |
| 4 | `validation` | Explicit val / auto-split fraction / none. | explicit → `data.val.annotations` + `data.val.images` (COCO) or `data.hf.split_val` (HF — **now wired end-to-end; see §12**); auto-split → `data.val_split.fraction`; none → (omit both) | Defaulted (auto-split 0.1 offered as default) |
| 5 | `domain` | Domain: natural / medical / satellite / microscopy / none. Then intensity: safe / medium / aggressive. | `data.augmentations.preset` **and** `train.loss.preset` (same value); `data.augmentations.intensity` | Defaulted (natural / medium) |
| 6 | `class_imbalance` | Auto-detected tier (accept/override). See §6.2. | `train.loss.class_imbalance` | Defaulted (auto-calculated; never a raw prompt) |
| 7 | `peft_sizing` | If CUDA: offer VRAM auto-size (see §6.1). Else / declined: `peft.method` (lora/qlora), accept default `r=16`. | auto-size → `peft.method`, `peft.r`, `train.batch_size`, `train.grad_accum_steps`, `model.dtype`; manual → `peft.method` | **Required** (`peft.method` always set; rest defaulted) |
| 8 | `epochs` | `train.epochs`? | `train.epochs` | **Required** (silently set to `1` in `eval` mode — schema requires it, eval ignores it) |
| 9 | `model_weights` | Checkpoint path (see §6.3). Then existing `_maybe_download_weights`. | `model.local_dir`, `model.checkpoint_file` (only when user supplies a path or a glob hit) | Defaulted (blank → schema defaults + download) |

Notes:

- **Step 1 (`run_mode`)** is the one step whose fragment is *not* a config key. It returns `{}` and stashes the choice on `ctx` (add a `run_mode: str` field to `Ctx`). It is modeled as a `WizardStep` anyway so the registry remains the single source of flow order. The emit function reads `ctx.run_mode` to print the correct launch command (`csp train`/`csp run`/`csp eval`).
- **Step 4 (`validation`)** when mode is `eval` or `run` and the user picks "none": print a one-line discouraged-but-allowed note ("eval/run needs a validation set to score against; selecting none means eval will have nothing to evaluate"). Do not block.
- **Step 5 (`domain`)** sets the same preset string into BOTH `data.augmentations.preset` and `train.loss.preset` from one choice, by design (a "domain" is one concept spanning aug + loss). `none` is a valid domain choice and maps to both presets being `none`.
- **Step 6 (`class_imbalance`)** is gated to train/run mode and categories loadable (§6.2 `when`). It is never a blind prompt: it shows a detected tier the user accepts or overrides.
- **Fields left at schema default (not prompted):** `run.output_dir`, `run.seed`, `model.name`, `model.dtype` (unless set by auto-size), `model.revision`/`device`, `data.image_size`, `data.channels`/`channel_semantics`, `data.text_prompt`, `data.normalize`, `data.limit`, `peft.alpha`/`dropout`/`scope`/`target_modules`/`bias`/`qlora`, all of `train.*` except `epochs`/`batch_size`/`grad_accum_steps`, all of `eval.*`, `tracking.*`, `export.*`.

---

## §5 Data flow: answers dict → render template → validate → emit (Workstream 1)

### 5.1 Fragment shapes (illustrative)

Each `ask` returns a fragment; the driver deep-merges. After a full local-COCO + auto-split + train-mode + manual-LoRA run, `ctx.answers` looks like:

```yaml
run:
  name: my-run
data:
  format: coco
  train:
    annotations: data/train.json
    images: data/train/
  val_split:
    fraction: 0.1
  augmentations:
    preset: medical
    intensity: medium
peft:
  method: lora
train:
  epochs: 10
  loss:
    preset: medical
    class_imbalance: moderate
```

(`model.*` keys appear only if step 9 set them; `peft.r`/`train.batch_size`/etc. appear only if auto-size ran. `data.prompt_mode` is never an answer — the template hardcodes `prompt_mode: text`.) The render stage (§5.2) maps this answers dict onto the `config_full.yaml` placeholders; it does not emit the answers dict directly.

### 5.2 Render

`render(answers, *, run_mode) -> str` maps the answers dict onto the `config_full.yaml` placeholder set and calls `string.Template(...).substitute(...)`:

- Scalar placeholders take answers values or schema defaults (e.g. `$run_name`, `$peft_method`, `$epochs`, `$aug_preset`, `$loss_preset`, `$aug_intensity`, `$class_imbalance`).
- **Block placeholders** are computed for the two branching dimensions so the chosen branch is active YAML and the alternatives are present-but-commented:
  - **Dataset format:** the chosen format (`coco` or `hf`) renders as live `data:` keys; the other format renders as a commented-out block, so the file documents both shapes.
  - **Validation mode:** the chosen mode renders live (explicit `val:` / `val_split:` / no-val) and the other two render commented-out, so a user can switch validation strategy by editing comments. **For an HF dataset, the "explicit" mode renders `hf.split_val: <name>` (not a `val:` block) and that mode is now honored at runtime — see §12. The HF "explicit" render previously dropped `split_val` and the validation system ignored it; §12 fixes both.**
- `prompt_mode: text` is hardcoded in the template body (not a placeholder, not an answer) per #126's text-primary invariant.
- Advanced knobs the wizard never prompts render as commented scaffolds carrying their schema defaults, so the output is comprehensive (§2 "Comprehensive-output rationale").

### 5.3 Validate

`validate(rendered: str)` parses and validates the rendered string via `load_config`:

- This is the "emitted config always loads" guarantee, and it validates the **exact bytes about to be written** (render → validate the string → only then write). Because the wizard already validated each answer locally (§9), a failure here is a defensive backstop. If `load_config` raises (`ConfigError` wrapping a Pydantic `ValidationError`), the wizard prints the error plus the collected `answers` dict (for debugging) and exits non-zero **without writing any file**.
- Validating via `load_config` (not bare `model_validate`) exercises the same loader path a real run uses, including the header comment the emit step prepends — the rendered string fed to `load_config` is identical to the file body.
- Path strings are rendered as the user typed them (relative or absolute). `load_config` resolves paths relative to the config file at run time via `_resolve_paths`; for validation the strings only need to parse. The emitted file carries paths verbatim, matching how the flag-driven template behaves.

### 5.4 Emit

`emit(rendered, output, force, *, run_mode, launch_cmd)`:

1. Refuse if `output.exists() and not force` (already checked pre-prompt in §7; re-checked here defensively — but the pre-prompt check is the real gate so answers aren't wasted).
2. Compose the file body:
   - Line 1: `# Generated by `custom-sam-peft init --interactive` on YYYY-MM-DD`
   - Line 2: `# Launch: <exact command>` — e.g. `custom-sam-peft train --config config.yaml` (mode-dependent: `train` / `run` / `eval`), using the actual `--output` path as `--config`.
   - Blank line, then the rendered `config_full.yaml` body from §5.2 (every section, chosen branches active, alternatives commented, advanced scaffolds).
3. `output.write_text(body)`.
4. Print `wrote <output>` and the exact launch command (so the user can copy-paste).

The 2-line header comment + exact launch command prefix the rendered template. Because §5.3 validated the rendered string (header included), the bytes written here are byte-identical to the validated string.

---

## §6 The three smart steps (Workstream 1)

### 6.1 VRAM auto-size (step 7, `peft_sizing`)

- **`when`:** `ctx.cuda_available` is true. (The opt-in confirm happens *inside* `ask`, not in `when`, so the manual fallback is reachable in the same step.)
- **Behavior:** `ask` asks `ask_confirm("Auto-size the PEFT config to your GPU's VRAM?", default=True)`.
  - **Yes:** call `decide_preset(image_size)` where `image_size` is `ctx.answers["data"].get("image_size", 1008)` (schema default 1008). This is **READ-ONLY** — `decide_preset` runs an analytic estimate (and consults a calibration cache if present); it NEVER probes the GPU and carries no OOM risk. Apply `decision.config_patch` as the fragment. Print `decision.label()` so the user sees the choice and its provenance (`analytic estimate` vs `calibrated YYYY-MM-DD`).
    - **`config_patch` shape after Workstream 2** is `{"model": {"dtype": ...}, "peft": {"method": ..., "r": ...}, "train": {"batch_size": ..., "grad_accum_steps": ...}}` — no `gradient_checkpointing` key. The wizard applies it as-is; no stripping needed (this is why the two workstreams are bundled).
    - On `RuntimeError` ("nothing fits" / CUDA / env-var) → print a one-line notice ("could not auto-size: <reason>; falling back to manual") and **fall through to the manual prompt** below.
  - **No (or fell through):** `ask_choice("PEFT method?", ["lora", "qlora"], default="lora")` → `{"peft": {"method": <choice>}}`. Accept the schema default `r=16` (do not prompt for rank — tight prompt set).
- **#148 forward-compat:** the wizard depends ONLY on `decide_preset(image_size) -> PresetDecision` and `PresetDecision.config_patch`. #148's reworking of `decide_preset`'s internals (lookup table vs. opt-in live probe) keeps this facade stable, so the wizard needs no changes when #148 lands. The wizard MUST NOT call `calibrate` or `_run_probe`.

### 6.2 `infer_class_imbalance` (step 6, `class_imbalance`)

**Location:** in `setup_wizard.py`. Rationale: it is wizard-specific (its output is a `ClassImbalance` tier shown for accept/override), and it only needs the public pycocotools-backed primitives already exported from `data/coco.py` (`_load_coco_index`, `_build_category_remap`) plus pycocotools `getAnnIds`/`loadAnns`. Putting it under `data/` would imply it's part of the dataset pipeline, which it is not. Keeping it co-located with the step that consumes it is clearer.

- **`when`:** `ctx.run_mode in {"train", "run"}` AND categories/counts are loadable (the step attempts the count inside `ask`; the `when` gate just suppresses it in pure-eval mode where loss config is irrelevant). For HF datasets, attempt category counts; fall back if not locally available.
- **Algorithm (mirrors the per-class-frequency precedent in `data/subset.py::_stratified_indices`):**
  1. Load the COCO index (`_load_coco_index(annotations)`), build the category remap (`_build_category_remap`).
  2. Count **instances per category**: iterate images via `getImgIds`, `loadAnns(getAnnIds(imgIds=[img]))`, filter `iscrowd == 0` (matching `_drop_crowd_only_images`), tally per dense category id.
  3. Over categories **present** (count > 0), compute the ratio `R = max_count / min_count`.
  4. Map `R` to a tier via NAMED module constants (tunable):

     ```python
     IMBALANCE_MODERATE_RATIO = 3.0   # R < 3 → balanced
     IMBALANCE_SEVERE_RATIO   = 10.0  # 3 <= R < 10 → moderate; R >= 10 → severe
     ```

     - `R < IMBALANCE_MODERATE_RATIO` → `"balanced"`
     - `IMBALANCE_MODERATE_RATIO <= R < IMBALANCE_SEVERE_RATIO` → `"moderate"`
     - `R >= IMBALANCE_SEVERE_RATIO` → `"severe"`
- **Output:** show the detected tier and the ratio (e.g. `detected class imbalance: moderate (max/min instance ratio = 4.2)`), then `ask_choice` to accept the detected tier (as default) or override to any of `balanced`/`moderate`/`severe`. Returns `{"train": {"loss": {"class_imbalance": <tier>}}}`.
- **Failure handling:** on ANY failure (HF dataset not locally available, unreadable annotations, zero present categories making `min_count == 0`, pycocotools error) → default to `"balanced"` with a one-line notice (`could not auto-detect class imbalance (<reason>); defaulting to balanced — override in config if needed`). Still returns a fragment so the field is set explicitly. (Note: `balanced` is also the schema default; emitting it explicitly documents that the wizard considered it.)

### 6.3 Model checkpoint / weights path (step 9, `model_weights`)

One prompt, exact text:

> Path to an existing SAM 3.1 checkpoint (.pt)? Leave blank to use `models/sam3.1` and download if missing.

- **Non-blank path:**
  - Resolve and require it to be an existing file. If not a file → re-ask (`ask_text` with a path-existence validator).
  - Set `model.local_dir` = the file's parent directory, `model.checkpoint_file` = the filename. Returns `{"model": {"local_dir": <parent>, "checkpoint_file": <name>}}`.
- **Blank:**
  - Default to `models/sam3.1` + `sam3.1_multiplex.pt` (the schema defaults).
  - **Shallow glob:** search `models/**/<checkpoint_file>` (i.e. `models/**/sam3.1_multiplex.pt`). If found elsewhere, set `model.local_dir` to that match's parent and return that fragment. If not found, return `{}` (leave the schema defaults) and let the existing `_maybe_download_weights` fetch it.
- After the wizard writes the config, `init` calls `_maybe_download_weights(output, download_weights=..., yes=...)` exactly as the flag-driven path does — honoring `--download-weights/--no-download-weights` and `--yes`.

---

## §7 `init` command surface (Workstream 1)

`init` gains one flag; everything else is preserved.

```
custom-sam-peft init [--interactive/-i] [--output PATH] [--force]
                     [--download-weights/--no-download-weights] [--yes]
                     [--template ...] [--preset ...] [--intensity ...] [--class-imbalance ...]
```

- **`--interactive` / `-i`** (`bool`, default `False`): when set, run the wizard. The flag-driven options (`--template`, `--preset`, `--intensity`, `--class-imbalance`) are ignored in interactive mode (the wizard collects those interactively); document this in the flag help.
- **`--template`** (`coco-text-lora` / `coco-text-qlora`, flag-driven path): the CLI surface is preserved — both values are still accepted — but both now map onto the **`$peft_method` placeholder of the unified `config_full.yaml`** (`coco-text-lora`→`lora`, `coco-text-qlora`→`qlora`) rather than selecting one of two template files. The two legacy templates no longer exist (§1, §8.7).
- **Honored in interactive mode:** `--output`, `--force`, `--download-weights/--no-download-weights`, `--yes`.
- **Pre-flight order in `init` when `--interactive` is set (BEFORE any prompt):**
  1. **TTY check.** If `not sys.stdin.isatty()` → hard error up front: `typer.BadParameter("interactive setup needs a TTY; use the flag-driven `custom-sam-peft init …` instead")`. (Plain `csp init` non-TTY behavior is unchanged — it does not require a TTY.)
  2. **Output-exists check.** If `output.exists() and not force` → `typer.BadParameter` (same message/`param_hint="--output"` as today's `FileExistsError` path). Checked here so the user does not answer ~7 prompts only to be refused at write time.
  3. Build `Ctx(answers={}, cuda_available=torch.cuda.is_available())`, run the driver, render, validate, emit, then `_maybe_download_weights`.
- **Without `--interactive`:** `init` calls `run_init(...)` as before, with the **CLI surface unchanged but the output changed**: `run_init` now renders the unified `config_full.yaml` (mapping `--template` onto `$peft_method` plus flag-driven defaults for placeholders the flags do not supply), so flag-driven `init` ALSO emits the comprehensive file. Both paths share the one template — this reverses the earlier "flag-driven init byte-for-byte unchanged" non-goal (see §1).

---

## §8 Workstream 2: full gradient-checkpointing removal

GC is abandoned (#60/#89/#127 closed), already a no-op on the current SAM 3.1 revision (the warning in `sam3.py`), and main ships it off. Remove the user knob, the runtime lever, and the presets search dimension. **No migration, no back-compat shim.**

### 8.1 `config/schema.py`

Remove `ModelConfig.gradient_checkpointing` entirely (field + its `TODO(#60)` comment). Because `_Strict` sets `extra="forbid"`, any YAML still carrying `model.gradient_checkpointing:` now **fails to load** with a Pydantic "extra fields not permitted" error. This is the breaking change; it is intentional and documented in §11. No shim, no alias, no deprecation warning.

### 8.2 `presets.py`

- `PresetDecision`: remove the `gradient_checkpointing: bool` field.
- `config_patch`: drop `"gradient_checkpointing"` from the `"model"` sub-dict (leaving `{"model": {"dtype": ...}, "peft": {...}, "train": {...}}`).
- `_candidates()`: collapse to ckpt-off only — remove the `ckpt ∈ {False, True}` dimension; candidates become `(method, r, batch)` triples (the search space halves to 192 candidates).
- `_sort_key`: drop the `0 if not ckpt else 1` tiebreaker tuple element; key becomes `(method-rank, -r, -batch)`.
- Delete the `CKPT_FACTOR` constant.
- `_activation_bytes(image_size, batch, cache)`: drop the `ckpt` parameter and the `factor = CKPT_FACTOR if ckpt else 1.0` logic (activation = `per * batch`).
- `_predicted_bytes(...)`: drop the `ckpt` parameter; update both `train` and `eval` branches and all internal callsites.
- `label()`: remove the `ckpt={on|off}` token from the formatted string.
- `decide_preset`: update the feasible-candidate unpacking (`method, r, batch, predicted` — no `ckpt`); update the "nothing fits" `RuntimeError` message to drop "ckpt=on": new text references `QLoRA r=4 batch=1` only (e.g. `… SAM 3.1 needs ≈Y GiB even at QLoRA r=4 batch=1. Use a larger GPU.`); update the `min_needed = _predicted_bytes("qlora", 4, 1, image_size, cache)` call.
- `decide_eval_batch_size`: update its `_predicted_bytes(...)` calls (remove the `ckpt=False` argument).
- `to_json`/`from_json`: no special handling needed once the field is gone (they use `asdict`/`**d`), but the round-trip test must no longer reference `gradient_checkpointing`. Verify no stray references remain.

### 8.3 `models/sam3.py`

Remove the entire `if cfg.gradient_checkpointing:` block (the `set_grad_checkpointing` branch and the no-op warning). Nothing replaces it.

### 8.4 `train/loop.py`

- `OomState`: remove the `gradient_checkpointing: bool = False` field.
- `_train_step_with_oom_ladder`: remove the `if not state.gradient_checkpointing:` rung (the GC-enable branch and its `OomEvent`). The ladder becomes two rungs: **halve micro_batch** → (at `micro_batch == 1`) **raise**.
- Final raise message: drop "gradient_checkpointing=on". New text: `f"OOM at step {state.step} after micro_batch=1. Use a larger GPU or smaller image_size."`
- Update the `OomState`/helper docstrings that mention "gradient_checkpointing toggles at most once per run".

### 8.5 `train/types.py`

- `OomEvent.action`: `Literal["microbatch_halved", "grad_ckpt_enabled"]` → `Literal["microbatch_halved"]`.
- Remove the `new_gradient_checkpointing: bool` field.
- Update the docstring to describe the single remaining rung.

### 8.6 `runs/bundle.py`

- `_preset_block`: remove `ckpt_word = "on" if preset.gradient_checkpointing else "off"` and the `gradient_checkpointing={ckpt_word}` clause from the `- Method:` line (now `… grad_accum={...}, bf16`).
- `_oom_edge_note`: remove the `ckpt_event` lookup and the `, gradient_checkpointing enabled at step S` clause. The note becomes just `OOM retries: N — final micro_batch=M`.

### 8.7 Templates

No separate GC edit to template files: the template consolidation in Workstream 1 (§1, §7) already deletes `coco_text_lora.yaml` and `coco_text_qlora.yaml`, and the unified `config_full.yaml` that replaces them simply ships **without** a `gradient_checkpointing` line. So GC removal touches no template directly — the GC-free template state falls out of WS1's consolidation.

### 8.8 Docs

- `docs/config-schema.md`: remove the `model.gradient_checkpointing` table row.
- **Leave historical specs under `docs/superpowers/specs/` untouched** — they are point-in-time records (notably `2026-05-22-algo-vram-preset-design.md`, which documents the now-removed GC rung; it is not edited).

---

## §9 Error handling & edge cases (Workstream 1)

| Condition | Behavior |
|-----------|----------|
| `--interactive` + non-TTY (`not sys.stdin.isatty()`) | Hard error up front: `BadParameter("interactive setup needs a TTY; use the flag-driven `custom-sam-peft init …`")`. Checked before any prompt. |
| Plain `csp init` (no `--interactive`) non-TTY | Unchanged — does not require a TTY. |
| `--output` exists & no `--force` | `BadParameter` (same as today's `FileExistsError`→`--output` path), checked BEFORE prompting so answers aren't wasted. |
| Per-answer invalid input (choice not in set, path doesn't exist, non-positive int for epochs) | Immediate re-ask via the prompt primitive's `validate`/membership loop. Never raises mid-wizard for recoverable input. |
| Final `load_config(rendered)` fails | Defensive backstop: print the `ConfigError`/Pydantic error + the collected `answers` dict, exit non-zero, write NOTHING. (Should be unreachable given per-answer validation; if it fires it's a wizard or template bug.) |
| `Ctrl-C` mid-wizard (`KeyboardInterrupt`) | Nothing written — the file is emitted only at the very end, after validation. Propagates as a clean abort. |
| `decide_preset` `RuntimeError` (nothing fits / env-var) | Notice + fall back to manual `peft.method` prompt (§6.1). |
| `infer_class_imbalance` failure (HF not local, unreadable annotations, zero present cats) | Default `balanced` + one-line notice (§6.2). |
| Checkpoint path given but not a file | Re-ask (§6.3). |

`epochs` validation: must parse to a positive int (`PositiveInt` in schema). In `eval` mode the step is skipped and `train.epochs` is set to `1` silently (schema requires it; eval ignores it).

---

## §10 Testing strategy

Per-project rule: CPU-testable cases live on CPU; GPU tests are reserved for real-only failure modes. Honored — only ONE existing GPU test is adjusted; nothing new lands on GPU.

### 10.1 Wizard (CPU) — `tests/unit/cli/test_setup_wizard.py` (new)

Drive prompts via `monkeypatch` of the prompt primitives (preferred — deterministic) or Typer `CliRunner` with `input=`. Cases:

| Test | Asserts |
|------|---------|
| `test_step_fragment_shapes` | each `ask` returns a nested dict fragment (never a bare scalar); deep-merge composes them correctly. |
| `test_when_gating_skips_class_imbalance_in_eval_mode` | step 6 `when` returns False when `ctx.run_mode == "eval"`. |
| `test_when_gating_skips_vram_autosize_without_cuda` | step 7 takes the manual branch when `cuda_available` is False. |
| `test_non_tty_hard_errors` | `--interactive` with `stdin.isatty()` patched False → `BadParameter`, no prompts, no file. |
| `test_output_exists_without_force_errors_before_prompting` | pre-flight refuses; no prompt primitive is called. |
| `test_output_force_overwrites` | `--force` allows overwrite. |
| `test_happy_path_local_coco_autosplit_reloads` | full local-COCO + auto-split + train run; emitted YAML re-loads via `load_config` and equals expected `TrainConfig`. |
| `test_happy_path_hf_reloads` | full HF-dataset run; emitted YAML re-loads via `load_config`. |
| `test_emit_header_and_launch_command` | emitted file starts with the 2-line header + correct `csp <mode>` launch comment for each of train/run/eval. |
| `test_emit_is_comprehensive_and_reloads` | emitted file has ALL top-level sections present; the chosen dataset-format and validation branches are active AND the alternative branches are present-but-commented; `prompt_mode: text` is emitted though never prompted; the rendered output re-loads via `load_config`. |
| `test_validate_backstop_exits_nonzero_no_file` | inject an answers dict that renders to invalid config → backstop prints error + answers, exits non-zero, writes nothing. |
| `test_ctrl_c_writes_nothing` | `KeyboardInterrupt` raised mid-wizard leaves no file. |

### 10.2 `infer_class_imbalance` (CPU) — same test module or `test_infer_class_imbalance.py`

Tiny synthetic COCO annotation JSONs written to `tmp_path`:

| Test | Synthetic data | Asserts |
|------|----------------|---------|
| `test_balanced_below_3x` | per-cat counts e.g. 10/10/12 (R≈1.2) | tier `"balanced"`. |
| `test_moderate_3x_to_10x` | counts 10/40 (R=4) | tier `"moderate"`. |
| `test_severe_at_or_above_10x` | counts 5/100 (R=20) | tier `"severe"`. |
| `test_thresholds_are_boundary_exact` | counts giving R exactly 3.0 and 10.0 | 3.0 → moderate, 10.0 → severe (boundary mapping). |
| `test_unreadable_annotations_defaults_balanced` | path to nonexistent / malformed file | `"balanced"` + notice, no raise. |
| `test_iscrowd_excluded_from_counts` | iscrowd=1 instances present | excluded from the ratio. |

### 10.3 VRAM auto-size step (CPU) — `decide_preset` monkeypatched

| Test | Asserts |
|------|---------|
| `test_vram_autosize_applies_config_patch` | `decide_preset` patched to return a `PresetDecision`; step fragment equals `decision.config_patch`; `label()` printed. |
| `test_vram_autosize_runtime_error_falls_back_to_manual` | `decide_preset` raises `RuntimeError` → manual `peft.method` prompt path taken; notice printed. |
| `test_vram_autosize_config_patch_has_no_gradient_checkpointing` | the applied fragment's `model` sub-dict has no `gradient_checkpointing` key (cross-workstream guard). |

No real CUDA in any wizard test.

### 10.4 GC removal (CPU)

| File | Adjustment |
|------|-----------|
| `tests/unit/test_model_config.py` | assert `model.gradient_checkpointing:` in YAML now raises (extra-forbidden); drop any default-value assertion on the removed field. |
| `tests/unit/test_presets.py` | drop GC assertions; `config_patch` `model` sub-dict has no `gradient_checkpointing`; `label()` has no `ckpt=` token; search space no longer enumerates ckpt; "nothing fits" message updated; `to_json`/`from_json` round-trip without the field. |
| `tests/unit/test_trainer_oom_retry.py` | **Rewrite to the 2-rung ladder.** Remove `test_oom_after_microbatch_1_enables_ckpt` and `test_oom_ckpt_toggle_is_once`; rewrite `test_oom_after_ckpt_enabled_raises` → `test_oom_after_microbatch_1_raises` (asserts `RuntimeError("OOM at step … after micro_batch=1. …")`, no GC clause). Keep microbatch-halving, stickiness, zero-grad-once, event-propagation tests. |
| `tests/unit/test_train_types.py` | `OomEvent.action` is `Literal["microbatch_halved"]`; no `new_gradient_checkpointing` field. |
| `tests/unit/runs/test_bundle.py` | `## Preset` block has no `gradient_checkpointing=` clause; `_oom_edge_note` output has no "gradient_checkpointing enabled at step S" clause. |
| `tests/unit/test_data_transforms.py` | drop any `gradient_checkpointing` reference in fixture configs. |
| `tests/integration/test_load_sam31_real.py` | drop GC-related assertions / config keys. |
| `tests/integration/test_cli_run.py` | drop GC from any generated/asserted config. |

### 10.5 GPU

| File | Adjustment |
|------|-----------|
| `tests/gpu/test_multiplex_vram.py` | adjust the single existing multiplex VRAM test to the GC-free `PresetDecision`/`config_patch` shape. No new GPU test. |

### 10.6 Flag-driven `init` retargeted to the unified template (CPU) — `tests/unit/test_cli_init.py`

Updated for the single unified `config_full.yaml` (the two legacy template files no longer exist):

| Test | Asserts |
|------|---------|
| both `--template` values accepted | `coco-text-lora` and `coco-text-qlora` both succeed and map onto `$peft_method` (`lora` / `qlora`) on the one template. |
| emitted config is comprehensive + valid | flag-driven output now has all top-level sections and re-loads via `load_config`; update any assertion that expected the old minimal per-template output. |
| qlora block present | `--template coco-text-qlora` renders the `qlora:` sub-block in the emitted file. |
| no stale template filenames | drop any test reference to `coco_text_lora.yaml` / `coco_text_qlora.yaml`; assert against `config_full.yaml` instead. |

---

## §11 Migration & breaking-change stance

**Pre-1.0. The GC-knob removal is a clean breaking change. No shim, no deprecation path. Users update their YAML.**

| What breaks | Who | How they notice |
|-------------|-----|-----------------|
| `model.gradient_checkpointing:` in any YAML | anyone with an existing config carrying the key | `ConfigError` wrapping a Pydantic "extra fields not permitted" error at load (because `_Strict` forbids extras). Fix: delete the line. |
| `PresetDecision.gradient_checkpointing` field removed | anyone constructing/serializing `PresetDecision` directly (only `presets.py` + sidecar `preset.json`) | `TypeError` on init / absent JSON key. Stale `preset.json` files from prior runs will fail `from_json` (a new run regenerates them). |
| `OomEvent.action == "grad_ckpt_enabled"` removed | anyone matching on that literal | the value is never produced; matches are dead code. |

Release-notes copy (3 steps):

1. **Delete `model.gradient_checkpointing` from your config YAML.** The knob is gone; keeping it now fails config load. It was a no-op on the current SAM 3.1 revision anyway.
2. **Re-run any saved preset sidecars.** `preset.json` files from older runs no longer deserialize; a new `run` regenerates them.
3. **The OOM auto-retry ladder no longer enables gradient checkpointing.** It now halves the micro-batch and, at micro-batch 1, raises with guidance to use a larger GPU or smaller `image_size`.

### Forward-compat with #148

The wizard's VRAM step depends only on the public `decide_preset(image_size) -> PresetDecision` facade and `PresetDecision.config_patch`. #148 will rework `decide_preset`'s internals (lookup table vs. opt-in live probe) but preserve that signature and the 3-section `config_patch` shape. Therefore this PR introduces no coupling to #148, and #148 can land later without touching `setup_wizard.py`.

### Rollback

Revert the PR. Workstream 1 adds the wizard (`setup_wizard.py` + the `--interactive` branch) and consolidates templates (deletes `coco_text_lora.yaml` / `coco_text_qlora.yaml`, adds `config_full.yaml`, repoints `run_init` at it). Reverting restores the two legacy template files and the old per-template `run_init` output alongside removing the wizard. Workstream 2's GC removal revert restores the schema field, the presets ckpt dimension, the sam3 no-op block, and the OOM GC rung; it touches no template (the GC-free state lived in the unified template, which the WS1 revert handles). The two workstreams are one logical change and must revert as a unit to keep the schema, presets, and templates consistent.

---

## §12 Amendment: HF explicit-validation wiring (post-review)

**Status:** post-Checkpoint-B amendment. Closes a design defect surfaced in review. **Resolution chosen: "wire `hf.split_val` for real"** — extend the validation system so an HF config that names `hf.split_val` actually runs validation against that HF split (`mode='explicit'`), and have the wizard render it. Same pre-1.0, no-shim stance as §11.

### 12.1 The defect

The §4-row-4 / §5.2 promise that an HF dataset can pick **explicit** validation was a no-op end-to-end:

- `data/val_source.py::resolve_val_source` — the authority on validation mode — chose the mode ONLY from `cfg.data.val_split` (→ `auto_split`) or `cfg.data.val` (→ `explicit`), else `none`. It never consulted `cfg.data.hf.split_val`. So an HF + "explicit" config silently resolved to **`none`** and never validated during training.
- `eval/runner.py` rejected `--split val` for an HF config (`raise ValueError("--split val requires data.val or data.val_split …")`), so `eval --split val` on an HF + split_val config **errored**.
- The wizard's `render._dataset_block` HF branch **dropped** the collected `split_val` entirely (it never emitted `split_val:`), and additionally emitted a spurious COCO-shaped `train:` block under `data:` for HF datasets.

`_ask_validation` already collected the HF val split correctly (`{"data": {"hf": {"split_val": split}}}`); the failure was downstream (schema gating + render + resolver + eval gate).

### 12.2 The opt-in problem and chosen signal

`HFDatasetConfig.split_val` defaulted to `"validation"` — a non-empty string ALWAYS present. Gating "split_val present → explicit val" naively would make EVERY HF config validate, breaking the "none" option. An opt-in signal is required.

**Chosen opt-in:** change `HFDatasetConfig.split_val: str = "validation"` → `str | None = None`. Rationale:

- The old default `"validation"` was **inert for gating** (the resolver ignored `split_val` entirely), so flipping the default to `None` changes **no current runtime behavior**: an HF config with no `val`/`val_split` resolved to `none` before AND after.
- With the new gating, **`split_val is not None`** is the explicit opt-in. The wizard's HF-explicit path sets it; HF-none leaves it `None`; HF-auto-split uses `data.val_split` (unchanged, and carves val out of `split_train`).
- The HF builder's `else` branch (`data/hf.py` ~line 414) reads `hf_cfg["split_val"]` only on the explicit eval path (no `_resolved_image_ids`, no COCO `val`). Under the new gating that path is reached **only when `split_val is not None`**, so the builder never reads `None`. This invariant holds for both train-time val building (`train/runner.py` builds the eval dataset only when `vs.mode != "none"`) and `eval --split val` (only reached past the relaxed gate, which requires `split_val is not None`). No builder change is needed.
- Pre-1.0 breaking change, no shim, consistent with §11. Existing code relying on the `"validation"` default is updated (see §12.7).

### 12.3 Schema (`config/schema.py`)

- `HFDatasetConfig.split_val: str = "validation"` → `split_val: str | None = None`.
- **Retarget the existing `_check_hf_split_val_compat` validator** from the `"validation"` sentinel to the new `None` sentinel. It currently rejects `val_split` + a *customized* `split_val` via `self.hf.split_val != "validation"`; change that condition to `self.hf.split_val is not None`. The error message stays accurate ("data.hf.split_val cannot be customized when data.val_split is set; auto-split carves the val set from data.hf.split_train. Remove split_val or remove val_split.").
- `_check_val_modes` (the `val` ⊻ `val_split` mutual-exclusivity rule) is unchanged.

**Mutual-exclusivity rules (recommended, enforced in `DataConfig` after-validators):**

| Pair | Rule | Where |
|------|------|-------|
| `val` & `val_split` | mutually exclusive (existing) | `_check_val_modes` (unchanged) |
| `val_split` & `hf.split_val` (set) | mutually exclusive — auto-split owns the val set; a named HF val split would contradict it | `_check_hf_split_val_compat` (retargeted to `is not None`) |
| `val` & `hf.split_val` | **NOT guarded.** `data.val` is COCO-only and `hf.split_val` is HF-only; `data.format` selects exactly one builder, so the unused one is dead config. Adding a guard is YAGNI; leave it. (The wizard never emits both.) |

### 12.4 `resolve_val_source` (`data/val_source.py`) — new dispatch priority

Insert an HF-explicit branch into the existing 1-2-3-4 dispatch. New ordering:

1. `run_dir/val_source.json` exists → resume (unchanged).
2. `cfg.data.val_split is not None` → `auto_split` (unchanged). *(Takes priority over split_val; the schema validator already forbids `val_split` + a set `split_val`, so this branch and #4 are mutually exclusive by construction.)*
3. `cfg.data.val is not None` → `explicit` (COCO, unchanged).
4. **NEW:** `cfg.data.format == "hf" and cfg.data.hf is not None and cfg.data.hf.split_val is not None` → `explicit`.
5. else → `none`.

The new HF-explicit branch yields **the same `ValSource(mode="explicit", …)`** the COCO `val` branch yields: `train_ids=None`, `val_ids=None` (full-split, no id filtering — the builder reads the whole `split_val`), and all the auto-split-only fields `None`. The two explicit sources share `mode='explicit'`; downstream consumers (`train/runner.py` builds val whenever `vs.mode != "none"`, `_log_val_source` logs "explicit") need no special-casing. (Optional polish: `_log_val_source`'s `mode=="explicit"` line currently logs "explicit (cfg.data.val)"; broaden to "explicit (cfg.data.val or data.hf.split_val)" so the HF case isn't mislabeled — cosmetic, log-only.)

### 12.5 `eval/runner.py` — relax the `--split val` gate

The gate at ~line 104 currently reads:

```python
if split == "val" and cfg.data.val is None and cfg.data.val_split is None:
    raise ValueError("--split val requires data.val or data.val_split in config; got neither.")
```

Relax it to also accept HF + `split_val`:

```python
_hf_val = cfg.data.format == "hf" and cfg.data.hf is not None and cfg.data.hf.split_val is not None
if split == "val" and cfg.data.val is None and cfg.data.val_split is None and not _hf_val:
    raise ValueError(
        "--split val requires data.val, data.val_split, or data.hf.split_val in config; got none."
    )
```

The existing `val_dataset is None` build path (lines ~109-118) needs no change: for HF-explicit there is no `_resolved_image_ids` and no `val`, so the HF builder's `else` branch reads `split_val` for `pipeline="eval"` — exactly the intended split.

### 12.6 No change needed downstream

- **`train/runner.py`** — already builds the eval dataset whenever `vs.mode != "none"` (~line 106). The new HF-explicit `mode='explicit'` is picked up automatically. No edit.
- **`data/hf.py`** — the `else` branch already reads `split_val` on the explicit eval path. Under the new gating it is reached only when `split_val is not None`, so it never reads `None`. No edit, no guard.

### 12.7 Existing code/tests that rely on the old `"validation"` default

| Location | Reliance | Required update |
|----------|----------|-----------------|
| `src/custom_sam_peft/config/schema.py:475` | `_check_hf_split_val_compat` uses `self.hf.split_val != "validation"` as the "customized" check | retarget to `self.hf.split_val is not None` (§12.3) |
| `tests/unit/test_data_schema_extensions.py:85` | `assert cfg.split_val == "validation"` (default-value assertion) | change to `assert cfg.split_val is None` |
| `tests/unit/test_config_schema.py:310` (`test_hf_split_val_default_with_val_split_validates`) | builds `hf={"name": …}` (relying on default `split_val="validation"`) + `val_split`, expects VALIDATE | still valid under `None` default (None + val_split passes the retargeted validator); keep the test, update its comment from `# default split_val="validation"` to `# default split_val=None` |
| `tests/unit/test_config_schema.py:296` (`test_hf_split_val_custom_with_val_split_rejected`) | `split_val="custom_val"` + `val_split`, expects rejection | unchanged — `"custom_val"` is not None, still rejected |

No source consumer outside the schema validator and the HF builder reads `split_val`. The init template comment string (`init_cmd.py:70`) and the wizard's COCO-branch comment scaffold are literal documentation text, not gating — they keep showing `split_val: validation` as an illustrative example in the commented HF-alternative block.

### 12.8 Wizard render fix (`cli/setup_wizard.py`)

`render`'s `_dataset_block` HF branch must (a) stop emitting the spurious COCO `train:` block under `data:`, and (b) render `split_val: <name>` under `hf:` when the answers carry it. `_ask_validation` is unchanged (its HF-explicit return `{"data": {"hf": {"split_val": split}}}` is already correct). The validation step keeps offering all three modes for HF.

Exact rendered shapes (2-space indent under `data:`):

- **HF + explicit** (`data.hf.split_val` set, no `val`, no `val_split`): `_dataset_block` emits `format: hf`, `hf:` with `name:` AND `split_val: <name>`; `_validation_block` must NOT render a contradictory active "no-val" line. Since HF-explicit sets neither `data.val` nor `data.val_split`, `_validation_block`'s current dispatch would fall to the `noval_active` branch and emit `# no-val mode: …` as the ACTIVE line — wrong. Fix `_validation_block` to detect HF-explicit (format == "hf" and `hf.split_val` present) and render an active note pointing at `split_val` instead, e.g. `# validation: HF split 'data.hf.split_val' (set above) is used as the val set.` The split name lives in the dataset block; the validation block just must not claim no-val.
- **HF + auto-split** (`data.val_split` set): `_dataset_block` emits `format: hf` + `hf: name:` (no `split_val`); `_validation_block` renders the active `val_split:` block. Unchanged behavior.
- **HF + none** (neither set): `_dataset_block` emits `format: hf` + `hf: name:` (no `split_val`); `_validation_block` renders the active no-val note. Unchanged behavior.

The commented HF-alternative scaffold in the COCO `_dataset_block` branch may keep showing `# split_val: validation` as illustrative example text.

### 12.9 Tests (CPU only)

All new coverage is CPU-testable at the config / dataclass / resolver level — **do not load a real HF dataset.** Test the mode decision and gate logic, not data loading.

- **Schema default** (`tests/unit/test_data_schema_extensions.py`): update line 85 to `assert cfg.split_val is None`.
- **Schema validator** (`tests/unit/test_config_schema.py`): keep the two existing `val_split` + `split_val` tests (comment tweak per §12.7); add a test that HF + `split_val` set + NO `val_split`/`val` validates cleanly.
- **`resolve_val_source`** (`tests/unit/` — new or in the val-source test module): HF config with `hf.split_val="myval"`, no `val`/`val_split` → `resolve_val_source(cfg).mode == "explicit"` and `train_ids is None and val_ids is None`; HF config with `split_val=None` → `mode == "none"`.
- **`eval/runner` gate** (`tests/unit/`): HF + `split_val` set + `--split val` no longer raises the "requires data.val or data.val_split" `ValueError` (assert the gate passes — exercise just the gate, monkeypatching/short-circuiting the build+load so no model or dataset is loaded); HF + `split_val=None` + `--split val` still raises.
- **Wizard `render`** (`tests/unit/cli/test_setup_wizard.py`): HF + explicit answers (`{"data": {"format": "hf", "hf": {"name": "org/ds", "split_val": "myval"}}, …}`) → rendered output contains `split_val: myval`, contains no active no-val line, and **re-loads via `load_config` with `cfg.data.hf.split_val == "myval"`** (and, by §12.4, would resolve to explicit mode); add/keep an HF-none render test asserting no `split_val:` line is emitted.

### 12.10 Docs (`docs/config-schema.md`)

Update the `data.hf.split_val` row: default `null` (was `"validation"`), and describe the explicit-HF-val behavior — "HF split name used as the validation set; when set (and `data.val_split` is unset), validation runs against this split (mode='explicit'). Null → no HF-driven validation."

### 12.11 Risks / edge-cases for the implementer

- **`val_split` vs `split_val` mutual exclusivity** is enforced by the retargeted `_check_hf_split_val_compat` (`split_val is not None` while `val_split` set → reject). This is the load-bearing rule; the resolver's branch-2-before-branch-4 ordering relies on it so the two never both fire. Verify the validator retarget keeps `test_hf_split_val_custom_with_val_split_rejected` green and `test_hf_split_val_default_with_val_split_validates` green.
- **`model_dump()` carries `split_val: None`** into `data_cfg_dict` for HF none/auto-split runs. Confirm the HF builder never indexes `hf_cfg["split_val"]` on those paths (it does not: none → eval dataset not built; auto-split → `_resolved_image_ids` set → `split_train` branch).
- **`_minimal_dict()` HF fixtures** in `test_config_schema.py` that omit `split_val` now get `None` instead of `"validation"`; confirm none assert the old default.
