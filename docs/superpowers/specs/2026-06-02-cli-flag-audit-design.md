# CLI Flag-Surface Audit + Consistency Cleanup — Design

**Status:** Draft (2026-06-02)
**Issue:** [#115](https://github.com/NguyenJus/custom-sam-peft/issues/115) —
`chore(cli): audit flag surface — intuitiveness, coverage gaps, cross-command
consistency` (priority:low)
**Scope:** `src/custom_sam_peft/cli/` — all eight subcommands (`train`, `run`,
`eval`, `export`, `init`, `doctor`, `predict`, `calibrate`), the shared option
vocabulary, config discovery, and the convenience-flag override path. Touches
`config/loader.py` (the existing `apply_overrides` seam, unchanged), `README.md`,
and the command strings emitted by `setup_wizard` / `_interactive`.
**Relationship:** Reuses the current tree-walk config discovery via one shared
helper so issue [#249](https://github.com/NguyenJus/custom-sam-peft/issues/249)
(self-describing checkpoints) can later upgrade that single helper.

---

## 1. Overview & Motivation

The CLI grew to eight commands incrementally. The newest, `predict`, shipped
with the full modern flag kit — validated `Choice` types, validation callbacks,
a `--dry-run` preview, and an explicit `--device` / `--seed` / `--dtype` triad.
The older commands (`train`, `run`, `eval`, `export`) were never retrofitted, so
the surface drifted: `run` has no `--override`, `eval` accepts `--split` as an
unvalidated string, three commands lack `-v`, and `--progress` is re-declared as
a bare `str` in every command instead of a shared validated type.

This work does two things. First, it **audits** the surface and records the
result as durable spec content: a flag-classification table and a task→flag-path
coverage matrix. Second, it **implements** the consistency fixes the audit
implies, decomposed into three phases (foundation refactor, additive
standardization, breaking cleanup + propagation).

The codebase's design priorities, in order, are: **final accuracy > user-facing
simplicity >> training speed.** A speed-only benefit is a weak reason to keep a
config knob. The project's goal is to let a user finetune a SAM-family
foundation model on their own data with a single local or cloud GPU (CPU is not
a supported training target). Several decisions below — dropping `predict
--dtype`, deriving `--merge-adapter` — follow directly from "user-facing
simplicity beats a speed-only knob."

### 1.1 Implementation framework (verified)

The CLI is built on **Typer**, not raw Click. Every command parameter is a
`typer.Option(...)`. Validated choice flags are expressed today as
`typer.Option(..., click_type=click.Choice([...]))` (e.g. `predict --save-masks`,
`--device`, `--dtype`), and per-value validation uses Typer `callback=`
functions (`predict --score-threshold`, `--top-k`, `--batch-size`,
`--checkpoint`). The shared-vocabulary plan below uses Typer's
`Annotated[T, typer.Option(...)]` form (supported since typer ≥ 0.12), which is
the idiomatic single-source-of-truth mechanism for Typer parameters.

---

## 2. Current Surface (audit baseline)

Verified against the worktree at commit `1f7cfaf` (PR #248). Inventory
corrections versus the task brief are flagged inline as **[drift]**.

- **train** (`train_cmd.py`): `--config` (required), `--override` (repeatable),
  `--resume` (optional-value via the `_ResumeAwareGroup` patch), `--time-limit`
  (`DURATION` metavar), `--eval`, `--export`, `-v`/`--verbose`, `--progress`
  (bare `str`, `metavar=MODE`, unvalidated — `resolve_mode` interprets it).
- **run** (`run_cmd.py`): `--config` (required), `--resume`, `--time-limit`,
  `--finalize`, `-v`/`--verbose`, `--progress`, `--visualize/--no-visualize`
  (default `True`). **No `--override`** (the headline gap). Auto-inits the
  config if the path is missing.
- **eval** (`eval_cmd.py`): `--config` (signature `Optional`, but the body
  raises `typer.BadParameter("--config is required")` unless `--interactive`),
  `--checkpoint` (omit ⇒ baseline zero-shot), `--split` (plain `str`, default
  `"val"`; help says `val | test`; **[drift]** it *is* validated in the body —
  `if split not in ("val", "test"): raise typer.BadParameter` — just not
  enum-*typed* at the parser layer), `--output`, `--save-predictions/--no-`,
  `--visualize/--no-`, `--export`, `-v`, `--progress`, `-i`/`--interactive`.
- **export** (`export_cmd.py`): `--checkpoint` (required), `--merge`, `--output`
  (`Optional`; conditionally required when not `--merge` — **[drift]** the
  required-when-not-merging raise lives in `run_export` in `runs/bundle.py` as a
  `ValueError`, surfaced by the CLI as `typer.BadParameter(param_hint=
  "--output")`, not as a manual raise inside `export_cmd.py`), `--config`
  (`Optional`; auto-discovered via `_discover_config` tree-walk), `-v`,
  `--progress`.
- **init** (`init_cmd.py`): `--template` (`coco-text-lora` | `coco-text-qlora`),
  `--preset` (`natural|medical|satellite|microscopy|none|custom`,
  `case_sensitive=False`, choices only in help prose — validated post-parse in
  `run_init` against `get_args(Preset)`), `--intensity` (`safe|medium|aggressive`,
  same pattern), `--class-imbalance` (`balanced|moderate|severe`, validated via
  `typer.BadParameter` in `run_init`), `--output` (default `config.yaml`),
  `--force`, `--download-weights/--no-`, `--yes` (no `-y` short form),
  `-i`/`--interactive`. **No `-v`.**
- **doctor** (`doctor_cmd.py`): `--weights-path`, `--json`, `--config` (a
  config-*validation* feature: loads + validates, reports resolved dataset
  sizes + resolved augs/normalization/losses). **No `-v`.**
- **predict** (`predict_cmd.py`): `--images` (req), `--prompts` (req),
  `--output` (req), `--checkpoint` (callback-validated), `--merge-adapter/--no-`
  (default `True`), `--config`, `--score-threshold` (callback `[0,1]`),
  `--top-k` (callback positive int), `--save-masks` (`Choice rle|png|none`),
  `--visualize`, `--device` (`Choice auto|cuda|cpu`), `--dtype` (`Choice
  auto|bfloat16|float32`), `--batch-size` (`auto`|positive int via callback),
  `--seed` (default `0` — **[drift]** note `run.seed` schema default is `42`;
  this `0` is itself a cross-command inconsistency the audit records),
  `--dry-run`, `-v`, `--progress`, `-i`/`--interactive`.
- **calibrate** (`calibrate_cmd.py`): `--output` (cache file, default cache
  filename), `--force`, `--config` (default `config.yaml`). **No `-v`.**

---

## 3. Goals & Non-Goals

### 3.1 Goals

1. A single source of truth for the shared flag vocabulary (names + help) so a
   future command cannot drift the way `predict` drifted from `train`.
2. Validated choices everywhere a flag has a fixed value set (`--progress`,
   `eval --split`, `init` tier flags), expressed as real `Enum`s rather than
   prose in help text.
3. Close the coverage gaps: `--override` on `run`; `-v` on `doctor`/`init`/
   `calibrate`; `-y` on `init`; `--dry-run` on `train`/`run`/`eval`;
   convenience `--name`/`--output-dir` on `train`/`run`.
4. Coherent `--config` semantics across the three command families (launch
   input vs. discoverable override vs. validation feature).
5. Level `predict` *down* to the lean rule: drop `--device`, `--seed`,
   `--dtype`, `--merge-adapter` in favor of overrides / auto-resolution /
   derivation. Do not level the other commands *up* to predict's verbosity.
6. A cross-command consistency test that mechanically asserts the standard
   vocabulary — the structural guard against the next `predict`-style drift.
7. The audit deliverables themselves (classification table + coverage matrix)
   as durable spec content.

### 3.2 Non-Goals (from #115)

- Migrating off Typer.
- Shell completion.
- Building a new TUI / wizard (`setup_wizard` already exists).
- Error-formatting / exit-code cleanup.
- Notebook FORM-cell ergonomics.

---

## 4. Detailed Design

### 4.1 Architecture: `cli/_options.py`

Create `src/custom_sam_peft/cli/_options.py` as the single source of truth for
the shared flag vocabulary, exposed as `Annotated[T, typer.Option(...)]` type
aliases. Each command annotates its parameters with these aliases instead of
re-declaring `typer.Option(...)` inline.

Aliases to define (at minimum):

| Alias | Underlying type | Option spec |
|---|---|---|
| `ConfigArg` | `Path` | positional `typer.Argument`, optional, the canonical launch input for `train`/`run` (§4.2) |
| `ConfigOpt` | `Path \| None` | `--config` hidden alias (train/run) / discoverable override (eval/export/predict) / validation feature (doctor) — see §4.2 |
| `VerboseOpt` | `bool` | `-v` / `--verbose`, default `False`, help "Enable DEBUG logging." |
| `OverrideOpt` | `list[str]` | `--override`, repeatable, help "Override config keys: dotted.key=value." |
| `ProgressOpt` | `Progress` (Enum) | `--progress`, default `Progress.auto`, `metavar=MODE` |
| `DryRunOpt` | `bool` | `--dry-run`, default `False` |
| `NameOpt` | `str \| None` | `--name`, convenience for `run.name` (§4.5) |
| `OutputDirOpt` | `Path \| None` | `--output-dir`, convenience for `run.output_dir` (§4.5) |

Help text and option names live with the alias; no command re-states them. A
command may still override a *default* (e.g. `eval` keeps `--config` optional
where `train` promotes it to a positional argument), but the *name* and *help*
come from `_options.py`.

`_options.py` also houses the two new `Enum`s (§4.4) and is the import target
for `merge_cli_overrides` (§4.5) and the shared discovery helper (§4.3) so the
consistency test (§4.7) and every command import from one module.

### 4.2 `--config` tiering (the rationale)

`--config` means three different things. The audit makes the three explicit and
aligns each family's surface to its meaning.

**Tier A — launch input (`train`, `run`).** The config is the *only*
description of the run that exists before a run directory or resolved
`config.yaml` snapshot exists. It is therefore required. **Decision:** promote
it to an **optional positional argument** as the canonical form (`csp run
config.yaml`), and keep `--config` as a **hidden alias** for back-compat.
Rationale: PR #244 was forced to "correct" the natural `csp run config.yaml`
form to `csp run --config config.yaml` only because the code did not support a
positional. Restoring the positional is the intuitive form; the hidden `--config`
alias keeps #244's examples and every emitted command working unchanged. The
positional is *optional at the parser* so the `run` auto-init path
(`run` writes a default config when the path is missing) and a future
config-discovery extension stay possible; absence of both positional and alias
is rejected in the body with the existing required-config error.

**Tier B — discoverable override (`eval`, `export`, `predict`).** These commands
operate on a checkpoint. The trainer writes a resolved, round-trippable
`TrainConfig` snapshot to `<run_dir>/config.yaml` (PR #248 made this a faithful
`model_dump` that reloads through `load_config`; note val-source provenance is
*no longer* written into `config.yaml` — it lives in `val_source.json` and the
tracker hparams). So `--config` here is an *optional override*: when omitted, it
is auto-discovered by walking up from the checkpoint to the nearest sibling /
ancestor `config.yaml`.

- `export` already does this via `_discover_config`.
- **Fix:** `eval` currently force-raises `--config is required` whenever a
  non-interactive invocation omits `--config`. Change `eval` to
  *discover-then-fallback* exactly like `export` **when `--checkpoint` is
  given**. Baseline eval (no `--checkpoint`) has no checkpoint to discover from,
  so it still genuinely requires `--config` — keep that raise for *that case
  only*.
- `predict` is the **exception** and is deliberately *exempt* from the shared
  discovery helper. It parses `--config` directly today (in `_resolve_config`)
  and tolerates its absence: predict resolves the base model from the adapter's
  `base_model_name_or_path` and never *requires* a full `TrainConfig` to run.
  Routing predict through the *raising* `discover_config` (§4.3) would be a
  **regression** — it would make `predict` fail on a bare adapter directory that
  was copied out of its run dir (no ancestor `config.yaml`), a case predict
  handles fine today. So predict keeps `--config` as a direct optional override
  with no tree-walk; its existing precedence (adapter `base_model_name_or_path` >
  `--config` model.name > builtin default) is unchanged.

**Lift `_discover_config` into one shared helper** (§4.3) that `eval` and
`export` call (not `predict` — see above). Do **not** invent new discovery
logic — reuse the exact tree-walk. Issue #249 will later upgrade this single
helper (and both consumers benefit).

**Tier C — validation feature (`doctor`).** `doctor --config` loads + validates
a config and reports resolved sizes/augs/normalization/losses. It is a feature,
not a dependency. **Decision:** leave it exactly as-is.

### 4.3 Shared config-discovery helper

Lift the body of `export_cmd._discover_config` verbatim into
`cli/_options.py` (or a sibling `cli/_discovery.py` imported by `_options.py`)
as:

```python
def discover_config(checkpoint: Path) -> Path:
    """Walk up from *checkpoint* to the nearest sibling/ancestor config.yaml."""
```

Same tree-walk, same `typer.BadParameter("could not auto-discover config.yaml
above {checkpoint}; pass --config", param_hint="--config")` on miss. `export`
and `eval` (checkpoint case) call it; `predict` does **not** (§4.2 — it tolerates
a missing config and must not gain a raise-on-miss). `export_cmd._discover_config`
becomes a thin re-export or is removed in favor of the shared name. This is the
seam #249 upgrades.

### 4.4 Validation / enums

Define two `Enum`s in `_options.py`:

- `class Progress(str, Enum): auto = "auto"; on = "on"; off = "off"; plain =
  "plain"`. `ProgressOpt` types every command's `--progress` against it, so all
  commands inherit validated choices + help. The command bodies pass
  `progress.value` (or `None` when `auto`) into the existing `resolve_mode(...)`
  contract unchanged.
- `class Split(str, Enum): val = "val"; test = "test"`. `eval --split` types
  against it. The body's manual `if split not in (...)` raise is removed (the
  parser now rejects bad values), and the `cast(Literal["val","test"], split)`
  becomes `split.value`. See verify-item §6.1 on whether `train` is also a valid
  split.

For `init`'s tier flags, replace the prose-only choices with validated types so
the parser rejects bad values before `run_init`:

- `--preset` → an `Enum` mirroring the schema `Preset`
  (`natural|medical|satellite|microscopy|none|custom`), `case_sensitive=False`.
- `--intensity` → an `Enum` mirroring `Intensity` (`safe|medium|aggressive`).
- `--class-imbalance` → an `Enum` mirroring `ClassImbalance`
  (`balanced|moderate|severe`).

The post-parse validation inside `run_init` (`get_args(Preset)` etc.) stays as a
defensive belt-and-suspenders check (`run_init` is also called from `run`'s
auto-init path and tests), but the CLI parser becomes the first line of defense.
Keep `.lower()` normalization where `case_sensitive=False` already applied.

### 4.5 Convenience flags + `merge_cli_overrides`

Add a helper to `_options.py`:

```python
def merge_cli_overrides(
    explicit_overrides: list[str],
    *,
    name: str | None,
    output_dir: Path | None,
) -> list[str]:
    """Append synthesized `dotted.key=value` overrides for convenience flags.

    Raises typer.BadParameter on conflict: if a convenience flag and an
    explicit --override both target the same dotted key.
    """
```

It synthesizes `run.name=<name>` and `run.output_dir=<output_dir>` and appends
them to the user's `--override` list, then the command passes the merged list
into the existing `load_config(config, overrides=...)` → `apply_overrides` path
**unchanged**. **Rule: error-on-conflict.** If the user passes both `--name foo`
and `--override run.name=bar`, raise `typer.BadParameter` — never silently pick
a precedence. Conflict detection compares the dotted key on the left of each
explicit `--override`'s first `=` against `run.name` / `run.output_dir`.

**Add `--name` and `--output-dir` to `train` and `run` only.** They are the
common-enough run-identity knobs that justify a dedicated flag. Everything else
stays override-only.

**Do not add `--device` or `--seed` as dedicated flags anywhere.** Device
pinning (`model.device`) is a rare multi-GPU power-user need; seed (`run.seed`)
is power-user repro. Both remain override-only (`--override model.device=cuda:1`,
`--override run.seed=123`).

### 4.6 `predict` cleanups (level down to the lean rule)

- **Drop `--device`.** The config-less rare multi-GPU case is served by
  `--override model.device=...` once predict honors a discovered/explicit config
  (predict resolves device from `auto` → `cuda if available else cpu` today;
  that auto behavior is retained). Removing the flag deletes the `Choice` and the
  `PredictOptions.device` plumbing's CLI surface (the field may stay internal).
- **Drop `--seed`.** Inference is near-deterministic; the rare repro need is
  `--override run.seed=...`. (This also removes the `--seed 0` vs `run.seed=42`
  inconsistency the audit flagged.) `run_predict` keeps seeding internally with a
  fixed default.
- **Drop `--dtype`; make dtype always-`auto`.** The runtime already resolves
  `auto` correctly: `predict.runner._resolve_config` maps `auto` → `bfloat16` on
  CUDA, `float32` on CPU; the train/eval runtime additionally coerces
  `bfloat16` → `float16` below compute capability 8.0 via
  `coerce_dtype_for_capability`. `float32`-forcing is exotic and not
  CLI-worthy. **Decision:** predict's `_resolve_config` currently does *not* call
  `coerce_dtype_for_capability`, so on a sub-CC-8.0 CUDA card it would request raw
  `bfloat16`. Route predict's `auto` resolution **through
  `coerce_dtype_for_capability`** (bf16 → fp16 below CC 8.0) so the lean
  always-`auto` surface is correct on every supported card, including sub-8.0
  GPUs. See verify-item §6.4 for the exact integration point.
- **Drop `--merge-adapter`; derive from PEFT method.** LoRA merges its deltas
  into the base weights (a speed win, no result change); QLoRA stays unmerged to
  avoid the 4-bit dequant memory blowup. `detect_adapter_kind` already
  distinguishes the two; derive `merge = (kind == "lora")` instead of taking a
  flag. **See verify-item §6.3** — confirm that adapter merge changes only
  speed/memory and never inference results before making it implicit.

These four removals are *breaking* for `predict`'s CLI but land in Phase 2
(predict has no positional-config change and these flags are net removals). They
realize "user-facing simplicity beats a speed-only knob."

### 4.7 Cross-command consistency test (the structural guard)

Add `tests/cli/test_flag_consistency.py` that introspects each command's
compiled Typer/Click params (via `typer.main.get_command(app)` →
`command.params`) and asserts the standard vocabulary:

- Every **training-family** command (`train`, `run`) exposes: a config input
  (positional `config` *or* `--config` alias), `--override`, `-v`/`--verbose`,
  `--progress` (typed as `Progress`), and `--dry-run`.
- Every command that takes `--progress` uses the shared `Progress` enum (no bare
  `str`).
- Every command exposes `-v`/`--verbose` (after Phase 2 adds it to
  `doctor`/`init`/`calibrate`).
- `--config`, where present, carries the help text from `ConfigOpt`/`ConfigArg`.

The test is parametrized over the command set so adding a command without the
vocabulary fails CI. This is explicitly the guard that prevents the next
`predict`-style drift.

---

## 5. Audit Deliverables

### 5.1 Flag-classification table

Every flag classified as one of: **keep** (no change) · **rework/help**
(rename-or-rework or help-clarity only) · **promote-positional** · **add to
other commands** · **drop/implicit**.

| Command | Flag | Classification | Note |
|---|---|---|---|
| train | `--config` | promote-positional | optional positional + hidden `--config` alias |
| train | `--override` | keep | becomes `OverrideOpt` alias |
| train | `--resume` | keep | optional-value patch retained |
| train | `--time-limit` | keep | command-specific |
| train | `--eval` / `--export` | keep | pipeline-composition toggles |
| train | `-v` | keep | becomes `VerboseOpt` alias |
| train | `--progress` | rework/help | typed as `Progress` enum |
| train | `--name` / `--output-dir` | add to other commands | new convenience flags (train+run) |
| train | `--dry-run` | add to other commands | new |
| run | `--config` | promote-positional | same as train |
| run | `--override` | add to other commands | **headline gap — new** |
| run | `--resume` / `--time-limit` / `--finalize` | keep | command-specific |
| run | `--visualize/--no-visualize` | keep | command-specific composition |
| run | `-v` | keep | alias |
| run | `--progress` | rework/help | `Progress` enum |
| run | `--name` / `--output-dir` / `--dry-run` | add to other commands | new |
| eval | `--config` | rework/help | discover-then-fallback (checkpoint case); required only for baseline |
| eval | `--checkpoint` | keep | omit ⇒ baseline |
| eval | `--split` | rework/help | typed as `Split` enum |
| eval | `--output` | rework/help | help states it is a results dir |
| eval | `--save-predictions` / `--visualize` / `--export` | keep | command-specific |
| eval | `-v` | keep | alias |
| eval | `--progress` | rework/help | `Progress` enum |
| eval | `--dry-run` | add to other commands | new |
| eval | `-i`/`--interactive` | keep | intentional (§5.3) |
| export | `--checkpoint` | keep | required |
| export | `--merge` | rework/help | output now always required (§5.4) |
| export | `--output` | rework/help | now always required; help states it is a dir |
| export | `--config` | rework/help | shared `discover_config` helper |
| export | `-v` | keep | alias |
| export | `--progress` | rework/help | `Progress` enum |
| init | `--template` | rework/help | typed enum (optional) / keep |
| init | `--preset` / `--intensity` / `--class-imbalance` | rework/help | typed `Enum`s |
| init | `--output` | rework/help | help states it is a config dest |
| init | `--force` / `--download-weights` | keep | command-specific |
| init | `--yes` | rework/help | add `-y` short form |
| init | `-v` | add to other commands | **new** |
| init | `-i`/`--interactive` | keep | intentional (§5.3) |
| doctor | `--weights-path` / `--json` | keep | command-specific |
| doctor | `--config` | keep | validation feature (Tier C) |
| doctor | `-v` | add to other commands | **new** (orthogonal to `--json`) |
| predict | `--images` / `--prompts` / `--output` | keep | required core |
| predict | `--checkpoint` | keep | callback-validated |
| predict | `--merge-adapter` | drop/implicit | derived from PEFT method (§4.6) |
| predict | `--config` | keep | direct optional override; tolerates absence, no tree-walk (§4.2) |
| predict | `--score-threshold` / `--top-k` / `--batch-size` | keep | callback-validated |
| predict | `--save-masks` / `--visualize` | keep | command-specific |
| predict | `--device` | drop/implicit | → `--override model.device` |
| predict | `--dtype` | drop/implicit | always-auto (§4.6, §6.4) |
| predict | `--seed` | drop/implicit | → `--override run.seed` |
| predict | `--dry-run` | keep | the template for other commands |
| predict | `-v` / `--progress` | keep | aliases |
| predict | `-i`/`--interactive` | keep | intentional (§5.3) |
| calibrate | `--output` / `--force` / `--config` | keep | command-specific |
| calibrate | `-v` | add to other commands | **new** |

### 5.2 Coverage matrix (common task → flag path)

"must edit YAML" means there is no flag and the value lives only in the config
file. After this work:

| Task | Flag path (after) | Before |
|---|---|---|
| Train from a config | `csp train config.yaml` | `csp train --config config.yaml` |
| Full pipeline (train+eval+export) | `csp run config.yaml` | `csp run --config config.yaml` |
| Override any config key | `--override a.b=c` (all of train/run/eval via load_config) | train only; **run had no `--override`** |
| Name a run | `--name my-run` (train/run) | `--override run.name=...` only |
| Pick output dir | `--output-dir runs/exp1` (train/run) | `--override run.output_dir=...` only |
| Resume latest | `--resume` (train/run) | same |
| Wall-clock budget | `--time-limit 2h30m` (train/run) | same |
| Pin a GPU | `--override model.device=cuda:1` | `predict --device` / else must edit YAML |
| Set seed | `--override run.seed=123` | `predict --seed` / else must edit YAML |
| Force fp32 inference | must edit `model.dtype` in YAML | `predict --dtype float32` |
| Evaluate a checkpoint | `csp eval --checkpoint PATH` (config auto-discovered) | `--config` was forced-required |
| Baseline (zero-shot) eval | `csp eval --config cfg.yaml` | same |
| Choose eval split | `--split test` (validated) | unvalidated string |
| Preview without running | `--dry-run` (train/run/eval/predict) | predict only |
| Export adapter | `csp export --checkpoint PATH --output DIR` | `--output` conditionally required |
| Export merged weights | `csp export --checkpoint PATH --merge --output DIR` | `--output` optional when `--merge` |
| Predict on images | `csp predict --images D --prompts "a,b" --output O` | same |
| Apply adapter at predict | `--checkpoint PATH` (merge derived from kind) | `--merge-adapter` flag |
| Verbose logs (any command) | `-v` (all eight) | missing on doctor/init/calibrate |
| Skip init prompt | `csp init -y` | `--yes` (no short form) |
| Validate a config | `csp doctor --config cfg.yaml` | same |
| Probe VRAM | `csp calibrate --config cfg.yaml` | same |

### 5.3 Document-as-intentional (deliberate non-changes)

- `-i`/`--interactive` exists on `eval`/`init`/`predict` but **not** `train`/
  `run` — by design. Interactive flows *build* a command/config; `train`/`run`
  *consume* the built config. An interactive `train` would overlap `init
  --interactive` / `setup_wizard`.
- Per-command pipeline toggles (`--eval`/`--export` on `train`, `--export` on
  `eval`, `--finalize`/`--visualize` on `run`) are inherently command-specific
  composition flags. There is no uniformity to extract; they stay as-is.

### 5.4 Help-clarity-only (no rename)

`--output` denotes different nouns across commands: a run/results *directory*
(`train` implicit, `eval`, `export`), a config-file *destination* (`init`), and
a cache *file* (`calibrate`). **Do not unify the name.** Each command's
`--output` help text must state (a) what kind of path it is and (b) whether it
is created. This is the only change to `--output`.

### 5.5 `export --merge` rework

Today `--output` is optional and, when omitted with `--merge`, `run_export`
defaults the merged weights to `<checkpoint.parent>/merged` (`run_dir = checkpoint.parent`),
while the non-merge path raises `ValueError("output is required when not
merging")`. This conditional-required shape is confusing.

**Decision:** make `--output` **always required** on `export`. `--merge` then
writes merged full-model weights to (a derived path under) `--output`; the
non-merge path copies the adapter to `--output`. Drop the conditional-required
ValueError. **See verify-item §6.2** for the exact landing path of merged
weights relative to `--output` (e.g. `--output` itself vs. `--output/merged`) —
confirm against `save_merged` before finalizing, and keep the emitted path in
the success message accurate.

---

## 6. Verify-During-Implementation Items

1. **`eval --split train`** — confirm whether the data pipeline supports a
   `train` split for eval. If yes, `Split` = `{val, test, train}`; if not,
   `Split` stays `{val, test}`. Default `{val, test}` unless proven otherwise.
2. **`export --merge` landing path** — confirm the exact on-disk path
   `save_merged` writes to relative to the now-always-required `--output`, and
   make the CLI success message print that path.
3. **Adapter merge is result-neutral** — confirm LoRA merge changes only
   speed/memory and never inference outputs (gates dropping
   `predict --merge-adapter`). QLoRA-unmerged is the safe default by derivation.
4. **predict dtype `auto` on every supported GPU** — route `_resolve_config`'s
   `auto` path through `coerce_dtype_for_capability` (bf16 → fp16 below CC 8.0)
   so it is correct on every supported card, including sub-8.0 GPUs (decided).
   Confirm the exact integration point in `_resolve_config` / the predict runtime
   and that the coerced dtype flows into model load. This is now an
   implementation task, not an open question.
5. **Emitted-command propagation** — find every place `setup_wizard` /
   `_interactive` emit copy-paste commands and update them for the positional
   form. Known sites: `_interactive.py` line ~250 (`custom-sam-peft
   {train|run|eval} --config {output}` via `_LAUNCH_VERB`), ~344 / ~362 (eval
   commands), ~498/~511 (predict command assembly). The hidden `--config` alias
   means leaving these unchanged is *correct* too — but Phase 3 should restore
   the natural positional form in user-facing examples.
6. **README quickstart + command table** — restore the natural positional form
   (`csp run config.yaml`, `csp train config.yaml`) in the quickstart (lines
   ~59–64) and the command table (lines ~88–95). The hidden `--config` alias
   keeps #244's examples working; the table should show the positional as
   canonical.

---

## 7. Phasing Intent + Interface Contracts

The planner produces the detailed phased plan; this section fixes the phase
boundaries and the interface contract each phase exposes.

### Phase 1 — Foundation (pure refactor, ZERO behavior change)

Create `_options.py` (the `Annotated` aliases + `Progress` and `Split` enums),
`merge_cli_overrides` (with conflict detection), and the shared `discover_config`
helper. Refactor **all eight** commands (including `predict`) to consume the
shared aliases without changing any current behavior — the `--progress` enum
must accept the same string values it does today, `discover_config` must be
byte-for-byte the current tree-walk, and no flag is added or removed yet. Add the
consistency test (§4.7) asserting only the *currently true* vocabulary, then
tighten it in later phases.

**Interface contract exposed:** `cli/_options.py` exports the alias names
(`ConfigArg`, `ConfigOpt`, `VerboseOpt`, `OverrideOpt`, `ProgressOpt`,
`DryRunOpt`, `NameOpt`, `OutputDirOpt`), the `Progress` and `Split` enums, the
`merge_cli_overrides(explicit_overrides, *, name, output_dir) -> list[str]`
signature, and `discover_config(checkpoint: Path) -> Path`. Later phases import
exclusively from this module.

### Phase 2 — Additive standardization

Consumes Phase 1's seam. Add: `--override` to `run`; `--name`/`--output-dir` to
`train`/`run` (via `merge_cli_overrides`); `-v` to `doctor`/`init`/`calibrate`;
`-y` to `init`; `--dry-run` to `train`/`run`/`eval`; the `Split` enum on `eval`;
eval discover-then-fallback (checkpoint case); the `init` tier-flag enums. Strip
`predict`'s `--device`/`--seed`/`--dtype`/`--merge-adapter` (auto / derived /
override replace them; gated on verify-items §6.3, §6.4). Tighten the
consistency test to assert the newly-standard vocabulary.

**Interface contract exposed:** all non-breaking surface additions are present;
`train`/`run` still take `--config` (no positional yet); `predict`'s lean surface
is final. Phase 3 only changes config *form* (positional) and `export --merge`.

### Phase 3 — Breaking cleanup + propagation

Consumes Phase 2's surface. Promote `--config` to an optional positional on
`train`/`run` with the hidden `--config` alias. Rework `export --merge`
(`--output` always required; drop the conditional ValueError; verify landing
path §6.2). Update `setup_wizard`/`_interactive` emitted commands and the README
quickstart + command table to the positional form. Finalize the audit doc's
"after" state (this spec already encodes it).

**Interface contract exposed:** the canonical user-facing form is positional;
the hidden alias preserves back-compat. No later phase.

---

## 8. Testing Strategy

- **Consistency test** (§4.7) — the structural guard; parametrized over the
  command set; tightened phase by phase. CPU-only, parser-introspection,
  no model load.
- **Per-flag unit tests** via Typer's `CliRunner`: `run --override a.b=c` reaches
  `apply_overrides`; `--name`/`--output-dir` synthesize the right override and
  `--name foo --override run.name=bar` raises `BadParameter`; `eval --split bad`
  is rejected by the parser; `eval --checkpoint PATH` (no `--config`) discovers
  the sibling `config.yaml`; baseline `eval` (no checkpoint, no config) still
  raises; `init --preset bogus` rejected; `-v`/`-y` present where added;
  positional `train config.yaml` and `train --config config.yaml` both parse.
- **predict-removal tests:** `predict --device cuda` / `--dtype float32` /
  `--seed 1` / `--merge-adapter` now error as unknown options (or are accepted as
  no-ops only if a deprecation shim is chosen — default is hard removal); a LoRA
  checkpoint merges and a QLoRA checkpoint does not, by derivation.
- **export test:** `--output` required in both `--merge` and non-merge modes;
  merged-weights landing path matches §6.2; success message prints it.
- **Regression:** run the full suite (a required-field/positional-arg change can
  ripple beyond the named files); grep every constructor of `PredictOptions`
  after dropping its CLI-surfaced fields.
- Tests live under `tests/cli/`; use `-o "addopts="` to bypass the global
  coverage gate when running the CLI subset in the inner loop. These tests are
  CPU-only and must not pull in GPU fixtures.

---

## 9. Deferred to Follow-Up Issues (out of scope)

The implementation may file these via `gh issue create`:

- `--profile` — a net-new throughput probe; pairs with the cost-estimator (#109).
- `doctor --check <name>` — run a single diagnostic check.
- `--quiet` / granular log levels beyond `-v`.

---

## 10. Relationship to #249

Issue #249 will make checkpoints self-describing (embed the resolved config
inside the checkpoint directory, replacing the tree-walk). This audit
deliberately reuses the **current** tree-walk discovery via one shared
`discover_config` helper (§4.3), so #249 later upgrades that single helper and
both consumers (`eval`, `export`) benefit without further changes (`predict` is
exempt — §4.2). This is why §4.3 forbids inventing new discovery logic.
