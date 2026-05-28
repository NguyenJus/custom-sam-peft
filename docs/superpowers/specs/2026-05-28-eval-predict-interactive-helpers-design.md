# Per-command interactive helpers for `eval` / `predict` + PEFT-from-checkpoint

**Issue:** [#172 — Setup wizard: eval/predict ask training-only prompts that the checkpoint already implies](https://github.com/NguyenJus/custom-sam-peft/issues/172).
**Release:** pre-1.0 minor bump (new `eval -i` / `predict -i` features + one breaking wizard change → MINOR).
**Status:** locked design, single PR, no back-compat shims.
**Sibling spec:** [`2026-05-26-interactive-setup-wizard-design.md`](2026-05-26-interactive-setup-wizard-design.md) (the `init` wizard this work extends and reshapes). This spec reuses that one's `WizardStep`/`Ctx`/prompt-primitive vocabulary and its CPU-first testing conventions; §12 maps the relationship explicitly.

`csp init --interactive` (shipped in commit `84bc83f`) asks the full training prompt battery for every run mode. Issue #172 observes that for `eval` — and a not-yet-existing `predict` flow — most of those prompts are inert or already implied by the trained adapter checkpoint. The issue proposed three init-centric moves (a new `predict` run mode, eval-config reuse, PEFT-from-checkpoint). Brainstorming **pivoted the architecture**: rather than piling more run-modes onto `init`, this work splits the interactive surface into **per-command helpers** — `init -i` stays a training-config generator, and `eval`/`predict` each grow their own `-i` flag backed by a new shared infra module. The PEFT-from-checkpoint goal lands as a centralized adapter-introspection seam (§7) so eval, predict, and the wizard all infer the method from the checkpoint instead of `cfg.peft.method`.

This is a single cohesive PR with four bundled, interdependent workstreams:

1. **Shrink `init -i`** from `train|run|eval` to `train|run` and remove the now-dead eval-only prompt gating (§4).
2. **`csp eval --interactive`** — a reuse-vs-baseline helper that either prints a runnable `csp eval` command against an existing config + adapter, or emits a baseline (zero-shot) eval config (§5), backed by eval-runner changes that make `--checkpoint` optional and infer PEFT from the checkpoint (§6).
3. **`csp predict --interactive`** — a runnable-command builder that collects adapter, channels, and common knobs, writing a thin `--config` only when channels/model differ from defaults (§8).
4. **Centralize the PEFT/adapter introspection seam** (§7) into `peft_adapters/` so the three current sentinel-file duplications collapse onto one canonical discovery function, and `eval`/`predict`/wizard adapter-peeking share it.

They are bundled because (a) `eval -i` reuse and `predict -i` both need the same adapter-peek helpers (§7) and the same extracted prompt primitives + shared steps (§3); (b) the eval-runner PEFT-from-checkpoint change (§6) depends on the centralized seam (§7); and (c) shrinking `init -i` (§4) and adding the new helpers both edit the wizard module surface, so a single PR avoids two rounds of conflict in `setup_wizard.py`.

---

## §1 Scope & non-goals

### In scope

| File | Command(s) | Change |
|------|-----------|--------|
| `src/custom_sam_peft/cli/_interactive.py` | shared | **New module.** Houses the prompt primitives (`ask_text`, `ask_choice`, `ask_confirm` — moved verbatim from `setup_wizard.py`), the `WizardStep` / `Ctx` / `_deep_merge` / `run_wizard` driver vocabulary, reusable steps (`dataset_source`, `validation`, `model_weights`), the adapter-peek helpers (delegating to §7's seam), the TTY guard, and emit/launch-command helpers. See §3 for the exact symbol split + import graph. |
| `src/custom_sam_peft/cli/setup_wizard.py` | `init -i` | `RunMode` literal `train\|run\|eval` → `train\|run` (§4). Import primitives + shared steps + driver from `_interactive` instead of defining them; keep the init-specific train/run step list, `render`, and `generate_config`. Drop the eval branch of the `validation` no-val note, the `class_imbalance` eval-exclusion `when`, and the `epochs` `when=run_mode != "eval"` gate (all become unconditional / removed since only train/run remain). |
| `src/custom_sam_peft/cli/init_cmd.py` | `init` | No behavioral change to the flag-driven path. `setup_wizard` still imports `UNIFIED_TEMPLATE` + `_build_loss_overrides_block` from here; keep those exports. (Import-graph note: the `init -i` branch lazy-imports `setup_wizard`; `setup_wizard` imports `_interactive`; `_interactive` must NOT import `init_cmd` at module scope — see §3.) |
| `src/custom_sam_peft/cli/eval_cmd.py` | `eval`, `eval -i` | `--checkpoint` becomes **optional** (was required `...` → `None` default, §6). Add `--interactive` / `-i` flag, TTY-guarded, dispatching to the new `eval -i` helper (§5). |
| `src/custom_sam_peft/eval/runner.py` | `eval` | `run_eval`: `checkpoint=None and artifacts=None` → baseline (no adapter load) path; output-dir fallback to `cfg.run.output_dir` then cwd; replace `make_peft_method(...).load_from_disk(...)` + `_load_channel_adapter(...)` with `train.checkpoint.load_adapter(wrapper, ckpt)` (sentinel dispatch); advisory `WARNING` when `cfg.peft.method` disagrees with the detected kind. (§6) |
| `src/custom_sam_peft/cli/predict_cmd.py` | `predict`, `predict -i` | Add `--interactive` / `-i` flag, TTY-guarded, dispatching to the new `predict -i` helper (§8). No change to the existing flag surface. |
| `src/custom_sam_peft/peft_adapters/__init__.py` | seam | **New module-level** `discover_method_from_checkpoint(dir) -> "lora" \| "qlora"` (sentinel-file convention). **Relocate** `read_adapter_base_model_name` here from `predict/adapter_load.py`. (§7) |
| `src/custom_sam_peft/predict/adapter_load.py` | `predict` | `detect_adapter_kind` and `read_adapter_base_model_name` become thin delegators to the §7 seam (public names preserved; `detect_adapter_kind` keeps raising `typer.BadParameter` for the missing-`adapter_config.json` case; lazy-import discipline preserved). |
| `src/custom_sam_peft/train/checkpoint.py` | train/eval | `load_adapter`'s inline `(path / _QLORA_META_FILENAME).exists()` sentinel check → call the §7 canonical discovery function. No behavior change. |
| Tests (see §10) | all | New `_interactive` / `eval -i` / `predict -i` / seam tests; retarget `tests/unit/cli/test_setup_wizard.py` to the 2-mode `RunMode`; eval-runner baseline + PEFT-inference tests. |

### Out of scope

- **No new init run-modes.** This reverses issue #172's literal proposal #1; the pivot to per-command helpers is the agreed design.
- **No #164 / #165 work.** Both are already merged/closed: the `data.limit` prompt (#164) and the reworded class-imbalance prompt (#165) are present in `setup_wizard.py` today and carry over unchanged into the train/run flow.
- **No predict config beyond `model.name` + `data.channels` + `data.channel_semantics`.** Those are the only fields `predict/runner.py::_resolve_config` consumes from `--config`; the `predict -i` helper writes nothing else and writes nothing at all for the common RGB + default-model case (§8).
- **No change to the `csp predict` CLI flag surface.** `top_k` / `device` / `dtype` / `batch_size` / `seed` are not prompted (they stay at CLI defaults; the helper documents they are addable as flags — §8). The flag definitions in `predict_cmd.py` are untouched apart from the additive `-i` flag.
- **No removal of `EvalArtifacts.peft_method`.** Grep confirms it is still read (`eval/runner.py:97`, set by `trainer.py:462`, asserted in three test files); it stays. See §7 for the analysis.
- **No deprecation / migration shim for the dropped `init -i` eval mode.** Pre-1.0; shipped only in `84bc83f`; nothing depends on it (§11).
- **No GPU tests.** Every case here is CPU-testable (§10); GPU is reserved for real-only failure modes per project convention.

---

## §2 Architectural approach — per-command helpers, not init run-modes

Issue #172 framed the work as more `init` run-modes (`predict` added; `eval` reusing a config). Brainstorming rejected that: `init`'s job is to **generate a training config**, and bolting eval/predict onto its `RunMode` selector forces a single prompt list to serve three semantically different jobs (train a config, point eval at an existing config + adapter, build a predict command). The agreed architecture is **one interactive helper per command**, sharing infrastructure:

```text
        ┌───────────────────────── shared infra ─────────────────────────┐
        │  src/custom_sam_peft/cli/_interactive.py                        │
        │  • prompt primitives: ask_text / ask_choice / ask_confirm       │
        │  • WizardStep / Ctx / _deep_merge / run_wizard driver           │
        │  • reusable steps: dataset_source, validation, model_weights    │
        │  • adapter-peek: peek_adapter(dir) → (kind, base_model_name)    │
        │      (delegates to peft_adapters seam, §7)                      │
        │  • require_tty(); _launch_command()/_header() emit helpers      │
        └────────────────────────────────────────────────────────────────┘
              ▲                    ▲                         ▲
              │                    │                         │
   setup_wizard.py          eval_cmd.py + helper      predict_cmd.py + helper
   (init -i, train|run)     (eval -i: reuse|baseline) (predict -i: build cmd)
```

Each command's `-i` branch:

1. **Guards TTY first** (before any prompt) via the shared `require_tty()`, mirroring `init`'s existing guard (`init_cmd.py:221`): `not sys.stdin.isatty()` → `typer.BadParameter`.
2. **Collects answers** via the shared primitives and (where applicable) the shared steps + driver.
3. **Emits / prints** its command-specific output: `init -i` writes a validated training config and prints `csp train|run`; `eval -i` either prints a runnable `csp eval` command (reuse path, writes nothing) or writes a validated baseline config and prints `csp eval --split val` (baseline path); `predict -i` prints a runnable `csp predict` command and writes a thin `--config` only when needed (§8).

The three helpers differ in **what they emit**, not in **how they prompt** — so the prompt machinery, the dataset/validation/model steps, the adapter-peek, and the TTY guard all live once in `_interactive.py`. The PEFT-from-checkpoint goal becomes a property of the centralized adapter seam (§7): wherever a checkpoint dir is introspected (eval load, predict load, the wizards' adapter-peek), the method is discovered from the sentinel-file convention, never from `cfg.peft.method`.

### Import graph (no cycles)

- `_interactive.py` imports only: `typer`, stdlib, `config.loader.load_config`, `config.schema` types, and — **lazily, inside the adapter-peek function body** — the `peft_adapters` seam (§7). It MUST NOT import `init_cmd`, `setup_wizard`, `eval_cmd`, or `predict_cmd` at module scope (those import it, not vice-versa).
- `setup_wizard.py` imports `_interactive` (primitives, driver, shared steps) at module scope, and continues to import `UNIFIED_TEMPLATE` + `_build_loss_overrides_block` from `init_cmd` at module scope (today's arrangement; `init_cmd` only imports `setup_wizard` lazily inside `init()`, so this stays acyclic).
- `eval_cmd.py` / `predict_cmd.py` import their helper functions from `_interactive` (or a thin per-command helper module — implementer's choice, but the spec assumes the helpers live in `_interactive.py` to keep the seam in one place). The CLI command functions lazy-import the helper inside the `if interactive:` branch, mirroring `init`'s lazy `from custom_sam_peft.cli import setup_wizard`.

## §3 `_interactive.py` — the shared module

The split is a refactor-and-extend of today's `setup_wizard.py`: symbols that are NOT init-specific move verbatim into `_interactive.py`; init-specific code stays in `setup_wizard.py` and imports from `_interactive`. No prompt wording or behavior changes for the moved symbols.

### Symbols that MOVE to `_interactive.py` (verbatim unless noted)

| Symbol | Source today | Notes |
|--------|--------------|-------|
| `ask_text` | `setup_wizard.py:57` | verbatim. |
| `ask_choice` | `setup_wizard.py:77` | verbatim. |
| `ask_confirm` | `setup_wizard.py:92` | verbatim. |
| `WizardStep` | `setup_wizard.py:41` | verbatim dataclass. |
| `Ctx` | `setup_wizard.py:32` | verbatim, but `RunMode` becomes shared (below). The `eval`-baseline path constructs `Ctx` with `run_mode="eval"` only as an internal driver hint; see §5.2. |
| `_deep_merge` | `setup_wizard.py:48` | verbatim. |
| `run_wizard` | `setup_wizard.py:444` | verbatim driver (iterate a passed-in `STEPS` list). |
| `_ask_dataset_source` → `dataset_source` step builder | `setup_wizard.py:285` | the COCO/HF format + path(s) step. Reused by `init -i` (train/run) and `eval -i` baseline. Verbatim ask-body. |
| `_ask_validation` → `validation` step builder | `setup_wizard.py:295` | explicit / auto-split / none, COCO + HF aware. Reused by `init -i` and `eval -i` baseline. See §3.1 on the no-val note. |
| `_ask_model_weights` → `model_weights` step builder | `setup_wizard.py:406` | checkpoint path / blank-with-glob. Reused by `init -i` and `eval -i` baseline. Verbatim. |
| `validate` | `setup_wizard.py:459` | round-trips rendered bytes through `load_config` via a temp file. Reused by any helper that writes a config. |
| `_launch_command` / `_header` / `_LAUNCH_VERB` | `setup_wizard.py:456`, `470`, `474` | generalized: `_LAUNCH_VERB` already maps `train`/`run`/`eval`; add `predict` is **not** needed because predict prints a hand-assembled command (§8). `_header`'s "Generated by" line takes the generating command as a parameter so eval-baseline can say `csp eval --interactive`. |

### Shared `RunMode` and the new TTY guard

```python
# _interactive.py
RunMode = Literal["train", "run", "eval"]  # superset; init -i narrows to train|run (§4)

def require_tty() -> None:
    """Raise typer.BadParameter if stdin is not a TTY. Call BEFORE any prompt."""
    import sys
    if not sys.stdin.isatty():
        raise typer.BadParameter(
            "interactive mode needs a TTY; use the flag-driven command instead"
        )
```

`RunMode` keeps `eval` because `_interactive`'s shared `validation` step and `_launch_command` still reason about an eval target (the `eval -i` baseline path drives the shared steps with `run_mode="eval"`). It is `setup_wizard`'s *step list* and *`init` selector* that narrow to `train|run` (§4) — the shared type stays a superset so eval-baseline can reuse the validation no-val note logic.

### Adapter-peek helper (used by `eval -i` reuse + `predict -i`)

```python
# _interactive.py
def peek_adapter(checkpoint_dir: Path) -> tuple[str, str | None]:
    """Return (pretty_method_name, base_model_name) for an adapter checkpoint dir.

    Lazy-imports the peft_adapters seam (§7) so the import stays off any hot path.
    method := discover_method_from_checkpoint(dir)  → "lora" | "qlora"
    pretty := method_pretty_name(method)            → "LoRA" | "QLoRA"
    base   := read_adapter_base_model_name(dir)     → str | None
    """
    from custom_sam_peft.peft_adapters import (
        discover_method_from_checkpoint,
        method_pretty_name,
        read_adapter_base_model_name,
    )
    method = discover_method_from_checkpoint(checkpoint_dir)
    return method_pretty_name(method), read_adapter_base_model_name(checkpoint_dir)
```

`peek_adapter` is the single place the wizards inspect a checkpoint dir; it never opens the model. The caller validates dir existence + `adapter_config.json` presence *before* calling (the validators below), so `peek_adapter` operates on a known-good dir. `method_pretty_name` already exists (`peft_adapters/__init__.py:149`).

### New validators (small, in `_interactive.py`)

```python
def validate_checkpoint_dir(s: str) -> str | None:
    """ask_text validator: re-ask unless s is a dir containing adapter_config.json."""
    p = Path(s)
    if p.is_dir() and (p / "adapter_config.json").is_file():
        return None
    return f"{s} is not an adapter checkpoint dir (missing adapter_config.json)"

def validate_config_with_eval_split(s: str) -> str | None:
    """ask_text validator for eval-reuse: re-ask unless s load_config's AND has a
    val/val_split/test (something to score). Warns-and-re-asks otherwise."""
    # load_config(Path(s)); on ConfigError return the message.
    # Then require: cfg.data.val is not None OR cfg.data.val_split is not None
    #   OR (cfg.data.format == "hf" and cfg.data.hf and cfg.data.hf.split_val is not None)
    #   OR cfg.data.test is not None
    # else return "config has no val/test split to evaluate; pick a config with one".
```

The eval-split predicate mirrors `eval/runner.py`'s `--split val` gate (`runner.py:107-114`) plus the `--split test` requirement (`runner.py:115`), so a config that passes the validator can run `csp eval --split val` or `--split test`.

### What STAYS in `setup_wizard.py`

`UNIFIED_TEMPLATE`/`_build_loss_overrides_block` import (from `init_cmd`); the init-specific step ask-functions (`_ask_run_mode`, `_ask_run_name`, `_ask_domain`, `_ask_class_imbalance`, `_ask_peft_sizing`, `_ask_epochs`, `infer_class_imbalance` + ratio helpers); the init `STEPS` list (now `train|run` only, §4); `render` + its block helpers (`_model_block`, `_dataset_block`, `_validation_block`, `_qlora_block`, `_aug_overrides_block`); `emit`; `generate_config`. `setup_wizard` re-imports the moved primitives/steps/driver from `_interactive`. (Tests that currently do `sw.ask_text`, `sw.WizardStep`, etc. — see §10 — either update to `_interactive` or rely on `setup_wizard` re-exporting them; the spec recommends re-exporting the moved names from `setup_wizard` so the existing test import surface stays stable, then adding direct `_interactive` tests for the new symbols.)

### §3.1 The `validation` step's no-val note across helpers

Today `_ask_validation` prints an eval/run discouragement note when mode is `eval`/`run` and the user picks "none" (`setup_wizard.py:299-303`). In the shared step this stays keyed on `ctx.run_mode in {"eval", "run"}`. For `init -i` (now `train|run` only) the `run` branch still fires; the `eval` branch fires only when `eval -i` baseline drives the shared step with `run_mode="eval"`. The wording is unchanged.

## §4 `csp init --interactive` shrinks to train|run

`init -i` stops offering `eval`. The exact edits in `setup_wizard.py`:

1. **`RunMode` selector narrows.** `_ask_run_mode` (`setup_wizard.py:275`) currently offers `["train", "run", "eval"]`; change to `["train", "run"]` (default `"train"`). The shared `RunMode` type in `_interactive.py` stays the `train|run|eval` superset (§3) — only `init`'s offered choices and the steps below narrow. `Ctx.run_mode` default stays `"train"`.
2. **`epochs` step un-gates.** Today `WizardStep("epochs", _ask_epochs, when=lambda ctx: ctx.run_mode != "eval")` (`setup_wizard.py:439`). With `eval` gone from `init`, `run_mode` is always `train` or `run`, so the `when` is always true — **remove the `when`** (epochs always asked). The "silently set to 1 in eval mode" behavior in `render` (`setup_wizard.py:250`, `epochs = ... .get("epochs", 1)`) becomes unreachable from `init -i` but stays harmless as a default; keep it (it also serves the eval-baseline render, §5.3).
3. **`class_imbalance` step un-gates.** Today `when=lambda ctx: ctx.run_mode in {"train", "run"}` (`setup_wizard.py:436`). Both remaining modes satisfy it, so the `when` is now always true — **remove the eval-exclusion `when`** (the step's internal "could not auto-detect → balanced" fallback already handles non-COCO/HF, so it is safe to always run for train/run).
4. **`validation` no-val note.** The shared step keeps `ctx.run_mode in {"eval", "run"}` (§3.1). For `init -i` only the `run` arm is reachable, so the note still fires for `run` + "none". No edit beyond using the shared step.

Everything else in `init -i`'s train/run flow is **unchanged**, including:

- the already-merged `data.limit` prompt (#164) and the reworded class-imbalance prompt (#165) — both present today and untouched;
- the VRAM auto-size step (`_ask_peft_sizing`), domain/intensity step, model-weights step;
- `render` + the unified `config_full.yaml` template;
- `generate_config`, `validate`, `emit`, and the `_maybe_download_weights` hand-off in `init_cmd.init()`;
- the pre-flight TTY + output-exists checks in `init_cmd.init()` (now `init_cmd` may call `_interactive.require_tty()` instead of its inline check — cosmetic; the message is unchanged in spirit. Implementer may keep the existing inline check to minimize churn).

The init `STEPS` list (`setup_wizard.py:427`) keeps all nine steps; only the two `when` gates change (removed) and the `run_mode` choice list shrinks.

## §5 `csp eval --interactive`

New `--interactive` / `-i` flag on `eval_cmd.py`. When set:

1. `configure_logging(verbose)` as today.
2. `require_tty()` (§3) — **before any prompt**.
3. Run the helper `run_eval_interactive(...)` (in `_interactive.py`), which branches on the first prompt and returns nothing (it prints / writes directly). The `eval -i` branch returns before the normal `run_eval` invocation — interactive mode never *runs* an eval; it prints a command (reuse) or writes a config (baseline).

### §5.1 First prompt — reuse vs baseline

```text
Evaluate a trained adapter, or baseline zero-shot SAM?  [reuse/baseline]
```

`ask_choice("Evaluate a trained adapter, or baseline zero-shot SAM?", ["reuse", "baseline"], default="reuse")`.

### §5.2 Reuse path (trained adapter) — prints a command, writes NOTHING

Prompt flow (ordered). "Required" = always asked.

| # | Step | Prompt | Target | Required / Defaulted |
|---|------|--------|--------|----------------------|
| R1 | `config_path` | `Path to your existing training config (.yaml)?` | local var | **Required.** Validated with `validate_config_with_eval_split` (§3): must `load_config` AND carry a val/val_split/test (re-ask with a warning otherwise). |
| R2 | `checkpoint_dir` | `Path to the adapter checkpoint directory?` | local var | **Required.** Validated with `validate_checkpoint_dir` (§3): dir containing `adapter_config.json` (re-ask otherwise). Then `peek_adapter(dir)` (§3) prints `detected adapter: <LoRA\|QLoRA>, base model: <name or "(unspecified)">`. |
| R3 | `split` | `Which split? [val/test]` | local var | Defaulted `val`. If the chosen config lacks the chosen split's source, re-ask / warn (R1's validator already guaranteed at least one exists). |

Output: **write no file.** Print the runnable command:

```text
csp eval --config <config_path> --checkpoint <checkpoint_dir> --split <split>
```

The peek (R2) is purely informational — the method is inferred at run time by the eval-runner's sentinel dispatch (§6), so the printed command carries no `--peft-method` and the user need not know lora vs qlora.

### §5.3 Baseline path (no adapter) — writes a config, prints a `csp eval` command

Framing: a baseline eval answers "I have my dataset (train + val) and want zero-shot SAM numbers on the val split *before* training." So `data.train` is legitimately present (it is a required schema field — `DataConfig.train` has no default, `schema.py:381`) and there is no need for a dummy/stub train block: the user really does have a train set; they just are not training yet.

The baseline path drives the **shared steps** (§3) with `run_mode="eval"` and a minimal step list:

| # | Step | Prompt | Target | Required / Defaulted |
|---|------|--------|--------|----------------------|
| B1 | `dataset_source` | Local COCO or HuggingFace? Then path(s). | COCO → `data.format="coco"` + `data.train.{annotations,images}`; HF → `data.format="hf"` + `data.hf.name` | **Required** |
| B2 | `validation` | Explicit val / auto-split fraction / none. | explicit → `data.val.*` (COCO) or `data.hf.split_val` (HF); auto-split → `data.val_split.fraction`; none → omit | Defaulted (auto-split 0.1). `run_mode="eval"` → the no-val note (§3.1) fires on "none". |
| B3 | `model_weights` | Checkpoint path / blank → `models/sam3.1` + glob. | `model.local_dir` / `model.checkpoint_file` | Defaulted (blank → schema default + glob) |

**Skipped entirely** vs `init -i`: `run_mode` (fixed to eval internally), `run_name` (eval needs a `run.name` — emit a fixed default `baseline-eval`, not prompted), `domain` (augmentation/loss presets are inert for eval — `pipeline="eval"` builds with no augmentation and eval computes metrics not loss), `peft_sizing` (no adapter to size; `peft.method` left at schema default — see below), `class_imbalance` (loss-only), `epochs` (training-only).

Emit a config by rendering the existing unified template (`render(answers, run_mode="eval")`, reusing `setup_wizard.render`):

- dataset + validation + model blocks from B1–B3;
- `run.name: baseline-eval` (placeholder default);
- `eval` section at schema defaults (the template's `eval:` block);
- training-only knobs left at schema defaults and **inert for eval**: `peft.method` renders the template's `$peft_method` — set it to the schema/template default `lora` (it is never read for dispatch on the baseline path since no checkpoint is loaded; `run_eval` skips the adapter entirely — §6); `epochs` renders the `render` default `1`; augmentations/loss render their `natural`/`balanced` defaults.
- **Validate** the rendered bytes via `validate(...)` (§3) before writing.
- Respect `--output` / `--force` (same `emit` refusal as `init -i`).

Output: write the config, then print:

```text
csp eval --config <output> --split <val|test>
```

with **no `--checkpoint`** (baseline = no adapter). The split in the printed command matches B2: if the user picked explicit/auto-split it is `--split val`; the spec keeps it simple and always prints `--split val` for the baseline (B2 always yields a val source or the no-val note warned the user). `--split test` is not offered on the baseline path (baseline is a pre-training sanity check on val); a user wanting test can edit the printed command.

> **Reuse of `render`:** the baseline path renders the *same* `config_full.yaml` the train/run wizard uses, so the emitted file is comprehensive (every section, alternatives commented) — consistent with the prior spec's "config meant to be tweaked" rationale. The training-only sections are present but harmless for eval.

## §6 eval-runner changes — optional checkpoint + PEFT-from-checkpoint

Two changes to `eval/runner.py` plus one to `eval_cmd.py`. The `run`/artifacts path is **untouched** (the trained method always matches its saved checkpoint, so sentinel detection agrees with `artifacts.peft_method`).

### §6.1 `eval_cmd.py` — `--checkpoint` becomes optional (additive)

Today (`eval_cmd.py:23`):

```python
checkpoint: Path = typer.Option(..., "--checkpoint", help="Path to adapter checkpoint."),
```

Change to optional:

```python
checkpoint: Path | None = typer.Option(
    None, "--checkpoint", help="Path to adapter checkpoint. Omit to evaluate baseline (zero-shot) SAM."
),
```

This is additive — supplying `--checkpoint` is unchanged; omitting it newly selects the baseline path in `run_eval`. The `--output` help string (`eval_cmd.py:27`, "defaults to checkpoint.parent") should be reworded to note the baseline fallback (§6.3). The `ValueError → BadParameter(param_hint="--checkpoint")` wrap (`eval_cmd.py:74-75`) stays.

### §6.2 `run_eval` — baseline path when `checkpoint is None and artifacts is None`

Today the `artifacts is None` branch (`runner.py:99-104`) requires a checkpoint:

```python
else:
    if checkpoint is None:
        raise ValueError("run_eval requires either 'checkpoint' or 'artifacts' to be provided.")
    resolved_checkpoint = checkpoint
    resolved_peft_method = cfg.peft.method
    resolved_run_dir = None
```

New behavior: `checkpoint is None and artifacts is None` is the **baseline** case (not an error). Restructure so `resolved_checkpoint` may be `None`:

```python
if artifacts is not None:
    resolved_checkpoint = artifacts.checkpoint_path
    resolved_run_dir = artifacts.run_dir
else:
    resolved_checkpoint = checkpoint            # may be None → baseline
    resolved_run_dir = None
```

`resolved_peft_method` is **no longer needed** on the standalone path (it was only used to build `_peft_method` for dispatch); remove the `make_peft_method(resolved_peft_method)` call at `runner.py:106` (see §6.4). The docstring's "neither ``checkpoint`` nor ``artifacts`` provided" `Raises` line (`runner.py:92`) is removed; add a note that `checkpoint=None` (standalone) evaluates baseline SAM.

The `--split val` gate (`runner.py:107-114`) and the `--split test` gate (`runner.py:115-116`) are **unchanged** — baseline still needs something to score against. The dataset-build block (`runner.py:118-129`) is unchanged.

### §6.3 Adapter load — sentinel dispatch + advisory warning + output-dir fallback

Replace the standalone load block (`runner.py:131-138`):

```python
if model is None:
    wrapper = load_sam31(
        cfg.model, channels=cfg.data.channels, channel_semantics=cfg.data.channel_semantics
    )
    _peft_method.load_from_disk(wrapper, resolved_checkpoint)
    _load_channel_adapter(wrapper, resolved_checkpoint)
else:
    wrapper = model
```

with:

```python
if model is None:
    wrapper = load_sam31(
        cfg.model, channels=cfg.data.channels, channel_semantics=cfg.data.channel_semantics
    )
    if resolved_checkpoint is not None:
        from custom_sam_peft.peft_adapters import discover_method_from_checkpoint
        detected = discover_method_from_checkpoint(resolved_checkpoint)
        if cfg.peft.method != detected:
            _LOG.warning(
                "cfg.peft.method=%r but the checkpoint at %s is %r; "
                "loading the checkpoint's method (config value ignored for eval dispatch).",
                cfg.peft.method, resolved_checkpoint, detected,
            )
        load_adapter(wrapper, resolved_checkpoint)   # train.checkpoint.load_adapter
    # else: baseline — no adapter load, no channel-adapter restore.
else:
    wrapper = model
```

- `load_adapter` is `train.checkpoint.load_adapter(wrapper, path)` (`checkpoint.py:117`), which already does sentinel-based LoRA/QLoRA dispatch **and** restores the channel adapter (`_load_channel_adapter`, `checkpoint.py:123`). This removes both the `make_peft_method(...).load_from_disk(...)` call and the separate `_load_channel_adapter(...)` call from `eval/runner.py`. Import `load_adapter` (rename the existing `from custom_sam_peft.train.checkpoint import _load_channel_adapter` import — see §6.4).
- The advisory `WARNING` is purely informational; eval proceeds with the detected method. It uses `discover_method_from_checkpoint` (§7) only to compute the message — `load_adapter` itself re-derives the kind from the sentinel, so the warning and the actual dispatch read the same file convention.
- This **drops eval's `cfg.peft.method` read for dispatch**; the value now only feeds the advisory comparison.

**Output-dir fallback** (`runner.py:160-165`). Today:

```python
out = (
    output_dir if output_dir is not None
    else (resolved_run_dir if resolved_run_dir is not None else resolved_checkpoint.parent)
)
```

`resolved_checkpoint.parent` crashes when `resolved_checkpoint is None`. New precedence: explicit `output_dir` → `resolved_run_dir` → `resolved_checkpoint.parent` (when a checkpoint exists) → `cfg.run.output_dir` → cwd:

```python
if output_dir is not None:
    out = output_dir
elif resolved_run_dir is not None:
    out = resolved_run_dir
elif resolved_checkpoint is not None:
    out = resolved_checkpoint.parent
else:
    out = Path(cfg.run.output_dir) if cfg.run.output_dir else Path.cwd()
```

`cfg.run.output_dir` exists with default `"./runs"` (`schema.py:109`), so the cwd arm is only a defensive fallback for an empty string. (Implementer may simplify to `Path(cfg.run.output_dir)` since the schema default guarantees non-empty; the cwd arm documents intent.)

### §6.4 Imports to adjust in `eval/runner.py`

- Remove `from custom_sam_peft.peft_adapters import make_peft_method` (`runner.py:23`) — no longer used on the standalone path. (Confirm no other use in the file; grep shows only the one call at `runner.py:106`.)
- Change `from custom_sam_peft.train.checkpoint import _load_channel_adapter` (`runner.py:24`) → `from custom_sam_peft.train.checkpoint import load_adapter`. `_load_channel_adapter` is no longer called directly (it runs inside `load_adapter`).
- Add the lazy `from custom_sam_peft.peft_adapters import discover_method_from_checkpoint` inside the load block (only reached when a checkpoint exists), keeping `peft_adapters` off the baseline hot path — consistent with the predict runner's lazy-import discipline.

### §6.5 Backward compatibility

Configs that still carry `peft.method` load and eval **fine**: the value is ignored for dispatch and used only for the advisory mismatch warning. `peft.method` stays a **required train-only** schema field (`PEFTConfig.method`, `schema.py:488`, no default) — this spec does NOT relax it. A standalone `csp eval --config c.yaml --checkpoint ckpt` where `c.yaml` says `peft.method: lora` but `ckpt` is QLoRA now logs a warning and loads QLoRA correctly (previously it would mis-dispatch via `make_peft_method("lora").load_from_disk`).

## §7 PEFT seam centralization (`peft_adapters/__init__.py`)

The sentinel-file convention (`custom_sam_peft_qlora.json` present → QLoRA, else LoRA) is currently checked in **three** places with duplicated logic:

1. `predict/adapter_load.py::detect_adapter_kind` (`adapter_load.py:32-44`) — `_QLORA_SENTINEL` constant, raises `typer.BadParameter` when `adapter_config.json` is also missing.
2. `train/checkpoint.py::load_adapter` (`checkpoint.py:118-124`) — inline `(path / _QLORA_META_FILENAME).exists()`.
3. eval (after §6) — would be a fourth inline check if not centralized.

All collapse onto one canonical module-level discovery function.

### §7.1 New canonical discovery function

```python
# peft_adapters/__init__.py
def discover_method_from_checkpoint(adapter_dir: Path) -> str:
    """Discover the PEFT method of an unknown checkpoint dir from the sentinel file.

    Convention: custom_sam_peft_qlora.json present → 'qlora', else 'lora'.
    This is DISCOVERY (no prior expectation). Contrast detect_method_from_checkpoint
    (an INSTANCE method on LoraAdapter/QloraAdapter) which VERIFIES a *known* method
    and raises CheckpointError on contradiction.

    Returns 'lora' or 'qlora'. Does not validate adapter_config.json presence —
    callers that need that check do it separately (e.g. predict's detect_adapter_kind).
    """
    return "qlora" if (adapter_dir / _QLORA_META_FILENAME).is_file() else "lora"
```

Naming rationale (the locked design requires a non-colliding name): the existing instance method is `detect_method_from_checkpoint` and it **verifies** (raises if the dir contradicts the adapter the instance represents). The new module-level function **discovers** (no expectation, never raises on a well-formed dir). `discover_` vs `detect_` keeps them distinct and documents the operation difference; the docstrings cross-reference each other. `_QLORA_META_FILENAME` already lives in this module (`__init__.py:29`).

### §7.2 Relocate `read_adapter_base_model_name` into `peft_adapters`

Move `read_adapter_base_model_name` (currently `predict/adapter_load.py:87-98`, reads `adapter_config.json:base_model_name_or_path`) into `peft_adapters/__init__.py`, so all "what is in this checkpoint dir?" introspection lives with the adapters. The function body is verbatim (it uses only `json` + `Path`, no torch/bnb, so no import-discipline concern at module scope). The `_LORA_CONFIG = "adapter_config.json"` constant moves with it (or is inlined).

### §7.3 Delegators in `predict/adapter_load.py` (public names preserved)

`predict/runner.py` imports `detect_adapter_kind` and `read_adapter_base_model_name` from `predict/adapter_load.py` (`runner.py:153,277,320`), and `tests/predict/test_adapter_detect.py` imports both from there. Keep both public names as **thin delegators**:

```python
# predict/adapter_load.py
def detect_adapter_kind(checkpoint_dir: Path) -> AdapterKind:
    """Return 'qlora' | 'lora'. Raises typer.BadParameter if adapter_config.json absent."""
    if not (checkpoint_dir / _LORA_CONFIG).is_file():
        raise typer.BadParameter(
            f"--checkpoint must contain adapter_config.json (checked: {checkpoint_dir})"
        )
    from custom_sam_peft.peft_adapters import discover_method_from_checkpoint
    return cast(AdapterKind, discover_method_from_checkpoint(checkpoint_dir))

def read_adapter_base_model_name(checkpoint_dir: Path) -> str | None:
    from custom_sam_peft.peft_adapters import read_adapter_base_model_name as _impl
    return _impl(checkpoint_dir)
```

- `detect_adapter_kind` keeps its `typer.BadParameter`-on-missing-`adapter_config.json` semantics (the canonical `discover_` function does NOT validate `adapter_config.json` presence — it only checks the qlora sentinel — so the delegator does the `adapter_config.json` check itself, preserving behavior; `tests/predict/test_adapter_detect.py:43-46` asserts this raise).
- The `peft_adapters` imports stay **lazy (inside the function bodies)**, preserving `adapter_load.py`'s documented discipline that the base-model-only hot path never imports `peft_adapters` (`adapter_load.py:10-12`). `read_adapter_base_model_name` is pure-JSON so could import at module scope, but keeping it lazy is uniform and harmless.
- `AdapterKind = Literal["lora", "qlora"]` and `_QLORA_SENTINEL` stay defined in `adapter_load.py` for the type alias; `_QLORA_SENTINEL` is now unused by the delegator and may be deleted (the canonical function uses `peft_adapters._QLORA_META_FILENAME`). Implementer: delete `_QLORA_SENTINEL` if no longer referenced.

### §7.4 `train/checkpoint.py::load_adapter` uses the canonical function

Replace the inline check (`checkpoint.py:118-124`):

```python
def load_adapter(wrapper: Sam3Wrapper, path: Path) -> Sam3Wrapper:
    if (path / _QLORA_META_FILENAME).exists():
        load_qlora(wrapper, path)
    else:
        load_lora(wrapper, path)
    _load_channel_adapter(wrapper, path)
    return wrapper
```

with a call to the canonical discovery function:

```python
def load_adapter(wrapper: Sam3Wrapper, path: Path) -> Sam3Wrapper:
    from custom_sam_peft.peft_adapters import discover_method_from_checkpoint
    if discover_method_from_checkpoint(path) == "qlora":
        load_qlora(wrapper, path)
    else:
        load_lora(wrapper, path)
    _load_channel_adapter(wrapper, path)
    return wrapper
```

`checkpoint.py` already imports `load_lora`/`load_qlora` and `make_peft_method`; the local `_QLORA_META_FILENAME` constant (`checkpoint.py:35`) becomes unused by `load_adapter` but is still referenced elsewhere? Grep: it is used only at `checkpoint.py:119`. After this edit it is unused — delete it (or keep as documentation; spec recommends delete to avoid drift, since the canonical source of the constant is now `peft_adapters`). Behavior is identical.

### §7.5 The wizards reuse the seam (no fourth copy)

`_interactive.peek_adapter` (§3) calls `discover_method_from_checkpoint` + `method_pretty_name` + `read_adapter_base_model_name`. `eval/runner.py` (§6.3) calls `discover_method_from_checkpoint` for the advisory warning and `train.checkpoint.load_adapter` (which calls it again internally) for the actual load. No new inline sentinel checks land anywhere.

### §7.6 `EvalArtifacts.peft_method` — kept (not dead)

Spec-phase check per the locked design: `EvalArtifacts.peft_method` is **still read** after unification:

- `eval/runner.py:97` reads `artifacts.peft_method` on the `run`/`train` artifacts path. §6 leaves the artifacts branch untouched (it only drops `resolved_peft_method` on the *standalone* branch). So `artifacts.peft_method` is no longer consumed by `run_eval` for dispatch — but the field is also asserted by the trainer→evaluator seam tests (`tests/integration/test_trainer_evaluator_seam.py:162,188`) and the extensibility test (`tests/integration/test_peft_extensibility.py:140`), and unit-tested directly (`tests/unit/test_eval_artifacts.py:13,26`). The trainer sets it (`trainer.py:462`).

Verdict: **keep `EvalArtifacts.peft_method`.** It documents the method the trainer used and is part of the trainer→evaluator hand-off contract that integration tests enforce. Removing it would break those tests for no benefit. The §6 change simply stops *dispatching* on it (the artifacts checkpoint always matches its method, so even if the runner wanted to sentinel-detect on that path it would agree). The field is not dead; do not remove it.

> Optional polish (not required): after §6, `run_eval`'s artifacts branch no longer assigns `resolved_peft_method` at all (it was the only reader). Confirm the artifacts branch compiles without that local. The field on `EvalArtifacts` is independent of the local and stays.

## §8 `csp predict --interactive`

New `--interactive` / `-i` flag on `predict_cmd.py`. When set:

1. `configure_logging(verbose)` as today.
2. `require_tty()` (§3) — **before any prompt**.
3. Run the helper `run_predict_interactive(...)` (in `_interactive.py`), which collects answers, optionally writes a thin config, prints the full runnable `csp predict` command, and returns. Interactive mode never *runs* inference — it builds a command.

The user's locked choice is "collect everything needed for a runnable command." Prompt flow (ordered):

| # | Step | Prompt | Maps to | Required / Defaulted |
|---|------|--------|---------|----------------------|
| P1 | `checkpoint` | `Adapter checkpoint directory? Leave blank for baseline (no adapter).` | `--checkpoint <dir>` or omitted | Defaulted (blank → baseline). Non-blank validated with `validate_checkpoint_dir` (§3); then `peek_adapter` prints `detected adapter: <LoRA\|QLoRA>, base model: <name>`. |
| P2 | `channels` | `Number of input image channels?` | thin-config `data.channels` | Defaulted `3`. Positive-int validator. |
| P3 | `channel_semantics` | `Channel semantics? [rgb/rgba/grayscale/freeform]` | thin-config `data.channel_semantics` | Defaulted `rgb`. `ask_choice` over `CHANNEL_SEMANTIC_NAMES`. |
| P4 | `merge_adapter` | `Merge adapter weights before inference?` | `--merge-adapter` / `--no-merge-adapter` | Defaulted `yes`. Only asked when P1 gave a checkpoint (no adapter → nothing to merge). |
| P5 | `score_threshold` | `Minimum score to keep a prediction [0.0-1.0]?` | `--score-threshold <f>` | Defaulted `0.3` (CLI default). Unit-interval validator. |
| P6 | `save_masks` | `Mask output format? [rle/png/none]` | `--save-masks <fmt>` | Defaulted `rle`. |
| P7 | `visualize` | `Write per-image overlay PNGs?` | `--visualize` (flag emitted only when yes) | Defaulted `no`. |
| P8 | `images` | `Images: dir / glob / manifest / single file?` | `--images <path>` | **Required** (per-run arg). |
| P9 | `prompts` | `Class prompts (comma-separated) or path to a one-per-line file?` | `--prompts <spec>` | **Required** (per-run arg). |
| P10 | `output` | `Output directory?` | `--output <dir>` | **Required** (per-run arg). |

Not prompted (stay at their CLI defaults; the helper prints a one-line note that they are addable as flags): `--top-k` (100), `--device` (auto), `--dtype` (auto), `--batch-size` (auto), `--seed` (0). This keeps the prompt set tight while the printed command remains runnable.

### §8.1 Thin config — written ONLY when needed

`predict/runner.py::_resolve_config` (`runner.py:111-219`) reads exactly three fields from `--config`: `model.name`, `data.channels`, `data.channel_semantics` (`runner.py:129-146`). The `predict -i` helper writes a thin config **only when at least one of these differs from the predict defaults**:

- channels ≠ 3, OR
- channel_semantics ≠ `rgb`, OR
- a non-default model is chosen (the helper may offer an optional model-name prompt; if it does not, model.name only differs when the adapter's `base_model_name_or_path` differs from `facebook/sam3.1` — but that is read from the adapter at run time by `_resolve_config`, not from `--config`, so it does **not** require a thin config).

For the common **RGB + default-model** case (channels=3, semantics=rgb), **write nothing** and print a `--config`-free command.

When a thin config IS written, it contains only the consumed fields:

```yaml
# Generated by `csp predict --interactive` on YYYY-MM-DD
model:
  name: facebook/sam3.1
data:
  channels: 4
  channel_semantics: rgba
```

> This thin file is NOT a `TrainConfig` and is NOT validated via `load_config` — `_resolve_config` parses it with `yaml.safe_load` and reads only those keys (`runner.py:133-146`). The helper just validates `channel_semantics ∈ CHANNEL_SEMANTIC_NAMES` (P3's `ask_choice` already enforces this) so the runtime check at `runner.py:181-185` cannot fire. Respect `--output`-style overwrite safety: if writing a thin config, default its path to `predict-config.yaml` (or a name the helper chooses) and refuse to overwrite an existing file without confirmation (re-using the `emit` overwrite-refusal pattern). The thin-config path is then passed as `--config` in the printed command.

### §8.2 Printed command

Assemble and print the full command from the collected answers, e.g.:

```text
csp predict --images <images> --prompts <prompts> --output <output> \
  --checkpoint <dir> --merge-adapter --score-threshold 0.3 --save-masks rle [--visualize] \
  [--config predict-config.yaml]
```

Rules:

- `--checkpoint` omitted when baseline (P1 blank); `--merge-adapter`/`--no-merge-adapter` emitted only with a checkpoint (P4).
- `--visualize` emitted only when P7 = yes.
- `--config <thin>` emitted only when §8.1 wrote one.
- Flags left at defaults (`--top-k`, `--device`, `--dtype`, `--batch-size`, `--seed`) are NOT emitted — the command relies on the CLI defaults; the helper notes they can be appended.
- Quote any path/prompt argument that contains spaces (the helper is responsible for shell-safe rendering, e.g. via `shlex.quote`).

## §9 Error handling & edge cases

| Condition | Command | Behavior |
|-----------|---------|----------|
| `-i` + non-TTY (`not sys.stdin.isatty()`) | eval, predict | Hard error up front via `require_tty()`: `typer.BadParameter("interactive mode needs a TTY; use the flag-driven command instead")`. Checked BEFORE any prompt. Mirrors `init`'s existing guard. |
| Plain `csp eval` / `csp predict` (no `-i`) non-TTY | eval, predict | Unchanged — neither requires a TTY today; behavior preserved. |
| eval-reuse: config path does not `load_config` | eval `-i` (R1) | Re-ask via `validate_config_with_eval_split`: print the `ConfigError` message, prompt again. |
| eval-reuse: config loads but has no val/val_split/test | eval `-i` (R1) | Re-ask with `"config has no val/test split to evaluate; pick a config with one"`. |
| eval-reuse: checkpoint dir missing `adapter_config.json` | eval `-i` (R2), predict `-i` (P1) | Re-ask via `validate_checkpoint_dir`: `"<dir> is not an adapter checkpoint dir (missing adapter_config.json)"`. `peek_adapter` is never called on a bad dir. |
| eval-reuse: chosen split absent from config | eval `-i` (R3) | Re-ask / warn. R1's validator guarantees at least one split exists, so this only fires when the user picks the split the config lacks (e.g. `test` with only a `val`). |
| eval-baseline: user picks "none" validation | eval `-i` (B2) | One-line discouraged-but-allowed note (`run_mode="eval"` no-val note, §3.1). Not blocked. The rendered baseline config then has no val source; `csp eval --split val` against it will raise the runner's `--split val` gate at run time — the note warns the user up front. |
| eval / predict `-i`: `--output` thin/baseline file exists w/o `--force` | eval `-i` baseline, predict `-i` thin config | `emit`-style refusal: `typer.BadParameter("refusing to overwrite existing <path>; pass --force")` (eval baseline reuses `emit`; predict thin-config reuses the same refusal). Checked at write time. |
| `Ctrl-C` mid-helper (`KeyboardInterrupt`) | eval, predict | Nothing written, nothing partially printed beyond prompts. eval-reuse writes no file ever; eval-baseline / predict thin-config write only at the very end after all prompts — interrupt before that leaves no file. Propagates as a clean abort. |
| Standalone `csp eval` with neither `--checkpoint` nor a config val source | eval (non-interactive) | The `--split val` gate raises `ValueError("--split val requires data.val, data.val_split, or data.hf.split_val …")` → `BadParameter` (existing behavior; baseline still needs something to score). |
| Standalone `csp eval --checkpoint ckpt` where `cfg.peft.method` ≠ detected | eval (non-interactive) | Advisory `WARNING` logged; eval loads the checkpoint's actual method and proceeds (§6.3). Not an error. |
| Baseline eval, no `--output`, no checkpoint | eval (non-interactive) | Output dir falls back to `cfg.run.output_dir` (default `./runs`) then cwd (§6.3) — no `NoneType.parent` crash. |
| predict `-i`: invalid `channel_semantics` | predict `-i` (P3) | `ask_choice` re-asks (membership-checked against `CHANNEL_SEMANTIC_NAMES`); the thin config can never carry an invalid semantic. |

Per-answer recoverable input (bad path, out-of-range threshold, non-positive channels) re-asks via the prompt primitive's `validate` loop — never raises mid-helper, consistent with the prior spec's §9.

## §10 Testing strategy

CPU-first per project convention. Every case here is CPU-testable; **no new GPU tests**. Interactive prompts are driven by monkeypatching the prompt primitives (`ask_text`/`ask_choice`/`ask_confirm`) in `_interactive` — deterministic, per the prior spec's §10. No real model, dataset, or HF download loads in any unit test.

### §10.1 `_interactive.py` — `tests/unit/cli/test_interactive.py` (new)

| Test | Asserts |
|------|---------|
| `test_prompt_primitives_moved` | `ask_text`/`ask_choice`/`ask_confirm` importable from `_interactive`; re-ask-on-invalid behavior intact (membership, validate-loop). |
| `test_require_tty_non_tty_raises` | `sys.stdin.isatty` patched False → `require_tty()` raises `typer.BadParameter`; patched True → returns None. |
| `test_shared_steps_fragment_shapes` | `dataset_source` / `validation` / `model_weights` each return nested-dict fragments (COCO + HF variants); `_deep_merge` composes them. |
| `test_validate_checkpoint_dir` | dir with `adapter_config.json` → None; dir without → error string; non-dir → error string. |
| `test_validate_config_with_eval_split` | config with `val` / `val_split` / `hf.split_val` / `test` → None (each); config with none → error string; non-loadable config → `ConfigError` message string. Build configs with `TrainConfig.model_validate` / tmp YAML. |
| `test_peek_adapter` | synthetic lora dir → `("LoRA", base)`; synthetic qlora dir (sentinel file written) → `("QLoRA", base)`; base read from `adapter_config.json:base_model_name_or_path`, None when absent. |

### §10.2 init retarget — `tests/unit/cli/test_setup_wizard.py` (update)

| Test | Asserts |
|------|---------|
| `test_run_mode_offers_only_train_run` | `_ask_run_mode` choices are `["train", "run"]`; `eval` is NOT offered (patch `ask_choice` to capture the choices list, or assert via the choice set). |
| `test_epochs_always_asked` | `epochs` step has no `eval`-exclusion `when` (always runs for train/run). |
| `test_class_imbalance_always_asked_train_run` | `class_imbalance` step runs for both train and run (no eval-exclusion). |
| existing happy-path + render + emit tests | still pass against `train`/`run`; the existing `run_mode="eval"` render test (`test_setup_wizard.py:315`) is repurposed/kept for the eval-baseline render (it exercises `render(..., run_mode="eval")`, which the eval-baseline path reuses — see §10.3). The existing `Ctx(..., run_mode="eval")` construction (`test_setup_wizard.py:241`) stays valid since the shared `RunMode` superset still includes `eval`. |
| moved-symbol imports | if tests reference `sw.ask_text` / `sw.WizardStep` etc., they pass via `setup_wizard` re-exporting the moved names (§3); otherwise retarget the import to `_interactive`. |

### §10.3 `eval -i` helper (CPU) — `tests/unit/cli/test_eval_interactive.py` (new)

| Test | Asserts |
|------|---------|
| `test_reuse_prints_command_writes_nothing` | monkeypatch prompts to choose `reuse`, valid config path, valid lora checkpoint dir, split `val`. Assert stdout contains `csp eval --config <cfg> --checkpoint <ckpt> --split val` and **no file** is created in `tmp_path` beyond the inputs. |
| `test_reuse_peek_prints_method` | reuse path with a qlora checkpoint dir → output mentions `QLoRA` and the base model name. |
| `test_baseline_emits_reloadable_config` | choose `baseline`, COCO dataset, auto-split val, blank weights. Assert the written file `load_config`s cleanly and `cfg.data.val_split` is set, `cfg.peft.method` is the default; printed command is `csp eval --config <out> --split val` with no `--checkpoint`. |
| `test_baseline_resolves_to_no_adapter_eval` | the emitted baseline config + `checkpoint=None` is accepted by `run_eval`'s baseline path (exercise just the gate + dispatch decision, monkeypatching `load_sam31` / `Evaluator` / dataset builder so nothing real loads). |
| `test_non_tty_hard_errors` | `eval -i` with `stdin.isatty()` patched False → `BadParameter`, no prompt primitive called, no file. |
| `test_output_exists_without_force` | baseline path, `--output` exists, no `--force` → `BadParameter` refusal at write time. |
| `test_ctrl_c_writes_nothing` | `KeyboardInterrupt` mid-prompt → no file. |

### §10.4 eval-runner (CPU) — `tests/unit/test_eval_runner.py` + `test_eval_runner_gate.py` (update/add)

| Test | Asserts |
|------|---------|
| `test_peft_inferred_lora_without_cfg_method` | synthetic checkpoint dir (LoRA: `adapter_config.json`, no sentinel) → `run_eval(cfg, checkpoint=dir, model=None)` dispatches `load_lora` even when `cfg.peft.method="qlora"`; monkeypatch `load_lora`/`load_qlora`/`load_sam31`/`Evaluator`/builder. |
| `test_peft_inferred_qlora_without_cfg_method` | synthetic qlora dir (sentinel file written) → dispatches `load_qlora` even when `cfg.peft.method="lora"`. |
| `test_peft_mismatch_logs_warning` | `cfg.peft.method` ≠ detected → a `WARNING` is logged (caplog) and the **detected** method loads. |
| `test_checkpoint_none_skips_adapter_load` | `run_eval(cfg, checkpoint=None, model=None)` (baseline) → no `load_lora`/`load_qlora`/`_load_channel_adapter` call; `load_sam31` called once; `Evaluator.evaluate_and_save` called. Monkeypatch all of these. |
| `test_baseline_output_dir_fallback` | baseline (`checkpoint=None`, `output_dir=None`, `artifacts=None`) → out dir resolves to `cfg.run.output_dir` (no `NoneType.parent` crash). |
| existing `test_run_eval_dispatches_qlora_from_disk` | update: the standalone qlora path now dispatches via `train.checkpoint.load_adapter` (sentinel) rather than `make_peft_method(...).load_from_disk`; the synthetic dir must carry the qlora sentinel so detection picks qlora. Keep asserting `load_qlora` + channel-adapter restore run. |
| `test_eval_runner_gate.py` | unchanged — the `--split val` gate logic is untouched (§6.2). |

### §10.5 `predict -i` helper (CPU) — `tests/unit/cli/test_predict_interactive.py` (new)

| Test | Asserts |
|------|---------|
| `test_command_assembly_baseline_rgb` | monkeypatch prompts: blank checkpoint, channels=3, semantics=rgb, threshold default, save-masks rle, no visualize, images/prompts/output values. Assert printed command has `--images/--prompts/--output`, NO `--checkpoint`, NO `--merge-adapter`, NO `--config`, NO `--visualize`. **No file written.** |
| `test_command_assembly_with_checkpoint` | checkpoint dir given → command has `--checkpoint <dir>` and `--merge-adapter`; peek output mentions detected method. |
| `test_thin_config_emitted_for_non_rgb` | channels=4, semantics=rgba → a thin config is written containing only `model.name` + `data.channels` + `data.channel_semantics`; printed command references it via `--config`. The thin file parses with `yaml.safe_load` and carries exactly those keys. |
| `test_thin_config_not_emitted_for_rgb` | channels=3, semantics=rgb → no thin config written; command has no `--config`. |
| `test_visualize_flag_emitted_when_yes` | P7=yes → command contains `--visualize`. |
| `test_non_tty_hard_errors` | `predict -i` non-TTY → `BadParameter`, no prompts, no file. |
| `test_thin_config_overwrite_refused` | thin-config target exists, no force/confirm → refusal. |

### §10.6 `peft_adapters` seam (CPU) — `tests/unit/test_peft_method_protocol.py` / `tests/predict/test_adapter_detect.py` (update/add)

| Test | Asserts |
|------|---------|
| `test_discover_method_lora` | dir with only `adapter_config.json` (no sentinel) → `discover_method_from_checkpoint(dir) == "lora"`. |
| `test_discover_method_qlora` | dir with `custom_sam_peft_qlora.json` → `"qlora"`. |
| `test_discover_does_not_validate_adapter_config` | dir with sentinel but no `adapter_config.json` → still `"qlora"` (discovery does not require `adapter_config.json`; that check lives in `detect_adapter_kind`). |
| `test_detect_adapter_kind_delegates_and_still_validates` | `predict.adapter_load.detect_adapter_kind` agrees with `discover_method_from_checkpoint` for lora/qlora dirs AND still raises `typer.BadParameter` on a dir missing `adapter_config.json` (`tests/predict/test_adapter_detect.py:43-46` semantics preserved). |
| `test_read_base_model_name_delegates` | `predict.adapter_load.read_adapter_base_model_name` and the relocated `peft_adapters.read_adapter_base_model_name` return the same value for a fixture lora dir; None when the file/key is absent (`tests/predict/test_adapter_detect.py:132-147` cases preserved). |
| `test_train_checkpoint_load_adapter_uses_discover` | `train.checkpoint.load_adapter` dispatches `load_qlora` for a sentinel dir and `load_lora` otherwise (monkeypatch the loaders); behavior unchanged from today (`tests/unit/test_train_checkpoint.py` qlora/lora dirs). |

> Existing `tests/integration/test_trainer_evaluator_seam.py` and `test_peft_extensibility.py` assertions on `EvalArtifacts.peft_method` stay green (the field is kept, §7.6).

## §11 Migration & breaking-change stance

Pre-1.0. One clean breaking change, the rest additive or compatible. No shims, consistent with the prior spec's §11.

| What changes | Class | Who notices | How |
|--------------|-------|-------------|-----|
| `init --interactive` drops the `eval` run mode | **Breaking** | Anyone who scripted `init -i` expecting an `eval` option | The `run_mode` prompt offers only `train`/`run`. Shipped only in commit `84bc83f` (the most recent feature commit); nothing in-tree depends on it; no automation could rely on an interactive prompt's choice set. Replacement: `csp eval --interactive` (§5). No shim. |
| `csp eval --checkpoint` becomes optional | **Additive** | Anyone running `csp eval` | Supplying `--checkpoint` is unchanged; omitting it newly selects the baseline (zero-shot) eval. No existing invocation breaks. |
| eval infers PEFT method from the checkpoint, not `cfg.peft.method` | **Compatible** | Anyone with a config carrying `peft.method` | Configs still load + eval. The value is ignored for dispatch and used only for an advisory mismatch `WARNING`. A previously *mis-dispatching* config (method disagrees with checkpoint) now loads correctly instead of failing — a strict improvement. `peft.method` stays a required train-only schema field. |
| `read_adapter_base_model_name` relocated to `peft_adapters` | **Compatible** | `predict/adapter_load.py` importers | The public name stays exported from `predict/adapter_load.py` as a delegator (§7.3); existing imports (`predict/runner.py`, `tests/predict/test_adapter_detect.py`) keep working. |
| `detect_adapter_kind` / `load_adapter` now delegate to the canonical seam | **Compatible** | internal callers | Same return values + same `typer.BadParameter` raise for `detect_adapter_kind`; `train.checkpoint.load_adapter` behavior identical. Pure refactor. |

No release-notes migration steps are required beyond noting the `init -i` eval mode moved to `csp eval -i`. The PEFT-inference change is a behavior *improvement* (mis-dispatch becomes correct dispatch + warning), not a break.

### Rollback

Revert the PR. It is one logical change: the new `_interactive.py`, the `eval -i` / `predict -i` flags + helpers, the `init -i` shrink, the eval-runner baseline + sentinel-dispatch changes, and the `peft_adapters` seam centralization revert together. Reverting restores `init -i`'s `eval` mode, the required `--checkpoint`, eval's `make_peft_method(cfg.peft.method)` dispatch, and the three inline sentinel checks. No data/schema migration to undo (`peft.method`, `EvalArtifacts.peft_method`, and the schema are unchanged).

---

## §12 Relationship to the prior wizard spec

This spec is a sibling to [`2026-05-26-interactive-setup-wizard-design.md`](2026-05-26-interactive-setup-wizard-design.md) and depends on the machinery that spec introduced.

**Carried over unchanged (reused, not re-specified):**

- The `WizardStep` / `Ctx` / `_deep_merge` / `run_wizard` registry-driver model (prior §3) — moved to `_interactive.py` (§3 here) so all three helpers share it.
- The three prompt primitives (`ask_text` / `ask_choice` / `ask_confirm`) and their re-ask-on-invalid contract (prior §3).
- The "render the unified `config_full.yaml` → validate the exact bytes via `load_config` → emit with a 2-line header" pipeline (prior §2, §5) — the eval-baseline path (§5.3) reuses `setup_wizard.render` + `validate` + `emit` verbatim.
- The TTY-guard-before-prompt and `--output`/`--force` pre-flight discipline (prior §7, §9) — generalized into `require_tty()` (§3) and reused by `eval -i` / `predict -i`.
- The CPU-first, monkeypatch-the-primitives testing approach (prior §10) — §10 here follows it; no GPU tests.
- The shared steps `dataset_source`, `validation`, `model_weights` (prior §4 rows 3/4/9) — extracted into `_interactive.py` and reused by both `init -i` and `eval -i` baseline.

**Changed by this spec:**

- The prior spec's §4 row 1 modeled `run_mode` as `train`/`run`/`eval`. This spec narrows `init`'s offered modes to `train`/`run` (§4) and moves eval interactivity into a dedicated command helper. The shared `RunMode` *type* keeps `eval` as a superset (§3) so the eval-baseline path can drive the shared validation step.
- The prior spec assumed `eval` would be an init mode that asked the full battery (with `domain`/`peft_sizing` noise — the exact complaint in #172). This spec replaces that with the reuse/baseline `eval -i` helper (§5) that asks only what eval needs.
- PEFT method selection: the prior `peft_sizing` step (prior §6.1) sets `peft.method` for *training*. For eval/predict this spec discovers the method from the checkpoint (§6, §7) and never prompts for it.

**Not touched by this spec** (owned by the prior spec, left as-is): the VRAM auto-size step, the `infer_class_imbalance` detector, the gradient-checkpointing removal (prior Workstream 2), and the `config_full.yaml` template body. The eval-baseline path renders that same template; it does not modify it.
