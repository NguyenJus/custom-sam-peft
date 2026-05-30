# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added — eval GT-vs-Pred visualization

- **eval**: new `eval.visualize` (bool, default `true`) and `eval.visualize_count`
  (int, default `10`) config knobs. On the final/standalone eval path (`csp eval`,
  `csp run`'s eval phase, `csp train --eval`), eval now writes one
  `Ground Truth | Prediction` composite PNG per variety-weighted sampled image
  under `<output>/visualizations/`, with per-class color legend. Predictions are
  the Hungarian mask-only matched 1:1 set per class.
- **cli**: `csp eval --visualize/--no-visualize` (tri-state; defers to config when
  unset) and `csp run --visualize/--no-visualize` (default on). The in-loop
  training eval is unchanged.

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

### Removed — box_hint localization-hint curriculum (#88)

- **train**: removed the `box_hint` curriculum and the `BoxHintSchedule`
  config model (`train.box_hint.*`). Training is now text-only.
- **Changed**: `SupportPrompts` is retained as a field-less reserved extension
  seam (#126 §12) for future mask/point hints; `Sam3Wrapper.forward(support=)`
  stays as a no-op. Inference is unchanged (already text-only).
- **Note**: resume tolerates pre-removal checkpoints — a stale `box_hint_p`
  key in an old `training_state.pt` is ignored.
- **Note**: any config carrying `train.box_hint:` now fails to load with a
  Pydantic `extra_forbidden` error; delete the block from your YAML.

---

## [0.12.0] — 2026-05-23

### Added — SAM 3.1 multiplex forward (issue #22)

- **feat**: one forward per ≤16-class group in train, eval, and predict. New
  `train.multiplex.classes_per_forward` (1..16, default 16). New
  `eval.batch_size: int | "auto"` (default `"auto"`). New `--batch-size auto`
  (default) for `csp predict`.

### Performance

- **perf**: Multi-class training/eval workloads (COCO ≥80 classes, LVIS) see
  significantly higher throughput; see PR description for
  `scripts/bench_multiplex_throughput.py` numbers.

### Breaking (numeric)

- Per-step loss magnitudes shift vs prior versions. The `LossConfig` defaults
  (`w_mask=w_obj=w_presence=1`) are unchanged; re-validate manual tunings.
- Per-step RNG draw order shifts at K>1; runs are not seed-bit-equivalent to
  <0.12.0 for K>1. Bit-equivalence holds at `train.multiplex.classes_per_forward=1`.

### Escape hatch

- Set `train.multiplex.classes_per_forward: 1` to recover the per-class
  iteration order within the same code path.

## [0.11.0] — 2026-05-23

### Breaking — v0.x debt paydown ("hardening pass", issue #26)

This release rewrites the YAML schema, CLI surface, and internal seams to
make the user-facing API small and obvious. Upgrade by editing your YAML
manually against the rename table below — there is intentionally no
migration tool (pre-1.0; README already declares this).

#### YAML field renames

| Old | New | Notes |
| --- | --- | --- |
| `train.lr` | `train.learning_rate` | `"lr"` is an abbreviation; `"learning_rate"` matches the concept and is self-documenting. |

The following fields were **considered** for rename but kept as-is:

| Field | Decision |
| --- | --- |
| `train.batch_size` | No rename — already consistent with common ML convention. |
| `run.output_dir` | No rename — already consistent. |
| `tracking.wandb.project` | No rename — already consistent. |

#### Removed fields

- `EvalConfig.metrics` — was silently ignored by `compute_coco_map`; removed.
  Re-introduction tracked in follow-up issue (see below).
- `BoxHintSchedule.early_stop_p_threshold` — was unused; removed pending
  early-stopper implementation.

#### Demoted fields (no longer user-set; hardcoded as internal defaults)

The following config dataclasses are now internal-only and have been moved to
`src/custom_sam_peft/config/_internal.py`. Import from `config._internal`
in new code; the old names remain re-exported from `config.schema` for
backward compatibility through this PR.

- `MatcherWeights` — box supervision is deferred; `lambda_l1` / `lambda_giou`
  are now hardcoded constants.
- `LossConfig` — `focal_gamma` / `focal_alpha` are never set by users; now
  hardcoded.
- `WandbConfig` — rarely set by users; demoted to internal default.
- `ExportConfig` — single-field dataclass; demoted to internal.

#### CLI command flag changes

- `train` gains bare `--eval` and `--export` flags.
- `eval` gains a bare `--export` flag.
- `run` is now documented as an alias for `train --eval --export`.

#### New error taxonomy

- `CustomSamPeftError` — base class for all user-facing errors.
- `ConfigError` — missing, malformed, or invalid config value.
- `DataError` — dataset-loading or example-decoding failures.
- `ModelError` — model construction, patch-application, or adapter failures.
- `CheckpointError` — checkpoint read/write or resume-state mismatches.
- `EnvironmentError` — runtime precondition failures (HF gating, missing GPU,
  missing extra).
- CLI renders errors in a four-part shape (summary / expected / found / fix).
  Re-run with `-v` for the full traceback.

#### Internal refactors

- `Runtime` value object centralizes device + dtype + rank-awareness.
  `is_primary` and `world_size` fields are seam scaffolding for future
  DDP / FSDP work.
- `paths/` module owns the run-dir layout; no more string-joined
  `runs/.../checkpoints/` outside `paths/`.
- `_bootstrap.py` is the sole site for adapter registration, seeding, and
  logging configuration.
- `_patch_*` functions each live in their own file under
  `src/custom_sam_peft/models/_patches/`; `Sam3Patches.apply` is the
  single application site.
- `EvalArtifacts` is the seam between `Trainer` and `Evaluator`.
- PEFT method-string branches (`if peft.method == "lora": ...`) replaced
  by `PEFTMethod` protocol dispatch throughout.
- Static guards in CI enforce: no method-string branches outside
  `peft_adapters/`, no `.to(device)` outside collator + `runtime/`, no
  string-joined checkpoint paths outside `paths/`.

### See also

- Audit inventory: `docs/superpowers/specs/2026-05-21-hardening-audit-inventory.md`
- Config schema reference: `docs/config-schema.md`
- Design spec: `docs/superpowers/specs/2026-05-21-hardening-pass-design.md`
- Implementation plan: `docs/superpowers/plans/2026-05-21-hardening-pass.md`
- Tracking issue: [#26](https://github.com/NguyenJus/Efficient-SAM3-Finetuning/issues/26)
