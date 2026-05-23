# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
