# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

<!-- Add entries for the next milestone here. -->

## [v0.9.0] — 2026-06-02

### Added — v0.9.0 provenance gate and config resync

- **tracker**: local-disk experiment tracker is now the default; resume-dir resolution fixed.
- **tensorboard**: opt-in TensorBoard extra added (`#206`).
- **peft**: in-projection concept scope added for PEFT (`#230`).
- **ci**: no-uncited-default provenance gate enforces citation or `# tbd:` tag on every new hyperparam (`#192`).
- **config**: config-schema resynced to current field set (`#239`); README refreshed (`#200`).
- **security**: 25 code-scanning alerts resolved; `GH_TOKEN` removed from Colab notebook.

## [v0.8.0] — 2026-06-01

### Added — v0.8.0 GPU re-architecture and eval panel

- **gpu**: test suite re-architected around RTX 5070 Ti / sm_120 (`#211`); Colab T4 confirmations closed (`#139`/`#193`).
- **vram**: `calibrate` VRAM probe now survives dirty-OOM on sm_120; ladder hardened against sm_120 "device not ready" surface (`#208`).
- **eval**: eval-viz gains an original-image panel alongside GT-vs-Pred composites.
- **deps**: TensorBoard promoted to a base dependency.
- **train**: three-regime step prediction implemented (`#128`).
- **wizard**: run-mode default changed from `train` to `run` (`#223`).

## [v0.7.0] — 2026-05-31

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

### Changed — VRAM K-autosize, plateau ladder, and wall-clock limit

- **vram**: VRAM K-autosize with split-activation model and cc-aware attention
  (materialized H·N² term only when cc < 8.0); calibrate-and-climb strategy (`#203`).
- **train**: plateau-response LR ladder with best-checkpoint-as-final (`#197`).
- **train**: wall-clock time limit with resumable stop (`#198`).
- **train**: unified OOM ladder covering trainer/eval/predict paths (`#181`).
- **config**: literature-cited defaults; epoch count aligned to convergence regime (`#120`).

## [v0.6.0] — 2026-05-28

### Breaking — text-primary prompt invariant (#126)

- **schema**: removed the `data.prompt_mode` field. Any config that carries
  `prompt_mode:` (any value) now fails at load with a Pydantic
  `extra_forbidden` error. Migration: delete the line from your YAML.
- **api**: removed the `Sam3Wrapper.forward(..., box_hints=...)` kwarg. The
  forward is text-only; an optional `support=` parameter (a `SupportPrompts`
  reserved seam) is accepted but ignored. See "Removed — box_hint
  localization-hint curriculum (#88)" below.
- **types**: removed `BoxPrompts` and `PromptMode`. `Prompts` is now an alias
  for `TextPrompts`.
- **trainer/CLI**: removed three hand-rolled `prompt_mode == "bbox"` guards
  (`train/trainer.py`, `cli/train_cmd.py`, `cli/run_cmd.py`) — the schema is
  the sole gate.

## [v0.5.0] — 2026-05-28

### Added — interactive wizard and config simplifications

- **wizard**: interactive setup wizard added (`#149`); full gradient-checkpointing support removed to simplify training path.
- **resume**: bare `--resume` now auto-resolves the latest checkpoint (`#156`).
- **config**: `image_size` knob removed — always 1008 (`#158`).
- **eval**: OOM caps added to eval path (`#153`).
- **fix**: config-path-vs-CWD resolution fixed (`#151`); augmentation box-alignment fixed (`#150`).

## [v0.4.0] — 2026-05-24

### Added — SAM 3.1 multiplex forward and N-channel adapter

- **train/eval/predict**: SAM 3.1 multiplex forward (`#22`) — one forward per ≤16-class group.
- **model**: N-channel input via learned channel adapter (`#111`).
- **loss**: domain-aware loss presets (`#112`).
- **checkpoint**: QLoRA checkpoint disk-load support (`#98`).
- **test**: 3-tier GPU test taxonomy established; 3 real-GPU bugs fixed (`#138`).

## [v0.3.0] — 2026-05-23

### Added — hardening pass and new subsystems

- **schema/CLI**: hardening pass — schema/CLI/seam rewrite, `train.lr` → `train.learning_rate`,
  demoted internal dataclasses, new error taxonomy (`#26`/`#90`).
- **versioning**: hatch-vcs dynamic versioning (`#103`).
- **ci**: GHCR Docker publish workflow (`#34`).
- **cli**: `csp predict` subcommand (`#74`).
- **vram**: analytic VRAM-tier preset selection + OOM auto-retry (`#36`).
- **data**: `data.limit` knob (`#72`); no-val / auto-split support (`#71`).
- **train**: augmentation presets (`#75`); progress bars (`#76`).

## [v0.2.0] — 2026-05-21

### Added — first GPU-tested release

- First GPU-validated release; project renamed to `custom-sam-peft`.
- CI hardened; draft-PR workflow skips non-essential jobs.
- Minimized GPU-gated test surface; smoke-test harness established.

## [v0.1.0] — 2026-05-18

### Added — initial milestone baseline

- Package skeleton, COCO + HF data subsystem, SAM 3.1 loader/wrapper/matcher/open-vocab losses,
  PEFT LoRA + QLoRA adapters with save/load, and training-loop scaffold.
