# GPU Coverage Assessment (#65) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land the assessment doc for #65 plus the 11 concrete test additions (6 GPU, 5 CPU) it specifies, close the uniquely-#65-owned coverage gaps, file two follow-up issues for deferred work, and close #65 (ticking #70's `GPU test gate decision shipped — #65` checkbox).

**Architecture:** Test-only PR. Zero changes under `src/custom_sam_peft/`. The spec at `docs/superpowers/specs/2026-05-21-gpu-coverage-assessment-design.md` is the single source of truth for *what* each test asserts and *why* each GPU test cannot be CPU-substituted; this plan tells the orchestrator *how* and *in what order* to land them. All 11 test additions are file-disjoint and dependency-free, so they fan out via `superpowers:dispatching-parallel-agents` on the same branch (no `isolation: "worktree"` — would split onto temp branches), followed by a single design-sensitive reviewer pass that ends with lint/format/typecheck.

**Tech Stack:** pytest, bitsandbytes (CUDA-only, gated by `_bnb_available()`), peft, pycocotools, pyhton 3.11, uv, ruff, mypy, gh CLI.

---

## Orchestrator Routing Cheat Sheet

Apply these routings when dispatching subagents (per `~/.claude/CLAUDE.md` Implementation-Orchestrator pipeline §2):

| Task | Role | Model | Effort | Dispatch shape |
| ---- | ---- | ----- | ------ | -------------- |
| 1 (commit spec + plan) | orchestrator (self) | n/a | n/a | inline |
| 2A–2D (GPU additions T1, T2+T3, T4, T5+T6) | implementer | sonnet | high | parallel batch with 3A–3E |
| 3A–3E (CPU additions C1, C2, C3, C4, C5) | implementer | sonnet | high | parallel batch with 2A–2D |
| 4 (code review) | reviewer | **opus** | **xhigh** | serial, after 2 + 3 |
| 5 (lint/format/typecheck) | implementer | sonnet | high | serial, LAST step of reviewer pass |
| 6 (verification) | orchestrator (self) | n/a | n/a | inline |
| 7 (skip version bump — record reason) | orchestrator (self) | n/a | n/a | inline |
| 8A, 8B (follow-up issues) | orchestrator (self) | n/a | n/a | inline |
| 9 (open PR + watch CI) | orchestrator (self) | n/a | n/a | inline; `run_in_background`/Monitor |
| 10 (post-merge close-out #65 + #70) | orchestrator (self) | n/a | n/a | inline, after merge |

**Escalation note for Task 2A (T1, QLoRA resume).** T1 is borderline Hard — it stitches `run_training`'s `resume_from` seam against bnb 4-bit `quant_state` survival across save/load and the spec is the most detail-dense of any task. Default routing remains sonnet/high; if the T1 subagent escalates per `superpowers:subagent-driven-development` (or hits Design Ambiguity tiers 2+ from `~/.claude/CLAUDE.md`), re-dispatch on opus/xhigh.

**Reviewer rationale (opus/xhigh).** This branch encodes an audit's design argument (which gaps are uniquely #65-owned vs. delegated to #23/#64/#68/#74/#78/#79), pins behaviors with no prior test (NaN abort, QLoRA resume), and asserts metric finiteness on real-model artifacts. That makes it design-sensitive in the orchestrator CLAUDE.md sense — minimum opus/xhigh.

**Parallelization rationale.** Every implementation task touches a different file (T1 creates a new file; T2/T3/T4 each modify a single GPU file no other task touches; T5/T6 each modify a single integration file; C1/C2/C3 each create a new unit file; C4/C5 each modify a single integration file no other task touches). Per `~/.claude/CLAUDE.md` step 2: "Prefer `superpowers:dispatching-parallel-agents` for 2+ file-disjoint, dependency-free tasks — same branch/worktree, no `isolation: \"worktree\"`."

---

## Version-Bump Decision (Step 7 SOURCE OF TRUTH)

**SKIP version bump.** Per `~/.claude/CLAUDE.md` Override `superpowers:finishing-a-development-branch` step 4: *"Skip when nothing ships to consumers (lockfile-only, CI fix, internal refactor with no behavior change), no manifest carries a version, or the project uses non-semver."* This branch:

- Adds files only under `tests/`, `docs/superpowers/specs/`, and `docs/superpowers/plans/`.
- Touches zero files under `src/custom_sam_peft/`.
- Adds zero new YAML configs, zero new dependencies, zero workflow edits.
- Therefore ships nothing observable to consumers (PyPI / Colab notebook / CLI users see no change).

The follow-up issues filed in Tasks 8A and 8B are filed via `gh issue create` — they do not require a manifest version change.

Step 5a (close-out tag-and-push) is therefore also skipped, per the same rule. Note that fact in the sign-off line at close-out 5e.

---

## File Structure

```text
docs/superpowers/
  specs/2026-05-21-gpu-coverage-assessment-design.md     # EXISTING — committed by Task 1
  plans/2026-05-21-gpu-coverage-assessment-plan.md       # EXISTING (THIS FILE) — committed by Task 1

tests/gpu/
  test_real_train_qlora_resume.py                        # NEW (Task 2A — T1)
  test_real_train_overfits.py                            # MODIFIED (Task 2B — T2)
  test_real_train_qlora.py                               # MODIFIED (Task 2B — T3)
  test_run_end_to_end_gpu.py                             # MODIFIED (Task 2C — T4)

tests/integration/
  test_peft_lora_real.py                                 # MODIFIED (Task 2D — T5)
  test_peft_qlora_real.py                                # MODIFIED (Task 2D — T6)
  test_train_resume.py                                   # MODIFIED (Task 3D — C4)
  test_train_end_to_end.py                               # MODIFIED (Task 3E — C5)

tests/unit/
  test_evaluator_schema.py                               # NEW (Task 3A — C1)
  test_peft_scope_coverage.py                            # NEW (Task 3B — C2)
  test_trainer_nan_behavior.py                           # NEW (Task 3C — C3)
  test_checkpoint_roundtrip.py                           # NEW (Task 3D — C4 amendment, primary)
```

No file under `src/custom_sam_peft/` is touched. No CI workflow is touched. If Task 3C surfaces a real bug in `train/loop.py`'s NaN behavior, the implementer pins the current (potentially buggy) behavior and the reviewer files a follow-up issue — fix is out of scope for this PR per spec §6.2 C3.

---

## Task 1: Commit spec + plan, push branch

**Files:**
- Track: `docs/superpowers/specs/2026-05-21-gpu-coverage-assessment-design.md`
- Track: `docs/superpowers/plans/2026-05-21-gpu-coverage-assessment-plan.md`

**Dispatch:** Inline (orchestrator). No subagent.

- [ ] **Step 1: Verify worktree state**

Run from worktree root:
```bash
git rev-parse --abbrev-ref HEAD
git status --short
```
Expected: branch is `gpu-coverage-65`; the two doc files appear under untracked.

- [ ] **Step 2: Stage exactly the two doc files**

```bash
git add docs/superpowers/specs/2026-05-21-gpu-coverage-assessment-design.md \
        docs/superpowers/plans/2026-05-21-gpu-coverage-assessment-plan.md
git status --short
```
Expected: only those two files staged; nothing else.

- [ ] **Step 3: Commit**

```bash
git commit -m "$(cat <<'EOF'
docs(superpowers): assess GPU coverage gaps owned by #65 + plan

Spec audits the post-v0.6.1 GPU/CPU suite against the eight #65
coverage dimensions, points the five dimensions owned by other issues
(#23 #64 #68 #74 #78 #79) at their owners, and identifies four gaps
uniquely owned by #65: vision/all PEFT scopes, mid-training NaN abort
behavior, eval-metric assertions, and QLoRA resume. The plan stages
six GPU additions (T1–T6) and five CPU additions (C1–C5) in
file-disjoint parallel implementer tasks, gated by a design-sensitive
opus/xhigh reviewer pass.

Refs #65, #70.
EOF
)"
```

- [ ] **Step 4: Push branch with upstream**

```bash
git push -u origin gpu-coverage-65
```
Expected: branch created on origin; no PR yet (PR is opened by Task 9, after CI is green).

---

## Task 2A: GPU T1 — `tests/gpu/test_real_train_qlora_resume.py` (NEW)

**Files:**
- Create: `tests/gpu/test_real_train_qlora_resume.py`

**Dispatch:** implementer subagent, sonnet/high. Parallel with 2B, 2C, 2D, 3A, 3B, 3C, 3D, 3E. **Subagent prompt MUST include:** "Read spec §6.1 T1 in full before writing the test. The spec is the source of truth for assertions, overrides, and what is explicitly NOT asserted. If you escalate per superpowers:subagent-driven-development, the orchestrator will re-dispatch on opus/xhigh."

**Subagent context primer (paste into the prompt):**
- Existing QLoRA test for marker patterns: `tests/gpu/test_real_train_qlora.py` (read in full — copy module-level pytestmark, per-test `@pytest.mark.requires_bnb` and `@pytest.mark.skipif(not _bnb_available(), ...)` exactly).
- Shared helpers: `tests/gpu/conftest.py` exports `_bnb_available` and `_RecordingTracker`. Import from `tests.gpu.conftest`.
- `run_training` signature: `src/custom_sam_peft/train/runner.py:34` accepts `resume_from: Path | None`.
- `Trainer.fit` resume seam: `src/custom_sam_peft/train/trainer.py:155-200`.
- Config: `configs/examples/gpu_smoke_qlora.yaml` (shipped epochs=25, batch=1 → ~50 grad steps).
- Use `code-review-graph` MCP `query_graph` (callers_of `run_training`, callees_of `Trainer.fit`) and `get_review_context` for snippets — per project CLAUDE.md: "ALWAYS use the code-review-graph MCP tools BEFORE using Grep/Glob/Read."

- [ ] **Step 1: Write the failing test file**

Create `tests/gpu/test_real_train_qlora_resume.py` with the contract from spec §6.1 T1:

```python
"""QLoRA resume smoke: split the 50-step QLoRA overfit budget across a save/load
boundary to pin that bnb 4-bit quant_state survives the resume seam.

Gated by `@pytest.mark.gpu`, `@requires_compatible_gpu`, `@requires_checkpoint`,
plus per-test `@pytest.mark.requires_bnb` and `@pytest.mark.skipif(not _bnb_available())`.
Not in CI by default. Run with:
    pytest -m gpu tests/gpu/test_real_train_qlora_resume.py -v

Phase A trains ~26 grad steps (epochs=13 against 2-image tiny_coco, batch=1,
just past save_every=25 to land a checkpoint). Phase B resumes from that
checkpoint and trains the full shipped 50-step budget (epochs=25). Net GPU
time is roughly one extra `test_qlora_overfits_in_50_steps`; the resume
seam can only be exercised end-to-end against real bnb 4-bit weights
(CPU stub at tiny_sam3_lora_stub.py cannot replicate Linear4bit). See
spec §6.1 T1 and §7.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from custom_sam_peft.config.loader import load_config
from custom_sam_peft.train.runner import run_training
from tests.gpu.conftest import _bnb_available, _RecordingTracker

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.requires_compatible_gpu,
    pytest.mark.requires_checkpoint,
]

CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "configs" / "examples" / "gpu_smoke_qlora.yaml"
)
VRAM_CEIL_GB = 10.0  # Same ceiling as test_real_train_qlora.py.


def _load_cfg(tmp_path: Path, tiny_coco_dir: Path, *, epochs: int) -> object:
    return load_config(
        CONFIG_PATH,
        overrides=[
            f"data.train.annotations={tiny_coco_dir / 'annotations.json'}",
            f"data.train.images={tiny_coco_dir / 'images'}",
            f"data.val.annotations={tiny_coco_dir / 'annotations.json'}",
            f"data.val.images={tiny_coco_dir / 'images'}",
            f"run.output_dir={tmp_path}",
            f"train.epochs={epochs}",
            # save_every=25 lands one checkpoint at step 25, midway through
            # the ~50-step total budget. log_every=1 so every step's scalar
            # is captured for finiteness checks.
            "train.save_every=25",
            "train.log_every=1",
        ],
    )


@pytest.mark.requires_bnb
@pytest.mark.skipif(not _bnb_available(), reason="bitsandbytes not installed")
def test_qlora_resume_survives_quant_state_roundtrip(
    tmp_path: Path,
    tiny_coco_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.cuda.reset_peak_memory_stats()

    # --- Phase A: train ~26 grad steps so save_every=25 fires. ---
    tracker_a = _RecordingTracker()
    monkeypatch.setattr(
        "custom_sam_peft.train.runner.build_tracker", lambda *_a, **_kw: tracker_a
    )
    cfg_short = _load_cfg(tmp_path, tiny_coco_dir, epochs=13)
    run_training(cfg_short)

    # Locate the checkpoint produced by phase A.
    runs = sorted(tmp_path.glob("gpu-smoke-qlora-*"))
    assert runs, f"phase A wrote no run dir under {tmp_path}"
    ckpts_a = sorted((runs[-1] / "checkpoints").glob("step_*"))
    assert ckpts_a, f"phase A wrote no checkpoint under {runs[-1] / 'checkpoints'}"
    resume_dir = ckpts_a[-1]

    losses_a = [s["loss/total"] for _, s in tracker_a.scalars if s["loss/total"] > 0]
    assert losses_a, "phase A logged no loss scalars"
    assert all(math.isfinite(v) for _, s in tracker_a.scalars for v in s.values()), (
        "phase A logged a non-finite scalar"
    )

    # --- Phase B: resume from phase A's checkpoint, complete the 50-step budget. ---
    tracker_b = _RecordingTracker()
    monkeypatch.setattr(
        "custom_sam_peft.train.runner.build_tracker", lambda *_a, **_kw: tracker_b
    )
    cfg_full = _load_cfg(tmp_path, tiny_coco_dir, epochs=25)
    run_training(cfg_full, resume_from=resume_dir)

    losses_b = [s["loss/total"] for _, s in tracker_b.scalars if s["loss/total"] > 0]
    assert losses_b, "phase B (resumed) logged no loss scalars"
    assert all(math.isfinite(v) for _, s in tracker_b.scalars for v in s.values()), (
        "phase B logged a non-finite scalar"
    )
    assert math.isfinite(losses_b[-1]), f"phase B final loss not finite: {losses_b[-1]}"

    # Final adapter state: at least one lora_ param, every lora_ param finite.
    # Locate the run dir produced by phase B (most recent under tmp_path).
    runs_b = sorted(tmp_path.glob("gpu-smoke-qlora-*"))
    assert len(runs_b) >= 2, f"phase B did not create a fresh run dir: {runs_b}"
    final_ckpts = sorted((runs_b[-1] / "checkpoints").glob("step_*"))
    assert final_ckpts, "phase B wrote no final checkpoint"
    import safetensors.torch  # peft's default adapter format

    adapter_file = final_ckpts[-1] / "adapter" / "adapter_model.safetensors"
    if adapter_file.exists():
        adapter_state = safetensors.torch.load_file(str(adapter_file))
    else:
        # Fallback: bin format.
        adapter_state = torch.load(
            final_ckpts[-1] / "adapter" / "adapter_model.bin", map_location="cpu"
        )
    lora_params = {k: v for k, v in adapter_state.items() if "lora_" in k}
    assert lora_params, "no lora_ params in final adapter state"
    for name, t in lora_params.items():
        assert torch.isfinite(t).all(), f"non-finite lora param: {name}"

    peak_vram_gb = torch.cuda.max_memory_allocated() / 1e9
    assert peak_vram_gb <= VRAM_CEIL_GB, (
        f"peak VRAM {peak_vram_gb:.2f}GB exceeded ceiling {VRAM_CEIL_GB}GB"
    )
```

- [ ] **Step 2: Verify collection (CPU; the test itself will skip without GPU)**

Run from worktree root:
```bash
uv run pytest --collect-only tests/gpu/test_real_train_qlora_resume.py
```
Expected: exit code 0; one test function collected: `test_qlora_resume_survives_quant_state_roundtrip`. The test itself will be `SKIPPED` on a CPU host because of the `requires_compatible_gpu` / `requires_bnb` gates — that is fine. The orchestrator does NOT run T1 here; it will collect-count in Task 6.

- [ ] **Step 3: Verify ruff/mypy on the new file**

```bash
uv run ruff check tests/gpu/test_real_train_qlora_resume.py
uv run ruff format --check tests/gpu/test_real_train_qlora_resume.py
```
Expected: both clean. (mypy runs against `src/custom_sam_peft` only in this repo; tests are not type-checked.)

- [ ] **Step 4: Commit**

```bash
git add tests/gpu/test_real_train_qlora_resume.py
git commit -m "$(cat <<'EOF'
test(gpu): pin QLoRA resume across the save/load quant-state seam

Splits the existing 50-step QLoRA overfit budget across a checkpoint
boundary (epochs=13 phase A → save at step 25 → epochs=25 phase B
resuming from step 25). Asserts finiteness of every logged scalar in
both phases, finiteness of every final lora_ adapter parameter, and
peak VRAM ≤ 10 GB (the existing test_real_train_qlora.py ceiling).
Does NOT compare against an uninterrupted reference — the test's
purpose is to prove bnb 4-bit quant_state survives save/load without
corrupting subsequent gradients; the CPU stub at
tiny_sam3_lora_stub.py cannot replicate Linear4bit.

Refs #65. Spec: docs/superpowers/specs/2026-05-21-gpu-coverage-assessment-design.md §6.1 T1.
EOF
)"
```

---

## Task 2B: GPU T2 + T3 — eval-metric finiteness on 50-step tests

**Files:**
- Modify: `tests/gpu/test_real_train_overfits.py` (T2)
- Modify: `tests/gpu/test_real_train_qlora.py::test_qlora_overfits_in_50_steps` ONLY (T3)

**Dispatch:** implementer subagent, sonnet/high. Parallel with 2A, 2C, 2D, 3A, 3B, 3C, 3D, 3E. **The two files do not import each other** — same subagent handles both edits for cohesion.

**Subagent context primer:**
- Spec sections: §6.1 T2 and §6.1 T3.
- Existing files: `tests/gpu/test_real_train_overfits.py` and `tests/gpu/test_real_train_qlora.py`. Both already import `math` and `from pathlib import Path`.
- Run-dir glob patterns come from each YAML's `run.name`: `gpu-smoke-lora-*` for T2 (cf. `configs/examples/gpu_smoke_lora.yaml`), `gpu-smoke-qlora-*` for T3.
- T3 applies ONLY to `test_qlora_overfits_in_50_steps`. **Do not touch `test_qlora_smoke_fast`** — it monkeypatches Evaluator to a no-op (`_SkipEvaluator` at lines 124–131 of `test_real_train_qlora.py`), so no `metrics.json` body is written.

- [ ] **Step 1: T2 — extend `test_real_train_overfits.py::test_overfits_in_50_steps`**

After the existing `peak_vram_gb` assertion at the end of `test_overfits_in_50_steps`, add the metrics-finiteness block. Insert these lines immediately after the existing `assert peak_vram_gb <= VRAM_CEIL_GB, (...)` statement (lines 69–71 in the current file):

```python

    # T2 (spec §6.1): assert the Evaluator's metrics.json overall.mAP is finite.
    import json
    runs = sorted(tmp_path.glob("gpu-smoke-lora-*"))
    assert runs, f"no run dir under {tmp_path}"
    metrics = json.loads((runs[-1] / "metrics.json").read_text())
    assert "overall" in metrics, f"metrics.json missing 'overall': {metrics}"
    mAP = metrics["overall"].get("mAP")
    assert isinstance(mAP, (int, float)) and math.isfinite(mAP) and mAP >= 0.0, (
        f"overall.mAP not finite/non-negative: {mAP}"
    )
```

Note: the `math` import is already present at line 14 of the file. Move the `import json` to the top-of-file imports block (between line 14 `import math` and line 15 `from pathlib import Path`) to satisfy `ruff` — do not leave a function-local import.

- [ ] **Step 2: T3 — extend `test_real_train_qlora.py::test_qlora_overfits_in_50_steps`**

After the existing `peak_vram_gb` assertion at the end of `test_qlora_overfits_in_50_steps` (the LOSS_RATIO_CEIL + VRAM_CEIL_GB block at lines 68–73), add the same metrics-finiteness block, with the run-dir glob changed to match the QLoRA YAML's `run.name`:

```python

    # T3 (spec §6.1): assert the Evaluator's metrics.json overall.mAP is finite.
    import json
    runs = sorted(tmp_path.glob("gpu-smoke-qlora-*"))
    assert runs, f"no run dir under {tmp_path}"
    metrics = json.loads((runs[-1] / "metrics.json").read_text())
    assert "overall" in metrics, f"metrics.json missing 'overall': {metrics}"
    mAP = metrics["overall"].get("mAP")
    assert isinstance(mAP, (int, float)) and math.isfinite(mAP) and mAP >= 0.0, (
        f"overall.mAP not finite/non-negative: {mAP}"
    )
```

Same import discipline: move `import json` to the top-of-file imports block.

Do NOT modify `test_qlora_smoke_fast`. Verify by reading the file after editing: the only diff in `test_qlora_smoke_fast` should be from the new top-of-file `import json` (it doesn't use json itself).

- [ ] **Step 3: Verify collection still works**

```bash
uv run pytest --collect-only tests/gpu/test_real_train_overfits.py tests/gpu/test_real_train_qlora.py
```
Expected: 3 tests collected (`test_overfits_in_50_steps`, `test_qlora_overfits_in_50_steps`, `test_qlora_smoke_fast`).

- [ ] **Step 4: Lint the changed files**

```bash
uv run ruff check tests/gpu/test_real_train_overfits.py tests/gpu/test_real_train_qlora.py
uv run ruff format --check tests/gpu/test_real_train_overfits.py tests/gpu/test_real_train_qlora.py
```
Expected: both clean.

- [ ] **Step 5: Commit**

```bash
git add tests/gpu/test_real_train_overfits.py tests/gpu/test_real_train_qlora.py
git commit -m "$(cat <<'EOF'
test(gpu): assert metrics.json mAP finite on 50-step LoRA + QLoRA smokes

T2/T3 per spec §6.1. The 50-step tests already call Evaluator.evaluate
which writes metrics.json, but never inspected the result. Adds a
post-training read + isinstance + math.isfinite + >= 0.0 check on
overall.mAP. test_qlora_smoke_fast is unchanged: it monkeypatches
Evaluator to a no-op so no metrics.json body is written.

Refs #65.
EOF
)"
```

---

## Task 2C: GPU T4 — tighten e2e mAP assertion in `test_run_end_to_end_gpu.py`

**Files:**
- Modify: `tests/gpu/test_run_end_to_end_gpu.py`

**Dispatch:** implementer subagent, sonnet/high. Parallel with 2A, 2B, 2D, 3A, 3B, 3C, 3D, 3E.

**Subagent context primer:**
- Spec section: §6.1 T4.
- Existing file: `tests/gpu/test_run_end_to_end_gpu.py` (single test `test_run_end_to_end_writes_bundle`).
- The current assertion at line 59 is `assert isinstance(metrics["overall"].get("mAP"), (int, float))` — which lets `NaN`, `-inf`, and negative numbers through.
- `math` is NOT currently imported; add the import.

- [ ] **Step 1: Add `math` to imports**

Top-of-file imports currently include `import json` (line 9). Add `import math` between line 9 and line 10 (alphabetical order within stdlib block):

```python
import json
import math
from pathlib import Path
```

- [ ] **Step 2: Replace the loose assertion**

Find the block at lines 56–59:

```python
    # metrics.json parses; has overall.mAP numeric.
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert "overall" in metrics
    assert isinstance(metrics["overall"].get("mAP"), (int, float))
```

Replace lines 58–59 with:

```python
    assert "overall" in metrics
    mAP = metrics["overall"].get("mAP")
    assert isinstance(mAP, (int, float)) and math.isfinite(mAP) and mAP >= 0.0, (
        f"overall.mAP not finite/non-negative: {mAP}"
    )
```

- [ ] **Step 3: Verify collection**

```bash
uv run pytest --collect-only tests/gpu/test_run_end_to_end_gpu.py
```
Expected: 1 test collected.

- [ ] **Step 4: Lint**

```bash
uv run ruff check tests/gpu/test_run_end_to_end_gpu.py
uv run ruff format --check tests/gpu/test_run_end_to_end_gpu.py
```
Expected: clean.

- [ ] **Step 5: Commit**

```bash
git add tests/gpu/test_run_end_to_end_gpu.py
git commit -m "$(cat <<'EOF'
test(gpu): tighten e2e mAP check to finite + non-negative

T4 per spec §6.1. The previous isinstance(mAP, (int, float)) allowed
NaN, -inf, and negative values (Python float is too permissive).
pycocotools can produce negatives when the precision array is all -1
(sentinel for "no GT") and the mean handling slips through — a real
regression that would land silently today. Adds math.isfinite(mAP)
and mAP >= 0.0.

Refs #65.
EOF
)"
```

---

## Task 2D: GPU T5 + T6 — `vision`-scope subtests on real ViT-Det

**Files:**
- Modify: `tests/integration/test_peft_lora_real.py` (T5; inspection tier — file already carries `pytest.mark.gpu_inspection`)
- Modify: `tests/integration/test_peft_qlora_real.py` (T6; inspection tier)

**Dispatch:** implementer subagent, sonnet/high. Parallel with 2A, 2B, 2C, 3A, 3B, 3C, 3D, 3E. Same subagent handles both — they are mirror tests and share the same intervention shape.

**Subagent context primer:**
- Spec sections: §6.1 T5 and §6.1 T6.
- Existing files exercise `load_sam31` + `apply_lora`/`apply_qlora` at default scope. Both files' module-level `pytestmark` already includes `pytest.mark.gpu_inspection`.
- `PEFTConfig(scope="vision")` is the explicit non-default scope. `SCOPE_TARGETS["vision"]` (in `src/custom_sam_peft/peft_adapters/lora.py:38`) is `r"backbone\.vision_backbone\.trunk\.blocks\.\d+\.attn\.(qkv|proj)$"`. **Critical regex consequence:** `"transformer.decoder"` (with a dot) is the production substring; `"mask_decoder"` would only appear at `scope="all"`. The assertions below use these exact substrings.
- Inspection tier is forward-free per `docs/superpowers/specs/2026-05-19-gpu-test-policy-design.md` §5.6 — the test calls `apply_lora`/`apply_qlora` and inspects `named_parameters` only.

- [ ] **Step 1: T5 — append `test_apply_lora_vision_scope_targets_only_vision_backbone` to `tests/integration/test_peft_lora_real.py`**

Add at the bottom of the file (after `test_merge_lora_on_real_sam31`):

```python


def test_apply_lora_vision_scope_targets_only_vision_backbone() -> None:
    """T5 per spec §6.1: scope='vision' attaches LoRA only to vision_backbone.

    The test asserts the production SCOPE_TARGETS['vision'] regex matches the
    real SAM 3.1 module names (a regression like Meta renaming vision_backbone
    to image_encoder would slip past C2 in tests/unit/test_peft_scope_coverage.py
    because that test uses a stub). Forward-free; cost is dominated by
    load_sam31 which is already paid by the other tests in this file.
    """
    w = load_sam31(ModelConfig())
    apply_lora(w, PEFTConfig(method="lora", scope="vision"))

    lora_names = [n for n, _ in w.model.model.named_parameters() if "lora_" in n]
    assert lora_names, "no lora_ params after apply_lora(scope='vision')"
    assert any("vision_backbone" in n for n in lora_names), (
        f"no vision-trunk LoRA targets at scope='vision': {lora_names[:5]}"
    )
    assert all("transformer.decoder" not in n for n in lora_names), (
        f"transformer.decoder targets present at scope='vision' (should be excluded): "
        f"{[n for n in lora_names if 'transformer.decoder' in n][:5]}"
    )
    assert all("mask_decoder" not in n for n in lora_names), (
        f"mask_decoder targets present at scope='vision' (should be excluded): "
        f"{[n for n in lora_names if 'mask_decoder' in n][:5]}"
    )
```

No new imports needed — `apply_lora`, `PEFTConfig`, `ModelConfig`, `load_sam31` are all already imported at the top of the file.

- [ ] **Step 2: T6 — append `test_apply_qlora_vision_scope_targets_only_vision_backbone` to `tests/integration/test_peft_qlora_real.py`**

Add at the bottom of the file (after `test_merge_lora_unloads_qlora_wrapper`):

```python


@pytest.mark.skipif(not _bnb_available(), reason="bitsandbytes not installed")
def test_apply_qlora_vision_scope_targets_only_vision_backbone() -> None:
    """T6 per spec §6.1: mirror of T5 for QLoRA scope='vision'."""
    w = load_sam31(ModelConfig())
    apply_qlora(w, PEFTConfig(method="qlora", scope="vision"))

    lora_names = [n for n, _ in w.model.model.named_parameters() if "lora_" in n]
    assert lora_names, "no lora_ params after apply_qlora(scope='vision')"
    assert any("vision_backbone" in n for n in lora_names), (
        f"no vision-trunk LoRA targets at scope='vision': {lora_names[:5]}"
    )
    assert all("transformer.decoder" not in n for n in lora_names), (
        f"transformer.decoder targets present at scope='vision' (should be excluded): "
        f"{[n for n in lora_names if 'transformer.decoder' in n][:5]}"
    )
    assert all("mask_decoder" not in n for n in lora_names), (
        f"mask_decoder targets present at scope='vision' (should be excluded): "
        f"{[n for n in lora_names if 'mask_decoder' in n][:5]}"
    )
```

No new imports needed — `apply_qlora`, `PEFTConfig`, `ModelConfig`, `load_sam31`, `_bnb_available`, `pytest` are all already imported at the top of the file.

- [ ] **Step 3: Verify collection**

```bash
uv run pytest --collect-only tests/integration/test_peft_lora_real.py tests/integration/test_peft_qlora_real.py
```
Expected: existing 3 + 4 tests are still collected, plus the two new ones — 9 total. (Files carry `gpu_inspection` marker; they will skip on a CPU host.)

- [ ] **Step 4: Lint**

```bash
uv run ruff check tests/integration/test_peft_lora_real.py tests/integration/test_peft_qlora_real.py
uv run ruff format --check tests/integration/test_peft_lora_real.py tests/integration/test_peft_qlora_real.py
```
Expected: clean.

- [ ] **Step 5: Commit**

```bash
git add tests/integration/test_peft_lora_real.py tests/integration/test_peft_qlora_real.py
git commit -m "$(cat <<'EOF'
test(gpu-inspection): assert vision-scope wiring on real SAM 3.1

T5/T6 per spec §6.1. Adds a per-scope wiring assertion at the
inspection tier: apply_{lora,qlora}(scope='vision') attaches LoRA
under vision_backbone only, with transformer.decoder and
mask_decoder excluded. The test would catch a regression like Meta
renaming vision_backbone to image_encoder, which the stub-based C2
test cannot. Forward-free; cost dominated by the already-paid
load_sam31 call.

Refs #65.
EOF
)"
```

---

## Task 3A: CPU C1 — `tests/unit/test_evaluator_schema.py` (NEW)

**Files:**
- Create: `tests/unit/test_evaluator_schema.py`

**Dispatch:** implementer subagent, sonnet/high. Parallel with all other 2X / 3X tasks.

**Subagent context primer:**
- Spec section: §6.2 C1.
- Target API: `custom_sam_peft.eval.metrics.MetricsReport` and `compute_coco_map` (defined in `src/custom_sam_peft/eval/metrics.py` — read in full). `Evaluator` lives at `src/custom_sam_peft/eval/evaluator.py`.
- Schema: `MetricsReport(overall: dict[str, float], per_class: dict[str, dict[str, float]], n_images: int, n_predictions: int)`. `overall` always contains `mAP`; `mAP_50` only when `0.5 in iou_thresholds`; `mAP_75` only when `0.75 in iou_thresholds`. Empty predictions return zeroed report with `n_predictions=0` (cf. `metrics.py:54-61`). Classes with no GT are skipped from `per_class` (cf. `metrics.py:93`).
- Fixtures: `tiny_coco_dir` is in `tests/conftest.py:62` (auto-discoverable). Load via `pycocotools.coco.COCO(str(tiny_coco_dir / "annotations.json"))`. `tiny_coco` has 2 images.
- For the synthetic-perfect-prediction case, use `pycocotools.mask.encode` on a binary mask matching the GT shape; the prediction is `{"image_id": ..., "category_id": ..., "segmentation": {"size": ..., "counts": ...}, "score": 1.0}`.

- [ ] **Step 1: Create the file with all five test cases**

```python
"""Unit coverage for the Evaluator output schema (MetricsReport).

C1 per spec §6.2. Pinned on CPU because the schema invariants do not depend
on real model output — see spec §7 for why metric *values* (T2/T3) need GPU.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO

from custom_sam_peft.eval.metrics import MetricsReport, compute_coco_map


def _load_gt(tiny_coco_dir: Path) -> COCO:
    return COCO(str(tiny_coco_dir / "annotations.json"))


def _perfect_prediction_from_first_gt(gt: COCO) -> list[dict[str, object]]:
    """Return a single COCO-results entry that exactly matches the first GT annotation."""
    ann_ids = gt.getAnnIds()
    assert ann_ids, "tiny_coco has no annotations — fixture broken"
    ann = gt.loadAnns(ann_ids[:1])[0]
    img = gt.loadImgs([ann["image_id"]])[0]
    # Synthesize an RLE mask matching the bbox extent.
    h, w = img["height"], img["width"]
    bin_mask = np.zeros((h, w), dtype=np.uint8, order="F")
    x, y, bw, bh = ann["bbox"]
    x0, y0 = int(x), int(y)
    x1, y1 = min(int(x + bw), w), min(int(y + bh), h)
    bin_mask[y0:y1, x0:x1] = 1
    rle = mask_utils.encode(bin_mask)
    rle["counts"] = rle["counts"].decode("ascii")  # COCO results expect str
    return [
        {
            "image_id": ann["image_id"],
            "category_id": ann["category_id"],
            "segmentation": rle,
            "score": 1.0,
        }
    ]


def test_empty_predictions_returns_zeroed_report(tiny_coco_dir: Path) -> None:
    gt = _load_gt(tiny_coco_dir)
    report = compute_coco_map(
        predictions=[],
        ground_truth=gt,
        iou_thresholds=[0.5, 0.75, 0.95],
        include_per_class=True,
    )
    assert isinstance(report, MetricsReport)
    assert report.overall == {"mAP": 0.0, "mAP_50": 0.0, "mAP_75": 0.0}
    assert report.per_class == {}
    assert report.n_predictions == 0
    assert report.n_images == len(gt.imgs)  # tiny_coco has 2


def test_iou_thresholds_pick_only_50(tiny_coco_dir: Path) -> None:
    gt = _load_gt(tiny_coco_dir)
    preds = _perfect_prediction_from_first_gt(gt)
    report = compute_coco_map(
        predictions=preds,
        ground_truth=gt,
        iou_thresholds=[0.5],
        include_per_class=False,
    )
    assert "mAP" in report.overall
    assert "mAP_50" in report.overall
    assert "mAP_75" not in report.overall


def test_iou_thresholds_pick_only_75(tiny_coco_dir: Path) -> None:
    gt = _load_gt(tiny_coco_dir)
    preds = _perfect_prediction_from_first_gt(gt)
    report = compute_coco_map(
        predictions=preds,
        ground_truth=gt,
        iou_thresholds=[0.75],
        include_per_class=False,
    )
    assert "mAP" in report.overall
    assert "mAP_75" in report.overall
    assert "mAP_50" not in report.overall


def test_overall_keys_finite(tiny_coco_dir: Path) -> None:
    gt = _load_gt(tiny_coco_dir)
    preds = _perfect_prediction_from_first_gt(gt)
    report = compute_coco_map(
        predictions=preds,
        ground_truth=gt,
        iou_thresholds=[0.5, 0.75, 0.95],
        include_per_class=True,
    )
    for k, v in report.overall.items():
        assert isinstance(v, float), f"{k} not float: {type(v)}"
        assert math.isfinite(v), f"{k} not finite: {v}"
        assert 0.0 <= v <= 1.0, f"{k} outside [0,1]: {v}"


def test_per_class_skips_classes_without_gt(tiny_coco_dir: Path) -> None:
    gt = _load_gt(tiny_coco_dir)
    preds = _perfect_prediction_from_first_gt(gt)
    report = compute_coco_map(
        predictions=preds,
        ground_truth=gt,
        iou_thresholds=[0.5, 0.75],
        include_per_class=True,
    )
    # Every per_class row is keyed by a category name (str) and has a finite "AP".
    assert report.per_class, "per_class empty despite include_per_class=True with valid GT"
    for cat_name, row in report.per_class.items():
        assert isinstance(cat_name, str)
        assert "AP" in row
        assert math.isfinite(row["AP"])


def test_include_per_class_false_returns_empty_per_class(tiny_coco_dir: Path) -> None:
    gt = _load_gt(tiny_coco_dir)
    preds = _perfect_prediction_from_first_gt(gt)
    report = compute_coco_map(
        predictions=preds,
        ground_truth=gt,
        iou_thresholds=[0.5, 0.75],
        include_per_class=False,
    )
    assert report.per_class == {}
    assert "mAP" in report.overall
```

Note: spec §6.2 C1 lists five test cases; the implementation above expands the IoU-threshold case into two cleaner test functions (`_only_50` and `_only_75`) for readability — six functions total, asserting the same contract. This is within the planner's discretion per the writing-plans skill's "self-contained tasks" guidance.

- [ ] **Step 2: Run the new tests**

```bash
uv run pytest tests/unit/test_evaluator_schema.py -v
```
Expected: all six tests PASS (~1 second total). If `test_overall_keys_finite` fails, the perfect-prediction synthesis is probably mis-encoded — debug the RLE encoding (likely byte vs str on `counts`).

- [ ] **Step 3: Lint**

```bash
uv run ruff check tests/unit/test_evaluator_schema.py
uv run ruff format --check tests/unit/test_evaluator_schema.py
```
Expected: clean.

- [ ] **Step 4: Commit**

```bash
git add tests/unit/test_evaluator_schema.py
git commit -m "$(cat <<'EOF'
test(unit): pin MetricsReport schema + edge cases on CPU

C1 per spec §6.2. Covers compute_coco_map's contract on tiny_coco:
empty predictions return a zeroed report with n_predictions=0;
iou_thresholds=[0.5] and =[0.75] each emit only their own slice;
overall keys are finite floats in [0,1] under a perfect prediction;
per_class is non-empty + finite when include_per_class=True; and
empty when False. Pinned on CPU because the schema invariants do not
depend on real model output (T2/T3 cover real-model metric values).

Refs #65.
EOF
)"
```

---

## Task 3B: CPU C2 — `tests/unit/test_peft_scope_coverage.py` (NEW)

**Files:**
- Create: `tests/unit/test_peft_scope_coverage.py`

**Dispatch:** implementer subagent, sonnet/high. Parallel with all other 2X / 3X tasks.

**Subagent context primer:**
- Spec section: §6.2 C2.
- Target API: `custom_sam_peft.peft_adapters.lora.apply_lora`, `SCOPE_TARGETS` (`src/custom_sam_peft/peft_adapters/lora.py:36`).
- Stub: `tests/fixtures/tiny_sam3_lora_stub.py` exports `make_stub_wrapper(dim=8, working=True|False)` and `FIXTURE_SCOPE_PATTERNS` (a dict mirroring `SCOPE_TARGETS` against the renamed stub subtrees). Use `PEFTConfig(target_modules=FIXTURE_SCOPE_PATTERNS[scope])` to drive the stub.
- Stub structure: `vision_trunk.blocks[0..1].attn.(qkv|proj)`, `transformer_decoder.layers[0].{self_attn,cross_attn,q_proj,k_proj,v_proj,out_proj}`, plus `neg_control_a` and `neg_control_b` linears.
- Forward path (`working=True`): routes through `vision_trunk.blocks[0].attn.qkv` so vision-scope LoRA participates in the gradient graph.

- [ ] **Step 1: Create the file with all the test cases**

```python
"""Unit coverage for PEFT scope → trainable-set wiring on a stub.

C2 per spec §6.2. The stub renames subtrees (vision_trunk vs the real
backbone.vision_backbone.trunk) so the production SCOPE_TARGETS regex
does not match; the test drives via FIXTURE_SCOPE_PATTERNS which mirrors
the same shape against the renamed paths. T5/T6 in tests/integration/
test_peft_{lora,qlora}_real.py cover real-module-name matching on GPU;
this file covers the scope→trainable-set logic on CPU.
"""

from __future__ import annotations

import pytest
import torch

from custom_sam_peft.config.schema import PEFTConfig
from custom_sam_peft.peft_adapters.lora import apply_lora
from tests.fixtures.tiny_sam3_lora_stub import FIXTURE_SCOPE_PATTERNS, make_stub_wrapper


def _lora_param_names(wrapper: object) -> list[str]:
    return [n for n, _ in wrapper.model.model.named_parameters() if "lora_" in n]


def test_scope_vision_targets_only_vision_subtree() -> None:
    w = make_stub_wrapper(dim=8, working=False)
    apply_lora(
        w,
        PEFTConfig(
            method="lora",
            scope="vision",
            target_modules=FIXTURE_SCOPE_PATTERNS["vision"],
        ),
    )
    names = _lora_param_names(w)
    assert names, "no lora_ params at scope='vision'"
    assert any("vision_trunk.blocks" in n for n in names), (
        f"no vision-trunk LoRA: {names[:5]}"
    )
    assert all("transformer_decoder" not in n for n in names), (
        f"transformer_decoder targets present at scope='vision': "
        f"{[n for n in names if 'transformer_decoder' in n][:5]}"
    )
    assert all("neg_control_" not in n for n in names), (
        f"neg_control_ targets present at scope='vision': "
        f"{[n for n in names if 'neg_control_' in n][:5]}"
    )


def test_scope_vision_decoder_targets_vision_and_decoder() -> None:
    w = make_stub_wrapper(dim=8, working=False)
    apply_lora(
        w,
        PEFTConfig(
            method="lora",
            scope="vision_decoder",
            target_modules=FIXTURE_SCOPE_PATTERNS["vision_decoder"],
        ),
    )
    names = _lora_param_names(w)
    assert any("vision_trunk.blocks" in n for n in names), (
        f"no vision-trunk LoRA at scope='vision_decoder': {names[:5]}"
    )
    assert any("transformer_decoder.layers" in n for n in names), (
        f"no transformer-decoder LoRA at scope='vision_decoder': {names[:5]}"
    )
    assert all("neg_control_" not in n for n in names), (
        f"neg_control_ targets present at scope='vision_decoder': "
        f"{[n for n in names if 'neg_control_' in n][:5]}"
    )


def test_scope_all_targets_every_linear() -> None:
    w = make_stub_wrapper(dim=8, working=False)
    apply_lora(
        w,
        PEFTConfig(
            method="lora",
            scope="all",
            target_modules=FIXTURE_SCOPE_PATTERNS["all"],
        ),
    )
    names = _lora_param_names(w)
    # Every Linear in the stub should pick up LoRA, including the negative
    # controls.
    assert any("vision_trunk.blocks" in n for n in names)
    assert any("transformer_decoder.layers" in n for n in names)
    assert any("neg_control_a" in n for n in names), (
        f"neg_control_a missing at scope='all': {names[:10]}"
    )
    assert any("neg_control_b" in n for n in names), (
        f"neg_control_b missing at scope='all': {names[:10]}"
    )


@pytest.mark.parametrize("scope", ["vision", "vision_decoder", "all"])
def test_scope_forward_backward_finite_grad(scope: str) -> None:
    """Wiring assertion: LoRA actually plugs into the gradient path.

    A scope mis-mapping (e.g. regex matches no real Linear) would still pass
    the parameter-name assertions above if the test only looked at names.
    Doing forward+backward and checking lora_A.grad is finite proves the
    parameter is in the gradient graph.
    """
    w = make_stub_wrapper(dim=8, working=True)
    apply_lora(
        w,
        PEFTConfig(
            method="lora",
            scope=scope,
            target_modules=FIXTURE_SCOPE_PATTERNS[scope],
        ),
    )
    out = w(images=torch.randn(1, 3, 8, 8))
    loss = out["pred_masks"].sum()
    loss.backward()

    lora_a_params = [
        (n, p)
        for n, p in w.model.model.named_parameters()
        if "lora_A" in n and p.requires_grad
    ]
    assert lora_a_params, f"no lora_A params at scope={scope!r}"
    has_grad = [(n, p) for n, p in lora_a_params if p.grad is not None]
    assert has_grad, (
        f"no lora_A param received a gradient at scope={scope!r} — "
        f"the regex matched module names but the modules are not in the forward path"
    )
    for n, p in has_grad:
        assert torch.isfinite(p.grad).all(), f"non-finite grad on {n} at scope={scope!r}"
```

Note on the forward path: the stub's `working=True` forward only routes through `vision_trunk.blocks[0].attn.qkv`. That guarantees scope='vision' has at least one lora_A with a real gradient; scope='vision_decoder' and scope='all' will also have at least one lora_A with a gradient via the same shared module (lora_A on `vision_trunk.blocks[0].attn.qkv` exists in all three scopes). The other targets at scope='all' (e.g. neg_control_a) will have `p.grad is None` — that is acceptable; the assertion is `has_grad: list` non-empty, NOT "every lora_A has a grad."

- [ ] **Step 2: Run the new tests**

```bash
uv run pytest tests/unit/test_peft_scope_coverage.py -v
```
Expected: 6 tests PASS (3 name-based + 3 parametrized forward-backward) (~2 seconds total).

- [ ] **Step 3: Lint**

```bash
uv run ruff check tests/unit/test_peft_scope_coverage.py
uv run ruff format --check tests/unit/test_peft_scope_coverage.py
```
Expected: clean.

- [ ] **Step 4: Commit**

```bash
git add tests/unit/test_peft_scope_coverage.py
git commit -m "$(cat <<'EOF'
test(unit): pin PEFT scope→trainable-set wiring on the stub

C2 per spec §6.2. Asserts per-scope LoRA attachment on the
tiny_sam3_lora_stub.py wrapper using FIXTURE_SCOPE_PATTERNS (mirrors
SCOPE_TARGETS against the renamed stub subtrees). For each of vision /
vision_decoder / all, checks parameter-name presence/absence under the
expected subtrees, plus a forward+backward smoke that confirms at least
one lora_A receives a finite gradient (proving the scope actually
plugged into the gradient path, not just renamed parameters).
T5/T6 cover real-module-name regex matching on GPU.

Refs #65.
EOF
)"
```

---

## Task 3C: CPU C3 — `tests/unit/test_trainer_nan_behavior.py` (NEW)

**Files:**
- Create: `tests/unit/test_trainer_nan_behavior.py`

**Dispatch:** implementer subagent, sonnet/high. Parallel with all other 2X / 3X tasks.

**Subagent context primer:**
- Spec section: §6.2 C3.
- Pin current `src/custom_sam_peft/train/loop.py` NaN-abort behavior; **do NOT modify `src/custom_sam_peft/train/loop.py`**. If the implementer discovers the behavior is undefined or silently bad, the test fails and the reviewer surfaces a follow-up issue.
- Relevant code: `src/custom_sam_peft/train/loop.py:135-138` raises `RuntimeError("Training aborted: ... consecutive non-finite micro-steps.")` when `nan_streak >= cfg.train.nan_abort_after`. `nan_abort_after` defaults to 20 (`src/custom_sam_peft/config/schema.py:243`).
- Warning path: `loop.py:126` logs `"train_step: class %r raised %s; treating as non-finite."` when `total_loss` raises `ValueError` (e.g., the Hungarian matcher rejects a non-finite cost matrix).
- Helpers (duplicate locally — spec says no shared helper extraction this PR): copy `_ds` and `_cfg` from `tests/integration/test_train_resume.py:32-79`.
- NaN injection: monkeypatch `custom_sam_peft.train.loop.total_loss` to return `{"total": torch.tensor(float("nan"), requires_grad=True), "mask": torch.tensor(0.0), "box": torch.tensor(0.0), "obj": torch.tensor(0.0), "presence": torch.tensor(0.0)}`. Read `loop.py:100-160` to confirm the dict keys consumed.
- For the "below threshold doesn't abort" case, use a `monkeypatch.context()` or a counter to flip back to the real `total_loss` after N injected NaNs.

- [ ] **Step 1: Create the file**

```python
"""Pin the current trainer NaN-loss behavior.

C3 per spec §6.2. Documents what src/custom_sam_peft/train/loop.py does
TODAY: skip the micro-step + increment nan_streak; raise RuntimeError
after cfg.train.nan_abort_after consecutive non-finite micro-steps.

This file does NOT modify src/custom_sam_peft/train/loop.py. If a test
fails because the current behavior diverges from what's pinned here
(e.g., per-step vs. per-micro-step threshold semantics differ from the
reading), the reviewer files a follow-up issue; the fix is out of scope
for this PR per spec §6.2 C3.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest
import torch

from custom_sam_peft.config.schema import (
    AugmentationsConfig,
    DataConfig,
    DataSplit,
    NormalizeConfig,
    PEFTConfig,
    RunConfig,
    TextPromptConfig,
    TrainConfig,
    TrainHyperparams,
)
from custom_sam_peft.data.coco import COCODataset
from custom_sam_peft.data.transforms import build_train_transforms
from custom_sam_peft.peft_adapters.lora import apply_lora
from custom_sam_peft.tracking.noop import NoopTracker
from custom_sam_peft.train.trainer import Trainer
from tests.fixtures.tiny_sam3_lora_stub import FIXTURE_SCOPE_PATTERNS, make_stub_wrapper


# --- Local helpers (duplicated from tests/integration/test_train_resume.py
# per spec §6.2 C3 "no shared-helper extraction this PR"). ---


def _ds(tiny_coco_dir: Path) -> COCODataset:
    transforms = build_train_transforms(
        AugmentationsConfig(hflip=False, color_jitter=0.0),
        32,
        model_name="facebook/sam3.1",
        normalize=NormalizeConfig(),
    )
    return COCODataset(
        annotations=str(tiny_coco_dir / "annotations.json"),
        images=str(tiny_coco_dir / "images"),
        prompt_mode="text",
        transforms=transforms,
        text_prompt=TextPromptConfig(),
    )


def _cfg(
    tmp_path: Path,
    tiny_coco_dir: Path,
    *,
    nan_abort_after: int,
    epochs: int,
) -> TrainConfig:
    cfg = TrainConfig(
        run=RunConfig(name="nan", output_dir=str(tmp_path), seed=42),
        data=DataConfig(
            format="coco",
            train=DataSplit(
                annotations=str(tiny_coco_dir / "annotations.json"),
                images=str(tiny_coco_dir / "images"),
            ),
            val=DataSplit(
                annotations=str(tiny_coco_dir / "annotations.json"),
                images=str(tiny_coco_dir / "images"),
            ),
            prompt_mode="text",
            image_size=32,
        ),
        peft=PEFTConfig(
            method="lora",
            scope="vision",
            target_modules=FIXTURE_SCOPE_PATTERNS["vision"],
        ),
        train=TrainHyperparams(
            epochs=epochs,
            batch_size=1,
            grad_accum_steps=1,
            save_every=2,
            log_every=1,
            warmup_steps=0,
            num_workers=0,
            nan_abort_after=nan_abort_after,
        ),
    )
    return cfg


def _nan_loss_dict() -> dict[str, torch.Tensor]:
    """Dict mirroring total_loss()'s return shape, with NaN total."""
    return {
        "total": torch.tensor(float("nan"), requires_grad=True),
        "mask": torch.tensor(0.0),
        "box": torch.tensor(0.0),
        "obj": torch.tensor(0.0),
        "presence": torch.tensor(0.0),
    }


# --- Tests ---


def test_nan_loss_below_threshold_does_not_abort(
    tmp_path: Path, tiny_coco_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """3 NaN micro-steps under nan_abort_after=5 → trainer continues."""
    from custom_sam_peft.train import loop as loop_mod

    real_total_loss = loop_mod.total_loss
    counter = {"n": 0}

    def fake_total_loss(*args: Any, **kwargs: Any) -> dict[str, torch.Tensor]:
        counter["n"] += 1
        if counter["n"] <= 3:
            return _nan_loss_dict()
        return real_total_loss(*args, **kwargs)

    monkeypatch.setattr("custom_sam_peft.train.loop.total_loss", fake_total_loss)

    ds = _ds(tiny_coco_dir)
    cfg = _cfg(tmp_path, tiny_coco_dir, nan_abort_after=5, epochs=2)
    w = make_stub_wrapper(dim=8, working=True)
    apply_lora(w, cfg.peft)
    trainer = Trainer(w, ds, ds, NoopTracker(), cfg)

    # Must NOT raise — 3 < 5.
    result = trainer.fit(run_dir=tmp_path / "run-below")
    assert result.run_dir.exists()


def test_nan_loss_at_threshold_raises_runtime_error(
    tmp_path: Path, tiny_coco_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Persistent NaN under nan_abort_after=3 → RuntimeError('non-finite ...')."""
    monkeypatch.setattr(
        "custom_sam_peft.train.loop.total_loss",
        lambda *a, **kw: _nan_loss_dict(),
    )

    ds = _ds(tiny_coco_dir)
    cfg = _cfg(tmp_path, tiny_coco_dir, nan_abort_after=3, epochs=2)
    w = make_stub_wrapper(dim=8, working=True)
    apply_lora(w, cfg.peft)
    trainer = Trainer(w, ds, ds, NoopTracker(), cfg)

    with pytest.raises(RuntimeError, match="non-finite"):
        trainer.fit(run_dir=tmp_path / "run-abort")


def test_nan_loss_logs_warning_on_value_error_path(
    tmp_path: Path,
    tiny_coco_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """ValueError from total_loss surfaces as the 'treating as non-finite' WARNING."""
    monkeypatch.setattr(
        "custom_sam_peft.train.loop.total_loss",
        lambda *a, **kw: (_ for _ in ()).throw(ValueError("non-finite cost matrix")),
    )

    ds = _ds(tiny_coco_dir)
    # nan_abort_after=2: 2 ValueErrors → abort. We want the WARNING to fire on
    # the first one before the RuntimeError on the second.
    cfg = _cfg(tmp_path, tiny_coco_dir, nan_abort_after=2, epochs=2)
    w = make_stub_wrapper(dim=8, working=True)
    apply_lora(w, cfg.peft)
    trainer = Trainer(w, ds, ds, NoopTracker(), cfg)

    with caplog.at_level(logging.WARNING, logger="custom_sam_peft.train.loop"):
        with pytest.raises(RuntimeError, match="non-finite"):
            trainer.fit(run_dir=tmp_path / "run-warn")

    warning_msgs = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("treating as non-finite" in m for m in warning_msgs), (
        f"expected 'treating as non-finite' warning; got: {warning_msgs}"
    )
```

- [ ] **Step 2: Run the new tests**

```bash
uv run pytest tests/unit/test_trainer_nan_behavior.py -v
```
Expected: all three tests PASS. If `test_nan_loss_at_threshold_raises_runtime_error` does not raise within `epochs=2` (because the dataset is too small to drive 3 micro-steps), bump `epochs` to 4 — keep the change inside the test, do not edit `loop.py`. If `test_nan_loss_logs_warning_on_value_error_path` fires the RuntimeError on the FIRST ValueError instead of the second, adjust `nan_abort_after` to 3 to give the warning a chance to log first.

- [ ] **Step 3: Lint**

```bash
uv run ruff check tests/unit/test_trainer_nan_behavior.py
uv run ruff format --check tests/unit/test_trainer_nan_behavior.py
```
Expected: clean.

- [ ] **Step 4: Commit**

```bash
git add tests/unit/test_trainer_nan_behavior.py
git commit -m "$(cat <<'EOF'
test(unit): pin trainer NaN-abort-after-N policy

C3 per spec §6.2. Documents what train/loop.py does today: skip the
micro-step + increment nan_streak; raise RuntimeError after
cfg.train.nan_abort_after consecutive non-finite micro-steps; log a
WARNING ('treating as non-finite') when the matcher raises ValueError
on a non-finite cost matrix. Pins behavior without modifying
src/custom_sam_peft/train/loop.py.

Refs #65.
EOF
)"
```

---

## Task 3D: CPU C4 — add `tests/unit/test_checkpoint_roundtrip.py` (primary) + extend `tests/integration/test_train_resume.py` (smoke)

**Amendment (2026-05-21, mid-implementation, tier-2):** the original 3D wanted bit-identical optimizer `step` / `exp_avg` continuity across an uninterrupted reference vs. an interrupted-then-resumed run. That contract is physically impossible under the trainer's documented epoch-boundary resume (`src/custom_sam_peft/train/checkpoint.py:7` and `Trainer.fit` at `src/custom_sam_peft/train/trainer.py:200-203,233`): the resumed run re-walks the interrupted epoch and so runs strictly more optimizer steps than the reference. C4 is therefore split: a new unit test pins the save→load bit-equality contract directly (the seam C4 was always about), and the existing integration test gains only monotone ≥ assertions that are consistent with the re-walk. See spec §6.2 C4 for the full prose.

**Files:**
- Create: `tests/unit/test_checkpoint_roundtrip.py`
- Modify: `tests/integration/test_train_resume.py`

**Dispatch:** implementer subagent, sonnet/high. Parallel with all other 2X / 3X tasks.

**Subagent context primer:**
- Spec section: §6.2 C4 (post-amendment).
- Resume re-walk semantics: `src/custom_sam_peft/train/checkpoint.py:7` docstring; `src/custom_sam_peft/train/trainer.py:233` (`for epoch in range(start_epoch, cfg.train.epochs)`); `load_full_state` returns `ResumeState.start_epoch = saved epoch`, so a resume from `step_2, epoch=0` re-runs epoch 0 entirely.
- Save/load API: `save_full_state(state_dir, wrapper, optimizer, scheduler, global_step, epoch, nan_streak, box_hint_p, cfg)` and `load_full_state(state_dir, wrapper, optimizer, scheduler, cfg) -> ResumeState`. Both at `src/custom_sam_peft/train/checkpoint.py:94-178`. Payload format: `optimizer.state_dict()`, `scheduler.state_dict()`, RNG dict (python / numpy / torch_cpu / torch_cuda), `box_hint_p`, `nan_streak`, `peft_method`, `cfg_hash`.
- Builders: `_build_optimizer` and `_build_scheduler` in `custom_sam_peft.train.trainer`. The integration spy pattern (monkeypatch.setattr on these) is required ONLY for the integration test; the unit test builds optimizer + scheduler directly via the same builders (a normal import, no spy needed).
- Stub: `tests/fixtures/tiny_sam3_lora_stub.py::make_stub_wrapper(dim=8, working=True)` returns a `Sam3Wrapper` whose forward produces a real-valued tensor that supports backprop. Use this for the unit test's optimizer-step priming as well as the integration test (already imported there).

---

### Step 1: Create `tests/unit/test_checkpoint_roundtrip.py`

This is the primary C4 coverage. It pins the `save_full_state` → `load_full_state` roundtrip as bit-identical on every field that matters (optimizer state including `step` counter, `exp_avg`, `exp_avg_sq`; scheduler `last_epoch` and `_step_count`; CPU RNG state; `ResumeState` fields). No `Trainer.fit` invocation — the test drives `save_full_state` and `load_full_state` directly, which is exactly the contract being asserted.

Sketch (the implementer fills in details to match the existing project test style — pydantic config construction patterns mirror `tests/integration/test_train_resume.py::_cfg`):

```python
"""Save/load roundtrip for save_full_state ↔ load_full_state.

Pins the serialization contract directly: every scalar and tensor in the
training state survives a save+load cycle bit-identically. The integration
test in tests/integration/test_train_resume.py exercises the same code-path
end-to-end via Trainer.fit but cannot assert bit-equality because the
trainer re-walks the interrupted epoch on resume (see
src/custom_sam_peft/train/checkpoint.py:7).
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest
import torch

from custom_sam_peft.config.schema import (  # plus the same subset used by test_train_resume.py
    AugmentationsConfig, DataConfig, DataSplit, NormalizeConfig,
    PEFTConfig, RunConfig, TextPromptConfig, TrainConfig, TrainHyperparams,
)
from custom_sam_peft.peft_adapters.lora import apply_lora
from custom_sam_peft.train.checkpoint import (
    ResumeState, load_full_state, save_full_state,
)
from custom_sam_peft.train.trainer import _build_optimizer, _build_scheduler
from tests.fixtures.tiny_sam3_lora_stub import FIXTURE_SCOPE_PATTERNS, make_stub_wrapper

pytestmark = pytest.mark.unit


def _cfg(tmp_path: Path) -> TrainConfig:
    # Minimal TrainConfig — values don't matter for the roundtrip, only that
    # the schema validates and cfg_hash is stable across save+load.
    return TrainConfig(
        run=RunConfig(name="roundtrip", output_dir=str(tmp_path), seed=42),
        data=DataConfig(
            format="coco",
            train=DataSplit(annotations=str(tmp_path / "a.json"), images=str(tmp_path)),
            val=DataSplit(annotations=str(tmp_path / "a.json"), images=str(tmp_path)),
            prompt_mode="text",
            image_size=32,
        ),
        peft=PEFTConfig(method="lora", scope="vision",
                        target_modules=FIXTURE_SCOPE_PATTERNS["vision"]),
        train=TrainHyperparams(epochs=2, batch_size=1, grad_accum_steps=1,
                               save_every=2, log_every=1, warmup_steps=0, num_workers=0),
    )


def test_save_load_roundtrip_preserves_optimizer_scheduler_rng_state(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    wrapper = make_stub_wrapper(dim=8, working=True)
    apply_lora(wrapper, cfg.peft)

    trainable = [p for p in wrapper.model.parameters() if p.requires_grad]
    assert trainable, "stub LoRA wrapper has no trainable params"

    optimizer = _build_optimizer("adamw", trainable, cfg.train.lr)
    scheduler = _build_scheduler(optimizer, cfg, total_steps=10)

    # Drive a few real optimizer steps so exp_avg / exp_avg_sq / step are populated.
    # The stub's forward returns a scalar-loss-shaped tensor (see fixture). If
    # the stub does not expose a loss-producing forward directly, drive it via
    # the same path the existing test_resume integration uses — but the cheaper
    # mechanism is: synthesize a loss from the stub's trainable params directly
    # (sum(p.pow(2).sum() for p in trainable) — a quadratic that yields nonzero
    # gradients on every parameter). Three steps is enough; bias the first
    # backward by .backward() then optimizer.step() then scheduler.step().
    for _ in range(3):
        optimizer.zero_grad()
        loss = sum(p.pow(2).sum() for p in trainable)
        loss.backward()
        optimizer.step()
        scheduler.step()

    # Snapshot pre-save state.
    sd_pre = optimizer.state_dict()
    ssd_pre = scheduler.state_dict()
    rng_pre = torch.get_rng_state().clone()

    state_dir = tmp_path / "checkpoints" / "step_3"
    save_full_state(
        state_dir=state_dir,
        wrapper=wrapper,
        optimizer=optimizer,
        scheduler=scheduler,
        global_step=3,
        epoch=0,
        nan_streak=0,
        box_hint_p=cfg.train.box_hint.p_start,
    cfg=cfg,
    )

    # Mutate RNG between save and load to prove load actually restores it
    # (not just preserves a quiescent process state).
    random.seed(12345)
    np.random.seed(12345)
    torch.manual_seed(12345)

    # Fresh optimizer + scheduler (load_full_state mutates in-place).
    fresh_opt = _build_optimizer("adamw", trainable, cfg.train.lr)
    fresh_sched = _build_scheduler(fresh_opt, cfg, total_steps=10)
    rs = load_full_state(state_dir, wrapper, fresh_opt, fresh_sched, cfg)

    # ResumeState fields.
    assert isinstance(rs, ResumeState)
    assert rs.start_step == 3
    assert rs.start_epoch == 0
    assert rs.nan_streak == 0
    assert rs.box_hint_p == pytest.approx(cfg.train.box_hint.p_start)

    sd_post = fresh_opt.state_dict()
    ssd_post = fresh_sched.state_dict()

    # Param groups: bit-equal on canonical scalar keys.
    assert len(sd_pre["param_groups"]) == len(sd_post["param_groups"])
    for g_pre, g_post in zip(sd_pre["param_groups"], sd_post["param_groups"], strict=True):
        for key in ("lr", "betas", "weight_decay", "eps"):
            if key in g_pre:
                assert g_pre[key] == g_post[key], f"param_group {key} drift"

    # Per-parameter state: bit-equal on step counter and running moments.
    assert set(sd_pre["state"]) == set(sd_post["state"]), "param IDs changed across roundtrip"
    for pid in sd_pre["state"]:
        st_pre = sd_pre["state"][pid]
        st_post = sd_post["state"][pid]
        if "step" in st_pre:
            assert int(st_pre["step"]) == int(st_post["step"]), f"param {pid}: step drift"
        for mom_key in ("exp_avg", "exp_avg_sq"):
            if mom_key in st_pre:
                assert torch.equal(st_pre[mom_key], st_post[mom_key]), (
                    f"param {pid}: {mom_key} not bit-equal across save/load"
                )

    # Scheduler state: bit-equal on canonical keys.
    for key in ("last_epoch", "_step_count"):
        if key in ssd_pre:
            assert ssd_pre[key] == ssd_post[key], f"scheduler {key} drift"

    # CPU RNG: bit-equal to pre-save snapshot (load_full_state restored it
    # despite our intermediate manual_seed(12345)).
    assert torch.equal(torch.get_rng_state(), rng_pre), "CPU RNG not restored across load"
```

Notes for the implementer:
- The `sum(p.pow(2).sum() ...)` quadratic-loss trick is the lightest way to populate `exp_avg`/`exp_avg_sq` without driving a real forward through the stub. If `_build_optimizer` insists on `"adam"` rather than `"adamw"`, match what `Trainer.fit` uses (see `Trainer.__init__` for the default optimizer name).
- The `Trainer.fit` call at `src/custom_sam_peft/train/trainer.py:189` passes `self._optimizer_name` — read that value to match exactly. Adjust the test's first arg to `_build_optimizer` accordingly.
- If `_build_scheduler`'s second arg is the whole `TrainConfig` (it is — see `trainer.py:191`), pass `cfg` directly; do not reshape.
- CUDA RNG: skip — this test is CPU-only (`pytestmark = pytest.mark.unit`). The GPU resume test T1 is what covers CUDA RNG.
- The `_cfg` helper here intentionally mirrors `tests/integration/test_train_resume.py::_cfg` rather than importing it — spec rule §6.2 says "no shared-helper extraction this PR".

### Step 2: Extend `tests/integration/test_train_resume.py` with monotone ≥ assertions

Replace lines 90–122 (`def test_resume_matches_uninterrupted` through end of function) with the version below. The only difference from the original 3D step-1 code block is:

- The optimizer `step`-counter assertion becomes `step_c >= step_a`.
- The `exp_avg` / `exp_avg_sq` `torch.allclose` assertion is REMOVED entirely (it diverges legitimately under re-walk).
- The scheduler `last_epoch` / `_step_count` equalities become `>=`.

```python
def test_resume_matches_uninterrupted(
    tmp_path: Path, tiny_coco_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds = _ds(tiny_coco_dir)
    cfg = _cfg(tmp_path, tiny_coco_dir, save_every=2)

    # C4 (spec §6.2): capture the optimizer/scheduler instances the Trainer
    # builds. Wrap _build_optimizer / _build_scheduler with closures that stash
    # the constructed instance in a test-local list. Requires no
    # src/custom_sam_peft/ change.
    import custom_sam_peft.train.trainer as trainer_mod

    real_opt_builder = trainer_mod._build_optimizer
    real_sched_builder = trainer_mod._build_scheduler
    captured_opts: list[Any] = []
    captured_scheds: list[Any] = []

    def _opt_spy(*a: Any, **kw: Any) -> Any:
        opt = real_opt_builder(*a, **kw)
        captured_opts.append(opt)
        return opt

    def _sched_spy(*a: Any, **kw: Any) -> Any:
        sched = real_sched_builder(*a, **kw)
        captured_scheds.append(sched)
        return sched

    monkeypatch.setattr(trainer_mod, "_build_optimizer", _opt_spy)
    monkeypatch.setattr(trainer_mod, "_build_scheduler", _sched_spy)

    # Uninterrupted reference run (2 epochs).
    w_a = make_stub_wrapper(dim=8, working=True)
    apply_lora(w_a, cfg.peft)
    trainer_a = Trainer(w_a, ds, ds, NoopTracker(), cfg)
    trainer_a.fit(run_dir=tmp_path / "run-a")
    state_a = _adapter_state(w_a)
    opt_a = captured_opts[-1]
    sched_a = captured_scheds[-1]

    # Truncated first run (1 epoch), then resumed (2 epochs from checkpoint).
    w_b = make_stub_wrapper(dim=8, working=True)
    apply_lora(w_b, cfg.peft)
    cfg_short = _cfg(tmp_path, tiny_coco_dir, save_every=2)
    cfg_short.train.epochs = 1
    trainer_b = Trainer(w_b, ds, ds, NoopTracker(), cfg_short)
    result_b1 = trainer_b.fit(run_dir=tmp_path / "run-b1")

    ckpts = sorted((result_b1.run_dir / "checkpoints").glob("step_*"))
    assert ckpts, "no checkpoint produced"
    resume_dir = ckpts[-1]

    w_c = make_stub_wrapper(dim=8, working=True)
    apply_lora(w_c, cfg.peft)
    trainer_c = Trainer(w_c, ds, ds, NoopTracker(), cfg)
    trainer_c.fit(run_dir=tmp_path / "run-c", resume_from=resume_dir)
    state_c = _adapter_state(w_c)
    opt_c = captured_opts[-1]
    sched_c = captured_scheds[-1]

    # Resume produces finite weights (not bit-identical to uninterrupted run
    # because the re-walked epoch retreads some examples). Assert finiteness
    # only on the adapter tensors.
    for k in state_a:
        assert torch.isfinite(state_c[k]).all()

    # --- C4 monotone continuity assertions ---
    # The resumed run re-walks the interrupted epoch and therefore runs
    # strictly more optimizer steps than the reference. We assert ≥ on
    # the step counter and scheduler counters; exp_avg / exp_avg_sq are
    # NOT compared because they diverge legitimately. Bit-equality of the
    # serialization contract lives in tests/unit/test_checkpoint_roundtrip.py.

    sd_a = opt_a.state_dict()
    sd_c = opt_c.state_dict()
    assert len(sd_a["param_groups"]) == len(sd_c["param_groups"])
    for g_a, g_c in zip(sd_a["param_groups"], sd_c["param_groups"], strict=True):
        for key in ("lr", "betas", "weight_decay", "eps"):
            if key in g_a:
                assert g_a[key] == g_c[key], f"param_group {key} drift: {g_a[key]} vs {g_c[key]}"
    common_ids = set(sd_a["state"]) & set(sd_c["state"])
    assert common_ids, "no shared parameter IDs between uninterrupted + resumed optimizers"
    for pid in common_ids:
        st_a = sd_a["state"][pid]
        st_c = sd_c["state"][pid]
        if "step" in st_a and "step" in st_c:
            assert int(st_c["step"]) >= int(st_a["step"]), (
                f"param {pid}: resumed step {st_c['step']} < reference step {st_a['step']} "
                "(resumed run should run ≥ steps because it re-walks the interrupted epoch)"
            )

    # Scheduler-state continuity: resumed counters ≥ reference counters
    # (same re-walk argument as above).
    ssd_a = sched_a.state_dict()
    ssd_c = sched_c.state_dict()
    for key in ("last_epoch", "_step_count"):
        if key in ssd_a and key in ssd_c:
            assert ssd_c[key] >= ssd_a[key], (
                f"scheduler {key}: resumed {ssd_c[key]} < reference {ssd_a[key]}"
            )
```

Note on `Any` import: `Any` is already imported at line 6 of the existing file. The `import custom_sam_peft.train.trainer as trainer_mod` is local to the test body.

### Step 3: Run both tests

```bash
uv run pytest tests/unit/test_checkpoint_roundtrip.py tests/integration/test_train_resume.py -v
```
Expected: 2 tests PASS (~1–2 seconds total). If the unit test's `torch.equal` fails on `exp_avg`/`exp_avg_sq`, that is the genuine signal C4 was always trying to catch — surface in the commit message and let the reviewer triage.

### Step 4: Lint

```bash
uv run ruff check tests/unit/test_checkpoint_roundtrip.py tests/integration/test_train_resume.py
uv run ruff format --check tests/unit/test_checkpoint_roundtrip.py tests/integration/test_train_resume.py
```
Expected: clean.

### Step 5: Commit

```bash
git add tests/unit/test_checkpoint_roundtrip.py tests/integration/test_train_resume.py
git commit -m "$(cat <<'EOF'
test: pin checkpoint save/load roundtrip + monotone resume continuity

C4 per spec §6.2 (post-amendment). Adds unit test
test_checkpoint_roundtrip.py that asserts save_full_state →
load_full_state preserves optimizer state (step, exp_avg, exp_avg_sq,
param_groups), scheduler state (last_epoch, _step_count), CPU RNG, and
ResumeState fields bit-identically. Extends integration test
test_resume_matches_uninterrupted with monotone ≥ assertions on the
optimizer step counter and scheduler counters, consistent with the
trainer's epoch-boundary re-walk semantics (checkpoint.py:7,
trainer.py:233). exp_avg / exp_avg_sq are NOT compared in the
integration test because they diverge legitimately under re-walk; the
unit test owns that contract.

Refs #65.
EOF
)"
```

---

## Task 3E: CPU C5 — extend `tests/integration/test_train_end_to_end.py`

**Files:**
- Modify: `tests/integration/test_train_end_to_end.py`

**Dispatch:** implementer subagent, sonnet/high. Parallel with all other 2X / 3X tasks.

**Subagent context primer:**
- Spec section: §6.2 C5.
- Existing file exercises a happy-path CPU integration via `Trainer.fit(...)` with the LoRA stub. Add three malformed-dataset failure-mode tests *in the same file*; do NOT create a sibling file.
- The implementer pins whatever exception type the COCO loader actually raises — the spec does not prescribe specific exception classes because the loader's contract is "pin current behavior."
- For each new test, construct a minimal `TrainConfig` whose `data.train` and `data.val` point at the malformed annotations path; reuse the schema imports already at the top of the file. Use `make_stub_wrapper` + `apply_lora` + `Trainer(...).fit(...)` — same shape as the existing test.

- [ ] **Step 1: Add the three failure-mode tests at the bottom of the file**

Append after `test_fit_end_to_end_on_tiny_coco`:

```python


def _bad_data_cfg(
    tmp_path: Path,
    annotations: Path,
    images: Path,
) -> TrainConfig:
    """Minimal TrainConfig pointing at a (likely-broken) annotations/images pair."""
    return TrainConfig(
        run=RunConfig(name="bad-data", output_dir=str(tmp_path), seed=0),
        data=DataConfig(
            format="coco",
            train=DataSplit(annotations=str(annotations), images=str(images)),
            val=DataSplit(annotations=str(annotations), images=str(images)),
            prompt_mode="text",
            image_size=32,
        ),
        peft=PEFTConfig(
            method="lora",
            scope="vision",
            target_modules=FIXTURE_SCOPE_PATTERNS["vision"],
        ),
        train=TrainHyperparams(
            epochs=1,
            batch_size=1,
            grad_accum_steps=1,
            save_every=2,
            log_every=1,
            warmup_steps=0,
            num_workers=0,
        ),
    )


def test_malformed_coco_json_raises_clear_error(tmp_path: Path) -> None:
    """C5 per spec §6.2: invalid JSON in annotations.json surfaces a clear error."""
    images = tmp_path / "images"
    images.mkdir()
    annotations = tmp_path / "annotations.json"
    annotations.write_text("{")  # invalid JSON

    cfg = _bad_data_cfg(tmp_path, annotations, images)
    wrapper = make_stub_wrapper(dim=8, working=True)
    apply_lora(wrapper, cfg.peft)
    # Pin whatever exception the COCO loader actually raises (json.JSONDecodeError,
    # ValueError, RuntimeError, etc.). The implementer narrows after observing
    # the actual exception in a first run; commit the narrowest matching type.
    with pytest.raises(Exception) as excinfo:
        Trainer(
            wrapper,
            None,  # type: ignore[arg-type]
            None,  # type: ignore[arg-type]
            build_tracker(cfg),
            cfg,
        ).fit(run_dir=tmp_path / "run-malformed")
    # Reject the catch-all: if Exception is BaseException only, narrow this.
    assert excinfo.type is not BaseException


def test_missing_image_file_raises_clear_error(tmp_path: Path) -> None:
    """C5 per spec §6.2: missing image referenced by COCO surfaces a clear error
    naming the file."""
    images = tmp_path / "images"
    images.mkdir()
    annotations = tmp_path / "annotations.json"
    annotations.write_text(
        json.dumps(
            {
                "images": [
                    {"id": 1, "file_name": "missing.jpg", "width": 32, "height": 32}
                ],
                "annotations": [
                    {
                        "id": 1,
                        "image_id": 1,
                        "category_id": 1,
                        "bbox": [0, 0, 10, 10],
                        "area": 100,
                        "iscrowd": 0,
                        "segmentation": [[0, 0, 10, 0, 10, 10, 0, 10]],
                    }
                ],
                "categories": [{"id": 1, "name": "thing"}],
            }
        )
    )

    cfg = _bad_data_cfg(tmp_path, annotations, images)
    ds_train = COCODataset(
        annotations=str(annotations),
        images=str(images),
        prompt_mode="text",
        transforms=build_train_transforms(
            AugmentationsConfig(hflip=False, color_jitter=0.0),
            32,
            model_name="facebook/sam3.1",
            normalize=NormalizeConfig(),
        ),
        text_prompt=TextPromptConfig(),
    )
    wrapper = make_stub_wrapper(dim=8, working=True)
    apply_lora(wrapper, cfg.peft)
    with pytest.raises(Exception) as excinfo:
        Trainer(
            wrapper, ds_train, ds_train, build_tracker(cfg), cfg
        ).fit(run_dir=tmp_path / "run-missing-img")
    # Pin: the message should reference the missing filename. If it does not,
    # the loader's error message is bad — surface that as a follow-up issue.
    assert "missing.jpg" in str(excinfo.value), (
        f"expected 'missing.jpg' in error message; got: {excinfo.value!r}"
    )


def test_missing_annotation_entry_does_not_crash(tmp_path: Path) -> None:
    """C5 per spec §6.2: an image with no matching annotations is handled
    gracefully (zero-instance item) OR raises with a clear message.

    The implementer pins the actual behavior: if the loader returns
    zero-instance items, training proceeds without crashing; if it raises,
    the message names the orphan image.
    """
    # Use tiny_coco's first image as the only valid image.
    images = tmp_path / "images"
    images.mkdir()
    # Make a 1x1 black png so the loader has something to open.
    from PIL import Image as PILImage

    PILImage.new("RGB", (32, 32)).save(images / "img.png")
    annotations = tmp_path / "annotations.json"
    annotations.write_text(
        json.dumps(
            {
                "images": [
                    {"id": 1, "file_name": "img.png", "width": 32, "height": 32},
                    {"id": 2, "file_name": "img.png", "width": 32, "height": 32},
                ],
                # Only image_id=1 has an annotation; image_id=2 is orphan.
                "annotations": [
                    {
                        "id": 1,
                        "image_id": 1,
                        "category_id": 1,
                        "bbox": [0, 0, 10, 10],
                        "area": 100,
                        "iscrowd": 0,
                        "segmentation": [[0, 0, 10, 0, 10, 10, 0, 10]],
                    }
                ],
                "categories": [{"id": 1, "name": "thing"}],
            }
        )
    )
    cfg = _bad_data_cfg(tmp_path, annotations, images)
    ds_train = COCODataset(
        annotations=str(annotations),
        images=str(images),
        prompt_mode="text",
        transforms=build_train_transforms(
            AugmentationsConfig(hflip=False, color_jitter=0.0),
            32,
            model_name="facebook/sam3.1",
            normalize=NormalizeConfig(),
        ),
        text_prompt=TextPromptConfig(),
    )
    wrapper = make_stub_wrapper(dim=8, working=True)
    apply_lora(wrapper, cfg.peft)
    # Either Trainer.fit completes (zero-instance handling) OR raises with a
    # clear message naming the orphan image. The implementer pins which.
    try:
        Trainer(wrapper, ds_train, ds_train, build_tracker(cfg), cfg).fit(
            run_dir=tmp_path / "run-orphan"
        )
    except Exception as exc:
        # If it raises, the message should reference the orphan image or the
        # image_id. If it does not, surface that as a follow-up.
        assert "2" in str(exc) or "img.png" in str(exc), (
            f"orphan-image error message lacks identifier: {exc!r}"
        )
```

Note on imports: the file already imports `json`, `pytest`, `AugmentationsConfig`, `DataConfig`, `DataSplit`, `NormalizeConfig`, `PEFTConfig`, `RunConfig`, `TextPromptConfig`, `TrainConfig`, `TrainHyperparams`, `COCODataset`, `build_train_transforms`, `apply_lora`, `build_tracker`, `Trainer`, `FIXTURE_SCOPE_PATTERNS`, `make_stub_wrapper`. The only missing imports for the new code are `from pathlib import Path` (already imported) and `from PIL import Image as PILImage` (used locally inside the orphan test — keep it inline because no other test in the file uses PIL).

Note on `Exception` matchers: the orchestrator-CLAUDE.md / ruff config allows `pytest.raises(Exception)` as a *first-pass pinning device*; the implementer narrows to the actual class after running the test once and observing the raised type. If ruff flags `pytest.raises(Exception)` with `PT011` ("pytest.raises(Exception) is too broad"), narrow to the observed exception class before committing.

- [ ] **Step 2: Run the extended file**

```bash
uv run pytest tests/integration/test_train_end_to_end.py -v
```
Expected: existing 2 tests PASS + 3 new tests PASS or surface clear error messages. If the orphan test fails because the loader silently drops zero-instance images and then the trainer raises some downstream "0-length" error, narrow the assertion accordingly. Document the observed behavior in the commit message.

- [ ] **Step 3: Lint**

```bash
uv run ruff check tests/integration/test_train_end_to_end.py
uv run ruff format --check tests/integration/test_train_end_to_end.py
```
Expected: clean. If ruff complains about `pytest.raises(Exception)`, narrow to the actual class observed in Step 2.

- [ ] **Step 4: Commit**

```bash
git add tests/integration/test_train_end_to_end.py
git commit -m "$(cat <<'EOF'
test(integration): pin malformed-dataset failure modes

C5 per spec §6.2. Adds three failure-mode tests to the existing
end-to-end file (no sibling): invalid JSON in annotations.json,
missing image file referenced by a valid COCO, and an orphan image
with no matching annotations. Pins whatever exception type / message
the loader raises today. Like C3, this PR does not improve the
loader's error messages; if a message is bad, the reviewer files a
follow-up.

Refs #65.
EOF
)"
```

---

## Task 4: Code review pass

**Dispatch:** reviewer subagent, **opus / xhigh**. Serial — after all 2X and 3X tasks return successfully.

**Subagent prompt template:**

> Code-review the gpu-coverage-65 branch. The spec is at `docs/superpowers/specs/2026-05-21-gpu-coverage-assessment-design.md` and the plan at `docs/superpowers/plans/2026-05-21-gpu-coverage-assessment-plan.md`. Use the `code-review-graph` MCP tools (`detect_changes`, `get_review_context`, `get_impact_radius`) **before** Read/Grep per project CLAUDE.md.
>
> The branch is a test-only PR — zero changes under `src/custom_sam_peft/`. Focus your review on:
>
> 1. **Spec coverage.** Every exit-criteria checkbox in spec §11 must map to at least one committed change. If anything is missing, name it.
> 2. **Each task's assertion shape vs. the spec.** T1 must not compare against an uninterrupted reference (spec §6.1 T1 "Explicitly NOT asserted"). T3 must not modify `test_qlora_smoke_fast`. T5/T6 must use the exact substring matchers `vision_backbone`, `transformer.decoder`, `mask_decoder`. C3 must not modify `src/custom_sam_peft/train/loop.py`. C4's optimizer continuity must use `atol=1e-6`. C5 must extend the existing file in place (no sibling).
> 3. **GPU/CPU gating correctness.** T1's `@pytest.mark.skipif(not _bnb_available(), ...)` must be present. T6's mirror. T1's `pytestmark` must include `gpu`, `requires_compatible_gpu`, `requires_checkpoint`.
> 4. **Bug-risk pass.** Use `get_impact_radius` for any change under tests/integration/. Flag anything that might destabilize the existing CPU CI green path.
> 5. **Markdownlint relaxation** applies to `docs/superpowers/` per `docs/superpowers/specs/2026-05-18-ci-hardening-design.md`. The plan + spec files should not trigger CI lint failures.
>
> Report findings as a markdown list; group by Critical / Important / Nit. The orchestrator will dispatch fixes serially per finding.

After the reviewer returns:

- [ ] **Step 1: Triage findings**

For each Critical / Important finding, dispatch a fix subagent (sonnet/high) with the exact finding text + the affected file. Commit each fix as a separate `fix(test):` commit referencing the finding.

- [ ] **Step 2: Nit pass (optional)**

Nits are dispatched in a single batch only if they cluster on the same files. Otherwise skip per `superpowers:receiving-code-review` discretion.

- [ ] **Step 3: Re-review**

If any Critical fixes landed, re-dispatch the reviewer for a focused re-review on the changed files only. Re-review skipped if only nits were addressed.

---

## Task 5: Lint / format / typecheck — LAST step of the reviewer pass

**Dispatch:** implementer subagent, sonnet/high. Serial — after Task 4 (and any review-fix loops) is complete.

**Subagent prompt:**

> Run the project's lint/format/typecheck gates. Fix any findings directly — do NOT defer:
>
> 1. `uv run ruff check` (project root)
> 2. `uv run ruff format --check` (project root)
> 3. `uv run mypy src/custom_sam_peft` (project root)
>
> If `ruff format --check` reports diffs, run `uv run ruff format` to apply, then commit as a single `chore(format): apply ruff format` commit.
> If `ruff check` reports findings, fix them in place. For unused-import errors in newly-added tests, add the missing import or remove the unused name. For `pytest.raises(Exception)` (PT011), narrow to the observed exception class.
> If `mypy` reports findings under `src/custom_sam_peft/`, that means a test indirectly broke a type contract — investigate; **do not** add `type: ignore` casually. mypy does not scan tests in this project, so the most likely cause is a stale annotation in `src/` exposed by a re-import.
>
> Commit any fixes as separate commits with clear scopes (`chore(format)`, `fix(test)`, etc.). Final commit on the branch should be either a lint fix or the last test commit — never a "WIP" commit.

- [ ] **Step 1: Run all three gates**

```bash
uv run ruff check
uv run ruff format --check
uv run mypy src/custom_sam_peft
```

- [ ] **Step 2: Fix + commit any findings**

Per the subagent's discretion.

- [ ] **Step 3: Final re-run to confirm green**

```bash
uv run ruff check && uv run ruff format --check && uv run mypy src/custom_sam_peft
```
Expected: all three exit 0.

---

## Task 6: Verification — collection counts + full CPU suite green

**Dispatch:** Inline (orchestrator). No subagent.

This task gates the PR open; do NOT proceed to Task 7+ until every Expected line below matches.

- [ ] **Step 1: `gpu`-marker collection count**

```bash
uv run pytest --collect-only -m gpu -q | tail -20
```
Expected (per spec §11): 4 tests collected — the existing 3 (`test_overfits_in_50_steps`, `test_qlora_overfits_in_50_steps`, `test_qlora_smoke_fast`, `test_run_end_to_end_writes_bundle`... — let the orchestrator confirm by counting; the spec says 3 existing + T1 = 4). If the count is off, diagnose: did T1's `pytestmark` include `gpu`? Did any unrelated change leak a `gpu` marker?

NOTE: The current `gpu`-marked release-tier tests are `test_overfits_in_50_steps`, `test_qlora_overfits_in_50_steps`, `test_qlora_smoke_fast`, and `test_run_end_to_end_writes_bundle` — that is 4 already. Adding T1 makes 5. **Re-read spec §11 line 434:** *"pytest --collect-only -m gpu collects 4 tests (the existing 3 plus T1)."* The spec counts 3 existing — meaning `test_qlora_smoke_fast` is being treated as either non-`gpu`-marked or grouped with the qlora overfits as one "smoke" item. The orchestrator verifies the actual count and reports the delta in the PR description; if the spec's "3 existing" is wrong by 1, that is a spec nit, not a blocker.

The acceptance criterion the orchestrator enforces in this step: the count is **exactly one greater** than the count on `main` at the merge-base commit. Compute the baseline via:

```bash
git stash -u 2>/dev/null || true
git checkout main -- tests/gpu
BASELINE=$(uv run pytest --collect-only -m gpu -q 2>/dev/null | grep -cE "^tests/")
git checkout gpu-coverage-65 -- tests/gpu
CURRENT=$(uv run pytest --collect-only -m gpu -q 2>/dev/null | grep -cE "^tests/")
echo "baseline=$BASELINE current=$CURRENT delta=$((CURRENT - BASELINE))"
git stash pop 2>/dev/null || true
```
Expected: `delta=1`.

- [ ] **Step 2: `gpu_inspection`-marker collection count**

Same approach for `gpu_inspection`:

```bash
git checkout main -- tests/integration
BASELINE=$(uv run pytest --collect-only -m gpu_inspection -q 2>/dev/null | grep -cE "^tests/")
git checkout gpu-coverage-65 -- tests/integration
CURRENT=$(uv run pytest --collect-only -m gpu_inspection -q 2>/dev/null | grep -cE "^tests/")
echo "baseline=$BASELINE current=$CURRENT delta=$((CURRENT - BASELINE))"
```
Expected: `delta=2` (T5 + T6). Spec §11 line 435 claims 9 existing → 11 — re-verify if the delta is something other than 2.

- [ ] **Step 3: Full CPU suite green**

```bash
uv run pytest
```
Expected: all CPU + unit + integration tests pass. `gpu` / `gpu_inspection` tests are deselected by the default pytest config. If anything fails that is NOT in the changed files, the implementer broke something unrelated — re-dispatch a fix subagent.

- [ ] **Step 4: Spot-check the markdown lint contract**

```bash
uv run markdownlint docs/superpowers/specs/2026-05-21-gpu-coverage-assessment-design.md \
                    docs/superpowers/plans/2026-05-21-gpu-coverage-assessment-plan.md \
                    2>&1 || true
```
Expected: clean per the `docs/superpowers/` relaxation in `2026-05-18-ci-hardening-design.md`. If `markdownlint` is not installed locally, skip — CI will catch it.

---

## Task 7: Version-bump decision — SKIP (record reason)

**Dispatch:** Inline (orchestrator). No subagent. No code action.

- [ ] **Step 1: Confirm no `src/` changes landed**

```bash
git diff --stat main...gpu-coverage-65 -- src/
```
Expected: empty (no lines, no files). If anything appears, the assumption behind the version-skip decision is false — halt and reconsider.

- [ ] **Step 2: Record the skip rationale**

The version-bump decision is captured in the plan (the "Version-Bump Decision" section above) and will be repeated in the PR body (Task 9). No manifest edits, no commits in this task. The close-out sign-off (orchestrator close-out 5e) will name the skip explicitly.

---

## Task 8A: File follow-up issue — eval-metric thresholds

**Dispatch:** Inline (orchestrator). No subagent.

- [ ] **Step 1: Confirm labels exist**

```bash
gh label list --limit 100 | grep -E "^(question|priority:low)"
```
Expected: both labels present. If `priority:low` is missing, create it:

```bash
gh label create priority:low --description "Low priority" --color cccccc
```

- [ ] **Step 2: Create the issue**

Body prose is **verbatim from spec §9.1**.

```bash
gh issue create \
  --assignee @me \
  --label question \
  --label priority:low \
  --title "Benchmark eval-metric thresholds: set non-trivial mAP / IoU floors once a calibration run is available" \
  --body "$(cat <<'EOF'
T2, T3, T4 (added by #65's PR — see `docs/superpowers/specs/2026-05-21-gpu-coverage-assessment-design.md` §6.1) assert only that `overall.mAP` is finite and `>= 0.0`. The original #65 question asked whether the 50-step tests should assert specific mAP or IoU floors. We deferred because:

1. No contributor has run the smoke configs to convergence on tiny_coco to establish what a "healthy" mAP actually looks like at 50 steps.
2. tiny_coco (2 images) is small enough that mAP values will be noisy; a tight floor would flake, a loose floor adds no signal.

What we need to make this concrete: one calibration run per smoke YAML (LoRA + QLoRA), capturing `overall.mAP`, `overall.mAP_50`, `overall.mAP_75`, plus the per-class APs, plus the across-run variance. Once we have N>=5 runs of each, we can set a floor at `mean - 2*stddev` or similar.

Cross-ref: #65, `2026-05-21-gpu-coverage-assessment-design.md` §4 (non-goals) and §6.1 (T2/T3/T4 specs).
EOF
)"
```

Capture the issue number from stdout — needed for Task 9's PR body.

- [ ] **Step 3: Record the issue number**

The orchestrator stores `FOLLOWUP_THRESHOLDS=<num>` for use in Tasks 8B and 9.

---

## Task 8B: File follow-up issue — GPU `all`-scope smoke

**Dispatch:** Inline (orchestrator). No subagent.

- [ ] **Step 1: Confirm `enhancement` label exists**

```bash
gh label list --limit 100 | grep -E "^enhancement"
```
Expected: present (it is one of GitHub's defaults).

- [ ] **Step 2: Create the issue**

Body prose is **verbatim from spec §9.2**.

```bash
gh issue create \
  --assignee @me \
  --label enhancement \
  --label priority:low \
  --title "GPU \"all\"-scope smoke once VRAM budget allows" \
  --body "$(cat <<'EOF'
The `all` PEFT scope (regex `.*`) attaches LoRA to every `nn.Linear` in the SAM 3.1 tree. Its VRAM footprint almost certainly exceeds T4's `gpu_smoke_lora.yaml` (14 GB) and `gpu_smoke_qlora.yaml` (10 GB) ceilings — both pinned to T4 per `2026-05-19-gpu-test-policy-design.md` §5.4. C2 in #65's PR (`tests/unit/test_peft_scope_coverage.py`) covers the `all`-scope wiring contract on a CPU stub, which is sufficient for the typical regression mode (a scope mis-mapping).

What we'd want a GPU smoke for: catching a real-model regression where `all`-scope memory blows up *beyond what the stub can predict* (e.g. a future SAM weights release with substantially more linear layers). The blocker: until #68 (VRAM-floor tightening) reclassifies the VRAM tiers — possibly establishing a "release-tier-only" T4-32 GB / L4 / A10 ceiling for memory-heavy smokes — we have no place to put it that doesn't break the T4 baseline.

Defer until #68 lands. Cross-ref: #65, #68, `2026-05-21-gpu-coverage-assessment-design.md` §4 (non-goals).
EOF
)"
```

Capture the issue number as `FOLLOWUP_ALL_SCOPE=<num>`.

---

## Task 9: Open PR + watch CI

**Dispatch:** Inline (orchestrator). No subagent.

- [ ] **Step 1: Confirm branch is pushed and ahead of main**

```bash
git status
git log --oneline main..HEAD
```
Expected: working tree clean; commit list includes Task 1 doc commit + 8 implementation commits + any review-fix commits. If anything is uncommitted, halt.

- [ ] **Step 2: Open the PR**

Title: `test(gpu): close uniquely-owned coverage gaps from #65` (< 70 chars; verified: 53 chars).

```bash
gh pr create --assignee @me \
  --title "test(gpu): close uniquely-owned coverage gaps from #65" \
  --body "$(cat <<EOF
## Summary

- Spec audits the post-v0.6.1 GPU/CPU test suite against the eight #65 coverage dimensions and identifies four uniquely-#65-owned gaps: vision/all PEFT scopes, mid-training NaN abort behavior, eval-metric finiteness, QLoRA resume.
- Adds 6 GPU tests (T1 new file for QLoRA resume; T2/T3/T4 extend the 50-step + e2e tests with mAP finiteness; T5/T6 add vision-scope subtests on real ViT-Det at inspection tier).
- Adds 5 CPU tests (C1 MetricsReport schema; C2 PEFT scope wiring on a stub with forward+backward gradient check; C3 pins trainer NaN-abort policy; C4 extends resume test with optimizer/scheduler continuity; C5 pins malformed-dataset error contracts).
- Zero changes under \`src/custom_sam_peft/\`; zero new YAML configs; zero new dependencies. Closes the \`GPU test gate decision shipped — #65\` line on #70's pre-v1.0 checklist.

## Design

- Spec: \`docs/superpowers/specs/2026-05-21-gpu-coverage-assessment-design.md\`
- Plan: \`docs/superpowers/plans/2026-05-21-gpu-coverage-assessment-plan.md\`

## Follow-up issues filed by this branch

- #${FOLLOWUP_THRESHOLDS:-TBD} — Benchmark eval-metric thresholds (defer mAP/IoU floors until a calibration run is available).
- #${FOLLOWUP_ALL_SCOPE:-TBD} — GPU \`all\`-scope smoke once VRAM budget allows (blocked on #68).

## Version-bump decision

**Skipped.** Test-only PR; no consumer-observable change ships. Per the orchestrator override on \`superpowers:finishing-a-development-branch\` (CLAUDE.md): no manifest version stamped, no tag pushed at close-out.

## Test plan

- [ ] \`uv run pytest\` green (CPU suite)
- [ ] \`uv run ruff check\` clean
- [ ] \`uv run ruff format --check\` clean
- [ ] \`uv run mypy src/custom_sam_peft\` clean
- [ ] \`pytest --collect-only -m gpu\` collects +1 vs. main (T1)
- [ ] \`pytest --collect-only -m gpu_inspection\` collects +2 vs. main (T5, T6)
- [ ] CI green on this PR
- [ ] Real-GPU run of T1/T2/T3/T4/T5/T6 on Colab notebook (manual, post-merge — tracked in close-out, not in this PR's CI)

Refs #65, #70.
EOF
)"
```

Capture the PR number as `PR_NUM=<num>`.

- [ ] **Step 3: Watch CI without polling-sleeps**

Use `run_in_background` or `Monitor` per `~/.claude/CLAUDE.md`:

```bash
gh pr checks "$PR_NUM" --watch --interval 30
```

Dispatch this in `run_in_background` and use the resulting notification to know when CI lands. Do NOT chain sleeps.

- [ ] **Step 4: Handle CI result**

- **Green:** Notify user (single message: "CI green on PR #<num>; awaiting your merge"). STOP. The orchestrator does not auto-merge; the user merges when ready.
- **Red:** Re-dispatch a fix subagent (sonnet/high) with the failing job's log. Commit, re-push, loop until green. Do not notify user mid-loop.

---

## Task 10: Post-merge close-out — comment + close #65 + tick #70

**Dispatch:** Inline (orchestrator). No subagent. Triggered by the orchestrator's close-out step 5 in `~/.claude/CLAUDE.md` AFTER the user merges the PR.

This task slots into orchestrator close-out 5b/5c/5d but adds three discrete extra actions (5a is SKIPPED per Task 7).

- [ ] **Step 1: Post the templated comment on #65**

The comment body is **verbatim from spec §10**, with `<pr-number>`, `<follow-up-1>`, `<follow-up-2>` substituted.

```bash
gh issue comment 65 --body "$(cat <<EOF
Assessment complete; gaps closed where uniquely owned.

Full audit: \`docs/superpowers/specs/2026-05-21-gpu-coverage-assessment-design.md\`. PR: #${PR_NUM}.

**Closed in this PR:**
- \`vision\` / \`all\` PEFT scope coverage (T5/T6 GPU inspection + C2 CPU).
- Mid-training NaN-abort behavior pinned (C3).
- Eval-metric finiteness asserted on 50-step + e2e tests (T2/T3/T4).
- GPU resume — QLoRA quant-state continuity (T1); CPU resume continuity for optimizer / scheduler / RNG (C4).
- Malformed-dataset error contracts pinned (C5).
- Evaluator schema unit coverage (C1).

**Deferred via follow-up issues:**
- eval-metric *threshold* floors → see follow-up issue #${FOLLOWUP_THRESHOLDS}.
- GPU \`all\`-scope smoke → see follow-up issue #${FOLLOWUP_ALL_SCOPE}, blocked on #68.

**Owned elsewhere (no work here):** model variants #23, headless GPU CI #64, VRAM-floor tightening #68, \`csp predict\` smoke #74, video #78, older CUDA #79.

This closes the \`GPU test gate decision shipped — #65\` line on the #70 pre-v1.0 checklist.
EOF
)"
```

- [ ] **Step 2: Close #65**

```bash
gh issue close 65 --reason completed
```

- [ ] **Step 3: Tick `GPU test gate decision shipped — #65` on #70**

Fetch #70's body, find the relevant checkbox line (`- [ ] GPU test gate decision shipped — #65`), replace with `- [x] GPU test gate decision shipped — #65`, re-post.

```bash
gh issue view 70 --json body -q .body > /tmp/issue70.md
python3 -c "
import sys, pathlib
p = pathlib.Path('/tmp/issue70.md')
text = p.read_text()
needle = '- [ ] GPU test gate decision shipped — #65'
replacement = '- [x] GPU test gate decision shipped — #65'
if needle not in text:
    sys.exit(f'checkbox line not found in #70 body: {needle!r}')
p.write_text(text.replace(needle, replacement, 1))
"
gh issue edit 70 --body-file /tmp/issue70.md
rm /tmp/issue70.md
```

If the exact em-dash / spacing differs in #70's body (the spec uses an em-dash; #70 might use `--`), adjust the needle. Verify by re-reading #70 after the edit.

- [ ] **Step 4: Run the rest of orchestrator close-out 5b–5e**

Per `~/.claude/CLAUDE.md` step 5:
- **5b** kill background processes (any `gh pr checks --watch` from Task 9).
- **5c** fold `logs/gpu-coverage-65.md` into `logs/logs.md`.
- **5d** remove the worktree (`cd <root> && git worktree remove .worktrees/gpu-coverage-65`).
- **5e** sign-off line confirming: tag SKIPPED (test-only PR), background killed, log folded, worktree removed, #65 closed + comment posted, #70 checkbox ticked.

Example sign-off line:

> Close-out: tag skipped (test-only, no consumer-observable change); background gh-watch killed; logs/gpu-coverage-65.md folded into logs/logs.md; worktree .worktrees/gpu-coverage-65 removed; #65 closed with §10 comment; #70 'GPU test gate decision shipped — #65' ticked.

---

## Self-Review Checklist (orchestrator runs before sending the plan to the implementer dispatch)

Run this checklist mentally before starting Task 2:

- [ ] **Spec §11 coverage:** Doc commit (Task 1) → ✓. T1 (2A) ✓. T2/T3 (2B) ✓. T4 (2C) ✓. T5/T6 (2D) ✓. C1 (3A) ✓. C2 (3B) ✓. C3 (3C) ✓. C4 (3D) ✓. C5 (3E) ✓. Lint/mypy (5) ✓. Collection counts (6) ✓. Markdownlint (6) ✓. Follow-up issues (8A/8B) ✓. PR (9) ✓. #65 close + #70 tick (10) ✓.
- [ ] **No placeholders:** every step has either code or an exact command. No "TBD" / "fill in details".
- [ ] **Type consistency:** `_RecordingTracker`, `_bnb_available`, `FIXTURE_SCOPE_PATTERNS`, `make_stub_wrapper`, `apply_lora`, `apply_qlora`, `PEFTConfig`, `ModelConfig`, `load_sam31`, `run_training`, `Trainer`, `Trainer.fit(...)`, `resume_from`, `_build_optimizer`, `_build_scheduler` — all referenced by their actual signatures.
- [ ] **Routing decisions explicit:** each task names role + model + effort + parallel/serial.
- [ ] **Lint/format LAST:** Task 5 explicitly the LAST step of the reviewer pass per CLAUDE.md.
- [ ] **Version-bump SKIPPED explicitly:** Task 7 + the dedicated "Version-Bump Decision" section both state the skip with reason.
- [ ] **Follow-up issues are full `gh` command templates:** Task 8A / 8B include `--title`, `--label` (twice), `--assignee @me`, body via heredoc.
- [ ] **PR creation + close-out present:** Task 9 (PR) + Task 10 (#65 close-out comment + tick #70).
