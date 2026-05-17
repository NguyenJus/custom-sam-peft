# Colab GPU Integration Fix v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make all 9 GPU integration tests pass on Colab T4 by (a) rebasing `worktree-fix+colab-bpe-gzip` onto `origin/main` (PR #14) and re-applying our adapter implementation into PR #14's new `box_hints`-aware signature, (b) wrapping the adapter's bf16-weight forward path in `torch.autocast` to resolve the float32-vs-bf16 mismatch inside SAM 3.1's geometry encoder, and (c) pinning `torchao>=0.16.0` in the Colab install cell to satisfy peft's lazy torchao version check.

**Architecture:** Three surgical commits on a freshly-rebased branch. The rebase itself is one work item (Task 1) and is the only place adapter logic is touched re Problem 1. Problem 2 (dtype) is a small autocast wrap inside `_Sam3ImageAdapter.forward`. Problem 3 (torchao) is a one-line addition to the notebook's install cell. No `pyproject.toml` changes, no new project dependencies.

**Tech stack:** Python 3.13, PyTorch 2.4+, HuggingFace `peft` 0.19.x, Meta `sam3`, `pytest`, `ruff`. No new project deps; one Colab-install-cell pin (`torchao>=0.16.0`).

**Reference spec:** `docs/superpowers/specs/2026-05-17-colab-gpu-integration-fix-v2-design.md`

---

## Pre-flight checks

Run these once before starting Task 1:

```bash
# 1. Confirm you are in the worktree.
git -C /home/justin/projects/Efficient-SAM3-Finetuning/.claude/worktrees/fix+colab-bpe-gzip rev-parse --show-toplevel
# Expected: /home/justin/projects/Efficient-SAM3-Finetuning/.claude/worktrees/fix+colab-bpe-gzip

# 2. Confirm branch tip and that origin/main has been fetched.
git -C /home/justin/projects/Efficient-SAM3-Finetuning/.claude/worktrees/fix+colab-bpe-gzip status
git -C /home/justin/projects/Efficient-SAM3-Finetuning/.claude/worktrees/fix+colab-bpe-gzip fetch origin main
git -C /home/justin/projects/Efficient-SAM3-Finetuning/.claude/worktrees/fix+colab-bpe-gzip log --oneline -1
# Expected branch tip: dec482b chore(logs): record task-5 push
git -C /home/justin/projects/Efficient-SAM3-Finetuning/.claude/worktrees/fix+colab-bpe-gzip log --oneline origin/main -1
# Expected: 5071c00 feat(train): training loop — Trainer, train_step, checkpoint, box-hint curriculum (#14)

# 3. Confirm the working tree is CLEAN (no uncommitted changes).
git -C /home/justin/projects/Efficient-SAM3-Finetuning/.claude/worktrees/fix+colab-bpe-gzip status --porcelain
# Expected: no output

# 4. Confirm the existing unit baseline is 201 passing pre-rebase.
uv run pytest tests/unit -q --no-cov 2>&1 | tail -3
# Expected last line: "201 passed, <N> warnings in <T>s"

# 5. Confirm sam3 helpers still import.
uv run python -c "import sam3; from sam3.model.data_misc import FindStage; from sam3.model.geometry_encoders import Prompt; from sam3.model.box_ops import box_xyxy_to_cxcywh; print('OK')"
# Expected: OK
```

If any pre-flight check fails, STOP and investigate. Do not start Task 1 on a dirty tree.

---

## File map (what gets touched)

| File | Action | Owning task |
| --- | --- | --- |
| `src/esam3/models/sam3.py` | Modify during rebase conflict resolution (Task 1) and again with autocast (Task 2). | 1, 2 |
| `notebooks/colab_gpu_tests.ipynb` | Add `torchao>=0.16.0` to install cell + comment update. | 3 |
| `logs/log.md` | Append one entry per task. | 1, 2, 3, 4 |

No other files touched. The new files added by PR #14 (`src/esam3/train/*`, `tests/unit/test_train_*`, `tests/unit/test_trainer_*`, `tests/unit/test_box_hint_schedule.py`, `tests/unit/test_geometric_prompt_builder.py`, `tests/unit/test_sam3_wrapper_box_hints.py`, `tests/integration/test_train_*.py`, `tests/gpu/test_real_train_overfits.py`) come in via the rebase replay and are NOT modified by this plan.

---

## Task 1: Rebase onto origin/main and reconcile the adapter

**Difficulty:** H
**Subagent:** `implementer` (Sonnet/high). Surgical multi-step git operation + careful conflict resolution in one file; high consequence (could lose the gzip fix and/or break PR #14's `_build_geometric_prompt`).

**Files:**
- Modify: `src/esam3/models/sam3.py` (conflict resolution only — no new logic beyond stitching our v1 Task 2 forward body into main's new signature)
- Append: `logs/log.md` (new entry)

**Expected diff size after the rebase commit lands:** net `src/esam3/models/sam3.py` changes vs `origin/main`'s version: ~+10 / -15 lines (drop `_resolve_bpe_path` ~13 lines + drop `bpe_path=` 1 line; reshape adapter `__init__` to accept `image_size` +1 line; replace `NotImplementedError` body with ~25-line forward body; `from sam3.model.data_misc import FindStage` +1 line).

### Scope

Rebase `worktree-fix+colab-bpe-gzip` onto `origin/main`. Resolve the conflict in `src/esam3/models/sam3.py` by:
- Taking main's NEW imports, `_build_geometric_prompt`, `Sam3Wrapper` (new signature + new `_validate_inputs`), and adapter docstring.
- KEEPING our gzip fix: delete `_resolve_bpe_path()` and the `bpe_path=str(bpe_path)` kwarg in `build_sam3_image_model(...)`.
- REPLACING the adapter `forward` body (`NotImplementedError` in main) with our v1 Task 2 implementation, ADAPTED to:
  - The new signature (`box_hints` kwarg, `image_size` from `self.image_size`).
  - Use `_build_geometric_prompt(...)` instead of `_get_dummy_prompt(num_prompts=1)`.
  - Add fallback to a manual zero-length `Prompt` when the builder returns `None`.
- ADDING `image_size` as a constructor arg on `_Sam3ImageAdapter` (default `1008`).
- UPDATING `load_sam31` so `adapter = _Sam3ImageAdapter(raw_model, image_size=1008)` matches the wrapper's `image_size=1008`.

The autocast wrap is NOT in this task — Task 2 adds it as a separate commit so the dtype fix is reviewable in isolation.

### Reference

- Spec sections: §6.1 (rebase strategy), §6.2 (adapter forward body), §6.5 (fate of `dec482b`).
- v1 spec §4: original Task 2 recipe.
- Source citations (verify by reading, do NOT modify):
  - main version of `src/esam3/models/sam3.py`: `git show origin/main:src/esam3/models/sam3.py`
  - `.venv/lib/python3.13/site-packages/sam3/model/sam3_image.py:547-553` (`_get_dummy_prompt`)
  - `.venv/lib/python3.13/site-packages/sam3/model/geometry_encoders.py:83-238` (`Prompt` constructor)

### Steps

- [ ] **Step 1: Snapshot the current tip and main tip for the log.**

```bash
git -C /home/justin/projects/Efficient-SAM3-Finetuning/.claude/worktrees/fix+colab-bpe-gzip log --oneline -6 > /tmp/pre-rebase-tip.txt
git -C /home/justin/projects/Efficient-SAM3-Finetuning/.claude/worktrees/fix+colab-bpe-gzip log --oneline origin/main -1 > /tmp/origin-main-tip.txt
cat /tmp/pre-rebase-tip.txt /tmp/origin-main-tip.txt
```

Expected:
```
dec482b chore(logs): record task-5 push
09bde5b test(integration): realign LoRA name substrings to real SAM 3.1 naming
4c4686f fix(peft): pin SCOPE_TARGETS to real SAM 3.1 module names
dd76a20 docs: spec + plan for Colab GPU integration fix
ab8b0b9 fix(models): implement _Sam3ImageAdapter.forward via forward_grounding
517ff6a fix(models): drop bpe_path override so sam3 uses its bundled gzipped vocab
5071c00 feat(train): training loop — Trainer, train_step, checkpoint, box-hint curriculum (#14)
```

- [ ] **Step 2: Start the rebase non-interactively.**

```bash
git -C /home/justin/projects/Efficient-SAM3-Finetuning/.claude/worktrees/fix+colab-bpe-gzip rebase origin/main
```

Expected behavior: rebase stops with a conflict on the FIRST commit being replayed (`517ff6a fix(models): drop bpe_path override`). The conflict is in `src/esam3/models/sam3.py`.

If the rebase succeeds without conflict (unlikely), STOP — origin/main's content may have shifted; investigate before proceeding.

- [ ] **Step 3: Resolve the FIRST conflict (517ff6a, gzip-fix).**

`git status` should show:
```
both modified:   src/esam3/models/sam3.py
```

Open `src/esam3/models/sam3.py`. Conflict markers will appear around `_resolve_bpe_path` and the `build_sam3_image_model(...)` call.

Resolution: Accept main's structure as the base, then DELETE these two regions:

1. The entire `_resolve_bpe_path` function definition (~13 lines, lines ~180-191 of main's version):
   ```python
   def _resolve_bpe_path(cfg: ModelConfig) -> Path:
       """The BPE merges file is shipped alongside the checkpoint in the HF repo."""
       if cfg.local_dir is None:
           raise FileNotFoundError("ModelConfig.local_dir is None; cannot resolve BPE path.")
       path = Path(cfg.local_dir) / "merges.txt"
       if not path.exists():
           raise FileNotFoundError(
               f"SAM 3.1 BPE merges file not found at {path}. Re-download the checkpoint "
               f"directory from {cfg.name}."
           )
       return path
   ```

2. In `load_sam31`, delete the line `bpe_path = _resolve_bpe_path(cfg)` AND the kwarg `bpe_path=str(bpe_path),` in the `build_sam3_image_model(...)` call.

After deletion, `load_sam31`'s `build_sam3_image_model` call must look exactly like:

```python
raw_model = sam3.build_sam3_image_model(
    device=device,
    eval_mode=False,  # training mode — gradients flow.
    checkpoint_path=str(ckpt_path),
    load_from_HF=False,
    enable_segmentation=True,
    enable_inst_interactivity=False,
    compile=False,
)
```

Mark the file resolved and continue:

```bash
git -C /home/justin/projects/Efficient-SAM3-Finetuning/.claude/worktrees/fix+colab-bpe-gzip add src/esam3/models/sam3.py
git -C /home/justin/projects/Efficient-SAM3-Finetuning/.claude/worktrees/fix+colab-bpe-gzip rebase --continue
```

The editor will pop up with the commit message of `517ff6a` (`fix(models): drop bpe_path override...`). Save and close it unchanged.

- [ ] **Step 4: Resolve the SECOND conflict (ab8b0b9, adapter forward).**

The rebase will stop again at `ab8b0b9 fix(models): implement _Sam3ImageAdapter.forward via forward_grounding`. Conflict in the same file `src/esam3/models/sam3.py`.

Conflict region: `_Sam3ImageAdapter` class body. Main has the docstring + `NotImplementedError` body; our commit has the v1 working body. Resolution combines:

- Main's docstring (the one describing the `box_hints` kwarg and pointing to the recipe).
- A REPLACED constructor that adds `image_size`:
  ```python
  def __init__(self, model: nn.Module, image_size: int = 1008) -> None:
      super().__init__()
      self.model = model
      self.image_size = image_size
  ```
- A REPLACED `forward` body using main's signature, our v1 Task 2 logic, and main's `_build_geometric_prompt` helper. NO autocast wrap yet — that's Task 2.

The complete `_Sam3ImageAdapter` after this conflict resolution:

```python
class _Sam3ImageAdapter(nn.Module):
    """Adapt raw Sam3Image to the (images, prompts, box_hints) calling convention.

    Sam3Image's training-mode forward (``forward_grounding``) expects
    ``(backbone_out, find_input, find_target, geometric_prompt)``, none of which
    are raw image tensors or our ``Prompts`` dataclasses.  This adapter holds the
    inner ``Sam3Image`` and orchestrates the conversion.

    The ``box_hints`` kwarg routes per-image absolute-pixel xyxy box hints
    through ``_build_geometric_prompt`` into Meta's ``Prompt`` container.  When
    every entry is ``None`` (or the kwarg itself is ``None``), the builder
    returns ``None`` and we substitute Meta's zero-length-seq dummy.

    ``image_size`` must match the wrapper's image_size; ``load_sam31`` plumbs
    it through the constructor.
    """

    def __init__(self, model: nn.Module, image_size: int = 1008) -> None:
        super().__init__()
        self.model = model
        self.image_size = image_size

    def forward(
        self,
        images: Tensor,
        prompts: list[Prompts],
        box_hints: list[Tensor | None] | None = None,
    ) -> dict[str, Tensor]:
        if not all(isinstance(p, TextPrompts) for p in prompts):
            raise ValueError("_Sam3ImageAdapter only supports TextPrompts in v0")
        class_names = [p.classes[0] for p in prompts]
        if len(set(class_names)) > 1:
            raise ValueError(
                "All prompts in a batch must share the same class name "
                "(SAM 3.1 forward_grounding runs one text prompt per call); "
                f"got {class_names}"
            )
        device = images.device
        b = images.shape[0]
        backbone_out = self.model.backbone.forward_image(images)
        text_outputs = self.model.backbone.forward_text([class_names[0]], device=device)
        backbone_out.update(text_outputs)
        find_input = FindStage(
            img_ids=torch.arange(b, device=device, dtype=torch.long),
            text_ids=torch.zeros(b, device=device, dtype=torch.long),
            input_boxes=None,
            input_boxes_mask=None,
            input_boxes_label=None,
            input_points=None,
            input_points_mask=None,
        )
        gp = _build_geometric_prompt(
            box_hints if box_hints is not None else [None] * b,
            self.image_size,
            device,
        )
        if gp is None:
            gp = Prompt(
                box_embeddings=torch.zeros(0, b, 4, device=device),
                box_mask=torch.zeros(b, 0, device=device, dtype=torch.bool),
            )
        outputs: dict[str, Tensor] = self.model.forward_grounding(
            backbone_out=backbone_out,
            find_input=find_input,
            find_target=None,
            geometric_prompt=gp,
        )
        return outputs
```

Also: at the TOP of the file, ensure `from sam3.model.data_misc import FindStage` is present (main does not import it; we re-add it). The full import section must look like:

```python
import sam3
import torch
from sam3.model.box_ops import box_xyxy_to_cxcywh
from sam3.model.data_misc import FindStage
from sam3.model.geometry_encoders import Prompt
from torch import Tensor, nn

from esam3.config.schema import ModelConfig
from esam3.data.base import BoxPrompts, Prompts, TextPrompts
```

(`BoxPrompts` is imported because `Sam3Wrapper._validate_inputs` references it — leave that alone.)

Update `load_sam31` (the body of `_setup_device_and_mode` section near the bottom) to pass `image_size=1008`:

```python
adapter = _Sam3ImageAdapter(raw_model, image_size=1008)
return Sam3Wrapper(adapter, image_size=1008, mask_size=288)
```

Mark resolved and continue:

```bash
git -C /home/justin/projects/Efficient-SAM3-Finetuning/.claude/worktrees/fix+colab-bpe-gzip add src/esam3/models/sam3.py
git -C /home/justin/projects/Efficient-SAM3-Finetuning/.claude/worktrees/fix+colab-bpe-gzip rebase --continue
```

The editor pops up with `ab8b0b9`'s message. Save unchanged.

- [ ] **Step 5: Drive the remaining commits to completion.**

The remaining commits (`dd76a20 docs`, `4c4686f fix(peft) SCOPE_TARGETS`, `09bde5b test(integration) substring renames`, `dec482b chore(logs)`) should apply cleanly.

```bash
# (no extra commands needed — `git rebase --continue` keeps going if no further conflicts).
```

If ANY of these stops with a conflict, STOP and investigate. None should — origin/main did not touch `lora.py`, `tests/fixtures/tiny_sam3_lora_stub.py`, `tests/unit/test_peft_lora.py`, `tests/integration/test_peft_lora_real.py`, `tests/integration/test_peft_qlora_real.py`, or `docs/superpowers/`.

If a conflict appears on `logs/log.md` (origin/main does not modify it — but in case), resolve by taking OUR version verbatim.

- [ ] **Step 6: Verify rebase completed cleanly.**

```bash
git -C /home/justin/projects/Efficient-SAM3-Finetuning/.claude/worktrees/fix+colab-bpe-gzip status
# Expected: "On branch worktree-fix+colab-bpe-gzip" + "nothing to commit, working tree clean"
git -C /home/justin/projects/Efficient-SAM3-Finetuning/.claude/worktrees/fix+colab-bpe-gzip log --oneline -7
# Expected (top-down): 5 reapplied commits (with new SHAs) + 5071c00 + d2cef37 + ...
```

- [ ] **Step 7: Sanity-check the resolved sam3.py.**

```bash
grep -n "from sam3.model.data_misc import FindStage" src/esam3/models/sam3.py
# Expected: 1 match.

grep -n "_resolve_bpe_path\|bpe_path=" src/esam3/models/sam3.py
# Expected: 0 matches.

grep -n "_build_geometric_prompt" src/esam3/models/sam3.py
# Expected: at least 2 matches (function def + adapter call).

grep -n "image_size: int = 1008" src/esam3/models/sam3.py
# Expected: at least 2 matches (Sam3Wrapper.__init__ + _Sam3ImageAdapter.__init__).

grep -n "NotImplementedError\|IMPLEMENTOR" src/esam3/models/sam3.py
# Expected: 0 matches.

grep -n "_Sam3ImageAdapter only supports TextPrompts" src/esam3/models/sam3.py
# Expected: 1 match (prompt-validation guard preserved).

grep -n "All prompts in a batch must share the same class name" src/esam3/models/sam3.py
# Expected: 1 match.

grep -n "adapter = _Sam3ImageAdapter(raw_model, image_size=1008)" src/esam3/models/sam3.py
# Expected: 1 match.
```

If ANY assertion fails, STOP. Run `git rebase --abort` and report to the user.

- [ ] **Step 8: Lint and format.**

```bash
uv run ruff check src/esam3/models/sam3.py
uv run ruff format --check src/esam3/models/sam3.py
```

Fix any reported issue in-place (edit the file, do NOT amend the rebase commit — if a fix is needed, leave it for Task 2's commit since Task 2 also edits this file).

If neither check raises, proceed.

- [ ] **Step 9: Run the post-rebase unit baseline.**

```bash
uv run pytest tests/unit -q --no-cov 2>&1 | tail -3
```

Record the test count (e.g. "278 passed"). Pin this as the **new baseline** in `logs/log.md` (Step 11). It is expected to be significantly larger than 201 because origin/main added training-loop unit tests.

If ANY test FAILS (not just count differs), STOP and investigate. Possible failure modes:
- `tests/unit/test_geometric_prompt_builder.py::*` — should pass; if not, our adapter changes broke `_build_geometric_prompt`.
- `tests/unit/test_sam3_wrapper_box_hints.py::*` — uses `tests/fixtures/tiny_sam3_stub.py`'s `TinySam3Stub`. Confirm `TinySam3Stub` accepts `box_hints` (PR #14 added it).
- `tests/unit/test_peft_lora.py::*` — should still report 20 tests passing (the rename Task 3 of v1 stays in place).
- `tests/unit/test_stubs_raise.py::*` (added by PR #14) — should pass.

- [ ] **Step 10: Run linting on the new package layout.**

```bash
uv run ruff check src tests
uv run ruff format --check src tests
```

Both should pass with no output (or only known excepted warnings). Fix in-place if needed.

- [ ] **Step 11: Append to `logs/log.md`.**

Add an entry (replace `<UTC-ISO8601>` with the actual timestamp; replace `<N>` with the actual baseline count from Step 9):

```
[<UTC-ISO8601>] [implementer] task-1 v2: rebased worktree-fix+colab-bpe-gzip onto origin/main (5071c00); resolved sam3.py conflict to keep gzip-fix + re-apply forward-grounding adapter into PR #14's box_hints signature; new unit baseline: <N> passing
```

If `logs/log.md` does not exist after the rebase (it should — created in v1 Task 1), create it with a single line header `# Log` and add the entry below.

- [ ] **Step 12: DO NOT push yet.**

The push happens at Task 5 after Tasks 2 and 3 land. Leave the branch local-only for now.

### Definition of Done

- [ ] `git status` reports a clean working tree.
- [ ] `git log --oneline -1` shows a Task-1 commit replayed from the rebase, NOT a merge commit (rebase, not merge).
- [ ] `git log --oneline origin/main..HEAD` shows exactly 5 commits (4 functional + 1 historical log entry).
- [ ] `grep -c "_resolve_bpe_path" src/esam3/models/sam3.py` returns 0.
- [ ] `grep -c "bpe_path=" src/esam3/models/sam3.py` returns 0.
- [ ] `grep -c "_build_geometric_prompt" src/esam3/models/sam3.py` returns ≥ 2.
- [ ] `grep -c "NotImplementedError\|IMPLEMENTOR" src/esam3/models/sam3.py` returns 0.
- [ ] `_Sam3ImageAdapter.__init__` takes `(self, model, image_size=1008)`.
- [ ] `_Sam3ImageAdapter.forward` takes `(self, images, prompts, box_hints=None)`.
- [ ] `_Sam3ImageAdapter.forward` body does NOT yet contain `torch.autocast` (Task 2 adds it).
- [ ] `_Sam3ImageAdapter.forward` body contains the two prompt-validation guards (TextPrompts check, single-class check).
- [ ] `load_sam31` constructs `_Sam3ImageAdapter(raw_model, image_size=1008)`.
- [ ] `ruff check src/esam3/models/sam3.py` exits 0.
- [ ] `ruff format --check src/esam3/models/sam3.py` exits 0.
- [ ] `uv run pytest tests/unit -q --no-cov` reports ALL tests passing at the new post-rebase baseline (count recorded in `logs/log.md`); ZERO regressions.
- [ ] `logs/log.md` has a new Task-1 v2 entry pinning the baseline.

### Verification (commands)

```bash
git -C /home/justin/projects/Efficient-SAM3-Finetuning/.claude/worktrees/fix+colab-bpe-gzip status
git -C /home/justin/projects/Efficient-SAM3-Finetuning/.claude/worktrees/fix+colab-bpe-gzip log --oneline origin/main..HEAD
grep -c "_resolve_bpe_path\|bpe_path=" src/esam3/models/sam3.py    # expect 0
grep -c "NotImplementedError\|IMPLEMENTOR" src/esam3/models/sam3.py # expect 0
grep -n "image_size: int = 1008" src/esam3/models/sam3.py           # expect 2
grep -c "torch.autocast" src/esam3/models/sam3.py                    # expect 0 (Task 2 adds it)
uv run ruff check src/esam3/models/sam3.py
uv run ruff format --check src/esam3/models/sam3.py
uv run pytest tests/unit -q --no-cov 2>&1 | tail -3
```

### Rollback

If anything goes wrong during the rebase BEFORE Step 6 completes:

```bash
git -C /home/justin/projects/Efficient-SAM3-Finetuning/.claude/worktrees/fix+colab-bpe-gzip rebase --abort
```

If something goes wrong AFTER the rebase completes locally but before pushing (still local-only branch state in this task):

```bash
# Look at reflog to find the pre-rebase tip.
git -C /home/justin/projects/Efficient-SAM3-Finetuning/.claude/worktrees/fix+colab-bpe-gzip reflog -10
# Reset to the pre-rebase tip (dec482b, or whatever the reflog shows).
git -C /home/justin/projects/Efficient-SAM3-Finetuning/.claude/worktrees/fix+colab-bpe-gzip reset --hard <pre-rebase-SHA>
```

NEVER use `git reset --hard` without first inspecting the reflog. Confirm with the user before this rollback.

### Commit

The rebase creates the commits via replay; no manual commit step is needed for Task 1. Steps 10-11 simply WRITE the log entry, which lands in the NEXT task's commit (Task 2's commit includes the Task-1 log entry, or we add a one-line commit at the end of Task 1 just for the log — your choice. For simplicity, **add a one-line commit at the end of Task 1**):

```bash
git add logs/log.md
git commit -m "$(cat <<'EOF'
chore(logs): record rebase onto origin/main and new unit baseline

Pin the post-rebase unit baseline pinned by Task 1 of
docs/superpowers/plans/2026-05-17-colab-gpu-integration-fix-v2.md.
EOF
)"
```

---

## Task 2: Wrap the adapter forward in `torch.autocast(bfloat16)`

**Difficulty:** M
**Subagent:** `implementer` (Sonnet/high). Single-file edit, but the autocast contract is subtle (predicate, dtype, scope of the `with` block) and the test is GPU-only — so the engineer must reason about the change carefully, not lean on local tests.

**Files:**
- Modify: `src/esam3/models/sam3.py` — wrap the adapter's forward body in `torch.autocast(...)`.
- Append: `logs/log.md`

**Expected diff size:** +5 / -1 lines (one `with` line introduced; existing body indented by one level; one closing dedent).

### Scope

Add `torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda"))` around the body of `_Sam3ImageAdapter.forward` that touches model weights. The prompt-validation guards (TextPrompts check, single-class check) stay OUTSIDE the autocast context — they're pure Python control flow and don't need it.

Specifically, the autocast wraps:
- `backbone_out = self.model.backbone.forward_image(images)`
- `text_outputs = self.model.backbone.forward_text(...)`
- `backbone_out.update(text_outputs)` — no model op, but harmless inside
- `find_input = FindStage(...)` — also no model op, harmless inside; keep inside to minimize diff churn
- `gp = _build_geometric_prompt(...)` — pure tensor construction; inside is fine
- `if gp is None: gp = Prompt(...)` — same
- `outputs = self.model.forward_grounding(...)` — THE call that needs autocast

Return statement happens OUTSIDE the `with` block (returning out of `with` works either way, but indenting the return inside is fine too — leave it outside for clarity).

### Reference

- Spec sections: §3.4 (why autocast resolves the bug), §3.6 (chosen fix), §6.3 (why autocast wraps the whole body), §6.6 (acceptance for the autocast scope).
- PyTorch autocast docs: https://pytorch.org/docs/stable/amp.html#torch.autocast

### Steps

- [ ] **Step 1: Confirm Task 1 landed.**

```bash
grep -n "_Sam3ImageAdapter.forward\|def forward" src/esam3/models/sam3.py | head -10
grep -c "torch.autocast" src/esam3/models/sam3.py
# Expected: 0 (autocast not yet added)
```

If `torch.autocast` already appears, STOP — someone has added it already.

- [ ] **Step 2: Apply the autocast wrap.**

Find the `_Sam3ImageAdapter.forward` method body (the one introduced by Task 1). Wrap the body starting from the line that reads `backbone_out = self.model.backbone.forward_image(images)` through the line that reads `outputs: dict[str, Tensor] = self.model.forward_grounding(...)` (inclusive) in a `with torch.autocast(...)` block.

The resulting forward method must look exactly like:

```python
def forward(
    self,
    images: Tensor,
    prompts: list[Prompts],
    box_hints: list[Tensor | None] | None = None,
) -> dict[str, Tensor]:
    if not all(isinstance(p, TextPrompts) for p in prompts):
        raise ValueError("_Sam3ImageAdapter only supports TextPrompts in v0")
    class_names = [p.classes[0] for p in prompts]
    if len(set(class_names)) > 1:
        raise ValueError(
            "All prompts in a batch must share the same class name "
            "(SAM 3.1 forward_grounding runs one text prompt per call); "
            f"got {class_names}"
        )
    device = images.device
    b = images.shape[0]
    with torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=(device.type == "cuda"),
    ):
        backbone_out = self.model.backbone.forward_image(images)
        text_outputs = self.model.backbone.forward_text([class_names[0]], device=device)
        backbone_out.update(text_outputs)
        find_input = FindStage(
            img_ids=torch.arange(b, device=device, dtype=torch.long),
            text_ids=torch.zeros(b, device=device, dtype=torch.long),
            input_boxes=None,
            input_boxes_mask=None,
            input_boxes_label=None,
            input_points=None,
            input_points_mask=None,
        )
        gp = _build_geometric_prompt(
            box_hints if box_hints is not None else [None] * b,
            self.image_size,
            device,
        )
        if gp is None:
            gp = Prompt(
                box_embeddings=torch.zeros(0, b, 4, device=device),
                box_mask=torch.zeros(b, 0, device=device, dtype=torch.bool),
            )
        outputs: dict[str, Tensor] = self.model.forward_grounding(
            backbone_out=backbone_out,
            find_input=find_input,
            find_target=None,
            geometric_prompt=gp,
        )
    return outputs
```

Key contract:
- `device_type=device.type` (computed dynamically from `images.device`).
- `dtype=torch.bfloat16` (explicit, not default).
- `enabled=(device.type == "cuda")` (CPU paths skip autocast entirely).
- Return statement is OUTSIDE the `with` block.

- [ ] **Step 3: Lint and format.**

```bash
uv run ruff check src/esam3/models/sam3.py
uv run ruff format --check src/esam3/models/sam3.py
```

Fix any reported issue in-place. The most likely lint complaint is line-length on the `torch.autocast(...)` line — keep it as a multi-line call (matches the spec snippet above).

- [ ] **Step 4: Run unit tests.**

```bash
uv run pytest tests/unit -q --no-cov 2>&1 | tail -3
```

Expected: same count as Task 1's recorded baseline. ZERO regressions. The CPU-only fixture `TinySam3Stub` never hits the autocast path (its device is CPU; `enabled=False`).

If any test fails, the likely failure mode is `tests/unit/test_sam3_wrapper_box_hints.py` — it constructs a wrapper around `TinySam3Stub` and calls forward; confirm that path bypasses autocast on CPU.

- [ ] **Step 5: Append to `logs/log.md`.**

```
[<UTC-ISO8601>] [implementer] task-2 v2: wrapped _Sam3ImageAdapter.forward body in torch.autocast(bfloat16) on CUDA to fix float32-vs-bf16 mismatch in sam3 geometry encoder
```

- [ ] **Step 6: Commit.**

```bash
git -C /home/justin/projects/Efficient-SAM3-Finetuning/.claude/worktrees/fix+colab-bpe-gzip add src/esam3/models/sam3.py logs/log.md
git -C /home/justin/projects/Efficient-SAM3-Finetuning/.claude/worktrees/fix+colab-bpe-gzip commit -m "$(cat <<'EOF'
fix(models): wrap adapter forward in torch.autocast(bfloat16) for CUDA

SAM 3.1's geometry encoder synthesizes a default-float32 zero-points
tensor inside Prompt._init_point (sam3/model/geometry_encoders.py:299)
whenever the dummy Prompt provides only box_embeddings. With
load_sam31(cfg.dtype="bfloat16") the model's points_direct_project is a
bf16-weight nn.Linear, and F.linear raises `mat1 and mat2 must have the
same dtype, but got Float and BFloat16` even on a zero-length input.

Wrap the adapter's bf16-touching forward body in
torch.autocast(device_type=device.type, dtype=torch.bfloat16,
enabled=(device.type == "cuda")) so PyTorch downcasts mismatched inputs
on dispatch. CPU paths skip autocast (TinySam3Stub fixtures unaffected).

Unblocks tests/integration/test_load_sam31_real.py::test_load_sam31_forward_to_canonical
on Colab T4.
EOF
)"
```

### Definition of Done

- [ ] `src/esam3/models/sam3.py` contains exactly one `torch.autocast` call, inside `_Sam3ImageAdapter.forward`.
- [ ] The `torch.autocast` call uses `device_type=device.type`, `dtype=torch.bfloat16`, `enabled=(device.type == "cuda")`.
- [ ] The two prompt-validation guards (TextPrompts check, single-class check) are OUTSIDE the autocast block.
- [ ] The `forward_grounding` call is INSIDE the autocast block.
- [ ] The `return outputs` statement is OUTSIDE the autocast block.
- [ ] `ruff check src/esam3/models/sam3.py` exits 0.
- [ ] `ruff format --check src/esam3/models/sam3.py` exits 0.
- [ ] `uv run pytest tests/unit -q --no-cov` reports the same passing count as Task 1's recorded baseline; ZERO regressions.
- [ ] `logs/log.md` has a Task-2 v2 entry.
- [ ] One new commit on the branch tip with the message above.

### Verification (commands)

```bash
grep -c "torch.autocast" src/esam3/models/sam3.py        # expect 1
grep -n "dtype=torch.bfloat16" src/esam3/models/sam3.py  # expect 1 (the autocast)
grep -n "enabled=(device.type" src/esam3/models/sam3.py   # expect 1
uv run ruff check src/esam3/models/sam3.py
uv run ruff format --check src/esam3/models/sam3.py
uv run pytest tests/unit -q --no-cov 2>&1 | tail -3
```

### Rollback

```bash
git -C /home/justin/projects/Efficient-SAM3-Finetuning/.claude/worktrees/fix+colab-bpe-gzip reset --hard HEAD~1
# Restores Task 1's state.
```

---

## Task 3: Pin `torchao>=0.16.0` in the Colab install cell

**Difficulty:** L
**Subagent:** `implementer-simple` (Haiku/high). Single notebook cell edit, no logic, single line addition to an install command plus a comment update.

**Files:**
- Modify: `notebooks/colab_gpu_tests.ipynb` (one code cell — the `%pip install` cell, cell index 4)
- Append: `logs/log.md`

**Expected diff size:** +1 install-line continuation + ~7 lines of comment expansion in the same cell. The cell's other lines stay unchanged.

### Scope

Add `"torchao>=0.16.0"` to the existing `%pip install -e ...` line in the Colab notebook's install cell. Update the comment block above the install line to document WHY the torchao pin is needed.

This is a notebook-only change. Do NOT touch `pyproject.toml`. Local dev does not install torchao (and should not — we don't use it).

### Reference

- Spec sections: §4 (peft × torchao investigation), §4.4 (chosen fix).
- Notebook path: `notebooks/colab_gpu_tests.ipynb`, Cell 4 (the `%pip install` cell).

### Steps

- [ ] **Step 1: Inspect the current install cell.**

The notebook is a JSON file. Use `python -c` to print the install cell so you can see its exact current content:

```bash
python -c "import json; nb = json.load(open('notebooks/colab_gpu_tests.ipynb')); print(''.join(nb['cells'][4]['source']))"
```

Expected output ends with these lines (the install command):

```
%pip install -e ".[qlora,dev,tensorboard]" \
    "numpy==1.26.4" "scipy==1.13.1" "transformers==5.0.0" \
    "huggingface_hub>=1.15"
!python -c "import esam3; print('esam3 OK:', esam3.__file__)"
```

If the cell index is different (the notebook has been restructured), find the right cell first:

```bash
python -c "import json; nb = json.load(open('notebooks/colab_gpu_tests.ipynb')); [print(i, ''.join(c.get('source', []))[:80]) for i, c in enumerate(nb['cells'])]"
```

The cell whose source starts with `# Cell 3: Install runtime` is the right one.

- [ ] **Step 2: Edit the cell.**

Open `notebooks/colab_gpu_tests.ipynb` in your editor (notebooks are JSON; modern editors handle them as text or via a notebook UI). Make TWO changes to the install cell:

**Change A** (extend the comment block):

Find the existing paragraph that ends with:

```
# Re-evaluate these pins if sam3 ever relaxes its numpy bound. The
# transformers pin is the version Colab preinstalls; bumping it is fine as
# long as the bumped version is also in the kernel's site-packages.
```

Insert immediately AFTER it (before the `%pip install` line):

```
#
# torchao>=0.16.0 is pinned because peft 0.19+ lazily checks the installed
# torchao version on every LoRA-eligible nn.Linear dispatch
# (peft.tuners.lora.torchao calls peft.import_utils.is_torchao_available,
# which RAISES ImportError if torchao is installed AND `< 0.16.0`). Colab
# base images preinstall torchao 0.10.0, so apply_lora and apply_qlora hit
# this ImportError before any of our code runs. We do NOT use torchao
# ourselves (QLoRA uses bitsandbytes); the pin only upgrades the
# preinstalled package past peft's gate. Drop this pin once Colab ships
# torchao >= 0.16.0 by default.
```

**Change B** (add the pin to the install line):

Replace:

```
%pip install -e ".[qlora,dev,tensorboard]" \
    "numpy==1.26.4" "scipy==1.13.1" "transformers==5.0.0" \
    "huggingface_hub>=1.15"
```

With:

```
%pip install -e ".[qlora,dev,tensorboard]" \
    "numpy==1.26.4" "scipy==1.13.1" "transformers==5.0.0" \
    "huggingface_hub>=1.15" \
    "torchao>=0.16.0"
```

Save the notebook.

- [ ] **Step 3: Verify JSON validity.**

```bash
python -c "import json; json.load(open('notebooks/colab_gpu_tests.ipynb')); print('OK')"
```

Expected: `OK`. If the JSON is invalid (typo in the edit), STOP and re-edit.

- [ ] **Step 4: Verify the cell content.**

```bash
python -c "import json; nb = json.load(open('notebooks/colab_gpu_tests.ipynb')); src = ''.join(nb['cells'][4]['source']); print('torchao>=0.16.0' in src, 'is_torchao_available' in src)"
```

Expected: `True True` (both the install pin and the explanatory comment with `is_torchao_available` are present).

- [ ] **Step 5: Run unit tests (sanity).**

```bash
uv run pytest tests/unit -q --no-cov 2>&1 | tail -3
```

Expected: same count as Task 1/2's baseline. No regression. The notebook change has no effect on tests (it's not imported anywhere).

- [ ] **Step 6: Append to `logs/log.md`.**

```
[<UTC-ISO8601>] [implementer] task-3 v2: pinned torchao>=0.16.0 in notebooks/colab_gpu_tests.ipynb install cell to clear peft's lazy version gate on Colab
```

- [ ] **Step 7: Commit.**

```bash
git -C /home/justin/projects/Efficient-SAM3-Finetuning/.claude/worktrees/fix+colab-bpe-gzip add notebooks/colab_gpu_tests.ipynb logs/log.md
git -C /home/justin/projects/Efficient-SAM3-Finetuning/.claude/worktrees/fix+colab-bpe-gzip commit -m "$(cat <<'EOF'
fix(colab): pin torchao>=0.16.0 to clear peft's lazy version gate

peft 0.19+ calls is_torchao_available() in its LoRA dispatcher every time
_create_new_module wraps an nn.Linear. The function raises ImportError if
torchao is installed AND < 0.16.0. Colab T4 images preinstall torchao
0.10.0, so apply_lora and apply_qlora fail before our code runs — 7 of 9
GPU integration tests blocked at module-import time.

Pin torchao>=0.16.0 on the existing %pip install line in the Colab
notebook so the upgrade happens in the same resolver pass as the existing
numpy/scipy/transformers pins (otherwise pip backtracks). We do NOT use
torchao ourselves (QLoRA path uses bitsandbytes); pyproject.toml stays
clean, local dev installs no torchao.

Unblocks tests/integration/test_peft_*_real.py::* on Colab T4.
EOF
)"
```

### Definition of Done

- [ ] `notebooks/colab_gpu_tests.ipynb` is valid JSON.
- [ ] The install cell contains `"torchao>=0.16.0"` on the same `%pip install` line as the existing pins.
- [ ] The comment block above the install line contains a paragraph mentioning `peft`, `is_torchao_available`, and `0.16.0`.
- [ ] `pyproject.toml` is byte-identical to its pre-task state.
- [ ] `uv run pytest tests/unit -q --no-cov` baseline unchanged from Task 1/2.
- [ ] `logs/log.md` has a Task-3 v2 entry.
- [ ] One new commit on the branch tip.

### Verification (commands)

```bash
python -c "import json; json.load(open('notebooks/colab_gpu_tests.ipynb')); print('JSON OK')"
python -c "import json; nb = json.load(open('notebooks/colab_gpu_tests.ipynb')); src = ''.join(nb['cells'][4]['source']); assert 'torchao>=0.16.0' in src; assert 'is_torchao_available' in src; print('content OK')"
git -C /home/justin/projects/Efficient-SAM3-Finetuning/.claude/worktrees/fix+colab-bpe-gzip diff --name-only HEAD~1 HEAD
# Expected: notebooks/colab_gpu_tests.ipynb + logs/log.md
git -C /home/justin/projects/Efficient-SAM3-Finetuning/.claude/worktrees/fix+colab-bpe-gzip diff HEAD~1 HEAD -- pyproject.toml
# Expected: empty (no changes)
```

### Rollback

```bash
git -C /home/justin/projects/Efficient-SAM3-Finetuning/.claude/worktrees/fix+colab-bpe-gzip reset --hard HEAD~1
```

---

## Task 4: Push branch and trigger Colab verification

**Difficulty:** L
**Subagent:** Main thread (no subagent needed — push + ask user to run the notebook).

**Files:** None modified.

### Scope

Push the rebased + fixed branch to remote. Because the rebase rewrote history relative to `origin/worktree-fix+colab-bpe-gzip`, the push requires `--force-with-lease` (NOT `--force`). The PR #13 description will be updated by GitHub automatically.

Then request the user run `notebooks/colab_gpu_tests.ipynb` end-to-end on Colab T4 and report the result.

### Reference

- Spec §10 acceptance criterion 7.

### Steps

- [ ] **Step 1: Sanity-check the branch state.**

```bash
git -C /home/justin/projects/Efficient-SAM3-Finetuning/.claude/worktrees/fix+colab-bpe-gzip status
# Expected: clean working tree.

git -C /home/justin/projects/Efficient-SAM3-Finetuning/.claude/worktrees/fix+colab-bpe-gzip log --oneline origin/main..HEAD
# Expected (top-down):
#   <new SHA> fix(colab): pin torchao>=0.16.0 to clear peft's lazy version gate
#   <new SHA> fix(models): wrap adapter forward in torch.autocast(bfloat16) for CUDA
#   <new SHA> chore(logs): record rebase onto origin/main and new unit baseline
#   <new SHA> chore(logs): record task-5 push                    [from dec482b replay]
#   <new SHA> test(integration): realign LoRA name substrings... [from 09bde5b replay]
#   <new SHA> fix(peft): pin SCOPE_TARGETS to real SAM 3.1...    [from 4c4686f replay]
#   <new SHA> docs: spec + plan for Colab GPU integration fix    [from dd76a20 replay]
#   <new SHA> fix(models): implement _Sam3ImageAdapter.forward...[from ab8b0b9 replay]
#   <new SHA> fix(models): drop bpe_path override...             [from 517ff6a replay]
```

(Total: 9 commits between origin/main and HEAD — 6 from the rebased original + 3 new from Tasks 1/2/3 in this plan.)

- [ ] **Step 2: Fetch latest origin state for `--force-with-lease` safety.**

```bash
git -C /home/justin/projects/Efficient-SAM3-Finetuning/.claude/worktrees/fix+colab-bpe-gzip fetch origin worktree-fix+colab-bpe-gzip
```

If `origin/worktree-fix+colab-bpe-gzip` has commits we DON'T have locally (i.e., someone pushed to the branch while we were rebasing), STOP and ask the user.

- [ ] **Step 3: Push with `--force-with-lease`.**

```bash
git -C /home/justin/projects/Efficient-SAM3-Finetuning/.claude/worktrees/fix+colab-bpe-gzip push --force-with-lease
```

Why `--force-with-lease` instead of `--force`: the rebase rewrote the SHAs of every commit on our branch, so a regular `git push` is rejected. `--force-with-lease` succeeds only if the remote branch is at the SHA we last fetched — i.e., no surprise commits from someone else. This is safer than `--force` while still allowing the rewrite.

If push is rejected:
- Inspect the remote state: `git log --oneline origin/worktree-fix+colab-bpe-gzip -5`
- If new commits appeared, STOP and ask the user (do NOT `--force` blindly).
- Otherwise, the lease may have expired (rare); rerun `git fetch` and retry.

- [ ] **Step 4: Verify PR #13 picked up the new commits.**

```bash
gh pr view 13 --json commits --jq '.commits[-3:] | .[] | .messageHeadline'
```

Expected: the last 3 commit headlines are (newest first):
- `fix(colab): pin torchao>=0.16.0 to clear peft's lazy version gate`
- `fix(models): wrap adapter forward in torch.autocast(bfloat16) for CUDA`
- `chore(logs): record rebase onto origin/main and new unit baseline`

- [ ] **Step 5: Append to `logs/log.md` AND commit the entry.**

The push is done; this final log entry just records the state.

```
[<UTC-ISO8601>] [implementer] task-4 v2: pushed rebased branch with --force-with-lease; PR #13 updated; awaiting Colab T4 verification
```

```bash
git -C /home/justin/projects/Efficient-SAM3-Finetuning/.claude/worktrees/fix+colab-bpe-gzip add logs/log.md
git -C /home/justin/projects/Efficient-SAM3-Finetuning/.claude/worktrees/fix+colab-bpe-gzip commit -m "$(cat <<'EOF'
chore(logs): record v2 push and Colab verification ask

End of v2 work items; awaiting Colab T4 result.
EOF
)"
git -C /home/justin/projects/Efficient-SAM3-Finetuning/.claude/worktrees/fix+colab-bpe-gzip push
```

(Plain `git push` here; this commit is on top of the already-pushed `--force-with-lease`'d branch.)

- [ ] **Step 6: Notify the user.**

Reply with:

1. The list of new commit SHAs (top 4 of `git log --oneline -4`).
2. PR #13 URL.
3. **Colab instructions** — exact text to send the user:

> Open `notebooks/colab_gpu_tests.ipynb` on Colab.
> - Runtime → Change runtime type → **T4 GPU** (or better).
> - In Cell 1 of the notebook, ensure `BRANCH = "worktree-fix+colab-bpe-gzip"`.
> - Runtime → Restart session (to clear any cached preinstalled `torchao 0.10.0`).
> - Runtime → Run all.
> - When the final `bash scripts/run_gpu_tests.sh` cell finishes, copy the last pytest summary line (e.g. `9 passed in 187s`) and paste it back here.
>
> If the suite reports fewer than 9 passes, ALSO paste:
> - The full last 30 lines of the failed test's traceback.
> - The output of the `!pip show torchao peft bitsandbytes` cell if you can run it (or run it as a new cell after Cell 3).

### Definition of Done

- [ ] `git push --force-with-lease` exits 0.
- [ ] `git push` (the plain final-log push) exits 0.
- [ ] `gh pr view 13` shows the new commits at the tip.
- [ ] User has been notified with the Colab instructions.

### Rollback

The push is irreversible from our side (the rebased history is the new shared truth on remote). If something is wrong, the rollback is "fix it forward":

1. Make the corrective change locally.
2. Commit.
3. `git push`.

Do NOT attempt to undo the rebase via another rewrite — confirm with the user first.

---

## Task 5: Triage Colab failures (contingent)

**Difficulty:** Conditional — L if a simple env-pin tweak; M if the dtype or torchao fix needs revision.
**Subagent:** `implementer` (Sonnet/high) ONLY if invoked; otherwise skip.

### Scope

Execute ONLY if the Colab T4 run reports any failure among the 9 integration tests after Task 4 push. Otherwise this plan is done.

### Decision tree

- **Failure mode 1: `RuntimeError: mat1 and mat2 must have the same dtype` STILL appears after Task 2.**
  - Likely cause: autocast not engaging (e.g., wrong `device_type` detection, or torchscript-traced sub-module not respecting autocast).
  - Diagnostic step: ask the user to capture `print(torch.is_autocast_enabled())` inside `_Sam3ImageAdapter.forward` just before `forward_grounding`. If `False`, autocast didn't engage; investigate `device.type` value.
  - Fix candidate: ADD a defensive `next(self.model.parameters()).dtype` check and explicitly cast all synthesized tensors (`find_input` Long tensors are fine; the dummy `Prompt`'s `box_embeddings` may need `.to(dtype=...)`). Re-push with Option (C) layered on top of (A).

- **Failure mode 2: `ImportError: Found an incompatible version of torchao` STILL appears after Task 3.**
  - Likely cause: Colab's pip resolver picked a torchao version that's not compatible with installed torch (rare — torchao publishes wheels for current torch).
  - Diagnostic step: ask the user for `!pip show torchao` output and the `%pip install` cell's resolver output.
  - Fix candidate: switch to Option (iii) from spec §4.3 — pin `peft<0.19` in the notebook install line as a fallback. STOP and ask the user before applying.

- **Failure mode 3: A different `ImportError` or `AttributeError` from `peft.tuners.lora` (e.g., torchao installed fine but some new code path errors).**
  - Likely cause: peft 0.19 has additional optional deps that interact poorly with the upgraded torchao.
  - Fix candidate: capture the trace and reopen a follow-up ticket. STOP and ask the user.

- **Failure mode 4: `test_load_sam31_forward_to_canonical` passes shape contract but raises a NEW assertion (e.g., `presence_logit_dec` key missing).**
  - Likely cause: spec §8 risk — `dec_presence_out is None` in this build.
  - Out of scope. Reopen as a separate spec on `src/esam3/models/matching.py`.

- **Failure mode 5: `test_apply_lora_on_real_sam31_under_trainable_budget` fails with trainable-ratio >= 5%.**
  - Likely cause: v1 spec §7 risk (vision_decoder scope too wide).
  - Fix candidate: narrow `SCOPE_TARGETS["vision_decoder"]` second pattern to drop `ca_text`. Re-push.

- **Failure mode 6: `apply_qlora` fails with a bitsandbytes error.**
  - Out of scope for this plan; new ticket.

### Steps (only when invoked)

- [ ] **Step 1: Read the failure trace from the user.**

- [ ] **Step 2: Match against the decision tree above.**

- [ ] **Step 3: Apply the fix candidate, commit, push, ask the user to re-run Colab.**

- [ ] **Step 4: Append a Task-5 entry to `logs/log.md`.**

### Definition of Done

- [ ] All 9 Colab integration tests pass on the next run.
- [ ] `logs/log.md` records the triage.

### Rollback

If the fix candidate makes things WORSE on Colab, `git revert <commit>` and ask the user.

---

## Final acceptance

A correct implementation of this plan satisfies:

1. `worktree-fix+colab-bpe-gzip` is rebased onto `origin/main` (`5071c00 feat(train): training loop`), then has 3 new commits on top: rebase log, dtype-fix (autocast), torchao notebook pin, plus a final log-only commit from Task 4.
2. `uv run pytest tests/unit -q --no-cov` reports ZERO regressions vs the post-rebase baseline (which Task 1 pinned in `logs/log.md`).
3. `ruff check src tests` and `ruff format --check src tests` pass.
4. On Colab T4: `bash scripts/run_gpu_tests.sh` reports all 9 tests passing under `requires_compatible_gpu and requires_checkpoint`.
5. `logs/log.md` contains at least 4 new append-only entries (Task 1, 2, 3, 4).
6. `pyproject.toml` is byte-identical to pre-plan state.
7. No emojis anywhere in the diff.
8. No source files outside `src/esam3/models/sam3.py` (Tasks 1, 2) and `notebooks/colab_gpu_tests.ipynb` (Task 3) and `logs/log.md` (every task) are modified.
