# SAM 3.1 multiplex forward — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the per-class serialized forward in train, eval, and predict with SAM 3.1's native multiplex forward (one forward per ≤16-class group), gated by a single `MULTIPLEX_CAP = 16` constant and a configurable `classes_per_forward` knob.

**Architecture:** Two new constants + one new config dataclass underpin three sites of change. The wrapper/adapter (`models/sam3.py`) accept up to 16 classes per `TextPrompts` and emit `(B·K, …)` outputs; the trainer (`train/loop.py`) iterates `_chunked` class groups and calls `total_loss` once per group on the flattened batch; the evaluator (`eval/evaluator.py`) and predict CLI (`predict/runner.py`) mirror that shape, with a forward-only sibling of `decide_preset` (`presets.decide_eval_batch_size`) resolving an `EvalConfig.batch_size: int | "auto"` and a matching `PredictOptions.batch_size: int | "auto"`. Spec: `docs/superpowers/specs/2026-05-23-multiplex-forward-design.md`.

**Tech Stack:** PyTorch + Meta's `sam3` library, Pydantic v2 schema, Python `dataclasses`, pytest (CPU markers + `gpu` / `requires_compatible_gpu`).

---

## Conventions

- Spec references use `§N` (matches the design doc).
- Each task names exact files; line numbers cite the spec / current source where stable. Implementers should grep on the surrounding identifier if a number drifts.
- TDD ordering: write failing test → run → minimal impl → run → commit. Every code task uses that loop.
- Commits land on branch `multiplex-forward-22`. Implementer commits during implementation are exempt from the lint gate; the final reviewer/orchestrator runs lint+format before opening the PR.
- "Parallelizable with: [task IDs]" means the listed tasks touch disjoint files and have no shared-state ordering constraint with this task; the orchestrator may dispatch them concurrently to file-disjoint agents.
- The codebase ships `pytest` markers `gpu` and `requires_compatible_gpu` (see `pyproject.toml:118-124`); spec §9 calls one of them `requires_gpu` — interpret that as the `gpu` + `requires_compatible_gpu` pair that today's `tests/integration/test_load_sam31_real.py` already uses.

---

## File map

Files this plan touches, with each file's responsibility:

| File | Responsibility |
|------|----------------|
| `src/custom_sam_peft/models/sam3.py` | Adds `MULTIPLEX_CAP = 16`; lifts the K=1 raise; adapter builds `(B·K, …)` rows; `_build_geometric_prompt` parameterized on `n_cols`. (§4, §13 ACs 1-3) |
| `src/custom_sam_peft/config/schema.py` | New `MultiplexConfig`; new `TrainHyperparams.multiplex` field; `EvalConfig.batch_size` field. (§7, §13 ACs 8, 10) |
| `src/custom_sam_peft/config/_internal.py` | `LossConfig` docstring update. (§7, §13 AC 11) |
| `src/custom_sam_peft/models/losses.py` | No interface change; just absorbs `(B·K_g, …)` inputs. (§3) |
| `src/custom_sam_peft/train/loop.py` | Per-class loop replaced with per-group loop; per-(image, class) Bernoulli; OOM-ladder closure on flat layout; NaN/denom policy switch from class-count to group-count. (§5, §13 ACs 4-6) |
| `src/custom_sam_peft/eval/evaluator.py` | `_iter_predictions` flat `(image_chunk × group)` loop; `_row_outputs` helper; new `_eval_forward_with_oom_ladder`. (§6, §13 ACs 7, 9) |
| `src/custom_sam_peft/eval/runner.py` | Resolves `cfg.eval.batch_size == "auto"` via `decide_eval_batch_size`. (§6, §13 AC 8) |
| `src/custom_sam_peft/predict/runner.py` | `PredictOptions.batch_size: int | "auto"` with default `"auto"`; flat `(image_chunk × group)` loop. (§6, §13 AC 13) |
| `src/custom_sam_peft/cli/predict_cmd.py` | Forwards string `"auto"` sentinel through. (§7) |
| `src/custom_sam_peft/presets.py` | New `decide_eval_batch_size`; extend `_predicted_bytes` with `mode` param; `forward_only_factor = 0.25`. (§8, §13 AC 12) |
| `tests/unit/test_sam3_wrapper.py` (extend) | 1..16 accept, 0/17 reject, mismatched-class reject, K=1 still passes. (§9, §13 AC 14) |
| `tests/unit/test_sam3_adapter.py` (new) | `(B, K) ∈ {(1,1),(2,3),(4,16)}` `img_ids`/`text_ids` assembly. (§9) |
| `tests/unit/test_geometric_prompt_builder.py` (extend) | `n_cols ≠ len(box_hints)` raises; shapes scale to `(N_max, B·K, 4)`. (§9) |
| `tests/unit/test_train_step.py` (extend) | Multiplex mock returns `(B·K_g, Q, …)`; one `total_loss` per group; backward `/(G·grad_accum)`; closure builds flat hint list. (§9) |
| `tests/unit/test_train_loop_legacy_k1.py` (new) | K=1 RNG-order + numeric regression guard. (§9, §13 AC 14) |
| `tests/unit/test_train_loop_multiplex.py` (new) | K=4; auto-chunk INFO log; `StepResult.n_classes` unchanged. (§9) |
| `tests/unit/test_evaluator.py` (extend) | Flat iteration assertions; `_row_outputs` postprocess. (§9) |
| `tests/unit/test_eval_oom_ladder.py` (new) | Synthetic OOM mid-chunk; B halved once; ≤1 warn. (§9) |
| `tests/unit/test_decide_eval_batch_size.py` (new) | Mocked CUDA + cache; analytic fallback; CPU fallback. (§9) |
| `tests/predict/test_runner_smoke.py` (extend) | Flat iteration; warmup single-image / single-class. (§9) |
| `tests/integration/test_load_sam31_real.py` (extend) | One assertion: real K=8 multiplex forward emits `pred_logits.shape[0] == B*8` and finite outputs. (§9, §13 AC 16) |
| `tests/gpu/test_multiplex_vram.py` (new) | Real `decide_eval_batch_size` at 1008; peak ≤ 4× predicted_bytes. (§9, §13 AC 15) |
| `scripts/bench_multiplex_throughput.py` (new) | Wall-clock K=1 vs K=16 on COCO-80 mini-fixture. (§9) |
| `CHANGELOG.md` | New `## [0.8.0] — 2026-05-23` section with the four entries from §11. (§13 AC 17) |

---

## Task ordering and parallelization map

```
Phase 0  (foundation): T1, T2, T3                       — Parallelizable as a group.
Phase 1  (wrapper):    T4 → T5 → T6                     — Sequential (all in models/sam3.py).
Phase 2  (training):   T7 → T8                          — Sequential (both in train/loop.py).
Phase 3  (eval):       T9 → T10                         — Sequential (both in eval/evaluator.py + eval/runner.py).
Phase 4  (predict):    T11                              — Depends on T4 (wrapper) + T12 (presets sibling).
Phase 5  (vram math):  T12                              — Parallelizable with T7-T11 if they don't touch presets.py.
Phase 6  (gpu tests):  T13, T14                         — Parallelizable with each other after T4-T12 land.
Phase 7  (bench/docs): T15, T16                         — Parallelizable; depend on the above.
```

Tight ordering rules:
- T1-T3 land before any task that imports `MULTIPLEX_CAP` or `MultiplexConfig`.
- T4-T6 must precede T7 because the trainer's tests mock the new wrapper signature.
- T9-T10 may run in parallel with T7-T8 (different files) provided both depend on T1-T3 having merged first.
- T12 (`presets.py`) may run in parallel with T7-T11 (different file).
- T15 and T16 should run last so the changelog reflects all landed work and the benchmark script uses the final API.

---

## Task 1: Add `MULTIPLEX_CAP = 16` constant in `models/sam3.py`

**Files:**
- Modify: `src/custom_sam_peft/models/sam3.py` (top-level, near `Sam3Wrapper`)

**Dependencies:** none. **Parallelizable with:** T2, T3.

**Acceptance:** spec §13 AC 2 (single source of truth for the cap).

- [ ] **Step 1: Write the failing test**

File: `tests/unit/test_sam3_wrapper.py` — append at end:

```python
def test_multiplex_cap_constant_exists() -> None:
    from custom_sam_peft.models.sam3 import MULTIPLEX_CAP

    assert MULTIPLEX_CAP == 16
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_sam3_wrapper.py::test_multiplex_cap_constant_exists -v`
Expected: FAIL with `ImportError: cannot import name 'MULTIPLEX_CAP'`.

- [ ] **Step 3: Write minimal implementation**

In `src/custom_sam_peft/models/sam3.py`, immediately above `class Sam3Wrapper` (current line 165):

```python
# SAM 3.1's multiplex forward is trained at K ≤ 16 class prompts per call.
# This is a model property, not a tunable. Trainer/evaluator/predict cite
# this constant for chunking; see docs/superpowers/specs/2026-05-23-multiplex-forward-design.md §4.
MULTIPLEX_CAP: int = 16
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_sam3_wrapper.py::test_multiplex_cap_constant_exists -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/custom_sam_peft/models/sam3.py tests/unit/test_sam3_wrapper.py
git commit -m "feat(models): add MULTIPLEX_CAP=16 constant (#22)"
```

---

## Task 2: Add `MultiplexConfig` schema + `TrainHyperparams.multiplex` field

**Files:**
- Modify: `src/custom_sam_peft/config/schema.py` (place `MultiplexConfig` near `BoxHintSchedule`, around line 359; add `multiplex` field on `TrainHyperparams` around line 389-409)
- Test: `tests/unit/test_config_schema.py`

**Dependencies:** none. **Parallelizable with:** T1, T3.

**Acceptance:** spec §13 AC 10.

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_config_schema.py`:

```python
def test_multiplex_config_defaults() -> None:
    from custom_sam_peft.config.schema import MultiplexConfig

    cfg = MultiplexConfig()
    assert cfg.classes_per_forward == 16


def test_multiplex_config_validates_range() -> None:
    import pytest
    from pydantic import ValidationError

    from custom_sam_peft.config.schema import MultiplexConfig

    with pytest.raises(ValidationError):
        MultiplexConfig(classes_per_forward=0)
    with pytest.raises(ValidationError):
        MultiplexConfig(classes_per_forward=17)


def test_train_hyperparams_has_multiplex_default() -> None:
    from custom_sam_peft.config.schema import MultiplexConfig, TrainHyperparams

    th = TrainHyperparams(epochs=1)
    assert isinstance(th.multiplex, MultiplexConfig)
    assert th.multiplex.classes_per_forward == 16
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_config_schema.py -v -k "multiplex"`
Expected: FAIL (no `MultiplexConfig` symbol; no `multiplex` field).

- [ ] **Step 3: Write minimal implementation**

In `src/custom_sam_peft/config/schema.py`:

1. Add to `__all__` (alphabetical order in the existing list):

   ```python
   "MultiplexConfig",
   ```

2. Insert `MultiplexConfig` immediately after `BoxHintSchedule` (current end ~line 386):

   ```python
   class MultiplexConfig(_Strict):
       """Multiplex forward knobs.

       classes_per_forward: number of class prompts per multiplex forward pass.
       Capped at SAM 3.1's MULTIPLEX_CAP=16 (in src/custom_sam_peft/models/sam3.py).
       Setting 1 reduces to the legacy per-class regime within the same code path.
       """

       classes_per_forward: int = Field(default=16, ge=1, le=16)
   ```

3. Inside `TrainHyperparams` (current line 389), in the `--- advanced ---` section, append:

   ```python
   multiplex: MultiplexConfig = Field(default_factory=MultiplexConfig)
   ```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_config_schema.py -v -k "multiplex"`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/custom_sam_peft/config/schema.py tests/unit/test_config_schema.py
git commit -m "feat(config): add MultiplexConfig + TrainHyperparams.multiplex (#22)"
```

---

## Task 3: Update `LossConfig` docstring

**Files:**
- Modify: `src/custom_sam_peft/config/_internal.py:34-57`
- Test: `tests/unit/test_loss_config.py`

**Dependencies:** none. **Parallelizable with:** T1, T2.

**Acceptance:** spec §13 AC 11.

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_loss_config.py`:

```python
def test_loss_config_docstring_drops_legacy_claim() -> None:
    from custom_sam_peft.config._internal import LossConfig

    doc = LossConfig.__doc__ or ""
    assert "one forward pass per class prompt" not in doc
    assert "multiplex" in doc.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_loss_config.py::test_loss_config_docstring_drops_legacy_claim -v`
Expected: FAIL (the substring is still present today, per `_internal.py:46-47`).

- [ ] **Step 3: Write minimal implementation**

In `src/custom_sam_peft/config/_internal.py`, replace the two-sentence block at the end of `LossConfig`'s docstring (currently lines 46-47):

```
    No `w_cls`: discrimination across classes comes from running one forward
    pass per class prompt. `w_presence` weights the image-level
    "any-instance-of-this-class-present?" supervision.
```

with:

```
    No `w_cls`: SAM 3.1's multiplex forward provides open-vocabulary
    discrimination directly via per-text-embedding queries; per-class
    `w_cls` is unneeded. `w_presence` weights the image-level
    "any-instance-of-this-class-present?" supervision.
```

Defaults below the docstring are unchanged.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_loss_config.py::test_loss_config_docstring_drops_legacy_claim -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/custom_sam_peft/config/_internal.py tests/unit/test_loss_config.py
git commit -m "docs(config): update LossConfig docstring to reflect multiplex (#22)"
```

---

## Task 4: Lift the K=1 raise in `Sam3Wrapper._validate_inputs`

**Files:**
- Modify: `src/custom_sam_peft/models/sam3.py:229-234` (the `len(p.classes) != 1` block) and any docstring on `Sam3Wrapper` claiming "exactly one class name".
- Test: `tests/unit/test_sam3_wrapper.py`

**Dependencies:** T1 (uses `MULTIPLEX_CAP`).

**Acceptance:** spec §13 AC 1.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_sam3_wrapper.py`:

```python
import pytest
import torch

from custom_sam_peft.data.base import TextPrompts
from custom_sam_peft.models.sam3 import MULTIPLEX_CAP, Sam3Wrapper


def _imgs(b: int) -> torch.Tensor:
    return torch.zeros(b, 3, 8, 8)


def test_validate_inputs_accepts_K_between_1_and_cap() -> None:
    for k in (1, 5, MULTIPLEX_CAP):
        prompts = [TextPrompts(classes=[f"c{i}" for i in range(k)])] * 2
        Sam3Wrapper._validate_inputs(_imgs(2), prompts, None)


def test_validate_inputs_rejects_K_zero() -> None:
    with pytest.raises(ValueError, match="MULTIPLEX_CAP"):
        Sam3Wrapper._validate_inputs(_imgs(1), [TextPrompts(classes=[])], None)


def test_validate_inputs_rejects_K_over_cap() -> None:
    too_many = [f"c{i}" for i in range(MULTIPLEX_CAP + 1)]
    with pytest.raises(ValueError, match="MULTIPLEX_CAP"):
        Sam3Wrapper._validate_inputs(_imgs(1), [TextPrompts(classes=too_many)], None)


def test_validate_inputs_rejects_mismatched_class_lists_across_batch() -> None:
    prompts = [TextPrompts(classes=["cat", "dog"]), TextPrompts(classes=["dog", "cat"])]
    with pytest.raises(ValueError, match="same.*class"):
        Sam3Wrapper._validate_inputs(_imgs(2), prompts, None)


def test_validate_inputs_k1_still_passes() -> None:
    Sam3Wrapper._validate_inputs(
        _imgs(3),
        [TextPrompts(classes=["cat"]) for _ in range(3)],
        None,
    )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_sam3_wrapper.py -v -k "validate_inputs"`
Expected: FAIL on the >1-class cases (current code raises on `len(p.classes) != 1`).

- [ ] **Step 3: Write minimal implementation**

In `src/custom_sam_peft/models/sam3.py:_validate_inputs`, replace the `isinstance(p, TextPrompts) and len(p.classes) != 1` block (lines 229-234) with two checks. The first check enforces `1 <= K <= MULTIPLEX_CAP`; the second enforces shared class list. Place the new code where the old block sat:

```python
            if isinstance(p, TextPrompts):
                if not (1 <= len(p.classes) <= MULTIPLEX_CAP):
                    raise ValueError(
                        f"TextPrompts must contain 1..{MULTIPLEX_CAP} classes per "
                        f"call (got {len(p.classes)}). Configure "
                        f"train.multiplex.classes_per_forward to bound K."
                    )

        # After the per-prompt loop, enforce shared class list for TextPrompts.
        if first is TextPrompts:
            ref = tuple(cast(TextPrompts, prompts[0]).classes)
            for p in prompts[1:]:
                if tuple(cast(TextPrompts, p).classes) != ref:
                    raise ValueError(
                        "All TextPrompts in a batch must carry the same class "
                        "list in the same order (multiplex forward assumes a "
                        "shared K-prompt vocabulary)."
                    )
```

Add `cast` to the existing `typing` import if it is not already imported (sam3.py already imports `cast` on line 17 — confirm before edit).

Also update the `Sam3Wrapper` docstring (lines 165-190) line that says "each image's prompt MUST contain exactly one class name" to "each image's prompt may contain 1..MULTIPLEX_CAP class names; all prompts in a batch must share the same class list".

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_sam3_wrapper.py -v -k "validate_inputs"`
Expected: all PASS (5 cases).

- [ ] **Step 5: Commit**

```bash
git add src/custom_sam_peft/models/sam3.py tests/unit/test_sam3_wrapper.py
git commit -m "feat(wrapper): accept 1..MULTIPLEX_CAP classes per TextPrompts (#22)"
```

---

## Task 5: Parameterize `_build_geometric_prompt` on `n_cols`

**Files:**
- Modify: `src/custom_sam_peft/models/sam3.py:108-162` (function signature + body) and every existing caller (only `_Sam3ImageAdapter.forward` at sam3.py:342).
- Test: `tests/unit/test_geometric_prompt_builder.py`

**Dependencies:** T4 (touch same file; sequential to avoid merge churn).

**Acceptance:** spec §13 AC 3 (geometric prompt shape `(N_boxes_max, B·K, 4)`).

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_geometric_prompt_builder.py`:

```python
import pytest
import torch

from custom_sam_peft.models.sam3 import _build_geometric_prompt


def test_build_geometric_prompt_n_cols_must_match_len_box_hints() -> None:
    hints = [None, None, None]
    with pytest.raises(ValueError, match="n_cols"):
        _build_geometric_prompt(hints, n_cols=4, image_size=1008, device=torch.device("cpu"))


def test_build_geometric_prompt_produces_n_cols_columns() -> None:
    # 2 images × 3 classes = 6 columns (B·K layout).
    hints = [torch.tensor([[0.0, 0.0, 10.0, 10.0]]) for _ in range(6)]
    out = _build_geometric_prompt(
        hints, n_cols=6, image_size=1008, device=torch.device("cpu")
    )
    assert out is not None
    # box_embeddings: (N_max, n_cols, 4); box_mask: (n_cols, N_max).
    assert out.box_embeddings.shape == (1, 6, 4)
    assert out.box_mask.shape == (6, 1)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_geometric_prompt_builder.py -v -k "n_cols"`
Expected: FAIL (function does not accept `n_cols`).

- [ ] **Step 3: Write minimal implementation**

Change the signature at `models/sam3.py:108`:

```python
def _build_geometric_prompt(
    box_hints: list[Tensor | None],
    n_cols: int,
    image_size: int,
    device: torch.device,
) -> Prompt | None:
```

Inside the function (today's lines 129-162):
- Add at the top after the `all(h is None …)` check: `if len(box_hints) != n_cols: raise ValueError(f"len(box_hints)={len(box_hints)} must equal n_cols={n_cols}")`.
- Replace `b = len(box_hints)` (current line 132) with `b = n_cols`.
- Body stays otherwise unchanged; outputs become `(N_max, n_cols, 4)` and `(n_cols, N_max)` automatically.

Update the caller at `models/sam3.py:342-346` (will be revisited in T6) to pass `n_cols=b` for now (preserves K=1 behavior until T6 widens it):

```python
gp = _build_geometric_prompt(
    box_hints if box_hints is not None else [None] * b,
    n_cols=b,
    image_size=self.image_size,
    device=device,
)
```

- [ ] **Step 4: Run tests to verify they pass**

Run all builder tests: `pytest tests/unit/test_geometric_prompt_builder.py -v`
Expected: PASS (existing tests still pass because their `n_cols == len(box_hints)`).

- [ ] **Step 5: Commit**

```bash
git add src/custom_sam_peft/models/sam3.py tests/unit/test_geometric_prompt_builder.py
git commit -m "refactor(wrapper): parameterize _build_geometric_prompt on n_cols (#22)"
```

---

## Task 6: Multiplex `_Sam3ImageAdapter.forward` to `(B·K, …)` rows

**Files:**
- Modify: `src/custom_sam_peft/models/sam3.py:309-360` (`_Sam3ImageAdapter.forward`).
- Test: `tests/unit/test_sam3_adapter.py` (NEW file)

**Dependencies:** T1 (uses `MULTIPLEX_CAP`), T4 (validation must already accept K>1), T5 (`_build_geometric_prompt` accepts `n_cols`).

**Acceptance:** spec §13 AC 3.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_sam3_adapter.py`:

```python
"""Unit tests for _Sam3ImageAdapter.forward multiplex assembly.

We mock the inner model's backbone.forward_image / backbone.forward_text /
forward_grounding so the test exercises only the adapter's input shaping
(img_ids, text_ids, geometric_prompt column count).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch

from custom_sam_peft.data.base import TextPrompts
from custom_sam_peft.models.sam3 import _Sam3ImageAdapter


def _make_fake_inner(captured: dict[str, object]) -> MagicMock:
    inner = MagicMock()
    inner.parameters.return_value = iter([torch.zeros(1, dtype=torch.float32)])
    inner.backbone.forward_image.return_value = {"feat": torch.zeros(1)}
    inner.backbone.forward_text.return_value = {"text_feat": torch.zeros(1)}

    def _grounding(*, backbone_out, find_input, find_target, geometric_prompt):
        captured["find_input"] = find_input
        captured["geometric_prompt"] = geometric_prompt
        # Return dummy outputs shaped (B·K, Q, *) — Q=2 here.
        n_rows = find_input.img_ids.shape[0]
        return {
            "pred_logits": torch.zeros(n_rows, 2, 1),
            "pred_boxes": torch.zeros(n_rows, 2, 4),
            "pred_masks": torch.zeros(n_rows, 2, 4, 4),
            "presence_logit_dec": torch.zeros(n_rows, 1),
        }

    inner.forward_grounding.side_effect = _grounding
    return inner


@pytest.mark.parametrize("b,k", [(1, 1), (2, 3), (4, 16)])
def test_adapter_builds_img_text_ids_image_major(b: int, k: int) -> None:
    captured: dict[str, object] = {}
    inner = _make_fake_inner(captured)
    adapter = _Sam3ImageAdapter(inner, image_size=8)

    images = torch.zeros(b, 3, 8, 8)
    classes = [f"c{i}" for i in range(k)]
    prompts = [TextPrompts(classes=classes) for _ in range(b)]

    out = adapter(images, prompts, box_hints=None)

    find_input = captured["find_input"]
    # image-major / class-minor: img_ids = arange(B).repeat_interleave(K)
    assert torch.equal(
        find_input.img_ids,
        torch.arange(b).repeat_interleave(k),
    )
    # text_ids = arange(K).repeat(B)
    assert torch.equal(
        find_input.text_ids,
        torch.arange(k).repeat(b),
    )
    # output first dim is B·K
    assert out["pred_logits"].shape[0] == b * k


@pytest.mark.parametrize("b,k", [(2, 3), (4, 16)])
def test_adapter_calls_forward_text_once_with_k_names(b: int, k: int) -> None:
    captured: dict[str, object] = {}
    inner = _make_fake_inner(captured)
    adapter = _Sam3ImageAdapter(inner, image_size=8)

    classes = [f"c{i}" for i in range(k)]
    prompts = [TextPrompts(classes=classes) for _ in range(b)]
    adapter(torch.zeros(b, 3, 8, 8), prompts, box_hints=None)

    # forward_text called exactly once with the K class names.
    assert inner.backbone.forward_text.call_count == 1
    args, kwargs = inner.backbone.forward_text.call_args
    # First positional arg is the list of class names.
    assert args[0] == classes
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_sam3_adapter.py -v`
Expected: FAIL — today's adapter only handles K=1 and calls `forward_text([class_names[0]])`.

- [ ] **Step 3: Write minimal implementation**

Replace the body of `_Sam3ImageAdapter.forward` (sam3.py:309-360) with multiplex assembly. Key shape changes:

```python
    def forward(
        self,
        images: Tensor,
        prompts: list[Prompts],
        box_hints: list[Tensor | None] | None = None,
    ) -> dict[str, Tensor]:
        if not all(isinstance(p, TextPrompts) for p in prompts):
            raise ValueError("_Sam3ImageAdapter only supports TextPrompts in v0")
        text_prompts = cast(list[TextPrompts], prompts)
        # Validator (Sam3Wrapper._validate_inputs) guarantees a shared class list.
        classes = list(text_prompts[0].classes)
        k = len(classes)
        device = images.device
        b = images.shape[0]
        model_dtype = next(self.model.parameters()).dtype

        backbone_out = self.model.backbone.forward_image(images)  # type: ignore[union-attr, operator]
        text_outputs = self.model.backbone.forward_text(  # type: ignore[union-attr, operator]
            classes, device=device
        )
        backbone_out.update(text_outputs)

        find_input = FindStage(
            img_ids=torch.arange(b, device=device, dtype=torch.long).repeat_interleave(k),
            text_ids=torch.arange(k, device=device, dtype=torch.long).repeat(b),
            input_boxes=None,
            input_boxes_mask=None,
            input_boxes_label=None,
            input_points=None,
            input_points_mask=None,
        )
        n_cols = b * k
        gp = _build_geometric_prompt(
            box_hints if box_hints is not None else [None] * n_cols,
            n_cols=n_cols,
            image_size=self.image_size,
            device=device,
        )
        if gp is None:
            gp = Prompt(
                box_embeddings=torch.zeros(0, n_cols, 4, device=device, dtype=model_dtype),
                box_mask=torch.zeros(n_cols, 0, device=device, dtype=torch.bool),
                point_embeddings=torch.zeros(0, n_cols, 2, device=device, dtype=model_dtype),
                point_mask=torch.zeros(n_cols, 0, device=device, dtype=torch.bool),
            )
        outputs: dict[str, Tensor] = self.model.forward_grounding(  # type: ignore[operator]
            backbone_out=backbone_out,
            find_input=find_input,
            find_target=None,
            geometric_prompt=gp,
        )
        return outputs
```

Notes:
- The internal mismatch check (today `if len(set(class_names)) > 1`) is gone — `Sam3Wrapper._validate_inputs` now owns that contract.
- `box_hints` semantics widen: callers must now pass a length-`B·K` flat list ordered image-major / class-minor (or `None`); validator's contract update is documented on the wrapper docstring.

Update `Sam3Wrapper`'s contract docstring (lines 165-190) to reflect the new `box_hints` length-`B·K` requirement.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_sam3_adapter.py tests/unit/test_sam3_wrapper.py tests/unit/test_geometric_prompt_builder.py -v`
Expected: PASS. Also re-run `tests/unit/test_sam3_wrapper_box_hints.py` — must still pass (single-class path still works).

- [ ] **Step 5: Commit**

```bash
git add src/custom_sam_peft/models/sam3.py tests/unit/test_sam3_adapter.py
git commit -m "feat(adapter): multiplex forward assembles (B*K, ...) rows (#22)"
```

---

## Task 7: Replace train_step per-class loop with per-group loop

**Files:**
- Modify: `src/custom_sam_peft/train/loop.py:171-324` (`train_step`); add `_chunked` helper near top of file.
- Test: `tests/unit/test_train_step.py` (extend); new `tests/unit/test_train_loop_multiplex.py`; new `tests/unit/test_train_loop_legacy_k1.py`.

**Dependencies:** T1, T2, T4, T6.

**Acceptance:** spec §13 ACs 4, 5, 6.

This is the largest task. It rewrites the inner loop, the OOM-ladder closure, the NaN policy, and the accumulator denominator. Land all sub-steps in one commit so the file is never half-converted.

- [ ] **Step 1: Write the failing tests**

(a) Extend `tests/unit/test_train_step.py`. Find the existing main-path test (look for the first test that mocks `model(...)` to return `(1, Q, 1)`-shaped logits, around the test asserting `class_losses` is summed across `classes_in_batch`). Replace the mock's return shapes from `(B, Q, *)` to `(B*K_g, Q, *)` and assert:

```python
def test_train_step_one_total_loss_per_group(monkeypatch) -> None:
    """One total_loss call per chunked group; backward divides by G*grad_accum."""
    # Build a batch with two distinct classes -> K_total = 2.
    # With classes_per_forward=16 (default), G == 1, K_g == 2.
    # Mock model.forward to assert it is called exactly once with K=2 prompts.
    # Mock total_loss to count invocations and inspect args.
    ...
    assert model.call_count == 1
    assert len(prompts_passed[0].classes) == 2
    assert total_loss_call_count == 1
    # backward divided by G (==1) * grad_accum
    assert observed_scale == pytest.approx(1.0 / (1 * cfg.train.grad_accum_steps))
```

(b) Extend the existing OOM-ladder test in `tests/unit/test_train_step.py` (or `test_trainer_oom_retry.py`) — the closure must build a length-`|micro|·K_g` flat hint list:

```python
def test_oom_ladder_closure_builds_flat_K_g_hint_list(monkeypatch) -> None:
    """When the ladder halves B from 2 -> 1, the closure passes 1*K_g hints."""
    # Build a 2-image batch with K_g = 3 classes in one group.
    # Force first attempt OOM; assert second attempt receives 1*3 = 3 hints.
    ...
    assert len(captured_hints_lists[-1]) == 3
```

(c) New file `tests/unit/test_train_loop_legacy_k1.py`:

```python
"""Regression guard: at classes_per_forward=1, train_step is RNG-order and
numerically equivalent to today's per-class loop. Locked decision §10 R3.

Strategy:
  - Seed before train_step.
  - Use a deterministic Bernoulli probe: monkeypatch `random.random` to record
    every call's index, then check the order matches the legacy `for c: for i:`
    sequence (which collapses to one class per group at K=1).
  - Use a mocked model returning a fixed loss; check the final accum dict's
    `total` value matches what the legacy code produces (numerically).
"""

import random
from unittest.mock import MagicMock

import pytest
import torch

from custom_sam_peft.config.schema import MultiplexConfig, TrainConfig
from custom_sam_peft.train.loop import train_step
# ... (the test builds a minimal TrainConfig with multiplex.classes_per_forward=1,
#      runs train_step, and asserts ordered RNG draws + total loss value)
```

(d) New file `tests/unit/test_train_loop_multiplex.py`:

```python
"""Multiplex behavior with K_total > MULTIPLEX_CAP triggers auto-chunk INFO log."""

import logging
import pytest
import torch

from custom_sam_peft.models.sam3 import MULTIPLEX_CAP
# ... mocks + cfg with K_total = 20 (> MULTIPLEX_CAP=16) -> G==2 groups

def test_auto_chunk_logs_once_when_classes_exceed_cap(caplog) -> None:
    caplog.set_level(logging.INFO)
    # ... call train_step
    msgs = [r.message for r in caplog.records if "multiplex auto-chunk" in r.message]
    assert len(msgs) == 1


def test_step_result_n_classes_equals_K_total() -> None:
    # ... K_total = 5, default cap = 16 -> G = 1
    assert result.n_classes == 5  # K_total, not G

def test_K_4_multiplex_calls_model_once() -> None:
    # K_total = 4, classes_per_forward = 16 -> one model call with K=4
    assert mock_model.call_count == 1
    assert len(prompts_passed[0].classes) == 4
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```
pytest tests/unit/test_train_step.py tests/unit/test_train_loop_legacy_k1.py tests/unit/test_train_loop_multiplex.py -v
```
Expected: FAIL on the new assertions; the new files fail collection (test methods missing implementations until you copy in the bodies).

- [ ] **Step 3: Write minimal implementation**

In `src/custom_sam_peft/train/loop.py`:

(a) Add a `_chunked` helper near the top (after imports, before `OomState`):

```python
def _chunked(seq: list[str], n: int) -> list[list[str]]:
    """Split seq into consecutive chunks of size ≤ n. Preserves order."""
    if n <= 0:
        raise ValueError(f"_chunked: n must be positive; got {n}")
    return [seq[i : i + n] for i in range(0, len(seq), n)]
```

(b) Add module-level `_AUTO_CHUNK_LOGGED = False` flag and a helper that logs once per run.

(c) Rewrite `train_step` body (current lines 198-323). Pseudocode (full implementation follows the pattern below):

```python
classes_in_batch = sorted({c for p in prompts for c in p.classes})
if not classes_in_batch:
    return StepResult.empty(...)

from custom_sam_peft.models.sam3 import MULTIPLEX_CAP

effective_K = min(cfg.train.multiplex.classes_per_forward, MULTIPLEX_CAP)
groups = _chunked(classes_in_batch, effective_K)
G = len(groups)

global _AUTO_CHUNK_LOGGED
if len(classes_in_batch) > MULTIPLEX_CAP and not _AUTO_CHUNK_LOGGED:
    _LOG.info(
        "multiplex auto-chunk: classes_in_batch=%d > MULTIPLEX_CAP=%d -> %d groups",
        len(classes_in_batch), MULTIPLEX_CAP, G,
    )
    _AUTO_CHUNK_LOGGED = True

accum = {"mask": 0.0, "box": 0.0, "obj": 0.0, "presence": 0.0, "total": 0.0}
finite_group_count = 0
n_hint_applied = 0

if oom_state is not None:
    oom_state.step = global_step

for group in groups:
    K_g = len(group)
    prompts_g = [TextPrompts(classes=list(group)) for _ in range(B)]

    # Image-major, class-minor: hints[i*K_g + j] is image i, class group[j].
    hints_g: list[Tensor | None] = []
    targets_g: list[list[Instance]] = []
    for i in range(B):
        for c in group:
            c_dense = class_names.index(c)
            row_targets = [inst for inst in targets[i] if inst.class_id == c_dense]
            targets_g.append(row_targets)
            if row_targets and random.random() < p_t:  # noqa: S311
                hints_g.append(to_device(torch.stack([inst.box for inst in row_targets]), runtime))
                n_hint_applied += 1
            else:
                hints_g.append(None)

    group_losses: dict[str, Tensor] | None = None
    group_scaled: Tensor | None = None
    is_finite = False
    try:
        if oom_state is not None:
            # OOM ladder closure on flat-row layout
            _last_group_losses: list[dict[str, Tensor]] = []

            def _forward_group(
                _model, micro_indices,
                _prompts_g=prompts_g, _targets_g=targets_g, _hints_g=hints_g,
                _K_g=K_g, _G=G, _grad_accum=cfg.train.grad_accum_steps,
                _out=_last_group_losses, _pm=_peft_method,
            ):
                micro_prompts = [_prompts_g[i] for i in micro_indices]
                # Slice the flat lists at image granularity: image i contributes
                # rows [i*K_g : (i+1)*K_g].
                row_slices: list[int] = []
                for i in micro_indices:
                    row_slices.extend(range(i * _K_g, (i + 1) * _K_g))
                micro_targets = [_targets_g[r] for r in row_slices]
                micro_hints = [_hints_g[r] for r in row_slices]
                micro_imgs = images[micro_indices]
                with _autocast_ctx(cfg, _pm):
                    micro_out = _model(micro_imgs, micro_prompts, box_hints=micro_hints)
                    micro_losses = total_loss(micro_out, micro_targets, cfg.train.loss)
                _out.clear()
                _out.append(micro_losses)
                return micro_losses["total"] / (_G * _grad_accum)

            image_indices = list(range(B))
            _train_step_with_oom_ladder(
                model, image_indices, oom_state, forward_call=_forward_group
            )
            group_losses = _last_group_losses[0] if _last_group_losses else None
            if group_losses is not None:
                group_scaled_val = group_losses["total"] / (G * cfg.train.grad_accum_steps)
                is_finite = bool(torch.isfinite(group_scaled_val))
        else:
            with _autocast_ctx(cfg, _peft_method):
                out = model(images, prompts_g, box_hints=hints_g)
                group_losses = total_loss(out, targets_g, cfg.train.loss)
            group_scaled = group_losses["total"] / (G * cfg.train.grad_accum_steps)
            is_finite = bool(torch.isfinite(group_scaled))
    except ValueError as exc:
        _LOG.warning("train_step: group %r raised %s; treating as non-finite.", group, exc)
        is_finite = False

    if is_finite and group_losses is not None:
        if oom_state is None and group_scaled is not None:
            group_scaled.backward()
        finite_group_count += 1
        for k in ("mask", "box", "obj", "presence", "total"):
            accum[k] += float(group_losses[k].detach())

skipped = finite_group_count == 0
new_streak = nan_streak + 1 if skipped else 0
if new_streak >= cfg.train.nan_abort_after:
    raise RuntimeError(f"Training aborted: {new_streak} consecutive non-finite micro-steps.")

# optimizer.step() block unchanged.

return StepResult(
    losses={k: v / max(finite_group_count, 1) for k, v in accum.items()},
    p_t=p_t,
    n_hint_applied=n_hint_applied,
    n_classes=len(classes_in_batch),  # contract unchanged: K_total
    grad_norm=grad_norm,
    skipped=skipped,
    nan_streak=new_streak,
    images_processed=B,
)
```

The `_ScalarWindow.update`'s denominator at line 359 (`r.n_classes * max(r.images_processed, 1)`) stays — `n_classes` still tracks `K_total`, so the denominator still totals `B · K_total` hint-applied attempts. No change to `_ScalarWindow`.

The `_AUTO_CHUNK_LOGGED` flag is module-level. For test isolation, expose a reset helper (e.g. `_reset_auto_chunk_log()`) used in test fixtures.

- [ ] **Step 4: Run tests to verify they pass**

Run:
```
pytest tests/unit/test_train_step.py tests/unit/test_train_loop_multiplex.py tests/unit/test_train_loop_legacy_k1.py tests/unit/test_trainer_oom_retry.py tests/unit/test_trainer_nan_behavior.py -v
```
Expected: PASS. Re-run all `tests/unit/test_trainer_*.py` to confirm no regressions in the wider training-loop test surface.

- [ ] **Step 5: Commit**

```bash
git add src/custom_sam_peft/train/loop.py tests/unit/test_train_step.py tests/unit/test_train_loop_multiplex.py tests/unit/test_train_loop_legacy_k1.py
git commit -m "feat(train): one forward + one total_loss per multiplex group (#22)"
```

---

## Task 8: Verify trainer integration tests still pass

**Files:** none modified; runs the existing end-to-end CPU integration tests.

**Dependencies:** T7.

**Acceptance:** spec §13 ACs 4-6 confirmed end-to-end on CPU; spec §10 R3 (K=1 equivalence) holds.

- [ ] **Step 1: Run the CPU integration suite**

```
pytest tests/integration/test_train_end_to_end.py tests/integration/test_train_then_eval.py tests/integration/test_train_resume.py tests/integration/test_trainer_evaluator_seam.py -v
```

- [ ] **Step 2: If any fail**

Diagnose. Common causes:
- Stale mocks expecting `model(...)` called K_total times → adjust mocks to expect G calls.
- `StepResult.losses` magnitude assertions → update tolerance or assertion to track the §10 R1 shift.
- `n_classes` assertions remain unchanged (still K_total).

Fix and commit:
```bash
git add tests/integration/...
git commit -m "test(integration): align train integration tests with multiplex (#22)"
```

- [ ] **Step 3: Final smoke run**

```
pytest tests/unit tests/integration -v -q
```
All green before proceeding to T9.

---

## Task 9: Rewrite `Evaluator._iter_predictions` flat over (image_chunk × group)

**Files:**
- Modify: `src/custom_sam_peft/eval/evaluator.py:108-169` (`_iter_predictions`); add `_row_outputs` helper; add `_eval_forward_with_oom_ladder` module-private helper.
- Test: `tests/unit/test_evaluator.py` (extend); `tests/unit/test_eval_oom_ladder.py` (NEW)

**Dependencies:** T1, T6.

**Acceptance:** spec §13 ACs 7, 9.

- [ ] **Step 1: Write the failing tests**

(a) Extend `tests/unit/test_evaluator.py`. Identify the existing test that asserts model is called per (image, class). Replace its assertion with the flat-iteration shape:

```python
def test_iter_predictions_iterates_image_chunks_x_groups(monkeypatch) -> None:
    """Evaluator iterates (image_chunk, class_group) flat; one model call per pair."""
    # batch_size = 2 (resolved), 4 images, 3 classes, MULTIPLEX_CAP=16 -> 1 group per chunk.
    # Expect ceil(4/2) * 1 = 2 model calls.
    ...
    assert model.call_count == 2
    # Each call carries K_g = 3 class names.
    for args, _ in model.call_args_list:
        prompts_arg = args[1]
        assert all(len(p.classes) == 3 for p in prompts_arg)


def test_row_outputs_returns_single_row_dict() -> None:
    """_row_outputs(outputs, r) returns a per-row dict shaped (1, ...) for postprocess."""
    from custom_sam_peft.eval.evaluator import _row_outputs
    outputs = {
        "pred_logits": torch.zeros(6, 2, 1),
        "pred_boxes": torch.zeros(6, 2, 4),
        "pred_masks": torch.zeros(6, 2, 4, 4),
        "presence_logit_dec": torch.zeros(6, 1),
    }
    row = _row_outputs(outputs, r=3)
    assert row["pred_logits"].shape == (1, 2, 1)
    assert row["pred_boxes"].shape == (1, 2, 4)
```

(b) Create `tests/unit/test_eval_oom_ladder.py`:

```python
"""_eval_forward_with_oom_ladder: synthetic OOM mid-chunk halves B sticky;
single _LOG.warning per evaluate() call."""

import logging
import pytest
import torch

from custom_sam_peft.eval.evaluator import _eval_forward_with_oom_ladder, _OomCounter


def test_oom_halves_batch_size_sticky_and_warns_once(caplog) -> None:
    # Synthetic forward raising torch.cuda.OutOfMemoryError on first attempt at B=4.
    # Assert: ladder halves to 2 (or 1), single _LOG.warning emitted.
    caplog.set_level(logging.WARNING)
    ...
    warns = [r for r in caplog.records if "eval OOM" in r.message]
    assert len(warns) == 1


def test_oom_raises_at_B1_floor() -> None:
    """Persistent OOM at B=1 raises a RuntimeError."""
    ...
    with pytest.raises(RuntimeError, match="OOM"):
        ...
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```
pytest tests/unit/test_evaluator.py tests/unit/test_eval_oom_ladder.py -v
```
Expected: FAIL — current code is per-image-per-class and no `_eval_forward_with_oom_ladder` helper exists.

- [ ] **Step 3: Write minimal implementation**

In `src/custom_sam_peft/eval/evaluator.py`:

(a) Add module-private helpers near the top of the file (after the existing helper imports, around line 30):

```python
def _chunked(seq, n):
    """Tiny local helper; mirrors train/loop.py:_chunked."""
    if n <= 0:
        raise ValueError(f"_chunked: n must be positive; got {n}")
    return [seq[i : i + n] for i in range(0, len(seq), n)]


def _row_outputs(outputs: dict[str, Tensor], r: int) -> dict[str, Tensor]:
    """Slice multiplex outputs at row r, preserving the batch dim (size 1)."""
    return {k: v[r : r + 1] for k, v in outputs.items()}


def _eval_forward_with_oom_ladder(
    model: Any,
    images: Tensor,
    prompts: list[Any],
    *,
    state: dict[str, Any],  # mutable: {"batch_size": int, "warned": bool}
) -> dict[str, Tensor]:
    """One multiplex forward with sticky-B-halving on OOM.

    No grad-checkpoint rung (eval is under no_grad). Halves the image
    dimension within the chunk; on OOM at B=1, raises RuntimeError.
    state["batch_size"] persists across calls so halving is sticky for
    the rest of the evaluate() call. state["warned"] caps log spam at one.
    """
    while True:
        try:
            return model(images, prompts, box_hints=None)
        except torch.cuda.OutOfMemoryError as oom_err:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if state["batch_size"] > 1:
                state["batch_size"] //= 2
                if not state["warned"]:
                    _LOG.warning(
                        "eval OOM — halving batch_size to %d", state["batch_size"]
                    )
                    state["warned"] = True
                # caller is expected to slice the chunk smaller and retry; here we
                # raise to bubble up so the outer loop re-chunks at the new size.
                raise
            raise RuntimeError(
                "eval OOM at batch_size=1; use a larger GPU or smaller image_size."
            ) from oom_err
```

NOTE on contract: the ladder *raises* on OOM after halving so the outer chunking loop re-issues with the smaller batch. The single-warn invariant is held by `state["warned"]`. This is simpler than the train ladder because there is no `/n_micro` math; the outer iterator re-chunks naturally.

(b) Rewrite `_iter_predictions` (lines 108-169). The new body iterates `(image_chunk, class_group)`:

```python
def _iter_predictions(
    self, model: Any, examples: Sequence[Example], dataset: Dataset
) -> list[dict[str, object]]:
    cfg = self.cfg
    was_training = bool(getattr(model, "training", False))
    if hasattr(model, "eval"):
        model.eval()
    try:
        param_device = next(model.parameters()).device
    except (StopIteration, AttributeError):
        param_device = torch.device("cpu")
    eval_runtime = Runtime(device=param_device, dtype=torch.float32)

    from custom_sam_peft.models.sam3 import MULTIPLEX_CAP

    # cfg.batch_size is already resolved by run_eval (T10) — int here.
    state = {"batch_size": cfg.batch_size, "warned": False}

    predictions: list[dict[str, object]] = []
    log_every_n = max(1, len(examples) // 50)
    P.reset_inner(total=len(examples))
    img_idx_global = 0
    try:
        with torch.no_grad():
            i = 0
            while i < len(examples):
                # Re-chunk based on the (possibly halved) state["batch_size"].
                bs = state["batch_size"]
                image_chunk = list(examples[i : i + bs])
                images_t = to_device(
                    torch.stack([ex.image for ex in image_chunk]), eval_runtime
                )
                advanced_i = False
                for group in _chunked(dataset.class_names, MULTIPLEX_CAP):
                    K_g = len(group)
                    prompts_g = [TextPrompts(classes=list(group)) for _ in image_chunk]
                    try:
                        outputs = _eval_forward_with_oom_ladder(
                            model, images_t, prompts_g, state=state
                        )
                    except torch.cuda.OutOfMemoryError:
                        # state["batch_size"] was halved; re-chunk from i.
                        break
                    for r in range(len(image_chunk) * K_g):
                        ii, kk = divmod(r, K_g)
                        ex = image_chunk[ii]
                        original_hw = (
                            int(ex.image.shape[-2]),
                            int(ex.image.shape[-1]),
                        )
                        int_id = _int_image_id(ex.image_id)
                        cat_idx = dataset.class_names.index(group[kk])
                        entries = queries_to_coco_results(
                            _row_outputs(outputs, r),
                            int_id,
                            cat_idx + 1,
                            original_hw,
                            cfg.mask_threshold,
                        )
                        predictions.extend(entries)
                else:
                    # No break: we completed all groups for this image_chunk.
                    advanced_i = True
                if advanced_i:
                    i += len(image_chunk)
                    img_idx_global += len(image_chunk)
                    P.advance_inner(len(image_chunk))
                    if img_idx_global % log_every_n == 0:
                        P.update_postfix(it_s=float(img_idx_global))
    finally:
        if was_training and hasattr(model, "train"):
            model.train()
    return predictions
```

`P.advance_inner` takes no count today — the existing single-step `advance_inner()` is called once per image; we mirror that by calling it in a loop or by adding a count param. **Implementer:** check the current `progress` API in `cli/_progress.py`; preserve its semantics.

(c) `cfg.batch_size` resolution: the next task (T10) ensures `EvalConfig.batch_size` is an int by the time `_iter_predictions` runs. For now, treat `cfg.batch_size` as `int` here; T10 wires the `"auto"` resolution.

- [ ] **Step 4: Run tests to verify they pass**

Run:
```
pytest tests/unit/test_evaluator.py tests/unit/test_eval_oom_ladder.py -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/custom_sam_peft/eval/evaluator.py tests/unit/test_evaluator.py tests/unit/test_eval_oom_ladder.py
git commit -m "feat(eval): flat (image_chunk, group) loop + OOM ladder (#22)"
```

---

## Task 10: `EvalConfig.batch_size` field + `run_eval` resolves `"auto"`

**Files:**
- Modify: `src/custom_sam_peft/config/schema.py:412-420` (`EvalConfig`).
- Modify: `src/custom_sam_peft/eval/runner.py:56-169` (`run_eval`).
- Test: `tests/unit/test_evaluator_schema.py`; `tests/unit/test_eval_runner.py`.

**Dependencies:** T9 (evaluator must consume `cfg.batch_size` as int); T12 (`decide_eval_batch_size` exists).

**Acceptance:** spec §13 AC 8.

- [ ] **Step 1: Write the failing tests**

(a) `tests/unit/test_evaluator_schema.py` — add:

```python
def test_eval_config_batch_size_default_auto() -> None:
    from custom_sam_peft.config.schema import EvalConfig

    cfg = EvalConfig()
    assert cfg.batch_size == "auto"


def test_eval_config_batch_size_accepts_positive_int() -> None:
    from custom_sam_peft.config.schema import EvalConfig

    assert EvalConfig(batch_size=4).batch_size == 4


def test_eval_config_batch_size_rejects_zero() -> None:
    import pytest
    from pydantic import ValidationError

    from custom_sam_peft.config.schema import EvalConfig

    with pytest.raises(ValidationError):
        EvalConfig(batch_size=0)
```

(b) `tests/unit/test_eval_runner.py` — add:

```python
def test_run_eval_resolves_auto_via_decide_eval_batch_size(monkeypatch) -> None:
    """run_eval calls presets.decide_eval_batch_size when cfg.eval.batch_size == 'auto'."""
    called = {}

    def _fake_decide(image_size, classes_per_forward=16):
        called["image_size"] = image_size
        called["k"] = classes_per_forward
        return (3, 1, "analytic")

    monkeypatch.setattr("custom_sam_peft.presets.decide_eval_batch_size", _fake_decide)
    # ... build cfg with batch_size='auto' + minimal model/dataset stubs
    # ... call run_eval
    assert called["image_size"] == cfg.data.image_size
    assert called["k"] == 16
    # Evaluator instance received batch_size=3
    ...


def test_run_eval_cpu_fallback_logs_info(caplog) -> None:
    """On CPU, decide_eval_batch_size returns 1; run_eval logs once."""
    import logging
    caplog.set_level(logging.INFO)
    # ... force CUDA-unavailable path
    # ... run_eval
    assert any("eval.batch_size=auto on CPU" in r.message for r in caplog.records)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_evaluator_schema.py tests/unit/test_eval_runner.py -v -k "batch_size or resolves_auto or cpu_fallback"`
Expected: FAIL.

- [ ] **Step 3: Write minimal implementation**

(a) In `schema.py`, add to `EvalConfig` (after `save_predictions` at line 420):

```python
    batch_size: PositiveInt | Literal["auto"] = "auto"
```

(b) In `eval/runner.py:run_eval`, just before `evaluator = Evaluator(eval_cfg)` (line 139), resolve `"auto"`:

```python
    eval_cfg = cfg.eval
    if save_predictions is not None:
        eval_cfg = eval_cfg.model_copy(update={"save_predictions": save_predictions})

    if eval_cfg.batch_size == "auto":
        from custom_sam_peft.presets import decide_eval_batch_size

        bs, _, _ = decide_eval_batch_size(cfg.data.image_size, classes_per_forward=16)
        eval_cfg = eval_cfg.model_copy(update={"batch_size": bs})
```

`decide_eval_batch_size` is responsible for logging the CPU-fallback INFO message (per T12).

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_evaluator_schema.py tests/unit/test_eval_runner.py -v -k "batch_size or resolves_auto or cpu_fallback"`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/custom_sam_peft/config/schema.py src/custom_sam_peft/eval/runner.py tests/unit/test_evaluator_schema.py tests/unit/test_eval_runner.py
git commit -m "feat(eval): EvalConfig.batch_size knob + run_eval resolves auto (#22)"
```

---

## Task 11: `PredictOptions.batch_size: int | "auto"` + flat predict loop

**Files:**
- Modify: `src/custom_sam_peft/predict/runner.py:47-71` (`PredictOptions` dataclass — add default `"auto"` on `batch_size`); rewrite the forward loop at `predict/runner.py:340-410`.
- Modify: `src/custom_sam_peft/cli/predict_cmd.py` (forward string `"auto"` sentinel; remove `_validate_positive_int` callback for `--batch-size` if it conflicts).
- Test: `tests/predict/test_runner_smoke.py` (extend); `tests/predict/test_cli_predict.py` (extend).

**Dependencies:** T6, T9, T12.

**Acceptance:** spec §13 AC 13.

- [ ] **Step 1: Write the failing tests**

(a) Extend `tests/predict/test_runner_smoke.py`:

```python
def test_predict_options_batch_size_default_auto() -> None:
    """PredictOptions.batch_size dataclass default is 'auto'."""
    from custom_sam_peft.predict.runner import PredictOptions

    # Use dataclass.fields to inspect the default (don't construct, since
    # PredictOptions has many required fields).
    import dataclasses
    fields = {f.name: f for f in dataclasses.fields(PredictOptions)}
    assert fields["batch_size"].default == "auto"


def test_run_predict_resolves_auto(monkeypatch, tmp_path) -> None:
    """run_predict resolves 'auto' once at entry via decide_eval_batch_size."""
    ...


def test_run_predict_flat_loop_iterates_image_chunks_x_groups(monkeypatch, tmp_path) -> None:
    """Flat (image_chunk, group) iteration; warmup is still single-image / single-class."""
    ...
```

(b) Extend `tests/predict/test_cli_predict.py`:

```python
def test_cli_predict_accepts_auto_batch_size(tmp_path) -> None:
    # Invoke `csp predict --batch-size auto ...` (dry-run) and assert it doesn't reject.
    ...
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/predict/test_runner_smoke.py tests/predict/test_cli_predict.py -v -k "auto or flat"`
Expected: FAIL.

- [ ] **Step 3: Write minimal implementation**

(a) `predict/runner.py:67` — change `batch_size: int` to:

```python
    batch_size: int | Literal["auto"] = "auto"
```

Important: `PredictOptions` is currently a frozen dataclass with NO defaults on any field (per its docstring). Adding a default to ONE field forces it to the end of the field order. Either:
- Re-order: move `batch_size` to the bottom of the dataclass field list (after `verbose`), or
- Make ALL fields use `field(default=…)` (more invasive).
Choose the first; document in the dataclass docstring that `batch_size` carries the only field-level default for API-surface reasons.

Update the import at the top of the file to include `Literal` (already imported).

(b) Rewrite the forward loop at `predict/runner.py:352-410`. Sketch:

```python
# Resolve "auto" once at entry.
if opts.batch_size == "auto":
    from custom_sam_peft.presets import decide_eval_batch_size

    bs, _, _ = decide_eval_batch_size(rcfg.image_size, classes_per_forward=16)
else:
    bs = int(opts.batch_size)

# Warmup is unchanged — single image, single class.

# Flat loop: (image_chunk, class_group).
from custom_sam_peft.models.sam3 import MULTIPLEX_CAP

def _chunked(seq, n):
    return [seq[i : i + n] for i in range(0, len(seq), n)]

all_predictions: list[dict[str, object]] = []
for chunk_paths in _chunked(image_paths, bs):
    # Open + transform each image in the chunk; stack to (B, 3, H, W).
    imgs = []
    metas = []  # parallel list of (image_id, orig_h, orig_w, stem)
    for img_path in chunk_paths:
        try:
            from PIL import Image as _PILImage
            pil_img = _PILImage.open(img_path).convert("RGB")
        except Exception as exc:
            logger.warning("Skipping unreadable image %s: %s", img_path, exc)
            continue
        orig_h, orig_w = pil_img.height, pil_img.width
        image_id = _int_image_id(img_path)
        id_to_path[image_id] = img_path.resolve()
        id_to_stem[image_id] = img_path.stem
        originals[image_id] = (orig_h, orig_w)
        img_np = np.array(pil_img)
        transformed = transforms(image=img_np, bboxes=[], class_labels=[])
        imgs.append(transformed["image"].to(rcfg.device, dtype=rcfg.dtype))
        metas.append((image_id, orig_h, orig_w))
    if not imgs:
        continue
    img_batch = torch.stack(imgs, dim=0)

    for group in _chunked(prompts, MULTIPLEX_CAP):
        K_g = len(group)
        prompts_g = [TextPrompts(classes=list(group)) for _ in metas]
        try:
            with torch.no_grad():
                outputs = model(img_batch, prompts_g, box_hints=None)
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                logger.error(
                    "OOM: consider --no-merge-adapter (QLoRA), --batch-size 1, or --device cpu"
                )
            raise

        for r in range(len(metas) * K_g):
            ii, kk = divmod(r, K_g)
            image_id, orig_h, orig_w = metas[ii]
            class_idx_one_based = prompts.index(group[kk]) + 1
            entries = queries_to_coco_results(
                _row_outputs(outputs, r),
                image_id=image_id,
                category_id=class_idx_one_based,
                original_hw=(orig_h, orig_w),
                mask_threshold=0.0,
            )
            entries = [e for e in entries if cast(float, e["score"]) >= opts.score_threshold]
            entries.sort(key=lambda e: cast(float, e["score"]), reverse=True)
            entries = entries[: opts.top_k]
            all_predictions.extend(entries)
    n_successful += len(metas)
    P.advance_inner(len(metas))
```

`_row_outputs` lives in `eval/evaluator.py`; import it: `from custom_sam_peft.eval.evaluator import _row_outputs`.

(c) `cli/predict_cmd.py`:
- Change the `--batch-size` option to accept the string `"auto"` or a positive int. The existing `_validate_positive_int` callback rejects non-ints; replace with a callback that accepts `"auto"` OR an int ≥ 1.
- Pass the resolved value through to `PredictOptions(... batch_size=batch_size_resolved)` as either `"auto"` (string) or the int.

Sketch:

```python
def _validate_batch_size(value: str) -> int | str:
    if value == "auto":
        return "auto"
    try:
        n = int(value)
    except ValueError as exc:
        raise typer.BadParameter("--batch-size must be 'auto' or a positive int") from exc
    if n < 1:
        raise typer.BadParameter(f"must be >= 1, got {n}")
    return n


batch_size: str = typer.Option(
    "auto",
    "--batch-size",
    callback=_validate_batch_size,
    help="Images per forward pass: 'auto' or a positive int.",
),
```

When forwarding to `PredictOptions`, keep the value as-is (the `int | Literal["auto"]` annotation accepts both).

- [ ] **Step 4: Run tests to verify they pass**

Run:
```
pytest tests/predict/ tests/integration/test_cli_run.py -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/custom_sam_peft/predict/runner.py src/custom_sam_peft/cli/predict_cmd.py tests/predict/test_runner_smoke.py tests/predict/test_cli_predict.py
git commit -m "feat(predict): batch_size='auto' + flat (image_chunk, group) loop (#22)"
```

---

## Task 12: `presets.decide_eval_batch_size` + `_predicted_bytes` mode param

**Files:**
- Modify: `src/custom_sam_peft/presets.py` (add `forward_only_factor` constant; extend `_predicted_bytes` with `mode: Literal["train", "eval"]`; add public `decide_eval_batch_size`).
- Test: `tests/unit/test_decide_eval_batch_size.py` (NEW); `tests/unit/test_presets.py` (extend for `_predicted_bytes` mode).

**Dependencies:** none (file-disjoint with T7-T11). **Parallelizable with:** T7, T8, T9, T10, T11.

**Acceptance:** spec §13 AC 12.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_decide_eval_batch_size.py`:

```python
"""decide_eval_batch_size: forward-only VRAM math; calibrated/analytic/CPU paths."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from unittest.mock import patch

import pytest
import torch


def test_decide_eval_batch_size_cpu_fallback(caplog, monkeypatch) -> None:
    caplog.set_level(logging.INFO)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    from custom_sam_peft.presets import decide_eval_batch_size

    bs, predicted_bytes, provenance = decide_eval_batch_size(1024)
    assert bs == 1
    assert predicted_bytes == 0
    assert provenance == "analytic"
    msgs = [r.message for r in caplog.records if "eval.batch_size=auto on CPU" in r.message]
    assert len(msgs) == 1


def test_decide_eval_batch_size_analytic_no_cache(monkeypatch, tmp_path) -> None:
    """Without a calibration cache, the analytic estimate runs at BASE_ACTIVATION_AT_1024."""
    # mock torch.cuda.is_available + get_device_properties + get_device_name
    ...
    bs, _, provenance = decide_eval_batch_size(1024)
    assert provenance == "analytic"
    assert bs >= 1


def test_decide_eval_batch_size_caps_search_at_64(monkeypatch) -> None:
    """Search space is B in [1, 64]; never returns B > 64 even on huge GPUs."""
    ...
    bs, _, _ = decide_eval_batch_size(1024)
    assert bs <= 64


def test_decide_eval_batch_size_uses_calibrated_cache(monkeypatch, tmp_path) -> None:
    """With a matching calibration cache, provenance='calibrated' and the cached
    activation_bytes_per_example is multiplied by forward_only_factor=0.25."""
    ...
    bs, _, provenance = decide_eval_batch_size(1024)
    assert provenance == "calibrated"


def test_predicted_bytes_eval_mode_excludes_optimizer_and_adapter(monkeypatch) -> None:
    """In mode='eval', _predicted_bytes skips _optimizer_bytes and _adapter_bytes,
    and scales activations by forward_only_factor."""
    from custom_sam_peft.presets import _predicted_bytes

    train_bytes = _predicted_bytes(
        "lora", r=4, batch=1, ckpt=False, image_size=1024, cache=None, mode="train"
    )
    eval_bytes = _predicted_bytes(
        "lora", r=4, batch=1, ckpt=False, image_size=1024, cache=None, mode="eval"
    )
    # eval drops optimizer state + adapter weights; activations scaled by 0.25.
    assert eval_bytes < train_bytes
```

Append to `tests/unit/test_presets.py`:

```python
def test_predicted_bytes_train_mode_unchanged() -> None:
    """Existing train-mode callers stay byte-identical after the mode param."""
    from custom_sam_peft.presets import _predicted_bytes

    # Default mode='train' — same value as the pre-change signature.
    n = _predicted_bytes("lora", r=4, batch=1, ckpt=False, image_size=1024, cache=None)
    # And the explicit mode='train' matches.
    assert n == _predicted_bytes(
        "lora", r=4, batch=1, ckpt=False, image_size=1024, cache=None, mode="train"
    )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_decide_eval_batch_size.py tests/unit/test_presets.py -v -k "decide_eval or predicted_bytes_eval or predicted_bytes_train"`
Expected: FAIL.

- [ ] **Step 3: Write minimal implementation**

In `src/custom_sam_peft/presets.py`:

(a) Add constant under the `=== CONSTANTS ===` block (after `BASE_ACTIVATION_AT_1024` at line 52):

```python
# Forward-only memory is roughly 1/4 of the train-step probe (train captures
# forward + backward + retained graph; eval captures only forward, no graph).
# Empirically calibrated; conservative — under-estimating eval memory pushes
# us toward smaller B, and the eval OOM ladder catches over-prediction.
# Spec §8.
forward_only_factor: float = 0.25
```

(b) Extend `_predicted_bytes` (line 168) with a `mode` parameter:

```python
def _predicted_bytes(
    method: str,
    r: int,
    batch: int,
    ckpt: bool,
    image_size: int,
    cache: dict[str, Any] | None,
    mode: Literal["train", "eval"] = "train",
) -> int:
    if mode == "train":
        return (
            _model_bytes(method)
            + _adapter_bytes(r)
            + _optimizer_bytes(r)
            + _activation_bytes(image_size, batch, ckpt, cache)
            + WORKSPACE_BYTES
        )
    # mode == "eval": no optimizer, no adapter bytes (the eval-side caller is
    # measuring base + forward activations); activations scaled by
    # forward_only_factor.
    activations = int(_activation_bytes(image_size, batch, ckpt, cache) * forward_only_factor)
    return _model_bytes(method) + activations + WORKSPACE_BYTES
```

(c) Add `decide_eval_batch_size` as a public sibling of `decide_preset`, near the bottom of the file:

```python
def decide_eval_batch_size(
    image_size: int,
    classes_per_forward: int = 16,
) -> tuple[int, int, Literal["calibrated", "analytic"]]:
    """Pick the largest forward-only batch size that fits within the eval VRAM budget.

    Returns (batch_size, predicted_bytes, provenance).

    K_eval is fixed at `classes_per_forward` (default 16); per-row activation
    scales with B only — image-encoder cost is shared across class prompts in
    a multiplex forward. Joint (B, K_eval) tuning is a §12 follow-up.

    On CPU: returns (1, 0, "analytic") and logs once.

    Spec: design §8.
    """
    if not isinstance(image_size, int) or image_size <= 0:
        raise ValueError("image_size must be a positive integer")
    if not torch.cuda.is_available():
        _LOG.info("eval.batch_size=auto on CPU -> falling back to 1")
        return 1, 0, "analytic"

    props = torch.cuda.get_device_properties(0)
    total = int(props.total_memory)
    gpu_name = torch.cuda.get_device_name(0)

    headroom = _headroom_bytes()
    budget = total - headroom

    cache, _ = _load_cache(image_size, gpu_name)
    provenance: Literal["calibrated", "analytic"] = (
        "calibrated" if cache is not None else "analytic"
    )

    best_bs = 1
    best_predicted = _predicted_bytes(
        "lora", r=4, batch=1, ckpt=False, image_size=image_size, cache=cache, mode="eval"
    )
    for batch in range(1, 65):  # B in [1, 64]
        pb = _predicted_bytes(
            "lora", r=4, batch=batch, ckpt=False, image_size=image_size, cache=cache, mode="eval"
        )
        if pb <= budget:
            best_bs = batch
            best_predicted = pb
    return best_bs, best_predicted, provenance
```

Notes:
- `r=4` is a sentinel for the cache lookup (matches train probing).
- `_load_cache`, `_headroom_bytes`, `_predicted_bytes` are reused verbatim except for the `mode="eval"` argument.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_decide_eval_batch_size.py tests/unit/test_presets.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/custom_sam_peft/presets.py tests/unit/test_decide_eval_batch_size.py tests/unit/test_presets.py
git commit -m "feat(presets): decide_eval_batch_size sibling + forward_only_factor (#22)"
```

---

## Task 13: Extend `tests/integration/test_load_sam31_real.py` with a K=8 multiplex assertion

**Files:**
- Modify: `tests/integration/test_load_sam31_real.py` (append one new test).

**Dependencies:** T4, T6.

**Acceptance:** spec §13 AC 16.

This is a GPU/integration test guarded by `requires_checkpoint` + `requires_compatible_gpu` (matching the file's existing module-level `pytestmark`). It costs one extra forward, no new fixtures.

- [ ] **Step 1: Append the new test**

Append to `tests/integration/test_load_sam31_real.py`:

```python
def test_load_sam31_multiplex_K8_forward() -> None:
    """Real K=8 multiplex forward emits pred_logits.shape[0] == B*8 and finite outputs.

    Per spec §13 AC 16. Confirms (B*K, ...) row layout end-to-end on real weights.
    """
    cfg = ModelConfig(device="cuda", gradient_checkpointing=False, dtype="bfloat16")
    wrapper = load_sam31(cfg)
    wrapper.eval()
    b = 2
    k = 8
    image = torch.zeros(b, 3, 1008, 1008, dtype=torch.bfloat16, device="cuda")
    classes = [f"class_{i}" for i in range(k)]
    prompts = [TextPrompts(classes=classes) for _ in range(b)]
    with torch.no_grad():
        outputs = wrapper(image, prompts)
    assert outputs["pred_logits"].shape[0] == b * k
    assert torch.isfinite(outputs["pred_logits"]).all()
    assert torch.isfinite(outputs["pred_boxes"]).all()
```

- [ ] **Step 2: Confirm the test is correctly gated**

The file's `pytestmark` (line 14-18) already applies `requires_checkpoint` + `requires_compatible_gpu` + `gpu_inspection`. New test inherits — no extra decorator needed.

- [ ] **Step 3: Commit**

```bash
git add tests/integration/test_load_sam31_real.py
git commit -m "test(integration): assert real K=8 multiplex forward shape (#22)"
```

(Verifying on a GPU happens in the orchestrator's CI run, not here.)

---

## Task 14: New `tests/gpu/test_multiplex_vram.py` — peak ≤ 4× predicted_bytes

**Files:**
- Create: `tests/gpu/test_multiplex_vram.py`.

**Dependencies:** T6, T12.

**Acceptance:** spec §13 AC 15.

- [ ] **Step 1: Create the test file**

```python
"""GPU regression: real K=16 multiplex forward at decide_eval_batch_size's
choice for image_size=1008 runs without OOM; peak ≤ 4× predicted_bytes.

The 4× ceiling is a conservative regression guard, not a tightness check —
see spec §9 for the calibration-constant note.
"""

from __future__ import annotations

import pytest
import torch

from custom_sam_peft.config.schema import ModelConfig
from custom_sam_peft.data.base import TextPrompts
from custom_sam_peft.models.sam3 import MULTIPLEX_CAP, load_sam31
from custom_sam_peft.presets import decide_eval_batch_size

pytestmark = [
    pytest.mark.requires_checkpoint,
    pytest.mark.requires_compatible_gpu,
    pytest.mark.gpu,
]


def test_real_K16_forward_at_chosen_B_within_predicted_envelope() -> None:
    image_size = 1008
    bs, predicted_bytes, _ = decide_eval_batch_size(
        image_size, classes_per_forward=MULTIPLEX_CAP
    )

    cfg = ModelConfig(device="cuda", gradient_checkpointing=False, dtype="bfloat16")
    wrapper = load_sam31(cfg)
    wrapper.eval()

    images = torch.zeros(bs, 3, image_size, image_size, dtype=torch.bfloat16, device="cuda")
    classes = [f"class_{i}" for i in range(MULTIPLEX_CAP)]
    prompts = [TextPrompts(classes=classes) for _ in range(bs)]

    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        outputs = wrapper(images, prompts)
    peak = torch.cuda.max_memory_allocated()

    assert outputs["pred_logits"].shape[0] == bs * MULTIPLEX_CAP
    assert peak <= 4 * predicted_bytes, (
        f"peak={peak} > 4 * predicted_bytes={4 * predicted_bytes}; "
        "either forward_only_factor underestimates eval memory or this GPU "
        "is over the empirical envelope. See spec §9 calibration note."
    )
```

- [ ] **Step 2: Commit**

```bash
git add tests/gpu/test_multiplex_vram.py
git commit -m "test(gpu): K=16 multiplex VRAM regression guard (#22)"
```

---

## Task 15: Create `scripts/bench_multiplex_throughput.py`

**Files:**
- Create: `scripts/bench_multiplex_throughput.py`.

**Dependencies:** T4-T12 (uses the full multiplex stack).

**Acceptance:** spec §9 benchmark; not gated in CI.

- [ ] **Step 1: Create the script**

The script loads the real SAM 3.1 checkpoint, builds a small COCO-80 mini-fixture (or accepts a dataset path argument), and times:
- K=1 path (per-class loop) — wall-clock micro-step over a 4-image batch with 80 classes.
- K=16 path (multiplex) — wall-clock micro-step over the same batch.
Reports steps/sec and per-image throughput; prints a 1-line summary the PR description can quote.

Skeleton:

```python
#!/usr/bin/env python
"""Benchmark K=1 vs K=16 multiplex training throughput. NOT in CI."""

from __future__ import annotations

import argparse
import time

import torch

# (imports: load_sam31, TextPrompts, total_loss, etc.)


def _bench(
    wrapper,
    images,
    class_names,
    classes_per_forward: int,
    n_steps: int = 5,
) -> float:
    """Returns mean wall-clock seconds per micro-step."""
    ...


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-size", type=int, default=1008)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--n-classes", type=int, default=80)
    parser.add_argument("--n-steps", type=int, default=5)
    args = parser.parse_args()

    # build synthetic data + load model
    # bench k=1, k=16
    # print summary
    print(
        f"K=1:  {sec_k1:.3f} s/step  ({args.batch / sec_k1:.2f} img/s)\n"
        f"K=16: {sec_k16:.3f} s/step ({args.batch / sec_k16:.2f} img/s)\n"
        f"speedup: {sec_k1 / sec_k16:.1f}x"
    )


if __name__ == "__main__":
    main()
```

The script is documentation/benchmark only; no tests required. Implementer should include a 5-line module docstring noting CI exclusion.

- [ ] **Step 2: Commit**

```bash
git add scripts/bench_multiplex_throughput.py
git commit -m "bench: K=1 vs K=16 multiplex throughput script (#22)"
```

---

## Task 16: Changelog entry

**Files:**
- Modify: `CHANGELOG.md` (add `## [0.8.0] — 2026-05-23` section above the existing `## [0.11.0]` section, or whichever date is correct on merge day).

**Dependencies:** all prior tasks. Run last.

**Acceptance:** spec §13 AC 17; spec §11 wording.

- [ ] **Step 1: Insert the new section**

Following the project's existing changelog style (Keep-a-Changelog; see the top of `CHANGELOG.md`), insert immediately under the `---` separator above the most recent version block:

```markdown
## [0.8.0] — YYYY-MM-DD

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
  <0.8.0 for K>1. Bit-equivalence holds at `train.multiplex.classes_per_forward=1`.

### Escape hatch

- Set `train.multiplex.classes_per_forward: 1` to recover the per-class
  iteration order within the same code path.
```

Date placeholder: the orchestrator fills it on PR merge day.

- [ ] **Step 2: Commit**

```bash
git add CHANGELOG.md
git commit -m "docs(changelog): v0.8.0 — multiplex forward (#22)"
```

---

## Final verification

Before opening the PR:

- [ ] Run the full CPU test suite: `pytest tests/unit tests/integration tests/predict -v -q`
- [ ] Lint + format: `ruff check src/ tests/ && ruff format --check src/ tests/`
- [ ] If you have GPU access, run: `pytest tests/integration/test_load_sam31_real.py tests/gpu/test_multiplex_vram.py -v` (else: leave CI/orchestrator to surface failures).
- [ ] Re-skim spec §13 ACs 1-17; every AC should map to at least one merged commit.
- [ ] Spec §11 changelog entries are all present in `CHANGELOG.md`.
