# Hardening Follow-up 91-100 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Spec:** [`docs/superpowers/specs/2026-05-22-hardening-followup-91-100-design.md`](../specs/2026-05-22-hardening-followup-91-100-design.md)
**Issues:** [#91](https://github.com/NguyenJus/custom-sam-peft/issues/91), [#92](https://github.com/NguyenJus/custom-sam-peft/issues/92), [#93](https://github.com/NguyenJus/custom-sam-peft/issues/93), [#94](https://github.com/NguyenJus/custom-sam-peft/issues/94), [#95](https://github.com/NguyenJus/custom-sam-peft/issues/95), [#96](https://github.com/NguyenJus/custom-sam-peft/issues/96), [#97](https://github.com/NguyenJus/custom-sam-peft/issues/97), [#99](https://github.com/NguyenJus/custom-sam-peft/issues/99), [#100](https://github.com/NguyenJus/custom-sam-peft/issues/100)
**Branch:** `hardening-followup-91-100`

**Goal:** Sweep PR resolving 9 of the 10 `hardening-followup` issues (skipping #98 which is split). Inline two demoted config groups (#92, #93), drive `make_peft_method` from registry (#100), delete dead `flatten_metrics_report` (#95), rename and relocate the notebook helper (#97, #99), and add a SAM-3 bump checklist README (#96). #91 and #94 close tracked-only.

**Architecture:** Six sequential commits ordered by risk-decay (code with heavy test coverage first → deletions → renames → moves → docs). Each commit is independently revertable. The rename in commit 4 must land before the move in commit 5 so `git mv` produces a pure path-change diff.

**Tech Stack:** Python 3.12, pytest (CPU-only, no `pytest.mark.gpu`), ruff, mypy, existing project stack. No new runtime deps.

---

## Drift notes (planner-verified against branch HEAD)

These are corrections to spec line numbers based on inspecting the worktree:

- **#92, §2.2:** Spec says `losses.py:170-171`; verified at lines 170-171 (matches). `MatcherWeights` fields at `config/_internal.py:28-29` (matches).
- **#92 test impact (drift):** Spec acceptance lists `tests/unit/test_matching.py` and "any test that poked `MatcherWeights.lambda_l1 / lambda_giou`". Sweep with `rg` returns two **additional** tests asserting the dataclass fields that the spec did not name:
  - `tests/unit/test_loss_config.py:18-19` — asserts `MatcherWeights().lambda_l1 == 0.0` and `lambda_giou == 0.0`.
  - `tests/unit/test_box_hint_schedule.py:66-67` — asserts the same on a re-instantiated `MatcherWeights()`.
  Both of these will fail once the fields are removed. Plan handles them in Task 1.
- **#93 test impact (drift):** Same `tests/unit/test_loss_config.py:38-39` asserts `LossConfig().focal_gamma == 2.0` and `focal_alpha == 0.25`. Will fail once removed. Plan handles in Task 1.
- **#93, §2.3:** Spec says focal call at `losses.py:189-190`; verified at lines 189-190.
- **#95, §2.5:** Spec says `flatten_metrics_report` at lines 40-55; verified at 40-55. `__all__` at line 17 (matches). `TYPE_CHECKING` import at lines 11-15 (matches).
- **#97, §2.7:** Spec says `resolve_hf_token` at `notebook_helpers.py:54`; verified at line 54.
- **#100, §2.9:** Spec says if/elif at `peft_adapters/__init__.py:129-146`; verified — `def make_peft_method` at line 129, body if/elif at 138-146. `LoraAdapter` at line 70, `QloraAdapter` at line 92 (matches spec). Existing `@register("peft", "lora")` at `peft_adapters/lora.py:88`, `@register("peft", "qlora")` at `peft_adapters/qlora.py:249` (matches).

---

## File Map

### Modified files

```
src/custom_sam_peft/config/_internal.py          # drop 4 fields (Task 1)
src/custom_sam_peft/models/losses.py             # add _FOCAL_* consts, inline matcher zeros (Task 1)
src/custom_sam_peft/peft_adapters/__init__.py    # registry-driven factory + decorators (Task 2)
src/custom_sam_peft/tracking/__init__.py         # remove flatten_metrics_report + TYPE_CHECKING import (Task 3)
src/custom_sam_peft/notebook_helpers.py          # rename function + module docstring (Task 4) → DELETED in Task 5
tests/unit/test_loss_config.py                   # drop assertions on removed fields (Task 1)
tests/unit/test_box_hint_schedule.py             # drop lambda_l1/giou assertions (Task 1)
tests/unit/test_losses.py                        # add focal regression-guard test (Task 1)
tests/unit/test_notebook_helpers.py              # update imports + rename test fns (Task 4) → MOVED in Task 5
notebooks/custom_sam_peft_train.ipynb            # update import + call site (Task 4) → import path (Task 5)
```

### New files

```
notebooks/_lib/__init__.py                       # empty (Task 5)
notebooks/_lib/notebook_helpers.py               # moved from src/ (Task 5)
tests/unit/notebooks/__init__.py                 # empty (Task 5)
tests/unit/notebooks/test_notebook_helpers.py    # moved from tests/unit/ (Task 5)
src/custom_sam_peft/models/_patches/README.md    # SAM-3 bump checklist (Task 6)
```

### Deleted files

```
src/custom_sam_peft/notebook_helpers.py          # (Task 5; moved to notebooks/_lib/)
tests/unit/test_tracking_flatten.py              # (Task 3)
tests/unit/test_notebook_helpers.py              # (Task 5; moved to tests/unit/notebooks/)
```

---

## Routing summary (for orchestrator)

| Task | Suggested impl model | Reviewer | Notes |
| --- | --- | --- | --- |
| 0 | n/a | n/a | Pre-flight; orchestrator runs the commands. |
| 1 | sonnet/high | sonnet/high | Code edit touching config + losses + 3 tests. |
| 2 | sonnet/high | **opus/xhigh** | Registry seam touches a discriminator the trainer also uses (#100 review uplift). |
| 3 | sonnet/high | sonnet/high | Pure deletion; low risk. |
| 4 | sonnet/high | sonnet/high | Rename across notebook JSON; verify imports. |
| 5 | sonnet/high | sonnet/high | File move via `git mv`; sys.path edit in notebook. |
| 6 | **haiku/high** | sonnet/high | Doc-only README. |
| 7 | n/a | n/a | Manual gate + PR open; orchestrator runs commands. |

**Parallelization:** All six code/doc commits are **sequential by design** (§3 risk-decay ordering, Task 5 depends on Task 4). No parallel candidates inside any single task.

---

## Task 0: Verify clean baseline

**Files:** none (commands only)

This task is a pre-flight gate. Orchestrator runs these directly; no subagent dispatch needed.

- [ ] **Step 0a: Confirm working tree clean**

```bash
git status
```

Expected: branch `hardening-followup-91-100`. Only spec + plan files (already committed). No other modifications. If dirty, halt.

- [ ] **Step 0b: Confirm baseline unit tests pass**

```bash
uv run pytest tests/unit -x -q --no-cov
```

Expected: all green. If anything is red, halt and surface — Task 1 cannot validate against a broken baseline.

- [ ] **Step 0c: Confirm ruff is clean**

```bash
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/
```

Expected: both clean.

---

## Task 1: Inline MatcherWeights and LossConfig demoted constants (#92, #93)

**Files:**
- Modify: `src/custom_sam_peft/config/_internal.py` (lines 28-29, 55-57)
- Modify: `src/custom_sam_peft/models/losses.py` (add module constants near top; update lines 170-171, 189-190)
- Modify: `tests/unit/test_loss_config.py` (remove assertions on dropped fields)
- Modify: `tests/unit/test_box_hint_schedule.py` (remove lambda_l1/giou assertions at 66-67)
- Modify: `tests/unit/test_losses.py` (add focal regression-guard test)

**Objective:** Drop `lambda_l1`, `lambda_giou` (MatcherWeights) and `focal_gamma`, `focal_alpha` (LossConfig) per §2.2 and §2.3. Inline the four values at their call sites (literals for matcher zeros, module constants for focal). Add a regression-guard test that the focal call site applies `gamma=2.0`/`alpha=0.25`.

- [ ] **Step 1a: Add the failing focal regression-guard test**

Append to `tests/unit/test_losses.py`:

```python
def test_total_loss_applies_focal_constants() -> None:
    """Regression-guard: focal_gamma/alpha were demoted from LossConfig to module
    constants. Verify the call site still passes gamma=2.0, alpha=0.25 to
    objectness_loss after the demotion (audit Section E, #93).
    """
    from unittest.mock import patch

    import torch

    from custom_sam_peft.config.schema import LossConfig
    from custom_sam_peft.models.losses import total_loss

    raw = {
        "pred_logits": torch.zeros(1, 4),
        "pred_boxes": torch.zeros(1, 4, 4),
        "pred_masks": torch.zeros(1, 4, 8, 8),
        "presence_logit_dec": torch.zeros(1),
    }
    targets: list[list[object]] = [[]]

    with patch(
        "custom_sam_peft.models.losses.objectness_loss",
        wraps=lambda obj_logits, matched_mask, gamma=2.0, alpha=0.25: torch.zeros(()),
    ) as spy:
        total_loss(raw, targets, LossConfig())

    assert spy.call_count == 1
    _args, kwargs = spy.call_args
    assert kwargs["gamma"] == 2.0, f"expected gamma=2.0, got {kwargs.get('gamma')!r}"
    assert kwargs["alpha"] == 0.25, f"expected alpha=0.25, got {kwargs.get('alpha')!r}"
```

- [ ] **Step 1b: Run the new test, confirm it currently passes (because focal_* still come from cfg)**

```bash
uv run pytest tests/unit/test_losses.py::test_total_loss_applies_focal_constants -v --no-cov
```

Expected: PASS (the cfg currently supplies 2.0 / 0.25 — this test passes before and after the demotion; it just guards against silent drift).

- [ ] **Step 1c: Drop `lambda_l1` and `lambda_giou` from MatcherWeights**

In `src/custom_sam_peft/config/_internal.py`, change the `MatcherWeights` dataclass (lines 17-30) to:

```python
@dataclass
class MatcherWeights:
    """Internal config — not user-set.

    Per-term cost weight for the Hungarian matcher.

    v0 defaults are mask-only; lambda_l1 / lambda_giou were demoted to inline
    literal 0.0 at the construction site in losses.py (audit Section E,
    YAGNI demote — no config sets them).
    """

    lambda_mask: float = 5.0
```

- [ ] **Step 1d: Drop `focal_gamma` and `focal_alpha` from LossConfig**

In the same file, edit the `LossConfig` dataclass (lines 33-57). Remove lines 55-57:

```python
    # focal_gamma and focal_alpha are demoted internal constants.
    focal_gamma: float = 2.0
    focal_alpha: float = 0.25
```

Update the surrounding docstring (lines 41-46) to note focal_* demotion:

```python
@dataclass
class LossConfig:
    """Internal config — not user-set.

    Loss-mix weights for SAM 3.1 training.

    Most fields are demoted internal constants per audit Section E. Only
    w_mask, w_obj, w_presence, and matcher_weights survive as advanced
    settings read by the training loop, but none are exposed to the YAML
    user schema — they are hardcoded defaults here. focal_gamma and
    focal_alpha were demoted to module-level constants in
    models/losses.py (#93).

    No `w_cls`: discrimination across classes comes from running one forward
    pass per class prompt. `w_presence` weights the image-level
    "any-instance-of-this-class-present?" supervision.
    """
```

- [ ] **Step 1e: Add module-level focal constants and update call sites in `losses.py`**

In `src/custom_sam_peft/models/losses.py`, add constants near the top of the file (after the imports block, before `_dice_loss` at line 18):

```python
# Demoted from LossConfig to module-level constants per audit Section E (#93).
# These are not user-tunable: the values were the same across every config
# example. If you need to tune focal weights, re-promote with a YAML schema
# change and a tracked-feature issue.
_FOCAL_GAMMA = 2.0
_FOCAL_ALPHA = 0.25
```

Replace the `HungarianMatcher` construction at lines 169-173 with:

```python
    matcher = HungarianMatcher(
        lambda_l1=0.0,
        lambda_giou=0.0,
        lambda_mask=cfg.matcher_weights.lambda_mask,
    )
```

Replace the focal call at lines 186-191 with:

```python
        "obj": objectness_loss(
            canonical.obj_logits,
            matched_mask,
            gamma=_FOCAL_GAMMA,
            alpha=_FOCAL_ALPHA,
        ),
```

- [ ] **Step 1f: Update `tests/unit/test_loss_config.py`**

Remove the dropped-field assertions. Replace lines 16-22:

```python
def test_matcher_weights_defaults() -> None:
    w = MatcherWeights()
    assert w.lambda_l1 == 0.0
    assert w.lambda_giou == 0.0
    assert w.lambda_mask == 5.0
    # No lambda_cls — open-vocab head has no per-class classification.
    assert not hasattr(w, "lambda_cls")
```

with:

```python
def test_matcher_weights_defaults() -> None:
    w = MatcherWeights()
    assert w.lambda_mask == 5.0
    # No lambda_cls — open-vocab head has no per-class classification.
    assert not hasattr(w, "lambda_cls")
    # lambda_l1 / lambda_giou were demoted to inline literals in losses.py (#92).
    assert not hasattr(w, "lambda_l1")
    assert not hasattr(w, "lambda_giou")
```

Replace `test_loss_config_defaults` at lines 31-42:

```python
def test_loss_config_defaults() -> None:
    cfg = LossConfig()
    assert cfg.w_mask == 1.0
    # w_box was demoted earlier (audit Section E).
    assert cfg.w_box == 0.0
    assert cfg.w_obj == 1.0
    assert cfg.w_presence == 1.0
    assert isinstance(cfg.matcher_weights, MatcherWeights)
    # focal_gamma / focal_alpha demoted to module-level constants in losses.py (#93).
    assert not hasattr(cfg, "focal_gamma")
    assert not hasattr(cfg, "focal_alpha")
    # No w_cls — open-vocab head has no per-class classification.
    assert not hasattr(cfg, "w_cls")
```

- [ ] **Step 1g: Update `tests/unit/test_box_hint_schedule.py`**

Replace lines 63-68:

```python
def test_matcher_weights_default_box_terms_are_zero() -> None:
    """v0 matcher is mask-only by default."""
    w = MatcherWeights()
    assert w.lambda_l1 == 0.0
    assert w.lambda_giou == 0.0
    assert w.lambda_mask == 5.0  # unchanged
```

with:

```python
def test_matcher_weights_default_is_mask_only() -> None:
    """v0 matcher is mask-only by default (lambda_l1/giou demoted to inline 0.0 in losses.py, #92)."""
    w = MatcherWeights()
    assert w.lambda_mask == 5.0
    assert not hasattr(w, "lambda_l1")
    assert not hasattr(w, "lambda_giou")
```

- [ ] **Step 1h: Run all affected tests**

```bash
uv run pytest tests/unit/test_loss_config.py tests/unit/test_box_hint_schedule.py tests/unit/test_losses.py tests/unit/test_matching.py -v --no-cov
```

Expected: all pass. `test_matching.py` is unchanged — it constructs `HungarianMatcher` directly with explicit values and is unaffected.

- [ ] **Step 1i: Run full unit suite as a regression gate**

```bash
uv run pytest tests/unit -x -q --no-cov
```

Expected: all green. If a test elsewhere accessed `cfg.focal_gamma` / `cfg.focal_alpha` / `MatcherWeights().lambda_l1|giou`, fix it (the drift sweep above caught the known cases; this gate catches anything missed).

- [ ] **Step 1j: ruff + format**

```bash
uv run ruff check src/custom_sam_peft/config/_internal.py src/custom_sam_peft/models/losses.py tests/unit/test_loss_config.py tests/unit/test_box_hint_schedule.py tests/unit/test_losses.py
uv run ruff format src/custom_sam_peft/config/_internal.py src/custom_sam_peft/models/losses.py tests/unit/test_loss_config.py tests/unit/test_box_hint_schedule.py tests/unit/test_losses.py
```

Expected: clean.

- [ ] **Step 1k: Confirm no stragglers**

```bash
rg -n "lambda_l1|lambda_giou" src/
rg -n "focal_gamma|focal_alpha" src/
```

Expected: `lambda_l1|lambda_giou` matches in `models/matching.py` (constructor + scoring) and the literal `0.0` lines in `losses.py` only. No matches at all for `focal_gamma|focal_alpha`.

- [ ] **Step 1l: Commit**

```bash
git add src/custom_sam_peft/config/_internal.py src/custom_sam_peft/models/losses.py tests/unit/test_loss_config.py tests/unit/test_box_hint_schedule.py tests/unit/test_losses.py
git commit -m "chore(config): inline MatcherWeights.lambda_l1/giou and LossConfig.focal_* constants (#92, #93)"
```

**Acceptance** (lifted from §2.2 + §2.3):
- `MatcherWeights` has exactly one field: `lambda_mask: float = 5.0`.
- `losses.py` matcher construction passes literal `0.0` for `lambda_l1` and `lambda_giou`.
- `_FOCAL_GAMMA = 2.0` and `_FOCAL_ALPHA = 0.25` exist as module-level constants in `losses.py`, each with the demoted-from-config comment.
- `LossConfig` no longer has `focal_gamma` or `focal_alpha` attributes.
- `rg -n "lambda_l1|lambda_giou" src/` returns only `models/matching.py` (constructor + scoring) and the `0.0` literals in `losses.py`.
- `tests/unit/test_losses.py::test_total_loss_applies_focal_constants` passes — guards against silent constant drift.
- `tests/unit/test_loss_config.py` and `tests/unit/test_box_hint_schedule.py` no longer assert on the dropped fields; instead assert `not hasattr`.

---

## Task 2: Drive `make_peft_method` from registry (#100)

**Files:**
- Modify: `src/custom_sam_peft/peft_adapters/__init__.py` (imports, decorators on lines 70 + 92, factory body at 129-146, module docstring at 1-14)

**Objective:** Replace the if/elif branches in `make_peft_method` with a `lookup("peft_method", method)` call. Decorate `LoraAdapter` and `QloraAdapter` with `@register("peft_method", ...)` at their definition sites. Re-raise `RegistryError` as `ValueError` to preserve the existing error contract that `test_make_peft_method_unknown_raises` asserts.

**Critical:** Use the **new** `"peft_method"` namespace, not the existing `"peft"` namespace. The existing `"peft"` bucket already holds `apply_lora` / `apply_qlora` callables (signature `(wrapper, cfg) → Sam3Wrapper`); `make_peft_method` returns a `PEFTMethod` protocol instance constructed with `()`. Two different callables with different signatures and return types — separate registry keys (spec §2.9 "Why a new namespace").

- [ ] **Step 2a: Confirm the existing tests pass before edit**

```bash
uv run pytest tests/unit/test_peft_method_protocol.py -v --no-cov
```

Expected: all green, including `test_make_peft_method_lora`, `test_make_peft_method_qlora`, `test_make_peft_method_unknown_raises` at lines 130-144.

- [ ] **Step 2b: Update imports in `src/custom_sam_peft/peft_adapters/__init__.py`**

Replace lines 16-21 (current imports):

```python
from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from custom_sam_peft.errors import CheckpointError
```

with:

```python
from __future__ import annotations

from pathlib import Path
from typing import Protocol, cast, runtime_checkable

from custom_sam_peft._registry import RegistryError, lookup, register
from custom_sam_peft.errors import CheckpointError
```

- [ ] **Step 2c: Update the module docstring (lines 1-14)**

Replace lines 1-14 with:

```python
"""PEFT adapter package.

Documented seam: trainers, evaluators, and checkpoint code interact with
PEFT adapters through the ``PEFTMethod`` protocol below. They must not
branch on ``cfg.peft.method`` strings.

Registered factories:
  ``lookup("peft", "lora")``         → ``apply_lora``         (wrapper, cfg) → Sam3Wrapper
  ``lookup("peft", "qlora")``        → ``apply_qlora``        (wrapper, cfg) → Sam3Wrapper
  ``lookup("peft_method", "lora")``  → ``LoraAdapter``        () → PEFTMethod
  ``lookup("peft_method", "qlora")`` → ``QloraAdapter``       () → PEFTMethod

For method-dispatch decisions (optimizer, autocast, checkpoint detection)
call the appropriate ``LoraAdapter`` or ``QloraAdapter`` instance methods
instead of testing ``cfg.peft.method``.
"""
```

- [ ] **Step 2d: Decorate `LoraAdapter` at line 70**

Replace `class LoraAdapter:` (line 70) with:

```python
@register("peft_method", "lora")
class LoraAdapter:
```

- [ ] **Step 2e: Decorate `QloraAdapter` at line 92**

Replace `class QloraAdapter:` (line 92) with:

```python
@register("peft_method", "qlora")
class QloraAdapter:
```

- [ ] **Step 2f: Replace the `make_peft_method` body at lines 129-146**

Replace the entire function:

```python
def make_peft_method(method: str) -> PEFTMethod:
    """Return the PEFTMethod instance for the given peft.method string.

    This is the single factory that maps the string from cfg.peft.method to
    a protocol instance. Call it once during run setup (e.g. in Trainer.__init__
    or run_eval) and pass the instance through rather than passing cfg.peft.method.

    Resolves via the @register("peft_method", ...) registry — adding a new
    adapter requires only a @register decorator on the new class, no edits here.

    Raises ValueError for unknown method strings.
    """
    try:
        adapter_cls = lookup("peft_method", method)
    except RegistryError as exc:
        raise ValueError(
            f"Unknown peft.method {method!r}; expected 'lora' or 'qlora'. "
            "Register additional adapters via @register('peft_method', '<name>')."
        ) from exc
    return cast(PEFTMethod, adapter_cls())
```

- [ ] **Step 2g: Run the three factory tests**

```bash
uv run pytest tests/unit/test_peft_method_protocol.py::test_make_peft_method_lora tests/unit/test_peft_method_protocol.py::test_make_peft_method_qlora tests/unit/test_peft_method_protocol.py::test_make_peft_method_unknown_raises -v --no-cov
```

Expected: all three pass. The `unknown_raises` test asserts `pytest.raises(ValueError, match=r"Unknown peft\.method")` — the new `raise ValueError ... from exc` preserves the contract.

- [ ] **Step 2h: Run the full `test_peft_method_protocol.py` plus bootstrap and registry tests**

```bash
uv run pytest tests/unit/test_peft_method_protocol.py tests/unit/test_bootstrap.py tests/unit/test_registry.py -v --no-cov
```

Expected: all green. Importing `custom_sam_peft.peft_adapters` fires the new `@register("peft_method", ...)` decorators because the adapter classes live directly in `peft_adapters/__init__.py`. `_bootstrap.py` already imports `peft_adapters.lora` and `peft_adapters.qlora`, which transitively imports the package and fires the decorators.

- [ ] **Step 2i: Run full unit suite as a regression gate**

```bash
uv run pytest tests/unit -x -q --no-cov
```

Expected: all green. In particular, the existing `lookup("peft", "lora")` callers at `train/runner.py:115` are untouched and still receive `apply_lora`.

- [ ] **Step 2j: ruff + format**

```bash
uv run ruff check src/custom_sam_peft/peft_adapters/__init__.py
uv run ruff format src/custom_sam_peft/peft_adapters/__init__.py
```

Expected: clean.

- [ ] **Step 2k: Confirm the factory has no `if method ==` literals**

```bash
rg -n 'if method ==' src/custom_sam_peft/peft_adapters/__init__.py
```

Expected: no matches inside `make_peft_method`. (Note: `method_pretty_name` at line 114-126 retains its if/elif — that is **out of scope**; only `make_peft_method` is migrated per §2.9 + §6 "No registry refactor beyond `make_peft_method`".)

- [ ] **Step 2l: Commit**

```bash
git add src/custom_sam_peft/peft_adapters/__init__.py
git commit -m "chore(peft): drive make_peft_method from registry (#100)"
```

**Acceptance** (lifted from §2.9):
- `make_peft_method` body uses `lookup("peft_method", method)`; no `if method ==` literals inside it.
- `LoraAdapter` and `QloraAdapter` carry `@register("peft_method", "lora")` / `@register("peft_method", "qlora")` decorators.
- `make_peft_method("lora")` returns a `LoraAdapter` instance (existing test passes unchanged).
- `make_peft_method("qlora")` returns a `QloraAdapter` instance (existing test passes unchanged).
- `make_peft_method("unknown")` raises `ValueError` matching `r"Unknown peft\.method"` (existing test passes unchanged).
- Existing `lookup("peft", ...)` callers at `train/runner.py:115` are untouched and still receive `apply_lora` / `apply_qlora`.
- No new tests required.

---

## Task 3: Delete `flatten_metrics_report` (#95)

**Files:**
- Modify: `src/custom_sam_peft/tracking/__init__.py` (remove function at 40-55, `__all__` entry at 17, TYPE_CHECKING import at 11-15)
- Delete: `tests/unit/test_tracking_flatten.py`

**Objective:** Delete the dead `flatten_metrics_report` helper and its test file per §2.5. Rationale: zero `src/` callers; the audit (Section J5) flagged it as dead; the "wire it in" alternative is YAGNI.

- [ ] **Step 3a: Delete the test file**

```bash
git rm tests/unit/test_tracking_flatten.py
```

- [ ] **Step 3b: Edit `src/custom_sam_peft/tracking/__init__.py`**

Replace the full current file with:

```python
"""Tracking subsystem — Tracker Protocol, build_tracker factory."""

from __future__ import annotations

from typing import cast

from custom_sam_peft._registry import lookup
from custom_sam_peft.config.schema import TrainConfig
from custom_sam_peft.tracking.base import Tracker

__all__ = ["Tracker", "build_tracker"]


def build_tracker(cfg: TrainConfig) -> Tracker:
    """Resolve cfg.tracking.backend to a concrete Tracker.

    Imports the chosen backend module lazily so missing optional extras only
    surface when that backend is actually requested. The @register decorator
    in each backend module wires the factory into _registry on first import.
    """
    backend = cfg.tracking.backend  # Literal["tensorboard", "wandb", "none"]
    if backend == "tensorboard":
        from custom_sam_peft.tracking import tensorboard as _tb  # noqa: F401
    elif backend == "wandb":
        from custom_sam_peft.tracking import wandb as _wb  # noqa: F401
    elif backend == "none":
        from custom_sam_peft.tracking import noop as _noop  # noqa: F401
    else:  # pragma: no cover — pydantic Literal rejects this at config-load
        raise ValueError(f"unknown tracking.backend: {backend!r}")
    factory = lookup("tracker", backend)
    return cast(Tracker, factory(cfg))
```

Changes vs current file:
- Removed the `TYPE_CHECKING` block (lines 11-15) — `MetricsReport` was only used by the deleted helper.
- Removed `flatten_metrics_report` from `__all__` (line 17).
- Removed the `flatten_metrics_report` function (lines 40-55).
- Removed `TYPE_CHECKING` from the typing import; kept `cast`.

- [ ] **Step 3c: Confirm no orphan references**

```bash
rg -n "flatten_metrics_report" src/ tests/
```

Expected: zero matches.

- [ ] **Step 3d: Run all tracking tests**

```bash
uv run pytest tests/unit/test_tracking_build.py tests/unit/test_tracking_noop.py tests/unit/test_tracking_protocol.py tests/unit/test_tracking_tensorboard.py tests/unit/test_tracking_wandb.py -v --no-cov
```

Expected: all green. `test_tracking_flatten.py` is gone (deleted) and was the sole consumer of `flatten_metrics_report`.

- [ ] **Step 3e: Run full unit suite as a regression gate**

```bash
uv run pytest tests/unit -x -q --no-cov
```

Expected: all green.

- [ ] **Step 3f: ruff + format**

```bash
uv run ruff check src/custom_sam_peft/tracking/__init__.py
uv run ruff format src/custom_sam_peft/tracking/__init__.py
```

Expected: clean.

- [ ] **Step 3g: Commit**

```bash
git add src/custom_sam_peft/tracking/__init__.py tests/unit/test_tracking_flatten.py
git commit -m "chore(tracking): delete unused flatten_metrics_report (#95)"
```

**Acceptance** (lifted from §2.5):
- `rg -n "flatten_metrics_report" src/ tests/` returns no matches.
- `tracking/__init__.py:__all__` is `["Tracker", "build_tracker"]`.
- `tests/unit/test_tracking_flatten.py` does not exist.
- `uv run pytest tests/unit/test_tracking_*` is green.

---

## Task 4: Rename `resolve_hf_token` in `notebook_helpers.py` + document duplication (#97)

**Files:**
- Modify: `src/custom_sam_peft/notebook_helpers.py` (line 54 function name; add module docstring contrast paragraph)
- Modify: `tests/unit/test_notebook_helpers.py` (line 15 import; test function names if desired)
- Modify: `notebooks/custom_sam_peft_train.ipynb` (cell line 34 import; cell line 42 call site)

**Objective:** Rename `resolve_hf_token` → `resolve_hf_token_for_notebook` in `notebook_helpers.py` per §2.7. Add a module docstring paragraph explicitly contrasting the two `resolve_hf_token*` functions. Update the two callers (test + notebook).

**Critical:** This task does the **rename only**. The file is still at `src/custom_sam_peft/notebook_helpers.py` after this commit; the move to `notebooks/_lib/` happens in Task 5. Sequencing rename → move keeps the move commit as a pure path diff (clean `git mv` rename detection).

- [ ] **Step 4a: Edit `src/custom_sam_peft/notebook_helpers.py`**

Replace the module docstring (lines 1-5):

```python
"""Helpers used by `notebooks/custom_sam_peft_train.ipynb` for env detection,
local-checkpoint short-circuit, and HF-token resolution.

CLI never imports this module. Tests and the notebook do.

Note: ``utils/huggingface.py::resolve_hf_token`` is the silent best-effort
resolver used by ``download_model`` — it returns the token or ``None`` and
never raises. ``notebook_helpers.py::resolve_hf_token_for_notebook`` (below)
is an env-aware resolver for notebook contexts: it short-circuits when a
local checkpoint is present and raises ``RuntimeError`` with Colab- or
RunPod-specific instructions when the token is missing. The two are
deliberately not merged; their failure semantics differ.
"""
```

Rename the function at line 54:

```python
def resolve_hf_token_for_notebook(env: Env, local_present: bool) -> str | None:
```

No body changes — same logic, same docstring body.

- [ ] **Step 4b: Update `tests/unit/test_notebook_helpers.py`**

Replace the import block (lines 12-16):

```python
from custom_sam_peft.notebook_helpers import (
    check_local_checkpoint,
    detect_env,
    resolve_hf_token_for_notebook,
)
```

Replace every occurrence of `resolve_hf_token(` with `resolve_hf_token_for_notebook(` in the file (the eight test functions starting at line 65). The cleanest approach is `sed -i 's/resolve_hf_token(/resolve_hf_token_for_notebook(/g' tests/unit/test_notebook_helpers.py` then sanity-check with `git diff`. Test function names may also be renamed (`test_resolve_hf_token_*` → `test_resolve_hf_token_for_notebook_*`) but this is optional — implementer choice per spec §2.7.

- [ ] **Step 4c: Update the notebook JSON**

`notebooks/custom_sam_peft_train.ipynb` is a JSON file with cell source as a list of string lines. Use Edit / Read on the file directly:
- Around line 31-37: update the `from custom_sam_peft.notebook_helpers import ( ... resolve_hf_token, ... )` import. Change `resolve_hf_token` → `resolve_hf_token_for_notebook` (preserving the trailing comma and `\n`).
- Around line 42: update the call site `token = resolve_hf_token(env, local_present)` → `token = resolve_hf_token_for_notebook(env, local_present)`.

After editing, verify the notebook still parses as valid JSON:

```bash
python -c "import json; json.load(open('notebooks/custom_sam_peft_train.ipynb'))"
```

Expected: parses without error (exits 0).

- [ ] **Step 4d: Confirm two distinct function names exist**

```bash
rg -n "def resolve_hf_token" src/custom_sam_peft/
```

Expected: exactly two lines:
- `src/custom_sam_peft/utils/huggingface.py:...: def resolve_hf_token(...)`
- `src/custom_sam_peft/notebook_helpers.py:54: def resolve_hf_token_for_notebook(...)`

- [ ] **Step 4e: Run the notebook-helper tests**

```bash
uv run pytest tests/unit/test_notebook_helpers.py -v --no-cov
```

Expected: all pass with the renamed import.

- [ ] **Step 4f: Run full unit suite as a regression gate**

```bash
uv run pytest tests/unit -x -q --no-cov
```

Expected: all green.

- [ ] **Step 4g: ruff + format**

```bash
uv run ruff check src/custom_sam_peft/notebook_helpers.py tests/unit/test_notebook_helpers.py
uv run ruff format src/custom_sam_peft/notebook_helpers.py tests/unit/test_notebook_helpers.py
```

Expected: clean.

- [ ] **Step 4h: Commit**

```bash
git add src/custom_sam_peft/notebook_helpers.py tests/unit/test_notebook_helpers.py notebooks/custom_sam_peft_train.ipynb
git commit -m "chore(hf): rename resolve_hf_token in notebook_helpers, clarify duplication (#97)"
```

**Acceptance** (lifted from §2.7):
- `rg -n "def resolve_hf_token" src/custom_sam_peft/` returns two distinct names: `resolve_hf_token` (in `utils/huggingface.py`) and `resolve_hf_token_for_notebook` (in `notebook_helpers.py`).
- `tests/unit/test_notebook_helpers.py` imports the renamed function and tests pass.
- The notebook JSON's import and call sites use `resolve_hf_token_for_notebook`.
- `notebook_helpers.py` module docstring contains the contrast paragraph.
- No semantics change in either function body.

---

## Task 5: Move `notebook_helpers.py` → `notebooks/_lib/` (#99)

**Files:**
- Delete: `src/custom_sam_peft/notebook_helpers.py`
- Create: `notebooks/_lib/__init__.py` (empty)
- Create: `notebooks/_lib/notebook_helpers.py` (moved via `git mv`)
- Delete: `tests/unit/test_notebook_helpers.py`
- Create: `tests/unit/notebooks/__init__.py` (empty)
- Create: `tests/unit/notebooks/test_notebook_helpers.py` (moved via `git mv`)
- Modify: `notebooks/custom_sam_peft_train.ipynb` (import path: add `sys.path` prepend + change import)

**Objective:** Move `notebook_helpers.py` out of the installed package per §2.8 (it has zero `src/` callers; CLI never imports it). Co-locate it with its sole consumer under `notebooks/_lib/`. **Do NOT move `presets.py`** — its Section J premise is stale (now has 3 `src/` callers: `cli/run_cmd.py:30`, `cli/calibrate_cmd.py:24,33`, `runs/bundle.py:36`).

**Critical:** Use `git mv` for both moves so git's rename detection produces a clean diff. Because Task 4 renamed the function body, the move commit shows only path changes.

- [ ] **Step 5a: Move the source file via `git mv`**

```bash
mkdir -p notebooks/_lib
touch notebooks/_lib/__init__.py
git mv src/custom_sam_peft/notebook_helpers.py notebooks/_lib/notebook_helpers.py
```

- [ ] **Step 5b: Move the test file via `git mv`**

```bash
mkdir -p tests/unit/notebooks
touch tests/unit/notebooks/__init__.py
git mv tests/unit/test_notebook_helpers.py tests/unit/notebooks/test_notebook_helpers.py
```

- [ ] **Step 5c: Update the test import**

In `tests/unit/notebooks/test_notebook_helpers.py`, change the import block (was line 12-16 after Task 4):

```python
from custom_sam_peft.notebook_helpers import (
    check_local_checkpoint,
    detect_env,
    resolve_hf_token_for_notebook,
)
```

to:

```python
# The notebook_helpers module is co-located with the notebook under
# notebooks/_lib/ (#99). Prepend the repo's notebooks/ dir to sys.path
# so this test can import it the same way the notebook does.
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT / "notebooks"))

from _lib.notebook_helpers import (  # noqa: E402
    check_local_checkpoint,
    detect_env,
    resolve_hf_token_for_notebook,
)
```

The `parents[3]` walk: `test_notebook_helpers.py` → `notebooks/` → `unit/` → `tests/` → repo root. Verify by printing if needed.

- [ ] **Step 5d: Update the notebook JSON imports**

`notebooks/custom_sam_peft_train.ipynb`: the notebook already does `sys.path` manipulation in its setup cells (per spec §2.8). The cleanest seam is to prepend the notebook's parent `notebooks/` to `sys.path` and import from `_lib.notebook_helpers`.

In the cell currently containing `from custom_sam_peft.notebook_helpers import (...)` (around JSON line 31), do **both**:

1. **Above the import**, add a `sys.path` prepend line. The notebook already imports `sys` elsewhere in the setup; if not in this cell, add it. Insert a line like:

```python
import sys, pathlib; sys.path.insert(0, str(pathlib.Path.cwd() / "notebooks"))
```

(Notebook cells typically run from the repo root in Colab; `cwd() / "notebooks"` is the correct prefix. If the notebook's working directory convention differs, adapt to match the surrounding setup cells.)

2. **Change the import** from:

```python
from custom_sam_peft.notebook_helpers import (
    check_local_checkpoint,
    detect_env,
    resolve_hf_token_for_notebook,
)
```

to:

```python
from _lib.notebook_helpers import (
    check_local_checkpoint,
    detect_env,
    resolve_hf_token_for_notebook,
)
```

The call site (`token = resolve_hf_token_for_notebook(env, local_present)`) is unchanged.

- [ ] **Step 5e: Smoke-verify the notebook still parses**

```bash
python -c "import json; json.load(open('notebooks/custom_sam_peft_train.ipynb'))"
```

Expected: exits 0.

- [ ] **Step 5f: Confirm the source file is gone and the new location works**

```bash
test ! -f src/custom_sam_peft/notebook_helpers.py && echo "OK: deleted"
test -f notebooks/_lib/notebook_helpers.py && echo "OK: created"
test -f notebooks/_lib/__init__.py && echo "OK: __init__"
test -f tests/unit/notebooks/test_notebook_helpers.py && echo "OK: test moved"
test -f tests/unit/notebooks/__init__.py && echo "OK: test __init__"
```

Expected: five `OK` lines.

- [ ] **Step 5g: Verify `presets.py` was NOT moved**

```bash
test -f src/custom_sam_peft/presets.py && echo "OK: presets.py kept"
rg -n "from custom_sam_peft.presets" src/
```

Expected: `OK: presets.py kept`, and three `src/` callers reported: `cli/run_cmd.py:30`, `cli/calibrate_cmd.py:24` (and `33`), `runs/bundle.py:36`.

- [ ] **Step 5h: Run the moved test**

```bash
uv run pytest tests/unit/notebooks/ -v --no-cov
```

Expected: all eight `resolve_hf_token_for_notebook*` + three `detect_env*` + three `check_local_checkpoint*` tests pass. Pytest's rootdir-relative collection picks up the new subdirectory because `tests/unit/notebooks/__init__.py` exists.

- [ ] **Step 5i: Confirm no stragglers reference the old path**

```bash
rg -n "custom_sam_peft.notebook_helpers" src/ tests/ notebooks/
```

Expected: zero matches. (The notebook now imports from `_lib.notebook_helpers`; the test imports the same way.)

- [ ] **Step 5j: Run full unit suite**

```bash
uv run pytest tests/unit -x -q --no-cov
```

Expected: all green.

- [ ] **Step 5k: ruff + format**

```bash
uv run ruff check notebooks/_lib/notebook_helpers.py tests/unit/notebooks/test_notebook_helpers.py
uv run ruff format notebooks/_lib/notebook_helpers.py tests/unit/notebooks/test_notebook_helpers.py
```

Expected: clean.

- [ ] **Step 5l: Commit**

```bash
git add src/custom_sam_peft/notebook_helpers.py \
        notebooks/_lib/__init__.py notebooks/_lib/notebook_helpers.py \
        tests/unit/test_notebook_helpers.py \
        tests/unit/notebooks/__init__.py tests/unit/notebooks/test_notebook_helpers.py \
        notebooks/custom_sam_peft_train.ipynb
git commit -m "chore(notebooks): move notebook_helpers.py to notebooks/_lib (#99)"
```

**Acceptance** (lifted from §2.8):
- `src/custom_sam_peft/notebook_helpers.py` does not exist.
- `notebooks/_lib/notebook_helpers.py` exists with identical body (modulo the §2.7 rename).
- `notebooks/_lib/__init__.py` exists.
- `tests/unit/notebooks/test_notebook_helpers.py` exists and `uv run pytest tests/unit/notebooks/` is green.
- `src/custom_sam_peft/presets.py` still exists; its three `src/` callers still resolve.
- The Colab notebook parses as valid JSON; the import cell uses `from _lib.notebook_helpers import ...`.

---

## Task 6: Add `_patches/README.md` SAM-3 bump checklist (#96)

**Files:**
- Create: `src/custom_sam_peft/models/_patches/README.md`

**Objective:** Doc-only addition per §2.6. No Python changes. List each of the 8 patch files with a one-liner and provide a 5-item "When SAM-3 bumps" checklist.

**Model/effort note:** This is non-code (docs). Use **haiku/high** for implementation; sonnet/high reviewer.

- [ ] **Step 6a: Verify the patch files exist as enumerated**

```bash
ls src/custom_sam_peft/models/_patches/
```

Expected output (8 patch files + `__init__.py`):

```
__init__.py
addmm_act_grad_safe.py
encode_prompt_dtype.py
forward_grounding_skip_matching.py
mha_input_dtype.py
module_input_dtype.py
pos_enc_dtype.py
roi_align_dtype.py
text_pool_dtype.py
```

- [ ] **Step 6b: Write `src/custom_sam_peft/models/_patches/README.md`**

```markdown
# SAM-3 Patches

This directory holds in-process monkey-patches we apply to the upstream SAM-3
codebase. Each patch is narrow, targeted, and exists because a real failure
mode (dtype mismatch, autograd shape error, wrong dispatch path) surfaced in
training or eval against the pinned upstream checkpoint.

`models/sam3.py::load_sam31` wires each patch into the wrapper's
`_apply_patches` step. The patches are import-side-effect-free until that
function calls them.

## Patch index

| File | What it patches |
| --- | --- |
| `addmm_act_grad_safe.py` | Guards `addmm` autograd path against an upstream activation-grad shape mismatch. |
| `encode_prompt_dtype.py` | Forces prompt-encoder activations to the wrapper's compute dtype to prevent fp16/bf16 cast mismatches. |
| `forward_grounding_skip_matching.py` | Skips the upstream grounding matcher path that we replace with our own Hungarian matcher. |
| `mha_input_dtype.py` | Casts MHA inputs to a consistent dtype across Q/K/V projections. |
| `module_input_dtype.py` | Generic input-dtype harmonizer for modules that drop kwargs through. |
| `pos_enc_dtype.py` | Aligns positional-encoding dtype with the surrounding activation dtype. |
| `roi_align_dtype.py` | Forces ROI-Align inputs to fp32 (kernel only supports fp32; see `2026-05-22-fix-roi-align-dtype-mismatch.md`). |
| `text_pool_dtype.py` | Aligns text-pool projection dtype with the text-encoder output. |

## When SAM-3 bumps

Whenever the pinned SAM-3 checkpoint or vendored source version changes,
walk through this checklist before merging the bump:

1. Re-run `tests/gpu/` against the new SAM-3 checkpoint.
2. For each patch in this directory: open the corresponding upstream source
   file (`vendor/sam3/...` or the pinned pip dep), confirm the line numbers
   and function signatures the patch targets still exist.
3. If a target moved: update the patch's line / signature reference. If a
   target was removed: open an issue tagged `sam3-bump` to delete the patch.
4. Confirm `models/sam3.py::load_sam31` still wires each patch into the
   wrapper's `_apply_patches` step.
5. Update the SAM-3 checkpoint SHA pin in
   `src/custom_sam_peft/presets.py::_current_sam3_checkpoint_sha` (the
   analytic VRAM cache uses this to invalidate prior calibrations).
```

- [ ] **Step 6c: Confirm the README exists and contains the required sections**

```bash
test -f src/custom_sam_peft/models/_patches/README.md && echo OK
rg -c "^## Patch index|^## When SAM-3 bumps" src/custom_sam_peft/models/_patches/README.md
```

Expected: `OK` then `2` (both headers present).

- [ ] **Step 6d: Confirm no Python files were touched**

```bash
git status --short -- 'src/custom_sam_peft/**/*.py'
```

Expected: no entries — this commit is doc-only.

- [ ] **Step 6e: Commit**

```bash
git add src/custom_sam_peft/models/_patches/README.md
git commit -m "docs(models): add _patches/README.md SAM-3 version-bump checklist (#96)"
```

**Acceptance** (lifted from §2.6):
- `src/custom_sam_peft/models/_patches/README.md` exists.
- Lists all 8 patch files with one-liners.
- Includes the 5-item bump checklist.
- No source-code changes in the commit.

---

## Task 7: Manual gate + open PR

**Files:** none (commands + PR draft)

This task is the manual gate from §4 followed by opening the PR per §5. Orchestrator runs these commands directly; no subagent dispatch needed.

- [ ] **Step 7a: Run the full non-GPU test suite**

```bash
uv run pytest -m "not gpu" --no-cov
```

Expected: all green. If anything is red, halt and surface — do NOT open the PR.

- [ ] **Step 7b: Ruff lint and format checks**

```bash
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/
```

Expected: both clean.

- [ ] **Step 7c: Confirm no orphan references to the renamed / deleted symbols**

```bash
rg -n "flatten_metrics_report" src/ tests/
rg -n "custom_sam_peft.notebook_helpers" src/ tests/ notebooks/
```

Expected: both return no matches. (The notebook now imports from `_lib.notebook_helpers`.)

- [ ] **Step 7d: Smoke-verify the notebook parses**

```bash
python -c "import json; json.load(open('notebooks/custom_sam_peft_train.ipynb'))"
```

Expected: exits 0.

- [ ] **Step 7e: Push the branch**

```bash
git push -u origin hardening-followup-91-100
```

- [ ] **Step 7f: Open the PR**

Use this PR description (verbatim footer per §5):

```bash
gh pr create \
  --assignee @me \
  --label hardening-followup \
  --title "chore: hardening follow-ups #91–#100 (except #98)" \
  --body "$(cat <<'EOF'
Sweep PR resolving 9 of the 10 `hardening-followup` issues opened off the v0.7.0 hardening audit (Section J). #98 (QLoRA checkpoint disk-load) is split out for a separate spec/PR — follow-up to come.

**Spec:** [`docs/superpowers/specs/2026-05-22-hardening-followup-91-100-design.md`](docs/superpowers/specs/2026-05-22-hardening-followup-91-100-design.md)
**Plan:** [`docs/superpowers/plans/2026-05-22-hardening-followup-91-100-plan.md`](docs/superpowers/plans/2026-05-22-hardening-followup-91-100-plan.md)

## Commits

1. `chore(config): inline MatcherWeights.lambda_l1/giou and LossConfig.focal_* constants (#92, #93)`
2. `chore(peft): drive make_peft_method from registry (#100)`
3. `chore(tracking): delete unused flatten_metrics_report (#95)`
4. `chore(hf): rename resolve_hf_token in notebook_helpers, clarify duplication (#97)`
5. `chore(notebooks): move notebook_helpers.py to notebooks/_lib (#99)`
6. `docs(models): add _patches/README.md SAM-3 version-bump checklist (#96)`

## Test plan

- `uv run pytest -m "not gpu" --no-cov` — green
- `uv run ruff check src/ tests/` — clean
- `uv run ruff format --check src/ tests/` — clean
- Notebook JSON parses; import cell uses `from _lib.notebook_helpers import ...`

## Closes

Closes #91 — EvalConfig.metrics: tracked-only; field already removed.
Closes #92 — MatcherWeights.lambda_l1/giou inlined.
Closes #93 — LossConfig.focal_* inlined.
Closes #94 — early_stop_p_threshold: tracked-only; field already removed.
Closes #95 — flatten_metrics_report deleted.
Closes #96 — _patches/README.md SAM-3 bump checklist added.
Closes #97 — resolve_hf_token duplication renamed + documented.
Closes #99 — notebook_helpers.py moved to notebooks/_lib/ (presets.py kept — premise stale).
Closes #100 — make_peft_method driven from registry.

#98 (QLoRA checkpoint disk-load) is split out for a separate spec/PR — follow-up to come.
EOF
)"
```

- [ ] **Step 7g: Surface the PR URL**

Capture and report the PR URL printed by `gh pr create`. Orchestrator then enters its idle phase per `CLAUDE.md` Implementation-Orchestrator pipeline step 3.

**Acceptance:**
- All §4 manual gate commands return green.
- PR is open against `main` from `hardening-followup-91-100` with the verbatim Closes footer above.
- PR has `@me` as assignee and the `hardening-followup` label.

---

## Closes footer reference (for orchestrator)

The PR description above includes the verbatim `Closes` block from spec §5. Re-listed here so it can be copied without re-reading the spec:

```
Closes #91 — EvalConfig.metrics: tracked-only; field already removed.
Closes #92 — MatcherWeights.lambda_l1/giou inlined.
Closes #93 — LossConfig.focal_* inlined.
Closes #94 — early_stop_p_threshold: tracked-only; field already removed.
Closes #95 — flatten_metrics_report deleted.
Closes #96 — _patches/README.md SAM-3 bump checklist added.
Closes #97 — resolve_hf_token duplication renamed + documented.
Closes #99 — notebook_helpers.py moved to notebooks/_lib/ (presets.py kept — premise stale).
Closes #100 — make_peft_method driven from registry.
```

#98 stays open.
