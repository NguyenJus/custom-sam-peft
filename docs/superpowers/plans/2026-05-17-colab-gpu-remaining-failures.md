# Colab GPU Remaining Failures Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to execute this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Take the Colab T4 GPU integration suite from **4 of 9 passing** (post-PR #13) to **9 of 9 passing**, without regressing the local 240-pass unit baseline.

**Architecture:** Four surgical commits on a new branch cut from `main` after PR #13 lands. Three commits touch source files (`qlora.py`, `lora.py`, `sam3.py`); one commit touches the QLoRA integration test predicate. No new runtime dependencies. No notebook changes. No `pyproject.toml` changes.

**Tech stack:** Python 3.13, PyTorch 2.4+, HuggingFace `peft` 0.19.x, bitsandbytes (Colab T4 only), Meta `sam3`, `pytest`, `ruff`.

**Reference spec:** `docs/superpowers/specs/2026-05-17-colab-gpu-remaining-failures-design.md`
**Sibling plan (matches format and tone):** `docs/superpowers/plans/2026-05-17-colab-gpu-integration-fix-v2.md`
**Parent PR (baseline merged):** [#15](https://github.com/JustinNguyen64/Efficient-SAM3-Finetuning/pull/15) — supersedes the rebased version of PR #13.
**Pre-state on Colab T4:** 4 of 9 passing.
**Post-state target:** 9 of 9 passing.

---

## Decisions (resolutions to the spec's 7 open questions)

These are the planner's binding answers — the implementer does NOT re-derive them.

**Q1 (Issue 1 Option choice): Option (C) — monkey-patch sam3's `_encode_xy` to honor input dtype.**
Rationale: After reading `.venv/lib/python3.13/site-packages/sam3/model/position_encoding.py:60-77`, the fp32 leak is unambiguous: `dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, ...)` forces all downstream products to fp32 regardless of `x.dtype`. PR #13 already removed the broad-scope autocast wrap (commit 68f7c19) because the inner `decoder.py:75-77 forward_ffn` explicitly `enabled=False`'s autocast and re-burned the bf16-vs-fp32 collision when LayerNorm output was promoted. Re-introducing any autocast scope (Option A) re-burns that collision — **forbidden by spec §3.5**. Option (B) (fp32 forward) would balloon T4 VRAM (verified Q2 below). Option (C) — a tiny localized monkey-patch in `src/esam3/models/sam3.py` `load_sam31` that wraps `pos_enc._encode_xy` to cast its output to `x.dtype` — touches NO sam3 source, NO decoder autocast scope, and is testable on CPU via a small stub.

**Q2 (Issue 1 / Option B fitness — fp32 SAM 3.1 on T4): refuted as fallback option.**
Rationale: T4 has 15 GB VRAM. Meta's SAM 3.1 is ~660 M parameters; at fp32 that's ~2.6 GB of weights alone, plus activations through a 1008x1008 vision trunk easily exceeds the remaining budget under our current ungradient-checkpointed config (`ModelConfig(gradient_checkpointing=False)`). v2 spec §3.3 noted the bf16 path was chosen specifically to fit T4. **Option (B) is rejected; we go with Option (C).**

**Q3 (Issue 3 / `compute_dtype` audit): `compute_dtype` is still on the `Linear4bit` module — leave `_infer_compute_dtype_from_wrapper` alone.**
Rationale: Verified by reading `.venv/lib/python3.13/site-packages/peft/tuners/lora/bnb.py:580` in peft 0.19.1 — `target_base_layer.compute_dtype` is still the canonical accessor. No fallback needed.

**Q4 (Issue 3 / `quant_state` lazy-population & the correct attribute path): use `module.weight.quant_type` (NOT `module.weight.quant_state.quant_type`) as primary; fallback to `module.quant_type` (legacy).**
Rationale: Verified by reading `.venv/lib/python3.13/site-packages/peft/tuners/lora/bnb.py:582` — peft itself reads `target_base_layer.weight.quant_type` directly. `quant_type` is a `Params4bit` constructor argument and is set eagerly (independent of `.to(cuda)` / first-forward). `quant_state` (the `QuantState` object that holds absmax tables) IS lazy, but we don't need it for `quant_type` — the field lives on the `Params4bit` parameter itself. This makes the spec §5.3's `weight.quant_state.quant_type` framing incorrect; the planner's correction lands here.

**Q5 (Issue 4 / peft helper reuse — `get_delta_weight`): not applicable; the real fix is upstream of the merge path.**
Rationale: Reading `.venv/lib/python3.13/site-packages/peft/tuners/lora/model.py:226` reveals that `_create_new_module` reads `loaded_in_4bit = getattr(self.model, "is_loaded_in_4bit", False)` and forwards it to the `dispatch_bnb_4bit` dispatcher (`peft/tuners/lora/bnb.py:567-587`). That dispatcher only fires when `loaded_in_4bit=True` AND the target is a `bnb.nn.Linear4bit`. **Our `apply_qlora` swaps the Linears manually but never sets `is_loaded_in_4bit = True` on the base model.** Therefore peft falls through to `dispatch_default` and constructs a generic `peft.tuners.lora.layer.Linear` wrapper — whose `merge()` at line 871 does `base_layer.weight.data += delta_weight`, blowing up on the packed 4-bit `(1572864,)` shape. The fix is **upstream**: set `base.is_loaded_in_4bit = True` BEFORE calling `get_peft_model`. peft then auto-dispatches to `peft.tuners.lora.bnb.Linear4bit`, whose own `merge()` at line 351 already implements the correct dequant-then-add-then-repack path. This is dramatically simpler than the spec §6.4 explicit-dequant algorithm and avoids re-deriving LoRA math.

**Q6 (Issue 4 / post-merge wrapper strip strategy): N/A given Q5.**
Rationale: With Q5's fix, `peft_model.merge_and_unload()` works correctly out-of-the-box; it dequants base weights to compute_dtype, repacks the merged result as a `Params4bit` (per peft bnb.py:398), then unloads the LoRA wrappers. The acceptance criterion `wrapper.peft_model is None AND no Linear4bit modules remain` is partly affected: peft bnb's merge REPACKS as `Params4bit` (still quantized), so `Linear4bit` modules WILL remain after `merge_and_unload`. **Decision: the test assertion at `test_peft_qlora_real.py:122` needs to be relaxed** — the QLoRA merge is "fold deltas into the dequantized base, then quantize back", not "dequantize and stay dequantized". Update the test to assert structural correctness (`peft_model is None`) without requiring Linear4bit removal. This is a test fix that piggybacks onto Task 3.

**Q7 (Branching and PR strategy): ONE PR with 4 commits, one commit per Issue.**
Rationale: spec §7.7 recommendation; matches v2's structure (4 commits in PR #13). Colab verification is per-suite, so amortize one Colab run across the four fixes.

---

## Pre-flight checks

Run these once before starting Task 1.

```bash
# 1. Confirm worktree.
git -C /home/justin/projects/Efficient-SAM3-Finetuning/.claude/worktrees/fix+colab-remaining-failures rev-parse --show-toplevel
# Expected: /home/justin/projects/Efficient-SAM3-Finetuning/.claude/worktrees/fix+colab-remaining-failures

# 2. Confirm clean tree on top of PR #13's merged tip.
git -C /home/justin/projects/Efficient-SAM3-Finetuning/.claude/worktrees/fix+colab-remaining-failures status --porcelain
# Expected: empty

git -C /home/justin/projects/Efficient-SAM3-Finetuning/.claude/worktrees/fix+colab-remaining-failures log --oneline -3
# Expected top line: ac46bfa docs(specs): drop colab GPU remaining failures spec
# (or the commit that adds the parent spec; both 697412f PR #13 and 5071c00 PR #14 must be ancestors).

# 3. Confirm unit baseline is 240 / 1 (will become the lower bound; new tests added by this plan
#    push the count up but never down).
uv run pytest tests/unit -q --no-cov 2>&1 | tail -3
# Expected last line shape: "240 passed, 1 skipped in <T>s"

# 4. Confirm peft is 0.19.x (required for the bnb dispatch path we're using).
uv run python -c "import peft; print(peft.__version__)"
# Expected: 0.19.1 (or any 0.19.x).

# 5. Confirm bitsandbytes is NOT installed locally (correct — local box is GTX 1080, no bnb).
uv run python -c "import bitsandbytes" 2>&1 | tail -1
# Expected: "ModuleNotFoundError: No module named 'bitsandbytes'"
# This is fine: bnb code paths are gated by _bnb_available() and only run on Colab T4.

# 6. Confirm sam3 helpers still import.
uv run python -c "import sam3; from sam3.model.position_encoding import PositionEmbeddingSine; print('OK')"
# Expected: OK
```

If any pre-flight check (except #5) fails, STOP and investigate.

---

## File map (what gets touched)

| File | Action | Owning task |
| --- | --- | --- |
| `src/esam3/peft_adapters/qlora.py` | Modify `_infer_quant_type_from_wrapper`; add `is_loaded_in_4bit = True` flag in `apply_qlora` and `load_qlora`. | 1, 3 |
| `tests/integration/test_peft_qlora_real.py` | Tighten `_has_plain_nn_linear` predicate to ignore LoRA adapter Linears; relax merge assertion (Q6). | 2, 3 |
| `tests/unit/test_peft_qlora.py` (NEW) | CPU unit tests for `_has_plain_nn_linear` predicate and `_infer_quant_type_from_wrapper` fallback chain. | 2, 1 |
| `src/esam3/peft_adapters/lora.py` | No code change — the upstream `is_loaded_in_4bit` flag in qlora.py routes peft to the working `bnb.Linear4bit.merge()` path, leaving `merge_lora` delegating to `peft_model.merge_and_unload()` as before. | (Task 3 verifies; no edit needed.) |
| `src/esam3/models/sam3.py` | Add a monkey-patch helper that wraps `pos_enc._encode_xy` to cast outputs to input dtype; install it inside `load_sam31` after model construction. | 4 |
| `tests/unit/test_sam3_pos_enc_patch.py` (NEW) | CPU smoke test for the `_encode_xy` dtype-cast monkey-patch. | 4 |
| `logs/log.md` | Append one entry per task. | 1, 2, 3, 4, 5 |

No other files are modified. **`pyproject.toml`, `notebooks/colab_gpu_tests.ipynb`, and `src/esam3/peft_adapters/lora.py` stay byte-identical to their pre-plan state.**

---

## Task 1: Fix `_infer_quant_type_from_wrapper` (Issue 3)

**Difficulty:** L
**Subagent:** `implementer` (Sonnet/high). Single-file source change + one new CPU unit test; small but the fallback path needs to be correct.

**Files:**
- Modify: `src/esam3/peft_adapters/qlora.py` (`_infer_quant_type_from_wrapper` body only).
- Create: `tests/unit/test_peft_qlora.py` (or append to it if it exists; new test for the fallback chain).
- Append: `logs/log.md`.

**Expected diff size:** `qlora.py` +~10 / -1 lines. New unit-test file: ~50 lines.

### Scope

Update `_infer_quant_type_from_wrapper` (`src/esam3/peft_adapters/qlora.py:164-174`) to read `module.weight.quant_type` as the primary attribute, with `module.quant_type` as a legacy fallback. Raise a `RuntimeError` with a diagnostic if neither path yields a `str`.

`_infer_compute_dtype_from_wrapper` (`qlora.py:177-192`) stays unchanged (see Decision Q3).

### Reference

- Spec §5 (Problem 3) in full.
- Decisions Q3 and Q4 above.
- Source citations (read but do NOT modify):
  - `.venv/lib/python3.13/site-packages/peft/tuners/lora/bnb.py:580-582` (peft itself reads `target_base_layer.weight.quant_type`).
  - `.venv/lib/python3.13/site-packages/peft/utils/integrations.py:103` (uses `weight.quant_state` for dequant — separate attribute from `quant_type`).

### Steps

- [ ] **Step 1: Read the current function body.**

```bash
sed -n '164,193p' src/esam3/peft_adapters/qlora.py
```

Confirm it matches the spec §5.2 trace.

- [ ] **Step 2: Replace the function body.**

Replace lines 164-174 of `src/esam3/peft_adapters/qlora.py` with:

```python
def _infer_quant_type_from_wrapper(wrapper: Sam3Wrapper) -> str:
    """Read the quant_type from the first Linear4bit module in the wrapped base.

    In current bitsandbytes (the version installed on Colab alongside torch >= 2.4),
    `quant_type` lives on the Params4bit weight (`module.weight.quant_type`), not on
    the Linear4bit module. The legacy attribute `module.quant_type` is also checked
    as a fallback for older bnb builds the original tests were written against.
    """
    bnb = _import_bnb()
    assert wrapper.peft_model is not None
    for module in wrapper.peft_model.modules():
        if isinstance(module, bnb.nn.Linear4bit):
            # Primary: bnb >= the Params4bit-quant_type refactor.
            weight = getattr(module, "weight", None)
            qt = getattr(weight, "quant_type", None) if weight is not None else None
            if isinstance(qt, str):
                return qt
            # Fallback: legacy bnb where Linear4bit carried quant_type directly.
            qt_legacy = getattr(module, "quant_type", None)
            if isinstance(qt_legacy, str):
                return qt_legacy
            raise RuntimeError(
                "save_qlora: could not infer quant_type from Linear4bit module. "
                f"module repr: {module!r}; "
                f"bnb.__version__={getattr(bnb, '__version__', '<unknown>')}; "
                "expected `module.weight.quant_type` (current) or `module.quant_type` (legacy) "
                "to be a str."
            )
    raise RuntimeError(
        "save_qlora: wrapper.peft_model contains no Linear4bit modules; "
        "this should not happen after apply_qlora"
    )
```

Leave `_infer_compute_dtype_from_wrapper` (the next function) entirely untouched.

- [ ] **Step 3: Create the CPU unit test scaffold.**

Create `tests/unit/test_peft_qlora.py` if it does not exist; otherwise append to it. The test must NOT import `bitsandbytes` (the local dev box has no bnb), so use lightweight stand-ins.

```python
"""CPU unit tests for QLoRA helper internals.

These tests do NOT import bitsandbytes; they stand in fakes that mimic the
`Linear4bit` / `Params4bit` shape just enough to exercise the attribute-read
fallbacks in `_infer_quant_type_from_wrapper`.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest
from torch import nn


@pytest.fixture
def fake_bnb(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """Install a tiny fake `bitsandbytes` module exposing `bnb.nn.Linear4bit`."""
    fake = types.ModuleType("bitsandbytes")
    fake_nn = types.ModuleType("bitsandbytes.nn")

    class _FakeLinear4bit(nn.Module):
        def __init__(
            self,
            *,
            weight_quant_type: str | None = None,
            module_quant_type: str | None = None,
        ) -> None:
            super().__init__()
            # Mimic a Params4bit weight with `.quant_type` directly on the weight.
            weight = nn.Parameter(nn.functional.normalize(nn.Linear(2, 2).weight))
            if weight_quant_type is not None:
                weight.quant_type = weight_quant_type  # type: ignore[attr-defined]
            self.weight = weight
            if module_quant_type is not None:
                self.quant_type = module_quant_type  # type: ignore[attr-defined]

    fake_nn.Linear4bit = _FakeLinear4bit  # type: ignore[attr-defined]
    fake.nn = fake_nn  # type: ignore[attr-defined]
    fake.__version__ = "0.fake.0"  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "bitsandbytes", fake)
    monkeypatch.setitem(sys.modules, "bitsandbytes.nn", fake_nn)
    return fake


class _FakeWrapper:
    """Stand-in for Sam3Wrapper holding a `peft_model` attribute."""

    def __init__(self, peft_model: nn.Module) -> None:
        self.peft_model = peft_model


def test_infer_quant_type_primary_path(fake_bnb: types.ModuleType) -> None:
    from esam3.peft_adapters.qlora import _infer_quant_type_from_wrapper

    fake_linear4bit = fake_bnb.nn.Linear4bit(weight_quant_type="nf4")  # type: ignore[attr-defined]
    model = nn.Sequential(fake_linear4bit)
    wrapper: Any = _FakeWrapper(model)
    assert _infer_quant_type_from_wrapper(wrapper) == "nf4"


def test_infer_quant_type_legacy_fallback(fake_bnb: types.ModuleType) -> None:
    from esam3.peft_adapters.qlora import _infer_quant_type_from_wrapper

    fake_linear4bit = fake_bnb.nn.Linear4bit(module_quant_type="fp4")  # type: ignore[attr-defined]
    model = nn.Sequential(fake_linear4bit)
    wrapper: Any = _FakeWrapper(model)
    assert _infer_quant_type_from_wrapper(wrapper) == "fp4"


def test_infer_quant_type_raises_when_both_paths_missing(
    fake_bnb: types.ModuleType,
) -> None:
    from esam3.peft_adapters.qlora import _infer_quant_type_from_wrapper

    fake_linear4bit = fake_bnb.nn.Linear4bit()  # no quant_type set anywhere
    model = nn.Sequential(fake_linear4bit)
    wrapper: Any = _FakeWrapper(model)
    with pytest.raises(RuntimeError, match="could not infer quant_type"):
        _infer_quant_type_from_wrapper(wrapper)
```

- [ ] **Step 4: Lint and format.**

```bash
uv run ruff check src/esam3/peft_adapters/qlora.py tests/unit/test_peft_qlora.py
uv run ruff format --check src/esam3/peft_adapters/qlora.py tests/unit/test_peft_qlora.py
```

Fix any reports in place.

- [ ] **Step 5: Run unit tests.**

```bash
uv run pytest tests/unit/test_peft_qlora.py -v --no-cov 2>&1 | tail -10
uv run pytest tests/unit -q --no-cov 2>&1 | tail -3
```

Expected: 3 new tests pass; total count now `243 passed, 1 skipped`.

- [ ] **Step 6: Append to `logs/log.md`.**

```
[<UTC-ISO8601>] [implementer] task-1 remaining-failures: fixed _infer_quant_type_from_wrapper to read module.weight.quant_type (primary) with module.quant_type legacy fallback; added 3 CPU unit tests via fake bnb fixture
```

- [ ] **Step 7: Commit.**

```bash
git add src/esam3/peft_adapters/qlora.py tests/unit/test_peft_qlora.py logs/log.md
git commit -m "$(cat <<'EOF'
fix(qlora): read quant_type from Params4bit weight, not Linear4bit module

In current bitsandbytes (the version installed on Colab T4 alongside torch
>= 2.4), `quant_type` is no longer a direct attribute on `bnb.nn.Linear4bit`.
It moved onto the `Params4bit` parameter (`module.weight.quant_type`),
verified against peft 0.19's own access at peft/tuners/lora/bnb.py:582.

`_infer_quant_type_from_wrapper` raised `AttributeError: 'Linear4bit' object
has no attribute 'quant_type'` on Colab, blocking save_qlora /
save_load_qlora_roundtrip in tests/integration/test_peft_qlora_real.py.

Read `module.weight.quant_type` as primary, fall back to `module.quant_type`
for older bnb (the version this code was originally written against), and
raise a diagnostic RuntimeError naming the bnb version and module repr if
neither path yields a str.

Adds 3 CPU unit tests via a fake-bnb fixture covering primary path, legacy
fallback, and the error branch. `_infer_compute_dtype_from_wrapper` is
unchanged: `compute_dtype` still lives on the Linear4bit module per peft
0.19's own usage at peft/tuners/lora/bnb.py:580.

Unblocks:
- tests/integration/test_peft_qlora_real.py::test_save_qlora_writes_adapter_and_metadata
- tests/integration/test_peft_qlora_real.py::test_save_load_qlora_roundtrip
EOF
)"
```

### Definition of Done

- [ ] `src/esam3/peft_adapters/qlora.py::_infer_quant_type_from_wrapper` reads `module.weight.quant_type` first.
- [ ] Legacy fallback to `module.quant_type` is present.
- [ ] Both-missing case raises `RuntimeError` whose message mentions `quant_type` and bnb version.
- [ ] `_infer_compute_dtype_from_wrapper` body is byte-identical to its pre-task state.
- [ ] `tests/unit/test_peft_qlora.py` contains the 3 new tests; all pass.
- [ ] `uv run pytest tests/unit -q --no-cov` reports `243 passed, 1 skipped` (or pre-plan-baseline + 3); ZERO regressions.
- [ ] `ruff check` / `ruff format --check` both pass for `src/esam3/peft_adapters/qlora.py` and `tests/unit/test_peft_qlora.py`.
- [ ] `logs/log.md` has the Task-1 entry.
- [ ] One new commit on branch tip.

### Verification (commands)

```bash
grep -n "module.weight.quant_type\|module.quant_type\|could not infer quant_type" src/esam3/peft_adapters/qlora.py
# Expected: 3+ matches (primary read, fallback read, error message).
uv run pytest tests/unit/test_peft_qlora.py -v --no-cov 2>&1 | tail -10
uv run pytest tests/unit -q --no-cov 2>&1 | tail -3
uv run ruff check src/esam3/peft_adapters/qlora.py tests/unit/test_peft_qlora.py
uv run ruff format --check src/esam3/peft_adapters/qlora.py tests/unit/test_peft_qlora.py
```

### Rollback

```bash
git reset --hard HEAD~1
# Restores pre-task state.
```

---

## Task 2: Tighten `_has_plain_nn_linear` to ignore LoRA adapter modules (Issue 2)

**Difficulty:** L
**Subagent:** `implementer-simple` (Haiku/high). Single test-file change + one CPU unit test; pure logic with no side effects.

**Files:**
- Modify: `tests/integration/test_peft_qlora_real.py` (`_has_plain_nn_linear` predicate at lines 49-51 only).
- Append: `tests/unit/test_peft_qlora.py` (one new CPU unit test for the tightened predicate).
- Append: `logs/log.md`.

**Expected diff size:** `test_peft_qlora_real.py` +~15 / -3 lines (replace the 3-line predicate with a name-scoped walker). `test_peft_qlora.py` +~50 lines.

### Scope

Replace the over-coarse predicate that catches LoRA adapter Linears (`lora_A`, `lora_B`) along with legitimate base-Linear leaks. The replacement walks `named_modules()` and skips any plain `nn.Linear` whose qualified name contains a known LoRA adapter sub-path.

### Reference

- Spec §4 (Problem 2) in full.
- Source: `tests/integration/test_peft_qlora_real.py:49-51`.

### Steps

- [ ] **Step 1: Read the current predicate.**

```bash
sed -n '49,52p' tests/integration/test_peft_qlora_real.py
```

- [ ] **Step 2: Replace the predicate.**

Replace the 3-line `_has_plain_nn_linear` definition with the name-scoped version:

```python
# LoRA adapter sub-paths whose internal nn.Linear modules are EXPECTED and must
# not be flagged as a "base Linear leak". peft.tuners.lora.layer.Linear stores
# the trainable LoRA matrices in `lora_A.<adapter>` and `lora_B.<adapter>`
# nn.Linear submodules; the QLoRA recipe leaves these in full precision while
# the base layer is bnb.nn.Linear4bit. Add `lora_embedding_A`, `lora_embedding_B`,
# and `lora_magnitude_vector` defensively for future LoRA variants (Embedding
# adapters, DoRA) even though we do not configure them today.
_LORA_ADAPTER_PATH_TOKENS = (
    "lora_A",
    "lora_B",
    "lora_embedding_A",
    "lora_embedding_B",
    "lora_magnitude_vector",
)


def _has_plain_nn_linear(module: nn.Module) -> bool:
    """True if any nn.Linear remains in the BASE tree (NOT under a LoRA adapter path).

    Subclasses of nn.Linear (e.g. bnb.nn.Linear4bit) are ignored via `type(m) is`.
    Plain nn.Linear modules whose qualified name from `named_modules()` contains
    any token in `_LORA_ADAPTER_PATH_TOKENS` are also ignored: they belong to
    LoRA's full-precision adapter, not the base.
    """
    for name, m in module.named_modules():
        if type(m) is not nn.Linear:
            continue
        if any(tok in name for tok in _LORA_ADAPTER_PATH_TOKENS):
            continue
        return True
    return False
```

Keep `_LORA_ADAPTER_PATH_TOKENS` at module scope so the new CPU unit test in Step 3 can import and re-use it.

- [ ] **Step 3: Add the CPU unit test.**

Append to `tests/unit/test_peft_qlora.py` (created in Task 1) the following test. It mirrors the spec §4.4 acceptance scaffold: a tiny tree with one base `nn.Linear` left intact + one swapped to a sentinel + a fake LoRA wrapper carrying `lora_A` / `lora_B` `nn.Linear` children.

```python
def test_has_plain_nn_linear_ignores_lora_adapter_children() -> None:
    """The tightened predicate must ignore lora_A/lora_B nn.Linears but flag base leaks."""
    from tests.integration.test_peft_qlora_real import _has_plain_nn_linear

    # Fake LoRA adapter wrapper: holds a Linear4bit-shape sentinel as base, plus
    # full-precision lora_A / lora_B nn.Linear adapters (mimicking peft.tuners.lora.bnb.Linear4bit).
    class _Linear4bitSentinel(nn.Linear):
        """Subclass of nn.Linear (mimics bnb.nn.Linear4bit subclassing)."""

    class _FakeLoraWrapper(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.base_layer = _Linear4bitSentinel(4, 4)
            self.lora_A = nn.ModuleDict({"default": nn.Linear(4, 2, bias=False)})
            self.lora_B = nn.ModuleDict({"default": nn.Linear(2, 4, bias=False)})

    # Case 1: all base Linears already swapped (only Linear4bitSentinel + lora adapters).
    # Expected: predicate returns False.
    clean = nn.Sequential(_FakeLoraWrapper(), _FakeLoraWrapper())
    assert not _has_plain_nn_linear(clean), (
        "predicate must not flag lora_A/lora_B adapter Linears as base leaks"
    )

    # Case 2: introduce a real base-Linear leak alongside the LoRA-wrapped layers.
    # Expected: predicate returns True (the leaked plain nn.Linear is NOT under a lora_* path).
    leaked = nn.Sequential(_FakeLoraWrapper(), nn.Linear(4, 4))
    assert _has_plain_nn_linear(leaked), (
        "predicate must still flag a true base nn.Linear leak"
    )
```

Note: the import `from tests.integration.test_peft_qlora_real import _has_plain_nn_linear` works because tests are run from repo root with `pytest`, which puts `tests/` on `sys.path`. If a `tests/__init__.py` does not exist, copy the predicate body inline rather than importing across packages. Decision: prefer the import; if it fails, fall back to inline duplication and add a `tests/__init__.py` only if doing so does not break the existing test layout (verify `uv run pytest tests/unit -q --no-cov` stays green either way).

- [ ] **Step 4: Lint and format.**

```bash
uv run ruff check tests/integration/test_peft_qlora_real.py tests/unit/test_peft_qlora.py
uv run ruff format --check tests/integration/test_peft_qlora_real.py tests/unit/test_peft_qlora.py
```

- [ ] **Step 5: Run unit tests.**

```bash
uv run pytest tests/unit -q --no-cov 2>&1 | tail -3
```

Expected: count now `244 passed, 1 skipped` (Task 1's 3 + Task 2's 1 = +4 total over the pre-plan 240 baseline).

- [ ] **Step 6: Append to `logs/log.md`.**

```
[<UTC-ISO8601>] [implementer] task-2 remaining-failures: tightened _has_plain_nn_linear in tests/integration/test_peft_qlora_real.py to skip lora_A/lora_B adapter paths; added CPU unit test
```

- [ ] **Step 7: Commit.**

```bash
git add tests/integration/test_peft_qlora_real.py tests/unit/test_peft_qlora.py logs/log.md
git commit -m "$(cat <<'EOF'
test(qlora): tighten _has_plain_nn_linear to skip LoRA adapter Linears

The post-apply_qlora module tree contains plain nn.Linear instances inside
peft.tuners.lora.layer.Linear's `lora_A` / `lora_B` adapter dicts (LoRA
matrices live in full precision; quantization is only for the base layer).
The previous predicate flagged those adapter Linears as "base leaks" and
failed assert not _has_plain_nn_linear(base) at line 61, even though every
base nn.Linear had been correctly swapped to bnb.nn.Linear4bit.

Tighten the predicate to walk named_modules() and ignore any plain
nn.Linear whose qualified name contains a LoRA adapter token
(`lora_A`, `lora_B`, plus defensive `lora_embedding_A` / `lora_embedding_B` /
`lora_magnitude_vector` for future Embedding-LoRA and DoRA variants).

Adds a CPU unit test that constructs a fake LoRA wrapper around a
Linear4bit-shaped sentinel and asserts: (a) the predicate ignores the
lora_A/lora_B children, (b) it still flags a true base nn.Linear leak.

Unblocks:
- tests/integration/test_peft_qlora_real.py::test_apply_qlora_swaps_every_linear_and_attaches_lora
EOF
)"
```

### Definition of Done

- [ ] `tests/integration/test_peft_qlora_real.py::_has_plain_nn_linear` walks `named_modules()` (not `modules()`).
- [ ] `_LORA_ADAPTER_PATH_TOKENS` constant is defined and contains at least the 5 tokens listed.
- [ ] `apply_qlora` production code is byte-identical to its pre-task state (sanity: `git diff HEAD~1 HEAD -- src/esam3/peft_adapters/qlora.py` is empty for Task 2's commit).
- [ ] `tests/unit/test_peft_qlora.py::test_has_plain_nn_linear_ignores_lora_adapter_children` exists and passes.
- [ ] `uv run pytest tests/unit -q --no-cov` reports `244 passed, 1 skipped` (or +1 from Task 1's count); ZERO regressions.
- [ ] `ruff check` / `ruff format --check` pass for both changed files.
- [ ] `logs/log.md` has the Task-2 entry.
- [ ] One new commit on branch tip.

### Verification (commands)

```bash
grep -n "_LORA_ADAPTER_PATH_TOKENS\|named_modules" tests/integration/test_peft_qlora_real.py
# Expected: at least 2 matches each (token defn + usage in predicate).
uv run pytest tests/unit/test_peft_qlora.py -v --no-cov 2>&1 | tail -8
uv run pytest tests/unit -q --no-cov 2>&1 | tail -3
uv run ruff check tests/integration/test_peft_qlora_real.py tests/unit/test_peft_qlora.py
uv run ruff format --check tests/integration/test_peft_qlora_real.py tests/unit/test_peft_qlora.py
git diff HEAD~1 HEAD -- src/esam3/peft_adapters/qlora.py
# Expected: empty (no qlora.py changes in this commit).
```

### Rollback

```bash
git reset --hard HEAD~1
```

---

## Task 3: Make peft dispatch to `bnb.Linear4bit.merge()` by setting `is_loaded_in_4bit` (Issue 4)

**Difficulty:** M
**Subagent:** `implementer` (Sonnet/high). Two-line production fix in `apply_qlora` and `load_qlora`, but the reasoning is subtle (correct dispatcher routing), and the test-fix piggyback (Q6) requires care to preserve the test's intent.

**Files:**
- Modify: `src/esam3/peft_adapters/qlora.py` (`apply_qlora` and `load_qlora` — add `base.is_loaded_in_4bit = True` AFTER the Linear-to-Linear4bit swap and BEFORE `get_peft_model` / `PeftModel.from_pretrained`).
- Modify: `tests/integration/test_peft_qlora_real.py` (relax the post-merge assertion per Q6).
- Append: `tests/unit/test_peft_qlora.py` (one new CPU unit test asserting the flag is set after a fake `apply_qlora` call against the fake-bnb fixture).
- Append: `logs/log.md`.

**Expected diff size:** `qlora.py` +~4 / -0 lines (two flag assignments with a comment block). `test_peft_qlora_real.py` +~6 / -3 lines (replace the `not _has_linear4bit_modules(base)` assert with a structural-only check). `test_peft_qlora.py` +~30 lines.

### Scope

**Production change (qlora.py):** Set `base.is_loaded_in_4bit = True` immediately after `_replace_with_bnb_linear4bit` and BEFORE `prepare_model_for_kbit_training` + `get_peft_model`. Apply the same change in `load_qlora` before `PeftModel.from_pretrained`. This causes peft's `_create_new_module` (`peft/tuners/lora/model.py:226`) to pass `loaded_in_4bit=True` into `dispatch_bnb_4bit` (`peft/tuners/lora/bnb.py:567-587`), which then wraps each `Linear4bit` in `peft.tuners.lora.bnb.Linear4bit` (which has a correct `merge()` at line 351) instead of falling through to `dispatch_default`'s generic `peft.tuners.lora.layer.Linear` (which is what blew up at line 871).

**Test change (test_peft_qlora_real.py):** With the upstream fix, `peft.tuners.lora.bnb.Linear4bit.merge()` repacks the merged result as a `Params4bit` (peft bnb.py:398) — so `Linear4bit` modules **remain present** after `merge_and_unload()`. The current test asserts they are absent, which contradicts peft's design. Relax that assertion: keep `assert w.peft_model is None` (structural correctness) and drop the `not _has_linear4bit_modules(base)` assertion in favor of a comment explaining the QLoRA merge semantics (deltas folded; quantization preserved).

`src/esam3/peft_adapters/lora.py::merge_lora` stays unchanged (its delegation to `peft_model.merge_and_unload()` is correct once the dispatcher picks the right wrapper class).

### Reference

- Spec §6 (Problem 4) in full.
- Decisions Q5 and Q6 above.
- Source citations (read but do NOT modify):
  - `.venv/lib/python3.13/site-packages/peft/tuners/lora/model.py:221-230` (kwargs assembly; line 226 is the `loaded_in_4bit` read).
  - `.venv/lib/python3.13/site-packages/peft/tuners/lora/bnb.py:567-587` (`dispatch_bnb_4bit`).
  - `.venv/lib/python3.13/site-packages/peft/tuners/lora/bnb.py:351-408` (`peft.tuners.lora.bnb.Linear4bit.merge()` — already correct).
  - `.venv/lib/python3.13/site-packages/peft/tuners/lora/layer.py:871` (the generic `lora.Linear.merge` line that currently raises).
  - `.venv/lib/python3.13/site-packages/peft/utils/other.py:160` (where `prepare_model_for_kbit_training` reads `is_loaded_in_4bit` — but does NOT set it, hence the need to set it ourselves).

### Steps

- [ ] **Step 1: Apply the production fix in `apply_qlora`.**

In `src/esam3/peft_adapters/qlora.py`, find the `apply_qlora` body. Immediately AFTER the line `_replace_with_bnb_linear4bit(base, quant_names, cfg.qlora)` and BEFORE the line `lora_target_names = _resolve_targets(base, cfg, linear_types=(bnb.nn.Linear4bit,))`, insert:

```python
    # Tell peft that the base model is now 4-bit quantized. peft's
    # _create_new_module dispatcher reads getattr(self.model, "is_loaded_in_4bit",
    # False) (peft/tuners/lora/model.py:226) and forwards it into the
    # dispatch_bnb_4bit predicate (peft/tuners/lora/bnb.py:576). Without this
    # flag, dispatch_bnb_4bit returns None and peft falls through to
    # dispatch_default, which constructs a generic lora.layer.Linear wrapper
    # whose merge() at lora/layer.py:871 does `base_layer.weight.data +=
    # delta_weight` — incompatible with a packed Params4bit weight. Setting
    # the flag routes peft to lora.bnb.Linear4bit, whose merge() correctly
    # dequantizes, folds the LoRA delta, and repacks.
    base.is_loaded_in_4bit = True  # type: ignore[attr-defined]
```

- [ ] **Step 2: Apply the same fix in `load_qlora`.**

In the same file, find `load_qlora`. Immediately AFTER the line `_replace_with_bnb_linear4bit(base, quant_names, qcfg)` and BEFORE the `prepare_model_for_kbit_training(...)` call, insert:

```python
    # See apply_qlora for the rationale. Required so PeftModel.from_pretrained's
    # internal _create_new_module dispatches to lora.bnb.Linear4bit instead of
    # the generic lora.layer.Linear.
    base.is_loaded_in_4bit = True  # type: ignore[attr-defined]
```

- [ ] **Step 3: Relax the merge assertion in the integration test (Q6).**

In `tests/integration/test_peft_qlora_real.py`, find `test_merge_lora_dequantizes_qlora_wrapper` (lines ~114-122). Replace the body's last 3 lines:

From:
```python
    assert w.peft_model is None
    base = w.model.model
    assert not _has_linear4bit_modules(base), "Linear4bit modules remain after merge"
```

To:
```python
    # Structural acceptance after merge:
    #   1. peft_model must be detached (we unloaded the LoRA wrappers).
    #   2. Linear4bit modules MAY remain: peft.tuners.lora.bnb.Linear4bit.merge()
    #      dequantizes the base weight, folds the LoRA delta, then re-packs as
    #      Params4bit (peft/tuners/lora/bnb.py:398). Quantization is preserved
    #      across the merge; only the LoRA wrapper is unloaded. Asserting
    #      Linear4bit absence here would contradict peft's documented design.
    assert w.peft_model is None
    base = w.model.model
    assert _has_linear4bit_modules(base), (
        "Linear4bit modules should remain (peft re-packs the merged weight as "
        "Params4bit; quantization is preserved across merge_and_unload)"
    )
```

Also rename the docstring or test name if appropriate — the test name `test_merge_lora_dequantizes_qlora_wrapper` is now misleading. Decision: rename to `test_merge_lora_unloads_qlora_wrapper`. Update all references (single test, no other callers).

- [ ] **Step 4: Add a CPU unit test that exercises the flag-set behavior.**

Append to `tests/unit/test_peft_qlora.py`:

```python
def test_apply_qlora_sets_is_loaded_in_4bit_flag_on_base(
    fake_bnb: types.ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """apply_qlora must set base.is_loaded_in_4bit = True before peft dispatch.

    This is a tight unit-level smoke: we mock out the peft side (get_peft_model
    is a no-op) and only verify the flag mutation. The Colab integration test
    then verifies the downstream consequence (correct merge path).
    """
    # Build a stand-in Sam3Wrapper whose .model.model is a tiny module containing
    # an nn.Linear that will be "swapped" by the fake bnb path.
    inner = nn.Sequential(nn.Linear(4, 4))

    class _FakeModelHolder:
        def __init__(self) -> None:
            self.model = inner

    class _Sam3WrapperLike:
        def __init__(self) -> None:
            self.model = _FakeModelHolder()
            self.peft_model = None

    wrapper: Any = _Sam3WrapperLike()

    # Stub out _replace_with_bnb_linear4bit to a no-op (we don't have real bnb here);
    # the production code at this point would have done the swap, then must set the flag.
    from esam3.peft_adapters import qlora as qlora_mod

    monkeypatch.setattr(
        qlora_mod, "_replace_with_bnb_linear4bit", lambda base, names, qcfg: None
    )
    monkeypatch.setattr(qlora_mod, "_collect_linear_names", lambda base: ["0"])

    # Stub out the peft entry points to avoid pulling peft at unit-test time.
    import sys

    fake_peft = types.ModuleType("peft")

    def _fake_get_peft_model(base: Any, lora_cfg: Any) -> Any:
        # Side effect: peft reads is_loaded_in_4bit during _create_new_module.
        # We snapshot it here so the test can assert it was set BEFORE this call.
        _fake_get_peft_model.observed_flag = getattr(base, "is_loaded_in_4bit", False)
        return base

    _fake_get_peft_model.observed_flag = False
    fake_peft.LoraConfig = lambda **kw: types.SimpleNamespace(**kw)  # type: ignore[attr-defined]
    fake_peft.get_peft_model = _fake_get_peft_model  # type: ignore[attr-defined]
    fake_peft.prepare_model_for_kbit_training = lambda base, **kw: base  # type: ignore[attr-defined]
    fake_peft.PeftModel = type("PeftModel", (), {})  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "peft", fake_peft)

    # Also stub _resolve_targets so we don't traverse a non-bnb tree.
    monkeypatch.setattr(qlora_mod, "_resolve_targets", lambda base, cfg, linear_types=None: ["0"])

    from esam3.config.schema import PEFTConfig

    qlora_mod.apply_qlora(wrapper, PEFTConfig(method="qlora"))

    assert _fake_get_peft_model.observed_flag is True, (
        "apply_qlora must set base.is_loaded_in_4bit = True BEFORE get_peft_model"
    )
```

- [ ] **Step 5: Lint and format.**

```bash
uv run ruff check src/esam3/peft_adapters/qlora.py tests/integration/test_peft_qlora_real.py tests/unit/test_peft_qlora.py
uv run ruff format --check src/esam3/peft_adapters/qlora.py tests/integration/test_peft_qlora_real.py tests/unit/test_peft_qlora.py
```

- [ ] **Step 6: Run unit tests.**

```bash
uv run pytest tests/unit -q --no-cov 2>&1 | tail -3
```

Expected: `245 passed, 1 skipped` (one more new test).

- [ ] **Step 7: Append to `logs/log.md`.**

```
[<UTC-ISO8601>] [implementer] task-3 remaining-failures: set base.is_loaded_in_4bit=True in apply_qlora and load_qlora so peft dispatches to lora.bnb.Linear4bit (correct merge); relaxed merge test to match peft's repack-as-Params4bit semantics
```

- [ ] **Step 8: Commit.**

```bash
git add src/esam3/peft_adapters/qlora.py tests/integration/test_peft_qlora_real.py tests/unit/test_peft_qlora.py logs/log.md
git commit -m "$(cat <<'EOF'
fix(qlora): set is_loaded_in_4bit so peft dispatches to bnb.Linear4bit merge

peft's _create_new_module reads `getattr(self.model, "is_loaded_in_4bit",
False)` (peft/tuners/lora/model.py:226) and forwards it into dispatch_bnb_4bit
(peft/tuners/lora/bnb.py:567-587). Our apply_qlora and load_qlora swap
nn.Linear → bnb.nn.Linear4bit manually but never set that flag, so peft falls
through to dispatch_default and wraps each Linear4bit in a generic
peft.tuners.lora.layer.Linear. That generic class's merge() at
peft/tuners/lora/layer.py:871 does `base_layer.weight.data += delta_weight`,
which raises `RuntimeError: The size of tensor a (1572864) must match the size
of tensor b (3072)` because Params4bit's data is a packed 1-D uint8 blob, not
an (out_features, in_features) fp tensor.

Set `base.is_loaded_in_4bit = True` in both apply_qlora and load_qlora,
between the Linear4bit swap and the peft entry point. peft then routes to
peft.tuners.lora.bnb.Linear4bit (peft/tuners/lora/bnb.py:311), whose merge()
at line 351 correctly dequantizes via dequantize_bnb_weight, folds the LoRA
delta, and repacks as Params4bit (line 398). No code-side dequant-then-merge
is needed; peft already implements it.

Side effect documented in the merge integration test: Linear4bit modules
REMAIN after merge_and_unload (peft re-packs the merged weight as Params4bit;
quantization is preserved). The previous test asserted Linear4bit absence,
contradicting peft's design. Relax to require structural correctness
(peft_model is None) + presence of Linear4bit (quantization preserved).
Rename test to test_merge_lora_unloads_qlora_wrapper to match the new
semantics.

Adds a CPU unit test using a fake peft.get_peft_model to confirm the flag is
set BEFORE the peft entry point fires.

Unblocks:
- tests/integration/test_peft_qlora_real.py::test_merge_lora_unloads_qlora_wrapper
EOF
)"
```

### Definition of Done

- [ ] `src/esam3/peft_adapters/qlora.py::apply_qlora` sets `base.is_loaded_in_4bit = True` after the Linear4bit swap and before `get_peft_model`.
- [ ] `src/esam3/peft_adapters/qlora.py::load_qlora` sets `base.is_loaded_in_4bit = True` after the Linear4bit swap and before `PeftModel.from_pretrained`.
- [ ] `src/esam3/peft_adapters/lora.py::merge_lora` is byte-identical to its pre-task state.
- [ ] `tests/integration/test_peft_qlora_real.py::test_merge_lora_unloads_qlora_wrapper` (renamed from `test_merge_lora_dequantizes_qlora_wrapper`) asserts `w.peft_model is None` AND `_has_linear4bit_modules(base)`.
- [ ] `tests/unit/test_peft_qlora.py::test_apply_qlora_sets_is_loaded_in_4bit_flag_on_base` exists and passes.
- [ ] `uv run pytest tests/unit -q --no-cov` reports `245 passed, 1 skipped` (or +1 from Task 2's count).
- [ ] `ruff check` / `ruff format --check` pass for all 3 changed files.
- [ ] `logs/log.md` has the Task-3 entry.
- [ ] One new commit on branch tip.

### Verification (commands)

```bash
grep -n "is_loaded_in_4bit" src/esam3/peft_adapters/qlora.py
# Expected: 2 assignments (one in apply_qlora, one in load_qlora) + comment block lines.
grep -n "test_merge_lora_unloads_qlora_wrapper\|test_merge_lora_dequantizes_qlora_wrapper" tests/integration/test_peft_qlora_real.py
# Expected: only the *unloads* name appears (1 match); *dequantizes* gone (0 matches).
uv run pytest tests/unit/test_peft_qlora.py -v --no-cov 2>&1 | tail -8
uv run pytest tests/unit -q --no-cov 2>&1 | tail -3
git diff HEAD~1 HEAD -- src/esam3/peft_adapters/lora.py
# Expected: empty (lora.py unchanged).
```

### Rollback

```bash
git reset --hard HEAD~1
```

If after pushing, the Colab T4 run still raises `RuntimeError: The size of tensor a (...) must match the size of tensor b (...)` at peft/tuners/lora/layer.py:871, the flag is not being honored. Diagnose: ask the user to run on Colab:

```python
import bitsandbytes as bnb
from esam3.config.schema import ModelConfig, PEFTConfig
from esam3.models.sam3 import load_sam31
from esam3.peft_adapters.qlora import apply_qlora
w = load_sam31(ModelConfig())
apply_qlora(w, PEFTConfig(method="qlora"))
from peft.tuners.lora.bnb import Linear4bit as PeftBnbLinear4bit
counts = {"bnb_wrapper": 0, "generic_wrapper": 0}
for m in w.peft_model.modules():
    if isinstance(m, PeftBnbLinear4bit):
        counts["bnb_wrapper"] += 1
    elif m.__class__.__name__ == "Linear" and m.__class__.__module__ == "peft.tuners.lora.layer":
        counts["generic_wrapper"] += 1
print(counts)
```

`counts["bnb_wrapper"]` must be > 0 and `counts["generic_wrapper"]` should be 0. If both are 0, peft is not seeing any LoRA targets. If `generic_wrapper > 0`, the flag set is not happening before peft dispatch — investigate ordering.

---

## Task 4: Monkey-patch sam3's `_encode_xy` to honor input dtype (Issue 1)

**Difficulty:** M
**Subagent:** `implementer` (Sonnet/high). Single-file production change; reasoning around the v2 decoder-burn must be respected (no autocast).

**Files:**
- Modify: `src/esam3/models/sam3.py` (add `_patch_pos_enc_dtype` helper near the bottom of the file; call it from `load_sam31` after the dtype cast and before constructing the adapter).
- Create: `tests/unit/test_sam3_pos_enc_patch.py` (CPU smoke test for the patch).
- Append: `logs/log.md`.

**Expected diff size:** `sam3.py` +~30 / -0 lines (one new helper function + one call inside `load_sam31`). New unit test ~50 lines.

### Scope

After PR #13's empty-Prompt fallback already supplies explicit `point_embeddings` of `model_dtype` (sam3.py lines 241-244), the dtype mismatch has moved one level deeper: `sam3.model.geometry_encoders.PointGeometryEncoder._encode_points` calls `self.pos_enc._encode_xy(...)`, which forces fp32 via `dim_t = torch.arange(..., dtype=torch.float32, ...)` (`sam3/model/position_encoding.py:66`). The fp32 output then hits a bf16-weight `self.points_pos_enc_project` Linear at line 623 → mismatch.

Fix: monkey-patch the bound `_encode_xy` method on every `PositionEmbeddingSine` instance reachable from the loaded sam3 model, wrapping it to cast its outputs to the input's dtype. Install the patch inside `load_sam31` AFTER `raw_model.to(dtype=...)` so the patch sees the final weight dtype context (though the patch itself reads dtype from the runtime input, not from any baked-in value).

This is Option (C) from spec §3.4. It does NOT introduce any autocast scope, so it cannot re-trigger the v2 `decoder.py:75-80 forward_ffn` collision (verified by reading those lines: the explicit `enabled=False` is an outer autocast disable, irrelevant to a localized output cast).

### Reference

- Spec §3 (Problem 1) in full.
- Decisions Q1 and Q2 above.
- Source citations (read but do NOT modify):
  - `.venv/lib/python3.13/site-packages/sam3/model/position_encoding.py:60-77` (`_encode_xy` — the fp32 leak).
  - `.venv/lib/python3.13/site-packages/sam3/model/position_encoding.py:79-94` (`encode_boxes`, `encode_points` — both call `_encode_xy`, so patching `_encode_xy` covers both).
  - `.venv/lib/python3.13/site-packages/sam3/model/geometry_encoders.py:589-630` (`_encode_points`, the consumer that feeds `_encode_xy` output into a bf16 Linear).
  - `.venv/lib/python3.13/site-packages/sam3/model/decoder.py:75-80` (`forward_ffn` with `enabled=False` autocast — confirm our patch does NOT add any autocast scope, hence cannot re-trigger the v2 collision).

### Steps

- [ ] **Step 1: Read the current `load_sam31` body to locate the insertion point.**

```bash
sed -n '255,291p' src/esam3/models/sam3.py
```

Expected: `load_sam31` ends with `adapter = _Sam3ImageAdapter(raw_model, image_size=1008); return Sam3Wrapper(...)`. The patch call goes BETWEEN `raw_model = raw_model.to(dtype=...)` and `adapter = _Sam3ImageAdapter(...)`.

- [ ] **Step 2: Add the `_patch_pos_enc_dtype` helper.**

Insert this helper function near the bottom of `src/esam3/models/sam3.py`, between `_Sam3ImageAdapter` and `load_sam31` (or above `_Sam3ImageAdapter` — wherever import order lets the helper be referenced from `load_sam31`):

```python
def _patch_pos_enc_dtype(model: nn.Module) -> None:
    """Wrap every PositionEmbeddingSine._encode_xy to honor input dtype.

    sam3's `PositionEmbeddingSine._encode_xy`
    (sam3/model/position_encoding.py:60-77) builds its frequency table as
    `dim_t = torch.arange(..., dtype=torch.float32, ...)` regardless of the
    input dtype. Downstream broadcasts produce fp32 output, which then feeds a
    bf16-weight `points_pos_enc_project` Linear in
    `PointGeometryEncoder._encode_points` (sam3/model/geometry_encoders.py:623)
    and raises `RuntimeError: mat1 and mat2 must have the same dtype` on Colab
    T4 with `ModelConfig(dtype="bfloat16")`. This is true even for zero-length
    point sequences because `F.linear` validates dtypes regardless of seq len.

    We wrap each `_encode_xy` method to cast its (pos_x, pos_y) outputs to the
    dtype of the input `x` tensor BEFORE returning. The bound method is
    replaced via `MethodType` on each PositionEmbeddingSine instance so the
    patch persists across forward calls and survives `.to(device)` /
    `.to(dtype)` (only parameters move; methods do not).

    This is a localized stop-gap. The right long-term fix is upstream in
    sam3's pos-enc to honor input dtype directly (tracked as a follow-up
    in logs/TODO.md). Re-evaluate every sam3 upgrade.

    Notes:
    - We use a per-instance MethodType replacement (NOT class-level monkey-patch)
      to avoid affecting other consumers of sam3 in the same process.
    - We do NOT introduce any `torch.autocast` scope; doing so re-triggered the
      bf16-vs-fp32 collision inside `sam3/model/decoder.py::forward_ffn`'s
      `with torch.amp.autocast(enabled=False)` region during PR #13's v2 work.
      The cast-on-return approach side-steps that entirely.
    """
    from types import MethodType

    from sam3.model.position_encoding import PositionEmbeddingSine

    patched_count = 0
    for submodule in model.modules():
        if not isinstance(submodule, PositionEmbeddingSine):
            continue
        if getattr(submodule, "_esam3_pos_enc_dtype_patched", False):
            continue
        original = submodule._encode_xy

        def _encode_xy_dtype_aware(self, x, y, _orig=original):  # type: ignore[no-untyped-def]
            pos_x, pos_y = _orig(x, y)
            return pos_x.to(dtype=x.dtype), pos_y.to(dtype=x.dtype)

        submodule._encode_xy = MethodType(_encode_xy_dtype_aware, submodule)
        submodule._esam3_pos_enc_dtype_patched = True  # idempotency marker
        patched_count += 1

    logger.info("Patched %d PositionEmbeddingSine._encode_xy callsites for dtype awareness.", patched_count)
```

Key design choices:
- We capture `original` via `_orig=original` in a default arg so the closure is correctly bound per-instance (Python's late-binding gotcha avoidance).
- We mark each patched instance with `_esam3_pos_enc_dtype_patched = True` so repeated `load_sam31` calls (e.g., in tests) don't double-wrap.
- We rebind the bound method on the INSTANCE, not the class — keeps other consumers of `sam3` in the process clean.

- [ ] **Step 3: Call the helper from `load_sam31`.**

In `load_sam31`, AFTER the `cfg.dtype` cast block (the `if cfg.dtype == "bfloat16": raw_model = raw_model.to(...)` block at lines ~284-287) and BEFORE `adapter = _Sam3ImageAdapter(raw_model, image_size=1008)`, insert:

```python
    # Cast PositionEmbeddingSine._encode_xy outputs to input dtype to avoid
    # fp32 inputs feeding bf16 Linear weights in the geometry encoder.
    # See _patch_pos_enc_dtype for full rationale.
    _patch_pos_enc_dtype(raw_model)
```

- [ ] **Step 4: Add the CPU unit smoke test.**

Create `tests/unit/test_sam3_pos_enc_patch.py`:

```python
"""CPU unit test for the PositionEmbeddingSine._encode_xy dtype-cast patch.

The patch lives in src/esam3/models/sam3.py::_patch_pos_enc_dtype. It rebinds
the bound method on each PositionEmbeddingSine instance so its (pos_x, pos_y)
outputs are cast to the input tensor's dtype before returning.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from sam3.model.position_encoding import PositionEmbeddingSine

from esam3.models.sam3 import _patch_pos_enc_dtype


def test_pos_enc_patch_casts_outputs_to_input_dtype() -> None:
    pos_enc = PositionEmbeddingSine(num_pos_feats=8)

    # Pre-patch: outputs are fp32 regardless of input dtype.
    x = torch.randn(3, dtype=torch.bfloat16)
    y = torch.randn(3, dtype=torch.bfloat16)
    px_pre, py_pre = pos_enc._encode_xy(x, y)
    assert px_pre.dtype == torch.float32
    assert py_pre.dtype == torch.float32

    # Apply the patch via a wrapping nn.Module.
    holder = nn.Sequential(pos_enc)
    _patch_pos_enc_dtype(holder)

    # Post-patch: outputs honor the input dtype.
    px_post, py_post = pos_enc._encode_xy(x, y)
    assert px_post.dtype == torch.bfloat16, f"expected bf16, got {px_post.dtype}"
    assert py_post.dtype == torch.bfloat16, f"expected bf16, got {py_post.dtype}"


def test_pos_enc_patch_is_idempotent() -> None:
    pos_enc = PositionEmbeddingSine(num_pos_feats=8)
    holder = nn.Sequential(pos_enc)
    _patch_pos_enc_dtype(holder)
    first_bound = pos_enc._encode_xy
    _patch_pos_enc_dtype(holder)  # second call: must be a no-op
    second_bound = pos_enc._encode_xy
    assert first_bound is second_bound, (
        "second _patch_pos_enc_dtype call must not re-wrap an already-patched instance"
    )


@pytest.mark.parametrize("input_dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_pos_enc_patch_preserves_numerical_content_for_fp32(
    input_dtype: torch.dtype,
) -> None:
    """For fp32 input, post-patch output must equal pre-patch output bitwise.

    For lower-precision dtypes we only check the dtype is honored
    (numerical content differs by the bf16/fp16 truncation, which is the
    intended behavior).
    """
    pos_enc = PositionEmbeddingSine(num_pos_feats=8)
    x = torch.tensor([0.1, 0.5, 0.9], dtype=input_dtype)
    y = torch.tensor([0.2, 0.6, 0.8], dtype=input_dtype)
    px_pre, py_pre = pos_enc._encode_xy(x, y)

    holder = nn.Sequential(pos_enc)
    _patch_pos_enc_dtype(holder)
    px_post, py_post = pos_enc._encode_xy(x, y)

    assert px_post.dtype == input_dtype
    assert py_post.dtype == input_dtype
    if input_dtype == torch.float32:
        assert torch.equal(px_pre, px_post)
        assert torch.equal(py_pre, py_post)
```

- [ ] **Step 5: Lint and format.**

```bash
uv run ruff check src/esam3/models/sam3.py tests/unit/test_sam3_pos_enc_patch.py
uv run ruff format --check src/esam3/models/sam3.py tests/unit/test_sam3_pos_enc_patch.py
```

The most likely lint complaint is on the `def _encode_xy_dtype_aware(self, x, y, _orig=original)` line (B008 mutable default — but `original` is a method, not a mutable). If ruff flags it, add `# noqa: B008` or refactor to `functools.partial`. Prefer keeping the closure form for readability.

- [ ] **Step 6: Run unit tests.**

```bash
uv run pytest tests/unit/test_sam3_pos_enc_patch.py -v --no-cov 2>&1 | tail -15
uv run pytest tests/unit -q --no-cov 2>&1 | tail -3
```

Expected: 5 new tests pass (1 + 1 + 3 from parametrize); total `250 passed, 1 skipped`.

- [ ] **Step 7: Append to `logs/log.md`.**

```
[<UTC-ISO8601>] [implementer] task-4 remaining-failures: monkey-patched PositionEmbeddingSine._encode_xy to cast outputs to input dtype in load_sam31 (Option C from spec §3.4); avoids autocast which re-triggers v2 decoder.forward_ffn collision; CPU unit tests cover patch + idempotency
```

Then append to `logs/TODO.md`:

```
[<UTC-ISO8601>] [planner] colab-remaining-failures: upstream PR to Meta sam3 to make PositionEmbeddingSine._encode_xy honor input dtype natively (Option D from spec §3.4). Until merged and a sam3 release lands, the monkey-patch in src/esam3/models/sam3.py::_patch_pos_enc_dtype remains the stop-gap. Re-evaluate on every sam3 version bump.
```

- [ ] **Step 8: Commit.**

```bash
git add src/esam3/models/sam3.py tests/unit/test_sam3_pos_enc_patch.py logs/log.md
git commit -m "$(cat <<'EOF'
fix(models): patch sam3 PositionEmbeddingSine._encode_xy to honor input dtype

sam3's `PositionEmbeddingSine._encode_xy`
(sam3/model/position_encoding.py:60-77) builds its frequency table as
`dim_t = torch.arange(..., dtype=torch.float32, ...)` regardless of input
dtype. Downstream broadcasts produce fp32 output, which then feeds a bf16
`points_pos_enc_project` Linear in `PointGeometryEncoder._encode_points`
(sam3/model/geometry_encoders.py:623) and raises `RuntimeError: mat1 and
mat2 must have the same dtype` on Colab T4 with `ModelConfig(dtype="bfloat16")`.
True even for zero-length point sequences because F.linear validates dtype
regardless of seq len.

Add `_patch_pos_enc_dtype(raw_model)` to load_sam31 that walks every
PositionEmbeddingSine instance reachable from the loaded model and rebinds
`_encode_xy` (via MethodType on the instance, not the class) to cast outputs
to the input's dtype. Marks each patched instance with
`_esam3_pos_enc_dtype_patched = True` for idempotency.

Why monkey-patch and not autocast: PR #13 commit 68f7c19 removed an autocast
wrap because the broad scope re-burned the bf16-vs-fp32 collision inside
`sam3/model/decoder.py::forward_ffn`'s `with torch.amp.autocast(enabled=False)`
region. This cast-on-return approach introduces zero new autocast scopes and
cannot re-trigger that collision.

CPU unit tests verify the patch casts to bf16/fp16/fp32 correctly, preserves
fp32 numerical content bitwise, and is idempotent across repeated calls.

Stop-gap. Tracked in logs/TODO.md for an upstream sam3 PR.

Unblocks:
- tests/integration/test_load_sam31_real.py::test_load_sam31_forward_to_canonical
EOF
)"
```

### Definition of Done

- [ ] `src/esam3/models/sam3.py` contains a `_patch_pos_enc_dtype` helper.
- [ ] `load_sam31` calls `_patch_pos_enc_dtype(raw_model)` AFTER the dtype cast and BEFORE constructing the adapter.
- [ ] `src/esam3/models/sam3.py` contains ZERO `torch.autocast` calls (verified by `grep -c "torch.autocast" src/esam3/models/sam3.py` returning 0).
- [ ] `tests/unit/test_sam3_pos_enc_patch.py` contains the 3 test functions (the parametrize counts as 1; total assertions across 5 calls).
- [ ] All new tests pass.
- [ ] `uv run pytest tests/unit -q --no-cov` reports `250 passed, 1 skipped` (or +5 from Task 3's count).
- [ ] `ruff check` / `ruff format --check` pass for both changed files.
- [ ] `logs/log.md` has the Task-4 entry.
- [ ] `logs/TODO.md` has the upstream-PR follow-up entry.
- [ ] One new commit on branch tip.

### Verification (commands)

```bash
grep -n "_patch_pos_enc_dtype\|_esam3_pos_enc_dtype_patched" src/esam3/models/sam3.py
# Expected: at least 4 matches (function def + call + marker assignment + marker check).
grep -c "torch.autocast" src/esam3/models/sam3.py
# Expected: 0 (CRITICAL — we must NOT introduce any autocast).
uv run pytest tests/unit/test_sam3_pos_enc_patch.py -v --no-cov 2>&1 | tail -10
uv run pytest tests/unit -q --no-cov 2>&1 | tail -3
uv run ruff check src/esam3/models/sam3.py tests/unit/test_sam3_pos_enc_patch.py
uv run ruff format --check src/esam3/models/sam3.py tests/unit/test_sam3_pos_enc_patch.py
```

### Rollback

```bash
git reset --hard HEAD~1
```

If the Colab T4 run after Task 5 still raises the dtype mismatch:
- Diagnose: the patch may not be visible to a re-loaded sam3 instance. Ask the user to add a print: `print(getattr(raw_model_some_pos_enc, "_esam3_pos_enc_dtype_patched", "NOT-SET"))` after `load_sam31`. If `NOT-SET`, the walker missed it — investigate the module hierarchy.
- Fallback: switch to Option (B) (run the integration test with `dtype="float32"`) and accept the test-only memory hit. STOP and ask the user before applying.

---

## Task 5: Push branch, open PR, trigger Colab verification

**Difficulty:** L
**Subagent:** Main thread (push + ask user; no subagent needed).

**Files:** None modified.

### Scope

Push the 4-commit branch to remote, open a draft PR, and request the user run `notebooks/colab_gpu_tests.ipynb` end-to-end on Colab T4. Record the result.

### Steps

- [ ] **Step 1: Sanity-check the branch state.**

```bash
git -C /home/justin/projects/Efficient-SAM3-Finetuning/.claude/worktrees/fix+colab-remaining-failures status
# Expected: clean.

git -C /home/justin/projects/Efficient-SAM3-Finetuning/.claude/worktrees/fix+colab-remaining-failures log --oneline origin/main..HEAD
# Expected (top-down):
#   <SHA> fix(models): patch sam3 PositionEmbeddingSine._encode_xy to honor input dtype
#   <SHA> fix(qlora): set is_loaded_in_4bit so peft dispatches to bnb.Linear4bit merge
#   <SHA> test(qlora): tighten _has_plain_nn_linear to skip LoRA adapter Linears
#   <SHA> fix(qlora): read quant_type from Params4bit weight, not Linear4bit module
# (4 commits, one per task; if the branch has more, investigate before pushing.)

uv run pytest tests/unit -q --no-cov 2>&1 | tail -3
# Expected: 250 passed, 1 skipped (or pre-plan-baseline + 10).
uv run ruff check src tests
uv run ruff format --check src tests
# Expected: both pass.
```

If anything is dirty or counts are wrong, STOP.

- [ ] **Step 2: Push.**

```bash
git push -u origin HEAD
```

The branch is new (cut from `main`), so plain `git push -u origin HEAD` works. NO force flags.

- [ ] **Step 3: Open a draft PR.**

```bash
gh pr create --draft --base main --title "fix(colab): close remaining 5 GPU integration failures" --body "$(cat <<'EOF'
## Summary

Take the Colab T4 GPU integration suite from 4 of 9 passing (post-PR #13) to
9 of 9. Four independent fixes, one per failing-test category.

- **Issue 1 (sam3 fp32 leak)**: monkey-patch
  `PositionEmbeddingSine._encode_xy` in `load_sam31` to cast outputs to input
  dtype. Avoids autocast (which re-triggers the v2 `decoder.forward_ffn`
  collision). Unblocks `test_load_sam31_forward_to_canonical`.
- **Issue 2 (LoRA adapter Linears mis-flagged)**: tighten
  `_has_plain_nn_linear` in the integration test to skip `lora_A` / `lora_B`
  adapter paths. Unblocks
  `test_apply_qlora_swaps_every_linear_and_attaches_lora`.
- **Issue 3 (`quant_type` moved off Linear4bit)**: read
  `module.weight.quant_type` (primary) with `module.quant_type` fallback in
  `_infer_quant_type_from_wrapper`. Unblocks
  `test_save_qlora_writes_adapter_and_metadata` and
  `test_save_load_qlora_roundtrip`.
- **Issue 4 (peft dispatch picked wrong wrapper for Linear4bit)**: set
  `base.is_loaded_in_4bit = True` in `apply_qlora` / `load_qlora` so peft
  dispatches to `peft.tuners.lora.bnb.Linear4bit` (correct merge) instead of
  the generic `lora.layer.Linear` (which blew up on the packed 4-bit weight
  shape). Test assertion relaxed because peft re-packs the merged weight as
  `Params4bit` (quantization preserved). Unblocks
  `test_merge_lora_unloads_qlora_wrapper`.

`pyproject.toml`, `notebooks/colab_gpu_tests.ipynb`, and
`src/esam3/peft_adapters/lora.py` are byte-identical to their pre-plan
state. No new runtime dependencies.

Local unit suite: 250 passed / 1 skipped (was 240 / 1; +10 new CPU unit tests
covering the fallback chain, predicate, flag-set, and pos-enc patch).

Reference plan: `docs/superpowers/plans/2026-05-17-colab-gpu-remaining-failures.md`.
Reference spec: `docs/superpowers/specs/2026-05-17-colab-gpu-remaining-failures-design.md`.

## Test plan

- [ ] Local: `uv run pytest tests/unit -q --no-cov` reports `250 passed, 1 skipped`.
- [ ] Local: `uv run ruff check src tests` and `uv run ruff format --check src tests` both pass.
- [ ] Colab T4: `bash scripts/run_gpu_tests.sh` reports `9 passed` for `requires_compatible_gpu and requires_checkpoint` markers.
- [ ] PR diff: `pyproject.toml`, `notebooks/colab_gpu_tests.ipynb`, and `src/esam3/peft_adapters/lora.py` show ZERO changes.
EOF
)"
```

- [ ] **Step 4: Append to `logs/log.md` and push the final log commit.**

```
[<UTC-ISO8601>] [implementer] task-5 remaining-failures: pushed branch and opened draft PR; awaiting Colab T4 verification
```

```bash
git add logs/log.md
git commit -m "chore(logs): record colab-remaining-failures push"
git push
```

- [ ] **Step 5: Notify the user with Colab instructions.**

> Open `notebooks/colab_gpu_tests.ipynb` on Colab T4.
>
> - Runtime → Change runtime type → **T4 GPU**.
> - In Cell 1, set `BRANCH = "<this-branch-name>"`.
> - Runtime → Restart session.
> - Runtime → Run all.
> - When the final `bash scripts/run_gpu_tests.sh` cell finishes, copy the last pytest summary line and paste it back.
>
> If the suite reports fewer than 9 passes, ALSO paste:
> - The full last 40 lines of the failing test's traceback.
> - The output of `!pip show bitsandbytes peft torchao` (or run it as a new cell).
> - For Issue 4 specifically: run the diagnostic snippet at the end of Task 3's "Rollback" section to confirm peft is dispatching to `bnb.Linear4bit`.

### Definition of Done

- [ ] Branch pushed to origin.
- [ ] Draft PR opened against `main`.
- [ ] PR description includes the 4 unblocked test names.
- [ ] User has been notified with Colab instructions and the diagnostic snippets.
- [ ] `logs/log.md` has the Task-5 entry.

### Rollback

The branch is on a draft PR, isolated from `main`. If Colab fails, fix forward (commit, push). Do NOT close the PR until the cause is understood.

---

## Verification matrix (which task fixes which test)

| Failing test | Owning task | Fix mechanism |
| --- | --- | --- |
| `test_load_sam31_real.py::test_load_sam31_forward_to_canonical` | Task 4 | Monkey-patch `_encode_xy` dtype cast in `load_sam31`. |
| `test_peft_qlora_real.py::test_apply_qlora_swaps_every_linear_and_attaches_lora` | Task 2 | Predicate skips `lora_A` / `lora_B` adapter paths. |
| `test_peft_qlora_real.py::test_save_qlora_writes_adapter_and_metadata` | Task 1 | Read `module.weight.quant_type` (primary) with legacy fallback. |
| `test_peft_qlora_real.py::test_save_load_qlora_roundtrip` | Task 1 | Same as above. |
| `test_peft_qlora_real.py::test_merge_lora_unloads_qlora_wrapper` (renamed) | Task 3 | Set `is_loaded_in_4bit=True` so peft dispatches to `bnb.Linear4bit.merge`; test asserts repacked Params4bit. |

| Passing test (must stay green) | Risk source | Mitigation |
| --- | --- | --- |
| `test_load_sam31_real.py::test_load_sam31_returns_wrapper` | Task 4 (touches `load_sam31`). | Patch helper is additive; if model loads pre-patch, it loads post-patch. CPU smoke test verifies. |
| `test_peft_lora_real.py::test_apply_lora_on_real_sam31_under_trainable_budget` | None — Task 3 sets a flag only on the QLoRA path; LoRA path doesn't call apply_qlora. | Verification: confirm `is_loaded_in_4bit` is only set in `qlora.py`, never in `lora.py`. |
| `test_peft_lora_real.py::test_save_load_roundtrip_on_real_sam31` | None — Task 3 doesn't touch `lora.py`. | Same. |
| `test_peft_lora_real.py::test_merge_lora_on_real_sam31` | None — `merge_lora` body unchanged; for non-4bit `peft_model.merge_and_unload()` still delegates correctly. | Same; explicitly assert `git diff` for `lora.py` is empty in Task 3 DoD. |

---

## Out of scope

The following are explicitly NOT in this plan. Each comes with the reason for deferral.

| Item | Why deferred |
| --- | --- |
| Re-pin or change any of v2's Colab install-cell pins (torchao>=0.16.0, numpy==1.26.4, scipy==1.13.1, transformers==5.0.0, huggingface_hub>=1.15). | Already correct on the PR #13 baseline; touching them risks Colab resolver churn. Spec §2.2 forbids notebook changes. |
| Modify `pyproject.toml`. | No new runtime dependencies; spec §2.1 forbids. |
| Touch the 4 already-passing Colab tests (`test_load_sam31_returns_wrapper`, `test_apply_lora_on_real_sam31_under_trainable_budget`, `test_save_load_roundtrip_on_real_sam31`, `test_merge_lora_on_real_sam31`). | Not failing; touching them risks regressing the green slice. Spec §2.2. |
| Refactor `Sam3Wrapper` / `_Sam3ImageAdapter` architecture (merge them, expose a non-adapter forward, plumb `image_size` via a registry). | Current shape works; refactor when a third caller demands a new shape. Spec §9. |
| Upstream a sam3 PR fixing `_encode_xy` to honor input dtype (Option D from spec §3.4). | Tracked in `logs/TODO.md` as a long-tail follow-up; not blocking the Colab green-suite goal on Meta's review timeline. |
| Numerical equivalence tests for bf16 vs fp32 forward. | Spec §9 (v2 spec §7.3 also deferred this). |
| Supporting LoRA-on-non-4bit and LoRA-on-4bit mixed merge in `merge_lora`. | QLoRA path is uniformly Linear4bit; mixed graphs are theoretical. Spec §9. |
| Re-introducing any `torch.autocast` scope inside `_Sam3ImageAdapter.forward`. | Re-triggers v2's `decoder.forward_ffn` bf16-vs-fp32 collision; PR #13 commit 68f7c19 removed it for that reason. Spec §3.5. |
| Box-prompt path through `_Sam3ImageAdapter`. | v1 spec §8; still deferred. |
| Multi-class-per-batch forward. | v1 spec §8; still deferred. |
| Pinning a peft version (e.g. `peft<0.19`) as a workaround. | Spec §9; we're staying on peft 0.19+. |
| Replacing peft's `merge_and_unload` for all LoRA cases. | Task 3's fix routes via peft's correct path; no replacement needed. Spec §9. |

---

## Final acceptance

A correct implementation satisfies:

1. New branch cut from `main` AFTER PR #13's merge tip. `git log --oneline origin/main..HEAD` shows 4 source-fix commits (Tasks 1-4) + 1 log-only commit (Task 5).
2. All 5 previously-failing Colab integration tests pass on T4.
3. All 4 previously-passing Colab integration tests still pass.
4. `uv run pytest tests/unit -q --no-cov` reports at least 250 passed, 1 skipped (the pre-plan baseline 240 + 10 new CPU unit tests added by Tasks 1, 2, 3, 4). ZERO regressions.
5. `ruff check src tests` and `ruff format --check src tests` both pass.
6. `pyproject.toml` is byte-identical to its pre-plan state.
7. `notebooks/colab_gpu_tests.ipynb` is byte-identical to its pre-plan state.
8. `src/esam3/peft_adapters/lora.py` is byte-identical to its pre-plan state.
9. `src/esam3/models/sam3.py` contains ZERO `torch.autocast` calls.
10. `logs/log.md` has 5 new entries (Tasks 1-5). `logs/TODO.md` has the upstream-sam3 follow-up entry from Task 4.
11. No emojis anywhere in the diff.
12. Draft PR open against `main` with the body described in Task 5.

---

## Rollback (whole-plan)

If the Colab T4 run after Task 5 still reports < 9 passes and the per-task rollback paths in Tasks 1-4 do not isolate the root cause:

1. Do NOT revert any commits without user approval.
2. Capture the failing trace(s) and rerun the Task 3 dispatcher-diagnostic snippet to confirm peft routing.
3. If Issue 1 is still raising the same dtype error AND `_esam3_pos_enc_dtype_patched` is reported as set on the affected `PositionEmbeddingSine`, fall back to Option (B) from spec §3.4: change `tests/integration/test_load_sam31_real.py::test_load_sam31_forward_to_canonical` to use `ModelConfig(dtype="float32")` AND lower the test image size from `(1, 3, 1008, 1008)` to fit the T4 VRAM budget (start with 512x512; document the deviation in a code comment and a `logs/TODO.md` entry). STOP before applying — confirm with the user first.
4. If Issue 3 is still raising AttributeError, the bnb on Colab may have moved `quant_type` again. Ask the user for `python -c "import bitsandbytes; m = bnb.nn.Linear4bit(4, 4); print(dir(m), dir(m.weight))"` output and read the actual attribute layout before adjusting.
5. For Issue 4, the dispatcher diagnostic in Task 3 Rollback is decisive. If `bnb_wrapper` is 0 after the flag-set, the flag is being lost (perhaps `prepare_model_for_kbit_training` returns a NEW module without the attribute) — verify by inspecting the returned object's `is_loaded_in_4bit` before the `get_peft_model` call. If lost, set the flag on the return value too.

---

## Spec amendments needed

These are items the planner identified during the read-through that the spec did not anticipate. They are NOT applied here (spec is frozen for this session); the user should fold them into the spec before the next iteration if any further work is needed.

1. **Spec §5.3 / §5.4 hypothesis correction.** The spec hypothesizes `quant_type` lives on `weight.quant_state.quant_type`. Reading `peft/tuners/lora/bnb.py:582` shows peft itself accesses `target_base_layer.weight.quant_type` directly — the attribute is on the `Params4bit` parameter (`weight`), NOT inside `quant_state`. The planner's Decision Q4 corrects this. The spec should be updated to reflect the actual attribute path.

2. **Spec §6 framing is more complex than necessary.** The spec proposes implementing an explicit dequant-then-merge path inside `merge_lora`. After reading the peft 0.19 source, the root cause is the dispatcher flag (`is_loaded_in_4bit`), not a missing peft feature: `peft.tuners.lora.bnb.Linear4bit.merge()` (peft bnb.py:351) already correctly handles dequant-merge-repack. Setting the flag is a 2-line fix that routes peft to its own correct implementation. The spec should be amended to document this dispatcher gotcha and shrink §6.4 to the upstream fix.

3. **Spec §6.7 acceptance contradiction.** The spec asserts that after `merge_lora` on a QLoRA wrapper, "no `bnb.nn.Linear4bit` modules remain". This contradicts peft's documented design: `lora.bnb.Linear4bit.merge()` re-packs the result as `Params4bit` (peft bnb.py:398), so Linear4bit modules MUST remain after merge. The planner's Decision Q6 relaxes this. The spec should be amended to express the correct post-merge invariant: structural correctness (`peft_model is None`) + quantization preserved, NOT dequantization.

4. **Spec §6.7 "merged_module(x) vs pre_merge_wrapper(x)" CPU test.** This is hard to set up without a real bnb (the local dev box doesn't have it). The planner did NOT include this test; it would require a faked Linear4bit with a faked dequant. Recommendation: defer to a separate ticket dedicated to QLoRA numerical-equivalence testing on a Turing+ box (GPU CI). The current 250-test CPU baseline is strong enough; the Colab integration test catches structural breakage.

5. **PR #13's autocast removal not reflected in the spec.** The spec §3.3 describes the v2 autocast wrap as still present. The actual PR #13 history shows commit `68f7c19` removed the autocast wrap entirely and replaced it with explicit `model_dtype` tensor construction in the empty-Prompt fallback. This is critical context for Issue 1: the spec implies the autocast is in place; the reality is that PR #13 abandoned the autocast approach. The planner's Decision Q1 (Option C) is the only safe path given this history. The spec should be amended to reflect the actual post-PR #13 state.
