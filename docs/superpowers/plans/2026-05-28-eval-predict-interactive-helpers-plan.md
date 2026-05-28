# Per-command interactive helpers for `eval` / `predict` + PEFT-from-checkpoint Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split the interactive CLI surface into per-command helpers — extract shared prompt machinery into `_interactive.py`, shrink `init -i` to `train|run`, add `eval -i` (reuse|baseline) and `predict -i` (command builder), and centralize PEFT-method discovery into one `peft_adapters` seam so eval/predict/wizard infer the method from the checkpoint instead of `cfg.peft.method`.

**Architecture:** Four bundled, interdependent workstreams in one PR. The `peft_adapters` discovery seam (§7) lands first as the foundation, then the `_interactive.py` extraction (§3) which all helpers depend on, then the `init -i` shrink (§4), the eval-runner baseline + sentinel-dispatch changes (§6) and `eval -i` helper (§5), and the `predict -i` helper (§8). Tests are CPU-only, driving prompts by monkeypatching the moved prompt primitives.

**Tech Stack:** Python 3, Typer + rich (CLI/prompts), Pydantic v2 (schema/validation), `string.Template` (config render), `shlex` (shell-safe command assembly), pytest + pytest-cov (TDD, 80% gate), ruff + mypy + markdownlint-cli2 + yamllint (CI gates).

---

## Sequencing rationale (read before starting)

1. **Seam first (§7).** `discover_method_from_checkpoint` + the relocated `read_adapter_base_model_name` in `peft_adapters/__init__.py` are foundational: the eval-runner change (§6), `_interactive.peek_adapter` (§3), the `predict/adapter_load.py` delegators (§7.3), and `train/checkpoint.py::load_adapter` (§7.4) all call into them. Build Phase 1 before anything that imports the seam.
2. **`_interactive.py` extraction second (§3).** The shared prompt primitives, driver, steps, validators, `require_tty`, and `peek_adapter` are imported by the `init -i` shrink (§4), `eval -i` (§5), and `predict -i` (§8). Phase 2 lands the module + re-exports before the helpers.
3. **`setup_wizard.py` is shared by §3 and §4.** Phase 2 (extraction) and Phase 3 (`init -i` shrink) both edit `setup_wizard.py` — serialize them (Phase 3 after Phase 2).
4. **eval-runner (§6) depends on the seam (§7) only.** Phase 4 lands once Phase 1 is in; it is file-disjoint from `predict -i` (§8).
5. **`eval -i` helper (§5) depends on `_interactive` (§3) + the seam (§7) + the eval-runner change (§6)** because its baseline path reuses `setup_wizard.render`/`validate`/`emit` and its reuse path calls `peek_adapter`. Phase 5 after Phases 1–4.
6. **`predict -i` helper (§8) depends on `_interactive` (§3) + the seam (§7)** only — file-disjoint from eval work once those land. Phase 6 may run in parallel with Phase 5 (different test/helper files, both append to `_interactive.py` — serialize the two `_interactive.py`-appending tasks if dispatched together).
7. **Parallelization is called out per phase.** Within a phase, file-disjoint dependency-free tasks may run in parallel; same-file or chained tasks are serialized.

### Breaking-change note (NOT a migration step — spec §11)

This PR is one clean breaking change plus additive/compatible changes, **no shim, no migration**:

- `init --interactive` no longer offers the `eval` run mode (the `run_mode` prompt offers only `train`/`run`). Shipped only in commit `84bc83f`; nothing in-tree depends on it; replacement is `csp eval --interactive`.
- `csp eval --checkpoint` becomes optional (additive — omitting it selects the baseline path).
- eval infers the PEFT method from the checkpoint sentinel, not `cfg.peft.method` (compatible — a previously mis-dispatching config now loads correctly with an advisory warning).
- `read_adapter_base_model_name` relocates to `peft_adapters` but stays exported from `predict/adapter_load.py` as a delegator (compatible).

These are consequences to document, not tasks to mitigate.

---

## Resolved ambiguities

These are minimal decisions made where the spec left a detail open; none re-litigate the locked design.

1. **Helper module home.** The spec (§2 import-graph note) lets the implementer place `run_eval_interactive` / `run_predict_interactive` in `_interactive.py` or a thin per-command helper module, but "the spec assumes the helpers live in `_interactive.py`." This plan puts both helpers in `_interactive.py` (single seam, fewer files).
2. **Baseline `run.name`.** §5.3 says emit a fixed `baseline-eval`. The plan injects `{"run": {"name": "baseline-eval"}}` into the answers dict before `render`, so `setup_wizard.render` substitutes it via `$run_name`.
3. **Baseline printed split.** §5.3 says always print `--split val` for the baseline. The plan hardcodes `val` in the baseline printed command.
4. **`predict -i` thin-config path name.** §8.1 lets the helper choose a name; the plan uses `predict-config.yaml` in the cwd (overwrite-refused without confirmation).
5. **`predict -i` model-name prompt.** §8.1 makes an optional model-name prompt the implementer's choice; the plan does **not** add one (model.name differs only via the adapter's `base_model_name_or_path`, which `_resolve_config` reads at run time — no thin config needed for it). The thin config is written only when channels ≠ 3 or semantics ≠ rgb.
6. **`Ctx` import in `setup_wizard`.** Since `Ctx`/`RunMode`/`WizardStep` move to `_interactive`, `setup_wizard` re-exports them (`from custom_sam_peft.cli._interactive import Ctx, RunMode, WizardStep, ask_text, ...`) so the existing `tests/unit/cli/test_setup_wizard.py` import surface (`sw.Ctx`, `sw.ask_text`, etc.) stays stable.
7. **`csp eval --config` becomes optional.** The spec (§6.1) only names `--checkpoint` as becoming optional, but adding the `-i` flag (§5) requires the early interactive branch to run before Typer enforces a required `--config` (interactive mode prompts for the config path itself). The plan makes `--config` optional too and adds an explicit `if config is None: raise BadParameter("--config is required")` guard on the non-interactive path, preserving the non-interactive contract. The same pattern is NOT needed for `csp predict` (Task 13 keeps `--images`/`--prompts`/`--output` required and the tests pass dummy values), per §1's "no change to the predict flag surface."

---

## File structure

**New files:**

- `src/custom_sam_peft/cli/_interactive.py` — shared prompt primitives, `WizardStep`/`Ctx`/`RunMode`/`_deep_merge`/`run_wizard`, reusable steps (`dataset_source`, `validation`, `model_weights`), `require_tty`, `peek_adapter`, validators (`validate_checkpoint_dir`, `validate_config_with_eval_split`), emit/launch helpers, and the `run_eval_interactive` / `run_predict_interactive` helpers.
- `tests/unit/cli/test_interactive.py` — `_interactive` primitives/validators/peek/require_tty tests.
- `tests/unit/cli/test_eval_interactive.py` — `eval -i` helper tests.
- `tests/unit/cli/test_predict_interactive.py` — `predict -i` helper tests.

**Modified files:**

- `src/custom_sam_peft/peft_adapters/__init__.py` — add `discover_method_from_checkpoint`; relocate `read_adapter_base_model_name`.
- `src/custom_sam_peft/predict/adapter_load.py` — `detect_adapter_kind` + `read_adapter_base_model_name` become thin delegators.
- `src/custom_sam_peft/train/checkpoint.py` — `load_adapter` uses the canonical discovery function.
- `src/custom_sam_peft/cli/setup_wizard.py` — import moved symbols from `_interactive`; shrink `RunMode` selector + un-gate steps.
- `src/custom_sam_peft/cli/eval_cmd.py` — `--checkpoint` optional; `--config` optional (with a non-interactive presence guard — see Resolved ambiguities #7); add `--interactive`/`-i`.
- `src/custom_sam_peft/eval/runner.py` — baseline path, sentinel dispatch, advisory warning, output-dir fallback.
- `src/custom_sam_peft/cli/predict_cmd.py` — add `--interactive`/`-i`.

**Modified tests:**

- `tests/unit/cli/test_setup_wizard.py` — retarget to 2-mode `RunMode`.
- `tests/predict/test_adapter_detect.py` — seam-delegation cases.
- `tests/unit/test_train_checkpoint.py` — `load_adapter` discover-dispatch case.
- `tests/unit/test_eval_runner.py` — baseline + PEFT-inference + sentinel-dispatch retarget.

---

## Phase 1 — PEFT seam centralization (§7)

> Foundational. Tasks 1, 2, 3, 4 share no source files with each other EXCEPT that Tasks 2/3/4 all import the Task 1 function — so Task 1 lands first, then Tasks 2 (predict delegators), 3 (train checkpoint), 4 (relocate base-model-name) can run in parallel. Task 4 edits the same file as Task 1 (`peft_adapters/__init__.py`) — serialize Task 4 after Task 1.

### Task 1: Add `discover_method_from_checkpoint` to `peft_adapters/__init__.py`

**Spec ref:** §7.1.

**Files:**

- Modify: `src/custom_sam_peft/peft_adapters/__init__.py` (after `method_pretty_name`, line ~161)
- Test: `tests/unit/test_peft_method_protocol.py`

- [ ] **Step 1: Write the failing discovery tests**

Append to `tests/unit/test_peft_method_protocol.py`:

```python
def test_discover_method_lora(tmp_path: Path) -> None:
    from custom_sam_peft.peft_adapters import discover_method_from_checkpoint

    (tmp_path / "adapter_config.json").write_text("{}")
    assert discover_method_from_checkpoint(tmp_path) == "lora"


def test_discover_method_qlora(tmp_path: Path) -> None:
    from custom_sam_peft.peft_adapters import discover_method_from_checkpoint

    (tmp_path / "custom_sam_peft_qlora.json").write_text("{}")
    assert discover_method_from_checkpoint(tmp_path) == "qlora"


def test_discover_does_not_validate_adapter_config(tmp_path: Path) -> None:
    """Discovery only checks the qlora sentinel; it does NOT require adapter_config.json."""
    from custom_sam_peft.peft_adapters import discover_method_from_checkpoint

    (tmp_path / "custom_sam_peft_qlora.json").write_text("{}")
    # no adapter_config.json present
    assert discover_method_from_checkpoint(tmp_path) == "qlora"
```

Ensure `from pathlib import Path` is imported at the top of the test file (add it if absent).

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_peft_method_protocol.py -k discover -v`
Expected: FAIL with `ImportError: cannot import name 'discover_method_from_checkpoint'`.

- [ ] **Step 3: Implement the canonical discovery function**

In `src/custom_sam_peft/peft_adapters/__init__.py`, add immediately after `method_pretty_name` (it must be importable at module scope; `Path` is already imported at line 23):

```python
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

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_peft_method_protocol.py -k discover -v`
Expected: PASS (all three discovery tests green).

- [ ] **Step 5: Commit**

```bash
git add src/custom_sam_peft/peft_adapters/__init__.py tests/unit/test_peft_method_protocol.py
git commit -m "feat(peft): add discover_method_from_checkpoint canonical seam (#172)"
```

### Task 2: `predict/adapter_load.py::detect_adapter_kind` delegates to the seam

**Spec ref:** §7.3.

**Files:**

- Modify: `src/custom_sam_peft/predict/adapter_load.py` (`detect_adapter_kind`, lines 32-44)
- Test: `tests/predict/test_adapter_detect.py`

- [ ] **Step 1: Write the failing delegation test**

Append to `tests/predict/test_adapter_detect.py` (the lora/qlora dir fixtures `_LORA_DIR`/`_QLORA_DIR`/`_BAD_DIR` already exist at the top of the file; reuse them):

```python
def test_detect_adapter_kind_delegates_and_still_validates() -> None:
    """detect_adapter_kind agrees with the canonical seam for lora/qlora dirs
    AND still raises typer.BadParameter on a dir missing adapter_config.json."""
    from custom_sam_peft.peft_adapters import discover_method_from_checkpoint

    assert detect_adapter_kind(_LORA_DIR) == discover_method_from_checkpoint(_LORA_DIR)
    assert detect_adapter_kind(_QLORA_DIR) == discover_method_from_checkpoint(_QLORA_DIR)
    with pytest.raises(typer.BadParameter, match=r"adapter_config\.json"):
        detect_adapter_kind(_BAD_DIR)
```

- [ ] **Step 2: Run the tests to verify they pass-or-fail honestly**

Run: `uv run pytest tests/predict/test_adapter_detect.py -k "detect_adapter_kind" -v`
Expected: the new `test_detect_adapter_kind_delegates_and_still_validates` PASSES even before the refactor (current logic already returns the same kinds and raises on `_BAD_DIR`) — this is a characterization test that must keep passing through the refactor. The existing `test_detect_adapter_kind_lora/qlora/missing` also pass. Confirm green, then refactor in Step 3 and re-run to prove no behavior change.

- [ ] **Step 3: Make `detect_adapter_kind` a thin delegator**

In `src/custom_sam_peft/predict/adapter_load.py`, replace the `detect_adapter_kind` body (lines 32-44) with:

```python
def detect_adapter_kind(checkpoint_dir: Path) -> AdapterKind:
    """Return "qlora" if the QLoRA sentinel file is present, else "lora".

    Delegates kind discovery to the canonical peft_adapters seam, but still
    raises typer.BadParameter if adapter_config.json is absent (i.e. the
    directory does not look like any known adapter checkpoint). The canonical
    discover_method_from_checkpoint does NOT validate adapter_config.json, so
    that check stays here.
    """
    if not (checkpoint_dir / _LORA_CONFIG).is_file() and not (
        checkpoint_dir / _QLORA_SENTINEL
    ).is_file():
        raise typer.BadParameter(
            f"--checkpoint must contain adapter_config.json (checked: {checkpoint_dir})"
        )
    from custom_sam_peft.peft_adapters import discover_method_from_checkpoint

    return cast(AdapterKind, discover_method_from_checkpoint(checkpoint_dir))
```

Add `cast` to the typing import at the top of the file: change `from typing import Literal` to `from typing import Literal, cast`. The `_QLORA_SENTINEL`/`_LORA_CONFIG` constants stay (still used here). Note the missing-config check preserves the original semantics: a dir with the qlora sentinel but no `adapter_config.json` is still a valid qlora checkpoint, so we only raise when BOTH are absent (matching the original `is_file()` fall-through that raised only when neither was present).

- [ ] **Step 4: Run the tests to verify they still pass**

Run: `uv run pytest tests/predict/test_adapter_detect.py -k "detect_adapter_kind" -v`
Expected: PASS — same return values, same `BadParameter` raise; behavior is identical.

- [ ] **Step 5: Commit**

```bash
git add src/custom_sam_peft/predict/adapter_load.py tests/predict/test_adapter_detect.py
git commit -m "refactor(predict): detect_adapter_kind delegates to peft_adapters seam"
```

### Task 3: `train/checkpoint.py::load_adapter` uses the canonical discovery function

**Spec ref:** §7.4.

**Files:**

- Modify: `src/custom_sam_peft/train/checkpoint.py` (`load_adapter`, lines 117-124; `_QLORA_META_FILENAME` constant, line 35)
- Test: `tests/unit/test_train_checkpoint.py`

- [ ] **Step 1: Write the failing dispatch test**

Append to `tests/unit/test_train_checkpoint.py`:

```python
def test_load_adapter_uses_discover_qlora(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """load_adapter dispatches load_qlora when the qlora sentinel is present."""
    from custom_sam_peft.train import checkpoint as ckpt_mod

    (tmp_path / "custom_sam_peft_qlora.json").write_text("{}")
    calls: list[str] = []
    monkeypatch.setattr(ckpt_mod, "load_qlora", lambda w, p: calls.append("qlora"))
    monkeypatch.setattr(ckpt_mod, "load_lora", lambda w, p: calls.append("lora"))
    monkeypatch.setattr(ckpt_mod, "_load_channel_adapter", lambda w, p: None)
    wrapper = MagicMock()
    ckpt_mod.load_adapter(wrapper, tmp_path)
    assert calls == ["qlora"]


def test_load_adapter_uses_discover_lora(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """load_adapter dispatches load_lora when the qlora sentinel is absent."""
    from custom_sam_peft.train import checkpoint as ckpt_mod

    (tmp_path / "adapter_config.json").write_text("{}")
    calls: list[str] = []
    monkeypatch.setattr(ckpt_mod, "load_qlora", lambda w, p: calls.append("qlora"))
    monkeypatch.setattr(ckpt_mod, "load_lora", lambda w, p: calls.append("lora"))
    monkeypatch.setattr(ckpt_mod, "_load_channel_adapter", lambda w, p: None)
    wrapper = MagicMock()
    ckpt_mod.load_adapter(wrapper, tmp_path)
    assert calls == ["lora"]
```

Ensure `from unittest.mock import MagicMock` is imported in the test file (add if absent).

- [ ] **Step 2: Run the tests to verify they pass-or-fail honestly**

Run: `uv run pytest tests/unit/test_train_checkpoint.py -k "load_adapter_uses_discover" -v`
Expected: PASS even before the refactor (the current inline `(path / _QLORA_META_FILENAME).exists()` check already produces this dispatch) — characterization test. Confirm green, refactor in Step 3, re-run to prove no behavior change.

- [ ] **Step 3: Replace the inline sentinel check with the canonical function**

In `src/custom_sam_peft/train/checkpoint.py`, replace `load_adapter` (lines 117-124) with:

```python
def load_adapter(wrapper: Sam3Wrapper, path: Path) -> Sam3Wrapper:
    """LoRA vs QLoRA dispatch via the canonical peft_adapters discovery seam."""
    from custom_sam_peft.peft_adapters import discover_method_from_checkpoint

    if discover_method_from_checkpoint(path) == "qlora":
        load_qlora(wrapper, path)
    else:
        load_lora(wrapper, path)
    _load_channel_adapter(wrapper, path)
    return wrapper
```

The local `_QLORA_META_FILENAME = "custom_sam_peft_qlora.json"` constant at line 35 is now unused by `load_adapter`. Grep to confirm no other use in the file:

Run: `grep -n "_QLORA_META_FILENAME" src/custom_sam_peft/train/checkpoint.py`
Expected: only the definition at line 35 remains (the `load_adapter` use is gone). Delete the line-35 definition to avoid drift (the canonical source of the constant is now `peft_adapters`).

- [ ] **Step 4: Run the tests to verify they still pass**

Run: `uv run pytest tests/unit/test_train_checkpoint.py -v`
Expected: PASS — the new dispatch tests plus all existing checkpoint tests (save/load, channel-adapter, resume mismatch) stay green; behavior is identical.

- [ ] **Step 5: Commit**

```bash
git add src/custom_sam_peft/train/checkpoint.py tests/unit/test_train_checkpoint.py
git commit -m "refactor(train): load_adapter uses discover_method_from_checkpoint seam"
```

### Task 4: Relocate `read_adapter_base_model_name` into `peft_adapters` + delegator

**Spec ref:** §7.2, §7.3.

**Files:**

- Modify: `src/custom_sam_peft/peft_adapters/__init__.py` (add the relocated function)
- Modify: `src/custom_sam_peft/predict/adapter_load.py` (`read_adapter_base_model_name`, lines 87-98)
- Test: `tests/predict/test_adapter_detect.py`, `tests/unit/test_peft_method_protocol.py`

> Edits `peft_adapters/__init__.py` (shared with Task 1) — run after Task 1.

- [ ] **Step 1: Write the failing relocation + delegation tests**

Append to `tests/unit/test_peft_method_protocol.py`:

```python
def test_peft_adapters_read_base_model_name(tmp_path: Path) -> None:
    import json

    from custom_sam_peft.peft_adapters import read_adapter_base_model_name

    (tmp_path / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "facebook/sam3.1"})
    )
    assert read_adapter_base_model_name(tmp_path) == "facebook/sam3.1"


def test_peft_adapters_read_base_model_name_absent(tmp_path: Path) -> None:
    from custom_sam_peft.peft_adapters import read_adapter_base_model_name

    assert read_adapter_base_model_name(tmp_path) is None
```

Append to `tests/predict/test_adapter_detect.py`:

```python
def test_read_base_model_name_delegates() -> None:
    """The predict delegator and the relocated peft_adapters impl agree."""
    from custom_sam_peft.peft_adapters import (
        read_adapter_base_model_name as _impl,
    )

    assert read_adapter_base_model_name(_LORA_DIR) == _impl(_LORA_DIR)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_peft_method_protocol.py -k read_base_model -v`
Expected: FAIL with `ImportError: cannot import name 'read_adapter_base_model_name'` from `peft_adapters`.

- [ ] **Step 3: Move the function into `peft_adapters/__init__.py`**

In `src/custom_sam_peft/peft_adapters/__init__.py`, add `import json` to the top-of-file imports (it currently imports `from pathlib import Path` and typing only). Add the relocated function after `discover_method_from_checkpoint`:

```python
_LORA_CONFIG_FILENAME = "adapter_config.json"


def read_adapter_base_model_name(adapter_dir: Path) -> str | None:
    """Read base_model_name_or_path from adapter_config.json, or return None.

    Returns None if the file is absent or the key is missing. Pure JSON read
    (no torch/bnb), safe to call from any path.
    """
    config_path = adapter_dir / _LORA_CONFIG_FILENAME
    if not config_path.is_file():
        return None
    with config_path.open(encoding="utf-8") as fh:
        data: dict[str, object] = json.load(fh)
    value = data.get("base_model_name_or_path")
    return str(value) if value is not None else None
```

- [ ] **Step 4: Make the predict copy a thin delegator**

In `src/custom_sam_peft/predict/adapter_load.py`, replace `read_adapter_base_model_name` (lines 87-98) with:

```python
def read_adapter_base_model_name(checkpoint_dir: Path) -> str | None:
    """Read base_model_name_or_path from adapter_config.json, or return None.

    Thin delegator to the relocated peft_adapters implementation (spec §7.2).
    Import stays lazy to match this module's import discipline.
    """
    from custom_sam_peft.peft_adapters import (
        read_adapter_base_model_name as _impl,
    )

    return _impl(checkpoint_dir)
```

The `import json` at the top of `adapter_load.py` is now unused (the relocated function owns the JSON read). Grep to confirm:

Run: `grep -n "json\." src/custom_sam_peft/predict/adapter_load.py`
Expected: no matches → remove the `import json` line (line 16). If any match remains, leave the import.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_peft_method_protocol.py tests/predict/test_adapter_detect.py -v`
Expected: PASS — `peft_adapters.read_adapter_base_model_name` and the predict delegator return identical values; existing `tests/predict/test_adapter_detect.py:132-147` base-model cases still green.

- [ ] **Step 6: Commit**

```bash
git add src/custom_sam_peft/peft_adapters/__init__.py src/custom_sam_peft/predict/adapter_load.py tests/unit/test_peft_method_protocol.py tests/predict/test_adapter_detect.py
git commit -m "refactor(peft): relocate read_adapter_base_model_name; predict delegates"
```

---

## REVIEW CHECKPOINT A — seam centralized

- [ ] Run: `uv run pytest tests/unit/test_peft_method_protocol.py tests/predict/test_adapter_detect.py tests/unit/test_train_checkpoint.py -q`
      Expected: all PASS.
- [ ] Run: `! grep -rn "_QLORA_SENTINEL\|_QLORA_META_FILENAME\|adapter_config.json" src/custom_sam_peft/predict/runner.py src/custom_sam_peft/eval/runner.py`
      Expected: no inline sentinel/config checks leaked into the runners yet (they call the seam, not raw constants) — the only inline sentinel constants now live in `peft_adapters/__init__.py` (`_QLORA_META_FILENAME`) and `predict/adapter_load.py` (`_QLORA_SENTINEL`/`_LORA_CONFIG`, used by the validating delegator).
- [ ] Dispatch a code-review subagent (min sonnet/high) over the Phase 1 diff: confirm `discover_method_from_checkpoint` does NOT validate `adapter_config.json`, the `detect_adapter_kind` delegator preserves the missing-config raise, `train.checkpoint.load_adapter` dispatch is unchanged, and the relocated `read_adapter_base_model_name` is byte-identical in behavior.

---

## Phase 2 — `_interactive.py` extraction (§3)

> Single new module + `setup_wizard.py` re-export edit + new test file. One task; all extraction sub-edits serialized (they share `_interactive.py` and `setup_wizard.py`).

### Task 5: Extract shared machinery into `_interactive.py`; re-export from `setup_wizard`

**Spec ref:** §3 (symbols-that-move table), §2 (import graph).

**Files:**

- Create: `src/custom_sam_peft/cli/_interactive.py`
- Modify: `src/custom_sam_peft/cli/setup_wizard.py` (remove moved symbols; import + re-export from `_interactive`)
- Create: `tests/unit/cli/test_interactive.py`

> The moved symbols (`ask_text`, `ask_choice`, `ask_confirm`, `WizardStep`, `Ctx`, `RunMode`, `_deep_merge`, `run_wizard`, the shared step ask-functions `_ask_dataset_source`/`_ask_validation`/`_ask_model_weights`, `validate`, `_LAUNCH_VERB`, `_launch_command`, `_header`) move VERBATIM. `setup_wizard.py` then imports them back so its existing test surface (`sw.ask_text`, `sw.WizardStep`, `sw.Ctx`, `sw.run_wizard`, `sw.validate`, `sw._deep_merge`) keeps working unchanged.

- [ ] **Step 1: Write a failing test that the symbols are importable from `_interactive`**

Create `tests/unit/cli/test_interactive.py`:

```python
"""Tests for the shared interactive module (CPU-only; prompts monkeypatched)."""

from __future__ import annotations

from pathlib import Path

import pytest

from custom_sam_peft.cli import _interactive as itv


def test_prompt_primitives_importable() -> None:
    assert callable(itv.ask_text)
    assert callable(itv.ask_choice)
    assert callable(itv.ask_confirm)
    assert callable(itv.run_wizard)
    assert hasattr(itv, "WizardStep")
    assert hasattr(itv, "Ctx")


def test_ask_choice_reasks_on_invalid(monkeypatch: pytest.MonkeyPatch) -> None:
    answers = iter(["bogus", "coco"])
    monkeypatch.setattr(itv.typer, "prompt", lambda *a, **k: next(answers))
    out: list[str] = []
    monkeypatch.setattr(itv.typer, "echo", lambda msg="", *a, **k: out.append(str(msg)))
    assert itv.ask_choice("Format?", ["coco", "hf"], default="coco") == "coco"
    assert any("choose one of" in line for line in out)


def test_deep_merge_nested() -> None:
    dst = {"data": {"format": "coco"}}
    itv._deep_merge(dst, {"data": {"val_split": {"fraction": 0.1}}})
    assert dst == {"data": {"format": "coco", "val_split": {"fraction": 0.1}}}


def test_shared_steps_return_fragments(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(itv, "ask_choice", lambda *a, **k: "coco")
    answers = iter(["ann.json", "imgs/"])
    monkeypatch.setattr(itv, "ask_text", lambda *a, **k: next(answers))
    ctx = itv.Ctx(answers={}, cuda_available=False)
    frag = itv._ask_dataset_source(ctx)
    assert frag == {"data": {"format": "coco", "train": {"annotations": "ann.json", "images": "imgs/"}}}
```

Also confirm the existing `tests/unit/cli/test_setup_wizard.py` still imports `from custom_sam_peft.cli import setup_wizard as sw` and references `sw.ask_text` / `sw.WizardStep` / `sw.Ctx` / `sw.run_wizard` / `sw.validate` / `sw._deep_merge` — these must keep working after the move via re-export (Step 3).

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/unit/cli/test_interactive.py -v`
Expected: FAIL with `ModuleNotFoundError: custom_sam_peft.cli._interactive`.

- [ ] **Step 3: Create `_interactive.py` with the moved symbols**

Create `src/custom_sam_peft/cli/_interactive.py`. Move these symbols VERBATIM from `setup_wizard.py` (the bodies are unchanged; only the home module changes):

```python
"""Shared interactive-CLI machinery for `init -i`, `eval -i`, and `predict -i`.

Prompt primitives, the WizardStep/Ctx/run_wizard registry-driver, reusable
steps (dataset_source, validation, model_weights), the TTY guard, adapter-peek,
small validators, and the per-command interactive helpers. See
docs/superpowers/specs/2026-05-28-eval-predict-interactive-helpers-design.md.

Import discipline (spec §2): this module imports only typer, stdlib,
config.loader/config.schema, and — LAZILY, inside function bodies — the
peft_adapters seam (§7). It MUST NOT import init_cmd / setup_wizard / eval_cmd /
predict_cmd at module scope (those import it, not vice-versa).
"""

from __future__ import annotations

import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Literal

import typer

from custom_sam_peft.config.loader import load_config

RunMode = Literal["train", "run", "eval"]  # superset; init -i narrows to train|run (§4)


@dataclass
class Ctx:
    answers: dict[str, Any]
    cuda_available: bool
    run_mode: RunMode = "train"
    categories: list[str] | None = None
    category_counts: dict[str, int] | None = None


@dataclass(frozen=True)
class WizardStep:
    id: str
    ask: Callable[[Ctx], dict[str, Any]]
    when: Callable[[Ctx], bool] = field(default=lambda ctx: True)


def _deep_merge(dst: dict[str, Any], src: dict[str, Any]) -> None:
    """Recursively merge src into dst. Nested dicts merge; scalars/lists overwrite."""
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_merge(dst[k], v)
        else:
            dst[k] = v


def ask_text(
    prompt: str,
    *,
    default: str | None = None,
    validate: Callable[[str], str | None] | None = None,
) -> str:
    """Free-text prompt; re-asks on validate failure. validate returns an error string or None."""
    while True:
        value = (
            typer.prompt(prompt, default=default) if default is not None else typer.prompt(prompt)
        )
        value = str(value).strip()
        if validate is not None:
            err = validate(value)
            if err is not None:
                typer.echo(err)
                continue
        return value


def ask_choice(prompt: str, choices: list[str], *, default: str | None = None) -> str:
    """Membership-checked choice; re-asks on invalid."""
    rendered = f"{prompt} [{'/'.join(choices)}]"
    while True:
        value = (
            typer.prompt(rendered, default=default)
            if default is not None
            else typer.prompt(rendered)
        )
        value = str(value).strip()
        if value in choices:
            return value
        typer.echo(f"choose one of: {', '.join(choices)}")


def ask_confirm(prompt: str, *, default: bool = True) -> bool:
    return typer.confirm(prompt, default=default)


def _ask_dataset_source(ctx: Ctx) -> dict[str, Any]:
    fmt = ask_choice("Dataset format?", ["coco", "hf"], default="coco")
    if fmt == "coco":
        ann = ask_text("Path to COCO train annotations (.json)?")
        imgs = ask_text("Path to COCO train images dir?")
        return {"data": {"format": "coco", "train": {"annotations": ann, "images": imgs}}}
    name = ask_text("HuggingFace dataset name (org/dataset)?")
    return {"data": {"format": "hf", "hf": {"name": name}}}


def _ask_validation(ctx: Ctx) -> dict[str, Any]:
    fmt = ctx.answers.get("data", {}).get("format", "coco")
    mode = ask_choice("Validation?", ["explicit", "auto-split", "none"], default="auto-split")
    if mode == "none":
        if ctx.run_mode in {"eval", "run"}:
            typer.echo(
                "note: eval/run needs a validation set to score against; "
                "selecting none means eval will have nothing to evaluate."
            )
        return {}
    if mode == "auto-split":

        def _fraction(s: str) -> str | None:
            try:
                f = float(s)
            except ValueError:
                return "fraction must be a number"
            return None if 0.0 < f <= 0.5 else "fraction must be in (0, 0.5]"

        frac = ask_text("Auto-split fraction (0<f<=0.5)?", default="0.1", validate=_fraction)
        return {"data": {"val_split": {"fraction": float(frac)}}}
    if fmt == "hf":
        split = ask_text("HF validation split name?", default="validation")
        return {"data": {"hf": {"split_val": split}}}
    ann = ask_text("Path to COCO val annotations (.json)?")
    imgs = ask_text("Path to COCO val images dir?")
    return {"data": {"val": {"annotations": ann, "images": imgs}}}


def _ask_model_weights(ctx: Ctx) -> dict[str, Any]:
    def _is_file_or_blank(s: str) -> str | None:
        if s == "":
            return None
        return None if Path(s).is_file() else f"no file at {s}"

    raw = ask_text(
        "Path to an existing SAM 3.1 checkpoint (.pt)? Leave blank to use "
        "`models/sam3.1` and download if missing.",
        default="",
        validate=_is_file_or_blank,
    )
    if raw:
        p = Path(raw)
        return {"model": {"local_dir": str(p.parent), "checkpoint_file": p.name}}
    hits = sorted(Path("models").glob("**/sam3.1_multiplex.pt")) if Path("models").is_dir() else []
    if hits:
        return {"model": {"local_dir": str(hits[0].parent)}}
    return {}


def run_wizard(ctx: Ctx, steps: list[WizardStep]) -> dict[str, Any]:
    """Iterate the passed-in step list, merging each enabled step's fragment."""
    for step in steps:
        if step.when(ctx):
            fragment = step.ask(ctx)
            _deep_merge(ctx.answers, fragment)
    return ctx.answers


def validate(rendered: str) -> None:
    """Validate the exact bytes via load_config by round-tripping through a temp file."""
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write(rendered)
        tmp = Path(f.name)
    try:
        load_config(tmp)
    finally:
        tmp.unlink(missing_ok=True)


_LAUNCH_VERB = {"train": "train", "run": "run", "eval": "eval"}


def _launch_command(output: Path, run_mode: RunMode) -> str:
    return f"custom-sam-peft {_LAUNCH_VERB[run_mode]} --config {output}"


def _header(launch: str, generating_command: str = "custom-sam-peft init --interactive") -> str:
    return (
        f"# Generated by `{generating_command}` on {date.today().isoformat()}\n"
        f"# Launch: {launch}\n\n"
    )
```

> **Signature change:** `run_wizard` now takes the `steps` list as a parameter (today it reads a module-global `STEPS`). This lets each helper pass its own step list. `setup_wizard` passes its own `STEPS` (Task 7). `_header` gains a `generating_command` param (default unchanged) so `eval -i` baseline can label `csp eval --interactive` (§3, §5.3).

- [ ] **Step 4: Edit `setup_wizard.py` to import + re-export the moved symbols**

In `src/custom_sam_peft/cli/setup_wizard.py`:

1. Delete the now-moved definitions: `RunMode` (line 29), `Ctx` (32-38), `WizardStep` (41-45), `_deep_merge` (48-54), `ask_text` (57-74), `ask_choice` (77-89), `ask_confirm` (92-93), `_ask_dataset_source` (285-292), `_ask_validation` (295-321), `_ask_model_weights` (406-424), `validate` (459-467), `_LAUNCH_VERB` (456), `_launch_command` (470-471), `_header` (474-478).
2. Add the import (at module scope, alongside the existing `from custom_sam_peft.cli.init_cmd import ...`):

```python
from custom_sam_peft.cli._interactive import (
    Ctx,
    RunMode,
    WizardStep,
    _ask_dataset_source,
    _ask_model_weights,
    _ask_validation,
    _deep_merge,
    _header,
    _launch_command,
    ask_choice,
    ask_confirm,
    ask_text,
    run_wizard,
    validate,
)
```

3. Update `run_wizard` call sites in `setup_wizard.py` to pass `STEPS`: `generate_config` calls `run_wizard(ctx)` (line 501) → `run_wizard(ctx, STEPS)`.
4. Keep `setup_wizard`'s own step builders (`_ask_run_mode`, `_ask_run_name`, `_ask_domain`, `_ask_class_imbalance`, `_ask_peft_sizing`, `_ask_epochs`, `infer_class_imbalance` + ratio helpers), the `STEPS` list, `render` + block helpers, `emit`, `generate_config`, and the `from custom_sam_peft.cli.init_cmd import UNIFIED_TEMPLATE, _build_loss_overrides_block` import.
5. Remove now-unused imports in `setup_wizard.py` if they are no longer referenced after the move (`tempfile`, `date` move to `_interactive`; confirm via `grep -n "tempfile\.\|date\."` and remove dead imports). Keep `string`, `files`, `Path`, `Literal`, `typer`, `load_config` if still used by the retained code (`render` uses `string` + `files`; `validate` moved so `load_config` may become unused in `setup_wizard` — grep and remove if so).

Because `_interactive._ask_dataset_source` / `_ask_validation` / `_ask_model_weights` are imported into `setup_wizard`, the `STEPS` list entries that reference them (lines 430, 431, 440) keep working unchanged.

- [ ] **Step 5: Run both test files to verify green**

Run: `uv run pytest tests/unit/cli/test_interactive.py tests/unit/cli/test_setup_wizard.py -v`
Expected: PASS — the new `_interactive` tests pass; the existing `setup_wizard` tests (which use `sw.ask_text`, `sw.WizardStep`, `sw.Ctx`, `sw.run_wizard`, `sw.validate`, `sw._deep_merge`) keep passing via the re-exports. Note: the existing `test_setup_wizard.py` calls `sw.run_wizard(ctx)` with one arg (line 229) — that call site is updated in Task 7; if it fails here on the new 2-arg signature, defer that test fix to Task 7 and confirm the rest are green, OR update the call site now to `sw.run_wizard(ctx, sw.STEPS)`.

- [ ] **Step 6: Run mypy + ruff on the touched files**

Run: `uv run ruff check src/custom_sam_peft/cli/_interactive.py src/custom_sam_peft/cli/setup_wizard.py && uv run mypy src/custom_sam_peft/cli/_interactive.py src/custom_sam_peft/cli/setup_wizard.py`
Expected: clean (no unused-import findings; no type errors).

- [ ] **Step 7: Commit**

```bash
git add src/custom_sam_peft/cli/_interactive.py src/custom_sam_peft/cli/setup_wizard.py tests/unit/cli/test_interactive.py tests/unit/cli/test_setup_wizard.py
git commit -m "refactor(cli): extract shared interactive machinery into _interactive.py"
```

### Task 6: Add `require_tty`, validators, and `peek_adapter` to `_interactive.py`

**Spec ref:** §3 (require_tty, peek_adapter, validators), §9 (TTY guard).

**Files:**

- Modify: `src/custom_sam_peft/cli/_interactive.py` (append new functions)
- Test: `tests/unit/cli/test_interactive.py`

> Edits `_interactive.py` (shares with Task 5) — run after Task 5.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/cli/test_interactive.py`:

```python
import typer


def test_require_tty_non_tty_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(itv.sys.stdin, "isatty", lambda: False)
    with pytest.raises(typer.BadParameter, match="TTY"):
        itv.require_tty()


def test_require_tty_tty_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(itv.sys.stdin, "isatty", lambda: True)
    assert itv.require_tty() is None


def test_validate_checkpoint_dir(tmp_path: Path) -> None:
    good = tmp_path / "ckpt"
    good.mkdir()
    (good / "adapter_config.json").write_text("{}")
    assert itv.validate_checkpoint_dir(str(good)) is None
    bad = tmp_path / "empty"
    bad.mkdir()
    assert itv.validate_checkpoint_dir(str(bad)) is not None
    assert itv.validate_checkpoint_dir(str(tmp_path / "missing")) is not None


def test_validate_config_with_eval_split(tmp_path: Path) -> None:
    import textwrap

    def _write(body: str) -> Path:
        p = tmp_path / f"{abs(hash(body))}.yaml"
        p.write_text(textwrap.dedent(body))
        return p

    base = """
    run: {name: r}
    model: {name: facebook/sam3.1, local_dir: models/sam3.1, checkpoint_file: c.pt}
    data:
      format: coco
      train: {annotations: t.json, images: t/}
      VAL_BLOCK
    peft: {method: lora, r: 16, alpha: 32, dropout: 0.05}
    train:
      epochs: 1
      loss: {preset: natural, class_imbalance: balanced}
    """
    with_val = _write(base.replace("VAL_BLOCK", "val: {annotations: v.json, images: v/}"))
    assert itv.validate_config_with_eval_split(str(with_val)) is None
    no_val = _write(base.replace("      VAL_BLOCK\n", ""))
    assert itv.validate_config_with_eval_split(str(no_val)) is not None
    assert itv.validate_config_with_eval_split(str(tmp_path / "nope.yaml")) is not None


def test_peek_adapter_lora(tmp_path: Path) -> None:
    import json

    (tmp_path / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "facebook/sam3.1"})
    )
    pretty, base = itv.peek_adapter(tmp_path)
    assert pretty == "LoRA"
    assert base == "facebook/sam3.1"


def test_peek_adapter_qlora(tmp_path: Path) -> None:
    import json

    (tmp_path / "adapter_config.json").write_text(json.dumps({}))
    (tmp_path / "custom_sam_peft_qlora.json").write_text("{}")
    pretty, base = itv.peek_adapter(tmp_path)
    assert pretty == "QLoRA"
    assert base is None
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/unit/cli/test_interactive.py -k "require_tty or validate_checkpoint or validate_config or peek_adapter" -v`
Expected: FAIL — `require_tty`/`validate_checkpoint_dir`/`validate_config_with_eval_split`/`peek_adapter` are not defined.

- [ ] **Step 3: Implement the new functions in `_interactive.py`**

Append to `src/custom_sam_peft/cli/_interactive.py`. (`ConfigError` import is added lazily inside the validator to keep module imports tight; `load_config` is already imported at module scope.)

```python
def require_tty() -> None:
    """Raise typer.BadParameter if stdin is not a TTY. Call BEFORE any prompt."""
    if not sys.stdin.isatty():
        raise typer.BadParameter(
            "interactive mode needs a TTY; use the flag-driven command instead"
        )


def validate_checkpoint_dir(s: str) -> str | None:
    """ask_text validator: None unless s is a dir containing adapter_config.json."""
    p = Path(s)
    if p.is_dir() and (p / "adapter_config.json").is_file():
        return None
    return f"{s} is not an adapter checkpoint dir (missing adapter_config.json)"


def validate_config_with_eval_split(s: str) -> str | None:
    """ask_text validator for eval-reuse: None when s load_config's AND carries a
    val / val_split / hf.split_val / test source; else an error string."""
    from custom_sam_peft.errors import ConfigError

    try:
        cfg = load_config(Path(s))
    except ConfigError as exc:
        return str(exc)
    has_split = (
        cfg.data.val is not None
        or cfg.data.val_split is not None
        or (cfg.data.format == "hf" and cfg.data.hf is not None and cfg.data.hf.split_val is not None)
        or cfg.data.test is not None
    )
    if has_split:
        return None
    return "config has no val/test split to evaluate; pick a config with one"


def peek_adapter(checkpoint_dir: Path) -> tuple[str, str | None]:
    """Return (pretty_method_name, base_model_name) for a known-good adapter dir.

    The caller validates dir existence + adapter_config.json presence (via
    validate_checkpoint_dir) BEFORE calling, so this operates on a good dir and
    never opens the model. Lazy-imports the peft_adapters seam (§7).
    """
    from custom_sam_peft.peft_adapters import (
        discover_method_from_checkpoint,
        method_pretty_name,
        read_adapter_base_model_name,
    )

    method = discover_method_from_checkpoint(checkpoint_dir)
    return method_pretty_name(method), read_adapter_base_model_name(checkpoint_dir)
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/unit/cli/test_interactive.py -k "require_tty or validate_checkpoint or validate_config or peek_adapter" -v`
Expected: PASS — TTY guard raises/returns correctly; validators accept good configs/dirs and reject bad; peek returns `(LoRA, base)` / `(QLoRA, None)`.

- [ ] **Step 5: Commit**

```bash
git add src/custom_sam_peft/cli/_interactive.py tests/unit/cli/test_interactive.py
git commit -m "feat(cli): add require_tty, eval/checkpoint validators, peek_adapter"
```

---

## Phase 3 — `init -i` shrinks to `train|run` (§4)

> Edits `setup_wizard.py` (shares the file with Phase 2 — serialize after Phase 2).

### Task 7: Narrow the `init` `RunMode` selector + un-gate `epochs` / `class_imbalance`

**Spec ref:** §4.

**Files:**

- Modify: `src/custom_sam_peft/cli/setup_wizard.py` (`_ask_run_mode` line 276; `STEPS` list lines 427-441; the `run_wizard(ctx)` call site)
- Test: `tests/unit/cli/test_setup_wizard.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/unit/cli/test_setup_wizard.py`:

```python
def test_run_mode_offers_only_train_run(monkeypatch) -> None:
    captured: dict[str, list[str]] = {}

    def _fake_choice(prompt, choices, *, default=None):
        captured["choices"] = list(choices)
        return "train"

    monkeypatch.setattr(sw, "ask_choice", _fake_choice)
    ctx = sw.Ctx(answers={}, cuda_available=False)
    sw._ask_run_mode(ctx)
    assert captured["choices"] == ["train", "run"]


def test_epochs_step_always_runs() -> None:
    step = next(s for s in sw.STEPS if s.id == "epochs")
    for mode in ("train", "run"):
        ctx = sw.Ctx(answers={}, cuda_available=False, run_mode=mode)
        assert step.when(ctx) is True


def test_class_imbalance_step_runs_for_train_and_run() -> None:
    step = next(s for s in sw.STEPS if s.id == "class_imbalance")
    for mode in ("train", "run"):
        ctx = sw.Ctx(answers={"data": {"format": "coco"}}, cuda_available=False, run_mode=mode)
        assert step.when(ctx) is True
```

Also: the existing `test_when_gating_skips_class_imbalance_in_eval_mode` (line 239) asserts the `class_imbalance` step is gated OFF in `eval` mode. After un-gating (the step always runs for the remaining `train`/`run`), that test's premise no longer holds for `init`. Update it: since the shared `RunMode` superset still includes `eval` but `init` no longer offers it, the `class_imbalance` `when` is removed entirely (always true) — change `test_when_gating_skips_class_imbalance_in_eval_mode` to assert `step.when(ctx) is True` for `run_mode="eval"` (the step has no `when` gate now), OR delete it as redundant with `test_class_imbalance_step_runs_for_train_and_run`. Delete it (redundant). Likewise the existing `test_setup_wizard.py` `sw.run_wizard(ctx)` call site (line ~229) must become `sw.run_wizard(ctx, sw.STEPS)` for the new 2-arg signature (Task 5 Step 5 deferred it here).

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/unit/cli/test_setup_wizard.py -k "run_mode_offers or epochs_step_always or class_imbalance_step_runs" -v`
Expected: FAIL — `_ask_run_mode` still offers `["train", "run", "eval"]`; `epochs` step still has the `when=lambda ctx: ctx.run_mode != "eval"` gate; `class_imbalance` still has the `when=lambda ctx: ctx.run_mode in {"train", "run"}` gate (which passes for train/run but the test for `epochs` fails because the `when` is non-trivial).

- [ ] **Step 3: Narrow the run-mode selector**

In `src/custom_sam_peft/cli/setup_wizard.py`, change `_ask_run_mode` (line 276):

```python
def _ask_run_mode(ctx: Ctx) -> dict[str, Any]:
    ctx.run_mode = ask_choice("Run mode?", ["train", "run"], default="train")  # type: ignore[assignment]
    return {}
```

- [ ] **Step 4: Un-gate the `epochs` and `class_imbalance` steps**

In the `STEPS` list (lines 427-441), remove both `when` gates so the steps always run:

```python
STEPS: list[WizardStep] = [
    WizardStep("run_mode", _ask_run_mode),
    WizardStep("run_name", _ask_run_name),
    WizardStep("dataset_source", _ask_dataset_source),
    WizardStep("validation", _ask_validation),
    WizardStep("domain", _ask_domain),
    WizardStep("class_imbalance", _ask_class_imbalance),
    WizardStep("peft_sizing", _ask_peft_sizing),
    WizardStep("epochs", _ask_epochs),
    WizardStep("model_weights", _ask_model_weights),
]
```

Update the `run_wizard` call in `generate_config` (if not already done in Task 5) to pass `STEPS`: `answers = run_wizard(ctx, STEPS)`.

The `render` default `epochs = answers.get("train", {}).get("epochs", 1)` (line 250) stays — it is unreachable from `init -i` now (epochs always asked) but still serves the eval-baseline render (§5.3). No edit.

- [ ] **Step 5: Run to verify it passes**

Run: `uv run pytest tests/unit/cli/test_setup_wizard.py -v`
Expected: PASS — run-mode offers only `train`/`run`; `epochs` + `class_imbalance` steps always run; existing render/emit/class-imbalance tests stay green (the eval-mode render test `test_render_eval_mode_defaults_epochs_to_1` still works because `render(..., run_mode="eval")` is still callable — `RunMode` superset keeps `eval`; the eval-mode `emit` loop also still works).

- [ ] **Step 6: Commit**

```bash
git add src/custom_sam_peft/cli/setup_wizard.py tests/unit/cli/test_setup_wizard.py
git commit -m "feat(init)!: shrink interactive run-mode to train|run; un-gate epochs/imbalance"
```

---

## Phase 4 — eval-runner baseline + PEFT-from-checkpoint (§6)

> `eval/runner.py` + `eval_cmd.py`. Depends on Phase 1 (the seam). File-disjoint from Phases 5/6. Tasks 8 and 9 share `eval/runner.py` is FALSE — Task 8 edits `eval_cmd.py`, Task 9 edits `eval/runner.py`; they can run in parallel but Task 9's tests exercise the runner directly.

### Task 8: `eval_cmd.py` — `--checkpoint` becomes optional

**Spec ref:** §6.1.

**Files:**

- Modify: `src/custom_sam_peft/cli/eval_cmd.py` (`config` option line 22; `checkpoint` option line 23; `--output` help line 27; add a non-interactive `--config` presence check)
- Test: `tests/unit/cli/test_eval_cmd.py` (new) or extend an existing eval CLI test

> This task is small and edits only `eval_cmd.py`. It can run in parallel with Task 9 (which edits `eval/runner.py`).
>
> **Why `--config` also becomes optional:** Task 11 adds `--interactive`, which must dispatch BEFORE `--config` is read (interactive mode prompts for the config path itself). Typer enforces required options at parse time, so a required `--config` would make `csp eval --interactive` (no `--config`) a usage error before the early branch runs. Making `--config` optional (with a non-interactive presence check) squares this with the additive-flag intent (§1) — the non-interactive contract is preserved by the explicit check.

- [ ] **Step 1: Write the failing CLI test**

Create `tests/unit/cli/test_eval_cmd.py`:

```python
"""Tests for the `csp eval` CLI option surface (CPU-only)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from typer.testing import CliRunner

from custom_sam_peft.cli.main import app

runner = CliRunner()


def test_eval_checkpoint_optional_invokes_baseline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Omitting --checkpoint must NOT be a CLI usage error; run_eval is called
    with checkpoint=None (baseline)."""
    cfg = tmp_path / "c.yaml"
    cfg.write_text("placeholder")
    monkeypatch.setattr("custom_sam_peft.cli.eval_cmd.load_config", lambda p: MagicMock())
    captured: dict[str, object] = {}

    def _fake_run_eval(cfg, **kw):
        captured.update(kw)
        report = MagicMock()
        report.overall = {}
        return report

    monkeypatch.setattr("custom_sam_peft.cli.eval_cmd.run_eval", _fake_run_eval)
    result = runner.invoke(app, ["eval", "--config", str(cfg), "--split", "val"])
    assert result.exit_code == 0, result.output
    assert captured["checkpoint"] is None
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/unit/cli/test_eval_cmd.py -v`
Expected: FAIL — `--checkpoint` is currently `typer.Option(...)` (required), so omitting it is a usage error (exit code 2) and `run_eval` is never reached.

- [ ] **Step 3: Make `--config` and `--checkpoint` optional + add a non-interactive `--config` check**

In `src/custom_sam_peft/cli/eval_cmd.py`, change the `config` option (line 22) and `checkpoint` option (line 23) to optional:

```python
    config: Path | None = typer.Option(None, "--config", help="Path to config YAML."),
    checkpoint: Path | None = typer.Option(
        None,
        "--checkpoint",
        help="Path to adapter checkpoint. Omit to evaluate baseline (zero-shot) SAM.",
    ),
```

Reword the `--output` help (line 27) to note the baseline fallback:

```python
    output: Path | None = typer.Option(
        None,
        "--output",
        help="Output dir; defaults to checkpoint.parent, else cfg.run.output_dir.",
    ),
```

In the `evaluate` body, BEFORE `cfg = load_config(config)` (line 50), guard the non-interactive `--config` requirement (the interactive early branch from Task 11 returns before this):

```python
    if config is None:
        raise typer.BadParameter("--config is required", param_hint="--config")
```

The `run_eval(cfg, checkpoint=checkpoint, ...)` call (line 67) passes `checkpoint` through unchanged (now possibly `None`). The `ValueError → BadParameter(param_hint="--checkpoint")` wrap (lines 74-75) stays.

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/unit/cli/test_eval_cmd.py -v`
Expected: PASS — omitting `--checkpoint` reaches `run_eval` with `checkpoint=None`; the explicit non-interactive `--config` check still rejects `csp eval` (no `--config`, no `--interactive`) with a `BadParameter`. Add a guard test:

```python
def test_eval_config_required_non_interactive() -> None:
    result = runner.invoke(app, ["eval", "--split", "val"])
    assert result.exit_code != 0
    assert "config" in result.output.lower()
```

- [ ] **Step 5: Commit**

```bash
git add src/custom_sam_peft/cli/eval_cmd.py tests/unit/cli/test_eval_cmd.py
git commit -m "feat(eval): make --checkpoint optional (baseline zero-shot eval)"
```

### Task 9: `run_eval` — baseline path, sentinel dispatch, advisory warning, output-dir fallback

**Spec ref:** §6.2, §6.3, §6.4, §6.5, §7.5.

**Files:**

- Modify: `src/custom_sam_peft/eval/runner.py` (imports lines 23-24; standalone resolve block 99-104; dispatch block 131-138; output-dir block 160-165; docstring 90-93)
- Test: `tests/unit/test_eval_runner.py`

> Edits `eval/runner.py`. Depends on Phase 1 (the seam). File-disjoint from Task 8.

- [ ] **Step 1: Write the failing tests**

Add to `tests/unit/test_eval_runner.py` (reuse the existing `_make_cfg` helper and monkeypatch pattern from the file's top — note `_make_cfg` sets `cfg.peft.method`):

```python
def test_peft_inferred_lora_overrides_cfg_method(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A lora checkpoint (no sentinel) dispatches load_lora even when cfg says qlora."""
    cfg = _make_cfg(peft_method="qlora")
    (tmp_path / "adapter_config.json").write_text("{}")
    calls: list[str] = []
    monkeypatch.setattr("custom_sam_peft.peft_adapters.lora.load_lora", lambda *a, **k: calls.append("lora"))
    monkeypatch.setattr("custom_sam_peft.peft_adapters.qlora.load_qlora", lambda *a, **k: calls.append("qlora"))
    monkeypatch.setattr("custom_sam_peft.eval.runner._load_channel_adapter", lambda *a, **k: None)
    monkeypatch.setattr("custom_sam_peft.eval.runner.load_sam31", lambda _m, **_kw: MagicMock())
    monkeypatch.setattr(
        "custom_sam_peft.eval.runner.lookup",
        lambda kind, name: (lambda cfg_dict, **kw: MagicMock()),
    )
    ev = MagicMock()
    ev.evaluate_and_save.return_value = MagicMock(overall={})
    monkeypatch.setattr("custom_sam_peft.eval.runner.Evaluator", lambda _c: ev)
    run_eval(cfg, checkpoint=tmp_path, split="val", output_dir=tmp_path)
    assert calls == ["lora"]


def test_peft_inferred_qlora_overrides_cfg_method(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _make_cfg(peft_method="lora")
    (tmp_path / "custom_sam_peft_qlora.json").write_text("{}")
    (tmp_path / "adapter_config.json").write_text("{}")
    calls: list[str] = []
    monkeypatch.setattr("custom_sam_peft.peft_adapters.lora.load_lora", lambda *a, **k: calls.append("lora"))
    monkeypatch.setattr("custom_sam_peft.peft_adapters.qlora.load_qlora", lambda *a, **k: calls.append("qlora"))
    monkeypatch.setattr("custom_sam_peft.eval.runner._load_channel_adapter", lambda *a, **k: None)
    monkeypatch.setattr("custom_sam_peft.eval.runner.load_sam31", lambda _m, **_kw: MagicMock())
    monkeypatch.setattr(
        "custom_sam_peft.eval.runner.lookup",
        lambda kind, name: (lambda cfg_dict, **kw: MagicMock()),
    )
    ev = MagicMock()
    ev.evaluate_and_save.return_value = MagicMock(overall={})
    monkeypatch.setattr("custom_sam_peft.eval.runner.Evaluator", lambda _c: ev)
    run_eval(cfg, checkpoint=tmp_path, split="val", output_dir=tmp_path)
    assert calls == ["qlora"]


def test_peft_mismatch_logs_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    cfg = _make_cfg(peft_method="qlora")  # config says qlora; dir is lora
    (tmp_path / "adapter_config.json").write_text("{}")
    monkeypatch.setattr("custom_sam_peft.eval.runner.load_adapter", lambda *a, **k: None)
    monkeypatch.setattr("custom_sam_peft.eval.runner.load_sam31", lambda _m, **_kw: MagicMock())
    monkeypatch.setattr(
        "custom_sam_peft.eval.runner.lookup",
        lambda kind, name: (lambda cfg_dict, **kw: MagicMock()),
    )
    ev = MagicMock()
    ev.evaluate_and_save.return_value = MagicMock(overall={})
    monkeypatch.setattr("custom_sam_peft.eval.runner.Evaluator", lambda _c: ev)
    with caplog.at_level("WARNING"):
        run_eval(cfg, checkpoint=tmp_path, split="val", output_dir=tmp_path)
    assert any("checkpoint" in r.message and "lora" in r.message for r in caplog.records)


def test_checkpoint_none_skips_adapter_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _make_cfg()
    load_calls: list[str] = []
    monkeypatch.setattr("custom_sam_peft.eval.runner.load_adapter", lambda *a, **k: load_calls.append("adapter"))
    monkeypatch.setattr("custom_sam_peft.eval.runner._load_channel_adapter", lambda *a, **k: load_calls.append("channel"))
    sam_calls: list[int] = []
    monkeypatch.setattr("custom_sam_peft.eval.runner.load_sam31", lambda _m, **_kw: (sam_calls.append(1), MagicMock())[1])
    monkeypatch.setattr(
        "custom_sam_peft.eval.runner.lookup",
        lambda kind, name: (lambda cfg_dict, **kw: MagicMock()),
    )
    ev = MagicMock()
    ev.evaluate_and_save.return_value = MagicMock(overall={})
    monkeypatch.setattr("custom_sam_peft.eval.runner.Evaluator", lambda _c: ev)
    run_eval(cfg, checkpoint=None, split="val", output_dir=tmp_path)
    assert load_calls == []  # no adapter / channel-adapter load on baseline
    assert sam_calls == [1]  # base model loaded once
    assert ev.evaluate_and_save.called


def test_baseline_output_dir_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _make_cfg()
    cfg.run.output_dir = str(tmp_path / "runs")
    monkeypatch.setattr("custom_sam_peft.eval.runner.load_sam31", lambda _m, **_kw: MagicMock())
    monkeypatch.setattr(
        "custom_sam_peft.eval.runner.lookup",
        lambda kind, name: (lambda cfg_dict, **kw: MagicMock()),
    )
    captured: dict[str, object] = {}
    ev = MagicMock()

    def _save(wrapper, dataset, out):
        captured["out"] = out
        return MagicMock(overall={})

    ev.evaluate_and_save.side_effect = _save
    monkeypatch.setattr("custom_sam_peft.eval.runner.Evaluator", lambda _c: ev)
    run_eval(cfg, checkpoint=None, split="val", output_dir=None)
    assert str(captured["out"]) == str(tmp_path / "runs")  # no NoneType.parent crash
```

Note: `_make_cfg` (top of file) sets `cfg.eval.batch_size`. If it does not, set `cfg.eval.batch_size = 1` in `_make_cfg` (or in each new test) so the auto-batch branch (`runner.py:144`) is skipped. Confirm the existing helper already returns a usable `cfg.eval`; if `cfg.eval.batch_size == "auto"` it will hit `decide_eval_batch_size` — monkeypatch that too or set a concrete int.

Also retarget the existing `test_run_eval_dispatches_qlora_from_disk` (line 33): the standalone qlora path now dispatches via `train.checkpoint.load_adapter` (sentinel) rather than `make_peft_method(...).load_from_disk`. Update it so the synthetic dir carries the qlora sentinel (`(tmp_path / "custom_sam_peft_qlora.json").write_text("{}")`), and assert `load_qlora` + `_load_channel_adapter` still run (the channel-adapter restore now happens INSIDE `load_adapter` — so monkeypatch `custom_sam_peft.eval.runner.load_adapter` to a spy that records the call, OR keep monkeypatching `load_qlora` + `_load_channel_adapter` at their source modules since `load_adapter` calls them). The existing `test_run_eval_lora_calls_load_channel_adapter` (line 80) similarly: the dir needs `adapter_config.json` and no sentinel; assert the channel-adapter restore happens via `load_adapter`.

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/unit/test_eval_runner.py -v`
Expected: FAIL — `run_eval(checkpoint=None)` currently raises `ValueError("run_eval requires either ...")`; the inference/warning tests fail because dispatch still goes through `make_peft_method(cfg.peft.method).load_from_disk`; the output-dir test crashes on `resolved_checkpoint.parent` when `resolved_checkpoint is None`.

- [ ] **Step 3: Update imports in `eval/runner.py`**

In `src/custom_sam_peft/eval/runner.py`:

- Remove line 23: `from custom_sam_peft.peft_adapters import make_peft_method` (no longer used — confirm with `grep -n "make_peft_method" src/custom_sam_peft/eval/runner.py` → only line 106 used it, which is removed below).
- Change line 24: `from custom_sam_peft.train.checkpoint import _load_channel_adapter` → import BOTH `load_adapter` and `_load_channel_adapter` (the latter stays imported so the existing tests can monkeypatch `custom_sam_peft.eval.runner._load_channel_adapter`):

```python
from custom_sam_peft.train.checkpoint import _load_channel_adapter, load_adapter
```

- [ ] **Step 4: Rewrite the standalone resolve block (baseline allowed)**

Replace lines 95-104 (the `if artifacts is not None: ... else: ...` resolve block):

```python
    # Resolve checkpoint and run_dir. On the standalone path, checkpoint may be
    # None → baseline (zero-shot) eval with no adapter load.
    if artifacts is not None:
        resolved_checkpoint = artifacts.checkpoint_path
        resolved_run_dir = artifacts.run_dir
    else:
        resolved_checkpoint = checkpoint  # may be None → baseline
        resolved_run_dir = None
```

Then DELETE line 106 (`_peft_method = make_peft_method(resolved_peft_method)`) — `resolved_peft_method` is no longer assigned or used on either branch. The `_hf_val` / `--split val` gate (lines 107-114) and `--split test` gate (115-116) and dataset-build block (118-129) are UNCHANGED.

- [ ] **Step 5: Rewrite the adapter-load dispatch block (sentinel + advisory warning)**

Replace lines 131-138 (the `if model is None: ... else: wrapper = model` block):

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
                    "cfg.peft.method=%r but the checkpoint at %s is %r; loading the "
                    "checkpoint's method (config value ignored for eval dispatch).",
                    cfg.peft.method,
                    resolved_checkpoint,
                    detected,
                )
            load_adapter(wrapper, resolved_checkpoint)
        # else: baseline — no adapter load, no channel-adapter restore.
    else:
        wrapper = model
```

`load_adapter` (imported in Step 3) does the sentinel-based LoRA/QLoRA dispatch AND the channel-adapter restore internally, so the separate `_peft_method.load_from_disk(...)` + `_load_channel_adapter(...)` calls are both gone from `run_eval`.

- [ ] **Step 6: Rewrite the output-dir fallback**

Replace lines 160-165 (the `out = (...)` expression):

```python
    # Output dir: explicit → artifacts.run_dir → checkpoint.parent → cfg.run.output_dir → cwd.
    if output_dir is not None:
        out = output_dir
    elif resolved_run_dir is not None:
        out = resolved_run_dir
    elif resolved_checkpoint is not None:
        out = resolved_checkpoint.parent
    else:
        out = Path(cfg.run.output_dir) if cfg.run.output_dir else Path.cwd()
```

- [ ] **Step 7: Update the docstring**

In `run_eval`'s docstring (lines 90-93), remove the `ValueError: neither checkpoint nor artifacts provided.` line and add a note that `checkpoint=None` (standalone) evaluates baseline SAM:

```python
    Raises:
        ValueError: split == 'test' and cfg.data.test is None.

    When ``checkpoint`` is None on the standalone path (and ``artifacts`` is also
    None), no adapter is loaded — evaluates baseline (zero-shot) SAM.
```

- [ ] **Step 8: Run to verify it passes**

Run: `uv run pytest tests/unit/test_eval_runner.py tests/unit/test_eval_runner_gate.py -v`
Expected: PASS — inference dispatches the detected method regardless of `cfg.peft.method`; mismatch logs a WARNING; `checkpoint=None` skips adapter load; output-dir falls back to `cfg.run.output_dir` with no crash; the `--split val`/`--split test` gate tests (`test_eval_runner_gate.py`) are untouched and still green; retargeted qlora/lora dispatch tests pass.

- [ ] **Step 9: Run the trainer→evaluator integration tests (EvalArtifacts.peft_method kept)**

Run: `uv run pytest tests/integration/test_trainer_evaluator_seam.py tests/integration/test_peft_extensibility.py tests/unit/test_eval_artifacts.py -v`
Expected: PASS — `EvalArtifacts.peft_method` is untouched (§7.6); the artifacts branch still reads `artifacts.checkpoint_path`/`run_dir`; nothing references the removed `resolved_peft_method` local.

- [ ] **Step 10: Commit**

```bash
git add src/custom_sam_peft/eval/runner.py tests/unit/test_eval_runner.py
git commit -m "feat(eval): baseline path + infer PEFT method from checkpoint sentinel"
```

---

## Phase 5 — `csp eval --interactive` (§5)

> Appends `run_eval_interactive` to `_interactive.py`, wires `eval_cmd.py`. Depends on Phases 1 (seam), 2 (`_interactive`), 4 (eval-runner baseline). Shares `_interactive.py` with Phase 6 — serialize the two appending tasks if dispatched together.

### Task 10: `run_eval_interactive` helper (reuse + baseline paths)

**Spec ref:** §5, §5.1, §5.2, §5.3, §9.

**Files:**

- Modify: `src/custom_sam_peft/cli/_interactive.py` (append `run_eval_interactive`)
- Test: `tests/unit/cli/test_eval_interactive.py` (new)

> Appends to `_interactive.py` (shared with Task 12). Depends on Tasks 5/6 (`_interactive` primitives + peek + validators), Task 9 (eval-runner baseline), and `setup_wizard.render`/`emit`. `_interactive` must import `render`/`emit` from `setup_wizard` LAZILY inside the helper body (setup_wizard imports `_interactive` at module scope, so a module-scope reverse import would cycle — §2).

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/cli/test_eval_interactive.py`:

```python
"""Tests for the `eval -i` helper (CPU-only; prompts monkeypatched)."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from custom_sam_peft.cli import _interactive as itv
from custom_sam_peft.config.loader import load_config


def _write_cfg(tmp_path: Path) -> Path:
    p = tmp_path / "train.yaml"
    p.write_text(
        textwrap.dedent("""
        run: {name: r, output_dir: ./runs, seed: 42}
        model: {name: facebook/sam3.1, local_dir: models/sam3.1, checkpoint_file: c.pt, dtype: bfloat16}
        data:
          format: coco
          train: {annotations: t.json, images: t/}
          val: {annotations: v.json, images: v/}
          prompt_mode: text
          image_size: 1008
        peft: {method: lora, r: 16, alpha: 32, dropout: 0.05}
        train:
          epochs: 1
          batch_size: 1
          grad_accum_steps: 8
          loss: {preset: natural, class_imbalance: balanced}
        eval: {iou_thresholds: [0.5]}
        tracking: {backend: tensorboard}
        export: {merge: false}
        """)
    )
    return p


def _lora_ckpt(tmp_path: Path) -> Path:
    d = tmp_path / "ckpt"
    d.mkdir()
    (d / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": "facebook/sam3.1"}))
    return d


def test_reuse_prints_command_writes_nothing(tmp_path, monkeypatch, capsys) -> None:
    cfg = _write_cfg(tmp_path)
    ckpt = _lora_ckpt(tmp_path)
    text_answers = iter([str(cfg), str(ckpt)])
    monkeypatch.setattr(itv, "ask_choice", lambda prompt, choices, **k: "reuse" if "Evaluate" in prompt else "val")
    monkeypatch.setattr(itv, "ask_text", lambda *a, **k: next(text_answers))
    before = set(tmp_path.iterdir())
    itv.run_eval_interactive(output=None, force=False)
    out = capsys.readouterr().out
    assert f"custom-sam-peft eval --config {cfg} --checkpoint {ckpt} --split val" in out
    assert set(tmp_path.iterdir()) == before  # nothing new written


def test_reuse_peek_prints_method(tmp_path, monkeypatch, capsys) -> None:
    cfg = _write_cfg(tmp_path)
    ckpt = _lora_ckpt(tmp_path)
    (ckpt / "custom_sam_peft_qlora.json").write_text("{}")  # make it qlora
    text_answers = iter([str(cfg), str(ckpt)])
    monkeypatch.setattr(itv, "ask_choice", lambda prompt, choices, **k: "reuse" if "Evaluate" in prompt else "val")
    monkeypatch.setattr(itv, "ask_text", lambda *a, **k: next(text_answers))
    itv.run_eval_interactive(output=None, force=False)
    out = capsys.readouterr().out
    assert "QLoRA" in out
    assert "facebook/sam3.1" in out


def test_baseline_emits_reloadable_config(tmp_path, monkeypatch, capsys) -> None:
    out_cfg = tmp_path / "baseline.yaml"
    monkeypatch.setattr(
        itv, "ask_choice",
        lambda prompt, choices, **k: {"Evaluate": "baseline", "Dataset": "coco", "Validation": "auto-split"}.get(
            next((kw for kw in ("Evaluate", "Dataset", "Validation") if kw in prompt), ""), choices[0]
        ),
    )
    text_answers = iter(["ann.json", "imgs/", "0.1", ""])
    monkeypatch.setattr(itv, "ask_text", lambda *a, **k: next(text_answers))
    monkeypatch.setattr(itv, "ask_confirm", lambda *a, **k: True)
    itv.run_eval_interactive(output=out_cfg, force=False)
    assert out_cfg.is_file()
    cfg = load_config(out_cfg)
    assert cfg.data.val_split is not None
    assert cfg.run.name == "baseline-eval"
    out = capsys.readouterr().out
    assert f"custom-sam-peft eval --config {out_cfg} --split val" in out
    assert "--checkpoint" not in out


def test_output_exists_without_force(tmp_path, monkeypatch) -> None:
    out_cfg = tmp_path / "baseline.yaml"
    out_cfg.write_text("existing\n")
    monkeypatch.setattr(
        itv, "ask_choice",
        lambda prompt, choices, **k: {"Evaluate": "baseline", "Dataset": "coco", "Validation": "auto-split"}.get(
            next((kw for kw in ("Evaluate", "Dataset", "Validation") if kw in prompt), ""), choices[0]
        ),
    )
    text_answers = iter(["ann.json", "imgs/", "0.1", ""])
    monkeypatch.setattr(itv, "ask_text", lambda *a, **k: next(text_answers))
    monkeypatch.setattr(itv, "ask_confirm", lambda *a, **k: True)
    import typer

    with pytest.raises(typer.BadParameter, match="refusing to overwrite"):
        itv.run_eval_interactive(output=out_cfg, force=False)
    assert out_cfg.read_text() == "existing\n"
```

(The reuse `test_baseline_resolves_to_no_adapter_eval` and `test_ctrl_c_writes_nothing` from spec §10.3 are covered by Task 9's `test_checkpoint_none_skips_adapter_load` and by the "writes only at the end" structure respectively; add a `test_ctrl_c_writes_nothing` here too: monkeypatch `ask_text` to raise `KeyboardInterrupt` on first call, assert no file and `pytest.raises(KeyboardInterrupt)`.)

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/unit/cli/test_eval_interactive.py -v`
Expected: FAIL — `run_eval_interactive` is not defined.

- [ ] **Step 3: Implement `run_eval_interactive` in `_interactive.py`**

Append to `src/custom_sam_peft/cli/_interactive.py`:

```python
def run_eval_interactive(*, output: Path | None, force: bool) -> None:
    """Interactive `csp eval`: reuse a trained adapter (print a command) or build a
    baseline (zero-shot) eval config (write a config + print a command).

    Never runs an eval; prints/writes only. Caller guards the TTY first (§5).
    """
    mode = ask_choice(
        "Evaluate a trained adapter, or baseline zero-shot SAM?",
        ["reuse", "baseline"],
        default="reuse",
    )
    if mode == "reuse":
        _eval_reuse()
    else:
        _eval_baseline(output=output, force=force)


def _eval_reuse() -> None:
    config_path = ask_text(
        "Path to your existing training config (.yaml)?",
        validate=validate_config_with_eval_split,
    )
    checkpoint_dir = ask_text(
        "Path to the adapter checkpoint directory?",
        validate=validate_checkpoint_dir,
    )
    pretty, base = peek_adapter(Path(checkpoint_dir))
    typer.echo(f"detected adapter: {pretty}, base model: {base or '(unspecified)'}")
    split = ask_choice("Which split?", ["val", "test"], default="val")
    typer.echo(
        f"custom-sam-peft eval --config {config_path} "
        f"--checkpoint {checkpoint_dir} --split {split}"
    )


def _eval_baseline(*, output: Path | None, force: bool) -> None:
    from custom_sam_peft.cli.setup_wizard import emit, render  # lazy (avoid import cycle)

    out = output if output is not None else Path("baseline-eval.yaml")
    ctx = Ctx(answers={"run": {"name": "baseline-eval"}}, cuda_available=False, run_mode="eval")
    steps = [
        WizardStep("dataset_source", _ask_dataset_source),
        WizardStep("validation", _ask_validation),
        WizardStep("model_weights", _ask_model_weights),
    ]
    answers = run_wizard(ctx, steps)
    rendered = render(answers, run_mode="eval")
    validate(rendered)
    emit(rendered, out, force=force, run_mode="eval")
    typer.echo(f"custom-sam-peft eval --config {out} --split val")
```

> **`emit` overwrite refusal:** `setup_wizard.emit` already raises `typer.BadParameter("refusing to overwrite existing <path>; pass --force")` when `output.exists() and not force` (§5.3, §9). The baseline path relies on it. **Header note:** `emit` calls `_header` with the default `generating_command`; this prints `Generated by custom-sam-peft init --interactive`. Per §5.3 the eval-baseline header should say `csp eval --interactive`. Pass it through: change `_eval_baseline` to build the body itself rather than calling `emit`, OR (simpler) leave `emit` as-is and accept the default header — the spec marks the custom generating-command label as a nicety (§3 row for `_header`), not a hard requirement. **Decision (resolved ambiguity):** keep it simple — call `emit` (default header). If review wants the eval-specific label, `emit`/`_header` already accept it; wire `generating_command="custom-sam-peft eval --interactive"` through a small `emit` kwarg in a follow-up. Note this in the commit message.

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/unit/cli/test_eval_interactive.py -v`
Expected: PASS — reuse prints the runnable command and writes nothing; the qlora peek prints `QLoRA` + base model; baseline writes a reloadable config (`val_split` set, `run.name == "baseline-eval"`) and prints `--split val` with no `--checkpoint`; overwrite-without-force refuses; ctrl-C writes nothing.

- [ ] **Step 5: Commit**

```bash
git add src/custom_sam_peft/cli/_interactive.py tests/unit/cli/test_eval_interactive.py
git commit -m "feat(eval): add eval --interactive reuse/baseline helper"
```

### Task 11: Wire `--interactive`/`-i` into `eval_cmd.py`

**Spec ref:** §5 (steps 1-3), §9 (TTY guard).

**Files:**

- Modify: `src/custom_sam_peft/cli/eval_cmd.py` (add `interactive` option + early branch)
- Test: `tests/unit/cli/test_eval_cmd.py`

> Edits `eval_cmd.py` (shared with Task 8) — run after Task 8 and Task 10.

- [ ] **Step 1: Write the failing tests**

Add to `tests/unit/cli/test_eval_cmd.py`:

```python
def test_eval_interactive_dispatches_helper(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("custom_sam_peft.cli._interactive.sys.stdin.isatty", lambda: True)
    called: list[dict] = []
    monkeypatch.setattr(
        "custom_sam_peft.cli._interactive.run_eval_interactive",
        lambda **kw: called.append(kw),
    )
    result = runner.invoke(app, ["eval", "--interactive"])
    assert result.exit_code == 0, result.output
    assert len(called) == 1


def test_eval_interactive_non_tty_hard_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("custom_sam_peft.cli._interactive.sys.stdin.isatty", lambda: False)
    called: list[dict] = []
    monkeypatch.setattr(
        "custom_sam_peft.cli._interactive.run_eval_interactive",
        lambda **kw: called.append(kw),
    )
    result = runner.invoke(app, ["eval", "--interactive"])
    assert result.exit_code != 0
    assert "tty" in result.output.lower()
    assert called == []
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/unit/cli/test_eval_cmd.py -k interactive -v`
Expected: FAIL — `eval` has no `--interactive` flag (Typer usage error / unknown option).

- [ ] **Step 3: Add the flag + early branch**

In `src/custom_sam_peft/cli/eval_cmd.py`, add an `interactive` option to `evaluate(...)` (after `verbose`):

```python
    interactive: bool = typer.Option(
        False,
        "--interactive",
        "-i",
        help="Build an eval command (reuse a trained adapter) or a baseline eval config.",
    ),
```

At the top of the `evaluate` body, right after `configure_logging(verbose)` and BEFORE the `split` validation, the non-interactive `--config` check (Task 8 Step 3), and `load_config` (so interactive mode never touches `--config`/`--checkpoint`):

```python
    if interactive:
        from custom_sam_peft.cli import _interactive

        _interactive.require_tty()
        _interactive.run_eval_interactive(output=output, force=False)
        return
```

> Ordering: this early branch must precede the `if config is None: raise BadParameter("--config is required")` check added in Task 8, so `csp eval --interactive` (no `--config`) reaches the helper instead of erroring. The `evaluate` command has no `--force` flag today; the eval-baseline path passes `force=False`, relying on `emit`'s overwrite refusal. A user re-running with an existing `--output` gets the `BadParameter` refusal — they pick a new path. (Adding a `--force` flag is out of scope; the printed-command paths never write over inputs.)

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/unit/cli/test_eval_cmd.py -v`
Expected: PASS — `--interactive` dispatches the helper after the TTY guard; non-TTY hard-errors before the helper runs; the Task 8 optional-checkpoint test still green.

- [ ] **Step 5: Commit**

```bash
git add src/custom_sam_peft/cli/eval_cmd.py tests/unit/cli/test_eval_cmd.py
git commit -m "feat(eval): wire --interactive/-i flag with TTY guard"
```

---

## Phase 6 — `csp predict --interactive` (§8)

> Appends `run_predict_interactive` to `_interactive.py`, wires `predict_cmd.py`. Depends on Phases 1 (seam), 2 (`_interactive`). File-disjoint from Phase 5 except for `_interactive.py` (serialize the appends).

### Task 12: `run_predict_interactive` helper (command builder + thin config)

**Spec ref:** §8, §8.1, §8.2, §9.

**Files:**

- Modify: `src/custom_sam_peft/cli/_interactive.py` (append `run_predict_interactive`)
- Test: `tests/unit/cli/test_predict_interactive.py` (new)

> Appends to `_interactive.py` (shared with Task 10). Depends on Tasks 5/6 (primitives + peek + validators). Serialize the `_interactive.py` appends if dispatching Tasks 10 and 12 together.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/cli/test_predict_interactive.py`:

```python
"""Tests for the `predict -i` helper (CPU-only; prompts monkeypatched)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from custom_sam_peft.cli import _interactive as itv


def _lora_ckpt(tmp_path: Path) -> Path:
    d = tmp_path / "ckpt"
    d.mkdir()
    (d / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": "facebook/sam3.1"}))
    return d


def _drive(monkeypatch, *, checkpoint="", channels="3", semantics="rgb", merge=True,
           threshold="0.3", save_masks="rle", visualize=False,
           images="imgs/", prompts="cat,dog", output="out/"):
    choice_map = {"semantics": semantics, "Mask output": save_masks}
    text_iter = iter([checkpoint, channels, threshold, images, prompts, output])

    def _ask_choice(prompt, choices, **k):
        if "semantics" in prompt:
            return semantics
        if "Mask output" in prompt:
            return save_masks
        return choices[0]

    def _ask_text(prompt, **k):
        return next(text_iter)

    monkeypatch.setattr(itv, "ask_choice", _ask_choice)
    monkeypatch.setattr(itv, "ask_text", _ask_text)
    monkeypatch.setattr(itv, "ask_confirm", lambda prompt, **k: merge if "Merge" in prompt else visualize)


def test_command_assembly_baseline_rgb(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    _drive(monkeypatch, checkpoint="", channels="3", semantics="rgb")
    itv.run_predict_interactive(force=False)
    out = capsys.readouterr().out
    assert "--images imgs/" in out
    assert "--prompts cat,dog" in out
    assert "--output out/" in out
    assert "--checkpoint" not in out
    assert "--merge-adapter" not in out
    assert "--config" not in out
    assert "--visualize" not in out
    assert not (tmp_path / "predict-config.yaml").exists()


def test_command_assembly_with_checkpoint(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    ckpt = _lora_ckpt(tmp_path)
    _drive(monkeypatch, checkpoint=str(ckpt), channels="3", semantics="rgb", merge=True)
    itv.run_predict_interactive(force=False)
    out = capsys.readouterr().out
    assert f"--checkpoint {ckpt}" in out
    assert "--merge-adapter" in out
    assert "LoRA" in out  # peek output


def test_thin_config_emitted_for_non_rgb(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    _drive(monkeypatch, checkpoint="", channels="4", semantics="rgba")
    itv.run_predict_interactive(force=False)
    thin = tmp_path / "predict-config.yaml"
    assert thin.is_file()
    raw = yaml.safe_load(thin.read_text())
    assert raw["data"]["channels"] == 4
    assert raw["data"]["channel_semantics"] == "rgba"
    assert raw["model"]["name"] == "facebook/sam3.1"
    out = capsys.readouterr().out
    assert "--config" in out and "predict-config.yaml" in out


def test_thin_config_not_emitted_for_rgb(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    _drive(monkeypatch, checkpoint="", channels="3", semantics="rgb")
    itv.run_predict_interactive(force=False)
    assert not (tmp_path / "predict-config.yaml").exists()
    assert "--config" not in capsys.readouterr().out


def test_visualize_flag_emitted_when_yes(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    _drive(monkeypatch, checkpoint="", visualize=True)
    itv.run_predict_interactive(force=False)
    assert "--visualize" in capsys.readouterr().out


def test_thin_config_overwrite_refused(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "predict-config.yaml").write_text("existing\n")
    _drive(monkeypatch, checkpoint="", channels="4", semantics="rgba")
    import typer

    with pytest.raises(typer.BadParameter, match="refusing to overwrite"):
        itv.run_predict_interactive(force=False)
    assert (tmp_path / "predict-config.yaml").read_text() == "existing\n"
```

> The prompt-order in `_drive`'s `text_iter` is: checkpoint (P1), channels (P2), score_threshold (P5), images (P8), prompts (P9), output (P10). `semantics` (P3) and `save_masks` (P6) come via `ask_choice`; `merge_adapter` (P4, only when checkpoint) and `visualize` (P7) via `ask_confirm`. The implementation in Step 3 must prompt in exactly this order so the iterator lines up.

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/unit/cli/test_predict_interactive.py -v`
Expected: FAIL — `run_predict_interactive` is not defined.

- [ ] **Step 3: Implement `run_predict_interactive` in `_interactive.py`**

Append to `src/custom_sam_peft/cli/_interactive.py`. Add `import shlex` to the module's top-of-file imports.

```python
_PREDICT_DEFAULT_MODEL = "facebook/sam3.1"
_THIN_CONFIG_NAME = "predict-config.yaml"


def run_predict_interactive(*, force: bool) -> None:
    """Interactive `csp predict`: collect adapter/channels/knobs, optionally write a
    thin --config (only when channels/semantics differ from defaults), and print a
    runnable command. Never runs inference. Caller guards the TTY first (§8).
    """
    from custom_sam_peft.data.channel_semantics import CHANNEL_SEMANTIC_NAMES

    # P1 — checkpoint (blank → baseline)
    checkpoint = ask_text(
        "Adapter checkpoint directory? Leave blank for baseline (no adapter).",
        default="",
        validate=lambda s: None if s == "" else validate_checkpoint_dir(s),
    )
    if checkpoint:
        pretty, base = peek_adapter(Path(checkpoint))
        typer.echo(f"detected adapter: {pretty}, base model: {base or '(unspecified)'}")

    # P2 — channels
    def _positive_int(s: str) -> str | None:
        try:
            return None if int(s) > 0 else "channels must be a positive integer"
        except ValueError:
            return "channels must be a positive integer"

    channels = int(ask_text("Number of input image channels?", default="3", validate=_positive_int))

    # P3 — channel semantics
    semantics = ask_choice(
        "Channel semantics?", list(CHANNEL_SEMANTIC_NAMES), default="rgb"
    )

    # P4 — merge (only when a checkpoint was given)
    merge = ask_confirm("Merge adapter weights before inference?", default=True) if checkpoint else True

    # P5 — score threshold
    def _unit(s: str) -> str | None:
        try:
            f = float(s)
        except ValueError:
            return "score must be a number in [0.0, 1.0]"
        return None if 0.0 <= f <= 1.0 else "score must be in [0.0, 1.0]"

    threshold = float(ask_text("Minimum score to keep a prediction [0.0-1.0]?", default="0.3", validate=_unit))

    # P6 — save-masks format
    save_masks = ask_choice("Mask output format?", ["rle", "png", "none"], default="rle")

    # P7 — visualize
    visualize = ask_confirm("Write per-image overlay PNGs?", default=False)

    # P8/P9/P10 — per-run args (required)
    images = ask_text("Images: dir / glob / manifest / single file?")
    prompts = ask_text("Class prompts (comma-separated) or path to a one-per-line file?")
    output = ask_text("Output directory?")

    # Thin config only when channels/semantics differ from predict defaults (§8.1).
    config_path: Path | None = None
    if channels != 3 or semantics != "rgb":
        thin = Path(_THIN_CONFIG_NAME)
        if thin.exists() and not force:
            raise typer.BadParameter(
                f"refusing to overwrite existing {thin}; pass --force",
                param_hint="--config",
            )
        thin.write_text(
            f"# Generated by `custom-sam-peft predict --interactive` on {date.today().isoformat()}\n"
            f"model:\n  name: {_PREDICT_DEFAULT_MODEL}\n"
            f"data:\n  channels: {channels}\n  channel_semantics: {semantics}\n"
        )
        config_path = thin

    # Assemble + print the runnable command (shell-safe).
    parts = [
        "custom-sam-peft",
        "predict",
        "--images",
        shlex.quote(images),
        "--prompts",
        shlex.quote(prompts),
        "--output",
        shlex.quote(output),
    ]
    if checkpoint:
        parts += ["--checkpoint", shlex.quote(checkpoint)]
        parts += ["--merge-adapter"] if merge else ["--no-merge-adapter"]
    parts += ["--score-threshold", str(threshold), "--save-masks", save_masks]
    if visualize:
        parts += ["--visualize"]
    if config_path is not None:
        parts += ["--config", shlex.quote(str(config_path))]
    typer.echo(" ".join(parts))
    typer.echo(
        "note: --top-k, --device, --dtype, --batch-size, --seed stay at defaults; "
        "append them as flags if you need to override them."
    )
```

> The thin config is intentionally NOT validated via `load_config` — `predict/runner.py::_resolve_config` parses it with `yaml.safe_load` and reads only `model.name` / `data.channels` / `data.channel_semantics` (§8.1). `ask_choice` already constrains `semantics ∈ CHANNEL_SEMANTIC_NAMES`, so the runtime check at `predict/runner.py:181-185` can never fire. Per-run args use `shlex.quote` for shell-safe rendering (§8.2).

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/unit/cli/test_predict_interactive.py -v`
Expected: PASS — baseline RGB prints a `--config`-free, `--checkpoint`-free command and writes nothing; checkpoint path adds `--checkpoint`/`--merge-adapter` and prints the peek; non-RGB writes a thin config with exactly the three keys and references it via `--config`; RGB writes nothing; `--visualize` appears only when chosen; overwrite-without-force refuses.

- [ ] **Step 5: Commit**

```bash
git add src/custom_sam_peft/cli/_interactive.py tests/unit/cli/test_predict_interactive.py
git commit -m "feat(predict): add predict --interactive command builder + thin config"
```

### Task 13: Wire `--interactive`/`-i` into `predict_cmd.py`

**Spec ref:** §8 (steps 1-3), §9 (TTY guard).

**Files:**

- Modify: `src/custom_sam_peft/cli/predict_cmd.py` (add `interactive` option + early branch)
- Test: `tests/unit/cli/test_predict_cmd.py` (new)

> Edits `predict_cmd.py`. Depends on Task 12. File-disjoint from eval wiring.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/cli/test_predict_cmd.py`:

```python
"""Tests for the `csp predict` CLI option surface (CPU-only)."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from custom_sam_peft.cli.main import app

runner = CliRunner()


def test_predict_interactive_dispatches_helper(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("custom_sam_peft.cli._interactive.sys.stdin.isatty", lambda: True)
    called: list[dict] = []
    monkeypatch.setattr(
        "custom_sam_peft.cli._interactive.run_predict_interactive",
        lambda **kw: called.append(kw),
    )
    result = runner.invoke(app, ["predict", "--interactive"])
    assert result.exit_code == 0, result.output
    assert len(called) == 1


def test_predict_interactive_non_tty_hard_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("custom_sam_peft.cli._interactive.sys.stdin.isatty", lambda: False)
    called: list[dict] = []
    monkeypatch.setattr(
        "custom_sam_peft.cli._interactive.run_predict_interactive",
        lambda **kw: called.append(kw),
    )
    result = runner.invoke(app, ["predict", "--interactive"])
    assert result.exit_code != 0
    assert "tty" in result.output.lower()
    assert called == []
```

> `predict`'s `--images`/`--prompts`/`--output` are required options, so a plain `predict --interactive` (without them) would normally be a usage error. The early `interactive` branch must run BEFORE Typer enforces the other required options. Typer enforces required options at parse time, so making the early branch work requires the other options to be NON-required when `--interactive` is set. **Resolved ambiguity:** keep the flag surface unchanged (§1 out-of-scope: "No change to the `csp predict` CLI flag surface" — only the additive `-i`). To square this with required `--images/--prompts/--output`, the test invokes with dummy values OR the implementation makes those three `Optional` with `None` defaults and validates-presence only on the non-interactive path. Per §1 ("untouched apart from the additive `-i` flag"), DO NOT change their requiredness. Instead the test passes them: invoke `["predict", "--interactive", "--images", "x", "--prompts", "y", "--output", "z"]`. Update both tests above to include those three dummy options so the parse succeeds and the early `interactive` branch fires. (The helper ignores the flag-driven values and prompts for everything.)

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/unit/cli/test_predict_cmd.py -k interactive -v`
Expected: FAIL — `predict` has no `--interactive` flag.

- [ ] **Step 3: Add the flag + early branch**

In `src/custom_sam_peft/cli/predict_cmd.py`, add an `interactive` option to `predict(...)` (after `verbose`):

```python
    interactive: bool = typer.Option(
        False,
        "--interactive",
        "-i",
        help="Build a runnable predict command interactively (prompts for all inputs).",
    ),
```

At the top of the `predict` body, right after `configure_logging(verbose)` and BEFORE building `PredictOptions`:

```python
    if interactive:
        from custom_sam_peft.cli import _interactive

        _interactive.require_tty()
        _interactive.run_predict_interactive(force=False)
        return
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/unit/cli/test_predict_cmd.py -v`
Expected: PASS — `--interactive` dispatches the helper after the TTY guard; non-TTY hard-errors before the helper runs; existing predict flag surface untouched.

- [ ] **Step 5: Commit**

```bash
git add src/custom_sam_peft/cli/predict_cmd.py tests/unit/cli/test_predict_cmd.py
git commit -m "feat(predict): wire --interactive/-i flag with TTY guard"
```

---

## REVIEW CHECKPOINT B — all helpers complete

- [ ] Run: `uv run pytest tests/unit/cli/ tests/unit/test_eval_runner.py tests/unit/test_eval_runner_gate.py tests/unit/test_train_checkpoint.py tests/predict/test_adapter_detect.py tests/unit/test_peft_method_protocol.py -q`
      Expected: all PASS.
- [ ] Run: `! grep -rn "make_peft_method\|\.load_from_disk(" src/custom_sam_peft/eval/runner.py`
      Expected: no matches — eval no longer dispatches via `make_peft_method(cfg.peft.method).load_from_disk`; it uses `train.checkpoint.load_adapter` (sentinel) + the advisory `discover_method_from_checkpoint` comparison.
- [ ] Run: `! grep -n "import init_cmd\|import setup_wizard\|import eval_cmd\|import predict_cmd" src/custom_sam_peft/cli/_interactive.py`
      Expected: no matches at module scope (the only reverse imports — `setup_wizard.render`/`emit` in `_eval_baseline` — are lazy, inside the function body, per §2).
- [ ] Dispatch a code-review subagent (min sonnet/high; opus/xhigh for the eval-runner dispatch + `_interactive` seam, which are design-sensitive): confirm (a) eval infers the method from the checkpoint and only uses `cfg.peft.method` for the advisory warning, (b) the baseline path loads no adapter and falls back the output dir without crashing, (c) `_interactive` has no module-scope cycle into the CLI command modules, (d) the three helpers emit the right artifacts (init writes a train config, eval-reuse writes nothing, eval-baseline writes a validated config, predict writes a thin config only for non-RGB), (e) `EvalArtifacts.peft_method` is untouched.

---

## Phase 7 — Final verification (do not run during planning; these are plan steps)

### Task 14: Full-suite + lint + type + markdown verification

**Files:** none (verification only).

- [ ] **Step 1: Ruff lint**

Run: `uv run ruff check`
Expected: no findings (fix any before proceeding — common ones here: unused imports left in `setup_wizard.py` after the extraction, unused `cast`/`json` after the delegator refactors).

- [ ] **Step 2: Ruff format check**

Run: `uv run ruff format --check`
Expected: clean (run `uv run ruff format` to fix, then re-check).

- [ ] **Step 3: mypy**

Run: `uv run mypy src/custom_sam_peft`
Expected: no errors. (`_interactive.py` uses typed signatures; the `# type: ignore[assignment]` on `_ask_run_mode`'s `ctx.run_mode = ask_choice(...)` and the `cast(AdapterKind, ...)` in the delegator are pre-resolved in the task code.)

- [ ] **Step 4: FULL pytest suite (the 80% coverage gate only passes on the full suite)**

Run: `uv run pytest`
Expected: all PASS; `--cov-fail-under=80` satisfied. Do NOT run a subset for the gate — `addopts` enforces coverage across the whole run. If coverage dips below 80%, add focused CPU tests for any uncovered branch in `_interactive.py` (e.g. the eval-reuse `test` split path, the predict baseline-with-checkpoint merge=no path).

- [ ] **Step 5: yamllint (templates/configs untouched, but run for safety)**

Run: `uv run --with yamllint yamllint -c .config/yamllint.yml configs/`
Expected: clean. This PR writes no shipped YAML (the thin predict config and baseline eval config are generated at runtime into the user's cwd, not committed); `configs/examples/*.yaml` are untouched.

- [ ] **Step 6: markdownlint the plan + spec**

Run: `npx --yes markdownlint-cli2 --config .config/markdownlint-cli2.jsonc "docs/superpowers/plans/2026-05-28-eval-predict-interactive-helpers-plan.md" "docs/superpowers/specs/2026-05-28-eval-predict-interactive-helpers-design.md"`
Expected: no findings (fix any before the ready PR; CI lints all tracked `.md`).

- [ ] **Step 7: Commit any lint/format/type fixups**

```bash
git add -A
git commit -m "chore: lint/format/type fixups for eval/predict interactive helpers"
```

---

## Self-review (against the spec, after writing the plan)

- **§1 in-scope files — all represented:** `_interactive.py` (Tasks 5, 6, 10, 12), `setup_wizard.py` (Tasks 5, 7), `init_cmd.py` (no behavioral change — confirmed unchanged; the `init -i` shrink lands in `setup_wizard.py` per §4, and `init_cmd` keeps exporting `UNIFIED_TEMPLATE`/`_build_loss_overrides_block`), `eval_cmd.py` (Tasks 8, 11), `eval/runner.py` (Task 9), `predict_cmd.py` (Task 13), `peft_adapters/__init__.py` (Tasks 1, 4), `predict/adapter_load.py` (Tasks 2, 4), `train/checkpoint.py` (Task 3). Tests in §10 all represented (below).
- **§10 tests — all represented:** §10.1 `_interactive` → `tests/unit/cli/test_interactive.py` (Tasks 5, 6); §10.2 init retarget → `tests/unit/cli/test_setup_wizard.py` (Tasks 5, 7); §10.3 `eval -i` → `tests/unit/cli/test_eval_interactive.py` (Task 10); §10.4 eval-runner → `tests/unit/test_eval_runner.py` + `test_eval_runner_gate.py` (Task 9); §10.5 `predict -i` → `tests/unit/cli/test_predict_interactive.py` (Task 12); §10.6 seam → `tests/unit/test_peft_method_protocol.py` + `tests/predict/test_adapter_detect.py` + `tests/unit/test_train_checkpoint.py` (Tasks 1-4). **Plus** new CLI-surface tests `tests/unit/cli/test_eval_cmd.py` (Tasks 8, 11) and `tests/unit/cli/test_predict_cmd.py` (Task 13) — not named in §10 but required to drive the additive flags. Integration `test_trainer_evaluator_seam.py`/`test_peft_extensibility.py`/`test_eval_artifacts.py` stay green (Task 9 Step 9; §7.6 keeps `EvalArtifacts.peft_method`).
- **TDD:** every behavioral task writes the failing test first, runs it red, implements, runs it green. Pure refactors (Tasks 2, 3) use characterization tests (assert behavior unchanged) — written first, kept green through the edit. All CPU-only; prompts driven by monkeypatching the `_interactive` primitives; no GPU tests added.
- **Dependency graph / sequencing:** Phase 1 (seam, §7) is foundational and lands first; Phase 2 (`_interactive`, §3) before all helpers; Phase 3 (`init -i` shrink, §4) serialized after Phase 2 (shared `setup_wizard.py`); Phase 4 (eval-runner, §6) depends on Phase 1; Phase 5 (`eval -i`, §5) depends on Phases 1/2/4; Phase 6 (`predict -i`, §8) depends on Phases 1/2. Shared-file serializations called out: `peft_adapters/__init__.py` (Tasks 1+4), `setup_wizard.py` (Tasks 5+7), `eval_cmd.py` (Tasks 8+11), `_interactive.py` (Tasks 5/6/10/12). File-disjoint parallel opportunities flagged: Tasks 2/3 (after Task 1), Task 8 ∥ Task 9, Phase 5 ∥ Phase 6 (except the `_interactive.py` append serialization).
- **Resolved ambiguities** recorded up front (helper home, baseline run.name, baseline printed split, thin-config path name, no model-name prompt, re-export surface, eval-baseline header label). None re-litigate the locked decisions (per-command helpers; keep `EvalArtifacts.peft_method`; baseline = no-adapter eval; thin predict config only when non-RGB; `discover_method_from_checkpoint` naming).
- **No migration:** the breaking-change note is stated up front (the dropped `init -i` eval mode, optional `--checkpoint`, PEFT-from-checkpoint), not a task.
- **Final phase** runs ruff/format/mypy, the FULL pytest+coverage suite, yamllint, and the markdownlint gate so the branch is PR-ready.
