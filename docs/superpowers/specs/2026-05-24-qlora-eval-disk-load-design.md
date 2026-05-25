# spec/qlora-eval-disk-load — QLoRA checkpoint load from disk in eval/runner.py (#98)

**Status:** Draft (2026-05-24)
**Tracking:** [#98](https://github.com/NguyenJus/custom-sam-peft/issues/98) — labeled `hardening-followup`, split from the #91–#100 sweep PR (see [`2026-05-22-hardening-followup-91-100-design.md`](2026-05-22-hardening-followup-91-100-design.md)).
**Scope:** Wire `run_eval` to dispatch QLoRA disk-load through a new `PEFTMethod.load_from_disk` protocol method; restore the channel adapter in the same path; flip two blocking tests and extend one GPU round-trip test.

**Builds on:**
- [`2026-05-21-hardening-audit-inventory.md`](2026-05-21-hardening-audit-inventory.md) Section J item 8 — the original audit that opened #98.
- [`2026-05-17-peft-qlora-design.md`](2026-05-17-peft-qlora-design.md) — QLoRA adapter design; `load_qlora` defined there.
- [`2026-05-23-n-channel-input-design.md`](2026-05-23-n-channel-input-design.md) — channel adapter (PR #135); `_load_channel_adapter` defined there.

---

## 1. Problem Statement

### 1.1 Current behavior

`run_eval` in `src/custom_sam_peft/eval/runner.py` raises a `ValueError` when the caller does not supply a pre-loaded wrapper (`model is None`) and the configured PEFT method is QLoRA:

```python
# runner.py:106-110
_peft_method = make_peft_method(resolved_peft_method)
if model is None and not _peft_method.supports_checkpoint_load_from_disk():
    raise ValueError(
        f"checkpoint loading currently supports only LoRA adapters; "
        f"got peft.method={resolved_peft_method!r}"
    )
```

Then, when `model is None`, the adapter load is hardcoded to LoRA:

```python
# runner.py:129-133
if model is None:
    wrapper = load_sam31(
        cfg.model, channels=cfg.data.channels, channel_semantics=cfg.data.channel_semantics
    )
    load_lora(wrapper, resolved_checkpoint)
```

Two consequences:
1. `custom-sam-peft eval --config cfg.yaml --checkpoint /path/to/qlora/ckpt` raises before building any model.
2. Even when the `ValueError` guard is removed, `load_lora` would be called on a QLoRA checkpoint — wrong loader.

Additionally, the channel adapter (N-channel input, PR #135) is never restored in the eval standalone-load path (`model is None`). Unlike the `predict` path (`predict/adapter_load.py:70`) and train-resume path (`train/checkpoint.py:122`), the eval path builds an N-channel base but silently drops the saved `channel_adapter.pt`. This is a latent bug for any non-RGB checkpoint (no-op for `channels=3`).

### 1.2 Why this is a wiring change, not new core logic

All the hard machinery already exists and is battle-tested:

| Component | Location | Status |
|---|---|---|
| `load_qlora(wrapper, dirpath)` | `src/custom_sam_peft/peft_adapters/qlora.py:385` | Fully implemented; tested by GPU round-trip test |
| `load_lora(wrapper, dirpath)` | `src/custom_sam_peft/peft_adapters/lora.py:148` | Fully implemented; already wired in eval |
| `_load_channel_adapter(wrapper, dir)` | `src/custom_sam_peft/train/checkpoint.py:57` | Already called in predict and train-resume paths |

The `predict` path already dispatches both LoRA and QLoRA via `predict/adapter_load.py::load_adapter`, but that function is tightly coupled to the predict CLI (it takes `AdapterKind` = `"lora"|"qlora"` resolved from file-system detection, and imports `typer`). The eval path has its own invariant: `cfg.peft.method` is already known, so dispatch through the `PEFTMethod` protocol is cleaner and respects the documented seam.

### 1.3 Protocol seam

`src/custom_sam_peft/peft_adapters/__init__.py:30-70` defines the `PEFTMethod` Protocol. The package docstring (lines 1–16) and `make_peft_method` docstring (line 135) both explicitly state: _"Trainers, evaluators, and checkpoint code call these methods instead of branching on `cfg.peft.method` strings."_ Adding `load_from_disk` to the protocol extends this seam to disk-load dispatch.

---

## 2. Locked Design Decisions

All three decisions below are final. Do not redesign; note genuine contradictions or gaps in §8.

### Decision 1 — New `PEFTMethod.load_from_disk` protocol method

Add `load_from_disk(self, wrapper, dirpath) -> Sam3Wrapper` to the `PEFTMethod` Protocol and implement it on both adapter classes. Implementations import `load_lora` / `load_qlora` lazily inside the method body (to preserve the isolation contract: LoRA-only users must never import `bitsandbytes`).

**Why not reuse `predict/adapter_load.py`:** that function detects the kind from the filesystem and imports `typer`; `run_eval` already knows the method from `cfg.peft.method`. Protocol dispatch is cleaner and avoids a cross-subsystem import.

**Why not string-branch in `runner.py`:** the protocol seam exists precisely to prevent this pattern; adding a `if method == "qlora"` branch in runner would be a regression.

### Decision 2 — Restore channel adapter in the eval disk-load path

Call `_load_channel_adapter(wrapper, resolved_checkpoint)` in `run_eval` after `_peft_method.load_from_disk(...)`. This is orthogonal to PEFT method — it is **not** baked into `load_from_disk`. Callers of `load_from_disk` in other contexts (if any) should call `_load_channel_adapter` separately if needed.

**Why here, not inside `load_from_disk`:** `load_from_disk` is a pure PEFT-adapter concern; channel-adapter restore is a storage-layout concern that already lives in `run_eval`'s caller scope. Mixing them would break the single-responsibility of the protocol method.

### Decision 3 — Tests

Three buckets: CPU unit tests (new + flipped), GPU round-trip extension (existing test, no new GPU test file).

---

## 3. File-by-File Changes

### 3.1 `src/custom_sam_peft/peft_adapters/__init__.py`

**Current import structure (lines 1–24):**
```python
from __future__ import annotations

from pathlib import Path
from typing import Protocol, cast, runtime_checkable

from custom_sam_peft._registry import RegistryError, lookup, register
from custom_sam_peft.errors import CheckpointError
```

No `bitsandbytes` import at module level — confirmed. The isolation contract is already maintained. `load_lora` and `load_qlora` must be imported lazily inside the new method bodies.

#### Change A — Add `load_from_disk` to the `PEFTMethod` Protocol (lines 63–70)

Current `PEFTMethod` ends at:
```python
    def supports_checkpoint_load_from_disk(self) -> bool:
        """Return True if this method can load a checkpoint from disk without
        a pre-loaded model wrapper.

        LoRA returns True. QLoRA returns False (requires a live wrapper with
        quantized base; disk-only load is deferred to a follow-up PR).
        """
        ...
```

**After** — append the following method to the `PEFTMethod` Protocol body (after `supports_checkpoint_load_from_disk`, before the closing of the class):

```python
    def load_from_disk(self, wrapper: Any, dirpath: Any) -> Any:
        """Load a checkpoint from disk into a freshly-built wrapper.

        Rebuilds the PEFT-adapted model from the saved checkpoint directory
        (``dirpath``), mutating ``wrapper`` in place. Returns ``wrapper``.

        LoRA implementation delegates to ``load_lora(wrapper, dirpath)``.
        QLoRA implementation delegates to ``load_qlora(wrapper, dirpath)``,
        which reconstructs the 4-bit quantized base from saved metadata before
        loading the LoRA adapter weights.

        Both implementations import their respective loaders lazily inside the
        method body so that LoRA-only users never import bitsandbytes.
        """
        ...
```

`Any` is used for the type hints in the Protocol body to avoid a forward-reference cycle (the `Sam3Wrapper` type lives in `models.sam3` and is not imported in `__init__.py`). The concrete implementations in `LoraAdapter` and `QloraAdapter` use `Sam3Wrapper` explicitly in their local imports.

Note: `Any` is already imported at line 21 (`from typing import Protocol, cast, runtime_checkable`). Add `Any` to that import.

#### Change B — Update `QloraAdapter.supports_checkpoint_load_from_disk` docstring and return value (lines 115–116)

Current:
```python
    def supports_checkpoint_load_from_disk(self) -> bool:
        return False
```

Current docstring on the Protocol method (lines 63–69):
```
LoRA returns True. QLoRA returns False (requires a live wrapper with
quantized base; disk-only load is deferred to a follow-up PR).
```

**After** — change to `return True` and update the Protocol docstring to remove the deferred-PR language. The updated Protocol docstring should read:

```
LoRA returns True. QLoRA returns True (load_qlora reconstructs the 4-bit
quantized base from saved custom_sam_peft_qlora.json metadata, then loads
the LoRA adapter weights via PeftModel.from_pretrained).
```

The `QloraAdapter.supports_checkpoint_load_from_disk` body becomes:
```python
    def supports_checkpoint_load_from_disk(self) -> bool:
        return True
```

#### Change C — Add `LoraAdapter.load_from_disk` implementation (after line 93)

Insert after the existing `LoraAdapter.supports_checkpoint_load_from_disk` method:

```python
    def load_from_disk(self, wrapper: Any, dirpath: Any) -> Any:
        from custom_sam_peft.peft_adapters.lora import load_lora

        return load_lora(wrapper, dirpath)
```

#### Change D — Add `QloraAdapter.load_from_disk` implementation (after line 116)

Insert after the updated `QloraAdapter.supports_checkpoint_load_from_disk` method:

```python
    def load_from_disk(self, wrapper: Any, dirpath: Any) -> Any:
        from custom_sam_peft.peft_adapters.qlora import load_qlora

        return load_qlora(wrapper, dirpath)
```

#### Change E — Update module docstring (lines 1–16)

The existing module docstring already documents `supports_checkpoint_load_from_disk` implicitly by listing the protocol methods. No new entry is strictly required for `load_from_disk` in the "Registered factories" table (it is a protocol method, not a factory). However, add a note after the registered factories block documenting the new method:

```
For disk-load dispatch in evaluators/CLI:
  ``_peft_method.load_from_disk(wrapper, dirpath)``  → delegates to load_lora or load_qlora
```

This mirrors the existing documentation style.

---

### 3.2 `src/custom_sam_peft/eval/runner.py`

**Current imports (lines 1–24):**

```python
from custom_sam_peft._registry import lookup
from custom_sam_peft.config.schema import TrainConfig
from custom_sam_peft.data.base import Dataset
from custom_sam_peft.data.val_source import resolve_val_source
from custom_sam_peft.eval._artifacts import EvalArtifacts
from custom_sam_peft.eval.evaluator import Evaluator
from custom_sam_peft.eval.metrics import MetricsReport
from custom_sam_peft.models.sam3 import MULTIPLEX_CAP, load_sam31
from custom_sam_peft.peft_adapters import make_peft_method
from custom_sam_peft.peft_adapters.lora import load_lora
```

#### Change A — Remove the `load_lora` import; add `_load_channel_adapter` import (lines 23–24)

Remove:
```python
from custom_sam_peft.peft_adapters.lora import load_lora
```

Add:
```python
from custom_sam_peft.train.checkpoint import _load_channel_adapter
```

`load_lora` is no longer called directly in `runner.py` — dispatch goes through `_peft_method.load_from_disk`. The `_load_channel_adapter` private helper is an intentional shared seam (it is already imported the same way in `predict/adapter_load.py:68`).

#### Change B — Remove the `ValueError` guard (lines 106–110)

Current:
```python
    _peft_method = make_peft_method(resolved_peft_method)
    if model is None and not _peft_method.supports_checkpoint_load_from_disk():
        raise ValueError(
            f"checkpoint loading currently supports only LoRA adapters; "
            f"got peft.method={resolved_peft_method!r}"
        )
```

**After:**
```python
    _peft_method = make_peft_method(resolved_peft_method)
```

The guard is deleted entirely. `supports_checkpoint_load_from_disk` now returns `True` for both LoRA and QLoRA; the guard would always pass and adds no value. Remove it.

#### Change C — Replace `load_lora` call with `_peft_method.load_from_disk` and add channel-adapter restore (lines 129–133)

Current:
```python
    if model is None:
        wrapper = load_sam31(
            cfg.model, channels=cfg.data.channels, channel_semantics=cfg.data.channel_semantics
        )
        load_lora(wrapper, resolved_checkpoint)
```

**After:**
```python
    if model is None:
        wrapper = load_sam31(
            cfg.model, channels=cfg.data.channels, channel_semantics=cfg.data.channel_semantics
        )
        _peft_method.load_from_disk(wrapper, resolved_checkpoint)
        _load_channel_adapter(wrapper, resolved_checkpoint)
```

`_load_channel_adapter` is a no-op when `channel_adapter.pt` is absent (RGB checkpoints), so existing LoRA tests and checkpoints without a channel adapter are unaffected.

#### Change D — Update `run_eval` docstring

The docstring currently includes in the `Raises` section:
```
ValueError: cfg.peft.method != 'lora' AND model is None (QLoRA load
    from disk is not yet supported; pre-loaded wrappers bypass this).
```

Remove this entry entirely. QLoRA disk-load is now supported. The remaining `Raises` entries (`ValueError: split == 'test' and cfg.data.test is None` and `ValueError: neither checkpoint nor artifacts provided`) are unchanged.

---

### 3.3 `src/custom_sam_peft/peft_adapters/lora.py` — No changes

`lora.py` is not modified. The new `LoraAdapter.load_from_disk` delegates to the existing `load_lora` function defined at line 148. No changes needed in `lora.py` itself.

---

### 3.4 `src/custom_sam_peft/peft_adapters/qlora.py` — No changes

`qlora.py` is not modified. The new `QloraAdapter.load_from_disk` delegates to the existing `load_qlora` function defined at line 385. No changes needed in `qlora.py` itself.

---

### 3.5 `src/custom_sam_peft/train/checkpoint.py` — No changes

`checkpoint.py` is not modified. `_load_channel_adapter` (defined at line 57) is imported directly into `runner.py` as a shared helper. Its signature and semantics are unchanged.

---

## 4. Test Plan

### 4.1 Rationale for CPU-mock strategy

`bitsandbytes` 4-bit quantization is a GPU/real-model failure mode (NF4 kernels require CUDA). All new CPU tests mock `load_qlora` and `load_lora` at the point of call rather than exercising the real quantization path. The existing GPU round-trip test (`tests/integration/test_peft_qlora_real.py`) already covers the real `load_qlora` path; this spec extends that test rather than duplicating it.

### 4.2 CPU unit tests — `tests/unit/test_peft_method_protocol.py`

All changes in this file are in-place edits; no new test file is created.

#### Change A — Flip `test_qlora_adapter_supports_checkpoint_load_from_disk_false` (line 106)

Current (line 106–107):
```python
def test_qlora_adapter_supports_checkpoint_load_from_disk_false() -> None:
    assert QloraAdapter().supports_checkpoint_load_from_disk() is False
```

**After** — rename and flip the assertion:
```python
def test_qlora_adapter_supports_checkpoint_load_from_disk_true() -> None:
    assert QloraAdapter().supports_checkpoint_load_from_disk() is True
```

This mirrors the existing `test_lora_adapter_supports_checkpoint_load_from_disk_true` at line 70.

#### Change B — Add protocol structure test for `load_from_disk`

After the existing `test_peft_method_protocol_declares_supports_checkpoint_load_from_disk` test (line 49–50), add:

```python
def test_peft_method_protocol_declares_load_from_disk() -> None:
    assert hasattr(PEFTMethod, "load_from_disk")
```

#### Change C — Add `LoraAdapter.load_from_disk` delegation test

Add after the existing LoraAdapter tests (after line 87), in the section marked `# 2. LoraAdapter`:

```python
def test_lora_adapter_load_from_disk_delegates_to_load_lora(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """LoraAdapter.load_from_disk must call load_lora with (wrapper, dirpath) and return its result."""
    from unittest.mock import MagicMock

    fake_wrapper = MagicMock()
    sentinel = MagicMock()

    monkeypatch.setattr(
        "custom_sam_peft.peft_adapters.lora.load_lora",
        lambda w, d: (sentinel if w is fake_wrapper and d == tmp_path else None),
    )

    result = LoraAdapter().load_from_disk(fake_wrapper, tmp_path)
    assert result is sentinel
```

Note: because `LoraAdapter.load_from_disk` uses a lazy import (`from custom_sam_peft.peft_adapters.lora import load_lora`), monkeypatching must target the attribute on the module rather than the method's local name. Patch `custom_sam_peft.peft_adapters.lora.load_lora` (the module-level name that the lazy import resolves to).

#### Change D — Add `QloraAdapter.load_from_disk` delegation test

Add after the existing QloraAdapter tests (after line 122), in the section marked `# 3. QloraAdapter`:

```python
def test_qlora_adapter_load_from_disk_delegates_to_load_qlora(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """QloraAdapter.load_from_disk must call load_qlora with (wrapper, dirpath) and return its result."""
    from unittest.mock import MagicMock

    fake_wrapper = MagicMock()
    sentinel = MagicMock()

    monkeypatch.setattr(
        "custom_sam_peft.peft_adapters.qlora.load_qlora",
        lambda w, d: (sentinel if w is fake_wrapper and d == tmp_path else None),
    )

    result = QloraAdapter().load_from_disk(fake_wrapper, tmp_path)
    assert result is sentinel
```

Same lazy-import note as Change C above applies here.

---

### 4.3 CPU unit tests — `tests/unit/test_eval_runner.py`

All changes are in-place edits to the existing file.

#### Change A — Flip `test_run_eval_rejects_non_lora_peft` (line 35–38)

Current:
```python
def test_run_eval_rejects_non_lora_peft(tmp_path: Path) -> None:
    cfg = _make_cfg(peft_method="qlora")
    with pytest.raises(ValueError, match="lora"):
        run_eval(cfg, checkpoint=tmp_path, split="val")
```

**After** — rename and flip to assert QLoRA dispatches successfully:

```python
def test_run_eval_dispatches_qlora_from_disk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """run_eval with peft_method='qlora' and model=None must dispatch via load_from_disk
    (calling load_qlora) and call _load_channel_adapter, without raising."""
    from unittest.mock import MagicMock

    cfg = _make_cfg(peft_method="qlora")

    qlora_loader_calls: list[tuple[object, object]] = []
    channel_adapter_calls: list[tuple[object, object]] = []

    def fake_load_qlora(wrapper: object, dirpath: object) -> object:
        qlora_loader_calls.append((wrapper, dirpath))
        return wrapper

    def fake_load_channel_adapter(wrapper: object, dirpath: object) -> None:
        channel_adapter_calls.append((wrapper, dirpath))

    monkeypatch.setattr(
        "custom_sam_peft.peft_adapters.qlora.load_qlora", fake_load_qlora
    )
    monkeypatch.setattr(
        "custom_sam_peft.eval.runner._load_channel_adapter", fake_load_channel_adapter
    )
    monkeypatch.setattr(
        "custom_sam_peft.eval.runner.lookup",
        lambda *_a, **_kw: lambda *a, **kw: MagicMock(__len__=lambda self: 0, class_names=[]),
    )
    monkeypatch.setattr("custom_sam_peft.eval.runner.load_sam31", lambda _m, **_kw: MagicMock())
    fake_report = MagicMock()
    monkeypatch.setattr(
        "custom_sam_peft.eval.runner.Evaluator",
        lambda _cfg: MagicMock(evaluate_and_save=MagicMock(return_value=fake_report)),
    )

    result = run_eval(cfg, checkpoint=tmp_path, split="val", output_dir=tmp_path)

    assert result is fake_report
    assert len(qlora_loader_calls) == 1, "load_qlora must be called exactly once"
    assert len(channel_adapter_calls) == 1, "_load_channel_adapter must be called exactly once"
    # Verify dirpath is the resolved checkpoint.
    _, dirpath = qlora_loader_calls[0]
    assert dirpath == tmp_path
```

#### Change B — Add `_load_channel_adapter` assertion for the LoRA path

The existing `test_run_eval_dispatches_dataset_via_registry` test (lines 47–79) monkeypatches `load_lora` but does not assert `_load_channel_adapter` is called. Extend it (or add a separate test) to verify the channel adapter is restored on the LoRA path too.

Add a new test after `test_run_eval_rejects_test_split_when_data_test_none`:

```python
def test_run_eval_lora_calls_load_channel_adapter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """run_eval with peft_method='lora' and model=None must call _load_channel_adapter."""
    from unittest.mock import MagicMock

    cfg = _make_cfg(peft_method="lora")
    channel_adapter_calls: list[tuple[object, object]] = []

    monkeypatch.setattr(
        "custom_sam_peft.eval.runner._load_channel_adapter",
        lambda wrapper, dirpath: channel_adapter_calls.append((wrapper, dirpath)),
    )
    monkeypatch.setattr(
        "custom_sam_peft.eval.runner.lookup",
        lambda *_a, **_kw: lambda *a, **kw: MagicMock(__len__=lambda self: 0, class_names=[]),
    )
    monkeypatch.setattr("custom_sam_peft.eval.runner.load_sam31", lambda _m, **_kw: MagicMock())
    fake_report = MagicMock()
    monkeypatch.setattr(
        "custom_sam_peft.eval.runner.Evaluator",
        lambda _cfg: MagicMock(evaluate_and_save=MagicMock(return_value=fake_report)),
    )

    run_eval(cfg, checkpoint=tmp_path, split="val", output_dir=tmp_path)

    assert len(channel_adapter_calls) == 1, "_load_channel_adapter must be called exactly once"
    _, dirpath = channel_adapter_calls[0]
    assert dirpath == tmp_path
```

**Important — REQUIRED fix, not optional:** Six existing tests patch `custom_sam_peft.eval.runner.load_lora` (lines 63, 101, 159, 230, 290, 328 in `tests/unit/test_eval_runner.py`). After Change A in §3.2 removes the `load_lora` import from `runner.py`, that attribute no longer exists in the `runner` module namespace. `monkeypatch.setattr` with a string target **raises `AttributeError` by default** (`raising=True`) when the attribute is absent — so all six tests will **error at collection/setup time**, not silently no-op. The implementer MUST repoint all six patches to `custom_sam_peft.peft_adapters.lora.load_lora`. Because `LoraAdapter.load_from_disk` performs a lazy `from custom_sam_peft.peft_adapters.lora import load_lora` at call time, patching the name on the `lora` module is resolved correctly by the lazy import — so the mock is honored. (Alternatively, patch `custom_sam_peft.peft_adapters.__init__.LoraAdapter.load_from_disk` directly; repointing `load_lora` is preferred as the smaller diff.) This repoint applies to every one of the six listed tests, and the acceptance gate `uv run pytest -m "not gpu"` must be green afterward.

---

### 4.4 CPU unit test — `tests/unit/test_cli.py`

#### Change A — Flip `test_eval_command_rejects_qlora_method` (lines 192–215)

Current (lines 192–215):
```python
def test_eval_command_rejects_qlora_method(tmp_path: Path) -> None:
    """custom_sam_peft eval --checkpoint errors when peft.method is not lora."""
    ...
    assert result.exit_code != 0
    assert "qlora" in _plain(result.output).lower() or "only lora" in _plain(result.output).lower()
```

**After** — rename and flip. The CLI command must no longer reject QLoRA. The simplest flip verifies the command either exits 0 (if the model-load happens cleanly with mocks) or exits with an error that is NOT about QLoRA rejection. Since the CLI calls `run_eval` which now calls `load_sam31` + `load_qlora`, which will fail without a real checkpoint, the expected exit code is non-zero (file-not-found from `load_qlora` trying to open `custom_sam_peft_qlora.json`) — but the error must NOT be the old "checkpoint loading currently supports only LoRA adapters" message.

```python
def test_eval_command_accepts_qlora_method(tmp_path: Path) -> None:
    """custom_sam_peft eval --checkpoint no longer rejects peft.method=qlora.

    The command will fail for other reasons (no real checkpoint on disk), but the
    failure must NOT be the old 'only LoRA adapters' guard. QLoRA is now accepted
    and dispatched via QloraAdapter.load_from_disk.
    """
    from custom_sam_peft.cli.main import app

    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        """
run: {name: t, output_dir: ./runs, seed: 0}
data:
  format: coco
  train: {annotations: t.json, images: t/}
  val: {annotations: v.json, images: v/}
  prompt_mode: text
peft: {method: qlora}
train: {epochs: 1}
"""
    )
    local_runner = CliRunner()
    result = local_runner.invoke(
        app,
        ["eval", "--config", str(cfg_path), "--checkpoint", str(tmp_path)],
    )
    # Must NOT contain the old rejection message.
    assert "checkpoint loading currently supports only LoRA" not in _plain(result.output)
    assert "only lora" not in _plain(result.output).lower()
```

---

### 4.5 GPU test extension — `tests/integration/test_peft_qlora_real.py`

**Extend `test_save_load_qlora_roundtrip` only. No new GPU test file.**

The existing test (lines 104–128) does:
1. `apply_qlora(w1, ...)` → `save_qlora(w1, tmp_path)` → snapshots LoRA param names+values → `del w1` + GC → `load_sam31(w2)` → `load_qlora(w2, tmp_path)` → asserts LoRA param-name+value parity (atol=0.0).

This already exercises `load_qlora` from disk. Extend it with a **forward-output-parity assertion** to confirm the reconstructed model computes the same outputs as the original.

**Memory constraint:** host RAM is ~12GB (WSL). The test already deletes `w1` before constructing `w2` for this reason. The forward output from `w1` must be captured BEFORE `del w1` and stored as a small tensor on CPU (not the model itself).

**Extension — insert before `del w1`:**

```python
    # Capture a forward output from w1 before deleting it.
    # The input must be deterministic and minimal: a single-image batch at
    # the smallest spatial size SAM 3.1 accepts (1, C, H, W) where C=3 (RGB),
    # H=W=1024 (SAM 3.1's canonical input size; see load_sam31 image_size default).
    # Use eval mode + fixed seed to suppress dropout and any stochastic ops.
    import torch as _torch

    _torch.manual_seed(0)
    w1.model.eval()
    _dummy_input = _torch.zeros(1, 3, 1024, 1024, device="cuda", dtype=_torch.float32)
    with _torch.no_grad():
        _out_w1 = w1.model(_dummy_input)
    # Store only the output tensor on CPU; do NOT keep w1 alive.
    _out_w1_cpu = _out_w1.detach().cpu() if isinstance(_out_w1, _torch.Tensor) else None
    # (If the model returns a dict/list, the implementer must adapt this to extract
    # a representative scalar tensor, e.g. _out_w1["pred_masks"][0].detach().cpu().)
```

**Extension — after `load_qlora(w2, tmp_path)`, append:**

```python
    # Forward-output parity: w2 must produce the same output as w1 on the same input.
    if _out_w1_cpu is not None:
        _torch.manual_seed(0)
        w2.model.eval()
        _dummy_input2 = _torch.zeros(1, 3, 1024, 1024, device="cuda", dtype=_torch.float32)
        with _torch.no_grad():
            _out_w2 = w2.model(_dummy_input2)
        _out_w2_cpu = _out_w2.detach().cpu() if isinstance(_out_w2, _torch.Tensor) else None
        assert _out_w2_cpu is not None, "w2 forward output unexpectedly None"
        assert _torch.allclose(_out_w1_cpu, _out_w2_cpu, atol=1e-4, rtol=1e-4), (
            f"forward output mismatch after load_qlora roundtrip; "
            f"max abs diff={(_out_w1_cpu - _out_w2_cpu).abs().max().item():.6f}"
        )
```

**Tolerance choice:** `atol=1e-4, rtol=1e-4` is appropriate for bfloat16/float16 computations after a quantize→save→dequant→load cycle. The saved LoRA weights have atol=0.0 (exact), but the 4-bit base re-quantization introduces small dequant rounding errors in the forward. The implementer must verify this tolerance is sufficient on the GTX 1080 (sm_61, float16 compute_dtype) and tighten or loosen as needed.

**Note on forward input shape:** The implementer must inspect `Sam3Wrapper.model.forward` (or `w1.model.forward`) to confirm the exact input signature. SAM 3.1's forward may expect a dict of inputs rather than a raw tensor. If so, adapt the dummy input accordingly. The above uses `w1.model(...)` to call the `_Sam3ImageAdapter.forward` or equivalent; the implementer should use the same call pattern that the eval `Evaluator` uses and adapt the dummy input to match.

The existing `@pytest.mark.skipif(not _bnb_available(), ...)`, `pytestmark` markers (`requires_checkpoint`, `requires_compatible_gpu`, `gpu_local`), and the `del w1 / gc.collect() / torch.cuda.empty_cache()` sequence are all **unchanged**.

---

## 5. Acceptance Criteria

The reviewer can verify this checklist mechanically:

- [ ] `QloraAdapter().supports_checkpoint_load_from_disk()` returns `True` (was `False`).
- [ ] `LoraAdapter().load_from_disk` and `QloraAdapter().load_from_disk` exist and pass `isinstance(adapter, PEFTMethod)` checks.
- [ ] `LoraAdapter().load_from_disk(wrapper, dirpath)` calls `load_lora(wrapper, dirpath)` (verified by unit test with mock).
- [ ] `QloraAdapter().load_from_disk(wrapper, dirpath)` calls `load_qlora(wrapper, dirpath)` (verified by unit test with mock).
- [ ] `run_eval` with `peft_method='qlora'` and `model=None` no longer raises `ValueError` (verified by renamed/flipped unit test).
- [ ] `run_eval` with `peft_method='qlora'` and `model=None` calls `load_qlora` (verified by unit test with mock).
- [ ] `run_eval` with `peft_method='qlora'` and `model=None` calls `_load_channel_adapter(wrapper, resolved_checkpoint)` (verified by unit test with mock).
- [ ] `run_eval` with `peft_method='lora'` and `model=None` calls `_load_channel_adapter(wrapper, resolved_checkpoint)` (verified by unit test with mock).
- [ ] `run_eval` with `model=<pre-loaded>` does NOT call `_load_channel_adapter` (the `if model is None` guard remains; verified by existing `test_run_eval_accepts_prebuilt_val_dataset_and_model`).
- [ ] `custom_sam_peft eval --config <qlora-cfg.yaml> --checkpoint <dir>` does NOT produce "checkpoint loading currently supports only LoRA" in its output (verified by renamed/flipped CLI test).
- [ ] `peft_adapters/__init__.py` does not import `bitsandbytes` at module level (verify: `grep "bitsandbytes" src/custom_sam_peft/peft_adapters/__init__.py` returns no results).
- [ ] `peft_adapters/__init__.py` does not import `load_lora` or `load_qlora` at module level (both remain lazy inside the new method bodies).
- [ ] `tests/unit/test_peft_method_protocol.py::test_qlora_adapter_supports_checkpoint_load_from_disk_true` passes.
- [ ] `tests/unit/test_peft_method_protocol.py::test_eval_runner_does_not_branch_on_method_name` still passes (runner.py must contain no `.method ==` literal).
- [ ] `uv run pytest -m "not gpu"` — fully green (all CPU tests pass) with coverage ≥ 80% (the `--cov-fail-under=80` gate in `pyproject.toml addopts`; do NOT pass `--no-cov` on this final run).
- [ ] `uv run ruff check src/ tests/` — clean.
- [ ] `uv run ruff format --check src/ tests/` — clean.
- [ ] `uv run mypy src/custom_sam_peft` — clean (CI gate, ci.yml:44; covers the new `load_from_disk` Protocol method + `Any`-typed implementations).
- [ ] (GPU only) `tests/integration/test_peft_qlora_real.py::test_save_load_qlora_roundtrip` passes with the forward-output-parity assertion added.

---

## 6. Out of Scope

The following are explicitly excluded from this PR:

- **No changes to the training path.** Training already dispatches correctly through `train/checkpoint.py::save_adapter` / `load_adapter`.
- **No changes to the predict path.** `predict/adapter_load.py::load_adapter` already supports QLoRA via file-system detection; this PR does not touch it.
- **No new GPU test file.** The extension is to the existing `test_peft_qlora_real.py`.
- **No changes to `detect_method_from_checkpoint` semantics.** It remains unchanged on both adapter classes.
- **No `cfg.qlora.upcast_norms` knob.** Out of scope per `qlora.py:209` comment.
- **No changes to `EvalArtifacts` seam.** The `artifacts` path in `run_eval` (lines 94–103) is unchanged; it already receives a resolved `peft_method` string and the same `_peft_method.load_from_disk` dispatch will apply when `model is None` (which is rare in that path, since `custom_sam_peft run` always passes `model=<pre-loaded>`).
- **No changes to `predict/adapter_load.py`.** The predict path uses file-system detection (`custom_sam_peft_qlora.json` presence) rather than config-driven dispatch; this design difference is intentional and is unchanged.

---

## 7. PR Shape & Commit Sequencing

Single PR on branch `worktree-qlora-eval-disk-load-98`, targets `main`. Two commits, ordered by test coverage then cleanup:

| # | Commit message | Touches |
|---|---|---|
| 1 | `feat(peft): add load_from_disk protocol method; wire QLoRA eval disk-load + channel adapter (#98)` | `src/custom_sam_peft/peft_adapters/__init__.py`, `src/custom_sam_peft/eval/runner.py` |
| 2 | `test(peft): flip qlora-reject tests; add load_from_disk + channel-adapter assertions; extend GPU roundtrip (#98)` | `tests/unit/test_peft_method_protocol.py`, `tests/unit/test_eval_runner.py`, `tests/unit/test_cli.py`, `tests/integration/test_peft_qlora_real.py` |

**PR description footer:**
```
Closes #98 — QLoRA checkpoint disk-load in eval/runner.py via PEFTMethod.load_from_disk.
```

---

## 8. Open Questions

None. The design is locked. If the implementer encounters a contradiction between this spec and the code, note it here and surface to the spec author before proceeding.

---

## 9. Implementation Plan

This section is the seam consumed by `superpowers:writing-plans`.

**Step 1.** In `src/custom_sam_peft/peft_adapters/__init__.py`:
- Add `Any` to the `from typing import ...` import (line 21).
- Add `load_from_disk` method to `PEFTMethod` Protocol per §3.1 Change A.
- Flip `QloraAdapter.supports_checkpoint_load_from_disk` to return `True` and update Protocol docstring per §3.1 Change B.
- Add `LoraAdapter.load_from_disk` per §3.1 Change C.
- Add `QloraAdapter.load_from_disk` per §3.1 Change D.
- Update module docstring per §3.1 Change E.

**Step 2.** In `src/custom_sam_peft/eval/runner.py`:
- Replace `from custom_sam_peft.peft_adapters.lora import load_lora` with `from custom_sam_peft.train.checkpoint import _load_channel_adapter` per §3.2 Change A.
- Delete the `ValueError` guard (lines 106–110) per §3.2 Change B.
- Replace `load_lora(wrapper, resolved_checkpoint)` with `_peft_method.load_from_disk(wrapper, resolved_checkpoint)` followed by `_load_channel_adapter(wrapper, resolved_checkpoint)` per §3.2 Change C.
- Update `run_eval` docstring per §3.2 Change D.

**Step 3.** Commit message per §7 row 1. Run `uv run pytest tests/unit/ -m "not gpu"` to confirm green before proceeding.

**Step 4.** In `tests/unit/test_peft_method_protocol.py`:
- Flip and rename `test_qlora_adapter_supports_checkpoint_load_from_disk_false` per §4.2 Change A.
- Add `test_peft_method_protocol_declares_load_from_disk` per §4.2 Change B.
- Add `test_lora_adapter_load_from_disk_delegates_to_load_lora` per §4.2 Change C.
- Add `test_qlora_adapter_load_from_disk_delegates_to_load_qlora` per §4.2 Change D.

**Step 5.** In `tests/unit/test_eval_runner.py`:
- Flip and rename `test_run_eval_rejects_non_lora_peft` per §4.3 Change A.
- Add `test_run_eval_lora_calls_load_channel_adapter` per §4.3 Change B.
- **Required:** repoint all six existing patches of `custom_sam_peft.eval.runner.load_lora` (lines 63, 101, 159, 230, 290, 328) to `custom_sam_peft.peft_adapters.lora.load_lora` — they will otherwise error with `AttributeError` once the import is removed (see §4.3 Change B note).

**Step 6.** In `tests/unit/test_cli.py`:
- Flip and rename `test_eval_command_rejects_qlora_method` per §4.4 Change A.

**Step 7.** In `tests/integration/test_peft_qlora_real.py`:
- Extend `test_save_load_qlora_roundtrip` with forward-output capture before `del w1` and parity assertion after `load_qlora(w2, ...)` per §4.5. Adapt the dummy input shape to match the actual `Sam3Wrapper` forward signature (inspect before implementing).

**Step 8.** Commit message per §7 row 2. Run the full manual gate per §5.

**Step 9.** Open the PR with the description footer per §7.
