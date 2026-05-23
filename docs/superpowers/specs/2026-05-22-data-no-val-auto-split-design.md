# spec/data-no-val-auto-split — No-val mode & train→train/val auto-split (issue #71)

**Status:** Draft (2026-05-22)
**Tracking:** [#71](https://github.com/NguyenJus/custom-sam-peft/issues/71) — *feat(data): support missing validation set — no-val mode + train→train/val auto-split*
**Scope:** Allow training without a pre-supplied validation set, in two modes:
1. **No-val** — user opts out of validation; training proceeds with no eval-during-training; the end-of-run eval, image panel, and bundle samples are skipped.
2. **Auto-split** — user supplies only `data.train`; the loader carves train into train+val for the run using deterministic Sechidis-style iterative multi-label stratification.

`data.val` becomes optional; a new `data.val_split` block selects auto-split; setting neither resolves to no-val mode with a single WARN. The split is recorded once per run in `<run_dir>/val_source.json` and is authoritative on resume.

**Builds on:** [`2026-05-16-data-loading-design.md`](2026-05-16-data-loading-design.md) (COCO/HF adapters, `Dataset` protocol), [`2026-05-17-training-loop-design.md`](2026-05-17-training-loop-design.md) (Trainer lifecycle, cfg-drift WARN precedent in `train/checkpoint.py::load_full_state` line 167), [`2026-05-17-eval-design.md`](2026-05-17-eval-design.md) (`Evaluator`), [`2026-05-18-cli-design.md`](2026-05-18-cli-design.md) (runner/CLI layering), [`2026-05-18-simplify-ux-design.md`](2026-05-18-simplify-ux-design.md) (`custom_sam_peft run` orchestration + bundle).

---

## 1. Goals & Non-Goals

### 1.1 Goals

- Make `data.val` optional. `val: null`, `val:` omitted, and absence of both `val` and `val_split` all resolve to **no-val mode**, identically.
- Add `data.val_split: { fraction, seed }`. Setting it triggers a deterministic, multi-label-stratified carve of `data.train` into train+val for the run.
- A single source of truth — `<run_dir>/val_source.json` — records the resolved split and is reused verbatim on resume. The split survives dataset edits between runs by design.
- No-val mode degrades gracefully: `eval_every` is a no-op, no image panels, no end-of-run eval, no bundle samples; `summary.md` still ships.
- `csp doctor --config <yaml>` reports the resolved val mode without materializing datasets.

### 1.2 Non-goals (deferred — see §8)

- Folder dataset adapter (#33) — adopts the same hooks when it lands.
- Multiplex training (#22) — per-source split policy decided in #22's own design.
- Cross-validation / k-fold.
- Test-set auto-split (`data.test` stays explicit-only).
- Strict per-image stability across dataset edits (hash-based assignment). The run-dir saved split is the practical reproducibility hook.
- "Best-val" checkpoint selection. No such mechanism exists in the trainer today (`train/trainer.py` line 207–220 writes every `save_every` boundary unconditionally); nothing to fall back from.

---

## 2. Architecture overview

Two new modules, eight modified.

| File | Disposition | Purpose |
| --- | --- | --- |
| `src/custom_sam_peft/data/splitter.py` | **New** | Pure helpers: `stratified_split(items, fraction, seed) → SplitResult`. Sechidis 2011 iterative multi-label stratification. No IO, no torch, no torch-dataset deps. ~80 LOC. |
| `src/custom_sam_peft/data/val_source.py` | **New** | `ValSource` dataclass + `resolve_val_source(cfg, run_dir=None) → ValSource` resolver; `save_val_source` / `load_val_source` for `<run_dir>/val_source.json`; private `_enumerate_coco_items` / `_enumerate_hf_items` producing `list[SplittableItem]` from a `DataConfig` without decoding images. ~150 LOC. |
| `src/custom_sam_peft/config/schema.py` | Modify | `DataConfig.val: DataSplit \| None = None`; new `DataConfig.val_split: ValSplitConfig \| None = None`; new `ValSplitConfig` model; two `model_validator(mode="after")` checks (mutual exclusion + HF compatibility). |
| `src/custom_sam_peft/data/coco.py` | Modify | `COCODataset.__init__` accepts `image_ids: Iterable[int] \| None = None`; `build_coco` injects from `cfg["_resolved_image_ids"][pipeline]` when present. |
| `src/custom_sam_peft/data/hf.py` | Modify | `HFDataset.__init__` accepts `row_indices: Iterable[int] \| None = None`; `build_hf` injects symmetrically. |
| `src/custom_sam_peft/train/runner.py` | Modify | Orchestrates: `resolve_val_source` → `save_val_source` → inject `_resolved_image_ids` → build train/val datasets → pass `val_ds: Dataset \| None` to Trainer. |
| `src/custom_sam_peft/train/trainer.py` | Modify | `val_ds: Dataset \| None`; eval/panel/end-of-run-eval guards; `metrics.json` writes a "no validation set" note in no-val mode. |
| `src/custom_sam_peft/train/loop.py` | Modify (trivial) | Drop the dead-passed `val_ds` parameter from `run_epoch` (line 239); update the single call site in `train/trainer.py` line 246. |
| `src/custom_sam_peft/eval/runner.py` | Modify | New `--split val` guard symmetric with the existing `split == "test"` guard at line 81; standalone-eval auto-split support (call `resolve_val_source` and inject `_resolved_image_ids` when val isn't pre-built). |
| `src/custom_sam_peft/runs/bundle.py` | Modify | `write_bundle(ctx, metrics_report: MetricsReport \| None, val_dataset: Dataset \| None, model_wrapper)`. New `_write_summary_no_val(ctx)` path: writes `summary.md` only, no `samples/` directory. |
| `src/custom_sam_peft/cli/run_cmd.py` | Modify | Read saved `val_source.json` from `train_result.run_dir`; build val dataset only when mode != "none"; skip `run_eval` when val_dataset is None; pass through to `write_bundle` (which handles the None case). |
| `src/custom_sam_peft/cli/doctor_cmd.py` | Modify | Add `--config PATH` Typer option; new "Data" rich table when set. |
| `src/custom_sam_peft/diagnostics.py` | Modify | New `DataReport` dataclass; `DoctorReport.data: DataReport \| None`; `run_doctor(weights_path=..., config_path: Path \| None = None)`. |
| `configs/examples/coco_text_no_val.yaml` | **New** | Demonstrates no-val mode. |
| `configs/examples/coco_text_auto_split.yaml` | **New** | Demonstrates auto-split mode. |
| `src/custom_sam_peft/cli/templates/coco_text_lora.yaml`, `coco_text_qlora.yaml` | Modify | Add commented `val_split:` block under `data:`. |

### 2.1 Data flow (auto-split path)

```
load_config(cfg)
    │
    ▼
resolve_val_source(cfg, run_dir=<resume_dir or None>)
    │   ├─ run_dir / val_source.json exists  → load_val_source (resume path)
    │   └─ else                              → _enumerate_*_items → stratified_split
    ▼
save_val_source(vs, run_dir)              # writes <run_dir>/val_source.json
    │
    ▼
data_cfg_dict = cfg.data.model_dump()
data_cfg_dict["_resolved_image_ids"] = {"train": vs.train_ids, "eval": vs.val_ids}
    │
    ▼
train_ds = builder(data_cfg_dict, pipeline="train")   # builder picks ids out of dict
val_ds   = builder(data_cfg_dict, pipeline="eval")
    │
    ▼
Trainer(model, train_ds, val_ds, tracker, cfg).fit(run_dir=run_dir, resume_from=...)
```

For **explicit** mode the resolver returns `ValSource(mode="explicit", ...)` without enumerating items; the runner does not inject `_resolved_image_ids`, and the adapters take their existing fast path (full split).

For **none** mode the runner builds only the train dataset; `val_ds = None` flows through Trainer / `cli/run_cmd.py` / `runs/bundle.py`.

---

## 3. Schema

### 3.1 New types

```python
class ValSplitConfig(_Strict):
    """Auto-split parameters. Used when DataConfig.val_split is set.

    Carves data.train into train+val deterministically. In v0:
      - stratification is always-on Sechidis multi-label iterative;
        not configurable.
      - split unit is always 'image'; not configurable. Splitting by
        annotation can leak the same image into both sides.
    """

    fraction: float = Field(default=0.1, gt=0.0, le=0.5)
    seed: int | None = None  # None → inherit run.seed at resolve time
```

### 3.2 `DataConfig` diff

```python
class DataConfig(_Strict):
    format: DataFormat
    train: DataSplit
    val: DataSplit | None = None             # was: required DataSplit (schema.py line 132)
    val_split: ValSplitConfig | None = None  # new
    test: DataSplit | None = None
    hf: HFDatasetConfig | None = None
    prompt_mode: PromptMode
    image_size: PositiveInt = 1024
    augmentations: AugmentationsConfig = Field(default_factory=AugmentationsConfig)
    text_prompt: TextPromptConfig = Field(default_factory=TextPromptConfig)
    normalize: NormalizeConfig = Field(default_factory=NormalizeConfig)

    @model_validator(mode="after")
    def _check_format_specific(self) -> DataConfig:  # unchanged from line 141
        if self.format == "hf" and self.hf is None:
            raise ValueError("data.hf is required when data.format == 'hf'")
        return self

    @model_validator(mode="after")
    def _check_val_modes(self) -> DataConfig:
        if self.val is not None and self.val_split is not None:
            raise ValueError(
                "data.val and data.val_split are mutually exclusive. "
                "Set one to provide a validation set, neither for no-val mode."
            )
        return self

    @model_validator(mode="after")
    def _check_hf_split_val_compat(self) -> DataConfig:
        if (
            self.format == "hf"
            and self.val_split is not None
            and self.hf is not None
            and self.hf.split_val != "validation"
        ):
            raise ValueError(
                "data.hf.split_val cannot be customized when data.val_split is set; "
                "auto-split carves the val set from data.hf.split_train. "
                "Remove split_val or remove val_split."
            )
        return self
```

### 3.3 Resolved modes

| YAML state | Resolved mode |
| --- | --- |
| `data.val: {annotations: …, images: …}` | `explicit` |
| `data.val_split: {fraction: …}` | `auto_split` |
| `data.val: null`, `val:` key omitted, or neither `val` nor `val_split` | `none` (WARN once at training start) |
| Both `data.val` and `data.val_split` set | schema error from `_check_val_modes` |

`val: null` and the omitted-key case resolve **identically** to `none` — they both leave the field as Python `None` after pydantic load, and the resolver only inspects the runtime value. Documented in the example YAMLs.

### 3.4 Backwards compatibility

Making `val` optional is a strict broadening: every previously-validating config still validates with `val` present. No deprecation shim is needed. Configs that previously failed validation because of a missing `val` block (i.e., users who couldn't run before) now resolve to no-val mode.

---

## 4. Splitter algorithm (`data/splitter.py`)

Pure, no IO, no torch. Implemented in-tree rather than via `iterative-stratification` — ~80 LOC we can own; no narrow-purpose runtime dep; aligned with the project's existing pattern.

### 4.1 Public API

```python
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class SplittableItem:
    image_id: str                  # opaque string id (COCO int → str; HF row index → str)
    class_ids: frozenset[int]      # dense class ids present in this image


@dataclass(frozen=True)
class SplitResult:
    train_ids: tuple[str, ...]                       # sorted for determinism
    val_ids: tuple[str, ...]                         # sorted for determinism
    realized_fraction: float                         # len(val) / len(items); 0.0 when N == 0
    per_class_counts: dict[int, tuple[int, int]]     # class_id → (train_count, val_count)
    missing_in_val: tuple[int, ...]                  # classes with >= 2 train+val total but 0 in val


def stratified_split(
    items: Sequence[SplittableItem],
    fraction: float,
    seed: int,
) -> SplitResult:
    """Sechidis 2011 iterative multi-label stratification.

    Deterministic given (items, fraction, seed): items are sorted by
    image_id before processing so caller ordering does not matter.
    """
```

### 4.2 Algorithm — Sechidis 2011 (ECML PKDD) iterative stratification

1. **Sort input.** `items_sorted = sorted(items, key=lambda it: it.image_id)`. Caller order does not matter.
2. **Quotas.** Let `N = len(items_sorted)`, `V = round(N * fraction)`, `T = N - V`. For each class `c` with total count `n_c`, set `v_c = round(n_c * fraction)` and `t_c = n_c - v_c`.
3. **Initial ordering.** Sort items by `(min_class_count_in_item, rng.random())` ascending, where `min_class_count_in_item = min(n_c for c in item.class_ids)` and empty-class items use `math.inf`. The seeded `random.Random(seed)` provides the tiebreaker. Rarest-class items are placed first.
4. **Greedy placement.** Walk items in that order. For each item:
   - If one side is at capacity (`remaining[side] == 0`), place in the other.
   - Else score each side as `score[side] = max((quota[side][c] for c in item.class_ids), default=remaining[side])`. Place in the higher-scoring side.
   - Ties: prefer the side with larger `remaining`. Still tied: seeded coin flip via the same `rng`.
   - Decrement `remaining[chosen]` and every `quota[chosen][c]`.
5. **Post-checks.** Compute `realized_fraction = len(val) / max(N, 1)`. For each class `c` with `n_c >= 2` and zero val assignments, record in `missing_in_val`. `train_ids` and `val_ids` are sorted before being returned (stable shape regardless of placement order).

**Determinism contract:** identical `(items, fraction, seed)` ⇒ bit-identical `SplitResult`.

### 4.3 Edge cases (specified, not implicit)

| Case | Behavior |
| --- | --- |
| `N == 0` | `SplitResult(train_ids=(), val_ids=(), realized_fraction=0.0, per_class_counts={}, missing_in_val=())`. |
| `N == 1` | All to train (`round(1 * 0.1) == 0` for the default fraction). `realized_fraction = 0.0`. |
| Items with empty `class_ids` (HF row with no boxes) | Placed by capacity score only; never appear in `per_class_counts`; never contribute to `missing_in_val`. |
| `fraction` so small that `round(N * fraction) == 0` | `val_ids = ()`. `realized_fraction = 0.0`. Resolver WARNs (§4.5). |
| Adversarial tiny multi-label set where exact-fraction is impossible | `realized_fraction` deviates; caller (resolver) decides whether to WARN. |

### 4.4 Complexity

`O(N · C_avg)`. For COCO train2017 (~118k images × ~3 classes/image) the splitter runs in under one second on a single core; not on the training-step hot path.

### 4.5 Resolver-time WARN policy (consumes splitter output)

The resolver in `data/val_source.py` (not the splitter itself) emits:

- INFO: `"auto-split: fraction=0.10, realized=train=N/val=M (X.XX%); coverage=P/Q classes in val [missing: {ids}]"`.
- WARN if `abs(realized_fraction - fraction_requested) / fraction_requested > 0.2` OR `len(val_ids) < 8`.
- WARN once if `missing_in_val` is non-empty.

---

## 5. Val source resolver (`data/val_source.py`)

### 5.1 Types

```python
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from custom_sam_peft.config.schema import DataConfig, TrainConfig


@dataclass(frozen=True)
class ValSource:
    mode: Literal["explicit", "auto_split", "none"]
    train_ids: tuple[str, ...] | None       # None for explicit/none; populated for auto_split
    val_ids: tuple[str, ...] | None         # None for explicit/none; populated for auto_split
    realized_fraction: float | None         # auto_split only
    per_class_counts: dict[int, tuple[int, int]] | None
    missing_in_val: tuple[int, ...] | None
    fraction_requested: float | None        # auto_split only
    seed_used: int | None                   # auto_split only; resolved from val_split.seed or run.seed


def resolve_val_source(cfg: TrainConfig, *, run_dir: Path | None = None) -> ValSource: ...
def save_val_source(vs: ValSource, run_dir: Path) -> None: ...
def load_val_source(run_dir: Path) -> ValSource | None: ...
```

### 5.2 `resolve_val_source` — four-case dispatch

1. **`run_dir is not None and (run_dir / "val_source.json").exists()`** → `load_val_source(run_dir)`. Returns the saved record verbatim. INFO: `"resumed: reusing saved val_source.json (mode=…)"`. No re-enumeration, no re-stratification.
2. **`cfg.data.val_split is not None`** → enumerate items via `_enumerate_{coco,hf}_items(cfg.data)`; resolve `seed_used = cfg.data.val_split.seed if cfg.data.val_split.seed is not None else cfg.run.seed`; call `stratified_split(items, cfg.data.val_split.fraction, seed_used)`; construct `ValSource(mode="auto_split", ...)`.
3. **`cfg.data.val is not None`** → `ValSource(mode="explicit", train_ids=None, val_ids=None, realized_fraction=None, per_class_counts=None, missing_in_val=None, fraction_requested=None, seed_used=None)`.
4. **Else** → `ValSource(mode="none", ...all None...)`. WARN once.

The resolver itself emits the INFO/WARN log lines documented in §4.5; the trainer does not re-log.

**Seed override semantics.** `val_split.seed = None` means "inherit at resolve time". The resolver captures the resolved integer into `ValSource.seed_used`. The persistence record (§6) carries the integer, not `None` — so a resume + `run.seed` change does not silently re-stratify.

### 5.3 Resolution INFO/WARN at training start

The trainer's job is to call `_log_val_source(vs)` once after `resolve_val_source` returns (or after `load_val_source` in the resume path). Implementation idiomatic:

```python
def _log_val_source(vs: ValSource) -> None:
    if vs.mode == "explicit":
        _LOG.info("val source: explicit (cfg.data.val)")
    elif vs.mode == "auto_split":
        n_train, n_val = len(vs.train_ids), len(vs.val_ids)
        total = n_train + n_val
        pct = 100.0 * vs.realized_fraction
        covered = sum(1 for (t, v) in vs.per_class_counts.values() if v > 0)
        total_classes = len(vs.per_class_counts)
        _LOG.info(
            "val source: auto-split fraction=%.2f, realized=train=%d/val=%d (%.2f%%); "
            "coverage=%d/%d classes in val",
            vs.fraction_requested, n_train, n_val, pct, covered, total_classes,
        )
        if vs.missing_in_val:
            _LOG.warning("auto-split: %d classes missing from val: %s",
                         len(vs.missing_in_val), list(vs.missing_in_val))
        if (
            abs(vs.realized_fraction - vs.fraction_requested) / vs.fraction_requested > 0.2
            or n_val < 8
        ):
            _LOG.warning("auto-split: realized fraction deviates from requested or val is small")
    else:  # mode == "none"
        _LOG.warning(
            "training without validation set; eval_every is a no-op, end-of-run "
            "eval and bundle samples are skipped. Use data.val to provide one or "
            "data.val_split to auto-split."
        )
```

### 5.4 Enumeration helpers

Both are cheap — they read indices/metadata, never decode images.

```python
def _enumerate_coco_items(data_cfg: DataConfig) -> list[SplittableItem]:
    """Reuses _load_coco_index + _drop_crowd_only_images + _build_category_remap
    from data/coco.py (lines 27, 37, 46). Each SplittableItem.image_id is
    str(int_image_id); class_ids is the frozenset of dense ids present after
    crowd filtering.
    """
```

```python
def _enumerate_hf_items(data_cfg: DataConfig) -> list[SplittableItem]:
    """Reuses hf_load_dataset(data_cfg.hf.name, split=data_cfg.hf.split_train)
    from data/hf.py line 122. Walks data_cfg.hf.field_map.category for each
    row to populate class_ids. image_id = str(row_index).
    """
```

`_enumerate_coco_items` reuses the existing crowd filter so a no-instance image cannot inflate train and pull down val coverage.

### 5.5 `save_val_source` / `load_val_source`

`save_val_source(vs, run_dir)` writes `<run_dir>/val_source.json` with `mode`, `fraction_requested`, `seed_used`, `realized_fraction`, `n_train`, `n_val`, `per_class_counts`, `missing_in_val`, `train_ids`, `val_ids`. Atomic write: write to `<run_dir>/val_source.json.tmp` then `os.replace`.

`load_val_source(run_dir)` returns `None` if the file does not exist; otherwise rehydrates a `ValSource` with `tuple()` for the id lists and a `dict[int, tuple[int, int]]` for `per_class_counts` (JSON serializes keys as strings; the loader re-casts).

---

## 6. Adapter integration

### 6.1 `COCODataset.__init__` — additive `image_ids`

```python
class COCODataset:
    def __init__(
        self,
        annotations: str,
        images: str,
        prompt_mode: Literal["text", "bbox"],
        *,
        transforms: Any,
        text_prompt: TextPromptConfig,
        seed: int = 0,
        image_ids: Iterable[int] | None = None,   # new
    ) -> None:
        ...
        kept, ann_index, dropped = _drop_crowd_only_images(self._coco)
        if image_ids is not None:
            requested = {int(x) for x in image_ids}
            kept_set = set(kept)
            missing = requested - kept_set
            if missing:
                raise ValueError(
                    f"COCODataset: {len(missing)} image_ids requested but not present "
                    f"(or dropped as iscrowd-only): {sorted(missing)[:10]}…"
                )
            # Preserve sorted order (matches the existing _drop_crowd_only_images contract).
            self._image_ids = [i for i in kept if i in requested]
        else:
            self._image_ids = kept
        self._ann_index = ann_index
        ...
```

Behavior unchanged when `image_ids is None`. When set, the resulting dataset's `__len__` equals `len(requested ∩ crowd_filtered)`; missing ids are a loud `ValueError`.

### 6.2 `HFDataset.__init__` — additive `row_indices`

```python
class HFDataset:
    def __init__(
        self,
        name: str,
        split: str,
        prompt_mode: Literal["text", "bbox"],
        *,
        transforms: Any,
        text_prompt: TextPromptConfig,
        field_map: HFFieldMap,
        seed: int = 0,
        row_indices: Iterable[int] | None = None,  # new
    ) -> None:
        ...
        self._ds = hf_load_dataset(name, split=split)
        _validate_required_fields(self._ds, field_map)
        self._class_names = _resolve_class_names(self._ds, field_map)
        if row_indices is not None:
            self._index_map: list[int] | None = [int(i) for i in row_indices]
            invalid = [i for i in self._index_map if i < 0 or i >= len(self._ds)]
            if invalid:
                raise ValueError(
                    f"HFDataset: {len(invalid)} row_indices out of range "
                    f"[0, {len(self._ds)}): first few = {invalid[:10]}"
                )
        else:
            self._index_map = None

    def __len__(self) -> int:
        return len(self._index_map) if self._index_map is not None else len(self._ds)

    def __getitem__(self, i: int) -> Example:
        row_i = self._index_map[i] if self._index_map is not None else i
        # ... existing body reads self._ds[row_i] and uses row_i as image_id source ...
```

`image_id` in the returned `Example` uses the underlying dataset row index (so it matches the splitter's `image_id` namespace), not the post-subset position.

### 6.3 Builder injection from `_resolved_image_ids`

The dataset registry signature `(cfg: dict, *, model_name, pipeline)` is unchanged. The runner injects a private namespaced key into the dict it produces from `cfg.data.model_dump()`:

```python
# inside data/coco.py::build_coco, after the existing image_size / normalize / text_prompt setup:
resolved = (cfg.get("_resolved_image_ids") or {}).get(pipeline)  # 'train' or 'eval'
return COCODataset(
    annotations=split["annotations"],
    images=split["images"],
    prompt_mode=cfg["prompt_mode"],
    transforms=transforms,
    text_prompt=text_prompt,
    image_ids=[int(s) for s in resolved] if resolved is not None else None,
)
```

```python
# inside data/hf.py::build_hf, symmetric:
resolved = (cfg.get("_resolved_image_ids") or {}).get(pipeline)
return HFDataset(
    name=hf_cfg["name"],
    split=split,
    prompt_mode=cfg["prompt_mode"],
    transforms=transforms,
    text_prompt=text_prompt,
    field_map=field_map,
    row_indices=[int(s) for s in resolved] if resolved is not None else None,
)
```

**Why a private dict key, not a widened builder signature.** `DataConfig` extends `_Strict` (`extra="forbid"` — schema.py line 28–29). User-supplied `_resolved_image_ids` would fail validation. By contrast, `cfg.data.model_dump()` returns a plain dict; the runner mutates it post-dump. The leading underscore documents the field as runner-injected, never user-set. The builder contract therefore reads: *"`cfg` is the dict produced by `cfg.data.model_dump()`, optionally augmented by the runner with `_resolved_image_ids: {'train': [...], 'eval': [...]}`."*

Alternative considered: widen `dataset` builder signature to `(cfg, *, model_name, pipeline, image_ids=None)`. Rejected — the registry's `lookup("dataset", ...)` result is invoked in three call sites (`train/runner.py` line 27, `eval/runner.py` line 89, `cli/run_cmd.py` line 36) and would tangle the new optional parameter through each, plus break the registry's uniform signature.

### 6.4 Runner orchestration — `train/runner.py`

```python
from custom_sam_peft.data.val_source import (
    ValSource, resolve_val_source, save_val_source, _log_val_source,
)


def _build_dataset_from_dict(
    data_cfg_dict: dict[str, Any], cfg: TrainConfig, pipeline: str
) -> Dataset:
    builder = lookup("dataset", cfg.data.format)
    return cast(Dataset, builder(data_cfg_dict, model_name=cfg.model.name, pipeline=pipeline))


def run_training(
    cfg: TrainConfig,
    *,
    resume_from: Path | None = None,
) -> RunResult:
    run_dir = make_run_dir(cfg)
    # On resume, look for val_source.json in the run dir that owns the checkpoint
    # (checkpoints live at <run_dir>/checkpoints/step_N/).
    resume_run_dir = resume_from.parent.parent if resume_from is not None else None
    vs = resolve_val_source(cfg, run_dir=resume_run_dir)
    save_val_source(vs, run_dir)
    _log_val_source(vs)

    data_cfg_dict = cfg.data.model_dump()
    if vs.mode == "auto_split":
        data_cfg_dict["_resolved_image_ids"] = {
            "train": list(vs.train_ids),
            "eval":  list(vs.val_ids),
        }

    train_ds = _build_dataset_from_dict(data_cfg_dict, cfg, "train")
    val_ds: Dataset | None = (
        None if vs.mode == "none"
        else _build_dataset_from_dict(data_cfg_dict, cfg, "eval")
    )

    wrapper: Any = load_sam31(cfg.model)
    peft_factory = lookup("peft", cfg.peft.method)
    peft_factory(wrapper, cfg.peft)
    tracker = build_tracker(cfg)
    trainer = Trainer(wrapper, train_ds, val_ds, tracker, cfg)
    return trainer.fit(run_dir=run_dir, resume_from=resume_from)
```

The existing `_build_dataset(cfg, pipeline)` (runner.py line 25) is replaced by `_build_dataset_from_dict`. `make_run_dir` (runner.py line 17) is unchanged. The `cast(Dataset, ...)` and `Any` typing follow the existing convention.

Resume note: a resume reuses the *original* run_dir's saved split. The new `run_dir` from `make_run_dir(cfg)` gets the same `val_source.json` copied into it (via `save_val_source(vs, run_dir)` after the resolver returns the loaded record) so the resumed run is self-describing.

---

## 7. Trainer / Eval / Bundle / CLI no-val path

### 7.1 Trainer (`train/trainer.py`)

```python
class Trainer:
    def __init__(
        self,
        model: Sam3Wrapper,
        train_ds: Dataset,
        val_ds: Dataset | None,            # was: Dataset (line 132)
        tracker: Tracker,
        cfg: TrainConfig,
    ) -> None:
        if cfg.data.prompt_mode == "bbox":
            raise ValueError(...)          # unchanged (line 136–141)
        self.model = model
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.tracker = tracker
        self.cfg = cfg
        self._optimizer_name = _resolve_optimizer_name(cfg)
        if cfg.train.optimizer == "auto":
            _LOG.info("optimizer=auto resolved to %s (peft.method=%s)",
                      self._optimizer_name, cfg.peft.method)
        if val_ds is None:
            _LOG.info(
                "training without validation set; eval_every is a no-op, "
                "end-of-run eval and bundle samples are skipped."
            )
```

**`fit()` changes:**

- Line 186 (`val_examples = [self.val_ds[i] for i in range(min(4, len(self.val_ds)))]`) → `val_examples: list[Any] = [] if self.val_ds is None else [self.val_ds[i] for i in range(min(4, len(self.val_ds)))]`.
- `on_eval` closure (line 222): top of body → `if self.val_ds is None: return`. (Defensive — `run_epoch` will not reach this when `eval_every` triggers, see below.)
- Line 257 — end-of-run eval — wrap in `if self.val_ds is not None: full_report = Evaluator(cfg.eval).evaluate(self.model, self.val_ds)`.
- `metrics.json` write (line 258–271): branch.
  - `val_ds is not None and full_report is not None` → unchanged.
  - Otherwise → write `{"note": "no validation set provided", "global_step": global_step, "epoch": cfg.train.epochs - 1, "box_hint_p_final": _box_hint_p(global_step, cfg.train.box_hint)}`.
- Return: `RunResult.final_metrics = full_report` (already `MetricsReport | None`; remains `None` here — line 42).

**Tracker hparams injection.** Just before `self.tracker.start_run(...)` (line 167), the trainer reads the saved `val_source.json` from `run_dir` (since the runner saved it before constructing the trainer) and folds the val-mode fields into the config dict passed to the tracker:

```python
cfg_dict = cfg.model_dump(mode="json")
vs_path = run_dir / "val_source.json"
if vs_path.exists():
    saved = json.loads(vs_path.read_text())
    cfg_dict["val_source"] = {
        "mode": saved["mode"],
        "fraction_requested": saved.get("fraction_requested"),
        "realized_fraction": saved.get("realized_fraction"),
        "n_train": saved.get("n_train"),
        "n_val": saved.get("n_val"),
    }
self.tracker.start_run(run_dir, cfg_dict, resume_from)
```

The `Tracker.start_run` `config` param is already `dict[str, Any]` (tracking/base.py line 21) — no protocol change.

### 7.2 `run_epoch` parameter cleanup (`train/loop.py`)

The current `run_epoch` signature (line 227) takes `val_ds: Any` but does not use it — eval is invoked via the `on_eval` closure (line 266) which captures `self.val_ds` in `trainer.py`. The trainer call site is the only caller. Drop the parameter:

```python
def run_epoch(
    model: Sam3Wrapper,
    loader: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    tracker: Tracker,
    cfg: TrainConfig,
    run_dir: Path,
    epoch: int,
    global_step: int,
    nan_streak: int,
    class_names: list[str],
    on_checkpoint: Callable[[int, int, float, int], None],
    on_eval: Callable[[int], None],
) -> tuple[int, int]:
    ...
```

Trainer line 246 (`self.val_ds`) drops out of the `run_epoch(...)` argument list. No behavioral change for the explicit path.

Additionally, `run_epoch` skips the `on_eval` call when `val_ds` is absent — but since the on_eval callback already self-guards (§7.1), no extra logic is needed in the loop. The on_eval call at line 266–267 remains as-is.

### 7.3 Image panel skip

`_log_image_panel` (trainer.py line 282) already short-circuits on `if not val_examples: return` (line 288). In no-val mode `val_examples = []` (set in §7.1) → the panel skips without further changes. The `on_checkpoint` closure at line 207 still fires the image-panel call (line 220); it short-circuits cleanly.

### 7.4 Eval CLI (`eval/runner.py`)

Two changes:

**A. `--split val` guard** symmetric with the existing `split == "test"` check at line 81:

```python
if split == "val" and cfg.data.val is None and cfg.data.val_split is None:
    raise ValueError(
        "--split val requires data.val or data.val_split in config; got neither."
    )
```

This sits *between* the existing peft check (line 76) and the existing test guard (line 81).

**B. Standalone-eval auto-split support.** When `val_dataset is None` and `cfg.data.val_split is not None`, the function recomputes the split (no `run_dir` available in this codepath) and threads `_resolved_image_ids` into `cfg_dict`:

```python
if val_dataset is None:
    cfg_dict = cfg.data.model_dump()
    if split == "test":
        cfg_dict["val"] = cfg_dict["test"]
    elif split == "val" and cfg.data.val_split is not None:
        from custom_sam_peft.data.val_source import resolve_val_source
        vs = resolve_val_source(cfg, run_dir=None)
        cfg_dict["_resolved_image_ids"] = {"eval": list(vs.val_ids)}
    builder = lookup("dataset", cfg.data.format)
    dataset = cast(Dataset, builder(cfg_dict, model_name=cfg.model.name, pipeline="eval"))
```

**Reproducibility note.** Standalone `csp eval --split val` against an auto-split config reproduces the in-training split iff the underlying dataset (COCO annotation file or HF dataset version) is unchanged. The split is `(items, fraction, seed)`-deterministic; if items shift, the split shifts. This is the documented limitation — users who need exact reproducibility against a *specific run's* val set should pass `--run-dir` (future enhancement, not part of this spec) or load the run's `val_source.json` manually.

### 7.5 Bundle (`runs/bundle.py`)

Signature widens; the function gains an early no-val branch.

```python
def write_bundle(
    ctx: BundleContext,
    metrics_report: MetricsReport | None,            # was: MetricsReport
    val_dataset: Dataset | None,                     # was: Dataset
    model_wrapper: Any,
) -> None:
    if val_dataset is None:
        _write_summary_no_val(ctx)
        return
    # ... existing body (line 275 onward) unchanged ...
```

`_write_summary_no_val` writes `<run_dir>/summary.md` with:

```markdown
# {ctx.config_path.parent.name} — no-val

## Run
- Start:  {start}
- End:    {end}
- Duration: {hh:mm:ss}

## Hardware
- GPU:  {gpu_name}
- VRAM: {vram_gb} GB

## Preset
- Applied: {preset_label or 'manual'}

## Outputs
- Adapter: adapter
- Merged:  {merged status, same logic as the full path}
- Config:  {config_path.name}

## Validation
No validation set; this run did not produce mAP or per-example IoU.
Tracker scalars and training-loss curve are at the configured TB run dir.

## Edge cases
- {any from ctx, e.g., merged_export_error}
```

No `samples/` directory is created in no-val mode. The headline shows `no-val` instead of a numeric mAP — disambiguates summary scrapers.

### 7.6 `cli/run_cmd.py`

```python
def _orchestrate(cfg: TrainConfig, resume: Path | None) -> int:
    start_ts = datetime.now(UTC)
    try:
        train_result = run_training(cfg, resume_from=resume)
    except Exception as exc:
        rprint(f"[red]train failed[/red] {exc}")
        raise typer.Exit(code=1) from exc
    run_dir = train_result.run_dir
    adapter_path = train_result.adapter_path

    # Decide val mode from the saved record — same source of truth the trainer used.
    from custom_sam_peft.data.val_source import load_val_source
    vs = load_val_source(run_dir)
    assert vs is not None, "runner must have saved val_source.json"

    wrapper: Any = load_sam31(cfg.model)
    load_adapter(wrapper, adapter_path)

    val_dataset: Dataset | None = None
    report: Any = None
    per_example_iou: list[float] = []
    if vs.mode != "none":
        val_dataset = _build_val_dataset(cfg, vs)
        try:
            report, per_example_iou = cast(
                tuple[Any, list[float]],
                run_eval(
                    cfg,
                    checkpoint=adapter_path,
                    output_dir=run_dir,
                    val_dataset=val_dataset,
                    model=wrapper,
                    return_per_example_iou=True,
                ),
            )
        except Exception as exc:
            rprint(f"[red]eval failed[/red] run_dir={run_dir} — {exc}")
            raise typer.Exit(code=1) from exc

    end_ts = datetime.now(UTC)

    merged_dir: Path | None = None
    merged_export_error: str | None = None
    if cfg.export.merge:
        ...  # unchanged

    ctx = BundleContext(
        run_dir=run_dir,
        config_path=run_dir / "config.yaml",
        start_ts=start_ts,
        end_ts=end_ts,
        preset_label=os.environ.get("CUSTOM_SAM_PEFT_PRESET_LABEL"),
        per_example_iou=per_example_iou,
        merged_dir=merged_dir,
        merged_export_error=merged_export_error,
    )
    try:
        write_bundle(ctx, report, val_dataset=val_dataset, model_wrapper=wrapper)
    except Exception as exc:
        rprint(f"[red]bundle failed[/red] run_dir={run_dir} — {exc}")
        raise typer.Exit(code=1) from exc

    mAP_str = f"{report.overall.get('mAP', float('nan')):.4f}" if report is not None else "n/a (no val)"
    rprint(
        f"[green]done[/green] run_dir={run_dir} adapter={adapter_path} "
        f"merged={(merged_dir or merged_export_error or 'skipped')} "
        f"summary={run_dir / 'summary.md'} mAP={mAP_str}"
    )
    return 0


def _build_val_dataset(cfg: TrainConfig, vs: ValSource) -> Dataset:
    """Build the val dataset using the same image ids the trainer used."""
    data_cfg_dict = cfg.data.model_dump()
    if vs.mode == "auto_split":
        data_cfg_dict["_resolved_image_ids"] = {"eval": list(vs.val_ids)}
    builder = lookup("dataset", cfg.data.format)
    return cast(Dataset, builder(data_cfg_dict, model_name=cfg.model.name, pipeline="eval"))
```

The existing `_build_val_dataset(cfg)` (run_cmd.py line 34) is replaced by `_build_val_dataset(cfg, vs)`.

### 7.7 Doctor (`cli/doctor_cmd.py` + `diagnostics.py`)

New `DataReport` and `--config` option. Doctor never invokes the splitter — it reads the resolved mode from cfg only.

```python
# diagnostics.py
@dataclass(frozen=True)
class DataReport:
    val_mode: Literal["explicit", "auto_split", "none"]
    val_path: str | None                 # populated only for explicit mode
    val_split_fraction: float | None     # populated only for auto_split
    val_split_seed: int | None           # populated only for auto_split (resolved against run.seed)


@dataclass(frozen=True)
class DoctorReport:
    ...                                  # existing fields
    data: DataReport | None = None       # populated only when --config is passed


def run_doctor(
    *,
    weights_path: Path | None = None,
    config_path: Path | None = None,
) -> DoctorReport:
    ...                                  # existing body
    data: DataReport | None = None
    if config_path is not None:
        from custom_sam_peft.config.loader import load_config
        cfg = load_config(config_path)
        if cfg.data.val_split is not None:
            seed = cfg.data.val_split.seed if cfg.data.val_split.seed is not None else cfg.run.seed
            data = DataReport(
                val_mode="auto_split",
                val_path=None,
                val_split_fraction=cfg.data.val_split.fraction,
                val_split_seed=seed,
            )
        elif cfg.data.val is not None:
            data = DataReport(
                val_mode="explicit",
                val_path=cfg.data.val.annotations,
                val_split_fraction=None,
                val_split_seed=None,
            )
        else:
            data = DataReport(
                val_mode="none",
                val_path=None,
                val_split_fraction=None,
                val_split_seed=None,
            )
    return DoctorReport(..., data=data)
```

```python
# cli/doctor_cmd.py — new CLI flag, new table
def doctor(
    weights_path: Path | None = typer.Option(None, "--weights-path", ...),
    config_path: Path | None = typer.Option(
        None, "--config", help="Optional config YAML; enables the Data table."
    ),
    json_output: bool = typer.Option(False, "--json", ...),
) -> None:
    report = run_doctor(weights_path=weights_path, config_path=config_path)
    ...
```

```python
# _render_table — new branch
if report.data is not None:
    d = Table(title="Data", show_header=False, box=None)
    d.add_row("val mode", report.data.val_mode)
    if report.data.val_path is not None:
        d.add_row("val path", report.data.val_path)
    if report.data.val_split_fraction is not None:
        d.add_row("val_split.fraction", f"{report.data.val_split_fraction:.3f}")
        d.add_row("val_split.seed", str(report.data.val_split_seed))
    console.print(d)
```

Cost: one `load_config` call. No dataset materialization, no splitter invocation. Verified by test (§9) that mocks `_enumerate_coco_items` and asserts zero calls.

---

## 8. Run-dir record & resume

### 8.1 `<run_dir>/val_source.json` — three schemas

The file is written exactly once per run, by `save_val_source(vs, run_dir)`, called from `run_training` *before* `Trainer.fit` begins (so the trainer can read it for tracker hparams).

```json
// auto_split
{
  "mode": "auto_split",
  "fraction_requested": 0.1,
  "seed_used": 42,
  "realized_fraction": 0.103,
  "n_train": 894,
  "n_val": 103,
  "per_class_counts": {"0": [85, 9], "1": [120, 14]},
  "missing_in_val": [],
  "train_ids": ["123", "127", "131"],
  "val_ids": ["145", "203"]
}
```

```json
// explicit
{
  "mode": "explicit",
  "fraction_requested": null,
  "seed_used": null,
  "realized_fraction": null,
  "n_train": null,
  "n_val": null,
  "per_class_counts": null,
  "missing_in_val": null,
  "train_ids": null,
  "val_ids": null
}
```

```json
// none
{
  "mode": "none",
  "fraction_requested": null,
  "seed_used": null,
  "realized_fraction": null,
  "n_train": null,
  "n_val": null,
  "per_class_counts": null,
  "missing_in_val": null,
  "train_ids": null,
  "val_ids": null
}
```

`per_class_counts` JSON-serializes the int keys as strings; `load_val_source` re-casts to `int`.

### 8.2 Resume flow

`run_training(cfg, resume_from=<run_dir>/checkpoints/step_N/)`:

1. `resume_run_dir = resume_from.parent.parent` — the run dir that owns the checkpoint.
2. `vs = resolve_val_source(cfg, run_dir=resume_run_dir)` → resolver finds `val_source.json` there and returns the saved record.
3. `make_run_dir(cfg)` creates a fresh `run_dir` (existing behavior).
4. `save_val_source(vs, run_dir)` writes the *same* record into the new run dir so the resumed run is self-describing.
5. Trainer proceeds with the saved partition.

### 8.3 Invariant

An auto-split run **never re-stratifies on resume**. The saved `(train_ids, val_ids)` partition is authoritative for the lifetime of the original run_dir, and is propagated forward into any resume's new run_dir verbatim.

### 8.4 Config drift on resume

The resolver compares the saved record against the current cfg and WARNs (does not abort) on:

- `vs_saved.fraction_requested != cfg.data.val_split.fraction` (when both modes are `auto_split`).
- `vs_saved.seed_used != (cfg.data.val_split.seed or cfg.run.seed)` (auto_split).
- `vs_saved.mode != <mode that current cfg would resolve to>` (any mode change).

WARN message: `"resumed run uses saved val_source.json; cfg.data.val_split is ignored on resume"`. Pattern mirrors the existing `cfg_hash` mismatch WARN in `train/checkpoint.py::load_full_state` line 167–171. The saved record always wins by design.

### 8.5 Tracker hparams

Trainer injects four keys into the dict passed to `tracker.start_run` (already `dict[str, Any]`; no protocol change):

| Key | Type | Source |
| --- | --- | --- |
| `val_source.mode` | `"explicit" \| "auto_split" \| "none"` | `vs.mode` |
| `val_source.fraction_requested` | `float \| None` | `vs.fraction_requested` |
| `val_source.realized_fraction` | `float \| None` | `vs.realized_fraction` |
| `val_source.n_train` | `int \| None` | `len(vs.train_ids)` if populated |
| `val_source.n_val` | `int \| None` | `len(vs.val_ids)` if populated |

---

## 9. Testing strategy

CPU-only unless noted. Honors the `feedback_gpu_vs_cpu_testing` policy — no GPU tests added.

### 9.1 Splitter unit — `tests/unit/test_splitter.py` (new)

| Test | Asserts |
| --- | --- |
| Determinism | `stratified_split(items, 0.2, 42)` called twice → bit-identical `SplitResult`. |
| Order independence | `random.shuffle(items)` then split → same result. |
| Single-class realized fraction | 100 items, 1 class each, `fraction=0.1` → `realized_fraction` within `±0.01` of `0.1`. |
| Multi-class coverage | 50 items, 10 classes, one rare class with 2 items, `fraction=0.2`, fixed seed → rare class in both `train` and `val`. |
| Empty class set | 5 items each with `class_ids=frozenset()`, `fraction=0.2` → 1 val, 4 train; `missing_in_val == ()`; no crash. |
| Edge sizes | `N=0` → empty result; `N=1` → all to train; `N=2, fraction=0.5` → 1+1. |
| Quota deviation | Adversarial small set where exact-fraction is impossible → `realized_fraction` ∈ `[0, 1)` and `missing_in_val` accurate. |

### 9.2 Resolver unit — `tests/unit/test_val_source.py` (new)

| Test | Asserts |
| --- | --- |
| Mode dispatch | Three TrainConfigs (val / val_split / neither) → `vs.mode` ∈ {explicit, auto_split, none}; WARN once for `none` (capture via `caplog`). |
| Round-trip | `save_val_source(vs, tmp_path); load_val_source(tmp_path)` returns structurally equal `ValSource` (incl. `per_class_counts` re-cast to int keys). |
| Resume preference | Pre-existing `val_source.json` in `run_dir`; `resolve_val_source(cfg, run_dir=run_dir)` returns saved record even when `cfg.data.val_split.fraction` differs from saved; WARN emitted. |
| COCO enumeration | On `tests/fixtures/tiny_coco/` → produces `SplittableItem`s with correct dense class ids; crowd-only images excluded. |
| HF enumeration | On the existing HF stub (used by `tests/unit/test_data_hf.py`) → row indices populated; class ids per row. |
| Seed inheritance | `val_split.seed=None, run.seed=7` → `vs.seed_used == 7`. |

### 9.3 Schema — extend `tests/unit/test_config_schema.py`

1. `val: null` validates (no longer rejected).
2. Both `val` and `val_split` → `ValidationError` (helpful message).
3. `val_split.fraction = 0.6` → `ValidationError` (> 0.5 bound).
4. `val_split.fraction = 0` / negative → `ValidationError`.
5. `format=hf`, `val_split` set, `hf.split_val="custom"` → `ValidationError` from `_check_hf_split_val_compat`.
6. `format=hf`, `val_split` set, `hf.split_val="validation"` (default) → validates.
7. Neither `val` nor `val_split` → validates (WARN happens at resolve, not validation).

### 9.4 Adapter integration — extend `tests/unit/test_data_coco.py` and `test_data_hf.py`

1. `COCODataset(image_ids=[a, b, c])` on `tiny_coco` → `len == 3`; `__getitem__(0).image_id ∈ {a, b, c}`.
2. `COCODataset(image_ids=[999999])` → `ValueError` naming the missing id.
3. `HFDataset(row_indices=[0, 2])` → `len == 2`; row 0 of subset maps to underlying row 0; subset image_id stays in the underlying-row namespace.
4. `HFDataset(row_indices=[-1])` and `row_indices=[len(ds)+1]` → `ValueError`.
5. **Image-level leak invariant.** Run `stratified_split` on tiny_coco SplittableItems with `fraction=0.5`; assert `set(train_ids) & set(val_ids) == set()`.

### 9.5 Trainer no-val — new `tests/unit/test_trainer_no_val.py`

1. `Trainer(model, train_ds, val_ds=None, tracker, cfg).fit(run_dir=...)` against `TinySam3Stub` completes; `RunResult.final_metrics is None`; `metrics.json` parses to `{"note": "no validation set provided", "global_step": ..., "epoch": ..., "box_hint_p_final": ...}`.
2. `on_eval` never invoked when `val_ds is None` — assert via a mock `Evaluator` and `assert_not_called()`.
3. Image-panel never written (mock `tracker.log_images` and assert not called).

### 9.6 Bundle no-val — extend `tests/unit/runs/test_bundle.py`

1. `write_bundle(ctx, metrics_report=None, val_dataset=None, model_wrapper=Mock())` writes `summary.md`, no `samples/` directory created.
2. `summary.md` contains the literal `"No validation set"` line.
3. Headline reads `"... — no-val"`.

### 9.7 Runner — extend `tests/unit/test_train_runner.py`

1. End-to-end auto-split on `tiny_coco` (CPU + `TinySam3Stub`) → `val_source.json` exists in `run_dir`; `load_val_source(run_dir)` round-trips.
2. Resume reuses saved split: call `run_training` once with `val_split`, then again with `resume_from=<step_N>` — assert the second call's val dataset has the same image ids as the first. Side-effect check: monkeypatch `stratified_split` to raise; resume must not call it.

### 9.8 Doctor — extend `tests/unit/test_cli_doctor.py`

1. `csp doctor --config <yaml with val_split>` → output contains a "Data" section with `val mode auto_split` and the fraction/seed.
2. `csp doctor --config <yaml with explicit val>` → "Data" section shows `val mode explicit` and the val path.
3. `csp doctor --config <yaml with neither>` → "Data" section shows `val mode none`.
4. `csp doctor` (no `--config`) → no "Data" section; no dataset/splitter calls (monkeypatch `_enumerate_coco_items` and `stratified_split` to raise; assert they did not).

### 9.9 Eval — extend `tests/unit/test_eval_runner.py`

1. `run_eval(cfg, ..., split="val")` with `cfg.data.val is None` and `cfg.data.val_split is None` → `ValueError` containing `"--split val requires data.val or data.val_split"`.
2. `run_eval(cfg, ..., split="val")` with `cfg.data.val_split` set and `val_dataset=None` → builder receives `_resolved_image_ids` (mock the builder and assert the cfg dict it received).

### 9.10 Integration — extend `tests/integration/test_train_end_to_end.py`

1. Existing tests (explicit val) run unchanged.
2. New: end-to-end auto-split on `tiny_coco` finishes; assert `<run_dir>/val_source.json` exists with `mode=auto_split`; `metrics.json` exists.
3. New: end-to-end no-val run finishes; assert `<run_dir>/val_source.json` has `mode=none`; assert `summary.md` body contains "No validation set" via the CLI run path (`tests/integration/test_cli_run.py`).

### 9.11 GPU tests

No changes. Existing `tests/gpu/*` configs pin `data.val.{annotations,images}` explicitly via CLI overrides (see `tests/gpu/test_real_train_overfits.py` and friends), so they continue to exercise the explicit path. No GPU tests added.

### 9.12 Coverage gate

Maintain ≥80% coverage on `src/custom_sam_peft/data/` after this spec lands (current target). The two new modules contribute ~230 LOC of mostly straight-line code with full unit coverage per §9.1, §9.2.

---

## 10. Out of scope / future work

| Item | Reason / follow-up |
| --- | --- |
| Folder dataset adapter (#33) | Once it lands, its `Dataset` ctor picks up an `image_ids` (or `paths_subset`) parameter and the resolver gains `_enumerate_folder_items`. Not implemented here. |
| Multiplex training (#22) | Each per-source `Dataset` independently flows through `resolve_val_source`; the multi-source splitting policy (per-source vs pooled) is decided in #22's own design. Not implemented here. |
| Cross-validation / k-fold | Distinct feature; not gated by this. |
| Test-set auto-split | `data.test` remains explicit-only. Same rationale as #71's "Out of scope" section. |
| Strict per-image stability under dataset edits | Considered (hash-based assignment) and rejected: stratification was prioritized over add-one-image stability. The run-dir saved split records the partition; that solves the practical reproducibility need. Users who edit COCO between runs accept that the auto-split will move. |
| `csp eval --run-dir <path>` for exact run-val reproduction | Standalone `csp eval --split val` against an auto-split config recomputes from scratch (§7.4). Loading a specific run's `val_source.json` into eval is a future ergonomics enhancement. |
| Cosine / nonuniform val splits, per-attribute stratification | Sechidis multi-label iterative stratification is the only mode in v0. |

---

## 11. Implementation plan (numbered)

This section is the seam consumed by `superpowers:writing-plans`.

**Step 1.** Schema additions (§3): `ValSplitConfig`; `DataConfig.val: DataSplit | None = None`; `DataConfig.val_split: ValSplitConfig | None = None`; two `model_validator(mode="after")` methods. Unit tests per §9.3.

**Step 2.** Implement `data/splitter.py` (§4): `SplittableItem`, `SplitResult`, `stratified_split`. Unit tests per §9.1.

**Step 3.** Implement `data/val_source.py` (§5): `ValSource`, `resolve_val_source`, `save_val_source`, `load_val_source`, `_enumerate_coco_items`, `_enumerate_hf_items`, `_log_val_source`. Unit tests per §9.2.

**Step 4.** Adapter ctor parameter additions (§6.1, §6.2): `COCODataset(image_ids=...)`, `HFDataset(row_indices=...)`; builder injection from `cfg["_resolved_image_ids"]` (§6.3). Unit tests per §9.4 — including the image-level leak invariant.

**Step 5.** Trainer / loop wiring (§7.1, §7.2, §7.3): `val_ds: Dataset | None` in `Trainer`; guards in `fit()`; drop `val_ds` parameter from `run_epoch`; tracker hparams injection. Unit tests per §9.5.

**Step 6.** Runner orchestration (§6.4): `_build_dataset_from_dict`; `resolve_val_source` → `save_val_source` → inject `_resolved_image_ids` → build datasets. Unit tests per §9.7.

**Step 7.** Eval runner (§7.4): `--split val` guard, standalone auto-split support. Unit tests per §9.9.

**Step 8.** Bundle (§7.5): widen signature; `_write_summary_no_val`. Unit tests per §9.6.

**Step 9.** `cli/run_cmd.py` (§7.6): orchestration update; `_build_val_dataset(cfg, vs)`. End-to-end CLI test via `tests/integration/test_cli_run.py`.

**Step 10.** Doctor (§7.7): `--config` option, `DataReport`, "Data" table. Unit tests per §9.8.

**Step 11.** Example YAMLs + CLI templates: `configs/examples/coco_text_no_val.yaml`, `configs/examples/coco_text_auto_split.yaml`; uncommented `val_split:` reference block added to `src/custom_sam_peft/cli/templates/coco_text_lora.yaml` and `coco_text_qlora.yaml` as a commented section.

**Step 12.** Integration tests (§9.10): auto-split and no-val end-to-end on `tiny_coco`.

**Step 13.** Lint/format pass; ensure coverage gate (§9.12).
