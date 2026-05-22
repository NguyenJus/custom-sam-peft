# spec/data-subset-limit — Fast training runs on dataset subsets (issue #72)

**Status:** Draft (2026-05-22)
**Tracking:** [#72](https://github.com/NguyenJus/custom-sam-peft/issues/72) — *feat(data): support fast training runs on subsets of the full dataset*
**Scope:** Add a `data.limit` config knob so users can point at a real, full-size dataset and train on a deterministic small slice without editing COCO JSON files or maintaining a hand-curated `data/placeholder/` directory. The subset wrapper is format-agnostic (COCO and HF both supported), fully transparent to the trainer, and wires into the doctor command when `--config` is provided.

**Builds on:**
[`2026-05-16-data-loading-design.md`](2026-05-16-data-loading-design.md) (the `Dataset` Protocol, `COCODataset`/`HFDataset`, and `_build_dataset` seam this spec extends);
[`2026-05-18-cli-design.md`](2026-05-18-cli-design.md) and [`2026-05-19-hf-utils-design.md`](2026-05-19-hf-utils-design.md) (doctor command shape and `DoctorReport` extended here).

---

## 1. Goals & v0 Scope

Today the only way to do a quick sanity-check or profiling run on COCO is to edit the annotations JSON or keep a tiny `data/placeholder/` directory around. Issue #72 adds `data.limit` so users can write one extra config block and get a repeatable small slice of their real dataset — useful for debugging transforms, confirming the training loop runs to completion, and CI smoke runs that need real data without the full 118k-image cost.

**In scope:**

| Deliverable | Where |
| --- | --- |
| `LimitConfig` + `DataConfig.limit` | `src/custom_sam_peft/config/schema.py` |
| `SubsetDataset` wrapper + `resolve_subset_indices` pure function | `src/custom_sam_peft/data/subset.py` (new) |
| `image_class_labels` property on `COCODataset` (eager) and `HFDataset` (lazy + cached) | `src/custom_sam_peft/data/coco.py`, `hf.py` |
| Limit wrapping in `_build_dataset` + startup INFO log | `src/custom_sam_peft/train/runner.py` |
| `subset.json` run-start manifest | `src/custom_sam_peft/train/runner.py` |
| `csp doctor --config <path>` with new `DatasetResolution` section | `src/custom_sam_peft/diagnostics.py`, `cli/doctor_cmd.py` |
| Test suite (all CPU) | `tests/data/test_subset.py`, `test_coco_limit.py`, `test_hf_limit.py`, `tests/cli/test_doctor_config.py`, `tests/train/test_runner_limit.py` |
| Example YAML | `configs/examples/coco_text_lora_subset.yaml` |

**Out of scope (explicit):**

- `train.max_steps` — a separate mechanism, separate issue if wanted.
- Replacing `configs/examples/gpu_smoke_*.yaml` and `data/placeholder/` — enabled by this work but a separate cleanup PR.
- `data.limit.pool: bool` (cap-after-pool for multiplex sources, issue #22) — decision recorded in §6; field lands in #22, not here.
- Folder adapter `image_class_labels` (issue #33) — folder adapter not yet implemented; stratified degrades to random with a WARN.
- Auto-split integration (issue #71) — composition contract recorded in §6; wiring lands in #71.

---

## 2. Config Schema

`LimitConfig` mounts on `DataConfig` as `data.limit`.

```python
class LimitConfig(_Strict):
    train: int | float | None = None   # int >= 1 (cap), float in (0, 1] (fraction), None = no-op
    val:   int | float | None = None
    seed:  int = 42                    # distinct from run.seed; salts the subset RNG
    strategy: Literal["random", "stratified", "first_n"] = "random"
```

`DataConfig` grows one field: `limit: LimitConfig = Field(default_factory=LimitConfig)`.

A `@model_validator(mode="after")` on `LimitConfig` enforces:
- `bool` values for `train`/`val` are rejected — Pydantic v2 accepts `bool` as `int` by default; the validator checks `isinstance(v, bool)` before the numeric check and raises `ValueError("limit.train must not be a bool")`.
- `int` values must satisfy `>= 1`; `int < 1` raises.
- `float` values must satisfy `> 0.0` and `<= 1.0`; `float <= 0` and `float > 1` both raise.

`strategy: "stratified"` is always schema-legal. If the format builder cannot supply `image_class_labels`, the runtime degrades to `"random"` with a WARN (§3.2). Schema validation never rejects it.

When both `train` and `val` are `None` (the default `LimitConfig()`), no `SubsetDataset` is constructed and `_build_dataset` returns the inner dataset unchanged — zero behavior change for existing configs.

---

## 3. Sampling — pure function

### 3.1 `src/custom_sam_peft/data/subset.py`

The new module exposes two public surfaces: a wrapper class and a standalone sampling function.

```python
def resolve_subset_indices(
    n_total: int,
    limit: int | float,
    *,
    seed: int,
    strategy: Literal["random", "stratified", "first_n"],
    image_class_labels: Sequence[Sequence[int]] | None,
) -> list[int]:
    """Return sorted-ascending unique indices in [0, n_total).

    Length of result <= min(cap, n_total) where cap is derived from limit:
      int  → min(limit, n_total); warns if limit > n_total.
      float → max(1, round(limit * n_total)); 1.0 returns full range.
    """
```

**Cap resolution.** An `int` limit becomes the cap directly, clipped to `n_total` with a WARN logged when `limit > n_total` (`"limit.train=%d exceeds dataset size %d; using full dataset"`). A `float` limit becomes `max(1, round(fraction * n_total))`; fraction `1.0` yields the full range. The `SubsetDataset` is still constructed in that case — the caller in `_build_dataset` decides whether to wrap based on the config value being non-`None`, not on the resolved cap.

**`first_n`.** Returns `list(range(min(cap, n_total)))`. Stable, deterministic, ignores seed and `image_class_labels`. Useful for reproducible "always the first N images" workflows.

**`random`.** Seeds a local `random.Random` instance with the string `f"{seed}:{n_total}:{strategy}"`. Shuffles `range(n_total)`, takes the first `cap` indices, returns sorted ascending. The seed salt includes `n_total` so that re-runs on the same dataset pick the same subset, but changing the dataset size (e.g. adding a few images) changes the subset. This is intentional — `data.limit` is for fast iteration, not for maintaining a permanent fixed partition.

**`stratified`.** Multi-label proportional sampling via an iterative re-weighting approach (Sechidis et al. 2011). Tracks per-class remaining quota proportional to source class frequencies; at each step selects the image whose rarest still-needed class has the highest remaining deficit; tie-breaks by the class with the smallest current quota; among remaining ties, by image index (deterministic). After the greedy pass, if the quota leaves the total short, fills with a seeded random draw from the unselected pool. The implementation is approximately 50 lines of pure Python/NumPy with no new dependencies. If `image_class_labels is None`, logs a WARN and calls `resolve_subset_indices` recursively with `strategy="random"` and the same remaining arguments.

**Output contract.** The returned list has length `<= n_total`, indices are unique, and the list is sorted ascending. Sorting matters because DataLoader workers iterate in index order — preserving relative order keeps any inner-dataset prefetching useful.

### 3.2 `SubsetDataset` wrapper

```python
class SubsetDataset:
    def __init__(self, inner: Dataset, indices: list[int]) -> None: ...
    def __len__(self) -> int: ...              # len(self._indices)
    def __getitem__(self, i: int) -> Example: ...  # self._inner[self._indices[i]]
    @property
    def class_names(self) -> list[str]: ...    # self._inner.class_names
```

The wrapper is fully transparent to the trainer. It satisfies the `Dataset` Protocol from `data/base.py`. The inner dataset never sees the subset — indexing is entirely at the wrapper layer.

---

## 4. Per-format hooks for stratification labels

`resolve_subset_indices` needs per-image class labels when `strategy="stratified"`. Each format exposes `image_class_labels` as a property; the caller in `_build_dataset` does a `getattr(inner, "image_class_labels", None)` duck-type check before calling `resolve_subset_indices`.

**`COCODataset.image_class_labels` (eager).** Built once at the end of `__init__`, after `_ann_index` is populated. Returns `list[frozenset[int]]` of dense class IDs per image, in the same order as `_image_ids`. The data is already in memory (it's a dict lookup over `_ann_index`), so the cost is negligible.

**`HFDataset.image_class_labels` (lazy + cached).** Computed on first access via `@property` with a `_image_class_labels: list[frozenset[int]] | None = None` cache sentinel. On first access, if cache is `None`: logs one INFO line `"stratified subset: scanning N rows for class labels…"`, then scans every row's `objects.category` field via `_resolve_field`, builds the list, caches it. Subsequent accesses return the cached value. Called only when `strategy="stratified"` — the `_build_dataset` code never touches `image_class_labels` for the `random` or `first_n` strategies.

**Folder adapter (issue #33, not yet implemented).** The spec records the contract: when the folder adapter is eventually implemented, it must expose `image_class_labels` to unlock stratified sampling. If absent, `getattr` returns `None` and `resolve_subset_indices` logs a WARN and falls back to `random`.

---

## 5. Integration — `_build_dataset` in `train/runner.py`

`_build_dataset` is the single integration seam. After the format builder returns the inner dataset, the function checks `cfg.data.limit` and conditionally wraps:

```python
def _build_dataset(cfg: TrainConfig, pipeline: str) -> Dataset:
    builder = lookup("dataset", cfg.data.format)
    inner = cast(Dataset, builder(cfg.data.model_dump(), model_name=cfg.model.name, pipeline=pipeline))
    lim_cfg = cfg.data.limit
    limit_val = lim_cfg.train if pipeline == "train" else lim_cfg.val
    if limit_val is None:
        return inner
    labels = getattr(inner, "image_class_labels", None)
    indices = resolve_subset_indices(
        len(inner),
        limit_val,
        seed=lim_cfg.seed,
        strategy=lim_cfg.strategy,
        image_class_labels=labels,
    )
    _LOG.info(
        "data.limit applied: %s=%d/%d (strategy=%s, seed=%d)",
        pipeline, len(indices), len(inner), lim_cfg.strategy, lim_cfg.seed,
    )
    return SubsetDataset(inner, indices)
```

The log line format is `"data.limit applied: train=64/118287 (strategy=random, seed=42)"`.

**`subset.json` manifest.** After both datasets are built in `run_training`, the runner writes `<run_dir>/subset.json` before constructing the trainer. The file records resolved indices per side; a side is omitted if no limit applied to it:

```json
{
  "limit": {"train": 64, "val": 16, "seed": 42, "strategy": "random"},
  "train": {"n_total": 118287, "n_kept": 64, "indices": [12, 47, ...]},
  "val":   {"n_total": 5000,   "n_kept": 16,  "indices": [3, 91, ...]}
}
```

The indices are recorded so exact subset reproducibility can be verified or replayed. File size is a few KB even for cap=10 000. The `"limit"` block is always written when the file is produced (i.e., at least one side has a non-`None` limit). When both sides are `None`, the file is not written.

The `Tracker.start_run` call already receives the full config dict — `cfg.data.limit` fields are included automatically via `cfg.model_dump()`, so no additional tracking change is needed.

---

## 6. Composition — recorded contracts

**Issue #71 (auto-split).** Auto-split runs first, producing two `Dataset` instances; limit wrapping then applies independently to each side. No leakage occurs because auto-split decides train/val membership before either side sees a `SubsetDataset`. Issue #71 is not yet merged; this PR only handles the explicit-`train`+`val` path. When #71 lands, its PR threads `SubsetDataset` into the auto-split outputs using the same `_build_dataset` hook.

**Issue #22 (multiplex).** By default, `data.limit` applies per source (each source still contributes up to `cap` items). An optional `data.limit.pool: bool = false` field will switch to cap-after-pooling. That schema field lands in #22, not in this PR. The contract is recorded here so #22's implementer has a fixed interface to add to.

**Issue #33 (folder adapter).** Subset support is automatic via the wrapper once the folder adapter exposes `__len__` and `__getitem__`. Stratified support requires the `image_class_labels` property as described in §4; without it, the fallback to `random` applies.

---

## 7. `csp doctor --config <path>`

The doctor's existing contract (cheap, no-network, pure-environment) is preserved when `--config` is not provided. The `--config` path is new and explicitly heavy: it loads the YAML, validates the config, builds train and val datasets (which imports `pycocotools` or `datasets.load_dataset`), and reports resolved sizes.

`run_doctor` gains an optional `config_path: Path | None = None` parameter. When `None`, the new `dataset` field on `DoctorReport` is `None` and the CLI renderer skips the section — existing behavior is preserved bit-for-bit.

**New dataclass** added to `src/custom_sam_peft/diagnostics.py`:

```python
@dataclass(frozen=True)
class DatasetResolution:
    format: str
    train_total: int
    train_kept: int
    val_total: int
    val_kept: int
    limit_strategy: str
    limit_seed: int
    limit_train: int | float | None
    limit_val: int | float | None
```

`DoctorReport` gains `dataset: DatasetResolution | None = None` immediately before `issues`.

**Population in `run_doctor`.** When `config_path` is provided, `run_doctor` does:

1. Loads and validates the YAML via `load_config(config_path)`. On `ValidationError` or file-not-found, appends to `issues` and leaves `dataset=None`.
2. Calls `_build_dataset(cfg, "train")` and `_build_dataset(cfg, "eval")`. Catches any exception, appends `"couldn't build train/val dataset: <error>"` to `issues`, and leaves `dataset=None` — the doctor never crashes.
3. On success, constructs `DatasetResolution` using `len(train_ds)`, `len(val_ds)`, and the resolved `SubsetDataset._indices` lengths vs the inner dataset sizes. When no limit applies to a side, `train_total == train_kept`.

**CLI renderer.** `_render_table` adds a "Dataset" sub-table after the SAM 3.1 weights block when `report.dataset is not None`:

```python
ds = report.dataset
tbl = Table(title="Dataset", show_header=False, box=None)
tbl.add_row("format", ds.format)
tbl.add_row("train", f"{ds.train_kept}/{ds.train_total}")
tbl.add_row("val",   f"{ds.val_kept}/{ds.val_total}")
tbl.add_row("limit.strategy", ds.limit_strategy)
tbl.add_row("limit.seed", str(ds.limit_seed))
tbl.add_row("limit.train", str(ds.limit_train))
tbl.add_row("limit.val",   str(ds.limit_val))
console.print(tbl)
```

`--json` picks up the new field automatically via `dataclasses.asdict`.

The CLI gains `--config` as a new Typer option on the `doctor` command:

```python
config_path: Path | None = typer.Option(None, "--config", help="Load + validate a config YAML and report resolved dataset sizes.")
```

---

## 8. Testing

All tests are CPU-only per `feedback_gpu_vs_cpu_testing.md`. Real COCO/HF data never touches the test suite.

### 8.1 `tests/data/test_subset.py`

Pure-function and wrapper unit tests. No format-specific fixtures needed.

Schema validation cases (table-driven): `None` → valid; `int >= 1` → valid; `float in (0, 1]` → valid; `True`/`False` → `ValidationError`; `0` → `ValidationError`; `-1` → `ValidationError`; `0.0` → `ValidationError`; `1.1` → `ValidationError`. Both `train` and `val` fields tested.

`resolve_subset_indices` cases: `first_n` returns ascending range, respects cap, ignores seed. `random` returns correct count, sorted, unique, deterministic given same seed+n_total. `random` with different `n_total` returns a different set (salt behavior). `stratified` on a hand-crafted 20-image/4-class fixture returns correct count, respects class proportions. `stratified` with `image_class_labels=None` logs a WARN and produces a valid `random` result. Cap > n_total WARNs and returns all indices.

`SubsetDataset` delegation: `__len__` returns `len(indices)`, `__getitem__(i)` delegates to `inner[indices[i]]`, `class_names` delegates. Dataset Protocol structural check passes.

### 8.2 `tests/data/test_coco_limit.py`

Uses the existing tiny synthetic COCO fixture from `tests/fixtures/tiny_coco/`. Tests: int cap wraps correctly (resulting `len(ds)` matches cap); fraction cap rounds correctly; `strategy="stratified"` with a fixture that has multiple classes preserves all classes in the subset; `image_class_labels` is populated at init time and has length == `len(inner)`.

### 8.3 `tests/data/test_hf_limit.py`

Mocks `datasets.load_dataset` with an in-memory `datasets.Dataset.from_dict`. Tests: `image_class_labels` is NOT computed (cache sentinel stays `None`) when `strategy="random"` or `"first_n"` — verified by asserting `_image_class_labels is None` after building the dataset and calling `_build_dataset` with those strategies. `image_class_labels` IS computed on first access when `strategy="stratified"` and the INFO scan line is logged. Second access returns cached result without a second scan log.

### 8.4 `tests/cli/test_doctor_config.py`

Happy path: `csp doctor --config <path-to-valid-yaml>` prints a "Dataset" section with `train`/`val` rows. Bad config path (file does not exist): doctor exits 0 (issues surfaced, no crash), and `"config"` or `"load"` appears in the issues output. Un-buildable dataset (patch `_build_dataset` to raise): doctor exits 0 and `"couldn't build"` appears in issues. JSON output: `csp doctor --config … --json` produces `blob["dataset"]` with all fields; `blob["dataset"]` is `null` when no `--config` flag.

### 8.5 `tests/train/test_runner_limit.py`

Patches `lookup("dataset", …)` to return a stub that satisfies the `Dataset` Protocol. Both limits `None`: `_build_dataset` returns the inner dataset instance directly (no `SubsetDataset`). `train` limit set: `_build_dataset` returns a `SubsetDataset` with `len` == min(cap, stub_len). `subset.json` is written to `run_dir` after `run_training` when at least one side has a limit; not written when both are `None`.

Existing tests stay green via the default `LimitConfig()` keeping both `train` and `val` as `None`.

---

## 9. Example YAML

`configs/examples/coco_text_lora_subset.yaml` is a copy of `coco_text_lora.yaml` with a `data.limit` block added:

```yaml
data:
  format: coco
  train:
    annotations: data/coco/instances_train2017.json
    images: data/coco/train2017
  val:
    annotations: data/coco/instances_val2017.json
    images: data/coco/val2017
  prompt_mode: text
  image_size: 1024
  limit:
    train: 64
    val: 16
    seed: 42
    strategy: random
```

The file's header comment explains the purpose: "Demonstrates data.limit for fast sanity-check runs on a real COCO dataset. Remove or comment out the limit block for full training."

---

## 10. Deliverables

1. This spec at `docs/superpowers/specs/2026-05-22-data-subset-limit-design.md`.
2. `LimitConfig` + `DataConfig.limit` in `src/custom_sam_peft/config/schema.py`.
3. `src/custom_sam_peft/data/subset.py` — `SubsetDataset` + `resolve_subset_indices` (including multi-label stratified).
4. `image_class_labels` property on `COCODataset` (eager) and `HFDataset` (lazy + cached).
5. Updated `_build_dataset` in `src/custom_sam_peft/train/runner.py` — wrap, INFO log, `subset.json`.
6. `csp doctor --config` wired through `cli/doctor_cmd.py` → `diagnostics.run_doctor` → new `DatasetResolution` on `DoctorReport`.
7. All tests listed in §8.
8. `configs/examples/coco_text_lora_subset.yaml`.

---

## 11. Assumptions

1. `data.limit.seed` is intentionally separate from `run.seed`. Subset selection is a data concern; reproducibility of a subset should not change when the training seed changes (and vice versa).
2. The RNG salt `f"{seed}:{n_total}:{strategy}"` is the full contract for the `random` strategy. Changing any of these three values produces a different subset — this is documented behavior, not a bug.
3. Stratified sampling has no new runtime dependency. The iterative re-weighting algorithm uses only `list` and basic Python arithmetic; the multi-label proportional behavior on the test fixture (§8.1) is the acceptance criterion, not any particular external library.
4. `SubsetDataset` does not expose `coco_category_ids` or any other format-specific attribute. Callers that need `coco_category_ids` (the eval loop) must reach through `SubsetDataset._inner` or access the attribute before wrapping. The spec records this as a known limitation; eval is not affected in this PR because the eval path uses `class_names` not `coco_category_ids` at the data-layer boundary.
5. `_build_dataset` resolves `image_class_labels` from the inner dataset (before wrapping) so stratified sampling has access to the full-population class distribution. The selected subset may then be wrapped in `SubsetDataset`. The class labels list passed to `resolve_subset_indices` always has length == `len(inner)`.
6. The `subset.json` indices are the logical indices into the inner dataset (i.e., `SubsetDataset._indices`), not physical file positions. For COCO, index `i` maps to `_image_ids[i]`; for HF, index `i` maps to row `i` of the loaded split.
7. Doctor's `--config` path is "heavy by design" — it imports data dependencies and may trigger `datasets.load_dataset` network calls. The doc string and CLI help text will say so explicitly. The existing no-`--config` path remains cheap and network-free.

End of spec.
