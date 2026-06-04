# spec/train-val-test-split — 3-way train/val/test auto-split

**Status:** Draft (2026-06-04)
**Scope:** Generalize the existing 2-way train+val auto-split into a 3-way
train/val/test auto-split. A single labeled dataset (`data.train`) is carved
deterministically into train, val, and (optionally) test buckets in one joint
Sechidis-2011 stratification pass. Test is held out from training entirely and
evaluated only on demand via `csp eval --split test`.

**Builds on / supersedes:**
[`2026-05-22-data-no-val-auto-split-design.md`](2026-05-22-data-no-val-auto-split-design.md) —
the 2-way auto-split this spec generalizes. That design's `data.val_split`
(`ValSplitConfig`) is **removed** and replaced by `data.split` (`SplitConfig`);
the resolver module `data/val_source.py` (`ValSource` /
`resolve_val_source` / `val_source.json`) is **renamed** to `data/split_source.py`
(`SplitSource` / `resolve_split_source` / `split_source.json`).

**Breaking change.** This is a hard rename with **no deprecation alias and no
warning**. External and local configs that still set `data.val_split` get a clean
`_Strict` unknown-key validation error (`extra="forbid"`, schema.py line ~28).
In-flight runs are re-launched (see §9, resume).

---

## 1. Goals & Non-Goals

### 1.1 Goals

- Carve a single `data.train` pool into `{train, val, test}` deterministically,
  in one joint pass, generalizing the current 2-subset stratification.
- `data.split: { val, test, seed }` selects the carve. At least one of `val` /
  `test` must be set; either may be omitted independently.
- `test`-omitted reproduces today's 2-way train+val behavior *exactly*
  (the `t = 0` / test-absent fallback case).
- `val`-omitted (test-only) is **allowed** → trains without a validation set,
  identical to today's no-val mode for the val side: `eval_every` is a no-op, no
  early-stop, no end-of-run eval, no bundle samples.
- A single source of truth — `<run_dir>/split_source.json` — records the resolved
  3-way partition and is authoritative on resume (carried forward verbatim).
- One annotations file → one category remap (sparse→dense) + one class-name
  vocabulary, **shared across all three buckets** (guaranteed by the splitter
  enumerating from `data.train`).
- `csp eval --split test` resolves test ids from either explicit `data.test` or
  `data.split.test`.

### 1.2 Non-goals (deferred)

- Per-fraction `le=0.5` caps. The old cap is **dropped** (§3.3); replaced by a
  runtime small-bucket / deviation WARN.
- A `data.source` field. `data.train` is the shared anchor for both modes; the
  provenance discriminator (explicit `data.val`/`data.test` vs `data.split`) sits
  on top of it (§4, considered-and-rejected).
- Cross-validation / k-fold. Out of scope as in the predecessor.
- Strict per-image stability under dataset edits. The run-dir saved split is the
  practical reproducibility hook, unchanged from the predecessor.
- HF `split_val`/`split_test` test carving. `data.split` remains incompatible
  with `data.hf.split_val` (§4); HF test auto-split is out of scope.

---

## 2. Motivation

The predecessor delivered train→train/val auto-split. The common workflow of
"I have one labeled set; carve it into train, val, *and* a held-out test set I
score once at the end" still requires hand-splitting annotation files. This spec
closes that gap by promoting the 2-subset splitter to its native k-subset
(k≤3) form and threading a `test_ids` bucket through the resolver, the persisted
record, and the eval runner. The test bucket is **never** auto-evaluated — it is
strictly post-hoc via `csp eval --split test`, so it cannot leak into model
selection during training.

---

## 3. Schema

### 3.1 Removed: `ValSplitConfig`

`ValSplitConfig` (schema.py line ~369) and `DataConfig.val_split` (line ~463)
are **deleted**. No alias. A config setting `data.val_split` now fails `_Strict`
validation with an unknown-key error naming `val_split`.

### 3.2 New: `SplitConfig`

```python
class SplitConfig(_Strict):
    """3-way auto-split parameters. Used when DataConfig.split is set.

    Carves data.train into train + val + test deterministically via a single
    joint Sechidis-2011 multi-label stratification pass. At least one of
    val / test must be set and > 0.

    In v0 (inherited from the predecessor's locked decisions):
      - stratification is always-on Sechidis multi-label iterative; not
        configurable.
      - split unit is always 'image'; not configurable. Splitting by
        annotation can leak the same image across buckets.

    Spec: docs/superpowers/specs/2026-06-04-train-val-test-split-design.md §3.2.
    """

    val: float | None = None   # carve this fraction into val;  None → no val bucket
    test: float | None = None  # carve this fraction into test; None → no test bucket
    seed: int | None = None    # None → inherit run.seed at resolve time

    @model_validator(mode="after")
    def _check_fractions(self) -> SplitConfig:
        if self.val is None and self.test is None:
            raise ValueError(
                "data.split requires at least one of val / test to be set "
                "(both omitted carves nothing)."
            )
        for name in ("val", "test"):
            v = getattr(self, name)
            if v is not None and not (0.0 < v < 1.0):
                raise ValueError(
                    f"data.split.{name}={v!r} must be in (0.0, 1.0) (exclusive)."
                )
        carved = (self.val or 0.0) + (self.test or 0.0)
        if carved >= 1.0:
            raise ValueError(
                f"data.split.val + data.split.test = {carved} must be < 1.0 so the "
                "train bucket is non-empty."
            )
        return self
```

**Defaults.** `val`/`test`/`seed` all default to `None`. No numeric default
fraction ships (the predecessor's `fraction: float = 0.1` default is *not*
carried over — `val`/`test` are opt-in per-bucket). `seed: None → inherit
run.seed` follows the predecessor verbatim (cite:
[`2026-05-22-data-no-val-auto-split-design.md`](2026-05-22-data-no-val-auto-split-design.md) §3.1).

> The old `Field(gt=0.0, le=0.5)` per-fraction cap is intentionally **dropped**.
> A 60/20/20 or 50/30/20 carve is now legal. The `< 1.0` joint guard is the only
> hard bound; oversized buckets surface as a runtime WARN (§3.3, §6.5), not a
> schema error. (cite: locked brainstorm decision, 2026-06-04 — the `le=0.5` cap
> was a 2-way artifact that makes no sense once test exists.)

### 3.3 `DataConfig` diff

```python
class DataConfig(_Strict):
    format: DataFormat
    train: DataSplit                     # the SINGLE always-required source (§4)
    val: DataSplit | None = None
    split: SplitConfig | None = None     # was: val_split: ValSplitConfig | None
    ...
    # --- advanced ---
    test: DataSplit | None = None
    hf: HFDatasetConfig | None = None

    @model_validator(mode="after")
    def _check_split_modes(self) -> DataConfig:
        if self.split is not None and self.val is not None:
            raise ValueError(
                "data.split and data.val are mutually exclusive. Use data.split to "
                "carve val/test from data.train, or data.val for an explicit val set."
            )
        if self.split is not None and self.test is not None:
            raise ValueError(
                "data.split and data.test are mutually exclusive. Use data.split.test "
                "to carve a test set from data.train, or data.test for an explicit one."
            )
        return self

    @model_validator(mode="after")
    def _check_hf_split_compat(self) -> DataConfig:
        if (
            self.format == "hf"
            and self.split is not None
            and self.hf is not None
            and self.hf.split_val is not None
        ):
            raise ValueError(
                "data.hf.split_val cannot be customized when data.split is set; "
                "auto-split carves val/test from data.hf.split_train. "
                "Remove split_val or remove split."
            )
        return self
```

The existing `_check_val_modes` / `_check_hf_split_val_compat` validators
(schema.py lines ~546-568) are **replaced** by the two above (renamed and the
test-exclusivity clause added).

### 3.4 Resolved modes

| YAML state | Resolved val mode | Resolved test source |
| --- | --- | --- |
| `data.val: {…}` | `explicit` | `data.test` if set, else none |
| `data.split: {val: 0.1, test: 0.1}` | `auto_split` | auto_split (carved) |
| `data.split: {val: 0.1}` (test omitted) | `auto_split` | none (2-way fallback) |
| `data.split: {test: 0.1}` (val omitted) | `none` (WARN once) | auto_split (carved) |
| `data.split` + `data.val` | schema error (`_check_split_modes`) | — |
| `data.split` + `data.test` | schema error (`_check_split_modes`) | — |
| neither `val` nor `split` (COCO) | `none` (WARN once) | `data.test` if set, else none |

### 3.5 Example YAML (split-mode)

```yaml
data:
  format: coco
  train:
    annotations: data/coco/all.json
    images: data/coco/images
  split:
    val: 0.1
    test: 0.1
    seed: 0
```

Test-only minimal form: `data: {train: {annotations: all.json, ...}, split: {test: 0.2}}`.

### 3.6 Backwards compatibility

`data.split` *replaces* `data.val_split`. Configs that set `data.val_split` no
longer validate (clean `_Strict` error). This is the deliberate breaking-change
behavior — no shim, no alias, no deprecation warning. The in-repo example YAMLs
and emitters are migrated in the same PR (§8).

---

## 4. Anchor model & mutual exclusivity

**`data.train` is the single always-required source — "the labeled data you
start with."** In split-mode it is the whole pool that val/test are carved out
of; in explicit mode it is the pre-made train split. Both modes share this one
anchor; the provenance discriminator (whether val/test are *carved* or
*supplied*) lives in `data.split` vs `data.val`/`data.test`.

**Considered and rejected: a `data.source` field.** We deliberately do *not* add
a separate `data.source` to name "the pool to carve from." `data.train` already
*is* that pool in split-mode, and introducing a second field would create a
redundant two-anchor surface (which is authoritative when both are set?) and a
migration burden on every explicit config. Keeping `data.train` as the shared
anchor with the split block layered on top is simpler and keeps explicit configs
untouched.

**Mutual exclusivity** (generalizes the predecessor's `val` vs `val_split` rule):

- `data.split` is exclusive with explicit `data.val` (schema error).
- `data.split` is exclusive with explicit `data.test` (schema error) — new.
- `data.split` remains incompatible with `data.hf.split_val` (schema error).

**Shared-vocabulary guarantee.** Because the splitter enumerates items from
`data.train`'s single annotations file, the category remap (sparse→dense) and the
class-name vocabulary are computed once and shared identically across train, val,
and test buckets. There is no per-bucket remap drift; the dense class id space is
the same in all three. State this as a guaranteed property of split-mode (it falls
out of the enumeration; it is *not* guaranteed for explicit mode, where val/test
come from independent files).

---

## 5. Splitter — generalize to multi-subset Sechidis (joint single-pass)

`data/splitter.py` `stratified_split` is reworked from a 2-subset specialization
into the native k-subset form of Sechidis 2011 (cite: Sechidis, Tsoumakas,
Vlahavas 2011, *On the Stratification of Multi-Label Data*, ECML PKDD — the
algorithm the predecessor already cites). Items are distributed across
`{train, val, test}` in **one joint pass** with target proportions
`(1 − v − t, v, t)`.

**Why joint, not sequential two-pass.** A sequential carve (first pull val out,
then pull test out of the remainder) stratifies each bucket against a *different*
residual pool, degrading joint-stratification accuracy for rare classes. The
single joint pass stratifies rare classes across all present buckets
simultaneously, which is the accuracy-first choice (cite: locked brainstorm
decision, 2026-06-04; accuracy > simplicity per repo priority order).

### 5.1 Public API

```python
@dataclass(frozen=True)
class SplittableItem:           # UNCHANGED
    image_id: str
    class_ids: frozenset[int]


@dataclass(frozen=True)
class SplitResult:
    train_ids: tuple[str, ...]
    val_ids: tuple[str, ...]
    test_ids: tuple[str, ...]                                   # NEW
    realized_fraction: tuple[float, float]                      # (val, test) realized
    per_class_counts: dict[int, tuple[int, int, int]]           # (train, val, test)
    missing_in_val: tuple[int, ...]
    missing_in_test: tuple[int, ...]                            # NEW (mirror of missing_in_val)


def stratified_split(
    items: Sequence[SplittableItem],
    val_fraction: float,        # 0.0 ⇒ val bucket absent
    test_fraction: float,       # 0.0 ⇒ test bucket absent
    seed: int,
) -> SplitResult:
    """Sechidis 2011 joint multi-label stratification into train/val/test.

    Deterministic given (items, val_fraction, test_fraction, seed): items are
    sorted by image_id before processing so caller ordering does not matter.
    The (val_fraction=v, test_fraction=0) case reproduces the prior 2-way
    result bit-for-bit.
    """
```

> `realized_fraction` becomes a 2-tuple `(val, test)` and `per_class_counts` a
> 3-tuple `(train, val, test)` — both are breaking shape changes, but all
> consumers are internal (§7) and migrated in this PR.

### 5.2 Algorithm — joint k-subset placement

1. **Sort input.** `items_sorted = sorted(items, key=lambda it: it.image_id)`.
2. **Present buckets.** `buckets = ["train", "val"]`; append `"test"` iff
   `test_fraction > 0`. (`val` is always present as a bucket name even when
   `val_fraction == 0` so the 2-way `(v, t=0)` path stays structurally identical
   to today; a `val_fraction == 0` simply yields an empty `val_ids`.)
   Train proportion is `1 − val_fraction − test_fraction`.
3. **Per-bucket quotas.** `N = len(items_sorted)`. For each present bucket `b`
   with target fraction `f_b`, set `remaining[b] = round(N * f_b)` (train gets
   the remainder `N − sum(other buckets)` so totals reconcile). For each class
   `c` with total count `n_c`, set `quota[b][c] = round(n_c * f_b)` for val/test;
   train's per-class quota is `n_c − sum(val/test quotas)`.
4. **Initial ordering.** Sort items by `(min_class_count_in_item, rng.random())`
   ascending, empty-class items use `math.inf`. Seeded `random.Random(seed)`
   provides the tiebreak. Rarest-class items placed first. UNCHANGED.
5. **Greedy placement.** Walk items in that order. For each item, restrict to the
   **present** buckets that still have capacity (`remaining[b] > 0`); if only one
   has capacity, place there. Otherwise score each candidate bucket as
   `score[b] = max((quota[b][c] for c in item.class_ids), default=remaining[b])`
   and place in the highest-scoring bucket. Tie-break: prefer the bucket with the
   largest `remaining`; still tied → seeded RNG choice among the tied buckets
   (deterministic ordering of the tied set). Decrement `remaining[chosen]` and
   each `quota[chosen][c]`.
6. **Post-checks.** `realized_fraction = (len(val)/max(N,1), len(test)/max(N,1))`.
   `per_class_counts[c] = (train_count, val_count, test_count)`. For each class
   `c` with `n_c >= 2`: record in `missing_in_val` if `val_count == 0`, in
   `missing_in_test` if `test_count == 0` **and the test bucket is present**
   (test-absent ⇒ `missing_in_test == ()`). `train_ids`/`val_ids`/`test_ids` are
   sorted before return.

**Determinism contract:** identical `(items, val_fraction, test_fraction, seed)`
⇒ bit-identical `SplitResult`. The sort-by-`image_id` + seeded-RNG tiebreak +
sorted-output structure is **unchanged** from today.

**2-way fallback invariant:** `stratified_split(items, v, 0.0, seed)` must yield
`test_ids == ()`, `missing_in_test == ()`, `realized_fraction[1] == 0.0`, and
`train_ids`/`val_ids` **bit-identical** to the prior 2-arg
`stratified_split(items, v, seed)`. The greedy placement among `{train, val}`
(test absent) reduces to the existing two-side scoring. This invariant is
asserted by a test that pins the old fixtures (§10.1).

### 5.3 Edge cases

| Case | Behavior |
| --- | --- |
| `N == 0` | `SplitResult((), (), (), (0.0, 0.0), {}, (), ())`. |
| `N == 1` | All to train (rounding sends val/test quotas to 0 at small fractions). `realized_fraction == (0.0, 0.0)`. |
| `test_fraction == 0.0` | Test bucket absent; 2-way fallback (§5.2 invariant). |
| Empty `class_ids` items | Placed by capacity score only; never in `per_class_counts`; never in `missing_in_*`. |
| `round(N * f) == 0` for a present bucket | That bucket gets `()`. Resolver WARNs (§6.5). |
| Rare/singleton class (`n_c == 1`) | Placed greedily in one bucket; never recorded as missing in any bucket (the `n_c >= 2` guard). |
| Rare class `n_c == 2`, 3 buckets | Stratified jointly; may land in 2 of 3 buckets → recorded in the absent bucket's `missing_in_*`. |
| `val_fraction + test_fraction` near 1.0 | Schema guard (§3.2) forbids `>= 1.0`; tiny realized train surfaces as a WARN. |

### 5.4 Complexity

`O(N · C_avg · k)` with `k ≤ 3`. Unchanged order; still sub-second on COCO
train2017 and off the training hot path.

---

## 6. Resolved record — rename + extend (all internal)

The resolver module is **renamed** wholesale. No external contract exists; every
reader is in-repo (§7).

| Before | After |
| --- | --- |
| file `data/val_source.py` | `data/split_source.py` |
| dataclass `ValSource` | `SplitSource` |
| `resolve_val_source` | `resolve_split_source` |
| `save_val_source` / `load_val_source` | `save_split_source` / `load_split_source` |
| `_log_val_source` | `_log_split_source` |
| `ValMode` alias | `SplitMode` (values unchanged: `explicit`/`auto_split`/`none`) |
| persisted `<run_dir>/val_source.json` | `<run_dir>/split_source.json` |

The private enumerators (`_enumerate_items`, `_enumerate_coco_items`,
`_enumerate_hf_items`) keep their names and behavior (they already enumerate from
`data.train`).

### 6.1 `SplitSource` dataclass

```python
SplitMode = Literal["explicit", "auto_split", "none"]


@dataclass(frozen=True)
class SplitSource:
    mode: SplitMode                                          # val-side mode (unchanged semantics)
    train_ids: tuple[str, ...] | None
    val_ids: tuple[str, ...] | None
    test_ids: tuple[str, ...] | None                         # NEW; populated when split.test set
    realized_fraction: tuple[float, float] | None            # (val, test); auto_split only
    per_class_counts: dict[int, tuple[int, int, int]] | None # (train, val, test)
    missing_in_val: tuple[int, ...] | None
    missing_in_test: tuple[int, ...] | None                  # NEW
    val_fraction_requested: float | None                     # auto_split only
    test_fraction_requested: float | None                    # NEW; auto_split only
    seed_used: int | None
```

> `mode` continues to describe the **val side** only (`auto_split` when a val
> bucket was carved, `none` when val omitted). The test side is described by
> `test_ids is not None` / `test_fraction_requested`. This keeps the existing
> val-mode branching (trainer, bundle, run_cmd) working unchanged — a test-only
> split resolves to `mode="none"` (no val) with `test_ids` populated.

### 6.2 `resolve_split_source` dispatch

Generalizes the existing 5-case dispatch (val_source.py lines ~41-105):

1. `run_dir` has `split_source.json` → `load_split_source(run_dir)` (resume).
2. `cfg.data.split is not None` → enumerate items; resolve
   `seed_used = cfg.data.split.seed if not None else cfg.run.seed`; call
   `stratified_split(items, cfg.data.split.val or 0.0, cfg.data.split.test or 0.0,
   seed_used)`. Build `SplitSource` with:
   - `mode = "auto_split"` if `cfg.data.split.val` set, else `"none"`.
   - `val_ids = result.val_ids` (empty tuple when val omitted).
   - `test_ids = result.test_ids if cfg.data.split.test else None`.
   - `*_fraction_requested` from the config; counts/missing from the result.
3. `cfg.data.val is not None` → `mode="explicit"`, ids `None`,
   `test_ids = None` (explicit test is read from `cfg.data.test` directly by the
   eval runner, not threaded through the record).
4. HF `split_val` explicit → `mode="explicit"` (unchanged).
5. else → `mode="none"`, all `None`.

### 6.3 `save_split_source` / `load_split_source`

`save_split_source` writes `<run_dir>/split_source.json` (atomic tmp + os.replace,
unchanged) with the keys: `mode`, `val_fraction_requested`,
`test_fraction_requested`, `seed_used`, `realized_fraction` (a `[val, test]`
list), `n_train`, `n_val`, `n_test`, `per_class_counts`
(`{str(class): [train, val, test]}`), `missing_in_val`, `missing_in_test`,
`train_ids`, `val_ids`, `test_ids`. `load_split_source` re-casts string class
keys to int and tuples the id lists, mirroring today; new fields rehydrate
symmetrically; `n_test` derives from `len(test_ids)`.

### 6.4 `_log_split_source`

Mirrors today's val logging and **adds** a test branch:

- INFO when `auto_split`: existing val line, plus a test line when `test_ids` is
  populated: `"auto-split test: realized=test=%d (%.2f%%); coverage=%d/%d classes
  in test"`.
- WARN once if `missing_in_test` is non-empty (mirror of the `missing_in_val`
  WARN).
- The deviation/small-bucket WARN (§6.5) extended per-bucket.

### 6.5 WARN policy (replaces the dropped `le=0.5` cap)

The schema no longer caps fractions, so the resolver carries the safety net:

- WARN per present bucket if
  `abs(realized − requested) / requested > 0.2` OR the realized bucket size
  `< 8`. (cite: predecessor §4.5 thresholds, carried over and applied per-bucket.)
- WARN once per non-empty `missing_in_val` / `missing_in_test`.

### 6.6 No resume compat shim

There is **no** migration of an old `val_source.json` to `split_source.json`.
A run resumed against a pre-PR run dir simply won't find a `split_source.json`
(the old file name is not read), so the resolver falls through to re-resolving
from the current `cfg.data.split`. Because the same PR breaks config compat
(`val_split` no longer validates), in-flight runs must be re-launched with a
migrated config anyway. State this explicitly; do **not** add backward-read code.

---

## 7. File-by-file change list

Every entry below was enumerated from a `val_split`/`val_source` grep across
`src/`, `tests/`, `configs/`.

### 7.1 Core source

| File | Change |
| --- | --- |
| `src/custom_sam_peft/config/schema.py` | Remove `ValSplitConfig` + `DataConfig.val_split`. Add `SplitConfig` + `DataConfig.split`. Replace `_check_val_modes`/`_check_hf_split_val_compat` with `_check_split_modes`/`_check_hf_split_compat` (§3). |
| `src/custom_sam_peft/data/splitter.py` | Generalize `stratified_split` to 3-arg joint k-subset; add `test_ids`, `missing_in_test`; `per_class_counts` → 3-tuple; `realized_fraction` → 2-tuple (§5). Update the docstring spec pointer to this file. |
| `src/custom_sam_peft/data/val_source.py` → `data/split_source.py` | **Rename file + all symbols** (§6). Extend `SplitSource`/persistence/`_log_split_source` for the test bucket. Generalize `resolve_split_source` dispatch. |
| `src/custom_sam_peft/train/runner.py` | Update imports (`split_source` symbols). `resolve_split_source` → `save_split_source` → `_log_split_source`. Thread `test_ids` is *not* injected into training datasets (test held out); only `{"train", "eval"}` resolved ids are injected as today (lines ~96-103). |
| `src/custom_sam_peft/train/trainer.py` | Read `split_source.json` (was `val_source.json`) for tracker hparams (lines ~663-672). Rename the injected `cfg_dict["val_source"]` key to `cfg_dict["split_source"]`; add `test_fraction_requested` / `n_test` to the injected dict. |
| `src/custom_sam_peft/cli/run_cmd.py` | Update imports + `load_split_source` calls (lines ~55, 95, 129, 215, 240). `_build_val_dataset(cfg, vs)` unchanged in behavior (val side only). The `vs.mode != "none"` val gates are unchanged. |
| `src/custom_sam_peft/eval/runner.py` | `--split test` consumer (lines ~117-126): accept `data.split.test` as a source (§8 below). `--split val` guard already references `cfg.data.val_split` (line ~112) → rename to `cfg.data.split` and check it carved a val bucket. The auto-split val branch (lines ~127-130) uses `resolve_split_source`. |
| `src/custom_sam_peft/diagnostics.py` | `DataReport` (lines ~43-55, ~241-263): the `val_split_fraction`/`val_split_seed` fields read from `cfg.data.split` instead of `cfg.data.val_split`; add a test-fraction field. Rename references. |

### 7.2 Eval consumption detail (`eval/runner.py`, §8 of brainstorm)

Test is held out from training and **never** auto-evaluated. It is scored only via
`csp eval --split test`, which resolves test ids from **either** explicit
`data.test` **or** `data.split.test`, mirroring the dual-source pattern
`--split val` already uses (runner.py lines ~125-131):

```python
# guard (was: line ~117) — accept either source
if split == "test" and cfg.data.test is None and (
    cfg.data.split is None or cfg.data.split.test is None
):
    raise ValueError(
        "--split test requires data.test or data.split.test in config; got neither."
    )

# dataset build (was: lines ~125-130)
if split == "test" and cfg.data.test is not None:
    cfg_dict["val"] = cfg_dict["test"]            # explicit test → existing path
elif split == "test" and cfg.data.split is not None and cfg.data.split.test is not None:
    vs = resolve_split_source(cfg, run_dir=None)  # recompute (no run_dir in this path)
    assert vs.test_ids is not None  # noqa: S101 — split.test invariant
    cfg_dict["_resolved_image_ids"] = {"eval": list(vs.test_ids)}
elif split == "val" and cfg.data.split is not None:
    vs = resolve_split_source(cfg, run_dir=None)
    assert vs.val_ids is not None  # noqa: S101
    cfg_dict["_resolved_image_ids"] = {"eval": list(vs.val_ids)}
```

**Reproducibility note** (carried from predecessor §7.4): standalone `csp eval
--split {val,test}` against a split-mode config recomputes the split from scratch
and reproduces the in-training partition iff the underlying annotations file is
unchanged. The split is `(items, val_fraction, test_fraction, seed)`-deterministic.

### 7.3 Example configs + emitters (migration — §8)

| File | Change |
| --- | --- |
| `configs/examples/coco_text_auto_split.yaml` | `val_split: {fraction: 0.1}` → `split: {val: 0.1, test: 0.1, seed: null}`; update header comments + spec pointer to this file. |
| `configs/examples/coco_text_no_val.yaml` | Update comments referencing `data.val_split` → `data.split`; spec pointer. |
| `src/custom_sam_peft/cli/setup_wizard.py` | `_validation_block` (lines ~140-172): `data.get("val_split")` → `data.get("split")`; emit `split:\n  val: …\n  test: …` block; update alt comments (lines ~149-171). |
| `src/custom_sam_peft/cli/init_cmd.py` | `validation_block` template (lines ~82-91): commented `val_split:` → `split:` with `val`/`test`. |
| `src/custom_sam_peft/cli/_interactive.py` | `_fraction` prompt (lines ~171-181): emit `{"data": {"split": {"val": …}}}` and drop the `<= 0.5` validator bound. `validate_config_with_eval_split` (lines ~285-303): `cfg.data.val_split` → `cfg.data.split`. |

### 7.4 Tests (migration + new — §10)

All test files referencing `val_split`/`val_source` are migrated. From the grep:
`tests/cli/test_finalize.py`, `tests/cli/test_flag_consistency.py`,
`tests/cli/test_host_ram_cli.py`, `tests/cli/test_run_single_eval.py`,
`tests/cli/test_time_limit_cli.py`, `tests/integration/test_cli_run.py`,
`tests/integration/test_eval_visualize_integration.py`,
`tests/integration/test_train_end_to_end.py`,
`tests/unit/cli/test_eval_interactive.py`, `tests/unit/cli/test_interactive.py`,
`tests/unit/cli/test_run_cmd_val_limit.py`, `tests/unit/cli/test_setup_wizard.py`,
`tests/unit/test_cli_init.py`, `tests/unit/test_config_schema.py`,
`tests/unit/test_data_coco.py`, `tests/unit/test_eval_batch_size_cap.py`,
`tests/unit/test_eval_runner.py`, `tests/unit/test_eval_runner_gate.py`,
`tests/unit/test_eval_runner_semantic.py`, `tests/unit/test_load_sam31_callsites.py`,
`tests/unit/test_splitter.py`, `tests/unit/test_train_runner.py`,
`tests/unit/test_train_runner_limit.py`, `tests/unit/test_trainer_no_val.py`,
`tests/unit/test_val_source.py` (→ rename to `tests/unit/test_split_source.py`).
Each: rename `val_split`→`split` / `fraction`→`val`, rename
`val_source`/`ValSource`/`resolve_val_source` symbols, and `val_source.json`
→ `split_source.json` in fixtures/assertions.

---

## 8. Migration (in-repo, same PR)

The breaking rename is contained to one PR. Concretely:

1. Schema swap (§3) — `val_split`→`split` (and `fraction`→`val`).
2. Module rename (§6) — `git mv data/val_source.py data/split_source.py`; rename
   all symbols; rename persisted artifact.
3. Splitter generalization (§5).
4. Consumer updates (§7.1) — trainer hparams key, run_cmd loads, eval runner
   guards, diagnostics.
5. Example YAMLs + 3 emitter modules (§7.3).
6. Test migration + new tests (§7.4, §10).

No deprecation path, no dual-read shim, no alias. A `grep -r val_split src/ tests/
configs/` returning zero hits is the migration-complete signal.

---

## 9. Edge cases (consolidated)

| Case | Behavior |
| --- | --- |
| **Test-only** (`split: {test: 0.2}`) | `mode="none"` (no val) + `test_ids` populated. Trains with no val (eval_every no-op, no early-stop, no end-of-run eval, no bundle samples) — identical to today's no-val. Test scored only via `csp eval --split test`. |
| **2-way fallback** (`split: {val: 0.1}`) | `test_ids == ()`/absent; bit-identical to prior 2-way behavior (§5.2 invariant). |
| **Tiny buckets** (`round(N·f)==0`) | Empty bucket; resolver WARNs (§6.5); no crash. |
| **Singleton class** (`n_c==1`) | Greedy single placement; never in any `missing_in_*` (the `>=2` guard). |
| **Rare class across 3 buckets** (`n_c==2`, 3 buckets) | Joint-stratified; lands in ≤2 buckets → recorded in the absent bucket's `missing_in_*`. |
| **val + test → 0 train** | Schema guard `val + test < 1.0` rejects at load (§3.2); the residual tiny-train case WARNs at resolve. |
| **Resume finds no old json** | A run resumed against a pre-PR run dir finds no `split_source.json` (old name not read) → re-resolves from `cfg.data.split` (§6.6). In-flight runs re-launched. |
| **`split` + explicit `val`/`test`** | Schema error from `_check_split_modes` (§3.3). |
| **`split` + `hf.split_val`** | Schema error from `_check_hf_split_compat` (§3.3). |

---

## 10. Testing plan

CPU-only; no GPU tests added (honors the GPU-test policy).

### 10.1 Splitter — `tests/unit/test_splitter.py`

| Test | Asserts |
| --- | --- |
| 3-way determinism | `stratified_split(items, 0.2, 0.2, 7)` twice → bit-identical `SplitResult` (incl. `test_ids`, `missing_in_test`). |
| Order independence | shuffle then split → same result. |
| 3-way disjointness | `set(train) & set(val) == set(train) & set(test) == set(val) & set(test) == set()`; union == all ids. |
| Realized fractions | 100 single-class items, `(0.2, 0.2)` → `realized_fraction` within `±0.02` of `(0.2, 0.2)`. |
| Per-class 3-bucket coverage | rare class `n_c==2` over 3 buckets, fixed seed → recorded in the correct `missing_in_*`; counts are a faithful 3-tuple summing to `n_c`. |
| **2-way fallback** | `stratified_split(items, v, 0.0, seed)` → `test_ids == ()`, `missing_in_test == ()`, `realized_fraction[1] == 0.0`, and train/val ids match the pinned pre-PR fixture (regression-locks the predecessor behavior). |
| Edge sizes | `N=0` → all-empty; `N=1` → all train; `N=3, (1/3, 1/3)` → 1+1+1. |

### 10.2 Schema — `tests/unit/test_config_schema.py`

1. `data.val_split` set → `ValidationError` (unknown key — proves the hard rename).
2. `split: {val: 0.1, test: 0.1}` validates.
3. `split: {}` (both omitted) → `ValidationError` (`_check_fractions`).
4. `split.val = 0.6, split.test = 0.5` (sum ≥ 1.0) → `ValidationError`.
5. `split.val = 0` / `1.0` / negative → `ValidationError` (exclusive bounds).
6. `split.val = 0.6` alone (was illegal under `le=0.5`) → **validates** (cap dropped).
7. `split` + `data.val` → `ValidationError` (`_check_split_modes`).
8. `split` + `data.test` → `ValidationError` (`_check_split_modes`).
9. `format=hf`, `split` set, `hf.split_val="custom"` → `ValidationError`.

### 10.3 Resolver/record — `tests/unit/test_split_source.py` (renamed)

1. Mode dispatch: `split.val` set → `mode="auto_split"`; `split.test`-only →
   `mode="none"` + `test_ids` populated; explicit `val` → `explicit`; neither →
   `none` (WARN once).
2. Round-trip: `save_split_source` → `load_split_source` returns structurally
   equal `SplitSource` (3-tuple `per_class_counts` int keys; `test_ids`;
   `missing_in_test`; `realized_fraction` 2-tuple).
3. Resume preference: pre-existing `split_source.json` → returned verbatim even
   when cfg fractions differ.
4. COCO enumeration shared-vocab: one annotations file → identical dense ids
   across the three returned buckets.
5. Seed inheritance: `split.seed=None, run.seed=7` → `seed_used == 7`.
6. WARN policy: small/deviating bucket and non-empty `missing_in_test` each emit
   a WARN (caplog).

### 10.4 Eval runner — `tests/unit/test_eval_runner.py`

1. `--split test`, `data.test is None` and `data.split.test is None` →
   `ValueError` containing `"--split test requires data.test or data.split.test"`.
2. `--split test` with `data.split.test` set, `val_dataset=None` → builder
   receives `_resolved_image_ids={"eval": test_ids}` (mock builder, assert cfg dict).
3. `--split test` with explicit `data.test` → existing `cfg_dict["val"] =
   cfg_dict["test"]` path still taken (regression).
4. `--split val` with `data.split` carving a val bucket → builder receives the
   val ids.

### 10.5 Migration of example configs — `tests/unit/test_cli_init.py`, `tests/unit/cli/test_setup_wizard.py`, `tests/unit/cli/test_interactive.py`

1. Emitted configs load cleanly via `load_config` and contain `split:` not
   `val_split:`.
2. `configs/examples/coco_text_auto_split.yaml` parses to a `DataConfig` with
   `data.split.val`/`data.split.test` set (load-and-assert, guards the YAML
   migration).
3. `_interactive` auto-split prompt yields `{"data": {"split": {"val": …}}}`.

### 10.6 Trainer / run_cmd / integration

- `tests/unit/test_trainer_no_val.py`: test-only split (`mode="none"`) still
  produces the no-val `metrics.json` note and skips eval/panels (rename of
  `val_source.json` → `split_source.json` in assertions).
- `tests/unit/test_train_runner.py`: end-to-end split on `tiny_coco` writes
  `split_source.json` with `test_ids`; resume reuses it (monkeypatch
  `stratified_split` to raise on resume → must not be called).
- `tests/integration/test_train_end_to_end.py` + `test_cli_run.py`: split-mode
  run finishes; `split_source.json` exists; test bucket is NOT auto-evaluated
  (assert eval/bundle ran on val only); a follow-up `csp eval --split test`
  resolves from `split_source.json`/recompute and scores the held-out test.

### 10.7 Coverage gate

Maintain ≥80% on `src/custom_sam_peft/data/` (bypass the global
`--cov-fail-under` on subsets per repo convention). The generalized splitter and
extended resolver are mostly straight-line with full unit coverage (§10.1, §10.3).

---

## 11. Out of scope / future work

| Item | Reason |
| --- | --- |
| HF `split_test` carving | `data.split` incompatible with HF explicit split knobs; HF test auto-split deferred. |
| k>3 / arbitrary named subsets | The splitter is now k-subset-capable internally, but only `{train, val, test}` is exposed. |
| `csp eval --run-dir` exact-run test reproduction | Standalone eval recomputes; loading a specific run's `split_source.json` is a future ergonomics enhancement (carried from predecessor). |
| Per-image stability under dataset edits | Same rationale as predecessor — run-dir record is the reproducibility hook. |
| `data.source` anchor field | Considered and rejected (§4). |

---

## 12. Implementation plan (numbered)

Seam for `superpowers:writing-plans`.

1. **Schema** (§3): remove `ValSplitConfig`/`val_split`; add `SplitConfig`/`split`;
   swap the two validators. Tests §10.2.
2. **Splitter** (§5): generalize `stratified_split` to 3-arg joint k-subset; add
   `test_ids`/`missing_in_test`; 3-tuple counts; 2-tuple realized. Tests §10.1
   (incl. the 2-way fallback regression-lock).
3. **Resolver rename + extend** (§6): `git mv` to `split_source.py`; rename all
   symbols + persisted artifact; extend `SplitSource`/persistence/log for test.
   Tests §10.3.
4. **Consumers** (§7.1): trainer hparams key, run_cmd loads, eval runner guards
   (§7.2), diagnostics. Tests §10.4, §10.6.
5. **Migration** (§8): example YAMLs + `setup_wizard`/`init_cmd`/`_interactive`.
   Tests §10.5.
6. **Test migration** (§7.4): rename symbols/artifacts across all listed test
   files; add the new 3-way/test-source cases.
7. **Integration** (§10.6) + lint/format/coverage pass; `grep -r val_split src/
   tests/ configs/` returns zero.
