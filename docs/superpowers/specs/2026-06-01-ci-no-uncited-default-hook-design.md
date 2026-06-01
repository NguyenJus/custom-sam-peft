# Design Spec: CI No-Uncited-Default Hook + Inline-Tag Strip (#192)

- **Issue:** [#192](https://github.com/NguyenJus/custom-sam-peft/issues/192) — "CI:
  enforce no-uncited-default hook"
- **Parent:** [#120](https://github.com/NguyenJus/custom-sam-peft/issues/120) —
  the whole-codebase defaults audit that produced `docs/defaults-provenance.md`
  and explicitly deferred this CI hook to its own follow-up.
- **Date:** 2026-06-01
- **Status:** Locked design, ready for planning.

## Summary / Goal

Deliver, in a single PR, BOTH halves of the locked design:

1. A **completeness check** that fails when a trust-bearing default ships in an
   in-scope file without provenance in `docs/defaults-provenance.md`. The check
   is a pytest test (no new CI YAML), runs in the existing `test` job, and is
   locally runnable.
2. A **curated inline-tag strip** that makes `docs/defaults-provenance.md` the
   lone home for provenance — except for a small, explicitly enumerated set of
   "head-turner" defaults that retain a one-line inline note.

This is **document-and-tooling-only**. No default *value* changes (same
constraint as #120). The end state of the repo must be **green** under the new
check after the strip and doc migration are applied.

## Background & motivation

#120 audited every trust-bearing default value in the codebase and gave each a
provenance entry in `docs/defaults-provenance.md`, with terse inline
`# cite:` / `# tbd:` tags pointing into it. #192 (this work) asks for a guard so
that a *new* default added to an in-scope file cannot silently ship without
provenance.

During brainstorming the scope was deliberately widened beyond a bare guard to
also **de-clutter the inline tags**. The reason: the provenance *document*, not
the inline tags, is what gives a reader confidence that defaults are
research-grounded. Once a CI check guarantees doc completeness, the inline tags
stop being the primary code↔doc pointer and become redundant noise in most
files. So the same PR strips the now-redundant inline tags down to doc-only,
keeping inline notes only where a value genuinely turns a knowledgeable reader's
head (regression-bait).

## Principle

`docs/defaults-provenance.md` is the **complete registry and the single source
of truth** for default provenance. Inline tags are no longer the primary
code↔doc pointer. A small curated set of head-turner defaults retains an inline
note purely as a reader's "wait, that's intentional" guard, not as the canonical
provenance pointer.

## Scope

### Scope is auto-derived from the doc

The set of in-scope files is **derived from the `## <path>` section headers** in
`docs/defaults-provenance.md`. "Add a defaults file to the registry" therefore
means "add a `## <path>` section to the doc," and the check self-extends to
cover it. The checker enumerates the section headers, classifies each, and runs
the matching assertion. There is no second hard-coded list of paths anywhere.

### Section classification

Every `## <header>` in the doc is classified into exactly one of four classes:

| Class | Sections (current) | Check applied |
| --- | --- | --- |
| `prose` | `config/_internal.py`, `config/schema.py`, `data/channel_semantics.py`, `data/transforms.py`, `presets.py`, `predict/budget.py`, `tests/gpu/test_qlora_8gb_ceiling.py` | Symbol⇄row bijection (Assertion 1) |
| `table` | `data/aug_presets.py`, `models/losses/presets.py` | Tag-presence + legend resolution (Assertion 2) |
| `yaml` | `cli/templates/config_full.yaml` | Cross-link coverage (Assertion 3) |
| `prose-narrative` | `Verification Standard`, `Reference Training Profile` | **None** — not a check target |

The classifier must distinguish a file-section header (`## config/schema.py`)
from a prose-narrative header (`## Verification Standard`). The implementer picks
the discriminator and states it; the recommended discriminator is: a header is a
**file section** iff its text resolves to an existing file on disk (see
base-path resolution below); otherwise it is `prose-narrative`. `table` vs
`prose` vs `yaml` is then refined by the resolved file's path/extension:

- `*.yaml` → `yaml`.
- The two preset-table modules `data/aug_presets.py` and
  `models/losses/presets.py` → `table`. These two are identified by an explicit
  table-module allow-set in the checker (they are the only modules whose doc
  section is a legend + noteworthy-rows form rather than a row-per-symbol form).
- Every other resolved-file section → `prose`.

### Base-path resolution (edge case — call out for the implementer)

Section header text is **not uniformly rooted**. The checker must resolve each
header to a real file using these rules, in order:

1. Headers beginning `cli/templates/` resolve under
   `src/custom_sam_peft/cli/templates/` (e.g.
   `## cli/templates/config_full.yaml` →
   `src/custom_sam_peft/cli/templates/config_full.yaml`).
2. Headers beginning `tests/` resolve under the repo-root `tests/` directory
   (e.g. `## tests/gpu/test_qlora_8gb_ceiling.py` →
   `tests/gpu/test_qlora_8gb_ceiling.py` from repo root).
3. All other file-section headers resolve under `src/custom_sam_peft/` (e.g.
   `## config/schema.py` → `src/custom_sam_peft/config/schema.py`).

A header that matches none of the prose-narrative set and resolves to **no file
on disk** is a hard FAIL ("doc section names a path that does not exist").

### File-style split (central design decision — rationale)

The split below is deliberate and the spec records *why* each file is treated as
it is, so a future contributor does not "tidy up" the table modules into the
prose form (which would defeat the goal).

#### Prose / constant files → strip to doc-only

`config/schema.py`, `config/_internal.py`, `data/channel_semantics.py`,
`data/transforms.py`, `presets.py` (repo path
`src/custom_sam_peft/presets.py`), `predict/budget.py`.

The doc already carries a **per-symbol row for every default** in these files.
The inline tag is therefore fully redundant with its doc row. **Strip** the
inline `# cite:` / `# tbd:` tag from each default in these files down to
doc-only — the doc row stays, so nothing is lost. **Exception:** the curated
keep-list below retains its inline note.

`tests/gpu/test_qlora_8gb_ceiling.py` is *also* a prose section for the
completeness check (its `QLORA_8GB_CEIL_GB` constant has a doc row), but inline
rationale in a test file is normal practice and is **left as-is** — it is out of
the strip but in-scope for Assertion 1.

#### Dense table modules → keep the legend-letter system entirely

`data/aug_presets.py`, `models/losses/presets.py`.

The inline legend letters (`# (a)…(d)` in `aug_presets.py`; `# cite: (A)…(H)`
in `losses/presets.py`) are a **compact provenance index**, not clutter. The doc
does **not** carry a row per cell for these modules — it carries a *legend*
(letter → source) plus rows only for noteworthy / `tbd` cells. Stripping the
letters would force the doc into a verbose full per-cell table (12 cells × ~12
knobs), which defeats the de-clutter goal. So **keep the table legend-letter
system entirely — no strip in these modules.** Their doc sections keep the
two-tier legend-plus-noteworthy-rows structure.

#### YAML template → cross-link rows, keep intentional-divergence notes only

`cli/templates/config_full.yaml` (repo path under
`src/custom_sam_peft/cli/templates/`).

The template echoes schema defaults; its doc section is `cross-link` rows
pointing at the schema row for the same symbol. Leave a one-line inline note
**only** on values that intentionally **differ** from the schema default (see
keep-list). All other template scalars carry no inline tag (they already carry
none today and stay that way).

## The CI check — three assertions

The checker runs three assertions, dispatched by section class. Every failure
message must name the offending `file:symbol` (or `file:line` for table cells),
state what is missing, and give a one-line remediation.

### The "default surface" (definition for Assertion 1)

For `prose` sections, the **enforced default surface** is exactly:

- pydantic `Field(default=...)` and `Field(default_factory=...)` defaults,
- dataclass field defaults (`name: T = <default>`),
- module-level constant assignments (`NAME = <value>` and `NAME: T = <value>` at
  module top level).

The surface **deliberately excludes arbitrary in-function magic literals.**
Concretely, the `presets.py` rows that document in-function dtype/byte math —
`presets.py:_bytes_per_param_for_method (2.0)`,
`presets.py:_bytes_per_param_for_method (0.5)`,
`presets.py:_optimizer_bytes (*4 literal)` — are values that live **inside
function bodies**, not at module scope. The doc keeps documenting them (their
rows stay and are exempt from the doc→code direction; see below), but they are
**NOT** in the auto-enforced code→doc surface. Stating this exclusion
explicitly prevents the checker from demanding the implementer hoist
in-function literals to module constants just to satisfy the guard.

### Assertion 1 — prose/constant files: symbol⇄row bijection (both directions)

For every `prose` section:

- **code→doc.** Every default symbol present in the file's default surface
  (above) must have a matching doc row in that file's section, keyed by
  `file:symbol`. A surface symbol with no row → **FAIL** ("new undocumented
  default: `<file>:<symbol>` — add a row to the `## <section>` section of
  `docs/defaults-provenance.md`").
- **doc→code.** Every doc row in that section whose `Location` names a **surface
  symbol** must still resolve to an existing symbol in the file. A row pointing
  at a symbol that no longer exists → **FAIL** ("stale/orphaned provenance row:
  `<file>:<symbol>` — remove or update the row in `docs/defaults-provenance.md`").

  The doc→code direction applies only to rows whose `Location` denotes a surface
  symbol. Rows whose `Location` denotes an **in-function literal** (the three
  `presets.py` `(... literal)` rows above) are recognized by their parenthetical
  suffix and are **skipped** in the doc→code direction (they are intentionally
  out of the enforced surface). The implementer must detect these rows
  syntactically (a `Location` of the form `file:symbol (<literal-note>)`) and
  exempt them.

### Assertion 2 — table modules: tag-presence + legend resolution

For each `table` section's module, scope the check to the **`PRESET_TABLE` dict
literal** in that module (the cell-value lines between `PRESET_TABLE = {` and its
closing `}`; the module-level alias-assignment lines such as
`PRESET_TABLE[("microscopy","severe")] = dict(...)  # cite: (G)` in
`losses/presets.py` are also cell lines and are included). In `losses/presets.py`
the scope **also includes the `_LEGACY_DEFAULTS` base dict** (the cell-value
lines between `_LEGACY_DEFAULTS: dict[str, Any] = {` and its closing `}`): these
are the trust-bearing `preset == "none"` base values, so every one of them is a
preset-table value line for tag-presence purposes and must carry a tag like any
`PRESET_TABLE` cell. Then:

- **Tag presence.** Every value line inside the preset table must carry a
  recognized inline tag — a **legend letter** (`# (a)` bare form, or
  `# cite: (A)` / `# cite: (A,E)` parenthesized form), a `# cite: …` non-legend
  tag (e.g. `# cite: empirical`), or a `# tbd: …` tag. A value line with no
  recognized tag → **FAIL** ("untagged preset cell: `<file>:<line>` — add a
  legend letter, `# cite:`, or `# tbd:` tag").
- **Legend resolution.** Every legend letter actually used in the module's cells
  must be **defined in that module's doc legend** (the `### Legend` /
  `### Citation legend` sub-table under the module's `## <path>` section). A used
  letter with no legend definition → **FAIL** ("undefined legend letter
  `<letter>` used at `<file>:<line>` — define it in the legend under
  `## <section>` in `docs/defaults-provenance.md`").

The two table modules use **different legend-letter syntaxes** by current
convention (`aug_presets.py` uses bare `# (a)`; `losses/presets.py` uses
`# cite: (A)`). The tag recognizer must accept **both** forms. The set of legend
letters is per-module and case-sensitive (`aug_presets.py` lowercase `a–e` after
the new `(e)` row is added; `losses/presets.py` uppercase `A–H`); the checker
matches a module's used letters against that same module's doc legend only.

> **Bring-green note (resolved decision).** Several preset cells in **both**
> table modules carry **no** inline tag today, and Assertion 2 requires every
> value line to carry one. The governing rule is **T1: every preset-table cell
> carries a tag.** An "off" cell does **not** need a deep citation — a plain
> **justification** (why the value is deliberately off) suffices. But the bare
> off-cells must be tagged **honestly**, classified by *why* the value is off,
> rather than blanket-reusing `(a)` / `(d)` for everything. No cell *value*
> changes; this is a mechanical, value-preserving tagging edit the planner may
> phase with the Assertion-2 implementation.
>
> **`aug_presets.py` untagged off-cells** — the bare `False` booleans and `0.0`
> magnitudes (e.g. `"vflip": False`, `"rotate90": False`, `"blur": 0.0`,
> `"gauss_noise": 0.0`, `"rotate_arbitrary": 0.0`, `"stain_jitter": 0.0`,
> `"color_jitter": 0.0` in the gentler presets) are tagged by this four-way
> classification:
>
> 1. **Off booleans reflecting domain symmetry** (e.g. `"vflip": False` for
>    natural — no vertical symmetry; `"rotate90": False` for natural): existing
>    legend `(a)` "domain convention" already covers the on **and** off states of
>    the symmetry booleans. Tag these `# (a)`.
> 2. **Off magnitudes that are domain-not-applicable** (e.g. `"stain_jitter": 0.0`
>    outside the H&E domains — `LOCKED_OFF["satellite"]["stain_jitter"]` records
>    "stain_jitter is H&E-specific … satellite imagery is not H&E"; `"color_jitter":
>    0.0` for the medical/microscopy lock-offs already carrying `# (d)`): tag with
>    the existing **domain-justification** letter that fits — `# (d)` for the
>    clinical/diagnostic lock-offs already using it, and `# (a)` domain convention
>    for the H&E-not-applicable `stain_jitter` zeros whose rationale is a
>    domain fact rather than a clinical-laterality lock. These are justifications,
>    not citations.
> 3. **Off magnitudes simply below this preset's intensity tier** (e.g. `"blur":
>    0.0`, `"gauss_noise": 0.0`, `"rotate_arbitrary": 0.0` in the safe/medium
>    presets where the gentler tier just omits that augmentation): these are **not**
>    a domain convention — they are a recipe choice about intensity. **Add one new
>    aug legend letter** — `(e)`: "augmentation omitted at this preset's intensity
>    tier — recipe choice; no citation." Tag these cells `# (e)`. The doc migration
>    adds an `(e)` row to the `### Legend` table in the `## data/aug_presets.py`
>    section.
>
> **`losses/presets.py` untagged off-cells (undercount fix).** This module also
> has bare cells the earlier framing missed: every `boundary_weight: 0.0` across
> the presets, and the cells in the `_LEGACY_DEFAULTS` base dict (the
> `mask_family` / `box_family` / `obj_family` / `presence_family` / `w_mask` /
> `w_box` / `w_obj` / `w_presence` / `focal_gamma` / `focal_alpha` /
> `tversky_alpha` / `tversky_gamma` / `boundary_weight` lines used when
> `preset == "none"`). These are trust-bearing base values and must be tagged —
> they must **not** be left bare and must **not** be tagged `# cite: (H)` (the
> boundary-loss-ON citation, which only applies to `boundary_weight: 0.2`).
>
> 4. **`losses/presets.py` off-cells** — boundary loss is **off by default /
>    opt-in**: `boundary_weight: 0.0` is exactly the pre-#112 hardcoded behavior
>    (there was no boundary term before #112), so it folds cleanly into existing
>    legend `(B)` "preserved pre-#112 hardcoded behavior." Tag every
>    `boundary_weight: 0.0` cell `# cite: (B)`. The `_LEGACY_DEFAULTS` base-dict
>    cells are likewise the preserved pre-#112 defaults and tag `# cite: (B)`
>    (they duplicate the per-preset `(B)`-tagged family/weight values — that they
>    repeat already-tagged cells does not exempt them; as base values they are
>    trust-bearing and must each carry the tag). The `_LEGACY_DEFAULTS`
>    `focal_gamma: 2.0` / `focal_alpha: 0.25` carry `# cite: (A,C)` and the two
>    Dice-degenerate `tversky_alpha: 0.5` / `tversky_gamma: 1.0` carry `# cite:
>    (F)`, matching their per-preset siblings. `(B)` is the chosen
>    off-by-default / opt-in justification; no parallel new letter is needed since
>    `(B)` fits.
>
> Every chosen letter must be a defined legend letter (so legend resolution still
> passes): `(a)`, `(d)`, the new `(e)` for `aug_presets.py`; `(A)`, `(B)`, `(C)`,
> `(F)` for `losses/presets.py`. The existing `(d)` clinical-lockoff meaning
> stays distinct from `(e)` intensity-omission, and nothing is mislabeled.

### Assertion 3 — YAML template: cross-link coverage

For the `cli/templates/config_full.yaml` section: every shipped scalar in the
template that **echoes a schema default** (i.e. corresponds to a field that has a
schema default, regardless of whether the template value matches or intentionally
differs) must have a `cross-link` row in the template's doc section, keyed by
`config_full.yaml:<dotted.path>`. A template scalar that echoes a schema default
but has no `cross-link` row → **FAIL** ("template scalar
`config_full.yaml:<path>` has no cross-link row — add a `cross-link` row to the
`## cli/templates/config_full.yaml` section").

This is the **lightest** of the three assertions. If implementation cost is
high, the planner MAY stage Assertion 3 into a later phase, but it is **in
scope** and must be delivered and green by PR end. Resolving which template
scalars "echo a schema default" is by dotted-path correspondence to a
`config/schema.py` field that has a default; template-only structural keys with
no schema-default counterpart are not required to have a cross-link row.

### Failure-output contract (all three assertions)

Every failure message MUST:

1. name the offending `file:symbol` (Assertions 1, 3) or `file:line`
   (Assertion 2),
2. say what is missing (undocumented default / orphaned row / untagged cell /
   undefined legend letter / missing cross-link), and
3. give the one-line remediation (add a row to the right section / tag the cell /
   add a cross-link row).

When multiple violations exist, the check reports **all** of them (not just the
first) so a contributor fixes them in one pass.

## Enforcement form

- **Mechanism: a pytest test.** A test module — `tests/test_defaults_provenance.py`
  — drives the three assertions over the **real, current repo** and asserts the
  repo is clean. It runs automatically inside the existing `test` CI job
  (`uv run pytest`) and is locally runnable with no new CI YAML and no
  pre-commit framework.
- **Checker logic lives in a shared, importable, unit-testable module.** Put the
  parsing + assertion logic in `src/custom_sam_peft/_provenance_check.py`
  (importable as `custom_sam_peft._provenance_check`).

  **Rationale for `src/…` over `scripts/`:** the unit tests (below) need to drive
  the checker over *synthetic fixtures*, which requires importing the checker
  functions with stable import semantics. `scripts/` is not an importable
  package here, and the test suite already imports internal modules via
  `from custom_sam_peft.<...> import <...>` (e.g.
  `tests/unit/test_channel_semantics.py`). Placing the checker in the package
  matches that convention and keeps `coverage` accounting it like the rest of
  `src/`. The module is leading-underscore (`_provenance_check`) to mark it
  internal / not part of the public API.

  The checker module must expose its core as **pure functions** that take an
  explicit repo-root (or explicit doc-text + file-set) argument so the unit
  tests can point it at a temporary fixture tree rather than the live repo.

- **Standalone script wrapper is OPTIONAL.** A thin
  `scripts/check_defaults_provenance.py` (or similar) that imports the package
  module and prints failures for local / pre-commit convenience MAY be added,
  but is **not required** and **no pre-commit framework hook is mandated**.

## Doc migration (edits to `docs/defaults-provenance.md`)

These are the only edits to the provenance doc, and they are in scope:

1. **Reword the preamble.** State that the doc is the **home** for provenance;
   that inline `# cite:` / `# tbd:` tags are **no longer the primary code↔doc
   pointer**; and that a small curated set of head-turner defaults retains an
   inline note as a reader guard. The current preamble sentence ("Inline
   `# cite:` / `# tbd:` tags in the code are deliberately terse pointers into the
   rows below.") must be replaced to reflect the new principle.
2. **Reinterpret the `Tag` column.** The `Tag` column is redefined as the
   **provenance class** — one of `cite`, `tbd`, `index-only`, `cross-link` — and
   **no longer "mirrors the inline tag."** Update the row-schema legend bullet
   for `Tag` (currently: "the inline tag class applied (mirrors the code)…")
   accordingly. The existing per-row `Tag` cell *values* already express these
   classes and need not be rewritten cell-by-cell; only the column's
   **definition** changes.
3. **Add the `(e)` aug legend row.** The `## data/aug_presets.py` `### Legend`
   table gains one new row — `(e)`: "augmentation omitted at this preset's
   intensity tier — recipe choice; no citation" — so the intensity-omission
   off-magnitudes (`blur` / `gauss_noise` / `rotate_arbitrary` zeros in the
   gentler presets) resolve. This is **additive**; no existing legend row is
   reworded or deleted. The `(d)` clinical-lockoff row keeps its current meaning,
   distinct from the new `(e)`.
4. **Losses legend: no new letters needed.** The `losses/presets.py` off-cells
   (`boundary_weight: 0.0` and the `_LEGACY_DEFAULTS` base-dict cells) fold into
   the existing legend (`(B)` "preserved pre-#112 hardcoded behavior" for the
   off-by-default / opt-in cells, with `(A,C)` and `(F)` for the base-dict
   focal/Tversky cells per their per-preset siblings). The `### Citation legend`
   table is unchanged — no row added, none deleted.
5. **Do NOT delete any existing rows.** Every current row stays. The only row
   *addition* anywhere in the doc is the single `(e)` aug-legend row above; the
   newly tagged off-cells otherwise reuse existing legend letters and the
   existing legend + noteworthy-row structure, so they introduce no further new
   rows.

## Curated keep-list (reviewer's veto checklist — embed verbatim)

These are defaults in the **strip-category files** whose inline note **survives
the strip** because the value turns a knowledgeable reader's head / is
regression-bait. The reviewer treats this list as a veto checklist: the strip is
correct iff exactly these inline notes remain (plus the always-exempt
test-file and table-module cases). The previously-open items are resolved below
(user-confirmed at spec review); the planner/implementer treats each as final.

### Confident keep (inline note retained)

- **`config/schema.py: NormalizeConfig.mean` and `NormalizeConfig.std =
  [0.5, 0.5, 0.5]`** AND its source **`data/transforms.py:
  KNOWN_PROCESSOR_STATS["facebook/sam3.1"]`** — everyone expects ImageNet
  `(0.485, 0.456, 0.406)`; a prior audit already mis-"fixed" this to ImageNet.
  Prime regression-bait. Keep the inline note on all three.
- **`config/_internal.py: MatcherWeights.lambda_l1` and `lambda_giou = 0.0`** —
  read like disabled / broken box terms. Keep the inline note.
- **`config/schema.py: TrainHyperparams.lr_schedule = "plateau"`** — flipped from
  the expected `cosine` in #197; invites a "restore cosine" regression. Keep the
  inline note.
- **`data/transforms.py: _HED_FROM_RGB_MATRIX`** — the variable name reflects the
  INVERSE's math role, not the literal matrix; a genuine "wait, what?" without
  the note. Keep the inline note.

### Resolved keep/strip decisions (user-confirmed at spec review)

The five previously-open items are now **resolved**. The implementer treats each
decision below as final.

- **`config/schema.py: TextPromptConfig.negatives_per_image = 0`** → **STRIP** to
  doc-only. The schema-`0`-vs-template-`4` mismatch stays documented in the doc
  row and the `config_full.yaml` `cross-link` row; no inline note is retained on
  the schema field.
- **`config/schema.py: TextPromptConfig.k = 16` and
  `MultiplexConfig.classes_per_forward = 16`** → **KEEP** the inline note. Coupling
  invariant: both must equal `models/sam3.py:MULTIPLEX_CAP`; bumping above 16
  silently breaks. Both symbols keep the note (same invariant).
- **`config/schema.py: EvalConfig.mask_threshold = 0.0`** → **KEEP** the inline
  note. It is a LOGIT threshold; `0.0` reads as "off" to a probability-thinker.
- **`config/schema.py: PEFTConfig.scope = "vision_decoder"`** → **STRIP** to
  doc-only. The `focal_gamma 2.5 / 3.0` and `tversky_alpha 0.8` items raised
  alongside this are `models/losses/presets.py` **TABLE** cells where keep/strip
  is moot — their `# tbd: #191` tags stay per Assertion 2; no action.
- **`cli/templates/config_full.yaml`: the intentional divergences
  `text_prompt.negatives_per_image: 4` and
  `text_prompt.mode: present_plus_negatives`** → **KEEP** a one-line inline note on
  each (they intentionally differ from schema defaults `0` / `present`). This is
  consistent with stripping the schema-side `negatives_per_image = 0` note above:
  the divergence guard lives at the template, where a user edits, not on the
  schema field.

### Everything else → strip to doc-only

Framework-default values, canonical COCO / LoRA values, `index-only` structural
strings, and bare `# tbd:` numbers with no prior-violation → **STRIP** the inline
tag to doc-only. The doc row remains the provenance.

## Testing

### Unit tests (checker over synthetic fixtures)

In a unit test module (e.g. `tests/unit/test_provenance_check.py`), drive the
checker functions over small synthetic fixture trees (a temp dir with a tiny
`schema.py`-like file + a tiny provenance-doc string), asserting each of the
following independently:

1. A synthetic surface default **with** a matching doc row → **passes**.
2. A synthetic surface default **without** a doc row → **fails** (code→doc), and
   the failure message names the symbol and the "add a row" remediation.
3. An **orphaned doc row** (a row whose `Location` names a surface symbol that
   does not exist in the file) → **fails** (doc→code), message names the symbol
   and the "remove/update" remediation.
4. An **untagged table cell** (a `PRESET_TABLE` value line with no recognized
   tag) → **fails**, message names `file:line`.
5. An **undefined legend letter** (a cell using a letter not defined in the
   module's doc legend) → **fails**, message names the letter and `file:line`.
6. (Coverage of the in-function exclusion) A synthetic in-function literal that
   is **not** documented → **passes** (it is outside the enforced surface); a
   doc row of the `symbol (<literal>)` form whose base symbol is absent →
   **passes** in the doc→code direction (the row is exempt).

The unit tests must NOT depend on the live repo state — they construct their own
fixtures so they stay green regardless of future real-repo edits.

### Live check (the enforcement test over the real repo)

`tests/test_defaults_provenance.py` runs all three assertions over the **real,
current repo** and asserts zero violations. After the strip + bare-cell tagging +
doc migration are applied, this test must be **green**. This is the
acceptance gate: the post-strip repo passes its own completeness check.

### Existing-suite hygiene

- The new tests run under `uv run pytest` (the `test` job) with no new markers or
  CI YAML.
- The strip edits must not break `ruff check`, `ruff format --check`, or
  `mypy src/custom_sam_peft` (the `test` job's other steps).
- The doc edits must pass `markdownlint-cli2` under
  `.config/markdownlint-cli2.jsonc` (the `lint-hygiene` job lints `**/*.md`).

## Acceptance criteria

1. A checker module `src/custom_sam_peft/_provenance_check.py` exists, exposes
   pure functions taking an explicit repo-root / doc-text + file-set, and
   implements all three assertions with the failure-output contract.
2. `tests/test_defaults_provenance.py` runs the three assertions over the real
   repo and is **green** after the strip + bare-cell tagging + doc migration.
   Bare-cell tagging satisfies **T1** with **honest justifications**: every
   preset-table cell in both table modules carries a tag, with off-cells tagged by
   *why* they are off — off symmetry booleans `# (a)`; domain-not-applicable
   off-magnitudes the fitting existing domain letter (`# (d)` clinical lock-offs,
   `# (a)` H&E-not-applicable); intensity-tier-omission off-magnitudes the new
   `# (e)`; `losses/presets.py` off-by-default cells (`boundary_weight: 0.0`,
   `_LEGACY_DEFAULTS` base dict) `# cite: (B)` (never `(H)`). An off-cell's tag
   may be a citation-free justification; `(d)` stays reserved for clinical
   lock-offs and is not reused for intensity omission.
3. Unit tests over synthetic fixtures cover: documented-default pass,
   undocumented-default fail, orphaned-row fail, untagged-table-cell fail,
   undefined-legend-letter fail, and the in-function-literal exclusion.
4. Scope is auto-derived from the doc's `## <path>` headers; adding a doc section
   extends coverage with no second path list. Base-path resolution
   (`cli/templates/…`, `tests/…`, else `src/custom_sam_peft/…`) is implemented;
   a header resolving to no file is a hard fail.
5. `prose-narrative` sections (`Verification Standard`, `Reference Training
   Profile`) are excluded from all assertions.
6. The default surface excludes in-function magic literals; the three
   `presets.py` `(... literal)` doc rows are exempt in the doc→code direction.
7. The curated inline-tag strip is applied: confident-keep notes retained; the
   resolved keep/strip decisions applied (KEEP `k=16`/`classes_per_forward=16`,
   `mask_threshold=0.0`, and the `config_full.yaml` divergence notes; STRIP
   `negatives_per_image=0` and `PEFTConfig.scope`);
   everything else in the strip-category prose files stripped to doc-only; table
   modules' legend-letter system kept entirely (no strip — only the additive
   bare-off-cell tagging of item 2, including the new `(e)` aug legend letter and
   the `(B)` losses off-cell tags); test-file rationale left as-is.
8. The doc preamble is reworded to the new principle and the `Tag` column is
   redefined as the provenance class; no existing rows deleted.
9. Failure messages are actionable (name the offending `file:symbol` /
   `file:line`, what's missing, one-line remediation) and report all violations
   in one pass.
10. Out-of-scope items (below) are untouched. No default value changes.
11. Lint gates pass: `ruff check`, `ruff format --check`,
    `mypy src/custom_sam_peft`, and `markdownlint-cli2`.

## Out of scope (explicit)

- **Re-tuning / changing any default value.** Document-and-tooling-only — same
  constraint as #120.
- **Resolving the `#191` / `#193` / `#197` / `#142` `# tbd:` items.** The check
  treats a `# tbd:` tag as valid provenance; it does not demand the underlying
  number be nailed down.
- **Restructuring the table-module doc sections into full per-cell rows.** The
  legend-plus-noteworthy-rows form is deliberate and kept.
- **A mandated pre-commit framework hook.** The pytest test is the enforcement
  mechanism; a standalone script wrapper is optional.
- **Adding new CI YAML / jobs.** The check rides the existing `test` job.

## Open questions

None. All keep/strip items are user-confirmed at spec review (see "Resolved
keep/strip decisions"). The
bring-green tagging of bare off-cells in **both** table modules
(`aug_presets.py` and `losses/presets.py`) is resolved in the Assertion-2 note's
four-way honest-justification classification. The checker location
(`src/custom_sam_peft/_provenance_check.py`) is resolved with rationale in the
Enforcement section.
