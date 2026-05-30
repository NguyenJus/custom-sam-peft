# Defaults Provenance

This document is the source of truth for the provenance of every trust-bearing
default hyperparameter in `custom-sam-peft`. Inline `# cite:` / `# tbd:` tags in
the code are deliberately terse pointers into the rows below.

Umbrella `# tbd:` tracker: #191
(Every `# tbd: #191` tag and row points there.)

## Verification Standard

Every literature-backed value is verified against its *primary* source with a
captured quote + URL/DOI + exact equation/table/figure. Framework defaults link
the upstream docs and pin the observed version. Degenerate cases state the math
identity. Reference-implementation values cite the file/line they mirror.
Project numbers with no external source and no internal run are tagged
`# tbd: #191` — never fabricated.

Row schema (every section uses these six columns):

| Location | Value | Tag | Full reference | Verifying quote | Notes |
| --- | --- | --- | --- | --- | --- |

- **Location** — `file:symbol`.
- **Value** — the literal default.
- **Tag** — the inline tag class applied (mirrors the code), or `index-only` for
  untagged self-evident structural/string defaults.
- **Full reference** — authors, year, arXiv/DOI, exact Eq./Table/Fig.; or the
  upstream-doc URL + pinned version (framework defaults); or repo file/line
  (reference-impl).
- **Verifying quote** — short quote from the primary source establishing the
  value.
- **Notes** — caveats, degenerate-case identities, calibration run pointers,
  cross-links.

## config/_internal.py

| Location | Value | Tag | Full reference | Verifying quote | Notes |
| --- | --- | --- | --- | --- | --- |

## config/schema.py

| Location | Value | Tag | Full reference | Verifying quote | Notes |
| --- | --- | --- | --- | --- | --- |

## data/aug_presets.py

Legend letters used in the `aug_presets.py` module docstring resolve here.

| Location | Value | Tag | Full reference | Verifying quote | Notes |
| --- | --- | --- | --- | --- | --- |

## data/channel_semantics.py

| Location | Value | Tag | Full reference | Verifying quote | Notes |
| --- | --- | --- | --- | --- | --- |

## data/transforms.py

| Location | Value | Tag | Full reference | Verifying quote | Notes |
| --- | --- | --- | --- | --- | --- |

## presets.py

| Location | Value | Tag | Full reference | Verifying quote | Notes |
| --- | --- | --- | --- | --- | --- |

## cli/templates/config_full.yaml

Template-echoed literals; the authoritative provenance is the schema row for the
same symbol. This section cross-links the template slot to its schema row.

| Location | Value | Tag | Full reference | Verifying quote | Notes |
| --- | --- | --- | --- | --- | --- |

## models/losses/presets.py

### Citation legend (folded in from the module docstring)

| Letter | Source | Establishes |
| --- | --- | --- |

### Preset-table parameters

| Location | Value | Tag | Full reference | Verifying quote | Notes |
| --- | --- | --- | --- | --- | --- |

## Reference Training Profile

<!-- Owned by Deliverable 2 (epochs alignment). Populated in Phase 2. -->
