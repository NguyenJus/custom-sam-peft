"""Internal-only config sub-classes — not user-set.

These classes handle internal implementation details that do not belong in the
user-facing YAML schema. They are constructed from validated user-facing config
values or from hardcoded defaults. Users cannot set these fields directly in
their YAML configuration files.

Per audit Section G (OQ2): dataclass by default; Pydantic only when enum fields,
constrained ints/floats, or ≥3 end-user-set fields are present.

Internal sub-configs retained here: MatcherWeights, WandbConfig, ExportConfig.
LossConfig has been promoted to a Pydantic model in config.schema as part of
the #112 schema break.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MatcherWeights:
    """Internal config — not user-set.

    Per-term cost weights for the Hungarian matcher.

    v0 defaults are mask-only (box terms = 0) because v0 trains text-only
    with no box supervision. lambda_l1 and lambda_giou are demoted internal
    constants (audit Section E: YAGNI demote — no config sets them; always 0.0).
    """

    lambda_l1: float = 0.0
    lambda_giou: float = 0.0
    lambda_mask: float = 5.0


@dataclass
class WandbConfig:
    """Internal config — not user-set.

    Weights & Biases tracking configuration. Rarely set by users; no
    validation constraints needed.
    """

    project: str = "custom_sam_peft"
    entity: str | None = None


@dataclass
class ExportConfig:
    """Internal config — not user-set.

    Export options (single boolean field). Promoted to a dataclass per
    audit Section G: no enum, no constraints, 1 field.
    """

    merge: bool = False
