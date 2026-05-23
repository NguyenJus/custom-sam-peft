"""Internal-only config sub-classes — not user-set.

These classes handle internal implementation details that do not belong in the
user-facing YAML schema. They are constructed from validated user-facing config
values or from hardcoded defaults. Users cannot set these fields directly in
their YAML configuration files.

Per audit Section G (OQ2): dataclass by default; Pydantic only when enum fields,
constrained ints/floats, or ≥3 end-user-set fields are present.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MatcherWeights:
    """Internal config — not user-set.

    Per-term cost weight for the Hungarian matcher.

    v0 defaults are mask-only; the box-cost coefficients were demoted to inline
    literal 0.0 at the construction site in losses.py (audit Section E,
    #92, YAGNI demote — no config sets them).
    """

    lambda_mask: float = 5.0


@dataclass
class LossConfig:
    """Internal config — not user-set.

    Loss-mix weights for SAM 3.1 training.

    Most fields are demoted internal constants per audit Section E. Only
    w_mask, w_obj, w_presence, and matcher_weights survive as advanced
    settings read by the training loop, but none are exposed to the YAML
    user schema — they are hardcoded defaults here. The focal-loss
    hyperparameters were demoted to module-level constants in
    models/losses.py (#93).

    No `w_cls`: SAM 3.1's multiplex forward provides open-vocabulary
    discrimination directly via per-text-embedding queries; per-class
    `w_cls` is unneeded. `w_presence` weights the image-level
    "any-instance-of-this-class-present?" supervision.
    """

    w_mask: float = 1.0
    w_obj: float = 1.0
    w_presence: float = 1.0
    # w_box is demoted: always 0.0 in all examples; no box supervision in v0.
    w_box: float = 0.0
    matcher_weights: MatcherWeights = field(default_factory=MatcherWeights)


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
