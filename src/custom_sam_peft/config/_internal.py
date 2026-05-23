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

    Per-term cost weights for the Hungarian matcher.

    v0 defaults are mask-only (box terms = 0) because v0 trains text-only
    with no box supervision. lambda_l1 and lambda_giou are demoted internal
    constants (audit Section E: YAGNI demote — no config sets them; always 0.0).
    """

    lambda_l1: float = 0.0
    lambda_giou: float = 0.0
    lambda_mask: float = 5.0


@dataclass
class LossConfig:
    """Internal config — not user-set.

    Loss-mix weights and focal CE params for SAM 3.1 training.

    Most fields are demoted internal constants per audit Section E. Only
    w_mask, w_obj, w_presence, and matcher_weights survive as advanced
    settings read by the training loop, but none are exposed to the YAML
    user schema — they are hardcoded defaults here.

    No `w_cls`: discrimination across classes comes from running one forward
    pass per class prompt. `w_presence` weights the image-level
    "any-instance-of-this-class-present?" supervision.
    """

    w_mask: float = 1.0
    w_obj: float = 1.0
    w_presence: float = 1.0
    # w_box is demoted: always 0.0 in all examples; no box supervision in v0.
    w_box: float = 0.0
    matcher_weights: MatcherWeights = field(default_factory=MatcherWeights)
    # focal_gamma and focal_alpha are demoted internal constants.
    focal_gamma: float = 2.0
    focal_alpha: float = 0.25


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
