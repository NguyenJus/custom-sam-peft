"""SAM3.1 loader + forward wrapper. Implementation deferred to spec/model-loading."""

from __future__ import annotations

from typing import Any

from esam3.config.schema import ModelConfig


def load_sam31(cfg: ModelConfig) -> Any:
    """Load SAM3.1 from HuggingFace, applying dtype + grad-checkpointing flags."""
    raise NotImplementedError("filled in by spec: spec/model-loading")
