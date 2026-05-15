"""Variable-shape batch collator. Implementation deferred to spec/data-loading."""

from __future__ import annotations

from typing import Any

from esam3.data.base import Example


def collate_batch(examples: list[Example]) -> dict[str, Any]:
    raise NotImplementedError("filled in by spec: spec/data-loading")
