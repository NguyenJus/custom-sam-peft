# src/custom_sam_peft/data/_semantic_encode.py
"""SS5.2/SS4.5 GT-encoding: class_map JSON -> concept names + pixel-value->GT-label map.

Single source of truth shared by the mask_png and semantic-HF adapters. The
prompted concept order DEFINES the dense ids and the (K+1)-channel <-> GT-label
correspondence (spec SS5.2). Pure-Python (no torch).
"""

from __future__ import annotations

import json

# Recognized explicit-background class names (case-insensitive), SS4.5.
_BACKGROUND_NAMES = frozenset({"background", "bg", "none", "unlabeled"})


def build_value_to_label(
    class_map_path: str,
    *,
    ignore_index: int,
    background_class_name: str | None,
) -> tuple[list[str], dict[int, int], int]:
    """Return (class_names, value_to_label, ignore_index).

    - class_names: concept names in ASCENDING class_map pixel-value order, with any
      explicit background class removed. len == K.
    - value_to_label: pixel value -> GT label, where 0 == background, i+1 == concept
      with dense_id i. The configured ignore_index value is NOT placed in this map;
      callers remap it separately (it always wins, SS4.5).
    """
    with open(class_map_path, encoding="utf-8") as fh:
        raw: dict[str, str] = json.load(fh)
    # Sort by integer pixel value ascending.
    pairs = sorted(((int(v), name) for v, name in raw.items()), key=lambda kv: kv[0])

    bg_lower = background_class_name.lower() if background_class_name is not None else None

    def _is_background(name: str) -> bool:
        low = name.lower()
        if bg_lower is not None:
            return low == bg_lower
        return low in _BACKGROUND_NAMES

    class_names: list[str] = []
    value_to_label: dict[int, int] = {}
    for value, name in pairs:
        if _is_background(name):
            value_to_label[value] = 0  # background channel
            continue
        dense_id = len(class_names)
        class_names.append(name)
        value_to_label[value] = dense_id + 1  # +1 for the prepended background channel
    return class_names, value_to_label, ignore_index
