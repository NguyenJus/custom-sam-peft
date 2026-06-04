"""Best-effort num_classes helper for VRAM autosize (§5 of the pin-r/alpha spec).

Derives the class count from the dataset configuration without loading the full
dataset or the model. On any failure returns None and emits a single WARNING.

Public surface: ``infer_num_classes(data_cfg) -> int | None``.

Spec: 2026-06-04-pin-lora-rank-alpha-autosize-design.md §5.
"""

from __future__ import annotations

import logging
from typing import Any

_LOG = logging.getLogger(__name__)


def _num_classes_mask_png(data_cfg: dict[str, Any]) -> int | None:
    """Return class count from a mask_png data config.

    Reads the class_map JSON via build_value_to_label (cheap — JSON read only,
    no model or image loading). Spec §5.
    """
    # The mask_png config is nested: data_cfg["semantic"]["class_map"] or at top level.
    # Callers pass the full DataConfig.model_dump(); drill into the relevant sub-dict.
    semantic = data_cfg.get("semantic") or {}
    class_map_path: str | None = semantic.get("class_map")
    if not class_map_path:
        _LOG.warning("infer_num_classes(mask_png): no class_map in config; falling back to k_cap")
        return None
    try:
        from custom_sam_peft.data._semantic_encode import build_value_to_label

        class_names, _, _ = build_value_to_label(
            class_map_path, ignore_index=-1, background_class_name=None
        )
        return len(class_names)
    except Exception as exc:
        _LOG.warning(
            "infer_num_classes(mask_png): failed to read class_map %r (%s); falling back to k_cap",
            class_map_path,
            exc,
        )
        return None


def _num_classes_hf(data_cfg: dict[str, Any]) -> int | None:
    """Return class count from an HF data config.

    Resolves class names via _resolve_class_names on a lazily loaded HF dataset.
    Requires the ``datasets`` library and network/cache access to the dataset.
    Spec §5.
    """
    hf_cfg = data_cfg.get("hf") or {}
    name: str | None = hf_cfg.get("name")
    split: str = hf_cfg.get("split", "train")
    if not name:
        _LOG.warning("infer_num_classes(hf): no dataset name in config; falling back to k_cap")
        return None
    try:
        from custom_sam_peft.config.schema import HFFieldMap
        from custom_sam_peft.data.hf import _resolve_class_names

        try:
            from datasets import load_dataset as hf_load_dataset
        except ImportError as exc:
            _LOG.warning(
                "infer_num_classes(hf): datasets library not available (%s); falling back to k_cap",
                exc,
            )
            return None

        # Build a minimal field_map from the config, using HFFieldMap defaults.
        field_map_raw: dict[str, Any] = hf_cfg.get("field_map") or {}
        field_map = HFFieldMap(**field_map_raw) if field_map_raw else HFFieldMap()

        ds = hf_load_dataset(name, split=split)
        names = _resolve_class_names(ds, field_map)
        return len(names)
    except Exception as exc:
        _LOG.warning(
            "infer_num_classes(hf): failed to resolve class names for %r (%s); "
            "falling back to k_cap",
            name,
            exc,
        )
        return None


def infer_num_classes(data_cfg: dict[str, Any] | None) -> int | None:
    """Best-effort class count from a DataConfig dict (model_dump() format).

    Returns ``int`` ≥ 1 when successfully resolved, ``None`` on any failure.
    On failure emits a single WARNING and the sizer falls back to ``k_cap``
    (equivalent to today's unconstrained behavior). Must never hard-fail.

    Supported formats:
    - ``mask_png``: reads the class_map JSON (cheap, no model load). Spec §5.
    - ``hf``: resolves ClassLabel.names via _resolve_class_names. Spec §5.

    Args:
        data_cfg: the DataConfig dict (``DataConfig.model_dump()``). Pass None
            or an empty dict to get None back without a warning.
    """
    if not data_cfg:
        return None
    fmt: str | None = data_cfg.get("format")
    if fmt == "mask_png":
        return _num_classes_mask_png(data_cfg)
    if fmt == "hf":
        return _num_classes_hf(data_cfg)
    # Unknown format — silently skip; callers treat None as "use k_cap".
    return None
