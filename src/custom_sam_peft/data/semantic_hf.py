"""Semantic HF adapter (SS5.4).

Wraps a HuggingFace dataset that exposes a per-pixel label-map feature and
produces Example objects carrying SemanticTarget dense label maps.
Dispatched by build_hf when cfg['task'] == 'semantic'.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from custom_sam_peft.data.base import Example, SemanticTarget, TextPrompts
from custom_sam_peft.data.io import _coerce_to_channels

_LOG = logging.getLogger(__name__)


class SemanticHFDataset:
    """HuggingFace adapter for semantic segmentation (SS5.4).

    Reads a per-pixel label-map feature from an HF dataset and produces
    Example objects with SemanticTarget dense label maps.

    class_names: K concept names (background excluded).
    image_class_labels: per-image frozenset of present GT class ids
        (background=0 and ignore_index excluded) for stratified subset sampling.
    """

    def __init__(
        self,
        hf_dataset: Any,
        *,
        image_field: str,
        label_map_field: str | None,
        class_names: list[str],
        ignore_index: int,
        transforms: Any,
        channels: int,
        value_to_label: dict[int, int] | None = None,
    ) -> None:
        if label_map_field is None:
            raise ValueError(
                "SemanticHFDataset requires label_map_field to be set. "
                "Set data.hf.field_map.label_map to the HF feature name holding "
                "the (H,W) per-pixel label image."
            )
        self._ds = hf_dataset
        self._image_field = image_field
        self._label_map_field = label_map_field
        self._class_names = list(class_names)
        self._ignore_index = ignore_index
        self._transforms = transforms
        self._channels = channels
        self._image_class_labels_cache: list[frozenset[int]] | None = None

        # tbd: #113 -- direct-construction default assumes annotation already uses
        # {0=bg, 1..K=class}; the build_hf path always passes an explicit value_to_label.
        self._value_to_label: dict[int, int] | None = value_to_label

    # ------------------------------------------------------------------
    # Pixel-value -> GT-label
    # ------------------------------------------------------------------

    def _remap_pixel(self, v: int) -> int:
        """Map a single raw pixel value to its GT label integer."""
        if v == self._ignore_index:
            return self._ignore_index
        if self._value_to_label is not None:
            return self._value_to_label.get(v, 0)
        # Direct-construction default: {0=bg, 1..K=class}, unknown -> bg.
        k = len(self._class_names)
        if 1 <= v <= k:
            return v
        return 0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._ds)

    @property
    def class_names(self) -> list[str]:
        return list(self._class_names)

    @property
    def image_class_labels(self) -> list[frozenset[int]]:
        """Per-image dense class id sets for stratified subset sampling (lazy, memoized)."""
        if self._image_class_labels_cache is not None:
            return self._image_class_labels_cache
        result: list[frozenset[int]] = []
        for i in range(len(self._ds)):
            row = self._ds[i]
            lbl_obj = row[self._label_map_field]
            raw = _label_to_numpy(lbl_obj)
            present: set[int] = set()
            for v in np.unique(raw).tolist():
                gt = self._remap_pixel(int(v))
                if gt != 0 and gt != self._ignore_index:
                    present.add(gt)
            result.append(frozenset(present))
        self._image_class_labels_cache = result
        return self._image_class_labels_cache

    def __getitem__(self, i: int) -> Example:
        import torch

        row = self._ds[i]

        # --- image ---
        img_obj = row[self._image_field]
        np_img = _coerce_to_channels(img_obj, self._channels)

        # --- label map: raw (H,W) uint8/uint16, NO scaling ---
        lbl_obj = row[self._label_map_field]
        raw_lbl = _label_to_numpy(lbl_obj)

        # --- transforms ---
        if self._transforms is not None:
            out = self._transforms(
                image=np_img,
                bboxes=[],
                masks=[raw_lbl],
                class_labels=[],
                instance_idx=[],
            )
            image_tensor: Any = out["image"]
            lbl_out = out["masks"][0]
            if isinstance(lbl_out, torch.Tensor):
                transformed_lbl: np.ndarray[Any, Any] = lbl_out.numpy()
            else:
                transformed_lbl = np.asarray(lbl_out)
        else:
            # No-transform path: minimal HWC -> CHW float conversion.
            image_tensor = torch.from_numpy(np_img.astype(np.float32).transpose(2, 0, 1))
            transformed_lbl = np.asarray(raw_lbl)

        # --- vectorized pixel-value -> GT-label remap ---
        max_val = int(transformed_lbl.max()) if transformed_lbl.size > 0 else 0
        lut_size = max(max_val + 1, self._ignore_index + 1)
        lut = np.zeros(lut_size, dtype=np.int64)
        for v in range(lut_size):
            lut[v] = self._remap_pixel(v)
        clamped = np.clip(transformed_lbl.astype(np.int64), 0, lut_size - 1)
        remapped = lut[clamped]

        # --- text prompts: always 'all' for semantic task (SS5.6) ---
        prompts = TextPrompts(classes=list(self._class_names))

        semantic = SemanticTarget(
            labels=torch.from_numpy(remapped).to(torch.int64),
            ignore_index=self._ignore_index,
        )
        return Example(
            image=image_tensor,
            image_id=str(i),
            prompts=prompts,
            semantic=semantic,
        )


def _label_to_numpy(obj: Any) -> np.ndarray[Any, Any]:
    """Convert an HF label-map value (PIL Image or array) to (H,W) ndarray."""
    from PIL import Image as PILImage

    if isinstance(obj, PILImage.Image):
        return np.asarray(obj)
    arr = np.asarray(obj)
    if arr.ndim == 2:
        return arr
    # (H,W,1) -> squeeze
    if arr.ndim == 3 and arr.shape[2] == 1:
        return arr[:, :, 0]
    raise ValueError(
        f"SemanticHFDataset: label_map feature has unexpected shape {arr.shape}; "
        "expected (H,W) or (H,W,1)."
    )
