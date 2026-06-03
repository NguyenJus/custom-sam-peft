"""mask_png semantic data adapter (SS5.3).

Turns a paired image-dir + label-PNG-dir into Example objects carrying
SemanticTarget dense label maps. Registered as dataset format 'mask_png'.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal

import numpy as np
from PIL import Image

from custom_sam_peft._registry import register
from custom_sam_peft.config.schema import NormalizeConfig, TextPromptConfig
from custom_sam_peft.data._semantic_encode import build_value_to_label
from custom_sam_peft.data.base import Dataset, Example, SemanticTarget, TextPrompts
from custom_sam_peft.data.io import read_image

_LOG = logging.getLogger(__name__)

# Recognized pixel image extensions for enumeration.
_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

# One-time guard for non-'all' text_prompt mode warning.
_warned_non_all_mode = False


class MaskPngDataset:
    """Semantic segmentation dataset backed by paired image + label-PNG directories.

    class_names: K concept names in ascending pixel-value order (background excluded).
    image_class_labels: per-image frozenset of present GT class ids (background=0 and
    ignore_index excluded), matching COCODataset's pattern for stratified subset sampling.
    """

    def __init__(
        self,
        images_dir: str | Path,
        labels_dir: str | Path,
        *,
        class_map_path: str,
        ignore_index: int,
        label_suffix: str,
        transforms: Any,
        text_prompt: TextPromptConfig,
        channels: int,
        resolved_image_ids: frozenset[str] | None = None,
    ) -> None:
        self._images_dir = Path(images_dir)
        self._labels_dir = Path(labels_dir)
        self._label_suffix = label_suffix
        self._transforms = transforms
        self._text_prompt_cfg = text_prompt
        self._channels = channels

        self._class_names, self._value_to_label, self._ignore_index = build_value_to_label(
            class_map_path, ignore_index=ignore_index, background_class_name=None
        )

        # Enumerate image files (sorted for determinism), pair each to its label.
        img_paths = sorted(p for p in self._images_dir.iterdir() if p.suffix.lower() in _IMAGE_EXTS)
        missing: list[str] = []
        pairs: list[tuple[Path, Path]] = []
        for img_path in img_paths:
            lbl_path = self._labels_dir / (img_path.stem + self._label_suffix)
            if not lbl_path.exists():
                missing.append(img_path.stem)
            else:
                pairs.append((img_path, lbl_path))
        if missing:
            raise FileNotFoundError(
                f"mask_png: {len(missing)} images have no label; first few: {missing[:5]}"
            )

        # Auto-split filtering: if the trainer injected _resolved_image_ids, restrict
        # the dataset to images whose stem is in that set (mirrors COCODataset's
        # image_ids filter). Stems not in the resolved set are silently skipped —
        # this is intentional; the trainer is responsible for correctness of the split.
        if resolved_image_ids is not None:
            missing_ids = resolved_image_ids - {p.stem for p, _ in pairs}
            if missing_ids:
                first_few = sorted(missing_ids)[:10]
                raise ValueError(
                    f"MaskPngDataset: {len(missing_ids)} _resolved_image_ids not found "
                    f"in images_dir (first few stems): {first_few}"
                )
            pairs = [(p, lp) for p, lp in pairs if p.stem in resolved_image_ids]

        self._pairs = pairs

        # Eager per-image class label sets for stratified subset sampling —
        # mirrors COCODataset.image_class_labels.
        self.image_class_labels: list[frozenset[int]] = []
        for _img_path, lbl_path in self._pairs:
            raw = np.array(Image.open(lbl_path))
            present: set[int] = set()
            for v in np.unique(raw).tolist():
                gt = self._remap_pixel(int(v))
                if gt != 0 and gt != self._ignore_index:
                    present.add(gt)
            self.image_class_labels.append(frozenset(present))

    # ------------------------------------------------------------------
    # Pixel-value -> GT-label helper
    # ------------------------------------------------------------------

    def _remap_pixel(self, v: int) -> int:
        """Map a single raw pixel value to its GT label integer."""
        if v == self._ignore_index:
            return self._ignore_index
        return self._value_to_label.get(v, 0)  # unknown values -> background (0)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, i: int) -> Example:
        import torch

        img_path, lbl_path = self._pairs[i]

        # --- image ---
        np_img = read_image(img_path, self._channels)

        # --- label PNG: read raw class-index pixels, no normalization ---
        raw_lbl = np.array(Image.open(lbl_path))  # (H, W) uint8 or uint16

        # --- apply spatial transforms (image + dense label map aligned) ---
        # Pass the label map as a single-element masks list; Albumentations
        # applies INTER_NEAREST to masks, preserving class-index integrity.
        out = self._transforms(
            image=np_img,
            bboxes=[],
            masks=[raw_lbl],
            class_labels=[],
            instance_idx=[],
        )
        image_tensor: Any = out["image"]
        # ToTensorV2 converts masks to torch.Tensor; bring back to numpy for remapping.
        lbl_out = out["masks"][0]
        import torch as _torch

        if isinstance(lbl_out, _torch.Tensor):
            transformed_lbl: np.ndarray[Any, Any] = lbl_out.numpy()
        else:
            transformed_lbl = np.asarray(lbl_out)

        # --- vectorized pixel-value -> GT-label remap ---
        # Build a lookup array sized to the max value present in the label map.
        max_val = int(transformed_lbl.max()) if transformed_lbl.size > 0 else 0
        lut_size = max(max_val + 1, self._ignore_index + 1)
        lut = np.zeros(lut_size, dtype=np.int64)
        for v in range(lut_size):
            lut[v] = self._remap_pixel(v)
        # Clamp any values that exceed the LUT (shouldn't occur, but be safe).
        clamped = np.clip(transformed_lbl.astype(np.int64), 0, lut_size - 1)
        remapped = lut[clamped]

        # --- text prompts: always 'all' for semantic task (SS5.6) ---
        global _warned_non_all_mode
        if self._text_prompt_cfg.mode != "all" and not _warned_non_all_mode:
            _LOG.info(
                "task: semantic forces text-prompt mode 'all'; ignoring mode=%s",
                self._text_prompt_cfg.mode,
            )
            _warned_non_all_mode = True
        prompts = TextPrompts(classes=list(self._class_names))

        semantic = SemanticTarget(
            labels=torch.from_numpy(remapped).to(torch.int64),
            ignore_index=self._ignore_index,
        )
        return Example(
            image=image_tensor,
            image_id=img_path.stem,
            prompts=prompts,
            semantic=semantic,
        )

    @property
    def class_names(self) -> list[str]:
        return list(self._class_names)


@register("dataset", "mask_png")
def build_mask_png(
    cfg: dict[str, Any],
    *,
    model_name: str,
    pipeline: Literal["train", "eval"],
) -> Dataset:
    """Build a MaskPngDataset from a validated DataConfig dict.

    Mirrors build_coco: the caller passes cfg = DataConfig.model_dump() and
    selects the split via the 'pipeline' arg ('train' -> cfg['train'],
    'eval' -> cfg['val'] or cfg['train'] if val is absent).
    cfg['semantic'] must be present (enforced by the task<->data validator).
    """
    from custom_sam_peft.config.schema import AugmentationsConfig
    from custom_sam_peft.data.transforms import build_eval_transforms, build_train_transforms
    from custom_sam_peft.models.sam3 import SAM3_IMAGE_SIZE

    if pipeline not in ("train", "eval"):
        raise ValueError(f"pipeline must be 'train' or 'eval'; got {pipeline!r}")

    # Split selection mirrors build_coco exactly.
    if pipeline == "eval" and cfg.get("val") is None:
        split = cfg["train"]
    else:
        split_key = "train" if pipeline == "train" else "val"
        split = cfg[split_key]

    sem = cfg.get("semantic") or {}
    class_map: str = str(sem.get("class_map", ""))
    ignore_index: int = int(sem.get("ignore_index", 255))
    label_suffix: str = str(sem.get("label_suffix", "_labelIds.png"))

    normalize = NormalizeConfig.model_validate(cfg.get("normalize") or {})
    text_prompt = TextPromptConfig.model_validate(cfg.get("text_prompt") or {})
    channel_semantics: str = str(cfg.get("channel_semantics", "rgb"))

    if pipeline == "train":
        aug = AugmentationsConfig.model_validate(cfg.get("augmentations") or {})
        transforms = build_train_transforms(
            aug,
            SAM3_IMAGE_SIZE,
            model_name=model_name,
            normalize=normalize,
            channel_semantics=channel_semantics,
            channels=int(cfg.get("channels", 3)),
        )
    else:
        transforms = build_eval_transforms(
            SAM3_IMAGE_SIZE,
            model_name=model_name,
            normalize=normalize,
            channel_semantics=channel_semantics,
        )

    resolved_ids: frozenset[str] | None = None
    raw_resolved = (cfg.get("_resolved_image_ids") or {}).get(pipeline)
    if raw_resolved is not None:
        resolved_ids = frozenset(str(s) for s in raw_resolved)

    return MaskPngDataset(
        images_dir=split["images"],
        labels_dir=split["annotations"],
        class_map_path=class_map,
        ignore_index=ignore_index,
        label_suffix=label_suffix,
        transforms=transforms,
        text_prompt=text_prompt,
        channels=int(cfg.get("channels", 3)),
        resolved_image_ids=resolved_ids,
    )
