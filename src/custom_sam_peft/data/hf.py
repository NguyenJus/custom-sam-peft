"""HuggingFace `datasets` dataset adapter.

Uses a hybrid input contract: conventional dotted field paths with full
override via `HFFieldMap`. Class names come from a top-level `categories`
feature, or fall back to a `ClassLabel` inside the per-box category field.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any, Literal

import numpy as np

from custom_sam_peft._registry import register
from custom_sam_peft.config.schema import HFFieldMap, TextPromptConfig
from custom_sam_peft.data.base import Dataset, Example

_LOG = logging.getLogger(__name__)


class HFFieldError(KeyError):
    """Raised when the HF dataset does not contain a required field."""


def _resolve_field(row: dict[str, Any], dotted: str) -> Any:
    """Walk a dotted path against a row dict; raise `KeyError(dotted)` on miss."""
    node: Any = row
    parts = dotted.split(".")
    for part in parts:
        if not isinstance(node, dict) or part not in node:
            raise KeyError(dotted)
        node = node[part]
    return node


def _normalize_bbox(
    b: list[float] | tuple[float, ...], fmt: Literal["xywh", "xyxy"]
) -> tuple[float, float, float, float]:
    """Return `(x0, y0, x1, y1)`."""
    a, b1, c, d = (float(b[0]), float(b[1]), float(b[2]), float(b[3]))
    if fmt == "xywh":
        return (a, b1, a + c, b1 + d)
    return (a, b1, c, d)


def _validate_required_fields(ds: Any, field_map: HFFieldMap) -> None:
    """Read one row and ensure every required path resolves.

    Raises:
        HFFieldError: if any required path is missing; message contains the
            dotted path and the override key (`data.hf.field_map.<key>`).
    """
    if len(ds) == 0:
        return
    row = ds[0]
    required: list[tuple[str, str]] = [
        (field_map.image, "image"),
        (field_map.bbox, "bbox"),
        (field_map.category, "category"),
    ]
    for path, override_key in required:
        try:
            _resolve_field(row, path)
        except KeyError as e:
            raise HFFieldError(
                f"HF dataset is missing required field '{path}'. "
                f"Set data.hf.field_map.{override_key} to the correct dotted path."
            ) from e


def _resolve_class_names(ds: Any, field_map: HFFieldMap) -> list[str]:
    """Resolve dataset class names.

    Order of attempts:
      1. Top-level feature named `field_map.categories_feature` whose value is
         a `Sequence(ClassLabel)` or a `list[str]` per row.
      2. If absent, look for a `ClassLabel` feature at `<field_map.category>`
         inside `ds.features` and return its `names`.
    """
    feats = getattr(ds, "features", None)
    if feats is not None and field_map.categories_feature in feats:
        feat = feats[field_map.categories_feature]
        inner = getattr(feat, "feature", None)
        names = getattr(inner, "names", None)
        if names:
            return list(names)
        if len(ds) > 0:
            row_val = ds[0].get(field_map.categories_feature)
            if isinstance(row_val, list) and all(isinstance(x, str) for x in row_val):
                return list(row_val)
    if feats is not None:
        node: Any = feats
        for part in field_map.category.split("."):
            # Descend through any number of Sequence/List wrappers to reach a dict.
            while node is not None and not isinstance(node, dict):
                inner = getattr(node, "feature", None)
                if inner is None:
                    break
                node = inner
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                node = None
                break
        # Unwrap remaining Sequence/List wrappers to reach the ClassLabel.
        while node is not None and getattr(node, "names", None) is None:
            inner = getattr(node, "feature", None)
            if inner is None:
                break
            node = inner
        names = getattr(node, "names", None) if node is not None else None
        if names:
            return list(names)
    raise HFFieldError(
        "Cannot resolve class names. Set data.hf.field_map.categories_feature "
        "to a top-level Sequence(ClassLabel) feature, or use a ClassLabel-typed "
        "category field."
    )


from datasets import load_dataset as hf_load_dataset  # noqa: E402


class HFDataset:
    """HuggingFace `datasets` adapter."""

    def __init__(
        self,
        name: str,
        split: str,
        prompt_mode: Literal["text", "bbox"],
        *,
        transforms: Any,
        text_prompt: TextPromptConfig,
        field_map: HFFieldMap,
        seed: int = 0,
        row_indices: Iterable[int] | None = None,
    ) -> None:
        if prompt_mode not in ("text", "bbox"):
            raise ValueError(f"prompt_mode must be 'text' or 'bbox'; got {prompt_mode!r}")
        self._name = name
        self._split = split
        self._prompt_mode: Literal["text", "bbox"] = prompt_mode
        self._transforms = transforms
        self._text_prompt_cfg = text_prompt
        self._field_map = field_map
        self._seed = seed
        self._multiplex_cap = 16
        self._warned_truncation = False
        self._warned_masks_from_boxes = False
        self._image_class_labels: list[frozenset[int]] | None = None

        self._ds = hf_load_dataset(name, split=split)
        _validate_required_fields(self._ds, field_map)
        self._class_names = _resolve_class_names(self._ds, field_map)
        if row_indices is not None:
            self._index_map: list[int] | None = [int(i) for i in row_indices]
            invalid = [i for i in self._index_map if i < 0 or i >= len(self._ds)]
            if invalid:
                raise ValueError(
                    f"HFDataset: {len(invalid)} row_indices out of range "
                    f"[0, {len(self._ds)}): first few = {invalid[:10]}"
                )
        else:
            self._index_map = None

    def __len__(self) -> int:
        return len(self._index_map) if self._index_map is not None else len(self._ds)

    @property
    def image_class_labels(self) -> list[frozenset[int]]:
        """Per-image dense class id sets for stratified subset sampling.

        Computed lazily on first access; subsequent accesses return the cache.
        Emits exactly one INFO log per dataset instance when computed.
        """
        if self._image_class_labels is None:
            _LOG.info(
                "stratified subset: scanning %d rows for class labels…",
                len(self._ds),
            )
            cat_field = self._field_map.category
            result: list[frozenset[int]] = []
            for i in range(len(self._ds)):
                row = self._ds[i]
                raw = _resolve_field(row, cat_field)
                cats = [int(c) for c in raw] if isinstance(raw, list) else [int(raw)]
                result.append(frozenset(cats))
            self._image_class_labels = result
        return self._image_class_labels

    def __getitem__(self, i: int) -> Example:
        import random as _random

        import torch
        from PIL import Image as PILImage

        from custom_sam_peft.data.base import BoxPrompts, Instance, TextPrompts
        from custom_sam_peft.data.coco import _build_text_prompts

        row_i = self._index_map[i] if self._index_map is not None else i
        row = self._ds[row_i]
        img_obj = _resolve_field(row, self._field_map.image)
        if isinstance(img_obj, PILImage.Image):
            np_img = np.asarray(img_obj.convert("RGB"))
        else:
            np_img = np.asarray(img_obj)
            if np_img.ndim == 2:
                np_img = np.stack([np_img] * 3, axis=-1)
        h, w = int(np_img.shape[0]), int(np_img.shape[1])

        bboxes_raw = _resolve_field(row, self._field_map.bbox)
        classes = list(_resolve_field(row, self._field_map.category))
        bboxes_xyxy = [_normalize_bbox(list(b), self._field_map.bbox_format) for b in bboxes_raw]

        masks: list[np.ndarray[Any, Any]] = []
        seg_path = self._field_map.segmentation
        seg_resolved: Any = None
        if seg_path:
            try:
                seg_resolved = _resolve_field(row, seg_path)
            except KeyError:
                seg_resolved = None
        if seg_resolved is None:
            if not self._warned_masks_from_boxes:
                _LOG.warning(
                    "custom_sam_peft.data.hf: masks-from-boxes fallback used for dataset %r "
                    "(field_map.segmentation absent or None). Suppressing further warnings.",
                    self._name,
                )
                self._warned_masks_from_boxes = True
            for x0, y0, x1, y1 in bboxes_xyxy:
                m = np.zeros((h, w), dtype=np.uint8)
                xi0, yi0 = max(0, int(x0)), max(0, int(y0))
                xi1, yi1 = min(w, int(x1)), min(h, int(y1))
                if xi1 > xi0 and yi1 > yi0:
                    m[yi0:yi1, xi0:xi1] = 1
                masks.append(m)
        else:
            from custom_sam_peft.data.coco import _decode_segmentation

            for ann in seg_resolved:
                masks.append(_decode_segmentation({"segmentation": ann}, h, w).astype(np.uint8))

        out = self._transforms(
            image=np_img,
            bboxes=[list(b) for b in bboxes_xyxy],
            masks=masks,
            class_labels=classes,
        )
        image_tensor: torch.Tensor = out["image"]
        out_bboxes = list(out["bboxes"])
        out_masks = list(out["masks"])
        out_classes = [int(c) for c in out["class_labels"]]

        instances: list[Instance] = []
        for box, mask_np, cls in zip(out_bboxes, out_masks, out_classes, strict=True):
            instances.append(
                Instance(
                    mask=torch.from_numpy(np.asarray(mask_np).astype(bool)),
                    class_id=int(cls),
                    box=torch.tensor(box, dtype=torch.float32),
                )
            )

        image_id = str(row_i)
        if self._prompt_mode == "text":
            present = sorted(set(out_classes))
            rng = _random.Random(f"{self._seed}:{row_i}")  # noqa: S311 — deterministic seeded RNG for prompt sampling, not security
            prompts_list = _build_text_prompts(
                present_dense_ids=present,
                class_names=self._class_names,
                cfg=self._text_prompt_cfg,
                rng=rng,
                image_id=row_i,
            )
            if len(prompts_list) > self._multiplex_cap:
                if not self._warned_truncation:
                    _LOG.warning(
                        "custom_sam_peft.data.hf: image_id=%s requested %d text prompts; "
                        "truncating to %d. Suppressing further warnings.",
                        image_id,
                        len(prompts_list),
                        self._multiplex_cap,
                    )
                    self._warned_truncation = True
                prompts_list = prompts_list[: self._multiplex_cap]
            return Example(
                image=image_tensor,
                image_id=image_id,
                prompts=TextPrompts(classes=prompts_list),
                instances=instances,
            )

        order = sorted(
            range(len(instances)),
            key=lambda k: (
                instances[k].class_id,
                float(instances[k].box[0]),
                float(instances[k].box[1]),
            ),
        )
        if len(order) > self._multiplex_cap:
            if not self._warned_truncation:
                _LOG.warning(
                    "custom_sam_peft.data.hf: image_id=%s requested %d box prompts; "
                    "truncating to %d. Suppressing further warnings.",
                    image_id,
                    len(order),
                    self._multiplex_cap,
                )
                self._warned_truncation = True
            order = order[: self._multiplex_cap]
        kept_instances = [instances[k] for k in order]
        boxes_t = (
            torch.stack([inst.box for inst in kept_instances])
            if kept_instances
            else torch.zeros((0, 4))
        )
        class_ids_t = torch.tensor([inst.class_id for inst in kept_instances], dtype=torch.int64)
        return Example(
            image=image_tensor,
            image_id=image_id,
            prompts=BoxPrompts(boxes=boxes_t.to(torch.float32), class_ids=class_ids_t),
            instances=kept_instances,
        )

    @property
    def class_names(self) -> list[str]:
        return list(self._class_names)


@register("dataset", "hf")
def build_hf(
    cfg: dict[str, Any],
    *,
    model_name: str,
    pipeline: Literal["train", "eval"],
) -> Dataset:
    """Build an `HFDataset` from a validated DataConfig dict."""
    from custom_sam_peft.config.schema import AugmentationsConfig, NormalizeConfig
    from custom_sam_peft.data.transforms import build_eval_transforms, build_train_transforms

    if pipeline not in ("train", "eval"):
        raise ValueError(f"pipeline must be 'train' or 'eval'; got {pipeline!r}")
    hf_cfg = cfg["hf"]
    resolved = (cfg.get("_resolved_image_ids") or {}).get(pipeline)
    if pipeline == "eval" and cfg.get("val") is None and resolved is not None:
        split = hf_cfg["split_train"]
    else:
        split = hf_cfg["split_train"] if pipeline == "train" else hf_cfg["split_val"]
    image_size = int(cfg["image_size"])
    normalize = NormalizeConfig.model_validate(cfg.get("normalize", {}))
    text_prompt = TextPromptConfig.model_validate(cfg.get("text_prompt", {}))
    field_map = HFFieldMap.model_validate(hf_cfg.get("field_map", {}))
    if pipeline == "train":
        aug = AugmentationsConfig.model_validate(cfg.get("augmentations", {}))
        transforms = build_train_transforms(
            aug, image_size, model_name=model_name, normalize=normalize
        )
    else:
        transforms = build_eval_transforms(image_size, model_name=model_name, normalize=normalize)
    return HFDataset(
        name=hf_cfg["name"],
        split=split,
        prompt_mode=cfg["prompt_mode"],
        transforms=transforms,
        text_prompt=text_prompt,
        field_map=field_map,
        row_indices=[int(s) for s in resolved] if resolved is not None else None,
    )
