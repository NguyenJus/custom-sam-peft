"""COCO instance-JSON dataset adapter.

Backed by `pycocotools.coco.COCO` for index lookups and `pycocotools.mask` for
polygon/RLE decode. Sparse COCO category ids are remapped to a dense 0..C-1
namespace; the original sparse ids are preserved on `coco_category_ids` for
eval-time round-tripping.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pycocotools import mask as coco_mask
from pycocotools.coco import COCO

from custom_sam_peft._registry import register
from custom_sam_peft.config.schema import TextPromptConfig
from custom_sam_peft.data.base import Dataset, Example

_LOG = logging.getLogger(__name__)


def _load_coco_index(ann_path: str | Path) -> COCO:
    """Load a COCO annotations JSON via pycocotools (suppresses pycocotools prints)."""
    import contextlib
    import io

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        return COCO(str(ann_path))


def _build_category_remap(coco: COCO) -> tuple[list[int], dict[int, int], list[str]]:
    """Return `(sparse_ids_sorted, sparse_to_dense, class_names_in_dense_order)`."""
    cats = sorted(coco.dataset["categories"], key=lambda c: c["id"])
    sparse_ids = [int(c["id"]) for c in cats]
    names = [str(c["name"]) for c in cats]
    mapping = {sid: dense for dense, sid in enumerate(sparse_ids)}
    return sparse_ids, mapping, names


def _drop_crowd_only_images(
    coco: COCO,
) -> tuple[list[int], dict[int, list[dict[str, Any]]], int]:
    """Drop images that have zero non-crowd annotations.

    Returns `(image_ids_kept_sorted, ann_index_no_crowd, dropped_count)`.
    """
    kept: list[int] = []
    ann_index: dict[int, list[dict[str, Any]]] = {}
    dropped = 0
    for img_id in sorted(coco.getImgIds()):
        anns = coco.loadAnns(coco.getAnnIds(imgIds=[img_id]))
        non_crowd = [a for a in anns if int(a.get("iscrowd", 0)) == 0]
        if not non_crowd:
            dropped += 1
            continue
        kept.append(int(img_id))
        ann_index[int(img_id)] = non_crowd
    return kept, ann_index, dropped


def _decode_segmentation(ann: dict[str, Any], h: int, w: int) -> np.ndarray[Any, Any]:
    """Polygon or RLE -> (H, W) bool ndarray."""
    seg = ann["segmentation"]
    if isinstance(seg, list):
        rles = coco_mask.frPyObjects(seg, h, w)
        decoded = coco_mask.decode(rles)
    elif isinstance(seg, dict):
        decoded = coco_mask.decode(seg)
    else:
        raise TypeError(f"unsupported segmentation type: {type(seg).__name__}")
    if decoded.ndim == 3:
        decoded = decoded.sum(axis=2)
    return decoded.astype(bool)  # type: ignore[no-any-return]


def _build_text_prompts(
    present_dense_ids: list[int],
    class_names: list[str],
    cfg: TextPromptConfig,
    rng: random.Random,
    image_id: int,
) -> list[str]:
    """Apply the configured TextPromptMode. Output order:
    positives in ascending dense-id, then negatives in deterministic order.
    """
    present_sorted = sorted(set(present_dense_ids))
    positives = [class_names[i] for i in present_sorted]
    n = len(class_names)
    if cfg.mode == "present":
        return positives
    if cfg.mode == "all":
        return list(class_names)
    if cfg.mode == "present_plus_negatives":
        pool = [i for i in range(n) if i not in set(present_sorted)]
        negatives = rng.sample(pool, k=min(cfg.negatives_per_image, len(pool)))
        return positives + [class_names[i] for i in sorted(negatives)]
    if cfg.mode == "sampled_fixed_k":
        if len(positives) >= cfg.k:
            return positives[: cfg.k]
        pool = [i for i in range(n) if i not in set(present_sorted)]
        need = cfg.k - len(positives)
        negatives = rng.sample(pool, k=min(need, len(pool)))
        return positives + [class_names[i] for i in sorted(negatives)]
    raise ValueError(f"unknown text-prompt mode: {cfg.mode}")


class COCODataset:
    """COCO instance-JSON dataset.

    Sparse COCO category ids -> dense 0..C-1; images with only iscrowd=1
    annotations are dropped at construction; per-image multiplex capped at 16.
    """

    coco_category_ids: list[int]

    def __init__(
        self,
        annotations: str,
        images: str,
        prompt_mode: Literal["text", "bbox"],
        *,
        transforms: Any,
        text_prompt: TextPromptConfig,
        seed: int = 0,
    ) -> None:
        if prompt_mode not in ("text", "bbox"):
            raise ValueError(f"prompt_mode must be 'text' or 'bbox'; got {prompt_mode!r}")
        self._image_root = Path(images)
        self._prompt_mode: Literal["text", "bbox"] = prompt_mode
        self._transforms = transforms
        self._text_prompt_cfg = text_prompt
        self._seed = seed
        self._multiplex_cap = 16
        self._warned_truncation = False

        self._coco = _load_coco_index(annotations)
        sparse_ids, mapping, class_names = _build_category_remap(self._coco)
        self._coco_category_ids = sparse_ids
        self.coco_category_ids = sparse_ids
        self._cat_id_to_dense = mapping
        self._class_names = class_names

        kept, ann_index, dropped = _drop_crowd_only_images(self._coco)
        self._image_ids = kept
        self._ann_index = ann_index
        if dropped:
            _LOG.info(
                "custom_sam_peft.data.coco: dropped %d images (iscrowd-only) from %s",
                dropped,
                annotations,
            )
        _LOG.info(
            "custom_sam_peft.data.coco: loaded %d images, %d dense classes from %s",
            len(self._image_ids),
            len(self._class_names),
            annotations,
        )

    def __len__(self) -> int:
        return len(self._image_ids)

    def __getitem__(self, i: int) -> Example:
        import torch
        from PIL import Image

        image_id = self._image_ids[i]
        rec = self._coco.loadImgs([image_id])[0]
        img_path = self._image_root / rec["file_name"]
        with Image.open(img_path) as pil_img:
            np_img = np.asarray(pil_img.convert("RGB"))
        h, w = int(rec["height"]), int(rec["width"])

        anns = self._ann_index[image_id]
        bboxes_xyxy: list[list[float]] = []
        masks: list[np.ndarray[Any, Any]] = []
        class_labels: list[int] = []
        for ann in anns:
            x, y, bw, bh = ann["bbox"]
            bboxes_xyxy.append([float(x), float(y), float(x + bw), float(y + bh)])
            masks.append(_decode_segmentation(ann, h, w).astype(np.uint8))
            class_labels.append(self._cat_id_to_dense[int(ann["category_id"])])

        out = self._transforms(
            image=np_img,
            bboxes=bboxes_xyxy,
            masks=masks,
            class_labels=class_labels,
        )
        image_tensor: torch.Tensor = out["image"]
        out_bboxes: list[tuple[float, float, float, float]] = list(out["bboxes"])
        out_masks: list[np.ndarray[Any, Any]] = list(out["masks"])
        out_classes: list[int] = [int(c) for c in out["class_labels"]]

        from custom_sam_peft.data.base import BoxPrompts, Instance, TextPrompts

        instances: list[Instance] = []
        for box, mask_np, cls in zip(out_bboxes, out_masks, out_classes, strict=True):
            instances.append(
                Instance(
                    mask=torch.from_numpy(np.asarray(mask_np).astype(bool)),
                    class_id=int(cls),
                    box=torch.tensor(box, dtype=torch.float32),
                )
            )

        if self._prompt_mode == "text":
            present = sorted(set(out_classes))
            rng = random.Random(f"{self._seed}:{int(image_id)}")  # noqa: S311 — deterministic seeded RNG for prompt sampling, not security
            prompts_list = _build_text_prompts(
                present_dense_ids=present,
                class_names=self._class_names,
                cfg=self._text_prompt_cfg,
                rng=rng,
                image_id=int(image_id),
            )
            if len(prompts_list) > self._multiplex_cap:
                if not self._warned_truncation:
                    _LOG.warning(
                        "custom_sam_peft.data.coco: image_id=%s requested %d text prompts; "
                        "truncating to %d. Suppressing further warnings for this dataset.",
                        image_id,
                        len(prompts_list),
                        self._multiplex_cap,
                    )
                    self._warned_truncation = True
                prompts_list = prompts_list[: self._multiplex_cap]
            return Example(
                image=image_tensor,
                image_id=str(image_id),
                prompts=TextPrompts(classes=prompts_list),
                instances=instances,
            )

        # bbox mode
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
                    "custom_sam_peft.data.coco: image_id=%s requested %d box prompts; "
                    "truncating to %d. Suppressing further warnings for this dataset.",
                    image_id,
                    len(order),
                    self._multiplex_cap,
                )
                self._warned_truncation = True
            order = order[: self._multiplex_cap]
        kept_instances = [instances[k] for k in order]
        import torch as _torch

        boxes_t = (
            _torch.stack([inst.box for inst in kept_instances])
            if kept_instances
            else _torch.zeros((0, 4))
        )
        class_ids_t = _torch.tensor([inst.class_id for inst in kept_instances], dtype=_torch.int64)
        return Example(
            image=image_tensor,
            image_id=str(image_id),
            prompts=BoxPrompts(boxes=boxes_t.to(_torch.float32), class_ids=class_ids_t),
            instances=kept_instances,
        )

    @property
    def class_names(self) -> list[str]:
        return list(self._class_names)


@register("dataset", "coco")
def build_coco(
    cfg: dict[str, Any],
    *,
    model_name: str,
    pipeline: Literal["train", "eval"],
) -> Dataset:
    """Build a `COCODataset` from a validated DataConfig dict.

    The caller (trainer) chooses the split by passing the matching `train` or
    `val` sub-dict in `cfg["train"]` / `cfg["val"]`. Here `pipeline` selects the
    transform variant.
    """
    from custom_sam_peft.config.schema import AugmentationsConfig, NormalizeConfig, TextPromptConfig
    from custom_sam_peft.data.transforms import build_eval_transforms, build_train_transforms

    if pipeline not in ("train", "eval"):
        raise ValueError(f"pipeline must be 'train' or 'eval'; got {pipeline!r}")
    split_key = "train" if pipeline == "train" else "val"
    split = cfg[split_key]
    image_size = int(cfg["image_size"])
    normalize = NormalizeConfig.model_validate(cfg.get("normalize", {}))
    text_prompt = TextPromptConfig.model_validate(cfg.get("text_prompt", {}))
    if pipeline == "train":
        aug = AugmentationsConfig.model_validate(cfg.get("augmentations", {}))
        transforms = build_train_transforms(
            aug, image_size, model_name=model_name, normalize=normalize
        )
    else:
        transforms = build_eval_transforms(image_size, model_name=model_name, normalize=normalize)
    return COCODataset(
        annotations=split["annotations"],
        images=split["images"],
        prompt_mode=cfg["prompt_mode"],
        transforms=transforms,
        text_prompt=text_prompt,
    )
