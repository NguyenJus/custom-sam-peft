"""COCO instance-JSON dataset adapter.

Backed by `pycocotools.coco.COCO` for index lookups and `pycocotools.mask` for
polygon/RLE decode. Sparse COCO category ids are remapped to a dense 0..C-1
namespace; the original sparse ids are preserved on `coco_category_ids` for
eval-time round-tripping.
"""

from __future__ import annotations

import logging
import random
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pycocotools import mask as coco_mask
from pycocotools.coco import COCO

from custom_sam_peft._registry import register
from custom_sam_peft.config.schema import TextPromptConfig
from custom_sam_peft.data.base import Dataset, Example
from custom_sam_peft.data.io import read_image
from custom_sam_peft.data.tiling import Window, iter_windows, tiling_engaged

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
        *,
        transforms: Any,
        text_prompt: TextPromptConfig,
        seed: int = 0,
        image_ids: Iterable[int] | None = None,
        channels: int = 3,
    ) -> None:
        self._image_root = Path(images)
        self._channels = channels
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
        if image_ids is not None:
            requested = {int(x) for x in image_ids}
            kept_set = set(kept)
            missing = requested - kept_set
            if missing:
                first_few = sorted(missing)[:10]
                raise ValueError(
                    f"COCODataset: {len(missing)} image_ids requested but not present "
                    f"(or dropped as iscrowd-only): {first_few}"
                    f"{'…' if len(missing) > 10 else ''}"
                )
            self._image_ids = [i for i in kept if i in requested]
        else:
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

        # Pre-enumerate (image_id, window) samples (spec §5.3 / C6). A raster that
        # triggers tiling_engaged expands into one sample per iter_windows window;
        # otherwise a single full-image window. This is the POST-EXPANSION index
        # space that __len__ / _fetch_raw / __getitem__ and every length consumer
        # (data.limit subset cap, no-val auto-split, eval per-example alignment)
        # operate over.
        self._samples: list[tuple[int, Window]] = []
        for img_id in self._image_ids:
            h, w = self._image_hw(img_id)
            windows = iter_windows(h, w) if tiling_engaged(h, w) else [Window(0, 0, h, w)]
            for win in windows:
                self._samples.append((img_id, win))

        # Eager per-sample class label sets for stratified subset sampling.
        # Indexed over self._samples (NOT _image_ids) so resolve_subset_indices
        # sees labels aligned with the expanded len(dataset).
        self.image_class_labels: list[frozenset[int]] = [
            frozenset(
                self._cat_id_to_dense[int(ann["category_id"])]
                for ann in self._ann_index.get(img_id, [])
            )
            for img_id, _win in self._samples
        ]

    def _image_hw(self, image_id: int) -> tuple[int, int]:
        """Return ``(height, width)`` for *image_id* without decoding pixels.

        Prefers the COCO record's ``height``/``width`` fields; falls back to a
        header-only PIL read when either is missing (full-pixel decode at
        construction is deliberately avoided — this box is I/O-fragile).
        """
        rec = self._coco.loadImgs([image_id])[0]
        h, w = rec.get("height"), rec.get("width")
        if h and w:
            return int(h), int(w)
        from PIL import Image as PILImage

        with PILImage.open(self._image_root / rec["file_name"]) as im:
            pw, ph = im.size
        return int(ph), int(pw)

    def __len__(self) -> int:
        return len(self._samples)

    # ------------------------------------------------------------------
    # Internal pipeline helpers
    # ------------------------------------------------------------------

    def _fetch_raw(self, i: int) -> tuple[int, dict[str, Any], list[dict[str, Any]], Window]:
        """Return ``(image_id, img_record, annotations, window)`` for index *i*.

        *i* indexes the expanded ``self._samples`` space (one entry per tile
        window), not the raw image list.
        """
        image_id, window = self._samples[i]
        rec = self._coco.loadImgs([image_id])[0]
        anns = self._ann_index[image_id]
        return image_id, rec, anns, window

    def _decode_image(
        self, raw: tuple[int, dict[str, Any], list[dict[str, Any]], Window]
    ) -> np.ndarray[Any, Any]:
        """Load the image, decode to (H, W, C), and crop to the sample's window."""
        _image_id, rec, _anns, win = raw
        img_path = self._image_root / rec["file_name"]
        np_img = read_image(img_path, self._channels)
        return np_img[win.y0 : win.y0 + win.h, win.x0 : win.x0 + win.w]

    def _decode_targets(
        self,
        raw: tuple[int, dict[str, Any], list[dict[str, Any]], Window],
    ) -> tuple[list[list[float]], list[np.ndarray[Any, Any]], list[int]]:
        """Decode bounding boxes, masks, and class labels from *raw*, clipped to
        the sample's window (offset by ``-x0, -y0``; full masks cropped).

        Boxes are intersected with the window; instances whose clipped box has
        non-positive width or height are dropped. Surviving masks are cropped to
        the window crop. An empty post-clip window is a VALID negative — it simply
        yields zero instances. Albumentations later applies the same clip via
        ``BboxParams(min_area=0.0, min_visibility=0.0)``.

        Returns ``(bboxes_xyxy, masks, class_labels)`` in WINDOW-LOCAL coords.
        """
        image_id, _rec, anns, win = raw
        h, w = self._image_hw(image_id)
        bboxes_xyxy: list[list[float]] = []
        masks: list[np.ndarray[Any, Any]] = []
        class_labels: list[int] = []
        for ann in anns:
            x, y, bw, bh = ann["bbox"]
            # Intersect the full-image box with the window, then offset to local.
            x0 = max(float(x), float(win.x0))
            y0 = max(float(y), float(win.y0))
            x1 = min(float(x + bw), float(win.x0 + win.w))
            y1 = min(float(y + bh), float(win.y0 + win.h))
            if x1 <= x0 or y1 <= y0:
                continue  # box does not intersect this window — skip (valid negative)
            full_mask = _decode_segmentation(ann, h, w).astype(np.uint8)
            mask_crop = full_mask[win.y0 : win.y0 + win.h, win.x0 : win.x0 + win.w]
            bboxes_xyxy.append([x0 - win.x0, y0 - win.y0, x1 - win.x0, y1 - win.y0])
            masks.append(mask_crop)
            class_labels.append(self._cat_id_to_dense[int(ann["category_id"])])
        return bboxes_xyxy, masks, class_labels

    def _apply_transforms(
        self,
        np_img: np.ndarray[Any, Any],
        bboxes_xyxy: list[list[float]],
        masks: list[np.ndarray[Any, Any]],
        class_labels: list[int],
    ) -> tuple[Any, list[Any], list[Any], list[int]]:
        """Run the configured transform pipeline.

        Returns ``(image_tensor, out_bboxes, out_masks, out_classes)``.
        """
        out = self._transforms(
            image=np_img,
            bboxes=bboxes_xyxy,
            masks=masks,
            class_labels=class_labels,
            instance_idx=list(range(len(masks))),
        )
        image_tensor: Any = out["image"]
        out_bboxes: list[Any] = list(out["bboxes"])
        # Re-select masks by the original indices of the surviving bboxes so that
        # bboxes, masks, and class_labels stay parallel even when Albumentations
        # drops an out-of-frame bbox (which it also removes from class_labels but
        # not from the masks target, which is processed independently).
        out_masks: list[Any] = [out["masks"][int(idx)] for idx in out["instance_idx"]]
        out_classes: list[int] = [int(c) for c in out["class_labels"]]
        return image_tensor, out_bboxes, out_masks, out_classes

    def _pack_example(
        self,
        raw: tuple[int, dict[str, Any], list[dict[str, Any]], Window],
        image_tensor: Any,
        out_bboxes: list[Any],
        out_masks: list[Any],
        out_classes: list[int],
    ) -> Example:
        """Assemble `Instance` objects and return the final `Example`."""
        import torch

        from custom_sam_peft.data.base import Instance, TextPrompts

        image_id, _rec, _anns, _win = raw

        instances: list[Instance] = []
        for box, mask_np, cls in zip(out_bboxes, out_masks, out_classes, strict=True):
            instances.append(
                Instance(
                    mask=torch.from_numpy(np.asarray(mask_np).astype(bool)),
                    class_id=int(cls),
                    box=torch.tensor(box, dtype=torch.float32),
                )
            )

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

    def __getitem__(self, i: int) -> Example:
        raw = self._fetch_raw(i)
        np_img = self._decode_image(raw)
        bboxes_xyxy, masks, class_labels = self._decode_targets(raw)
        image_tensor, out_bboxes, out_masks, out_classes = self._apply_transforms(
            np_img, bboxes_xyxy, masks, class_labels
        )
        return self._pack_example(raw, image_tensor, out_bboxes, out_masks, out_classes)

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
    resolved = (cfg.get("_resolved_image_ids") or {}).get(pipeline)
    if pipeline == "eval" and cfg.get("val") is None and resolved is not None:
        split = cfg["train"]
    else:
        split_key = "train" if pipeline == "train" else "val"
        split = cfg[split_key]
    from custom_sam_peft.models.sam3 import SAM3_IMAGE_SIZE

    normalize = NormalizeConfig.model_validate(cfg.get("normalize", {}))
    text_prompt = TextPromptConfig.model_validate(cfg.get("text_prompt", {}))
    channel_semantics: str = str(cfg.get("channel_semantics", "rgb"))
    if pipeline == "train":
        aug = AugmentationsConfig.model_validate(cfg.get("augmentations", {}))
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
    return COCODataset(
        annotations=split["annotations"],
        images=split["images"],
        transforms=transforms,
        text_prompt=text_prompt,
        image_ids=[int(s) for s in resolved] if resolved is not None else None,
        channels=int(cfg.get("channels", 3)),
    )
