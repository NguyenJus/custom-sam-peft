"""Image augmentation + normalization pipelines (Albumentations).

Public API:
  - resolve_normalization(model_name, fallback) -> (mean, std)
  - build_eval_transforms(image_size, *, model_name, normalize) -> A.Compose
  - build_train_transforms(aug_cfg, image_size, *, model_name, normalize) -> A.Compose
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from custom_sam_peft.config.schema import AugmentationsConfig, NormalizeConfig

if TYPE_CHECKING:
    import albumentations as A

_LOG = logging.getLogger(__name__)


def resolve_normalization(
    model_name: str, fallback: NormalizeConfig
) -> tuple[list[float], list[float]]:
    """Try `AutoImageProcessor.from_pretrained(model_name, local_files_only=True)`.

    On success, read `image_mean` / `image_std`. On any of `(OSError, AttributeError,
    ValueError)`, return the fallback's (mean, std). Emits exactly one INFO log line.
    """
    from transformers import AutoImageProcessor

    try:
        proc = AutoImageProcessor.from_pretrained(model_name, local_files_only=True)  # type: ignore[no-untyped-call]
        mean = list(proc.image_mean)
        std = list(proc.image_std)
    except (OSError, AttributeError, ValueError):
        _LOG.info(
            "AutoImageProcessor cache miss for %r; falling back to NormalizeConfig "
            "(mean=%s, std=%s).",
            model_name,
            fallback.mean,
            fallback.std,
        )
        return list(fallback.mean), list(fallback.std)
    else:
        _LOG.info("Using image_mean/image_std from AutoImageProcessor for %r.", model_name)
        return mean, std


def build_eval_transforms(
    image_size: int,
    *,
    model_name: str,
    normalize: NormalizeConfig,
) -> A.Compose:
    """Deterministic eval pipeline: longest-edge resize -> top-left pad -> normalize -> ToTensor."""
    import albumentations as A
    import cv2
    from albumentations.pytorch import ToTensorV2

    mean, std = resolve_normalization(model_name, normalize)
    return A.Compose(
        [
            A.LongestMaxSize(max_size=image_size, interpolation=cv2.INTER_LINEAR),
            A.PadIfNeeded(
                min_height=image_size,
                min_width=image_size,
                border_mode=cv2.BORDER_CONSTANT,
                fill=0,
                fill_mask=0,
                position="top_left",
            ),
            A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["class_labels"],
            min_visibility=0.0,
            min_area=0.0,
        ),
    )


def build_train_transforms(
    aug_cfg: AugmentationsConfig,
    image_size: int,
    *,
    model_name: str,
    normalize: NormalizeConfig,
) -> A.Compose:
    """Train pipeline: resize+pad geometry + optional hflip + color jitter + normalize."""
    import albumentations as A
    import cv2
    from albumentations.pytorch import ToTensorV2

    mean, std = resolve_normalization(model_name, normalize)
    steps: list[object] = [
        A.LongestMaxSize(max_size=image_size, interpolation=cv2.INTER_LINEAR),
        A.PadIfNeeded(
            min_height=image_size,
            min_width=image_size,
            border_mode=cv2.BORDER_CONSTANT,
            fill=0,
            fill_mask=0,
            position="top_left",
        ),
    ]
    if aug_cfg.hflip:
        steps.append(A.HorizontalFlip(p=0.5))
    steps.append(
        A.ColorJitter(
            brightness=aug_cfg.color_jitter,
            contrast=aug_cfg.color_jitter,
            saturation=aug_cfg.color_jitter,
            hue=aug_cfg.color_jitter * 0.5,
            p=0.5,
        )
    )
    steps.append(A.Normalize(mean=mean, std=std, max_pixel_value=255.0))
    steps.append(ToTensorV2())
    return A.Compose(
        steps,
        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["class_labels"],
            min_visibility=0.0,
            min_area=0.0,
        ),
    )
