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

# Known-good (mean, std) per HF model name. Used as the offline fallback
# AND as a divergence sentinel against AutoImageProcessor on path 1.
#
# facebook/sam3.1: ImageNet stats. This matches what
# AutoImageProcessor.from_pretrained("facebook/sam3.1").image_mean/image_std
# returns; consistent with SAM/SAM2-class processors. Ratified by the
# 2026-05-21 config-defaults audit (supersedes the 2026-05-16 model-loading
# spec's [0.5, 0.5, 0.5] claim).
KNOWN_PROCESSOR_STATS: dict[str, tuple[list[float], list[float]]] = {
    "facebook/sam3.1": ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
}

# Element-wise absolute tolerance for table-vs-processor divergence detection
# on path 1. Loose enough to absorb float-serialization noise; tight enough
# to catch a real change (e.g. [0.5, 0.5, 0.5] diverges by >=0.014 per channel).
_STATS_DIVERGENCE_ATOL = 1e-3


def _stats_diverge(
    loaded: tuple[list[float], list[float]],
    table: tuple[list[float], list[float]],
) -> bool:
    """True if loaded and table differ on any channel of either vector beyond tolerance."""
    loaded_mean, loaded_std = loaded
    table_mean, table_std = table
    if len(loaded_mean) != len(table_mean) or len(loaded_std) != len(table_std):
        return True
    for lm, tm in zip(loaded_mean, table_mean, strict=True):
        if abs(lm - tm) > _STATS_DIVERGENCE_ATOL:
            return True
    for ls, ts in zip(loaded_std, table_std, strict=True):
        if abs(ls - ts) > _STATS_DIVERGENCE_ATOL:
            return True
    return False


def resolve_normalization(
    model_name: str, fallback: NormalizeConfig
) -> tuple[list[float], list[float]]:
    """Three-step resolution of (mean, std) for image normalization.

    1. Try ``transformers.AutoImageProcessor.from_pretrained(model_name,
       local_files_only=True)``. On success, read ``image_mean`` / ``image_std``.
       Before returning, look up ``model_name`` in :data:`KNOWN_PROCESSOR_STATS`.
       If the model is in the table and the loaded stats diverge beyond
       ``_STATS_DIVERGENCE_ATOL``, emit a WARNING naming both vectors.
       Otherwise emit INFO.
    2. On ``(OSError, AttributeError, ValueError)``, look up ``model_name`` in
       the table. If present, return the table values and emit WARNING.
    3. Otherwise (processor unavailable AND no table entry), return the user's
       ``fallback`` values and emit WARNING.

    Quality-regressing fallbacks must be loud; only path 1's happy path is INFO.
    """
    import transformers

    table_entry = KNOWN_PROCESSOR_STATS.get(model_name)

    try:
        proc = transformers.AutoImageProcessor.from_pretrained(  # type: ignore[no-untyped-call]
            model_name, local_files_only=True
        )
        mean = list(proc.image_mean)
        std = list(proc.image_std)
    except (OSError, AttributeError, ValueError):
        # Path 2 / Path 3
        if table_entry is not None:
            table_mean, table_std = table_entry
            _LOG.warning(
                "AutoImageProcessor unavailable for %r; using known-good stats "
                "(mean=%s, std=%s). Populate the HF cache to silence this warning.",
                model_name,
                table_mean,
                table_std,
            )
            return list(table_mean), list(table_std)
        _LOG.warning(
            "AutoImageProcessor unavailable for %r AND no known-good entry registered; "
            "using NormalizeConfig fallback (mean=%s, std=%s). Verify these are correct "
            "for this backbone.",
            model_name,
            fallback.mean,
            fallback.std,
        )
        return list(fallback.mean), list(fallback.std)

    # Path 1: processor loaded.
    if table_entry is not None and _stats_diverge((mean, std), table_entry):
        table_mean, table_std = table_entry
        _LOG.warning(
            "AutoImageProcessor for %r returned stats (mean=%s, std=%s) that diverge "
            "from KNOWN_PROCESSOR_STATS (mean=%s, std=%s) beyond tolerance %g. "
            "Using processor values; update the table if this divergence is expected.",
            model_name,
            mean,
            std,
            table_mean,
            table_std,
            _STATS_DIVERGENCE_ATOL,
        )
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
