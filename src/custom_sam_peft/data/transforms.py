"""Image augmentation + normalization pipelines (Albumentations).

Public API:
  - resolve_normalization(model_name, fallback) -> (mean, std)
  - resolve_normalization_with_path(model_name, fallback) -> (mean, std, path)
  - build_eval_transforms(image_size, *, model_name, normalize) -> A.Compose
  - build_train_transforms(aug_cfg, image_size, *, model_name, normalize) -> A.Compose
  - StainJitter: HED-space stain jitter Albumentations transform
"""

from __future__ import annotations

import logging
from typing import Literal

import albumentations as A
import numpy as np
from numpy.typing import NDArray

from custom_sam_peft.config.schema import AugmentationsConfig, NormalizeConfig

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

# ---------------------------------------------------------------------------
# StainJitter — HED-space color deconvolution for H&E histopathology
# (Ruifrok & Johnston 2001 / Tellez et al. 2018). Image-only Albumentations
# transform; masks/bboxes/keypoints pass through unchanged.
# ---------------------------------------------------------------------------

# Ruifrok & Johnston 2001 HED basis vectors (rows = stains: H, E, DAB).
_HED_FROM_RGB_MATRIX: NDArray[np.float32] = np.array(
    [
        [0.65, 0.70, 0.29],
        [0.07, 0.99, 0.11],
        [0.27, 0.57, 0.78],
    ],
    dtype=np.float32,
)
_HED_FROM_RGB_INV: NDArray[np.float32] = np.linalg.inv(_HED_FROM_RGB_MATRIX).astype(np.float32)

# Magnitude → Albumentations parameter projection constants — spec §8.1.
_GAUSS_NOISE_MAX_VAR: float = 0.05
_GAUSS_BLUR_MAX_SIGMA: float = 3.0


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


NormalizationPath = Literal["processor", "table-fallback", "config-fallback"]


def resolve_normalization_with_path(
    model_name: str, fallback: NormalizeConfig
) -> tuple[list[float], list[float], NormalizationPath]:
    """Three-step resolution of (mean, std) plus the path that fired.

    Path codes:
      - "processor":       loaded from AutoImageProcessor
      - "table-fallback":  processor unavailable, model in KNOWN_PROCESSOR_STATS
      - "config-fallback": processor unavailable, no table entry, user fallback

    Logging is unchanged from the legacy ``resolve_normalization``: WARN on
    fallback paths, WARN on table-vs-processor divergence, INFO on the happy
    path.
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
        if table_entry is not None:
            table_mean, table_std = table_entry
            _LOG.warning(
                "AutoImageProcessor unavailable for %r; using known-good stats "
                "(mean=%s, std=%s). Populate the HF cache to silence this warning.",
                model_name,
                table_mean,
                table_std,
            )
            return list(table_mean), list(table_std), "table-fallback"
        _LOG.warning(
            "AutoImageProcessor unavailable for %r AND no known-good entry registered; "
            "using NormalizeConfig fallback (mean=%s, std=%s). Verify these are correct "
            "for this backbone.",
            model_name,
            fallback.mean,
            fallback.std,
        )
        return list(fallback.mean), list(fallback.std), "config-fallback"

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
    return mean, std, "processor"


def resolve_normalization(
    model_name: str, fallback: NormalizeConfig
) -> tuple[list[float], list[float]]:
    """2-tuple wrapper kept for build_eval_transforms / build_train_transforms.

    Equivalent to dropping the path code from
    :func:`resolve_normalization_with_path`.
    """
    mean, std, _path = resolve_normalization_with_path(model_name, fallback)
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
    """Train pipeline: resolved presets → ordered Albumentations step list.

    See spec §8 for the canonical step ordering. The Albumentations objects
    appear in the compose iff the corresponding knob is enabled / > 0;
    knob = 0/False omits the step entirely (not p=0).
    """
    import albumentations as A
    import cv2
    from albumentations.pytorch import ToTensorV2

    from custom_sam_peft.data.aug_presets import resolve

    resolved = resolve(aug_cfg)
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
    if resolved.hflip:
        steps.append(A.HorizontalFlip(p=0.5))
    if resolved.vflip:
        steps.append(A.VerticalFlip(p=0.5))
    if resolved.rotate90:
        steps.append(A.RandomRotate90(p=0.5))
    if resolved.rotate_arbitrary > 0.0:
        steps.append(
            A.Affine(
                rotate=(-resolved.rotate_arbitrary, resolved.rotate_arbitrary),
                p=0.5,
                fit_output=False,
                fill=0,
                fill_mask=0,
            )
        )
    if resolved.gauss_noise > 0.0:
        # Albumentations 2.x replaced var_limit (variance, value-space) with std_range
        # (std as a fraction of max_pixel_value=255 here, since GaussNoise runs pre-Normalize).
        # We preserve the spec's per-knob scaling intent (magnitude * _GAUSS_NOISE_MAX_VAR) but
        # the numeric semantics differ: a knob of 1 caps std at ~12.75 pixel units, not the
        # sub-LSB variance the literal 1.x reading of the spec would produce. This is the
        # behaviorally meaningful interpretation.
        steps.append(
            A.GaussNoise(
                std_range=(0.0, resolved.gauss_noise * _GAUSS_NOISE_MAX_VAR),
                p=0.5,
            )
        )
    if resolved.blur > 0.0:
        steps.append(
            A.GaussianBlur(
                blur_limit=(3, 7),
                sigma_limit=(0.0, resolved.blur * _GAUSS_BLUR_MAX_SIGMA),
                p=0.5,
            )
        )
    if resolved.color_jitter > 0.0:
        v = resolved.color_jitter
        steps.append(
            A.ColorJitter(
                brightness=v,
                contrast=v,
                saturation=v,
                hue=v * 0.5,
                p=0.5,
            )
        )
    if resolved.stain_jitter > 0.0:
        steps.append(StainJitter(sigma=resolved.stain_jitter, p=0.5))
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


class StainJitter(A.ImageOnlyTransform):  # type: ignore[misc]
    """HED-space stain jitter for H&E histopathology images.

    Image-only — masks, bboxes, keypoints pass through unchanged. Implements
    the Tellez et al. (2018) / Ruifrok & Johnston (2001) color deconvolution:
    RGB → optical density → HED basis → per-channel affine perturbation →
    back to RGB.

    Identity at sigma=0 (the implementation short-circuits).

    Note: This class directly subclasses ``albumentations.ImageOnlyTransform``
    at module import time. The plan described a lazy ``__new__`` mixin pattern
    to defer the Albumentations import, but that pattern raises
    ``TypeError: __bases__ assignment: 'ImageOnlyTransform' deallocator differs
    from 'object'`` under Albumentations 2.0.8 (CPython's restriction on
    changing __bases__ for extension-type subclasses). Direct subclassing is
    used instead, which also means ``import albumentations`` is triggered
    whenever this module is imported.
    """

    def __init__(self, sigma: float = 0.0, p: float = 0.5) -> None:
        super().__init__(p=p)
        if sigma < 0:
            raise ValueError(f"StainJitter sigma must be >= 0, got {sigma}")
        self.sigma = float(sigma)

    def apply(self, img: NDArray[np.uint8], **params: object) -> NDArray[np.uint8]:
        """Apply HED-space stain perturbation to img (uint8 RGB, HWC)."""
        if self.sigma == 0.0:
            return img
        od = -np.log10((img.astype(np.float32) + 1.0) / 256.0)
        hed = od @ _HED_FROM_RGB_INV
        alpha = np.random.uniform(-self.sigma, self.sigma, size=3).astype(np.float32)
        beta = np.random.uniform(-self.sigma, self.sigma, size=3).astype(np.float32)
        hed = hed * (1.0 + alpha) + beta
        od_back = hed @ _HED_FROM_RGB_MATRIX
        out = 256.0 * np.power(10.0, -od_back) - 1.0
        return np.clip(out, 0.0, 255.0).astype(np.uint8)  # type: ignore[no-any-return]

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return ("sigma",)
