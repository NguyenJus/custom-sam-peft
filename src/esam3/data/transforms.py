"""Image augmentation + normalization pipelines (Albumentations).

Public API:
  - resolve_normalization(model_name, fallback) -> (mean, std)
  - build_eval_transforms(image_size, *, model_name, normalize) -> A.Compose
  - build_train_transforms(aug_cfg, image_size, *, model_name, normalize) -> A.Compose
"""

from __future__ import annotations

import logging

from esam3.config.schema import AugmentationsConfig, NormalizeConfig

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
        _LOG.info(
            "Using image_mean/image_std from AutoImageProcessor for %r.", model_name
        )
        return mean, std


def build_eval_transforms(
    image_size: int,
    *,
    model_name: str,
    normalize: NormalizeConfig,
) -> object:
    raise NotImplementedError("filled in by Task 9")


def build_train_transforms(
    aug_cfg: AugmentationsConfig,
    image_size: int,
    *,
    model_name: str,
    normalize: NormalizeConfig,
) -> object:
    raise NotImplementedError("filled in by Task 10")
