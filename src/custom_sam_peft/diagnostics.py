"""Cheap-to-run environment diagnostics. No heavy/optional imports."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import os
import platform
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import huggingface_hub


@dataclass(frozen=True)
class GpuInfo:
    index: int
    name: str
    capability: tuple[int, int]
    total_mib: int
    free_mib: int


@dataclass(frozen=True)
class WeightsInfo:
    path: Path
    exists: bool
    size_bytes: int | None


@dataclass(frozen=True)
class HuggingFaceAuthInfo:
    """Local-only HF token status. Mirrors custom_sam_peft.utils.huggingface.resolve_hf_token's
    probe order but reports the *source*, not the token value. Never hits the network.
    """

    token_source: Literal["env", "cache", "none"]
    has_token: bool


@dataclass(frozen=True)
class DataReport:
    """Validation source plan for the given config (no dataset materialization).

    Populated only when `run_doctor(config_path=...)` is called.

    Spec: docs/superpowers/specs/2026-05-22-data-no-val-auto-split-design.md §7.7.
    """

    val_mode: Literal["explicit", "auto_split", "none"]
    val_path: str | None
    val_split_fraction: float | None
    val_split_seed: int | None


@dataclass(frozen=True)
class DatasetResolution:
    format: str
    train_total: int
    train_kept: int
    val_total: int
    val_kept: int
    limit_strategy: str
    limit_seed: int
    limit_train: int | float | None
    limit_val: int | float | None


@dataclass(frozen=True)
class DoctorReport:
    python_version: str
    platform: str
    torch_version: str
    cuda_build: str | None
    cuda_available: bool
    gpus: list[GpuInfo]
    optional_deps: dict[str, str | None]
    core_versions: dict[str, str]
    sam3_weights: WeightsInfo
    hf_auth: HuggingFaceAuthInfo
    dataset: DatasetResolution | None = None
    issues: list[str] = field(default_factory=list)
    data: DataReport | None = None


_OPTIONAL = ("bitsandbytes", "wandb", "tensorboard")
_CORE = ("peft", "transformers", "sam3", "datasets", "pydantic", "typer")


def _version_or_none(name: str) -> str | None:
    if importlib.util.find_spec(name) is None:
        return None
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _required_version(name: str) -> str:
    return importlib.metadata.version(name)


def _gpus() -> list[GpuInfo]:
    import torch

    if not torch.cuda.is_available():
        return []
    out: list[GpuInfo] = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        free, total = torch.cuda.mem_get_info(i)
        out.append(
            GpuInfo(
                index=i,
                name=props.name,
                capability=(props.major, props.minor),
                total_mib=total // (1024 * 1024),
                free_mib=free // (1024 * 1024),
            )
        )
    return out


def _hf_auth_info() -> HuggingFaceAuthInfo:
    """Probe local HF token sources. Mirrors resolve_hf_token's order but
    reports the source, not the token value. Never hits the network.

    Deliberately NOT delegated to ``resolve_hf_token``: that returns the token
    string; we want to discriminate ``env`` vs ``cache``, which means probing
    each source explicitly.
    """
    if os.environ.get("HF_TOKEN"):
        return HuggingFaceAuthInfo(token_source="env", has_token=True)  # noqa: S106
    if huggingface_hub.get_token():
        return HuggingFaceAuthInfo(token_source="cache", has_token=True)  # noqa: S106
    return HuggingFaceAuthInfo(token_source="none", has_token=False)  # noqa: S106


def _default_weights_path() -> Path:
    from custom_sam_peft.config.schema import ModelConfig

    m = ModelConfig()
    return Path(m.local_dir or "") / m.checkpoint_file


def _build_dataset_for_doctor(config_path: Path, issues: list[str]) -> DatasetResolution | None:
    """Load config + build train/val datasets. Returns None and appends to issues on any error.

    Failure modes (all result in return None, exit code 0):
      - Config file not found or bad YAML  → appends "couldn't load config: <msg>"
      - Schema validation error             → appends "couldn't load config: <msg>"
      - Dataset build error                 → appends "couldn't build train/val dataset: <msg>"
    """
    from custom_sam_peft.config.loader import ConfigError, load_config
    from custom_sam_peft.data.subset import SubsetDataset
    from custom_sam_peft.train.runner import _build_dataset

    try:
        cfg = load_config(config_path)
    except (ConfigError, Exception) as e:
        issues.append(f"couldn't load config {config_path}: {e}")
        return None

    try:
        train_ds = _build_dataset(cfg, "train")
        val_ds = _build_dataset(cfg, "eval")
    except Exception as e:
        issues.append(f"couldn't build train/val dataset: {e}")
        return None

    train_total = len(train_ds._inner) if isinstance(train_ds, SubsetDataset) else len(train_ds)
    val_total = len(val_ds._inner) if isinstance(val_ds, SubsetDataset) else len(val_ds)
    lim = cfg.data.limit
    return DatasetResolution(
        format=cfg.data.format,
        train_total=train_total,
        train_kept=len(train_ds),
        val_total=val_total,
        val_kept=len(val_ds),
        limit_strategy=lim.strategy,
        limit_seed=lim.seed,
        limit_train=lim.train,
        limit_val=lim.val,
    )


def run_doctor(
    *,
    weights_path: Path | None = None,
    config_path: Path | None = None,
) -> DoctorReport:
    """Cheap-to-run environment audit.

    config_path is optional and heavy: loads the YAML, validates the config,
    builds train and val datasets (may trigger pycocotools or datasets.load_dataset),
    and also extracts the resolved val-source plan (mode, fraction, seed).
    The existing no-config path remains cheap and network-free.
    """
    import torch

    issues: list[str] = []

    if sys.version_info < (3, 12):  # noqa: UP036
        issues.append(f"python {sys.version_info.major}.{sys.version_info.minor} < 3.12")

    cuda_available = torch.cuda.is_available()
    if not cuda_available:
        issues.append("CUDA not available; training will run on CPU")

    optional = {name: _version_or_none(name) for name in _OPTIONAL}
    core = {name: _required_version(name) for name in _CORE}

    wp = weights_path or _default_weights_path()
    weights = WeightsInfo(
        path=wp,
        exists=wp.is_file(),
        size_bytes=(wp.stat().st_size if wp.is_file() else None),
    )
    if not weights.exists:
        issues.append(f"SAM 3.1 weights not found at {wp}")

    hf_auth = _hf_auth_info()
    if hf_auth.token_source == "none":  # noqa: S105
        issues.append(
            "no HuggingFace token found; gated repos like facebook/sam3.1 "
            "will not download (set HF_TOKEN or run `huggingface-cli login`)"
        )

    data: DataReport | None = None
    dataset_resolution: DatasetResolution | None = None
    if config_path is not None:
        from custom_sam_peft.config.loader import load_config

        try:
            cfg = load_config(config_path)
        except Exception:
            cfg = None
        if cfg is not None:
            if cfg.data.val_split is not None:
                seed = (
                    cfg.data.val_split.seed if cfg.data.val_split.seed is not None else cfg.run.seed
                )
                data = DataReport(
                    val_mode="auto_split",
                    val_path=None,
                    val_split_fraction=cfg.data.val_split.fraction,
                    val_split_seed=seed,
                )
            elif cfg.data.val is not None:
                data = DataReport(
                    val_mode="explicit",
                    val_path=cfg.data.val.annotations,
                    val_split_fraction=None,
                    val_split_seed=None,
                )
            else:
                data = DataReport(
                    val_mode="none",
                    val_path=None,
                    val_split_fraction=None,
                    val_split_seed=None,
                )
        dataset_resolution = _build_dataset_for_doctor(config_path, issues)

    return DoctorReport(
        python_version=sys.version.split()[0],
        platform=platform.platform(),
        torch_version=torch.__version__,
        cuda_build=torch.version.cuda,
        cuda_available=cuda_available,
        gpus=_gpus(),
        optional_deps=optional,
        core_versions=core,
        sam3_weights=weights,
        hf_auth=hf_auth,
        dataset=dataset_resolution,
        issues=issues,
        data=data,
    )
