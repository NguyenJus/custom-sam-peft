"""Cheap-to-run environment diagnostics. No heavy/optional imports."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import platform
import sys
from dataclasses import dataclass, field
from pathlib import Path


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
    issues: list[str] = field(default_factory=list)


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


def _default_weights_path() -> Path:
    from esam3.config.schema import ModelConfig

    m = ModelConfig()
    return Path(m.local_dir or "") / m.checkpoint_file


def run_doctor(*, weights_path: Path | None = None) -> DoctorReport:
    """Cheap-to-run environment audit."""
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
        issues=issues,
    )
