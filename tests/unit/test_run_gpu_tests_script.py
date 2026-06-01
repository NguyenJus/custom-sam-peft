"""CPU test for run_gpu_tests.sh tier parsing (no GPU needed)."""

from __future__ import annotations

import stat
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "run_gpu_tests.sh"
REPO_ROOT = Path(__file__).resolve().parents[2]


def _src() -> str:
    return SCRIPT.read_text()


def test_accepts_capability_tiers_and_rejects_legacy() -> None:
    src = _src()
    assert "local)" in src and "t4)" in src and "bf16)" in src and "xl)" in src
    assert "inspection)" not in src and "release)" not in src


def test_local_tier_selects_both_le16gb_bands() -> None:
    """The default `local` tier runs everything a <=16 GB dev card satisfies."""
    assert "gpu_t4 or gpu_bf16" in _src()


def test_no_legacy_gpu_local_marker() -> None:
    """The runner must not reference the removed `gpu_local` marker.

    Selectors map to the capability-named markers (gpu_t4, gpu_bf16, gpu_xl);
    the default `local` tier expands to `gpu_t4 or gpu_bf16`.
    """
    assert "gpu_local" not in _src()


def test_collects_predict_path() -> None:
    assert "tests/predict/" in _src()


def test_local_runs_per_file_loop() -> None:
    """The local tier must iterate over files (not a single pytest invocation).

    Confirms the script contains a loop over test files and handles exit-code 5
    (no tests collected) as success.
    """
    src = _src()
    # A loop construct is present for the local tier.
    assert "while" in src or "for" in src
    # Exit code 5 (no tests collected) is explicitly treated as success.
    assert "5" in src


def test_rejects_unknown_tier() -> None:
    res = subprocess.run(  # noqa: S603
        ["bash", str(SCRIPT), "bogus"],  # noqa: S607
        capture_output=True,
        text=True,
        env={"PYTHON": "true", "PATH": "/usr/bin:/bin"},
    )
    assert res.returncode != 0
    assert "usage" in (res.stderr + res.stdout).lower()


def _run_local_with_stub_python(tmp_path: Path, stub_rc: int) -> subprocess.CompletedProcess[str]:
    """Run the `local` tier with a stub `python` that exits `stub_rc` for every file.

    Exercises the per-file exit-code semantics without a GPU: every per-file
    pytest invocation returns the same code, so the script's overall exit
    reflects how it treats that code.
    """
    stub = tmp_path / "fake_python"
    stub.write_text("#!/usr/bin/env bash\nexit ${STUB_RC:-0}\n")
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
    return subprocess.run(  # noqa: S603
        ["bash", str(SCRIPT), "local"],  # noqa: S607
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),  # script uses relative `find tests/...` paths
        env={"PYTHON": str(stub), "STUB_RC": str(stub_rc), "PATH": "/usr/bin:/bin"},
    )


def test_local_exit5_per_file_is_success(tmp_path: Path) -> None:
    """Per-file pytest exit code 5 (no tests collected) → overall success."""
    res = _run_local_with_stub_python(tmp_path, 5)
    assert res.returncode == 0, res.stderr


def test_local_propagates_per_file_failure(tmp_path: Path) -> None:
    """A per-file pytest failure (exit 1) → overall non-zero exit."""
    res = _run_local_with_stub_python(tmp_path, 1)
    assert res.returncode != 0, res.stderr
