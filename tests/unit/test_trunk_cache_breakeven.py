"""CPU-only smoke tests for the trunk-feature-cache break-even calculator.

Tests the pure-arithmetic helpers in scripts/spike_trunk_cache_feasibility.py.
No GPU, no real model required.

Run with::

    uv run pytest tests/unit/test_trunk_cache_breakeven.py -o "addopts=" -p no:cacheprovider
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Import the spike script as a module without executing main().
# ---------------------------------------------------------------------------

_SPIKE_PATH = Path(__file__).parents[2] / "scripts" / "spike_trunk_cache_feasibility.py"


def _load_spike():
    """Dynamically load the spike script as a module."""
    spec = importlib.util.spec_from_file_location("spike_trunk_cache_feasibility", _SPIKE_PATH)
    if spec is None or spec.loader is None:
        pytest.fail(f"Cannot load spike script from {_SPIKE_PATH}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


@pytest.fixture(scope="module")
def spike():
    """Return the loaded spike module (imported once per test session)."""
    return _load_spike()


# ---------------------------------------------------------------------------
# bytes_per_image_fp16 — pure arithmetic.
# ---------------------------------------------------------------------------


class TestBytesPerImageFp16:
    def test_single_level_no_sam2(self, spike) -> None:
        # 1 level: (256, 10, 10) -> 256*10*10*2 = 51_200 bytes
        result = spike.bytes_per_image_fp16([(256, 10, 10)])
        assert result["backbone_fpn_bytes"] == 256 * 10 * 10 * 2
        assert result["sam2_fpn_bytes"] == 0
        assert result["total_bytes"] == result["backbone_fpn_bytes"]
        assert result["total_bytes_no_sam2"] == result["backbone_fpn_bytes"]

    def test_multi_level_no_sam2(self, spike) -> None:
        # 3 levels: (256, 4, 4), (256, 2, 2), (256, 1, 1)
        shapes = [(256, 4, 4), (256, 2, 2), (256, 1, 1)]
        expected = (256 * 4 * 4 + 256 * 2 * 2 + 256 * 1 * 1) * 2
        result = spike.bytes_per_image_fp16(shapes)
        assert result["backbone_fpn_bytes"] == expected
        assert result["total_bytes"] == expected
        assert result["total_bytes_no_sam2"] == expected

    def test_with_sam2(self, spike) -> None:
        fpn = [(256, 4, 4), (256, 2, 2)]
        sam2 = [(128, 4, 4)]
        result = spike.bytes_per_image_fp16(fpn, sam2_fpn_shapes=sam2)
        fpn_bytes = (256 * 4 * 4 + 256 * 2 * 2) * 2
        sam2_bytes = 128 * 4 * 4 * 2
        assert result["backbone_fpn_bytes"] == fpn_bytes
        assert result["sam2_fpn_bytes"] == sam2_bytes
        assert result["total_bytes"] == fpn_bytes + sam2_bytes
        assert result["total_bytes_no_sam2"] == fpn_bytes

    def test_empty_sam2_is_zero(self, spike) -> None:
        result = spike.bytes_per_image_fp16([(64, 8, 8)], sam2_fpn_shapes=[])
        assert result["sam2_fpn_bytes"] == 0

    def test_none_sam2_is_zero(self, spike) -> None:
        result = spike.bytes_per_image_fp16([(64, 8, 8)], sam2_fpn_shapes=None)
        assert result["sam2_fpn_bytes"] == 0

    def test_fp16_two_bytes_per_element(self, spike) -> None:
        # A single element tensor -> 2 bytes in fp16.
        result = spike.bytes_per_image_fp16([(1, 1, 1)])
        assert result["backbone_fpn_bytes"] == 2


# ---------------------------------------------------------------------------
# breakeven_table — pure arithmetic, deterministic output.
# ---------------------------------------------------------------------------


class TestBreakevenTable:
    _GB = 1024**3

    def test_fits_all_in_ram(self, spike) -> None:
        # Very small images: 1 byte per image (hypothetical); 1 image -> fits easily.
        table = spike.breakeven_table(1, 1, [1], host_ram_budget_gb=16.0)
        assert "YES" in table

    def test_does_not_fit_in_ram(self, spike) -> None:
        # 17 GiB per image, 1 image -> exceeds 16 GB budget.
        bytes_17gb = int(17 * self._GB)
        table = spike.breakeven_table(bytes_17gb, bytes_17gb, [1], host_ram_budget_gb=16.0)
        assert "NO " in table

    def test_disk_warn_fires(self, spike) -> None:
        # 60 GiB per image * 1 image = 60 GiB -> exceeds 50 GiB warn threshold.
        bytes_60gb = int(60 * self._GB)
        table = spike.breakeven_table(bytes_60gb, bytes_60gb, [1], disk_warn_threshold_gb=50.0)
        assert "WARN" in table

    def test_disk_warn_absent_below_threshold(self, spike) -> None:
        # 10 GiB per image * 1 image = 10 GiB -> below 50 GiB -> no WARN.
        bytes_10gb = int(10 * self._GB)
        table = spike.breakeven_table(bytes_10gb, bytes_10gb, [1], disk_warn_threshold_gb=50.0)
        assert "WARN" not in table

    def test_no_sam2_column_fits_but_sam2_does_not(self, spike) -> None:
        # no-sam2: 8 GiB per image -> fits 16 GB budget (1 image).
        # sam2:   17 GiB per image -> does NOT fit.
        bytes_8gb = int(8 * self._GB)
        bytes_17gb = int(17 * self._GB)
        table = spike.breakeven_table(bytes_8gb, bytes_17gb, [1], host_ram_budget_gb=16.0)
        # Both YES and NO should appear (one per column).
        assert "YES" in table
        assert "NO " in table

    def test_multiple_dataset_sizes(self, spike) -> None:
        # Smoke: multiple sizes print without error and contain all N values.
        bytes_1mb = 1 * 1024**2
        table = spike.breakeven_table(bytes_1mb, bytes_1mb, [100, 1000, 10000])
        assert "100" in table
        assert "1000" in table
        assert "10000" in table

    def test_returns_string(self, spike) -> None:
        table = spike.breakeven_table(1, 1, [1])
        assert isinstance(table, str)
        assert len(table) > 0

    def test_boundary_exactly_16gb(self, spike) -> None:
        # Exactly 16 GiB should be a YES (fits <= budget).
        bytes_16gb = int(16 * self._GB)
        table = spike.breakeven_table(bytes_16gb, bytes_16gb, [1], host_ram_budget_gb=16.0)
        assert "YES" in table

    def test_boundary_just_over_16gb(self, spike) -> None:
        # 16 GiB + 1 byte should be NO.
        bytes_over = int(16 * self._GB) + 1
        table = spike.breakeven_table(bytes_over, bytes_over, [1], host_ram_budget_gb=16.0)
        assert "NO " in table


# ---------------------------------------------------------------------------
# Standalone break-even CLI entrypoint (_breakeven_main).
# ---------------------------------------------------------------------------


class TestBreakevenMainCli:
    def test_prints_table_to_stdout(self, spike, capsys) -> None:
        ret = spike._breakeven_main(
            ["--bytes-no-sam2", "1048576", "--dataset-sizes", "100", "1000"]
        )
        assert ret == 0
        captured = capsys.readouterr()
        assert "Break-even" in captured.out
        assert "100" in captured.out
        assert "1000" in captured.out

    def test_missing_bytes_no_sam2_exits(self, spike) -> None:
        with pytest.raises(SystemExit):
            spike._breakeven_main(["--dataset-sizes", "100"])

    def test_bytes_with_sam2_defaults_to_no_sam2(self, spike, capsys) -> None:
        # When --bytes-with-sam2 is omitted, should not crash and should produce output.
        ret = spike._breakeven_main(["--bytes-no-sam2", "1048576", "--dataset-sizes", "1"])
        assert ret == 0

    def test_custom_ram_budget(self, spike, capsys) -> None:
        # 8 GB budget: 1 image * 10 GiB/image -> NO.
        bytes_10gb = str(10 * 1024**3)
        ret = spike._breakeven_main(
            [
                "--bytes-no-sam2",
                bytes_10gb,
                "--dataset-sizes",
                "1",
                "--ram-budget-gb",
                "8",
            ]
        )
        assert ret == 0
        captured = capsys.readouterr()
        assert "NO " in captured.out
