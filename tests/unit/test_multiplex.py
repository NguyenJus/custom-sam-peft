"""Tests for the torch-free multiplex index helper (spec §5.3, §8.5)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from custom_sam_peft.models import _multiplex
from custom_sam_peft.models._multiplex import multiplex_index_arrays


@pytest.mark.parametrize(("b", "k"), [(1, 1), (2, 3), (3, 2), (4, 1), (1, 5), (2, 2)])
def test_matches_torch_reference(b: int, k: int) -> None:
    """img_ids/text_ids equal the torch arange ordering used by the adapter."""
    img_ids, text_ids = multiplex_index_arrays(b, k)
    ref_img = torch.arange(b).repeat_interleave(k).numpy()
    ref_text = torch.arange(k).repeat(b).numpy()
    np.testing.assert_array_equal(img_ids, ref_img)
    np.testing.assert_array_equal(text_ids, ref_text)


@pytest.mark.parametrize(("b", "k"), [(1, 1), (2, 3), (5, 4)])
def test_shape_and_dtype(b: int, k: int) -> None:
    """Both arrays are 1-D of length b*k and int64."""
    img_ids, text_ids = multiplex_index_arrays(b, k)
    assert img_ids.shape == (b * k,)
    assert text_ids.shape == (b * k,)
    assert img_ids.dtype == np.int64
    assert text_ids.dtype == np.int64


def test_torch_free() -> None:
    """Loading and calling the helper module never imports torch.

    The module is loaded by file path via importlib to bypass the eager-import
    ``custom_sam_peft/__init__`` chain (which pulls torch); this proves the
    ``_multiplex`` module itself imports only numpy, per spec §8.4/§8.5.
    """
    module_path = Path(_multiplex.__file__)
    code = (
        "import sys, importlib.util; "
        f"spec = importlib.util.spec_from_file_location('_multiplex', {str(module_path)!r}); "
        "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); "
        "m.multiplex_index_arrays(2, 3); "
        "assert 'torch' not in sys.modules; print('OK')"
    )
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout
