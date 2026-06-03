"""C12: missing [dicom] extra raises an actionable RuntimeError (spec §8, §10)."""

from __future__ import annotations

import builtins

import pytest


def test_C12_missing_pydicom_raises_actionable(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name == "pydicom" or name.startswith("pydicom."):
            raise ImportError("No module named 'pydicom'")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    from custom_sam_peft.data.dicom_io import read_dcm_with_meta

    with pytest.raises(RuntimeError, match=r"pip install custom-sam-peft\[dicom\]"):
        read_dcm_with_meta("x.dcm", 1)
