import numpy as np
import pytest

pydicom = pytest.importorskip("pydicom")

from custom_sam_peft.data.dicom_io import read_dcm_with_meta  # noqa: E402


def _make_ct(
    tmp_path,
    stored,
    slope=1.0,
    intercept=-1024.0,
    photometric="MONOCHROME2",
    signed=0,
    window=None,
    name="ct.dcm",
):
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid

    ds = Dataset()
    ds.file_meta = FileMetaDataset()
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.file_meta.MediaStorageSOPClassUID = CTImageStorage
    ds.Rows, ds.Columns = stored.shape
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = photometric
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = signed
    ds.RescaleSlope = slope
    ds.RescaleIntercept = intercept
    ds.PixelSpacing = [1.0, 1.0]
    ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    ds.ImagePositionPatient = [0, 0, 0]
    ds.SeriesInstanceUID = generate_uid()
    ds.SOPInstanceUID = generate_uid()
    ds.FrameOfReferenceUID = generate_uid()
    if window is not None:
        ds.WindowCenter, ds.WindowWidth = window
    dtype = np.int16 if signed else np.uint16
    ds.PixelData = stored.astype(dtype).tobytes()
    p = tmp_path / name
    ds.save_as(p, enforce_file_format=True)
    return p


def test_C10_modality_lut_decodes_negative_hu(tmp_path):
    stored = np.full((4, 4), 24, np.int16)  # 24*1 + (-1024) = -1000 HU (air)
    p = _make_ct(tmp_path, stored, signed=1)
    pixels, meta = read_dcm_with_meta(p, 1)
    assert meta.kind == "dicom"
    assert meta.rescale == (1.0, -1024.0)
    assert pixels.min() < 0  # signed CT decodes negative HU


def test_C10_monochrome1_inverted(tmp_path):
    stored = np.array([[0, 100], [200, 300]], np.uint16)
    p1 = _make_ct(
        tmp_path, stored, slope=1.0, intercept=0.0, photometric="MONOCHROME1", name="m1.dcm"
    )
    p2 = _make_ct(
        tmp_path, stored, slope=1.0, intercept=0.0, photometric="MONOCHROME2", name="m2.dcm"
    )
    a1, _ = read_dcm_with_meta(p1, 1)
    a2, _ = read_dcm_with_meta(p2, 1)
    # MONOCHROME1 inverted relative to MONOCHROME2: argmin/argmax flip
    assert np.unravel_index(a1.argmax(), a1.shape[:2]) == np.unravel_index(
        a2.argmin(), a2.shape[:2]
    )
    # Full-array guarantee: inversion is exactly max - value (mathematical identity).
    assert np.allclose(a1, a2.max() - a2)


def test_voi_window_override(tmp_path):
    """Override window wins over file window; meta.voi_window reflects override; pixels differ."""
    stored = np.arange(0, 16, dtype=np.uint16).reshape(4, 4)
    file_window = (500.0, 1000.0)
    override_window = (100.0, 200.0)
    p = _make_ct(tmp_path, stored, slope=1.0, intercept=0.0, window=file_window)

    # Without override: meta.voi_window == file window
    pix_file, meta_file = read_dcm_with_meta(p, 1)
    assert meta_file.voi_window == file_window

    # With override: meta.voi_window == override
    pix_override, meta_override = read_dcm_with_meta(p, 1, voi_window=override_window)
    assert meta_override.voi_window == override_window

    # Override actually changes pixel values, not just metadata.
    assert not np.allclose(pix_file, pix_override)


def test_monochrome1_with_window_uses_hu_space_windowing(tmp_path):
    """Regression: VOI windowing must precede MONOCHROME1 inversion (DICOM PS3.3 §C.11.2).

    Build a MONOCHROME1 and MONOCHROME2 CT with the same stored pixels and a non-trivial
    window (center/width that actually clips).  After decoding:
      - Both see the same HU-space windowing.
      - MONOCHROME1 result == max(M2_result) - M2_result  (VOI then invert).
    Under the OLD (buggy) invert-then-VOI order this assertion FAILS because the window
    would have been applied to already-inverted values for M1.
    """
    # 4x4 ramp; slope=1, intercept=0 → HU == stored value.
    stored = np.arange(0, 16, dtype=np.uint16).reshape(4, 4)
    # Non-trivial window: center=7, width=6 → clips values outside [4, 10].
    window = (7.0, 6.0)

    p_m1 = _make_ct(
        tmp_path,
        stored,
        slope=1.0,
        intercept=0.0,
        photometric="MONOCHROME1",
        window=window,
        name="m1_win.dcm",
    )
    p_m2 = _make_ct(
        tmp_path,
        stored,
        slope=1.0,
        intercept=0.0,
        photometric="MONOCHROME2",
        window=window,
        name="m2_win.dcm",
    )

    m1_out, _ = read_dcm_with_meta(p_m1, 1)
    m2_out, _ = read_dcm_with_meta(p_m2, 1)

    # VOI-then-invert: MONOCHROME1 must equal (max - MONOCHROME2).
    assert np.allclose(m1_out, m2_out.max() - m2_out)
