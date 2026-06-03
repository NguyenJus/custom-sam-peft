import numpy as np
import pytest

pytest.importorskip("pydicom")
pytest.importorskip("nibabel")

import pydicom

from custom_sam_peft.data.dicom_io import group_series, series_affine
from custom_sam_peft.predict.writers import write_nifti_volume


def test_C11_series_groups_sorts_stacks_with_affine(tmp_path):
    import nibabel as nib

    # build 3 single-series slices at z=0,2,4 (out of order on disk)
    from tests.unit.test_dicom_decode import _make_ct  # reuse the fixture builder

    series = "1.2.3"
    paths = []
    for z in (4.0, 0.0, 2.0):  # deliberately unsorted
        p = _make_ct(tmp_path, np.full((4, 4), 10, np.int16), signed=1, name=f"s{z}.dcm")
        import pydicom

        ds = pydicom.dcmread(p)
        ds.SeriesInstanceUID = series
        ds.ImagePositionPatient = [0, 0, z]
        ds.save_as(p, enforce_file_format=True)
        paths.append(p)

    groups = group_series(paths)
    assert len(groups) == 1
    ordered = groups[series]
    zs = [float(ds.ImagePositionPatient[2]) for ds in ordered]
    assert zs == [0.0, 2.0, 4.0]  # sorted by position

    affine = series_affine(ordered)
    masks = [np.ones((4, 4), np.uint8) for _ in ordered]
    out = tmp_path / "vol.nii.gz"
    write_nifti_volume(masks, affine, out)
    vol = nib.load(str(out))
    assert vol.shape == (4, 4, 3)
    assert np.allclose(vol.affine, affine)


def _make_rle(binary_mask):
    import pycocotools.mask as mask_utils

    rle = mask_utils.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("ascii")
    return rle


def test_write_nifti_volumes_emits_one_volume_per_series(tmp_path):
    """Runner-emission helper: a 3-slice DICOM series → ONE .nii.gz of shape (H, W, 3)
    with the series affine, written under output/volumes/<series_uid>.nii.gz."""
    import nibabel as nib

    from custom_sam_peft.data.dicom_io import group_series, series_affine
    from custom_sam_peft.data.spatial_meta import SpatialMeta
    from custom_sam_peft.predict.writers import write_nifti_volumes
    from tests.unit.test_dicom_decode import _make_ct

    series = "9.9.9"
    H = W = 4
    # 3 slices at z=0,2,4 written out of order (z=4 first) to exercise sorting.
    id_to_path = {}
    id_to_meta = {}
    originals = {}
    for image_id, z in ((100, 4.0), (101, 0.0), (102, 2.0)):
        p = _make_ct(tmp_path, np.full((H, W), 10, np.int16), signed=1, name=f"img{image_id}.dcm")
        ds = pydicom.dcmread(p)
        ds.SeriesInstanceUID = series
        ds.ImagePositionPatient = [0, 0, z]
        ds.save_as(p, enforce_file_format=True)
        id_to_path[image_id] = p.resolve()
        id_to_meta[image_id] = SpatialMeta(
            kind="dicom",
            series_uid=series,
            position=[0, 0, z],
            orientation=[1, 0, 0, 0, 1, 0],
            pixel_spacing=(1.0, 1.0),
        )
        originals[image_id] = (H, W)

    # Distinct per-slice masks so the stacked volume is order-sensitive.
    mask_for = {
        100: np.eye(H, dtype=np.uint8),  # z=4 → top of stack
        101: np.ones((H, W), np.uint8),  # z=0 → bottom of stack
        102: np.zeros((H, W), np.uint8),  # z=2 → middle
    }
    mask_for[102][0, 0] = 1
    all_predictions = [
        {
            "image_id": iid,
            "category_id": 1,
            "segmentation": _make_rle(mask_for[iid]),
            "score": 0.9,
        }
        for iid in (100, 101, 102)
    ]

    write_nifti_volumes(all_predictions, id_to_meta, id_to_path, originals, tmp_path)

    out = tmp_path / "volumes" / f"{series}.nii.gz"
    assert out.exists(), "one .nii.gz per series must be written"
    vol = nib.load(str(out))
    assert vol.shape == (H, W, 3)

    # Affine round-trips the constructed series affine.
    ordered = group_series(list(id_to_path.values()))[series]
    expected_affine = series_affine(ordered)
    assert np.allclose(vol.affine, expected_affine)

    # Slices are stacked in geometric (z=0,2,4) order, i.e. by image_id 101,102,100.
    data = np.asarray(vol.dataobj)
    assert np.array_equal(data[:, :, 0], mask_for[101])  # z=0
    assert np.array_equal(data[:, :, 1], mask_for[102])  # z=2
    assert np.array_equal(data[:, :, 2], mask_for[100])  # z=4


def test_write_nifti_volumes_noop_on_empty_meta(tmp_path):
    """No-op when there are no DICOM metas (non-DICOM runs stay byte-for-byte)."""
    from custom_sam_peft.predict.writers import write_nifti_volumes

    entry = {
        "image_id": 1,
        "category_id": 1,
        "segmentation": _make_rle(np.ones((4, 4), np.uint8)),
        "score": 0.9,
    }
    write_nifti_volumes([entry], {}, {1: tmp_path / "x.dcm"}, {1: (4, 4)}, tmp_path)
    assert not (tmp_path / "volumes").exists()
