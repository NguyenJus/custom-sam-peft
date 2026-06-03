import dataclasses

from custom_sam_peft.data.spatial_meta import SpatialMeta


def test_geo_kind_carries_crs_affine_nodata():
    m = SpatialMeta(kind="geo", crs="EPSG:4326", affine=(1, 0, 0, 0, 1, 0), nodata=0.0)
    assert m.kind == "geo"
    assert m.crs == "EPSG:4326"
    assert m.orientation is None  # dicom-only field defaults None


def test_is_frozen():
    m = SpatialMeta(kind="geo")
    import pytest

    with pytest.raises(dataclasses.FrozenInstanceError):
        m.crs = "EPSG:3857"  # type: ignore[misc]
