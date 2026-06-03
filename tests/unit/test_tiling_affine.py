from rasterio.transform import from_origin

from custom_sam_peft.data.tiling import Window, tile_affine


def test_C8_tile_affine_is_parent_offset_by_origin():
    parent = from_origin(500000, 4600000, 10, 10)  # 10m pixels
    win = Window(y0=100, x0=200, h=300, w=400)
    child = tile_affine(parent, win)
    # world coord of tile pixel (0,0) == world coord of parent pixel (x0, y0)
    assert child * (0, 0) == parent * (win.x0, win.y0)
    # native-res: no scale change (a, e components identical)
    assert (child.a, child.e) == (parent.a, parent.e)
