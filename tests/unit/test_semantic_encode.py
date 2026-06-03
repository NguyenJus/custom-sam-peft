# tests/unit/test_semantic_encode.py
"""§5.2/§4.5 GT-encoding helper: class_map -> (class_names, value->label)."""

from __future__ import annotations

import json

from custom_sam_peft.data._semantic_encode import build_value_to_label


def test_ascending_pixel_value_order_drops_background(tmp_path):
    cm = tmp_path / "cm.json"
    cm.write_text(json.dumps({"0": "background", "2": "building", "1": "road"}))
    names, value_to_label, ignore = build_value_to_label(
        str(cm), ignore_index=255, background_class_name=None
    )
    # class_names = concept names in ascending pixel-value order, bg removed.
    assert names == ["road", "building"]  # value 1 -> road (dense 0), value 2 -> building (dense 1)
    # GT label = dense_id + 1; background -> 0; ignore stays ignore.
    assert value_to_label[0] == 0  # explicit background class
    assert value_to_label[1] == 1  # road -> channel 1
    assert value_to_label[2] == 2  # building -> channel 2
    assert ignore == 255


def test_recognized_background_names_case_insensitive(tmp_path):
    cm = tmp_path / "cm.json"
    cm.write_text(json.dumps({"0": "Unlabeled", "1": "road"}))
    names, value_to_label, _ = build_value_to_label(
        str(cm), ignore_index=255, background_class_name=None
    )
    assert names == ["road"]
    assert value_to_label[0] == 0  # "Unlabeled" recognized as background


def test_custom_background_class_name(tmp_path):
    cm = tmp_path / "cm.json"
    cm.write_text(json.dumps({"7": "void_region", "1": "road", "2": "tree"}))
    names, value_to_label, _ = build_value_to_label(
        str(cm), ignore_index=255, background_class_name="void_region"
    )
    assert names == ["road", "tree"]
    assert value_to_label[7] == 0  # custom bg -> channel 0


def test_no_background_class_all_concepts(tmp_path):
    cm = tmp_path / "cm.json"
    cm.write_text(json.dumps({"1": "road", "2": "tree"}))
    names, value_to_label, _ = build_value_to_label(
        str(cm), ignore_index=255, background_class_name=None
    )
    assert names == ["road", "tree"]
    assert value_to_label == {1: 1, 2: 2}  # nothing maps to channel 0 from data
