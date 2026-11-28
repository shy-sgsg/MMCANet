from __future__ import annotations

from experiments.phase_e_data_inventory import build_pair_inventory, build_reference_inventory


def test_pair_inventory_matches_only_available_single_look_and_gt25_files():
    metadata = [
        {
            "filename": "abc12_0",
            "direction": "ASCENDING",
            "latitud": "10.0",
            "longitud": "20.0",
        },
        {
            "filename": "abc12_1",
            "direction": "ASCENDING",
            "latitud": "10.0",
            "longitud": "20.0",
        },
        {
            "filename": "def34_0",
            "direction": "DESCENDING",
            "latitud": "-3.0",
            "longitud": "40.0",
        },
    ]
    sar_members = [
        "dataset/SENTINEL1/ASCENDING/VV/abc12_0.tif",
        "dataset/SENTINEL1/ASCENDING/VH/abc12_0.tif",
        "dataset/SENTINEL1/DESCENDING/VV/def34_0.tif",
        "dataset/SENTINEL1/ASCENDING/VV/zzzzz_0.tif",
    ]
    reference_members = [
        "fusion/ASCENDING/VV/GT_25/abc12.tif",
        "fusion/ASCENDING/VH/GT25/abc12.tif",
        "fusion/DESCENDING/VV/GT_25/def34.tif",
    ]

    result = build_pair_inventory(metadata, sar_members, reference_members)
    records = result["records"]

    assert [row["sample_id"] for row in records] == [
        "abc12_ASCENDING_VH",
        "abc12_ASCENDING_VV",
        "def34_DESCENDING_VV",
    ]
    assert records[0]["location_id"] == "abc12"
    assert records[0]["latitude"] == 10.0
    assert records[0]["input_member"].endswith("abc12_0.tif")
    assert records[0]["reference_member"].endswith("abc12.tif")
    assert result["audit"]["matched_pairs_with_metadata"] == 3
    assert result["audit"]["input_missing_metadata"] == 1


def test_pair_inventory_fails_on_conflicting_location_coordinates():
    metadata = [
        {"filename": "abc12_0", "direction": "ASCENDING", "latitud": "10", "longitud": "20"},
        {"filename": "abc12_1", "direction": "ASCENDING", "latitud": "10.1", "longitud": "20"},
    ]
    sar_members = ["dataset/SENTINEL1/ASCENDING/VV/abc12_0.tif"]
    reference_members = ["fusion/ASCENDING/VV/GT25/abc12.tif"]

    try:
        build_pair_inventory(metadata, sar_members, reference_members)
    except ValueError as error:
        assert "坐标不一致" in str(error)
    else:
        raise AssertionError("同一地点的冲突坐标必须失败关闭")


def test_reference_inventory_uses_metadata_matched_gt25_without_fabricating_inputs():
    metadata = [
        {
            "filename": "abc12_0",
            "direction": "ASCENDING",
            "latitud": "10.0",
            "longitud": "20.0",
        },
        {
            "filename": "def34_0",
            "direction": "DESCENDING",
            "latitud": "-3.0",
            "longitud": "40.0",
        },
    ]
    references = [
        "fusion/SENTINEL1/ASCENDING/VV/GT25/abc12.tif",
        "fusion/SENTINEL1/DESCENDING/VV/GT25/def34.tif",
        "fusion/SENTINEL1/ASCENDING/VV/GT25/zzzzz.tif",
        "fusion/SENTINEL1/ASCENDING/VV/GT20/abc12.tif",
    ]

    result = build_reference_inventory(metadata, references, polarization="VV")

    assert [row["sample_id"] for row in result["records"]] == [
        "abc12_ASCENDING_VV",
        "def34_DESCENDING_VV",
    ]
    assert result["records"][0]["latitude"] == 10.0
    assert result["records"][0]["reference_member"].endswith("GT25/abc12.tif")
    assert result["audit"]["gt25_reference_count"] == 3
    assert result["audit"]["matched_reference_count"] == 2
    assert result["audit"]["reference_missing_metadata"] == 1


def test_reference_inventory_rejects_duplicate_gt25_keys():
    metadata = [
        {"filename": "abc12_0", "direction": "ASCENDING", "latitud": "10", "longitud": "20"}
    ]
    references = [
        "fusion/SENTINEL1/ASCENDING/VV/GT25/abc12.tif",
        "other/SENTINEL1/ASCENDING/VV/GT25/abc12.tif",
    ]

    try:
        build_reference_inventory(metadata, references, polarization="VV")
    except ValueError as error:
        assert "重复" in str(error)
    else:
        raise AssertionError("GT25 清单重复键必须失败关闭")
