"""将公开 SAR 与 GT25 文件清单匹配到来源元数据，不读取影像载荷。"""
from __future__ import annotations

import math
import re
from pathlib import PurePosixPath
from typing import Iterable, Mapping


_LOCATION_ID = re.compile(r"^[A-Za-z0-9]{5}$")
_SINGLE_LOOK = re.compile(r"^([A-Za-z0-9]{5})_0\.tif$", re.IGNORECASE)
_GT25_FOLDERS = {"GT25", "GT_25", "GT-25"}


def _location_acquisition(filename: object) -> tuple[str, str]:
    stem = PurePosixPath(str(filename).replace("\\", "/")).stem
    if "_" not in stem:
        raise ValueError(f"元数据 filename 缺少采集序号：{filename}")
    location_id, acquisition_index = stem.rsplit("_", 1)
    if not _LOCATION_ID.fullmatch(location_id) or not acquisition_index.isdigit():
        raise ValueError(f"元数据 filename 格式无效：{filename}")
    return location_id, acquisition_index


def _direction_polarization(parts: tuple[str, ...]) -> tuple[str, str] | None:
    upper = tuple(part.upper() for part in parts)
    direction = next((value for value in upper if value in {"ASCENDING", "DESCENDING"}), None)
    polarization = next((value for value in upper if value in {"VV", "VH"}), None)
    if direction is None or polarization is None:
        return None
    return direction, polarization


def _metadata_coordinates(rows: Iterable[Mapping[str, object]]):
    coordinates: dict[tuple[str, str], tuple[float, float]] = {}
    for row in rows:
        location_id, _ = _location_acquisition(row.get("filename", ""))
        direction = str(row.get("direction", "")).strip().upper()
        if direction not in {"ASCENDING", "DESCENDING"}:
            raise ValueError(f"地点 {location_id} 的轨道方向无效")
        try:
            latitude = float(row["latitud"])
            longitude = float(row["longitud"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"地点 {location_id} 缺少合法经纬度") from exc
        if (
            not math.isfinite(latitude)
            or not math.isfinite(longitude)
            or not -90.0 <= latitude <= 90.0
            or not -180.0 <= longitude <= 180.0
        ):
            raise ValueError(f"地点 {location_id} 的经纬度越界或非有限")
        key = (location_id, direction)
        previous = coordinates.setdefault(key, (latitude, longitude))
        if abs(previous[0] - latitude) > 1e-6 or abs(previous[1] - longitude) > 1e-6:
            raise ValueError(f"同一地点与轨道方向的坐标不一致：{location_id} {direction}")
    return coordinates


def _input_members(member_paths: Iterable[str]):
    result = {}
    for raw_path in member_paths:
        normalized_path = str(raw_path).replace("\\", "/")
        parts = tuple(part for part in normalized_path.split("/") if part)
        if not any(part.upper() == "SENTINEL1" for part in parts):
            continue
        direction_pol = _direction_polarization(parts)
        if direction_pol is None:
            continue
        match = _SINGLE_LOOK.fullmatch(parts[-1])
        if match is None:
            continue
        location_id = match.group(1)
        direction, polarization = direction_pol
        key = (location_id, direction, polarization)
        if key in result:
            raise ValueError(f"SAR 单景文件清单出现重复样本：{key}")
        result[key] = normalized_path
    return result


def _gt25_members(member_paths: Iterable[str]):
    result = {}
    for raw_path in member_paths:
        normalized_path = str(raw_path).replace("\\", "/")
        parts = tuple(part for part in normalized_path.split("/") if part)
        if not any(part.upper() in _GT25_FOLDERS for part in parts[:-1]):
            continue
        direction_pol = _direction_polarization(parts)
        if direction_pol is None:
            continue
        location_id = PurePosixPath(parts[-1]).stem
        if "_" in location_id and location_id.rsplit("_", 1)[1].isdigit():
            location_id = location_id.rsplit("_", 1)[0]
        if not _LOCATION_ID.fullmatch(location_id):
            continue
        direction, polarization = direction_pol
        key = (location_id, direction, polarization)
        if key in result:
            raise ValueError(f"GT25 文件清单出现重复样本：{key}")
        result[key] = normalized_path
    return result


def build_pair_inventory(
    metadata_rows: Iterable[Mapping[str, object]],
    sar_member_paths: Iterable[str],
    reference_member_paths: Iterable[str],
) -> dict[str, object]:
    """只基于 CSV 与归档成员路径生成可用单景/GT25 配对记录及审计计数。"""
    coordinates = _metadata_coordinates(metadata_rows)
    input_members = _input_members(sar_member_paths)
    reference_members = _gt25_members(reference_member_paths)
    input_keys = set(input_members)
    reference_keys = set(reference_members)
    paired_keys = input_keys & reference_keys
    metadata_keys = {(location_id, direction) for location_id, direction in coordinates}

    records = []
    for location_id, direction, polarization in sorted(paired_keys):
        coordinate_key = (location_id, direction)
        if coordinate_key not in coordinates:
            continue
        latitude, longitude = coordinates[coordinate_key]
        records.append(
            {
                "sample_id": f"{location_id}_{direction}_{polarization}",
                "location_id": location_id,
                "latitude": latitude,
                "longitude": longitude,
                "orbit": direction,
                "polarization": polarization,
                "input_member": input_members[(location_id, direction, polarization)],
                "reference_member": reference_members[(location_id, direction, polarization)],
            }
        )

    audit = {
        "metadata_location_direction_count": len(metadata_keys),
        "sar_single_look_count": len(input_members),
        "gt25_reference_count": len(reference_members),
        "member_key_pair_count": len(paired_keys),
        "matched_pairs_with_metadata": len(records),
        "input_missing_reference": len(input_keys - reference_keys),
        "reference_missing_input": len(reference_keys - input_keys),
        "input_missing_metadata": sum(
            (location_id, direction) not in metadata_keys
            for location_id, direction, _ in input_keys
        ),
        "paired_missing_metadata": sum(
            (location_id, direction) not in metadata_keys
            for location_id, direction, _ in paired_keys
        ),
    }
    return {"records": records, "audit": audit}


def build_reference_inventory(
    metadata_rows: Iterable[Mapping[str, object]],
    reference_member_paths: Iterable[str],
    *,
    polarization: str = "VV",
) -> dict[str, object]:
    """将 GT25 成员与 CSV 地点/轨道元数据精确关联，不猜测 SAR 输入配对。"""
    selected_polarization = polarization.strip().upper()
    if selected_polarization not in {"VV", "VH"}:
        raise ValueError("极化必须为 VV 或 VH")

    coordinates = _metadata_coordinates(metadata_rows)
    reference_members = _gt25_members(reference_member_paths)
    selected = {
        key: path
        for key, path in reference_members.items()
        if key[2] == selected_polarization
    }
    records = []
    missing_metadata = 0
    for (location_id, direction, pol), member_path in sorted(selected.items()):
        coordinate_key = (location_id, direction)
        if coordinate_key not in coordinates:
            missing_metadata += 1
            continue
        latitude, longitude = coordinates[coordinate_key]
        records.append(
            {
                "sample_id": f"{location_id}_{direction}_{pol}",
                "location_id": location_id,
                "latitude": latitude,
                "longitude": longitude,
                "orbit": direction,
                "polarization": pol,
                "reference_member": member_path,
            }
        )

    return {
        "records": records,
        "audit": {
            "metadata_location_direction_count": len(coordinates),
            "gt25_reference_count": len(selected),
            "matched_reference_count": len(records),
            "reference_missing_metadata": missing_metadata,
            "polarization": selected_polarization,
        },
    }
