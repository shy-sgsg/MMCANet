"""全球 Sentinel-1 元数据空间分组与角色划分。"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterable, Mapping

import numpy as np

EARTH_RADIUS_M = 6_371_008.8
SPLIT_ROLES = ("train", "development", "confirmation")


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, left: int, right: int) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left == root_right:
            return
        if self.rank[root_left] < self.rank[root_right]:
            root_left, root_right = root_right, root_left
        self.parent[root_right] = root_left
        if self.rank[root_left] == self.rank[root_right]:
            self.rank[root_left] += 1


def _required_text(record: Mapping[str, object], key: str, sample_id: str) -> str:
    value = str(record.get(key, "")).strip()
    if not value:
        raise ValueError(f"样本 {sample_id} 缺少元数据字段 {key}")
    return value


def assign_spatial_groups(
    records: Iterable[Mapping[str, object]], *, separation_m: float = 8_000.0
) -> list[dict[str, object]]:
    """按地点标识和球面中心距离的连通分量形成空间组。

    只接受元数据记录，不读取或解码图像。8 km 默认间距覆盖 512×512、10 m
    网格影像的对角线范围，并留出少量定位误差余量。
    """
    rows = [dict(record) for record in records]
    if not rows:
        raise ValueError("空间分组输入不能为空")
    if (
        not math.isfinite(separation_m)
        or separation_m <= 0
        or separation_m > math.pi * EARTH_RADIUS_M
    ):
        raise ValueError("空间分组距离必须位于 (0, 地球半周长] 范围")

    sample_ids = [_required_text(row, "sample_id", "<未知>") for row in rows]
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("sample_id 必须唯一")
    location_ids = [
        _required_text(row, "location_id", sample_id)
        for row, sample_id in zip(rows, sample_ids, strict=True)
    ]
    latitudes = np.asarray([row.get("latitude") for row in rows], dtype=np.float64)
    longitudes = np.asarray([row.get("longitude") for row in rows], dtype=np.float64)
    if (
        not np.isfinite(latitudes).all()
        or not np.isfinite(longitudes).all()
        or np.any((latitudes < -90.0) | (latitudes > 90.0))
        or np.any((longitudes < -180.0) | (longitudes > 180.0))
    ):
        raise ValueError("经纬度必须有限且位于合法范围")

    union_find = _UnionFind(len(rows))
    first_by_location: dict[str, int] = {}
    for index, location_id in enumerate(location_ids):
        previous = first_by_location.setdefault(location_id, index)
        union_find.union(previous, index)

    lat_rad = np.deg2rad(latitudes)
    lon_rad = np.deg2rad(longitudes)
    angular_limit = separation_m / EARTH_RADIUS_M
    haversine_limit = math.sin(angular_limit / 2.0) ** 2
    for start in range(0, len(rows), 128):
        stop = min(start + 128, len(rows))
        delta_lat = lat_rad[None, :] - lat_rad[start:stop, None]
        delta_lon = lon_rad[None, :] - lon_rad[start:stop, None]
        haversine = (
            np.sin(delta_lat / 2.0) ** 2
            + np.cos(lat_rad[start:stop, None])
            * np.cos(lat_rad[None, :])
            * np.sin(delta_lon / 2.0) ** 2
        )
        local_rows, columns = np.where(haversine <= haversine_limit)
        for local_row, column in zip(local_rows, columns, strict=True):
            row_index = start + int(local_row)
            if row_index < int(column):
                union_find.union(row_index, int(column))

    components: dict[int, list[int]] = defaultdict(list)
    for index in range(len(rows)):
        components[union_find.find(index)].append(index)
    ordered_components = sorted(
        components.values(), key=lambda members: min(sample_ids[i] for i in members)
    )
    for group_number, members in enumerate(ordered_components, start=1):
        group_id = f"spatial-{group_number:05d}"
        for index in members:
            rows[index]["source_group_id"] = group_id
            rows[index]["sample_id"] = sample_ids[index]
    return sorted(rows, key=lambda row: str(row["sample_id"]))


def allocate_spatial_splits(
    records: Iterable[Mapping[str, object]],
    *,
    seed: int,
    fractions: tuple[float, float, float] = (0.70, 0.15, 0.15),
) -> dict[str, list[dict[str, object]]]:
    """按空间组分配 train/development/confirmation，并平衡轨道/极化计数。"""
    rows = [dict(record) for record in records]
    if not rows:
        raise ValueError("划分输入不能为空")
    if len(fractions) != len(SPLIT_ROLES) or any(
        not math.isfinite(value) or value <= 0 for value in fractions
    ):
        raise ValueError("train/development/confirmation 比例必须为有限正数")
    if not math.isclose(sum(fractions), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("train/development/confirmation 比例之和必须为 1")

    sample_ids = [_required_text(row, "sample_id", "<未知>") for row in rows]
    group_ids = [
        _required_text(row, "source_group_id", sample_id)
        for row, sample_id in zip(rows, sample_ids, strict=True)
    ]
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("sample_id 必须唯一")

    per_group: dict[str, dict[tuple[str, str], int]] = defaultdict(lambda: defaultdict(int))
    for row, sample_id, group_id in zip(rows, sample_ids, group_ids, strict=True):
        orbit = _required_text(row, "orbit", sample_id).upper()
        polarization = _required_text(row, "polarization", sample_id).upper()
        per_group[group_id][(orbit, polarization)] += 1
        row["orbit"] = orbit
        row["polarization"] = polarization

    if len(per_group) < len(SPLIT_ROLES):
        raise ValueError("至少需要三个独立空间组才能建立三个角色")

    total_by_stratum: dict[tuple[str, str], int] = defaultdict(int)
    for stratum_counts in per_group.values():
        for stratum, count in stratum_counts.items():
            total_by_stratum[stratum] += count
    targets = {
        role: {stratum: total * fraction for stratum, total in total_by_stratum.items()}
        for role, fraction in zip(SPLIT_ROLES, fractions, strict=True)
    }
    counts = {role: defaultdict(int) for role in SPLIT_ROLES}
    assigned_group_counts = {role: 0 for role in SPLIT_ROLES}

    ordered_group_ids = sorted(per_group)
    random_rank = {
        group_id: rank
        for rank, group_id in enumerate(np.random.default_rng(seed).permutation(ordered_group_ids))
    }
    ordered_group_ids.sort(
        key=lambda group_id: (-sum(per_group[group_id].values()), random_rank[group_id])
    )

    group_role: dict[str, str] = {}
    for position, group_id in enumerate(ordered_group_ids):
        remaining_groups = len(ordered_group_ids) - position
        empty_roles = [role for role in SPLIT_ROLES if assigned_group_counts[role] == 0]
        candidate_roles = (
            empty_roles if remaining_groups == len(empty_roles) else list(SPLIT_ROLES)
        )
        stratum_counts = per_group[group_id]
        role_scores = []
        for role in candidate_roles:
            score = 0.0
            for stratum, target in targets[role].items():
                current = counts[role][stratum]
                proposed = current + stratum_counts.get(stratum, 0)
                scale = max(target, 1.0)
                score += ((proposed - target) ** 2 - (current - target) ** 2) / scale
            role_scores.append((score, SPLIT_ROLES.index(role), role))
        selected_role = min(role_scores)[2]
        group_role[group_id] = selected_role
        assigned_group_counts[selected_role] += 1
        for stratum, count in stratum_counts.items():
            counts[selected_role][stratum] += count

    result = {role: [] for role in SPLIT_ROLES}
    for row in rows:
        role = group_role[str(row["source_group_id"])]
        row["split_role"] = role
        result[role].append(row)
    for role in SPLIT_ROLES:
        result[role].sort(key=lambda row: str(row["sample_id"]))
    return result
