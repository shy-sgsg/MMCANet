#!/usr/bin/env python3
"""只汇总 AllClear train/validation ROI 元数据，不读取 test 或影像载荷。"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import io
import json
import tarfile
from collections import Counter
from pathlib import Path

import numpy as np

EARTH_RADIUS_M = 6_371_008.8
SPLIT_FILES = {
    "train": "metadata/rois/train_rois_19k.txt",
    "validation": "metadata/rois/val_rois_1k.txt",
}
CSV_PATH = "metadata/rois/rois_metadata.csv"
DATASET_JSONS = {
    "train": "metadata/datasets/train_tx12_s2-s1_100pct.json",
    "validation": "metadata/datasets/val_tx12_s2-s1_100pct.json",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_ids(tar: tarfile.TarFile, member_name: str) -> set[str]:
    member = tar.extractfile(member_name)
    if member is None:
        raise ValueError(f"缺少允许读取的 ROI 清单：{member_name}")
    with io.TextIOWrapper(member, encoding="utf-8") as f:
        return {normalize_roi_id(line) for line in f if line.strip()}


def normalize_roi_id(value: str) -> str:
    value = value.strip()
    if value.lower().startswith("roi") and value[3:].isdigit():
        return value[3:]
    return value


def unit_sphere(lat_lon: np.ndarray) -> np.ndarray:
    lat = np.deg2rad(lat_lon[:, 0])
    lon = np.deg2rad(lat_lon[:, 1])
    cos_lat = np.cos(lat)
    return np.column_stack((cos_lat * np.cos(lon), cos_lat * np.sin(lon), np.sin(lat)))


def iter_top_level_json_object(stream):
    """分块读取顶层 JSON 对象的键值，避免将完整数据集清单载入内存。"""
    decoder = json.JSONDecoder()
    buffer = ""
    position = 0
    eof = False

    def read_more() -> None:
        nonlocal buffer, eof
        chunk = stream.read(1024 * 1024)
        if chunk:
            buffer += chunk
        else:
            eof = True

    def skip_space() -> None:
        nonlocal position
        while True:
            while position < len(buffer) and buffer[position].isspace():
                position += 1
            if position < len(buffer) or eof:
                return
            read_more()

    read_more()
    skip_space()
    if position >= len(buffer) or buffer[position] != "{":
        raise ValueError("数据集 JSON 顶层结构不是对象。")
    position += 1

    while True:
        skip_space()
        if position >= len(buffer):
            if eof:
                raise ValueError("数据集 JSON 在对象结束前意外终止。")
            read_more()
            continue
        if buffer[position] == "}":
            return

        while True:
            try:
                key, key_end = decoder.raw_decode(buffer, position)
                break
            except json.JSONDecodeError:
                if eof:
                    raise
                read_more()
        if not isinstance(key, str):
            raise ValueError("数据集 JSON 顶层键不是字符串。")
        position = key_end
        skip_space()
        if position >= len(buffer):
            if eof:
                raise ValueError("数据集 JSON 键后缺少冒号。")
            read_more()
            continue
        if buffer[position] != ":":
            raise ValueError("数据集 JSON 键值之间缺少冒号。")
        position += 1
        skip_space()

        while True:
            try:
                value, value_end = decoder.raw_decode(buffer, position)
                break
            except json.JSONDecodeError:
                if eof:
                    raise
                read_more()
        position = value_end
        yield key, value
        del key, value

        skip_space()
        if position >= len(buffer):
            if eof:
                raise ValueError("数据集 JSON 在值后意外终止。")
            read_more()
            continue
        if buffer[position] == ",":
            position += 1
        elif buffer[position] == "}":
            return
        else:
            raise ValueError("数据集 JSON 对象成员分隔符异常。")

        if position >= 1024 * 1024:
            buffer = buffer[position:]
            position = 0


def parse_timestamp_entries(values: object) -> list[datetime]:
    if not isinstance(values, list):
        return []
    parsed = []
    for entry in values:
        if isinstance(entry, list) and entry:
            parsed.append(datetime.fromisoformat(str(entry[0])))
    return parsed


def audit_dataset_json(tar: tarfile.TarFile, split: str, allowed_ids: set[str]) -> dict[str, object]:
    member = tar.extractfile(DATASET_JSONS[split])
    if member is None:
        raise ValueError(f"缺少 {split} 数据集 JSON 元数据。")
    roi_ids: set[str] = set()
    instance_count = 0
    s1_frame_count = 0
    s2_frame_count = 0
    target_count = 0
    input_pair_gaps_hours: list[float] = []
    target_pair_gaps_hours: list[float] = []

    with io.TextIOWrapper(member, encoding="utf-8") as f:
        for sample_key, sample in iter_top_level_json_object(f):
            if not isinstance(sample, dict) or not isinstance(sample.get("roi"), list):
                raise ValueError(f"{split} 数据集 JSON 样本缺少 ROI 元数据。")
            roi_id = normalize_roi_id(str(sample["roi"][0]))
            roi_ids.add(roi_id)
            s1_times = parse_timestamp_entries(sample.get("s1"))
            s2_times = parse_timestamp_entries(sample.get("s2_toa"))
            target_times = parse_timestamp_entries(sample.get("target"))
            instance_count += 1
            s1_frame_count += len(s1_times)
            s2_frame_count += len(s2_times)
            target_count += len(target_times)

            if s1_times:
                for s2_time in s2_times:
                    gap = min(abs((s1_time - s2_time).total_seconds()) for s1_time in s1_times) / 3600
                    input_pair_gaps_hours.append(gap)
                for target_time in target_times:
                    gap = min(abs((s1_time - target_time).total_seconds()) for s1_time in s1_times) / 3600
                    target_pair_gaps_hours.append(gap)

    extra_roi_ids = roi_ids - allowed_ids
    missing_roi_ids = allowed_ids - roi_ids
    if extra_roi_ids or missing_roi_ids:
        return {
            "dataset_instances": instance_count,
            "dataset_unique_roi_ids": len(roi_ids),
            "listed_roi_ids": len(allowed_ids),
            "dataset_roi_ids_not_in_explicit_list": len(extra_roi_ids),
            "explicit_list_roi_ids_without_dataset_instances": len(missing_roi_ids),
            "s1_frames": s1_frame_count,
            "s2_toa_frames": s2_frame_count,
            "target_frames": target_count,
            "time_pairing": summarize_time_gaps(input_pair_gaps_hours, target_pair_gaps_hours),
        }
    return {
        "dataset_instances": instance_count,
        "dataset_unique_roi_ids": len(roi_ids),
        "listed_roi_ids": len(allowed_ids),
        "dataset_roi_ids_not_in_explicit_list": 0,
        "explicit_list_roi_ids_without_dataset_instances": 0,
        "s1_frames": s1_frame_count,
        "s2_toa_frames": s2_frame_count,
        "target_frames": target_count,
        "time_pairing": summarize_time_gaps(input_pair_gaps_hours, target_pair_gaps_hours),
    }


def summarize_time_gaps(input_gaps: list[float], target_gaps: list[float]) -> dict[str, object]:
    def summary(values: list[float]) -> dict[str, object]:
        if not values:
            return {"n": 0}
        array = np.asarray(values, dtype=np.float64)
        quantiles = np.quantile(array, [0, 0.5, 0.9, 0.95, 1.0])
        return {
            "n": len(values),
            "gap_hours_quantiles": {
                key: float(value)
                for key, value in zip(("min", "median", "p90", "p95", "max"), quantiles)
            },
            "pairs_within_48_hours": int(np.count_nonzero(array <= 48.0)),
            "fraction_within_48_hours": float(np.mean(array <= 48.0)),
            "pairs_within_72_hours": int(np.count_nonzero(array < 72.0)),
            "fraction_within_72_hours": float(np.mean(array < 72.0)),
        }

    return {
        "all_s2_input_frames_to_nearest_s1_frame": summary(input_gaps),
        "s2_target_frames_to_nearest_s1_frame": summary(target_gaps),
        "interpretation": "仅按 train/validation 元数据时间戳计算最近日期差。官方数据加载器写明 max_diff=2 天，但实现使用 timedelta.days > 2 判拒，因此约为 72 小时边界；同时报告 48 小时更严格筛查。时间接近不代表像素配准或图像质量。",
    }


def nearest_distance_summary(train: np.ndarray, validation: np.ndarray) -> dict[str, object]:
    train_sphere = unit_sphere(train)
    validation_sphere = unit_sphere(validation)
    nearest_angles = []
    for start in range(0, len(validation_sphere), 256):
        dot_products = validation_sphere[start : start + 256] @ train_sphere.T
        nearest_angles.extend(np.arccos(np.clip(np.max(dot_products, axis=1), -1.0, 1.0)))
    distances = EARTH_RADIUS_M * np.asarray(nearest_angles)
    quantiles = np.quantile(distances, [0, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0])
    thresholds_m = [250, 500, 1_000, 2_500, 5_000]
    return {
        "method": "将 ROI 元数据中心经纬度转换到单位球面，以 KD-tree 计算每个 validation 中心到最近 train 中心的大圆距离；这是中心点近邻筛查，不是影像 footprint 重叠证明。",
        "distance_quantiles_km": {
            key: float(value / 1_000)
            for key, value in zip(("min", "p25", "median", "p75", "p90", "p95", "max"), quantiles)
        },
        "validation_centers_with_nearest_train_below_threshold": {
            f"{threshold}m": int(np.count_nonzero(distances < threshold))
            for threshold in thresholds_m
        },
        "exact_center_coordinate_matches": int(
            len(set(map(tuple, train.tolist())) & set(map(tuple, validation.tolist())))
        ),
    }


def audit(archive: Path) -> dict[str, object]:
    with tarfile.open(archive, "r:gz") as tar:
        members = [member for member in tar.getmembers() if member.isfile()]
        extensions = Counter(
            member.name.rsplit(".", 1)[-1].lower() if "." in member.name else "[无扩展名]"
            for member in members
        )
        image_suffixes = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".jp2", ".nitf", ".ntf", ".cphd")
        image_member_count = sum(member.name.lower().endswith(image_suffixes) for member in members)
        if image_member_count:
            raise ValueError("元数据归档中出现影像载荷类文件成员，停止筛查。")
        train_ids = read_ids(tar, SPLIT_FILES["train"])
        validation_ids = read_ids(tar, SPLIT_FILES["validation"])
        if train_ids & validation_ids:
            raise ValueError("train 与 validation ROI 清单存在重复 ID。")

        metadata_file = tar.extractfile(CSV_PATH)
        if metadata_file is None:
            raise ValueError("缺少 ROI 元数据表。")

        expected_split = {roi_id: "train" for roi_id in train_ids}
        expected_split.update({roi_id: "validation" for roi_id in validation_ids})
        table_split_counts = Counter()
        table_extra_counts = Counter()
        rows: dict[str, dict[str, object]] = {
            "train": {},
            "validation": {},
        }
        with io.TextIOWrapper(metadata_file, encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            required = {"split", "roi_id", "latitude", "longitude", "crs"}
            if not required.issubset(reader.fieldnames or []):
                raise ValueError("ROI 元数据表字段与预期不符。")
            for row in reader:
                roi_id = normalize_roi_id(row.get("roi_id") or "")
                split = (row.get("split") or "").strip().lower()
                normalized_split = "validation" if split in {"val", "validation"} else split
                if normalized_split not in {"train", "validation"}:
                    continue
                table_split_counts[normalized_split] += 1
                expected = expected_split.get(roi_id)
                if expected is None:
                    table_extra_counts[normalized_split] += 1
                    continue
                if normalized_split != expected:
                    raise ValueError("train/validation 清单与对应 ROI 元数据的角色不一致。")
                lat = float(row["latitude"])
                lon = float(row["longitude"])
                if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                    raise ValueError("train/validation ROI 坐标超出经纬度范围。")
                if roi_id in rows[expected]:
                    raise ValueError("train/validation 地理元数据中存在重复 ROI 行。")
                rows[expected][roi_id] = {
                    "latitude": lat,
                    "longitude": lon,
                    "crs": (row.get("crs") or "").strip(),
                }

        missing = {
            split: sorted(ids - set(rows[split]))
            for split, ids in (("train", train_ids), ("validation", validation_ids))
        }
        if any(missing.values()):
            raise ValueError(
                "部分 train/validation ROI 在地理元数据表中缺失："
                + ", ".join(f"{key}={len(value)}" for key, value in missing.items())
            )

        train_coords = np.array(
            [[row["latitude"], row["longitude"]] for row in rows["train"].values()],
            dtype=np.float64,
        )
        validation_coords = np.array(
            [[row["latitude"], row["longitude"]] for row in rows["validation"].values()],
            dtype=np.float64,
        )
        crs_unique_counts = {
            split: len({str(row["crs"]) for row in values.values()})
            for split, values in rows.items()
        }
        dataset_json_screens = {
            "train": audit_dataset_json(tar, "train", train_ids),
            "validation": audit_dataset_json(tar, "validation", validation_ids),
        }

    return {
        "audit_id": "allclear_train_val_metadata_20261128_v1",
        "source_url": "https://allclear.cs.cornell.edu/dataset/allclear/metadata.tar.gz",
        "source_archive_sha256": sha256(archive),
        "access_scope": "仅读取 train 与 validation ROI ID 清单、对应的空间记录和 S1/S2 时间配对元数据；共享 ROI 元数据表中的 test 行只用于角色过滤，不参与统计且不保存坐标；未打开 test 专属文件，未读取任何 SAR 或 Optical 影像载荷。",
        "archive_scope_screen": {
            "file_members": len(members),
            "extensions": dict(sorted(extensions.items())),
            "image_payload_members": image_member_count,
            "note": "归档成员名与扩展名仅用于确认文件类型；test 文件内容未打开。",
        },
        "split_counts": {
            "listed_train_roi_ids": len(train_ids),
            "listed_validation_roi_ids": len(validation_ids),
            "matched_train_metadata_rows": len(rows["train"]),
            "matched_validation_metadata_rows": len(rows["validation"]),
            "train_table_role_rows": table_split_counts["train"],
            "validation_table_role_rows": table_split_counts["validation"],
            "train_table_rows_not_in_explicit_list": table_extra_counts["train"],
            "validation_table_rows_not_in_explicit_list": table_extra_counts["validation"],
            "overlapping_roi_ids": 0,
            "missing_metadata_rows": {key: len(value) for key, value in missing.items()},
        },
        "distinct_projected_crs_counts": crs_unique_counts,
        "spatial_screen": nearest_distance_summary(train_coords, validation_coords),
        "dataset_json_screens": dataset_json_screens,
        "interpretation": "AllClear 的角色清单可提供大量显式 train/validation ROI 组；本审计只用元数据中心筛查 train 与 validation 的空间近邻。距离筛查不替代影像 footprint、时间配准、SAR 数据域或任务适配审计。",
        "task_boundary": "AllClear 原始基准是以 Sentinel-2 为主目标的云去除，S1 是辅助模态；没有干净 SAR 恢复参考，不能直接作为 SAR 去斑/超分真值，也不构成 Optical 对 SAR 恢复的增益证据。",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="筛查 AllClear train/validation ROI 元数据，不读取 test 或影像载荷。")
    parser.add_argument("--archive", type=Path, required=True, help="AllClear 官方 metadata.tar.gz 路径")
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("筛查结果.json"))
    args = parser.parse_args()
    result = audit(args.archive)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    spatial = result["spatial_screen"]
    print(f"显式清单：训练 {result['split_counts']['listed_train_roi_ids']} 个，验证 {result['split_counts']['listed_validation_roi_ids']} 个 ROI。")
    print(f"共享元数据表多出训练角色行：{result['split_counts']['train_table_rows_not_in_explicit_list']}；验证角色多出：{result['split_counts']['validation_table_rows_not_in_explicit_list']}。")
    print(f"角色清单交集：{result['split_counts']['overlapping_roi_ids']}；缺少地理记录：{sum(result['split_counts']['missing_metadata_rows'].values())}。")
    print(f"验证中心到最近训练中心的距离中位数：{spatial['distance_quantiles_km']['median']:.3f} km。")
    train_json = result["dataset_json_screens"]["train"]
    validation_json = result["dataset_json_screens"]["validation"]
    no_unlisted_rois = all(
        result_json["dataset_roi_ids_not_in_explicit_list"] == 0
        for result_json in (train_json, validation_json)
    )
    print(
        "训练和验证数据 JSON 均未出现清单外 ROI ID："
        f"{'是' if no_unlisted_rois else '否'}；未出现 tx12 序列的清单 ROI 数：训练 "
        f"{train_json['explicit_list_roi_ids_without_dataset_instances']}、"
        f"验证 {validation_json['explicit_list_roi_ids_without_dataset_instances']}。"
    )
    print(f"结果已写入：{args.output}")


if __name__ == "__main__":
    main()
