#!/usr/bin/env python3
"""对 AllClear train/validation 清单与已发布样本 JSON 做元数据交叉筛查。"""
from __future__ import annotations

import argparse
import csv
from datetime import UTC, datetime
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile

HERE = Path(__file__).resolve().parent
COMMON_DIR = HERE.parent / "allclear_train_val_metadata_20261128_v1"
sys.path.insert(0, str(COMMON_DIR))
import audit_allclear_train_val_metadata as common  # noqa: E402

TRAIN_LIST = "metadata/rois/train_rois_19k.txt"
VALIDATION_LIST = "metadata/rois/val_rois_1k.txt"
ROI_TABLE = "metadata/rois/rois_metadata.csv"
DATASET_FILES = {
    "train_tx12_s2_s1": "metadata/datasets/train_tx12_s2-s1_100pct.json",
    "train_tx3_s2_s1": "metadata/datasets/train_tx3_s2-s1_100pct.json",
    "train_tx3_s2_s1_landsat": "metadata/datasets/train_tx3_s2-s1-landsat_100pct.json",
    "validation_tx12_s2_s1": "metadata/datasets/val_tx12_s2-s1_100pct.json",
    "validation_tx3_s2_s1_landsat": "metadata/datasets/val_tx3_s2-s1-landsat_100pct.json",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_dataset_rois(tar: tarfile.TarFile, member_name: str) -> tuple[set[str], int]:
    member = tar.extractfile(member_name)
    if member is None:
        raise ValueError(f"缺少 train/validation 数据集 JSON：{member_name}")
    roi_ids: set[str] = set()
    instance_count = 0
    with io.TextIOWrapper(member, encoding="utf-8") as stream:
        for _, sample in common.iter_top_level_json_object(stream):
            if not isinstance(sample, dict) or not isinstance(sample.get("roi"), list):
                raise ValueError(f"样本缺少 ROI 元数据：{member_name}")
            roi_ids.add(common.normalize_roi_id(str(sample["roi"][0])))
            instance_count += 1
    return roi_ids, instance_count


def summarize_dataset(
    roi_ids: set[str],
    instance_count: int,
    listed_ids: set[str],
    extra_train_ids: set[str],
) -> dict[str, int]:
    return {
        "instances": instance_count,
        "unique_roi_ids": len(roi_ids),
        "listed_roi_ids": len(listed_ids),
        "listed_roi_ids_without_instances": len(listed_ids - roi_ids),
        "dataset_roi_ids_outside_explicit_list": len(roi_ids - listed_ids),
        "dataset_roi_ids_in_extra_train_table_rows": len(roi_ids & extra_train_ids),
    }


def audit(archive: Path) -> dict[str, object]:
    with tarfile.open(archive, "r:gz") as tar:
        train_ids = common.read_ids(tar, TRAIN_LIST)
        validation_ids = common.read_ids(tar, VALIDATION_LIST)
        if train_ids & validation_ids:
            raise ValueError("显式 train/validation 清单存在交集。")

        table_file = tar.extractfile(ROI_TABLE)
        if table_file is None:
            raise ValueError("缺少共享 ROI 元数据表。")
        role_rows = {"train": 0, "validation": 0}
        extra_train_rows = 0
        extra_train_ids: set[str] = set()
        with io.TextIOWrapper(table_file, encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            required = {"split", "roi_id"}
            if not required.issubset(reader.fieldnames or []):
                raise ValueError("共享 ROI 元数据表缺少 split 或 roi_id 字段。")
            for row in reader:
                role = (row.get("split") or "").strip().lower()
                role = "validation" if role in {"val", "validation"} else role
                if role not in role_rows:
                    continue
                role_rows[role] += 1
                roi_id = common.normalize_roi_id(row.get("roi_id") or "")
                if role == "train" and roi_id not in train_ids:
                    extra_train_rows += 1
                    extra_train_ids.add(roi_id)

        datasets: dict[str, dict[str, int]] = {}
        dataset_roi_sets: dict[str, set[str]] = {}
        for dataset_name, member_name in DATASET_FILES.items():
            roi_ids, instance_count = read_dataset_rois(tar, member_name)
            dataset_roi_sets[dataset_name] = roi_ids
            listed = validation_ids if dataset_name.startswith("validation_") else train_ids
            extras = extra_train_ids if dataset_name.startswith("train_") else set()
            datasets[dataset_name] = summarize_dataset(roi_ids, instance_count, listed, extras)

        missing_train_tx12 = train_ids - dataset_roi_sets["train_tx12_s2_s1"]
        missing_validation_tx12 = validation_ids - dataset_roi_sets["validation_tx12_s2_s1"]
        missing_train_in_other = (
            (missing_train_tx12 & dataset_roi_sets["train_tx3_s2_s1"])
            | (missing_train_tx12 & dataset_roi_sets["train_tx3_s2_s1_landsat"])
        )
        missing_validation_in_other = (
            missing_validation_tx12
            & dataset_roi_sets["validation_tx3_s2_s1_landsat"]
        )

        member_names = [TRAIN_LIST, VALIDATION_LIST, ROI_TABLE, *DATASET_FILES.values()]
        member_times = {}
        for name in member_names:
            member = tar.getmember(name)
            member_times[name] = datetime.fromtimestamp(member.mtime, tz=UTC).isoformat()

    return {
        "audit_id": "allclear_train_val_eligibility_crosscheck_20261128_v2",
        "source_url": "https://allclear.cs.cornell.edu/dataset/allclear/metadata.tar.gz",
        "source_archive_sha256": file_sha256(archive),
        "access_scope": "仅读取显式 train/validation 清单、共享 ROI 表中的 train/validation 角色与五份 train/validation 数据集 JSON；未打开 test 专属清单或 JSON，未读取任何影像载荷。",
        "protected_access_flags": {
            "confirmation_payload_accessed": False,
            "test_payload_accessed": False,
            "test_metrics_accessed": False,
            "optical_payload_accessed": False,
        },
        "read_members": member_times,
        "explicit_roster_counts": {
            "train": len(train_ids),
            "validation": len(validation_ids),
            "overlap": len(train_ids & validation_ids),
        },
        "shared_table_counts": {
            "train_role_rows": role_rows["train"],
            "validation_role_rows": role_rows["validation"],
            "train_rows_outside_explicit_roster": extra_train_rows,
            "distinct_train_roi_ids_outside_explicit_roster": len(extra_train_ids),
            "duplicate_extra_train_rows": extra_train_rows - len(extra_train_ids),
            "extra_train_ids_in_train_tx12_or_tx3_json": len(
                extra_train_ids
                & (
                    dataset_roi_sets["train_tx12_s2_s1"]
                    | dataset_roi_sets["train_tx3_s2_s1"]
                    | dataset_roi_sets["train_tx3_s2_s1_landsat"]
                )
            ),
        },
        "dataset_json_coverage": datasets,
        "crosscheck": {
            "train_rois_missing_tx12_but_present_in_any_train_tx3_json": len(missing_train_in_other),
            "validation_rois_missing_tx12_but_present_in_validation_tx3_json": len(missing_validation_in_other),
            "train_rois_without_any_checked_train_sequence": len(
                train_ids
                - dataset_roi_sets["train_tx12_s2_s1"]
                - dataset_roi_sets["train_tx3_s2_s1"]
                - dataset_roi_sets["train_tx3_s2_s1_landsat"]
            ),
            "validation_rois_without_any_checked_validation_sequence": len(
                validation_ids
                - dataset_roi_sets["validation_tx12_s2_s1"]
                - dataset_roi_sets["validation_tx3_s2_s1_landsat"]
            ),
            "roi_ids_or_coordinates_saved": False,
        },
        "interpretation": "34 条额外 train 角色地理记录是 34 个不同 ROI，且未出现在任何已检查的 train tx12/tx3 样本 JSON；它们不属于可执行的已发布训练样本。334/13 个没有 tx12 序列的 train/validation ROI 也没有出现在本归档列出的其他对应 tx3 JSON。该结果支持使用显式最终角色清单与实际样本 JSON 定义候选 cohort；不推断额外 ROI 的逐项剔除原因。",
        "task_boundary": "本审计只核查样本角色与任务 JSON 覆盖，不证明 SAR/Optical 像素配准、时间稳定性、SAR 目标质量或 Optical 对 SAR 恢复的增益。",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="交叉筛查 AllClear train/validation ROI 清单与样本 JSON，不读取 test 或影像载荷。"
    )
    parser.add_argument("--archive", type=Path, required=True, help="官方元数据归档路径")
    parser.add_argument(
        "--output", type=Path, default=HERE / "筛查结果.json", help="聚合结果保存路径"
    )
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        parser.error(f"拒绝覆盖已有结果路径：{args.output}；请指定新的输出路径。")
    result = audit(args.archive)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    counts = result["shared_table_counts"]
    crosscheck = result["crosscheck"]
    print(
        "角色差异：额外 train 行 "
        f"{counts['train_rows_outside_explicit_roster']}，其中不同 ROI "
        f"{counts['distinct_train_roi_ids_outside_explicit_roster']}，"
        f"出现在所查训练 JSON 的数量 {counts['extra_train_ids_in_train_tx12_or_tx3_json']}。"
    )
    print(
        "序列覆盖：缺少 tx12 且出现在其它所查序列 JSON 的 ROI，训练/验证为 "
        f"{crosscheck['train_rois_missing_tx12_but_present_in_any_train_tx3_json']}/"
        f"{crosscheck['validation_rois_missing_tx12_but_present_in_validation_tx3_json']}。"
    )
    print(
        "候选 ROI 清单数：所查 train/validation 序列 JSON 均未覆盖的数量为 "
        f"{crosscheck['train_rois_without_any_checked_train_sequence']}/"
        f"{crosscheck['validation_rois_without_any_checked_validation_sequence']}。"
    )
    print(f"聚合结果已写入：{args.output}")


if __name__ == "__main__":
    main()
