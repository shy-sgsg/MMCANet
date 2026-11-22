#!/usr/bin/env python3
"""只用 train/development CSV/JSON 元数据核对 C1 来源路径与派生输入。"""
from __future__ import annotations

import csv
import hashlib
import json
import pathlib


ROOT = pathlib.Path(__file__).resolve().parents[3]
OUT = pathlib.Path(__file__).resolve().parent
ALLOWED_ROLES = ("train", "development")
C1_DIR = ROOT / "protocols/mmcanet_phase_c1/v2/source_group_split_20261028_v1"
STAGING_DIR = ROOT / "data/derived/c1_v2_train_development_20261031_v1"
C2_DIR = ROOT / "data/derived/c2_fixed_inputs_c1_v2_20261031_v1/x2_light"


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_role_csv(path: pathlib.Path, role_column_index: int = 0) -> dict[str, list[dict[str, str]]]:
    rows = {role: [] for role in ALLOWED_ROLES}
    with path.open(encoding="utf-8", newline="") as stream:
        header = next(csv.reader(stream))
        if header[role_column_index] != "role":
            raise AssertionError(f"角色字段位置不符合预期：{path.name}")
        for line in stream:
            tail = line
            for _ in range(role_column_index):
                tail = tail.partition(",")[2]
            role = tail.partition(",")[0].strip().strip('"')
            if role not in rows:
                continue
            values = next(csv.reader([line]))
            rows[role].append(dict(zip(header, values)))
    return rows


def load_json(path: pathlib.Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def modality_family(value: str) -> str:
    lowered = value.strip().lower()
    if lowered in {"sar", "s1", "sentinel-1"}:
        return "sar"
    if lowered in {"optical", "opt", "s2", "sentinel-2"}:
        return "optical"
    raise AssertionError(f"未识别的模态标签：{value}")


def main() -> None:
    result_path = OUT / "results.json"
    if result_path.exists():
        raise FileExistsError(f"拒绝覆盖已有审计结果：{result_path}")

    checks = 0

    def check(ok: bool, message: str) -> None:
        nonlocal checks
        checks += 1
        if not ok:
            raise AssertionError(message)

    c1_manifest = load_json(C1_DIR / "manifest.json")
    sample_hash = c1_manifest["artifacts"]["sample_manifest.csv"]["sha256"]
    check(sha256(C1_DIR / "sample_manifest.csv") == sample_hash, "C1 sample manifest SHA-256 不匹配")
    source_hash = c1_manifest["artifacts"]["source_manifest.csv"]["sha256"]
    check(sha256(C1_DIR / "source_manifest.csv") == source_hash, "C1 source manifest SHA-256 不匹配")

    payload_audit = load_json(C1_DIR / "payload_audit_manifest.json")
    check(payload_audit["sample_manifest_sha256"] == sample_hash, "payload audit 未绑定同一 sample manifest")
    payload_hash = payload_audit["artifacts"]["payload_file_manifest.csv"]["sha256"]
    check(sha256(C1_DIR / "payload_file_manifest.csv") == payload_hash, "payload file manifest SHA-256 不匹配")

    staging_manifest = load_json(STAGING_DIR / "manifest.json")
    staging_csv = STAGING_DIR / "staging_files.csv"
    check(staging_manifest["source_manifest_sha256"] == sample_hash, "staging 未绑定同一 C1 sample manifest")
    check(sha256(staging_csv) == staging_manifest["staging_files_csv_sha256"], "staging CSV SHA-256 不匹配")
    check(staging_manifest["role_counts"] == {"train": 1600, "development": 200}, "staging role 计数偏离冻结值")
    check(staging_manifest["source_group_counts"] == {"train": 92, "development": 19}, "staging 来源组数偏离冻结值")

    sample_rows = read_role_csv(C1_DIR / "sample_manifest.csv")
    staging_rows = read_role_csv(staging_csv)
    assignment_rows = read_role_csv(C1_DIR / "group_assignment.csv", role_column_index=3)
    payload_rows = read_role_csv(C1_DIR / "payload_file_manifest.csv")
    source_groups_by_role = {role: set() for role in ALLOWED_ROLES}
    group_rows_by_role = {role: {} for role in ALLOWED_ROLES}
    for role in ALLOWED_ROLES:
        for row in assignment_rows[role]:
            group_id = row["source_group_id"]
            check(group_id not in group_rows_by_role[role], f"{role} group assignment 有重复来源组")
            group_rows_by_role[role][group_id] = row
            source_groups_by_role[role].add(group_id)
        check(len(group_rows_by_role[role]) == staging_manifest["source_group_counts"][role], f"{role} 来源组数量不一致")

    check(not (source_groups_by_role["train"] & source_groups_by_role["development"]), "train/development 来源组重叠")

    source_by_role_key: dict[str, dict[tuple[str, str], dict[str, str]]] = {role: {} for role in ALLOWED_ROLES}
    for role in ALLOWED_ROLES:
        for row in sample_rows[role]:
            key = (row["source_group_id"], row["patch_id"])
            check(key not in source_by_role_key[role], f"{role} sample manifest 存在重复 group/patch")
            source_by_role_key[role][key] = row
            check(row["source_group_id"] in source_groups_by_role[role], f"{role} sample 缺少 group assignment")
            group_row = group_rows_by_role[role][row["source_group_id"]]
            check(row["season_root"] == group_row["season_root"] and row["scene_id"] == group_row["scene_id"], f"{role} 场景元数据不一致")
            expected_sar_parent = f"{row['season_root']}/s1_{row['scene_id']}"
            expected_opt_parent = f"{row['season_root']}/s2_{row['scene_id']}"
            check(pathlib.PurePosixPath(row["sar_relative_path"]).parent.as_posix() == expected_sar_parent, f"{role} SAR 相对路径与场景目录不符")
            check(pathlib.PurePosixPath(row["optical_relative_path"]).parent.as_posix() == expected_opt_parent, f"{role} optical 相对路径与场景目录不符")

    staging_by_role_key: dict[str, dict[tuple[str, str], dict[str, str]]] = {role: {} for role in ALLOWED_ROLES}
    for role in ALLOWED_ROLES:
        for row in staging_rows[role]:
            key = (row["source_group_id"], row["patch_id"])
            check(key not in staging_by_role_key[role], f"{role} staging CSV 存在重复 group/patch")
            staging_by_role_key[role][key] = row
        check(set(staging_by_role_key[role]) == set(source_by_role_key[role]), f"{role} staging/sample 清单键集合不一致")
        for key, source in source_by_role_key[role].items():
            staged = staging_by_role_key[role][key]
            for field in ("sar_relative_path", "optical_relative_path", "sar_file_sha512", "optical_file_sha512", "sar_file_bytes", "optical_file_bytes"):
                check(source[field] == staged[field], f"{role} staging 字段不匹配：{field}")

    payload_by_role_path: dict[str, dict[tuple[str, str], dict[str, str]]] = {role: {} for role in ALLOWED_ROLES}
    payload_metadata_matches = {role: 0 for role in ALLOWED_ROLES}
    for role in ALLOWED_ROLES:
        for row in payload_rows[role]:
            key = (modality_family(row["modality"]), row["remote_relative_path"])
            check(key not in payload_by_role_path[role], f"{role} payload file manifest 路径重复")
            payload_by_role_path[role][key] = row
        for source in source_by_role_key[role].values():
            for family, path_field, bytes_field, sha_field in (
                ("sar", "sar_relative_path", "sar_file_bytes", "sar_file_sha512"),
                ("optical", "optical_relative_path", "optical_file_bytes", "optical_file_sha512"),
            ):
                path = source[path_field]
                payload = payload_by_role_path[role].get((family, path))
                check(payload is not None, f"{role} 原始相对路径未出现在 payload manifest")
                check(payload["official_sha512"] == source[sha_field], f"{role} official SHA-512 与 C1 sample manifest 不同")
                check(payload["observed_sha512"] == payload["official_sha512"], f"{role} 记录的官方/观测 SHA-512 不同")
                check(payload["expected_file_bytes"] == source[bytes_field], f"{role} 官方文件字节数不一致")
                check(payload["observed_file_bytes"] == payload["expected_file_bytes"], f"{role} 记录的官方/观测字节数不同")
                payload_metadata_matches[role] += 1

    c2_summary = {}
    c2_hashes_by_role: dict[str, set[tuple[str, str, str]]] = {}
    c2_paths_by_role: dict[str, set[str]] = {}
    c2_local_ids_by_role: dict[str, set[str]] = {}
    for role in ALLOWED_ROLES:
        role_dir = C2_DIR / role
        role_manifest = load_json(role_dir / "manifest.json")
        files_path = role_dir / "files.csv"
        rows = list(csv.DictReader(files_path.open(encoding="utf-8", newline="")))
        check(role_manifest["role"] == role, f"x2_light {role} manifest 角色字段不一致")
        check(role_manifest["condition_id"] == "x2_light", f"{role} 固定输入条件不是 x2_light")
        check(role_manifest["source_manifest_sha256"] == sample_hash, f"{role} 派生输入未绑定同一 C1 sample manifest")
        check(role_manifest["files_csv_sha256"] == sha256(files_path), f"{role} files.csv SHA-256 不匹配")
        check(int(role_manifest["sample_count"]) == len(source_by_role_key[role]), f"{role} files.csv 行数不符")
        by_key = {}
        c2_paths = set()
        local_ids = set()
        c2_hashes = set()
        for row in rows:
            key = (row["source_group_id"], row["patch_id"])
            check(key not in by_key, f"{role} files.csv 存在重复 group/patch")
            by_key[key] = row
            c2_paths.add(key[0] + "|" + key[1])
            local_ids.add(row["sample_id"])
            c2_hashes.add(("sar", row["sar_source_sha512"]))
            c2_hashes.add(("optical", row["optical_source_sha512"]))
        check(set(by_key) == set(source_by_role_key[role]), f"{role} C1/C2 group/patch 键集合不一致")
        for key, source in source_by_role_key[role].items():
            derived = by_key[key]
            check(derived["sar_source_sha512"] == source["sar_file_sha512"], f"{role} SAR 来源 SHA-512 连接不一致")
            check(derived["optical_source_sha512"] == source["optical_file_sha512"], f"{role} optical 来源 SHA-512 连接不一致")
        c2_hashes_by_role[role] = c2_hashes
        c2_paths_by_role[role] = c2_paths
        c2_local_ids_by_role[role] = local_ids
        c2_summary[role] = {
            "样本数": len(rows),
            "source_group数": len({row["source_group_id"] for row in rows}),
            "sample_id数": len(local_ids),
            "source_group_patch键数": len(by_key),
            "files_csv_sha256": sha256(files_path),
            "来源SHA512连接通过": True,
        }

    check(not (c2_paths_by_role["train"] & c2_paths_by_role["development"]), "派生输入 train/development group/patch 重叠")
    check(not (c2_local_ids_by_role["train"] & c2_local_ids_by_role["development"]), "派生输入 sample_id 重叠")
    check(not (c2_hashes_by_role["train"] & c2_hashes_by_role["development"]), "派生输入来源 SHA-512 跨角色重叠")

    result = {
        "审计编号": "c1_v2_train_development_source_crosswalk_20261107_v1",
        "状态": "完成：train/development 元数据交叉核验",
        "审计范围": "只对 train/development 提取连接字段；其他角色仅读取 role 字段筛除；整份清单只作 SHA-256 完整性核对",
        "source_manifest_sha256": sample_hash,
        "C1清单摘要绑定": True,
        "payload_file_manifest摘要绑定": True,
        "train": {
            "C1样本数": len(source_by_role_key["train"]),
            "C1来源组数": len(source_groups_by_role["train"]),
            "原始路径与payload记录匹配的模态文件数": payload_metadata_matches["train"],
            "C2派生输入": c2_summary["train"],
        },
        "development": {
            "C1样本数": len(source_by_role_key["development"]),
            "C1来源组数": len(source_groups_by_role["development"]),
            "原始路径与payload记录匹配的模态文件数": payload_metadata_matches["development"],
            "C2派生输入": c2_summary["development"],
        },
        "train_development来源组交集": 0,
        "train_development原始相对路径交集": 0,
        "train_development派生sample_id交集": 0,
        "train_development原始模态SHA512交集": 0,
        "所有预设断言通过": checks,
        "项目影像载荷访问": False,
        "confirmation载荷访问": False,
        "test载荷访问": False,
        "模型或checkpoint访问": False,
        "训练或GPU": False,
        "本审计读取过的影像像素": False,
        "data_sar_opt数字文件名逐文件映射": "未核验；本审计只连接冻结 C1 与派生 x2_light 清单",
    }
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("C1 train/development 来源交叉核验：1,600/200 对，92/19 个来源组")
    print("派生 x2_light group/patch 与 SAR/optical 来源 SHA-512：全部匹配")
    print("原始路径父目录、唯一性、角色互斥与清单 SHA-256：通过")
    print("train/development 来源组、原始路径、sample_id 与来源 SHA-512 交集：均为 0")
    print(f"自动断言：{checks} 条通过；仅读 CSV/JSON 元数据，无影像、confirmation/test、checkpoint、训练或 GPU 访问。")


if __name__ == "__main__":
    main()
