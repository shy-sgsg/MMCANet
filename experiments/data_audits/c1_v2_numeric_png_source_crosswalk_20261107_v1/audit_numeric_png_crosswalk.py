#!/usr/bin/env python3
"""使用既有哈希表核对 C1 v2 train/development 来源与数字 PNG 路径。"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = Path(__file__).resolve().parent
C1_DIR = ROOT / "protocols/mmcanet_phase_c1/v2/source_group_split_20261028_v1"
OLD_TABLE = ROOT / "experiments/data_audits/sen1_2_upstream_hash_crosswalk_20261019_v1/source_hash_matches.csv"
EXPECTED_SAMPLE_SHA256 = "6a278e97c161164d9440b5388c5b3fcda87746a56e26adfb7cdf658b94924e57"
ALLOWED_ROLES = {"train", "development"}
EXPECTED_ROLE_COUNTS = {"train": 1600, "development": 200}
HEX512 = re.compile(r"^[0-9a-f]{128}$")
NUMERIC_PNG = re.compile(r"^data/(sar|opt)/(\d+)\.png$", re.IGNORECASE)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_c1_train_development() -> tuple[
    dict[tuple[str, str, str, str], str],
    dict[tuple[str, str, str, str], str],
    Counter[str],
]:
    sample_path = C1_DIR / "sample_manifest.csv"
    manifest = json.loads((C1_DIR / "manifest.json").read_text(encoding="utf-8"))
    declared = manifest["artifacts"]["sample_manifest.csv"]["sha256"]
    observed = sha256(sample_path)
    if declared != EXPECTED_SAMPLE_SHA256 or observed != declared:
        raise ValueError("冻结 C1 sample_manifest.csv 摘要核对失败")

    by_hash: dict[tuple[str, str, str, str], str] = {}
    source_paths: dict[tuple[str, str, str, str], str] = {}
    role_counts: Counter[str] = Counter()
    with sample_path.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            role = row["role"]
            if role not in ALLOWED_ROLES:
                continue
            role_counts[role] += 1
            sample_key = (role, row["source_group_id"], row["patch_id"])
            for modality, path_field, hash_field in (
                ("sar", "sar_relative_path", "sar_file_sha512"),
                ("optical", "optical_relative_path", "optical_file_sha512"),
            ):
                source_hash = row[hash_field].lower()
                if not HEX512.fullmatch(source_hash):
                    raise ValueError("允许角色中存在格式无效的 SHA-512")
                key = (*sample_key, modality)
                if key in by_hash:
                    raise ValueError("允许角色中存在重复的样本/模态键")
                by_hash[key] = source_hash
                source_paths[key] = row[path_field]
    if dict(role_counts) != EXPECTED_ROLE_COUNTS:
        raise ValueError("冻结 C1 train/development 计数与预设值不符")
    return by_hash, source_paths, role_counts


def main() -> None:
    result_path = OUTPUT_DIR / "results.json"
    stdout_path = OUTPUT_DIR / "stdout.txt"
    if result_path.exists() or stdout_path.exists():
        raise FileExistsError("拒绝覆盖已有审计结果")
    if not OLD_TABLE.is_file():
        raise FileNotFoundError("既有来源哈希交叉表不存在")

    c1_hashes, source_paths, role_counts = read_c1_train_development()
    # 多个 C1 记录若共享同一模态摘要，则不把摘要当作唯一连接键。
    c1_targets: dict[tuple[str, str], list[tuple[str, str, str]]] = defaultdict(list)
    for (role, group_id, patch_id, modality), source_hash in c1_hashes.items():
        c1_targets[(modality, source_hash)].append((role, group_id, patch_id))

    candidates: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    matched_rows = 0
    with OLD_TABLE.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            # 对未命中记录只比较摘要字段；不读取或保存其 split、路径或上游来源。
            source_hash = row["local_sha512"].lower()
            if not any((modality, source_hash) in c1_targets for modality in ("sar", "optical")):
                continue
            signature = (row["modality"], source_hash)
            if signature not in c1_targets:
                continue
            matched_rows += 1
            if row["split"] not in {"fixed_train_dev2k", "fixed_val_dev2k"}:
                raise ValueError("旧交叉表命中行包含未允许的历史角色")
            candidates[signature].append(
                {
                    "本地路径": row["local_source"],
                    "上游路径": row["upstream_source"],
                    "上游唯一命中数": row["upstream_exact_match_count"],
                    "历史角色": row["split"],
                }
            )

    unique_matches = 0
    official_path_matches = 0
    numeric_png_matches = 0
    mapped_by_sample: dict[tuple[str, str, str], dict[str, str]] = defaultdict(dict)
    nonunique = 0
    for (role, group_id, patch_id, modality), source_hash in c1_hashes.items():
        signature = (modality, source_hash)
        if len(c1_targets[signature]) != 1:
            continue
        rows = candidates.get(signature, [])
        if len(rows) != 1:
            if len(rows) > 1:
                nonunique += 1
            continue
        unique_matches += 1
        row = rows[0]
        source_path = C1_DIR / "sample_manifest.csv"
        # C1 路径仅为允许角色解析后暂存的值，按样本键重新读取不会访问其他角色载荷。
        c1_source_path = source_paths[(role, group_id, patch_id, modality)]
        if row["上游唯一命中数"] == "1" and row["上游路径"] == c1_source_path:
            official_path_matches += 1
        match = NUMERIC_PNG.fullmatch(row["本地路径"])
        expected_root = "sar" if modality == "sar" else "opt"
        if match and match.group(1).lower() == expected_root:
            numeric_png_matches += 1
            mapped_by_sample[(role, group_id, patch_id)][modality] = row["本地路径"]

    complete_pairs = sum(
        1
        for modalities in mapped_by_sample.values()
        if set(modalities) == {"sar", "optical"}
        and Path(modalities["sar"]).name == Path(modalities["optical"]).name
    )
    total_samples = sum(role_counts.values())
    result = {
        "审计编号": "c1_v2_numeric_png_source_crosswalk_20261107_v1",
        "状态": "既有精确哈希交叉表未能恢复数字 PNG 映射",
        "审计范围": "仅连接冻结 C1 train/development 来源 SHA-512 与历史来源哈希表；不读取影像或其他角色路径/哈希",
        "C1_train_development样本数": dict(role_counts),
        "C1_train_development来源模态数": len(c1_hashes),
        "唯一SHA512命中数": unique_matches,
        "与C1官方来源路径一致数": official_path_matches,
        "本地数字PNG路径匹配数": numeric_png_matches,
        "SAR_optical同编号完整配对数": complete_pairs,
        "匹配历史表行数": matched_rows,
        "多重命中计数": nonunique,
        "C1_sample_manifest_SHA256": sha256(C1_DIR / "sample_manifest.csv"),
        "历史交叉表_SHA256": sha256(OLD_TABLE),
        "本地影像载荷访问": False,
        "confirmation载荷或路径哈希访问": False,
        "test载荷或结果访问": False,
        "模型_checkpoint_训练_GPU访问": False,
        "解释边界": "零精确 SHA-512 命中不证明像素不相同，也不排除重编码后的内容相同；不据此断言数据集身份或地理独立性",
    }
    lines = [
        "C1 v2 train/development 数字 PNG 来源交叉核验",
        f"允许角色样本：训练 {role_counts['train']} 对，开发 {role_counts['development']} 对；共 {len(c1_hashes)} 个来源模态。",
        f"既有 SHA-512 表命中 {matched_rows} 行；唯一连接 {unique_matches} 个模态；上游来源路径一致 {official_path_matches} 个。",
        f"可恢复的数字 PNG 路径 {numeric_png_matches} 个；SAR/optical 同编号完整配对 {complete_pairs}/{total_samples} 对。",
        "结论：这份历史精确字节哈希表不能恢复 C1 v2 train/development 到本地数字 PNG 名称的映射。",
        "本次只读取 CSV/JSON 元数据；未打开图像、数组、checkpoint、confirmation/test 载荷；未训练或使用 GPU。",
        "零命中不等于像素不同，也不排除文件重编码；不改变数据来源或阶段 gate 判断。",
    ]
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    stdout_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
