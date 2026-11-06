#!/usr/bin/env python3
"""核验冻结 C3-A train/development 的 SAR LR 与 target 固定数组。"""
from __future__ import annotations

import hashlib
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
RUN_DIRECTORY = Path(__file__).resolve().parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from MMCANetV2.phase_c2_data import C2FixedDataset, sha256_file
from MMCANetV2.phase_c3a_flow import SarTargetOnlyDataset

PROTOCOL_PATH = REPOSITORY_ROOT / "protocols/mmcanet_phase_c3a/sar_only_flow_c1_v2_20261105_v1.json"
EXPECTED_PROTOCOL_SHA256 = "147c361b3e91bc4cd8d76ce5c464ad9d2137f2d06b38f31a0bc7c7a7e93101a7"
MINIMUM_FREE_BYTES = 2 * 1024**3


def _git_state() -> dict[str, object]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT,
        capture_output=True, text=True, check=False,
    )
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=REPOSITORY_ROOT,
        capture_output=True, text=True, check=False,
    )
    return {
        "HEAD": commit.stdout.strip() if commit.returncode == 0 else None,
        "工作区有未提交内容": bool(status.stdout.strip()) if status.returncode == 0 else None,
    }


def _array_bytes(dataset: SarTargetOnlyDataset) -> dict[str, int]:
    totals = {"lr": 0, "target": 0}
    for row in dataset.rows:
        for name in totals:
            totals[name] += (dataset.root / name / f"{row['sample_id']}.npy").stat().st_size
    return totals


def run() -> dict[str, object]:
    manifest_path = RUN_DIRECTORY / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"预检运行清单已存在，拒绝覆盖：{manifest_path}")
    RUN_DIRECTORY.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "记录格式": "c3a_cpu_train_development_data_preflight_v1",
        "运行编号": "c3a_cpu_data_preflight_20261106_v1",
        "状态": "运行中",
        "开始时间 UTC": datetime.now(timezone.utc).isoformat(),
        "设备": "CPU",
        "确认集载荷访问": False,
        "测试集载荷访问": False,
        "测试集使用": False,
        "光学载荷读取": False,
        "数据角色": ["train", "development"],
        "载荷数组": ["lr", "target"],
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    try:
        free_bytes = shutil.disk_usage(REPOSITORY_ROOT).free
        if free_bytes < MINIMUM_FREE_BYTES:
            raise RuntimeError("项目分区余量低于 2 GiB，停止读取数据")
        protocol_hash = sha256_file(PROTOCOL_PATH)
        if protocol_hash != EXPECTED_PROTOCOL_SHA256:
            raise ValueError("冻结 C3-A 训练协议 SHA-256 不匹配")
        protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        source_manifest = REPOSITORY_ROOT / protocol["split"]["source_manifest_path"]
        source_hash = sha256_file(source_manifest)
        if source_hash != protocol["split"]["source_manifest_sha256"]:
            raise ValueError("冻结 C1 v2 来源清单 SHA-256 不匹配")
        fixed_root = REPOSITORY_ROOT / protocol["data"]["fixed_input_root"]
        role_records: dict[str, dict[str, object]] = {}
        payload_access = {
            "train": False,
            "development": False,
            "confirmation": False,
            "test": False,
            "optical": False,
        }
        manifest["载荷访问"] = payload_access
        cuda_initialized_before = torch.cuda.is_initialized()
        if cuda_initialized_before:
            raise RuntimeError("CPU 预检开始前 CUDA 已初始化")
        for role in ("train", "development"):
            payload_access[role] = True
            manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            fixed = C2FixedDataset(
                fixed_root / role,
                source_manifest,
                role=role,
                scale=2,
                verify_hashes_on_read=False,
            )
            if fixed.manifest.get("condition_id") != "x2_light":
                raise ValueError(f"{role} 固定输入不是冻结 x2_light 条件")
            dataset = SarTargetOnlyDataset(fixed)
            expected = protocol["split"][role]
            group_count = len({row["source_group_id"] for row in dataset.rows})
            if len(dataset) != int(expected["sample_count"]):
                raise ValueError(f"{role} 样本数与冻结协议不匹配")
            if group_count != int(expected["source_group_count"]):
                raise ValueError(f"{role} 来源组数与冻结协议不匹配")
            dataset.verify_all()
            role_records[role] = {
                "样本数": len(dataset),
                "来源组数": group_count,
                "已验证数组文件数": len(dataset) * 2,
                "已验证数组字节数": _array_bytes(dataset),
                "固定输入 manifest SHA-256": dataset.manifest_sha256,
                "固定输入 files.csv SHA-256": dataset.files_csv_sha256,
                "读取数组": ["lr", "target"],
            }
            print(
                f"已核验 {role}：{len(dataset)} 个样本、{group_count} 个来源组；"
                f"仅验证 SAR LR 与 noisy pseudo-target 数组。",
                flush=True,
            )
        cuda_initialized_after = torch.cuda.is_initialized()
        if cuda_initialized_after:
            raise RuntimeError("CPU 预检意外初始化 CUDA")
        manifest.update(
            {
                "状态": "通过",
                "结束时间 UTC": datetime.now(timezone.utc).isoformat(),
                "Python": sys.version,
                "PyTorch": torch.__version__,
                "NumPy": np.__version__,
                "平台": platform.platform(),
                "Git": _git_state(),
                "冻结训练协议路径": str(PROTOCOL_PATH.relative_to(REPOSITORY_ROOT)),
                "冻结训练协议 SHA-256": protocol_hash,
                "冻结来源清单 SHA-256": source_hash,
                "数据记录": role_records,
                "CUDA 预检前已初始化": cuda_initialized_before,
                "CUDA 预检后已初始化": cuda_initialized_after,
                "读取期间最低磁盘余量字节": MINIMUM_FREE_BYTES,
                "载荷访问": payload_access,
                "解释边界": [
                    "本次只验证冻结 train/development 的 lr 与 target 文件哈希、尺寸、dtype、有限性和值域。",
                    "未读取 optical 数组；来源 CSV 与 files.csv 中的光学身份/哈希仅作为元数据处理。",
                    "完整性通过不代表模型效果、SAR 辐射定标、物理散斑模型或独立泛化已获验证。",
                ],
                "代码 SHA-256": {
                    "MMCANetV2/phase_c2_data.py": sha256_file(REPOSITORY_ROOT / "MMCANetV2/phase_c2_data.py"),
                    "MMCANetV2/phase_c3a_flow.py": sha256_file(REPOSITORY_ROOT / "MMCANetV2/phase_c3a_flow.py"),
                    str(Path(__file__).resolve().relative_to(REPOSITORY_ROOT)): sha256_file(Path(__file__).resolve()),
                },
            }
        )
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return manifest
    except Exception as error:
        manifest.update(
            {
                "状态": "失败",
                "结束时间 UTC": datetime.now(timezone.utc).isoformat(),
                "失败类型": type(error).__name__,
                "失败原因": str(error),
            }
        )
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        raise


if __name__ == "__main__":
    result = run()
    print(
        f"train/development CPU 完整性预检通过：{sum(int(row['已验证数组文件数']) for row in result['数据记录'].values())} 个数组文件；"
        "confirmation/test/optical 载荷未读取。",
        flush=True,
    )
