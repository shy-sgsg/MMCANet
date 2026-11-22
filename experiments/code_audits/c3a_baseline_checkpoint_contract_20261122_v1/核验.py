#!/usr/bin/env python3
"""核对冻结 C3-A 协议与既有 C2 SAR-only checkpoint 的结构绑定。"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from MMCANetV2.phase_c2_matched import C2MatchedJoint


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    protocol_path = ROOT / "protocols/mmcanet_phase_c3a/sar_only_flow_c1_v2_20261105_v1.json"
    runner_path = ROOT / "scripts/run_phase_c3a_sar_flow.py"
    flow_path = ROOT / "MMCANetV2/phase_c3a_flow.py"
    c2_runner_path = ROOT / "MMCANetV2/phase_c2_runner.py"
    c2_model_path = ROOT / "MMCANetV2/phase_c2_matched.py"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol_sha = sha256(protocol_path)
    runner_text = runner_path.read_text(encoding="utf-8")
    runner_protocol_sha = re.search(
        r'EXPECTED_PROTOCOL_SHA256\s*=\s*["\']([0-9a-f]{64})["\']', runner_text
    )
    require(runner_protocol_sha is not None, "无法从 C3-A 入口读取冻结协议绑定")
    require(runner_protocol_sha.group(1) == protocol_sha, "C3-A 入口与冻结协议哈希不一致")

    source_manifest = ROOT / protocol["split"]["source_manifest_path"]
    source_sha = sha256(source_manifest)
    require(source_sha == protocol["split"]["source_manifest_sha256"], "C1 来源清单哈希不匹配")

    baseline = ROOT / protocol["comparison"]["baseline_run"]
    baseline_manifest_path = baseline / "manifest.json"
    baseline_manifest = json.loads(baseline_manifest_path.read_text(encoding="utf-8"))
    context = baseline_manifest["context"]
    require(baseline_manifest.get("status") == "complete", "C2 SAR-only 基线运行未完成")
    require(context.get("condition_id") == protocol["data"]["condition_id"], "基线退化条件不匹配")
    require(context.get("arm") == "sar_only" and context.get("optical_enabled") is False,
            "基线不是 SAR-only 条件")
    require(baseline_manifest.get("confirmation_payload_accessed") is False,
            "基线清单的 confirmation 载荷标记异常")
    require(baseline_manifest.get("test_set_accessed") is False
            and baseline_manifest.get("test_set_used") is False,
            "基线清单的 test 标记异常")

    fixed_input_hashes: dict[str, dict[str, str]] = {}
    fixed_root = ROOT / protocol["data"]["fixed_input_root"]
    for role in ("train", "development"):
        role_root = fixed_root / role
        manifest_sha = sha256(role_root / "manifest.json")
        files_sha = sha256(role_root / "files.csv")
        expected = context["input_records"][role]
        require(manifest_sha == expected["manifest_sha256"], f"{role} 固定输入 manifest 哈希不匹配")
        require(files_sha == expected["files_csv_sha256"], f"{role} 固定输入 files.csv 哈希不匹配")
        fixed_input_hashes[role] = {
            "manifest_sha256": manifest_sha,
            "files_csv_sha256": files_sha,
        }

    checkpoint_path = baseline / "checkpoints/best.pt"
    checkpoint_sha = sha256(checkpoint_path)
    require(checkpoint_sha == protocol["comparison"]["baseline_checkpoint_sha256"],
            "C2 SAR-only checkpoint 与冻结协议哈希不匹配")
    require(checkpoint_sha == baseline_manifest.get("best_checkpoint_sha256"),
            "C2 SAR-only checkpoint 与基线运行清单哈希不匹配")
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    require(checkpoint.get("epoch") == baseline_manifest.get("best_epoch"),
            "checkpoint epoch 与基线运行清单不匹配")
    require(checkpoint.get("context", {}).get("condition_id") == protocol["data"]["condition_id"],
            "checkpoint 条件与冻结协议不匹配")

    model_config = context["model_config"]
    model = C2MatchedJoint(scale=2, optical_enabled=False, **model_config)
    incompatible = model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    require(not incompatible.missing_keys and not incompatible.unexpected_keys,
            "checkpoint 严格加载仍有缺失或多余参数")
    finite_tensors = all(
        not torch.is_floating_point(value) or bool(torch.isfinite(value).all())
        for value in checkpoint["model_state_dict"].values()
    )
    require(finite_tensors, "checkpoint 含非有限浮点参数")
    model.eval()
    with torch.inference_mode():
        synthetic_output = model(torch.zeros((1, 1, 128, 128), dtype=torch.float32))
    require(tuple(synthetic_output.shape) == (1, 1, 256, 256), "合成零输入输出尺寸不匹配")
    require(bool(torch.isfinite(synthetic_output).all()), "合成零输入输出含非有限值")

    flow_run = ROOT / "experiments/phase_c3a_sar_only_flow_c1_v2_20261105_v1"
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    result = {
        "结论": "通过",
        "检查范围": "协议、运行清单、固定输入元数据、checkpoint 结构和合成零输入输出尺寸",
        "仓库提交": head,
        "Python 版本": sys.version.split()[0],
        "PyTorch 版本": torch.__version__,
        "CUDA 运行时版本": torch.version.cuda,
        "CUDA_VISIBLE_DEVICES": "已由命令设为空字符串",
        "CUDA 是否初始化": torch.cuda.is_initialized(),
        "C3-A 协议 SHA-256": protocol_sha,
        "C3-A 入口绑定协议": runner_protocol_sha.group(1),
        "C1 来源清单 SHA-256": source_sha,
        "固定输入元数据哈希": fixed_input_hashes,
        "基线运行状态": baseline_manifest["status"],
        "基线最优轮次": checkpoint["epoch"],
        "基线模型参数量": sum(parameter.numel() for parameter in model.parameters()),
        "checkpoint SHA-256": checkpoint_sha,
        "checkpoint 严格加载": "通过",
        "合成零输入输出尺寸": list(synthetic_output.shape),
        "合成零输入输出有限": True,
        "C3-A Flow 运行目录已存在": flow_run.exists(),
        "读取数组载荷": False,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
