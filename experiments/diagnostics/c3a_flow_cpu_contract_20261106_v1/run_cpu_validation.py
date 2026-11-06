#!/usr/bin/env python3
"""以合成张量验证 C3-A Heun 数值阶与逐样本源噪声确定性。"""
from __future__ import annotations

import hashlib
import json
import math
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch import nn

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
RUN_DIRECTORY = Path(__file__).resolve().parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from MMCANetV2.phase_c3a_flow import sample_heun
from scripts.run_phase_c3a_sar_flow import _batch_noise, _noise_for_sample

PROTOCOL = REPOSITORY_ROOT / "protocols/mmcanet_phase_c3a/sar_only_flow_c1_v2_20261105_v1.json"
EXPECTED_PROTOCOL_SHA256 = "147c361b3e91bc4cd8d76ce5c464ad9d2137f2d06b38f31a0bc7c7a7e93101a7"


class LinearVelocity(nn.Module):
    """提供解析解已知的合成速度场 dx/dt=x。"""

    def __init__(self) -> None:
        super().__init__()
        self.function_evaluations = 0

    def forward(self, state: torch.Tensor, condition: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        self.function_evaluations += 1
        return state


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_state() -> dict[str, object]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "head": commit.stdout.strip() if commit.returncode == 0 else None,
        "dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
    }


def run() -> dict[str, object]:
    protocol_hash = _sha256(PROTOCOL)
    if protocol_hash != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError("冻结 C3-A 协议 SHA-256 与登记值不符")

    condition = torch.zeros((1, 1, 1, 1), dtype=torch.float32, device="cpu")
    source_noise = torch.ones_like(condition)
    exact_solution = 0.1 * math.e
    steps_list = [16, 32, 64]
    errors: list[float] = []
    function_evaluations: list[int] = []
    for steps in steps_list:
        model = LinearVelocity()
        prediction = sample_heun(
            model,
            condition,
            source_noise,
            steps=steps,
            noise_sigma=0.1,
        )
        errors.append(abs(float(prediction.item()) - exact_solution))
        function_evaluations.append(model.function_evaluations)
    error_ratios = [errors[i] / errors[i + 1] for i in range(len(errors) - 1)]
    if function_evaluations != [2 * steps for steps in steps_list]:
        raise AssertionError("Heun 模型调用次数不等于 2×积分步数")
    if not all(3.8 < ratio < 4.2 for ratio in error_ratios):
        raise AssertionError(f"Heun 二阶误差收敛范围不符：{error_ratios}")

    seed = 20261105
    sample_a = _noise_for_sample("合成样本甲", seed=seed, device=torch.device("cpu"))
    replay_a = _noise_for_sample("合成样本甲", seed=seed, device=torch.device("cpu"))
    ordered = _batch_noise(["合成样本甲", "合成样本乙"], seed=seed, device=torch.device("cpu"))
    reversed_order = _batch_noise(["合成样本乙", "合成样本甲"], seed=seed, device=torch.device("cpu"))
    noise_repeatable = torch.equal(sample_a, replay_a)
    noise_batch_order_invariant = torch.equal(ordered[0:1], reversed_order[1:2])
    noise_sample_specific = not torch.equal(ordered[0:1], ordered[1:2])
    if not all((noise_repeatable, noise_batch_order_invariant, noise_sample_specific)):
        raise AssertionError("逐样本源噪声确定性检查失败")

    cuda_initialized = torch.cuda.is_initialized()
    if cuda_initialized:
        raise AssertionError("CPU 数值诊断意外初始化 CUDA")

    source_files = [
        REPOSITORY_ROOT / "MMCANetV2/phase_c3a_flow.py",
        REPOSITORY_ROOT / "scripts/run_phase_c3a_sar_flow.py",
        REPOSITORY_ROOT / "scripts/evaluate_c3a_multisample.py",
        REPOSITORY_ROOT / "tests/test_phase_c3a_flow.py",
        REPOSITORY_ROOT / "tests/test_c3a_multisample_eval.py",
        Path(__file__).resolve(),
    ]
    return {
        "记录格式": "c3a_flow_cpu_contract_diagnostic_v1",
        "运行编号": "c3a_flow_cpu_contract_20261106_v1",
        "状态": "通过",
        "类型": "合成张量数值与确定性软件诊断；不是模型性能实验",
        "创建时间 UTC": datetime.now(timezone.utc).isoformat(),
        "设备": "CPU",
        "CUDA 已初始化": cuda_initialized,
        "Git": _git_state(),
        "环境": {
            "Python": sys.version,
            "PyTorch": torch.__version__,
            "NumPy": __import__("numpy").__version__,
            "平台": platform.platform(),
        },
        "设计参数": {
            "微分方程": "dx/dt=x",
            "初值": 0.1,
            "解析终值": exact_solution,
            "Heun 步数": steps_list,
            "误差比验收区间": [3.8, 4.2],
            "源噪声种子": seed,
        },
        "结果": {
            "Heun 绝对误差": errors,
            "步数翻倍误差比": error_ratios,
            "模型调用次数": function_evaluations,
            "噪声重复一致": noise_repeatable,
            "噪声不依赖 batch 顺序": noise_batch_order_invariant,
            "不同样本噪声不同": noise_sample_specific,
        },
        "协议": {
            "路径": str(PROTOCOL.relative_to(REPOSITORY_ROOT)),
            "SHA-256": protocol_hash,
        },
        "代码 SHA-256": {str(path.relative_to(REPOSITORY_ROOT)): _sha256(path) for path in source_files},
        "数据载荷访问": {
            "train": False,
            "development": False,
            "confirmation": False,
            "test": False,
            "optical": False,
        },
        "解释边界": [
            "线性合成向量场只验证 Heun 实现的数值阶，不证明训练模型的图像重建质量。",
            "逐样本源噪声检查只验证 seed/sample_id 映射，不构成采样多样性或后验校准证据。",
        ],
    }


if __name__ == "__main__":
    manifest = run()
    manifest_path = RUN_DIRECTORY / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    errors = manifest["结果"]["Heun 绝对误差"]
    ratios = manifest["结果"]["步数翻倍误差比"]
    print("CPU 数值诊断：通过")
    print("Heun 步数：", manifest["设计参数"]["Heun 步数"])
    print("Heun 绝对误差：", [format(value, ".10g") for value in errors])
    print("步数翻倍误差比：", [format(value, ".6f") for value in ratios])
    print("模型调用次数：", manifest["结果"]["模型调用次数"], "（均为 2×步数）")
    print("固定样本噪声：重复一致、batch 顺序不影响、不同样本不同")
    print("CUDA 已初始化：", manifest["CUDA 已初始化"])
