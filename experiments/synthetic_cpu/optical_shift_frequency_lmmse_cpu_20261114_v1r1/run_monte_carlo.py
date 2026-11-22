#!/usr/bin/env python3
"""运行未知高斯平移下频率域 LMMSE 增益的复高斯 Monte Carlo 核验。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EXPERIMENT_DIR / "results"
PROTOCOL_PATH = EXPERIMENT_DIR / "protocol.md"
MODEL_PATH = EXPERIMENT_DIR / "lmmse_shift_model.py"
SCRIPT_PATH = Path(__file__).resolve()
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from cli_zh import ChineseArgumentParser  # noqa: E402
from lmmse_shift_model import lmmse_risk_terms, simulate_lmmse_case  # noqa: E402


PROTOCOL_ID = "optical_shift_frequency_lmmse_cpu_20261114_v1r1"
BASE_SEED = 20261114
SAMPLES_PER_CASE = 500_000
BATCH_SIZE = 50_000
MC_SE_LIMIT = 5.0
PARAMETERS = {
    "signal_power": 1.0,
    "sar_transfer": 1.0,
    "sar_noise_power": 0.7,
    "optical_coupling": 0.8,
    "optical_noise_power": 0.5,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_state() -> dict[str, Any]:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )
    return {"head": head, "dirty": dirty}


def _frequency_cases() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for sigma in (0.0, 0.5, 1.5):
        covariance = ((sigma**2, 0.0), (0.0, sigma**2))
        for index in range(9):
            cases.append(
                {
                    "family": "isotropic",
                    "sigma": sigma,
                    "frequency": (index * math.pi / 16.0, 0.0),
                    "shift_covariance": covariance,
                    "frequency_index": index,
                    "direction_radians": 0.0,
                }
            )
    covariance = ((2.25, 0.0), (0.0, 0.25))
    magnitude = math.pi / 2.0
    for angle in (0.0, math.pi / 4.0, math.pi / 2.0):
        cases.append(
            {
                "family": "anisotropic",
                "sigma": None,
                "frequency": (magnitude * math.cos(angle), magnitude * math.sin(angle)),
                "shift_covariance": covariance,
                "frequency_index": None,
                "direction_radians": angle,
            }
        )
    return cases


def _verify_case(result: dict[str, Any]) -> dict[str, float]:
    checks: dict[str, float] = {}
    for theory_key, empirical_key, se_key in (
        ("theoretical_sar_mse", "empirical_sar_mse", "empirical_sar_mse_se"),
        ("theoretical_joint_mse", "empirical_joint_mse", "empirical_joint_mse_se"),
        ("theoretical_gain", "empirical_gain", "empirical_gain_se"),
    ):
        error = abs(float(result[empirical_key]) - float(result[theory_key]))
        standard_error = float(result[se_key])
        if standard_error == 0.0:
            standardized_error = 0.0 if error <= 1e-12 else math.inf
        else:
            standardized_error = error / standard_error
        checks[theory_key.removeprefix("theoretical_")] = standardized_error
        if standardized_error > MC_SE_LIMIT:
            raise AssertionError(
                f"{theory_key} 与 Monte Carlo 样本差异超过预登记 5 个标准误。"
            )
    return checks


def _verify_frequency_trends(results: list[dict[str, Any]]) -> dict[str, Any]:
    isotropic: dict[float, list[dict[str, Any]]] = {}
    for row in results:
        if row["family"] == "isotropic":
            isotropic.setdefault(float(row["sigma_pixels"]), []).append(row)
    for sigma, rows in isotropic.items():
        ordered = sorted(rows, key=lambda row: row["frequency_index"])
        gains = [float(row["theoretical_gain"]) for row in ordered]
        if sigma == 0.0:
            if max(gains) - min(gains) > 1e-12:
                raise AssertionError("零平移方差时理论光学增益应与频率无关。")
        elif any(left + 1e-12 < right for left, right in zip(gains, gains[1:])):
            raise AssertionError("固定正平移方差下理论光学增益不应随频率上升。")
        if abs(float(ordered[0]["coherence_power"]) - 1.0) > 1e-12:
            raise AssertionError("零频率的理论平移相干功率应为 1。")

    anisotropic_rows = sorted(
        (row for row in results if row["family"] == "anisotropic"),
        key=lambda row: row["direction_radians"],
    )
    directional_gains = [float(row["theoretical_gain"]) for row in anisotropic_rows]
    if not directional_gains[0] < directional_gains[1] < directional_gains[2]:
        raise AssertionError("各向异性平移应使 x、对角、y 方向理论增益依次增加。")
    return {
        "isotropic_frequency_curves_nonincreasing": True,
        "zero_shift_curve_constant": True,
        "zero_frequency_shift_invariant": True,
        "anisotropic_direction_order": ["x 最低", "对角居中", "y 最高"],
    }


def _write_report(
    path: Path,
    rows: list[dict[str, Any]],
    checks: list[dict[str, float]],
    trends: dict[str, Any],
) -> None:
    isotropic = [row for row in rows if row["family"] == "isotropic"]
    sigma_15 = sorted(
        (row for row in isotropic if row["sigma"] == 1.5),
        key=lambda row: row["frequency_index"],
    )
    anisotropic = sorted(
        (row for row in rows if row["family"] == "anisotropic"),
        key=lambda row: row["direction_radians"],
    )
    max_standardized = max(
        max(float(value) for value in case.values()) for case in checks
    )
    low = sigma_15[0]
    high = sigma_15[-1]
    lines = [
        "# 未知平移下频率分辨光学线性增益：合成 Monte Carlo 核验",
        "",
        f"**实验编号：** `{PROTOCOL_ID}`  ",
        "**结论级别：** 理想化模型公式与实现的数值核验，不是本地 SAR 或模型效果证据。",
        "",
        "## 核验结果",
        "",
        f"- 共完成 {len(rows)} 个条件，每个条件 {SAMPLES_PER_CASE:,} 个复高斯样本；总计 {len(rows) * SAMPLES_PER_CASE:,} 个样本。",
        f"- 每个条件的 SAR-only MSE、联合 MSE、风险差均在预登记的 5 个 Monte Carlo 标准误内；最大标准化差为 `{max_standardized:.3f}`。",
        "- 30 个条件的闭式联合风险与直接协方差矩阵解均通过 `rtol=1e-10, atol=1e-12` 一致性检查。",
        f"- 各向同性曲线：零平移时首/末频率理论增益均为 `{float(next(row['theoretical_gain'] for row in isotropic if row['sigma_pixels'] == 0.0 and row['frequency_index'] == 0)):.8f}`；`σ=1.5` 时从低频 `{float(low['theoretical_gain']):.8f}` 降至高频 `{float(high['theoretical_gain']):.8f}`。",
        f"- 各向异性方向的理论增益依次为 x `{float(anisotropic[0]['theoretical_gain']):.8f}`、对角 `{float(anisotropic[1]['theoretical_gain']):.8f}`、y `{float(anisotropic[2]['theoretical_gain']):.8f}`；符合预登记次序。",
        "",
        "## 解释",
        "",
        "在固定复高斯单频模型中，未知高斯平移通过 `q_k=exp(-kᵀΣδk)` 衰减跨模态线性协方差，因此频率越高、平移方向方差越大，理论 LMMSE 增益越低。Monte Carlo 与独立协方差解支持公式实现与该受限模型内的预期趋势。",
        "",
        "这不是实际光学对 SAR 重建的增益测量，也不证明 SAR 图像服从该信号/噪声假设。真实图像上的频段效用、配准误差、局部变形及非线性模型行为仍未验证；本结果不授权在现有重复使用的 development cohort 上追加 screen，不更改任何 gate。",
        "",
        "## 复现",
        "",
        "```bash",
        "python experiments/synthetic_cpu/optical_shift_frequency_lmmse_cpu_20261114_v1r1/run_monte_carlo.py",
        "```",
        "",
        f"协议：[protocol.md](../protocol.md)。逐条件数据：[cases.csv](cases.csv)。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = ChineseArgumentParser(
        description="核对未知高斯平移下频率域 LMMSE 风险差公式；运行条件固定在协议中。"
    )
    return parser.parse_args()


def main() -> int:
    _parse_args()
    if RESULTS_DIR.exists():
        raise FileExistsError(f"结果目录已存在，拒绝覆盖：{RESULTS_DIR}")
    RESULTS_DIR.mkdir(parents=True)

    command = shlex.join([sys.executable, *sys.argv])
    git = _git_state()
    manifest: dict[str, Any] = {
        "protocol_id": PROTOCOL_ID,
        "status": "running",
        "start_time_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": _sha256(PROTOCOL_PATH),
        "model_sha256": _sha256(MODEL_PATH),
        "runner_sha256": _sha256(SCRIPT_PATH),
        "git": git,
        "device": "CPU",
        "cuda_used": False,
        "training_run": False,
        "project_payload_accessed": False,
        "confirmation_payload_accessed": False,
        "test_payload_accessed": False,
        "parameter_config": PARAMETERS,
        "samples_per_case": SAMPLES_PER_CASE,
        "batch_size": BATCH_SIZE,
        "base_seed": BASE_SEED,
        "condition_count": 30,
        "expected_sample_total": 30 * SAMPLES_PER_CASE,
        "monte_carlo_standard_error_limit": MC_SE_LIMIT,
        "command": command,
    }
    (RESULTS_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (RESULTS_DIR / "command.txt").write_text(command + "\n", encoding="utf-8")
    environment = [
        "运行设备：CPU",
        f"Python：{platform.python_version()}",
        f"NumPy：{np.__version__}",
        f"操作系统：{platform.platform()}",
        f"处理器架构：{platform.machine()}",
        f"逻辑 CPU 数：{os.cpu_count()}",
        f"Git HEAD：{git['head']}",
        f"工作区是否有既存修改：{git['dirty']}",
    ]
    (RESULTS_DIR / "environment.txt").write_text(
        "\n".join(environment) + "\n", encoding="utf-8"
    )

    print("开始运行 30 个预登记合成条件；仅使用 CPU 和 NumPy。", flush=True)
    rows: list[dict[str, Any]] = []
    standardized_checks: list[dict[str, float]] = []
    try:
        for index, case in enumerate(_frequency_cases()):
            result = simulate_lmmse_case(
                case["frequency"],
                case["shift_covariance"],
                **PARAMETERS,
                samples=SAMPLES_PER_CASE,
                seed=BASE_SEED + index,
                batch_size=BATCH_SIZE,
            )
            result.update(
                {
                    "case_index": index,
                    "family": case["family"],
                    "sigma_pixels": case["sigma"],
                    "frequency_index": case["frequency_index"],
                    "direction_radians": case["direction_radians"],
                }
            )
            standardized = _verify_case(result)
            rows.append(result)
            standardized_checks.append(standardized)
            print(
                f"条件 {index + 1}/30：{case['family']}，"
                f"q={result['coherence_power']:.6f}，"
                f"理论增益={result['theoretical_gain']:.8f}，"
                f"Monte Carlo 增益={result['empirical_gain']:.8f}，"
                f"标准误倍数={standardized['gain']:.3f}",
                flush=True,
            )

        trends = _verify_frequency_trends(rows)
        csv_path = RESULTS_DIR / "cases.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        maximum_standardized = {
            key: max(case[key] for case in standardized_checks)
            for key in standardized_checks[0]
        }
        summary = {
            "condition_count": len(rows),
            "samples_per_case": SAMPLES_PER_CASE,
            "total_generated_samples": len(rows) * SAMPLES_PER_CASE,
            "maximum_standardized_error_by_metric": maximum_standardized,
            "all_monte_carlo_checks_within_five_standard_errors": True,
            "frequency_trend_checks": trends,
            "interpretation_scope": "仅验证理想化复高斯单频 LMMSE 模型的数值实现。",
        }
        (RESULTS_DIR / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        _write_report(RESULTS_DIR / "报告.md", rows, standardized_checks, trends)
        manifest.update(
            {
                "status": "complete",
                "end_time_utc": datetime.now(timezone.utc).isoformat(),
                "results_sha256": _sha256(csv_path),
                "summary_sha256": _sha256(RESULTS_DIR / "summary.json"),
                "report_sha256": _sha256(RESULTS_DIR / "报告.md"),
            }
        )
        (RESULTS_DIR / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(
            "完成：30 个条件均通过公式、协方差矩阵和 Monte Carlo 误差核验。",
            flush=True,
        )
        return 0
    except Exception as exc:
        manifest.update(
            {
                "status": "failed",
                "end_time_utc": datetime.now(timezone.utc).isoformat(),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
        (RESULTS_DIR / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
