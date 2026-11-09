#!/usr/bin/env python3
"""已知二维双峰分布上的生成样本评价量诊断。"""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


RUN_DIR = Path(__file__).resolve().parent
PROTOCOL_PATH = RUN_DIR / "protocol.json"
SEED = 20261109
REPEATS = 30
SAMPLE_COUNT = 256
REFERENCE_COUNT = 2048
MODE_CENTERS = np.asarray([[-3.0, 0.0], [3.0, 0.0]])
MODE_STD = np.asarray([0.5, 0.75])
TRUE_MEAN = np.zeros(2, dtype=np.float64)
TRUE_COVARIANCE = np.diag([9.25, 0.5625])
SUPPORT_RADIUS = 1.75


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_text(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=RUN_DIR, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "不可用"


def sample_mixture(rng: np.random.Generator, count: int, left_weight: float = 0.5) -> np.ndarray:
    left = rng.random(count) < left_weight
    centers = np.where(left[:, None], MODE_CENTERS[0], MODE_CENTERS[1])
    return centers + rng.normal(size=(count, 2)) * MODE_STD


def pairwise_distances(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    differences = left[:, None, :] - right[None, :, :]
    return np.sqrt(np.sum(differences * differences, axis=-1))


def evaluate(samples: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    cross_distance = pairwise_distances(samples, reference)
    within_distance = pairwise_distances(samples, samples)
    left_mass = float(np.mean(samples[:, 0] < 0.0))
    right_mass = float(np.mean(samples[:, 0] > 0.0))
    boundary_mass = float(np.mean(samples[:, 0] == 0.0))
    near_mode = np.min(
        np.linalg.norm(samples[:, None, :] - MODE_CENTERS[None, :, :], axis=-1), axis=1
    ) <= SUPPORT_RADIUS
    upper_triangle = np.triu_indices(samples.shape[0], k=1)

    return {
        "总体均值误差": float(np.linalg.norm(np.mean(samples, axis=0) - TRUE_MEAN)),
        "能量分数": float(np.mean(cross_distance) - 0.5 * np.mean(within_distance)),
        "成对样本距离": float(np.mean(within_distance[upper_triangle])),
        "模态质量误差": float(
            0.5 * (abs(left_mass - 0.5) + abs(right_mass - 0.5)) + boundary_mass
        ),
        "高概率模态支持率": float(np.mean(near_mode)),
        "左模态样本比例": left_mass,
        "右模态样本比例": right_mass,
        "分界线样本比例": boundary_mass,
    }


def summarize(raw_rows: list[dict[str, object]]) -> dict[str, dict[str, dict[str, float]]]:
    methods = sorted({str(row["候选分布"]) for row in raw_rows})
    metric_names = [
        "总体均值误差",
        "能量分数",
        "成对样本距离",
        "模态质量误差",
        "高概率模态支持率",
        "左模态样本比例",
        "右模态样本比例",
        "分界线样本比例",
    ]
    output: dict[str, dict[str, dict[str, float]]] = {}
    for method in methods:
        rows = [row for row in raw_rows if row["候选分布"] == method]
        output[method] = {}
        for metric in metric_names:
            values = np.asarray([float(row[metric]) for row in rows], dtype=np.float64)
            output[method][metric] = {
                "均值": float(np.mean(values)),
                "重复间标准差": float(np.std(values, ddof=1)),
            }
    return output


def main() -> None:
    started = datetime.now(timezone.utc).isoformat()
    rng = np.random.default_rng(SEED)
    labels = {
        "真实双峰抽样": "真实双峰分布抽样",
        "均值塌缩": "真实总体均值处的完全塌缩点质量",
        "矩匹配单峰高斯": "匹配真实总体均值和协方差的单峰高斯",
        "偏置双峰抽样": "左右模态权重为0.8/0.2的偏置双峰分布",
    }
    rows: list[dict[str, object]] = []

    for repeat in range(REPEATS):
        reference = sample_mixture(rng, REFERENCE_COUNT)
        candidates = {
            "真实双峰抽样": sample_mixture(rng, SAMPLE_COUNT),
            "均值塌缩": np.repeat(TRUE_MEAN[None, :], SAMPLE_COUNT, axis=0),
            "矩匹配单峰高斯": rng.normal(
                loc=TRUE_MEAN,
                scale=np.sqrt(np.diag(TRUE_COVARIANCE)),
                size=(SAMPLE_COUNT, 2),
            ),
            "偏置双峰抽样": sample_mixture(rng, SAMPLE_COUNT, left_weight=0.8),
        }
        for method, samples in candidates.items():
            metrics = evaluate(samples, reference)
            if not all(np.isfinite(value) for value in metrics.values()):
                raise RuntimeError(f"第 {repeat + 1} 轮的 {method} 出现非有限评价值")
            rows.append({"重复编号": repeat + 1, "候选分布": labels[method], **metrics})

    csv_path = RUN_DIR / "逐重复结果.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize(rows)
    result = {
        "实验名称": "已知双峰分布下的生成结果评价量诊断",
        "随机种子": SEED,
        "独立重复数": REPEATS,
        "每个候选集合样本数": SAMPLE_COUNT,
        "每次重复的独立真实评价样本数": REFERENCE_COUNT,
        "真实总体均值": TRUE_MEAN.tolist(),
        "真实总体协方差": TRUE_COVARIANCE.tolist(),
        "模态支持半径": SUPPORT_RADIUS,
        "候选分布汇总": summary,
        "配对能量分数差_相对真实双峰抽样": {},
    }
    reference_label = labels["真实双峰抽样"]
    reference_energy = np.asarray(
        [float(row["能量分数"]) for row in rows if row["候选分布"] == reference_label]
    )
    for method, label in labels.items():
        method_energy = np.asarray(
            [float(row["能量分数"]) for row in rows if row["候选分布"] == label]
        )
        paired_delta = method_energy - reference_energy
        result["配对能量分数差_相对真实双峰抽样"][label] = {
            "均值": float(np.mean(paired_delta)),
            "重复间标准差": float(np.std(paired_delta, ddof=1)),
        }

    results_path = RUN_DIR / "结果.json"
    results_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    if not np.allclose(TRUE_COVARIANCE, np.diag([9.25, 0.5625])):
        raise RuntimeError("双峰混合分布的总体协方差定义不一致")
    collapsed = summary[labels["均值塌缩"]]
    if collapsed["总体均值误差"]["均值"] != 0.0:
        raise RuntimeError("均值塌缩候选没有保持预设总体均值")
    if collapsed["成对样本距离"]["均值"] != 0.0:
        raise RuntimeError("均值塌缩候选应具有零成对样本距离")

    finished = datetime.now(timezone.utc).isoformat()
    manifest = {
        "运行编号": "c3a_distribution_metric_sanity_cpu_20261109_v1",
        "运行类型": "合成数据CPU诊断；非模型运行",
        "开始时间UTC": started,
        "结束时间UTC": finished,
        "命令": "python experiments/synthetic_cpu/c3a_distribution_metric_sanity_cpu_20261109_v1/run.py",
        "代码提交": git_text("rev-parse", "HEAD"),
        "工作树有改动": bool(git_text("status", "--porcelain")),
        "Python版本": sys.version.split()[0],
        "NumPy版本": np.__version__,
        "平台": platform.platform(),
        "设备": "CPU",
        "随机种子": SEED,
        "确认测试载荷访问": False,
        "项目影像载荷访问": False,
        "模型与检查点访问": False,
        "输入": ["protocol.json中的合成分布定义"],
        "文件SHA256": {
            "协议": sha256(PROTOCOL_PATH),
            "脚本": sha256(Path(__file__).resolve()),
            "逐重复结果": sha256(csv_path),
            "汇总结果": sha256(results_path),
        },
        "评价器版本": "本目录run.py；评价对象与冻结C3-A v3分离",
    }
    (RUN_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print("合成分布评价诊断已完成。")
    print(f"候选分布：{len(labels)}；独立重复：{REPEATS}；每组样本：{SAMPLE_COUNT}。")
    print(f"逐重复结果：{csv_path.relative_to(Path.cwd())}")
    print(f"汇总结果：{results_path.relative_to(Path.cwd())}")
    print(f"运行清单：{(RUN_DIR / 'manifest.json').relative_to(Path.cwd())}")
    for method, label in labels.items():
        values = summary[label]
        print(
            f"{label}：均值误差={values['总体均值误差']['均值']:.4f}，"
            f"能量分数={values['能量分数']['均值']:.4f}，"
            f"模态支持率={values['高概率模态支持率']['均值']:.4f}，"
            f"模态质量误差={values['模态质量误差']['均值']:.4f}。"
        )


if __name__ == "__main__":
    main()
