#!/usr/bin/env python3
"""已知二维双峰分布下的评价样本数敏感性诊断。"""

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
SEED = 20261110
REPEATS = 100
SAMPLE_COUNTS = [4, 8, 16, 32, 64, 128, 256]
REFERENCE_COUNT = 2048
CENTERS = np.asarray([[-3.0, 0.0], [3.0, 0.0]])
STD = np.asarray([0.5, 0.75])
TRUE_MEAN = np.zeros(2, dtype=np.float64)
TRUE_SCALE = np.sqrt(np.asarray([9.25, 0.5625]))
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
    centers = np.where(left[:, None], CENTERS[0], CENTERS[1])
    return centers + rng.normal(size=(count, 2)) * STD


def pairwise_distances(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    differences = left[:, None, :] - right[None, :, :]
    return np.sqrt(np.sum(differences * differences, axis=-1))


def energy_score(samples: np.ndarray, reference: np.ndarray) -> float:
    cross = pairwise_distances(samples, reference)
    within = pairwise_distances(samples, samples)
    return float(np.mean(cross) - 0.5 * np.mean(within))


def support_and_mass(samples: np.ndarray) -> tuple[float, float]:
    distance = np.linalg.norm(samples[:, None, :] - CENTERS[None, :, :], axis=-1)
    support = float(np.mean(np.min(distance, axis=1) <= SUPPORT_RADIUS))
    left = float(np.mean(samples[:, 0] < 0.0))
    right = float(np.mean(samples[:, 0] > 0.0))
    boundary = float(np.mean(samples[:, 0] == 0.0))
    mass_error = 0.5 * (abs(left - 0.5) + abs(right - 0.5)) + boundary
    return support, mass_error


def main() -> None:
    started = datetime.now(timezone.utc).isoformat()
    rng = np.random.default_rng(SEED)
    candidate_names = ["匹配矩单峰高斯", "0.8/0.2偏置双峰"]
    raw_rows: list[dict[str, object]] = []

    for repeat in range(REPEATS):
        reference = sample_mixture(rng, REFERENCE_COUNT)
        correct = sample_mixture(rng, max(SAMPLE_COUNTS))
        candidates = {
            "匹配矩单峰高斯": rng.normal(size=(max(SAMPLE_COUNTS), 2))
            * TRUE_SCALE,
            "0.8/0.2偏置双峰": sample_mixture(
                rng, max(SAMPLE_COUNTS), left_weight=0.8
            ),
        }
        for count in SAMPLE_COUNTS:
            correct_score = energy_score(correct[:count], reference)
            for name in candidate_names:
                samples = candidates[name][:count]
                score = energy_score(samples, reference)
                support, mass_error = support_and_mass(samples)
                raw_rows.append(
                    {
                        "重复编号": repeat + 1,
                        "样本数": count,
                        "候选分布": name,
                        "正确抽样能量分数": correct_score,
                        "候选能量分数": score,
                        "配对能量分数差": score - correct_score,
                        "正确抽样分数更低": int(score > correct_score),
                        "高概率模态支持率": support,
                        "模态质量偏差": mass_error,
                    }
                )

    csv_path = RUN_DIR / "逐重复结果.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(raw_rows[0]))
        writer.writeheader()
        writer.writerows(raw_rows)

    summary: dict[str, dict[str, dict[str, float]]] = {}
    for name in candidate_names:
        summary[name] = {}
        for count in SAMPLE_COUNTS:
            subset = [
                row
                for row in raw_rows
                if row["候选分布"] == name and row["样本数"] == count
            ]
            deltas = np.asarray([float(row["配对能量分数差"]) for row in subset])
            summary[name][str(count)] = {
                "配对分数差均值": float(np.mean(deltas)),
                "配对分数差标准差": float(np.std(deltas, ddof=1)),
                "配对分数差第10百分位": float(np.percentile(deltas, 10)),
                "配对分数差中位数": float(np.percentile(deltas, 50)),
                "配对分数差第90百分位": float(np.percentile(deltas, 90)),
                "正确抽样分数更低的重复比例": float(
                    np.mean([int(row["正确抽样分数更低"]) for row in subset])
                ),
                "候选高概率模态支持率均值": float(
                    np.mean([float(row["高概率模态支持率"]) for row in subset])
                ),
                "候选模态质量偏差均值": float(
                    np.mean([float(row["模态质量偏差"]) for row in subset])
                ),
            }

    result = {
        "实验名称": "已知双峰分布下的评价样本数敏感性诊断",
        "随机种子": SEED,
        "独立重复数": REPEATS,
        "评价样本数": SAMPLE_COUNTS,
        "每次重复的独立真实评价样本数": REFERENCE_COUNT,
        "汇总": summary,
    }
    results_path = RUN_DIR / "结果.json"
    results_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    if len(raw_rows) != REPEATS * len(SAMPLE_COUNTS) * len(candidate_names):
        raise RuntimeError("逐重复记录数量不符合预登记协议")
    for row in raw_rows:
        if not all(
            np.isfinite(float(row[key]))
            for key in ("正确抽样能量分数", "候选能量分数", "配对能量分数差", "高概率模态支持率", "模态质量偏差")
        ):
            raise RuntimeError("结果中出现非有限值")

    finished = datetime.now(timezone.utc).isoformat()
    manifest = {
        "运行编号": "c3a_distribution_draw_count_cpu_20261109_v1",
        "运行类型": "合成数据CPU诊断；非模型运行",
        "开始时间UTC": started,
        "结束时间UTC": finished,
        "命令": "python experiments/synthetic_cpu/c3a_distribution_draw_count_cpu_20261109_v1/run.py",
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
        "文件SHA256": {
            "协议": sha256(PROTOCOL_PATH),
            "脚本": sha256(Path(__file__).resolve()),
            "逐重复结果": sha256(csv_path),
            "汇总结果": sha256(results_path),
        },
    }
    (RUN_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print("评价样本数敏感性诊断已完成。")
    print(f"候选分布：{len(candidate_names)}；独立重复：{REPEATS}；样本数：{SAMPLE_COUNTS}。")
    print(f"逐重复记录：{len(raw_rows)} 条；结果仅来自合成二维分布。")
    for name in candidate_names:
        for count in SAMPLE_COUNTS:
            values = summary[name][str(count)]
            print(
                f"{name}，样本数 {count}：配对能量分数差 "
                f"{values['配对分数差均值']:.4f} ± {values['配对分数差标准差']:.4f}；"
                f"正确抽样分数更低比例 {values['正确抽样分数更低的重复比例']:.2f}。"
            )
    print(f"报告产物目录：{RUN_DIR.relative_to(Path.cwd())}")


if __name__ == "__main__":
    main()
