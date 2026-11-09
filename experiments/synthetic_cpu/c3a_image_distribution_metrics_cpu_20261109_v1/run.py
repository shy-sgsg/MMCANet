#!/usr/bin/env python3
"""归一化图像域中多样本分布评价量的已知案例检查。"""

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
SEED = 20261111
REPEATS = 100
DRAW_COUNT = 16
HEIGHT = 32
WIDTH = 32
LOOK_SHAPE = 4.0
NOISE_SCALE = 1.0 / LOOK_SHAPE


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


def make_signal() -> np.ndarray:
    axis = np.linspace(-1.0, 1.0, HEIGHT, dtype=np.float64)
    xx, yy = np.meshgrid(axis, axis)
    region = ((xx > 0.1) & (yy > -0.35)).astype(np.float64)
    broad_target = np.exp(-((xx + 0.35) ** 2 + (yy - 0.25) ** 2) / 0.025)
    point_target = np.exp(-((xx - 0.55) ** 2 + (yy + 0.42) ** 2) / 0.008)
    signal = 0.11 + 0.045 * region + 0.16 * broad_target + 0.10 * point_target
    return np.clip(signal, 0.0, 1.0)


def gamma_noise(
    rng: np.random.Generator, shape: float, scale: float, count: int
) -> np.ndarray:
    return rng.gamma(shape=shape, scale=scale, size=(count, HEIGHT, WIDTH))


def pixel_crps(samples: np.ndarray, observation: np.ndarray) -> float:
    cross = np.mean(np.abs(samples - observation[None, :, :]), axis=0)
    pairwise = np.mean(
        np.abs(samples[:, None, :, :] - samples[None, :, :, :]), axis=(0, 1)
    )
    return float(np.mean(cross - 0.5 * pairwise))


def normalized_image_energy(samples: np.ndarray, observation: np.ndarray) -> float:
    flattened = samples.reshape(samples.shape[0], -1)
    target = observation.reshape(-1)
    cross = np.linalg.norm(flattened - target[None, :], axis=1) / np.sqrt(target.size)
    differences = flattened[:, None, :] - flattened[None, :, :]
    within = np.linalg.norm(differences, axis=2) / np.sqrt(target.size)
    return float(np.mean(cross) - 0.5 * np.mean(within))


def variogram_score(samples: np.ndarray, observation: np.ndarray) -> float:
    forecast_horizontal = np.mean(
        np.abs(samples[:, :, 1:] - samples[:, :, :-1]), axis=0
    )
    target_horizontal = np.abs(observation[:, 1:] - observation[:, :-1])
    forecast_vertical = np.mean(
        np.abs(samples[:, 1:, :] - samples[:, :-1, :]), axis=0
    )
    target_vertical = np.abs(observation[1:, :] - observation[:-1, :])
    errors = np.concatenate(
        [
            (forecast_horizontal - target_horizontal).ravel(),
            (forecast_vertical - target_vertical).ravel(),
        ]
    )
    return float(np.mean(errors * errors))


def score(samples: np.ndarray, observation: np.ndarray) -> dict[str, float]:
    return {
        "逐像素CRPS": pixel_crps(samples, observation),
        "归一化整图能量分数": normalized_image_energy(samples, observation),
        "邻接变差分数": variogram_score(samples, observation),
    }


def main() -> None:
    started = datetime.now(timezone.utc).isoformat()
    rng = np.random.default_rng(SEED)
    signal = make_signal()
    labels = {
        "正确Gamma边缘与独立像素": "正确候选",
        "均值点估计塌缩": "点估计塌缩",
        "正确Gamma边缘与共享整图因子": "同边缘错误空间依赖",
        "均值1方差加倍的Gamma边缘": "过分散乘性噪声",
    }
    raw_rows: list[dict[str, object]] = []

    for repeat in range(REPEATS):
        observation_raw = signal * gamma_noise(rng, LOOK_SHAPE, NOISE_SCALE, 1)[0]
        observation = np.clip(observation_raw, 0.0, 1.0)
        true_samples_raw = signal[None, :, :] * gamma_noise(
            rng, LOOK_SHAPE, NOISE_SCALE, DRAW_COUNT
        )
        shared_factor = rng.gamma(
            shape=LOOK_SHAPE, scale=NOISE_SCALE, size=(DRAW_COUNT, 1, 1)
        )
        shared_samples_raw = signal[None, :, :] * shared_factor
        overdispersed_raw = signal[None, :, :] * gamma_noise(
            rng, LOOK_SHAPE / 2.0, 2.0 / LOOK_SHAPE, DRAW_COUNT
        )
        candidates_raw = {
            "正确Gamma边缘与独立像素": true_samples_raw,
            "均值点估计塌缩": np.repeat(signal[None, :, :], DRAW_COUNT, axis=0),
            "正确Gamma边缘与共享整图因子": shared_samples_raw,
            "均值1方差加倍的Gamma边缘": overdispersed_raw,
        }
        metrics_by_candidate: dict[str, dict[str, float]] = {}
        for name, samples_raw in candidates_raw.items():
            samples = np.clip(samples_raw, 0.0, 1.0)
            metrics = score(samples, observation)
            metrics["输入裁剪比例"] = float(
                np.mean((samples_raw < 0.0) | (samples_raw > 1.0))
            )
            if not all(np.isfinite(value) for value in metrics.values()):
                raise RuntimeError(f"第 {repeat + 1} 轮的{name}出现非有限评价值")
            metrics_by_candidate[name] = metrics

        correct = metrics_by_candidate["正确Gamma边缘与独立像素"]
        for name, metrics in metrics_by_candidate.items():
            row: dict[str, object] = {"重复编号": repeat + 1, "候选分布": labels[name]}
            for metric_name, value in metrics.items():
                row[metric_name] = value
                row[f"{metric_name}与正确候选之差"] = value - correct[metric_name]
            raw_rows.append(row)

    csv_path = RUN_DIR / "逐重复结果.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(raw_rows[0]))
        writer.writeheader()
        writer.writerows(raw_rows)

    metric_names = ["逐像素CRPS", "归一化整图能量分数", "邻接变差分数"]
    summary: dict[str, dict[str, dict[str, float]]] = {}
    paired_summary: dict[str, dict[str, dict[str, float]]] = {}
    for name, label in labels.items():
        rows = [row for row in raw_rows if row["候选分布"] == label]
        summary[label] = {}
        paired_summary[label] = {}
        for metric_name in [*metric_names, "输入裁剪比例"]:
            values = np.asarray([float(row[metric_name]) for row in rows])
            summary[label][metric_name] = {
                "均值": float(np.mean(values)),
                "重复间标准差": float(np.std(values, ddof=1)),
            }
            differences = np.asarray(
                [float(row[f"{metric_name}与正确候选之差"]) for row in rows]
            )
            paired_summary[label][metric_name] = {
                "与正确候选配对差均值": float(np.mean(differences)),
                "配对差标准误": float(np.std(differences, ddof=1) / np.sqrt(REPEATS)),
                "候选分数高于正确候选的重复比例": float(np.mean(differences > 0.0)),
            }

    result = {
        "实验名称": "归一化图像域中的多样本分布评价量已知案例",
        "随机种子": SEED,
        "独立重复数": REPEATS,
        "图像尺寸": [HEIGHT, WIDTH],
        "每个预测集合draw数": DRAW_COUNT,
        "Gamma形状参数": LOOK_SHAPE,
        "Gamma尺度参数": NOISE_SCALE,
        "候选分布汇总": summary,
        "相对正确候选的配对比较": paired_summary,
        "解释边界": "受控归一化表示域数学构造，不代表定标SAR物理噪声。"
    }
    results_path = RUN_DIR / "结果.json"
    results_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    if len(raw_rows) != REPEATS * len(labels):
        raise RuntimeError("逐重复记录数量不符合预登记协议")
    finished = datetime.now(timezone.utc).isoformat()
    manifest = {
        "运行编号": "c3a_image_distribution_metrics_cpu_20261109_v1",
        "运行类型": "合成归一化图像域CPU诊断；非模型运行",
        "开始时间UTC": started,
        "结束时间UTC": finished,
        "命令": "python experiments/synthetic_cpu/c3a_image_distribution_metrics_cpu_20261109_v1/run.py",
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

    print("归一化图像域分布指标已知案例诊断完成。")
    print(f"图像尺寸：{HEIGHT}×{WIDTH}；预测draw数：{DRAW_COUNT}；独立重复：{REPEATS}。")
    print(f"逐重复记录：{len(raw_rows)} 条；所有运算使用CPU和合成数据。")
    for label, metrics in paired_summary.items():
        if label == labels["正确Gamma边缘与独立像素"]:
            continue
        print(
            f"{label}：逐像素CRPS配对差={metrics['逐像素CRPS']['与正确候选配对差均值']:.7f}，"
            f"整图能量配对差={metrics['归一化整图能量分数']['与正确候选配对差均值']:.7f}，"
            f"邻接变差配对差={metrics['邻接变差分数']['与正确候选配对差均值']:.7f}。"
        )
    print(f"产物目录：{RUN_DIR.relative_to(Path.cwd())}")


if __name__ == "__main__":
    main()
