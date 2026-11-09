"""用合成高斯 copula 检查邻接与多尺度变差分数的空间依赖敏感性。"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


RUN_DIR = Path(__file__).resolve().parent
HEIGHT = 24
WIDTH = 24
DRAW_COUNT = 16
REPEATS = 300
LAGS = (1, 2, 4, 8)
DATA_SEED = 20261109
BOOTSTRAP_SEED = 20261110
BOOTSTRAP_REPEATS = 2_000
CORRELATION = 0.8


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def covariance_cholesky(family: str) -> np.ndarray:
    coordinates = np.arange(HEIGHT, dtype=np.float64)
    distance = np.abs(coordinates[:, None] - coordinates[None, :])
    if family == "AR(1)":
        covariance = CORRELATION**distance
    elif family == "RBF":
        length_scale = math.sqrt(-0.5 / math.log(CORRELATION))
        covariance = np.exp(-0.5 * (distance / length_scale) ** 2)
    else:
        raise ValueError(f"未知协方差族：{family}")
    return np.linalg.cholesky(covariance)


def make_fields(cholesky: np.ndarray, base_noise: np.ndarray) -> np.ndarray:
    gaussian = np.matmul(np.matmul(cholesky, base_noise), cholesky.T)
    # sigmoid 把同一标准正态边际映射到 (0,1)，避免额外裁剪改变边际。
    return 1.0 / (1.0 + np.exp(-gaussian))


def pixelwise_crps(samples: np.ndarray, observation: np.ndarray) -> float:
    draw_count = samples.shape[0]
    cross = np.mean(np.abs(samples - observation[None]), axis=0)
    pairwise_upper_sum = np.zeros_like(observation)
    for first in range(draw_count):
        for second in range(first + 1, draw_count):
            pairwise_upper_sum += np.abs(samples[first] - samples[second])
    return float(np.mean(cross - pairwise_upper_sum / draw_count**2))


def image_energy_score(samples: np.ndarray, observation: np.ndarray) -> float:
    flattened = samples.reshape(samples.shape[0], -1)
    target = observation.reshape(-1)
    draw_count = flattened.shape[0]
    pairwise_upper_sum = 0.0
    for first in range(draw_count):
        for second in range(first + 1, draw_count):
            pairwise_upper_sum += float(np.linalg.norm(flattened[first] - flattened[second]))
    cross = sum(float(np.linalg.norm(draw - target)) for draw in flattened)
    return float(
        cross / (draw_count * math.sqrt(target.size))
        - pairwise_upper_sum / (draw_count**2 * math.sqrt(target.size))
    )


def variogram_score(
    samples: np.ndarray, observation: np.ndarray, lag: int
) -> float:
    forecast_parts: list[np.ndarray] = []
    observed_parts: list[np.ndarray] = []
    if samples.shape[2] > lag:
        forecast_parts.append(
            np.mean(
                np.abs(samples[:, :, lag:] - samples[:, :, :-lag]), axis=0
            ).ravel()
        )
        observed_parts.append(
            np.abs(observation[:, lag:] - observation[:, :-lag]).ravel()
        )
    if samples.shape[1] > lag:
        forecast_parts.append(
            np.mean(
                np.abs(samples[:, lag:, :] - samples[:, :-lag, :]), axis=0
            ).ravel()
        )
        observed_parts.append(
            np.abs(observation[lag:, :] - observation[:-lag, :]).ravel()
        )
    forecast = np.concatenate(forecast_parts)
    observed = np.concatenate(observed_parts)
    return float(np.mean((observed - forecast) ** 2))


def score_set(samples: np.ndarray, observation: np.ndarray) -> dict[str, float]:
    values = {
        "逐像素 CRPS": pixelwise_crps(samples, observation),
        "整图能量分数": image_energy_score(samples, observation),
    }
    lag_scores = {
        lag: variogram_score(samples, observation, lag) for lag in LAGS
    }
    values["单尺度变差 h=1"] = lag_scores[1]
    values["单尺度变差 h=2"] = lag_scores[2]
    values["单尺度变差 h=4"] = lag_scores[4]
    values["单尺度变差 h=8"] = lag_scores[8]
    values["多尺度变差 h=1,2,4,8"] = float(np.mean(list(lag_scores.values())))
    return values


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault((record["真值族"], record["评分量"]), []).append(record)

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    summaries: dict[str, Any] = {}
    for (truth_family, metric), rows in grouped.items():
        differences = np.asarray(
            [row["错误预测分数减正确预测分数"] for row in rows],
            dtype=np.float64,
        )
        bootstrap_means = np.empty(BOOTSTRAP_REPEATS, dtype=np.float64)
        for index in range(BOOTSTRAP_REPEATS):
            selection = rng.integers(0, differences.size, size=differences.size)
            bootstrap_means[index] = differences[selection].mean()
        summaries.setdefault(truth_family, {})[metric] = {
            "重复数": int(differences.size),
            "错误减正确平均分": float(differences.mean()),
            "平均分差95百分位区间": [
                float(np.quantile(bootstrap_means, 0.025)),
                float(np.quantile(bootstrap_means, 0.975)),
            ],
            "正确模型逐重复排序比例": float(np.mean(differences > 0.0)),
            "平分比例": float(np.mean(differences == 0.0)),
        }
    return summaries


def main() -> None:
    if (RUN_DIR / "results.json").exists() or (RUN_DIR / "per_trial.csv").exists():
        raise FileExistsError(f"结果文件已存在，拒绝覆盖：{RUN_DIR}")

    started_utc = datetime.now(timezone.utc).isoformat()
    rng = np.random.default_rng(DATA_SEED)
    cholesky = {
        family: covariance_cholesky(family) for family in ("AR(1)", "RBF")
    }
    records: list[dict[str, Any]] = []
    total = 2 * REPEATS
    completed = 0

    for truth_family in ("AR(1)", "RBF"):
        competing_family = "RBF" if truth_family == "AR(1)" else "AR(1)"
        for repeat in range(REPEATS):
            base_noise = rng.standard_normal((DRAW_COUNT + 1, HEIGHT, WIDTH))
            observation = make_fields(cholesky[truth_family], base_noise[:1])[0]
            paired_draw_noise = base_noise[1:]
            predictions = {
                family: make_fields(cholesky[family], paired_draw_noise)
                for family in ("AR(1)", "RBF")
            }
            scored = {
                family: score_set(predictions[family], observation)
                for family in ("AR(1)", "RBF")
            }
            for metric, correct_score in scored[truth_family].items():
                wrong_score = scored[competing_family][metric]
                records.append(
                    {
                        "真值族": truth_family,
                        "重复编号": repeat + 1,
                        "评分量": metric,
                        "正确预测族": truth_family,
                        "竞争预测族": competing_family,
                        "正确预测分数": correct_score,
                        "竞争预测分数": wrong_score,
                        "错误预测分数减正确预测分数": wrong_score - correct_score,
                        "正确排序": wrong_score > correct_score,
                    }
                )
            completed += 1
            if completed % 100 == 0 or completed == total:
                print(f"合成评分进度：{completed}/{total} 个目标。", flush=True)

    summary = summarize(records)
    results = {
        "实验编号": RUN_DIR.name,
        "输入": {
            "图像尺寸": [HEIGHT, WIDTH],
            "预测样本数": DRAW_COUNT,
            "每个真值方向重复数": REPEATS,
            "一阶相关系数": CORRELATION,
            "变差 lag": list(LAGS),
            "预测底噪配对": True,
        },
        "汇总规则": {
            "差值方向": "错误预测分数减正确预测分数；正值表示正确预测获得更低分",
            "bootstrap重复数": BOOTSTRAP_REPEATS,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "重抽样单位": "每个真值方向内的独立目标重复",
        },
        "各真值方向汇总": summary,
    }
    results_path = RUN_DIR / "results.json"
    results_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    csv_path = RUN_DIR / "per_trial.csv"
    fieldnames = list(records[0])
    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    protocol_path = RUN_DIR / "protocol.md"
    script_path = Path(__file__).resolve()
    manifest = {
        "实验编号": RUN_DIR.name,
        "状态": "完成",
        "开始时间_UTC": started_utc,
        "结束时间_UTC": datetime.now(timezone.utc).isoformat(),
        "运行命令": "python experiments/synthetic_cpu/c3a_spatial_variogram_multiscale_cpu_20261109_v1/run.py",
        "Python版本": sys.version.split()[0],
        "NumPy版本": np.__version__,
        "操作系统": platform.platform(),
        "设备": "CPU；脚本未导入 PyTorch，也未调用 GPU",
        "随机种子": DATA_SEED,
        "协议_SHA256": sha256_file(protocol_path),
        "脚本_SHA256": sha256_file(script_path),
        "结果_SHA256": sha256_file(results_path),
        "逐重复CSV_SHA256": sha256_file(csv_path),
        "项目数据访问": False,
        "模型或检查点访问": False,
        "确认集载荷访问": False,
        "测试载荷访问": False,
        "训练或推理": False,
        "备注": "纯合成空间相关场评分诊断；不改变 C3-A v3 或任何阶段 gate。",
    }
    (RUN_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    for truth_family, metrics in summary.items():
        print(f"真值为 {truth_family}：", flush=True)
        for metric in ("逐像素 CRPS", "单尺度变差 h=1", "多尺度变差 h=1,2,4,8", "整图能量分数"):
            item = metrics[metric]
            print(
                f"  {metric}：平均配对分差 {item['错误减正确平均分']:.8f}，"
                f"95%区间 [{item['平均分差95百分位区间'][0]:.8f}, "
                f"{item['平均分差95百分位区间'][1]:.8f}]，"
                f"正确排序比例 {item['正确模型逐重复排序比例']:.3f}。",
                flush=True,
            )
    print(f"报告数据已写入：{RUN_DIR}", flush=True)


if __name__ == "__main__":
    main()
