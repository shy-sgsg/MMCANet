"""单亮点漏检的图幅尺度敏感性 CPU 诊断。"""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
from scripts.distribution_metrics import (  # noqa: E402
    adjacent_variogram_score,
    normalized_image_energy_score,
    pixelwise_crps,
)

if "torch" in sys.modules:
    raise RuntimeError("本实验不得导入 PyTorch。")

OUT_DIR = Path(__file__).resolve().parent
PROTOCOL_PATH = OUT_DIR / "protocol.json"
RAW_PATH = OUT_DIR / "逐重复结果.csv"
SUMMARY_PATH = OUT_DIR / "汇总.json"
MANIFEST_PATH = OUT_DIR / "运行清单.json"

SEED = 2026110902
SIZES = (64, 128, 256)
REPEATS = 300
DRAW_COUNT = 16
BACKGROUND = 0.04
PEAK = 0.45
SHAPE = 16.0
SCALE = 1.0 / SHAPE
THRESHOLD = 0.25
BOOTSTRAPS = 5000
SCORE_FIELDS = ("逐像素_CRPS", "整图能量分数", "邻接变差分数")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _score(samples: np.ndarray, observation: np.ndarray) -> dict[str, float]:
    values = {
        "逐像素_CRPS": pixelwise_crps(samples, observation),
        "整图能量分数": normalized_image_energy_score(samples, observation),
        "邻接变差分数": adjacent_variogram_score(samples, observation),
    }
    if not all(np.isfinite(value) for value in values.values()):
        raise RuntimeError("评分出现非有限值。")
    return values


def _bootstrap_interval(
    values: np.ndarray, rng: np.random.Generator
) -> list[float]:
    indices = rng.integers(0, values.size, size=(BOOTSTRAPS, values.size))
    means = np.mean(values[indices], axis=1)
    low, high = np.quantile(means, (0.025, 0.975))
    return [float(low), float(high)]


def main() -> None:
    for output_path in (RAW_PATH, SUMMARY_PATH, MANIFEST_PATH):
        if output_path.exists():
            raise FileExistsError(f"输出文件已存在，为保护旧结果而停止：{output_path}")
    with PROTOCOL_PATH.open("r", encoding="utf-8") as stream:
        protocol = json.load(stream)
    if list(SIZES) != protocol["场景"]["图幅"]:
        raise RuntimeError("脚本图幅与预登记协议不一致。")
    if protocol["运行设置"]["每种图幅独立重复数"] != REPEATS:
        raise RuntimeError("脚本重复数与预登记协议不一致。")

    rng = np.random.default_rng(SEED)
    bootstrap_rng = np.random.default_rng(SEED + 1)
    gamma_check_rng = np.random.default_rng(SEED + 2)
    gamma_check = gamma_check_rng.gamma(SHAPE, SCALE, size=500_000)
    gamma_mean = float(np.mean(gamma_check))
    gamma_variance = float(np.var(gamma_check))
    if abs(gamma_mean - 1.0) >= 0.005:
        raise RuntimeError("Gamma抽样均值未通过预登记数值检查。")
    if abs(gamma_variance - 1.0 / SHAPE) >= 0.001:
        raise RuntimeError("Gamma抽样方差未通过预登记数值检查。")

    records: list[dict[str, float | int | str]] = []
    for size in SIZES:
        target = np.full((size, size), BACKGROUND, dtype=np.float64)
        center = (size // 2, size // 2)
        target[center] = PEAK
        missing = np.full((size, size), BACKGROUND, dtype=np.float64)

        for repeat in range(REPEATS):
            observation_multiplier = rng.gamma(SHAPE, SCALE, size=(size, size))
            raw_observation = target * observation_multiplier
            observation = np.clip(raw_observation, 0.0, 1.0)
            forecast_multipliers = rng.gamma(
                SHAPE, SCALE, size=(DRAW_COUNT, size, size)
            )

            for condition, scene in (("正确", target), ("漏检", missing)):
                raw_samples = scene[None, :, :] * forecast_multipliers
                clip_fraction = float(np.mean(raw_samples > 1.0))
                samples = np.clip(raw_samples, 0.0, 1.0)
                point_values = samples[:, center[0], center[1]]
                records.append(
                    {
                        "图幅": size,
                        "亮点密度": 1.0 / (size * size),
                        "重复": repeat + 1,
                        "预测条件": condition,
                        **_score(samples, observation),
                        "中心亮点预测均值": float(np.mean(point_values)),
                        "中心亮点检出率": float(np.mean(point_values >= THRESHOLD)),
                        "预测裁剪比例": clip_fraction,
                        "目标裁剪比例": float(np.mean(raw_observation > 1.0)),
                    }
                )

    with RAW_PATH.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)

    summary: dict[str, object] = {
        "实验名称": protocol["实验名称"],
        "随机种子": SEED,
        "重复数": REPEATS,
        "每集合draw数": DRAW_COUNT,
        "Gamma抽样核验": {
            "样本数": int(gamma_check.size),
            "样本均值": gamma_mean,
            "理论均值": 1.0,
            "样本方差": gamma_variance,
            "理论方差": 1.0 / SHAPE,
        },
        "各图幅结果": {},
        "逐图幅配对差": {},
        "输出记录数": len(records),
    }

    for size in SIZES:
        rows = [row for row in records if row["图幅"] == size]
        correct = [row for row in rows if row["预测条件"] == "正确"]
        missing = [row for row in rows if row["预测条件"] == "漏检"]
        density = 1.0 / (size * size)
        summary["各图幅结果"][str(size)] = {
            "亮点密度": density,
            "正确条件": {
                "中心亮点预测均值": float(
                    np.mean([float(row["中心亮点预测均值"]) for row in correct])
                ),
                "中心亮点检出率": float(
                    np.mean([float(row["中心亮点检出率"]) for row in correct])
                ),
                "分数": {
                    field: float(np.mean([float(row[field]) for row in correct]))
                    for field in SCORE_FIELDS
                },
            },
            "漏检条件": {
                "中心亮点预测均值": float(
                    np.mean([float(row["中心亮点预测均值"]) for row in missing])
                ),
                "中心亮点检出率": float(
                    np.mean([float(row["中心亮点检出率"]) for row in missing])
                ),
                "分数": {
                    field: float(np.mean([float(row[field]) for row in missing]))
                    for field in SCORE_FIELDS
                },
            },
        }
        comparison: dict[str, object] = {}
        for field in SCORE_FIELDS:
            differences = np.asarray(
                [
                    float(error[field]) - float(reference[field])
                    for reference, error in zip(correct, missing, strict=True)
                ]
            )
            comparison[field] = {
                "漏检减正确配对均值": float(np.mean(differences)),
                "95%区间": _bootstrap_interval(differences, bootstrap_rng),
                "正差比例": float(np.mean(differences > 0.0)),
            }
        summary["逐图幅配对差"][str(size)] = comparison

    with SUMMARY_PATH.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
        stream.write("\n")

    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    branch = subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    manifest = {
        "实验名称": protocol["实验名称"],
        "运行日期": "2026-11-09",
        "代码提交": commit,
        "分支": branch,
        "工作区原有未提交状态": bool(status),
        "运行命令": "CUDA_VISIBLE_DEVICES='' python experiments/synthetic_cpu/c3a_bright_scatterer_resolution_sweep_cpu_20261109_v1/run.py",
        "运行设备": "CPU",
        "运行环境": {
            "Python": platform.python_version(),
            "NumPy": np.__version__,
            "平台": platform.platform(),
            "Python可执行文件": sys.executable,
            "PyTorch已导入": "torch" in sys.modules,
        },
        "随机种子": SEED,
        "图幅": list(SIZES),
        "每图幅重复数": REPEATS,
        "每集合draw数": DRAW_COUNT,
        "项目数据/模型/checkpoint输入": "无；仅使用协议规定的合成数组。",
        "confirmation/test访问": False,
        "输出记录数": len(records),
        "文件SHA256": {
            "协议": _sha256(PROTOCOL_PATH),
            "脚本": _sha256(Path(__file__)),
            "逐重复结果": _sha256(RAW_PATH),
            "汇总": _sha256(SUMMARY_PATH),
        },
    }
    with MANIFEST_PATH.open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
        stream.write("\n")

    print(f"实验完成：{len(records)} 条逐重复记录，三种图幅，CPU运行。")
    print("图幅 | 亮点密度 | CRPS漏检差 | 能量漏检差 | 变差漏检差 | 正确亮点检出率")
    for size in SIZES:
        comparison = summary["逐图幅配对差"][str(size)]
        hit_rate = summary["各图幅结果"][str(size)]["正确条件"]["中心亮点检出率"]
        print(
            f"{size}×{size} | {1.0/(size*size):.8f} | "
            f"{comparison['逐像素_CRPS']['漏检减正确配对均值']:.9f} | "
            f"{comparison['整图能量分数']['漏检减正确配对均值']:.9f} | "
            f"{comparison['邻接变差分数']['漏检减正确配对均值']:.9f} | "
            f"{hit_rate:.4f}"
        )
    print("产物：逐重复结果.csv、汇总.json、运行清单.json")


if __name__ == "__main__":
    main()
