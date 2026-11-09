"""稀疏亮点退化下多样本评价量的合成 CPU 诊断。"""

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

SEED = 2026110901
REPEATS = 300
DRAW_COUNT = 16
HEIGHT = 64
WIDTH = 64
BACKGROUND = 0.04
PEAK = 0.45
ATTENUATED_PEAK = 0.30
GAMMA_SHAPE = 16.0
GAMMA_SCALE = 1.0 / GAMMA_SHAPE
THRESHOLD = 0.25
BOOTSTRAPS = 5000

TRUE_POINTS = ((12, 12), (12, 48), (48, 12), (48, 48))
SPURIOUS_POINTS = ((24, 24), (24, 40), (40, 24), (40, 40))
SCENARIOS = ("正确", "压暗", "漏检", "右移一像素", "增加伪亮点")
SCORE_FIELDS = ("逐像素_CRPS", "整图能量分数", "邻接变差分数")
LOCAL_FIELDS = (
    "亮点均值",
    "亮点幅度绝对误差",
    "精确位置检出率",
    "三乘三邻域检出率",
    "真实目标区外伪亮点率",
    "背景中位数",
    "预测裁剪比例",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _scene(points: tuple[tuple[int, int], ...], intensity: float) -> np.ndarray:
    image = np.full((HEIGHT, WIDTH), BACKGROUND, dtype=np.float64)
    for row, col in points:
        image[row, col] = intensity
    return image


def _scenario_scenes() -> dict[str, np.ndarray]:
    shifted = tuple((row, col + 1) for row, col in TRUE_POINTS)
    return {
        "正确": _scene(TRUE_POINTS, PEAK),
        "压暗": _scene(TRUE_POINTS, ATTENUATED_PEAK),
        "漏检": np.full((HEIGHT, WIDTH), BACKGROUND, dtype=np.float64),
        "右移一像素": _scene(shifted, PEAK),
        "增加伪亮点": _scene(TRUE_POINTS + SPURIOUS_POINTS, PEAK),
    }


def _evaluate_local(samples: np.ndarray) -> dict[str, float]:
    point_values = samples[:, [p[0] for p in TRUE_POINTS], [p[1] for p in TRUE_POINTS]]
    neighborhood_hits = []
    for row, col in TRUE_POINTS:
        neighborhood = samples[:, row - 1 : row + 2, col - 1 : col + 2]
        neighborhood_hits.append(np.max(neighborhood, axis=(1, 2)) >= THRESHOLD)

    target_zone = np.zeros((HEIGHT, WIDTH), dtype=bool)
    for row, col in TRUE_POINTS:
        target_zone[row - 1 : row + 2, col - 1 : col + 2] = True
    background_values = samples[:, ~target_zone]

    peak_mean = float(np.mean(point_values))
    return {
        "亮点均值": peak_mean,
        "亮点幅度绝对误差": abs(peak_mean - PEAK),
        "精确位置检出率": float(np.mean(point_values >= THRESHOLD)),
        "三乘三邻域检出率": float(np.mean(np.asarray(neighborhood_hits))),
        "真实目标区外伪亮点率": float(np.mean(background_values >= THRESHOLD)),
        "背景中位数": float(np.median(background_values)),
    }


def _evaluate_scores(samples: np.ndarray, observation: np.ndarray) -> dict[str, float]:
    scores = {
        "逐像素_CRPS": pixelwise_crps(samples, observation),
        "整图能量分数": normalized_image_energy_score(samples, observation),
        "邻接变差分数": adjacent_variogram_score(samples, observation),
    }
    if not all(np.isfinite(value) for value in scores.values()):
        raise RuntimeError("评分出现非有限值。")
    return scores


def _bootstrap_interval(values: np.ndarray, rng: np.random.Generator) -> list[float]:
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
    if protocol["运行设置"]["独立重复数"] != REPEATS:
        raise RuntimeError("脚本重复数与预登记协议不一致。")

    rng = np.random.default_rng(SEED)
    bootstrap_rng = np.random.default_rng(SEED + 1)
    scenes = _scenario_scenes()
    true_scene = scenes["正确"]

    # 独立核验乘性随机变量的预登记均值与方差参数化。
    gamma_check_rng = np.random.default_rng(SEED + 2)
    gamma_check = gamma_check_rng.gamma(
        shape=GAMMA_SHAPE, scale=GAMMA_SCALE, size=500_000
    )
    gamma_mean = float(np.mean(gamma_check))
    gamma_variance = float(np.var(gamma_check))
    if abs(gamma_mean - 1.0) >= 0.005:
        raise RuntimeError("Gamma抽样均值未通过预登记数值检查。")
    if abs(gamma_variance - 1.0 / GAMMA_SHAPE) >= 0.001:
        raise RuntimeError("Gamma抽样方差未通过预登记数值检查。")

    records: list[dict[str, float | int | str]] = []
    clipping_by_scenario: dict[str, list[float]] = {name: [] for name in SCENARIOS}
    target_clipping: list[float] = []

    for repeat in range(REPEATS):
        observation_multiplier = rng.gamma(
            shape=GAMMA_SHAPE, scale=GAMMA_SCALE, size=(HEIGHT, WIDTH)
        )
        raw_observation = true_scene * observation_multiplier
        target_clipping.append(float(np.mean(raw_observation > 1.0)))
        observation = np.clip(raw_observation, 0.0, 1.0)

        # 同一重复内共享噪声实现，形成配对的场景条件比较。
        forecast_multipliers = rng.gamma(
            shape=GAMMA_SHAPE,
            scale=GAMMA_SCALE,
            size=(DRAW_COUNT, HEIGHT, WIDTH),
        )

        for scenario in SCENARIOS:
            raw_samples = scenes[scenario][None, :, :] * forecast_multipliers
            clipping = float(np.mean(raw_samples > 1.0))
            clipping_by_scenario[scenario].append(clipping)
            samples = np.clip(raw_samples, 0.0, 1.0)
            scores = _evaluate_scores(samples, observation)
            local = _evaluate_local(samples)
            records.append(
                {
                    "重复": repeat + 1,
                    "预测条件": scenario,
                    **scores,
                    **local,
                    "预测裁剪比例": clipping,
                    "目标裁剪比例": target_clipping[-1],
                }
            )

    with RAW_PATH.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)

    by_scenario = {
        scenario: [record for record in records if record["预测条件"] == scenario]
        for scenario in SCENARIOS
    }
    correct = by_scenario["正确"]
    summary: dict[str, object] = {
        "实验名称": protocol["实验名称"],
        "随机种子": SEED,
        "独立重复数": REPEATS,
        "每集合draw数": DRAW_COUNT,
        "场景": {},
        "相对正确条件的配对差": {},
        "Gamma参数抽样核验": {
            "样本数": int(gamma_check.size),
            "样本均值": gamma_mean,
            "理论均值": 1.0,
            "样本方差": gamma_variance,
            "理论方差": 1.0 / GAMMA_SHAPE,
        },
        "裁剪比例": {
            "目标均值": float(np.mean(target_clipping)),
            "各预测条件均值": {
                key: float(np.mean(values))
                for key, values in clipping_by_scenario.items()
            },
        },
        "输出记录数": len(records),
    }

    for scenario, rows in by_scenario.items():
        scenario_summary: dict[str, object] = {}
        for field in SCORE_FIELDS + LOCAL_FIELDS:
            values = np.asarray([float(row[field]) for row in rows])
            scenario_summary[field] = {
                "均值": float(np.mean(values)),
                "95%区间": _bootstrap_interval(values, bootstrap_rng),
            }
        summary["场景"][scenario] = scenario_summary

    for scenario in SCENARIOS[1:]:
        comparison: dict[str, object] = {}
        for field in SCORE_FIELDS + LOCAL_FIELDS:
            differences = np.asarray(
                [
                    float(alternative[field]) - float(reference[field])
                    for reference, alternative in zip(
                        correct, by_scenario[scenario], strict=True
                    )
                ]
            )
            comparison[field] = {
                "配对均值差": float(np.mean(differences)),
                "95%区间": _bootstrap_interval(differences, bootstrap_rng),
                "替代条件高于正确条件比例": float(np.mean(differences > 0.0)),
            }
        summary["相对正确条件的配对差"][scenario] = comparison

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
        "运行命令": "CUDA_VISIBLE_DEVICES='' python experiments/synthetic_cpu/c3a_bright_scatterer_metric_sensitivity_cpu_20261109_v1/run.py",
        "运行设备": "CPU",
        "运行环境": {
            "Python": platform.python_version(),
            "NumPy": np.__version__,
            "平台": platform.platform(),
            "Python可执行文件": sys.executable,
            "PyTorch已导入": "torch" in sys.modules,
        },
        "随机种子": SEED,
        "重复数": REPEATS,
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

    print(f"实验完成：{len(records)} 条逐重复记录，CPU运行。")
    print("条件 | CRPS配对差 | 能量分数配对差 | 变差分数配对差 | 精确亮点检出率")
    for scenario in SCENARIOS[1:]:
        paired = summary["相对正确条件的配对差"][scenario]
        peak_hit = summary["场景"][scenario]["精确位置检出率"]["均值"]
        print(
            f"{scenario} | "
            f"{paired['逐像素_CRPS']['配对均值差']:.7f} | "
            f"{paired['整图能量分数']['配对均值差']:.7f} | "
            f"{paired['邻接变差分数']['配对均值差']:.7f} | "
            f"{peak_hit:.4f}"
        )
    print("产物：逐重复结果.csv、汇总.json、运行清单.json")


if __name__ == "__main__":
    main()
