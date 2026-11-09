"""用各向异性高斯场诊断变差分数对横纵相关错配的敏感性。"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


RUN_DIR = Path(__file__).resolve().parent
SCORING_RUNNER = (
    RUN_DIR.parent
    / "c3a_spatial_variogram_multiscale_cpu_20261109_v1"
    / "run.py"
)
TRUTHS = {
    "横向相关更强": {"x": 0.75, "y": 0.35},
    "纵向相关更强": {"x": 0.35, "y": 0.75},
}
REPEATS = 300
DRAW_COUNT = 16
HEIGHT = 24
WIDTH = 24
DATA_SEED = 20261112
BOOTSTRAP_SEED = 20261113
BOOTSTRAP_REPEATS = 2_000


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_scoring_implementation() -> Any:
    spec = importlib.util.spec_from_file_location(
        "spatial_variogram_base", SCORING_RUNNER
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("无法加载已核验的评分实现")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def covariance_cholesky(correlation: float) -> np.ndarray:
    coordinates = np.arange(HEIGHT, dtype=np.float64)
    distance = np.abs(coordinates[:, None] - coordinates[None, :])
    covariance = correlation**distance
    return np.linalg.cholesky(covariance)


def make_fields(
    cholesky_x: np.ndarray, cholesky_y: np.ndarray, noise: np.ndarray
) -> np.ndarray:
    gaussian = np.matmul(np.matmul(cholesky_y, noise), cholesky_x.T)
    return 1.0 / (1.0 + np.exp(-gaussian))


def bootstrap_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in records:
        groups.setdefault((row["真值方向"], row["评分量"]), []).append(row)

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    output: dict[str, Any] = {}
    metric_names = sorted({row["评分量"] for row in records})
    for truth in TRUTHS:
        output[truth] = {}
        for metric in metric_names:
            rows = groups[(truth, metric)]
            delta = np.asarray(
                [row["错误预测分数减正确预测分数"] for row in rows],
                dtype=np.float64,
            )
            bootstrap_means = np.empty(BOOTSTRAP_REPEATS, dtype=np.float64)
            for index in range(BOOTSTRAP_REPEATS):
                selection = rng.integers(0, delta.size, size=delta.size)
                bootstrap_means[index] = delta[selection].mean()
            output[truth][metric] = {
                "重复数": int(delta.size),
                "错误减正确平均分": float(delta.mean()),
                "平均分差95百分位区间": [
                    float(np.quantile(bootstrap_means, 0.025)),
                    float(np.quantile(bootstrap_means, 0.975)),
                ],
                "正确模型逐目标排序比例": float(np.mean(delta > 0.0)),
                "平分比例": float(np.mean(delta == 0.0)),
            }
    return output


def main() -> None:
    if (RUN_DIR / "results.json").exists() or (RUN_DIR / "per_trial.csv").exists():
        raise FileExistsError(f"结果文件已存在，拒绝覆盖：{RUN_DIR}")

    started_utc = datetime.now(timezone.utc).isoformat()
    scoring = load_scoring_implementation()
    cholesky_x = {
        correlation: covariance_cholesky(correlation)
        for correlation in (0.35, 0.75)
    }
    cholesky_y = cholesky_x
    rng = np.random.default_rng(DATA_SEED)
    records: list[dict[str, Any]] = []

    for truth, correlation in TRUTHS.items():
        competitor = {"x": correlation["y"], "y": correlation["x"]}
        correct_scores = []
        wrong_scores = []
        for repeat in range(REPEATS):
            base_noise = rng.standard_normal((DRAW_COUNT + 1, HEIGHT, WIDTH))
            observation = make_fields(
                cholesky_x[correlation["x"]],
                cholesky_y[correlation["y"]],
                base_noise[:1],
            )[0]
            forecast_noise = base_noise[1:]
            correct = make_fields(
                cholesky_x[correlation["x"]],
                cholesky_y[correlation["y"]],
                forecast_noise,
            )
            wrong = make_fields(
                cholesky_x[competitor["x"]],
                cholesky_y[competitor["y"]],
                forecast_noise,
            )
            correct_scores.append(scoring.score_set(correct, observation))
            wrong_scores.append(scoring.score_set(wrong, observation))
            for metric, correct_score in correct_scores[-1].items():
                wrong_score = wrong_scores[-1][metric]
                records.append(
                    {
                        "真值方向": truth,
                        "重复编号": repeat + 1,
                        "评分量": metric,
                        "正确预测横向相关": correlation["x"],
                        "正确预测纵向相关": correlation["y"],
                        "竞争预测横向相关": competitor["x"],
                        "竞争预测纵向相关": competitor["y"],
                        "正确预测分数": correct_score,
                        "竞争预测分数": wrong_score,
                        "错误预测分数减正确预测分数": wrong_score - correct_score,
                        "正确排序": wrong_score > correct_score,
                    }
                )
        print(f"方向错配诊断进度：{truth}，完成 {REPEATS}/{REPEATS} 个目标。", flush=True)

    summary = bootstrap_summary(records)
    results = {
        "实验编号": RUN_DIR.name,
        "评分实现": "复用 c3a_spatial_variogram_multiscale_cpu_20261109_v1/run.py",
        "输入": {
            "图像尺寸": [HEIGHT, WIDTH],
            "横纵相关值": [0.35, 0.75],
            "真值方向": list(TRUTHS),
            "每个真值方向重复数": REPEATS,
            "预测样本数": DRAW_COUNT,
            "变差 lag": [1, 2, 4, 8],
        },
        "汇总规则": {
            "差值方向": "错误预测分数减正确预测分数；正值表示正确预测得分较低",
            "bootstrap重复数": BOOTSTRAP_REPEATS,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "重抽样单位": "每个真值方向内的独立目标",
        },
        "条件汇总": summary,
    }
    results_path = RUN_DIR / "results.json"
    results_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    csv_path = RUN_DIR / "per_trial.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    manifest = {
        "实验编号": RUN_DIR.name,
        "状态": "完成",
        "开始时间_UTC": started_utc,
        "结束时间_UTC": datetime.now(timezone.utc).isoformat(),
        "运行命令": "python experiments/synthetic_cpu/c3a_spatial_variogram_anisotropy_cpu_20261109_v1/run.py",
        "Python版本": sys.version.split()[0],
        "NumPy版本": np.__version__,
        "操作系统": platform.platform(),
        "设备": "CPU；未导入 PyTorch，也未调用 GPU",
        "协议_SHA256": sha256_file(RUN_DIR / "protocol.md"),
        "脚本_SHA256": sha256_file(Path(__file__).resolve()),
        "评分实现_SHA256": sha256_file(SCORING_RUNNER),
        "结果_SHA256": sha256_file(results_path),
        "逐重复CSV_SHA256": sha256_file(csv_path),
        "项目数据访问": False,
        "模型或检查点访问": False,
        "确认集载荷访问": False,
        "测试载荷访问": False,
        "训练或推理": False,
        "备注": "独立方向错配诊断；不改变多样本补充设计、C3-A v3 或任何 gate。",
    }
    (RUN_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    for truth, metrics in summary.items():
        multi = metrics["多尺度变差 h=1,2,4,8"]
        ci = multi["平均分差95百分位区间"]
        h1 = metrics["单尺度变差 h=1"]["正确模型逐目标排序比例"]
        print(
            f"{truth}：h=1 排序 {h1:.3f}；多尺度排序 "
            f"{multi['正确模型逐目标排序比例']:.3f}，"
            f"平均分差95%区间 [{ci[0]:.8f}, {ci[1]:.8f}]。",
            flush=True,
        )
    print(f"结果已写入：{RUN_DIR}", flush=True)


if __name__ == "__main__":
    main()
