"""检验多尺度变差分数对匹配一阶相关强度的敏感性。"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


RUN_DIR = Path(__file__).resolve().parent
BASELINE_RUNNER = (
    RUN_DIR.parent
    / "c3a_spatial_variogram_multiscale_cpu_20261109_v1"
    / "run.py"
)
CORRELATIONS = (0.4, 0.6, 0.8, 0.9)
FAMILIES = ("AR(1)", "RBF")
REPEATS = 300
DRAW_COUNT = 16
HEIGHT = 24
WIDTH = 24
LAGS = (1, 2, 4, 8)
DATA_SEED = 20261110
BOOTSTRAP_SEED = 20261111
BOOTSTRAP_REPEATS = 2_000


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_scoring_implementation() -> Any:
    spec = importlib.util.spec_from_file_location("spatial_variogram_base", BASELINE_RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError("无法加载已核验的评分实现")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.HEIGHT = HEIGHT
    module.WIDTH = WIDTH
    module.DRAW_COUNT = DRAW_COUNT
    module.LAGS = LAGS
    return module


def bootstrap_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    groups: dict[tuple[float, str, str], list[dict[str, Any]]] = {}
    for row in records:
        groups.setdefault((row["一阶相关"], row["真值族"], row["评分量"]), []).append(row)

    output: dict[str, Any] = {}
    for correlation in CORRELATIONS:
        correlation_key = f"{correlation:.1f}"
        output[correlation_key] = {}
        for truth in FAMILIES:
            output[correlation_key][truth] = {}
            for metric in sorted({row["评分量"] for row in records}):
                rows = groups[(correlation, truth, metric)]
                delta = np.asarray(
                    [row["错误预测分数减正确预测分数"] for row in rows],
                    dtype=np.float64,
                )
                bootstrap_means = np.empty(BOOTSTRAP_REPEATS, dtype=np.float64)
                for index in range(BOOTSTRAP_REPEATS):
                    selection = rng.integers(0, delta.size, size=delta.size)
                    bootstrap_means[index] = delta[selection].mean()
                output[correlation_key][truth][metric] = {
                    "重复数": int(delta.size),
                    "错误减正确平均分": float(delta.mean()),
                    "平均分差95百分位区间": [
                        float(np.quantile(bootstrap_means, 0.025)),
                        float(np.quantile(bootstrap_means, 0.975)),
                    ],
                    "正确模型逐重复排序比例": float(np.mean(delta > 0.0)),
                    "平分比例": float(np.mean(delta == 0.0)),
                }
    return output


def main() -> None:
    if (RUN_DIR / "results.json").exists() or (RUN_DIR / "per_trial.csv").exists():
        raise FileExistsError(f"结果文件已存在，拒绝覆盖：{RUN_DIR}")
    started_utc = datetime.now(timezone.utc).isoformat()
    scoring = load_scoring_implementation()
    rng = np.random.default_rng(DATA_SEED)
    records: list[dict[str, Any]] = []
    total_targets = len(CORRELATIONS) * len(FAMILIES) * REPEATS
    completed = 0

    for correlation in CORRELATIONS:
        scoring.CORRELATION = correlation
        cholesky = {
            family: scoring.covariance_cholesky(family) for family in FAMILIES
        }
        for truth in FAMILIES:
            competitor = "RBF" if truth == "AR(1)" else "AR(1)"
            for repeat in range(REPEATS):
                base_noise = rng.standard_normal((DRAW_COUNT + 1, HEIGHT, WIDTH))
                observation = scoring.make_fields(cholesky[truth], base_noise[:1])[0]
                predictions = {
                    family: scoring.make_fields(cholesky[family], base_noise[1:])
                    for family in FAMILIES
                }
                scores = {
                    family: scoring.score_set(predictions[family], observation)
                    for family in FAMILIES
                }
                for metric, correct_score in scores[truth].items():
                    wrong_score = scores[competitor][metric]
                    records.append(
                        {
                            "一阶相关": correlation,
                            "真值族": truth,
                            "重复编号": repeat + 1,
                            "评分量": metric,
                            "正确预测族": truth,
                            "竞争预测族": competitor,
                            "正确预测分数": correct_score,
                            "竞争预测分数": wrong_score,
                            "错误预测分数减正确预测分数": wrong_score - correct_score,
                            "正确排序": wrong_score > correct_score,
                        }
                    )
                completed += 1
        print(f"相关强度敏感性进度：ρ={correlation:.1f}，累计 {completed}/{total_targets} 个目标。", flush=True)

    summary = bootstrap_summary(records)
    results = {
        "实验编号": RUN_DIR.name,
        "评分实现": "复用 c3a_spatial_variogram_multiscale_cpu_20261109_v1/run.py 的评分函数",
        "输入": {
            "图像尺寸": [HEIGHT, WIDTH],
            "一阶相关": list(CORRELATIONS),
            "协方差族": list(FAMILIES),
            "每个相关强度和真值族重复数": REPEATS,
            "预测样本数": DRAW_COUNT,
            "变差 lag": list(LAGS),
        },
        "汇总规则": {
            "差值方向": "错误预测分数减正确预测分数；正值表示正确预测获得更低分",
            "bootstrap重复数": BOOTSTRAP_REPEATS,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "重抽样单位": "每个相关强度/真值族条件内的独立目标",
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
        "运行命令": "python experiments/synthetic_cpu/c3a_spatial_variogram_correlation_sweep_cpu_20261109_v1/run.py",
        "Python版本": sys.version.split()[0],
        "NumPy版本": np.__version__,
        "操作系统": platform.platform(),
        "设备": "CPU；未导入 PyTorch，也未调用 GPU",
        "协议_SHA256": sha256_file(RUN_DIR / "protocol.md"),
        "脚本_SHA256": sha256_file(Path(__file__).resolve()),
        "评分实现_SHA256": sha256_file(BASELINE_RUNNER),
        "结果_SHA256": sha256_file(results_path),
        "逐重复CSV_SHA256": sha256_file(csv_path),
        "项目数据访问": False,
        "模型或检查点访问": False,
        "确认集载荷访问": False,
        "测试载荷访问": False,
        "训练或推理": False,
        "备注": "上一项合成空间依赖诊断的参数敏感性延伸；不改变 C3-A v3 或任何 gate。",
    }
    (RUN_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    for correlation_key, truths in summary.items():
        for truth, metrics in truths.items():
            h1 = metrics["单尺度变差 h=1"]["正确模型逐重复排序比例"]
            multi = metrics["多尺度变差 h=1,2,4,8"]
            ci = multi["平均分差95百分位区间"]
            print(
                f"ρ={correlation_key}、真值={truth}："
                f"h=1 排序 {h1:.3f}；多尺度排序 {multi['正确模型逐重复排序比例']:.3f}，"
                f"平均分差95%区间 [{ci[0]:.8f}, {ci[1]:.8f}]。",
                flush=True,
            )
    print(f"结果已写入：{RUN_DIR}", flush=True)


if __name__ == "__main__":
    main()
