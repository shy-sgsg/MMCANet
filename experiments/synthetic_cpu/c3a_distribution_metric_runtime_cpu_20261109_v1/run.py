#!/usr/bin/env python3
"""测量候选多样本评分量处理单张合成图像的 CPU 耗时。"""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.distribution_metrics import (
    adjacent_variogram_score,
    normalized_image_energy_score,
    pixelwise_crps,
)


RUN_DIR = Path(__file__).resolve().parent
PROTOCOL_PATH = RUN_DIR / "protocol.json"
SEED = 20261109
REPEATS = 7
DRAW_COUNT = 16
HEIGHT = 256
WIDTH = 256


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


def main() -> None:
    started = datetime.now(timezone.utc).isoformat()
    rng = np.random.default_rng(SEED)
    samples = rng.random((DRAW_COUNT, HEIGHT, WIDTH), dtype=np.float64)
    observation = rng.random((HEIGHT, WIDTH), dtype=np.float64)
    scorers = {
        "逐像素CRPS": pixelwise_crps,
        "整图能量分数": normalized_image_energy_score,
        "邻接变差分数": adjacent_variogram_score,
    }

    for scorer in scorers.values():
        value = scorer(samples, observation)
        if not np.isfinite(value):
            raise RuntimeError("预热阶段返回非有限分数")

    rows: list[dict[str, object]] = []
    summaries: dict[str, dict[str, float]] = {}
    for name, scorer in scorers.items():
        durations_ms: list[float] = []
        for repeat in range(1, REPEATS + 1):
            start_ns = time.perf_counter_ns()
            value = scorer(samples, observation)
            duration_ms = (time.perf_counter_ns() - start_ns) / 1_000_000.0
            if not np.isfinite(value):
                raise RuntimeError(f"{name} 第 {repeat} 次计时返回非有限分数")
            durations_ms.append(duration_ms)
            rows.append(
                {
                    "评分量": name,
                    "重复编号": repeat,
                    "耗时毫秒": duration_ms,
                    "评分值": value,
                }
            )

        values = np.asarray(durations_ms, dtype=np.float64)
        summaries[name] = {
            "中位数毫秒": float(np.median(values)),
            "第25百分位毫秒": float(np.percentile(values, 25)),
            "第75百分位毫秒": float(np.percentile(values, 75)),
            "最小毫秒": float(np.min(values)),
            "最大毫秒": float(np.max(values)),
        }

    csv_path = RUN_DIR / "逐次计时.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    result = {
        "实验名称": "多样本评分工具的单图像 CPU 运行时间诊断",
        "随机种子": SEED,
        "预测集合形状": [DRAW_COUNT, HEIGHT, WIDTH],
        "观测形状": [HEIGHT, WIDTH],
        "预热次数": 1,
        "每项计时重复": REPEATS,
        "评分量汇总": summaries,
        "解释限制": "仅为当前主机上的合成数组计时，不外推完整开发集或 GPU 运行时间。",
    }
    result_path = RUN_DIR / "结果.json"
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    finished = datetime.now(timezone.utc).isoformat()
    manifest = {
        "运行编号": "c3a_distribution_metric_runtime_cpu_20261109_v1",
        "运行类型": "合成数组 CPU 评分工具耗时诊断；非模型运行",
        "开始时间UTC": started,
        "结束时间UTC": finished,
        "命令": "python experiments/synthetic_cpu/c3a_distribution_metric_runtime_cpu_20261109_v1/run.py",
        "代码提交": git_text("rev-parse", "HEAD"),
        "工作树有改动": bool(git_text("status", "--porcelain")),
        "Python版本": sys.version.split()[0],
        "NumPy版本": np.__version__,
        "平台": platform.platform(),
        "处理器说明": platform.processor() or "系统未提供处理器名称",
        "设备": "CPU",
        "随机种子": SEED,
        "项目数据访问": False,
        "模型与检查点访问": False,
        "确认测试载荷访问": False,
        "文件SHA256": {
            "协议": sha256(PROTOCOL_PATH),
            "脚本": sha256(Path(__file__).resolve()),
            "逐次计时": sha256(csv_path),
            "汇总结果": sha256(result_path),
        },
    }
    (RUN_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print("多样本评分工具 CPU 耗时诊断完成。")
    print(f"合成输入：{DRAW_COUNT} 张 {HEIGHT}×{WIDTH} 图像；每项计时 {REPEATS} 次。")
    print(f"计时记录：{len(rows)} 条；没有读取项目数据或启动模型/GPU。")
    for name, summary in summaries.items():
        print(
            f"{name}：中位数 {summary['中位数毫秒']:.3f} 毫秒，"
            f"四分位距 {summary['第25百分位毫秒']:.3f}–"
            f"{summary['第75百分位毫秒']:.3f} 毫秒。"
        )
    print(f"产物目录：{RUN_DIR.relative_to(Path.cwd())}")


if __name__ == "__main__":
    main()
