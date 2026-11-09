"""局部亮点阈值的解析敏感性诊断；仅使用 Python 标准库。"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path


OUTPUT_DIR = Path(__file__).resolve().parent
PROTOCOL_PATH = OUTPUT_DIR / "protocol.json"
CSV_PATH = OUTPUT_DIR / "逐阈值结果.csv"
SUMMARY_PATH = OUTPUT_DIR / "汇总.json"
REPORT_PATH = OUTPUT_DIR / "报告.md"
MANIFEST_PATH = OUTPUT_DIR / "运行清单.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def gamma_integer_shape_survival(threshold: float, mean: float, shape: int) -> float:
    """计算均值已知且形状为正整数的 Gamma 分布右尾概率。"""
    if threshold < 0 or mean <= 0 or shape <= 0:
        raise ValueError("阈值须非负，均值与形状须为正。")
    if threshold == 0:
        return 1.0

    x = threshold * shape / mean
    log_terms = [
        -x + k * math.log(x) - math.lgamma(k + 1)
        for k in range(shape)
    ]
    max_log = max(log_terms)
    log_probability = max_log + math.log(
        sum(math.exp(value - max_log) for value in log_terms)
    )
    if log_probability < math.log(float.fromhex("0x0.0000000000001p-1022")):
        return 0.0
    return min(1.0, math.exp(log_probability))


def probability_at_least_one(probability: float, trials: int) -> float:
    if not 0.0 <= probability <= 1.0 or trials < 0:
        raise ValueError("概率或试验次数无效。")
    if probability == 1.0:
        return 1.0 if trials else 0.0
    if trials == 0 or probability == 0.0:
        return 0.0
    return -math.expm1(trials * math.log1p(-probability))


def run() -> None:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    settings = protocol["固定场景"]
    height, width = settings["图幅"]
    background = settings["背景均值"]
    peaks = settings["目标均值"]
    shapes = settings["Gamma形状"]
    thresholds = settings["阈值"]
    draws = settings["预测draw数"]
    target_pixels = settings["目标像素数"]
    background_pixels = height * width - target_pixels
    fields = [
        "Gamma形状",
        "目标均值",
        "背景均值",
        "阈值",
        "单draw目标命中概率",
        "16draw目标至少一次命中概率",
        "单背景像素单draw误报概率",
        "整图16draw至少一次背景误报概率",
        "单图期望背景误报数",
    ]

    rows: list[dict[str, float | int]] = []
    for shape in shapes:
        for peak in peaks:
            for threshold in thresholds:
                target_hit = gamma_integer_shape_survival(threshold, peak, shape)
                background_false = gamma_integer_shape_survival(
                    threshold, background, shape
                )
                family_trials = background_pixels * draws
                row = {
                    "Gamma形状": shape,
                    "目标均值": peak,
                    "背景均值": background,
                    "阈值": threshold,
                    "单draw目标命中概率": target_hit,
                    "16draw目标至少一次命中概率": probability_at_least_one(
                        target_hit, draws
                    ),
                    "单背景像素单draw误报概率": background_false,
                    "整图16draw至少一次背景误报概率": probability_at_least_one(
                        background_false, family_trials
                    ),
                    "单图期望背景误报数": background_false * family_trials,
                }
                if not all(
                    math.isfinite(float(row[field]))
                    and 0.0 <= float(row[field]) <= 1.0
                    for field in fields[4:8]
                ):
                    raise ValueError("解析结果包含非有限值或概率越界。")
                rows.append(row)

    # 检查解析式特例：shape=1、mean=1 对应指数分布。
    for threshold in (0.1, 1.0, 4.0):
        observed = gamma_integer_shape_survival(threshold, 1.0, 1)
        expected = math.exp(-threshold)
        if not math.isclose(observed, expected, rel_tol=1e-13, abs_tol=1e-15):
            raise AssertionError("Gamma 右尾解析式未通过指数分布特例核验。")
    observed = gamma_integer_shape_survival(0.25, 1.0, 4)
    expected = math.exp(-1.0) * (1.0 + 1.0 + 0.5 + 1.0 / 6.0)
    if not math.isclose(observed, expected, rel_tol=1e-13, abs_tol=1e-15):
        raise AssertionError("Gamma 形状为 4 的已知右尾值核验失败。")

    # 阈值升高时，目标命中率和背景误报率均不得升高。
    for shape in shapes:
        for peak in peaks:
            selected = [
                row for row in rows
                if row["Gamma形状"] == shape and row["目标均值"] == peak
            ]
            for lower, upper in zip(selected, selected[1:]):
                if lower["单draw目标命中概率"] + 1e-14 < upper["单draw目标命中概率"]:
                    raise AssertionError("目标命中率不满足阈值单调性。")
        selected_background = [
            next(
                row for row in rows
                if row["Gamma形状"] == shape
                and row["目标均值"] == peaks[0]
                and row["阈值"] == threshold
            )
            for threshold in thresholds
        ]
        for lower, upper in zip(selected_background, selected_background[1:]):
            if lower["单背景像素单draw误报概率"] + 1e-14 < upper["单背景像素单draw误报概率"]:
                raise AssertionError("背景误报率不满足阈值单调性。")

    with CSV_PATH.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    threshold_index = thresholds.index(0.25)
    at_fixed_threshold = []
    for shape in shapes:
        for peak in peaks:
            at_fixed_threshold.append(
                next(
                    row for row in rows
                    if row["Gamma形状"] == shape
                    and row["目标均值"] == peak
                    and row["阈值"] == thresholds[threshold_index]
                )
            )

    summary = {
        "实验名称": protocol["实验名称"],
        "计算类型": "整数形状Gamma右尾概率的解析计算；未生成或读取影像。",
        "目标条件数": len(peaks) * len(shapes),
        "阈值数": len(thresholds),
        "输出记录数": len(rows),
        "固定阈值0.25结果": [
            {
                "Gamma形状": row["Gamma形状"],
                "目标均值": row["目标均值"],
                "单draw目标命中概率": row["单draw目标命中概率"],
                "16draw目标至少一次命中概率": row["16draw目标至少一次命中概率"],
                "单背景像素单draw误报概率": row["单背景像素单draw误报概率"],
                "整图16draw至少一次背景误报概率": row["整图16draw至少一次背景误报概率"],
                "单图期望背景误报数": row["单图期望背景误报数"],
            }
            for row in at_fixed_threshold
        ],
        "解释边界": "只在独立 Gamma 乘性数学构造与给定参数下成立；不估计项目噪声，也不代表真实模型、SAR 物理或正式评价阈值。",
        "验证": {
            "指数分布特例通过": True,
            "阈值单调性通过": True,
            "概率范围检查通过": True,
            "图像生成": False,
            "PyTorch导入": False,
            "项目载荷访问": False,
            "confirmation_test访问": False,
        },
    }
    SUMMARY_PATH.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    report_lines = [
        "# 局部亮点护栏阈值敏感性解析诊断",
        "",
        "## 协议与计算",
        "",
        "固定背景均值为 0.04，目标均值为 0.12、0.20、0.30、0.45，Gamma 形状为 4、16、64，阈值从 0.10 到 0.50（步长 0.05）。假设 64×64 图幅只有一个目标像素，每个预测集合含 16 个 draw，像素与 draw 相互独立。",
        "",
        "使用整数形状 Gamma 分布右尾解析式计算概率；本实验没有生成影像或抽样噪声。整图误报概率按 4,095 个背景像素乘 16 个 draw 的独立性计算。该设定只用于数学敏感性检查。",
        "",
        "## 固定阈值 0.25 的结果",
        "",
        "| Gamma 形状 | 目标均值 | 单 draw 命中率 | 16 draw 至少一次命中率 | 背景单像素误报率 | 整图至少一次误报率 | 单图期望背景误报数 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in at_fixed_threshold:
        report_lines.append(
            "| {shape} | {peak:.2f} | {hit:.6g} | {any_hit:.6g} | {fp:.6g} | {any_fp:.6g} | {count:.6g} |".format(
                shape=row["Gamma形状"],
                peak=row["目标均值"],
                hit=row["单draw目标命中概率"],
                any_hit=row["16draw目标至少一次命中概率"],
                fp=row["单背景像素单draw误报概率"],
                any_fp=row["整图16draw至少一次背景误报概率"],
                count=row["单图期望背景误报数"],
            )
        )
    hit_values = [float(row["单draw目标命中概率"]) for row in at_fixed_threshold]
    fwer_values = [
        float(row["整图16draw至少一次背景误报概率"])
        for row in at_fixed_threshold
    ]
    report_lines.extend(
        [
            "",
            "## 判断与边界",
            "",
            f"在阈值 0.25 下，单 draw 目标命中率跨全部目标均值与 Gamma 形状从 {min(hit_values):.6g} 到 {max(hit_values):.6g}；按独立像素假设得到的整图至少一次背景误报概率从 {min(fwer_values):.6g} 到 {max(fwer_values):.6g}。命中与误报会随目标幅值和噪声形状变化，故固定阈值不应被视为可迁移常数。",
            "",
            "- 完整 108 个组合（12 组目标/噪声参数、9 个阈值）见逐阈值 CSV；未按结果选择阈值，也未据此提出推荐值。",
            "- 整图误报概率依赖像素和 draw 独立假设；真实 SAR 的空间相关性、显示归一化及模型输出均未建模。",
            "- 不修改局部统计代码、C3-A v3、正式评价协议、模型或阶段 gate；不推断真实 SAR 命中/误报表现。",
            "- 仅使用 Python 标准库在 CPU 解析计算；没有导入 PyTorch，也没有访问项目图像、模型、checkpoint、confirmation/test 载荷或测量值。",
            "",
            "## 文件",
            "",
            "- [预登记协议](protocol.json)",
            "- [运行脚本](run.py)",
            "- [逐阈值结果](逐阈值结果.csv)",
            "- [汇总结果](汇总.json)",
            "- [运行清单](运行清单.json)",
        ]
    )
    REPORT_PATH.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    manifest = {
        "实验名称": protocol["实验名称"],
        "运行日期": "2026-11-09",
        "代码提交": "bc3bc0c3834a7b60bc59573a3c4b508bbb393164",
        "分支": "main",
        "工作区原有未提交状态": True,
        "运行命令": "python experiments/synthetic_cpu/c3a_bright_scatterer_threshold_sensitivity_cpu_20261109_v1/run.py",
        "运行设备": "CPU",
        "运行环境": {
            "Python": platform.python_version(),
            "平台": platform.platform(),
            "Python可执行文件": sys.executable,
            "依赖": "Python 标准库",
            "PyTorch已导入": False,
        },
        "随机性": "无随机抽样；按预登记 Gamma 右尾闭式计算。",
        "输出记录数": len(rows),
        "输入边界": {
            "项目数据或模型": False,
            "confirmation或test访问": False,
            "图像生成": False,
            "训练或推理": False,
            "GPU": False,
        },
        "文件SHA256": {
            "协议": sha256_file(PROTOCOL_PATH),
            "脚本": sha256_file(Path(__file__).resolve()),
            "逐阈值结果": sha256_file(CSV_PATH),
            "汇总": sha256_file(SUMMARY_PATH),
            "报告": sha256_file(REPORT_PATH),
        },
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(f"解析诊断完成：{len(rows)} 条记录，CPU运行。")
    print("固定阈值 0.25：目标均值 | Gamma形状 | 单draw命中率 | 整图至少一次背景误报率")
    for row in at_fixed_threshold:
        print(
            f"{row['目标均值']:.2f} | {row['Gamma形状']} | "
            f"{row['单draw目标命中概率']:.6g} | "
            f"{row['整图16draw至少一次背景误报概率']:.6g}"
        )


if __name__ == "__main__":
    run()
