#!/usr/bin/env python3
"""在冻结 C2 v2 screen 因 median TBR 未定义时，生成不参与门槛的补充开发集诊断。"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for entry in (ROOT, SCRIPTS):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import evaluate_phase_c2_screen as frozen_screen  # noqa: E402
from MMCANetV2.phase_c2_data import (  # noqa: E402
    C2FixedDataset,
    sha256_file,
    verify_frozen_c2_source_manifest,
)
from MMCANetV2.phase_c2_runner import audit_c2_cross_condition_parity  # noqa: E402
from MMCANetV2.phase_c2_screen import (  # noqa: E402
    C2_CONTROL_ALIASES,
    C2_CONTROL_IDS,
    C2_DIRECTIONAL_METRICS,
    C2_HIGHER_IS_BETTER,
    C2_RETURN_GUARDRAIL_METRICS,
    build_cross_group_permutation,
    evaluate_c2_diagnostic_sample_controls,
    summarize_group_deltas,
)

_UNAVAILABLE_METRICS = frozenset(
    {"tbr_median_abs_error", "abs_log_median_tbr_ratio_error"}
)
_AVAILABLE_METRICS = tuple(
    metric for metric in C2_DIRECTIONAL_METRICS if metric not in _UNAVAILABLE_METRICS
)
_RUN_ID = "c2_non_gate_diagnostic_v1"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_nested(path: Path, other: Path) -> bool:
    return path == other or path.is_relative_to(other) or other.is_relative_to(path)


def _write_json(path: Path, value: Mapping[str, object], *, exclusive: bool = False) -> None:
    mode = "x" if exclusive else "w"
    with path.open(mode, encoding="utf-8") as stream:
        json.dump(dict(value), stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _write_rows(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"拒绝写入空结果表：{path.name}")
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for name in row:
            if name not in seen:
                fields.append(name)
                seen.add(name)
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())


def _build_partial_contrasts(
    method_rows: Sequence[Mapping[str, object]],
    protocol: Mapping[str, object],
    expected_identities: Sequence[tuple[str, str, str]],
) -> tuple[list[dict[str, object]], dict[str, int]]:
    by_key: dict[tuple[str, str, str], Mapping[str, object]] = {}
    condition_ids = set(protocol["conditions"])
    undefined_counts = {condition: 0 for condition in condition_ids}
    undefined_ids: set[tuple[str, str]] = set()
    for row in method_rows:
        key = (str(row["condition_id"]), str(row["control"]), str(row["sample_id"]))
        if key in by_key:
            raise ValueError("补充诊断发现重复的条件/控制/样本行")
        by_key[key] = row
        if row.get("confirmation_payload_accessed") is not False or row.get("test_set_accessed") is not False or row.get("test_set_used") is not False:
            raise ValueError("补充诊断的逐图访问边界标记无效")
        if row.get("tbr_median_status") != "defined":
            undefined_ids.add((key[0], key[2]))
    for condition, _sample, _group in expected_identities:
        undefined_counts[condition] += int((condition, _sample) in undefined_ids)

    expected_samples = {sample for _condition, sample, _group in expected_identities}
    if len(expected_samples) != len(expected_identities) // len(condition_ids):
        raise ValueError("补充诊断的开发样本身份不唯一")
    expected_keys = {
        (condition, control, sample)
        for condition in condition_ids
        for control in C2_CONTROL_IDS
        for sample in expected_samples
    }
    if set(by_key) != expected_keys:
        raise ValueError("补充诊断的条件/控制/开发样本覆盖与冻结集合不一致")

    output: list[dict[str, object]] = []
    for condition, condition_spec in protocol["conditions"].items():
        condition_identities = {
            sample: group
            for cond, sample, group in expected_identities
            if cond == condition
        }
        if set(condition_identities) != expected_samples:
            raise ValueError("跨条件开发样本集合不一致")
        for contrast in protocol["contrasts"]:
            candidate = C2_CONTROL_ALIASES[str(contrast["candidate"])]
            comparator = C2_CONTROL_ALIASES[str(contrast["comparator"])]
            contrast_id = f"{contrast['candidate']}_vs_{contrast['comparator']}"
            for sample in sorted(expected_samples):
                candidate_row = by_key[(condition, candidate, sample)]
                comparator_row = by_key[(condition, comparator, sample)]
                expected_identity = (condition_identities[sample], str(candidate_row["patch_id"]))
                for row in (candidate_row, comparator_row):
                    if (
                        str(row["source_group_id"]) != expected_identity[0]
                        or str(row["patch_id"]) != expected_identity[1]
                    ):
                        raise ValueError("同一配对样本的来源组或 patch 身份不一致")
                result: dict[str, object] = {
                    "condition_id": condition,
                    "scale": int(condition_spec["scale"]),
                    "contrast_id": contrast_id,
                    "candidate_control": candidate,
                    "comparator_control": comparator,
                    "sample_id": sample,
                    "source_group_id": expected_identity[0],
                    "patch_id": expected_identity[1],
                    "unavailable_median_tbr_metrics": ";".join(sorted(_UNAVAILABLE_METRICS)),
                }
                for metric in _AVAILABLE_METRICS:
                    candidate_value = float(candidate_row[metric])
                    comparator_value = float(comparator_row[metric])
                    if not math.isfinite(candidate_value) or not math.isfinite(comparator_value):
                        raise ValueError(f"补充诊断发现非有限指标：{metric}")
                    delta = candidate_value - comparator_value
                    gain = delta if metric in C2_HIGHER_IS_BETTER else -delta
                    result[f"candidate_{metric}"] = candidate_value
                    result[f"comparator_{metric}"] = comparator_value
                    result[f"delta_{metric}"] = delta
                    result[f"gain_{metric}"] = gain
                output.append(result)
    return output, undefined_counts


def _summarize_partial_contrasts(
    paired_rows: Sequence[Mapping[str, object]], protocol: Mapping[str, object]
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    samples = int(protocol["statistics"]["bootstrap_samples"])
    seed = int(protocol["statistics"]["bootstrap_seed"])
    summary_rows: list[dict[str, object]] = []
    group_rows: list[dict[str, object]] = []
    for condition in sorted(protocol["conditions"]):
        contrast_ids = sorted(
            {
                str(row["contrast_id"])
                for row in paired_rows
                if row["condition_id"] == condition
            }
        )
        for contrast_id in contrast_ids:
            rows = [
                row
                for row in paired_rows
                if row["condition_id"] == condition and row["contrast_id"] == contrast_id
            ]
            for metric in _AVAILABLE_METRICS:
                direction = "higher_is_better" if metric in C2_HIGHER_IS_BETTER else "lower_is_better"
                metric_rows = [
                    {
                        "source_group_id": row["source_group_id"],
                        "candidate": row[f"candidate_{metric}"],
                        "comparator": row[f"comparator_{metric}"],
                    }
                    for row in rows
                ]
                summary = summarize_group_deltas(
                    metric_rows, direction, samples=samples, seed=seed
                )
                summary_rows.append(
                    {
                        "condition_id": condition,
                        "contrast_id": contrast_id,
                        "metric": metric,
                        "benefit_direction": direction,
                        "gate_eligible": False,
                        **summary,
                    }
                )
                grouped: dict[str, list[float]] = {}
                for row in rows:
                    grouped.setdefault(str(row["source_group_id"]), []).append(
                        float(row[f"gain_{metric}"])
                    )
                for group, gains in sorted(grouped.items()):
                    group_mean = float(np.mean(gains))
                    group_rows.append(
                        {
                            "condition_id": condition,
                            "contrast_id": contrast_id,
                            "metric": metric,
                            "source_group_id": group,
                            "sample_count": len(gains),
                            "group_mean_gain": group_mean,
                            "positive_gain": group_mean > 0,
                            "gate_eligible": False,
                        }
                    )
    return summary_rows, group_rows


def _report_text(
    *,
    protocol: Mapping[str, object],
    summary_rows: Sequence[Mapping[str, object]],
    undefined_counts: Mapping[str, int],
    method_count: int,
    paired_count: int,
) -> str:
    lines = [
        "# C2 v2 开发集补充诊断（非冻结门槛）",
        "",
        "## 目的与限制",
        "",
        "原 C2 v2 配对 screen 因必需的 median TBR 在少数开发样本上未定义，按冻结规则 fail-closed。此补充运行保留全部开发样本，仅把 `tbr_median_abs_error` 和 `abs_log_median_tbr_ratio_error` 标记为不可用；不加 epsilon、不删样本、不作晋级或资源决策。",
        "",
        f"协议：`{protocol['protocol_id']}`。结果是单 seed、19 个开发来源组上的描述性观察；同一 development 已用于训练期 checkpoint 选择，不能作为独立泛化证据。",
        "",
        "## 未定义中位数覆盖",
        "",
        "| 条件 | median TBR 未定义的开发样本数 |",
        "|---|---:|",
    ]
    for condition in sorted(undefined_counts):
        lines.append(f"| {condition} | {undefined_counts[condition]} / 200 |")
    lines.extend(
        [
            "",
            "## 开发集 PSNR 配对差值",
            "",
            "正值表示候选方法优于对照；bootstrap 按来源组等权。",
            "",
            "| 条件 | 对比 | 来源组平均增益（dB） | 95% 来源组 bootstrap 区间 | 正增益来源组 |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for condition in sorted(protocol["conditions"]):
        for contrast_id in ("optical_guided_matched_vs_sar_only", "optical_guided_matched_vs_optical_guided_permuted"):
            row = next(
                item
                for item in summary_rows
                if item["condition_id"] == condition
                and item["contrast_id"] == contrast_id
                and item["metric"] == "psnr"
            )
            lines.append(
                f"| {condition} | {contrast_id} | {float(row['group_mean_gain']):+.4f} | "
                f"[{float(row['bootstrap_ci_low']):+.4f}, {float(row['bootstrap_ci_high']):+.4f}] | "
                f"{row['positive_group_count']} / {row['group_count']} |"
            )
    primary_guardrails = tuple(
        metric
        for metric in ("ssim", "epi_corr", *C2_RETURN_GUARDRAIL_METRICS)
        if metric not in _UNAVAILABLE_METRICS
    )
    lines.extend(
        [
            "",
            "## 解释边界",
            "",
            f"本次生成 {method_count} 条逐图控制指标与 {paired_count} 条逐图配对对比。所有已定义指标均覆盖完整的 200 个 development 身份；median TBR 两项明确缺失，因而冻结 C2 v2 决策函数没有运行。",
            "该诊断不改写原失败的 screen、不改变冻结协议、不构成 continue/stop/inconclusive 决策、不晋级 B1，也不支持独立泛化或辐射物理结论。",
            f"未定义 median TBR 之外的结构/回波护栏包括：{', '.join(primary_guardrails)}。完整结果见 CSV。",
            "",
            "## 访问边界",
            "",
            "只读取 train/development 固定输入和八组 best checkpoint；confirmation/test 影像、测量和指标均未读取。",
            "",
        ]
    )
    return "\n".join(lines)


def run_non_gate_diagnostic(
    *,
    source_manifest: str | Path,
    input_root: str | Path,
    matrix_root: str | Path,
    failed_screen_manifest: str | Path,
    output_root: str | Path,
) -> dict[str, object]:
    source_manifest = Path(source_manifest).resolve()
    input_root = Path(input_root).resolve()
    matrix_root = Path(matrix_root).resolve()
    failed_screen_manifest = Path(failed_screen_manifest).resolve()
    output_root = Path(output_root).resolve()
    protocol, protocol_sha256 = verify_frozen_c2_source_manifest(source_manifest)
    matrix_path = matrix_root / "manifest.json"
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    specs = frozen_screen.validate_c2_screen_matrix_manifest(matrix, protocol, protocol_sha256)
    if not input_root.is_dir():
        raise FileNotFoundError("C2 固定输入根目录不存在")
    if output_root.exists():
        raise FileExistsError("拒绝覆盖已存在的补充诊断目录")
    if any(
        _is_nested(output_root, path)
        for path in (source_manifest, input_root, matrix_root, failed_screen_manifest)
    ):
        raise ValueError("补充诊断输出目录不得与来源、输入或训练结果路径重叠")

    failed_screen = json.loads(failed_screen_manifest.read_text(encoding="utf-8"))
    if (
        failed_screen.get("status") != "failed"
        or "Target background median is zero or too small" not in str(failed_screen.get("failure_message", ""))
        or failed_screen.get("protocol_sha256") != protocol_sha256
        or failed_screen.get("source_manifest_sha256") != protocol.get("source_manifest_sha256")
        or failed_screen.get("confirmation_payload_accessed") is not False
        or failed_screen.get("test_set_accessed") is not False
        or failed_screen.get("test_set_used") is not False
    ):
        raise ValueError("指定的原配对 screen 失败记录与本补充诊断不匹配")

    condition_ids = tuple(protocol["conditions"])
    current_inputs = frozen_screen._current_development_input_records(input_root, condition_ids)
    frozen_inputs = matrix["identity"].get("input_records")
    if not isinstance(frozen_inputs, Mapping) or any(
        frozen_inputs.get(key) != record for key, record in current_inputs.items()
    ):
        raise ValueError("当前 development 输入与训练矩阵记录不一致")
    parity = audit_c2_cross_condition_parity(
        {condition: input_root / condition / "development" for condition in condition_ids},
        source_manifest,
        role="development",
    )
    resource_preflight = frozen_screen.preflight_c2_cuda_inference(protocol)

    datasets: dict[str, C2FixedDataset] = {}
    for condition in condition_ids:
        scale = int(protocol["conditions"][condition]["scale"])
        dataset = C2FixedDataset(
            input_root / condition / "development",
            source_manifest,
            role="development",
            scale=scale,
            verify_hashes_on_read=False,
        )
        if len(dataset) != int(protocol["roles"]["development"]["samples"]):
            raise ValueError(f"{condition} 的开发样本数量与冻结协议不符")
        dataset.verify_all()
        datasets[condition] = dataset

    reference_rows = datasets[condition_ids[0]].rows
    expected_identities = [
        (condition_ids[0], row["sample_id"], row["source_group_id"])
        for row in reference_rows
    ]
    for condition in condition_ids[1:]:
        rows = datasets[condition].rows
        if [(row["sample_id"], row["source_group_id"], row["patch_id"]) for row in rows] != [
            (row["sample_id"], row["source_group_id"], row["patch_id"])
            for row in reference_rows
        ]:
            raise ValueError("四条件 development 样本身份或顺序不一致")
        expected_identities.extend(
            (condition, row["sample_id"], row["source_group_id"]) for row in rows
        )
    permutation_rows = build_cross_group_permutation(
        [{"role": "development", **row} for row in reference_rows],
        offset=int(protocol["permutation"]["offset"]),
    )
    permutation = {
        str(row["receiver_sample_id"]): str(row["donor_sample_id"])
        for row in permutation_rows
    }
    if len(permutation) != len(reference_rows):
        raise ValueError("冻结的跨来源组 optical 错配没有覆盖全部开发样本")

    output_root.mkdir(parents=True, exist_ok=False)
    execution_manifest: dict[str, object] = {
        "format": "c2_non_gate_diagnostic_v1",
        "status": "running",
        "diagnostic_id": _RUN_ID,
        "purpose": "describe_defined_metrics_after_frozen_fail_closed_screen_failure",
        "gate_eligible": False,
        "c2_v2_gate_decision": "not_evaluated",
        "excluded_undefined_metrics": sorted(_UNAVAILABLE_METRICS),
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_sha256,
        "source_manifest_sha256": protocol["source_manifest_sha256"],
        "failed_screen_manifest": str(failed_screen_manifest),
        "failed_screen_manifest_sha256": sha256_file(failed_screen_manifest),
        "matrix_manifest_sha256": sha256_file(matrix_path),
        "input_records": current_inputs,
        "parity_audit": parity,
        "cuda_preflight": resource_preflight,
        "git": frozen_screen._git_facts(),
        "environment": frozen_screen._environment_facts(),
        "argv": sys.argv,
        "training_run_count": len(specs),
        "development_sample_count": len(reference_rows),
        "development_source_group_count": len({row["source_group_id"] for row in reference_rows}),
        "completed_conditions": [],
        "confirmation_payload_accessed": False,
        "test_set_accessed": False,
        "test_set_used": False,
        "started_utc": _utc_now(),
    }
    execution_manifest_path = output_root / "execution_manifest.json"
    _write_json(execution_manifest_path, execution_manifest, exclusive=True)
    started = time.perf_counter()
    method_rows: list[dict[str, object]] = []
    try:
        torch.use_deterministic_algorithms(bool(protocol["training"]["deterministic_algorithms"]))
        torch.backends.cudnn.benchmark = bool(protocol["training"]["cudnn_benchmark"])
        torch.cuda.reset_peak_memory_stats(torch.device("cuda:0"))
        for condition in condition_ids:
            rows = frozen_screen._evaluate_condition(
                condition_id=condition,
                dataset=datasets[condition],
                matrix_root=matrix_root,
                matrix=matrix,
                protocol=protocol,
                protocol_sha256=protocol_sha256,
                permutation=permutation,
                sample_evaluator=evaluate_c2_diagnostic_sample_controls,
            )
            if len(rows) != len(reference_rows) * len(C2_CONTROL_IDS):
                raise ValueError(f"{condition} 的四控制逐图行数不完整")
            method_rows.extend(rows)
            execution_manifest["completed_conditions"].append(condition)
            execution_manifest["method_metric_row_count"] = len(method_rows)
            execution_manifest["updated_utc"] = _utc_now()
            _write_json(execution_manifest_path, execution_manifest)

        expected_method_count = len(condition_ids) * len(reference_rows) * len(C2_CONTROL_IDS)
        if len(method_rows) != expected_method_count:
            raise ValueError("补充诊断的逐图控制行数不符合预期")
        paired_rows, undefined_counts = _build_partial_contrasts(
            method_rows, protocol, expected_identities
        )
        summary_rows, group_rows = _summarize_partial_contrasts(paired_rows, protocol)
        _write_rows(output_root / "per_image_metrics.csv", method_rows)
        _write_rows(output_root / "per_image_contrasts.csv", paired_rows)
        _write_rows(output_root / "group_summary.csv", summary_rows)
        _write_rows(output_root / "source_group_metrics.csv", group_rows)
        _write_rows(output_root / "permutation.csv", permutation_rows)
        report_path = output_root / "diagnostic_report.md"
        report_path.write_text(
            _report_text(
                protocol=protocol,
                summary_rows=summary_rows,
                undefined_counts=undefined_counts,
                method_count=len(method_rows),
                paired_count=len(paired_rows),
            ),
            encoding="utf-8",
        )
        files = [
            "per_image_metrics.csv",
            "per_image_contrasts.csv",
            "group_summary.csv",
            "source_group_metrics.csv",
            "permutation.csv",
            "diagnostic_report.md",
        ]
        execution_manifest.update(
            status="complete",
            metric_summary_count=len(summary_rows),
            per_image_metric_row_count=len(method_rows),
            per_image_contrast_row_count=len(paired_rows),
            undefined_median_tbr_sample_count_by_condition=undefined_counts,
            gate_eligible=False,
            c2_v2_gate_decision="not_evaluated",
            files={name: sha256_file(output_root / name) for name in files},
            runtime_seconds=time.perf_counter() - started,
            peak_allocated_memory_bytes=int(torch.cuda.max_memory_allocated(torch.device("cuda:0"))),
            ended_utc=_utc_now(),
        )
        _write_json(execution_manifest_path, execution_manifest)
        print(f"C2 非门槛开发集诊断完成：{output_root}", flush=True)
        print(f"逐图控制记录：{len(method_rows)}；配对对比：{len(paired_rows)}", flush=True)
        print("冻结 C2 v2 门槛判定：未运行；确认集/test 载荷访问：否。", flush=True)
        return execution_manifest
    except Exception as exc:
        execution_manifest.update(
            status="failed",
            failure_type=type(exc).__name__,
            failure_message=str(exc),
            runtime_seconds=time.perf_counter() - started,
            failed_utc=_utc_now(),
        )
        _write_json(execution_manifest_path, execution_manifest)
        raise


def main() -> None:
    parser = frozen_screen.ChineseArgumentParser(description=__doc__, add_help=False)
    parser._optionals.title = "可选参数"
    parser.add_argument("-h", "--help", action="help", help="显示此帮助并退出")
    parser.add_argument("--source-manifest", type=Path, required=True, help="冻结 C1 v2 来源清单")
    parser.add_argument("--input-root", type=Path, required=True, help="固定 C2 输入目录")
    parser.add_argument("--matrix-root", type=Path, required=True, help="冻结 C2 八组训练目录")
    parser.add_argument("--failed-screen-manifest", type=Path, required=True, help="原 fail-closed screen 的执行记录")
    parser.add_argument("--output-root", type=Path, required=True, help="全新的非门槛诊断输出目录")
    args = parser.parse_args()
    try:
        run_non_gate_diagnostic(
            source_manifest=args.source_manifest,
            input_root=args.input_root,
            matrix_root=args.matrix_root,
            failed_screen_manifest=args.failed_screen_manifest,
            output_root=args.output_root,
        )
    except Exception as exc:
        print(f"C2 非门槛诊断失败：{exc}", file=sys.stderr)
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
