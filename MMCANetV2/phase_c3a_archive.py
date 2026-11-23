"""C3-A development 生成样本归档及表示域高值目标代理指标。"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np


METRIC_FIELDS = (
    "高值区域绝对误差",
    "高值区域均值比",
    "高值区域峰值比",
    "高值饱和比例",
)
REQUIRED_BINDINGS = (
    "training_protocol_sha256",
    "evaluation_protocol_sha256",
    "archive_protocol_sha256",
    "source_manifest_sha256",
    "development_input_manifest_sha256",
    "flow_checkpoint_sha256",
    "baseline_checkpoint_sha256",
)
PROTECTED_FLAGS = (
    "confirmation_payload_accessed",
    "test_set_accessed",
    "test_set_used",
    "optical_payload_read",
)


def bright_proxy_metrics(
    prediction: np.ndarray, target: np.ndarray, *, quantile: float = 0.99
) -> dict[str, float | int]:
    """按 noisy pseudo-target 的固定高分位掩膜计算表示域亮值代理。"""
    pred = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    if pred.ndim != 2 or truth.ndim != 2 or pred.shape != truth.shape:
        raise ValueError("高值代理的预测和目标必须是尺寸相同的二维图像")
    if not np.isfinite(pred).all() or not np.isfinite(truth).all():
        raise ValueError("高值代理输入含非有限值")
    if truth.min() < 0.0 or truth.max() > 1.0 or pred.min() < 0.0 or pred.max() > 1.0:
        raise ValueError("高值代理输入必须位于归一化表示域 [0,1]")
    if not math.isfinite(float(quantile)) or not 0.0 < float(quantile) < 1.0:
        raise ValueError("高值目标分位必须位于 (0,1)")
    threshold = float(np.quantile(truth, float(quantile), method="linear"))
    mask = truth >= threshold
    count = int(mask.sum())
    if count == 0:
        raise ValueError("目标高值 mask 为空")
    target_mean = float(truth[mask].mean(dtype=np.float64))
    target_peak = float(truth[mask].max())
    if target_mean <= 1e-12 or target_peak <= 1e-12:
        raise ValueError("目标高值区域为零，不能计算表示域亮值比")
    return {
        "目标分位阈值": threshold,
        "目标高值像素数": count,
        "目标高值区域占比": float(count / truth.size),
        "高值区域绝对误差": float(np.mean(np.abs(pred[mask] - truth[mask]), dtype=np.float64)),
        "高值区域均值比": float(pred[mask].mean(dtype=np.float64) / target_mean),
        "高值区域峰值比": float(pred[mask].max() / target_peak),
        "高值饱和比例": float(np.mean(pred[mask] >= 1.0 - 1e-6)),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_arrays(
    sample_ids: Sequence[str],
    source_group_ids: Sequence[str],
    patch_ids: Sequence[str],
    targets: np.ndarray,
    flow_raw: np.ndarray,
    baseline_raw: np.ndarray,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...], np.ndarray, np.ndarray, np.ndarray]:
    identities = tuple(str(value) for value in sample_ids)
    groups = tuple(str(value) for value in source_group_ids)
    patches = tuple(str(value) for value in patch_ids)
    count = len(identities)
    if count == 0 or len(groups) != count or len(patches) != count:
        raise ValueError("样本 ID、来源组 ID 和 patch ID 必须等长且非空")
    if len(set(identities)) != count or any(not value.strip() for value in identities + groups + patches):
        raise ValueError("样本/来源组/patch 身份为空或样本 ID 重复")
    target_array = np.asarray(targets)
    flow_array = np.asarray(flow_raw)
    baseline_array = np.asarray(baseline_raw)
    if target_array.ndim != 4 or target_array.shape[1] != 1:
        raise ValueError("targets 必须为 [N,1,H,W]")
    if flow_array.ndim != 5 or flow_array.shape[1:3] != (4, 1):
        raise ValueError("flow_raw 必须为 [N,4,1,H,W]，对应冻结的四次 draw")
    if baseline_array.ndim != 4 or baseline_array.shape[1] != 1:
        raise ValueError("baseline_raw 必须为 [N,1,H,W]")
    if target_array.shape[0] != count or flow_array.shape[0] != count or baseline_array.shape[0] != count:
        raise ValueError("归档数组身份数与 ID 清单不一致")
    if target_array.shape[1:] != baseline_array.shape[1:] or flow_array.shape[2:] != target_array.shape[1:]:
        raise ValueError("Flow、SAR-only baseline 和目标的通道/空间尺寸不一致")
    for label, array in (("targets", target_array), ("flow_raw", flow_array), ("baseline_raw", baseline_array)):
        if array.dtype != np.float32 or not np.isfinite(array).all():
            raise ValueError(f"{label} 必须为有限 float32 数组")
    if target_array.min() < 0.0 or target_array.max() > 1.0:
        raise ValueError("noisy pseudo-target 超出冻结范围 [0,1]")
    return identities, groups, patches, target_array, flow_array, baseline_array


def _csv_write(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _identity_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["方法"], row["样本 ID"], row["来源组 ID"]), []).append(row)
    output: list[dict[str, Any]] = []
    for (method, sample_id, group_id), members in sorted(grouped.items()):
        record: dict[str, Any] = {"方法": method, "样本 ID": sample_id, "来源组 ID": group_id}
        for field in METRIC_FIELDS:
            record[field] = float(np.mean([float(item[field]) for item in members], dtype=np.float64))
        output.append(record)
    return output


def _group_rows(identity_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in identity_rows:
        grouped.setdefault((row["方法"], row["来源组 ID"]), []).append(row)
    output: list[dict[str, Any]] = []
    for (method, group_id), members in sorted(grouped.items()):
        record: dict[str, Any] = {"方法": method, "来源组 ID": group_id, "身份数": len(members)}
        for field in METRIC_FIELDS:
            record[field] = float(np.mean([float(item[field]) for item in members], dtype=np.float64))
        output.append(record)
    return output


def _paired_group_bootstrap(
    identity_rows: list[dict[str, Any]], *, seed: int, repeats: int
) -> dict[str, Any]:
    by_method = {
        method: {row["样本 ID"]: row for row in identity_rows if row["方法"] == method}
        for method in ("C3-A SAR-only Flow", "C2 SAR-only 确定性基线")
    }
    if set(by_method["C3-A SAR-only Flow"]) != set(by_method["C2 SAR-only 确定性基线"]):
        raise ValueError("高值代理 Flow 与 baseline 身份集合不一致")
    sample_to_group = {
        row["样本 ID"]: row["来源组 ID"] for row in by_method["C3-A SAR-only Flow"].values()
    }
    group_ids = sorted(set(sample_to_group.values()))
    if len(group_ids) < 2:
        raise ValueError("高值代理来源组 bootstrap 至少需要两个来源组")
    result: dict[str, Any] = {
        "来源组数": len(group_ids),
        "bootstrap 重复数": repeats,
        "bootstrap seed": seed,
        "抽样单位": "来源组；先对身份内 draw 求均值，再对来源组等权",
        "指标": {},
    }
    generator = np.random.default_rng(seed)
    for field in METRIC_FIELDS:
        group_delta = []
        for group_id in group_ids:
            sample_ids = [sample_id for sample_id, value in sample_to_group.items() if value == group_id]
            flow_mean = np.mean([float(by_method["C3-A SAR-only Flow"][sid][field]) for sid in sample_ids])
            baseline_mean = np.mean([float(by_method["C2 SAR-only 确定性基线"][sid][field]) for sid in sample_ids])
            group_delta.append(float(flow_mean - baseline_mean))
        values = np.asarray(group_delta, dtype=np.float64)
        draws = generator.integers(0, len(values), size=(repeats, len(values)))
        bootstrap_means = values[draws].mean(axis=1)
        result["指标"][field] = {
            "Flow 减基线来源组等权差": float(values.mean()),
            "bootstrap 95% 下限": float(np.quantile(bootstrap_means, 0.025)),
            "bootstrap 95% 上限": float(np.quantile(bootstrap_means, 0.975)),
        }
    return result


def write_c3a_sample_archive(
    output_directory: str | Path,
    *,
    sample_ids: Sequence[str],
    source_group_ids: Sequence[str],
    patch_ids: Sequence[str],
    targets: np.ndarray,
    flow_raw: np.ndarray,
    baseline_raw: np.ndarray,
    metadata: dict[str, Any],
    bootstrap_seed: int = 20261106,
    bootstrap_repeats: int = 10000,
) -> dict[str, Any]:
    """只创建新目录并归档四次 draw、基线预测及描述性亮值代理。"""
    started = time.perf_counter()
    archive_started_utc = datetime.now(timezone.utc).isoformat()
    destination = Path(output_directory)
    if destination.exists():
        raise FileExistsError(f"样本归档目录已存在：{destination}")
    if not isinstance(metadata, dict):
        raise ValueError("归档绑定元数据必须为对象")
    missing = [key for key in REQUIRED_BINDINGS if not isinstance(metadata.get(key), str) or len(metadata[key]) != 64]
    if missing:
        raise ValueError(f"归档缺少 SHA-256 绑定：{', '.join(missing)}")
    if any(metadata.get(key) is not False for key in PROTECTED_FLAGS):
        raise ValueError("归档绑定必须明确声明 optical/confirmation/test 未访问")
    if isinstance(bootstrap_seed, bool) or not isinstance(bootstrap_seed, int):
        raise ValueError("bootstrap seed 必须为整数")
    if isinstance(bootstrap_repeats, bool) or not isinstance(bootstrap_repeats, int) or bootstrap_repeats < 100:
        raise ValueError("来源组 bootstrap 重复数至少为 100")
    identities, groups, patches, target_array, flow_array, baseline_array = _validate_arrays(
        sample_ids, source_group_ids, patch_ids, targets, flow_raw, baseline_raw
    )

    flow_clipped = np.clip(flow_array, 0.0, 1.0).astype(np.float32, copy=False)
    baseline_clipped = np.clip(baseline_array, 0.0, 1.0).astype(np.float32, copy=False)
    destination.mkdir(parents=True, exist_ok=False)
    archive_path = destination / "generated_samples.npz"
    np.savez_compressed(
        archive_path,
        sample_ids=np.asarray(identities, dtype=np.str_),
        source_group_ids=np.asarray(groups, dtype=np.str_),
        patch_ids=np.asarray(patches, dtype=np.str_),
        draw_indices=np.arange(4, dtype=np.int8),
        flow_raw=flow_array,
        flow_clipped=flow_clipped,
        baseline_raw=baseline_array,
        baseline_clipped=baseline_clipped,
    )

    per_draw_rows: list[dict[str, Any]] = []
    for index, sample_id in enumerate(identities):
        target = target_array[index, 0]
        for draw_index in range(4):
            per_draw_rows.append(
                {
                    "方法": "C3-A SAR-only Flow",
                    "样本 ID": sample_id,
                    "来源组 ID": groups[index],
                    "patch ID": patches[index],
                    "draw 序号": draw_index,
                    **bright_proxy_metrics(flow_clipped[index, draw_index, 0], target),
                }
            )
        per_draw_rows.append(
            {
                "方法": "C2 SAR-only 确定性基线",
                "样本 ID": sample_id,
                "来源组 ID": groups[index],
                "patch ID": patches[index],
                "draw 序号": "确定性",
                **bright_proxy_metrics(baseline_clipped[index, 0], target),
            }
        )
    per_draw_path = destination / "bright_proxy_per_draw.csv"
    per_draw_fields = ["方法", "样本 ID", "来源组 ID", "patch ID", "draw 序号", "目标分位阈值", "目标高值像素数", "目标高值区域占比", *METRIC_FIELDS]
    _csv_write(per_draw_path, per_draw_rows, per_draw_fields)
    identities_rows = _identity_rows(per_draw_rows)
    group_rows = _group_rows(identities_rows)
    _csv_write(
        destination / "bright_proxy_source_group_summary.csv",
        group_rows,
        ["方法", "来源组 ID", "身份数", *METRIC_FIELDS],
    )
    bootstrap = _paired_group_bootstrap(identities_rows, seed=bootstrap_seed, repeats=bootstrap_repeats)
    (destination / "bright_proxy_group_bootstrap.json").write_text(
        json.dumps(bootstrap, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    runtime_path = destination / "runtime.json"
    runtime = {
        "归档开始 UTC": archive_started_utc,
        "归档数据写入秒数": time.perf_counter() - started,
        "Flow 评价秒数": metadata.get("flow_runtime_seconds"),
        "SAR-only 基线评价秒数": metadata.get("baseline_runtime_seconds"),
        "Flow 峰值 GPU 已分配字节": metadata.get("flow_peak_gpu_memory_allocated_bytes"),
        "SAR-only 基线峰值 GPU 已分配字节": metadata.get("baseline_peak_gpu_memory_allocated_bytes"),
        "评价开始前 GPU 空闲字节": metadata.get("gpu_free_memory_before_evaluation_bytes"),
    }
    runtime_path.write_text(json.dumps(runtime, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    manifest: dict[str, Any] = {
        "format": "c3a_development_sample_archive_v1",
        "sample_count": len(identities),
        "source_group_count": len(set(groups)),
        "draws_per_identity": 4,
        "样本数": len(identities),
        "来源组数": len(set(groups)),
        "每身份 Flow draw 数": 4,
        "generated_draw_count": len(identities) * 4,
        "亮值代理定义": "每身份 noisy pseudo-target 第 99 百分位阈值，含并列像素；归一化表示域描述，不作物理散射体解释。",
        "选择或阶段 gate 改变": False,
        "confirmation_payload_accessed": False,
        "test_set_accessed": False,
        "test_set_used": False,
        "optical_payload_read": False,
        "metadata": metadata,
        "bootstrap_seed": bootstrap_seed,
        "bootstrap_repeats": bootstrap_repeats,
        "files_sha256": {
            "generated_samples.npz": _sha256(archive_path),
            "bright_proxy_per_draw.csv": _sha256(per_draw_path),
            "bright_proxy_source_group_summary.csv": _sha256(destination / "bright_proxy_source_group_summary.csv"),
            "bright_proxy_group_bootstrap.json": _sha256(destination / "bright_proxy_group_bootstrap.json"),
            "runtime.json": _sha256(runtime_path),
        },
        "archive_sha256": _sha256(archive_path),
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return manifest
