"""执行 SARptical 空间分组 SAR-only 表示域 pilot。"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import platform
import random
import shlex
import shutil
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MMCANetV2.phase_c3a_metrics import edge_metrics, sobel_magnitude, ssim
from MMCANetV2.phase_c3a_flow import SarConditionalFlow, sample_heun
from experiments.phase_d1_edsr_20261124_v1.edsr_sar import (
    build_edsr_sar,
    load_published_edsr_class,
)
from experiments.published_baselines_sar_only_20261123_v1 import train_swinir_sar
from experiments.phase_d_sarptical_spatial_pilot_20261125 import pilot_data


METHODS = ("bicubic", "edsr", "swinir", "flow_bridge")
TRAINED_METHODS = ("edsr", "swinir", "flow_bridge")
SEED = 20261125
BOOTSTRAP_SEED = 20261125
EPOCHS = 30
BATCH_SIZE = 4
PATCH_LR = 48
LEARNING_RATE = 1e-4
FLOW_SIGMA = 0.5
FLOW_STEPS = 8
FLOW_DRAWS = 4
BOOTSTRAP_DRAWS = 10_000
METRIC_COLUMNS = (
    "psnr_db",
    "ssim",
    "rmse",
    "epi_corr",
    "edge_strength_ratio",
    "edge_strength_abs_error",
    "bright_mean_abs_error",
    "bright_abs_error_p95",
    "background_bias",
    "background_abs_bias",
    "lr_reprojection_rmse",
)


def write_json(path: Path, value: Any) -> None:
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"拒绝写入空 CSV：{path}")
    with Path(path).open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def append_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path = Path(path)
    exists = path.exists()
    with path.open("a" if exists else "x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        if not exists:
            writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())


def sha256_file(path: Path) -> str:
    return pilot_data.sha256_file(Path(path))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_edrs_model() -> nn.Module:
    """加载 vendored EDSR 并移除 [0,1] 头，适配无界标准化 dB 域。"""
    model = build_edsr_sar(load_published_edsr_class())
    model.core.mean = torch.zeros_like(model.core.mean)
    model.output = nn.Identity()
    model.experiment_config.update(
        {
            "身份": "TASK_ADAPTED_PUBLISHED",
            "输出适配": "无界线性输出；关闭 sigmoid；标准化 SAR dB 表示域",
            "输入均值": 0.0,
            "初始化": "随机初始化，不使用作者权重",
        }
    )
    return model


def load_swinir_model() -> nn.Module:
    """加载已审阅 SwinIR 主体，改用单通道无界线性输出。"""
    model = train_swinir_sar.build_swinir(train_swinir_sar.load_swinir_class())
    model.output = nn.Identity()
    model.experiment_config.update(
        {
            "身份": "TASK_ADAPTED_PUBLISHED",
            "输出适配": "无界线性输出；关闭 sigmoid；标准化 SAR dB 表示域",
            "初始化": "随机初始化，不使用作者权重",
        }
    )
    return model


def build_model(method: str) -> nn.Module:
    if method == "edsr":
        return load_edrs_model()
    if method == "swinir":
        return load_swinir_model()
    if method == "flow_bridge":
        model = SarConditionalFlow(base_channels=16, time_dimension=64)
        model.experiment_config = {
            "身份": "PROPOSED_METHOD_CANDIDATE",
            "方法": "双三次锚定条件 Flow Bridge（探索性候选）",
            "base_channels": 16,
            "time_dimension": 64,
            "sigma": FLOW_SIGMA,
            "heun_steps": FLOW_STEPS,
            "输出": "标准化 SAR dB 表示域速度场",
        }
        return model
    raise ValueError(f"未知训练方法：{method}")


def parameter_count(model: nn.Module) -> dict[str, int]:
    return {
        "total": sum(parameter.numel() for parameter in model.parameters()),
        "trainable": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
    }


def flow_bridge_pair(
    target: Tensor,
    anchor: Tensor,
    time: Tensor,
    noise: Tensor,
    *,
    sigma: float = FLOW_SIGMA,
) -> tuple[Tensor, Tensor]:
    """构造 x0=anchor+sigma*noise 的直线条件 flow bridge。"""
    if not math.isfinite(float(sigma)) or sigma <= 0:
        raise ValueError("Flow Bridge sigma 必须为有限正数")
    if target.shape != anchor.shape or target.shape != noise.shape or time.shape != (target.shape[0],):
        raise ValueError("Flow Bridge target/anchor/noise/time 的张量契约不匹配")
    if not all(torch.isfinite(value).all() for value in (target, anchor, noise, time)):
        raise ValueError("Flow Bridge 输入含非有限值")
    source = anchor + float(sigma) * noise
    fraction = time[:, None, None, None].to(dtype=target.dtype)
    state = (1.0 - fraction) * source + fraction * target
    velocity = target - source
    if not torch.isfinite(state).all() or not torch.isfinite(velocity).all():
        raise FloatingPointError("Flow Bridge 路径产生非有限张量")
    return state, velocity


def train_batch(
    method: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    lr: Tensor,
    target: Tensor,
    anchor: Tensor,
    *,
    flow_generator: torch.Generator | None = None,
    sigma: float = FLOW_SIGMA,
) -> dict[str, float]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    if method in {"edsr", "swinir"}:
        prediction = model(lr)
        if prediction.shape != target.shape:
            raise ValueError(f"{method} 输出与 HR target 尺寸不匹配")
        loss = F.l1_loss(prediction, target)
    elif method == "flow_bridge":
        if flow_generator is None:
            flow_generator = torch.Generator(device=target.device).manual_seed(SEED)
        time = torch.rand(
            (target.shape[0],), device=target.device, dtype=target.dtype, generator=flow_generator
        )
        noise = torch.randn(target.shape, device=target.device, dtype=target.dtype, generator=flow_generator)
        state, velocity = flow_bridge_pair(target, anchor, time, noise, sigma=sigma)
        prediction = model(state, anchor, time)
        if prediction.shape != velocity.shape:
            raise ValueError("Flow Bridge 速度场输出尺寸不匹配")
        loss = F.mse_loss(prediction, velocity)
    else:
        raise ValueError(f"不支持训练方法：{method}")
    if not torch.isfinite(loss):
        raise FloatingPointError(f"{method} 训练损失含 NaN 或 Inf")
    loss.backward()
    squared_norm = torch.zeros((), device=target.device, dtype=torch.float64)
    for parameter in model.parameters():
        if parameter.grad is not None:
            if not torch.isfinite(parameter.grad).all():
                raise FloatingPointError(f"{method} 梯度含 NaN 或 Inf")
            squared_norm += parameter.grad.detach().double().square().sum()
    gradient_norm = float(torch.sqrt(squared_norm).cpu())
    if not math.isfinite(gradient_norm):
        raise FloatingPointError(f"{method} 梯度范数不是有限数")
    optimizer.step()
    return {"loss": float(loss.detach().cpu()), "gradient_norm": gradient_norm}


def make_epoch_schedule(
    records: list[dict[str, Any]], *, seed: int, patch_lr: int = PATCH_LR
) -> list[dict[str, Any]]:
    """为一个 epoch 固定样本顺序、裁剪坐标和同步几何增强。"""
    if not records:
        raise ValueError("训练记录为空")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(records))
    schedule = []
    for index in order:
        lr = records[int(index)]["lr"]
        height, width = lr.shape[-2:]
        if patch_lr > height or patch_lr > width:
            raise ValueError("训练裁剪尺寸大于 LR 图像")
        schedule.append(
            {
                "record_index": int(index),
                "top": int(rng.integers(0, height - patch_lr + 1)),
                "left": int(rng.integers(0, width - patch_lr + 1)),
                "horizontal_flip": bool(rng.integers(0, 2)),
                "vertical_flip": bool(rng.integers(0, 2)),
                "transpose": bool(rng.integers(0, 2)),
            }
        )
    return schedule


def schedule_digest(schedule: list[list[dict[str, Any]]]) -> str:
    payload = json.dumps(schedule, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _patch(record: dict[str, Any], row: dict[str, Any], patch_lr: int) -> tuple[np.ndarray, ...]:
    top, left = int(row["top"]), int(row["left"])
    lr = record["lr"][:, top : top + patch_lr, left : left + patch_lr]
    hr_top, hr_left, patch_hr = top * 2, left * 2, patch_lr * 2
    target = record["target"][:, hr_top : hr_top + patch_hr, hr_left : hr_left + patch_hr]
    anchor = record["anchor"][:, hr_top : hr_top + patch_hr, hr_left : hr_left + patch_hr]
    if row["horizontal_flip"]:
        lr, target, anchor = lr[..., ::-1], target[..., ::-1], anchor[..., ::-1]
    if row["vertical_flip"]:
        lr, target, anchor = lr[..., ::-1, :], target[..., ::-1, :], anchor[..., ::-1, :]
    if row["transpose"]:
        lr, target, anchor = lr.transpose(0, 2, 1), target.transpose(0, 2, 1), anchor.transpose(0, 2, 1)
    return tuple(np.ascontiguousarray(value, dtype=np.float32) for value in (lr, target, anchor))


def _batch_from_schedule(
    records: list[dict[str, Any]], rows: list[dict[str, Any]], *, patch_lr: int, device: torch.device
) -> tuple[Tensor, Tensor, Tensor]:
    patches = [_patch(records[row["record_index"]], row, patch_lr) for row in rows]
    lr, target, anchor = [np.stack([sample[index] for sample in patches]) for index in range(3)]
    return tuple(torch.from_numpy(array).to(device) for array in (lr, target, anchor))


def _psnr(prediction: np.ndarray, target: np.ndarray, data_range: float) -> float:
    mse = float(np.mean((prediction.astype(np.float64) - target.astype(np.float64)) ** 2))
    if not math.isfinite(mse):
        raise FloatingPointError("PSNR 均方误差不是有限数")
    return math.inf if mse == 0 else float(10.0 * math.log10(float(data_range) ** 2 / mse))


def group_balanced_mean(rows: list[dict[str, Any]], metric: str = "psnr_db") -> float:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = row.get(metric)
        if value is None or value == "NA":
            continue
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError(f"来源组汇总指标 {metric} 含非有限值")
        grouped[str(row["source_group_id"])].append(numeric)
    if not grouped:
        raise ValueError(f"没有可用于来源组等权汇总的 {metric} 结果")
    return float(np.mean([np.mean(values) for values in grouped.values()]))


def _fixed_noise(sample_id: str, *, draw_index: int, size: tuple[int, int] = (112, 112)) -> np.ndarray:
    key = f"{SEED}:{sample_id}:draw:{draw_index}".encode("utf-8")
    seed = int.from_bytes(hashlib.sha256(key).digest()[:8], "little")
    return np.random.default_rng(seed).standard_normal((1, *size), dtype=np.float32)


def _predict_batch(
    model: nn.Module,
    method: str,
    records: list[dict[str, Any]],
    *,
    start: int,
    end: int,
    device: torch.device,
    draw_index: int = 0,
) -> Tensor:
    lr = torch.from_numpy(np.stack([record["lr"] for record in records[start:end]])).to(device)
    anchor = torch.from_numpy(np.stack([record["anchor"] for record in records[start:end]])).to(device)
    if method in {"edsr", "swinir"}:
        return model(lr)
    noises = np.stack(
        [_fixed_noise(record["sample_id"], draw_index=draw_index)[0] for record in records[start:end]]
    )[:, None]
    noise = torch.from_numpy(noises).to(device)
    return sample_heun(model, anchor, noise, steps=FLOW_STEPS, noise_sigma=FLOW_SIGMA)


def _selection_rows(
    model: nn.Module,
    method: str,
    records: list[dict[str, Any]],
    *,
    device: torch.device,
    data_range: float,
) -> list[dict[str, Any]]:
    model.eval()
    rows = []
    with torch.inference_mode():
        for start in range(0, len(records), BATCH_SIZE):
            end = min(start + BATCH_SIZE, len(records))
            prediction = _predict_batch(
                model, method, records, start=start, end=end, device=device, draw_index=0
            ).float().cpu().numpy()
            for local, index in enumerate(range(start, end)):
                target = records[index]["target"][0]
                rows.append(
                    {
                        "sample_id": records[index]["sample_id"],
                        "source_group_id": records[index]["source_group_id"],
                        "psnr_db": _psnr(prediction[local, 0], target, data_range),
                    }
                )
    return rows


def write_prediction_manifest(
    prediction_dir: Path,
    sample_ids: list[str],
    *,
    method: str = "unspecified",
    draw_index: int | None = None,
) -> Path:
    prediction_dir = Path(prediction_dir)
    manifest = {
        "schema_version": "sarptical_prediction_manifest_v1",
        "method": method,
        "draw_index": draw_index,
        "sample_ids": list(sample_ids),
        "prediction_shape": [112, 112],
        "prediction_dtype": "float32",
        "test_set_used": False,
        "confirmation_payload_accessed": False,
        "optical_payload_loaded": False,
    }
    path = prediction_dir / "inference_manifest.json"
    write_json(path, manifest)
    return path


def _metric_masks(anchor: np.ndarray, evaluation_config: dict[str, Any]) -> dict[str, np.ndarray]:
    edge_source = sobel_magnitude(anchor)
    high_threshold = float(evaluation_config["high_return_threshold"])
    background_threshold = float(evaluation_config["background_threshold"])
    edge_threshold = float(evaluation_config["edge_threshold"])
    if not high_threshold > background_threshold:
        raise ValueError("固定高回波阈值必须大于背景阈值")
    return {
        "high_return": anchor >= high_threshold,
        "background": anchor <= background_threshold,
        "edge": edge_source >= edge_threshold,
    }


def metric_row(
    prediction: np.ndarray,
    record: dict[str, Any],
    evaluation_config: dict[str, Any],
) -> dict[str, Any]:
    prediction = np.asarray(prediction)
    target = np.asarray(record["target"], dtype=np.float32)
    if target.shape == (1, 112, 112):
        target = target[0]
    anchor = np.asarray(record["anchor"], dtype=np.float32)
    if anchor.shape == (1, 112, 112):
        anchor = anchor[0]
    lr = np.asarray(record["lr"], dtype=np.float32)
    if lr.shape != (1, 56, 56):
        raise ValueError(f"{record['sample_id']} LR 形状不匹配")
    if (
        prediction.shape != (112, 112)
        or prediction.dtype != np.float32
        or target.shape != (112, 112)
        or anchor.shape != (112, 112)
    ):
        raise ValueError(f"{record['sample_id']} prediction/target/anchor 尺寸或类型错误")
    if any(not np.isfinite(value).all() for value in (prediction, target, anchor, lr)):
        raise ValueError(f"{record['sample_id']} 评价输入含 NaN 或 Inf")
    data_range = float(evaluation_config["data_range"])
    if not math.isfinite(data_range) or data_range <= 0:
        raise ValueError("评价 data_range 必须为正的有限数")

    difference = prediction.astype(np.float64) - target.astype(np.float64)
    mse = float(np.mean(difference**2))
    if not math.isfinite(mse) or mse <= 0:
        raise ValueError(f"{record['sample_id']} PSNR 均方误差无效")
    masks = _metric_masks(anchor, evaluation_config)
    row: dict[str, Any] = {
        "sample_id": record["sample_id"],
        "source_group_id": record["source_group_id"],
        "psnr_db": float(10.0 * math.log10(data_range**2 / mse)),
        "ssim": ssim(prediction, target, data_range=data_range),
        "rmse": float(math.sqrt(mse)),
        "lr_reprojection_rmse": None,
    }
    prediction_tensor = torch.from_numpy(np.ascontiguousarray(prediction))[None, None]
    projected_lr = F.interpolate(
        prediction_tensor,
        size=(56, 56),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    )[0, 0].numpy()
    row["lr_reprojection_rmse"] = float(np.sqrt(np.mean((projected_lr - lr[0]) ** 2)))

    minimum_mask_pixels = int(evaluation_config.get("minimum_mask_pixels", 16))
    if minimum_mask_pixels < 1:
        raise ValueError("固定 mask 最小像素数必须为正数")
    edge_mask = masks["edge"]
    row["edge_mask_pixels"] = int(edge_mask.sum())
    if int(edge_mask.sum()) >= minimum_mask_pixels:
        try:
            edge = edge_metrics(prediction, target, edge_mask)
            gt_edge = sobel_magnitude(target)[edge_mask]
            pred_edge = sobel_magnitude(prediction)[edge_mask]
            row["epi_corr"] = float(edge["epi_corr"])
            row["edge_strength_ratio"] = float(edge["edge_strength_ratio"])
            row["edge_strength_abs_error"] = float(abs(pred_edge.mean() - gt_edge.mean()))
        except ValueError:
            row["epi_corr"] = "NA"
            row["edge_strength_ratio"] = "NA"
            row["edge_strength_abs_error"] = "NA"
    else:
        row["epi_corr"] = "NA"
        row["edge_strength_ratio"] = "NA"
        row["edge_strength_abs_error"] = "NA"

    high_mask = masks["high_return"]
    row["high_return_mask_pixels"] = int(high_mask.sum())
    if int(high_mask.sum()) >= minimum_mask_pixels:
        high_error = np.abs(difference[high_mask])
        row["bright_mean_abs_error"] = float(high_error.mean())
        row["bright_abs_error_p95"] = float(np.percentile(high_error, 95, method="linear"))
    else:
        row["bright_mean_abs_error"] = "NA"
        row["bright_abs_error_p95"] = "NA"

    background_mask = masks["background"]
    row["background_mask_pixels"] = int(background_mask.sum())
    row["background_bias"] = (
        float(difference[background_mask].mean())
        if int(background_mask.sum()) >= minimum_mask_pixels
        else "NA"
    )
    row["background_abs_bias"] = (
        float(abs(row["background_bias"])) if row["background_bias"] != "NA" else "NA"
    )
    for metric in METRIC_COLUMNS:
        value = row.get(metric)
        if value is None or value == "NA" or not math.isfinite(float(value)):
            if value != "NA":
                raise FloatingPointError(f"{record['sample_id']} 指标 {metric} 非有限")
    return row


def evaluate_saved_predictions(
    records: list[dict[str, Any]],
    prediction_dir: Path,
    prediction_manifest: Path,
    evaluation_config: dict[str, Any],
    expected_sample_ids: list[str],
) -> list[dict[str, Any]]:
    """按冻结顺序、文件名集合、尺寸与有限性 fail-closed 评价。"""
    prediction_dir = Path(prediction_dir)
    expected_sample_ids = list(expected_sample_ids)
    if len(expected_sample_ids) != len(set(expected_sample_ids)):
        raise ValueError("评价清单含重复 sample_id")
    record_ids = [record["sample_id"] for record in records]
    if record_ids != expected_sample_ids:
        raise ValueError("评价 records 顺序与冻结 manifest 不一致")
    try:
        with Path(prediction_manifest).open("r", encoding="utf-8") as stream:
            manifest = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("推理 manifest 缺失或无法解析") from exc
    if manifest.get("sample_ids") != expected_sample_ids:
        raise ValueError("推理 manifest 样本顺序与冻结 manifest 不一致")
    if manifest.get("test_set_used") is not False or manifest.get("confirmation_payload_accessed") is not False:
        raise ValueError("推理 manifest 的保护集 flags 不安全")
    expected = set(expected_sample_ids)
    actual = {path.stem for path in prediction_dir.glob("*.npy")}
    if actual != expected:
        missing, extra = sorted(expected - actual), sorted(actual - expected)
        raise ValueError(f"prediction 文件集合不一致；缺少={missing[:3]}，额外={extra[:3]}")
    rows = []
    for record in records:
        path = prediction_dir / f"{record['sample_id']}.npy"
        try:
            prediction = np.load(path, allow_pickle=False)
        except Exception as exc:
            raise ValueError(f"prediction 无法读取：{record['sample_id']}") from exc
        if prediction.dtype != np.float32 or prediction.shape != (112, 112):
            raise ValueError(f"prediction 尺寸或类型错误：{record['sample_id']}")
        if not np.isfinite(prediction).all():
            raise ValueError(f"prediction 含 NaN 或 Inf：{record['sample_id']}")
        rows.append(metric_row(prediction, record, evaluation_config))
    return rows


def paired_group_bootstrap(
    candidate_rows: list[dict[str, Any]],
    reference_rows: list[dict[str, Any]],
    metric: str,
    *,
    seed: int = BOOTSTRAP_SEED,
    draws: int = BOOTSTRAP_DRAWS,
) -> dict[str, Any]:
    candidate = {row["sample_id"]: row for row in candidate_rows}
    reference = {row["sample_id"]: row for row in reference_rows}
    if not candidate or set(candidate) != set(reference):
        raise ValueError("配对 bootstrap 的 sample_id 集合不一致或为空")
    groups: dict[str, list[float]] = defaultdict(list)
    for sample_id, row in candidate.items():
        base = reference[sample_id]
        if row["source_group_id"] != base["source_group_id"]:
            raise ValueError(f"配对 bootstrap 的来源组不匹配：{sample_id}")
        left, right = row.get(metric), base.get(metric)
        if left == "NA" or right == "NA" or left is None or right is None:
            continue
        delta = float(left) - float(right)
        if not math.isfinite(delta):
            raise ValueError(f"配对 bootstrap 指标 {metric} 含非有限值")
        groups[str(row["source_group_id"])].append(delta)
    if not groups:
        return {
            "metric": metric,
            "group_count": 0,
            "sample_count": 0,
            "group_balanced_delta": None,
            "ci95_lower": None,
            "ci95_upper": None,
            "group_deltas": {},
            "bootstrap_seed": int(seed),
            "bootstrap_draws": int(draws),
            "unavailable_reason": "没有有效的配对样本",
        }
    group_names = sorted(groups)
    group_deltas = np.asarray([np.mean(groups[name]) for name in group_names], dtype=np.float64)
    if draws < 1:
        raise ValueError("bootstrap 次数必须为正数")
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(group_deltas), size=(draws, len(group_deltas)))
    bootstrap_values = group_deltas[samples].mean(axis=1)
    return {
        "metric": metric,
        "group_count": len(group_names),
        "sample_count": int(sum(map(len, groups.values()))),
        "group_balanced_delta": float(group_deltas.mean()),
        "ci95_lower": float(np.quantile(bootstrap_values, 0.025, method="linear")),
        "ci95_upper": float(np.quantile(bootstrap_values, 0.975, method="linear")),
        "group_deltas": {name: float(value) for name, value in zip(group_names, group_deltas)},
        "bootstrap_seed": int(seed),
        "bootstrap_draws": int(draws),
        "unavailable_reason": None,
    }


def fixed_evaluation_config(train_records: list[dict[str, Any]]) -> dict[str, Any]:
    if not train_records:
        raise ValueError("训练记录为空，不能冻结评价范围")
    targets = np.concatenate([record["target"].reshape(-1) for record in train_records]).astype(np.float64)
    anchors = np.concatenate([record["anchor"].reshape(-1) for record in train_records]).astype(np.float64)
    edge_values = np.concatenate(
        [sobel_magnitude(record["anchor"][0]).reshape(-1) for record in train_records]
    ).astype(np.float64)
    data_range = float(
        np.percentile(targets, 99.5, method="linear")
        - np.percentile(targets, 0.5, method="linear")
    )
    config = {
        "data_range": data_range,
        "data_range_definition": "train 标准化 target 的 P99.5−P0.5",
        "high_return_threshold": float(np.percentile(anchors, 99.0, method="linear")),
        "high_return_definition": "train bicubic LR anchor 像素 P99",
        "background_threshold": float(np.percentile(anchors, 10.0, method="linear")),
        "background_definition": "train bicubic LR anchor 像素 P10",
        "edge_threshold": float(np.percentile(edge_values, 90.0, method="linear")),
        "edge_definition": "train bicubic LR anchor Sobel 幅值 P90",
        "minimum_mask_pixels": 16,
        "sobel_implementation": "OpenCV Sobel ksize=3, BORDER_REFLECT_101",
        "ssim_implementation": "11×11 Gaussian，sigma=1.5，裁去 5 像素边界",
        "confirmation_payload_accessed": False,
        "test_set_used": False,
        "optical_payload_loaded": False,
    }
    if not math.isfinite(data_range) or data_range <= 0:
        raise ValueError("train 目标的评价 data_range 无效")
    if config["high_return_threshold"] <= config["background_threshold"]:
        raise ValueError("train 固定高回波与背景阈值次序错误")
    return config


def _save_prediction(path: Path, prediction: np.ndarray) -> None:
    value = np.asarray(prediction, dtype=np.float32)
    if value.shape != (112, 112) or not np.isfinite(value).all():
        raise ValueError(f"拒绝保存尺寸错误或非有限 prediction：{path.name}")
    with path.open("xb") as stream:
        np.save(stream, value, allow_pickle=False)


def infer_method(
    method: str,
    model: nn.Module | None,
    records: list[dict[str, Any]],
    *,
    output_dir: Path,
    evaluation_config: dict[str, Any],
    device: torch.device,
    draw_index: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, np.ndarray]]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    if model is not None:
        model.eval()
    expected_ids = [record["sample_id"] for record in records]
    predictions: dict[str, np.ndarray] = {}
    batch_seconds: list[float] = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        for start in range(0, len(records), BATCH_SIZE):
            end = min(start + BATCH_SIZE, len(records))
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            if method == "bicubic":
                output = torch.from_numpy(
                    np.stack([record["anchor"] for record in records[start:end]])
                ).to(device)
            elif model is not None:
                output = _predict_batch(
                    model,
                    "flow_bridge" if method.startswith("flow_bridge") else method,
                    records,
                    start=start,
                    end=end,
                    device=device,
                    draw_index=draw_index,
                )
            else:
                raise ValueError(f"{method} 缺少模型")
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
            batch_seconds.extend([elapsed / (end - start)] * (end - start))
            output_np = output.detach().float().cpu().numpy()
            if output_np.shape != (end - start, 1, 112, 112):
                raise ValueError(f"{method} 输出尺寸错误：{output_np.shape}")
            for local, index in enumerate(range(start, end)):
                sample_id = records[index]["sample_id"]
                prediction = np.ascontiguousarray(output_np[local, 0], dtype=np.float32)
                _save_prediction(output_dir / f"{sample_id}.npy", prediction)
                predictions[sample_id] = prediction

    manifest_path = write_prediction_manifest(
        output_dir,
        expected_ids,
        method=method,
        draw_index=draw_index if method.startswith("flow_bridge") else None,
    )
    rows = evaluate_saved_predictions(
        records,
        output_dir,
        manifest_path,
        evaluation_config,
        expected_ids,
    )
    runtime = {
        "sample_count": len(rows),
        "inference_seconds_total": float(sum(batch_seconds)),
        "inference_seconds_median_per_sample": float(np.median(batch_seconds)),
        "cuda_peak_allocated_mib": float(torch.cuda.max_memory_allocated(device) / 2**20)
        if device.type == "cuda"
        else 0.0,
        "cuda_peak_reserved_mib": float(torch.cuda.max_memory_reserved(device) / 2**20)
        if device.type == "cuda"
        else 0.0,
        "sample_ids_sha256": hashlib.sha256("\n".join(expected_ids).encode("utf-8")).hexdigest(),
    }
    return rows, runtime, predictions


def _group_rows(method: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["source_group_id"])].append(row)
    summary = []
    for group_id in sorted(grouped):
        group_rows = grouped[group_id]
        summary_row: dict[str, Any] = {
            "method": method,
            "source_group_id": group_id,
            "sample_count": len(group_rows),
        }
        for metric in METRIC_COLUMNS:
            values = [
                float(row[metric])
                for row in group_rows
                if row.get(metric) not in (None, "NA")
            ]
            summary_row[metric] = float(np.mean(values)) if values else "NA"
            summary_row[f"{metric}_valid_count"] = len(values)
        summary.append(summary_row)
    return summary


def summarize_comparisons(
    methods: dict[str, list[dict[str, Any]]],
    *,
    reference_method: str,
) -> list[dict[str, Any]]:
    if reference_method not in methods:
        raise ValueError(f"缺少比较参考方法：{reference_method}")
    rows = []
    reference_rows = methods[reference_method]
    for method, candidate_rows in methods.items():
        if method == reference_method:
            continue
        for metric in METRIC_COLUMNS:
            effect = paired_group_bootstrap(
                candidate_rows,
                reference_rows,
                metric,
                seed=BOOTSTRAP_SEED,
                draws=BOOTSTRAP_DRAWS,
            )
            rows.append(
                {
                    "candidate_method": method,
                    "reference_method": reference_method,
                    **effect,
                }
            )
    return rows


def _write_sample_set_statistics(
    path: Path,
    sample_ids: list[str],
    predictions_by_draw: list[dict[str, np.ndarray]],
) -> list[dict[str, Any]]:
    if len(predictions_by_draw) != FLOW_DRAWS:
        raise ValueError("Flow sample-set 统计必须包含四个 draw")
    rows = []
    for sample_id in sample_ids:
        stack = np.stack([draw[sample_id] for draw in predictions_by_draw]).astype(np.float64)
        rows.append(
            {
                "sample_id": sample_id,
                "draw_count": len(predictions_by_draw),
                "pixel_variance_mean": float(stack.var(axis=0, ddof=1).mean()),
                "pixel_variance_p95": float(np.percentile(stack.var(axis=0, ddof=1), 95, method="linear")),
                "mean_pairwise_absolute_difference": float(
                    np.mean(
                        [
                            np.mean(np.abs(stack[left] - stack[right]))
                            for left in range(FLOW_DRAWS)
                            for right in range(left + 1, FLOW_DRAWS)
                        ]
                    )
                ),
            }
        )
    write_csv(path, rows)
    return rows


def _run_inference_and_evaluation(
    run_dir: Path,
    records: list[dict[str, Any]],
    models: dict[str, nn.Module],
    evaluation_config: dict[str, Any],
    *,
    device: torch.device,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    prediction_root = run_dir / "predictions"
    metric_root = run_dir / "metrics"
    metric_root.mkdir(parents=True, exist_ok=False)
    prediction_root.mkdir(parents=True, exist_ok=False)
    all_rows: dict[str, list[dict[str, Any]]] = {}
    runtime: dict[str, Any] = {"inference": {}}

    for method in ("bicubic", "edsr", "swinir"):
        model = None if method == "bicubic" else models[method]
        rows, timing, _ = infer_method(
            method,
            model,
            records,
            output_dir=prediction_root / method,
            evaluation_config=evaluation_config,
            device=device,
        )
        for row in rows:
            row["method"] = method
            row["draw_index"] = ""
        all_rows[method] = rows
        runtime["inference"][method] = timing
        write_csv(metric_root / f"{method}_per_image.csv", rows)

    draw_rows = []
    predictions_by_draw = []
    for draw_index in range(FLOW_DRAWS):
        rows, timing, predictions = infer_method(
            "flow_bridge",
            models["flow_bridge"],
            records,
            output_dir=prediction_root / "flow_bridge" / f"draw_{draw_index:02d}",
            evaluation_config=evaluation_config,
            device=device,
            draw_index=draw_index,
        )
        for row in rows:
            row["method"] = "flow_bridge_single_draw"
            row["draw_index"] = draw_index
        draw_rows.append(rows)
        predictions_by_draw.append(predictions)
        runtime["inference"][f"flow_bridge_draw_{draw_index:02d}"] = timing
        write_csv(metric_root / f"flow_bridge_draw_{draw_index:02d}_per_image.csv", rows)

    all_rows["flow_bridge_single_draw"] = draw_rows[0]
    _write_sample_set_statistics(
        metric_root / "flow_bridge_sample_set.csv",
        [record["sample_id"] for record in records],
        predictions_by_draw,
    )

    mean_dir = prediction_root / "flow_bridge" / "pixel_mean"
    mean_dir.mkdir(parents=True, exist_ok=False)
    for sample_id in [record["sample_id"] for record in records]:
        mean_prediction = np.mean(
            [draw[sample_id] for draw in predictions_by_draw], axis=0, dtype=np.float64
        ).astype(np.float32)
        _save_prediction(mean_dir / f"{sample_id}.npy", mean_prediction)
    write_prediction_manifest(
        mean_dir,
        [record["sample_id"] for record in records],
        method="flow_bridge_pixel_mean",
    )
    mean_rows = evaluate_saved_predictions(
        records,
        mean_dir,
        mean_dir / "inference_manifest.json",
        evaluation_config,
        [record["sample_id"] for record in records],
    )
    for row in mean_rows:
        row["method"] = "flow_bridge_pixel_mean"
        row["draw_index"] = ""
    all_rows["flow_bridge_pixel_mean"] = mean_rows
    write_csv(metric_root / "flow_bridge_pixel_mean_per_image.csv", mean_rows)

    group_rows = []
    for method, rows in all_rows.items():
        group_rows.extend(_group_rows(method, rows))
    write_csv(metric_root / "source_group_metrics.csv", group_rows)

    deterministic = ("bicubic", "edsr", "swinir")
    reference_method = max(deterministic, key=lambda method: group_balanced_mean(all_rows[method], "psnr_db"))
    methods_for_comparison = {
        method: all_rows[method]
        for method in ("bicubic", "edsr", "swinir", "flow_bridge_single_draw", "flow_bridge_pixel_mean")
    }
    comparison_rows = summarize_comparisons(methods_for_comparison, reference_method=reference_method)
    write_csv(metric_root / "paired_group_comparisons.csv", comparison_rows)
    return all_rows, {
        "inference": runtime["inference"],
        "reference_deterministic_method": reference_method,
        "group_metrics": group_rows,
        "comparisons": comparison_rows,
        "sample_set_rows": len(records),
    }


def replace_json(path: Path, value: Any) -> None:
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _disk_snapshot(path: Path) -> dict[str, Any]:
    usage = shutil.disk_usage(path)
    return {
        "path": str(path),
        "total_bytes": int(usage.total),
        "used_bytes": int(usage.used),
        "free_bytes": int(usage.free),
        "free_gib": float(usage.free / 2**30),
    }


def _nvidia_smi() -> str | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total,memory.used,memory.free,utilization.gpu",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=20,
        )
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _git_snapshot() -> dict[str, Any]:
    def run_git(arguments: list[str]) -> str:
        result = subprocess.run(
            ["git", *arguments], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True
        )
        return result.stdout

    status = run_git(["status", "--short", "--untracked-files=all"])
    return {
        "head": run_git(["rev-parse", "HEAD"]).strip(),
        "branch": run_git(["branch", "--show-current"]).strip(),
        "dirty": bool(status.strip()),
        "status_line_count": len(status.splitlines()),
        "status_short": status,
    }


def _environment_snapshot(device: torch.device) -> dict[str, Any]:
    try:
        import torchvision

        torchvision_version = torchvision.__version__
    except Exception as exc:
        torchvision_version = f"不可用：{type(exc).__name__}: {exc}"
    import scipy

    environment = {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "torch": torch.__version__,
        "torchvision": torchvision_version,
        "opencv": cv2.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "device": str(device),
        "nvidia_smi": _nvidia_smi(),
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        environment["gpu_name"] = properties.name
        environment["gpu_total_memory_bytes"] = int(total_bytes)
        environment["gpu_free_memory_at_start_bytes"] = int(free_bytes)
        environment["gpu_multiprocessors"] = int(properties.multi_processor_count)
    return environment


def _guardrail_review(
    flow_rows: list[dict[str, Any]],
    reference_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    guards = {}
    for metric in (
        "bright_mean_abs_error",
        "bright_abs_error_p95",
        "edge_strength_abs_error",
        "background_abs_bias",
    ):
        effect = paired_group_bootstrap(flow_rows, reference_rows, metric)
        worsened_groups = [
            group for group, delta in effect["group_deltas"].items() if delta > 0
        ]
        guards[metric] = {
            **effect,
            "worsened_group_count": len(worsened_groups),
            "worsened_groups": worsened_groups,
            "passed": effect["group_count"] == 8 and len(worsened_groups) <= 4,
            "rule": "8 个空间组至少有 4 个不恶化；5 个或更多组恶化视为一致恶化",
        }
    return guards


def _research_decision(
    rows_by_method: dict[str, list[dict[str, Any]]],
    inference_runtime: dict[str, Any],
    reference_method: str,
) -> dict[str, Any]:
    reference_rows = rows_by_method[reference_method]
    flow_rows = rows_by_method["flow_bridge_single_draw"]
    psnr = paired_group_bootstrap(flow_rows, reference_rows, "psnr_db")
    ssim_effect = paired_group_bootstrap(flow_rows, reference_rows, "ssim")
    epi = paired_group_bootstrap(flow_rows, reference_rows, "epi_corr")
    guardrails = _guardrail_review(flow_rows, reference_rows)
    conditions = {
        "delta_psnr_at_least_0_05_db": psnr["group_balanced_delta"] >= 0.05,
        "psnr_bootstrap_95ci_lower_above_zero": psnr["ci95_lower"] > 0,
        "delta_ssim_nonnegative": ssim_effect["group_balanced_delta"] >= 0,
        "delta_epi_nonnegative": epi["group_balanced_delta"] >= 0,
        "brightness_structure_background_guardrails": all(item["passed"] for item in guardrails.values()),
    }
    continued = all(conditions.values())
    flow_seconds = float(inference_runtime["flow_bridge_draw_00"]["inference_seconds_total"])
    reference_seconds = float(inference_runtime[reference_method]["inference_seconds_total"])
    return {
        "decision": "CONTINUE_TO_SEPARATE_PROTOCOL_REVIEW" if continued else "STOP_CURRENT_FLOW_BRIDGE_CONFIGURATION",
        "decision_scope": "仅决定该 SAR-only 单城区 pilot 是否值得另行设计复核；不改变 C1/C2/C3 阶段 gate，不授权 C3-B",
        "reference_method": reference_method,
        "flow_single_draw": {"psnr": psnr, "ssim": ssim_effect, "epi": epi},
        "guardrails": guardrails,
        "conditions": conditions,
        "inference_cost": {
            "flow_single_draw_seconds": flow_seconds,
            "reference_seconds": reference_seconds,
            "flow_to_reference_ratio": flow_seconds / reference_seconds if reference_seconds > 0 else None,
            "flow_four_draw_seconds": sum(
                float(inference_runtime[f"flow_bridge_draw_{index:02d}"]["inference_seconds_total"])
                for index in range(FLOW_DRAWS)
            ),
        },
        "limitations": [
            "全部结果来自 SARptical 柏林单城区 8 个 development 网格组。",
            "development 仅支持探索性决策，不代表独立城市或传感器泛化。",
            "SARptical dB noisy target 不是 clean SAR ground truth；本试验不证明真实去斑。",
            "Optical payload 未读取，不能支持 Optical 增量或配准效用主张。",
            "×2 bicubic 退化是表示域受控重建，不是 SAR 传感器前向模型。",
            "Flow 四次样本均值单独报告，不能替代单次生成质量或样本分布真实性。",
        ],
    }


def write_decision_review(
    path: Path,
    decision: dict[str, Any],
    summary: dict[str, Any],
    rows_by_method: dict[str, list[dict[str, Any]]],
) -> None:
    ref_method = decision["reference_method"]
    flow_rows = rows_by_method["flow_bridge_single_draw"]
    table_rows = []
    for method in ("bicubic", "edsr", "swinir", "flow_bridge_single_draw", "flow_bridge_pixel_mean"):
        method_rows = rows_by_method[method]
        table_rows.append(
            "| {method} | {psnr:.4f} | {ssim:.5f} | {epi} | {rmse:.5f} |".format(
                method=method,
                psnr=group_balanced_mean(method_rows, "psnr_db"),
                ssim=group_balanced_mean(method_rows, "ssim"),
                epi=(
                    f"{group_balanced_mean(method_rows, 'epi_corr'):.5f}"
                    if any(row.get("epi_corr") != "NA" for row in method_rows)
                    else "NA"
                ),
                rmse=group_balanced_mean(method_rows, "rmse"),
            )
        )
    status = "继续至单独的方案复核" if decision["decision"].startswith("CONTINUE") else "停止当前 Flow Bridge 配置"
    guard_lines = []
    for name, value in decision["guardrails"].items():
        if value["group_count"] < 8:
            guard_lines.append(
                f"- `{name}`：不可用，有效来源组 {value['group_count']}/8；"
                "未达到预设 8 组有效性要求，因此该护栏未通过。"
            )
        else:
            guard_lines.append(
                f"- `{name}`：来源组等权差 {value['group_balanced_delta']:+.6f}；"
                f"95% 区间 [{value['ci95_lower']:+.6f}, {value['ci95_upper']:+.6f}]；"
                f"恶化组 {value['worsened_group_count']}/8；{'通过' if value['passed'] else '未通过'}。"
            )
    content = "\n".join(
        [
            "# SARptical 空间 pilot 研究决策复核",
            "",
            f"**日期：** {dt.datetime.now().astimezone().isoformat(timespec='seconds')}",
            f"**决策：** {status}",
            "",
            "## 实验范围",
            "",
            "本复核只覆盖 SARptical 柏林单城区、SAR-only、按 250 m UTM 网格隔离的 train/development/confirmation 候选划分。仅 train 和 development 影像载荷进入实验；confirmation 只保留坐标、网格和样本名元数据。Optical payload 与项目历史 confirmation/test 均未读取。",
            "",
            "## 开发集结果",
            "",
            "指标按 8 个 development 空间组等权汇总；PSNR/SSIM/EPI/RMSE 对 noisy SAR pseudo-target 只表示本次表示域重建一致性。Flow 单次 draw 与四次 pixel-mean 分开列出。",
            "",
            "| 方法 | PSNR (dB) | SSIM | EPI | RMSE |",
            "|---|---:|---:|---:|---:|",
            *table_rows,
            "",
            f"Flow 单次 draw 相对最强确定性对照 `{ref_method}`：ΔPSNR={decision['flow_single_draw']['psnr']['group_balanced_delta']:+.4f} dB，95% bootstrap 区间 [{decision['flow_single_draw']['psnr']['ci95_lower']:+.4f}, {decision['flow_single_draw']['psnr']['ci95_upper']:+.4f}]；ΔSSIM={decision['flow_single_draw']['ssim']['group_balanced_delta']:+.5f}；ΔEPI={decision['flow_single_draw']['epi']['group_balanced_delta']:+.5f}。",
            "",
            "## 结构、亮回波和背景护栏",
            "",
            *guard_lines,
            "",
            "## 计算成本",
            "",
            f"Flow 单次 draw 的 development 总推理时间为 {decision['inference_cost']['flow_single_draw_seconds']:.2f} 秒；对照 `{ref_method}` 为 {decision['inference_cost']['reference_seconds']:.2f} 秒；四 draw 共 {decision['inference_cost']['flow_four_draw_seconds']:.2f} 秒。逐方法训练耗时、峰值显存、checkpoint 和运行环境见 `runtime.json`、`methods/*/train_log.csv` 与 checkpoint 元数据。",
            "",
            "## 预先冻结的门槛结果",
            "",
            *[f"- `{key}`：{'通过' if value else '未通过'}。" for key, value in decision["conditions"].items()],
            "",
            "## 证据边界和长期方向",
            "",
            *[f"- {item}" for item in decision["limitations"]],
            "- 本 pilot 不评价 Optical 信息的增量价值，不进入 C3-B，也不修改论文、论文表格或既有研究结论。",
            "- 若决策为继续，只允许据此制定新的独立方案并提交审阅；本结果本身不构成阶段晋级或论文泛化证据。",
            "",
            "## 产物索引",
            "",
            f"- 运行目录：`{summary['run_directory']}`",
            "- 逐样本指标：`metrics/*_per_image.csv`。",
            "- 来源组指标和配对 bootstrap：`metrics/source_group_metrics.csv`、`metrics/paired_group_comparisons.csv`。",
            "- 生成样本：`predictions/`；Flow 四次 draw 和 pixel-mean 分目录保存。",
            "- 配置、数据、环境、代码状态和资源：`config.json`、`manifest.json`、`environment.txt`、`git_state.txt`、`runtime.json`。",
        ]
    )
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(content)
        stream.write("\n")


def _write_environment_text(path: Path, environment: dict[str, Any]) -> None:
    lines = [
        "SARptical 空间 pilot 实验环境",
        f"开始时间：{dt.datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"Python：{environment['python']}",
        f"Python 可执行文件：{environment['python_executable']}",
        f"操作系统：{environment['platform']}",
        f"NumPy：{environment['numpy']}",
        f"SciPy：{environment['scipy']}",
        f"PyTorch：{environment['torch']}",
        f"torchvision：{environment['torchvision']}",
        f"OpenCV：{environment['opencv']}",
        f"CUDA 可用：{environment['cuda_available']}",
        f"PyTorch CUDA：{environment['cuda_version']}",
        f"cuDNN：{environment['cudnn_version']}",
        f"设备：{environment['device']}",
        f"GPU 信息：{environment.get('nvidia_smi')}",
        f"GPU 总显存：{environment.get('gpu_total_memory_bytes')}",
        f"启动时 GPU 空闲显存字节数（仅记录、不作门槛）：{environment.get('gpu_free_memory_at_start_bytes')}",
    ]
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")


def run_experiment(archive_path: Path, output_dir: Path) -> dict[str, Any]:
    archive_path = Path(archive_path).resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    expected_archive_hash = "044cfa600e301f3b1f3d72d38f2182c4e74453d9403a0aab66076d6308c567f9"
    if not archive_path.is_file():
        raise FileNotFoundError(f"SARptical 归档不存在：{archive_path}")
    if output_dir.exists():
        raise FileExistsError(f"拒绝覆盖已有运行目录：{output_dir}")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_disk_before = _disk_snapshot(output_dir.parent)
    root_disk_before = _disk_snapshot(PROJECT_ROOT)
    tmp_disk_before = _disk_snapshot(Path("/tmp"))
    estimated_peak_bytes = 3 * 2**30
    reserve_bytes = 5 * 2**30
    if output_disk_before["free_bytes"] < estimated_peak_bytes + reserve_bytes:
        raise OSError("运行磁盘未达到预计产物空间加 5 GiB 安全余量")
    if not torch.cuda.is_available():
        raise RuntimeError("当前进程不可见 CUDA；请在已授权的 GPU 提权环境运行本脚本")

    output_dir.mkdir(parents=False, exist_ok=False)
    started_at = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    started_clock = time.perf_counter()
    device = torch.device("cuda:0")
    torch.set_num_threads(4)
    environment = _environment_snapshot(device)
    config = {
        "run_id": "phase_d_sarptical_spatial_pilot_20261126_retry01",
        "研究问题": "双三次锚定条件 Flow Bridge 是否能在有限空间分组 SAR-only 表示域重建中改善最强确定性对照",
        "数据归档": str(archive_path),
        "expected_archive_sha256": expected_archive_hash,
        "output_directory": str(output_dir),
        "split_seed": pilot_data.SPLIT_SEED,
        "grid_size_meters": pilot_data.GRID_SIZE_METERS,
        "buffer_meters": pilot_data.BUFFER_METERS,
        "role_group_counts": pilot_data.EXPECTED_ROLE_GROUP_COUNTS,
        "role_sample_counts": pilot_data.EXPECTED_ROLE_SAMPLE_COUNTS,
        "methods": list(METHODS),
        "method_identity": {
            "bicubic": "FIXED_INTERPOLATION_BASELINE",
            "edsr": "TASK_ADAPTED_PUBLISHED",
            "swinir": "TASK_ADAPTED_PUBLISHED",
            "flow_bridge": "PROPOSED_METHOD_CANDIDATE",
        },
        "train_seed": SEED,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "patch_lr": PATCH_LR,
        "patch_hr": PATCH_LR * 2,
        "optimizer": "Adam",
        "learning_rate": LEARNING_RATE,
        "weight_decay": 0.0,
        "flow_sigma": FLOW_SIGMA,
        "flow_heun_steps": FLOW_STEPS,
        "flow_final_draws": FLOW_DRAWS,
        "checkpoint_selection": "每 epoch 在完整 development 上按来源组等权 PSNR 选择；Flow 使用固定 draw 0 噪声",
        "evaluation_bootstrap_seed": BOOTSTRAP_SEED,
        "evaluation_bootstrap_draws": BOOTSTRAP_DRAWS,
        "test_set_used": False,
        "confirmation_payload_accessed": False,
        "optical_payload_loaded": False,
        "论文文件修改": False,
    }

    try:
        write_json(output_dir / "config.json", config)
        command = shlex.join([sys.executable, *sys.argv])
        with (output_dir / "command.txt").open("x", encoding="utf-8") as stream:
            stream.write("执行命令：\n")
            stream.write(command + "\n")
        shutil.copyfile(
            PROJECT_ROOT / "docs/research/PHASE_D_SARPTICAL_SPATIAL_PILOT_DESIGN_20261125.md",
            output_dir / "protocol.md",
        )
        write_json(output_dir / "environment.json", environment)
        _write_environment_text(output_dir / "environment.txt", environment)
        git_snapshot = _git_snapshot()
        with (output_dir / "git_state.txt").open("x", encoding="utf-8") as stream:
            stream.write(f"提交：{git_snapshot['head']}\n分支：{git_snapshot['branch']}\n")
            stream.write(f"工作区是否含改动：{git_snapshot['dirty']}\n状态行数：{git_snapshot['status_line_count']}\n")
            stream.write("完整 git status --short --untracked-files=all：\n")
            stream.write(git_snapshot["status_short"])

        print("开始读取 SARptical 坐标与 SAR 文件名清单；不打开 confirmation 或 Optical 载荷。", flush=True)
        split = pilot_data.build_spatial_split(archive_path, seed=pilot_data.SPLIT_SEED)
        if split["source_archive_sha256"] != expected_archive_hash:
            raise ValueError("SARptical 归档 SHA-256 与冻结协议不一致")
        write_json(output_dir / "split_manifest.json", split)

        cache_started = time.perf_counter()
        input_manifest = pilot_data.prepare_fixed_inputs(
            archive_path,
            split,
            output_dir / "data_cache",
        )
        write_json(output_dir / "input_manifest.json", input_manifest)
        train_records = pilot_data.load_cached_records(output_dir / "data_cache", split, "train")
        development_records = pilot_data.load_cached_records(output_dir / "data_cache", split, "development")
        if len(train_records) != 916 or len(development_records) != 306:
            raise ValueError("固定输入载入后的 train/development 样本数量不符")
        evaluation_config = fixed_evaluation_config(train_records)
        write_json(output_dir / "evaluation_config.json", evaluation_config)
        cache_seconds = float(time.perf_counter() - cache_started)

        schedule = [
            make_epoch_schedule(train_records, seed=SEED + epoch, patch_lr=PATCH_LR)
            for epoch in range(EPOCHS)
        ]
        with (output_dir / "training_schedule.json").open("x", encoding="utf-8") as stream:
            json.dump(schedule, stream, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
            stream.write("\n")
        schedule_hash = schedule_digest(schedule)
        config["training_schedule_sha256"] = schedule_hash
        config["evaluation_config_sha256"] = sha256_file(output_dir / "evaluation_config.json")
        replace_json(output_dir / "config.json", config)

        code_paths = [
            Path(__file__),
            Path(__file__).with_name("pilot_data.py"),
            PROJECT_ROOT / "MMCANetV2/phase_c3a_flow.py",
            PROJECT_ROOT / "MMCANetV2/phase_c3a_metrics.py",
            PROJECT_ROOT / "experiments/phase_d1_edsr_20261124_v1/edsr_sar.py",
            PROJECT_ROOT / "experiments/published_baselines_sar_only_20261123_v1/train_swinir_sar.py",
        ]
        source_hashes = {str(path.relative_to(PROJECT_ROOT)): sha256_file(path) for path in code_paths}
        write_json(output_dir / "source_manifest.json", source_hashes)

        runtime: dict[str, Any] = {
            "status": "running",
            "started_at": started_at,
            "device": str(device),
            "environment": environment,
            "storage_before": {
                "output_volume": output_disk_before,
                "project_root_volume": root_disk_before,
                "tmp_volume": tmp_disk_before,
            },
            "estimated_peak_output_bytes": estimated_peak_bytes,
            "reserved_bytes": reserve_bytes,
            "data_cache_seconds": cache_seconds,
            "split_group_count": split["groups_total"],
            "train_sample_count": len(train_records),
            "development_sample_count": len(development_records),
            "schedule_sha256": schedule_hash,
            "methods": {},
            "confirmation_payload_accessed": False,
            "test_set_used": False,
            "optical_payload_loaded": False,
        }
        replace_json(output_dir / "runtime.json", runtime)

        print(
            f"输入准备完成：train {len(train_records)} 组图，development {len(development_records)} 组图；"
            f"归档和 split 哈希已记录。输出盘剩余 {output_disk_before['free_gib']:.1f} GiB。",
            flush=True,
        )
        models: dict[str, nn.Module] = {}
        training_summaries = []
        training_config = {
            "seed": SEED,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "patch_lr": PATCH_LR,
            "patch_hr": PATCH_LR * 2,
            "learning_rate": LEARNING_RATE,
            "flow_sigma": FLOW_SIGMA,
            "flow_steps": FLOW_STEPS,
            "optimizer": "Adam",
            "split_manifest_sha256": sha256_file(output_dir / "split_manifest.json"),
            "input_manifest_sha256": sha256_file(output_dir / "input_manifest.json"),
            "evaluation_config_sha256": sha256_file(output_dir / "evaluation_config.json"),
            "training_schedule_sha256": schedule_hash,
            "data_range": evaluation_config["data_range"],
        }
        for method in TRAINED_METHODS:
            print(f"开始训练 {method}；不因其他 CUDA 进程或当前空闲显存量拦截。", flush=True)
            trained = train_method(
                method,
                train_records,
                development_records,
                training_config,
                output_dir,
                device=device,
                schedule=schedule,
                data_range=float(evaluation_config["data_range"]),
            )
            models[method] = trained.pop("model")
            training_summaries.append(
                {
                    "method": method,
                    "best_epoch": trained["best_epoch"],
                    "best_development_group_balanced_psnr_db": trained["best_score"],
                    "optimizer_steps": trained["optimizer_steps"],
                    "parameter_count": trained["parameter_count"],
                    "training_seconds": float(sum(row["epoch_seconds"] for row in trained["train_log"])),
                    "cuda_peak_allocated_mib": float(
                        max(row["cuda_peak_allocated_mib"] for row in trained["train_log"])
                    ),
                    "cuda_peak_reserved_mib": float(
                        max(row["cuda_peak_reserved_mib"] for row in trained["train_log"])
                    ),
                }
            )
            runtime["methods"][method] = training_summaries[-1]
            runtime["storage_after_" + method] = _disk_snapshot(output_dir)
            replace_json(output_dir / "runtime.json", runtime)

        with torch.inference_mode():
            rows_by_method, inference_summary = _run_inference_and_evaluation(
                output_dir,
                development_records,
                models,
                evaluation_config,
                device=device,
            )
        all_metric_rows = []
        for method_rows in rows_by_method.values():
            all_metric_rows.extend(method_rows)
        write_csv(output_dir / "metrics" / "per_image_metrics.csv", all_metric_rows)
        reference_method = inference_summary["reference_deterministic_method"]
        decision = _research_decision(rows_by_method, inference_summary["inference"], reference_method)
        replace_json(output_dir / "decision.json", decision)
        write_decision_review(
            output_dir / "RESEARCH_DECISION_REVIEW.md",
            decision,
            {"run_directory": str(output_dir)},
            rows_by_method,
        )

        ended_at = dt.datetime.now().astimezone().isoformat(timespec="seconds")
        runtime.update(
            {
                "status": "complete",
                "ended_at": ended_at,
                "elapsed_seconds": float(time.perf_counter() - started_clock),
                "training": training_summaries,
                "inference": inference_summary["inference"],
                "best_deterministic_method": reference_method,
                "decision": decision["decision"],
                "storage_after": _disk_snapshot(output_dir),
                "storage_after_output_parent": _disk_snapshot(output_dir.parent),
                "confirmation_payload_accessed": False,
                "test_set_used": False,
                "optical_payload_loaded": False,
            }
        )
        replace_json(output_dir / "runtime.json", runtime)

        checkpoint_hashes = {}
        for method in TRAINED_METHODS:
            checkpoint_hashes[method] = {
                name: sha256_file(output_dir / "methods" / method / "checkpoints" / f"{name}.pt")
                for name in ("best", "last")
            }
        config_hash = sha256_file(output_dir / "config.json")
        manifest = {
            "run_id": config["run_id"],
            "status": "complete",
            "method": "bicubic + EDSR + SwinIR + bicubic-anchored Flow Bridge",
            "method_identities": config["method_identity"],
            "started_at": started_at,
            "ended_at": ended_at,
            "git_head": git_snapshot["head"],
            "git_branch": git_snapshot["branch"],
            "git_dirty": git_snapshot["dirty"],
            "git_status_line_count": git_snapshot["status_line_count"],
            "code_sha256": source_hashes,
            "config_sha256": config_hash,
            "split_manifest_sha256": sha256_file(output_dir / "split_manifest.json"),
            "input_manifest_sha256": sha256_file(output_dir / "input_manifest.json"),
            "evaluation_config_sha256": sha256_file(output_dir / "evaluation_config.json"),
            "training_schedule_sha256": schedule_hash,
            "checkpoint_sha256": checkpoint_hashes,
            "output_prediction_count": len(list((output_dir / "predictions").rglob("*.npy"))),
            "train_sample_count": len(train_records),
            "development_sample_count": len(development_records),
            "development_group_count": split["role_group_counts"]["development"],
            "best_deterministic_method": reference_method,
            "research_decision": decision["decision"],
            "environment": environment,
            "test_set_used": False,
            "confirmation_payload_accessed": False,
            "confirmation_metadata_recorded_only": True,
            "optical_payload_loaded": False,
            "paper_files_modified": False,
        }
        write_json(output_dir / "manifest.json", manifest)
        print(
            f"实验完成：最强确定性对照={reference_method}；Flow 决策={decision['decision']}。"
            f"全部指标、预测、checkpoint、资源与决策记录在 {output_dir}。",
            flush=True,
        )
        return {"manifest": manifest, "decision": decision, "runtime": runtime}
    except BaseException as exc:
        failure = {
            "status": "failed",
            "failed_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "traceback": traceback.format_exc(),
            "confirmation_payload_accessed": False,
            "test_set_used": False,
            "optical_payload_loaded": False,
            "storage_at_failure": _disk_snapshot(output_dir),
        }
        try:
            replace_json(output_dir / "failure.json", failure)
            with (output_dir / "failure.txt").open("x", encoding="utf-8") as stream:
                stream.write("本轮 SARptical pilot 失败证据\n")
                stream.write(f"异常类型：{type(exc).__name__}\n异常内容：{exc}\n\n")
                stream.write(failure["traceback"])
        except Exception:
            pass
        print(f"运行失败：{type(exc).__name__}：{exc}；已保留目录内可写的诊断证据。", flush=True)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SARptical 有限 SAR-only 空间 pilot")
    parser.add_argument(
        "--archive",
        type=Path,
        default=Path("/run/media/shy/C61C9A341C9A200B/Codex_JDSR_runs/SARptical_data_reacquire_direct_20261126.zip"),
        help="SARptical 官方数据归档路径",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "/run/media/shy/C61C9A341C9A200B/Codex_JDSR_runs/"
            "phase_d_sarptical_spatial_pilot_20261126_retry01"
        ),
        help="本轮唯一输出目录；目录已存在时拒绝覆盖",
    )
    args = parser.parse_args(argv)
    try:
        run_experiment(args.archive, args.output_dir)
    except Exception as exc:
        print(f"任务未完成：{type(exc).__name__}：{exc}", flush=True)
        return 1
    return 0


def _gradient_setup_log(model: nn.Module, config: dict[str, Any]) -> dict[str, Any]:
    return {
        "身份与结构": getattr(model, "experiment_config", {}),
        "训练配置": config,
        "参数": parameter_count(model),
    }


def _write_checkpoint(
    path: Path,
    *,
    method: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_epoch: int,
    best_score: float,
    config: dict[str, Any],
    flow_generator: torch.Generator,
) -> None:
    payload = {
        "format": "sarptical_spatial_pilot_checkpoint_v1",
        "method": method,
        "epoch": int(epoch),
        "best_epoch": int(best_epoch),
        "best_development_group_balanced_psnr_db": float(best_score),
        "config": config,
        "model_config": getattr(model, "experiment_config", {}),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "flow_generator_state": flow_generator.get_state(),
    }
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def train_method(
    method: str,
    train_records: list[dict[str, Any]],
    development_records: list[dict[str, Any]],
    config: dict[str, Any],
    run_dir: Path,
    *,
    device: torch.device,
    schedule: list[list[dict[str, Any]]],
    data_range: float,
) -> dict[str, Any]:
    if method not in TRAINED_METHODS:
        raise ValueError(f"{method} 不是可训练方法")
    seed_everything(int(config["seed"]))
    model = build_model(method).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]), weight_decay=0.0)
    flow_generator = torch.Generator(device=device).manual_seed(int(config["seed"]) + 71)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    method_dir = run_dir / "methods" / method
    checkpoint_dir = method_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    method_dir.mkdir(parents=True, exist_ok=True)
    write_json(method_dir / "model_config.json", _gradient_setup_log(model, config))
    train_rows = []
    best_score = -math.inf
    best_epoch = 0
    optimizer_steps = 0
    train_log_path = method_dir / "train_log.csv"
    validation_log_path = method_dir / "validation_metrics.csv"

    for epoch_index, epoch_schedule in enumerate(schedule):
        epoch_started = time.perf_counter()
        model.train()
        losses, gradient_norms = [], []
        for start in range(0, len(epoch_schedule), BATCH_SIZE):
            batch_rows = epoch_schedule[start : start + BATCH_SIZE]
            lr, target, anchor = _batch_from_schedule(
                train_records, batch_rows, patch_lr=int(config["patch_lr"]), device=device
            )
            result = train_batch(
                method,
                model,
                optimizer,
                lr,
                target,
                anchor,
                flow_generator=flow_generator,
                sigma=float(config["flow_sigma"]),
            )
            losses.append(result["loss"])
            gradient_norms.append(result["gradient_norm"])
            optimizer_steps += 1
        validation_rows = _selection_rows(
            model,
            method,
            development_records,
            device=device,
            data_range=data_range,
        )
        score = group_balanced_mean(validation_rows, "psnr_db")
        epoch = epoch_index + 1
        if score > best_score:
            best_score, best_epoch = score, epoch
            _write_checkpoint(
                checkpoint_dir / "best.pt",
                method=method,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_epoch=best_epoch,
                best_score=best_score,
                config=config,
                flow_generator=flow_generator,
            )
        _write_checkpoint(
            checkpoint_dir / "last.pt",
            method=method,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            best_epoch=best_epoch,
            best_score=best_score,
            config=config,
            flow_generator=flow_generator,
        )
        row = {
            "epoch": epoch,
            "training_loss_mean": float(np.mean(losses)),
            "gradient_norm_mean": float(np.mean(gradient_norms)),
            "optimizer_steps_total": optimizer_steps,
            "development_group_balanced_psnr_db": score,
            "epoch_seconds": float(time.perf_counter() - epoch_started),
            "cuda_peak_allocated_mib": float(torch.cuda.max_memory_allocated(device) / 2**20)
            if device.type == "cuda"
            else 0.0,
            "cuda_peak_reserved_mib": float(torch.cuda.max_memory_reserved(device) / 2**20)
            if device.type == "cuda"
            else 0.0,
        }
        train_rows.append(row)
        append_csv(train_log_path, [row])
        append_csv(validation_log_path, [{"epoch": epoch, **item} for item in validation_rows])
        print(
            f"[{method}] 第 {epoch:02d}/{len(schedule):02d} 轮："
            f"训练损失={row['training_loss_mean']:.6f}，开发组等权 PSNR={score:.4f} dB，"
            f"耗时={row['epoch_seconds']:.1f} 秒。",
            flush=True,
        )

    best = torch.load(checkpoint_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best["model_state_dict"], strict=True)
    return {
        "method": method,
        "model": model,
        "best_epoch": best_epoch,
        "best_score": best_score,
        "optimizer_steps": optimizer_steps,
        "train_log": train_rows,
        "parameter_count": parameter_count(model),
    }


if __name__ == "__main__":
    raise SystemExit(main())
