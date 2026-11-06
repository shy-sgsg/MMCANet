#!/usr/bin/env python3
"""对已训练的 C3-A 最佳 checkpoint 做 development 多次采样稳定性诊断。"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
METRIC_ROOT = ROOT / "Early-FusionCascadeJDSR"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(METRIC_ROOT) not in sys.path:
    sys.path.insert(0, str(METRIC_ROOT))

from eval_metrics import edge_metrics
from MMCANetV2.phase_c2_screen import build_fixed_masks
from MMCANetV2.phase_c3a_flow import SarConditionalFlow, SarTargetOnlyDataset, sample_heun, upsample_sar_condition
from scripts.run_phase_c3a_sar_flow import (
    BASELINE_PATH,
    INPUT_ROOT,
    SOURCE_MANIFEST,
    _active_cuda_processes,
    _csv_write,
    _environment,
    _evaluate,
    _json_write,
    _load_baseline,
    _metric_rows,
    _utc_now,
    fixed_input_files_sha256,
    fixed_input_manifest_sha256,
    sha256_file,
)

EVALUATION_PROTOCOL = ROOT / "protocols/mmcanet_phase_c3a/multi_sample_stability_c1_v2_20261105_v2.json"
TRAIN_PROTOCOL = ROOT / "protocols/mmcanet_phase_c3a/sar_only_flow_c1_v2_20261105_v1.json"
OUTPUT_PATH = ROOT / "experiments/diagnostics/c3a_multi_sample_eval_c1_v2_20261105_v2"
EXPECTED_EVALUATION_PROTOCOL_SHA256 = "8d886f26023f9d929265b3f70ad9f51314f94788b1e4a01f269af5c1f0a589fc"


def draw_noise_for_identity(
    sample_id: str,
    *,
    draw_index: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    if not sample_id or not isinstance(draw_index, int) or isinstance(draw_index, bool) or draw_index < 0:
        raise ValueError("采样身份和非负采样编号必须有效")
    digest = hashlib.sha256(f"{seed}\0{sample_id}\0{draw_index}".encode("utf-8")).digest()
    local_seed = int.from_bytes(digest[:8], "big") % (2**63 - 1)
    generator = torch.Generator(device=device)
    generator.manual_seed(local_seed)
    return torch.randn((1, 1, 256, 256), generator=generator, device=device)


def aggregate_multi_sample_metrics(
    draw_rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    *,
    expected_draws: int,
) -> list[dict[str, Any]]:
    if expected_draws < 2:
        raise ValueError("稳定性评价至少需要两次采样")
    flow_by_id: dict[str, list[dict[str, Any]]] = {}
    for row in draw_rows:
        sample_id = str(row.get("样本 ID", ""))
        draw_index = row.get("采样编号")
        if not sample_id or not isinstance(draw_index, int) or isinstance(draw_index, bool):
            raise ValueError("flow 逐次采样行缺少身份或整数采样编号")
        flow_by_id.setdefault(sample_id, []).append(row)

    baseline_by_id: dict[str, dict[str, Any]] = {}
    for row in baseline_rows:
        sample_id = str(row.get("样本 ID", ""))
        if not sample_id or sample_id in baseline_by_id:
            raise ValueError("baseline 样本 ID 为空或重复")
        baseline_by_id[sample_id] = row
    if set(flow_by_id) != set(baseline_by_id):
        raise ValueError("flow 与 baseline 的 development 身份集合不一致")

    summaries: list[dict[str, Any]] = []
    expected_indices = set(range(expected_draws))
    for sample_id in sorted(flow_by_id):
        rows = sorted(flow_by_id[sample_id], key=lambda row: row["采样编号"])
        indices = [row["采样编号"] for row in rows]
        if len(indices) != expected_draws:
            raise ValueError(f"样本 {sample_id} 的采样次数不符合协议")
        if len(set(indices)) != len(indices) or set(indices) != expected_indices:
            raise ValueError(f"样本 {sample_id} 存在重复采样编号或编号缺失")
        group_ids = {str(row["来源组 ID"]) for row in rows}
        patch_ids = {str(row["patch ID"]) for row in rows}
        baseline = baseline_by_id[sample_id]
        if (
            len(group_ids) != 1
            or len(patch_ids) != 1
            or str(baseline.get("来源组 ID")) != next(iter(group_ids))
            or str(baseline.get("patch ID")) != next(iter(patch_ids))
        ):
            raise ValueError(f"样本 {sample_id} 的 flow/baseline 身份/来源组/patch 不一致")

        psnr_values = np.asarray([float(row["PSNR"]) for row in rows], dtype=np.float64)
        ssim_values = np.asarray([float(row["SSIM"]) for row in rows], dtype=np.float64)
        epi_values = np.asarray([float(row["EPI"]) for row in rows], dtype=np.float64)
        edge_strength_values = np.asarray([float(row["边缘强度比"]) for row in rows], dtype=np.float64)
        pixel_std_values = np.asarray([float(row["预测像素标准差均值"]) for row in rows], dtype=np.float64)
        baseline_psnr = float(baseline["PSNR"])
        baseline_ssim = float(baseline["SSIM"])
        baseline_epi = float(baseline["EPI"])
        baseline_edge_strength = float(baseline["边缘强度比"])
        values = np.concatenate(
            (
                psnr_values,
                ssim_values,
                epi_values,
                edge_strength_values,
                pixel_std_values,
                [baseline_psnr, baseline_ssim, baseline_epi, baseline_edge_strength],
            )
        )
        if not np.isfinite(values).all():
            raise FloatingPointError(f"样本 {sample_id} 的多次采样指标含非有限值")
        group_id = next(iter(group_ids))
        patch_id = next(iter(patch_ids))
        summaries.append(
            {
                "样本 ID": sample_id,
                "来源组 ID": group_id,
                "patch ID": patch_id,
                "采样次数": expected_draws,
                "Flow 平均 PSNR": float(psnr_values.mean()),
                "Flow PSNR 样本标准差": float(psnr_values.std(ddof=1)),
                "Flow 最低 PSNR": float(psnr_values.min()),
                "Flow 最高 PSNR": float(psnr_values.max()),
                "Flow 平均 SSIM": float(ssim_values.mean()),
                "Flow SSIM 样本标准差": float(ssim_values.std(ddof=1)),
                "Flow 平均 EPI": float(epi_values.mean()),
                "Flow EPI 样本标准差": float(epi_values.std(ddof=1)),
                "Flow 平均边缘强度比": float(edge_strength_values.mean()),
                "Flow 边缘强度比样本标准差": float(edge_strength_values.std(ddof=1)),
                "平均预测像素标准差": float(pixel_std_values.mean()),
                "baseline PSNR": baseline_psnr,
                "平均配对 ΔPSNR": float(psnr_values.mean() - baseline_psnr),
                "baseline SSIM": baseline_ssim,
                "平均配对 ΔSSIM": float(ssim_values.mean() - baseline_ssim),
                "baseline EPI": baseline_epi,
                "平均配对 ΔEPI": float(epi_values.mean() - baseline_epi),
                "baseline 边缘强度比": baseline_edge_strength,
            }
        )
    return summaries


def group_bootstrap_summary(
    rows: list[dict[str, Any]],
    *,
    metric: str,
    seed: int,
    repeats: int,
) -> dict[str, Any]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        group = str(row.get("来源组 ID", ""))
        value = float(row[metric])
        if not group or not math.isfinite(value):
            raise ValueError("来源组统计要求有效身份和有限指标")
        grouped.setdefault(group, []).append(value)
    group_values = np.asarray([np.mean(grouped[key]) for key in sorted(grouped)], dtype=np.float64)
    if group_values.size < 2 or repeats < 1:
        raise ValueError("来源组 bootstrap 至少需要两个组和一次重采样")
    rng = np.random.default_rng(seed)
    selected = rng.integers(0, group_values.size, size=(repeats, group_values.size))
    bootstrap = group_values[selected].mean(axis=1)
    return {
        "指标": metric,
        "来源组等权均值": float(group_values.mean()),
        "来源组等权中位数": float(np.median(group_values)),
        "95% 区间下界": float(np.quantile(bootstrap, 0.025, method="linear")),
        "95% 区间上界": float(np.quantile(bootstrap, 0.975, method="linear")),
        "来源组数": int(group_values.size),
        "bootstrap 重复数": int(repeats),
        "bootstrap seed": int(seed),
    }


def _read_protocol() -> tuple[dict[str, Any], str]:
    raw = EVALUATION_PROTOCOL.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if EXPECTED_EVALUATION_PROTOCOL_SHA256 and digest != EXPECTED_EVALUATION_PROTOCOL_SHA256:
        raise ValueError("多次采样诊断协议哈希不匹配")
    protocol = json.loads(raw)
    if protocol.get("format") != "c3a_multi_sample_stability_protocol_v2":
        raise ValueError("多次采样诊断协议格式不匹配")
    return protocol, digest


def _check_resources(protocol: dict[str, Any]) -> tuple[torch.device, int, int]:
    if not torch.cuda.is_available():
        raise RuntimeError("多次采样诊断要求 CUDA GPU")
    other_processes = _active_cuda_processes()
    if other_processes:
        raise RuntimeError(f"GPU 仍有 {len(other_processes)} 个其他 CUDA 计算进程")
    device = torch.device("cuda:0")
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    threshold = int(protocol["resources"]["minimum_free_memory_gib"] * 1024**3)
    if free_bytes < threshold:
        raise RuntimeError(f"GPU 空闲显存 {free_bytes / 1024**3:.2f} GiB 低于协议门槛")
    return device, int(free_bytes), int(total_bytes)


def _build_flow_model(model_config: dict[str, Any]) -> SarConditionalFlow:
    if set(model_config) != {"architecture", "base_channels", "time_dimension"}:
        raise ValueError("C3-A Flow 模型配置键与冻结训练协议不一致")
    if model_config["architecture"] != "两层下采样的紧凑时间条件 U-Net":
        raise ValueError("C3-A Flow 架构标识与冻结训练协议不一致")
    return SarConditionalFlow(
        base_channels=int(model_config["base_channels"]),
        time_dimension=int(model_config["time_dimension"]),
    )


def _edge_metrics_with_mask(
    prediction: np.ndarray, target: np.ndarray, edge_mask: np.ndarray
) -> dict[str, float]:
    try:
        metrics = edge_metrics(prediction, target, edge_mask)
    except ValueError as error:
        raise ValueError(
            "固定 EPI/边缘强度指标未定义：边缘均值或相关分母为零/过小；按协议 fail-closed。"
        ) from error
    values = np.asarray([metrics["epi_corr"], metrics["edge_strength_ratio"]], dtype=np.float64)
    if not np.isfinite(values).all():
        raise FloatingPointError("固定 EPI/边缘强度指标含 NaN 或 Inf")
    return {"EPI": float(metrics["epi_corr"]), "边缘强度比": float(metrics["edge_strength_ratio"])}


def _compute_edge_metrics(
    prediction: np.ndarray, target: np.ndarray, sar_lr: np.ndarray
) -> dict[str, float]:
    masks, _ = build_fixed_masks(sar_lr, target, scale=2)
    pred = np.asarray(prediction)
    truth = np.asarray(target)
    if pred.ndim == 3 and pred.shape[0] == 1:
        pred = pred[0]
    if truth.ndim == 3 and truth.shape[0] == 1:
        truth = truth[0]
    return _edge_metrics_with_mask(pred, truth, masks["edge"])


def _build_fixed_edge_masks(
    dataset: SarTargetOnlyDataset,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    masks_by_id: dict[str, np.ndarray] = {}
    manifest_rows: list[dict[str, Any]] = []
    for index in range(len(dataset)):
        sample = dataset[index]
        sample_id = str(sample["name"])
        edge_mask, thresholds = build_fixed_masks(
            sample["lr"].numpy(), sample["target"].numpy(), scale=2
        )
        mask = edge_mask["edge"]
        digest = hashlib.sha256()
        digest.update(str(mask.shape).encode("utf-8"))
        digest.update(np.packbits(mask, bitorder="little").tobytes())
        if sample_id in masks_by_id:
            raise ValueError(f"development edge mask 身份重复：{sample_id}")
        masks_by_id[sample_id] = mask
        manifest_rows.append(
            {
                "样本 ID": sample_id,
                "来源组 ID": str(sample["source_group_id"]),
                "patch ID": str(sample["patch_id"]),
                "edge 阈值": float(thresholds["edge"]),
                "edge 像素数": int(mask.sum()),
                "edge 覆盖率": float(mask.mean()),
                "edge mask SHA-256": digest.hexdigest(),
            }
        )
    if len(masks_by_id) != len(dataset):
        raise ValueError("development edge mask 覆盖与固定身份数不一致")
    return masks_by_id, manifest_rows


def _edge_masks_for_batch(
    batch: dict[str, Any], masks_by_id: dict[str, np.ndarray]
) -> list[np.ndarray]:
    names = [str(name) for name in batch["name"]]
    try:
        return [masks_by_id[name] for name in names]
    except KeyError as error:
        raise ValueError(f"evaluation batch 缺少预计算 edge mask：{error.args[0]}") from error


def _batch_edge_metrics(
    prediction: torch.Tensor, target: torch.Tensor, edge_masks: list[np.ndarray]
) -> list[dict[str, float]]:
    predictions = prediction.clamp(0.0, 1.0).detach().cpu().numpy()
    targets = target.detach().cpu().numpy()
    if predictions.shape != targets.shape or len(edge_masks) != predictions.shape[0]:
        raise ValueError("固定结构指标的预测、目标和 edge mask batch 不一致")
    return [
        _edge_metrics_with_mask(predictions[index, 0], targets[index, 0], edge_masks[index])
        for index in range(predictions.shape[0])
    ]


def _validate_development_identity(
    dataset: SarTargetOnlyDataset, data_protocol: dict[str, Any]
) -> tuple[int, int]:
    sample_count = len(dataset.rows)
    sample_ids = [str(row.get("sample_id", "")) for row in dataset.rows]
    group_ids = {str(row.get("source_group_id", "")) for row in dataset.rows}
    if (
        sample_count != int(data_protocol["sample_count"])
        or len(set(sample_ids)) != sample_count
        or any(not sample_id for sample_id in sample_ids)
    ):
        raise ValueError("development 身份数或身份重复，与补充协议不符")
    if "" in group_ids or len(group_ids) != int(data_protocol["source_group_count"]):
        raise ValueError("development 来源组数或来源组标记与补充协议不符")
    return sample_count, len(group_ids)


def _set_evaluation_determinism(settings: dict[str, Any]) -> None:
    seed = int(settings["global_seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(bool(settings["deterministic_algorithms"]))
    torch.backends.cudnn.deterministic = bool(settings["cudnn_deterministic"])
    torch.backends.cudnn.benchmark = bool(settings["cudnn_benchmark"])


def _validate_training_run(run_dir: Path, protocol: dict[str, Any], protocol_hash: str) -> tuple[dict[str, Any], dict[str, Any]]:
    training_protocol = json.loads(TRAIN_PROTOCOL.read_text(encoding="utf-8"))
    training_protocol_hash = sha256_file(TRAIN_PROTOCOL)
    if training_protocol_hash != protocol["training_protocol_sha256"]:
        raise ValueError("训练协议哈希与多次采样诊断绑定值不匹配")
    run_manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    if run_manifest.get("status") != "complete" or run_manifest.get("protocol_sha256") != training_protocol_hash:
        raise ValueError("C3-A 训练尚未完成或协议哈希不匹配")
    for key in ("confirmation_payload_accessed", "test_set_accessed", "test_set_used", "optical_payload_read"):
        if run_manifest.get(key) is not False:
            raise ValueError(f"C3-A 训练 manifest 的 {key} 访问标记不是 false")
    input_manifest_path = run_dir / "input_manifest.json"
    if not input_manifest_path.is_file():
        raise ValueError("C3-A 训练缺少 input_manifest.json，拒绝读取 development 载荷")
    input_manifest = json.loads(input_manifest_path.read_text(encoding="utf-8"))
    source_manifest_path = training_protocol["split"]["source_manifest_path"]
    source_manifest_hash = training_protocol["split"]["source_manifest_sha256"]
    if input_manifest.get("source_manifest_path") != source_manifest_path:
        raise ValueError("C3-A input_manifest 来源清单路径与冻结协议不匹配")
    if (
        run_manifest.get("c1_source_manifest_sha256") != source_manifest_hash
        or input_manifest.get("source_manifest_sha256") != source_manifest_hash
    ):
        raise ValueError("C3-A 训练或 input_manifest 来源清单 SHA-256 与冻结协议不匹配")
    for key in ("confirmation_payload_accessed", "test_set_accessed", "test_set_used", "optical_payload_read"):
        if input_manifest.get(key) is not False:
            raise ValueError(f"C3-A input_manifest 的 {key} 访问标记不是 false")
    fixed_inputs = run_manifest.get("fixed_input_records")
    input_roles = input_manifest.get("roles")
    allowed_roles = {"train", "development"}
    if (
        not isinstance(fixed_inputs, dict)
        or not isinstance(input_roles, dict)
        or set(fixed_inputs) != allowed_roles
        or set(input_roles) != allowed_roles
    ):
        raise ValueError("C3-A input_manifest 只能包含 train/development 固定输入角色")
    if input_roles != fixed_inputs:
        raise ValueError("C3-A input_manifest roles 与训练 manifest 固定输入记录不一致")
    for role in ("train", "development"):
        expected_role = training_protocol["split"][role]
        fixed_role = fixed_inputs[role]
        if (
            fixed_role.get("sample_count") != expected_role["sample_count"]
            or fixed_role.get("source_group_count") != expected_role["source_group_count"]
        ):
            raise ValueError(f"C3-A {role} 固定输入样本数或来源组数与冻结协议不匹配")
        if fixed_role.get("payload_arrays_read") != ["lr", "target"]:
            raise ValueError(f"C3-A {role} 固定输入载荷记录必须且只能包含 lr/target")
    checkpoint_path = run_dir / "checkpoints/best.pt"
    checkpoint_hash = sha256_file(checkpoint_path)
    if checkpoint_hash != run_manifest.get("best_checkpoint_sha256"):
        raise ValueError("C3-A 最佳 checkpoint 哈希与训练 manifest 不匹配")
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    checkpoint_format = checkpoint.get("format")
    if checkpoint_format not in {
        "mmcanet_c3a_sar_only_flow_checkpoint_v1",
        "mmcanet_c3a_sar_only_flow_checkpoint_v2",
    }:
        raise ValueError("C3-A checkpoint 格式不匹配")
    if checkpoint.get("protocol_sha256") != training_protocol_hash:
        raise ValueError("C3-A checkpoint 训练协议哈希不匹配")
    if checkpoint.get("epoch") != run_manifest.get("best_epoch"):
        raise ValueError("C3-A 最佳 checkpoint 轮次与训练 manifest 不匹配")
    if checkpoint_format == "mmcanet_c3a_sar_only_flow_checkpoint_v2":
        resume_binding = run_manifest.get("resume_binding")
        if (
            run_manifest.get("format") != "mmcanet_c3a_sar_only_flow_run_v2"
            or not isinstance(resume_binding, dict)
            or checkpoint.get("resume_binding") != resume_binding
            or input_manifest.get("resume_binding") != resume_binding
        ):
            raise ValueError("C3-A checkpoint、input_manifest 与运行 manifest 的续训绑定不一致")
        if checkpoint.get("best_epoch") != run_manifest.get("best_epoch"):
            raise ValueError("C3-A checkpoint 最佳轮次与训练 manifest 不匹配")
        checkpoint_best_score = checkpoint.get("best_development_group_balanced_mean_psnr")
        manifest_best_score = run_manifest.get("best_development_group_balanced_mean_psnr")
        if (
            not isinstance(checkpoint_best_score, (int, float))
            or not isinstance(manifest_best_score, (int, float))
            or not math.isclose(float(checkpoint_best_score), float(manifest_best_score), rel_tol=0.0, abs_tol=0.0)
        ):
            raise ValueError("C3-A checkpoint 最佳指标与训练 manifest 不匹配")
    fixed = run_manifest.get("fixed_input_records", {}).get("development", {})
    if fixed.get("sample_count") != protocol["data"]["sample_count"]:
        raise ValueError("C3-A 训练 manifest 的 development 数量不匹配")
    if fixed.get("source_group_count") != protocol["data"]["source_group_count"]:
        raise ValueError("C3-A 训练 manifest 的 development 来源组数量不匹配")
    if not protocol_hash:
        raise ValueError("多次采样诊断协议哈希为空")
    return run_manifest, checkpoint


@torch.inference_mode()
def evaluate_multi_sample(
    model: SarConditionalFlow,
    loader: DataLoader,
    *,
    device: torch.device,
    seed: int,
    draws: int,
    sampler_steps: int,
    noise_sigma: float,
    edge_masks_by_id: dict[str, np.ndarray],
) -> tuple[list[dict[str, Any]], float]:
    model.eval()
    rows: list[dict[str, Any]] = []
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    for batch_index, batch in enumerate(loader, start=1):
        lr = batch["lr"].to(device, non_blocking=False)
        target = batch["target"].to(device, non_blocking=False)
        condition = upsample_sar_condition(lr, size=(256, 256))
        edge_masks = _edge_masks_for_batch(batch, edge_masks_by_id)
        clipped_predictions: list[torch.Tensor] = []
        batch_draw_rows: list[list[dict[str, Any]]] = []
        for draw_index in range(draws):
            noise = torch.cat(
                [
                    draw_noise_for_identity(
                        str(sample_id), draw_index=draw_index, seed=seed, device=device
                    )
                    for sample_id in batch["name"]
                ],
                dim=0,
            )
            raw = sample_heun(model, condition, noise, steps=sampler_steps, noise_sigma=noise_sigma)
            if raw.shape != target.shape or not torch.isfinite(raw).all():
                raise FloatingPointError("多次采样输出尺寸不符或含 NaN/Inf")
            clipped_predictions.append(raw.clamp(0.0, 1.0))
            draw_rows = _metric_rows(raw, raw, target, batch, method_name="C3-A SAR-only Flow")
            structural_rows = _batch_edge_metrics(raw, target, edge_masks)
            for row, structural in zip(draw_rows, structural_rows):
                row["采样编号"] = draw_index
                row.update(structural)
            batch_draw_rows.append(draw_rows)
        stacked = torch.stack(clipped_predictions, dim=0)
        pixel_std = stacked.std(dim=0, correction=1).mean(dim=(1, 2, 3)).detach().cpu().numpy()
        for draw_rows in batch_draw_rows:
            for index, row in enumerate(draw_rows):
                row["预测像素标准差均值"] = float(pixel_std[index])
                rows.append(row)
        print(
            f"多次采样评价进度：{batch_index}/{len(loader)} 批，"
            f"已完成 {len(rows) // draws} 个 development 身份、{draws} 次固定采样。",
            flush=True,
        )
    torch.cuda.synchronize(device)
    return rows, time.perf_counter() - started


@torch.inference_mode()
def evaluate_baseline_with_structure(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    edge_masks_by_id: dict[str, np.ndarray],
) -> tuple[list[dict[str, Any]], float]:
    model.eval()
    rows: list[dict[str, Any]] = []
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    for batch_index, batch in enumerate(loader, start=1):
        lr = batch["lr"].to(device, non_blocking=False)
        target = batch["target"].to(device, non_blocking=False)
        raw = model(lr)
        if raw.shape != target.shape or not torch.isfinite(raw).all():
            raise FloatingPointError("C2 SAR-only baseline 输出尺寸不符或含 NaN/Inf")
        batch_rows = _metric_rows(raw, raw, target, batch, method_name="C2 SAR-only")
        structural_rows = _batch_edge_metrics(
            raw, target, _edge_masks_for_batch(batch, edge_masks_by_id)
        )
        for row, structural in zip(batch_rows, structural_rows):
            row.update(structural)
        rows.extend(batch_rows)
        if batch_index % 10 == 0 or batch_index == len(loader):
            print(f"SAR-only baseline 评价进度：{batch_index}/{len(loader)} 批。", flush=True)
    torch.cuda.synchronize(device)
    return rows, time.perf_counter() - started


def _run(run_dir: Path) -> int:
    started_utc = _utc_now()
    protocol, protocol_hash = _read_protocol()
    if OUTPUT_PATH.exists():
        raise FileExistsError(f"诊断结果目录已存在：{OUTPUT_PATH}")
    device, free_bytes, total_bytes = _check_resources(protocol)
    run_manifest, checkpoint = _validate_training_run(run_dir, protocol, protocol_hash)
    run_dir = run_dir.resolve()
    run_dir_relative = run_dir.relative_to(ROOT)
    git_head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    git_status = subprocess.run(
        ["git", "status", "--short", "--untracked-files=all"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    command = [sys.executable, str(Path(__file__).resolve()), "--training-run-dir", str(run_dir)]
    code_paths = {
        "scripts/evaluate_c3a_multisample.py": Path(__file__).resolve(),
        "scripts/run_phase_c3a_sar_flow.py": ROOT / "scripts/run_phase_c3a_sar_flow.py",
        "MMCANetV2/phase_c3a_flow.py": ROOT / "MMCANetV2/phase_c3a_flow.py",
        "MMCANetV2/phase_c2_data.py": ROOT / "MMCANetV2/phase_c2_data.py",
        "Early-FusionCascadeJDSR/eval_metrics.py": ROOT / "Early-FusionCascadeJDSR/eval_metrics.py",
    }
    code_sha256 = {name: sha256_file(path) for name, path in code_paths.items()}
    OUTPUT_PATH.mkdir(parents=True, exist_ok=False)
    manifest: dict[str, Any] = {
        "format": "c3a_multi_sample_stability_result_v2",
        "run_id": OUTPUT_PATH.name,
        "status": "running",
        "started_utc": started_utc,
        "protocol_path": str(EVALUATION_PROTOCOL.relative_to(ROOT)),
        "protocol_sha256": protocol_hash,
        "training_run": str(run_dir_relative),
        "training_run_manifest_status": run_manifest["status"],
        "training_protocol_sha256": run_manifest["protocol_sha256"],
        "flow_checkpoint_sha256": sha256_file(run_dir / "checkpoints/best.pt"),
        "baseline_checkpoint_sha256": sha256_file(BASELINE_PATH / "checkpoints/best.pt"),
        "git_head": git_head,
        "git_status_dirty": bool(git_status.strip()),
        "code_sha256": code_sha256,
        "sample_count": None,
        "source_group_count": None,
        "draws_per_identity": int(protocol["sampling"]["draws_per_identity"]),
        "sampler_steps": int(protocol["sampling"]["sampler_steps"]),
        "function_evaluations_per_draw": int(protocol["sampling"]["function_evaluations_per_draw"]),
        "determinism": protocol["determinism"],
        "structural_metric_protocol": protocol["metrics"]["structural"],
        "development_payload_accessed": False,
        "baseline_runtime_seconds": None,
        "flow_runtime_seconds": None,
        "baseline_peak_gpu_memory_allocated_bytes": None,
        "flow_peak_gpu_memory_allocated_bytes": None,
        "gpu_free_memory_before_evaluation_bytes": free_bytes,
        "gpu_total_memory_bytes": total_bytes,
        "selection_or_checkpoint_update": False,
        "confirmation_payload_accessed": False,
        "test_set_accessed": False,
        "test_set_used": False,
        "optical_payload_read": False,
    }
    development_accessed = False
    try:
        _json_write(OUTPUT_PATH / "manifest.json", manifest)
        (OUTPUT_PATH / "command.txt").write_text(
            "多次采样诊断命令：\n" + " ".join(command) + "\n", encoding="utf-8"
        )
        (OUTPUT_PATH / "protocol.json").write_bytes(EVALUATION_PROTOCOL.read_bytes())
        (OUTPUT_PATH / "git_state.txt").write_text(
            f"HEAD：{git_head}\n工作区状态（运行开始前）：\n{git_status}", encoding="utf-8"
        )
        training_protocol = json.loads(TRAIN_PROTOCOL.read_text(encoding="utf-8"))
        _set_evaluation_determinism(protocol["determinism"])
        environment = _environment(device)
        environment["determinism"] = protocol["determinism"]
        _json_write(OUTPUT_PATH / "environment.json", environment)
        (OUTPUT_PATH / "environment.txt").write_text(
            "\n".join(f"{key}：{value}" for key, value in environment.items()) + "\n", encoding="utf-8"
        )
        source_hash = sha256_file(SOURCE_MANIFEST)
        if (
            source_hash != protocol["data"]["source_manifest_sha256"]
            or source_hash != training_protocol["split"]["source_manifest_sha256"]
        ):
            raise ValueError("C1 v2 冻结来源清单 SHA-256 不匹配")
        manifest["source_manifest_sha256"] = source_hash
        _json_write(OUTPUT_PATH / "manifest.json", manifest)
        dataset = SarTargetOnlyDataset(INPUT_ROOT / "development", SOURCE_MANIFEST, role="development", scale=2)
        development_accessed = True
        manifest["development_payload_accessed"] = True
        _json_write(OUTPUT_PATH / "manifest.json", manifest)
        dataset.verify_all()
        fixed = run_manifest["fixed_input_records"]["development"]
        if (
            dataset.manifest_sha256 != fixed.get("manifest_sha256")
            or dataset.files_csv_sha256 != fixed.get("files_csv_sha256")
        ):
            raise ValueError("C3-A 训练与多次采样诊断的 development 固定输入哈希不同")

        sample_count, source_group_count = _validate_development_identity(dataset, protocol["data"])
        manifest["development_input_manifest_sha256"] = fixed_input_manifest_sha256(dataset)
        manifest["development_input_files_csv_sha256"] = fixed_input_files_sha256(dataset)
        manifest["sample_count"] = sample_count
        manifest["source_group_count"] = source_group_count
        data_loader = DataLoader(
            dataset,
            batch_size=int(protocol["resources"]["batch_size"]),
            shuffle=False,
            num_workers=0,
            drop_last=False,
        )
        edge_masks_by_id, edge_mask_rows = _build_fixed_edge_masks(dataset)
        _csv_write(
            OUTPUT_PATH / "edge_mask_manifest.csv",
            edge_mask_rows,
            ["样本 ID", "来源组 ID", "patch ID", "edge 阈值", "edge 像素数", "edge 覆盖率", "edge mask SHA-256"],
        )
        manifest["edge_mask_manifest_sha256"] = sha256_file(OUTPUT_PATH / "edge_mask_manifest.csv")
        _json_write(OUTPUT_PATH / "manifest.json", manifest)
        baseline_model, _, _ = _load_baseline(device, dataset, training_protocol)
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        baseline_rows, baseline_runtime = evaluate_baseline_with_structure(
            baseline_model,
            data_loader,
            device=device,
            edge_masks_by_id=edge_masks_by_id,
        )
        baseline_peak = int(torch.cuda.max_memory_allocated(device))
        _csv_write(
            OUTPUT_PATH / "baseline_metrics.csv",
            baseline_rows,
            ["样本 ID", "来源组 ID", "patch ID", "PSNR", "SSIM", "EPI", "边缘强度比", "最终裁剪比例"],
        )
        baseline_parameter_count = sum(parameter.numel() for parameter in baseline_model.parameters())
        baseline_model.to("cpu")
        del baseline_model
        torch.cuda.empty_cache()

        model_config = training_protocol["method"]["model"]
        model = _build_flow_model(model_config).to(device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        flow_parameter_count = sum(parameter.numel() for parameter in model.parameters())
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        flow_rows, flow_runtime = evaluate_multi_sample(
            model,
            data_loader,
            device=device,
            seed=int(protocol["sampling"]["seed"]),
            draws=int(protocol["sampling"]["draws_per_identity"]),
            sampler_steps=int(protocol["sampling"]["sampler_steps"]),
            noise_sigma=float(protocol["sampling"]["noise_sigma"]),
            edge_masks_by_id=edge_masks_by_id,
        )
        flow_peak = int(torch.cuda.max_memory_allocated(device))
        identity_summary = aggregate_multi_sample_metrics(
            flow_rows,
            baseline_rows,
            expected_draws=int(protocol["sampling"]["draws_per_identity"]),
        )
        bootstrap_seed = int(protocol["statistics"]["bootstrap_seed"])
        bootstrap_repeats = int(protocol["statistics"]["bootstrap_repeats"])
        summaries = {
            metric: group_bootstrap_summary(
                identity_summary,
                metric=metric,
                seed=bootstrap_seed + offset,
                repeats=bootstrap_repeats,
            )
            for offset, metric in enumerate(
                (
                    "平均配对 ΔPSNR",
                    "平均配对 ΔSSIM",
                    "平均配对 ΔEPI",
                    "Flow PSNR 样本标准差",
                    "Flow SSIM 样本标准差",
                    "Flow EPI 样本标准差",
                    "Flow 平均边缘强度比",
                    "平均预测像素标准差",
                )
            )
        }

        _csv_write(
            OUTPUT_PATH / "flow_per_draw_metrics.csv",
            flow_rows,
            ["样本 ID", "来源组 ID", "patch ID", "采样编号", "PSNR", "SSIM", "EPI", "边缘强度比", "最终裁剪比例", "预测像素标准差均值"],
        )
        _csv_write(OUTPUT_PATH / "identity_summary.csv", identity_summary, list(identity_summary[0]))
        _json_write(OUTPUT_PATH / "group_bootstrap_summary.json", summaries)
        input_manifest = {
            "source_manifest_path": str(SOURCE_MANIFEST.relative_to(ROOT)),
            "source_manifest_sha256": sha256_file(SOURCE_MANIFEST),
            "development_fixed_manifest_sha256": manifest["development_input_manifest_sha256"],
            "development_fixed_files_csv_sha256": manifest["development_input_files_csv_sha256"],
            "edge_mask_manifest_sha256": manifest["edge_mask_manifest_sha256"],
            "sample_count": manifest["sample_count"],
            "source_group_count": manifest["source_group_count"],
            "payload_arrays_read": ["lr", "target"],
            "confirmation_payload_accessed": False,
            "test_set_accessed": False,
            "test_set_used": False,
            "optical_payload_read": False,
        }
        _json_write(OUTPUT_PATH / "input_manifest.json", input_manifest)
        finished_utc = _utc_now()
        runtime_summary = {
            "started_utc": started_utc,
            "finished_utc": finished_utc,
            "baseline_runtime_seconds": baseline_runtime,
            "flow_runtime_seconds": flow_runtime,
            "flow_runtime_seconds_per_draw_per_identity": flow_runtime / (len(dataset) * int(protocol["sampling"]["draws_per_identity"])),
        }
        _json_write(OUTPUT_PATH / "runtime.json", runtime_summary)
        output_names = [
            "baseline_metrics.csv",
            "edge_mask_manifest.csv",
            "environment.json",
            "environment.txt",
            "flow_per_draw_metrics.csv",
            "group_bootstrap_summary.json",
            "identity_summary.csv",
            "input_manifest.json",
            "protocol.json",
            "runtime.json",
            "command.txt",
            "git_state.txt",
        ]
        output_hashes = {name: sha256_file(OUTPUT_PATH / name) for name in output_names}
        manifest.update(
            {
                "status": "complete",
                "finished_utc": finished_utc,
                "source_manifest_sha256": sha256_file(SOURCE_MANIFEST),
                "flow_runtime_seconds": flow_runtime,
                "baseline_runtime_seconds": baseline_runtime,
                "flow_runtime_seconds_per_draw_per_identity": flow_runtime / (len(dataset) * int(protocol["sampling"]["draws_per_identity"])),
                "baseline_peak_gpu_memory_allocated_bytes": baseline_peak,
                "flow_peak_gpu_memory_allocated_bytes": flow_peak,
                "baseline_parameter_count": baseline_parameter_count,
                "flow_parameter_count": flow_parameter_count,
                "output_sample_count": len(flow_rows),
                "output_files": output_names,
                "output_sha256": output_hashes,
                "baseline_metrics_row_count": len(baseline_rows),
                "summary": summaries,
                "interpretation_zh": "同一最佳 checkpoint 在已反复使用的 development 上进行四次固定种子采样；EPI 与边缘强度比仅表示对 noisy pseudo-target 固定边缘区域的表示域一致性。结果是单训练 seed 的描述性诊断，不用于选轮、改 gate 或独立泛化主张。",
            }
        )
        _json_write(OUTPUT_PATH / "manifest.json", manifest)
        print(
            f"完成：{len(dataset)} 个 development 身份、{manifest['source_group_count']} 个来源组、"
            f"每个身份 {protocol['sampling']['draws_per_identity']} 次采样；仅输出描述性稳定性统计。",
            flush=True,
        )
        return 0
    except Exception as error:
        manifest.update(
            {
                "status": "failed",
                "finished_utc": _utc_now(),
                "failure_type": type(error).__name__,
                "failure_message": str(error),
                "development_payload_accessed": development_accessed,
            }
        )
        _json_write(OUTPUT_PATH / "manifest.json", manifest)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="C3-A development 多次采样稳定性诊断")
    parser.add_argument("--training-run-dir", type=Path, default=ROOT / "experiments/phase_c3a_sar_only_flow_c1_v2_20261105_v1")
    args = parser.parse_args()
    _run(args.training_run_dir.resolve())


if __name__ == "__main__":
    main()
