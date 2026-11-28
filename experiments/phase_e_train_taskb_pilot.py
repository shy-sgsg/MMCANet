"""在 Phase E v3 固定表示域合成任务上训练与评价 SAR-only Pilot。"""
from __future__ import annotations

import argparse
import csv
import copy
import hashlib
import json
import math
import os
import platform
import random
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn.functional as F
import cv2
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.phase_d1_edsr_20261124_v1.edsr_sar import (  # noqa: E402
    build_edsr_sar,
    load_published_edsr_class,
)
from experiments.phase_e_taskb_metrics import METRIC_NAMES, compute_taskb_metrics  # noqa: E402
from experiments.phase_e_taskb_models import (  # noqa: E402
    ConditionalResidualUNet,
    ResidualMeanCorrector,
    linear_beta_schedule,
)
from experiments.phase_e_taskb_operator import downsample_noisy_sar_x2  # noqa: E402
from experiments.published_baselines_sar_only_20261123_v1 import train_swinir_sar as swinir  # noqa: E402

SEED = 20261126
BOOTSTRAP_SEED = 20261127
MAX_EPOCHS = 30
MAX_STAGE_SECONDS = 4 * 60 * 60
MAX_TOTAL_SECONDS = 12 * 60 * 60
DDIM_STEPS = 20
DIFFUSION_STEPS = 1000
DEFAULT_PATCH = 64


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _append_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    exists = path.exists()
    with path.open("a", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        if not exists:
            writer.writeheader()
        writer.writerows(rows)
        stream.flush()


def _save_unique_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    with path.open("xb") as stream:
        torch.save(payload, stream)


def _save_predictions(directory: Path, predictions: dict[str, np.ndarray]) -> None:
    directory.mkdir(parents=True, exist_ok=False)
    for sample_id, prediction in predictions.items():
        path = directory / f"{sample_id}.npy"
        with path.open("xb") as stream:
            np.save(stream, np.asarray(prediction, dtype=np.float32), allow_pickle=False)


def _seed_for(label: str, base: int = SEED) -> int:
    material = f"{base}:{label}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") & ((1 << 63) - 1)


def _set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def _git_state() -> dict[str, Any]:
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--short", "--untracked-files=all"], cwd=PROJECT_ROOT,
            check=True, capture_output=True, text=True,
        ).stdout
        return {"head": head, "dirty": bool(status), "status_short": status}
    except (OSError, subprocess.CalledProcessError) as error:
        return {"head": None, "dirty": True, "error": str(error)}


def _gpu_snapshot(device: torch.device) -> dict[str, Any]:
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    try:
        smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True, timeout=15,
        ).stdout.strip().splitlines()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        smi = []
    return {
        "device": torch.cuda.get_device_name(device),
        "free_bytes": int(free_bytes),
        "total_bytes": int(total_bytes),
        "nvidia_smi": smi,
    }


def _assert_total_budget(started: float, next_stage: str) -> None:
    if time.monotonic() - started >= MAX_TOTAL_SECONDS:
        raise TimeoutError(f"达到本轮 12 小时 GPU 总预算，未开始 {next_stage}")


def _gpu_smoke(
    train_record: dict[str, Any], device: torch.device, output_dir: Path
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(SEED)
    lr_patch, target_patch = _crop_pair(train_record, patch_size=64, rng=rng)
    lr = torch.from_numpy(lr_patch[None]).to(device)
    target = torch.from_numpy(target_patch[None]).to(device)
    results = []

    def check(name: str, model: nn.Module, forward) -> None:
        model = model.to(device).train()
        output = forward(model)
        if output.shape != target.shape or not torch.isfinite(output).all():
            raise ValueError(f"{name} GPU smoke 输出契约失败")
        loss = F.l1_loss(output, target)
        loss.backward()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        if not gradients or any(not torch.isfinite(grad).all() for grad in gradients):
            raise FloatingPointError(f"{name} GPU smoke 梯度无效")
        state_path = output_dir / f"{name}_state.pt"
        with state_path.open("xb") as stream:
            torch.save(model.state_dict(), stream)
        restored = torch.load(state_path, map_location="cpu", weights_only=True)
        model.load_state_dict(restored, strict=True)
        results.append(
            {"method": name, "output_shape": list(output.shape),
             "loss": float(loss.detach().cpu()),
             "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
             "state_sha256": sha256_file(state_path)}
        )
        del model, output, loss, gradients, restored
        torch.cuda.empty_cache()

    edsr_model = build_edsr_sar(load_published_edsr_class())
    check("EDSR", edsr_model, lambda model: model(lr))
    swinir_model = swinir.build_swinir(swinir.load_swinir_class())
    check("SwinIR", swinir_model, lambda model: model(lr))

    base = torch.full_like(target, 0.5)
    corrector = ResidualMeanCorrector().to(device).train()
    check("均值校正器", corrector, lambda model: model(lr, base))
    diffusion = ConditionalResidualUNet().to(device).train()
    timestep = torch.tensor([137], device=device, dtype=torch.long)
    noisy_residual = torch.randn_like(target)
    check(
        "中心残差扩散",
        diffusion,
        lambda model: model(noisy_residual, base, lr, timestep),
    )
    result = {
        "status": "passed",
        "seed": SEED,
        "input_sample_id": train_record["sample_id"],
        "output_shape": [1, 1, 128, 128],
        "models": results,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "confirmation_payload_accessed": False,
        "test_payload_accessed": False,
        "optical_payload_loaded": False,
    }
    _write_json(output_dir / "smoke_manifest.json", result)
    print("CUDA 冒烟验证通过：EDSR、SwinIR、均值校正器与扩散网络均完成梯度和 checkpoint 往返。", flush=True)
    return result


def _load_records(input_manifest_path: Path) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    manifest = json.loads(input_manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("format") != "phase_e_gt25_synthetic_inputs_v3"
        or manifest.get("status") != "complete"
        or manifest.get("roles_allowed") != ["train", "development"]
        or manifest.get("confirmation_payload_accessed") is not False
        or manifest.get("test_payload_accessed") is not False
        or manifest.get("optical_payload_loaded") is not False
    ):
        raise ValueError("固定输入 manifest 不符合 Task B v3 权限合同")
    records: dict[str, list[dict[str, Any]]] = {"train": [], "development": []}
    for row in manifest.get("samples", []):
        role = row.get("split_role")
        if role not in records:
            raise ValueError("输入 manifest 包含未授权数据角色")
        if not row.get("source_group_id") or not row.get("sample_id"):
            raise ValueError("固定输入样本缺少 ID 或来源组")
        lr_path = Path(row["lr_npy"])
        target_path = Path(row["reference_npy"])
        if sha256_file(lr_path) != row.get("lr_npy_sha256"):
            raise ValueError(f"固定 LR 哈希不匹配：{row['sample_id']}")
        if sha256_file(target_path) != row.get("reference_npy_sha256"):
            raise ValueError(f"固定参考哈希不匹配：{row['sample_id']}")
        lr = np.load(lr_path, allow_pickle=False)
        target = np.load(target_path, allow_pickle=False)
        if lr.dtype != np.float32 or lr.shape != (1, 256, 256):
            raise ValueError(f"固定 LR shape/dtype 错误：{row['sample_id']}")
        if target.dtype != np.float32 or target.shape != (1, 512, 512):
            raise ValueError(f"固定目标 shape/dtype 错误：{row['sample_id']}")
        if (
            not np.isfinite(lr).all()
            or not np.isfinite(target).all()
            or lr.min() < 0
            or target.min() < 0
            or target.max() > 1
        ):
            raise ValueError(f"固定输入含非有限或越界值：{row['sample_id']}")
        records[role].append(
            {
                "sample_id": row["sample_id"],
                "source_group_id": row["source_group_id"],
                "lr": np.ascontiguousarray(lr),
                "target": np.ascontiguousarray(target),
                "lr_path": str(lr_path),
                "target_path": str(target_path),
            }
        )
    if len(records["train"]) != 327 or len(records["development"]) != 70:
        raise ValueError("train/development 样本数不符合已冻结的 327/70")
    train_groups = {row["source_group_id"] for row in records["train"]}
    development_groups = {row["source_group_id"] for row in records["development"]}
    if train_groups & development_groups:
        raise ValueError("train/development 来源组重叠")
    for role in records:
        records[role].sort(key=lambda row: row["sample_id"])
    return records, manifest


def _crop_aligned(
    record: dict[str, Any],
    high_resolution_arrays: tuple[np.ndarray, ...],
    *,
    patch_size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, ...]:
    lr = record["lr"]
    height, width = lr.shape[-2:]
    if patch_size > min(height, width):
        raise ValueError("LR patch 大于固定输入")
    top = int(rng.integers(0, height - patch_size + 1))
    left = int(rng.integers(0, width - patch_size + 1))
    lr_crop = lr[:, top : top + patch_size, left : left + patch_size]
    crops = [lr_crop]
    for array in high_resolution_arrays:
        value = np.asarray(array)
        if value.ndim != 3 or value.shape[0] != 1 or value.shape[-2:] != (512, 512):
            raise ValueError("对齐裁剪的 HR 数组必须为 [1,512,512]")
        crops.append(
            value[
                :, top * 2 : (top + patch_size) * 2,
                left * 2 : (left + patch_size) * 2,
            ]
        )
    if rng.integers(2):
        crops = [crop[..., ::-1] for crop in crops]
    if rng.integers(2):
        crops = [crop[..., ::-1, :] for crop in crops]
    if rng.integers(2):
        crops = [crop.transpose(0, 2, 1) for crop in crops]
    return tuple(np.ascontiguousarray(crop) for crop in crops)


def _crop_pair(
    record: dict[str, Any], *, patch_size: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    low, high = _crop_aligned(
        record, (record["target"],), patch_size=patch_size, rng=rng
    )
    return low, high


def _group_balanced_mean(
    rows: list[dict[str, Any]], metric: str = "psnr_db"
) -> float | None:
    groups: dict[str, list[float]] = {}
    if not rows:
        raise ValueError("开发集评价为空")
    for row in rows:
        value = row.get(metric)
        if value is None:
            continue
        groups.setdefault(str(row["source_group_id"]), []).append(float(value))
    if not groups:
        return None
    return float(np.mean([np.mean(values) for values in groups.values()]))


def _evaluate_model(
    model: nn.Module,
    records: list[dict[str, Any]],
    device: torch.device,
    *,
    base_predictions: dict[str, np.ndarray] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    model.eval()
    rows = []
    predictions = {}
    with torch.inference_mode():
        for record in records:
            sample_id = record["sample_id"]
            lr = torch.from_numpy(record["lr"][None]).to(device)
            if base_predictions is None:
                output = model(lr)
            else:
                base = torch.from_numpy(base_predictions[sample_id][None, None]).to(device)
                output = model(lr, base)
            prediction = output[0, 0].float().cpu().numpy()
            prediction = np.clip(prediction, 0.0, 1.0).astype(np.float32, copy=False)
            target = record["target"][0]
            metrics = compute_taskb_metrics(
                prediction, target, record["lr"][0], allow_undefined_epi=True
            )
            if any(
                value is not None and not math.isfinite(value)
                for value in metrics.values()
            ):
                raise FloatingPointError(f"开发指标非有限：{sample_id}")
            rows.append(
                {
                    "sample_id": sample_id,
                    "source_group_id": record["source_group_id"],
                    "epi_valid": metrics["epi_corr"] is not None,
                    **metrics,
                }
            )
            predictions[sample_id] = prediction
    return rows, predictions


def _save_stage_metrics(stage_dir: Path, rows: list[dict[str, Any]], *, epoch: int) -> float:
    summary = {
        "epoch": epoch,
        "development_group_balanced_psnr_db": _group_balanced_mean(rows),
        "development_image_count": len(rows),
        "development_group_count": len({row["source_group_id"] for row in rows}),
    }
    _append_csv(stage_dir / "development_epoch_summary.csv", [summary])
    return float(summary["development_group_balanced_psnr_db"])


def _save_epoch_checkpoint(
    checkpoint_dir: Path,
    *,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    method: str,
    extra: dict[str, Any] | None = None,
) -> Path:
    path = checkpoint_dir / f"epoch_{epoch:03d}.pt"
    payload = {
        "format": "phase_e_taskb_checkpoint_v1",
        "method": method,
        "epoch": epoch,
        "seed": SEED,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all(),
    }
    if extra:
        payload.update(extra)
    _save_unique_checkpoint(path, payload)
    return path


def _load_best_model(model: nn.Module, checkpoint_path: Path, device: torch.device) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)


def _write_stage_manifest(path: Path, manifest: dict[str, Any]) -> None:
    _write_json(path, manifest)


def train_supervised_baseline(
    *,
    method: str,
    model: nn.Module,
    train_records: list[dict[str, Any]],
    development_records: list[dict[str, Any]],
    device: torch.device,
    output_dir: Path,
    input_manifest_sha256: str,
    patch_size: int,
) -> tuple[nn.Module, Path, list[dict[str, Any]], dict[str, np.ndarray], dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir()
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.0)
    best_score = -math.inf
    best_checkpoint = None
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats(device)
    stage_manifest = {
        "method": method,
        "status": "running",
        "seed": SEED,
        "epoch_limit": MAX_EPOCHS,
        "patch_lr": patch_size,
        "batch_size": 1,
        "optimizer": "AdamW(lr=1e-4,weight_decay=0)",
        "loss": "L1",
        "input_manifest_sha256": input_manifest_sha256,
        "confirmation_payload_accessed": False,
        "test_payload_accessed": False,
        "optical_payload_loaded": False,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }
    _write_stage_manifest(output_dir / "manifest.json", stage_manifest)

    for epoch in range(1, MAX_EPOCHS + 1):
        if time.monotonic() - started >= MAX_STAGE_SECONDS:
            stage_manifest["status"] = "time_budget_reached"
            break
        model.train()
        epoch_rng = np.random.default_rng(SEED + epoch)
        order = epoch_rng.permutation(len(train_records))
        losses = []
        for index in order:
            lr_patch, target_patch = _crop_pair(
                train_records[int(index)], patch_size=patch_size, rng=epoch_rng
            )
            lr_tensor = torch.from_numpy(lr_patch[None]).to(device)
            target_tensor = torch.from_numpy(target_patch[None]).to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(lr_tensor)
            if prediction.shape != target_tensor.shape:
                raise ValueError(f"{method} 输出与目标 shape 不一致")
            loss = F.l1_loss(prediction, target_tensor)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"{method} loss 非有限")
            loss.backward()
            if any(
                parameter.grad is not None and not torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
            ):
                raise FloatingPointError(f"{method} 梯度非有限")
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        development_rows, _ = _evaluate_model(model, development_records, device)
        score = _save_stage_metrics(output_dir, development_rows, epoch=epoch)
        checkpoint = _save_epoch_checkpoint(
            checkpoint_dir,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            method=method,
            extra={"development_group_balanced_psnr_db": score},
        )
        if score > best_score:
            best_score = score
            best_checkpoint = checkpoint
            stage_manifest["best_epoch"] = epoch
            stage_manifest["best_score"] = score
            stage_manifest["best_checkpoint"] = str(checkpoint)
            stage_manifest["best_checkpoint_sha256"] = sha256_file(checkpoint)
        stage_manifest["last_epoch"] = epoch
        stage_manifest["last_checkpoint"] = str(checkpoint)
        stage_manifest["elapsed_seconds"] = time.monotonic() - started
        stage_manifest["peak_cuda_memory_bytes"] = int(torch.cuda.max_memory_allocated(device))
        stage_manifest["status"] = "running"
        _append_csv(
            output_dir / "train_log.csv",
            [{"epoch": epoch, "train_l1_mean": float(np.mean(losses)), "steps": len(losses)}],
        )
        _write_stage_manifest(output_dir / "manifest.json", stage_manifest)
        print(
            f"{method} 第 {epoch}/{MAX_EPOCHS} 轮：训练 L1={np.mean(losses):.6f}，"
            f"开发组均值 PSNR={score:.4f} dB；已用 {stage_manifest['elapsed_seconds']:.0f} 秒。",
            flush=True,
        )
        if stage_manifest["elapsed_seconds"] >= MAX_STAGE_SECONDS:
            stage_manifest["status"] = "time_budget_reached"
            break

    if best_checkpoint is None:
        raise RuntimeError(f"{method} 未完成任何有效 epoch")
    _load_best_model(model, best_checkpoint, device)
    final_rows, final_predictions = _evaluate_model(model, development_records, device)
    stage_manifest["status"] = (
        "completed" if stage_manifest.get("last_epoch") == MAX_EPOCHS else "time_budget_reached"
    )
    stage_manifest["final_best_epoch"] = stage_manifest["best_epoch"]
    stage_manifest["elapsed_seconds"] = time.monotonic() - started
    stage_manifest["peak_cuda_memory_bytes"] = int(torch.cuda.max_memory_allocated(device))
    _write_stage_manifest(output_dir / "manifest.json", stage_manifest)
    _append_csv(output_dir / "development_best_per_image.csv", final_rows)
    _save_predictions(output_dir / "predictions_best", final_predictions)
    return model, best_checkpoint, final_rows, final_predictions, stage_manifest


def _save_component_predictions(
    directory: Path, predictions: dict[str, np.ndarray]
) -> dict[str, dict[str, str]]:
    _save_predictions(directory, predictions)
    return {
        sample_id: {
            "path": str(directory / f"{sample_id}.npy"),
            "sha256": sha256_file(directory / f"{sample_id}.npy"),
        }
        for sample_id in predictions
    }


def _group_bootstrap(
    rows: list[dict[str, Any]],
    reference_rows: list[dict[str, Any]],
    *,
    method: str,
    reference: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    method_map = {row["sample_id"]: row for row in rows}
    reference_map = {row["sample_id"]: row for row in reference_rows}
    if set(method_map) != set(reference_map):
        raise ValueError("配对 bootstrap 的样本集合不一致")
    grouped: dict[str, list[dict[str, Any]]] = {}
    paired_rows = []
    for sample_id, current in method_map.items():
        baseline = reference_map[sample_id]
        if current["source_group_id"] != baseline["source_group_id"]:
            raise ValueError("配对 bootstrap 来源组不一致")
        item = {
            "sample_id": sample_id,
            "source_group_id": current["source_group_id"],
            "method": method,
            "reference": reference,
        }
        for metric in METRIC_NAMES:
            current_value = current.get(metric)
            reference_value = baseline.get(metric)
            if current_value is None or reference_value is None:
                item[f"{metric}_method"] = (
                    None if current_value is None else float(current_value)
                )
                item[f"{metric}_reference"] = (
                    None if reference_value is None else float(reference_value)
                )
                item[f"delta_{metric}"] = None
            else:
                item[f"{metric}_method"] = float(current_value)
                item[f"{metric}_reference"] = float(reference_value)
                item[f"delta_{metric}"] = float(current_value) - float(reference_value)
        paired_rows.append(item)
        grouped.setdefault(current["source_group_id"], []).append(item)

    group_names = sorted(grouped)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    summary_rows = []
    for metric in METRIC_NAMES:
        valid_by_group = {
            group: [row[f"delta_{metric}"] for row in grouped[group]
                    if row[f"delta_{metric}"] is not None]
            for group in group_names
        }
        valid_by_group = {group: values for group, values in valid_by_group.items() if values}
        group_values = np.asarray(
            [np.mean(values) for values in valid_by_group.values()], dtype=np.float64
        )
        valid_sample_count = sum(len(values) for values in valid_by_group.values())
        if len(group_values):
            draws = rng.integers(0, len(group_values), size=(10_000, len(group_values)))
            boot = group_values[draws].mean(axis=1)
            delta = float(group_values.mean())
            ci_low = float(np.quantile(boot, 0.025))
            ci_high = float(np.quantile(boot, 0.975))
        else:
            delta = ci_low = ci_high = None
        summary_rows.append(
            {
                "method": method,
                "reference": reference,
                "metric": metric,
                "source_group_count": len(valid_by_group),
                "total_source_group_count": len(group_names),
                "valid_sample_count": valid_sample_count,
                "missing_sample_count": len(method_map) - valid_sample_count,
                "group_balanced_delta": delta,
                "bootstrap_95ci_low": ci_low,
                "bootstrap_95ci_high": ci_high,
                "bootstrap_seed": BOOTSTRAP_SEED,
                "bootstrap_draws": 10_000 if len(group_values) else 0,
            }
        )
    return paired_rows, summary_rows


def _eval_and_save(
    method: str,
    predictions: dict[str, np.ndarray],
    records: list[dict[str, Any]],
    output_root: Path,
) -> list[dict[str, Any]]:
    rows = []
    lookup = {row["sample_id"]: row for row in records}
    if set(predictions) != set(lookup):
        raise ValueError(f"{method} 预测集合与 development 不一致")
    for sample_id in sorted(predictions):
        record = lookup[sample_id]
        metrics = compute_taskb_metrics(
            predictions[sample_id], record["target"][0], record["lr"][0],
            allow_undefined_epi=True,
        )
        rows.append(
            {"method": method, "sample_id": sample_id,
             "source_group_id": record["source_group_id"],
             "epi_valid": metrics["epi_corr"] is not None, **metrics}
        )
    _append_csv(output_root / "development_per_image_metrics.csv", rows)
    return rows


def _ddim_sample(
    model: ConditionalResidualUNet,
    mean_estimate: torch.Tensor,
    low_resolution: torch.Tensor,
    *,
    alphas_cumprod: torch.Tensor,
    seed: int,
) -> torch.Tensor:
    model.eval()
    generator = torch.Generator(device=mean_estimate.device).manual_seed(seed)
    residual = torch.randn(
        mean_estimate.shape,
        device=mean_estimate.device,
        dtype=mean_estimate.dtype,
        generator=generator,
    )
    indices = torch.linspace(
        DIFFUSION_STEPS - 1, 0, DDIM_STEPS, device=mean_estimate.device
    ).round().long()
    with torch.inference_mode():
        for index, timestep in enumerate(indices):
            time_tensor = timestep.expand(mean_estimate.shape[0])
            predicted_noise = model(residual, mean_estimate, low_resolution, time_tensor)
            alpha_t = alphas_cumprod[timestep]
            if index + 1 < len(indices):
                alpha_previous = alphas_cumprod[indices[index + 1]]
            else:
                alpha_previous = torch.ones((), device=mean_estimate.device)
            predicted_clean = (residual - torch.sqrt(1.0 - alpha_t) * predicted_noise) / torch.sqrt(alpha_t)
            residual = (
                torch.sqrt(alpha_previous) * predicted_clean
                + torch.sqrt(1.0 - alpha_previous) * predicted_noise
            )
    return torch.clamp(mean_estimate + residual, 0.0, 1.0)


class _ExponentialMovingAverage:
    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.model = copy.deepcopy(model).eval()
        self.decay = decay
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        current = model.state_dict()
        for name, average in self.model.state_dict().items():
            if torch.is_floating_point(average):
                average.mul_(self.decay).add_(current[name].detach(), alpha=1.0 - self.decay)
            else:
                average.copy_(current[name])


def _validate_resume_stage(
    source_run: Path,
    *,
    input_manifest_sha256: str,
    patch_size: int,
    allowed_checkpoint_roots: tuple[Path, ...] = (),
) -> dict[str, Any]:
    stage_dir = source_run / "candidate_components" / "centered_residual_diffusion"
    manifest_path = stage_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"源运行缺少扩散阶段清单：{manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "interrupted":
        raise ValueError("源扩散阶段必须已明确标记为意外中断")
    if manifest.get("input_manifest_sha256") != input_manifest_sha256:
        raise ValueError("恢复输入清单哈希与源阶段不一致")
    if manifest.get("patch_lr") != patch_size:
        raise ValueError("恢复 patch size 与源阶段不一致")
    fixed_values = {
        "seed": SEED,
        "epochs": 15,
        "diffusion_steps": DIFFUSION_STEPS,
        "beta_schedule": "linear(1e-4,0.02)",
        "sampling": {"method": "DDIM", "steps": DDIM_STEPS, "eta": 0},
        "ema_decay": 0.999,
    }
    for key, expected in fixed_values.items():
        if manifest.get(key) != expected:
            raise ValueError(f"恢复配置字段不匹配：{key}")
    for key in (
        "confirmation_payload_accessed",
        "test_payload_accessed",
        "optical_payload_loaded",
    ):
        if manifest.get(key) is not False:
            raise ValueError(f"源阶段保护数据标记不合规：{key}")

    last_epoch = manifest.get("last_epoch")
    best_epoch = manifest.get("best_epoch")
    if not isinstance(last_epoch, int) or not 1 <= last_epoch < manifest["epochs"]:
        raise ValueError("源阶段最后 checkpoint 轮次不在可恢复范围")
    if not isinstance(best_epoch, int) or not 1 <= best_epoch <= last_epoch:
        raise ValueError("源阶段最佳 checkpoint 轮次不合法")
    allowed_roots = tuple(
        dict.fromkeys((source_run.resolve(), *(root.resolve() for root in allowed_checkpoint_roots)))
    )

    def checked_checkpoint(path_key: str, hash_key: str, expected_epoch: int) -> Path:
        raw_path = manifest.get(path_key)
        if not raw_path:
            raise ValueError(f"源阶段缺少 {path_key}")
        checkpoint = Path(raw_path).resolve()
        if not any(checkpoint.is_relative_to(root) for root in allowed_roots) or not checkpoint.is_file():
            raise ValueError(f"{path_key} 不在允许的运行目录内或文件缺失")
        expected_sha = manifest.get(hash_key)
        if not expected_sha or sha256_file(checkpoint) != expected_sha:
            raise ValueError(f"{path_key} SHA256 与源清单不匹配")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if (
            payload.get("format") != "phase_e_taskb_checkpoint_v1"
            or payload.get("method") != "centered_residual_conditional_ddpm"
            or payload.get("seed") != SEED
            or payload.get("epoch") != expected_epoch
        ):
            raise ValueError(f"{path_key} checkpoint 元数据与源清单轮次不一致")
        return checkpoint

    resume_checkpoint = checked_checkpoint(
        "last_checkpoint", "last_checkpoint_sha256", last_epoch
    )
    best_checkpoint = checked_checkpoint(
        "best_checkpoint", "best_checkpoint_sha256", best_epoch
    )
    best_score = manifest.get("best_score")
    if not isinstance(best_score, (int, float)) or not math.isfinite(best_score):
        raise ValueError("源阶段最佳开发分数缺失或非有限")
    return {
        "manifest": manifest,
        "resume_epoch": last_epoch,
        "resume_checkpoint": resume_checkpoint,
        "best_epoch": best_epoch,
        "best_checkpoint": best_checkpoint,
        "best_score": float(best_score),
    }


def _load_diffusion_training_state(
    payload: dict[str, Any],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    ema: _ExponentialMovingAverage,
    *,
    expected_epoch: int,
    device: torch.device,
) -> None:
    if payload.get("format") != "phase_e_taskb_checkpoint_v1":
        raise ValueError("扩散 checkpoint 格式不匹配")
    if payload.get("method") != "centered_residual_conditional_ddpm":
        raise ValueError("扩散 checkpoint 方法不匹配")
    if payload.get("seed") != SEED:
        raise ValueError("扩散 checkpoint 随机种子不匹配")
    if payload.get("epoch") != expected_epoch:
        raise ValueError("扩散 checkpoint 轮次与恢复点不匹配")
    rng_keys = (
        "python_rng_state",
        "numpy_rng_state",
        "torch_rng_state",
        "cuda_rng_state_all",
    )
    if any(key not in payload for key in rng_keys):
        raise ValueError("扩散 checkpoint 缺少完整 RNG 状态")
    cuda_states = payload["cuda_rng_state_all"]
    if device.type == "cuda":
        if not torch.cuda.is_available() or len(cuda_states) != torch.cuda.device_count():
            raise ValueError("扩散 checkpoint CUDA RNG 数量与当前设备不匹配")

    model.load_state_dict(payload["model_state_dict"], strict=True)
    ema.model.load_state_dict(payload["ema_state_dict"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    random.setstate(payload["python_rng_state"])
    np.random.set_state(payload["numpy_rng_state"])
    torch.set_rng_state(payload["torch_rng_state"].cpu())
    if device.type == "cuda":
        torch.cuda.set_rng_state_all(cuda_states)


def _load_prediction_map(
    directory: Path,
    sample_ids: list[str],
    *,
    expected_shape: tuple[int, ...] = (512, 512),
) -> dict[str, np.ndarray]:
    expected = set(sample_ids)
    paths = {path.stem: path for path in directory.glob("*.npy")}
    if set(paths) != expected:
        missing = sorted(expected - set(paths))
        extra = sorted(set(paths) - expected)
        raise ValueError(
            f"预测文件集合与固定 split 不一致：缺失 {missing[:5]}，额外 {extra[:5]}"
        )
    predictions = {}
    for sample_id, path in paths.items():
        prediction = np.load(path, allow_pickle=False)
        if (
            prediction.shape != expected_shape
            or prediction.dtype != np.float32
            or not np.isfinite(prediction).all()
            or prediction.min() < 0
            or prediction.max() > 1
        ):
            raise ValueError(f"预测数组契约错误：{sample_id}")
        predictions[sample_id] = prediction
    return predictions


def _predict_frozen_base(
    model: nn.Module,
    records: list[dict[str, Any]],
    device: torch.device,
    destination: Path,
) -> dict[str, np.ndarray]:
    model.eval()
    predictions = {}
    with torch.inference_mode():
        for record in records:
            lr = torch.from_numpy(record["lr"][None]).to(device)
            prediction = model(lr)[0, 0].float().cpu().numpy()
            if prediction.shape != (512, 512) or not np.isfinite(prediction).all():
                raise ValueError(f"冻结 SwinIR 输出契约错误：{record['sample_id']}")
            predictions[record["sample_id"]] = np.clip(prediction, 0, 1).astype(np.float32)
    _save_predictions(destination, predictions)
    return predictions


def train_mean_corrector(
    *,
    train_records: list[dict[str, Any]],
    development_records: list[dict[str, Any]],
    train_base: dict[str, np.ndarray],
    development_base: dict[str, np.ndarray],
    device: torch.device,
    output_dir: Path,
    input_manifest_sha256: str,
    patch_size: int,
    candidate_deadline: float,
) -> tuple[ResidualMeanCorrector, Path, list[dict[str, Any]], dict[str, np.ndarray], dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir()
    model = ResidualMeanCorrector().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.0)
    best_score = -math.inf
    best_checkpoint = None
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats(device)
    manifest = {
        "method": "冻结 SwinIR + 确定性均值残差校正",
        "status": "running",
        "seed": SEED,
        "epochs": 15,
        "patch_lr": patch_size,
        "loss": "L1",
        "input_manifest_sha256": input_manifest_sha256,
        "confirmation_payload_accessed": False,
        "test_payload_accessed": False,
        "optical_payload_loaded": False,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }
    _write_json(output_dir / "manifest.json", manifest)

    for epoch in range(1, 16):
        if time.monotonic() - started >= MAX_STAGE_SECONDS or time.monotonic() >= candidate_deadline:
            manifest["status"] = "time_budget_reached"
            break
        model.train()
        rng = np.random.default_rng(SEED + 10_000 + epoch)
        order = rng.permutation(len(train_records))
        losses = []
        for index in order:
            record = train_records[int(index)]
            base_full = train_base[record["sample_id"]][None]
            lr_patch, target_patch, base_patch = _crop_aligned(
                record, (record["target"], base_full), patch_size=patch_size, rng=rng
            )
            sample_id = record["sample_id"]
            base_tensor = torch.from_numpy(base_patch[None]).to(device)
            lr_tensor = torch.from_numpy(lr_patch[None]).to(device)
            target_tensor = torch.from_numpy(target_patch[None]).to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(lr_tensor, base_tensor)
            loss = F.l1_loss(prediction, target_tensor)
            if not torch.isfinite(loss):
                raise FloatingPointError("均值校正 loss 非有限")
            loss.backward()
            if any(
                parameter.grad is not None and not torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
            ):
                raise FloatingPointError("均值校正梯度非有限")
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        dev_rows, _ = _evaluate_model(
            model, development_records, device, base_predictions=development_base
        )
        score = _save_stage_metrics(output_dir, dev_rows, epoch=epoch)
        checkpoint = _save_epoch_checkpoint(
            checkpoint_dir, epoch=epoch, model=model, optimizer=optimizer,
            method="residual_mean_corrector", extra={"development_group_balanced_psnr_db": score}
        )
        if score > best_score:
            best_score = score
            best_checkpoint = checkpoint
            manifest["best_epoch"] = epoch
            manifest["best_score"] = score
            manifest["best_checkpoint"] = str(checkpoint)
            manifest["best_checkpoint_sha256"] = sha256_file(checkpoint)
        manifest["last_epoch"] = epoch
        manifest["last_checkpoint"] = str(checkpoint)
        manifest["elapsed_seconds"] = time.monotonic() - started
        manifest["peak_cuda_memory_bytes"] = int(torch.cuda.max_memory_allocated(device))
        _append_csv(output_dir / "train_log.csv", [{"epoch": epoch, "train_l1_mean": float(np.mean(losses))}])
        _write_json(output_dir / "manifest.json", manifest)
        print(
            f"均值校正第 {epoch}/15 轮：训练 L1={np.mean(losses):.6f}，"
            f"开发组均值 PSNR={score:.4f} dB；已用 {manifest['elapsed_seconds']:.0f} 秒。",
            flush=True,
        )

    if best_checkpoint is None:
        raise RuntimeError("均值校正没有有效 checkpoint")
    _load_best_model(model, best_checkpoint, device)
    final_rows, final_predictions = _evaluate_model(
        model, development_records, device, base_predictions=development_base
    )
    manifest["status"] = "completed" if manifest.get("last_epoch") == 15 else "time_budget_reached"
    manifest["elapsed_seconds"] = time.monotonic() - started
    _write_stage_manifest(output_dir / "manifest.json", manifest)
    _append_csv(output_dir / "development_best_per_image.csv", final_rows)
    _save_predictions(output_dir / "predictions_best", final_predictions)
    return model, best_checkpoint, final_rows, final_predictions, manifest


def _evaluate_diffusion_draw(
    model: ConditionalResidualUNet,
    records: list[dict[str, Any]],
    mean_predictions: dict[str, np.ndarray],
    device: torch.device,
    alphas_cumprod: torch.Tensor,
    draw_index: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    rows = []
    predictions = {}
    for record in records:
        sample_id = record["sample_id"]
        mean_tensor = torch.from_numpy(mean_predictions[sample_id][None, None]).to(device)
        lr_tensor = torch.from_numpy(record["lr"][None]).to(device)
        seed = _seed_for(f"{sample_id}:generated_draw_{draw_index}", base=SEED + 100)
        prediction = _ddim_sample(
            model, mean_tensor, lr_tensor, alphas_cumprod=alphas_cumprod, seed=seed
        )[0, 0].cpu().numpy().astype(np.float32)
        metrics = compute_taskb_metrics(
            prediction, record["target"][0], record["lr"][0],
            allow_undefined_epi=True,
        )
        rows.append(
            {"sample_id": sample_id, "source_group_id": record["source_group_id"],
             "epi_valid": metrics["epi_corr"] is not None, **metrics}
        )
        predictions[sample_id] = prediction
    return rows, predictions


def train_centered_residual_diffusion(
    *,
    train_records: list[dict[str, Any]],
    development_records: list[dict[str, Any]],
    train_mean: dict[str, np.ndarray],
    development_mean: dict[str, np.ndarray],
    device: torch.device,
    output_dir: Path,
    input_manifest_sha256: str,
    patch_size: int,
    candidate_deadline: float,
    resume_state: dict[str, Any] | None = None,
) -> tuple[ConditionalResidualUNet, Path, list[dict[str, Any]], dict[str, np.ndarray], dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir()
    model = ConditionalResidualUNet().to(device)
    ema = _ExponentialMovingAverage(model, decay=0.999)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.0)
    betas = linear_beta_schedule(steps=DIFFUSION_STEPS, beta_start=1e-4, beta_end=0.02, device=device)
    alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
    best_score = -math.inf
    best_checkpoint = None
    best_rows = None
    first_epoch = 1
    if resume_state is not None:
        payload = torch.load(
            resume_state["resume_checkpoint"], map_location="cpu", weights_only=False
        )
        if not torch.equal(payload.get("betas"), betas.detach().cpu()):
            raise ValueError("恢复 checkpoint 的 beta 日程与冻结配置不一致")
        if not torch.equal(payload.get("alphas_cumprod"), alphas_cumprod.detach().cpu()):
            raise ValueError("恢复 checkpoint 的 alpha 日程与冻结配置不一致")
        _load_diffusion_training_state(
            payload, model, optimizer, ema,
            expected_epoch=resume_state["resume_epoch"], device=device,
        )
        first_epoch = resume_state["resume_epoch"] + 1
        best_score = resume_state["best_score"]
        best_checkpoint = resume_state["best_checkpoint"]
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats(device)
    manifest = {
        "method": "中心残差条件 DDPM epsilon-prediction",
        "status": "running",
        "seed": SEED,
        "epochs": 15,
        "patch_lr": patch_size,
        "diffusion_steps": DIFFUSION_STEPS,
        "beta_schedule": "linear(1e-4,0.02)",
        "sampling": {"method": "DDIM", "steps": DDIM_STEPS, "eta": 0},
        "ema_decay": 0.999,
        "loss": "epsilon MSE",
        "input_manifest_sha256": input_manifest_sha256,
        "confirmation_payload_accessed": False,
        "test_payload_accessed": False,
        "optical_payload_loaded": False,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }
    if resume_state is not None:
        manifest.update(
            {
                "resumed_from_checkpoint": str(resume_state["resume_checkpoint"]),
                "resumed_from_epoch": resume_state["resume_epoch"],
                "inherited_best_epoch": resume_state["best_epoch"],
                "inherited_best_checkpoint": str(resume_state["best_checkpoint"]),
                "inherited_best_score": best_score,
                "last_epoch": resume_state["resume_epoch"],
                "last_checkpoint": str(resume_state["resume_checkpoint"]),
                "last_checkpoint_sha256": sha256_file(resume_state["resume_checkpoint"]),
                "best_epoch": resume_state["best_epoch"],
                "best_checkpoint": str(resume_state["best_checkpoint"]),
                "best_checkpoint_sha256": sha256_file(resume_state["best_checkpoint"]),
                "best_score": best_score,
            }
        )
    _write_json(output_dir / "manifest.json", manifest)

    for epoch in range(first_epoch, 16):
        if time.monotonic() - started >= MAX_STAGE_SECONDS or time.monotonic() >= candidate_deadline:
            manifest["status"] = "time_budget_reached"
            break
        model.train()
        rng = np.random.default_rng(SEED + 20_000 + epoch)
        order = rng.permutation(len(train_records))
        losses = []
        for index in order:
            record = train_records[int(index)]
            sample_id = record["sample_id"]
            lr_patch, target_patch, mean_patch = _crop_aligned(
                record,
                (record["target"], train_mean[sample_id]),
                patch_size=patch_size,
                rng=rng,
            )
            residual = target_patch - mean_patch
            residual_tensor = torch.from_numpy(np.ascontiguousarray(residual[None])).to(device)
            mean_tensor = torch.from_numpy(np.ascontiguousarray(mean_patch[None])).to(device)
            lr_tensor = torch.from_numpy(lr_patch[None]).to(device)
            timestep = torch.randint(0, DIFFUSION_STEPS, (1,), device=device)
            noise = torch.randn_like(residual_tensor)
            alpha = alphas_cumprod[timestep].view(1, 1, 1, 1)
            noisy = torch.sqrt(alpha) * residual_tensor + torch.sqrt(1.0 - alpha) * noise
            optimizer.zero_grad(set_to_none=True)
            predicted_noise = model(noisy, mean_tensor, lr_tensor, timestep)
            loss = F.mse_loss(predicted_noise, noise)
            if not torch.isfinite(loss):
                raise FloatingPointError("扩散 epsilon loss 非有限")
            loss.backward()
            if any(
                parameter.grad is not None and not torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
            ):
                raise FloatingPointError("扩散网络梯度非有限")
            optimizer.step()
            ema.update(model)
            losses.append(float(loss.detach().cpu()))

        checkpoint = _save_epoch_checkpoint(
            checkpoint_dir, epoch=epoch, model=model, optimizer=optimizer,
            method="centered_residual_conditional_ddpm",
            extra={
                "ema_state_dict": ema.model.state_dict(),
                "betas": betas.detach().cpu(),
                "alphas_cumprod": alphas_cumprod.detach().cpu(),
            },
        )
        eval_rows = None
        score = None
        if epoch % 3 == 0 or epoch == 15:
            eval_rows, _ = _evaluate_diffusion_draw(
                ema.model, development_records, development_mean, device, alphas_cumprod
            )
            score = _group_balanced_mean(eval_rows)
            _append_csv(
                output_dir / "development_epoch_summary.csv",
                [{"epoch": epoch, "development_group_balanced_psnr_db": score,
                  "development_image_count": len(eval_rows)}],
            )
            if score > best_score:
                best_score = score
                best_checkpoint = checkpoint
                best_rows = eval_rows
                manifest["best_epoch"] = epoch
                manifest["best_score"] = score
                manifest["best_checkpoint"] = str(checkpoint)
                manifest["best_checkpoint_sha256"] = sha256_file(checkpoint)
        manifest["last_epoch"] = epoch
        manifest["last_checkpoint"] = str(checkpoint)
        manifest["last_checkpoint_sha256"] = sha256_file(checkpoint)
        manifest["elapsed_seconds"] = time.monotonic() - started
        manifest["peak_cuda_memory_bytes"] = int(torch.cuda.max_memory_allocated(device))
        _append_csv(output_dir / "train_log.csv", [{"epoch": epoch, "epsilon_mse_mean": float(np.mean(losses))}])
        _write_json(output_dir / "manifest.json", manifest)
        score_text = "本轮未做生成评价" if score is None else f"开发组均值 PSNR={score:.4f} dB"
        print(
            f"中心残差扩散第 {epoch}/15 轮：epsilon MSE={np.mean(losses):.6f}，"
            f"{score_text}；已用 {manifest['elapsed_seconds']:.0f} 秒。",
            flush=True,
        )
        if manifest["elapsed_seconds"] >= MAX_STAGE_SECONDS:
            manifest["status"] = "time_budget_reached"
            break

    if best_checkpoint is None:
        raise RuntimeError("中心残差扩散没有可评价的 checkpoint")
    best_payload = torch.load(best_checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(best_payload["model_state_dict"], strict=True)
    ema.model.load_state_dict(best_payload["ema_state_dict"], strict=True)
    final_rows, final_predictions = _evaluate_diffusion_draw(
        ema.model, development_records, development_mean, device, alphas_cumprod
    )
    manifest["status"] = "completed" if manifest.get("last_epoch") == 15 else "time_budget_reached"
    manifest["elapsed_seconds"] = time.monotonic() - started
    _write_stage_manifest(output_dir / "manifest.json", manifest)
    _append_csv(output_dir / "development_best_per_image.csv", final_rows)
    _save_predictions(output_dir / "predictions_single_draw", final_predictions)
    return ema.model, best_checkpoint, final_rows, final_predictions, manifest


def _load_model_from_checkpoint(model: nn.Module, checkpoint_path: Path, device: torch.device) -> nn.Module:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = payload.get("ema_state_dict", payload["model_state_dict"])
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def _write_final_results(
    output_root: Path,
    records: list[dict[str, Any]],
    method_predictions: dict[str, dict[str, np.ndarray]],
) -> dict[str, Any]:
    all_rows = []
    summaries = []
    group_rows = []
    for method, predictions in method_predictions.items():
        rows = _eval_and_save(method, predictions, records, output_root)
        all_rows.extend(rows)
        groups: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            groups.setdefault(row["source_group_id"], []).append(row)
        for group_id, values in sorted(groups.items()):
            group_rows.append(
                {"method": method, "source_group_id": group_id,
                 "sample_count": len(values),
                 "epi_valid_sample_count": sum(row["epi_corr"] is not None for row in values),
                 "epi_missing_sample_count": sum(row["epi_corr"] is None for row in values),
                 **{metric: _group_balanced_mean(values, metric) for metric in METRIC_NAMES}}
            )
        summaries.append(
            {"method": method, "source_group_count": len(groups),
             "sample_count": len(rows),
             "epi_valid_sample_count": sum(row["epi_corr"] is not None for row in rows),
             "epi_missing_sample_count": sum(row["epi_corr"] is None for row in rows),
             "epi_coverage": float(np.mean([row["epi_corr"] is not None for row in rows])),
             **{metric: _group_balanced_mean(rows, metric) for metric in METRIC_NAMES}}
        )
    _append_csv(output_root / "development_group_metrics.csv", group_rows)
    _append_csv(output_root / "development_method_summary.csv", summaries)

    reference_row = max(
        (row for row in summaries if row["method"] in {"EDSR", "SwinIR"}),
        key=lambda row: row["psnr_db"],
    )
    reference_method = reference_row["method"]
    paired = []
    bootstrap_rows = []
    for method in ("均值校正点估计", "DDIM_单次", "DDIM_四次抽样像素均值"):
        method_rows = [row for row in all_rows if row["method"] == method]
        baseline_rows = [row for row in all_rows if row["method"] == reference_method]
        paired_method, summary = _group_bootstrap(
            method_rows, baseline_rows, method=method, reference=reference_method
        )
        paired.extend(paired_method)
        bootstrap_rows.extend(summary)
    _append_csv(output_root / "development_paired_per_image.csv", paired)
    _append_csv(output_root / "development_group_bootstrap.csv", bootstrap_rows)

    summary_map = {row["method"]: row for row in summaries}
    candidate = summary_map["DDIM_单次"]
    psnr_boot = next(
        row for row in bootstrap_rows if row["method"] == "DDIM_单次" and row["metric"] == "psnr_db"
    )
    candidate_epi_complete = (
        candidate["epi_valid_sample_count"] == candidate["sample_count"]
        and reference_row["epi_valid_sample_count"] == reference_row["sample_count"]
    )
    meets = (
        candidate["psnr_db"] - reference_row["psnr_db"] >= 0.05
        and psnr_boot["bootstrap_95ci_low"] > 0
        and candidate["ssim"] >= reference_row["ssim"]
        and candidate_epi_complete
        and candidate["epi_corr"] is not None
        and reference_row["epi_corr"] is not None
        and candidate["epi_corr"] >= reference_row["epi_corr"]
        and abs(candidate["bright_mean_ratio"] - 1.0)
        <= abs(reference_row["bright_mean_ratio"] - 1.0)
        and abs(candidate["background_mean_ratio"] - 1.0)
        <= abs(reference_row["background_mean_ratio"] - 1.0)
        and candidate["reprojection_mae"] <= reference_row["reprojection_mae"]
    )
    return {
        "strongest_learning_baseline": reference_method,
        "strongest_baseline_summary": reference_row,
        "methods": summaries,
        "candidate_vs_strongest_psnr_delta": candidate["psnr_db"] - reference_row["psnr_db"],
        "candidate_psnr_bootstrap_95ci": [
            psnr_boot["bootstrap_95ci_low"], psnr_boot["bootstrap_95ci_high"]
        ],
        "candidate_epi_valid_sample_count": candidate["epi_valid_sample_count"],
        "candidate_epi_missing_sample_count": candidate["epi_missing_sample_count"],
        "candidate_epi_complete": candidate_epi_complete,
        "predeclared_continue_gate_met": meets,
        "decision_scope": "单 seed development pilot；探索性，不是独立确认",
    }


def run_resume_pilot(
    input_manifest_path: Path,
    source_run: Path,
    output_root: Path,
    *,
    patch_size: int = DEFAULT_PATCH,
) -> dict[str, Any]:
    if patch_size not in {48, 64}:
        raise ValueError("LR crop 仅允许冻结默认 64 或真实 OOM 后的 48 回退")
    if output_root.exists():
        raise FileExistsError(f"恢复输出目录已存在，拒绝覆盖：{output_root}")
    if not torch.cuda.is_available():
        raise RuntimeError("Phase E 恢复要求 CUDA，不回退 CPU")

    source_run = source_run.resolve()
    input_manifest_path = input_manifest_path.resolve()
    records, input_manifest = _load_records(input_manifest_path)
    input_manifest_sha = sha256_file(input_manifest_path)
    protected_flags = (
        "confirmation_payload_accessed",
        "test_payload_accessed",
        "optical_payload_loaded",
    )
    source_environment_path = source_run / "environment.json"
    source_environment = json.loads(source_environment_path.read_text(encoding="utf-8-sig"))
    if source_environment.get("input_manifest_sha256") != input_manifest_sha:
        raise ValueError("源运行环境记录的输入清单哈希不匹配")
    if any(source_environment.get(key) is not False for key in protected_flags):
        raise ValueError("源运行环境记录的保护数据标记不合规")

    artifact_run = source_run
    if not (artifact_run / "EDSR" / "manifest.json").is_file():
        parent_run_value = source_environment.get("parent_run")
        if not parent_run_value:
            raise ValueError("恢复运行缺少可追溯的基线产物父运行目录")
        artifact_run = Path(parent_run_value).resolve()
        if artifact_run.parent != source_run.parent:
            raise ValueError("基线产物父运行必须与恢复运行处于同一 pilot 根目录")
    artifact_environment_path = artifact_run / "environment.json"
    artifact_environment = json.loads(artifact_environment_path.read_text(encoding="utf-8-sig"))
    if artifact_environment.get("input_manifest_sha256") != input_manifest_sha:
        raise ValueError("基线产物父运行的输入清单哈希不匹配")
    if any(artifact_environment.get(key) is not False for key in protected_flags):
        raise ValueError("基线产物父运行的保护数据标记不合规")
    resume_state = _validate_resume_stage(
        source_run,
        input_manifest_sha256=input_manifest_sha,
        patch_size=patch_size,
        allowed_checkpoint_roots=(artifact_run,),
    )

    completed_manifests = {}
    for stage_name, relative in (
        ("EDSR", "EDSR/manifest.json"),
        ("SwinIR", "SwinIR/manifest.json"),
        ("mean_corrector", "candidate_components/mean_corrector/manifest.json"),
    ):
        manifest = json.loads((artifact_run / relative).read_text(encoding="utf-8"))
        if (
            manifest.get("status") != "completed"
            or manifest.get("input_manifest_sha256") != input_manifest_sha
            or manifest.get("seed") != SEED
            or manifest.get("patch_lr") != patch_size
            or any(manifest.get(key) is not False for key in protected_flags)
        ):
            raise ValueError(f"源运行阶段不满足恢复条件：{stage_name}")
        checkpoint = Path(manifest.get("best_checkpoint", "")).resolve()
        if (
            not checkpoint.is_relative_to(artifact_run)
            or not checkpoint.is_file()
            or sha256_file(checkpoint) != manifest.get("best_checkpoint_sha256")
        ):
            raise ValueError(f"源运行阶段最佳 checkpoint 校验失败：{stage_name}")
        completed_manifests[stage_name] = manifest

    development_ids = [record["sample_id"] for record in records["development"]]
    train_ids = [record["sample_id"] for record in records["train"]]
    train_mean = _load_prediction_map(
        artifact_run / "candidate_components" / "mean_predictions" / "train",
        train_ids,
        expected_shape=(1, 512, 512),
    )
    development_mean = _load_prediction_map(
        artifact_run / "candidate_components" / "mean_predictions" / "development",
        development_ids,
    )
    bicubic_predictions = _load_prediction_map(
        artifact_run / "Bicubic" / "predictions", development_ids
    )
    edsr_predictions = _load_prediction_map(
        artifact_run / "EDSR" / "predictions_best", development_ids
    )
    swinir_predictions = _load_prediction_map(
        artifact_run / "SwinIR" / "predictions_best", development_ids
    )

    output_root.mkdir(parents=True, exist_ok=False)
    _set_global_seed(SEED)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    disk_before = shutil.disk_usage(output_root.parent)
    gpu_before = _gpu_snapshot(device)
    parent_budget = source_environment.get("budget_accounting", {})
    candidate_budget_before_resume = parent_budget.get("candidate_budget_remaining_seconds")
    total_budget_before_resume = parent_budget.get("total_budget_remaining_seconds")
    source_elapsed = float(source_environment.get("elapsed_seconds", 0.0))
    if not isinstance(candidate_budget_before_resume, (int, float)) or not isinstance(
        total_budget_before_resume, (int, float)
    ):
        raise ValueError("恢复源运行缺少可核验的候选/总 GPU 预算账目")
    candidate_elapsed_before = MAX_STAGE_SECONDS - float(candidate_budget_before_resume) + source_elapsed
    total_elapsed_before = MAX_TOTAL_SECONDS - float(total_budget_before_resume) + source_elapsed
    mean_elapsed = float(completed_manifests["mean_corrector"]["elapsed_seconds"])
    diffusion_elapsed = float(resume_state["manifest"].get("elapsed_seconds", 0.0))
    edsr_elapsed = float(completed_manifests["EDSR"]["elapsed_seconds"])
    swinir_elapsed = float(completed_manifests["SwinIR"]["elapsed_seconds"])
    base_precompute_reserve = float(parent_budget.get("swinir_base_precompute_reserved_seconds", 0.0))
    interruption_gap_reserve = float(parent_budget.get("interruption_gap_reserved_seconds", 0.0))
    candidate_remaining = MAX_STAGE_SECONDS - candidate_elapsed_before
    final_evaluation_reserve = 60 * 60
    if candidate_remaining <= final_evaluation_reserve:
        raise TimeoutError("候选剩余预算不足以同时完成训练和固定生成评价")
    candidate_total_deadline = started + candidate_remaining
    candidate_training_deadline = candidate_total_deadline - final_evaluation_reserve
    if total_elapsed_before >= MAX_TOTAL_SECONDS:
        raise TimeoutError("源运行已达到本轮 12 小时 GPU 总预算")
    total_deadline = started + MAX_TOTAL_SECONDS - total_elapsed_before

    profile = {
        "status": "running",
        "run_type": "中心残差扩散断点恢复",
        "parent_run": str(source_run),
        "artifact_source_run": str(artifact_run),
        "resume_checkpoint": str(resume_state["resume_checkpoint"]),
        "resume_epoch": resume_state["resume_epoch"],
        "inherited_best_checkpoint": str(resume_state["best_checkpoint"]),
        "inherited_best_epoch": resume_state["best_epoch"],
        "inherited_best_score": resume_state["best_score"],
        "seed": SEED,
        "patch_lr": patch_size,
        "input_manifest_sha256": input_manifest_sha,
        "split_manifest_sha256": input_manifest.get("split_manifest_sha256"),
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "torch": torch.__version__,
        "torchvision": __import__("torchvision").__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu_before": gpu_before,
        "disk_free_before_bytes": disk_before.free,
        "git": _git_state(),
        "budget_accounting": {
            "logged_edsr_seconds": edsr_elapsed,
            "logged_swinir_seconds": swinir_elapsed,
            "logged_mean_corrector_seconds": mean_elapsed,
            "logged_diffusion_before_interruption_seconds": diffusion_elapsed,
            "candidate_budget_remaining_before_this_resume_seconds": float(candidate_budget_before_resume),
            "total_budget_remaining_before_this_resume_seconds": float(total_budget_before_resume),
            "source_run_elapsed_seconds": source_elapsed,
            "swinir_base_precompute_reserved_seconds": base_precompute_reserve,
            "interruption_gap_reserved_seconds": interruption_gap_reserve,
            "smoke_and_setup_reserved_seconds": 300.0,
            "candidate_budget_remaining_seconds": candidate_remaining,
            "final_evaluation_reserved_seconds": final_evaluation_reserve,
            "total_budget_remaining_seconds": MAX_TOTAL_SECONDS - total_elapsed_before,
        },
        "confirmation_payload_accessed": False,
        "test_payload_accessed": False,
        "optical_payload_loaded": False,
    }
    _write_json(output_root / "environment.json", profile)
    _write_json(
        output_root / "manifest.json",
        {
            "method": "中心残差条件 DDPM epsilon-prediction",
            "status": "running",
            "parent_run": str(source_run),
            "artifact_source_run": str(artifact_run),
            "input_manifest_sha256": input_manifest_sha,
            "seed": SEED,
            "patch_lr": patch_size,
            "resume_epoch": resume_state["resume_epoch"],
            "resume_checkpoint_sha256": sha256_file(resume_state["resume_checkpoint"]),
            "inherited_best_epoch": resume_state["best_epoch"],
            "inherited_best_checkpoint_sha256": sha256_file(resume_state["best_checkpoint"]),
            "confirmation_payload_accessed": False,
            "test_payload_accessed": False,
            "optical_payload_loaded": False,
        },
    )
    (output_root / "命令.txt").write_text(
        "恢复命令：\n"
        + " ".join([sys.executable, *sys.argv])
        + "\n",
        encoding="utf-8",
    )

    try:
        diffusion, diffusion_checkpoint, diffusion_rows, diffusion_single, diffusion_manifest = (
            train_centered_residual_diffusion(
                train_records=records["train"],
                development_records=records["development"],
                train_mean=train_mean,
                development_mean=development_mean,
                device=device,
                output_dir=output_root / "candidate_components" / "centered_residual_diffusion",
                input_manifest_sha256=input_manifest_sha,
                patch_size=patch_size,
                candidate_deadline=candidate_training_deadline,
                resume_state=resume_state,
            )
        )
        beta_schedule = linear_beta_schedule(
            steps=DIFFUSION_STEPS, beta_start=1e-4, beta_end=0.02, device=device
        )
        alphas_cumprod = torch.cumprod(1.0 - beta_schedule, dim=0)
        draw_predictions = []
        for draw_index in range(4):
            if time.monotonic() >= candidate_total_deadline:
                raise TimeoutError("中心残差扩散恢复达到 4 小时候选预算，未完成四次抽样")
            if draw_index == 0:
                predictions = diffusion_single
            else:
                _, predictions = _evaluate_diffusion_draw(
                    diffusion, records["development"], development_mean, device,
                    alphas_cumprod, draw_index=draw_index,
                )
            _save_predictions(
                output_root / "DDIM_四次抽样" / f"draw_{draw_index}", predictions
            )
            draw_predictions.append(predictions)
        mean4 = {
            sample_id: np.mean(
                [draw[sample_id] for draw in draw_predictions], axis=0, dtype=np.float64
            ).astype(np.float32)
            for sample_id in draw_predictions[0]
        }
        _save_predictions(output_root / "均值校正点估计" / "predictions", development_mean)
        _save_predictions(output_root / "DDIM_单次" / "predictions", diffusion_single)
        _save_predictions(output_root / "DDIM_四次抽样像素均值" / "predictions", mean4)
        method_predictions = {
            "Bicubic": bicubic_predictions,
            "EDSR": edsr_predictions,
            "SwinIR": swinir_predictions,
            "均值校正点估计": development_mean,
            "DDIM_单次": diffusion_single,
            "DDIM_四次抽样像素均值": mean4,
        }
        for draw_index, predictions in enumerate(draw_predictions):
            method_predictions[f"DDIM_draw_{draw_index}"] = predictions
        if time.monotonic() >= candidate_total_deadline or time.monotonic() >= total_deadline:
            raise TimeoutError("最终结果统计超过冻结的候选或总 GPU 预算")
        final_summary = _write_final_results(
            output_root, records["development"], method_predictions
        )
        elapsed = time.monotonic() - started
        profile.update(
            {
                "status": "completed",
                "elapsed_seconds": elapsed,
                "candidate_elapsed_budget_charge_seconds": candidate_elapsed_before + elapsed,
                "gpu_after": _gpu_snapshot(device),
                "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
                "disk_free_after_bytes": shutil.disk_usage(output_root.parent).free,
                "stage_manifests": {"centered_residual_diffusion": diffusion_manifest},
                "checkpoints": {
                    "EDSR": completed_manifests["EDSR"]["best_checkpoint"],
                    "SwinIR": completed_manifests["SwinIR"]["best_checkpoint"],
                    "mean_corrector": str(completed_manifests["mean_corrector"].get("best_checkpoint")),
                    "centered_residual_diffusion": str(diffusion_checkpoint),
                },
                "predeclared_decision": final_summary,
            }
        )
        _write_json(output_root / "manifest.json", {**profile, "method": "Phase E Task B 中心残差扩散恢复"})
        _write_json(output_root / "environment.json", profile)
        _write_json(output_root / "研究决策复核.json", final_summary)
        print(
            "扩散恢复试验完成："
            f"最强学习基线={final_summary['strongest_learning_baseline']}；"
            f"DDIM 单次 ΔPSNR={final_summary['candidate_vs_strongest_psnr_delta']:.4f} dB；"
            f"冻结继续门槛={'通过' if final_summary['predeclared_continue_gate_met'] else '未通过'}。",
            flush=True,
        )
        return final_summary
    except BaseException as error:
        profile.update(
            {
                "status": "failed",
                "elapsed_seconds": time.monotonic() - started,
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "cuda_oom": isinstance(error, torch.cuda.OutOfMemoryError),
                "gpu_after": _gpu_snapshot(device),
                "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
                "disk_free_after_bytes": shutil.disk_usage(output_root.parent).free,
            }
        )
        _write_json(output_root / "失败记录.json", profile)
        _write_json(output_root / "environment.json", profile)
        print(f"扩散恢复试验失败，已保存诊断：{type(error).__name__}；{error}", flush=True)
        raise


def run_pilot(
    input_manifest_path: Path,
    output_root: Path,
    *,
    patch_size: int = DEFAULT_PATCH,
) -> dict[str, Any]:
    if patch_size not in {48, 64}:
        raise ValueError("LR crop 仅允许冻结默认 64 或真实 OOM 后的 48 回退")
    if output_root.exists():
        raise FileExistsError(f"输出目录已存在，拒绝覆盖：{output_root}")
    if not torch.cuda.is_available():
        raise RuntimeError("Phase E Pilot 要求 CUDA，不回退 CPU")
    output_root.mkdir(parents=True, exist_ok=False)
    _set_global_seed(SEED)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    records, input_manifest = _load_records(input_manifest_path)
    input_manifest_sha = sha256_file(input_manifest_path)
    disk_before = shutil.disk_usage(output_root.parent)
    gpu_before = _gpu_snapshot(device)
    total_started = time.monotonic()
    profile = {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "torch": torch.__version__,
        "torchvision": __import__("torchvision").__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu_before": gpu_before,
        "disk_free_before_bytes": disk_before.free,
        "input_manifest_sha256": input_manifest_sha,
        "split_manifest_sha256": input_manifest.get("split_manifest_sha256"),
        "confirmation_payload_accessed": False,
        "test_payload_accessed": False,
        "optical_payload_loaded": False,
        "git": _git_state(),
    }
    _write_json(output_root / "environment.json", profile)
    (output_root / "命令.txt").write_text(
        "训练命令：\n" + " ".join([sys.executable, *sys.argv]) + "\n",
        encoding="utf-8",
    )

    try:
        profile["gpu_smoke"] = _gpu_smoke(
            records["train"][0], device, output_root / "GPU_smoke"
        )
        _write_json(output_root / "environment.json", profile)
        _assert_total_budget(total_started, "EDSR")
        edsr_class = load_published_edsr_class()
        edsr = build_edsr_sar(edsr_class)
        edsr, edsr_checkpoint, edsr_rows, edsr_predictions, edsr_manifest = train_supervised_baseline(
            method="EDSR SAR 任务适配",
            model=edsr,
            train_records=records["train"],
            development_records=records["development"],
            device=device,
            output_dir=output_root / "EDSR",
            input_manifest_sha256=input_manifest_sha,
            patch_size=patch_size,
        )

        _assert_total_budget(total_started, "SwinIR")
        swinir_class = swinir.load_swinir_class()
        swinir_model = swinir.build_swinir(swinir_class)
        swinir_model, swinir_checkpoint, swinir_rows, swinir_predictions, swinir_manifest = train_supervised_baseline(
            method="SwinIR SAR 任务适配",
            model=swinir_model,
            train_records=records["train"],
            development_records=records["development"],
            device=device,
            output_dir=output_root / "SwinIR",
            input_manifest_sha256=input_manifest_sha,
            patch_size=patch_size,
        )

        bicubic_predictions = {}
        for record in records["development"]:
            bicubic = cv2.resize(
                record["lr"][0], (512, 512), interpolation=cv2.INTER_CUBIC
            )
            bicubic_predictions[record["sample_id"]] = np.clip(bicubic, 0, 1).astype(np.float32)
        _save_predictions(output_root / "Bicubic" / "predictions", bicubic_predictions)

        _assert_total_budget(total_started, "生成候选")
        candidate_started = time.monotonic()
        candidate_train_deadline = candidate_started + 3.5 * 60 * 60
        base_root = output_root / "candidate_components" / "SwinIR_base"
        base_root.mkdir(parents=True, exist_ok=False)
        train_base = _predict_frozen_base(
            swinir_model, records["train"], device, base_root / "train"
        )
        development_base = _predict_frozen_base(
            swinir_model, records["development"], device, base_root / "development"
        )
        corrector, corrector_checkpoint, mean_rows, mean_predictions, mean_manifest = train_mean_corrector(
            train_records=records["train"],
            development_records=records["development"],
            train_base=train_base,
            development_base=development_base,
            device=device,
            output_dir=output_root / "candidate_components" / "mean_corrector",
            input_manifest_sha256=input_manifest_sha,
            patch_size=patch_size,
            candidate_deadline=candidate_train_deadline,
        )
        train_mean = {}
        for record in records["train"]:
            lr_tensor = torch.from_numpy(record["lr"][None]).to(device)
            base_tensor = torch.from_numpy(train_base[record["sample_id"]][None, None]).to(device)
            with torch.inference_mode():
                train_mean[record["sample_id"]] = corrector(lr_tensor, base_tensor)[0].cpu().numpy()
        dev_mean = mean_predictions
        mean_root = output_root / "candidate_components" / "mean_predictions"
        mean_root.mkdir(parents=True, exist_ok=False)
        for role, prediction_map in (("train", train_mean), ("development", dev_mean)):
            _save_predictions(mean_root / role, prediction_map)

        diffusion, diffusion_checkpoint, diffusion_rows, diffusion_single, diffusion_manifest = train_centered_residual_diffusion(
            train_records=records["train"],
            development_records=records["development"],
            train_mean=train_mean,
            development_mean=dev_mean,
            device=device,
            output_dir=output_root / "candidate_components" / "centered_residual_diffusion",
            input_manifest_sha256=input_manifest_sha,
            patch_size=patch_size,
            candidate_deadline=candidate_train_deadline,
        )
        beta_schedule = linear_beta_schedule(
            steps=DIFFUSION_STEPS, beta_start=1e-4, beta_end=0.02, device=device
        )
        alphas_cumprod = torch.cumprod(1.0 - beta_schedule, dim=0)

        draw_predictions = []
        for draw_index in range(4):
            if time.monotonic() - candidate_started >= MAX_STAGE_SECONDS:
                raise TimeoutError("生成候选达到 4 小时预算，未完成四次抽样评价")
            if draw_index == 0:
                predictions = diffusion_single
            else:
                _, predictions = _evaluate_diffusion_draw(
                    diffusion, records["development"], dev_mean, device,
                    alphas_cumprod, draw_index=draw_index,
                )
            draw_dir = output_root / "DDIM_四次抽样" / f"draw_{draw_index}"
            _save_predictions(draw_dir, predictions)
            draw_predictions.append(predictions)
        mean4 = {
            sample_id: np.mean(
                [draw[sample_id] for draw in draw_predictions], axis=0, dtype=np.float64
            ).astype(np.float32)
            for sample_id in draw_predictions[0]
        }
        _save_predictions(output_root / "均值校正点估计" / "predictions", mean_predictions)
        _save_predictions(output_root / "DDIM_单次" / "predictions", diffusion_single)
        _save_predictions(output_root / "DDIM_四次抽样像素均值" / "predictions", mean4)

        method_predictions = {
            "Bicubic": bicubic_predictions,
            "EDSR": edsr_predictions,
            "SwinIR": swinir_predictions,
            "均值校正点估计": mean_predictions,
            "DDIM_单次": diffusion_single,
            "DDIM_四次抽样像素均值": mean4,
        }
        for draw_index, predictions in enumerate(draw_predictions):
            method_predictions[f"DDIM_draw_{draw_index}"] = predictions
        if time.monotonic() - candidate_started >= MAX_STAGE_SECONDS:
            raise TimeoutError("生成候选达到 4 小时预算，停止后续评价")
        _assert_total_budget(total_started, "结果复核")
        final_summary = _write_final_results(
            output_root, records["development"], method_predictions
        )
        profile.update(
            {
                "status": "completed",
                "elapsed_seconds": time.monotonic() - total_started,
                "candidate_elapsed_seconds": time.monotonic() - candidate_started,
                "gpu_after": _gpu_snapshot(device),
                "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
                "disk_free_after_bytes": shutil.disk_usage(output_root.parent).free,
                "stage_manifests": {
                    "EDSR": edsr_manifest,
                    "SwinIR": swinir_manifest,
                    "mean_corrector": mean_manifest,
                    "centered_residual_diffusion": diffusion_manifest,
                },
                "checkpoints": {
                    "EDSR": str(edsr_checkpoint),
                    "SwinIR": str(swinir_checkpoint),
                    "mean_corrector": str(corrector_checkpoint),
                    "centered_residual_diffusion": str(diffusion_checkpoint),
                },
                "predeclared_decision": final_summary,
            }
        )
        _write_json(output_root / "environment.json", profile)
        _write_json(output_root / "研究决策复核.json", final_summary)
        print(
            "试验完成："
            f"最强学习基线={final_summary['strongest_learning_baseline']}；"
            f"DDIM 单次 ΔPSNR={final_summary['candidate_vs_strongest_psnr_delta']:.4f} dB；"
            f"冻结继续门槛={'通过' if final_summary['predeclared_continue_gate_met'] else '未通过'}。",
            flush=True,
        )
        return final_summary
    except BaseException as error:
        profile.update(
            {
                "status": "failed",
                "elapsed_seconds": time.monotonic() - total_started,
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "cuda_oom": isinstance(error, torch.cuda.OutOfMemoryError),
                "gpu_after": _gpu_snapshot(device),
                "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
                "disk_free_after_bytes": shutil.disk_usage(output_root.parent).free,
            }
        )
        _write_json(output_root / "失败记录.json", profile)
        print(f"试验失败，已保存诊断：{type(error).__name__}；{error}", flush=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="运行 Phase E SAR 单模态 Task B 试验")
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume-from-run", type=Path)
    parser.add_argument("--lr-patch-size", type=int, choices=(48, 64), default=DEFAULT_PATCH)
    args = parser.parse_args()
    if args.resume_from_run is None:
        run_pilot(args.input_manifest, args.output_dir, patch_size=args.lr_patch_size)
    else:
        run_resume_pilot(
            args.input_manifest,
            args.resume_from_run,
            args.output_dir,
            patch_size=args.lr_patch_size,
        )


if __name__ == "__main__":
    main()
