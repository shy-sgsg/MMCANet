#!/usr/bin/env python3
"""执行 C1 v2 train/development 上的 C3-A SAR-only 流匹配筛选。"""
from __future__ import annotations

import csv
import argparse
import hashlib
import json
import math
import os
import platform
import random
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
METRIC_ROOT = REPOSITORY_ROOT / "Early-FusionCascadeJDSR"
if str(METRIC_ROOT) not in sys.path:
    sys.path.insert(0, str(METRIC_ROOT))

from eval_metrics import ssim as project_ssim
from MMCANetV2.phase_c2_data import C2FixedDataset, sha256_file
from MMCANetV2.phase_c2_runner import _load_c2_model_class
from MMCANetV2.phase_c3a_flow import (
    SarConditionalFlow,
    SarTargetOnlyDataset,
    make_flow_training_pair,
    sample_heun,
    upsample_sar_condition,
)
from scripts.training_resume import save_checkpoint_atomic


PROTOCOL_PATH = REPOSITORY_ROOT / "protocols/mmcanet_phase_c3a/sar_only_flow_c1_v2_20261105_v1.json"
EXPECTED_PROTOCOL_SHA256 = "147c361b3e91bc4cd8d76ce5c464ad9d2137f2d06b38f31a0bc7c7a7e93101a7"
OUTPUT_PATH = REPOSITORY_ROOT / "experiments/phase_c3a_sar_only_flow_c1_v2_20261105_v1"
BASELINE_PATH = REPOSITORY_ROOT / "experiments/phase_c2_c1_v2_20261031_v1/matrix/runs/x2_light__sar_only"
SOURCE_MANIFEST = REPOSITORY_ROOT / "protocols/mmcanet_phase_c1/v2/source_group_split_20261028_v1/sample_manifest.csv"
INPUT_ROOT = REPOSITORY_ROOT / "data/derived/c2_fixed_inputs_c1_v2_20261031_v1/x2_light"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_write(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _csv_write(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _set_determinism(seed: int) -> tuple[torch.Generator, torch.Generator]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    loader_generator = torch.Generator(device="cpu")
    loader_generator.manual_seed(seed + 1)
    flow_generator = torch.Generator(device="cuda")
    flow_generator.manual_seed(seed + 2)
    return loader_generator, flow_generator


def _group_balanced(rows: list[dict[str, Any]], metric: str) -> float:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        group = str(row["来源组 ID"])
        value = float(row[metric])
        if not math.isfinite(value):
            raise FloatingPointError(f"开发指标 {metric} 出现非有限值")
        grouped.setdefault(group, []).append(value)
    if not grouped:
        raise ValueError("开发指标没有样本")
    return float(np.mean([np.mean(values) for values in grouped.values()]))


def _paired_group_bootstrap(
    rows: list[dict[str, Any]], *, metric: str, seed: int, repeats: int = 10000
) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        grouped.setdefault(str(row["来源组 ID"]), []).append(float(row[metric]))
    group_values = np.asarray([np.mean(grouped[key]) for key in sorted(grouped)], dtype=np.float64)
    if len(group_values) < 2 or not np.isfinite(group_values).all():
        raise ValueError("来源组配对 bootstrap 至少需要两个有限来源组")
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(group_values), size=(repeats, len(group_values)))
    samples = group_values[indices].mean(axis=1)
    return {
        "mean": float(group_values.mean()),
        "lower_95": float(np.quantile(samples, 0.025)),
        "upper_95": float(np.quantile(samples, 0.975)),
        "source_group_count": int(len(group_values)),
        "bootstrap_repeats": repeats,
    }


def _noise_for_sample(sample_id: str, *, seed: int, device: torch.device) -> torch.Tensor:
    digest = hashlib.sha256(f"{seed}\0{sample_id}".encode("utf-8")).digest()
    local_seed = int.from_bytes(digest[:8], "big") % (2**63 - 1)
    generator = torch.Generator(device=device)
    generator.manual_seed(local_seed)
    return torch.randn((1, 1, 256, 256), generator=generator, device=device)


def _batch_noise(sample_ids: list[str], *, seed: int, device: torch.device) -> torch.Tensor:
    return torch.cat([_noise_for_sample(sample_id, seed=seed, device=device) for sample_id in sample_ids])


def _metric_rows(
    prediction: torch.Tensor,
    raw_prediction: torch.Tensor,
    target: torch.Tensor,
    batch: dict[str, Any],
    *,
    method_name: str,
) -> list[dict[str, Any]]:
    clipped = prediction.clamp(0.0, 1.0)
    output_rows: list[dict[str, Any]] = []
    names = list(batch["name"])
    groups = list(batch["source_group_id"])
    patches = list(batch["patch_id"])
    for index, (sample_id, group_id, patch_id) in enumerate(zip(names, groups, patches)):
        pred = clipped[index, 0].detach().cpu().numpy().astype(np.float64, copy=False)
        truth = target[index, 0].detach().cpu().numpy().astype(np.float64, copy=False)
        raw = raw_prediction[index, 0]
        mse = float(np.mean((pred - truth) ** 2))
        if not math.isfinite(mse) or mse <= 0:
            raise FloatingPointError(f"{method_name} 的 PSNR 均方误差无效：{sample_id}")
        output_rows.append(
            {
                "样本 ID": str(sample_id),
                "来源组 ID": str(group_id),
                "patch ID": str(patch_id),
                "PSNR": float(10.0 * math.log10(1.0 / mse)),
                "SSIM": float(project_ssim(pred, truth, data_range=1.0)),
                "最终裁剪比例": float(((raw < 0.0) | (raw > 1.0)).float().mean().item()),
            }
        )
    return output_rows


@torch.inference_mode()
def _evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    method_name: str,
    run_seed: int,
    sampler_steps: int | None,
    noise_sigma: float,
) -> list[dict[str, Any]]:
    model.eval()
    rows: list[dict[str, Any]] = []
    for batch in loader:
        lr = batch["lr"].to(device, non_blocking=False)
        target = batch["target"].to(device, non_blocking=False)
        condition = upsample_sar_condition(lr, size=(256, 256))
        if sampler_steps is None:
            raw_prediction = model(lr)
        else:
            noise = _batch_noise(list(batch["name"]), seed=run_seed, device=device)
            raw_prediction = sample_heun(
                model,
                condition,
                noise,
                steps=sampler_steps,
                noise_sigma=noise_sigma,
            )
        if raw_prediction.shape != target.shape or not torch.isfinite(raw_prediction).all():
            raise FloatingPointError(f"{method_name} 输出尺寸不符或含 NaN/Inf")
        rows.extend(
            _metric_rows(
                raw_prediction,
                raw_prediction,
                target,
                batch,
                method_name=method_name,
            )
        )
    return rows


def _environment(device: torch.device) -> dict[str, Any]:
    properties = torch.cuda.get_device_properties(device)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    driver = None
    try:
        query = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if query.returncode == 0:
            driver = query.stdout.strip().splitlines()[0]
    except (OSError, subprocess.SubprocessError, IndexError):
        pass
    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "pytorch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "driver": driver,
        "device": properties.name,
        "device_total_memory_bytes": int(total_bytes),
        "device_free_memory_before_run_bytes": int(free_bytes),
        "device_count": torch.cuda.device_count(),
        "platform": platform.platform(),
        "numpy": np.__version__,
    }


def _active_cuda_processes() -> list[int]:
    query = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )
    if query.returncode != 0:
        raise RuntimeError("无法读取 CUDA 计算进程清单；按预登记规则停止启动")
    process_ids: list[int] = []
    for line in query.stdout.splitlines():
        value = line.strip()
        if value and value.isdigit():
            process_ids.append(int(value))
    return sorted(set(process_ids) - {os.getpid()})


def _load_baseline(
    device: torch.device,
    development_dataset: SarTargetOnlyDataset,
    protocol: dict[str, Any],
) -> tuple[torch.nn.Module, dict[str, Any], float]:
    manifest_path = BASELINE_PATH / "manifest.json"
    baseline_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    context = baseline_manifest.get("context", {})
    if context.get("condition_id") != "x2_light" or context.get("arm") != "sar_only":
        raise ValueError("C2 SAR-only 对照的条件或 arm 不匹配")
    if baseline_manifest.get("confirmation_payload_accessed") is not False:
        raise ValueError("C2 SAR-only 对照曾访问 confirmation 载荷")
    if baseline_manifest.get("test_set_accessed") is not False or baseline_manifest.get("test_set_used") is not False:
        raise ValueError("C2 SAR-only 对照的 test 标记不符合要求")
    expected_input = context.get("input_records", {}).get("development", {})
    fixed_input = development_dataset
    if (
        expected_input.get("manifest_sha256") != fixed_input_manifest_sha256(fixed_input)
        or expected_input.get("files_csv_sha256") != fixed_input_files_sha256(fixed_input)
    ):
        raise ValueError("C2 SAR-only 对照与 C3-A development 固定输入不一致")

    checkpoint_path = BASELINE_PATH / "checkpoints/best.pt"
    checkpoint_sha256 = sha256_file(checkpoint_path)
    expected_checkpoint_hash = protocol["comparison"]["baseline_checkpoint_sha256"]
    if checkpoint_sha256 != expected_checkpoint_hash or checkpoint_sha256 != baseline_manifest.get("best_checkpoint_sha256"):
        raise ValueError("C2 SAR-only 最佳 checkpoint 哈希不匹配")
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("epoch") != baseline_manifest.get("best_epoch"):
        raise ValueError("C2 SAR-only checkpoint 轮次与运行清单不一致")
    if checkpoint.get("context", {}).get("condition_id") != "x2_light":
        raise ValueError("C2 SAR-only checkpoint 条件不匹配")
    model_class = _load_c2_model_class()
    model_config = context.get("model_config", {})
    if set(model_config) != {"width", "blocks", "hr_width"}:
        raise ValueError("C2 SAR-only 基线 model_config 无效")
    model = model_class(scale=2, optical_enabled=False, **model_config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, baseline_manifest, float(baseline_manifest["best_score"])


def fixed_input_manifest_sha256(dataset: SarTargetOnlyDataset) -> str:
    return dataset.manifest_sha256


def fixed_input_files_sha256(dataset: SarTargetOnlyDataset) -> str:
    return dataset.files_csv_sha256


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行或恢复冻结的 C3-A SAR-only Flow 筛选。")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="仅从预定运行目录中最近一个完整 epoch 的 checkpoint 继续。",
    )
    return parser.parse_args(argv)


def _validate_output_mode(path: Path, *, resume: bool) -> None:
    if resume:
        if not path.is_dir():
            raise FileNotFoundError(f"续训要求已有运行目录：{path}")
        return
    if path.exists():
        raise FileExistsError(f"新实验目录已存在，拒绝覆盖：{path}")


def _resume_environment_signature(environment: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "python",
        "python_executable",
        "pytorch",
        "cuda_runtime",
        "cudnn",
        "driver",
        "device",
        "device_count",
    )
    return {key: environment.get(key) for key in fields}


def _resume_code_hashes() -> dict[str, str]:
    paths = (
        "MMCANetV2/phase_c3a_flow.py",
        "scripts/run_phase_c3a_sar_flow.py",
        "MMCANetV2/phase_c2_data.py",
        "MMCANetV2/phase_c2_runner.py",
        "MMCANet/src/MMCANetV2/phase_c2_matched.py",
    )
    return {path: sha256_file(REPOSITORY_ROOT / path) for path in paths}


def _build_resume_binding(
    *,
    protocol: dict[str, Any],
    source_manifest_sha256: str,
    fixed_input_records: dict[str, dict[str, Any]],
    environment: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    return {
        "source_manifest_sha256": source_manifest_sha256,
        "fixed_input_records": fixed_input_records,
        "code_sha256": _resume_code_hashes(),
        "environment_signature": _resume_environment_signature(environment),
        "training_options": {
            "protocol_sha256": hashlib.sha256(PROTOCOL_PATH.read_bytes()).hexdigest(),
            "training": protocol["training"],
            "method": protocol["method"],
            "device": str(device),
        },
    }


def _validate_resume_manifest(
    manifest: dict[str, Any],
    input_manifest: dict[str, Any],
    *,
    protocol_sha256: str,
    resume_binding: dict[str, Any],
) -> None:
    if manifest.get("format") != "mmcanet_c3a_sar_only_flow_run_v2":
        raise ValueError("运行 manifest 格式不支持严格续训")
    if manifest.get("status") not in {"running", "failed"}:
        raise ValueError(f"运行状态不允许续训：{manifest.get('status')!r}")
    if manifest.get("protocol_sha256") != protocol_sha256:
        raise ValueError("运行 manifest 的冻结协议哈希不匹配")
    if input_manifest.get("protocol_sha256") != protocol_sha256:
        raise ValueError("输入 manifest 的冻结协议哈希不匹配")
    if manifest.get("resume_binding") != resume_binding:
        raise ValueError("运行 manifest 的续训绑定信息不匹配")
    if input_manifest.get("resume_binding") != resume_binding:
        raise ValueError("输入 manifest 的续训绑定信息不匹配")
    expected_records = resume_binding["fixed_input_records"]
    if manifest.get("fixed_input_records") != expected_records:
        raise ValueError("运行 manifest 的固定输入记录不匹配")
    if input_manifest.get("roles") != expected_records:
        raise ValueError("输入 manifest 的角色记录不匹配")
    expected_source_hash = resume_binding["source_manifest_sha256"]
    if manifest.get("c1_source_manifest_sha256") != expected_source_hash:
        raise ValueError("运行 manifest 的来源清单哈希不匹配")
    if input_manifest.get("source_manifest_sha256") != expected_source_hash:
        raise ValueError("输入 manifest 的来源清单哈希不匹配")
    if input_manifest.get("source_manifest_path") != str(SOURCE_MANIFEST.relative_to(REPOSITORY_ROOT)):
        raise ValueError("输入 manifest 的来源清单路径不匹配")
    if set(input_manifest.get("roles", {})) != {"train", "development"}:
        raise ValueError("续训输入必须且只能包含 train/development")
    for role, record in input_manifest["roles"].items():
        if record.get("payload_arrays_read") != ["lr", "target"]:
            raise ValueError(f"续训 {role} 输入数组不是 SAR-only")
        if manifest.get("payload_access", {}).get(role) is not True:
            raise ValueError(f"运行记录未确认已读取 {role} 固定输入")
        if input_manifest.get("payload_access", {}).get(role) is not True:
            raise ValueError(f"输入记录未确认已读取 {role} 固定输入")
    protected_flags = (
        "confirmation_payload_accessed",
        "test_set_accessed",
        "test_set_used",
        "optical_payload_read",
    )
    for key in protected_flags:
        if manifest.get(key) is not False or input_manifest.get(key) is not False:
            label = "光学载荷" if key == "optical_payload_read" else key
            raise ValueError(f"续训记录未保持保护标记 {label}=false")


def _make_checkpoint(
    model: SarConditionalFlow,
    optimizer: torch.optim.Optimizer,
    *,
    epoch: int,
    score: float,
    protocol_sha256: str,
    loader_generator: torch.Generator,
    flow_generator: torch.Generator,
    best_score: float,
    best_epoch: int,
    train_log: list[dict[str, Any]],
    development_log: list[dict[str, Any]],
    resume_binding: dict[str, Any],
) -> dict[str, Any]:
    numpy_state = np.random.get_state()
    return {
        "format": "mmcanet_c3a_sar_only_flow_checkpoint_v2",
        "protocol_sha256": protocol_sha256,
        "epoch": epoch,
        "development_group_balanced_mean_psnr": float(score),
        "best_development_group_balanced_mean_psnr": float(best_score),
        "best_epoch": int(best_epoch),
        "resume_binding": resume_binding,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "python_rng_state": random.getstate(),
        "numpy_rng_state": {
            "generator": numpy_state[0],
            "state": torch.from_numpy(numpy_state[1].copy()),
            "position": int(numpy_state[2]),
            "has_gaussian": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [],
        "loader_generator_state": loader_generator.get_state(),
        "flow_generator_state": flow_generator.get_state(),
        "train_log": [dict(row) for row in train_log],
        "development_log": [dict(row) for row in development_log],
    }


def _validate_resume_checkpoint(
    checkpoint: dict[str, Any],
    *,
    protocol_sha256: str,
    resume_binding: dict[str, Any],
    total_epochs: int,
) -> None:
    if checkpoint.get("format") != "mmcanet_c3a_sar_only_flow_checkpoint_v2":
        raise ValueError("续训 checkpoint 格式不支持")
    if checkpoint.get("protocol_sha256") != protocol_sha256:
        raise ValueError("续训 checkpoint 协议哈希不匹配")
    if checkpoint.get("resume_binding") != resume_binding:
        raise ValueError("续训 checkpoint 绑定信息不匹配")
    epoch = checkpoint.get("epoch")
    best_epoch = checkpoint.get("best_epoch")
    if isinstance(epoch, bool) or not isinstance(epoch, int) or not 1 <= epoch <= total_epochs:
        raise ValueError("续训 checkpoint 轮次超出冻结协议")
    if isinstance(best_epoch, bool) or not isinstance(best_epoch, int) or not 1 <= best_epoch <= epoch:
        raise ValueError("续训 checkpoint 最佳轮次无效")
    score = checkpoint.get("development_group_balanced_mean_psnr")
    best_score = checkpoint.get("best_development_group_balanced_mean_psnr")
    if (
        isinstance(score, bool)
        or not isinstance(score, (int, float))
        or not math.isfinite(float(score))
        or isinstance(best_score, bool)
        or not isinstance(best_score, (int, float))
        or not math.isfinite(float(best_score))
    ):
        raise ValueError("续训 checkpoint 指标无效")
    if not isinstance(checkpoint.get("model_state_dict"), dict) or not isinstance(
        checkpoint.get("optimizer_state_dict"), dict
    ):
        raise ValueError("续训 checkpoint 缺少模型或优化器状态")
    numpy_state = checkpoint.get("numpy_rng_state")
    if (
        not isinstance(checkpoint.get("python_rng_state"), tuple)
        or not isinstance(numpy_state, dict)
        or not isinstance(numpy_state.get("state"), torch.Tensor)
        or not isinstance(checkpoint.get("torch_rng_state"), torch.Tensor)
        or not isinstance(checkpoint.get("loader_generator_state"), torch.Tensor)
        or not isinstance(checkpoint.get("flow_generator_state"), torch.Tensor)
    ):
        raise ValueError("续训 checkpoint 缺少全局或专用随机生成器状态")
    cuda_states = checkpoint.get("cuda_rng_state_all")
    if not isinstance(cuda_states, list) or not all(isinstance(state, torch.Tensor) for state in cuda_states):
        raise ValueError("续训 checkpoint CUDA 随机状态格式无效")
    device = str(resume_binding.get("training_options", {}).get("device", ""))
    if device.startswith("cuda"):
        expected_devices = resume_binding.get("environment_signature", {}).get("device_count")
        if not isinstance(expected_devices, int) or expected_devices < 1 or len(cuda_states) != expected_devices:
            raise ValueError("续训 checkpoint CUDA 随机状态数量与设备数不匹配")
    elif cuda_states:
        raise ValueError("CPU 合成续训状态不得包含 CUDA 随机状态")
    train_log = checkpoint.get("train_log")
    if not isinstance(train_log, list) or [row.get("轮次") for row in train_log] != list(range(1, epoch + 1)):
        raise ValueError("续训 checkpoint 训练日志与已完成轮次不一致")
    development_log = checkpoint.get("development_log")
    if not isinstance(development_log, list) or not development_log:
        raise ValueError("续训 checkpoint 缺少开发集逐图日志")
    if any(
        isinstance(row.get("轮次"), bool)
        or not isinstance(row.get("轮次"), int)
        or not 1 <= row["轮次"] <= epoch
        for row in development_log
    ):
        raise ValueError("续训 checkpoint 开发集日志轮次无效")


def _restore_checkpoint(
    checkpoint_path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loader_generator: torch.Generator,
    flow_generator: torch.Generator,
    protocol_sha256: str,
    resume_binding: dict[str, Any],
    total_epochs: int,
) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("续训 checkpoint 必须是字典")
    _validate_resume_checkpoint(
        checkpoint,
        protocol_sha256=protocol_sha256,
        resume_binding=resume_binding,
        total_epochs=total_epochs,
    )
    cuda_states = checkpoint["cuda_rng_state_all"]
    if cuda_states:
        if not torch.cuda.is_initialized() or len(cuda_states) != torch.cuda.device_count():
            raise RuntimeError("恢复 CUDA 随机状态前必须初始化相同数量的 CUDA 设备")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    loader_generator.set_state(checkpoint["loader_generator_state"].to(device="cpu", dtype=torch.uint8))
    flow_generator.set_state(checkpoint["flow_generator_state"].to(device="cpu", dtype=torch.uint8))
    random.setstate(checkpoint["python_rng_state"])
    numpy_state = checkpoint["numpy_rng_state"]
    np.random.set_state(
        (
            str(numpy_state["generator"]),
            numpy_state["state"].cpu().numpy().astype(np.uint32, copy=False),
            int(numpy_state["position"]),
            int(numpy_state["has_gaussian"]),
            float(numpy_state["cached_gaussian"]),
        )
    )
    torch.set_rng_state(checkpoint["torch_rng_state"].to(device="cpu", dtype=torch.uint8))
    if cuda_states:
        torch.cuda.set_rng_state_all([state.cpu() for state in cuda_states])
    return {
        "start_epoch": int(checkpoint["epoch"]) + 1,
        "best_score": float(checkpoint["best_development_group_balanced_mean_psnr"]),
        "best_epoch": int(checkpoint["best_epoch"]),
        "train_log": [dict(row) for row in checkpoint["train_log"]],
        "development_log": [dict(row) for row in checkpoint["development_log"]],
        "checkpoint": checkpoint,
    }


def _select_resume_checkpoint(
    checkpoint_directory: Path,
    *,
    protocol_sha256: str,
    resume_binding: dict[str, Any],
    total_epochs: int,
) -> tuple[Path, dict[str, Any]]:
    candidates: list[tuple[Path, dict[str, Any]]] = []
    for name in ("last.pt", "best.pt"):
        path = checkpoint_directory / name
        if not path.is_file():
            continue
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, dict):
            raise ValueError(f"续训 checkpoint 格式损坏：{path.name}")
        _validate_resume_checkpoint(
            checkpoint,
            protocol_sha256=protocol_sha256,
            resume_binding=resume_binding,
            total_epochs=total_epochs,
        )
        candidates.append((path, checkpoint))
    if not candidates:
        raise FileNotFoundError("续训运行目录中没有有效的 last.pt 或 best.pt")
    selected_path, selected = max(candidates, key=lambda item: int(item[1]["epoch"]))
    best_path = checkpoint_directory / "best.pt"
    selected_best_epoch = int(selected["best_epoch"])
    best_candidate = next((checkpoint for path, checkpoint in candidates if path == best_path), None)
    if best_candidate is None:
        raise FileNotFoundError("续训目录缺少 best.pt，不能安全保留最佳轮次")
    elif (
        int(best_candidate["epoch"]) != selected_best_epoch
        or int(best_candidate.get("best_epoch", -1)) != selected_best_epoch
        or not math.isclose(
            float(best_candidate["development_group_balanced_mean_psnr"]),
            float(selected["best_development_group_balanced_mean_psnr"]),
            rel_tol=0.0,
            abs_tol=0.0,
        )
        or not math.isclose(
            float(best_candidate["best_development_group_balanced_mean_psnr"]),
            float(selected["best_development_group_balanced_mean_psnr"]),
            rel_tol=0.0,
            abs_tol=0.0,
        )
    ):
        raise ValueError("best.pt 与最新 checkpoint 中记录的最佳轮次/分数不一致")
    last_path = checkpoint_directory / "last.pt"
    if selected_path != last_path:
        save_checkpoint_atomic(selected, last_path)
        selected_path = last_path
    return selected_path, selected


def _run() -> int:
    run_started_clock = time.perf_counter()
    started_at = _utc_now()
    args = _parse_args()
    protocol_bytes = PROTOCOL_PATH.read_bytes()
    protocol = json.loads(protocol_bytes)
    protocol_sha256 = hashlib.sha256(protocol_bytes).hexdigest()
    if protocol.get("protocol_id") != "mmcanet_c3a_sar_only_flow_c1_v2_20261105_v1":
        raise ValueError("C3-A protocol 身份不匹配")
    if protocol_sha256 != EXPECTED_PROTOCOL_SHA256:
        raise ValueError("C3-A protocol SHA-256 与代码内冻结值不一致")
    try:
        active_processes = _active_cuda_processes()
    except (OSError, subprocess.SubprocessError, RuntimeError) as error:
        print(f"未启动训练：{error}；没有读取影像。", flush=True)
        return 2
    if active_processes:
        print(
            f"未启动训练：检测到 {len(active_processes)} 个其他 CUDA 计算进程；为避免并发占用，训练未启动且没有读取影像。",
            flush=True,
        )
        return 2
    if torch.cuda.device_count() < 1 or not torch.cuda.is_available():
        print("未启动训练：当前不可见 CUDA 设备；尚未读取影像载荷。", flush=True)
        return 2
    device = torch.device("cuda:0")
    torch.cuda.init()
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    minimum_free_gpu_bytes = int(float(protocol["resources"]["minimum_free_memory_gib"]) * 1024**3)
    print(
        f"GPU 预检：{torch.cuda.get_device_name(device)}，空闲显存 {free_bytes / 1024**3:.2f}/{total_bytes / 1024**3:.2f} GiB。",
        flush=True,
    )
    if free_bytes < minimum_free_gpu_bytes:
        print(
            f"未启动训练：空闲显存低于预登记的 {minimum_free_gpu_bytes / 1024**3:.0f} GiB 门槛；没有创建运行目录或读取影像。",
            flush=True,
        )
        return 2
    _validate_output_mode(OUTPUT_PATH, resume=args.resume)
    git_head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    git_status = subprocess.run(
        ["git", "status", "--short", "--untracked-files=all"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    environment = _environment(device)
    if args.resume:
        run_manifest = json.loads((OUTPUT_PATH / "manifest.json").read_text(encoding="utf-8"))
    else:
        OUTPUT_PATH.mkdir(parents=True, exist_ok=False)
        (OUTPUT_PATH / "checkpoints").mkdir()
        (OUTPUT_PATH / "protocol.json").write_bytes(protocol_bytes)
        (OUTPUT_PATH / "git_state.txt").write_text(
            f"HEAD：{git_head}\n工作区状态（运行开始前）：\n{git_status}", encoding="utf-8"
        )
        run_manifest = {
            "format": "mmcanet_c3a_sar_only_flow_run_v2",
            "run_id": OUTPUT_PATH.name,
            "status": "running",
            "protocol_path": str(PROTOCOL_PATH.relative_to(REPOSITORY_ROOT)),
            "protocol_sha256": protocol_sha256,
            "started_utc": started_at,
            "finished_utc": None,
            "environment": environment,
            "command": [sys.executable, *sys.argv],
            "resume_history": [],
            "last_completed_epoch": 0,
            "confirmation_payload_accessed": False,
            "test_set_accessed": False,
            "test_set_used": False,
            "optical_payload_read": False,
            "payload_access": {
                "train": False,
                "development": False,
                "confirmation": False,
                "test": False,
                "optical": False,
            },
        }
        _json_write(OUTPUT_PATH / "manifest.json", run_manifest)
        (OUTPUT_PATH / "command.txt").write_text(shlex.join([sys.executable, *sys.argv]) + "\n", encoding="utf-8")
        (OUTPUT_PATH / "environment.txt").write_text(
            "\n".join(f"{key}：{value}" for key, value in environment.items()) + "\n", encoding="utf-8"
        )
    input_manifest_path = OUTPUT_PATH / "input_manifest.json"
    input_manifest: dict[str, Any] | None = None
    runtime_before_attempt = float(run_manifest.get("runtime_seconds", 0.0))
    try:
        source_hash = sha256_file(SOURCE_MANIFEST)
        if source_hash != protocol["split"]["source_manifest_sha256"]:
            raise ValueError("C1 v2 冻结来源清单 SHA-256 不匹配")
        fixed_datasets: dict[str, SarTargetOnlyDataset] = {}
        fixed_input_records: dict[str, dict[str, Any]] = {}
        for role in ("train", "development"):
            fixed_dataset = C2FixedDataset(
                INPUT_ROOT / role,
                SOURCE_MANIFEST,
                role=role,
                scale=2,
                verify_hashes_on_read=False,
            )
            if fixed_dataset.manifest.get("condition_id") != "x2_light":
                raise ValueError(f"{role} 固定输入不是 x2_light")
            only_sar_target = SarTargetOnlyDataset(fixed_dataset)
            expected = protocol["split"][role]
            actual_groups = len({row["source_group_id"] for row in only_sar_target.rows})
            if len(only_sar_target) != expected["sample_count"] or actual_groups != expected["source_group_count"]:
                raise ValueError(f"{role} 样本数或来源组数与冻结 C1 v2 不一致")
            fixed_datasets[role] = only_sar_target
            fixed_input_records[role] = {
                "root": str(INPUT_ROOT / role),
                "manifest_sha256": fixed_input_manifest_sha256(only_sar_target),
                "files_csv_sha256": fixed_input_files_sha256(only_sar_target),
                "sample_count": len(only_sar_target),
                "source_group_count": actual_groups,
                "payload_arrays_read": ["lr", "target"],
            }
        train_groups = {row["source_group_id"] for row in fixed_datasets["train"].rows}
        development_groups = {row["source_group_id"] for row in fixed_datasets["development"].rows}
        if train_groups & development_groups:
            raise ValueError("C1 v2 train/development 来源组出现交叉")

        resume_binding = _build_resume_binding(
            protocol=protocol,
            source_manifest_sha256=source_hash,
            fixed_input_records=fixed_input_records,
            environment=environment,
            device=device,
        )
        if args.resume:
            input_manifest = json.loads(input_manifest_path.read_text(encoding="utf-8"))
            _validate_resume_manifest(
                run_manifest,
                input_manifest,
                protocol_sha256=protocol_sha256,
                resume_binding=resume_binding,
            )
            selected_checkpoint_path, selected_checkpoint = _select_resume_checkpoint(
                OUTPUT_PATH / "checkpoints",
                protocol_sha256=protocol_sha256,
                resume_binding=resume_binding,
                total_epochs=int(protocol["training"]["epochs"]),
            )
            if int(run_manifest.get("last_completed_epoch", 0)) > int(selected_checkpoint["epoch"]):
                raise ValueError("运行 manifest 记录的完成轮次晚于有效 checkpoint")
        else:
            input_manifest = {
                "protocol_sha256": protocol_sha256,
                "source_manifest_path": str(SOURCE_MANIFEST.relative_to(REPOSITORY_ROOT)),
                "source_manifest_sha256": source_hash,
                "roles": fixed_input_records,
                "resume_binding": resume_binding,
                "confirmation_payload_accessed": False,
                "test_set_accessed": False,
                "test_set_used": False,
                "optical_payload_read": False,
                "payload_access": {"train": False, "development": False},
            }
            run_manifest.update(
                {
                    "c1_source_manifest_sha256": source_hash,
                    "fixed_input_records": fixed_input_records,
                    "resume_binding": resume_binding,
                }
            )
            _json_write(input_manifest_path, input_manifest)
            _json_write(OUTPUT_PATH / "manifest.json", run_manifest)

        for role in ("train", "development"):
            run_manifest["payload_access"][role] = True
            input_manifest["payload_access"][role] = True
            _json_write(input_manifest_path, input_manifest)
            _json_write(OUTPUT_PATH / "manifest.json", run_manifest)
            fixed_datasets[role].verify_all()
            print(
                f"已核验 {role}：{len(fixed_datasets[role])} 个样本、"
                f"{fixed_input_records[role]['source_group_count']} 个来源组；"
                "仅读取 SAR LR 与目标数组。",
                flush=True,
            )

        baseline_model, baseline_manifest, baseline_manifest_score = _load_baseline(
            device, fixed_datasets["development"], protocol
        )
        loader_generator, flow_generator = _set_determinism(int(protocol["training"]["seed"]))
        batch_size = int(protocol["training"]["batch_size"])
        train_loader = DataLoader(
            fixed_datasets["train"],
            batch_size=batch_size,
            shuffle=True,
            generator=loader_generator,
            num_workers=0,
            drop_last=False,
        )
        development_loader = DataLoader(
            fixed_datasets["development"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            drop_last=False,
        )
        baseline_rows = _evaluate(
            baseline_model,
            development_loader,
            device=device,
            method_name="C2 SAR-only",
            run_seed=int(protocol["training"]["seed"]),
            sampler_steps=None,
            noise_sigma=float(protocol["method"]["noise_sigma"]),
        )
        baseline_score = _group_balanced(baseline_rows, "PSNR")
        if not math.isclose(baseline_score, baseline_manifest_score, rel_tol=0.0, abs_tol=1e-4):
            raise ValueError(
                f"C2 开发集基线复算分数 {baseline_score:.8f} 与运行清单 {baseline_manifest_score:.8f} 不一致"
            )
        _csv_write(
            OUTPUT_PATH / "sar_only_baseline_development.csv",
            baseline_rows,
            ["样本 ID", "来源组 ID", "patch ID", "PSNR", "SSIM", "最终裁剪比例"],
        )

        model_config = protocol["method"]["model"]
        model = SarConditionalFlow(
            base_channels=int(model_config["base_channels"]),
            time_dimension=int(model_config["time_dimension"]),
        ).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(protocol["training"]["learning_rate"]),
            betas=tuple(float(value) for value in protocol["training"]["betas"]),
            weight_decay=float(protocol["training"]["weight_decay"]),
        )
        model_parameter_count = sum(parameter.numel() for parameter in model.parameters())
        train_log: list[dict[str, Any]] = []
        development_log: list[dict[str, Any]] = []
        best_score = -math.inf
        best_epoch = 0
        seed = int(protocol["training"]["seed"])
        start_epoch = 1
        previous_status = str(run_manifest["status"])
        if args.resume:
            resume_state = _restore_checkpoint(
                selected_checkpoint_path,
                model=model,
                optimizer=optimizer,
                loader_generator=loader_generator,
                flow_generator=flow_generator,
                protocol_sha256=protocol_sha256,
                resume_binding=resume_binding,
                total_epochs=int(protocol["training"]["epochs"]),
            )
            start_epoch = int(resume_state["start_epoch"])
            best_score = float(resume_state["best_score"])
            best_epoch = int(resume_state["best_epoch"])
            train_log = resume_state["train_log"]
            development_log = resume_state["development_log"]
            checkpoint_epoch = start_epoch - 1
            _csv_write(
                OUTPUT_PATH / "train_log.csv",
                train_log,
                [
                    "轮次", "训练速度场 MSE", "优化器步数", "训练样本数", "开发集来源组等权 PSNR",
                    "开发集来源组等权 SSIM", "开发集来源组等权裁剪比例", "轮次秒数",
                ],
            )
            _csv_write(
                OUTPUT_PATH / "development_metrics.csv",
                development_log,
                ["轮次", "样本 ID", "来源组 ID", "patch ID", "PSNR", "SSIM", "最终裁剪比例"],
            )
            history = run_manifest.setdefault("resume_history", [])
            if not isinstance(history, list):
                raise ValueError("运行 manifest 的续训历史格式无效")
            history.append(
                {
                    "恢复时间 UTC": _utc_now(),
                    "前次状态": previous_status,
                    "恢复自轮次": checkpoint_epoch,
                    "下一轮": start_epoch,
                    "checkpoint 路径": str(selected_checkpoint_path.relative_to(REPOSITORY_ROOT)),
                    "checkpoint SHA-256": sha256_file(selected_checkpoint_path),
                    "前次失败类型": run_manifest.get("failure_type"),
                    "前次失败信息": run_manifest.get("failure_message_zh_or_raw"),
                    "命令": shlex.join([sys.executable, *sys.argv]),
                }
            )
            run_manifest.pop("failure_type", None)
            run_manifest.pop("failure_message_zh_or_raw", None)
            run_manifest.pop("finished_utc", None)
            run_manifest.update(
                {
                    "status": "running",
                    "resumed_at_utc": _utc_now(),
                    "last_completed_epoch": checkpoint_epoch,
                    "last_checkpoint_sha256": sha256_file(selected_checkpoint_path),
                }
            )
            _json_write(OUTPUT_PATH / "manifest.json", run_manifest)

        for epoch in range(start_epoch, int(protocol["training"]["epochs"]) + 1):
            epoch_started = time.perf_counter()
            model.train()
            total_loss = 0.0
            total_samples = 0
            optimizer_steps = 0
            for batch in train_loader:
                lr = batch["lr"].to(device)
                target = batch["target"].to(device)
                condition = upsample_sar_condition(lr, size=(256, 256))
                time_value = torch.rand((target.shape[0],), generator=flow_generator, device=device)
                noise = torch.randn(target.shape, generator=flow_generator, device=device)
                state, velocity_target = make_flow_training_pair(
                    target,
                    condition,
                    time_value,
                    noise,
                    noise_sigma=float(protocol["method"]["noise_sigma"]),
                )
                optimizer.zero_grad(set_to_none=True)
                velocity = model(state, condition, time_value)
                loss = F.mse_loss(velocity, velocity_target)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"第 {epoch} 轮训练损失含 NaN 或 Inf")
                loss.backward()
                for name, parameter in model.named_parameters():
                    if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                        raise FloatingPointError(f"第 {epoch} 轮梯度含 NaN 或 Inf：{name}")
                optimizer.step()
                for name, parameter in model.named_parameters():
                    if not torch.isfinite(parameter).all():
                        raise FloatingPointError(f"第 {epoch} 轮参数含 NaN 或 Inf：{name}")
                total_loss += float(loss.detach().cpu()) * int(target.shape[0])
                total_samples += int(target.shape[0])
                optimizer_steps += 1
            if total_samples != len(fixed_datasets["train"]):
                raise ValueError(f"第 {epoch} 轮训练样本数不完整：{total_samples}")
            validation_rows = _evaluate(
                model,
                development_loader,
                device=device,
                method_name="C3-A SAR-only flow",
                run_seed=seed,
                sampler_steps=int(protocol["method"]["sampler_steps"]),
                noise_sigma=float(protocol["method"]["noise_sigma"]),
            )
            score = _group_balanced(validation_rows, "PSNR")
            ssim_score = _group_balanced(validation_rows, "SSIM")
            clipping_score = _group_balanced(validation_rows, "最终裁剪比例")
            for row in validation_rows:
                development_log.append({"轮次": epoch, **row})
            train_log.append(
                {
                    "轮次": epoch,
                    "训练速度场 MSE": total_loss / total_samples,
                    "优化器步数": optimizer_steps,
                    "训练样本数": total_samples,
                    "开发集来源组等权 PSNR": score,
                    "开发集来源组等权 SSIM": ssim_score,
                    "开发集来源组等权裁剪比例": clipping_score,
                    "轮次秒数": time.perf_counter() - epoch_started,
                }
            )
            improved = score > best_score
            if improved:
                best_score = score
                best_epoch = epoch
            checkpoint = _make_checkpoint(
                model,
                optimizer,
                epoch=epoch,
                score=score,
                protocol_sha256=protocol_sha256,
                loader_generator=loader_generator,
                flow_generator=flow_generator,
                best_score=best_score,
                best_epoch=best_epoch,
                train_log=train_log,
                development_log=development_log,
                resume_binding=resume_binding,
            )
            if improved:
                save_checkpoint_atomic(checkpoint, OUTPUT_PATH / "checkpoints/best.pt")
            save_checkpoint_atomic(checkpoint, OUTPUT_PATH / "checkpoints/last.pt")
            run_manifest.update(
                {
                    "status": "running",
                    "last_completed_epoch": epoch,
                    "best_epoch": best_epoch,
                    "best_development_group_balanced_mean_psnr": best_score,
                    "last_checkpoint_sha256": sha256_file(OUTPUT_PATH / "checkpoints/last.pt"),
                    "best_checkpoint_sha256": sha256_file(OUTPUT_PATH / "checkpoints/best.pt"),
                    "runtime_seconds": runtime_before_attempt + time.perf_counter() - run_started_clock,
                }
            )
            _json_write(OUTPUT_PATH / "manifest.json", run_manifest)
            print(
                f"第 {epoch}/{protocol['training']['epochs']} 轮：速度场 MSE={total_loss / total_samples:.6f}，"
                f"开发集来源组等权 PSNR={score:.4f} dB，SSIM={ssim_score:.5f}，裁剪比例={clipping_score:.4%}。",
                flush=True,
            )
            _csv_write(
                OUTPUT_PATH / "train_log.csv",
                train_log,
                ["轮次", "训练速度场 MSE", "优化器步数", "训练样本数", "开发集来源组等权 PSNR", "开发集来源组等权 SSIM", "开发集来源组等权裁剪比例", "轮次秒数"],
            )
            _csv_write(
                OUTPUT_PATH / "development_metrics.csv",
                development_log,
                ["轮次", "样本 ID", "来源组 ID", "patch ID", "PSNR", "SSIM", "最终裁剪比例"],
            )
        best_checkpoint = torch.load(OUTPUT_PATH / "checkpoints/best.pt", map_location=device, weights_only=True)
        model.load_state_dict(best_checkpoint["model_state_dict"], strict=True)
        flow_rows = _evaluate(
            model,
            development_loader,
            device=device,
            method_name="C3-A SAR-only flow 最佳轮次",
            run_seed=seed,
            sampler_steps=int(protocol["method"]["sampler_steps"]),
            noise_sigma=float(protocol["method"]["noise_sigma"]),
        )
        baseline_index = {row["样本 ID"]: row for row in baseline_rows}
        if len(baseline_index) != len(flow_rows):
            raise ValueError("C2 基线与 C3-A flow 的 development 样本数不一致")
        comparison_rows: list[dict[str, Any]] = []
        for flow_row in flow_rows:
            base = baseline_index.get(flow_row["样本 ID"])
            if base is None or base["来源组 ID"] != flow_row["来源组 ID"] or base["patch ID"] != flow_row["patch ID"]:
                raise ValueError("C2 基线与 C3-A flow 样本身份/来源组/patch 不一致")
            comparison_rows.append(
                {
                    **flow_row,
                    "基线 PSNR": base["PSNR"],
                    "配对 ΔPSNR": flow_row["PSNR"] - base["PSNR"],
                    "基线 SSIM": base["SSIM"],
                    "配对 ΔSSIM": flow_row["SSIM"] - base["SSIM"],
                    "基线裁剪比例": base["最终裁剪比例"],
                }
            )
        psnr_effect = _paired_group_bootstrap(
            comparison_rows,
            metric="配对 ΔPSNR",
            seed=seed + 100,
        )
        ssim_effect = _paired_group_bootstrap(
            comparison_rows,
            metric="配对 ΔSSIM",
            seed=seed + 101,
        )
        _csv_write(
            OUTPUT_PATH / "paired_development_comparison.csv",
            comparison_rows,
            ["样本 ID", "来源组 ID", "patch ID", "PSNR", "SSIM", "最终裁剪比例", "基线 PSNR", "配对 ΔPSNR", "基线 SSIM", "配对 ΔSSIM", "基线裁剪比例"],
        )
        minimum_effect = float(protocol["comparison"]["exploration_continuation_minimum_delta_psnr_db"])
        minimum_lower_bound = float(protocol["comparison"]["exploration_continuation_ci_lower_minimum"])
        minimum_ssim_effect = float(protocol["comparison"]["exploration_continuation_minimum_delta_ssim"])
        continue_signal = (
            psnr_effect["mean"] >= minimum_effect
            and psnr_effect["lower_95"] > minimum_lower_bound
            and ssim_effect["mean"] >= minimum_ssim_effect
        )
        run_manifest.update(
            {
                "status": "complete",
                "finished_utc": _utc_now(),
                "last_completed_epoch": int(protocol["training"]["epochs"]),
                "c1_source_manifest_sha256": source_hash,
                "fixed_input_records": fixed_input_records,
                "resume_binding": resume_binding,
                "baseline": {
                    "run_manifest": str((BASELINE_PATH / "manifest.json").relative_to(REPOSITORY_ROOT)),
                    "checkpoint_sha256": sha256_file(BASELINE_PATH / "checkpoints/best.pt"),
                    "selected_epoch": baseline_manifest.get("best_epoch"),
                    "recomputed_development_group_balanced_psnr": baseline_score,
                },
                "method": {
                    "model_parameter_count": model_parameter_count,
                    "best_epoch": best_epoch,
                    "best_development_group_balanced_psnr": best_score,
                    "sampler": f"{protocol['method']['sampler_steps']} 步 Heun；每步 2 次速度场评估",
                    "noise_sigma": float(protocol["method"]["noise_sigma"]),
                    "optical_input": False,
                },
                "paired_development_effect": {
                    "psnr": psnr_effect,
                    "ssim": ssim_effect,
                    "continued_to_second_seed": continue_signal,
                    "interpretation_zh": "复用 development 的单 seed 探索信号；不构成独立泛化证据或阶段晋级。",
                },
                "best_checkpoint_sha256": sha256_file(OUTPUT_PATH / "checkpoints/best.pt"),
                "last_checkpoint_sha256": sha256_file(OUTPUT_PATH / "checkpoints/last.pt"),
                "output_sample_count": len(comparison_rows),
                "runtime_seconds": runtime_before_attempt + time.perf_counter() - run_started_clock,
                "code_sha256": resume_binding["code_sha256"],
                "git_head": git_head,
                "git_status_dirty": bool(git_status.strip()),
                "peak_gpu_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            }
        )
        _json_write(
            OUTPUT_PATH / "runtime.json",
            {
                "started_utc": started_at,
                "finished_utc": run_manifest["finished_utc"],
                "runtime_seconds": run_manifest["runtime_seconds"],
                "peak_gpu_memory_allocated_bytes": run_manifest["peak_gpu_memory_allocated_bytes"],
            },
        )
        _json_write(OUTPUT_PATH / "manifest.json", run_manifest)
        print(
            f"完成：最佳轮次 {best_epoch}；flow−C2 SAR-only 来源组等权 ΔPSNR={psnr_effect['mean']:+.4f} dB，"
            f"95% 区间 [{psnr_effect['lower_95']:+.4f}, {psnr_effect['upper_95']:+.4f}]；"
            f"是否追加第二 seed={('是' if continue_signal else '否')}。结果仅属开发集探索。",
            flush=True,
        )
        return 0
    except Exception as error:
        run_manifest.update(
            {
                "status": "failed",
                "finished_utc": _utc_now(),
                "failure_type": type(error).__name__,
                "failure_message_zh_or_raw": str(error),
            }
        )
        _json_write(OUTPUT_PATH / "manifest.json", run_manifest)
        print(f"实验失败，诊断已保存在 {OUTPUT_PATH / 'manifest.json'}：{type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(_run())
