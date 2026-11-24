"""在 C1 v2 的 train/development 固定输入上训练 SAR-only SwinIR 适配版。"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import platform
import random
import shutil
import subprocess
import sys
import time
import types
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MMCANetV2.phase_c3a_metrics import build_fixed_masks, edge_metrics, ssim


SOURCE_ARCH = PROJECT_ROOT / "BasicSR/basicsr/archs/swinir_arch.py"
SOURCE_REGISTRY = PROJECT_ROOT / "BasicSR/basicsr/utils/registry.py"
SCALE = 2
TRAIN_SAMPLES = 1600
DEVELOPMENT_SAMPLES = 200
TRAIN_GROUPS = 92
DEVELOPMENT_GROUPS = 19
EXPECTED_INTERNAL_BASELINE_SHA256 = "8255b5229ca1e041cb59aed582d794bb67afdbe1601b6f4dc425c43ce27f7165"
METRIC_NAMES = (
    "psnr_db",
    "ssim",
    "epi_corr",
    "edge_strength_ratio",
    "bright_mean_ratio",
    "bright_p95_ratio",
    "background_mean_ratio",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_pair_arrays(root: str | Path, row: dict[str, str], scale: int) -> dict[str, np.ndarray]:
    """只读取并核验 SAR LR 与 HR noisy pseudo-target，不接受 optical 参数。"""
    root = Path(root)
    if scale not in (2, 4):
        raise ValueError("固定输入倍率无效")
    sample_id = row.get("sample_id", "")
    if not sample_id or Path(sample_id).name != sample_id:
        raise ValueError("固定输入 sample_id 无效")
    arrays: dict[str, np.ndarray] = {}
    for name in ("lr", "target"):
        path = root / name / f"{sample_id}.npy"
        if not path.is_file():
            raise ValueError(f"缺少固定 {name} 输入：{sample_id}")
        if sha256_file(path) != row.get(f"{name}_sha256"):
            raise ValueError(f"固定输入 {name} 哈希不匹配：{sample_id}")
        value = np.load(path, allow_pickle=False)
        if value.dtype != np.float32 or value.ndim != 3 or value.shape[0] != 1:
            raise ValueError(f"固定输入 {name} 形状或类型无效：{sample_id}")
        if not np.isfinite(value).all():
            raise ValueError(f"固定输入 {name} 含非有限值：{sample_id}")
        if name == "target" and (value.min() < 0 or value.max() > 1):
            raise ValueError(f"HR noisy pseudo-target 超出 [0,1]：{sample_id}")
        arrays[name] = np.ascontiguousarray(value)
    lr_height, lr_width = arrays["lr"].shape[-2:]
    if arrays["target"].shape[-2:] != (lr_height * scale, lr_width * scale):
        raise ValueError(f"固定 LR/HR 尺寸倍率不一致：{sample_id}")
    return arrays


def load_split(input_root: Path, role: str, expected_samples: int, expected_groups: int):
    if role not in {"train", "development"}:
        raise ValueError("本实验仅接受 train/development")
    root = input_root / role
    manifest_path = root / "manifest.json"
    files_path = root / "files.csv"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("format") != "c2_fixed_inputs_v1"
        or manifest.get("status") != "complete"
        or manifest.get("role") != role
        or manifest.get("scale") != SCALE
        or manifest.get("sample_count") != expected_samples
        or manifest.get("test_set_accessed") is not False
        or manifest.get("test_set_used") is not False
        or manifest.get("confirmation_payload_accessed") is not False
    ):
        raise ValueError(f"{role} 固定输入 manifest 不符合 C1 v2 安全合同")
    if sha256_file(files_path) != manifest.get("files_csv_sha256"):
        raise ValueError(f"{role} files.csv 哈希不匹配")
    with files_path.open(encoding="utf-8", newline="") as stream:
        rows = sorted(csv.DictReader(stream), key=lambda row: row["sample_id"])
    if len(rows) != expected_samples:
        raise ValueError(f"{role} 固定输入数量不匹配")
    identities = [row["sample_id"] for row in rows]
    if len(identities) != len(set(identities)):
        raise ValueError(f"{role} 固定输入 sample_id 重复")
    groups = {row.get("source_group_id", "") for row in rows}
    if "" in groups or len(groups) != expected_groups:
        raise ValueError(f"{role} 来源组数量不匹配")
    records = []
    for row in rows:
        arrays = load_pair_arrays(root, row, SCALE)
        records.append(
            {
                "sample_id": row["sample_id"],
                "source_group_id": row["source_group_id"],
                "patch_id": row["patch_id"],
                "lr": arrays["lr"],
                "target": arrays["target"],
            }
        )
    return records, {
        "role": role,
        "sample_count": len(records),
        "source_group_count": len(groups),
        "manifest_sha256": sha256_file(manifest_path),
        "files_csv_sha256": manifest["files_csv_sha256"],
        "test_set_accessed": False,
        "test_set_used": False,
        "confirmation_payload_accessed": False,
        "optical_payload_loaded": False,
    }


def load_swinir_class():
    """加载仓库中的原始 SwinIR 文件，绕过不完整 BasicSR 顶层导入。"""
    basicsr_root = PROJECT_ROOT / "BasicSR/basicsr"
    if not SOURCE_ARCH.is_file() or not SOURCE_REGISTRY.is_file():
        raise FileNotFoundError("仓库 vendored SwinIR 源文件缺失")

    # 此子目录的 BasicSR/__init__.py 会导入仓库未包含的 basicsr.data。
    # 用 namespace package 装载原始 SwinIR 所需的 registry，不修改 vendored 源码。
    basicsr = types.ModuleType("basicsr")
    basicsr.__path__ = [str(basicsr_root)]
    sys.modules["basicsr"] = basicsr
    utils = types.ModuleType("basicsr.utils")
    utils.__path__ = [str(basicsr_root / "utils")]
    sys.modules["basicsr.utils"] = utils
    registry_spec = importlib.util.spec_from_file_location(
        "basicsr.utils.registry", SOURCE_REGISTRY
    )
    if registry_spec is None or registry_spec.loader is None:
        raise ImportError("无法加载 SwinIR registry")
    registry_module = importlib.util.module_from_spec(registry_spec)
    registry_spec.loader.exec_module(registry_module)
    registry_module.ARCH_REGISTRY = registry_module.Registry("arch")
    sys.modules["basicsr.utils.registry"] = registry_module

    archs = types.ModuleType("basicsr.archs")
    archs.__path__ = [str(basicsr_root / "archs")]
    sys.modules["basicsr.archs"] = archs
    arch_util = types.ModuleType("basicsr.archs.arch_util")
    arch_util.to_2tuple = lambda value: value if isinstance(value, tuple) else (value, value)
    arch_util.trunc_normal_ = torch.nn.init.trunc_normal_
    sys.modules["basicsr.archs.arch_util"] = arch_util

    swinir_spec = importlib.util.spec_from_file_location(
        "basicsr.archs.swinir_arch", SOURCE_ARCH
    )
    if swinir_spec is None or swinir_spec.loader is None:
        raise ImportError("无法加载原始 SwinIR 源文件")
    swinir_module = importlib.util.module_from_spec(swinir_spec)
    swinir_spec.loader.exec_module(swinir_module)
    return swinir_module.SwinIR


def build_swinir(swinir_class: type[nn.Module]) -> nn.Module:
    class SAROnlySwinIR(nn.Module):
        def __init__(self):
            super().__init__()
            self.core = swinir_class(
                img_size=128,
                patch_size=1,
                in_chans=1,
                embed_dim=60,
                depths=(6,),
                num_heads=(6,),
                window_size=8,
                mlp_ratio=2.0,
                upscale=2,
                img_range=1.0,
                upsampler="pixelshuffledirect",
                resi_connection="1conv",
            )
            self.output = nn.Sigmoid()

        def forward(self, sar_lr: torch.Tensor) -> torch.Tensor:
            if sar_lr.ndim != 4 or sar_lr.shape[1] != 1:
                raise ValueError("SAR LR 输入必须为 [B,1,H,W]")
            if not torch.isfinite(sar_lr).all():
                raise ValueError("SAR LR 输入含非有限值")
            if any(size % 8 for size in sar_lr.shape[-2:]):
                raise ValueError("SwinIR 输入高宽必须被 8 整除")
            prediction = self.output(self.core(sar_lr))
            expected = (sar_lr.shape[0], 1, sar_lr.shape[-2] * 2, sar_lr.shape[-1] * 2)
            if tuple(prediction.shape) != expected:
                raise RuntimeError("SwinIR 输出尺寸不符合 x2 SAR 合同")
            return prediction

    model = SAROnlySwinIR()
    model.experiment_config = {
        "方法": "SwinIR 轻量结构的 SAR 单通道任务适配版",
        "结构": "embed_dim=60, depths=(6,), num_heads=(6,), window_size=8, mlp_ratio=2, pixelshuffledirect x2",
        "输入通道": 1,
        "输出通道": 1,
        "输出激活": "Sigmoid",
        "初始化": "随机初始化，不使用外部预训练权重",
        "上游源文件": "BasicSR/basicsr/archs/swinir_arch.py",
        "方法边界": "发布架构的任务适配实现，不是 SAR 专用论文权重或原论文完整复现",
    }
    return model


def _augment_crop(record: dict[str, Any], rng: np.random.Generator, patch_size: int):
    lr = record["lr"]
    target = record["target"]
    height, width = lr.shape[-2:]
    if patch_size > min(height, width):
        raise ValueError("随机裁剪大于 LR 输入")
    top = int(rng.integers(0, height - patch_size + 1))
    left = int(rng.integers(0, width - patch_size + 1))
    lr_crop = lr[:, top : top + patch_size, left : left + patch_size]
    hr_top, hr_left = top * SCALE, left * SCALE
    hr_size = patch_size * SCALE
    target_crop = target[:, hr_top : hr_top + hr_size, hr_left : hr_left + hr_size]
    if rng.integers(2):
        lr_crop, target_crop = lr_crop[..., ::-1], target_crop[..., ::-1]
    if rng.integers(2):
        lr_crop, target_crop = lr_crop[..., ::-1, :], target_crop[..., ::-1, :]
    if rng.integers(2):
        lr_crop, target_crop = lr_crop.transpose(0, 2, 1), target_crop.transpose(0, 2, 1)
    return np.ascontiguousarray(lr_crop), np.ascontiguousarray(target_crop)


def _psnr(prediction: np.ndarray, target: np.ndarray) -> float:
    mse = float(np.mean((prediction.astype(np.float64) - target.astype(np.float64)) ** 2))
    return float("inf") if mse == 0 else float(10.0 * math.log10(1.0 / mse))


def group_balanced_mean(rows: list[dict[str, Any]], metric: str = "psnr_db") -> float:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        grouped.setdefault(str(row["source_group_id"]), []).append(float(row[metric]))
    if not grouped:
        raise ValueError("开发集指标为空")
    return float(np.mean([np.mean(values) for values in grouped.values()]))


def evaluate_psnr(model: nn.Module, records: list[dict[str, Any]], device: torch.device):
    model.eval()
    rows = []
    with torch.inference_mode():
        for record in records:
            lr = torch.from_numpy(record["lr"][None]).to(device)
            target = record["target"][0]
            prediction = model(lr)[0, 0].float().cpu().numpy()
            rows.append(
                {
                    "sample_id": record["sample_id"],
                    "source_group_id": record["source_group_id"],
                    "psnr_db": _psnr(prediction, target),
                }
            )
    return rows


def _evaluate_full(model, records, device, prediction_dir: Path):
    model.eval()
    prediction_dir.mkdir(parents=True, exist_ok=False)
    rows = []
    inference_seconds = []
    max_repeat_diff = 0.0
    with torch.inference_mode():
        for index, record in enumerate(records):
            lr_np = record["lr"]
            target = record["target"][0]
            lr = torch.from_numpy(lr_np[None]).to(device)
            torch.cuda.synchronize()
            started = time.perf_counter()
            prediction_tensor = model(lr)
            torch.cuda.synchronize()
            inference_seconds.append(time.perf_counter() - started)
            if index == 0:
                repeated = model(lr)
                torch.cuda.synchronize()
                max_repeat_diff = float(
                    (prediction_tensor.float() - repeated.float()).abs().max().cpu()
                )
            prediction = prediction_tensor[0, 0].float().cpu().numpy()
            np.save(prediction_dir / f"{record['sample_id']}.npy", prediction, allow_pickle=False)
            masks, thresholds = build_fixed_masks(lr_np, record["target"], SCALE)
            edge = edge_metrics(prediction, target, masks["edge"])
            bright_pred = prediction[masks["high_return"]]
            bright_target = target[masks["high_return"]]
            background_pred = prediction[masks["background"]]
            background_target = target[masks["background"]]
            row = {
                "sample_id": record["sample_id"],
                "source_group_id": record["source_group_id"],
                "patch_id": record["patch_id"],
                "psnr_db": _psnr(prediction, target),
                "ssim": ssim(prediction, target, data_range=1.0),
                "epi_corr": edge["epi_corr"],
                "edge_strength_ratio": edge["edge_strength_ratio"],
                "bright_mean_ratio": float(bright_pred.mean() / (bright_target.mean() + 1e-12)),
                "bright_p95_ratio": float(
                    np.percentile(bright_pred, 95, method="linear")
                    / (np.percentile(bright_target, 95, method="linear") + 1e-12)
                ),
                "background_mean_ratio": float(
                    background_pred.mean() / (background_target.mean() + 1e-12)
                ),
                "high_return_threshold": thresholds["high_return"],
                "background_threshold": thresholds["background"],
            }
            if not all(math.isfinite(float(row[key])) for key in METRIC_NAMES):
                raise FloatingPointError(f"{record['sample_id']} 指标含非有限值")
            rows.append(row)
    return rows, {
        "image_count": len(rows),
        "inference_seconds_total": float(sum(inference_seconds)),
        "inference_seconds_median_per_image": float(np.median(inference_seconds)),
        "repeat_inference_max_abs_difference": max_repeat_diff,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]):
    if not rows:
        raise ValueError(f"拒绝写入空 CSV：{path}")
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _summary(rows: list[dict[str, Any]]):
    result = {
        "sample_count": len(rows),
        "source_group_count": len({row["source_group_id"] for row in rows}),
        "sample_mean": {},
        "sample_std": {},
        "group_balanced_mean": {},
    }
    for metric in METRIC_NAMES:
        values = np.asarray([float(row[metric]) for row in rows], dtype=np.float64)
        result["sample_mean"][metric] = float(values.mean())
        result["sample_std"][metric] = float(values.std(ddof=1))
        result["group_balanced_mean"][metric] = group_balanced_mean(rows, metric)
    return result


def _group_statistics(method_rows: dict[str, list[dict[str, Any]]]):
    result = []
    for method, rows in method_rows.items():
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            grouped.setdefault(row["source_group_id"], []).append(row)
        for group, values in sorted(grouped.items()):
            output = {"method": method, "source_group_id": group, "sample_count": len(values)}
            for metric in METRIC_NAMES:
                output[metric] = float(np.mean([float(row[metric]) for row in values]))
            result.append(output)
    return result


def _paired_comparison(swinir_rows, baseline_rows, seed: int):
    baseline_by_id = {row["sample_id"]: row for row in baseline_rows}
    if set(baseline_by_id) != {row["sample_id"] for row in swinir_rows}:
        raise ValueError("配对开发样本集合不一致")
    paired = []
    for row in swinir_rows:
        baseline = baseline_by_id[row["sample_id"]]
        output = {"sample_id": row["sample_id"], "source_group_id": row["source_group_id"]}
        for metric in METRIC_NAMES:
            output[f"swinir_{metric}"] = row[metric]
            output[f"internal_c2_{metric}"] = baseline[metric]
            output[f"delta_{metric}"] = float(row[metric]) - float(baseline[metric])
        paired.append(output)

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in paired:
        grouped.setdefault(row["source_group_id"], []).append(row)
    rng = np.random.default_rng(seed)
    summary = {}
    group_names = sorted(grouped)
    for metric in METRIC_NAMES:
        group_deltas = np.asarray(
            [np.mean([row[f"delta_{metric}"] for row in grouped[group]]) for group in group_names],
            dtype=np.float64,
        )
        draws = rng.choice(group_deltas, size=(10000, len(group_deltas)), replace=True).mean(axis=1)
        summary[metric] = {
            "group_balanced_delta": float(group_deltas.mean()),
            "group_bootstrap_95_ci": [
                float(np.quantile(draws, 0.025)),
                float(np.quantile(draws, 0.975)),
            ],
        }
    return paired, summary


def _baseline_model(checkpoint_path: Path, device: torch.device):
    from MMCANetV2.phase_c2_matched import C2MatchedJoint

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    context = checkpoint.get("context", {})
    if (
        context.get("condition_id") != "x2_light"
        or context.get("arm") != "sar_only"
        or context.get("optical_enabled") is not False
        or context.get("scale") != 2
        or checkpoint.get("format") != "c2_best_checkpoint_v1"
    ):
        raise ValueError("历史内部 SAR-only checkpoint 元数据不匹配")
    config = context.get("model_config", {})
    model = C2MatchedJoint(
        scale=2,
        optical_enabled=False,
        width=int(config["width"]),
        blocks=int(config["blocks"]),
        hr_width=int(config["hr_width"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.to(device).eval(), checkpoint


def _environment():
    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "pytorch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "platform": platform.platform(),
        "device": torch.cuda.get_device_name(0),
        "device_total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        "device_free_memory_bytes_before_run": int(torch.cuda.mem_get_info()[0]),
        "device_free_memory_bytes_after_run": None,
    }


def _git_state():
    def command(*args):
        return subprocess.check_output(["git", *args], cwd=PROJECT_ROOT, text=True).strip()

    status = command("status", "--short", "--untracked-files=all")
    return {
        "commit": command("rev-parse", "HEAD"),
        "branch": command("branch", "--show-current"),
        "dirty": bool(status),
        "status_entry_count": len(status.splitlines()),
    }


def run(args):
    input_root = Path(args.input_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"实验输出目录已存在，拒绝覆盖：{output_dir}")
    if not torch.cuda.is_available():
        raise RuntimeError("本实验固定要求 CUDA；当前 CUDA 不可用")
    if not input_root.is_dir():
        raise FileNotFoundError("C1 v2 固定输入目录不存在")
    baseline_checkpoint_path = Path(args.baseline_checkpoint).resolve()
    baseline_checkpoint_sha256 = sha256_file(baseline_checkpoint_path)
    if baseline_checkpoint_sha256 != EXPECTED_INTERNAL_BASELINE_SHA256:
        raise ValueError("内部 SAR-only baseline checkpoint 哈希不匹配")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda:0")
    environment = _environment()
    git = _git_state()

    train_records, train_manifest = load_split(
        input_root / "x2_light", "train", TRAIN_SAMPLES, TRAIN_GROUPS
    )
    development_records, development_manifest = load_split(
        input_root / "x2_light", "development", DEVELOPMENT_SAMPLES, DEVELOPMENT_GROUPS
    )
    train_groups = {row["source_group_id"] for row in train_records}
    development_groups = {row["source_group_id"] for row in development_records}
    if train_groups & development_groups:
        raise ValueError("train/development 来源场景组重叠")

    output_dir.mkdir(parents=True, exist_ok=False)
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir()
    protocol = {
        "协议编号": "SAR-only SwinIR 任务适配探索 v1",
        "研究对象": "C1 v2 固定 x2_light 输入下的 SAR LR 到 noisy HR pseudo-target 映射",
        "数据角色": {
            "train": {"样本数": TRAIN_SAMPLES, "来源组数": TRAIN_GROUPS},
            "development": {"样本数": DEVELOPMENT_SAMPLES, "来源组数": DEVELOPMENT_GROUPS},
            "confirmation_payload_accessed": False,
            "test_set_accessed": False,
            "optical_payload_loaded": False,
            "development_scope": "历史重复使用的开发集，只支持探索，不作为独立泛化证据",
        },
        "模型": {
            "结构": "SwinIR 轻量结构，embed_dim=60, depths=(6,), num_heads=(6,), window_size=8, mlp_ratio=2.0",
            "任务适配": "单通道 SAR 输入/输出、x2 pixelshuffledirect、Sigmoid 输出",
            "初始化": "随机初始化；不使用外部预训练权重",
            "实现来源": "仓库 BasicSR/basicsr/archs/swinir_arch.py",
            "方法边界": "发布结构的 SAR 任务适配实现；不是论文作者的 SAR 权重或完整复现",
        },
        "训练": {
            "seed": args.seed,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr_patch_size": args.patch_size,
            "target_patch_size": args.patch_size * SCALE,
            "loss": "L1",
            "optimizer": "Adam",
            "learning_rate": args.learning_rate,
            "weight_decay": 0.0,
            "augmentation": "仅 train 使用随机裁剪及水平翻转、垂直翻转、转置；development 不增强",
            "data_loader_workers": 0,
            "amp": False,
            "deterministic_algorithms": True,
        },
        "checkpoint_selection": "每轮完整 development 推理，按来源组等权的平均 PSNR 最大值选择；同分保留较早轮次",
        "final_evaluation": {
            "evaluator": "MMCANetV2/phase_c3a_metrics.py 的表示域指标",
            "metrics": list(METRIC_NAMES),
            "input_target_range": "HR target data_range=1.0；LR 保留冻结表示域数值，不额外归一化",
            "bright_mask": "固定 LR 99.5 百分位阈值并按 x2 最近邻展开",
            "background_mask": "固定 LR 20 百分位阈值并按 x2 最近邻展开",
            "edge_mask": "HR noisy pseudo-target Sobel 幅值 90 百分位",
            "paired_bootstrap": "先按来源组聚合配对差，再以来源组为单位进行 10000 次 bootstrap",
        },
        "resource_policy": "按用户明确指示，与当时已有 CUDA 评价进程并行启动；不采用 3 GiB 空闲显存门槛，以实际 CUDA OOM 为停止边界，不终止外部进程",
    }
    protocol_bytes = (json.dumps(protocol, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
    protocol_path = output_dir / "protocol.json"
    protocol_path.write_bytes(protocol_bytes)
    protocol_sha256 = hashlib.sha256(protocol_bytes).hexdigest()
    environment["device_free_memory_bytes_after_run"] = None
    (output_dir / "environment.json").write_text(
        json.dumps(environment, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    manifest = {
        "format": "sar_only_swinir_task_adaptation_run_v1",
        "run_id": output_dir.name,
        "status": "running",
        "started_unix": time.time(),
        "protocol_sha256": protocol_sha256,
        "git": git,
        "environment": environment,
        "input_root": str(input_root),
        "inputs": {"train": train_manifest, "development": development_manifest},
        "group_disjoint": True,
        "baseline_checkpoint_path": str(baseline_checkpoint_path),
        "baseline_checkpoint_sha256": baseline_checkpoint_sha256,
        "test_set_accessed": False,
        "test_set_used": False,
        "confirmation_payload_accessed": False,
        "optical_payload_loaded": False,
        "outputs": [],
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    model = None
    started_training = time.perf_counter()
    try:
        SwinIR = load_swinir_class()
        model = build_swinir(SwinIR).to(device)
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=0.0)
        best_score = -math.inf
        best_epoch = None
        train_log = []
        selection_log = []
        torch.cuda.reset_peak_memory_stats()

        for epoch in range(1, args.epochs + 1):
            model.train()
            rng = np.random.default_rng(args.seed + epoch)
            order_rng = np.random.default_rng(args.seed * 10 + epoch)
            order = order_rng.permutation(len(train_records))
            loss_sum = 0.0
            image_count = 0
            epoch_started = time.perf_counter()
            for start in range(0, len(order), args.batch_size):
                batch_records = [train_records[int(index)] for index in order[start : start + args.batch_size]]
                crops = [_augment_crop(record, rng, args.patch_size) for record in batch_records]
                lr_batch = torch.from_numpy(np.stack([pair[0] for pair in crops])).to(device)
                target_batch = torch.from_numpy(np.stack([pair[1] for pair in crops])).to(device)
                optimizer.zero_grad(set_to_none=True)
                prediction = model(lr_batch)
                loss = nn.functional.l1_loss(prediction, target_batch)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"第 {epoch} 轮训练损失不是有限值")
                loss.backward()
                gradient_checks = [
                    torch.isfinite(parameter.grad).all()
                    for parameter in model.parameters()
                    if parameter.grad is not None
                ]
                if gradient_checks and not bool(torch.stack(gradient_checks).all().item()):
                    raise FloatingPointError(f"第 {epoch} 轮梯度含非有限值")
                optimizer.step()
                loss_sum += float(loss.detach()) * len(batch_records)
                image_count += len(batch_records)

            selection_rows = evaluate_psnr(model, development_records, device)
            group_balanced_psnr = group_balanced_mean(selection_rows, "psnr_db")
            epoch_seconds = time.perf_counter() - epoch_started
            train_row = {
                "epoch": epoch,
                "train_l1_mean": loss_sum / image_count,
                "development_group_balanced_psnr_db": group_balanced_psnr,
                "development_sample_mean_psnr_db": float(np.mean([row["psnr_db"] for row in selection_rows])),
                "epoch_seconds": epoch_seconds,
                "peak_allocated_mib": torch.cuda.max_memory_allocated() / (1024**2),
                "peak_reserved_mib": torch.cuda.max_memory_reserved() / (1024**2),
            }
            train_log.append(train_row)
            selection_log.extend({"epoch": epoch, **row} for row in selection_rows)
            checkpoint = {
                "format": "sar_only_swinir_task_adaptation_checkpoint_v1",
                "epoch": epoch,
                "best_epoch": best_epoch,
                "best_development_group_balanced_psnr_db": best_score,
                "model_config": model.experiment_config,
                "optimizer_config": {"type": "Adam", "learning_rate": args.learning_rate, "weight_decay": 0.0},
                "seed": args.seed,
                "protocol_sha256": protocol_sha256,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state_all": torch.cuda.get_rng_state_all(),
            }
            torch.save(checkpoint, checkpoints_dir / "last.pt")
            if group_balanced_psnr > best_score:
                best_score = group_balanced_psnr
                best_epoch = epoch
                checkpoint["best_epoch"] = best_epoch
                checkpoint["best_development_group_balanced_psnr_db"] = best_score
                torch.save(checkpoint, checkpoints_dir / "best.pt")
            print(
                f"第 {epoch:02d}/{args.epochs} 轮：训练 L1={train_row['train_l1_mean']:.6f}，"
                f"开发组均衡 PSNR={group_balanced_psnr:.4f} dB，"
                f"峰值显存={train_row['peak_allocated_mib']:.1f} MiB，耗时={epoch_seconds:.1f} 秒",
                flush=True,
            )

        _write_csv(output_dir / "train_log.csv", train_log)
        _write_csv(output_dir / "development_selection_per_image.csv", selection_log)
        checkpoint = torch.load(checkpoints_dir / "best.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.to(device).eval()
        swinir_rows, swinir_runtime = _evaluate_full(
            model, development_records, device, output_dir / "development_predictions_swinir"
        )
        _write_csv(output_dir / "development_metrics_swinir.csv", swinir_rows)

        baseline_path = Path(args.baseline_checkpoint).resolve()
        baseline_model, baseline_checkpoint = _baseline_model(baseline_path, device)
        baseline_rows, baseline_runtime = _evaluate_full(
            baseline_model,
            development_records,
            device,
            output_dir / "development_predictions_internal_c2",
        )
        _write_csv(output_dir / "development_metrics_internal_c2.csv", baseline_rows)
        paired_rows, paired_summary = _paired_comparison(
            swinir_rows, baseline_rows, seed=args.seed + 999
        )
        _write_csv(output_dir / "paired_development_comparison.csv", paired_rows)
        group_stats = _group_statistics({"SwinIR_SAR_adaptation": swinir_rows, "internal_C2_SAR_only": baseline_rows})
        _write_csv(output_dir / "source_group_stats.csv", group_stats)

        summary = {
            "scope": "单 seed、重复使用的 development；仅探索，不作独立泛化或投稿主张",
            "checkpoint_selection": {"best_epoch": best_epoch, "development_group_balanced_psnr_db": best_score},
            "SwinIR_SAR_adaptation": _summary(swinir_rows),
            "internal_C2_SAR_only": _summary(baseline_rows),
            "paired_group_bootstrap": paired_summary,
            "baseline_lineage": "项目内部 C2MatchedJoint，不是发表模型或强外部 baseline",
            "stability": {
                "SwinIR_repeat_inference_max_abs_difference": swinir_runtime["repeat_inference_max_abs_difference"],
                "internal_C2_repeat_inference_max_abs_difference": baseline_runtime["repeat_inference_max_abs_difference"],
                "diffusion_sample_distribution_evaluated": False,
            },
            "interpretation_limits": [
                "开发集此前已重复使用，存在历史选择污染风险。",
                "单 seed 结果只适合作为探索观察。",
                "HR 目标为归一化 8-bit 表示域 noisy pseudo-target，不代表 clean SAR 或辐射定标量。",
                "SwinIR 为发布架构的 SAR 单通道任务适配实现；没有使用作者 SAR 权重，也不是完整复现。",
                "内部 C2 baseline 不等于已发表强 baseline。",
            ],
        }
        (output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

        torch.cuda.synchronize()
        runtime = {
            "training_seconds": time.perf_counter() - started_training,
            "training_epochs": args.epochs,
            "training_samples_per_epoch": len(train_records),
            "training_updates": args.epochs * math.ceil(len(train_records) / args.batch_size),
            "swinir_parameter_count": parameter_count,
            "internal_c2_parameter_count": sum(parameter.numel() for parameter in baseline_model.parameters()),
            "swinir_evaluation": swinir_runtime,
            "internal_c2_evaluation": baseline_runtime,
            "peak_allocated_mib": torch.cuda.max_memory_allocated() / (1024**2),
            "peak_reserved_mib": torch.cuda.max_memory_reserved() / (1024**2),
            "cuda_free_memory_bytes_after_run": int(torch.cuda.mem_get_info()[0]),
            "disk_free_bytes_after_run": shutil.disk_usage(output_dir).free,
        }
        (output_dir / "runtime.json").write_text(
            json.dumps(runtime, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        manifest.update(
            {
                "status": "complete",
                "ended_unix": time.time(),
                "best_epoch": best_epoch,
                "best_development_group_balanced_psnr_db": best_score,
                "runtime_path": "runtime.json",
                "protocol_path": "protocol.json",
                "code_sha256": sha256_file(Path(__file__)),
                "swinir_source_sha256": sha256_file(SOURCE_ARCH),
                "baseline_checkpoint_sha256_verified": sha256_file(baseline_path),
                "baseline_checkpoint_epoch": baseline_checkpoint["epoch"],
                "outputs": sorted(path.relative_to(output_dir).as_posix() for path in output_dir.rglob("*") if path.is_file()),
            }
        )
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"实验完成：最佳轮次 {best_epoch}，输出目录 {output_dir}", flush=True)
        return summary
    except BaseException as error:
        manifest["status"] = "failed"
        manifest["ended_unix"] = time.time()
        manifest["failure_type"] = type(error).__name__
        manifest["failure_message"] = str(error)
        manifest["checkpoint_preserved"] = (checkpoints_dir / "last.pt").is_file()
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        raise


def main():
    parser = argparse.ArgumentParser(description="训练可复现的 SAR-only SwinIR 任务适配实验")
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--baseline-checkpoint", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--patch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20261123)
    args = parser.parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.patch_size <= 0:
        parser.error("训练轮数、批次和裁剪尺寸必须为正整数")
    run(args)


if __name__ == "__main__":
    main()
