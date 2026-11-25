"""在 C1 v2 固定 train/development SAR 输入上训练 EDSR 任务适配基线。"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import random
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.phase_d1_edsr_20261124_v1.edsr_sar import (  # noqa: E402
    EDSR_SOURCE,
    build_edsr_sar,
    load_published_edsr_class,
)
from experiments.published_baselines_sar_only_20261123_v1 import train_swinir_sar as common  # noqa: E402

PROTOCOL = PROJECT_ROOT / "docs/research/PHASE_D1_EDSR_SAR_ADAPTATION_PROTOCOL_20261124.md"
DEFAULT_OUTPUT = PROJECT_ROOT / "experiments/runs/phase_d1_edsr_sar_20261124_v1"
SEED = 20261125
EPOCHS = 30
BATCH_SIZE = 4
PATCH_SIZE = 64
LEARNING_RATE = 1e-4
BOOTSTRAP_SEED = 20261125


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_state():
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--short", "--untracked-files=all"],
            cwd=PROJECT_ROOT, check=True, capture_output=True, text=True,
        ).stdout
        return {"head": head, "dirty": bool(status), "status_short": status}
    except (OSError, subprocess.CalledProcessError) as exc:
        return {"head": None, "dirty": True, "error": str(exc)}


def _driver_info():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            check=True, capture_output=True, text=True,
        )
        return result.stdout.strip().splitlines()[0]
    except (OSError, subprocess.CalledProcessError, IndexError):
        return None


def _metrics_for_saved_predictions(records, prediction_root: Path):
    rows = []
    expected = {record["sample_id"] for record in records}
    actual = {path.stem for path in prediction_root.glob("*.npy")}
    if actual != expected:
        raise ValueError("SwinIR 预测文件集合与固定 development 不完全一致")
    for record in records:
        sample_id = record["sample_id"]
        prediction = np.load(prediction_root / f"{sample_id}.npy", allow_pickle=False)
        if (
            prediction.dtype != np.float32
            or prediction.shape != (256, 256)
            or not np.isfinite(prediction).all()
            or prediction.min() < 0.0
            or prediction.max() > 1.0
        ):
            raise ValueError(f"SwinIR 预测数值契约错误：{sample_id}")
        target = record["target"][0]
        masks, thresholds = common.build_fixed_masks(record["lr"], record["target"], common.SCALE)
        edge = common.edge_metrics(prediction, target, masks["edge"])
        bright_pred, bright_target = prediction[masks["high_return"]], target[masks["high_return"]]
        background_pred = prediction[masks["background"]]
        background_target = target[masks["background"]]
        row = {
            "sample_id": sample_id,
            "source_group_id": record["source_group_id"],
            "patch_id": record["patch_id"],
            "psnr_db": common._psnr(prediction, target),
            "ssim": common.ssim(prediction, target, data_range=1.0),
            "epi_corr": edge["epi_corr"],
            "edge_strength_ratio": edge["edge_strength_ratio"],
            "bright_mean_ratio": float(bright_pred.mean() / (bright_target.mean() + 1e-12)),
            "bright_p95_ratio": float(
                np.percentile(bright_pred, 95, method="linear")
                / (np.percentile(bright_target, 95, method="linear") + 1e-12)
            ),
            "background_mean_ratio": float(background_pred.mean() / (background_target.mean() + 1e-12)),
            "high_return_threshold": thresholds["high_return"],
            "background_threshold": thresholds["background"],
        }
        if not all(math.isfinite(float(row[name])) for name in common.METRIC_NAMES):
            raise FloatingPointError(f"SwinIR 指标非有限：{sample_id}")
        rows.append(row)
    return rows


def _paired_group_bootstrap(method_rows, reference_rows):
    reference = {row["sample_id"]: row for row in reference_rows}
    if set(reference) != {row["sample_id"] for row in method_rows}:
        raise ValueError("EDSR 与 SwinIR 的 sample_id 集合不一致")
    groups = {}
    paired_rows = []
    for row in method_rows:
        baseline = reference[row["sample_id"]]
        if row["source_group_id"] != baseline["source_group_id"]:
            raise ValueError("EDSR 与 SwinIR 的来源组标识不一致")
        paired = {
            "sample_id": row["sample_id"],
            "source_group_id": row["source_group_id"],
        }
        for metric in common.METRIC_NAMES:
            paired[f"edsr_{metric}"] = float(row[metric])
            paired[f"swinir_{metric}"] = float(baseline[metric])
            paired[f"delta_{metric}"] = float(row[metric]) - float(baseline[metric])
        paired_rows.append(paired)
        groups.setdefault(row["source_group_id"], []).append(paired)

    result = {}
    group_names = sorted(groups)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    for metric in common.METRIC_NAMES:
        values = np.asarray(
            [np.mean([row[f"delta_{metric}"] for row in groups[group]]) for group in group_names],
            dtype=np.float64,
        )
        samples = rng.integers(0, len(values), size=(10_000, len(values)))
        draws = values[samples].mean(axis=1)
        result[metric] = {
            "来源组数": len(group_names),
            "来源组等权差": float(values.mean()),
            "来源组bootstrap_95CI": [
                float(np.quantile(draws, 0.025)),
                float(np.quantile(draws, 0.975)),
            ],
        }
    return paired_rows, result


def _write_csv(path: Path, rows):
    if not rows:
        raise ValueError(f"拒绝写入空表：{path.name}")
    with path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_checkpoint(path, model, optimizer, epoch, best_epoch, best_score):
    torch.save(
        {
            "format": "task_adapted_edsr_sar_checkpoint_v1",
            "epoch": int(epoch),
            "best_epoch": int(best_epoch),
            "best_development_group_balanced_psnr_db": float(best_score),
            "seed": SEED,
            "model_config": model.experiment_config,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": torch.cuda.get_rng_state_all(),
            "numpy_rng_state": np.random.get_state(),
            "python_rng_state": random.getstate(),
        },
        path,
    )


def _write_rows_incrementally(path: Path, fieldnames, rows):
    create = not path.exists()
    with path.open("a", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        if create:
            writer.writeheader()
        if isinstance(rows, dict):
            writer.writerow(rows)
        else:
            writer.writerows(rows)
        stream.flush()


def run(input_root: Path, output_dir: Path, swinir_prediction_root: Path):
    input_root = input_root.resolve()
    output_dir = output_dir.resolve()
    swinir_prediction_root = swinir_prediction_root.resolve()
    if output_dir.exists():
        raise FileExistsError(f"输出目录已存在，拒绝覆盖：{output_dir}")
    if not torch.cuda.is_available():
        raise RuntimeError("EDSR 冻结训练协议要求 CUDA，不回退 CPU")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    train_records, train_meta = common.load_split(input_root, "train", 1600, 92)
    dev_records, dev_meta = common.load_split(input_root, "development", 200, 19)
    if {row["source_group_id"] for row in train_records} & {
        row["source_group_id"] for row in dev_records
    }:
        raise ValueError("train/development 来源组有交集")
    if not swinir_prediction_root.is_dir():
        raise FileNotFoundError(f"缺少固定 SwinIR predictions：{swinir_prediction_root}")

    protocol_hash = sha256_file(PROTOCOL)
    edsr_source_hash = sha256_file(EDSR_SOURCE)
    registry_source = PROJECT_ROOT / "BasicSR/basicsr/utils/registry.py"
    registry_hash = sha256_file(registry_source)
    disk_before = shutil.disk_usage(output_dir.parent)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    output_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir()
    git_state = _git_state()
    model = None
    start_time = time.time()
    manifest = {
        "run_id": output_dir.name,
        "method": "EDSR SAR 单通道 x2 任务适配版",
        "method_identity": "TASK_ADAPTED_PUBLISHED",
        "status": "running",
        "seed": SEED,
        "protocol_sha256": protocol_hash,
        "edsr_source_sha256": edsr_source_hash,
        "registry_source_sha256": registry_hash,
        "git": git_state,
        "input_root": str(input_root),
        "train": train_meta,
        "development": dev_meta,
        "swinir_prediction_root": str(swinir_prediction_root),
        "test_set_accessed": False,
        "test_set_used": False,
        "confirmation_payload_accessed": False,
        "optical_payload_loaded": False,
        "gpu": torch.cuda.get_device_name(device),
        "driver": _driver_info(),
        "torch": torch.__version__,
        "torchvision": __import__("torchvision").__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "memory_total_bytes": int(torch.cuda.mem_get_info(device)[1]),
        "memory_free_before_bytes": int(torch.cuda.mem_get_info(device)[0]),
        "disk_free_before_bytes": int(disk_before.free),
        "completed_epochs": 0,
        "best_epoch": None,
        "cuda_oom": False,
        "command": [sys.executable, *sys.argv],
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "protocol.md").write_bytes(PROTOCOL.read_bytes())
    config = {
        "method": "EDSR SAR 单通道 ×2 TASK_ADAPTED_PUBLISHED",
        "seed": SEED,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "lr_patch": PATCH_SIZE,
        "target_patch": PATCH_SIZE * 2,
        "num_feat": 64,
        "num_block": 16,
        "res_scale": 1,
        "optimizer": "Adam",
        "learning_rate": LEARNING_RATE,
        "loss": "L1",
        "output": "Sigmoid [0,1]",
        "checkpoint_selection": "development 来源组等权 PSNR 最大值",
        "protocol_sha256": protocol_hash,
    }
    (output_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "command.txt").write_text(" ".join(manifest["command"]) + "\n", encoding="utf-8")
    (output_dir / "git_state.txt").write_text(
        f"HEAD：{git_state.get('head')}\n工作区 dirty：{git_state.get('dirty')}\n\n"
        + git_state.get("status_short", git_state.get("error", "")),
        encoding="utf-8",
    )
    environment = (
        f"Python：{platform.python_version()}\nPython 可执行文件：{sys.executable}\n"
        f"PyTorch：{torch.__version__}\ntorchvision：{__import__('torchvision').__version__}\n"
        f"CUDA 运行时：{torch.version.cuda}\ncuDNN：{torch.backends.cudnn.version()}\n"
        f"GPU：{torch.cuda.get_device_name(device)}\n驱动：{manifest['driver']}\n"
        f"启动前空闲显存 MiB：{manifest['memory_free_before_bytes'] / 2**20:.1f}\n"
        f"启动前分区可用 GiB：{disk_before.free / 2**30:.2f}\n"
    )
    (output_dir / "environment.txt").write_text(environment, encoding="utf-8")

    try:
        edsr_class = load_published_edsr_class()
        model = build_edsr_sar(edsr_class).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=0.0)
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        try:
            torch.cuda.reset_peak_memory_stats(device)
        except RuntimeError as exc:
            manifest["peak_stats_reset_error"] = str(exc)
        best_score = -math.inf
        best_epoch = 0
        train_fields = (
            "epoch", "train_l1_mean", "development_group_balanced_psnr_db",
            "development_sample_mean_psnr_db", "updates", "epoch_seconds",
            "peak_allocated_mib", "peak_reserved_mib",
        )
        log_path = output_dir / "train_log.csv"
        started = time.perf_counter()
        total_updates = 0

        for epoch in range(1, EPOCHS + 1):
            model.train()
            crop_rng = np.random.default_rng(SEED + epoch)
            order_rng = np.random.default_rng(SEED * 10 + epoch)
            order = order_rng.permutation(len(train_records))
            loss_sum = 0.0
            samples_seen = 0
            epoch_start = time.perf_counter()
            epoch_updates = 0
            for offset in range(0, len(order), BATCH_SIZE):
                batch = [train_records[int(index)] for index in order[offset : offset + BATCH_SIZE]]
                crops = [common._augment_crop(record, crop_rng, PATCH_SIZE) for record in batch]
                lr = torch.from_numpy(np.stack([pair[0] for pair in crops])).to(device)
                target = torch.from_numpy(np.stack([pair[1] for pair in crops])).to(device)
                optimizer.zero_grad(set_to_none=True)
                prediction = model(lr)
                loss = nn.functional.l1_loss(prediction, target)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"第 {epoch} 轮训练损失非有限")
                loss.backward()
                gradients = [p.grad for p in model.parameters() if p.grad is not None]
                if not gradients or not all(torch.isfinite(value).all() for value in gradients):
                    raise FloatingPointError(f"第 {epoch} 轮梯度缺失或非有限")
                optimizer.step()
                loss_sum += float(loss.detach()) * len(batch)
                samples_seen += len(batch)
                epoch_updates += 1
                total_updates += 1

            selection_rows = common.evaluate_psnr(model, dev_records, device)
            group_psnr = common.group_balanced_mean(selection_rows, "psnr_db")
            sample_psnr = float(np.mean([row["psnr_db"] for row in selection_rows]))
            epoch_seconds = time.perf_counter() - epoch_start
            row = {
                "epoch": epoch,
                "train_l1_mean": loss_sum / samples_seen,
                "development_group_balanced_psnr_db": group_psnr,
                "development_sample_mean_psnr_db": sample_psnr,
                "updates": epoch_updates,
                "epoch_seconds": epoch_seconds,
                "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20,
                "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20,
            }
            _write_rows_incrementally(log_path, train_fields, row)
            _write_rows_incrementally(
                output_dir / "development_selection_per_image.csv",
                ("epoch", "sample_id", "source_group_id", "psnr_db"),
                [
                    {"epoch": epoch, **selection}
                    for selection in selection_rows
                ],
            )
            if group_psnr > best_score:
                best_score, best_epoch = group_psnr, epoch
                _write_checkpoint(
                    checkpoint_dir / "best.pt", model, optimizer, epoch, best_epoch, best_score
                )
            _write_checkpoint(
                checkpoint_dir / "last.pt", model, optimizer, epoch, best_epoch, best_score
            )
            manifest.update(
                completed_epochs=epoch,
                best_epoch=best_epoch,
                best_development_group_balanced_psnr_db=best_score,
                total_updates=total_updates,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
                elapsed_seconds=time.perf_counter() - started,
            )
            manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            print(
                f"第 {epoch:02d}/{EPOCHS} 轮：训练 L1={row['train_l1_mean']:.6f}，"
                f"开发组等权 PSNR={group_psnr:.4f} dB，峰值显存={row['peak_allocated_mib']:.1f} MiB，"
                f"耗时={epoch_seconds:.1f} 秒。",
                flush=True,
            )

        best = torch.load(checkpoint_dir / "best.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(best["model_state_dict"], strict=True)
        model.to(device).eval()
        edsr_rows, edsr_runtime = common._evaluate_full(
            model, dev_records, device, output_dir / "development_predictions_edsr"
        )
        _write_csv(output_dir / "development_metrics_edsr.csv", edsr_rows)
        swinir_rows = _metrics_for_saved_predictions(dev_records, swinir_prediction_root)
        _write_csv(output_dir / "development_metrics_swinir_reference.csv", swinir_rows)
        paired_rows, paired_summary = _paired_group_bootstrap(edsr_rows, swinir_rows)
        _write_csv(output_dir / "paired_development_comparison.csv", paired_rows)
        group_rows = common._group_statistics({"EDSR_TASK_ADAPTED": edsr_rows, "SwinIR_TASK_ADAPTED": swinir_rows})
        _write_csv(output_dir / "source_group_stats.csv", group_rows)
        summary = {
            "scope": "单 seed、历史重复使用的 development；仅探索，不作独立泛化或投稿主张",
            "method_identity": "TASK_ADAPTED_PUBLISHED",
            "parameter_count": parameter_count,
            "development_group_balanced": {
                "edsr": common._summary(edsr_rows)["group_balanced_mean"],
                "swinir": common._summary(swinir_rows)["group_balanced_mean"],
            },
            "paired_edsr_minus_swinir": paired_summary,
            "checkpoint_selection": {
                "best_epoch": best_epoch,
                "best_development_group_balanced_psnr_db": best_score,
            },
            "interpretation_limits": [
                "C1 v2 development 曾重复用于训练/选择，只支撑探索观察。",
                "EDSR 保留已发表主干结构，但输入通道、表示范围和 Sigmoid 输出经过 SAR 任务适配。",
                "EDSR 未使用作者 SAR 权重，也不是原论文完整训练方案复现。",
                "HR 目标为归一化 8-bit 表示域 noisy pseudo-target，不代表 clean SAR。",
            ],
        }
        (output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        torch.cuda.synchronize(device)
        free_after, _ = torch.cuda.mem_get_info(device)
        disk_after = shutil.disk_usage(output_dir.parent)
        runtime = {
            "training_epochs": EPOCHS,
            "training_samples_per_epoch": len(train_records),
            "training_updates": total_updates,
            "training_seconds": time.perf_counter() - started,
            "parameter_count": parameter_count,
            "edsr_inference": edsr_runtime,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
            "memory_free_after_bytes": int(free_after),
            "disk_free_before_bytes": int(disk_before.free),
            "disk_free_after_bytes": int(disk_after.free),
        }
        (output_dir / "runtime.json").write_text(
            json.dumps(runtime, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        manifest.update(
            status="complete",
            ended_unix=time.time(),
            best_epoch=best_epoch,
            best_development_group_balanced_psnr_db=best_score,
            parameter_count=parameter_count,
            total_updates=total_updates,
            training_seconds=runtime["training_seconds"],
            edsr_source_sha256=edsr_source_hash,
            adapter_source_sha256=sha256_file(Path(__file__).with_name("edsr_sar.py")),
            training_code_sha256=sha256_file(Path(__file__).resolve()),
            best_checkpoint_sha256=sha256_file(checkpoint_dir / "best.pt"),
            last_checkpoint_sha256=sha256_file(checkpoint_dir / "last.pt"),
            summary_sha256=sha256_file(output_dir / "summary.json"),
            peak_allocated_bytes=runtime["peak_allocated_bytes"],
            peak_reserved_bytes=runtime["peak_reserved_bytes"],
            memory_free_after_bytes=int(free_after),
            disk_free_after_bytes=int(disk_after.free),
            cuda_oom=False,
            outputs=sorted(path.relative_to(output_dir).as_posix() for path in output_dir.rglob("*") if path.is_file()),
        )
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"EDSR 训练和评价完成；最佳轮次 {best_epoch}，输出目录：{output_dir}", flush=True)
        return summary
    except BaseException as exc:
        is_oom = isinstance(exc, torch.cuda.OutOfMemoryError) or (
            isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()
        )
        manifest.update(
            status="cuda_oom" if is_oom else "failed",
            failure_type=type(exc).__name__,
            failure_message=str(exc),
            traceback=traceback.format_exc(),
            ended_unix=time.time(),
            completed_epochs=manifest.get("completed_epochs", 0),
            cuda_oom=is_oom,
        )
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--swinir-prediction-root", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run(args.input_root, args.output_dir, args.swinir_prediction_root)


if __name__ == "__main__":
    main()
