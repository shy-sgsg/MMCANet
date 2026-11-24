"""按冻结设置恢复被中断的 SAR-only 扩散探索运行。"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import train_sar_only_diffusion as experiment
from train_swinir_sar import sha256_file


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _load_records(run_dir: Path, with_base: bool):
    manifest = _read_json(run_dir / "manifest.json")
    input_root = Path(manifest["input_root"])
    train, train_manifest = experiment.load_split(input_root / "x2_light", "train", 1600, 92)
    development, development_manifest = experiment.load_split(
        input_root / "x2_light", "development", 200, 19
    )
    for role, value in (("train", train_manifest), ("development", development_manifest)):
        expected = experiment.EXPECTED_INPUTS[role]
        if value["manifest_sha256"] != expected["manifest_sha256"] or value["files_csv_sha256"] != expected["files_csv_sha256"]:
            raise ValueError(f"{role} 固定输入哈希不匹配")
    if {row["source_group_id"] for row in train} & {row["source_group_id"] for row in development}:
        raise ValueError("train/development 来源组重叠")
    if with_base:
        for role, records in (("train", train), ("development", development)):
            base_dir = run_dir / "swinir_base_predictions" / role
            for record in records:
                path = base_dir / f"{record['sample_id']}.npy"
                if not path.is_file():
                    raise FileNotFoundError(f"缺失已保存的 SwinIR base 图：{path.name}")
                base = np.load(path, allow_pickle=False)
                if base.shape != record["target"].shape[-2:] or not np.isfinite(base).all():
                    raise ValueError(f"SwinIR base 图尺寸或数值无效：{path.name}")
                record["base"] = base[None].astype(np.float32, copy=False)
    return manifest, train, development


def _configuration(run_dir: Path):
    config = _read_json(run_dir / "config.json")
    config["patch_size"] = config["patch_size_lr"]
    return argparse.Namespace(**config)


def _swinir_reference_run(run_dir: Path):
    config = _read_json(run_dir / "config.json")
    if not config.get("swinir_run"):
        raise ValueError("冻结配置缺少 SwinIR 参考运行目录")
    return Path(config["swinir_run"])


def _parse_training_history(log_path: Path, variant: str, expected_epoch: int, total_epochs: int):
    pattern = re.compile(
        rf"^{re.escape(variant)} 第 (?P<epoch>\d+)/(?P<total>\d+) 轮：训练 epsilon MSE="
        r"(?P<train>[0-9.]+)，开发组均衡 MSE=(?P<development>[0-9.]+)，"
        r"峰值显存=(?P<allocated>[0-9.]+) MiB，耗时=(?P<seconds>[0-9.]+) 秒$"
    )
    if not log_path.is_file():
        raise FileNotFoundError(f"训练控制台日志不存在：{log_path}")
    parsed = {}
    for line_number, line in enumerate(log_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.startswith(f"{variant} 第 "):
            continue
        match = pattern.fullmatch(line.strip())
        if not match:
            raise ValueError(f"控制台日志第 {line_number} 行格式无法识别：{line}")
        values = match.groupdict()
        epoch, recorded_total = int(values["epoch"]), int(values["total"])
        if recorded_total != total_epochs:
            raise ValueError(f"控制台日志训练总轮数不匹配：{recorded_total} != {total_epochs}")
        if epoch in parsed:
            raise ValueError(f"控制台日志重复记录 {variant} 第 {epoch} 轮")
        parsed[epoch] = {
            "epoch": epoch,
            "train_epsilon_mse": float(values["train"]),
            "development_group_balanced_epsilon_mse": float(values["development"]),
            "epoch_seconds": float(values["seconds"]),
            "peak_allocated_mib": float(values["allocated"]),
            # 早期终端摘要只输出 allocated；reserved 在训练进程终止前没有落盘。
            "peak_reserved_mib": "",
        }
    expected = set(range(1, expected_epoch + 1))
    if set(parsed) != expected:
        missing = sorted(expected - set(parsed))
        extra = sorted(set(parsed) - expected)
        raise ValueError(f"{variant} 历史训练轮次不连续；缺失={missing}，超出 checkpoint={extra}")
    return [parsed[epoch] for epoch in range(1, expected_epoch + 1)]


def _deterministic_setup(seed: int):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def _load_ema(run_dir: Path, variant: str, device: torch.device):
    checkpoint_path = run_dir / variant / "checkpoints" / "best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    manifest = _read_json(run_dir / "manifest.json")
    if (
        checkpoint.get("format") != "sar_only_conditional_diffusion_v1"
        or checkpoint.get("variant") != variant
        or checkpoint.get("seed") != 20261124
        or checkpoint.get("protocol_sha256") != manifest["protocol_sha256"]
    ):
        raise ValueError(f"{variant} checkpoint 与冻结协议不匹配")
    model = experiment.ConditionalUNet().to(device)
    model.load_state_dict(checkpoint["ema_state_dict"], strict=True)
    return model.eval(), checkpoint


def _backup_partial_predictions(run_dir: Path, variant: str):
    prediction_dir = run_dir / "predictions" / variant
    if not prediction_dir.exists():
        return None
    count = len(list(prediction_dir.glob("*.npy")))
    suffix = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    backup = prediction_dir.with_name(f"{variant}_中断部分_{count}张_{suffix}")
    shutil.move(str(prediction_dir), str(backup))
    return backup.relative_to(run_dir).as_posix()


def _begin_stage(run_dir: Path, stage_id: str, description: str):
    manifest = _read_json(run_dir / "manifest.json")
    manifest["status"] = "running"
    manifest["active_recovery_stage"] = stage_id
    manifest["resumption_status"] = description
    _write_json(run_dir / "manifest.json", manifest)
    record = {
        "阶段": stage_id,
        "状态": "运行中",
        "说明": description,
        "开始时间 Unix": time.time(),
        "命令": " ".join(sys.argv),
        "CUBLAS_WORKSPACE_CONFIG": __import__("os").environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }
    recovery_dir = run_dir / "恢复记录"
    recovery_dir.mkdir(exist_ok=True)
    suffix = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    file_path = recovery_dir / f"{stage_id}_{suffix}.json"
    _write_json(file_path, record)
    return manifest, record, file_path


def _finish_stage(run_dir: Path, manifest, record, file_path: Path, extra=None):
    record["状态"] = "完成"
    record["结束时间 Unix"] = time.time()
    if extra:
        record.update(extra)
    _write_json(file_path, record)
    manifest["active_recovery_stage"] = None
    manifest.setdefault("completed_recovery_stages", []).append(file_path.relative_to(run_dir).as_posix())
    manifest["resumption_status"] = "恢复工作继续进行"
    _write_json(run_dir / "manifest.json", manifest)


def run_stage(stage_id: str, run_dir: Path):
    run_dir = run_dir.resolve()
    manifest_path = run_dir / "manifest.json"
    manifest, args = _read_json(manifest_path), _configuration(run_dir)
    protocol_path = PROJECT_ROOT / manifest["protocol_path"]
    if sha256_file(protocol_path) != manifest["protocol_sha256"]:
        raise ValueError("冻结扩散协议哈希已变化，拒绝恢复")
    if not torch.cuda.is_available():
        raise RuntimeError("扩散恢复阶段要求 CUDA")
    device = torch.device("cuda:0")
    _deterministic_setup(args.seed)
    need_base = stage_id in {"residual_train", "residual_sample", "finalize"}
    manifest, train, development = _load_records(run_dir, with_base=need_base)
    residual_scale = float(manifest.get("train_only_residual_scale", 0.0))
    residual_rms = float(manifest.get("train_only_residual_rms", 0.0))
    if need_base:
        if residual_scale <= 0 or residual_rms <= 0:
            squared_errors = [float(np.mean((r["target"] - r["base"]) ** 2)) for r in train]
            residual_rms = math.sqrt(float(np.mean(squared_errors)))
            residual_scale = 1.0 / residual_rms
            manifest["train_only_residual_rms"] = residual_rms
            manifest["train_only_residual_scale"] = residual_scale
            _write_json(manifest_path, manifest)

    if stage_id == "conditional_sample":
        description = "从已完整训练的 conditional EMA checkpoint 重建完整 200 张 4-draw 评价"
        manifest, record, record_path = _begin_stage(run_dir, stage_id, description)
        backup = _backup_partial_predictions(run_dir, "conditional")
        model, checkpoint = _load_ema(run_dir, "conditional", device)
        alpha_bar = torch.cumprod(
            1.0 - experiment.make_beta_schedule(args.diffusion_steps, 1e-4, 0.02).to(device), dim=0
        )
        rows, means, distributions, runtime = experiment._sample_development(
            "conditional", model, development, 1.0, alpha_bar, args, run_dir, device
        )
        method_dir = run_dir / "conditional"
        experiment._write_csv(method_dir / "development_per_draw_metrics.csv", rows)
        experiment._write_csv(method_dir / "development_pixel_mean_metrics.csv", means)
        experiment._write_csv(method_dir / "development_distribution_scores.csv", distributions)
        runtime.update(
            {
                "best_epoch": checkpoint["epoch"],
                "checkpoint_best_epoch": checkpoint["epoch"],
                "peak_allocated_mib": torch.cuda.max_memory_allocated() / (1024**2),
                "peak_reserved_mib": torch.cuda.max_memory_reserved() / (1024**2),
            }
        )
        _write_json(method_dir / "sampling_runtime.json", runtime)
        _finish_stage(run_dir, manifest, record, record_path, {"已归档部分预测": backup, "完整样本数": len(means)})
        print(f"Conditional Diffusion 完整采样完成：{len(means)} 张 development。", flush=True)
        return

    if stage_id == "residual_train":
        description = "从第 24 轮 last checkpoint 恢复 optimizer 与 RNG，继续完成剩余冻结训练轮次"
        manifest, record, record_path = _begin_stage(run_dir, stage_id, description)
        failed_dir = run_dir / "residual"
        backup = None
        checkpoint_dir = failed_dir / "checkpoints"
        last_path, best_path = checkpoint_dir / "last.pt", checkpoint_dir / "best.pt"
        if not last_path.is_file() or not best_path.is_file():
            raise FileNotFoundError("缺少 residual last/best checkpoint，拒绝从空状态覆盖训练")
        last_checkpoint = torch.load(last_path, map_location="cpu", weights_only=False)
        best_checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
        expected_epoch = int(last_checkpoint["epoch"])
        if expected_epoch >= args.epochs:
            raise ValueError(f"last checkpoint 已到第 {expected_epoch} 轮，无需续训")
        if best_checkpoint.get("epoch", args.epochs + 1) > expected_epoch:
            raise ValueError("best checkpoint 轮次晚于 last checkpoint，拒绝续训")

        console_logs = sorted(run_dir.parent.glob("训练进度_diffusion_*.log"))
        matching_history = []
        for candidate in console_logs:
            try:
                rows = _parse_training_history(candidate, "residual", expected_epoch, args.epochs)
            except (ValueError, FileNotFoundError):
                continue
            matching_history.append((candidate, rows))
        if len(matching_history) != 1:
            raise ValueError(f"无法唯一确定覆盖第 1–{expected_epoch} 轮的控制台日志，候选数={len(matching_history)}")
        console_log, recovered_logs = matching_history[0]
        experiment._write_csv(failed_dir / "train_log.csv", recovered_logs)

        # 只有第 24 轮的 best/last checkpoint 被保留；据此重算其逐图选择指标。
        best_model, _ = _load_ema(run_dir, "residual", device)
        alpha_bar = torch.cumprod(
            1.0 - experiment.make_beta_schedule(args.diffusion_steps, 1e-4, 0.02).to(device), dim=0
        )
        validation = experiment._fixed_validation(
            development, "residual", residual_scale, alpha_bar, args.seed + 4000, args.patch_size, device
        )
        selection_rows, recovered_score = experiment._validate_epoch(best_model, validation, device)
        if not math.isclose(
            recovered_score,
            float(best_checkpoint["best_development_group_balanced_epsilon_mse"]),
            rel_tol=1e-5,
            abs_tol=1e-7,
        ):
            raise ValueError("从 best checkpoint 重算的开发评分与 checkpoint 元数据不一致")
        experiment._write_csv(
            failed_dir / "development_selection_per_image.csv",
            [{"epoch": int(best_checkpoint["epoch"]), **row} for row in selection_rows],
        )

        model, best_epoch, best_score, runtime, method_dir = experiment._train_variant(
            "residual", train, development, residual_scale, run_dir, args, device,
            manifest["protocol_sha256"], resume=True
        )
        runtime.update({
            "best_epoch": best_epoch,
            "best_development_group_balanced_epsilon_mse": best_score,
            "resume_from_epoch": expected_epoch,
            "recovered_training_log": console_log.relative_to(run_dir.parent).as_posix(),
            "recovered_per_image_validation_epochs": [int(best_checkpoint["epoch"])],
            "per_image_validation_epochs_unavailable": list(range(1, int(best_checkpoint["epoch"]))),
            "interruption_cause": "训练进程在第 24 轮 checkpoint 后消失；内核日志未发现 OOM 记录，CUDA OOM 未被证实",
        })
        _write_json(method_dir / "training_runtime.json", runtime)
        _finish_stage(run_dir, manifest, record, record_path, {
            "最佳轮次": best_epoch,
            "开发组均衡 epsilon MSE": best_score,
            "恢复起始轮次": expected_epoch,
            "训练日志来源": console_log.relative_to(run_dir.parent).as_posix(),
            "历史逐图验证不可恢复轮次": list(range(1, int(best_checkpoint["epoch"]))),
            "归档启动失败空目录": backup,
        })
        print(f"Residual Diffusion 完成 30 轮训练，最佳轮次 {best_epoch}。", flush=True)
        return

    if stage_id == "residual_sample":
        description = "从残差扩散 EMA checkpoint 完成 development 4-draw DDIM 采样"
        manifest, record, record_path = _begin_stage(run_dir, stage_id, description)
        backup = _backup_partial_predictions(run_dir, "residual")
        model, checkpoint = _load_ema(run_dir, "residual", device)
        alpha_bar = torch.cumprod(
            1.0 - experiment.make_beta_schedule(args.diffusion_steps, 1e-4, 0.02).to(device), dim=0
        )
        rows, means, distributions, runtime = experiment._sample_development(
            "residual", model, development, residual_scale, alpha_bar, args, run_dir, device
        )
        method_dir = run_dir / "residual"
        experiment._write_csv(method_dir / "development_per_draw_metrics.csv", rows)
        experiment._write_csv(method_dir / "development_pixel_mean_metrics.csv", means)
        experiment._write_csv(method_dir / "development_distribution_scores.csv", distributions)
        runtime.update(
            {
                "best_epoch": checkpoint["epoch"],
                "checkpoint_best_epoch": checkpoint["epoch"],
                "peak_allocated_mib": torch.cuda.max_memory_allocated() / (1024**2),
                "peak_reserved_mib": torch.cuda.max_memory_reserved() / (1024**2),
            }
        )
        _write_json(method_dir / "sampling_runtime.json", runtime)
        _finish_stage(run_dir, manifest, record, record_path, {"已归档部分预测": backup, "完整样本数": len(means)})
        print(f"Residual Diffusion 完整采样完成：{len(means)} 张 development。", flush=True)
        return

    if stage_id == "finalize":
        description = "汇总两种扩散条件的逐图指标、来源组统计、配对 bootstrap 与资源记录"
        manifest, record, record_path = _begin_stage(run_dir, stage_id, description)
        draw_rows, mean_rows, distribution_rows, variant_results = [], [], [], {}
        for variant in ("conditional", "residual"):
            method_dir = run_dir / variant
            method_draw = list(csv.DictReader((method_dir / "development_per_draw_metrics.csv").open(encoding="utf-8-sig", newline="")))
            method_mean = list(csv.DictReader((method_dir / "development_pixel_mean_metrics.csv").open(encoding="utf-8-sig", newline="")))
            method_distribution = list(csv.DictReader((method_dir / "development_distribution_scores.csv").open(encoding="utf-8-sig", newline="")))
            draw_rows.extend(method_draw)
            mean_rows.extend(method_mean)
            distribution_rows.extend(method_distribution)
            train_log = list(csv.DictReader((method_dir / "train_log.csv").open(encoding="utf-8-sig", newline="")))
            checkpoint = torch.load(method_dir / "checkpoints/best.pt", map_location="cpu", weights_only=False)
            if (method_dir / "training_runtime.json").exists():
                training_runtime = _read_json(method_dir / "training_runtime.json")
            else:
                training_runtime = {
                    "training_seconds": float(sum(float(row["epoch_seconds"]) for row in train_log)),
                    "optimizer_updates": args.epochs * math.ceil(1600 / (args.batch_size * args.accumulation_steps)),
                    "parameter_count": sum(value.numel() for value in checkpoint["ema_state_dict"].values()),
                    "peak_allocated_mib": max(float(row["peak_allocated_mib"]) for row in train_log),
                    "peak_reserved_mib": max(float(row["peak_reserved_mib"]) for row in train_log),
                    "runtime_source": "由逐轮日志重建；原首轮进程在写入总 runtime.json 前中断",
                }
            sampling = _read_json(method_dir / "sampling_runtime.json")
            single = [row for row in method_draw if row["method"] == variant]
            variant_results[variant] = {
                "best_epoch": checkpoint["epoch"],
                "development_group_balanced_epsilon_mse": checkpoint["best_development_group_balanced_epsilon_mse"],
                "training": training_runtime,
                "sampling": sampling,
                "single_draw_group_balanced_metrics": {metric: experiment._group_balanced(single, metric) for metric in experiment.METRIC_NAMES},
                "single_draw_metric_standard_deviation_across_draws": experiment._draw_metric_spread(single),
                "group_balanced_pixel_mean_metrics": {metric: experiment._group_balanced(method_mean, metric) for metric in experiment.METRIC_NAMES},
                "mean_pixelwise_sample_std": float(np.mean([float(row["mean_pixelwise_sample_std"]) for row in method_mean])),
                "draw_pair_mean_absolute_difference": float(np.mean([float(row["draw_pair_mean_absolute_difference"]) for row in method_mean])),
                "output_clipped_fraction_mean": float(np.mean([float(row["output_clipped_fraction"]) for row in single])),
                "group_balanced_distribution_scores": {metric: experiment._group_balanced(method_distribution, metric) for metric in ("pixelwise_crps", "normalized_image_energy_score", "adjacent_variogram_score")},
                "checkpoint_sha256": {"best": sha256_file(method_dir / "checkpoints/best.pt"), "last": sha256_file(method_dir / "checkpoints/last.pt")},
            }

        experiment._write_csv(run_dir / "all_methods_per_draw_metrics.csv", draw_rows)
        experiment._write_csv(run_dir / "all_methods_pixel_mean_metrics.csv", mean_rows)
        experiment._write_csv(run_dir / "all_methods_distribution_scores.csv", distribution_rows)
        mean_draw_rows = experiment._average_draws_by_sample(draw_rows)
        experiment._write_csv(run_dir / "all_methods_mean_draw_metrics_per_image.csv", mean_draw_rows)
        experiment._write_csv(run_dir / "source_group_stats.csv", experiment._source_group_statistics(draw_rows, mean_rows, distribution_rows))

        swinir_run = _swinir_reference_run(run_dir)
        swinir_rows = list(csv.DictReader((swinir_run / "development_metrics_swinir.csv").open(encoding="utf-8-sig", newline="")))
        c2_rows = list(csv.DictReader((swinir_run / "development_metrics_internal_c2.csv").open(encoding="utf-8-sig", newline="")))
        comparisons, paired_rows, paired_mean_rows = {}, [], []
        for variant in ("conditional", "residual"):
            rows = [row for row in mean_rows if row["method"] == variant]
            paired, summary = experiment._paired_bootstrap(rows, swinir_rows, args.seed + 1000 + (variant == "residual"))
            for row in paired:
                row["method"] = variant
            paired_rows.extend(paired)
            c2_paired, c2_summary = experiment._paired_bootstrap(rows, c2_rows, args.seed + 2000 + (variant == "residual"))
            summary["versus_internal_C2"] = c2_summary
            comparisons[variant] = summary
            draw_rows_for_method = [row for row in mean_draw_rows if row["method"] == variant]
            draw_paired, draw_summary = experiment._paired_bootstrap(draw_rows_for_method, swinir_rows, args.seed + 1500 + (variant == "residual"))
            for row in draw_paired:
                row["method"] = variant
            paired_mean_rows.extend(draw_paired)
        experiment._write_csv(run_dir / "paired_group_bootstrap_vs_swinir.csv", paired_rows)
        experiment._write_csv(run_dir / "paired_mean_draw_metrics_vs_swinir.csv", paired_mean_rows)
        residual_rows = [row for row in mean_rows if row["method"] == "residual"]
        conditional_rows = [row for row in mean_rows if row["method"] == "conditional"]
        direct, direct_summary = experiment._paired_bootstrap(residual_rows, conditional_rows, args.seed + 3000)
        for row in direct:
            for metric in experiment.METRIC_NAMES:
                row[f"conditional_{metric}"] = row.pop(f"swinir_{metric}")
        comparisons["residual"]["versus_conditional_diffusion"] = direct_summary
        experiment._write_csv(run_dir / "paired_residual_vs_conditional.csv", direct)
        residual_draws = [row for row in mean_draw_rows if row["method"] == "residual"]
        conditional_draws = [row for row in mean_draw_rows if row["method"] == "conditional"]
        direct_draw, direct_draw_summary = experiment._paired_bootstrap(residual_draws, conditional_draws, args.seed + 3500)
        for row in direct_draw:
            for metric in experiment.METRIC_NAMES:
                row[f"conditional_{metric}"] = row.pop(f"swinir_{metric}")
        comparisons["residual"]["mean_draw_metrics_vs_conditional_diffusion"] = direct_draw_summary
        experiment._write_csv(run_dir / "paired_residual_vs_conditional_mean_draw_metrics.csv", direct_draw)

        swinir_checkpoint = torch.load(Path(manifest["swinir_checkpoint_path"]), map_location="cpu", weights_only=False)
        summary = {
            "interpretation_scope": "单 seed、历史重复使用 development 的探索观察；不构成独立泛化或校准不确定性证据",
            "train_only_residual_rms": residual_rms,
            "train_only_residual_scale": residual_scale,
            "SwinIR_SAR_deterministic_reference": {
                "checkpoint_sha256": manifest["swinir_checkpoint_sha256"],
                "best_epoch": swinir_checkpoint.get("best_epoch"),
                "group_balanced_metrics": {metric: experiment._group_balanced(swinir_rows, metric) for metric in experiment.METRIC_NAMES},
            },
            "internal_C2_deterministic_reference": {
                "checkpoint_sha256": "8255b5229ca1e041cb59aed582d794bb67afdbe1601b6f4dc425c43ce27f7165",
                "group_balanced_metrics": {metric: experiment._group_balanced(c2_rows, metric) for metric in experiment.METRIC_NAMES},
            },
            "diffusion_variants": variant_results,
            "paired_group_bootstrap_vs_swinir": comparisons,
            "limitations": [
                "4 draws 只支撑小样本描述性集合评分。",
                "CRPS、energy score 与 adjacent variogram score 只作为 noisy pseudo-target 上的经验评分，不证明分布校准。",
                "PSNR/SSIM 只衡量表示域 noisy pseudo-target 一致性。",
                "SwinIR 本身是发布架构任务适配版，非 SAR 专用论文实现。",
                "residual diffusion 依赖已在同一历史重复开发集上选择的 SwinIR checkpoint。",
                "运行曾在 conditional 采样阶段外部中断；原始 168 张部分输出保留，完整采样由确定性固定噪声重新生成。",
            ],
        }
        _write_json(run_dir / "summary.json", summary)
        training_seconds = {key: value["training"]["training_seconds"] for key, value in variant_results.items()}
        sampling_seconds = {key: value["sampling"].get("sampling_seconds_total") for key, value in variant_results.items()}
        runtime = {
            "training_seconds_by_variant": training_seconds,
            "sampling_seconds_by_variant": sampling_seconds,
            "measured_training_and_sampling_seconds": float(sum(training_seconds.values()) + sum(sampling_seconds.values())),
            "swinir_base_inference_seconds_train_and_development": None,
            "swinir_base_inference_time_note": "初始进程在写入总 runtime.json 前中断，未持久化该计时；不作估算。",
            "cuda_peak_allocated_mib": max(max(value["training"].get("peak_allocated_mib", 0.0), value["sampling"].get("peak_allocated_mib", 0.0)) for value in variant_results.values()),
            "cuda_peak_reserved_mib": max(max(value["training"].get("peak_reserved_mib", 0.0), value["sampling"].get("peak_reserved_mib", 0.0)) for value in variant_results.values()),
            "cuda_free_memory_bytes_after_finalize": int(torch.cuda.mem_get_info()[0]),
            "disk_free_bytes_after_finalize": shutil.disk_usage(run_dir).free,
            "parameters": sum(parameter.numel() for parameter in experiment.ConditionalUNet().parameters()),
            "optimizer_updates_per_variant": args.epochs * math.ceil(1600 / (args.batch_size * args.accumulation_steps)),
            "generated_image_count_per_variant": len(development) * args.draws,
            "prediction_array_shape_per_sample": [args.draws, 256, 256],
        }
        _write_json(run_dir / "runtime.json", runtime)
        manifest.update({
            "status": "complete",
            "ended_unix": time.time(),
            "active_recovery_stage": None,
            "resumption_status": "完成全部训练、推理、逐图和来源组评价",
            "code_sha256": sha256_file(SCRIPT_DIR / "train_sar_only_diffusion.py"),
            "recovery_code_sha256": sha256_file(Path(__file__)),
            "runtime_path": "runtime.json",
            "outputs": sorted(path.relative_to(run_dir).as_posix() for path in run_dir.rglob("*") if path.is_file()),
        })
        _write_json(manifest_path, manifest)
        record.update({"状态": "完成", "结束时间 Unix": time.time(), "输出数量": len(manifest["outputs"])})
        _write_json(record_path, record)
        print("扩散恢复与 Research Decision Review 数据准备完成。", flush=True)
        return

    raise ValueError(f"未知恢复阶段：{stage_id}")


def main():
    parser = argparse.ArgumentParser(description="按阶段恢复 SAR-only 扩散训练与评价")
    parser.add_argument("--stage", choices=("conditional_sample", "residual_train", "residual_sample", "finalize"), required=True)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    try:
        run_stage(args.stage, Path(args.run_dir))
    except BaseException as error:
        run_dir = Path(args.run_dir).resolve()
        manifest_path = run_dir / "manifest.json"
        if manifest_path.exists():
            manifest = _read_json(manifest_path)
            manifest["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
            manifest["last_recovery_error"] = {
                "stage": args.stage,
                "type": type(error).__name__,
                "message": str(error),
                "cuda_oom": isinstance(error, torch.cuda.OutOfMemoryError),
                "time_unix": time.time(),
            }
            _write_json(manifest_path, manifest)
        print(f"恢复阶段失败：{type(error).__name__}: {error}", flush=True)
        raise


if __name__ == "__main__":
    main()
