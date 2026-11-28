"""对冻结的 Task B 候选 A checkpoint 运行矩匹配观测似然引导 DDIM。"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn.functional as F

from experiments.phase_e_taskb_metrics import METRIC_NAMES
from experiments.phase_e_taskb_models import ConditionalResidualUNet
from experiments.phase_e_train_taskb_pilot import (
    BOOTSTRAP_SEED,
    DDIM_STEPS,
    DIFFUSION_STEPS,
    MAX_STAGE_SECONDS,
    SEED,
    _append_csv,
    _eval_and_save,
    _group_balanced_mean,
    _group_bootstrap,
    _git_state,
    _gpu_snapshot,
    _load_prediction_map,
    _load_records,
    _save_predictions,
    _seed_for,
    _write_json,
    sha256_file,
)


def gaussian_gamma_moment_nll(
    clean_image: torch.Tensor,
    low_resolution: torch.Tensor,
    *,
    variance_epsilon: float = 1e-6,
) -> torch.Tensor:
    """计算 Gamma(1,1) 乘性噪声经 2×2 面积平均后的矩匹配高斯 NLL。

    返回每张图的空间求和 NLL。四个独立 Exp(1) 加权和通常不是单一 Gamma，
    因此该函数只实现冻结协议规定的前两矩高斯近似。
    """
    if clean_image.ndim != 4 or low_resolution.ndim != 4:
        raise ValueError("似然输入必须为 BCHW 张量")
    if clean_image.shape[:2] != low_resolution.shape[:2] or clean_image.shape[1] != 1:
        raise ValueError("似然输入必须为对齐的单通道张量")
    if clean_image.shape[-2] != 2 * low_resolution.shape[-2] or clean_image.shape[-1] != 2 * low_resolution.shape[-1]:
        raise ValueError("高分辨率与 LR 的空间尺寸必须严格为 2 倍")
    if clean_image.device != low_resolution.device or clean_image.dtype != low_resolution.dtype:
        raise ValueError("似然输入 device/dtype 必须一致")
    if variance_epsilon <= 0 or not math.isfinite(variance_epsilon):
        raise ValueError("方差 epsilon 必须为有限正数")

    bounded = clean_image.clamp(1e-4, 1.0 - 1e-4)
    predicted_mean = F.avg_pool2d(bounded, kernel_size=2, stride=2)
    predicted_variance = F.avg_pool2d(bounded.square(), kernel_size=2, stride=2) / 4.0
    stable_variance = predicted_variance + variance_epsilon
    per_cell = 0.5 * (
        (low_resolution - predicted_mean).square() / stable_variance
        + torch.log(stable_variance)
    )
    return per_cell.flatten(start_dim=1).sum(dim=1)


def likelihood_guided_ddim_sample(
    model: torch.nn.Module,
    mean_estimate: torch.Tensor,
    low_resolution: torch.Tensor,
    alphas_cumprod: torch.Tensor,
    *,
    seed: int,
    steps: int = DDIM_STEPS,
) -> torch.Tensor:
    """按固定 unit-score correction 运行零训练观测似然引导 DDIM。"""
    if mean_estimate.ndim != 4 or low_resolution.ndim != 4:
        raise ValueError("DDIM 条件输入必须为 BCHW")
    if mean_estimate.shape[:2] != low_resolution.shape[:2] or mean_estimate.shape[1] != 1:
        raise ValueError("DDIM 条件输入必须为对齐的单通道张量")
    if mean_estimate.shape[-2] != 2 * low_resolution.shape[-2] or mean_estimate.shape[-1] != 2 * low_resolution.shape[-1]:
        raise ValueError("DDIM 条件输入空间倍率必须为 2")
    if alphas_cumprod.ndim != 1 or alphas_cumprod.numel() < steps or steps < 2:
        raise ValueError("DDIM alpha schedule 或步数无效")
    if mean_estimate.device != low_resolution.device or mean_estimate.device != alphas_cumprod.device:
        raise ValueError("DDIM 条件输入与 alpha schedule device 不一致")
    if mean_estimate.dtype != low_resolution.dtype or mean_estimate.dtype != alphas_cumprod.dtype:
        raise ValueError("DDIM 条件输入与 alpha schedule dtype 不一致")

    model.eval()
    generator = torch.Generator(device=mean_estimate.device).manual_seed(seed)
    residual = torch.randn(
        mean_estimate.shape,
        device=mean_estimate.device,
        dtype=mean_estimate.dtype,
        generator=generator,
    )
    indices = torch.linspace(
        alphas_cumprod.numel() - 1, 0, steps, device=mean_estimate.device
    ).round().long()
    if torch.unique(indices).numel() != steps:
        raise ValueError("DDIM 时间索引重复")

    for step_index, timestep in enumerate(indices):
        residual_state = residual.detach().requires_grad_(True)
        time_tensor = timestep.expand(mean_estimate.shape[0])
        alpha_t = alphas_cumprod[timestep]
        with torch.enable_grad():
            predicted_noise = model(
                residual_state, mean_estimate, low_resolution, time_tensor
            )
            if predicted_noise.shape != residual_state.shape or not torch.isfinite(predicted_noise).all():
                raise FloatingPointError("似然引导 DDIM epsilon 输出形状错误或含非有限值")
            predicted_clean_residual = (
                residual_state - torch.sqrt(1.0 - alpha_t) * predicted_noise
            ) / torch.sqrt(alpha_t)
            predicted_clean_image = mean_estimate + predicted_clean_residual
            nll = gaussian_gamma_moment_nll(predicted_clean_image, low_resolution)
            likelihood_gradient, = torch.autograd.grad(nll.sum(), residual_state)
            if not torch.isfinite(likelihood_gradient).all():
                raise FloatingPointError("似然引导梯度含非有限值")
            guided_noise = (
                predicted_noise.detach()
                + torch.sqrt(1.0 - alpha_t) * likelihood_gradient.detach()
            )

        predicted_clean_guided = (
            residual_state.detach() - torch.sqrt(1.0 - alpha_t) * guided_noise
        ) / torch.sqrt(alpha_t)
        if step_index + 1 < len(indices):
            alpha_previous = alphas_cumprod[indices[step_index + 1]]
        else:
            alpha_previous = torch.ones((), device=mean_estimate.device, dtype=mean_estimate.dtype)
        residual = (
            torch.sqrt(alpha_previous) * predicted_clean_guided
            + torch.sqrt(1.0 - alpha_previous) * guided_noise
        ).detach()

    return torch.clamp(mean_estimate + residual, 0.0, 1.0)


def _summarize_methods(
    output_root: Path,
    records: list[dict[str, Any]],
    method_predictions: dict[str, dict[str, np.ndarray]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    all_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []
    for method, predictions in method_predictions.items():
        rows = _eval_and_save(method, predictions, records, output_root)
        all_rows.extend(rows)
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            grouped.setdefault(row["source_group_id"], []).append(row)
        for group_id, values in sorted(grouped.items()):
            group_rows.append({
                "method": method,
                "source_group_id": group_id,
                "sample_count": len(values),
                "epi_valid_sample_count": sum(row["epi_corr"] is not None for row in values),
                "epi_missing_sample_count": sum(row["epi_corr"] is None for row in values),
                **{metric: _group_balanced_mean(values, metric) for metric in METRIC_NAMES},
            })
        summary_rows.append({
            "method": method,
            "source_group_count": len(grouped),
            "sample_count": len(rows),
            "epi_valid_sample_count": sum(row["epi_corr"] is not None for row in rows),
            "epi_missing_sample_count": sum(row["epi_corr"] is None for row in rows),
            "epi_coverage": float(np.mean([row["epi_corr"] is not None for row in rows])),
            **{metric: _group_balanced_mean(rows, metric) for metric in METRIC_NAMES},
        })
    _append_csv(output_root / "development_group_metrics.csv", group_rows)
    _append_csv(output_root / "development_method_summary.csv", summary_rows)
    return all_rows, summary_rows


def _predeclared_gate(
    candidate: dict[str, Any], baseline: dict[str, Any], psnr_bootstrap: dict[str, Any]
) -> tuple[bool, bool]:
    epi_complete = (
        candidate["epi_valid_sample_count"] == candidate["sample_count"]
        and baseline["epi_valid_sample_count"] == baseline["sample_count"]
    )
    meets = (
        candidate["psnr_db"] - baseline["psnr_db"] >= 0.05
        and psnr_bootstrap["bootstrap_95ci_low"] is not None
        and psnr_bootstrap["bootstrap_95ci_low"] > 0
        and candidate["ssim"] >= baseline["ssim"]
        and epi_complete
        and candidate["epi_corr"] is not None
        and baseline["epi_corr"] is not None
        and candidate["epi_corr"] >= baseline["epi_corr"]
        and abs(candidate["bright_mean_ratio"] - 1.0)
        <= abs(baseline["bright_mean_ratio"] - 1.0)
        and abs(candidate["background_mean_ratio"] - 1.0)
        <= abs(baseline["background_mean_ratio"] - 1.0)
        and candidate["reprojection_mae"] <= baseline["reprojection_mae"]
    )
    return meets, epi_complete


def run_guided_evaluation(
    input_manifest_path: Path,
    source_run: Path,
    output_root: Path,
) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"评估输出目录已存在，拒绝覆盖：{output_root}")
    if not torch.cuda.is_available():
        raise RuntimeError("Phase E 似然引导评价要求 CUDA")

    source_run = source_run.resolve()
    input_manifest_path = input_manifest_path.resolve()
    input_manifest_sha = sha256_file(input_manifest_path)
    preflight_started = time.monotonic()
    records, input_manifest = _load_records(input_manifest_path)
    source_manifest = json.loads((source_run / "manifest.json").read_text(encoding="utf-8-sig"))
    source_environment = json.loads((source_run / "environment.json").read_text(encoding="utf-8-sig"))
    source_stage = json.loads(
        (source_run / "candidate_components" / "centered_residual_diffusion" / "manifest.json")
        .read_text(encoding="utf-8-sig")
    )
    protected_flags = (
        "confirmation_payload_accessed", "test_payload_accessed", "optical_payload_loaded"
    )
    for name, metadata in (("候选 A manifest", source_manifest), ("候选 A environment", source_environment), ("候选 A 扩散阶段", source_stage)):
        if metadata.get("input_manifest_sha256") != input_manifest_sha:
            raise ValueError(f"{name} 输入清单哈希不匹配")
        if any(metadata.get(key) is not False for key in protected_flags):
            raise ValueError(f"{name} 保护数据访问标记不合规")
    if source_manifest.get("status") != "completed" or source_stage.get("status") != "completed":
        raise ValueError("候选 A 必须是完整成功运行")
    if source_stage.get("best_epoch") != 3 or source_stage.get("patch_lr") != 64:
        raise ValueError("候选 A best checkpoint 与冻结备选合同不符")
    if len(records["development"]) != 70:
        raise ValueError("固定 development 必须恰为 70 张")

    artifact_run = Path(source_environment["artifact_source_run"]).resolve()
    if artifact_run.parent != source_run.parent:
        raise ValueError("基线产物父目录必须与候选 A 位于同一 pilot 根目录")
    development_ids = [row["sample_id"] for row in records["development"]]
    development_mean = _load_prediction_map(
        source_run / "均值校正点估计" / "predictions", development_ids
    )
    reference_predictions = {
        "EDSR": _load_prediction_map(artifact_run / "EDSR" / "predictions_best", development_ids),
        "SwinIR": _load_prediction_map(artifact_run / "SwinIR" / "predictions_best", development_ids),
    }
    source_predictions = {
        "无引导_DDIM_单次": _load_prediction_map(
            source_run / "DDIM_单次" / "predictions", development_ids
        ),
        "无引导_DDIM_四次像素均值": _load_prediction_map(
            source_run / "DDIM_四次抽样像素均值" / "predictions", development_ids
        ),
    }
    for draw_index in range(4):
        source_predictions[f"无引导_DDIM_draw_{draw_index}"] = _load_prediction_map(
            source_run / "DDIM_四次抽样" / f"draw_{draw_index}", development_ids
        )

    checkpoint = Path(source_stage["best_checkpoint"]).resolve()
    if not checkpoint.is_file() or sha256_file(checkpoint) != source_stage.get("best_checkpoint_sha256"):
        raise ValueError("候选 A best checkpoint 文件或 SHA256 校验失败")
    checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if (
        checkpoint_payload.get("format") != "phase_e_taskb_checkpoint_v1"
        or checkpoint_payload.get("method") != "centered_residual_conditional_ddpm"
        or checkpoint_payload.get("seed") != SEED
        or checkpoint_payload.get("epoch") != 3
        or "ema_state_dict" not in checkpoint_payload
    ):
        raise ValueError("候选 A best checkpoint 元数据不匹配")

    output_root.parent.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=False, exist_ok=False)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    _set_seed_for_eval(SEED)
    torch.cuda.reset_peak_memory_stats(device)
    disk_before = shutil.disk_usage(output_root.parent)
    gpu_before = _gpu_snapshot(device)
    preflight_seconds = time.monotonic() - preflight_started
    total_budget_before = float(
        source_environment.get("budget_accounting", {}).get("total_budget_remaining_seconds", 0.0)
    ) - float(source_environment.get("elapsed_seconds", 0.0))
    if total_budget_before <= 0:
        raise TimeoutError("Phase E 本轮总 GPU 预算已耗尽")

    model = ConditionalResidualUNet().to(device)
    model.load_state_dict(checkpoint_payload["ema_state_dict"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    betas = torch.linspace(1e-4, 0.02, DIFFUSION_STEPS, device=device)
    alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
    started = time.monotonic()
    candidate_deadline = started + min(MAX_STAGE_SECONDS, total_budget_before)
    profile: dict[str, Any] = {
        "status": "running",
        "method": "矩匹配表示域观测似然引导 DDIM（零训练）",
        "candidate_id": "phase_e_taskb_likelihood_guided_ddim_v1",
        "candidate_a_run": str(source_run),
        "artifact_source_run": str(artifact_run),
        "best_checkpoint": str(checkpoint),
        "best_checkpoint_sha256": source_stage["best_checkpoint_sha256"],
        "best_epoch": 3,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "training_run": False,
        "seed": SEED,
        "draw_indices": [0, 1, 2, 3],
        "ddim_steps": DDIM_STEPS,
        "diffusion_steps": DIFFUSION_STEPS,
        "beta_schedule": "linear(1e-4,0.02)",
        "eta": 0,
        "guidance_coefficient": 1.0,
        "variance_epsilon": 1e-6,
        "likelihood": "由 Gamma(k=1,theta=1) 乘性噪声与 2x2 面积平均推导的前两矩高斯近似",
        "input_manifest_sha256": input_manifest_sha,
        "protocol_sha256": sha256_file(
            Path(__file__).resolve().parents[1]
            / "docs/research/PHASE_E_TASK_B_GLOBAL_LIKELIHOOD_GUIDED_DDIM_PROTOCOL_20261128_V1.md"
        ),
        "split_manifest_sha256": input_manifest.get("split_manifest_sha256"),
        "development_sample_count": len(development_ids),
        "development_source_group_count": len({r["source_group_id"] for r in records["development"]}),
        "preflight_seconds": preflight_seconds,
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": __import__("torchvision").__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu_before": gpu_before,
        "disk_free_before_bytes": disk_before.free,
        "git": _git_state(),
        "budget_seconds": {
            "candidate_limit_seconds": MAX_STAGE_SECONDS,
            "total_budget_before_candidate_seconds": total_budget_before,
        },
        "confirmation_payload_accessed": False,
        "test_payload_accessed": False,
        "optical_payload_loaded": False,
    }
    _write_json(output_root / "environment.json", profile)
    _write_json(output_root / "manifest.json", profile)
    (output_root / "命令.txt").write_text(
        "协议：PHASE_E_TASK_B_GLOBAL_LIKELIHOOD_GUIDED_DDIM_PROTOCOL_20261128_V1.md\n"
        + "命令：\n" + " ".join([sys.executable, *sys.argv]) + "\n",
        encoding="utf-8",
    )

    try:
        guided_draws: list[dict[str, np.ndarray]] = []
        draw_runtime = []
        for draw_index in range(4):
            if time.monotonic() >= candidate_deadline:
                raise TimeoutError("零训练似然引导候选达到冻结的 4 小时/总预算")
            draw_started = time.monotonic()
            predictions: dict[str, np.ndarray] = {}
            for sample_index, record in enumerate(records["development"], start=1):
                mean_tensor = torch.from_numpy(development_mean[record["sample_id"]][None, None]).to(device)
                lr_tensor = torch.from_numpy(record["lr"][None]).to(device)
                sample_seed = _seed_for(
                    f"{record['sample_id']}:generated_draw_{draw_index}", base=SEED + 100
                )
                prediction = likelihood_guided_ddim_sample(
                    model, mean_tensor, lr_tensor, alphas_cumprod, seed=sample_seed
                )[0, 0].cpu().numpy().astype(np.float32)
                if not np.isfinite(prediction).all():
                    raise FloatingPointError(f"似然引导输出非有限：{record['sample_id']}")
                predictions[record["sample_id"]] = prediction
                if sample_index % 10 == 0 or sample_index == len(records["development"]):
                    print(
                        f"似然引导第 {draw_index + 1}/4 次采样：已完成 {sample_index}/70 张。",
                        flush=True,
                    )
            _save_predictions(output_root / "DDIM_似然引导" / f"draw_{draw_index}", predictions)
            guided_draws.append(predictions)
            elapsed_draw = time.monotonic() - draw_started
            draw_runtime.append(elapsed_draw)
            print(f"似然引导第 {draw_index + 1}/4 次采样完成，用时 {elapsed_draw:.1f} 秒。", flush=True)

        guided_mean = {
            sample_id: np.mean(
                [draw[sample_id] for draw in guided_draws], axis=0, dtype=np.float64
            ).astype(np.float32)
            for sample_id in development_ids
        }
        _save_predictions(output_root / "DDIM_似然引导_四次抽样像素均值" / "predictions", guided_mean)
        method_predictions: dict[str, dict[str, np.ndarray]] = {
            **reference_predictions,
            "确定性均值校正": development_mean,
            **source_predictions,
            "似然引导_DDIM_单次": guided_draws[0],
            "似然引导_DDIM_四次像素均值": guided_mean,
        }
        for draw_index, predictions in enumerate(guided_draws):
            method_predictions[f"似然引导_DDIM_draw_{draw_index}"] = predictions

        all_rows, summary_rows = _summarize_methods(
            output_root, records["development"], method_predictions
        )
        summary_map = {row["method"]: row for row in summary_rows}
        reference_method = max(
            (name for name in ("EDSR", "SwinIR")),
            key=lambda name: summary_map[name]["psnr_db"],
        )
        bootstrap_rows: list[dict[str, Any]] = []
        paired_rows: list[dict[str, Any]] = []
        comparisons = (
            ("似然引导_DDIM_单次", reference_method),
            ("似然引导_DDIM_四次像素均值", reference_method),
            ("似然引导_DDIM_单次", "无引导_DDIM_单次"),
        )
        for method, reference in comparisons:
            current_rows = [row for row in all_rows if row["method"] == method]
            reference_rows = [row for row in all_rows if row["method"] == reference]
            paired, boot = _group_bootstrap(
                current_rows, reference_rows, method=method, reference=reference
            )
            paired_rows.extend(paired)
            bootstrap_rows.extend(boot)
        _append_csv(output_root / "development_paired_per_image.csv", paired_rows)
        _append_csv(output_root / "development_group_bootstrap.csv", bootstrap_rows)
        candidate_summary = summary_map["似然引导_DDIM_单次"]
        reference_summary = summary_map[reference_method]
        psnr_bootstrap = next(
            row for row in bootstrap_rows
            if row["method"] == "似然引导_DDIM_单次"
            and row["reference"] == reference_method
            and row["metric"] == "psnr_db"
        )
        gate_met, epi_complete = _predeclared_gate(
            candidate_summary, reference_summary, psnr_bootstrap
        )
        review = {
            "candidate": "矩匹配表示域观测似然引导 DDIM（零训练）",
            "strongest_learning_baseline": reference_method,
            "strongest_baseline_summary": reference_summary,
            "candidate_summary": candidate_summary,
            "candidate_vs_strongest_psnr_delta": candidate_summary["psnr_db"] - reference_summary["psnr_db"],
            "candidate_psnr_bootstrap_95ci": [
                psnr_bootstrap["bootstrap_95ci_low"], psnr_bootstrap["bootstrap_95ci_high"]
            ],
            "candidate_epi_complete": epi_complete,
            "candidate_epi_valid_sample_count": candidate_summary["epi_valid_sample_count"],
            "candidate_epi_missing_sample_count": candidate_summary["epi_missing_sample_count"],
            "predeclared_continue_gate_met": gate_met,
            "draw_runtime_seconds": draw_runtime,
            "draw_runtime_total_seconds": float(sum(draw_runtime)),
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_draws": 10_000,
            "decision_scope": "单 seed、已用于候选选择的 development；仅为表示域合成任务探索性判断",
            "likelihood_limit": "Gamma 加权和的前两矩高斯近似，不是精确密度或 SAR 物理似然",
        }
        elapsed = time.monotonic() - started
        profile.update({
            "status": "completed",
            "elapsed_seconds": elapsed,
            "sampler_draw_runtime_seconds": draw_runtime,
            "sampler_draw_runtime_total_seconds": float(sum(draw_runtime)),
            "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
            "gpu_after": _gpu_snapshot(device),
            "disk_free_after_bytes": shutil.disk_usage(output_root.parent).free,
            "budget_seconds_after_run": {
                "candidate_budget_remaining_seconds": max(0.0, MAX_STAGE_SECONDS - elapsed),
                "total_budget_remaining_seconds": max(0.0, total_budget_before - elapsed),
            },
            "protected_access_flags": {key: False for key in protected_flags},
            "research_decision": review,
        })
        _write_json(output_root / "manifest.json", profile)
        _write_json(output_root / "environment.json", profile)
        _write_json(output_root / "研究决策复核.json", review)
        print(
            "似然引导试验完成："
            f"最强学习基线={reference_method}；"
            f"DDIM 单次 ΔPSNR={review['candidate_vs_strongest_psnr_delta']:.4f} dB；"
            f"冻结继续门槛={'通过' if gate_met else '未通过'}。",
            flush=True,
        )
        return review
    except BaseException as error:
        profile.update({
            "status": "failed",
            "elapsed_seconds": time.monotonic() - started,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "cuda_oom": isinstance(error, torch.cuda.OutOfMemoryError),
            "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
            "gpu_after": _gpu_snapshot(device),
            "disk_free_after_bytes": shutil.disk_usage(output_root.parent).free,
        })
        _write_json(output_root / "失败记录.json", profile)
        _write_json(output_root / "environment.json", profile)
        print(f"似然引导试验失败，已保存诊断：{type(error).__name__}；{error}", flush=True)
        raise


def _set_seed_for_eval(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def main() -> None:
    parser = argparse.ArgumentParser(description="运行 Phase E Task B 表示域观测似然引导 DDIM development 评价")
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run_guided_evaluation(args.input_manifest, args.source_run, args.output_dir)


if __name__ == "__main__":
    main()
