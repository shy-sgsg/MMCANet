"""基于固定表示域 Gamma 观测能量执行后验引导 DDIM，并配对评价 SwinIR。"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import random
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.phase_d2_observation_aware_20261124_v1.observation_model import (  # noqa: E402
    observation_energy,
)

DEFAULT_TRAINING_RUN = PROJECT_ROOT / "experiments/runs/phase_d2_observation_aware_pilot_20261124_v1_retry01"
DEFAULT_SWINIR_RUN = PROJECT_ROOT / "experiments/published_baselines_sar_only_20261123_v1/运行_swinir_20261123_v3"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "experiments/runs/phase_d2_observation_aware_eval_20261124_v1"
PILOT_PROTOCOL = PROJECT_ROOT / "docs/research/PHASE_D2_OBSERVATION_AWARE_PILOT_PROTOCOL_20261124.md"
QUALITY_METRICS = (
    "psnr_db",
    "ssim",
    "rmse",
    "epi_corr",
    "edge_strength_ratio",
    "edge_strength_error",
    "bright_mean_ratio",
    "bright_p95_ratio",
    "background_mean_ratio",
)
OBSERVATION_METRICS = ("positive_nll", "zero_constraint", "zero_violation_rate")
ALL_METRICS = QUALITY_METRICS + OBSERVATION_METRICS
DRAW_BASE_SEEDS = (20261125, 20261126, 20261127, 20261128)


def ddim_posterior_sample(
    model,
    observation: torch.Tensor,
    alpha_bar: torch.Tensor,
    timesteps: torch.Tensor,
    seed: int,
    guidance_weight: float = 1.0,
    *,
    return_diagnostics: bool = False,
):
    """以观测能量梯度引导单个 256×256 HR 后验 draw。"""
    if observation.shape != (1, 1, 128, 128):
        raise ValueError("observation shape 必须为 [1,1,128,128]")
    if not torch.is_floating_point(observation) or not torch.isfinite(observation).all():
        raise ValueError("observation 必须为有限浮点张量")
    if torch.any(observation < 0.0):
        raise ValueError("observation 必须非负")
    if alpha_bar.ndim != 1 or alpha_bar.numel() < 2:
        raise ValueError("alpha_bar 必须是一维 schedule")
    if timesteps.ndim != 1 or timesteps.numel() < 2:
        raise ValueError("timesteps 必须是一维且至少包含两个时刻")
    time_values = [int(value) for value in timesteps.detach().cpu().tolist()]
    if (
        any(value < 0 or value >= alpha_bar.numel() for value in time_values)
        or time_values[-1] != 0
        or any(left <= right for left, right in zip(time_values, time_values[1:]))
    ):
        raise ValueError("timesteps 必须严格 descending 且以 0 结束")
    if not math.isfinite(float(guidance_weight)) or guidance_weight < 0.0:
        raise ValueError("guidance weight 必须有限且非负")
    if not (alpha_bar.device == observation.device and alpha_bar.dtype == observation.dtype):
        alpha_bar = alpha_bar.to(device=observation.device, dtype=observation.dtype)

    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    state = torch.randn((1, 1, 256, 256), generator=generator, dtype=observation.dtype)
    state = state.to(device=observation.device)
    zero_condition = torch.zeros_like(state)
    gradient_norms = []

    for index, step in enumerate(time_values):
        state = state.detach().requires_grad_(True)
        step_tensor = torch.full((1,), step, device=state.device, dtype=torch.long)
        alpha = alpha_bar[step]
        predicted_noise = model(state, zero_condition, step_tensor)
        if predicted_noise.shape != state.shape or not torch.isfinite(predicted_noise).all():
            raise FloatingPointError("prior epsilon 输出 shape/finite 无效")
        predicted_x0 = (state - (1.0 - alpha).sqrt() * predicted_noise) / alpha.sqrt()
        bounded_x0 = predicted_x0.clamp(-1.0, 1.0)
        energy = observation_energy((bounded_x0 + 1.0) * 0.5, observation)
        gradient = torch.autograd.grad(energy["total"], state, create_graph=False)[0]
        if not torch.isfinite(gradient).all():
            raise FloatingPointError("观测引导梯度非有限")
        gradient_norms.append(float(gradient.detach().norm().item()))
        guided_noise = predicted_noise + (1.0 - alpha).sqrt() * float(guidance_weight) * gradient
        guided_x0 = (
            state - (1.0 - alpha).sqrt() * guided_noise
        ) / alpha.sqrt()
        guided_x0 = guided_x0.clamp(-1.0, 1.0)
        if index + 1 == len(time_values):
            state = guided_x0.detach()
        else:
            previous_alpha = alpha_bar[time_values[index + 1]]
            state = (
                previous_alpha.sqrt() * guided_x0
                + (1.0 - previous_alpha).sqrt() * guided_noise
            ).detach()

    prediction = ((state + 1.0) * 0.5).clamp(0.0, 1.0)
    if not torch.isfinite(prediction).all():
        raise FloatingPointError("posterior prediction 含非有限值")
    if not return_diagnostics:
        return prediction
    with torch.no_grad():
        final_energy = observation_energy(prediction, observation)
    diagnostics = {
        key: float(value.detach().item()) for key, value in final_energy.items()
    }
    diagnostics["mean_guidance_gradient_norm"] = float(np.mean(gradient_norms))
    diagnostics["max_guidance_gradient_norm"] = float(np.max(gradient_norms))
    return prediction, diagnostics


def _image_metrics(prediction: np.ndarray, record: dict) -> dict[str, float]:
    from experiments.published_baselines_sar_only_20261123_v1.train_sar_only_diffusion import (
        _metrics_for_prediction,
    )

    value = np.asarray(prediction)
    target = record["target"][0]
    if value.shape != target.shape or value.dtype != np.float32:
        raise ValueError("prediction 必须为 float32 且与 target 同形")
    if not np.isfinite(value).all() or value.min() < 0.0 or value.max() > 1.0:
        raise ValueError("prediction 必须有限并位于 [0,1]")
    metrics = _metrics_for_prediction(value, record)
    metrics["rmse"] = float(np.sqrt(np.mean((value.astype(np.float64) - target) ** 2)))
    metrics["edge_strength_error"] = abs(float(metrics["edge_strength_ratio"]) - 1.0)
    return {name: float(metrics[name]) for name in QUALITY_METRICS}


def _group_balanced(rows: list[dict], metrics=ALL_METRICS) -> dict[str, float]:
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(row["source_group_id"], []).append(row)
    if not groups:
        raise ValueError("来源组统计输入为空")
    return {
        metric: float(np.mean([np.mean([float(row[metric]) for row in values]) for values in groups.values()]))
        for metric in metrics
    }


def _paired_group_bootstrap(method_rows, reference_rows, seed=20261124):
    reference = {row["sample_id"]: row for row in reference_rows}
    if set(reference) != {row["sample_id"] for row in method_rows}:
        raise ValueError("方法与 SwinIR 的 sample_id 集合不一致")
    if any(reference[row["sample_id"]]["source_group_id"] != row["source_group_id"] for row in method_rows):
        raise ValueError("配对方法的 source_group_id 不一致")
    groups: dict[str, list[dict]] = {}
    for row in method_rows:
        base = reference[row["sample_id"]]
        delta = {
            "sample_id": row["sample_id"],
            "source_group_id": row["source_group_id"],
            **{metric: float(row[metric]) - float(base[metric]) for metric in ALL_METRICS},
        }
        groups.setdefault(row["source_group_id"], []).append(delta)
    group_names = sorted(groups)
    summary = {}
    for metric_index, metric in enumerate(ALL_METRICS):
        values = np.asarray(
            [np.mean([row[metric] for row in groups[name]]) for name in group_names],
            dtype=np.float64,
        )
        rng = np.random.default_rng(seed + metric_index)
        indices = rng.integers(0, len(values), size=(10_000, len(values)))
        draws = values[indices].mean(axis=1)
        summary[metric] = {
            "group_count": len(group_names),
            "group_balanced_delta": float(values.mean()),
            "group_bootstrap_95_ci": [
                float(np.quantile(draws, 0.025)),
                float(np.quantile(draws, 0.975)),
            ],
        }
    return summary


def evaluate_development(predictions, records, baseline_predictions, observation_metrics):
    """分开汇总单 draw、pixel-mean 与固定 SwinIR 的来源组配对比较。"""
    sample_ids = {record["sample_id"] for record in records}
    if set(predictions) != sample_ids or set(baseline_predictions) != sample_ids:
        raise ValueError("prediction、baseline 与 development sample_id 集合必须完全一致")
    if set(observation_metrics) != sample_ids:
        raise ValueError("观测遥测与 development sample_id 集合必须完全一致")

    single_rows, mean_rows, baseline_rows = [], [], []
    for record in records:
        sample_id = record["sample_id"]
        group = record["source_group_id"]
        draw_stack = np.asarray(predictions[sample_id])
        if draw_stack.ndim != 3 or draw_stack.shape[0] != len(DRAW_BASE_SEEDS):
            raise ValueError("每个 development 样本必须有 4 个 [H,W] draw")
        draw_stats = observation_metrics[sample_id]["draws"]
        if len(draw_stats) != len(DRAW_BASE_SEEDS):
            raise ValueError("每个样本的观测 draw 遥测数必须为 4")
        for draw_index, prediction in enumerate(draw_stack):
            image_stats = _image_metrics(np.ascontiguousarray(prediction, dtype=np.float32), record)
            single_rows.append(
                {
                    "sample_id": sample_id,
                    "source_group_id": group,
                    "draw": draw_index,
                    **image_stats,
                    **{name: float(draw_stats[draw_index][name]) for name in OBSERVATION_METRICS},
                }
            )
        pixel_mean = draw_stack.astype(np.float64).mean(axis=0).astype(np.float32)
        pair_differences = [
            float(np.abs(draw_stack[left] - draw_stack[right]).mean())
            for left in range(len(draw_stack))
            for right in range(left + 1, len(draw_stack))
        ]
        pixel_std = draw_stack.astype(np.float64).std(axis=0)
        mean_stats = observation_metrics[sample_id]["pixel_mean"]
        mean_rows.append(
            {
                "sample_id": sample_id,
                "source_group_id": group,
                **_image_metrics(pixel_mean, record),
                **{name: float(mean_stats[name]) for name in OBSERVATION_METRICS},
                "mean_pixelwise_sample_std": float(pixel_std.mean()),
                "rms_pixelwise_sample_std": float(np.sqrt(np.mean(pixel_std**2))),
                "draw_pair_mean_absolute_difference": float(np.mean(pair_differences)),
            }
        )
        base = np.asarray(baseline_predictions[sample_id])
        if base.dtype != np.float32:
            base = base.astype(np.float32)
        baseline_stats = observation_metrics[sample_id]["baseline"]
        baseline_rows.append(
            {
                "sample_id": sample_id,
                "source_group_id": group,
                **_image_metrics(base, record),
                **{name: float(baseline_stats[name]) for name in OBSERVATION_METRICS},
            }
        )

    averaged_draw_rows = []
    for record in records:
        rows = [row for row in single_rows if row["sample_id"] == record["sample_id"]]
        averaged_draw_rows.append(
            {
                "sample_id": record["sample_id"],
                "source_group_id": record["source_group_id"],
                **{
                    metric: float(np.mean([row[metric] for row in rows]))
                    for metric in ALL_METRICS
                },
            }
        )
    summary = {
        "draw_count": len(DRAW_BASE_SEEDS),
        "single_draw_group_balanced": _group_balanced(single_rows),
        "pixel_mean_group_balanced": _group_balanced(mean_rows),
        "swinir_group_balanced": _group_balanced(baseline_rows),
        "paired_bootstrap": {
            "single_draw_mean_over_draws": _paired_group_bootstrap(averaged_draw_rows, baseline_rows),
            "pixel_mean": _paired_group_bootstrap(mean_rows, baseline_rows),
        },
        "source_group_count": len({row["source_group_id"] for row in mean_rows}),
        "development_sample_count": len(records),
    }
    group_rows = []
    for group in sorted({row["source_group_id"] for row in mean_rows}):
        for method, rows in (
            ("posterior_single_draw", [row for row in single_rows if row["source_group_id"] == group]),
            ("posterior_pixel_mean", [row for row in mean_rows if row["source_group_id"] == group]),
            ("swinir", [row for row in baseline_rows if row["source_group_id"] == group]),
        ):
            group_rows.append(
                {
                    "method": method,
                    "source_group_id": group,
                    "sample_count": len({row["sample_id"] for row in rows}),
                    **{metric: float(np.mean([row[metric] for row in rows])) for metric in ALL_METRICS},
                }
            )
    return {
        "single_draw_metrics": single_rows,
        "pixel_mean_metrics": mean_rows,
        "baseline_metrics": baseline_rows,
        "source_group_metrics": group_rows,
    }, summary


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_peak_memory_bytes(device: torch.device, reserved: bool = False) -> int | None:
    reader = torch.cuda.max_memory_reserved if reserved else torch.cuda.max_memory_allocated
    try:
        return int(reader(device))
    except RuntimeError:
        return None


def _write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"拒绝写入空指标表：{path.name}")
    with path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(args) -> dict:
    input_root = args.input_root.resolve()
    training_run = args.training_run.resolve()
    swinir_run = args.swinir_run.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"输出目录已存在，拒绝覆盖：{output_dir}")
    if not torch.cuda.is_available():
        raise RuntimeError("冻结后验评价要求 CUDA；不回退 CPU")

    from experiments.published_baselines_sar_only_20261123_v1.train_sar_only_diffusion import (
        ConditionalUNet,
        _stable_noise_seed,
        load_split,
        make_beta_schedule,
    )

    training_manifest = json.loads((training_run / "manifest.json").read_text(encoding="utf-8"))
    if (
        training_manifest.get("status") != "complete"
        or training_manifest.get("completed_epochs") != 5
        or training_manifest.get("test_set_accessed") is not False
        or training_manifest.get("confirmation_payload_accessed") is not False
        or training_manifest.get("optical_payload_loaded") is not False
    ):
        raise ValueError("Pilot manifest 未通过数据与完成状态检查")
    protocol_hash = _sha256(PILOT_PROTOCOL)
    if training_manifest.get("protocol_sha256") != protocol_hash:
        raise ValueError("训练 manifest 的 protocol SHA-256 不匹配")
    checkpoint_path = training_run / "checkpoints/best.pt"
    checkpoint_hash = _sha256(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("epoch") != training_manifest.get("best_epoch"):
        raise ValueError("best checkpoint epoch 与训练 manifest 不匹配")
    model = ConditionalUNet(base_channels=32, time_dim=128)
    model.load_state_dict(checkpoint["ema"], strict=True)
    model = model.cuda().eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    records, split_meta = load_split(input_root, "development", 200, 19)
    prediction_root = swinir_run / "development_predictions_swinir"
    baseline_predictions = {}
    baseline_hashes = {}
    for record in records:
        sample_id = record["sample_id"]
        path = prediction_root / f"{sample_id}.npy"
        if not path.is_file():
            raise FileNotFoundError(f"缺少 SwinIR development prediction：{sample_id}")
        baseline_hashes[sample_id] = _sha256(path)
        value = np.load(path, allow_pickle=False)
        if value.dtype != np.float32 or value.shape != (256, 256) or not np.isfinite(value).all():
            raise ValueError(f"SwinIR prediction 数值契约错误：{sample_id}")
        baseline_predictions[sample_id] = value
    if {p.stem for p in prediction_root.glob("*.npy")} != {r["sample_id"] for r in records}:
        raise ValueError("SwinIR prediction 文件集合与 development 不完全一致")

    output_dir.mkdir(parents=True, exist_ok=False)
    prediction_dir = output_dir / "predictions"
    prediction_dir.mkdir()
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    reset_error = None
    try:
        torch.cuda.reset_peak_memory_stats(device)
    except RuntimeError as exc:
        reset_error = str(exc)
    free_before, total_memory = torch.cuda.mem_get_info(device)
    disk_before = shutil.disk_usage(output_dir.parent)
    torch.manual_seed(20261124)
    torch.cuda.manual_seed_all(20261124)
    random.seed(20261124)
    np.random.seed(20261124)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    betas = make_beta_schedule(1000, 1e-4, 0.02).to(device)
    alpha_bar = torch.cumprod(1.0 - betas, dim=0)
    timesteps = torch.linspace(999, 0, 20, device=device).round().long().unique_consecutive()

    manifest = {
        "run_id": output_dir.name,
        "status": "running",
        "training_run": str(training_run),
        "training_checkpoint_sha256": checkpoint_hash,
        "training_code_sha256": training_manifest.get("code_sha256"),
        "protocol_sha256": protocol_hash,
        "sampling_timesteps": [int(value) for value in timesteps.cpu().tolist()],
        "draw_base_seeds": list(DRAW_BASE_SEEDS),
        "guidance_weight": 1.0,
        "input_root": str(input_root),
        "development": split_meta,
        "swinir_prediction_root": str(prediction_root),
        "swinir_prediction_sha256": baseline_hashes,
        "test_set_accessed": False,
        "test_set_used": False,
        "confirmation_payload_accessed": False,
        "optical_payload_loaded": False,
        "training_performed": False,
        "cuda_oom": False,
        "gpu": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "memory_total_bytes": int(total_memory),
        "memory_free_before_bytes": int(free_before),
        "peak_stats_reset_error": reset_error,
        "disk_free_before_bytes": int(disk_before.free),
        "completed_samples": 0,
        "completed_draws": 0,
        "analysis_code_sha256": _sha256(Path(__file__).resolve()),
        "observation_model_sha256": _sha256(Path(__file__).with_name("observation_model.py")),
        "command": [sys.executable, *sys.argv],
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    predictions, observation_metrics = {}, {}
    sample_runtime_seconds = []
    started = time.perf_counter()
    current_sample, current_draw = None, None
    try:
        for image_index, record in enumerate(records):
            sample_id = record["sample_id"]
            current_sample = sample_id
            observation = torch.from_numpy(record["lr"][None]).to(device)
            draw_images, draw_stats = [], []
            for draw_index, base_seed in enumerate(DRAW_BASE_SEEDS):
                current_draw = draw_index
                stable_seed = _stable_noise_seed(base_seed, sample_id, draw_index)
                torch.cuda.synchronize(device)
                draw_started = time.perf_counter()
                output, diagnostics = ddim_posterior_sample(
                    model,
                    observation,
                    alpha_bar,
                    timesteps,
                    stable_seed,
                    guidance_weight=1.0,
                    return_diagnostics=True,
                )
                torch.cuda.synchronize(device)
                sample_runtime_seconds.append(time.perf_counter() - draw_started)
                image = output[0, 0].float().cpu().numpy().astype(np.float32, copy=False)
                draw_images.append(image)
                draw_stats.append(
                    {name: diagnostics[name] for name in OBSERVATION_METRICS}
                    | {
                        "mean_guidance_gradient_norm": diagnostics["mean_guidance_gradient_norm"],
                        "max_guidance_gradient_norm": diagnostics["max_guidance_gradient_norm"],
                    }
                )
            image_stack = np.stack(draw_images).astype(np.float32, copy=False)
            np.save(prediction_dir / f"{sample_id}.npy", image_stack, allow_pickle=False)
            predictions[sample_id] = image_stack
            pixel_mean = image_stack.mean(axis=0, dtype=np.float64).astype(np.float32)
            with torch.no_grad():
                mean_terms = observation_energy(
                    torch.from_numpy(pixel_mean[None, None]).to(device), observation
                )
                baseline_terms = observation_energy(
                    torch.from_numpy(baseline_predictions[sample_id][None, None]).to(device),
                    observation,
                )
            observation_metrics[sample_id] = {
                "draws": draw_stats,
                "pixel_mean": {key: float(value.item()) for key, value in mean_terms.items()},
                "baseline": {key: float(value.item()) for key, value in baseline_terms.items()},
            }
            manifest.update(
                completed_samples=image_index + 1,
                completed_draws=(image_index + 1) * len(DRAW_BASE_SEEDS),
                current_sample_id=sample_id,
                current_draw_index=None,
                elapsed_seconds=time.perf_counter() - started,
                peak_allocated_bytes=_safe_peak_memory_bytes(device),
                peak_reserved_bytes=_safe_peak_memory_bytes(device, reserved=True),
            )
            manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            if (image_index + 1) % 10 == 0:
                print(
                    f"后验采样已完成 {image_index + 1}/200 张（{(image_index + 1) * 4}/800 个 draw）；"
                    f"当前峰值显存 {manifest['peak_allocated_bytes'] / 2**20:.1f} MiB。",
                    flush=True,
                )
        tables, summary = evaluate_development(
            predictions, records, baseline_predictions, observation_metrics
        )
        _write_rows(output_dir / "逐draw指标.csv", tables["single_draw_metrics"])
        _write_rows(output_dir / "像素均值指标.csv", tables["pixel_mean_metrics"])
        _write_rows(output_dir / "SwinIR基线指标.csv", tables["baseline_metrics"])
        _write_rows(output_dir / "来源组统计.csv", tables["source_group_metrics"])
        (output_dir / "汇总.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        torch.cuda.synchronize(device)
        free_after, _ = torch.cuda.mem_get_info(device)
        disk_after = shutil.disk_usage(output_dir.parent)
        manifest.update(
            status="complete",
            end_time_unix=time.time(),
            elapsed_seconds=time.perf_counter() - started,
            sampling_seconds_total=float(sum(sample_runtime_seconds)),
            sampling_seconds_median_per_draw=float(np.median(sample_runtime_seconds)),
            memory_free_after_bytes=int(free_after),
            disk_free_after_bytes=int(disk_after.free),
            peak_allocated_bytes=_safe_peak_memory_bytes(device),
            peak_reserved_bytes=_safe_peak_memory_bytes(device, reserved=True),
            cuda_oom=False,
            summary_sha256=_sha256(output_dir / "汇总.json"),
        )
    except Exception as exc:
        is_oom = isinstance(exc, torch.cuda.OutOfMemoryError) or (
            isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()
        )
        manifest.update(
            status="cuda_oom" if is_oom else "failed",
            end_time_unix=time.time(),
            current_sample_id=current_sample,
            current_draw_index=current_draw,
            cuda_oom=is_oom,
            error=str(exc),
            traceback=traceback.format_exc(),
            elapsed_seconds=time.perf_counter() - started,
            peak_allocated_bytes=_safe_peak_memory_bytes(device),
            peak_reserved_bytes=_safe_peak_memory_bytes(device, reserved=True),
        )
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        raise
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--training-run", type=Path, default=DEFAULT_TRAINING_RUN)
    parser.add_argument("--swinir-run", type=Path, default=DEFAULT_SWINIR_RUN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    run(parser.parse_args())
    print("200 张 development 的四 draw 后验采样与配对评价已完成。", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
