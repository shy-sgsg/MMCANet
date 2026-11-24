"""训练与评价 SAR-only Conditional Diffusion 和 SwinIR 残差扩散探索。"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import platform
import random
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MMCANetV2.phase_c3a_metrics import build_fixed_masks, edge_metrics, ssim
from scripts.distribution_metrics import (
    adjacent_variogram_score,
    normalized_image_energy_score,
    pixelwise_crps,
)
from experiments.published_baselines_sar_only_20261123_v1.train_swinir_sar import (
    METRIC_NAMES,
    _psnr,
    _write_csv,
    build_swinir,
    load_split,
    load_swinir_class,
    sha256_file,
)

EXPECTED_INPUTS = {
    "train": {
        "manifest_sha256": "64d335f06812859b96b884a5583092df76dd4b6fa181c3a842b18ed5d3bf6eeb",
        "files_csv_sha256": "97f35201125dcb3830e9d0583d3350159f0d6757056c7acf13c4c1a0910466ad",
    },
    "development": {
        "manifest_sha256": "de3101a78105c05653b9779342cabe98d61a9cddac6153b25a7d6347d0ef0da9",
        "files_csv_sha256": "70d762015b2b16a604d381758091ae3d282526a7137e9de51c6db827b60f2839",
    },
}


def make_beta_schedule(steps: int, beta_start: float, beta_end: float) -> torch.Tensor:
    if steps <= 1 or not 0.0 < beta_start < beta_end < 1.0:
        raise ValueError("扩散步数或 beta 端点无效")
    return torch.linspace(beta_start, beta_end, steps, dtype=torch.float32)


def q_sample(
    x0: torch.Tensor,
    timesteps: torch.Tensor,
    noise: torch.Tensor,
    alpha_bar: torch.Tensor,
) -> torch.Tensor:
    if x0.shape != noise.shape or timesteps.ndim != 1 or timesteps.shape[0] != x0.shape[0]:
        raise ValueError("前向扩散张量形状不匹配")
    values = alpha_bar.to(device=x0.device, dtype=x0.dtype)[timesteps]
    values = values.reshape(-1, 1, 1, 1)
    return values.sqrt() * x0 + (1.0 - values).sqrt() * noise


def timestep_embedding(timesteps: torch.Tensor, dimension: int) -> torch.Tensor:
    half = dimension // 2
    frequencies = torch.exp(
        -math.log(10000.0)
        * torch.arange(half, device=timesteps.device, dtype=torch.float32)
        / max(half - 1, 1)
    )
    angles = timesteps.float()[:, None] * frequencies[None, :]
    embedding = torch.cat((torch.sin(angles), torch.cos(angles)), dim=1)
    if dimension % 2:
        embedding = F.pad(embedding, (0, 1))
    return embedding


class TimeResidualBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, time_dimension: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, input_channels)
        self.conv1 = nn.Conv2d(input_channels, output_channels, 3, padding=1)
        self.time_projection = nn.Linear(time_dimension, output_channels)
        self.norm2 = nn.GroupNorm(8, output_channels)
        self.conv2 = nn.Conv2d(output_channels, output_channels, 3, padding=1)
        self.skip = (
            nn.Identity()
            if input_channels == output_channels
            else nn.Conv2d(input_channels, output_channels, 1)
        )

    def forward(self, value: torch.Tensor, time_embedding_value: torch.Tensor):
        hidden = self.conv1(F.silu(self.norm1(value)))
        hidden = hidden + self.time_projection(F.silu(time_embedding_value))[:, :, None, None]
        hidden = self.conv2(F.silu(self.norm2(hidden)))
        return hidden + self.skip(value)


class ConditionalUNet(nn.Module):
    """小型条件 epsilon-prediction U-Net，输入为 noisy image 与 condition。"""

    def __init__(self, base_channels: int = 32, time_dim: int = 128):
        super().__init__()
        if base_channels < 8 or base_channels % 8 or time_dim < 8 or time_dim % 2:
            raise ValueError("U-Net 通道或时间嵌入维度无效")
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 2),
            nn.SiLU(),
            nn.Linear(time_dim * 2, time_dim),
        )
        c1, c2, c3 = base_channels, base_channels * 2, base_channels * 4
        self.input = nn.Conv2d(2, c1, 3, padding=1)
        self.enc0a = TimeResidualBlock(c1, c1, time_dim)
        self.enc0b = TimeResidualBlock(c1, c1, time_dim)
        self.down0 = nn.Conv2d(c1, c2, 3, stride=2, padding=1)
        self.enc1 = TimeResidualBlock(c2, c2, time_dim)
        self.down1 = nn.Conv2d(c2, c3, 3, stride=2, padding=1)
        self.middle = TimeResidualBlock(c3, c3, time_dim)
        self.up1 = nn.Conv2d(c3, c2, 3, padding=1)
        self.dec1a = TimeResidualBlock(c2 * 2, c2, time_dim)
        self.dec1b = TimeResidualBlock(c2, c2, time_dim)
        self.up0 = nn.Conv2d(c2, c1, 3, padding=1)
        self.dec0a = TimeResidualBlock(c1 * 2, c1, time_dim)
        self.dec0b = TimeResidualBlock(c1, c1, time_dim)
        self.output_norm = nn.GroupNorm(8, c1)
        self.output = nn.Conv2d(c1, 1, 3, padding=1)

    def forward(
        self, noisy: torch.Tensor, condition: torch.Tensor, timesteps: torch.Tensor
    ) -> torch.Tensor:
        if noisy.ndim != 4 or condition.shape != noisy.shape or noisy.shape[1] != 1:
            raise ValueError("扩散 U-Net 输入必须为同形单通道张量")
        if any(size % 4 for size in noisy.shape[-2:]):
            raise ValueError("扩散 U-Net 高宽须能被 4 整除")
        time = self.time_mlp(timestep_embedding(timesteps, self.time_dim))
        first = self.enc0b(self.enc0a(self.input(torch.cat((noisy, condition), dim=1)), time), time)
        second = self.enc1(self.down0(first), time)
        middle = self.middle(self.down1(second), time)
        up_second = F.interpolate(middle, size=second.shape[-2:], mode="nearest")
        up_second = self.up1(up_second)
        decoded_second = self.dec1b(self.dec1a(torch.cat((up_second, second), dim=1), time), time)
        up_first = F.interpolate(decoded_second, size=first.shape[-2:], mode="nearest")
        up_first = self.up0(up_first)
        decoded_first = self.dec0b(self.dec0a(torch.cat((up_first, first), dim=1), time), time)
        return self.output(F.silu(self.output_norm(decoded_first)))


def _ema_update(ema: nn.Module, model: nn.Module, decay: float):
    with torch.no_grad():
        for ema_parameter, parameter in zip(ema.parameters(), model.parameters()):
            ema_parameter.mul_(decay).add_(parameter, alpha=1.0 - decay)
        for ema_buffer, buffer in zip(ema.buffers(), model.buffers()):
            ema_buffer.copy_(buffer)


def _condition_from_lr(lr: torch.Tensor) -> torch.Tensor:
    return F.interpolate(lr, scale_factor=2, mode="bicubic", align_corners=False)


def _training_group(records, order, rng, patch_size, variant):
    lr_crops, target_crops, base_crops = [], [], []
    scale = 2
    for index in order:
        record = records[int(index)]
        lr = record["lr"]
        target = record["target"]
        base = record.get("base")
        height, width = lr.shape[-2:]
        top = int(rng.integers(0, height - patch_size + 1))
        left = int(rng.integers(0, width - patch_size + 1))
        lr_crop = lr[:, top : top + patch_size, left : left + patch_size]
        hr_top, hr_left, hr_size = top * scale, left * scale, patch_size * scale
        target_crop = target[:, hr_top : hr_top + hr_size, hr_left : hr_left + hr_size]
        if variant == "residual":
            base_crop = base[:, hr_top : hr_top + hr_size, hr_left : hr_left + hr_size]
        if rng.integers(2):
            lr_crop, target_crop = lr_crop[..., ::-1], target_crop[..., ::-1]
            if variant == "residual":
                base_crop = base_crop[..., ::-1]
        if rng.integers(2):
            lr_crop, target_crop = lr_crop[..., ::-1, :], target_crop[..., ::-1, :]
            if variant == "residual":
                base_crop = base_crop[..., ::-1, :]
        if rng.integers(2):
            lr_crop, target_crop = lr_crop.transpose(0, 2, 1), target_crop.transpose(0, 2, 1)
            if variant == "residual":
                base_crop = base_crop.transpose(0, 2, 1)
        lr_crops.append(np.ascontiguousarray(lr_crop))
        target_crops.append(np.ascontiguousarray(target_crop))
        if variant == "residual":
            base_crops.append(np.ascontiguousarray(base_crop))
    return (
        torch.from_numpy(np.stack(lr_crops)),
        torch.from_numpy(np.stack(target_crops)),
        torch.from_numpy(np.stack(base_crops)) if base_crops else None,
    )


def _make_x0_and_condition(lr, target, base, variant, residual_scale, device):
    lr = lr.to(device)
    target = target.to(device)
    if variant == "conditional":
        return target.mul(2.0).sub(1.0), _condition_from_lr(lr)
    base = base.to(device)
    return (target - base) * residual_scale, base.mul(2.0).sub(1.0)


def _fixed_validation(records, variant, residual_scale, alpha_bar, seed, patch_size, device):
    values = []
    for record in records:
        lr, target = record["lr"], record["target"]
        base = record.get("base")
        top = (lr.shape[-2] - patch_size) // 2
        left = (lr.shape[-1] - patch_size) // 2
        lr_crop = torch.from_numpy(np.ascontiguousarray(lr[:, top : top + patch_size, left : left + patch_size]))[None]
        htop, hleft, hsize = top * 2, left * 2, patch_size * 2
        target_crop = torch.from_numpy(
            np.ascontiguousarray(target[:, htop : htop + hsize, hleft : hleft + hsize])
        )[None]
        base_crop = None
        if variant == "residual":
            base_crop = torch.from_numpy(
                np.ascontiguousarray(base[:, htop : htop + hsize, hleft : hleft + hsize])
            )[None]
        x0, condition = _make_x0_and_condition(
            lr_crop, target_crop, base_crop, variant, residual_scale, device
        )
        stable_seed = seed + int(hashlib.sha256(record["sample_id"].encode()).hexdigest()[:8], 16)
        generator = torch.Generator(device="cpu").manual_seed(stable_seed)
        timestep = torch.randint(0, len(alpha_bar), (1,), generator=generator, dtype=torch.long).to(device)
        noise = torch.randn(x0.shape, generator=generator, dtype=x0.dtype).to(device)
        noisy = q_sample(x0, timestep, noise, alpha_bar)
        values.append(
            {
                "sample_id": record["sample_id"],
                "source_group_id": record["source_group_id"],
                "noisy": noisy,
                "condition": condition,
                "timestep": timestep,
                "noise": noise,
            }
        )
    return values


def _validate_epoch(ema, validation, device):
    ema.eval()
    grouped = {}
    rows = []
    with torch.inference_mode():
        for value in validation:
            prediction = ema(value["noisy"], value["condition"], value["timestep"])
            error = float(F.mse_loss(prediction, value["noise"]).cpu())
            rows.append(
                {
                    "sample_id": value["sample_id"],
                    "source_group_id": value["source_group_id"],
                    "epsilon_mse": error,
                }
            )
            grouped.setdefault(value["source_group_id"], []).append(error)
    score = float(np.mean([np.mean(values) for values in grouped.values()]))
    return rows, score


def _save_checkpoint(path, *, variant, epoch, best_score, model, ema, optimizer, seed, scale, protocol_sha256):
    torch.save(
        {
            "format": "sar_only_conditional_diffusion_v1",
            "variant": variant,
            "epoch": epoch,
            "best_development_group_balanced_epsilon_mse": best_score,
            "model_config": {"type": "ConditionalUNet", "base_channels": 32, "time_dim": 128},
            "diffusion_config": {
                "steps": 1000,
                "beta_start": 1e-4,
                "beta_end": 0.02,
                "prediction": "epsilon",
                "ema_decay": 0.999,
                "residual_scale": scale,
            },
            "seed": seed,
            "protocol_sha256": protocol_sha256,
            "model_state_dict": model.state_dict(),
            "ema_state_dict": ema.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": torch.cuda.get_rng_state_all(),
        },
        path,
    )


def _train_variant(
    variant,
    records_train,
    records_development,
    residual_scale,
    output_dir,
    args,
    device,
    protocol_sha256,
    resume=False,
):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    model = ConditionalUNet().to(device)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=0.0)
    torch.cuda.reset_peak_memory_stats()
    betas = make_beta_schedule(args.diffusion_steps, 1e-4, 0.02).to(device)
    alpha_bar = torch.cumprod(1.0 - betas, dim=0)
    validation = _fixed_validation(
        records_development,
        variant,
        residual_scale,
        alpha_bar,
        args.seed + 4000,
        args.patch_size,
        device,
    )
    method_dir = output_dir / ("conditional" if variant == "conditional" else "residual")
    checkpoint_dir = method_dir / "checkpoints"
    start_epoch = 0
    if resume:
        last_path, best_path = checkpoint_dir / "last.pt", checkpoint_dir / "best.pt"
        if not last_path.is_file() or not best_path.is_file():
            raise FileNotFoundError("续训要求同时存在 last.pt 与 best.pt")
        last_checkpoint = torch.load(last_path, map_location="cpu", weights_only=False)
        best_checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
        for checkpoint, label in ((last_checkpoint, "last"), (best_checkpoint, "best")):
            if (
                checkpoint.get("format") != "sar_only_conditional_diffusion_v1"
                or checkpoint.get("variant") != variant
                or checkpoint.get("seed") != args.seed
                or checkpoint.get("protocol_sha256") != protocol_sha256
            ):
                raise ValueError(f"{label} checkpoint 与续训配置或冻结协议不匹配")
            diffusion_config = checkpoint.get("diffusion_config", {})
            if (
                diffusion_config.get("steps") != args.diffusion_steps
                or not math.isclose(float(diffusion_config.get("residual_scale", -1.0)), residual_scale)
            ):
                raise ValueError(f"{label} checkpoint 的扩散步数或 residual scale 不匹配")
        start_epoch = int(last_checkpoint["epoch"])
        best_epoch = int(best_checkpoint["epoch"])
        best_score = float(best_checkpoint["best_development_group_balanced_epsilon_mse"])
        if start_epoch >= args.epochs or best_epoch > start_epoch:
            raise ValueError(f"checkpoint 轮次无效：last={start_epoch} best={best_epoch} target={args.epochs}")
        model.load_state_dict(last_checkpoint["model_state_dict"], strict=True)
        ema.load_state_dict(last_checkpoint["ema_state_dict"], strict=True)
        optimizer.load_state_dict(last_checkpoint["optimizer_state_dict"])
        log_path = method_dir / "train_log.csv"
        selection_path = method_dir / "development_selection_per_image.csv"
        if not log_path.is_file() or not selection_path.is_file():
            raise FileNotFoundError("续训要求先恢复历史训练日志和 checkpoint 选择轮次的逐图指标")
        with log_path.open(encoding="utf-8-sig", newline="") as stream:
            logs = list(csv.DictReader(stream))
        with selection_path.open(encoding="utf-8-sig", newline="") as stream:
            selection_rows = list(csv.DictReader(stream))
        if [int(row["epoch"]) for row in logs] != list(range(1, start_epoch + 1)):
            raise ValueError("train_log.csv 与 last checkpoint 轮次不连续")
        if not selection_rows or not any(int(row["epoch"]) == best_epoch for row in selection_rows):
            raise ValueError("逐图选择日志缺少 best checkpoint 对应轮次")
        if len(last_checkpoint.get("cuda_rng_state_all", [])) != torch.cuda.device_count():
            raise ValueError("checkpoint CUDA RNG 设备数与当前设备不匹配")
        torch.set_rng_state(last_checkpoint["torch_rng_state"].cpu())
        torch.cuda.set_rng_state_all(last_checkpoint["cuda_rng_state_all"])
    else:
        method_dir.mkdir(parents=True, exist_ok=False)
        checkpoint_dir.mkdir()
        best_score, best_epoch = math.inf, None
        logs, selection_rows = [], []
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()

    for epoch in range(start_epoch + 1, args.epochs + 1):
        model.train()
        order_rng = np.random.default_rng(args.seed * 10 + epoch)
        crop_rng = np.random.default_rng(args.seed + epoch)
        order = order_rng.permutation(len(records_train))
        effective_batch = args.batch_size * args.accumulation_steps
        loss_values = []
        epoch_started = time.perf_counter()
        for group_start in range(0, len(order), effective_batch):
            group_indices = order[group_start : group_start + effective_batch]
            optimizer.zero_grad(set_to_none=True)
            group_size = len(group_indices)
            for micro_start in range(0, group_size, args.batch_size):
                micro_indices = group_indices[micro_start : micro_start + args.batch_size]
                lr_cpu, target_cpu, base_cpu = _training_group(
                    records_train, micro_indices, crop_rng, args.patch_size, variant
                )
                if variant == "conditional":
                    x0, condition = _make_x0_and_condition(
                        lr_cpu, target_cpu, None, variant, residual_scale, device
                    )
                else:
                    x0, condition = _make_x0_and_condition(
                        lr_cpu, target_cpu, base_cpu, variant, residual_scale, device
                    )
                timesteps = torch.randint(0, args.diffusion_steps, (len(micro_indices),), device=device)
                noise = torch.randn_like(x0)
                noisy = q_sample(x0, timesteps, noise, alpha_bar)
                prediction = model(noisy, condition, timesteps)
                loss = F.mse_loss(prediction, noise)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"{variant} 第 {epoch} 轮损失非有限")
                (loss * (len(micro_indices) / group_size)).backward()
                loss_values.extend([float(loss.detach())] * len(micro_indices))
            gradients = [
                torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
                if parameter.grad is not None
            ]
            if gradients and not bool(torch.stack(gradients).all().item()):
                raise FloatingPointError(f"{variant} 第 {epoch} 轮梯度含非有限值")
            optimizer.step()
            _ema_update(ema, model, 0.999)

        validation_rows, score = _validate_epoch(ema, validation, device)
        selection_rows.extend({"epoch": epoch, **row} for row in validation_rows)
        epoch_seconds = time.perf_counter() - epoch_started
        log_row = {
            "epoch": epoch,
            "train_epsilon_mse": float(np.mean(loss_values)),
            "development_group_balanced_epsilon_mse": score,
            "epoch_seconds": epoch_seconds,
            "peak_allocated_mib": torch.cuda.max_memory_allocated() / (1024**2),
            "peak_reserved_mib": torch.cuda.max_memory_reserved() / (1024**2),
        }
        logs.append(log_row)
        checkpoint_args = {
            "variant": variant,
            "epoch": epoch,
            "best_score": best_score,
            "model": model,
            "ema": ema,
            "optimizer": optimizer,
            "seed": args.seed,
            "scale": residual_scale,
            "protocol_sha256": protocol_sha256,
        }
        _save_checkpoint(checkpoint_dir / "last.pt", **checkpoint_args)
        if score < best_score:
            best_score, best_epoch = score, epoch
            checkpoint_args["best_score"] = best_score
            _save_checkpoint(checkpoint_dir / "best.pt", **checkpoint_args)
        _write_csv(method_dir / "train_log.csv", logs)
        _write_csv(method_dir / "development_selection_per_image.csv", selection_rows)
        print(
            f"{variant} 第 {epoch:02d}/{args.epochs} 轮：训练 epsilon MSE={log_row['train_epsilon_mse']:.6f}，"
            f"开发组均衡 MSE={score:.6f}，峰值显存={log_row['peak_allocated_mib']:.1f} MiB，"
            f"耗时={epoch_seconds:.1f} 秒",
            flush=True,
        )
    best = torch.load(checkpoint_dir / "best.pt", map_location="cpu", weights_only=False)
    ema.load_state_dict(best["ema_state_dict"], strict=True)
    ema.to(device).eval()
    training_runtime = {
        "training_seconds": float(sum(float(row["epoch_seconds"]) for row in logs)),
        "optimizer_updates": args.epochs * math.ceil(len(records_train) / (args.batch_size * args.accumulation_steps)),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "peak_allocated_mib": max(float(row["peak_allocated_mib"]) for row in logs),
        "peak_reserved_mib": max(
            [float(row["peak_reserved_mib"]) for row in logs if row.get("peak_reserved_mib") not in (None, "")]
            or [torch.cuda.max_memory_reserved() / (1024**2)]
        ),
        "elapsed_since_resume_start_seconds": time.perf_counter() - started,
    }
    return ema, best_epoch, best_score, training_runtime, method_dir


def _stable_noise_seed(run_seed, sample_id, draw):
    value = f"{run_seed}|{sample_id}|{draw}".encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "big") % (2**32 - 1)


def _ddim_sample(model, initial_noise, condition, alpha_bar, steps, clip_x0):
    sample = initial_noise
    total = len(alpha_bar)
    timesteps = torch.linspace(total - 1, 0, steps, device=sample.device).round().long().unique_consecutive()
    with torch.inference_mode():
        for index, timestep in enumerate(timesteps):
            t = timestep.expand(sample.shape[0])
            alpha = alpha_bar[timestep]
            predicted_noise = model(sample, condition, t)
            predicted_x0 = (sample - (1.0 - alpha).sqrt() * predicted_noise) / alpha.sqrt()
            if clip_x0:
                predicted_x0 = predicted_x0.clamp(-1.0, 1.0)
            if index + 1 == len(timesteps):
                sample = predicted_x0
            else:
                previous = timesteps[index + 1]
                alpha_previous = alpha_bar[previous]
                sample = alpha_previous.sqrt() * predicted_x0 + (1.0 - alpha_previous).sqrt() * predicted_noise
    return sample


def _metrics_for_prediction(prediction, record):
    target = record["target"][0]
    lr = record["lr"]
    masks, thresholds = build_fixed_masks(lr, record["target"], 2)
    edge = edge_metrics(prediction, target, masks["edge"])
    bright_prediction = prediction[masks["high_return"]]
    bright_target = target[masks["high_return"]]
    background_prediction = prediction[masks["background"]]
    background_target = target[masks["background"]]
    return {
        "psnr_db": _psnr(prediction, target),
        "ssim": ssim(prediction, target, data_range=1.0),
        "epi_corr": edge["epi_corr"],
        "edge_strength_ratio": edge["edge_strength_ratio"],
        "bright_mean_ratio": float(bright_prediction.mean() / (bright_target.mean() + 1e-12)),
        "bright_p95_ratio": float(
            np.percentile(bright_prediction, 95, method="linear")
            / (np.percentile(bright_target, 95, method="linear") + 1e-12)
        ),
        "background_mean_ratio": float(
            background_prediction.mean() / (background_target.mean() + 1e-12)
        ),
        "high_return_threshold": thresholds["high_return"],
        "background_threshold": thresholds["background"],
    }


def _sample_development(
    variant,
    model,
    records,
    residual_scale,
    alpha_bar,
    args,
    output_dir,
    device,
):
    prediction_dir = output_dir / "predictions" / variant
    prediction_dir.mkdir(parents=True, exist_ok=False)
    draw_rows, mean_rows, distribution_rows = [], [], []
    runtime_values = []
    repeat_error = 0.0
    model.eval()
    for image_index, record in enumerate(records):
        lr = torch.from_numpy(record["lr"][None]).to(device)
        base = torch.from_numpy(record["base"][None]).to(device) if variant == "residual" else None
        condition = _condition_from_lr(lr) if variant == "conditional" else base.mul(2.0).sub(1.0)
        images = []
        for draw in range(args.draws):
            seed = _stable_noise_seed(args.seed + 7000, record["sample_id"], draw)
            generator = torch.Generator(device="cpu").manual_seed(seed)
            initial = torch.randn((1, 1, *record["target"].shape[-2:]), generator=generator)
            initial = initial.to(device)
            torch.cuda.synchronize()
            started = time.perf_counter()
            normalized = _ddim_sample(
                model,
                initial,
                condition,
                alpha_bar,
                args.ddim_steps,
                clip_x0=(variant == "conditional"),
            )
            torch.cuda.synchronize()
            runtime_values.append(time.perf_counter() - started)
            if variant == "conditional":
                unbounded_prediction = normalized.add(1.0).mul(0.5)
                clipped_fraction = float(
                    ((unbounded_prediction < 0.0) | (unbounded_prediction > 1.0)).float().mean().cpu()
                )
                prediction = unbounded_prediction.clamp(0.0, 1.0)
            else:
                unbounded_prediction = base + normalized / residual_scale
                clipped_fraction = float(
                    ((unbounded_prediction < 0.0) | (unbounded_prediction > 1.0)).float().mean().cpu()
                )
                prediction = unbounded_prediction.clamp(0.0, 1.0)
            prediction = prediction[0, 0].float().cpu().numpy()
            images.append(prediction)
            metrics = _metrics_for_prediction(prediction, record)
            draw_rows.append(
                {
                    "method": variant,
                    "sample_id": record["sample_id"],
                    "source_group_id": record["source_group_id"],
                    "draw": draw,
                    "output_clipped_fraction": clipped_fraction,
                    **metrics,
                }
            )
            if image_index == 0 and draw == 0:
                repeat = _ddim_sample(
                    model,
                    initial,
                    condition,
                    alpha_bar,
                    args.ddim_steps,
                    clip_x0=(variant == "conditional"),
                )
                if variant == "conditional":
                    repeat = repeat.add(1.0).mul(0.5).clamp(0.0, 1.0)
                else:
                    repeat = (base + repeat / residual_scale).clamp(0.0, 1.0)
                repeat_error = float((repeat[0, 0].float().cpu() - torch.from_numpy(prediction)).abs().max())
        image_set = np.stack(images).astype(np.float32)
        np.save(prediction_dir / f"{record['sample_id']}.npy", image_set, allow_pickle=False)
        pixel_mean = image_set.mean(axis=0)
        mean_rows.append(
            {
                "method": variant,
                "sample_id": record["sample_id"],
                "source_group_id": record["source_group_id"],
                **_metrics_for_prediction(pixel_mean, record),
                "mean_pixelwise_sample_std": float(image_set.std(axis=0).mean()),
                "draw_pair_mean_absolute_difference": float(
                    np.mean([np.abs(image_set[i] - image_set[j]).mean() for i in range(len(images)) for j in range(i + 1, len(images))])
                ),
            }
        )
        observed = record["target"][0]
        distribution_rows.append(
            {
                "method": variant,
                "sample_id": record["sample_id"],
                "source_group_id": record["source_group_id"],
                "pixelwise_crps": pixelwise_crps(image_set, observed),
                "normalized_image_energy_score": normalized_image_energy_score(image_set, observed),
                "adjacent_variogram_score": adjacent_variogram_score(image_set, observed),
            }
        )
        if (image_index + 1) % 25 == 0:
            print(f"{variant} 已完成开发样本 {image_index + 1}/{len(records)}", flush=True)
    return draw_rows, mean_rows, distribution_rows, {
        "draw_count_per_image": args.draws,
        "ddim_steps": args.ddim_steps,
        "sampling_seconds_total": float(sum(runtime_values)),
        "sampling_seconds_median_per_draw": float(np.median(runtime_values)),
        "repeat_same_noise_max_abs_difference": repeat_error,
    }


def _group_balanced(rows, metric):
    groups = {}
    for row in rows:
        groups.setdefault(row["source_group_id"], []).append(float(row[metric]))
    return float(np.mean([np.mean(values) for values in groups.values()]))


def _draw_metric_spread(draw_rows):
    by_sample = {}
    for row in draw_rows:
        by_sample.setdefault(row["sample_id"], []).append(row)
    return {
        metric: float(
            np.mean(
                [
                    np.std([float(row[metric]) for row in rows], ddof=0)
                    for rows in by_sample.values()
                ]
            )
        )
        for metric in METRIC_NAMES
    }


def _average_draws_by_sample(draw_rows):
    """先在每幅图上对 draw 指标取算术均值，供图像级配对比较。"""
    grouped = {}
    for row in draw_rows:
        key = (row["method"], row["sample_id"], row["source_group_id"])
        grouped.setdefault(key, []).append(row)
    averaged = []
    for (method, sample_id, source_group_id), rows in sorted(grouped.items()):
        result = {
            "method": method,
            "sample_id": sample_id,
            "source_group_id": source_group_id,
        }
        for metric in METRIC_NAMES:
            result[metric] = float(np.mean([float(row[metric]) for row in rows]))
        averaged.append(result)
    return averaged


def _paired_bootstrap(method_rows, reference_rows, seed):
    reference = {row["sample_id"]: row for row in reference_rows}
    if set(reference) != {row["sample_id"] for row in method_rows}:
        raise ValueError("扩散与确定性 baseline 开发样本集合不一致")
    paired = []
    for row in method_rows:
        base = reference[row["sample_id"]]
        output = {"sample_id": row["sample_id"], "source_group_id": row["source_group_id"]}
        for metric in METRIC_NAMES:
            output[f"method_{metric}"] = row[metric]
            output[f"swinir_{metric}"] = base[metric]
            output[f"delta_{metric}"] = float(row[metric]) - float(base[metric])
        paired.append(output)
    groups = {}
    for row in paired:
        groups.setdefault(row["source_group_id"], []).append(row)
    group_names = sorted(groups)
    rng = np.random.default_rng(seed)
    summary = {}
    for metric in METRIC_NAMES:
        values = np.asarray(
            [np.mean([row[f"delta_{metric}"] for row in groups[name]]) for name in group_names],
            dtype=np.float64,
        )
        draws = rng.choice(values, size=(10000, len(values)), replace=True).mean(axis=1)
        summary[metric] = {
            "group_balanced_delta": float(values.mean()),
            "group_bootstrap_95_ci": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
        }
    return paired, summary


def _source_group_statistics(draw_rows, mean_rows, distribution_rows):
    output = []
    for method in ("conditional", "residual"):
        by_group_mean, by_group_draw, by_group_distribution = {}, {}, {}
        for row in mean_rows:
            if row["method"] == method:
                by_group_mean.setdefault(row["source_group_id"], []).append(row)
        for row in draw_rows:
            if row["method"] == method:
                by_group_draw.setdefault(row["source_group_id"], []).append(row)
        for row in distribution_rows:
            if row["method"] == method:
                by_group_distribution.setdefault(row["source_group_id"], []).append(row)
        for group in sorted(by_group_mean):
            means = by_group_mean[group]
            draws = by_group_draw[group]
            distributions = by_group_distribution[group]
            result = {
                "method": method,
                "source_group_id": group,
                "sample_count": len(means),
            }
            for metric in METRIC_NAMES:
                result[f"pixel_mean_{metric}"] = float(np.mean([float(row[metric]) for row in means]))
                result[f"single_draw_{metric}"] = float(np.mean([float(row[metric]) for row in draws]))
            for metric in ("pixelwise_crps", "normalized_image_energy_score", "adjacent_variogram_score"):
                result[metric] = float(np.mean([float(row[metric]) for row in distributions]))
            result["mean_pixelwise_sample_std"] = float(
                np.mean([float(row["mean_pixelwise_sample_std"]) for row in means])
            )
            result["draw_pair_mean_absolute_difference"] = float(
                np.mean([float(row["draw_pair_mean_absolute_difference"]) for row in means])
            )
            result["output_clipped_fraction_mean"] = float(
                np.mean([float(row["output_clipped_fraction"]) for row in draws])
            )
            output.append(result)
    return output


def run(args):
    input_root, swinir_run, output_dir = map(
        lambda value: Path(value).resolve(), (args.input_root, args.swinir_run, args.output_dir)
    )
    if not torch.cuda.is_available():
        raise RuntimeError("扩散 run 固定要求 CUDA")
    if output_dir.exists():
        raise FileExistsError(f"输出目录已存在，拒绝覆盖：{output_dir}")
    protocol_path = PROJECT_ROOT / "docs/research/SAR_ONLY_DIFFUSION_EXPERIMENT_PROTOCOL_20261123.md"
    protocol_sha256 = sha256_file(protocol_path)
    swinir_checkpoint_path = swinir_run / "checkpoints/best.pt"
    if not swinir_checkpoint_path.is_file():
        raise FileNotFoundError("SAR-only SwinIR best checkpoint 不存在")
    swinir_checkpoint_sha256 = sha256_file(swinir_checkpoint_path)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda:0")
    train_records, train_manifest = load_split(
        input_root / "x2_light", "train", 1600, 92
    )
    development_records, development_manifest = load_split(
        input_root / "x2_light", "development", 200, 19
    )
    for role, record in (("train", train_manifest), ("development", development_manifest)):
        expected = EXPECTED_INPUTS[role]
        if (
            record["manifest_sha256"] != expected["manifest_sha256"]
            or record["files_csv_sha256"] != expected["files_csv_sha256"]
        ):
            raise ValueError(f"{role} 固定输入不再匹配 C1 v2 x2_light 核验哈希")
    train_groups = {row["source_group_id"] for row in train_records}
    development_groups = {row["source_group_id"] for row in development_records}
    if train_groups & development_groups:
        raise ValueError("train/development 来源组重叠")

    def git_command(*command):
        return subprocess.check_output(["git", *command], cwd=PROJECT_ROOT, text=True).strip()

    git_status = git_command("status", "--short", "--untracked-files=all")
    git_state = {
        "提交": git_command("rev-parse", "HEAD"),
        "分支": git_command("branch", "--show-current"),
        "工作区脏状态": bool(git_status),
        "状态项数": len(git_status.splitlines()),
    }
    environment = {
        "Python": sys.version,
        "Python可执行文件": sys.executable,
        "PyTorch": torch.__version__,
        "CUDA运行时": torch.version.cuda,
        "cuDNN": torch.backends.cudnn.version(),
        "NumPy": np.__version__,
        "平台": platform.platform(),
        "GPU": torch.cuda.get_device_name(0),
        "GPU总显存字节": torch.cuda.get_device_properties(0).total_memory,
        "GPU启动前可用显存字节": int(torch.cuda.mem_get_info()[0]),
        "磁盘启动前可用字节": shutil.disk_usage(output_dir.parent).free,
    }
    configuration = {
        "input_root": str(input_root),
        "swinir_run": str(swinir_run),
        "output_dir": str(output_dir),
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "accumulation_steps": args.accumulation_steps,
        "effective_batch_size": args.batch_size * args.accumulation_steps,
        "patch_size_lr": args.patch_size,
        "learning_rate": args.learning_rate,
        "diffusion_steps": args.diffusion_steps,
        "ddim_steps": args.ddim_steps,
        "draws": args.draws,
        "beta_start": 1e-4,
        "beta_end": 0.02,
        "ema_decay": 0.999,
        "amp": False,
    }
    config_bytes = (json.dumps(configuration, sort_keys=True, indent=2) + "\n").encode("utf-8")

    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "config.json").write_bytes(config_bytes)
    (output_dir / "environment.json").write_text(
        json.dumps(environment, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "git_state.txt").write_text(
        "\n".join(f"{key}：{value}" for key, value in git_state.items()) + "\n",
        encoding="utf-8",
    )
    (output_dir / "command.txt").write_text(
        "环境变量：CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=20261124\n"
        + "命令："
        + shlex.join([sys.executable, *sys.argv])
        + "\n",
        encoding="utf-8",
    )
    manifest = {
        "format": "sar_only_conditional_diffusion_exploration_v1",
        "run_id": output_dir.name,
        "status": "running",
        "started_unix": time.time(),
        "protocol_path": str(protocol_path.relative_to(PROJECT_ROOT)),
        "protocol_sha256": protocol_sha256,
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "git": git_state,
        "environment": environment,
        "swinir_checkpoint_path": str(swinir_checkpoint_path),
        "swinir_checkpoint_sha256": swinir_checkpoint_sha256,
        "input_root": str(input_root),
        "inputs": {"train": train_manifest, "development": development_manifest},
        "test_set_accessed": False,
        "test_set_used": False,
        "confirmation_payload_accessed": False,
        "optical_payload_loaded": False,
        "resource_override": "用户要求不拦截并行评价任务、不使用 3 GiB 空闲显存门槛，以真实 OOM 为停止条件",
        "cuda_total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        "cuda_free_memory_bytes_before_run": int(torch.cuda.mem_get_info()[0]),
        "disk_free_bytes_before_run": shutil.disk_usage(output_dir).free,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    run_started = time.perf_counter()
    try:
        SwinIR = load_swinir_class()
        swinir = build_swinir(SwinIR).to(device)
        swinir_checkpoint = torch.load(swinir_checkpoint_path, map_location="cpu", weights_only=False)
        swinir.load_state_dict(swinir_checkpoint["model_state_dict"], strict=True)
        swinir.eval()
        base_dir = output_dir / "swinir_base_predictions"
        baseline_seconds = 0.0
        for role, records in (("train", train_records), ("development", development_records)):
            role_dir = base_dir / role
            role_dir.mkdir(parents=True, exist_ok=False)
            with torch.inference_mode():
                for index, record in enumerate(records):
                    lr = torch.from_numpy(record["lr"][None]).to(device)
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    base = swinir(lr)[0, 0].float().cpu().numpy()
                    torch.cuda.synchronize()
                    baseline_seconds += time.perf_counter() - started
                    record["base"] = base[None].astype(np.float32)
                    np.save(role_dir / f"{record['sample_id']}.npy", base.astype(np.float32), allow_pickle=False)
        squared_errors = [float(np.mean((record["target"] - record["base"]) ** 2)) for record in train_records]
        residual_rms = math.sqrt(float(np.mean(squared_errors)))
        if not math.isfinite(residual_rms) or residual_rms <= 1e-8:
            raise ValueError("train residual RMS 无效")
        residual_scale = 1.0 / residual_rms
        print(
            f"已核验 train={len(train_records)}/development={len(development_records)}；"
            f"SwinIR train-only residual RMS={residual_rms:.6f}；"
            f"当前空闲显存={torch.cuda.mem_get_info()[0] / 2**20:.1f} MiB",
            flush=True,
        )

        variant_results = {}
        all_draw_rows, all_mean_rows, all_distribution_rows = [], [], []
        for variant in ("conditional", "residual"):
            manifest["active_variant"] = variant
            manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            model, best_epoch, best_score, training_runtime, method_dir = _train_variant(
                variant,
                train_records,
                development_records,
                residual_scale,
                output_dir,
                args,
                device,
                protocol_sha256,
            )
            draw_rows, mean_rows, distribution_rows, sampling_runtime = _sample_development(
                variant,
                model,
                development_records,
                residual_scale,
                torch.cumprod(
                    1.0 - make_beta_schedule(args.diffusion_steps, 1e-4, 0.02).to(device), dim=0
                ),
                args,
                output_dir,
                device,
            )
            _write_csv(method_dir / "development_per_draw_metrics.csv", draw_rows)
            _write_csv(method_dir / "development_pixel_mean_metrics.csv", mean_rows)
            _write_csv(method_dir / "development_distribution_scores.csv", distribution_rows)
            all_draw_rows.extend(draw_rows)
            all_mean_rows.extend(mean_rows)
            all_distribution_rows.extend(distribution_rows)
            single_draw_rows = [row for row in draw_rows if row["method"] == variant]
            variant_results[variant] = {
                "best_epoch": best_epoch,
                "development_group_balanced_epsilon_mse": best_score,
                "training": training_runtime,
                "sampling": sampling_runtime,
                "single_draw_group_balanced_metrics": {
                    metric: _group_balanced(single_draw_rows, metric) for metric in METRIC_NAMES
                },
                "single_draw_metric_standard_deviation_across_draws": _draw_metric_spread(single_draw_rows),
                "group_balanced_pixel_mean_metrics": {
                    metric: _group_balanced(mean_rows, metric) for metric in METRIC_NAMES
                },
                "mean_pixelwise_sample_std": float(np.mean([row["mean_pixelwise_sample_std"] for row in mean_rows])),
                "draw_pair_mean_absolute_difference": float(np.mean([row["draw_pair_mean_absolute_difference"] for row in mean_rows])),
                "output_clipped_fraction_mean": float(
                    np.mean([row["output_clipped_fraction"] for row in single_draw_rows])
                ),
                "group_balanced_distribution_scores": {
                    metric: _group_balanced(distribution_rows, metric)
                    for metric in ("pixelwise_crps", "normalized_image_energy_score", "adjacent_variogram_score")
                },
                "checkpoint_sha256": {
                    "best": sha256_file(method_dir / "checkpoints/best.pt"),
                    "last": sha256_file(method_dir / "checkpoints/last.pt"),
                },
            }

        _write_csv(output_dir / "all_methods_per_draw_metrics.csv", all_draw_rows)
        _write_csv(output_dir / "all_methods_pixel_mean_metrics.csv", all_mean_rows)
        _write_csv(output_dir / "all_methods_distribution_scores.csv", all_distribution_rows)
        mean_draw_rows = _average_draws_by_sample(all_draw_rows)
        _write_csv(output_dir / "all_methods_mean_draw_metrics_per_image.csv", mean_draw_rows)
        _write_csv(
            output_dir / "source_group_stats.csv",
            _source_group_statistics(all_draw_rows, all_mean_rows, all_distribution_rows),
        )
        swinir_rows = list(csv.DictReader((swinir_run / "development_metrics_swinir.csv").open(encoding="utf-8-sig", newline="")))
        c2_rows = list(csv.DictReader((swinir_run / "development_metrics_internal_c2.csv").open(encoding="utf-8-sig", newline="")))
        comparison_rows, comparisons = [], {}
        mean_draw_comparison_rows = []
        for variant in ("conditional", "residual"):
            rows = [row for row in all_mean_rows if row["method"] == variant]
            paired, summary = _paired_bootstrap(rows, swinir_rows, args.seed + 1000 + (variant == "residual"))
            for row in paired:
                row["method"] = variant
            comparison_rows.extend(paired)
            comparisons[variant] = summary
            c2_paired, c2_summary = _paired_bootstrap(rows, c2_rows, args.seed + 2000 + (variant == "residual"))
            comparisons[variant]["versus_internal_C2"] = c2_summary
            del c2_paired
            draw_rows_for_method = [row for row in mean_draw_rows if row["method"] == variant]
            draw_paired, draw_summary = _paired_bootstrap(
                draw_rows_for_method,
                swinir_rows,
                args.seed + 1500 + (variant == "residual"),
            )
            for row in draw_paired:
                row["method"] = variant
            mean_draw_comparison_rows.extend(draw_paired)
            comparisons[variant]["mean_draw_metrics_vs_swinir"] = draw_summary
        _write_csv(output_dir / "paired_group_bootstrap_vs_swinir.csv", comparison_rows)
        _write_csv(
            output_dir / "paired_mean_draw_metrics_vs_swinir.csv",
            mean_draw_comparison_rows,
        )
        residual_rows = [row for row in all_mean_rows if row["method"] == "residual"]
        conditional_rows = [row for row in all_mean_rows if row["method"] == "conditional"]
        direct_paired, direct_summary = _paired_bootstrap(
            residual_rows, conditional_rows, args.seed + 3000
        )
        for row in direct_paired:
            for metric in METRIC_NAMES:
                row[f"conditional_{metric}"] = row.pop(f"swinir_{metric}")
        comparisons["residual"]["versus_conditional_diffusion"] = direct_summary
        _write_csv(output_dir / "paired_residual_vs_conditional.csv", direct_paired)

        residual_mean_draws = [row for row in mean_draw_rows if row["method"] == "residual"]
        conditional_mean_draws = [row for row in mean_draw_rows if row["method"] == "conditional"]
        draw_direct_paired, draw_direct_summary = _paired_bootstrap(
            residual_mean_draws, conditional_mean_draws, args.seed + 3500
        )
        for row in draw_direct_paired:
            for metric in METRIC_NAMES:
                row[f"conditional_{metric}"] = row.pop(f"swinir_{metric}")
        comparisons["residual"]["mean_draw_metrics_vs_conditional_diffusion"] = draw_direct_summary
        _write_csv(
            output_dir / "paired_residual_vs_conditional_mean_draw_metrics.csv",
            draw_direct_paired,
        )

        summary = {
            "interpretation_scope": "单 seed、历史重复使用 development 的探索观察；不构成独立泛化或校准不确定性证据",
            "train_only_residual_rms": residual_rms,
            "train_only_residual_scale": residual_scale,
            "SwinIR_SAR_deterministic_reference": {
                "checkpoint_sha256": swinir_checkpoint_sha256,
                "best_epoch": swinir_checkpoint.get("best_epoch"),
                "group_balanced_metrics": {
                    metric: _group_balanced(swinir_rows, metric) for metric in METRIC_NAMES
                },
            },
            "internal_C2_deterministic_reference": {
                "checkpoint_sha256": "8255b5229ca1e041cb59aed582d794bb67afdbe1601b6f4dc425c43ce27f7165",
                "group_balanced_metrics": {
                    metric: _group_balanced(c2_rows, metric) for metric in METRIC_NAMES
                },
            },
            "diffusion_variants": variant_results,
            "paired_group_bootstrap_vs_swinir": comparisons,
            "limitations": [
                "4 draws 只支撑小样本描述性集合评分。",
                "CRPS、energy score 与 adjacent variogram score 只作为 noisy pseudo-target 上的经验评分，不证明分布校准。",
                "PSNR/SSIM 只衡量表示域 noisy pseudo-target 一致性。",
                "SwinIR 本身是发布架构任务适配版，非 SAR 专用论文实现。",
                "residual diffusion 依赖已在同一历史重复开发集上选择的 SwinIR checkpoint。",
            ],
        }
        (output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        runtime = {
            "total_seconds": time.perf_counter() - run_started,
            "swinir_base_inference_seconds_train_and_development": baseline_seconds,
            "cuda_peak_allocated_mib": torch.cuda.max_memory_allocated() / (1024**2),
            "cuda_peak_reserved_mib": torch.cuda.max_memory_reserved() / (1024**2),
            "cuda_free_memory_bytes_after_run": int(torch.cuda.mem_get_info()[0]),
            "disk_free_bytes_after_run": shutil.disk_usage(output_dir).free,
            "parameters": sum(parameter.numel() for parameter in ConditionalUNet().parameters()),
            "optimizer_updates_per_variant": args.epochs
            * math.ceil(1600 / (args.batch_size * args.accumulation_steps)),
            "generated_image_count_per_variant": len(development_records) * args.draws,
            "prediction_array_shape_per_sample": [args.draws, 256, 256],
        }
        (output_dir / "runtime.json").write_text(
            json.dumps(runtime, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        manifest.update(
            {
                "status": "complete",
                "ended_unix": time.time(),
                "train_only_residual_rms": residual_rms,
                "train_only_residual_scale": residual_scale,
                "code_sha256": sha256_file(Path(__file__)),
                "swinir_source_sha256": sha256_file(PROJECT_ROOT / "BasicSR/basicsr/archs/swinir_arch.py"),
                "runtime_path": "runtime.json",
                "outputs": sorted(path.relative_to(output_dir).as_posix() for path in output_dir.rglob("*") if path.is_file()),
            }
        )
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"SAR-only diffusion 探索完成：{output_dir}", flush=True)
        return summary
    except BaseException as error:
        manifest.update(
            {
                "status": "failed",
                "ended_unix": time.time(),
                "failure_type": type(error).__name__,
                "failure_message": str(error),
                "checkpoint_files_preserved": sorted(
                    path.relative_to(output_dir).as_posix()
                    for path in output_dir.glob("*/checkpoints/*.pt")
                    if path.is_file()
                ),
            }
        )
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        raise


def main():
    parser = argparse.ArgumentParser(description="训练 SAR-only 条件扩散与 SwinIR 残差扩散")
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--swinir-run", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--accumulation-steps", type=int, default=1)
    parser.add_argument("--patch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--diffusion-steps", type=int, default=1000)
    parser.add_argument("--ddim-steps", type=int, default=20)
    parser.add_argument("--draws", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261124)
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.accumulation_steps, args.patch_size) <= 0:
        parser.error("训练轮数、批次、梯度累积和裁剪尺寸必须为正")
    if args.patch_size % 4 or args.diffusion_steps < 20 or args.ddim_steps < 2 or args.draws < 2:
        parser.error("扩散步数、U-Net patch 或 draws 设置无效")
    run(args)


if __name__ == "__main__":
    main()
