"""训练 SAR noisy pseudo-target 的无条件扩散先验（冻结 5 epoch Pilot）。"""
from __future__ import annotations

import argparse
import copy
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
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "experiments/runs/phase_d2_observation_aware_pilot_20261124_v1"
)
SEED = 20261124
EPOCHS = 5
BATCH_SIZE = 4
LR_PATCH = 64
BASE_CHANNELS = 32
TIME_DIM = 128
DIFFUSION_STEPS = 1000
BETA_START = 1e-4
BETA_END = 0.02
LEARNING_RATE = 1e-4
EMA_DECAY = 0.995


def prepare_training_batch(records, order, rng, lr_patch: int = LR_PATCH):
    """取对齐的 LR/HR patch；HR 映射到 [-1,1]，返回固定零条件。"""
    lr_crops, target_crops = [], []
    for index in order:
        record = records[int(index)]
        lr = record["lr"]
        target = record["target"]
        if lr.dtype != np.float32 or target.dtype != np.float32:
            raise ValueError("SAR LR/target 必须为 float32")
        if lr.ndim != 3 or target.ndim != 3 or lr.shape[0] != 1 or target.shape[0] != 1:
            raise ValueError("SAR LR/target 必须为单通道 [1,H,W]")
        height, width = lr.shape[-2:]
        if target.shape[-2:] != (height * 2, width * 2):
            raise ValueError("SAR target 尺寸必须是 LR 的 2 倍")
        if lr_patch <= 0 or lr_patch > min(height, width):
            raise ValueError("LR patch 尺寸超出影像范围")
        top = int(rng.integers(0, height - lr_patch + 1))
        left = int(rng.integers(0, width - lr_patch + 1))
        lr_crop = lr[:, top : top + lr_patch, left : left + lr_patch]
        hr_top, hr_left, hr_size = top * 2, left * 2, lr_patch * 2
        target_crop = target[
            :, hr_top : hr_top + hr_size, hr_left : hr_left + hr_size
        ]
        if rng.integers(2):
            lr_crop = np.flip(lr_crop, axis=-1)
            target_crop = np.flip(target_crop, axis=-1)
        if rng.integers(2):
            lr_crop = np.flip(lr_crop, axis=-2)
            target_crop = np.flip(target_crop, axis=-2)
        if rng.integers(2):
            lr_crop = np.swapaxes(lr_crop, -1, -2)
            target_crop = np.swapaxes(target_crop, -1, -2)
        lr_crops.append(np.ascontiguousarray(lr_crop))
        target_crops.append(np.ascontiguousarray(target_crop))
    _ = lr_crops  # LR crops define the paired coordinates; the prior does not condition on them.
    target_tensor = torch.from_numpy(np.stack(target_crops)).mul(2.0).sub(1.0)
    return target_tensor, torch.zeros_like(target_tensor)


def epsilon_training_loss(model, target, timesteps, noise, alpha_bar):
    if target.ndim != 4 or target.shape[1] != 1 or target.shape != noise.shape:
        raise ValueError("target/noise 必须为同形 [B,1,H,W]")
    if timesteps.shape != (target.shape[0],):
        raise ValueError("timesteps 必须为 [B]")
    values = alpha_bar.to(device=target.device, dtype=target.dtype)[timesteps]
    values = values.reshape(-1, 1, 1, 1)
    noisy = values.sqrt() * target + (1.0 - values).sqrt() * noise
    condition = torch.zeros_like(target)
    prediction = model(noisy, condition, timesteps)
    if prediction.shape != noise.shape:
        raise ValueError("epsilon prediction shape 与 noise 不一致")
    return F.mse_loss(prediction, noise)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _peak_memory_bytes(device: torch.device, reserved: bool = False) -> int | None:
    reader = torch.cuda.max_memory_reserved if reserved else torch.cuda.max_memory_allocated
    try:
        return int(reader(device))
    except RuntimeError:
        return None


def _load_components():
    from experiments.published_baselines_sar_only_20261123_v1.train_sar_only_diffusion import (
        ConditionalUNet,
        _ema_update,
        load_split,
        make_beta_schedule,
    )

    return ConditionalUNet, _ema_update, load_split, make_beta_schedule


def _group_balanced(values: list[tuple[str, float]]) -> float:
    by_group: dict[str, list[float]] = {}
    for group, value in values:
        by_group.setdefault(group, []).append(float(value))
    return float(np.mean([np.mean(group_values) for group_values in by_group.values()]))


def _validate_epoch(ema, records, alpha_bar, seed, device):
    ema.eval()
    results = []
    with torch.inference_mode():
        for record in records:
            lr, target = record["lr"], record["target"]
            patch = LR_PATCH
            top = (lr.shape[-2] - patch) // 2
            left = (lr.shape[-1] - patch) // 2
            hr = target[:, top * 2 : (top + patch) * 2, left * 2 : (left + patch) * 2]
            x0 = torch.from_numpy(np.ascontiguousarray(hr))[None].to(device).mul(2.0).sub(1.0)
            stable_seed = seed + int(hashlib.sha256(record["sample_id"].encode()).hexdigest()[:8], 16)
            generator = torch.Generator(device="cpu").manual_seed(stable_seed)
            timestep = torch.randint(0, len(alpha_bar), (1,), generator=generator)
            noise = torch.randn(x0.shape, generator=generator, dtype=x0.dtype)
            timestep = timestep.to(device)
            noise = noise.to(device)
            values = alpha_bar[timestep].reshape(-1, 1, 1, 1)
            noisy = values.sqrt() * x0 + (1.0 - values).sqrt() * noise
            predicted = ema(noisy, torch.zeros_like(noisy), timestep)
            loss = F.mse_loss(predicted, noise)
            results.append((record["source_group_id"], float(loss.item())))
    return _group_balanced(results)


def _save_checkpoint(path, *, model, ema, optimizer, epoch, best_metric, seed, np_rng):
    torch.save(
        {
            "model": model.state_dict(),
            "ema": ema.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": int(epoch),
            "best_development_epsilon_mse": float(best_metric),
            "seed": SEED,
            "torch_cpu_rng": torch.get_rng_state(),
            "torch_cuda_rng": torch.cuda.get_rng_state_all(),
            "numpy_rng_state": np_rng.bit_generator.state,
            "python_rng_state": random.getstate(),
        },
        path,
    )


def train(args: SimpleNamespace) -> Path:
    if args.output_dir.exists():
        raise FileExistsError(f"输出目录已存在，拒绝覆盖：{args.output_dir}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用；冻结 Pilot 不回退到 CPU")

    ConditionalUNet, ema_update, load_split, make_beta_schedule = _load_components()
    train_records, train_meta = load_split(args.input_root, "train", 1600, 92)
    dev_records, dev_meta = load_split(args.input_root, "development", 200, 19)
    if {r["source_group_id"] for r in train_records} & {
        r["source_group_id"] for r in dev_records
    }:
        raise ValueError("train/development 来源组重叠")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    checkpoints = args.output_dir / "checkpoints"
    checkpoints.mkdir()
    device = torch.device("cuda:0")
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np_rng = np.random.default_rng(SEED)
    random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.set_device(device)
    peak_stats_reset_error = None
    try:
        torch.cuda.reset_peak_memory_stats(device)
    except RuntimeError as exc:
        # 峰值统计只用于报告；统计 API 异常不应成为训练启动门槛。
        peak_stats_reset_error = str(exc)
    free_before, total_memory = torch.cuda.mem_get_info(device)
    disk_before = shutil.disk_usage(args.output_dir.parent)

    protocol_path = PROJECT_ROOT / "docs/research/PHASE_D2_OBSERVATION_AWARE_PILOT_PROTOCOL_20261124.md"
    protocol_hash = _sha256(protocol_path)
    config = {
        "run_id": args.output_dir.name,
        "method": "无条件 SAR 表示域 noisy pseudo-target 扩散先验；观测似然仅用于后验采样",
        "seed": SEED,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "lr_patch": LR_PATCH,
        "target_patch": LR_PATCH * 2,
        "base_channels": BASE_CHANNELS,
        "time_dim": TIME_DIM,
        "diffusion_steps": DIFFUSION_STEPS,
        "beta_start": BETA_START,
        "beta_end": BETA_END,
        "optimizer": "AdamW",
        "learning_rate": LEARNING_RATE,
        "weight_decay": 0.0,
        "ema_decay": EMA_DECAY,
        "validation_selection": "最低来源组等权 development epsilon MSE",
        "protocol_sha256": protocol_hash,
        "test_set_accessed": False,
        "test_set_used": False,
        "confirmation_payload_accessed": False,
        "optical_payload_loaded": False,
    }
    source_script = Path(__file__).resolve()
    (args.output_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "command.txt").write_text(
        " ".join([sys.executable, *sys.argv]) + "\n", encoding="utf-8"
    )
    environment = (
        f"Python：{platform.python_version()}\n"
        f"Python 可执行文件：{sys.executable}\n"
        f"PyTorch：{torch.__version__}\n"
        f"CUDA 运行时：{torch.version.cuda}\n"
        f"GPU：{torch.cuda.get_device_name(device)}\n"
        f"显存总量 MiB：{total_memory / 2**20:.1f}\n"
        f"启动前空闲显存 MiB：{free_before / 2**20:.1f}\n"
        f"启动前目标分区可用 GiB：{disk_before.free / 2**30:.2f}\n"
    )
    (args.output_dir / "environment.txt").write_text(environment, encoding="utf-8")
    manifest = {
        **config,
        "status": "running",
        "commit": "91ad31107e71f42e028cec984a3c02a62220a710",
        "git_dirty": True,
        "input_root": str(args.input_root),
        "output_dir": str(args.output_dir),
        "splits": {"train": train_meta, "development": dev_meta},
        "code_sha256": _sha256(source_script),
        "observation_model_sha256": _sha256(
            source_script.parent / "observation_model.py"
        ),
        "protocol_sha256": protocol_hash,
        "start_time_unix": time.time(),
        "gpu": torch.cuda.get_device_name(device),
        "memory_total_bytes": int(total_memory),
        "memory_free_before_bytes": int(free_before),
        "disk_free_before_bytes": int(disk_before.free),
        "peak_stats_reset_error": peak_stats_reset_error,
        "completed_epochs": 0,
        "cuda_oom": False,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    log_path = args.output_dir / "train_log.csv"
    fields = (
        "epoch",
        "train_epsilon_mse",
        "development_group_balanced_epsilon_mse",
        "gradient_norm_mean",
        "elapsed_seconds",
        "peak_allocated_mib",
        "peak_reserved_mib",
    )
    started = time.perf_counter()
    best_metric = math.inf
    best_epoch = 0
    try:
        model = ConditionalUNet(base_channels=BASE_CHANNELS, time_dim=TIME_DIM).to(device)
        ema = copy.deepcopy(model).eval()
        for parameter in ema.parameters():
            parameter.requires_grad_(False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.0)
        betas = make_beta_schedule(DIFFUSION_STEPS, BETA_START, BETA_END).to(device)
        alpha_bar = torch.cumprod(1.0 - betas, dim=0)
        with log_path.open("x", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for epoch in range(1, EPOCHS + 1):
                model.train()
                order = np_rng.permutation(len(train_records))
                losses, gradient_norms = [], []
                for start in range(0, len(order), BATCH_SIZE):
                    batch_order = order[start : start + BATCH_SIZE]
                    target, _ = prepare_training_batch(train_records, batch_order, np_rng)
                    target = target.to(device)
                    timesteps = torch.randint(
                        0, DIFFUSION_STEPS, (target.shape[0],), device=device
                    )
                    noise = torch.randn_like(target)
                    loss = epsilon_training_loss(model, target, timesteps, noise, alpha_bar)
                    if not torch.isfinite(loss):
                        raise FloatingPointError("train epsilon MSE 非有限")
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    norms = [p.grad.detach().norm(2) for p in model.parameters() if p.grad is not None]
                    gradient_norm = torch.linalg.vector_norm(torch.stack(norms), 2)
                    if not torch.isfinite(gradient_norm):
                        raise FloatingPointError("梯度范数非有限")
                    optimizer.step()
                    ema_update(ema, model, EMA_DECAY)
                    losses.append(float(loss.item()))
                    gradient_norms.append(float(gradient_norm.item()))
                dev_metric = _validate_epoch(ema, dev_records, alpha_bar, SEED, device)
                if not math.isfinite(dev_metric):
                    raise FloatingPointError("development epsilon MSE 非有限")
                row = {
                    "epoch": epoch,
                    "train_epsilon_mse": float(np.mean(losses)),
                    "development_group_balanced_epsilon_mse": dev_metric,
                    "gradient_norm_mean": float(np.mean(gradient_norms)),
                    "elapsed_seconds": time.perf_counter() - started,
                    "peak_allocated_mib": (
                        None if _peak_memory_bytes(device) is None
                        else _peak_memory_bytes(device) / 2**20
                    ),
                    "peak_reserved_mib": (
                        None if _peak_memory_bytes(device, reserved=True) is None
                        else _peak_memory_bytes(device, reserved=True) / 2**20
                    ),
                }
                writer.writerow(row)
                stream.flush()
                if dev_metric < best_metric:
                    best_metric, best_epoch = dev_metric, epoch
                    _save_checkpoint(
                        checkpoints / "best.pt",
                        model=model,
                        ema=ema,
                        optimizer=optimizer,
                        epoch=epoch,
                        best_metric=best_metric,
                        seed=SEED,
                        np_rng=np_rng,
                    )
                _save_checkpoint(
                    checkpoints / "last.pt",
                    model=model,
                    ema=ema,
                    optimizer=optimizer,
                    epoch=epoch,
                    best_metric=best_metric,
                    seed=SEED,
                    np_rng=np_rng,
                )
                manifest.update(
                    completed_epochs=epoch,
                    best_epoch=best_epoch,
                    best_development_epsilon_mse=best_metric,
                    peak_allocated_bytes=_peak_memory_bytes(device),
                    peak_reserved_bytes=_peak_memory_bytes(device, reserved=True),
                    elapsed_seconds=time.perf_counter() - started,
                )
                manifest_path.write_text(
                    json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                print(
                    f"Pilot epoch {epoch}/{EPOCHS}：train epsilon MSE={row['train_epsilon_mse']:.6f}；"
                    f"development 组等权={dev_metric:.6f}；"
                    f"梯度范数={row['gradient_norm_mean']:.4f}；"
                    f"峰值显存={row['peak_allocated_mib']:.1f} MiB。",
                    flush=True,
                )
        torch.cuda.synchronize(device)
        free_after, _ = torch.cuda.mem_get_info(device)
        disk_after = shutil.disk_usage(args.output_dir.parent)
        manifest.update(
            status="complete",
            end_time_unix=time.time(),
            memory_free_after_bytes=int(free_after),
            disk_free_after_bytes=int(disk_after.free),
            peak_allocated_bytes=_peak_memory_bytes(device),
            peak_reserved_bytes=_peak_memory_bytes(device, reserved=True),
            elapsed_seconds=time.perf_counter() - started,
            cuda_oom=False,
        )
    except Exception as exc:
        is_oom = isinstance(exc, torch.cuda.OutOfMemoryError) or (
            isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()
        )
        manifest.update(
            status="cuda_oom" if is_oom else "failed",
            end_time_unix=time.time(),
            completed_epochs=manifest.get("completed_epochs", 0),
            cuda_oom=is_oom,
            error=str(exc),
            traceback=traceback.format_exc(),
            peak_allocated_bytes=_peak_memory_bytes(device),
            peak_reserved_bytes=_peak_memory_bytes(device, reserved=True),
            elapsed_seconds=time.perf_counter() - started,
        )
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        raise
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return args.output_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
