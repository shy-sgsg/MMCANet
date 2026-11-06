"""C3-A 像素域 SAR-only 条件流匹配筛选模型。"""
from __future__ import annotations

import math
import hashlib
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import Dataset


def _groups(channels: int) -> int:
    for count in (8, 4, 2, 1):
        if channels % count == 0:
            return count
    return 1


def _check_image(name: str, value: Tensor) -> None:
    if value.ndim != 4 or value.shape[1] != 1:
        raise ValueError(f"{name} 必须为 [B,1,H,W]，实际为 {tuple(value.shape)}")
    if not torch.is_floating_point(value) or not torch.isfinite(value).all():
        raise ValueError(f"{name} 必须为有限浮点张量")


class SarTargetOnlyDataset(Dataset):
    """只读取冻结输入中的 SAR LR 与 pseudo-target，不打开 optical 文件。"""

    def __init__(self, fixed_dataset: object) -> None:
        if getattr(fixed_dataset, "role", None) not in {"train", "development"}:
            raise ValueError("C3-A 数据集只接受 C1 v2 train/development")
        if getattr(fixed_dataset, "scale", None) != 2:
            raise ValueError("C3-A 首轮只接受 x2 固定输入")
        if getattr(fixed_dataset, "manifest", {}).get("condition_id") != "x2_light":
            raise ValueError("C3-A 首轮只接受 x2_light 固定输入")
        self.root = Path(fixed_dataset.root)
        self.role = fixed_dataset.role
        self.rows = list(fixed_dataset.rows)
        self.manifest_sha256 = self._sha256(self.root / "manifest.json")
        self.files_csv_sha256 = self._sha256(self.root / "files.csv")
        self._all_sar_target_verified = False

    def __len__(self) -> int:
        return len(self.rows)

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def _load(self, row: Mapping[str, str], name: str, *, verify_hash: bool) -> np.ndarray:
        sample_id = row["sample_id"]
        path = self.root / name / f"{sample_id}.npy"
        if verify_hash and self._sha256(path) != row[f"{name}_sha256"]:
            raise ValueError(f"C3-A {name} 数组哈希不匹配：{sample_id}")
        array = np.load(path, allow_pickle=False)
        expected_shape = (1, 128, 128) if name == "lr" else (1, 256, 256)
        if array.shape != expected_shape or array.dtype != np.float32 or not np.isfinite(array).all():
            raise ValueError(f"C3-A {name} 数组尺寸、类型或有限性不符合要求：{sample_id}")
        if name == "target" and (array.min() < 0 or array.max() > 1):
            raise ValueError(f"C3-A noisy pseudo-target 不在固定范围 [0,1]：{sample_id}")
        if name == "lr" and array.min() < 0:
            raise ValueError(f"C3-A SAR LR 含负值，与冻结退化不一致：{sample_id}")
        return array

    def verify_all(self) -> None:
        """预先核验全部 train/development SAR 与目标文件；不访问 optical 载荷。"""
        self._all_sar_target_verified = False
        for row in self.rows:
            self._load(row, "lr", verify_hash=True)
            self._load(row, "target", verify_hash=True)
        self._all_sar_target_verified = True

    def __getitem__(self, index: int) -> dict[str, object]:
        if not self._all_sar_target_verified:
            raise RuntimeError("先调用 verify_all 核验 SAR 与目标数组")
        row = self.rows[index]
        lr = self._load(row, "lr", verify_hash=False)
        target = self._load(row, "target", verify_hash=False)
        return {
            "name": row["sample_id"],
            "source_group_id": row["source_group_id"],
            "patch_id": row["patch_id"],
            "lr": torch.from_numpy(np.ascontiguousarray(lr)),
            "target": torch.from_numpy(np.ascontiguousarray(target)),
        }


class _TimeEmbedding(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        if dimension < 4 or dimension % 2:
            raise ValueError("时间嵌入维度必须为不小于 4 的偶数")
        self.dimension = dimension
        self.projection = nn.Sequential(
            nn.Linear(dimension, dimension * 2),
            nn.SiLU(),
            nn.Linear(dimension * 2, dimension),
        )

    def forward(self, time: Tensor) -> Tensor:
        if time.ndim != 1 or not torch.isfinite(time).all() or (time < 0).any() or (time > 1).any():
            raise ValueError("time 必须是一维且位于 [0,1] 的有限张量")
        half = self.dimension // 2
        frequencies = torch.exp(
            torch.arange(half, device=time.device, dtype=time.dtype)
            * (-math.log(10000.0) / max(half - 1, 1))
        )
        angles = time[:, None] * frequencies[None, :]
        encoded = torch.cat((angles.sin(), angles.cos()), dim=1)
        return self.projection(encoded)


class _TimeResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dimension: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(_groups(in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(_groups(out_channels), out_channels)
        self.time_affine = nn.Linear(time_dimension, out_channels * 2)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, 1)
        )

    def forward(self, value: Tensor, time_embedding: Tensor) -> Tensor:
        hidden = self.conv1(F.silu(self.norm1(value)))
        scale, shift = self.time_affine(time_embedding).chunk(2, dim=1)
        hidden = self.norm2(hidden)
        hidden = hidden * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]
        hidden = self.conv2(F.silu(hidden))
        return (self.skip(value) + hidden) * (2.0**-0.5)


class SarConditionalFlow(nn.Module):
    """接收 noisy flow state、双线性 SAR 条件和时间，预测速度场。"""

    def __init__(self, *, base_channels: int = 16, time_dimension: int = 64) -> None:
        super().__init__()
        if base_channels < 8 or base_channels % 8:
            raise ValueError("base_channels 必须是不小于 8 的 8 的倍数")
        self.base_channels = base_channels
        self.time_dimension = time_dimension
        self.time_embedding = _TimeEmbedding(time_dimension)
        c1, c2, c3 = base_channels, base_channels * 2, base_channels * 4
        self.input = nn.Conv2d(3, c1, 3, padding=1)
        self.encoder0a = _TimeResidualBlock(c1, c1, time_dimension)
        self.encoder0b = _TimeResidualBlock(c1, c1, time_dimension)
        self.down0 = nn.Conv2d(c1, c2, 4, stride=2, padding=1)
        self.encoder1 = _TimeResidualBlock(c2, c2, time_dimension)
        self.down1 = nn.Conv2d(c2, c3, 4, stride=2, padding=1)
        self.middle0 = _TimeResidualBlock(c3, c3, time_dimension)
        self.middle1 = _TimeResidualBlock(c3, c3, time_dimension)
        self.merge1 = nn.Conv2d(c3 + c2, c2, 3, padding=1)
        self.decoder1 = _TimeResidualBlock(c2, c2, time_dimension)
        self.merge0 = nn.Conv2d(c2 + c1, c1, 3, padding=1)
        self.decoder0 = _TimeResidualBlock(c1, c1, time_dimension)
        self.output = nn.Conv2d(c1, 1, 3, padding=1)

    def forward(self, state: Tensor, condition: Tensor, time: Tensor) -> Tensor:
        _check_image("state", state)
        _check_image("condition", condition)
        if state.shape != condition.shape or state.device != condition.device:
            raise ValueError("state 与 condition 的 shape/device 必须相同")
        if state.dtype != condition.dtype:
            raise ValueError("state 与 condition 的 dtype 必须相同")
        if state.shape[-2] % 4 or state.shape[-1] % 4:
            raise ValueError("输入高宽必须能被 4 整除")
        if time.shape != (state.shape[0],) or time.device != state.device:
            raise ValueError("time 必须与 batch 对齐并位于同一设备")
        time_embedding = self.time_embedding(time.to(dtype=state.dtype))
        time_map = time[:, None, None, None].to(dtype=state.dtype).expand(
            -1, 1, state.shape[-2], state.shape[-1]
        )
        first = self.input(torch.cat((state, condition, time_map), dim=1))
        first = self.encoder0b(self.encoder0a(first, time_embedding), time_embedding)
        second = self.encoder1(self.down0(first), time_embedding)
        middle = self.middle1(self.middle0(self.down1(second), time_embedding), time_embedding)
        up_second = F.interpolate(middle, size=second.shape[-2:], mode="nearest")
        up_second = self.decoder1(self.merge1(torch.cat((up_second, second), dim=1)), time_embedding)
        up_first = F.interpolate(up_second, size=first.shape[-2:], mode="nearest")
        up_first = self.decoder0(self.merge0(torch.cat((up_first, first), dim=1)), time_embedding)
        velocity = self.output(F.silu(up_first))
        if not torch.isfinite(velocity).all():
            raise FloatingPointError("SAR-only flow 速度场含 NaN 或 Inf")
        return velocity


def upsample_sar_condition(sar_lr: Tensor, *, size: tuple[int, int]) -> Tensor:
    _check_image("sar_lr", sar_lr)
    if len(size) != 2 or min(size) <= 0:
        raise ValueError("size 必须是两个正整数")
    return F.interpolate(sar_lr, size=size, mode="bilinear", align_corners=False)


def make_flow_training_pair(
    target: Tensor,
    condition: Tensor,
    time: Tensor,
    noise: Tensor,
    *,
    noise_sigma: float = 0.1,
) -> tuple[Tensor, Tensor]:
    """按预登记直线路径构造 x_t 与目标速度，不对中间状态裁剪。"""
    _check_image("target", target)
    _check_image("condition", condition)
    _check_image("noise", noise)
    if target.shape != condition.shape or target.shape != noise.shape:
        raise ValueError("target、condition、noise 的 shape 必须相同")
    if target.device != condition.device or target.device != noise.device:
        raise ValueError("target、condition、noise 必须位于同一设备")
    if target.dtype != condition.dtype or target.dtype != noise.dtype:
        raise ValueError("target、condition、noise 的 dtype 必须相同")
    if time.shape != (target.shape[0],) or time.device != target.device:
        raise ValueError("time 必须与 batch 对齐并位于同一设备")
    if (
        not torch.is_floating_point(time)
        or not torch.isfinite(time).all()
        or (time < 0).any()
        or (time > 1).any()
    ):
        raise ValueError("time 必须是 [0,1] 内的有限浮点张量")
    if not math.isfinite(noise_sigma) or noise_sigma <= 0:
        raise ValueError("noise_sigma 必须为有限正数")
    source = condition + noise_sigma * noise
    fraction = time[:, None, None, None].to(dtype=target.dtype)
    state = (1.0 - fraction) * source + fraction * target
    velocity = target - source
    if not torch.isfinite(state).all() or not torch.isfinite(velocity).all():
        raise FloatingPointError("流匹配样本含 NaN 或 Inf")
    return state, velocity


@torch.inference_mode()
def sample_heun(
    model: nn.Module,
    condition: Tensor,
    noise: Tensor,
    *,
    steps: int = 8,
    noise_sigma: float = 0.1,
) -> Tensor:
    """使用固定噪声执行 Heun 积分；只在调用端最终裁剪输出。"""
    _check_image("condition", condition)
    _check_image("noise", noise)
    if condition.shape != noise.shape or condition.device != noise.device:
        raise ValueError("condition 与 noise 的 shape/device 必须相同")
    if condition.dtype != noise.dtype:
        raise ValueError("condition 与 noise 的 dtype 必须相同")
    if not isinstance(steps, int) or isinstance(steps, bool) or steps < 1:
        raise ValueError("steps 必须为正整数")
    if not math.isfinite(noise_sigma) or noise_sigma <= 0:
        raise ValueError("noise_sigma 必须为有限正数")
    state = condition + noise_sigma * noise
    batch = condition.shape[0]
    step_size = 1.0 / steps
    for index in range(steps):
        time = condition.new_full((batch,), index * step_size)
        next_time = condition.new_full((batch,), (index + 1) * step_size)
        velocity = model(state, condition, time)
        _check_velocity(velocity, state)
        euler = state + step_size * velocity
        next_velocity = model(euler, condition, next_time)
        _check_velocity(next_velocity, euler)
        state = state + 0.5 * step_size * (velocity + next_velocity)
    if not torch.isfinite(state).all():
        raise FloatingPointError("Heun 积分结果含 NaN 或 Inf")
    return state


def _check_velocity(velocity: Tensor, state: Tensor) -> None:
    """拒绝会触发隐式广播或类型提升的速度场输出。"""
    if not isinstance(velocity, Tensor):
        raise ValueError("模型速度场必须返回 Tensor")
    if velocity.shape != state.shape:
        raise ValueError(
            f"模型速度场 shape 必须与积分状态完全相同：{tuple(velocity.shape)} != {tuple(state.shape)}"
        )
    if velocity.device != state.device:
        raise ValueError("模型速度场 device 必须与积分状态相同")
    if velocity.dtype != state.dtype:
        raise ValueError("模型速度场 dtype 必须与积分状态相同")
    if not torch.is_floating_point(velocity) or not torch.isfinite(velocity).all():
        raise FloatingPointError("模型速度场必须为有限浮点张量")
