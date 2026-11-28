"""Task B 的确定性均值校正器与中心残差条件扩散网络。"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def linear_beta_schedule(
    *,
    steps: int = 1000,
    beta_start: float = 1e-4,
    beta_end: float = 0.02,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    if steps < 2 or not math.isfinite(beta_start) or not math.isfinite(beta_end):
        raise ValueError("扩散步数与 beta 必须有效")
    if not 0.0 < beta_start < beta_end < 1.0:
        raise ValueError("beta 必须满足 0 < beta_start < beta_end < 1")
    return torch.linspace(beta_start, beta_end, steps, dtype=torch.float32, device=device)


class _MeanResidualBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(width, width, 3, padding=1)
        self.conv2 = nn.Conv2d(width, width, 3, padding=1)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = self.conv2(self.activation(self.conv1(value)))
        return value + residual * 0.1


class ResidualMeanCorrector(nn.Module):
    """学习固定 SwinIR 点估计中可预测的均值残差。"""

    def __init__(self, *, width: int = 32, residual_blocks: int = 4) -> None:
        super().__init__()
        if width <= 0 or residual_blocks <= 0:
            raise ValueError("校正器通道数和残差块数量必须为正")
        self.input = nn.Conv2d(2, width, 3, padding=1)
        self.blocks = nn.Sequential(*(_MeanResidualBlock(width) for _ in range(residual_blocks)))
        self.output = nn.Conv2d(width, 1, 3, padding=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, low_resolution: torch.Tensor, base_estimate: torch.Tensor) -> torch.Tensor:
        if (
            low_resolution.ndim != 4
            or base_estimate.ndim != 4
            or low_resolution.shape[1] != 1
            or base_estimate.shape[1] != 1
        ):
            raise ValueError("均值校正器输入必须是单通道 BCHW")
        if low_resolution.shape[0] != base_estimate.shape[0]:
            raise ValueError("均值校正器输入 batch 不一致")
        expected = (low_resolution.shape[-2] * 2, low_resolution.shape[-1] * 2)
        if base_estimate.shape[-2:] != expected:
            raise ValueError("基线输出尺寸必须是 LR 的 ×2")
        if low_resolution.device != base_estimate.device:
            raise ValueError("均值校正器输入设备不一致")
        upsampled = F.interpolate(
            low_resolution,
            size=base_estimate.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        features = self.blocks(self.input(torch.cat((upsampled, base_estimate), dim=1)))
        return torch.clamp(base_estimate + self.output(features), 0.0, 1.0)


class _TimeResidualBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, time_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(8, input_channels)
        self.conv1 = nn.Conv2d(input_channels, output_channels, 3, padding=1)
        self.time_projection = nn.Linear(time_dim, output_channels)
        self.norm2 = nn.GroupNorm(8, output_channels)
        self.conv2 = nn.Conv2d(output_channels, output_channels, 3, padding=1)
        self.skip = (
            nn.Identity()
            if input_channels == output_channels
            else nn.Conv2d(input_channels, output_channels, 1)
        )

    def forward(self, value: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        hidden = self.conv1(F.silu(self.norm1(value)))
        hidden = hidden + self.time_projection(F.silu(time_embedding))[:, :, None, None]
        hidden = self.conv2(F.silu(self.norm2(hidden)))
        return (self.skip(value) + hidden) * (2**-0.5)


class ConditionalResidualUNet(nn.Module):
    """以均值估计和 SAR LR 为条件的 residual epsilon-prediction U-Net。"""

    def __init__(self, *, base_channels: int = 32, time_dim: int = 128) -> None:
        super().__init__()
        if base_channels <= 0 or base_channels % 8 or time_dim < 4 or time_dim % 2:
            raise ValueError("U-Net 通道数须为 8 的倍数，time_dim 须为正偶数")
        self.base_channels = base_channels
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim),
        )
        self.input = nn.Conv2d(3, base_channels, 3, padding=1)
        self.down0 = nn.ModuleList(
            [
                _TimeResidualBlock(base_channels, base_channels, time_dim),
                _TimeResidualBlock(base_channels, base_channels, time_dim),
            ]
        )
        self.downsample0 = nn.Conv2d(base_channels, base_channels * 2, 3, stride=2, padding=1)
        self.down1 = nn.ModuleList(
            [
                _TimeResidualBlock(base_channels * 2, base_channels * 2, time_dim),
                _TimeResidualBlock(base_channels * 2, base_channels * 2, time_dim),
            ]
        )
        self.downsample1 = nn.Conv2d(base_channels * 2, base_channels * 4, 3, stride=2, padding=1)
        self.down2 = nn.ModuleList(
            [
                _TimeResidualBlock(base_channels * 4, base_channels * 4, time_dim),
                _TimeResidualBlock(base_channels * 4, base_channels * 4, time_dim),
            ]
        )
        self.middle = nn.ModuleList(
            [
                _TimeResidualBlock(base_channels * 4, base_channels * 4, time_dim),
                _TimeResidualBlock(base_channels * 4, base_channels * 4, time_dim),
            ]
        )
        self.upsample1 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.up1 = nn.ModuleList(
            [
                _TimeResidualBlock(base_channels * 4, base_channels * 2, time_dim),
                _TimeResidualBlock(base_channels * 2, base_channels * 2, time_dim),
            ]
        )
        self.upsample0 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)
        self.up0 = nn.ModuleList(
            [
                _TimeResidualBlock(base_channels * 2, base_channels, time_dim),
                _TimeResidualBlock(base_channels, base_channels, time_dim),
            ]
        )
        self.output_norm = nn.GroupNorm(8, base_channels)
        self.output = nn.Conv2d(base_channels, 1, 3, padding=1)

    def _time_embedding(self, timesteps: torch.Tensor) -> torch.Tensor:
        half = self.time_dim // 2
        frequencies = torch.exp(
            -math.log(10_000.0)
            * torch.arange(half, device=timesteps.device, dtype=torch.float32)
            / max(half - 1, 1)
        )
        angles = timesteps.float()[:, None] * frequencies[None, :]
        return self.time_mlp(torch.cat((torch.sin(angles), torch.cos(angles)), dim=1))

    @staticmethod
    def _run_blocks(
        blocks: nn.ModuleList, value: torch.Tensor, time_embedding: torch.Tensor
    ) -> torch.Tensor:
        for block in blocks:
            value = block(value, time_embedding)
        return value

    def forward(
        self,
        noisy_residual: torch.Tensor,
        mean_estimate: torch.Tensor,
        low_resolution: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        tensors = (noisy_residual, mean_estimate, low_resolution)
        if any(value.ndim != 4 or value.shape[1] != 1 for value in tensors):
            raise ValueError("扩散条件必须为单通道 BCHW 张量")
        if len({value.shape[0] for value in tensors}) != 1:
            raise ValueError("扩散条件 batch 不一致")
        if mean_estimate.shape[-2:] != noisy_residual.shape[-2:]:
            raise ValueError("中心残差与均值估计空间尺寸不一致")
        if low_resolution.shape[-2:] != (
            noisy_residual.shape[-2] // 2,
            noisy_residual.shape[-1] // 2,
        ):
            raise ValueError("SAR LR 条件必须是 HR 的二分之一尺寸")
        if noisy_residual.shape[-2] % 4 or noisy_residual.shape[-1] % 4:
            raise ValueError("扩散 U-Net 的 HR 高宽必须能被 4 整除")
        if timesteps.ndim != 1 or timesteps.shape[0] != noisy_residual.shape[0]:
            raise ValueError("扩散 timestep 必须为 [B]")
        if any(value.device != noisy_residual.device for value in tensors) or timesteps.device != noisy_residual.device:
            raise ValueError("扩散条件和 timestep 必须位于同一设备")

        upsampled_lr = F.interpolate(
            low_resolution,
            size=noisy_residual.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        time_embedding = self._time_embedding(timesteps)
        level0 = self._run_blocks(
            self.down0,
            self.input(torch.cat((noisy_residual, mean_estimate, upsampled_lr), dim=1)),
            time_embedding,
        )
        level1 = self._run_blocks(self.down1, self.downsample0(level0), time_embedding)
        level2 = self._run_blocks(self.down2, self.downsample1(level1), time_embedding)
        middle = self._run_blocks(self.middle, level2, time_embedding)
        up1 = self.upsample1(middle)
        up1 = self._run_blocks(self.up1, torch.cat((up1, level1), dim=1), time_embedding)
        up0 = self.upsample0(up1)
        up0 = self._run_blocks(self.up0, torch.cat((up0, level0), dim=1), time_embedding)
        return self.output(F.silu(self.output_norm(up0)))
