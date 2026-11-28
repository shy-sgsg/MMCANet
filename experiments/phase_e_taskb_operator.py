"""Task B 中固定的 SAR 表示域 ×2 空间降采样算子。"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def downsample_noisy_sar_x2(image: torch.Tensor) -> torch.Tensor:
    """对单通道 SAR 表示应用固定 2×2 面积平均，得到 ×2 低分辨率输入。"""
    if image.ndim != 4 or image.shape[1] != 1:
        raise ValueError("SAR 输入必须为单通道 [B,1,H,W]")
    if not image.is_floating_point():
        raise ValueError("SAR 输入必须为浮点张量")
    if image.shape[-2] % 2 or image.shape[-1] % 2:
        raise ValueError("SAR 输入高宽必须为偶数，供固定 ×2 面积平均使用")
    if not torch.isfinite(image).all():
        raise ValueError("SAR 输入含 NaN 或 Inf")
    return F.avg_pool2d(image, kernel_size=2, stride=2)


def gamma_speckle_like_representation(
    reference: torch.Tensor,
    *,
    looks: float,
    generator: torch.Generator,
) -> torch.Tensor:
    """在非负归一化表示域乘单位均值 Gamma 噪声；不代表真实 SAR 强度噪声。"""
    if reference.ndim != 4 or reference.shape[1] != 1:
        raise ValueError("参考输入必须为单通道 [B,1,H,W]")
    if not reference.is_floating_point():
        raise ValueError("参考输入必须为浮点张量")
    if not math.isfinite(looks) or looks <= 0:
        raise ValueError("looks 必须为有限正数")
    if not torch.isfinite(reference).all() or torch.any(reference < 0):
        raise ValueError("参考输入必须有限且非负")
    if generator.device != reference.device:
        raise ValueError("随机数生成器与参考输入必须位于同一设备")
    concentration = torch.full_like(reference, float(looks))
    multiplier = torch._standard_gamma(concentration, generator=generator) / looks
    return reference * multiplier
