"""Phase E 表示域合成 Task B 的固定数据构建函数。"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from experiments.phase_e_taskb_operator import (
    downsample_noisy_sar_x2,
    gamma_speckle_like_representation,
)


def read_reference_tiff(path: str | Path) -> np.ndarray:
    """无拉伸地读取单波段参考 TIFF；来源范围核验由调用者完成。"""
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"参考 TIFF 无法解码：{path}")
    if image.ndim != 2:
        raise ValueError(f"参考 TIFF 必须为单波段灰度图：{path}")
    return image


def center_crop_reference(
    reference: np.ndarray, *, size: int = 512
) -> tuple[np.ndarray, dict[str, int]]:
    """按冻结窗口从参考栅格中心裁取固定正方形，不进行缩放。"""
    image = np.asarray(reference)
    if image.ndim != 2 or not np.issubdtype(image.dtype, np.number):
        raise ValueError("中心裁取输入必须为单通道数值影像")
    if size <= 0 or image.shape[0] < size or image.shape[1] < size:
        raise ValueError("参考影像尺寸小于固定中心裁取窗口")
    if not np.isfinite(image).all():
        raise ValueError("中心裁取输入含 NaN 或 Inf")
    row_start = (image.shape[0] - size) // 2
    col_start = (image.shape[1] - size) // 2
    window = {
        "row_start": row_start,
        "row_stop": row_start + size,
        "col_start": col_start,
        "col_stop": col_start + size,
    }
    return (
        np.ascontiguousarray(image[row_start : row_start + size, col_start : col_start + size]),
        window,
    )


def build_fixed_taskb_pair(
    reference_8bit: np.ndarray,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """从 [0,255] 参考确定性生成一份 Gamma-like ×2 表示域输入和目标。"""
    reference = np.asarray(reference_8bit)
    if reference.ndim != 2 or min(reference.shape, default=0) <= 0:
        raise ValueError("参考影像必须为非空二维单通道数组")
    if reference.shape[0] % 2 or reference.shape[1] % 2:
        raise ValueError("参考影像高宽必须为偶数")
    if not np.issubdtype(reference.dtype, np.number):
        raise ValueError("参考影像必须为数值数组")
    if not np.isfinite(reference).all():
        raise ValueError("参考影像含 NaN 或 Inf")
    if np.any(reference < 0) or np.any(reference > 255):
        raise ValueError("参考影像超出冻结的 [0,255] 范围")

    target = np.ascontiguousarray(reference.astype(np.float32) / 255.0)
    target_tensor = torch.from_numpy(target[None, None])
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    corrupted = gamma_speckle_like_representation(
        target_tensor, looks=1.0, generator=generator
    )
    low_resolution = downsample_noisy_sar_x2(corrupted)
    return (
        np.ascontiguousarray(low_resolution[0].numpy(), dtype=np.float32),
        np.ascontiguousarray(target[None], dtype=np.float32),
    )
