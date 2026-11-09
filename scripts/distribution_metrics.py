"""面向归一化图像域合成实验的多样本评分量。"""

from __future__ import annotations

import numpy as np


def _validate_inputs(
    samples: np.ndarray, observation: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    raw_samples = np.asarray(samples)
    raw_observation = np.asarray(observation)

    if raw_samples.ndim != 3:
        raise ValueError("samples 必须为三维数组 [K,H,W]")
    if raw_observation.ndim != 2:
        raise ValueError("observation 必须为二维数组 [H,W]")
    if raw_samples.shape[0] == 0:
        raise ValueError("至少需要一个预测样本")
    if raw_samples.shape[1:] != raw_observation.shape:
        raise ValueError("samples 与 observation 的空间尺寸必须一致")
    if raw_samples.shape[1] == 0 or raw_samples.shape[2] == 0:
        raise ValueError("图像的高和宽必须大于零")
    if (
        raw_samples.dtype.kind not in "iuf"
        or raw_observation.dtype.kind not in "iuf"
    ):
        raise ValueError("输入必须为实数数组")

    sample_values = raw_samples.astype(np.float64, copy=False)
    observation_values = raw_observation.astype(np.float64, copy=False)
    if not np.isfinite(sample_values).all() or not np.isfinite(observation_values).all():
        raise ValueError("输入不得包含 NaN 或 Inf")
    if (
        np.any(sample_values < 0.0)
        or np.any(sample_values > 1.0)
        or np.any(observation_values < 0.0)
        or np.any(observation_values > 1.0)
    ):
        raise ValueError("数值范围必须位于 [0, 1]；函数不会自动裁剪")

    return sample_values, observation_values


def pixelwise_crps(samples: np.ndarray, observation: np.ndarray) -> float:
    """计算逐像素经验 CRPS，再对图像像素取平均。"""
    sample_values, observation_values = _validate_inputs(samples, observation)
    draw_count = sample_values.shape[0]

    cross_term = np.mean(
        np.abs(sample_values - observation_values[None, :, :]), axis=0
    )
    pairwise_sum = np.zeros_like(observation_values)
    for first in range(draw_count):
        for second in range(first + 1, draw_count):
            pairwise_sum += np.abs(sample_values[first] - sample_values[second])

    score = cross_term - pairwise_sum / (draw_count * draw_count)
    return float(np.mean(score))


def normalized_image_energy_score(
    samples: np.ndarray, observation: np.ndarray
) -> float:
    """计算按像素数平方根归一化的整图经验能量分数。"""
    sample_values, observation_values = _validate_inputs(samples, observation)
    flattened = sample_values.reshape(sample_values.shape[0], -1)
    target = observation_values.reshape(-1)
    draw_count = flattened.shape[0]
    normalizer = np.sqrt(target.size)

    cross_sum = 0.0
    for draw in flattened:
        cross_sum += float(np.linalg.norm(draw - target))

    pairwise_sum = 0.0
    for first in range(draw_count):
        for second in range(first + 1, draw_count):
            pairwise_sum += float(np.linalg.norm(flattened[first] - flattened[second]))

    return float(
        cross_sum / (draw_count * normalizer)
        - pairwise_sum / (draw_count * draw_count * normalizer)
    )


def adjacent_variogram_score(
    samples: np.ndarray, observation: np.ndarray
) -> float:
    """比较预测集合与观测在水平、垂直相邻像素间的绝对差。"""
    sample_values, observation_values = _validate_inputs(samples, observation)
    height, width = observation_values.shape
    if height * (width - 1) + (height - 1) * width == 0:
        raise ValueError("邻接变差分数至少需要两个像素")

    forecast_differences: list[np.ndarray] = []
    target_differences: list[np.ndarray] = []
    if width > 1:
        forecast_differences.append(
            np.mean(
                np.abs(sample_values[:, :, 1:] - sample_values[:, :, :-1]), axis=0
            ).ravel()
        )
        target_differences.append(
            np.abs(observation_values[:, 1:] - observation_values[:, :-1]).ravel()
        )
    if height > 1:
        forecast_differences.append(
            np.mean(
                np.abs(sample_values[:, 1:, :] - sample_values[:, :-1, :]), axis=0
            ).ravel()
        )
        target_differences.append(
            np.abs(observation_values[1:, :] - observation_values[:-1, :]).ravel()
        )

    forecast = np.concatenate(forecast_differences)
    target = np.concatenate(target_differences)
    return float(np.mean((forecast - target) ** 2))
