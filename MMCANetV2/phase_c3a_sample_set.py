"""C3-A 多次生成的点估计与有限样本分布评分。"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def _validated_draws_and_target(
    draws: np.ndarray,
    target: np.ndarray,
    *,
    minimum_draws: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    sample_set = np.asarray(draws, dtype=np.float64)
    observation = np.asarray(target, dtype=np.float64)
    if sample_set.ndim != 4 or observation.ndim != 3:
        raise ValueError("draws 和 target 形状必须分别为 [K,C,H,W] 与 [C,H,W]")
    if sample_set.shape[0] < minimum_draws:
        raise ValueError(f"样本集合评分至少两次 draw，当前为 {sample_set.shape[0]} 次")
    if sample_set.shape[1:] != observation.shape:
        raise ValueError("draws 与 target 的通道或空间形状不一致")
    if not np.isfinite(sample_set).all() or not np.isfinite(observation).all():
        raise ValueError("draws 或 target 含非有限值")
    return sample_set, observation


def pixelwise_mean(draws: np.ndarray) -> np.ndarray:
    """按 draw 顺序以 float64 累加并返回逐像素均值点估计。"""
    sample_set = np.asarray(draws, dtype=np.float64)
    if sample_set.ndim != 4:
        raise ValueError("draws 形状必须为 [K,C,H,W]")
    if sample_set.shape[0] < 2:
        raise ValueError(f"像素均值点估计至少两次 draw，当前为 {sample_set.shape[0]} 次")
    if not np.isfinite(sample_set).all():
        raise ValueError("draws 含非有限值")
    return sample_set.mean(axis=0, dtype=np.float64)


def mean_pixelwise_sample_std(draws: np.ndarray) -> float:
    """计算各像素跨 draw 的样本标准差，再对像素等权平均。"""
    sample_set = np.asarray(draws, dtype=np.float64)
    if sample_set.ndim != 4:
        raise ValueError("draws 形状必须为 [K,C,H,W]")
    if sample_set.shape[0] < 2:
        raise ValueError(f"像素离散度至少需要两次 draw，当前为 {sample_set.shape[0]} 次")
    if not np.isfinite(sample_set).all():
        raise ValueError("draws 含非有限值")
    pixelwise_std = np.std(sample_set, axis=0, ddof=1, dtype=np.float64)
    summary = float(np.mean(pixelwise_std, dtype=np.float64))
    if not np.isfinite(summary):
        raise FloatingPointError("像素样本标准差不是有限值")
    return summary


def source_group_balanced_mean(
    *,
    sample_ids: Sequence[str],
    source_group_ids: Sequence[str],
    scores: Sequence[float],
) -> dict[str, object]:
    """先算每个来源组的样本均值，再对来源组等权平均。"""
    identities = tuple(sample_ids)
    groups = tuple(source_group_ids)
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or not identities or len(identities) != len(groups) or len(groups) != values.size:
        raise ValueError("样本身份、来源组和分数必须是等长的非空序列")
    if any(not isinstance(value, str) or not value.strip() for value in identities):
        raise ValueError("样本身份为空或类型无效")
    if len(set(identities)) != len(identities):
        raise ValueError("样本身份为空或重复")
    if any(not isinstance(value, str) or not value.strip() for value in groups):
        raise ValueError("来源组身份为空或类型无效")
    if not np.isfinite(values).all():
        raise ValueError("来源组评分含非有限值")

    grouped: dict[str, list[float]] = {}
    for group, value in zip(groups, values):
        grouped.setdefault(group, []).append(float(value))
    group_means = {
        group: float(np.mean(grouped[group], dtype=np.float64))
        for group in sorted(grouped)
    }
    balanced_mean = float(np.mean(tuple(group_means.values()), dtype=np.float64))
    return {
        "来源组等权均值": balanced_mean,
        "样本数": int(values.size),
        "来源组数": len(group_means),
        "各来源组均值": group_means,
    }


def _normalized_euclidean_distance(left: np.ndarray, right: np.ndarray) -> float:
    difference = (left - right).reshape(-1)
    return float(np.sqrt(np.mean(np.square(difference), dtype=np.float64)))


def energy_score(draws: np.ndarray, target: np.ndarray) -> float:
    """计算有限经验集合的归一化欧氏 energy score，分数越低越好。

    每幅多通道图像视为一个向量；欧氏距离除以像素维数平方根。
    第二项按完整 K×K 经验分布配对计算，包含零对角项。
    """
    sample_set, observation = _validated_draws_and_target(draws, target)
    observation_distance = np.mean(
        [_normalized_euclidean_distance(draw, observation) for draw in sample_set],
        dtype=np.float64,
    )
    pairwise_distance = np.mean(
        [
            _normalized_euclidean_distance(left, right)
            for left in sample_set
            for right in sample_set
        ],
        dtype=np.float64,
    )
    score = float(observation_distance - 0.5 * pairwise_distance)
    if not np.isfinite(score):
        raise FloatingPointError("energy score 不是有限值")
    return score


def fair_energy_score_iid(draws: np.ndarray, target: np.ndarray) -> float:
    """估计 IID draw 所代表的底层分布 energy score；分数越低越好。

    与 ``energy_score`` 的有限经验分布解释不同，本函数以 K(K-1) 个非对角
    有序成员对作 U 统计量。draw 必须视为同一条件分布的独立同分布样本；
    该无偏估计可能为负数，不能解释为有限经验分布本身的 proper score。
    """
    sample_set, observation = _validated_draws_and_target(draws, target)
    size = sample_set.shape[0]
    observation_distance = np.mean(
        [_normalized_euclidean_distance(draw, observation) for draw in sample_set],
        dtype=np.float64,
    )
    distinct_pair_distances = [
        _normalized_euclidean_distance(sample_set[left], sample_set[right])
        for left in range(size)
        for right in range(size)
        if left != right
    ]
    pairwise_distance = float(np.mean(distinct_pair_distances, dtype=np.float64))
    score = float(observation_distance - 0.5 * pairwise_distance)
    if not np.isfinite(score):
        raise FloatingPointError("fair energy score 不是有限值")
    return score


def _validate_offsets(offsets_yx: Sequence[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    try:
        offsets = tuple(tuple(offset) for offset in offsets_yx)
    except TypeError as exc:
        raise ValueError("variogram 偏移必须是非空的整数二元组集合") from exc
    if not offsets:
        raise ValueError("variogram 偏移集合不能为空")
    normalized: list[tuple[int, int]] = []
    for offset in offsets:
        if (
            len(offset) != 2
            or any(not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_)) for value in offset)
        ):
            raise ValueError("variogram 偏移必须是非空的整数二元组集合")
        dy, dx = int(offset[0]), int(offset[1])
        if dy == 0 and dx == 0:
            raise ValueError("variogram 偏移不能为零向量")
        normalized.append((dy, dx))
    if len(set(normalized)) != len(normalized):
        raise ValueError("variogram 偏移不能重复")
    return tuple(normalized)


def _matching_slices(height: int, width: int, dy: int, dx: int) -> tuple[slice, slice, slice, slice] | None:
    y_start = max(0, -dy)
    y_stop = min(height, height - dy)
    x_start = max(0, -dx)
    x_stop = min(width, width - dx)
    if y_start >= y_stop or x_start >= x_stop:
        return None
    return (
        slice(y_start, y_stop),
        slice(x_start, x_stop),
        slice(y_start + dy, y_stop + dy),
        slice(x_start + dx, x_stop + dx),
    )


def variogram_score(
    draws: np.ndarray,
    target: np.ndarray,
    *,
    offsets_yx: Sequence[tuple[int, int]] = (
        (0, 1),
        (1, 0),
        (1, 1),
        (1, -1),
        (0, 2),
        (2, 0),
    ),
    power: float = 0.5,
) -> float:
    """计算等权局部空间 variogram score，分数越低越好。"""
    sample_set, observation = _validated_draws_and_target(draws, target)
    offsets = _validate_offsets(offsets_yx)
    if isinstance(power, (bool, np.bool_)) or not np.isfinite(power) or not 0.0 < float(power) < 2.0:
        raise ValueError("variogram 幂指数必须位于 (0,2)")
    _, channels, height, width = sample_set.shape
    del channels
    offset_scores: list[float] = []
    for dy, dx in offsets:
        slices = _matching_slices(height, width, dy, dx)
        if slices is None:
            raise ValueError(f"偏移 {(dy, dx)} 没有有效像素对")
        y0, x0, y1, x1 = slices
        observed_increment = np.abs(observation[..., y0, x0] - observation[..., y1, x1]) ** float(power)
        forecast_increment = np.mean(
            np.abs(sample_set[..., y0, x0] - sample_set[..., y1, x1]) ** float(power),
            axis=0,
            dtype=np.float64,
        )
        offset_scores.append(float(np.mean(np.square(observed_increment - forecast_increment), dtype=np.float64)))
    score = float(np.mean(offset_scores, dtype=np.float64))
    if not np.isfinite(score):
        raise FloatingPointError("variogram score 不是有限值")
    return score


def fair_variogram_score_iid(
    draws: np.ndarray,
    target: np.ndarray,
    *,
    offsets_yx: Sequence[tuple[int, int]] = (
        (0, 1),
        (1, 0),
        (1, 1),
        (1, -1),
        (0, 2),
        (2, 0),
    ),
    power: float = 0.5,
) -> float:
    """估计 IID draw 所代表的底层 variogram score；分数越低越好。

    每个空间邻接对上，使用成员平方误差均值减去半数非对角成员差平方
    均值。该 U 统计量在 IID draw 假设下无偏，有限样本估计可以为负；
    它与 ``variogram_score`` 的经验分布 plug-in 解释不同。
    """
    sample_set, observation = _validated_draws_and_target(draws, target)
    offsets = _validate_offsets(offsets_yx)
    if isinstance(power, (bool, np.bool_)) or not np.isfinite(power) or not 0.0 < float(power) < 2.0:
        raise ValueError("variogram 幂指数必须位于 (0,2)")
    size = sample_set.shape[0]
    _, _, height, width = sample_set.shape
    offset_scores: list[float] = []
    for dy, dx in offsets:
        slices = _matching_slices(height, width, dy, dx)
        if slices is None:
            raise ValueError(f"偏移 {(dy, dx)} 没有有效像素对")
        y0, x0, y1, x1 = slices
        observed_increment = np.abs(observation[..., y0, x0] - observation[..., y1, x1]) ** float(power)
        forecast_increments = np.abs(
            sample_set[..., y0, x0] - sample_set[..., y1, x1]
        ) ** float(power)
        member_squared_error = np.square(forecast_increments - observed_increment[None, ...]).mean(
            axis=0, dtype=np.float64
        )
        member_pair_differences = forecast_increments[:, None, ...] - forecast_increments[None, ...]
        fair_pair_correction = np.square(member_pair_differences).sum(axis=(0, 1), dtype=np.float64) / (
            2.0 * size * (size - 1)
        )
        offset_scores.append(
            float(np.mean(member_squared_error - fair_pair_correction, dtype=np.float64))
        )
    score = float(np.mean(offset_scores, dtype=np.float64))
    if not np.isfinite(score):
        raise FloatingPointError("fair variogram score 不是有限值")
    return score


__all__ = [
    "energy_score",
    "fair_energy_score_iid",
    "fair_variogram_score_iid",
    "mean_pixelwise_sample_std",
    "pixelwise_mean",
    "source_group_balanced_mean",
    "variogram_score",
]
