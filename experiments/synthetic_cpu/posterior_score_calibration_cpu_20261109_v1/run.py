#!/usr/bin/env python3
"""运行已知稀疏混合后验下的 CRPS 与覆盖率 CPU 诊断。"""

from __future__ import annotations

import csv
import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np


OUTPUT_DIR = Path(__file__).resolve().parent
MASTER_SEED = 20261109
CASE_COUNT = 1024
BOOTSTRAP_COUNT = 2000
DIMENSION = 5
OBSERVATION_DIMENSION = 3
SPIKE_STD = 0.05
SLAB_STD = 1.0
INCLUSION_PROBABILITY = 0.18
NOISE_STD = 0.35
PROBABILITIES = (0.05, 0.25, 0.75, 0.95)
CONDITIONS = (("精确后验", 1.0), ("欠分散", 0.5), ("过分散", 1.5), ("均值塌缩", 0.0))


def normal_cdf(values: object) -> np.ndarray:
    """用 Abramowitz–Stegun 近似计算标准正态分布函数。"""
    x = np.asarray(values, dtype=np.float64)
    z = np.abs(x)
    t = 1.0 / (1.0 + 0.2316419 * z)
    polynomial = t * (
        0.319381530
        + t
        * (
            -0.356563782
            + t
            * (1.781477937 + t * (-1.821255978 + t * 1.330274429))
        )
    )
    tail = 0.3989422804014327 * np.exp(-0.5 * z * z) * polynomial
    return np.clip(np.where(x >= 0.0, 1.0 - tail, tail), 0.0, 1.0)


def absolute_normal_expectation(mean: object, std: object) -> np.ndarray:
    """计算正态随机变量绝对值的期望，允许标准差为零。"""
    mu, sigma = np.broadcast_arrays(
        np.asarray(mean, dtype=np.float64), np.asarray(std, dtype=np.float64)
    )
    if np.any(sigma < 0.0):
        raise ValueError("标准差不能为负")
    result = np.abs(mu)
    nondegenerate = sigma > 0.0
    if np.any(nondegenerate):
        z = mu[nondegenerate] / sigma[nondegenerate]
        density = 0.3989422804014327 * np.exp(-0.5 * z * z)
        result[nondegenerate] = sigma[nondegenerate] * (
            2.0 * density + z * (2.0 * normal_cdf(z) - 1.0)
        )
    return result


def mixture_crps(
    weights: object, means: object, variances: object, target: object
) -> float:
    """精确计算高斯混合的逐维平均 CRPS。"""
    w = np.asarray(weights, dtype=np.float64)
    mu = np.asarray(means, dtype=np.float64)
    var = np.asarray(variances, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    if mu.ndim != 2 or var.shape != mu.shape or y.shape != (mu.shape[1],):
        raise ValueError("高斯混合参数维度不一致")
    if w.shape != (mu.shape[0],) or np.any(w < 0.0) or not np.isclose(w.sum(), 1.0):
        raise ValueError("混合权重必须非负且和为 1")
    if np.any(var < 0.0) or not all(np.all(np.isfinite(v)) for v in (w, mu, var, y)):
        raise ValueError("高斯混合参数必须有限且方差非负")

    component_std = np.sqrt(var)
    first = np.sum(
        w[:, None] * absolute_normal_expectation(mu - y[None, :], component_std),
        axis=0,
    )
    pair_mean = mu[:, None, :] - mu[None, :, :]
    pair_std = np.sqrt(var[:, None, :] + var[None, :, :])
    pair_abs = absolute_normal_expectation(pair_mean, pair_std)
    pair_weight = w[:, None] * w[None, :]
    second = np.sum(pair_weight[:, :, None] * pair_abs, axis=(0, 1))
    return float(np.mean(first - 0.5 * second))


def mixture_quantile(
    weights: object,
    means: object,
    stds: object,
    probabilities: object,
) -> np.ndarray:
    """逐维求高斯混合分位数；返回 [分位数, 维度] 数组。"""
    w = np.asarray(weights, dtype=np.float64)
    mu = np.asarray(means, dtype=np.float64)
    sigma = np.asarray(stds, dtype=np.float64)
    probs = np.asarray(probabilities, dtype=np.float64)
    if mu.ndim != 2 or sigma.shape != mu.shape or w.shape != (mu.shape[0],):
        raise ValueError("高斯混合参数维度不一致")
    if np.any(w < 0.0) or not np.isclose(w.sum(), 1.0):
        raise ValueError("混合权重必须非负且和为 1")
    if np.any(sigma < 0.0) or np.any((probs <= 0.0) | (probs >= 1.0)):
        raise ValueError("标准差或分位概率无效")
    if not all(np.all(np.isfinite(v)) for v in (w, mu, sigma, probs)):
        raise ValueError("高斯混合参数必须有限")

    if np.all(sigma == 0.0) and np.allclose(mu, mu[0:1]):
        return np.repeat(mu[0:1, :], probs.size, axis=0)
    if np.any(sigma == 0.0):
        raise ValueError("分位数计算不支持连续与点质量混合")

    lower_bound = np.min(mu - 8.0 * sigma, axis=0)
    upper_bound = np.max(mu + 8.0 * sigma, axis=0)
    quantiles = []
    for probability in probs:
        lo = lower_bound.copy()
        hi = upper_bound.copy()
        for _ in range(64):
            mid = 0.5 * (lo + hi)
            cdf = np.sum(
                w[:, None] * normal_cdf((mid[None, :] - mu) / sigma), axis=0
            )
            below = cdf < probability
            lo = np.where(below, mid, lo)
            hi = np.where(below, hi, mid)
        quantiles.append(0.5 * (lo + hi))
    return np.stack(quantiles, axis=0)


def build_mixture_model(
    operator: object,
    spike_std: float,
    slab_std: float,
    inclusion_probability: float,
    noise_std: float,
) -> dict[str, np.ndarray]:
    """预计算稀疏高斯混合先验下的线性高斯后验成分。"""
    a = np.asarray(operator, dtype=np.float64)
    if a.ndim != 2 or min(a.shape) < 1:
        raise ValueError("观测算子必须是二维矩阵")
    if spike_std <= 0.0 or slab_std <= 0.0 or noise_std <= 0.0:
        raise ValueError("标准差必须为正")
    if not 0.0 < inclusion_probability < 1.0:
        raise ValueError("稀疏先验概率必须在 0 与 1 之间")

    observation_dimension, dimension = a.shape
    component_count = 1 << dimension
    prior_log_weights = np.empty(component_count, dtype=np.float64)
    observation_inverses = np.empty(
        (component_count, observation_dimension, observation_dimension), dtype=np.float64
    )
    observation_log_determinants = np.empty(component_count, dtype=np.float64)
    gains = np.empty((component_count, dimension, observation_dimension), dtype=np.float64)
    posterior_covariances = np.empty((component_count, dimension, dimension), dtype=np.float64)
    identity = np.eye(observation_dimension, dtype=np.float64)

    for code in range(component_count):
        active = np.array([(code >> index) & 1 for index in range(dimension)], dtype=bool)
        variances = np.where(active, slab_std**2, spike_std**2)
        active_count = int(active.sum())
        prior_log_weights[code] = (
            active_count * math.log(inclusion_probability)
            + (dimension - active_count) * math.log1p(-inclusion_probability)
        )
        observation_covariance = (a * variances[None, :]) @ a.T + noise_std**2 * identity
        sign, log_determinant = np.linalg.slogdet(observation_covariance)
        if sign <= 0.0:
            raise ArithmeticError("观测协方差不是正定矩阵")
        inverse = np.linalg.inv(observation_covariance)
        gain = (variances[:, None] * a.T) @ inverse
        covariance = np.diag(variances) - (gain @ a) * variances[None, :]
        covariance = 0.5 * (covariance + covariance.T)
        prior_log_weights[code] -= 0.5 * (
            observation_dimension * math.log(2.0 * math.pi) + log_determinant
        )
        observation_inverses[code] = inverse
        observation_log_determinants[code] = log_determinant
        gains[code] = gain
        posterior_covariances[code] = covariance

    return {
        "operator": a,
        "prior_log_weights": prior_log_weights,
        "observation_inverses": observation_inverses,
        "observation_log_determinants": observation_log_determinants,
        "gains": gains,
        "posterior_covariances": posterior_covariances,
        "noise_std": np.asarray(noise_std, dtype=np.float64),
    }


def posterior_components(
    observation: object, model: dict[str, np.ndarray]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """根据观测计算精确后验混合权重、均值与协方差。"""
    y = np.asarray(observation, dtype=np.float64)
    a = model["operator"]
    if y.shape != (a.shape[0],) or not np.all(np.isfinite(y)):
        raise ValueError("观测维度错误或包含非有限值")
    inverses = model["observation_inverses"]
    quadratic = np.einsum("i,cij,j->c", y, inverses, y)
    log_weights = model["prior_log_weights"] - 0.5 * quadratic
    log_weights -= np.max(log_weights)
    weights = np.exp(log_weights)
    weights /= weights.sum()
    means = np.einsum("cij,j->ci", model["gains"], y)
    covariances = model["posterior_covariances"].copy()
    if np.any(np.diagonal(covariances, axis1=1, axis2=2) < -1e-12):
        raise ArithmeticError("后验协方差出现负方差")
    return weights, means, covariances


def enumerate_posterior_components(
    observation: object,
    operator: object,
    spike_std: float,
    slab_std: float,
    inclusion_probability: float,
    noise_std: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """便利接口：枚举全部先验成分并计算指定观测的精确后验。"""
    model = build_mixture_model(
        operator, spike_std, slab_std, inclusion_probability, noise_std
    )
    return posterior_components(observation, model)


def make_operator(seed: int, dimension: int, observation_dimension: int) -> np.ndarray:
    """生成固定正交行观测矩阵。"""
    rng = np.random.default_rng(seed)
    basis, _ = np.linalg.qr(rng.standard_normal((dimension, observation_dimension)))
    return basis.T


def sample_joint_cases(
    rng: np.random.Generator,
    operator: np.ndarray,
    count: int,
    spike_std: float,
    slab_std: float,
    inclusion_probability: float,
    noise_std: float,
) -> tuple[np.ndarray, np.ndarray]:
    """从稀疏高斯混合先验与观测模型生成独立目标/观测对。"""
    dimension = operator.shape[1]
    active = rng.random((count, dimension)) < inclusion_probability
    stds = np.where(active, slab_std, spike_std)
    targets = rng.standard_normal((count, dimension)) * stds
    observations = targets @ operator.T + rng.normal(
        0.0, noise_std, size=(count, operator.shape[0])
    )
    return targets, observations


def posterior_mean(weights: np.ndarray, means: np.ndarray) -> np.ndarray:
    return np.sum(weights[:, None] * means, axis=0)


def transformed_mixture(
    weights: np.ndarray,
    means: np.ndarray,
    covariances: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """围绕后验均值收缩/扩张混合成分，返回边际方差。"""
    center = posterior_mean(weights, means)
    transformed_means = center[None, :] + alpha * (means - center[None, :])
    marginal_variances = alpha**2 * np.diagonal(covariances, axis1=1, axis2=2)
    return transformed_means, marginal_variances, np.sqrt(marginal_variances)


def percentile_interval(values: np.ndarray, seed: int) -> tuple[float, float]:
    """对配对案例差进行可复现的百分位 bootstrap。"""
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(BOOTSTRAP_COUNT, values.size), dtype=np.int32)
    means = np.mean(values[indices], axis=1)
    lower, upper = np.quantile(means, [0.025, 0.975])
    return float(lower), float(upper)


def run_experiment() -> dict[str, object]:
    sequence = np.random.SeedSequence(MASTER_SEED)
    data_sequence, bootstrap_sequence = sequence.spawn(2)
    data_rng = np.random.default_rng(data_sequence)
    operator = make_operator(MASTER_SEED, DIMENSION, OBSERVATION_DIMENSION)
    model = build_mixture_model(
        operator,
        SPIKE_STD,
        SLAB_STD,
        INCLUSION_PROBABILITY,
        NOISE_STD,
    )
    targets, observations = sample_joint_cases(
        data_rng,
        operator,
        CASE_COUNT,
        SPIKE_STD,
        SLAB_STD,
        INCLUSION_PROBABILITY,
        NOISE_STD,
    )

    per_case: dict[str, dict[str, np.ndarray]] = {
        name: {
            "crps": np.empty(CASE_COUNT, dtype=np.float64),
            "coverage_50": np.empty(CASE_COUNT, dtype=np.float64),
            "coverage_90": np.empty(CASE_COUNT, dtype=np.float64),
        }
        for name, _ in CONDITIONS
    }
    bootstrap_seed = int(bootstrap_sequence.generate_state(1, dtype=np.uint32)[0])
    for index, (target, observation) in enumerate(zip(targets, observations)):
        weights, component_means, covariances = posterior_components(observation, model)
        for name, alpha in CONDITIONS:
            means, variances, stds = transformed_mixture(
                weights, component_means, covariances, alpha
            )
            per_case[name]["crps"][index] = mixture_crps(
                weights, means, variances, target
            )
            quantiles = mixture_quantile(weights, means, stds, PROBABILITIES)
            inside_50 = (target >= quantiles[1]) & (target <= quantiles[2])
            inside_90 = (target >= quantiles[0]) & (target <= quantiles[3])
            per_case[name]["coverage_50"][index] = float(np.mean(inside_50))
            per_case[name]["coverage_90"][index] = float(np.mean(inside_90))

    comparison = {}
    for offset, (name, _) in enumerate(CONDITIONS[1:]):
        differences = per_case[name]["crps"] - per_case["精确后验"]["crps"]
        lower, upper = percentile_interval(differences, bootstrap_seed + offset)
        comparison[name] = {
            "平均_CRPS差_候选减精确": float(np.mean(differences)),
            "配对bootstrap_95%区间": [lower, upper],
        }

    metrics_path = OUTPUT_DIR / "逐案例结果.csv"
    with metrics_path.open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "案例序号",
                "条件",
                "逐维平均CRPS",
                "50%区间坐标覆盖比例",
                "90%区间坐标覆盖比例",
            ]
        )
        for index in range(CASE_COUNT):
            for name, _ in CONDITIONS:
                item = per_case[name]
                writer.writerow(
                    [
                        index,
                        name,
                        f"{item['crps'][index]:.12g}",
                        f"{item['coverage_50'][index]:.12g}",
                        f"{item['coverage_90'][index]:.12g}",
                    ]
                )

    exact_coverage_50 = float(np.mean(per_case["精确后验"]["coverage_50"]))
    exact_coverage_90 = float(np.mean(per_case["精确后验"]["coverage_90"]))
    checks = {
        "三种非校准条件CRPS差区间下界均大于零": all(
            value["配对bootstrap_95%区间"][0] > 0.0
            for value in comparison.values()
        ),
        "精确后验50%区间覆盖误差不超过0.04": abs(exact_coverage_50 - 0.5) <= 0.04,
        "精确后验90%区间覆盖误差不超过0.04": abs(exact_coverage_90 - 0.9) <= 0.04,
    }
    summary: dict[str, object] = {
        "协议": "posterior_score_calibration_cpu_20261109_v1",
        "主种子": MASTER_SEED,
        "独立案例数": CASE_COUNT,
        "目标维数": DIMENSION,
        "观测维数": OBSERVATION_DIMENSION,
        "稀疏先验纳入概率": INCLUSION_PROBABILITY,
        "尖峰标准差": SPIKE_STD,
        "宽峰标准差": SLAB_STD,
        "观测噪声标准差": NOISE_STD,
        "观测矩阵": operator.tolist(),
        "CRPS均值": {
            name: float(np.mean(values["crps"])) for name, values in per_case.items()
        },
        "区间覆盖率": {
            name: {
                "50%": float(np.mean(values["coverage_50"])),
                "90%": float(np.mean(values["coverage_90"])),
            }
            for name, values in per_case.items()
        },
        "配对CRPS比较": comparison,
        "预声明判据": checks,
        "全部判据通过": all(checks.values()),
        "解释边界": "仅为指定稀疏高斯混合线性逆问题的 CPU 评价方法诊断，不代表 SAR、项目模型或物理后验校准。",
    }
    return summary


def main() -> int:
    existing = [OUTPUT_DIR / "逐案例结果.csv", OUTPUT_DIR / "汇总.json"]
    if any(path.exists() for path in existing):
        print("拒绝覆盖：本次唯一输出目录中已存在结果文件。")
        return 2
    start = time.perf_counter()
    summary = run_experiment()
    elapsed = time.perf_counter() - start
    summary["运行秒数"] = round(elapsed, 3)
    summary["Python版本"] = platform.python_version()
    summary["NumPy版本"] = np.__version__
    with (OUTPUT_DIR / "汇总.json").open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(f"独立合成案例数：{CASE_COUNT}")
    print("平均 CRPS：")
    for name, score in summary["CRPS均值"].items():
        print(f"  {name}：{score:.6f}")
    print("区间覆盖率：")
    for name, coverage in summary["区间覆盖率"].items():
        print(f"  {name}：50%={coverage['50%']:.4f}，90%={coverage['90%']:.4f}")
    for name, difference in summary["配对CRPS比较"].items():
        lower, upper = difference["配对bootstrap_95%区间"]
        print(f"  {name} 减精确后验的 CRPS 差：{difference['平均_CRPS差_候选减精确']:.6f}，95%区间=[{lower:.6f}, {upper:.6f}]")
    for label, passed in summary["预声明判据"].items():
        print(f"判据{'通过' if passed else '未通过'}：{label}")
    print(f"运行耗时：{elapsed:.3f} 秒；结果目录：{OUTPUT_DIR}")
    return 0 if summary["全部判据通过"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
