"""计算高斯平移下的频率域线性最小均方误差及合成抽样结果。"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def _validated_frequency_and_covariance(
    frequency: Sequence[float],
    shift_covariance: Sequence[Sequence[float]],
) -> tuple[np.ndarray, np.ndarray]:
    k = np.asarray(frequency, dtype=np.float64)
    covariance = np.asarray(shift_covariance, dtype=np.float64)
    if k.shape != (2,) or not np.isfinite(k).all():
        raise ValueError("frequency 必须是两个有限角频率分量。")
    if covariance.shape != (2, 2) or not np.isfinite(covariance).all():
        raise ValueError("shift_covariance 必须是有限的 2×2 矩阵。")
    if not np.allclose(covariance, covariance.T, rtol=0.0, atol=1e-12):
        raise ValueError("shift_covariance 必须对称。")
    covariance = (covariance + covariance.T) * 0.5
    if float(np.linalg.eigvalsh(covariance).min()) < -1e-12:
        raise ValueError("shift_covariance 必须半正定。")
    return k, covariance


def gaussian_shift_coherence_power(
    frequency: Sequence[float],
    shift_covariance: Sequence[Sequence[float]],
) -> float:
    """返回零均值高斯平移的相干功率 ``q=exp(-kᵀΣk)``。"""

    k, covariance = _validated_frequency_and_covariance(frequency, shift_covariance)
    exponent = max(0.0, float(k @ covariance @ k))
    return float(np.exp(-exponent))


def lmmse_risk_terms(
    frequency: Sequence[float],
    shift_covariance: Sequence[Sequence[float]],
    *,
    signal_power: float,
    sar_transfer: complex,
    sar_noise_power: float,
    optical_coupling: complex,
    optical_noise_power: float,
) -> dict[str, float]:
    """计算 SAR-only 与联合线性估计的均方误差及其差值。"""

    q = gaussian_shift_coherence_power(frequency, shift_covariance)
    sx = float(signal_power)
    sn = float(sar_noise_power)
    su = float(optical_noise_power)
    h = complex(sar_transfer)
    c = complex(optical_coupling)
    if not np.isfinite((sx, sn, su, h.real, h.imag, c.real, c.imag)).all():
        raise ValueError("功率、传递系数和耦合系数必须是有限值。")
    if sx <= 0.0 or sn < 0.0 or su < 0.0:
        raise ValueError("signal_power 必须为正，噪声功率不能为负。")

    sar_observation_power = abs(h) ** 2 * sx + sn
    if sar_observation_power <= 0.0:
        raise ValueError("SAR 观测功率必须为正。")
    sar_mse = sx - abs(h) ** 2 * sx**2 / sar_observation_power
    sar_mse = max(0.0, float(sar_mse))

    denominator = (
        abs(c) ** 2 * sx
        + su
        - q * abs(h) ** 2 * abs(c) ** 2 * sx**2 / sar_observation_power
    )
    numerator = q * abs(c) ** 2 * sar_mse**2
    if denominator <= 1e-15:
        if numerator > 1e-15:
            raise ValueError("光学残差方差非正，无法定义该 LMMSE 条件。")
        risk_reduction = 0.0
    else:
        risk_reduction = float(numerator / denominator)
    joint_mse = sar_mse - risk_reduction
    if joint_mse < -1e-12:
        raise FloatingPointError("联合线性估计风险低于数值容差。")
    return {
        "coherence_power": q,
        "sar_mse": sar_mse,
        "joint_mse": max(0.0, float(joint_mse)),
        "risk_reduction": risk_reduction,
    }


def _complex_normal(
    rng: np.random.Generator,
    count: int,
    power: float,
) -> np.ndarray:
    scale = np.sqrt(power / 2.0)
    return scale * (rng.standard_normal(count) + 1j * rng.standard_normal(count))


def _normal_shift(
    rng: np.random.Generator,
    count: int,
    covariance: np.ndarray,
) -> np.ndarray:
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    root = eigenvectors @ np.diag(np.sqrt(np.maximum(eigenvalues, 0.0)))
    return rng.standard_normal((count, 2)) @ root.T


def _mean_and_standard_error(total: float, total_squared: float, count: int) -> tuple[float, float]:
    mean = total / count
    sample_variance = max(0.0, (total_squared - total * total / count) / (count - 1))
    return float(mean), float(np.sqrt(sample_variance / count))


def simulate_lmmse_case(
    frequency: Sequence[float],
    shift_covariance: Sequence[Sequence[float]],
    *,
    signal_power: float,
    sar_transfer: complex,
    sar_noise_power: float,
    optical_coupling: complex,
    optical_noise_power: float,
    samples: int,
    seed: int,
    batch_size: int = 50_000,
) -> dict[str, Any]:
    """以独立复高斯样本核对协方差解析式和 Monte Carlo 风险差。"""

    k, covariance = _validated_frequency_and_covariance(frequency, shift_covariance)
    if samples < 2 or batch_size <= 0:
        raise ValueError("samples 至少为 2，batch_size 必须为正数。")
    terms = lmmse_risk_terms(
        k,
        covariance,
        signal_power=signal_power,
        sar_transfer=sar_transfer,
        sar_noise_power=sar_noise_power,
        optical_coupling=optical_coupling,
        optical_noise_power=optical_noise_power,
    )

    sx = float(signal_power)
    sn = float(sar_noise_power)
    su = float(optical_noise_power)
    h = complex(sar_transfer)
    c = complex(optical_coupling)
    sy = abs(h) ** 2 * sx + sn
    q = terms["coherence_power"]
    phi = np.sqrt(q)

    # z=[Y,O] 的理论协方差与 E[z·X*]，独立核对 Schur 补形式。
    cross_y_optical = h * np.conjugate(c) * sx * phi
    covariance_z = np.array(
        [
            [sy, cross_y_optical],
            [np.conjugate(cross_y_optical), abs(c) ** 2 * sx + su],
        ],
        dtype=np.complex128,
    )
    covariance_zx = np.array([h * sx, c * sx * phi], dtype=np.complex128)
    weights = np.linalg.solve(covariance_z, covariance_zx)
    matrix_joint_mse = float(
        sx - np.vdot(covariance_zx, weights).real
    )
    if not np.isclose(matrix_joint_mse, terms["joint_mse"], rtol=1e-10, atol=1e-12):
        raise AssertionError("协方差矩阵解与风险差闭式公式不一致。")

    rng = np.random.default_rng(seed)
    sums = {
        "sar": 0.0,
        "sar_sq": 0.0,
        "joint": 0.0,
        "joint_sq": 0.0,
        "gain": 0.0,
        "gain_sq": 0.0,
    }
    remaining = samples
    while remaining:
        count = min(batch_size, remaining)
        x = _complex_normal(rng, count, sx)
        sar_noise = _complex_normal(rng, count, sn)
        optical_noise = _complex_normal(rng, count, su)
        shift = _normal_shift(rng, count, covariance)
        y = h * x + sar_noise
        phase = np.exp(-1j * (shift @ k))
        optical = phase * (c * x + optical_noise)

        sar_estimate = sx * np.conjugate(h) / sy * y
        joint_estimate = np.conjugate(weights[0]) * y + np.conjugate(weights[1]) * optical
        sar_error = np.abs(sar_estimate - x) ** 2
        joint_error = np.abs(joint_estimate - x) ** 2
        gain_sample = sar_error - joint_error
        for key, values in (
            ("sar", sar_error),
            ("joint", joint_error),
            ("gain", gain_sample),
        ):
            sums[key] += float(values.sum(dtype=np.float64))
            sums[f"{key}_sq"] += float(np.square(values).sum(dtype=np.float64))
        remaining -= count

    empirical_sar, empirical_sar_se = _mean_and_standard_error(
        sums["sar"], sums["sar_sq"], samples
    )
    empirical_joint, empirical_joint_se = _mean_and_standard_error(
        sums["joint"], sums["joint_sq"], samples
    )
    empirical_gain, empirical_gain_se = _mean_and_standard_error(
        sums["gain"], sums["gain_sq"], samples
    )
    return {
        "frequency_x": float(k[0]),
        "frequency_y": float(k[1]),
        "shift_covariance": covariance.tolist(),
        "coherence_power": q,
        "samples": int(samples),
        "seed": int(seed),
        "theoretical_sar_mse": terms["sar_mse"],
        "empirical_sar_mse": empirical_sar,
        "empirical_sar_mse_se": empirical_sar_se,
        "theoretical_joint_mse": terms["joint_mse"],
        "matrix_joint_mse": matrix_joint_mse,
        "empirical_joint_mse": empirical_joint,
        "empirical_joint_mse_se": empirical_joint_se,
        "theoretical_gain": terms["risk_reduction"],
        "empirical_gain": empirical_gain,
        "empirical_gain_se": empirical_gain_se,
        "gain_error_in_mc_se": (
            abs(empirical_gain - terms["risk_reduction"]) / empirical_gain_se
            if empirical_gain_se > 0.0
            else 0.0
        ),
    }


__all__ = [
    "gaussian_shift_coherence_power",
    "lmmse_risk_terms",
    "simulate_lmmse_case",
]
