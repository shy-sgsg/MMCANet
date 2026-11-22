from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_DIR = (
    ROOT
    / "experiments"
    / "synthetic_cpu"
    / "optical_shift_frequency_lmmse_cpu_20261114_v1r2"
)
sys.path.insert(0, str(EXPERIMENT_DIR))

from lmmse_shift_model import (  # noqa: E402
    gaussian_shift_coherence_power,
    lmmse_risk_terms,
)


def _direct_covariance_risks(
    frequency: tuple[float, float],
    covariance: tuple[tuple[float, float], tuple[float, float]],
    *,
    signal_power: float,
    sar_transfer: complex,
    sar_noise_power: float,
    optical_coupling: complex,
    optical_noise_power: float,
) -> tuple[float, float, float]:
    """独立构造观测协方差矩阵并用 Schur 补计算风险。"""
    k = np.asarray(frequency, dtype=np.float64)
    sigma = np.asarray(covariance, dtype=np.float64)
    q = math.exp(-max(0.0, float(k @ sigma @ k)))
    phase_mean_magnitude = math.sqrt(q)
    sx = float(signal_power)
    h = complex(sar_transfer)
    c = complex(optical_coupling)
    sy = abs(h) ** 2 * sx + float(sar_noise_power)
    cross_y_optical = h * sx * np.conjugate(c) * phase_mean_magnitude
    covariance_z = np.asarray(
        [
            [sy, cross_y_optical],
            [np.conjugate(cross_y_optical), abs(c) ** 2 * sx + optical_noise_power],
        ],
        dtype=np.complex128,
    )
    covariance_zx = np.asarray(
        [h * sx, c * sx * phase_mean_magnitude], dtype=np.complex128
    )
    weights = np.linalg.solve(covariance_z, covariance_zx)
    joint_mse = float(sx - np.vdot(covariance_zx, weights).real)
    sar_mse = float(sx - abs(h) ** 2 * sx**2 / sy)
    return q, sar_mse, joint_mse


def test_zero_optical_coupling_has_zero_incremental_risk_reduction() -> None:
    result = lmmse_risk_terms(
        (0.7, -0.2),
        ((0.4, 0.1), (0.1, 0.3)),
        signal_power=1.2,
        sar_transfer=1.1 - 0.6j,
        sar_noise_power=0.35,
        optical_coupling=0.0 + 0.0j,
        optical_noise_power=0.8,
    )

    assert result["risk_reduction"] == 0.0
    assert result["joint_mse"] == pytest.approx(result["sar_mse"], abs=1e-14)


def test_extreme_translation_coherence_zero_removes_optical_gain() -> None:
    covariance = ((1.0, 0.0), (0.0, 0.5))
    frequency = (40.0, 0.0)

    result = lmmse_risk_terms(
        frequency,
        covariance,
        signal_power=1.0,
        sar_transfer=0.9 + 0.3j,
        sar_noise_power=0.4,
        optical_coupling=0.7 - 0.2j,
        optical_noise_power=0.6,
    )

    assert result["coherence_power"] == 0.0
    assert result["risk_reduction"] == 0.0
    assert result["joint_mse"] == pytest.approx(result["sar_mse"], abs=1e-14)


def test_zero_translation_with_complex_coefficients_matches_independent_covariance() -> None:
    frequency = (0.83, -0.29)
    covariance = ((0.0, 0.0), (0.0, 0.0))
    parameters = {
        "signal_power": 1.1,
        "sar_transfer": 1.2 - 0.7j,
        "sar_noise_power": 0.27,
        "optical_coupling": -0.3 + 1.1j,
        "optical_noise_power": 0.4,
    }

    result = lmmse_risk_terms(frequency, covariance, **parameters)
    q, sar_mse, joint_mse = _direct_covariance_risks(
        frequency, covariance, **parameters
    )

    assert result["coherence_power"] == pytest.approx(q, abs=1e-15)
    assert result["coherence_power"] == 1.0
    assert result["sar_mse"] == pytest.approx(sar_mse, rel=1e-13, abs=1e-14)
    assert result["joint_mse"] == pytest.approx(joint_mse, rel=1e-13, abs=1e-14)
    assert result["risk_reduction"] == pytest.approx(
        sar_mse - joint_mse, rel=1e-13, abs=1e-14
    )


def test_near_singular_correlated_shift_covariance_is_stable() -> None:
    frequency = (1.0, -1.0)
    covariance = ((1.0, 1.0 - 1e-14), (1.0 - 1e-14, 1.0))
    parameters = {
        "signal_power": 1.0,
        "sar_transfer": 0.8 + 0.4j,
        "sar_noise_power": 0.5,
        "optical_coupling": 0.6 - 0.7j,
        "optical_noise_power": 0.9,
    }

    result = lmmse_risk_terms(frequency, covariance, **parameters)
    q, sar_mse, joint_mse = _direct_covariance_risks(
        frequency, covariance, **parameters
    )

    assert result["coherence_power"] == pytest.approx(q, rel=1e-13, abs=1e-15)
    assert math.isfinite(result["risk_reduction"])
    assert result["risk_reduction"] >= 0.0
    assert result["sar_mse"] == pytest.approx(sar_mse, rel=1e-13, abs=1e-14)
    assert result["joint_mse"] == pytest.approx(joint_mse, rel=1e-12, abs=1e-14)


@pytest.mark.parametrize(
    "sar_noise_power,optical_noise_power",
    [
        (1e-12, 0.8),
        (1e12, 0.8),
        (0.5, 1e-12),
        (0.5, 1e12),
    ],
)
def test_very_low_and_high_sar_or_optical_noise_keep_risk_terms_finite(
    sar_noise_power: float, optical_noise_power: float
) -> None:
    result = lmmse_risk_terms(
        (0.5, 0.2),
        ((0.2, 0.0), (0.0, 0.1)),
        signal_power=1.0,
        sar_transfer=1.0 - 0.25j,
        sar_noise_power=sar_noise_power,
        optical_coupling=0.5 + 0.75j,
        optical_noise_power=optical_noise_power,
    )

    assert all(math.isfinite(value) for value in result.values())
    assert 0.0 <= result["joint_mse"] <= result["sar_mse"] <= 1.0
    assert result["risk_reduction"] == pytest.approx(
        result["sar_mse"] - result["joint_mse"], rel=1e-12, abs=1e-14
    )


def test_near_singular_covariance_rejects_indefinite_perturbation() -> None:
    with pytest.raises(ValueError, match="半正定"):
        gaussian_shift_coherence_power(
            (1.0, -1.0), ((1.0, 1.0 + 1e-8), (1.0 + 1e-8, 1.0))
        )
