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
    simulate_lmmse_case,
)
import run_monte_carlo as runner  # noqa: E402


def _theoretical_result_rows() -> list[dict[str, object]]:
    rows = []
    for case in runner._frequency_cases():
        terms = lmmse_risk_terms(
            case["frequency"],
            case["shift_covariance"],
            signal_power=1.0,
            sar_transfer=1.0,
            sar_noise_power=0.7,
            optical_coupling=0.8,
            optical_noise_power=0.5,
        )
        rows.append(
            {
                "family": case["family"],
                "sigma_pixels": case["sigma"],
                "frequency_index": case["frequency_index"],
                "direction_radians": case["direction_radians"],
                "theoretical_gain": terms["risk_reduction"],
                "coherence_power": terms["coherence_power"],
            }
        )
    return rows


def test_gaussian_shift_coherence_uses_full_anisotropic_covariance() -> None:
    actual = gaussian_shift_coherence_power(
        (2.0, 1.0), ((0.25, 0.10), (0.10, 0.50))
    )

    assert actual == pytest.approx(math.exp(-1.9), rel=1e-14)


def test_lmmse_risk_reduction_matches_hand_derived_scalar_case() -> None:
    actual = lmmse_risk_terms(
        (1.0, 0.0),
        ((math.log(4.0), 0.0), (0.0, 0.0)),
        signal_power=1.0,
        sar_transfer=1.0,
        sar_noise_power=1.0,
        optical_coupling=1.0,
        optical_noise_power=1.0,
    )

    assert actual["coherence_power"] == pytest.approx(0.25, rel=1e-14)
    assert actual["sar_mse"] == pytest.approx(0.5, abs=1e-14)
    assert actual["risk_reduction"] == pytest.approx(1.0 / 30.0, rel=1e-14)
    assert actual["joint_mse"] == pytest.approx(7.0 / 15.0, rel=1e-14)


def test_lmmse_gain_does_not_increase_with_frequency_under_fixed_shift_noise() -> None:
    gains = [
        lmmse_risk_terms(
            (frequency, 0.0),
            ((1.0, 0.0), (0.0, 1.0)),
            signal_power=1.0,
            sar_transfer=1.0,
            sar_noise_power=0.7,
            optical_coupling=0.8,
            optical_noise_power=0.5,
        )["risk_reduction"]
        for frequency in (0.0, 0.25, 0.5, 1.0, 1.5)
    ]

    assert all(left >= right for left, right in zip(gains, gains[1:]))


def test_runner_frequency_trend_check_uses_serialized_sigma_field() -> None:
    summary = runner._verify_frequency_trends(_theoretical_result_rows())

    assert summary["isotropic_frequency_curves_nonincreasing"] is True


def test_runner_report_uses_serialized_sigma_field(tmp_path: Path) -> None:
    rows = _theoretical_result_rows()
    trends = runner._verify_frequency_trends(rows)
    report_path = tmp_path / "报告.md"

    runner._write_report(
        report_path,
        rows,
        [{"gain": 0.0} for _ in rows],
        trends,
    )

    report = report_path.read_text(encoding="utf-8")
    assert runner.PROTOCOL_ID in report
    assert "各向异性方向的理论增益" in report


def test_monte_carlo_gain_matches_theoretical_risk_within_sampling_error() -> None:
    result = simulate_lmmse_case(
        (0.7, -0.3),
        ((0.36, 0.0), (0.0, 0.16)),
        signal_power=1.0,
        sar_transfer=1.0,
        sar_noise_power=0.7,
        optical_coupling=0.8,
        optical_noise_power=0.5,
        samples=30_000,
        seed=20261114,
        batch_size=10_000,
    )

    assert abs(result["empirical_gain"] - result["theoretical_gain"]) <= (
        5.0 * result["empirical_gain_se"] + 1e-12
    )


@pytest.mark.parametrize(
    "frequency,covariance",
    [
        ((1.0,), ((1.0, 0.0), (0.0, 1.0))),
        ((1.0, 0.0), ((1.0, 0.5), (0.0, 1.0))),
        ((1.0, 0.0), ((1.0, 0.0), (0.0, -0.1))),
    ],
)
def test_shift_coherence_rejects_invalid_frequency_or_covariance(
    frequency: tuple[float, ...], covariance: tuple[tuple[float, ...], ...]
) -> None:
    with pytest.raises(ValueError):
        gaussian_shift_coherence_power(frequency, covariance)
