import numpy as np
import pytest

from experiments.published_baselines_sar_only_20261123_v1.analyze_residual_failure import (
    _load_sar,
    _load_draw_predictions,
    _load_prediction_image,
    frequency_energy_fractions,
    residual_alignment_metrics,
    sample_variability_metrics,
    _sha256,
)


def test_perfect_residual_correction_has_positive_alignment_and_zero_error():
    residual = np.array([[1.0, -1.0], [1.0, -1.0]], dtype=np.float32)

    metrics = residual_alignment_metrics(residual, residual)

    assert metrics["cosine"] == pytest.approx(1.0)
    assert metrics["pearson"] == pytest.approx(1.0)
    assert metrics["baseline_mse"] == pytest.approx(1.0)
    assert metrics["prediction_mse"] == pytest.approx(0.0)
    assert metrics["oracle_alpha"] == pytest.approx(1.0)
    assert metrics["oracle_mse"] == pytest.approx(0.0)


def test_reversed_residual_is_harmful_before_oracle_scaling():
    target_residual = np.ones((3, 3), dtype=np.float32)
    predicted_residual = -target_residual

    metrics = residual_alignment_metrics(target_residual, predicted_residual)

    assert metrics["cosine"] == pytest.approx(-1.0)
    assert metrics["dot_product"] < 0.0
    assert metrics["prediction_mse"] > metrics["baseline_mse"]
    assert metrics["oracle_alpha"] == pytest.approx(-1.0)
    assert metrics["oracle_mse"] == pytest.approx(0.0)


def test_zero_predicted_residual_leaves_base_mse_unchanged():
    target_residual = np.array([[0.5, -0.5]], dtype=np.float32)
    predicted_residual = np.zeros_like(target_residual)

    metrics = residual_alignment_metrics(target_residual, predicted_residual)

    assert np.isnan(metrics["cosine"])
    assert metrics["prediction_mse"] == pytest.approx(metrics["baseline_mse"])
    assert metrics["oracle_alpha"] == pytest.approx(0.0)
    assert metrics["oracle_mse"] == pytest.approx(metrics["baseline_mse"])


def test_constant_residual_energy_is_assigned_to_low_frequency_band():
    fractions = frequency_energy_fractions(np.ones((8, 8), dtype=np.float32))

    assert fractions["low_fraction"] == pytest.approx(1.0)
    assert fractions["mid_fraction"] == pytest.approx(0.0)
    assert fractions["high_fraction"] == pytest.approx(0.0)


def test_draw_variability_reports_pixel_std_and_pairwise_difference():
    draws = np.stack([np.zeros((2, 2)), np.ones((2, 2))]).astype(np.float32)

    metrics = sample_variability_metrics(draws)

    assert metrics["pixel_std_mean"] == pytest.approx(0.5)
    assert metrics["pixel_std_rms"] == pytest.approx(0.5)
    assert metrics["draw_pair_mean_absolute_difference"] == pytest.approx(1.0)


def test_residual_alignment_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="shape"):
        residual_alignment_metrics(np.zeros((2, 2)), np.zeros((2, 3)))


def test_lr_input_keeps_frozen_values_above_one(tmp_path):
    path = tmp_path / "lr.npy"
    source = np.array([[[0.1, 2.5]]], dtype=np.float32)
    np.save(path, source, allow_pickle=False)

    loaded = _load_sar(path, _sha256(path), "LR", unit_range=False)

    assert loaded.max() == pytest.approx(2.5)


def test_target_input_still_requires_unit_interval(tmp_path):
    path = tmp_path / "target.npy"
    np.save(path, np.array([[[0.1, 1.1]]], dtype=np.float32), allow_pickle=False)

    with pytest.raises(ValueError, match=r"\[0,1\]"):
        _load_sar(path, _sha256(path), "target", unit_range=True)


def test_swinir_prediction_loader_accepts_saved_two_dimensional_format(tmp_path):
    path = tmp_path / "base.npy"
    expected = np.full((8, 8), 0.25, dtype=np.float32)
    np.save(path, expected, allow_pickle=False)

    loaded = _load_prediction_image(path, expected.shape, "SwinIR")

    assert loaded.shape == expected.shape
    np.testing.assert_array_equal(loaded, expected)


def test_residual_prediction_loader_accepts_four_saved_two_dimensional_draws(tmp_path):
    path = tmp_path / "draws.npy"
    expected = np.zeros((4, 8, 8), dtype=np.float32)
    expected[2] = 0.75
    np.save(path, expected, allow_pickle=False)

    loaded = _load_draw_predictions(path, (8, 8))

    assert loaded.shape == (4, 8, 8)
    np.testing.assert_array_equal(loaded, expected)
