import numpy as np
import pytest

from MMCANetV2.phase_c3a_sample_set import (
    energy_score,
    mean_pixelwise_sample_std,
    pixelwise_mean,
    source_group_balanced_mean,
    variogram_score,
)


def test_pixelwise_mean_is_a_separate_point_estimate_not_a_draw():
    target = np.array([[[0.25, 0.75]]], dtype=np.float32)
    draws = np.array(
        [
            [[[0.125, 0.625]]],
            [[[0.375, 0.875]]],
        ],
        dtype=np.float32,
    )

    mean_image = pixelwise_mean(draws)

    np.testing.assert_allclose(mean_image, target, rtol=0.0, atol=1e-8)
    assert not any(np.array_equal(mean_image, draw) for draw in draws)


def test_pixelwise_sample_spread_is_separate_and_uses_sample_standard_deviation():
    target = np.array([[[0.25, 0.75]]], dtype=np.float64)
    perturbation = np.array([[[0.125, -0.125]]], dtype=np.float64)
    draws = np.stack((target + perturbation, target - perturbation), axis=0)

    mean_image = pixelwise_mean(draws)
    spread = mean_pixelwise_sample_std(draws)

    np.testing.assert_array_equal(mean_image, target)
    assert spread == pytest.approx(0.125 * np.sqrt(2.0), rel=1e-14)
    assert spread > 0.0
    with pytest.raises(ValueError, match="至少需要两次"):
        mean_pixelwise_sample_std(draws[:1])


def test_energy_score_matches_finite_empirical_ensemble_definition():
    draws = np.array([[[[0.0]]], [[[2.0]]]], dtype=np.float64)
    target = np.array([[[1.0]]], dtype=np.float64)

    assert energy_score(draws, target) == pytest.approx(0.5)


def test_energy_score_requires_multiple_finite_shape_matched_draws():
    target = np.zeros((1, 2, 2), dtype=np.float32)

    with pytest.raises(ValueError, match="至少两次"):
        energy_score(np.zeros((1, 1, 2, 2), dtype=np.float32), target)
    with pytest.raises(ValueError, match="有限"):
        energy_score(np.full((2, 1, 2, 2), np.nan, dtype=np.float32), target)
    with pytest.raises(ValueError, match="形状"):
        energy_score(np.zeros((2, 1, 2, 2), dtype=np.float32), np.zeros((1, 3, 2)))


def test_variogram_score_distinguishes_correct_local_increment_distribution():
    target = np.array([[[0.0, 1.0]]], dtype=np.float64)
    calibrated_draws = np.repeat(target[None, ...], repeats=2, axis=0)
    flat_draws = np.full_like(calibrated_draws, 0.5)

    calibrated = variogram_score(calibrated_draws, target, offsets_yx=((0, 1),), power=0.5)
    misspecified = variogram_score(flat_draws, target, offsets_yx=((0, 1),), power=0.5)

    assert calibrated == pytest.approx(0.0)
    assert misspecified == pytest.approx(1.0)


def test_variogram_score_rejects_invalid_offsets_and_nonfinite_values():
    target = np.zeros((1, 2, 2), dtype=np.float32)
    draws = np.zeros((2, 1, 2, 2), dtype=np.float32)

    with pytest.raises(ValueError, match="偏移"):
        variogram_score(draws, target, offsets_yx=((0, 0),))
    with pytest.raises(ValueError, match="没有有效像素对"):
        variogram_score(draws, target, offsets_yx=((8, 0),))
    draws[0, 0, 0, 0] = np.inf
    with pytest.raises(ValueError, match="有限"):
        variogram_score(draws, target, offsets_yx=((0, 1),))


def test_source_group_balanced_mean_weights_groups_equally_and_checks_identity():
    summary = source_group_balanced_mean(
        sample_ids=("样本甲", "样本乙", "样本丙"),
        source_group_ids=("场景甲", "场景甲", "场景乙"),
        scores=(0.0, 4.0, 10.0),
    )

    assert summary["来源组等权均值"] == pytest.approx(6.0)
    assert summary["样本数"] == 3
    assert summary["来源组数"] == 2
    assert summary["各来源组均值"] == {"场景甲": 2.0, "场景乙": 10.0}

    with pytest.raises(ValueError, match="身份为空或重复"):
        source_group_balanced_mean(
            sample_ids=("样本甲", "样本甲"),
            source_group_ids=("场景甲", "场景乙"),
            scores=(1.0, 2.0),
        )
    with pytest.raises(ValueError, match="等长"):
        source_group_balanced_mean(
            sample_ids=("样本甲", "样本乙"),
            source_group_ids=("场景甲",),
            scores=(1.0, 2.0),
        )
    with pytest.raises(ValueError, match="非有限"):
        source_group_balanced_mean(
            sample_ids=("样本甲", "样本乙"),
            source_group_ids=("场景甲", "场景乙"),
            scores=(1.0, np.nan),
        )
