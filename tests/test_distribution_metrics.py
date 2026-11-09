import numpy as np
import pytest

from scripts.distribution_metrics import (
    adjacent_variogram_score,
    normalized_image_energy_score,
    pixelwise_crps,
)


def test_three_scores_are_zero_for_an_exact_ensemble():
    observation = np.array([[0.0, 0.25], [0.75, 1.0]])
    samples = np.repeat(observation[None, :, :], repeats=3, axis=0)

    assert pixelwise_crps(samples, observation) == pytest.approx(0.0)
    assert normalized_image_energy_score(samples, observation) == pytest.approx(0.0)
    assert adjacent_variogram_score(samples, observation) == pytest.approx(0.0)


def test_pixelwise_crps_matches_two_draw_hand_calculation():
    samples = np.array([[[0.0]], [[1.0]]])
    observation = np.array([[0.25]])

    assert pixelwise_crps(samples, observation) == pytest.approx(0.25)


def test_normalized_image_energy_score_uses_sqrt_pixel_count_normalization():
    samples = np.array([[[0.0, 1.0]]])
    observation = np.array([[0.0, 0.0]])

    assert normalized_image_energy_score(samples, observation) == pytest.approx(
        np.sqrt(0.5)
    )


def test_adjacent_variogram_score_matches_single_edge_calculation():
    samples = np.array([[[0.0, 1.0]]])
    observation = np.array([[0.0, 0.0]])

    assert adjacent_variogram_score(samples, observation) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("samples", "observation", "message"),
    [
        (np.empty((0, 2, 2)), np.zeros((2, 2)), "至少需要一个预测样本"),
        (np.zeros((2, 2)), np.zeros((2, 2)), "samples 必须为三维数组"),
        (np.zeros((2, 2, 3)), np.zeros((2, 2)), "空间尺寸必须一致"),
        (np.full((1, 2, 2), np.nan), np.zeros((2, 2)), "不得包含 NaN 或 Inf"),
        (np.full((1, 2, 2), 1.1), np.zeros((2, 2)), "不会自动裁剪"),
    ],
)
def test_pixelwise_crps_rejects_invalid_inputs(samples, observation, message):
    with pytest.raises(ValueError, match=message):
        pixelwise_crps(samples, observation)


def test_adjacent_variogram_score_requires_at_least_one_neighbor_pair():
    with pytest.raises(ValueError, match="至少需要两个像素"):
        adjacent_variogram_score(np.array([[[0.5]]]), np.array([[0.5]]))
