import numpy as np
import pytest

from MMCANetV2.phase_c3a_metrics import edge_metrics, ssim


def test_ssim_and_edge_metrics_match_identical_known_image():
    image = np.zeros((20, 20), dtype=np.float64)
    image[3:12, 4:15] = 0.5
    image[8:17, 9:18] = 0.9
    mask = np.ones(image.shape, dtype=bool)

    assert ssim(image, image, data_range=1.0) == pytest.approx(1.0)
    scores = edge_metrics(image, image, mask)
    assert scores["epi_corr"] == pytest.approx(1.0)
    assert scores["edge_strength_ratio"] == pytest.approx(1.0)


def test_edge_metrics_reject_zero_target_gradient():
    image = np.zeros((16, 16), dtype=np.float64)
    mask = np.ones(image.shape, dtype=bool)

    with pytest.raises(ValueError, match="目标梯度强度"):
        edge_metrics(image, image, mask)


def test_ssim_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="尺寸"):
        ssim(np.zeros((16, 16)), np.zeros((15, 16)), data_range=1.0)
