from __future__ import annotations

import torch

from experiments.phase_e_taskb_operator import (
    downsample_noisy_sar_x2,
    gamma_speckle_like_representation,
)


def test_x2_operator_is_fixed_box_average_and_preserves_constants():
    constant = torch.full((2, 1, 8, 10), 0.37, dtype=torch.float32)
    result = downsample_noisy_sar_x2(constant)

    assert result.shape == (2, 1, 4, 5)
    torch.testing.assert_close(result, torch.full_like(result, 0.37))

    ramp = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
    expected = torch.tensor([[[[2.5, 4.5], [10.5, 12.5]]]], dtype=torch.float32)
    torch.testing.assert_close(downsample_noisy_sar_x2(ramp), expected)


def test_x2_operator_rejects_invalid_tensor_contracts():
    for invalid in (
        torch.zeros((1, 8, 8)),
        torch.zeros((1, 2, 8, 8)),
        torch.zeros((1, 1, 7, 8)),
        torch.full((1, 1, 8, 8), float("nan")),
    ):
        try:
            downsample_noisy_sar_x2(invalid)
        except ValueError:
            continue
        raise AssertionError("不符合输入契约的影像必须报错")


def test_representation_gamma_noise_has_fixed_unit_mean_and_inverse_looks_variance():
    reference = torch.ones((1, 1, 512, 512), dtype=torch.float32)
    measured = []
    for looks in (1.0, 4.0):
        generator = torch.Generator().manual_seed(20261126)
        corrupted = gamma_speckle_like_representation(
            reference, looks=looks, generator=generator
        )
        mean = corrupted.mean().item()
        variance = corrupted.var(unbiased=True).item()
        assert abs(mean - 1.0) < 0.01
        assert abs(variance - 1.0 / looks) < 0.015
        measured.append((mean, variance))
    assert measured[0][1] > measured[1][1]


def test_representation_gamma_noise_rejects_invalid_contracts():
    generator = torch.Generator().manual_seed(20261126)
    for reference, looks in (
        (torch.zeros((1, 1, 8, 8)), 0.0),
        (torch.zeros((1, 2, 8, 8)), 1.0),
        (torch.full((1, 1, 8, 8), -0.1), 1.0),
        (torch.full((1, 1, 8, 8), float("nan")), 1.0),
    ):
        try:
            gamma_speckle_like_representation(reference, looks=looks, generator=generator)
        except ValueError:
            continue
        raise AssertionError("不符合输入契约的合成退化必须报错")
