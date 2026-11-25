import numpy as np
import pytest
import torch

from experiments.phase_d2_observation_aware_20261124_v1.observation_model import (
    _energy_from_intermediates,
    apply_representation_operator,
    observation_energy,
)


def test_positive_gamma_energy_gradient_points_toward_observation():
    y = torch.tensor([0.75], dtype=torch.float32)
    below = torch.tensor([0.50], dtype=torch.float32, requires_grad=True)
    equal = torch.tensor([0.75], dtype=torch.float32, requires_grad=True)
    above = torch.tensor([1.00], dtype=torch.float32, requires_grad=True)

    loss_below = _energy_from_intermediates(torch.zeros_like(y), below, y)["positive_nll"]
    loss_equal = _energy_from_intermediates(torch.zeros_like(y), equal, y)["positive_nll"]
    loss_above = _energy_from_intermediates(torch.zeros_like(y), above, y)["positive_nll"]
    grad_below = torch.autograd.grad(loss_below, below)[0]
    grad_equal = torch.autograd.grad(loss_equal, equal)[0]
    grad_above = torch.autograd.grad(loss_above, above)[0]

    assert torch.isfinite(loss_below)
    assert grad_below.item() < 0
    assert grad_equal.item() == pytest.approx(0.0, abs=1e-6)
    assert grad_above.item() > 0



def test_positive_observation_has_zero_guidance_gradient_below_clamp_support():
    # 频域振铃可产生 z<0；当前正值 Gamma 项经两层 clamp 后在该区间没有梯度。
    z = torch.tensor([-0.1], dtype=torch.float32, requires_grad=True)
    a = z.clamp_min(0.0)
    observation = torch.tensor([0.4], dtype=torch.float32)

    positive_nll = _energy_from_intermediates(z, a, observation)["positive_nll"]
    gradient = torch.autograd.grad(positive_nll, z)[0]

    assert torch.isfinite(positive_nll)
    assert gradient.item() == pytest.approx(0.0, abs=1e-8)

def test_zero_observation_penalizes_only_positive_preclamp_prediction():
    y = torch.zeros(2, dtype=torch.float32)
    z = torch.tensor([-0.1, 0.2], dtype=torch.float32, requires_grad=True)
    a = z.clamp_min(0)

    terms = _energy_from_intermediates(z, a, y)
    gradient = torch.autograd.grad(terms["zero_constraint"], z)[0]

    assert terms["zero_constraint"].item() == pytest.approx(0.02)
    assert terms["zero_violation_rate"].item() == pytest.approx(0.5)
    assert gradient[0].item() == pytest.approx(0.0)
    assert gradient[1].item() > 0


def test_fixed_representation_operator_returns_preclamp_and_clamped_values():
    target = torch.linspace(0.0, 1.0, 256 * 256, dtype=torch.float32).reshape(1, 1, 256, 256)
    target.requires_grad_()

    z, a = apply_representation_operator(target)

    assert z.shape == (1, 1, 128, 128)
    assert a.shape == z.shape
    assert torch.isfinite(z).all() and torch.isfinite(a).all()
    assert torch.equal(a, z.clamp_min(0.0))
    assert torch.isfinite(torch.autograd.grad(a.sum(), target)[0]).all()


def test_observation_energy_accepts_lr_values_above_one_without_clipping():
    prediction = torch.full((1, 1, 256, 256), 0.5, dtype=torch.float32)
    observation = torch.full((1, 1, 128, 128), 1.5, dtype=torch.float32)

    terms = observation_energy(prediction, observation)

    assert observation.max().item() == pytest.approx(1.5)
    assert torch.isfinite(terms["total"])
    assert set(terms) == {"positive_nll", "zero_constraint", "zero_violation_rate", "total"}


def test_observation_energy_rejects_shape_and_nonfinite_inputs():
    prediction = torch.ones((1, 1, 256, 256), dtype=torch.float32)
    with pytest.raises(ValueError, match="shape"):
        observation_energy(prediction, torch.ones((1, 1, 127, 128)))
    invalid = prediction.clone()
    invalid[0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        observation_energy(invalid, torch.ones((1, 1, 128, 128)))
