from __future__ import annotations

import torch

from experiments.phase_e_taskb_models import (
    ConditionalResidualUNet,
    ResidualMeanCorrector,
    linear_beta_schedule,
)


def test_residual_mean_corrector_starts_as_identity_and_has_gradients():
    torch.manual_seed(11)
    lr = torch.rand((1, 1, 32, 32))
    base = torch.rand((1, 1, 64, 64))
    model = ResidualMeanCorrector(width=16, residual_blocks=2)

    prediction = model(lr, base)
    assert prediction.shape == base.shape
    torch.testing.assert_close(prediction, base)
    (prediction - 0.4).abs().mean().backward()
    assert model.output.weight.grad is not None
    assert torch.isfinite(model.output.weight.grad).all()


def test_conditional_residual_unet_predicts_noise_and_backpropagates():
    torch.manual_seed(12)
    residual = torch.randn((1, 1, 32, 32))
    mean_estimate = torch.rand((1, 1, 32, 32))
    lr = torch.rand((1, 1, 16, 16))
    timesteps = torch.tensor([137], dtype=torch.long)
    model = ConditionalResidualUNet(base_channels=16, time_dim=64)

    noise = model(residual, mean_estimate, lr, timesteps)
    assert noise.shape == residual.shape
    assert torch.isfinite(noise).all()
    noise.square().mean().backward()
    assert model.output.weight.grad is not None
    assert torch.isfinite(model.output.weight.grad).all()


def test_linear_beta_schedule_is_finite_monotonic_and_protocol_sized():
    betas = linear_beta_schedule(steps=1000, beta_start=1e-4, beta_end=0.02)

    assert betas.shape == (1000,)
    assert torch.isfinite(betas).all()
    assert torch.all(betas[1:] > betas[:-1])
    assert abs(betas[0].item() - 1e-4) < 1e-8
    assert abs(betas[-1].item() - 0.02) < 1e-7
